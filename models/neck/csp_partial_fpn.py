"""
CSPPartialFPN — 頸部，把 backbone 出來的三層特徵互相融合。

重點：
  - 裡面的 FPNStage 長得跟 backbone 的 stage 很像，但沒有 CA
  - 最高層 P5 那條會多過一個 SPP
  - 走雙向：先由上往下傳一遍，再由下往上傳一遍，讓大小物體的資訊互通

吃進 backbone 的 (P3, P4, P5)，吐出同樣三層、通道數不變，交給偵測頭。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbone.phdc_block import PHDCBlock
from ..backbone.csp_partial_stage import CBN
from ..layers.spp import SPP
from typing import Tuple


class FPNStage(nn.Module):
    """
    FPN 用的模塊，跟 backbone 的 stage 幾乎一樣，差別是沒有 CA。
    一樣是進來砍半通道、分兩條、接回來。
    只有最高層會把 use_spp 打開，在第二條的 PHDC 後面多塞一個 SPP。
    """
    def __init__(self, in_ch: int, out_ch: int, use_spp: bool = False):
        super().__init__()
        mid_ch = in_ch // 2

        # 進來先砍半通道（跟 backbone 的 stage 一樣套路）
        self.conv_in       = CBN(in_ch, mid_ch)

        self.branch1       = CBN(mid_ch, mid_ch)

        self.branch2_conv  = CBN(mid_ch, mid_ch)
        self.branch2_phdc  = PHDCBlock(mid_ch)
        self.use_spp       = use_spp
        if use_spp:
            self.spp       = SPP(mid_ch)

        self.conv_out = CBN(mid_ch * 2, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = self.conv_in(x)
        b1  = self.branch1(x)
        b2  = self.branch2_conv(x)
        b2  = self.branch2_phdc(b2)
        if self.use_spp:
            b2 = self.spp(b2)
        out = torch.cat([b1, b2], dim=1)
        return self.conv_out(out)


class CSPPartialFPN(nn.Module):
    """
    雙向的特徵金字塔：
      由上往下：P5 放大去跟 P4、P3 合（把高層的語意傳給低層）
      由下往上：P3 縮小去跟 P4、P5 合（把低層的細節傳給高層）
    """
    def __init__(self, in_channels: Tuple[int, int, int] = (128, 256, 512)):
        super().__init__()
        c3, c4, c5 = in_channels

        # --- 由上往下 ---
        self.fpn_p5    = FPNStage(c5,       c5, use_spp=True)
        self.lat_p5    = CBN(c5, c4)              # 把 P5 通道調成跟 P4 一樣才好相加
        self.fpn_p4_td = FPNStage(c4 + c4,  c4)
        self.lat_p4    = CBN(c4, c3)              # 把 P4 通道調成跟 P3 一樣
        self.fpn_p3_td = FPNStage(c3 + c3,  c3)

        # --- 由下往上 ---
        self.down_p3   = nn.Sequential(
            nn.Conv2d(c3, c3, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
        )
        self.fpn_p4_bu = FPNStage(c3 + c4,  c4)
        self.down_p4   = nn.Sequential(
            nn.Conv2d(c4, c4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4), nn.ReLU(inplace=True),
        )
        self.fpn_p5_bu = FPNStage(c4 + c5,  c5)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor,
                p5: torch.Tensor) -> Tuple[torch.Tensor, ...]:

        # --- 先由上往下 ---
        p5 = self.fpn_p5(p5)                              # B,c5,H5,W5

        p5_up = F.interpolate(self.lat_p5(p5),
                              size=p4.shape[2:], mode='nearest')
        p4 = self.fpn_p4_td(torch.cat([p4, p5_up], dim=1))

        p4_up = F.interpolate(self.lat_p4(p4),
                              size=p3.shape[2:], mode='nearest')
        p3 = self.fpn_p3_td(torch.cat([p3, p4_up], dim=1))

        # --- 再由下往上 ---
        p3_dn = self.down_p3(p3)
        p4 = self.fpn_p4_bu(torch.cat([p3_dn, p4], dim=1))

        p4_dn = self.down_p4(p4)
        p5 = self.fpn_p5_bu(torch.cat([p4_dn, p5], dim=1))

        return p3, p4, p5
