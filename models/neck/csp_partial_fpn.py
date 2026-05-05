"""
CSPPartialFPN：頸部網路
論文：CSPPartial-YOLO (IEEE JSTARS 2024), Section III-C, Fig.4

結構：
  - FPNStage 與 CSPPartialStage 相似，但不含 CA
  - 最高層 (P5) 在 branch2 通過 SPP
  - 雙向 FPN：Top-down pass → Bottom-up pass

輸入：(P3, P4, P5) from CSPPartialNet
輸出：(P3', P4', P5') 同 channel，供 detection head 使用
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
    FPN 頸部模塊（無 CA）。
    結構與 CSPPartialStage 相似：入口 CBN 先將通道減半，雙分支後 concat 再輸出。
    最高層特徵圖（use_spp=True）在 branch2 的 PHDC 之後使用 SPP。
    """
    def __init__(self, in_ch: int, out_ch: int, use_spp: bool = False):
        super().__init__()
        mid_ch = in_ch // 2

        # 入口 CBN：通道減半（與 CSPPartialStage 相同設計）
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
    雙向特徵金字塔：
      Top-down：P5 → P4 → P3（上採樣 + concat）
      Bottom-up：P3 → P4 → P5（下採樣 + concat）
    """
    def __init__(self, in_channels: Tuple[int, int, int] = (128, 256, 512)):
        super().__init__()
        c3, c4, c5 = in_channels

        # --- Top-down ---
        self.fpn_p5    = FPNStage(c5,       c5, use_spp=True)
        self.lat_p5    = CBN(c5, c4)              # 調整 P5 通道數以配合 P4
        self.fpn_p4_td = FPNStage(c4 + c4,  c4)
        self.lat_p4    = CBN(c4, c3)              # 調整 P4 通道數以配合 P3
        self.fpn_p3_td = FPNStage(c3 + c3,  c3)

        # --- Bottom-up ---
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

        # --- Top-down pass ---
        p5 = self.fpn_p5(p5)                              # B,c5,H5,W5

        p5_up = F.interpolate(self.lat_p5(p5),
                              size=p4.shape[2:], mode='nearest')
        p4 = self.fpn_p4_td(torch.cat([p4, p5_up], dim=1))

        p4_up = F.interpolate(self.lat_p4(p4),
                              size=p3.shape[2:], mode='nearest')
        p3 = self.fpn_p3_td(torch.cat([p3, p4_up], dim=1))

        # --- Bottom-up pass ---
        p3_dn = self.down_p3(p3)
        p4 = self.fpn_p4_bu(torch.cat([p3_dn, p4], dim=1))

        p4_dn = self.down_p4(p4)
        p5 = self.fpn_p5_bu(torch.cat([p4_dn, p5], dim=1))

        return p3, p4, p5
