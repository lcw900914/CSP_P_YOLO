"""
PHDC Block — 這篇論文省計算量的核心模組。

想法是：只挑一部分通道(Cp)去做卷積，其他通道原樣留著，
而且那部分通道用的是「不同空洞率串起來的」空洞卷積，
這樣既省 FLOPs 又能看到比較大的範圍。

一個 block 的流程：
  PartialConv(空洞卷積[1,2,5]) → 1×1 → BN+ReLU → 1×1 → 加回輸入(殘差)
"""

import torch
import torch.nn as nn


class PartialConv(nn.Module):
    """
    部分卷積：只對前 Cp 個通道做空洞卷積，後面的通道直接原封不動傳下去。
    那 Cp 個通道會連過三層空洞卷積（空洞率 1/2/5），中間不放 BN 跟 ReLU。
    """
    def __init__(self, channels: int, cp_ratio: float = 0.25,
                 dilation_rates=(1, 2, 5)):
        super().__init__()
        self.cp = max(1, int(channels * cp_ratio))  # 真正參與卷積的通道數

        # 三層空洞卷積串聯，無 BN / ReLU
        self.hdc = nn.Sequential(
            nn.Conv2d(self.cp, self.cp, 3,
                      padding=dilation_rates[0], dilation=dilation_rates[0],
                      bias=False),
            nn.Conv2d(self.cp, self.cp, 3,
                      padding=dilation_rates[1], dilation=dilation_rates[1],
                      bias=False),
            nn.Conv2d(self.cp, self.cp, 3,
                      padding=dilation_rates[2], dilation=dilation_rates[2],
                      bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, :self.cp]          # 前面這幾個通道拿去做空洞卷積
        x2 = x[:, self.cp:]          # 後面的通道不動，直接留著

        x1 = self.hdc(x1)
        return torch.cat([x1, x2], dim=1)   # 算完的跟沒動的接回去


class PHDCBlock(nn.Module):
    """
    一個 PHDC block：
      部分卷積 → 1×1 → BN → ReLU → 1×1 → 加回輸入
    最後那個殘差(加回輸入)是為了好訓練、避免梯度消失。
    """
    def __init__(self, channels: int, cp_ratio: float = 0.25):
        super().__init__()
        self.pconv = PartialConv(channels, cp_ratio)
        self.pw1   = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn    = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)
        self.pw2   = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pconv(x)
        out = self.pw1(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pw2(out)
        return out + x   # 加回原本的輸入（殘差）
