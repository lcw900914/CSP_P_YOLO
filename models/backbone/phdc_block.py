"""
PHDC Block：Partial Hybrid Dilated Convolution Block
論文：CSPPartial-YOLO (IEEE JSTARS 2024)

結構（依論文 Fig.8(b) 及 page 4 原文）：
  Input
    └─ PartialConv (HDC [1,2,5] on Cp channels, 無BN/ReLU)
    └─ PW1 (1×1 Conv)
    └─ BN + ReLU
    └─ PW2 (1×1 Conv)
    └─ + residual
"""

import torch
import torch.nn as nn


class PartialConv(nn.Layer if hasattr(nn, 'Layer') else nn.Module):
    pass


class PartialConv(nn.Module):
    """
    部分卷積：只對前 Cp 個通道做 HDC，其餘通道直接 Identity 傳遞。
    HDC 三層串聯，空洞率 [1, 2, 5]，層間無 BN/ReLU。
    """
    def __init__(self, channels: int, cp_ratio: float = 0.25,
                 dilation_rates=(1, 2, 5)):
        super().__init__()
        self.cp = max(1, int(channels * cp_ratio))  # 參與卷積的通道數

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
        x1 = x[:, :self.cp]          # 前 Cp 通道：做 HDC
        x2 = x[:, self.cp:]          # 後 C-Cp 通道：Identity

        x1 = self.hdc(x1)
        return torch.cat([x1, x2], dim=1)


class PHDCBlock(nn.Module):
    """
    PHDC Block：
      PConv(HDC) → PW1(1×1) → BN → ReLU → PW2(1×1) → + residual
    論文 page 4：
      'we apply pointwise convolution to the output of partial convolution,
       then follow up with a BatchNorm layer and a ReLU activation function,
       before finally restoring channel dimensionality using pointwise convolution.'
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
        return out + x   # 殘差連接
