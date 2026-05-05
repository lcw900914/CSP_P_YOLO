"""
CSPPartialStage
論文：CSPPartial-YOLO (IEEE JSTARS 2024), Section III-B-1, Fig.3

結構（依論文原文）：
  Input
    └─ CBN (1×1, 通道減半)
    ├─ Branch1: CBN (1×1)
    └─ Branch2: CBN (1×1) → PHDC Block × N
  Concat → CA Module → CBN (1×1, 通道匹配)

注意：論文明確指出「all CBNs use 1×1 convolution」
"""

import torch
import torch.nn as nn
from .phdc_block import PHDCBlock
from ..layers.coord_attention import CoordAttention


class CBN(nn.Module):
    """Conv (1×1) + BatchNorm + ReLU"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CSPPartialStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 n_blocks: int = 1, use_ca: bool = True):
        super().__init__()
        mid_ch = in_ch // 2

        # 入口 CBN：通道減半
        self.conv_in  = CBN(in_ch, mid_ch)

        # 兩條分支（輸入相同，均為 mid_ch）
        self.branch1  = CBN(mid_ch, mid_ch)

        self.branch2  = nn.Sequential(
            CBN(mid_ch, mid_ch),
            *[PHDCBlock(mid_ch) for _ in range(n_blocks)],
        )

        # CA（可選）
        self.use_ca   = use_ca
        if use_ca:
            self.ca   = CoordAttention(mid_ch * 2)

        # 出口 CBN：通道匹配
        self.conv_out = CBN(mid_ch * 2, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = self.conv_in(x)
        b1  = self.branch1(x)
        b2  = self.branch2(x)
        out = torch.cat([b1, b2], dim=1)
        if self.use_ca:
            out = self.ca(out)
        return self.conv_out(out)
