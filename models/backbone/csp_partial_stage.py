"""
CSPPartialStage — backbone 裡的一個 stage。

走 CSP 那一套：進來先把通道砍一半，然後分兩條路——
一條(branch1)幾乎原樣放著，另一條(branch2)丟進 PHDC 堆疊去抽特徵，
最後兩條接起來、過一下 CA，再調回需要的通道數輸出。
這種「一半留、一半算」的做法可以省掉不少計算。

小提醒：這裡所有的 CBN 都是 1×1 卷積。
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

        # 一進來先把通道砍一半
        self.conv_in  = CBN(in_ch, mid_ch)

        # 分兩條路，兩條的輸入都是 mid_ch
        self.branch1  = CBN(mid_ch, mid_ch)              # 這條基本上原樣放著

        self.branch2  = nn.Sequential(                   # 這條才是真的在抽特徵
            CBN(mid_ch, mid_ch),
            *[PHDCBlock(mid_ch) for _ in range(n_blocks)],
        )

        # CA 看要不要開
        self.use_ca   = use_ca
        if use_ca:
            self.ca   = CoordAttention(mid_ch * 2)

        # 出口再把通道調回外面要的數量
        self.conv_out = CBN(mid_ch * 2, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = self.conv_in(x)
        b1  = self.branch1(x)
        b2  = self.branch2(x)
        out = torch.cat([b1, b2], dim=1)
        if self.use_ca:
            out = self.ca(out)
        return self.conv_out(out)
