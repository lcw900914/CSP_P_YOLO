"""
CSPPartialNet：骨幹網路
論文：CSPPartial-YOLO (IEEE JSTARS 2024), Section III-C

架構：
  Stem → Stage0 → Down → Stage1 → Down → Stage2 → Down → Stage3
  輸出最後三個 Stage 的特徵圖：P3(128×128), P4(64×64), P5(32×32)
  PHDC Block 數量比 [1,1,3,1]（論文明確引用 ConvNeXt）
  通道配置參考 PP-YOLOE-R-S：[32, 64, 128, 256, 512]
"""

import torch
import torch.nn as nn
from .csp_partial_stage import CSPPartialStage, CBN
from typing import Tuple


class StemBlock(nn.Module):
    """初步降採樣：3×3 Conv stride=2，1024 → 512"""
    def __init__(self, in_ch: int = 3, out_ch: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownSample(nn.Module):
    """3×3 Conv stride=2 降採樣"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CSPPartialNet(nn.Module):
    """
    輸入：B × 3 × 1024 × 1024
    輸出：(P3, P4, P5)
      P3：B × 128 × 128 × 128
      P4：B × 256 ×  64 ×  64
      P5：B × 512 ×  32 ×  32
    """
    def __init__(self):
        super().__init__()

        # Stem：1024 → 512, ch: 3→32
        self.stem   = StemBlock(3, 32)
        self.down_stem = DownSample(32) # 512 → 256（讓後續 Stage 輸出對齊論文）

        # Stage0：256→256, ch: 32→64, n_blocks=1（此層輸出不送入 FPN）
        self.stage0 = CSPPartialStage(32,  64,  n_blocks=1, use_ca=True)
        self.down0  = DownSample(64)    # 256 → 128

        # Stage1 (P3)：128→128, ch: 64→128, n_blocks=1
        self.stage1 = CSPPartialStage(64,  128, n_blocks=1, use_ca=True)
        self.down1  = DownSample(128)   # 128 → 64

        # Stage2 (P4)：64→64, ch: 128→256, n_blocks=3
        self.stage2 = CSPPartialStage(128, 256, n_blocks=3, use_ca=True)
        self.down2  = DownSample(256)   # 64 → 32

        # Stage3 (P5)：32→32, ch: 256→512, n_blocks=1
        self.stage3 = CSPPartialStage(256, 512, n_blocks=1, use_ca=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x  = self.stem(x)       # B,32,512,512
        x  = self.down_stem(x)  # B,32,256,256

        x  = self.stage0(x)     # B,64,256,256
        x  = self.down0(x)      # B,64,128,128

        p3 = self.stage1(x)     # B,128,128,128
        x  = self.down1(p3)     # B,128,64,64

        p4 = self.stage2(x)     # B,256,64,64
        x  = self.down2(p4)     # B,256,32,32

        p5 = self.stage3(x)     # B,512,32,32

        return p3, p4, p5
