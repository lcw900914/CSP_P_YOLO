"""
CSPPartialNet — 骨幹，負責從圖片抽特徵。

流程就是一路往下抽：
  Stem → Stage0 → 降採樣 → Stage1 → 降採樣 → Stage2 → 降採樣 → Stage3
最後把後三個 stage 的特徵 P3/P4/P5 丟出去給 neck（分別是 /8、/16、/32 解析度）。
每個 stage 裡 PHDC block 的數量是 [1,1,3,1]，通道數沿用 PP-YOLOE-R-S 的配置。
"""

import torch
import torch.nn as nn
from .csp_partial_stage import CSPPartialStage, CBN
from typing import Tuple


class StemBlock(nn.Module):
    """開頭第一刀降採樣：3×3 stride=2 卷積，把圖縮一半"""
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
    """用 MaxPool 來縮小尺寸。改用 pool 而不是 stride 卷積，就是為了省 FLOPs"""
    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


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
        self.down_stem = DownSample(32) # 再縮一次到 256，這樣後面尺寸才對得上論文

        # Stage0：這層的輸出不會送給 FPN，純粹墊一層
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
