"""
SPP（Spatial Pyramid Pooling）— 用幾個不同大小的池化窗去看同一張特徵圖，
小窗看局部、大窗看全域，再把結果疊起來，等於一次抓到多種尺度的資訊。
放在 FPN 最高層。池化窗大小論文沒講，這裡先用 [1, 5, 9, 13]。
"""

import torch
import torch.nn as nn


class SPP(nn.Module):
    def __init__(self, channels: int, pool_sizes=(1, 5, 9, 13)):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
            for k in pool_sizes
        ])

        # 疊起來後通道會變 N 倍，用 1×1 壓回原本的數量
        self.conv = nn.Sequential(
            nn.Conv2d(channels * len(pool_sizes), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = [pool(x) for pool in self.pools]
        out = torch.cat(pooled, dim=1)
        return self.conv(out)
