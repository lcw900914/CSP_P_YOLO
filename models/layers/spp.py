"""
SPP：Spatial Pyramid Pooling
論文：CSPPartial-YOLO (IEEE JSTARS 2024)
用於 FPN 最高層特徵圖，融合局部與全局特徵。
池化核大小論文未說明，依復現指南先試 [1, 5, 9, 13]。
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

        # 拼接後通道 = channels * N，用 1×1 壓回原始通道數
        self.conv = nn.Sequential(
            nn.Conv2d(channels * len(pool_sizes), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = [pool(x) for pool in self.pools]
        out = torch.cat(pooled, dim=1)
        return self.conv(out)
