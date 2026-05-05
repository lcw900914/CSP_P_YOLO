"""
Coordinate Attention (CA) Module
論文：CSPPartial-YOLO (IEEE JSTARS 2024), Section III-B-2

依論文 Eq.(2)-(7) 實作：
  zh(h) = mean over W  → B,C,H,1
  zw(w) = mean over H  → B,C,1,W
  concat → B,C,H+W,1
  shared 1×1 conv → BN → ReLU (channel reduction by r)
  split → fh, fw
  各自 1×1 升回 C → sigmoid → 注意力圖
  output = x * gh * gw
"""

import torch
import torch.nn as nn


class CoordAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mid_ch = max(8, channels // reduction)

        # 共享 1×1 卷積（降維） + BN + ReLU
        self.shared_conv = nn.Conv2d(channels, mid_ch, 1, bias=False)
        self.bn          = nn.BatchNorm2d(mid_ch)
        self.act         = nn.ReLU(inplace=True)

        # 各方向升維回 C
        self.conv_h = nn.Conv2d(mid_ch, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid_ch, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Eq.(2): 沿 W 做平均，保留 H 位置資訊
        zh = x.mean(dim=3, keepdim=True)          # B,C,H,1

        # Eq.(3): 沿 H 做平均，保留 W 位置資訊
        zw = x.mean(dim=2, keepdim=True)          # B,C,1,W
        zw = zw.permute(0, 1, 3, 2)              # B,C,W,1

        # Eq.(4): concat → 共享 1×1 → BN → ReLU
        z = torch.cat([zh, zw], dim=2)            # B,C,H+W,1
        z = self.act(self.bn(self.shared_conv(z)))# B,mid,H+W,1

        # 分割回兩個方向
        fh = z[:, :, :H, :]                       # B,mid,H,1
        fw = z[:, :, H:, :]                       # B,mid,W,1
        fw = fw.permute(0, 1, 3, 2)              # B,mid,1,W

        # Eq.(5)(6): 升維 + Sigmoid
        gh = torch.sigmoid(self.conv_h(fh))       # B,C,H,1
        gw = torch.sigmoid(self.conv_w(fw))       # B,C,1,W

        # Eq.(7): y(i,j) = x(i,j) * gh(i) * gw(j)
        return x * gh * gw
