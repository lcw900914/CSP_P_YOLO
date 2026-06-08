"""
Coordinate Attention (CA) — 一種很省的注意力。

它不是直接學一張 2D 的注意力圖，而是拆成「橫」和「豎」兩個方向：
  - 沿著寬度平均 → 得到每一橫列的重要度
  - 沿著高度平均 → 得到每一直行的重要度
兩個方向各自算出一組 0~1 的權重，再乘回特徵圖上。
好處是計算量很小，又能保留位置感，對遙感這種有方向性的物體蠻有用。
"""

import torch
import torch.nn as nn


class CoordAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mid_ch = max(8, channels // reduction)

        # 先用 1×1 把通道壓小一點，省參數，後面兩個方向共用這層
        self.shared_conv = nn.Conv2d(channels, mid_ch, 1, bias=False)
        self.bn          = nn.BatchNorm2d(mid_ch)
        self.act         = nn.ReLU(inplace=True)

        # 兩個方向各自再把通道升回原本的數量
        self.conv_h = nn.Conv2d(mid_ch, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid_ch, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 沿寬度壓成一條，留下「每一橫列」的資訊
        zh = x.mean(dim=3, keepdim=True)          # B,C,H,1

        # 沿高度壓成一條，留下「每一直行」的資訊
        zw = x.mean(dim=2, keepdim=True)          # B,C,1,W
        zw = zw.permute(0, 1, 3, 2)              # B,C,W,1

        # 兩條接起來一起過共用的 1×1 + BN + ReLU
        z = torch.cat([zh, zw], dim=2)            # B,C,H+W,1
        z = self.act(self.bn(self.shared_conv(z)))# B,mid,H+W,1

        # 再拆回橫、豎兩個方向
        fh = z[:, :, :H, :]                       # B,mid,H,1
        fw = z[:, :, H:, :]                       # B,mid,W,1
        fw = fw.permute(0, 1, 3, 2)              # B,mid,1,W

        # 各自升回原通道、過 sigmoid 變成 0~1 的權重
        gh = torch.sigmoid(self.conv_h(fh))       # B,C,H,1
        gw = torch.sigmoid(self.conv_w(fw))       # B,C,1,W

        # 橫權重 × 豎權重，乘回原本的特徵圖
        return x * gh * gw
