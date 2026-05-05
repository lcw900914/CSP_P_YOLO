"""
旋轉框工具函數
格式：(cx, cy, w, h, angle)  angle 單位為弧度，範圍 [-π/2, 0)  (le90)

主要功能：
  - xywha_to_gaussian: 旋轉框 → Gaussian 參數（用於 ProbIoU）
  - make_anchors: anchor-free grid 座標生成
  - dist2rbox: 分佈預測 → 旋轉框（用於 decode）
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple


# ─────────────────────────────────────────
# Gaussian 編碼（用於 ProbIoU）
# ─────────────────────────────────────────

def xywha_to_gaussian(boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    將旋轉框轉換為 Gaussian 分佈參數。
    Args:
        boxes: (..., 5) = (cx, cy, w, h, angle_rad)
    Returns:
        mu:    (..., 2)   中心點
        sigma: (..., 2, 2) 協方差矩陣
    """
    cx, cy, w, h, angle = boxes.unbind(-1)

    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    # 半軸長度（標準差）
    w2 = (w / 2) ** 2
    h2 = (h / 2) ** 2

    # 協方差矩陣 Σ = R diag(w²/4, h²/4) R^T
    sigma_xx = cos_a ** 2 * w2 + sin_a ** 2 * h2
    sigma_yy = sin_a ** 2 * w2 + cos_a ** 2 * h2
    sigma_xy = cos_a * sin_a * (w2 - h2)

    mu    = torch.stack([cx, cy], dim=-1)
    sigma = torch.stack([
        torch.stack([sigma_xx, sigma_xy], dim=-1),
        torch.stack([sigma_xy, sigma_yy], dim=-1),
    ], dim=-2)

    return mu, sigma


def bhattacharyya_distance(
    mu1: torch.Tensor, sigma1: torch.Tensor,
    mu2: torch.Tensor, sigma2: torch.Tensor,
) -> torch.Tensor:
    """
    計算兩個 2D Gaussian 之間的 Bhattacharyya 距離。
    Bd = 1/8 * Δμ^T (Σavg)^-1 Δμ + 1/2 * log(|Σavg| / sqrt(|Σ1||Σ2|))
    """
    sigma_avg = (sigma1 + sigma2) / 2.0

    # 行列式
    def det2x2(m):
        return m[..., 0, 0] * m[..., 1, 1] - m[..., 0, 1] * m[..., 1, 0]

    # 逆矩陣 (2×2)
    def inv2x2(m):
        d = det2x2(m).clamp(min=1e-7).unsqueeze(-1).unsqueeze(-1)
        adj = torch.stack([
            torch.stack([ m[..., 1, 1], -m[..., 0, 1]], dim=-1),
            torch.stack([-m[..., 1, 0],  m[..., 0, 0]], dim=-1),
        ], dim=-2)
        return adj / d

    det1     = det2x2(sigma1).clamp(min=1e-7)
    det2     = det2x2(sigma2).clamp(min=1e-7)
    det_avg  = det2x2(sigma_avg).clamp(min=1e-7)

    # 馬氏距離項
    dmu      = (mu1 - mu2).unsqueeze(-1)           # (..., 2, 1)
    sigma_inv = inv2x2(sigma_avg)
    maha     = (dmu.transpose(-1, -2) @ sigma_inv @ dmu).squeeze(-1).squeeze(-1)

    bd = 0.125 * maha + 0.5 * torch.log(
        det_avg / (det1 * det2).sqrt().clamp(min=1e-7)
    )
    return bd.clamp(min=0.0)


# ─────────────────────────────────────────
# Anchor-free 格點生成
# ─────────────────────────────────────────

def make_anchors(
    feats: Tuple[torch.Tensor, ...],
    strides: Tuple[int, ...],
    offset: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    為每個 FPN 層級生成格點中心座標（原圖像素）。
    Returns:
        anchor_points: (N_total, 2)  格點中心 (x, y)
        stride_tensor: (N_total, 1)  對應的 stride
    """
    anchor_pts_list, stride_list = [], []
    for feat, stride in zip(feats, strides):
        _, _, h, w = feat.shape
        sx = torch.arange(w, device=feat.device) + offset  # (W,)
        sy = torch.arange(h, device=feat.device) + offset  # (H,)
        gy, gx = torch.meshgrid(sy, sx, indexing='ij')
        pts = torch.stack([gx.flatten(), gy.flatten()], dim=-1) * stride
        anchor_pts_list.append(pts)
        stride_list.append(torch.full((h * w, 1), stride,
                                      dtype=feat.dtype, device=feat.device))
    return torch.cat(anchor_pts_list), torch.cat(stride_list)


# ─────────────────────────────────────────
# 分佈解碼 → 旋轉框
# ─────────────────────────────────────────

def dist2rbox(
    pred_dist: torch.Tensor,
    pred_angle: torch.Tensor,
    anchor_points: torch.Tensor,
    reg_max: int = 16,
    stride = 1.0,
) -> torch.Tensor:
    """
    將 DFL 分佈預測 + 角度預測解碼為旋轉框 (cx, cy, w, h, angle)。
    Args:
        pred_dist:   (N, 4*(reg_max+1)) 或 (B, N, 4*(reg_max+1))
        pred_angle:  (N, 1) 或 (B, N, 1)   單位：弧度
        anchor_points: (N, 2)
        reg_max:     分佈的最大值
        stride:      對應 FPN 層的 stride（scalar 或 (N,1) tensor）
    Returns:
        (N, 5) 或 (B, N, 5) 格式 (cx, cy, w, h, angle)
    """
    # softmax 解碼各方向距離
    dist = pred_dist.reshape(*pred_dist.shape[:-1], 4, reg_max + 1)
    dist = F.softmax(dist, dim=-1)
    bins = torch.arange(reg_max + 1, dtype=dist.dtype,
                        device=dist.device)
    dist = (dist * bins).sum(-1) * stride  # (..., 4): [l, t, r, b] in pixels

    lt, rb = dist[..., :2], dist[..., 2:]
    x1y1   = anchor_points - lt
    x2y2   = anchor_points + rb
    cx     = (x1y1[..., 0] + x2y2[..., 0]) / 2
    cy     = (x1y1[..., 1] + x2y2[..., 1]) / 2
    w      = x2y2[..., 0] - x1y1[..., 0]
    h      = x2y2[..., 1] - x1y1[..., 1]

    angle = pred_angle.squeeze(-1)
    return torch.stack([cx, cy, w, h, angle], dim=-1)
