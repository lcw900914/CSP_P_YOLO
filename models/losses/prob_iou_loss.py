"""
ProbIoU Loss
論文：PP-YOLOE-R (Cao et al. 2022)
用於 CSPPartial-YOLO 旋轉框回歸（權重 2.5）

將旋轉框編碼為 2D Gaussian，計算 Bhattacharyya 距離。
ProbIoU = 1 - exp(-Bd)
Loss    = 1 - ProbIoU
"""

import torch
import torch.nn as nn
from ..utils.rotated_box import xywha_to_gaussian, bhattacharyya_distance


class ProbIoULoss(nn.Module):
    def __init__(self, reduction: str = 'sum'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor,
                weight: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred:   (N, 5) 預測旋轉框 (cx, cy, w, h, angle_rad)
            target: (N, 5) GT 旋轉框  (cx, cy, w, h, angle_rad)
            weight: (N,)   樣本權重（可選）
        Returns:
            scalar loss
        """
        mu1, sigma1 = xywha_to_gaussian(pred)
        mu2, sigma2 = xywha_to_gaussian(target)

        bd      = bhattacharyya_distance(mu1, sigma1, mu2, sigma2)
        prob_iou = torch.exp(-bd).clamp(min=1e-7)
        loss     = 1.0 - prob_iou

        if weight is not None:
            loss = loss * weight

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        return loss
