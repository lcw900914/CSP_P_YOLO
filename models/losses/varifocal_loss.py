"""
Varifocal Loss
論文：VarifocalNet (Zhang et al. 2021)
用於 CSPPartial-YOLO 分類分支（權重 1.0）

公式：
  正樣本 (q > 0): loss = -q * (q*log(p) + (1-q)*log(1-p))
  負樣本 (q = 0): loss = -α * p^γ * log(1-p)
其中 q 為預測框與 GT 的 IoU（作為軟標籤）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VarifocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0,
                 reduction: str = 'sum'):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, pred_score: torch.Tensor,
                target_score: torch.Tensor,
                weight: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred_score:   (N, num_classes) logits 或 sigmoid 後的預測
            target_score: (N, num_classes) 軟標籤 (IoU score for positives)
            weight:       (N,) 樣本權重（可選）
        """
        pred_sigmoid = torch.sigmoid(pred_score)
        target       = target_score.to(pred_score.dtype)

        # 正樣本：focal weight = q
        # 負樣本：focal weight = α * p^γ
        focal_weight = torch.where(
            target > 0,
            target,
            self.alpha * (pred_sigmoid - target).abs() ** self.gamma,
        )

        bce = F.binary_cross_entropy_with_logits(
            pred_score, target, reduction='none'
        )
        loss = focal_weight * bce

        if weight is not None:
            loss = loss * weight.unsqueeze(-1)

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        return loss
