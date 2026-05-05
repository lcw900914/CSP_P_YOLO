"""
Distribution Focal Loss (DFL)
論文：Generalized Focal Loss (Li et al. 2020)
用於 CSPPartial-YOLO 框回歸的分佈學習（權重 0.05）

將連續回歸目標 t 表示為兩個相鄰離散 bin 的線性組合。
DFL = -(⌈t⌉ - t) * log(P[⌊t⌋]) - (t - ⌊t⌋) * log(P[⌈t⌉])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionFocalLoss(nn.Module):
    def __init__(self, reg_max: int = 16, reduction: str = 'sum'):
        super().__init__()
        self.reg_max   = reg_max
        self.reduction = reduction

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor,
                weight: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred:   (N, 4*(reg_max+1)) 原始 logits（4方向 × 分佈bins）
            target: (N, 4) 連續目標值，已除以 stride，範圍 [0, reg_max]
            weight: (N,) 樣本權重（可選）
        Returns:
            scalar loss
        """
        N    = pred.shape[0]
        dist = pred.reshape(N, 4, self.reg_max + 1)   # (N, 4, bins)
        t    = target.clamp(0, self.reg_max)           # (N, 4)

        t_low  = t.long().clamp(0, self.reg_max - 1)  # ⌊t⌋
        t_high = (t_low + 1).clamp(0, self.reg_max)   # ⌈t⌉
        w_high = t - t_low.float()                     # t - ⌊t⌋
        w_low  = 1.0 - w_high                          # ⌈t⌉ - t

        log_prob = F.log_softmax(dist, dim=-1)         # (N, 4, bins)

        # 取出對應 bin 的 log 機率
        loss = -(w_low  * log_prob.gather(-1, t_low.unsqueeze(-1)).squeeze(-1)
               + w_high * log_prob.gather(-1, t_high.unsqueeze(-1)).squeeze(-1))
        # loss: (N, 4) → mean over 4 directions
        loss = loss.mean(dim=-1)   # (N,)

        if weight is not None:
            loss = loss * weight

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        return loss
