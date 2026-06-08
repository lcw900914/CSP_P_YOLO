"""
整個模型組裝在一起的地方。

三大塊串起來：
  backbone(CSPPartialNet) 抽特徵 → neck(CSPPartialFPN) 多尺度融合 → head(PPYOLOERHead) 出框
訓練時 forward 直接回傳 loss，推論時回傳解碼好的旋轉框。
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from .backbone.csp_partial_net import CSPPartialNet
from .neck.csp_partial_fpn import CSPPartialFPN
from .head.ppyoloe_r_head import PPYOLOERHead


class CSPPartialYOLO(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        reg_max:     int = 16,
        strides:     Tuple[int, ...] = (8, 16, 32),
    ):
        super().__init__()
        self.backbone = CSPPartialNet()
        self.neck     = CSPPartialFPN(in_channels=(128, 256, 512))
        self.head     = PPYOLOERHead(
            in_channels=(128, 256, 512),
            num_classes=num_classes,
            reg_max=reg_max,
            strides=strides,
        )

    def forward(self, x: torch.Tensor,
                gt_bboxes: Optional[List[torch.Tensor]] = None,
                gt_labels: Optional[List[torch.Tensor]] = None):
        # 先過 backbone 抽三層特徵，再丟進 neck 融合
        p3, p4, p5      = self.backbone(x)
        p3, p4, p5      = self.neck(p3, p4, p5)
        feats           = (p3, p4, p5)
        feat_shapes     = [(p.shape[2], p.shape[3]) for p in feats]

        # head 吐出三個分支的原始輸出
        cls_list, reg_list, ang_list = self.head(feats)

        if self.training and gt_bboxes is not None:
            return self.head.loss(
                cls_list, reg_list, ang_list,
                gt_bboxes, gt_labels, feat_shapes
            )
        else:
            return self.head.decode(
                cls_list, reg_list, ang_list, feat_shapes
            )
