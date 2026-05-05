"""
CSPPartial-YOLO：完整模型
Backbone: CSPPartialNet
Neck:     CSPPartialFPN
Head:     PPYOLOERHead
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
        # Backbone + Neck
        p3, p4, p5      = self.backbone(x)
        p3, p4, p5      = self.neck(p3, p4, p5)
        feats           = (p3, p4, p5)
        feat_shapes     = [(p.shape[2], p.shape[3]) for p in feats]

        # Head forward
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
