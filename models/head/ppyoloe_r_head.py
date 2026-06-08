"""
旋轉框偵測頭，直接沿用 PP-YOLOE-R 的設計。

每個 FPN 層級各有三個分支，分頭做三件事：
  - 分類：每個位置是哪一類
  - 框回歸：用 DFL 預測框四個邊到中心的距離（每邊一個 reg_max+1 的分布）
  - 角度：直接回歸一個弧度值

每個層級前面都先過一個 stem（兩層 DWConv + BN + SiLU）整理一下特徵。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from ..utils.rotated_box import make_anchors, dist2rbox
from ..losses.varifocal_loss import VarifocalLoss
from ..losses.prob_iou_loss import ProbIoULoss
from ..losses.dfl_loss import DistributionFocalLoss


class DWConvBNSiLU(nn.Module):
    """Depthwise Conv + BN + SiLU"""
    def __init__(self, ch: int, kernel: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, kernel, padding=kernel // 2,
                      groups=ch, bias=False),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PPYOLOERHead(nn.Module):
    def __init__(
        self,
        in_channels:  Tuple[int, ...]  = (128, 256, 512),
        num_classes:  int               = 4,
        reg_max:      int               = 16,
        strides:      Tuple[int, ...]   = (8, 16, 32),
        # Loss weights（依論文）
        loss_cls_w:   float             = 1.0,
        loss_reg_w:   float             = 2.5,
        loss_dfl_w:   float             = 0.05,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max     = reg_max
        self.strides     = strides
        self.loss_cls_w  = loss_cls_w
        self.loss_reg_w  = loss_reg_w
        self.loss_dfl_w  = loss_dfl_w

        # ── 每個 FPN 層級的 stem（各自獨立，不共享）──
        self.cls_stems = nn.ModuleList()
        self.reg_stems = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.ang_preds = nn.ModuleList()

        for in_ch in in_channels:
            self.cls_stems.append(nn.Sequential(
                DWConvBNSiLU(in_ch),
                DWConvBNSiLU(in_ch),
            ))
            self.reg_stems.append(nn.Sequential(
                DWConvBNSiLU(in_ch),
                DWConvBNSiLU(in_ch),
            ))
            self.cls_preds.append(
                nn.Conv2d(in_ch, num_classes, 1)
            )
            self.reg_preds.append(
                nn.Conv2d(in_ch, 4 * (reg_max + 1), 1)
            )
            self.ang_preds.append(
                nn.Conv2d(in_ch, 1, 1)     # 直接預測一個弧度值
            )

        # ── Loss ──
        self.loss_cls = VarifocalLoss(reduction='sum')
        self.loss_reg = ProbIoULoss(reduction='sum')
        self.loss_dfl = DistributionFocalLoss(reg_max=reg_max, reduction='sum')

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # cls bias 初始化（防止訓練初期 loss 爆炸）
        prior_prob = 0.01
        bias_init  = float(-torch.log(torch.tensor((1 - prior_prob) / prior_prob)))
        for layer in self.cls_preds:
            nn.init.constant_(layer.bias, bias_init)

    def forward_single(self, feat: torch.Tensor, idx: int):
        cls_feat = self.cls_stems[idx](feat)
        reg_feat = self.reg_stems[idx](feat)

        cls_logit = self.cls_preds[idx](cls_feat)   # B,C,H,W
        reg_dist  = self.reg_preds[idx](reg_feat)   # B,4*(reg_max+1),H,W
        angle     = self.ang_preds[idx](reg_feat)   # B,1,H,W

        return cls_logit, reg_dist, angle

    def forward(self, feats: Tuple[torch.Tensor, ...]) \
            -> Tuple[List, List, List]:
        cls_list, reg_list, ang_list = [], [], []
        for i, feat in enumerate(feats):
            cls, reg, ang = self.forward_single(feat, i)
            cls_list.append(cls)
            reg_list.append(reg)
            ang_list.append(ang)
        return cls_list, reg_list, ang_list

    # ── 推論解碼 ──────────────────────────────────
    @torch.no_grad()
    def decode(
        self,
        cls_list: List[torch.Tensor],
        reg_list: List[torch.Tensor],
        ang_list: List[torch.Tensor],
        feat_shapes: List[Tuple],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        解碼預測為旋轉框。
        Returns:
            boxes:  (B, N, 5) (cx, cy, w, h, angle)
            scores: (B, N, num_classes)
            anchors:(N, 2)
        """
        all_cls, all_boxes = [], []
        for cls, reg, ang, stride in zip(
                cls_list, reg_list, ang_list, self.strides):
            B, _, H, W = cls.shape
            cls_flat = cls.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
            reg_flat = reg.permute(0, 2, 3, 1).reshape(B, -1, 4 * (self.reg_max + 1))
            ang_flat = ang.permute(0, 2, 3, 1).reshape(B, -1, 1)
            dummy = [torch.zeros(1, 1, H, W, device=cls.device)]
            anch, _ = make_anchors(dummy, [stride])
            boxes = dist2rbox(reg_flat, ang_flat, anch, self.reg_max, stride)
            all_cls.append(cls_flat)
            all_boxes.append(boxes)

        cls_all   = torch.cat(all_cls,   dim=1)   # B, N_total, C
        boxes_all = torch.cat(all_boxes, dim=1)   # B, N_total, 5
        scores = torch.sigmoid(cls_all)

        # anchor_pts for reference
        dummy_feats = [torch.zeros(1, 1, h, w) for h, w in feat_shapes]
        anchor_pts, _ = make_anchors(dummy_feats, self.strides)
        anchor_pts = anchor_pts.to(cls_list[0].device)
        return boxes_all, scores, anchor_pts

    # ── 訓練損失 ──────────────────────────────────
    def loss(
        self,
        cls_list: List[torch.Tensor],
        reg_list: List[torch.Tensor],
        ang_list: List[torch.Tensor],
        gt_bboxes: List[torch.Tensor],   # List of (M_i, 5) per image
        gt_labels: List[torch.Tensor],   # List of (M_i,) per image
        feat_shapes: List[Tuple],
    ) -> dict:
        """
        簡化版 Task-Aligned Label Assignment (AABB) + Loss 計算。
        正樣本條件：anchor point 在 GT 框的 AABB 內。
        """
        from ..utils.rotated_box import xywha_to_gaussian, bhattacharyya_distance

        device = cls_list[0].device
        B      = cls_list[0].shape[0]

        # 生成 anchor 座標
        dummy_feats  = [torch.zeros(1, 1, h, w, device=device)
                        for h, w in feat_shapes]
        anchor_pts, stride_tensor = make_anchors(dummy_feats, self.strides)
        N = anchor_pts.shape[0]  # 總 anchor 數

        # 將各 FPN 輸出 flatten
        def flatten_preds(pred_list, dim_per_anchor):
            flat = []
            for p, stride in zip(pred_list, self.strides):
                _B, C, H, W = p.shape
                flat.append(
                    p.permute(0, 2, 3, 1).reshape(B, -1, dim_per_anchor)
                )
            return torch.cat(flat, dim=1)  # B, N, D

        cls_pred = flatten_preds(cls_list, self.num_classes)   # B,N,C
        reg_pred = flatten_preds(reg_list, 4*(self.reg_max+1)) # B,N,4*(r+1)
        ang_pred = flatten_preds(ang_list, 1)                  # B,N,1

        loss_cls_total = torch.tensor(0., device=device)
        loss_reg_total = torch.tensor(0., device=device)
        loss_dfl_total = torch.tensor(0., device=device)
        num_pos_total  = 0

        for b in range(B):
            gt_box = gt_bboxes[b].to(device)   # M,5
            gt_lbl = gt_labels[b].to(device)   # M,

            if gt_box.shape[0] == 0:
                # 無 GT：全負樣本
                loss_cls_total += self.loss_cls(
                    cls_pred[b],
                    torch.zeros_like(cls_pred[b])
                ) * self.loss_cls_w
                continue

            # ── AABB label assignment ──────────────────────────────────
            # 計算每個 anchor 到每個 GT 的 AABB overlap（近似旋轉框）
            ap = anchor_pts.unsqueeze(1)                       # N,1,2
            gc = gt_box[:, :2].unsqueeze(0)                    # 1,M,2
            gw = gt_box[:, 2].unsqueeze(0)                     # 1,M
            gh = gt_box[:, 3].unsqueeze(0)                     # 1,M
            dx = (ap[..., 0] - gc[..., 0]).abs()               # N,M
            dy = (ap[..., 1] - gc[..., 1]).abs()               # N,M
            inside = (dx < gw / 2) & (dy < gh / 2)            # N,M

            # 每個 anchor 選最近的 GT（若有多個 GT 覆蓋）
            pos_mask = inside.any(dim=1)                       # N, bool
            pos_idx  = inside.float().argmax(dim=1)            # N, 選哪個 GT

            num_pos = pos_mask.sum().item()
            if num_pos == 0:
                loss_cls_total += self.loss_cls(
                    cls_pred[b],
                    torch.zeros_like(cls_pred[b])
                ) * self.loss_cls_w
                continue

            num_pos_total += num_pos

            # ── 解碼正樣本的預測框 ─────────────────────────────────
            pos_reg    = reg_pred[b][pos_mask]                 # Np, 4*(r+1)
            pos_ang    = ang_pred[b][pos_mask]                 # Np, 1
            pos_anch   = anchor_pts[pos_mask]                  # Np, 2
            pos_stride = stride_tensor[pos_mask]               # Np, 1

            pred_boxes = dist2rbox(
                pos_reg, pos_ang, pos_anch, self.reg_max, pos_stride,
            )                                                  # Np, 5

            # 對應的 GT
            pos_gt_idx = pos_idx[pos_mask]
            tgt_boxes  = gt_box[pos_gt_idx]                   # Np, 5
            tgt_labels = gt_lbl[pos_gt_idx]                   # Np,

            # IoU score 作為分類軟標籤（Bhattacharyya-based ProbIoU）
            mu1, s1 = xywha_to_gaussian(pred_boxes.detach())
            mu2, s2 = xywha_to_gaussian(tgt_boxes)
            bd        = bhattacharyya_distance(mu1, s1, mu2, s2)
            iou_score = torch.exp(-bd).clamp(0, 1)            # Np,

            # 分類目標（one-hot × IoU score）
            cls_target = torch.zeros(N, self.num_classes, device=device)
            cls_target[pos_mask, tgt_labels] = iou_score

            # ── Losses ────────────────────────────────────────────
            loss_cls_total += self.loss_cls(
                cls_pred[b], cls_target
            ) * self.loss_cls_w

            loss_reg_total += self.loss_reg(
                pred_boxes, tgt_boxes, weight=iou_score
            ) * self.loss_reg_w

            # DFL target：各方向距離 / stride → [0, reg_max]
            with torch.no_grad():
                lt = pos_anch - tgt_boxes[:, :2] + \
                     torch.stack([tgt_boxes[:, 2], tgt_boxes[:, 3]], -1) / 2
                rb = tgt_boxes[:, :2] + \
                     torch.stack([tgt_boxes[:, 2], tgt_boxes[:, 3]], -1) / 2 \
                     - pos_anch
                dfl_target = torch.cat([lt, rb], dim=-1) / pos_stride
            loss_dfl_total += self.loss_dfl(
                pos_reg, dfl_target, weight=iou_score
            ) * self.loss_dfl_w

        # 正規化（依 PP-YOLOE-R，除以正樣本數）
        num_pos_total = max(num_pos_total, 1)
        return {
            'loss_cls': loss_cls_total / num_pos_total,
            'loss_reg': loss_reg_total / num_pos_total,
            'loss_dfl': loss_dfl_total / num_pos_total,
            'loss':     (loss_cls_total + loss_reg_total + loss_dfl_total)
                        / num_pos_total,
        }
