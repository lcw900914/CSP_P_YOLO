"""
CSPPartial-YOLO 評估腳本
計算：mAP（旋轉框 IoU@0.5）、FLOPs、Latency
"""

import sys
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from models.csp_partial_yolo import CSPPartialYOLO
from datasets.dota_dataset import build_dataloader


CLASSES = ['plane', 'large-vehicle', 'small-vehicle', 'ship']


# ─────────────────────────────────────────
# 旋轉框 IoU（基於 Shapely）
# ─────────────────────────────────────────

def rbox_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    計算兩個旋轉框的 IoU。
    box: (cx, cy, w, h, angle_rad)
    使用 Shapely Polygon 近似計算。
    """
    from shapely.geometry import Polygon
    import math

    def rbox_to_poly(b):
        cx, cy, w, h, a = b
        cos_a, sin_a = math.cos(a), math.sin(a)
        dx1, dy1 = w/2 * cos_a, w/2 * sin_a
        dx2, dy2 = h/2 * (-sin_a), h/2 * cos_a
        pts = [
            (cx - dx1 - dx2, cy - dy1 - dy2),
            (cx + dx1 - dx2, cy + dy1 - dy2),
            (cx + dx1 + dx2, cy + dy1 + dy2),
            (cx - dx1 + dx2, cy - dy1 + dy2),
        ]
        return Polygon(pts)

    p1 = rbox_to_poly(box1)
    p2 = rbox_to_poly(box2)
    if not p1.is_valid or not p2.is_valid:
        return 0.0
    inter = p1.intersection(p2).area
    union = p1.area + p2.area - inter
    return inter / (union + 1e-7)


def rbox_iou_batch(pred_boxes: np.ndarray,
                   gt_boxes: np.ndarray) -> np.ndarray:
    """
    (Np, 5) × (Ng, 5) → (Np, Ng) IoU 矩陣
    """
    Np, Ng = len(pred_boxes), len(gt_boxes)
    iou_mat = np.zeros((Np, Ng))
    for i in range(Np):
        for j in range(Ng):
            iou_mat[i, j] = rbox_iou(pred_boxes[i], gt_boxes[j])
    return iou_mat


# ─────────────────────────────────────────
# NMS（旋轉框）
# ─────────────────────────────────────────

def nms_rotated(boxes: np.ndarray, scores: np.ndarray,
                iou_thr: float = 0.1) -> np.ndarray:
    """
    旋轉框 NMS，返回保留的 index。
    使用 mmcv 的 nms_rotated 如果可用，否則 fallback 到 Shapely。
    """
    try:
        import torch
        from mmcv.ops import nms_rotated as mmcv_nms
        boxes_t  = torch.from_numpy(boxes).float()
        scores_t = torch.from_numpy(scores).float()
        # mmcv nms_rotated 期望 (cx,cy,w,h,angle_deg) 或 (cx,cy,w,h,angle_rad)
        # 使用角度弧度版本
        _, keep = mmcv_nms(boxes_t, scores_t, iou_thr)
        return keep.numpy()
    except Exception:
        # Shapely fallback（慢，但正確）
        order  = scores.argsort()[::-1]
        keep   = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            ious = np.array([rbox_iou(boxes[i], boxes[j]) for j in order[1:]])
            order = order[1:][ious < iou_thr]
        return np.array(keep)


# ─────────────────────────────────────────
# mAP 計算
# ─────────────────────────────────────────

def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute AP using smooth interpolation (PASCAL VOC 2010+, integral under PR curve)"""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_map(all_preds: dict, all_gts: dict,
                iou_thr: float = 0.5) -> dict:
    """
    all_preds[cls_id] = list of (img_id, score, box_5d)
    all_gts[cls_id]   = dict of img_id → list of box_5d
    """
    # 檢查 mmcv 是否可用（只測試一次）
    try:
        from mmcv.ops import box_iou_rotated as _mmcv_iou
        def _iou_matrix(pred_boxes, gt_boxes):
            p = torch.tensor(np.array(pred_boxes, dtype=np.float32))
            g = torch.tensor(np.array(gt_boxes,   dtype=np.float32))
            return _mmcv_iou(p, g).numpy()
        print('  [mAP] 使用 mmcv box_iou_rotated')
    except Exception:
        def _iou_matrix(pred_boxes, gt_boxes):
            return rbox_iou_batch(np.array(pred_boxes, dtype=np.float32),
                                  np.array(gt_boxes,   dtype=np.float32))
        print('  [mAP] 使用 Shapely fallback（較慢）')

    aps = {}
    for cls_id in range(len(CLASSES)):
        preds = sorted(all_preds.get(cls_id, []),
                       key=lambda x: -x[1])  # 按 score 降序
        gts   = all_gts.get(cls_id, {})
        matched = {img_id: [False] * len(boxes)
                   for img_id, boxes in gts.items()}
        tp_list, fp_list = [], []
        total_gt = sum(len(v) for v in gts.values())
        if total_gt == 0:
            aps[cls_id] = 0.0
            continue

        # ── 預先批次計算每張圖的 IoU 矩陣 ──────────────────
        # img_pred_idx[img_id] = 該圖在 preds 中的 index 列表（按 score 降序）
        img_pred_idx = defaultdict(list)
        for i, (img_id, score, pred_box) in enumerate(preds):
            img_pred_idx[img_id].append(i)

        iou_matrices = {}   # img_id → (M_pred × N_gt) numpy array
        for img_id, indices in img_pred_idx.items():
            gt_boxes = gts.get(img_id, [])
            if not gt_boxes:
                continue
            pred_boxes_arr = [preds[i][2] for i in indices]
            iou_matrices[img_id] = _iou_matrix(pred_boxes_arr, gt_boxes)

        # ── 按全局 score 降序走訪，查快取 IoU ───────────────
        img_cursor = defaultdict(int)   # 每張圖已查了幾行
        for img_id, score, pred_box in preds:
            gt_boxes = gts.get(img_id, [])
            if not gt_boxes:
                fp_list.append(1); tp_list.append(0)
                continue
            row  = img_cursor[img_id]
            ious = iou_matrices[img_id][row]
            img_cursor[img_id] += 1

            best_iou_idx = ious.argmax()
            best_iou     = ious[best_iou_idx]
            if best_iou >= iou_thr and not matched[img_id][best_iou_idx]:
                tp_list.append(1); fp_list.append(0)
                matched[img_id][best_iou_idx] = True
            else:
                tp_list.append(0); fp_list.append(1)

        tp = np.cumsum(tp_list)
        fp = np.cumsum(fp_list)
        rec = tp / total_gt
        pre = tp / (tp + fp + 1e-7)
        aps[cls_id] = compute_ap(rec, pre)
        print(f'  [{CLASSES[cls_id]}] AP={aps[cls_id]*100:.2f}%')

    mAP = np.mean(list(aps.values()))
    return {'mAP': mAP, 'AP_per_class': {CLASSES[k]: v for k, v in aps.items()}}


# ─────────────────────────────────────────
# 主評估流程
# ─────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device,
             score_thr: float = 0.05,
             nms_thr: float   = 0.1):
    model.eval()
    all_preds = defaultdict(list)
    all_gts   = defaultdict(dict)

    # 診斷：確認 mmcv nms_rotated 是否可用
    try:
        from mmcv.ops import nms_rotated as _test_nms
        import torch as _tt
        _tb = _tt.tensor([[100,100,50,30,0.1]], dtype=_tt.float32)
        _ts = _tt.tensor([0.9], dtype=_tt.float32)
        _test_nms(_tb, _ts, 0.5)
        print('  [NMS] mmcv nms_rotated OK')
    except Exception as e:
        print(f'  [NMS] mmcv 不可用，使用 Shapely fallback: {e}')

    total_batches = len(loader)
    for batch_idx, (imgs, gt_bboxes, gt_labels, stems) in enumerate(loader):
        imgs = imgs.to(device)
        boxes_b, scores_b, _ = model(imgs)

        for b in range(len(stems)):
            img_id  = stems[b]
            boxes   = boxes_b[b].cpu().numpy()    # N, 5
            scores  = scores_b[b].cpu().numpy()   # N, C

            # 收集 GT
            gt_box = gt_bboxes[b].numpy()
            gt_lbl = gt_labels[b].numpy()
            for cls_id in range(len(CLASSES)):
                mask = gt_lbl == cls_id
                all_gts[cls_id][img_id] = gt_box[mask].tolist()

            # 每類別做 NMS
            for cls_id in range(len(CLASSES)):
                cls_scores = scores[:, cls_id]
                pos = cls_scores > score_thr
                if pos.sum() == 0:
                    continue
                keep = nms_rotated(boxes[pos], cls_scores[pos], nms_thr)
                keep = np.asarray(keep).flatten().astype(np.int64)
                for k in keep:
                    b_np = boxes[pos][k]
                    s    = cls_scores[pos][k]
                    all_preds[cls_id].append((img_id, float(s), b_np.tolist()))

    return compute_map(all_preds, all_gts)


def compute_flops_latency(model, device, input_size=(1, 3, 1024, 1024),
                          n_warmup=10, n_runs=100):
    """計算 FLOPs 與推論延遲"""
    model.eval()
    x = torch.randn(*input_size).to(device)

    # FLOPs（使用 torch.profiler 或 thop）
    try:
        from thop import profile
        flops, params = profile(model, inputs=(x,), verbose=False)
        gflops = flops / 1e9
    except ImportError:
        gflops = None
        print('  thop 未安裝，跳過 FLOPs 計算（pip install thop）')

    # Latency（GPU 上量測）
    if device.type == 'cuda':
        torch.cuda.synchronize()
        for _ in range(n_warmup):
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        latency = np.mean(times)
    else:
        latency = None

    return gflops, latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_dir',  default='D:/cspyolo/data/dota/val')
    parser.add_argument('--weights',  default='D:/cspyolo/checkpoints/best_model.pt')
    parser.add_argument('--batch',    type=int, default=4)
    parser.add_argument('--workers',  type=int, default=4)
    parser.add_argument('--score_thr',type=float, default=0.05)
    parser.add_argument('--nms_thr',  type=float, default=0.1)
    parser.add_argument('--dota_only_val', action='store_true',
                        help='排除 soda_ 前綴圖片，只用 DOTA 原始圖')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CSPPartialYOLO(num_classes=4).to(device)
    ckpt  = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f'Loaded: {args.weights}')

    val_loader = build_dataloader(
        args.val_dir, batch_size=args.batch,
        augment=False, num_workers=args.workers,
        exclude_prefix='soda_' if args.dota_only_val else '',
    )

    # ── mAP ──────────────────────────────────────────────────
    print('Computing mAP ...')
    results = evaluate(model, val_loader, device,
                       args.score_thr, args.nms_thr)
    print(f"\nmAP@0.5 = {results['mAP']*100:.2f}%")
    for cls_name, ap in results['AP_per_class'].items():
        print(f"  {cls_name:20s}: {ap*100:.2f}%")

    # ── FLOPs & Latency ──────────────────────────────────────
    print('\nComputing FLOPs & Latency ...')
    gflops, latency = compute_flops_latency(model, device)
    if gflops is not None:
        print(f'FLOPs:   {gflops:.1f}G  (論文目標: 16.2G)')
    if latency is not None:
        print(f'Latency: {latency:.1f}ms  (論文目標: 23ms)')


if __name__ == '__main__':
    main()
