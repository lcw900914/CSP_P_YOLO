"""
CSPPartial-YOLO DOTA-style 評估腳本
與論文相同的評估協議：
  1. 推論每個 1024×1024 patch
  2. 利用 stem 中的 (px, py) 將預測框轉回原始影像座標
  3. 每張原始影像做全局旋轉框 NMS（去除 overlap 區重複預測）
  4. GT 同樣轉回原始座標後做去重 NMS
  5. 計算 smooth integral AP（VOC 2010+，與論文一致）

與 eval.py 的差異：
  eval.py     — per-patch 評估，每個 patch 獨立計算
  eval_dota.py — full-image 評估，還原至原始影像後再算 mAP
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

# 重用 eval.py 的工具函式
from eval import (
    CLASSES,
    rbox_iou,
    rbox_iou_batch,
    nms_rotated,
    compute_ap,
    compute_map,
)


# ─────────────────────────────────────────
# Stem 解析
# ─────────────────────────────────────────

def parse_stem(stem: str):
    """
    Patch 命名格式：{原始影像名}_{px}_{py}
    例如：P0001_0_512 → ('P0001', 0, 512)
    注意：原始影像名本身可能包含 '_'，因此從右側分割兩次。
    """
    parts = stem.rsplit('_', 2)
    if len(parts) == 3:
        try:
            px, py = int(parts[1]), int(parts[2])
            return parts[0], px, py
        except ValueError:
            pass
    return stem, 0, 0


# ─────────────────────────────────────────
# DOTA-style 評估主流程
# ─────────────────────────────────────────

@torch.no_grad()
def evaluate_dota(model, loader, device,
                  score_thr: float = 0.05,
                  nms_thr:   float = 0.1,
                  gt_dedup_thr: float = 0.5):
    """
    Parameters
    ----------
    gt_dedup_thr : GT 去重 IoU 閾值（overlap 區同一物件出現在多個 patch）
    """
    model.eval()

    # 診斷 mmcv NMS
    try:
        from mmcv.ops import nms_rotated as _test_nms
        import torch as _tt
        _tb = _tt.tensor([[100, 100, 50, 30, 0.1]], dtype=_tt.float32)
        _ts = _tt.tensor([0.9], dtype=_tt.float32)
        _test_nms(_tb, _ts, 0.5)
        print('  [NMS] mmcv nms_rotated OK')
    except Exception as e:
        print(f'  [NMS] mmcv 不可用，使用 Shapely fallback: {e}')

    # 以原始影像名為 key，收集全圖座標的預測與 GT
    # img_preds[orig][cls_id] = [(score, box_5d), ...]
    # img_gts  [orig][cls_id] = [box_5d, ...]
    img_preds = defaultdict(lambda: defaultdict(list))
    img_gts   = defaultdict(lambda: defaultdict(list))

    total_batches = len(loader)
    print(f'  [DBG] 開始迭代 DataLoader ({total_batches} batches)...', flush=True)

    for batch_idx, (imgs, gt_bboxes, gt_labels, stems) in enumerate(loader):
        if batch_idx % 50 == 0:
            print(f'  [DBG] batch {batch_idx}/{total_batches}', flush=True)
        imgs = imgs.to(device)

        boxes_b, scores_b, _ = model(imgs)

        for b in range(len(stems)):
            stem = stems[b]
            orig, px, py = parse_stem(stem)

            boxes  = boxes_b[b].cpu().numpy()    # (N, 5)  patch 座標
            scores = scores_b[b].cpu().numpy()   # (N, C)

            # ── GT 轉回全圖座標 ────────────────────────────
            gt_box = gt_bboxes[b].numpy().copy()   # (M, 5)
            gt_lbl = gt_labels[b].numpy()          # (M,)
            if gt_box.shape[0] > 0:
                gt_box[:, 0] += px   # cx
                gt_box[:, 1] += py   # cy
            for cls_id in range(len(CLASSES)):
                mask = gt_lbl == cls_id
                if mask.sum() > 0:
                    img_gts[orig][cls_id].extend(gt_box[mask].tolist())

            # ── 預測框轉回全圖座標（先做 patch NMS 再累積，節省記憶體）──
            boxes_full = boxes.copy()
            boxes_full[:, 0] += px
            boxes_full[:, 1] += py

            for cls_id in range(len(CLASSES)):
                cls_scores = scores[:, cls_id]
                pos = cls_scores > score_thr
                if pos.sum() == 0:
                    continue
                b_pos = boxes_full[pos]
                s_pos = cls_scores[pos]
                keep = nms_rotated(b_pos, s_pos, nms_thr)
                keep = np.asarray(keep).flatten().astype(np.int64)
                for k in keep:
                    img_preds[orig][cls_id].append(
                        (float(s_pos[k]), b_pos[k].tolist())
                    )

    print(f'  [DBG] 推論完成，共 {len(img_preds)} 張原始影像', flush=True)

    # ── 全局 NMS（去除 overlap 區重複預測）─────────────────
    all_preds = defaultdict(list)   # cls_id → [(orig, score, box)]
    for orig, cls_dict in img_preds.items():
        for cls_id, pred_list in cls_dict.items():
            if not pred_list:
                continue
            scores_arr = np.array([p[0] for p in pred_list], dtype=np.float32)
            boxes_arr  = np.array([p[1] for p in pred_list], dtype=np.float32)
            keep = nms_rotated(boxes_arr, scores_arr, nms_thr)
            keep = np.asarray(keep).flatten().astype(np.int64)
            for k in keep:
                all_preds[cls_id].append(
                    (orig, float(scores_arr[k]), boxes_arr[k].tolist())
                )

    # ── GT 去重（同一物件可能出現在多個 patch 的 GT 中）────
    all_gts = defaultdict(dict)     # cls_id → {orig: [box]}
    for orig, cls_dict in img_gts.items():
        for cls_id, gt_list in cls_dict.items():
            if not gt_list:
                continue
            gt_arr = np.array(gt_list, dtype=np.float32)
            # 用高分 dummy score + NMS 去重
            dummy = np.ones(len(gt_arr), dtype=np.float32)
            keep  = nms_rotated(gt_arr, dummy, iou_thr=gt_dedup_thr)
            keep  = np.asarray(keep).flatten().astype(np.int64)
            if cls_id not in all_gts:
                all_gts[cls_id] = {}
            all_gts[cls_id][orig] = gt_arr[keep].tolist()

    total_gt = {cls_id: sum(len(v) for v in d.values())
                for cls_id, d in all_gts.items()}
    print(f'  [DBG] GT 去重後各類數量: '
          + ', '.join(f'{CLASSES[c]}={n}' for c, n in total_gt.items()))

    return compute_map(all_preds, all_gts)


def compute_flops_latency(model, device, input_size=(1, 3, 1024, 1024),
                          n_warmup=10, n_runs=100):
    model.eval()
    x = torch.randn(*input_size).to(device)
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(x,), verbose=False)
        gflops = flops / 1e9
    except ImportError:
        gflops = None
        print('  thop 未安裝，跳過 FLOPs（pip install thop）')

    if device.type == 'cuda':
        torch.cuda.synchronize()
        for _ in range(n_warmup):
            with torch.no_grad():
                model(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        latency = np.mean(times)
    else:
        latency = None

    return gflops, latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_dir',     default='D:/cspyolo/data/dota/val')
    parser.add_argument('--weights',     default='D:/cspyolo/checkpoints_v2/best_model.pt')
    parser.add_argument('--batch',       type=int,   default=4)
    parser.add_argument('--workers',     type=int,   default=4)
    parser.add_argument('--score_thr',   type=float, default=0.05)
    parser.add_argument('--nms_thr',     type=float, default=0.1)
    parser.add_argument('--gt_dedup_thr',type=float, default=0.5,
                        help='GT 去重 IoU 閾值（0=不去重）')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CSPPartialYOLO(num_classes=4).to(device)
    ckpt  = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f'Loaded: {args.weights}')

    val_loader = build_dataloader(
        args.val_dir, batch_size=args.batch,
        augment=False, num_workers=args.workers,
    )

    # ── mAP（DOTA-style）─────────────────────────────────────
    print('\nComputing mAP (DOTA full-image style) ...')
    results = evaluate_dota(model, val_loader, device,
                            args.score_thr, args.nms_thr, args.gt_dedup_thr)
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
