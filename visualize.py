"""
視覺化推論結果：畫出旋轉框 + 類別標籤
用法：python visualize.py --weights D:/cspyolo/checkpoints_v2/best_model.pt
      --val_dir D:/cspyolo/data/dota/val --n 12 --out D:/cspyolo/vis
"""
import sys, argparse, math, random
import numpy as np
import torch
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.csp_partial_yolo import CSPPartialYOLO
from datasets.dota_dataset import DOTADataset, CLASSES
import torchvision.transforms.functional as TF

COLORS = {
    'plane':          (255,  80,  80),
    'large-vehicle':  ( 80, 200,  80),
    'small-vehicle':  ( 80, 150, 255),
    'ship':           (255, 200,  50),
}


def rbox_to_corners(cx, cy, w, h, angle):
    """旋轉框中心+尺寸+角度 → 4個頂點座標"""
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    dx1, dy1 =  w / 2 * cos_a,  w / 2 * sin_a
    dx2, dy2 = -h / 2 * sin_a,  h / 2 * cos_a
    pts = np.array([
        [cx - dx1 - dx2, cy - dy1 - dy2],
        [cx + dx1 - dx2, cy + dy1 - dy2],
        [cx + dx1 + dx2, cy + dy1 + dy2],
        [cx - dx1 + dx2, cy - dy1 + dy2],
    ], dtype=np.float32)
    return pts.reshape((-1, 1, 2)).astype(np.int32)


def draw_rbox(img, cx, cy, w, h, angle, label, score, color):
    pts = rbox_to_corners(cx, cy, w, h, angle)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)


def unnormalize(img_t):
    """Tensor → numpy BGR"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = (img_t.cpu() * std + mean).clamp(0, 1)
    img  = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',   default='D:/cspyolo/checkpoints_v2/best_model.pt')
    parser.add_argument('--val_dir',   default='D:/cspyolo/data/dota/val')
    parser.add_argument('--out',       default='D:/cspyolo/vis')
    parser.add_argument('--n',         type=int,   default=12,   help='輸出幾張圖')
    parser.add_argument('--score_thr', type=float, default=0.25)
    parser.add_argument('--nms_thr',   type=float, default=0.1)
    parser.add_argument('--seed',      type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = CSPPartialYOLO(num_classes=4).to(device)
    ckpt   = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded: {args.weights}")

    ds = DOTADataset(args.val_dir, augment=False)
    indices = random.sample(range(len(ds)), min(args.n, len(ds)))

    for rank, idx in enumerate(indices):
        img_t, gt_boxes, gt_labels, stem = ds[idx]
        inp = img_t.unsqueeze(0).to(device)

        boxes_b, scores_b, _ = model(inp)
        boxes  = boxes_b[0].cpu().numpy()   # (N, 5)
        scores = scores_b[0].cpu().numpy()  # (N, C)

        canvas = unnormalize(img_t)

        # ── 畫 GT（白色虛線）────────────────────────────────
        for i in range(gt_boxes.shape[0]):
            cx, cy, w, h, ang = gt_boxes[i].tolist()
            pts = rbox_to_corners(cx, cy, w, h, ang)
            cv2.polylines(canvas, [pts], True, (200, 200, 200), 1)

        # ── NMS + 畫預測框 ──────────────────────────────────
        try:
            from mmcv.ops import nms_rotated
            use_mmcv = True
        except Exception:
            use_mmcv = False

        for cls_id, cls_name in enumerate(CLASSES):
            cls_scores = scores[:, cls_id]
            pos = cls_scores > args.score_thr
            if pos.sum() == 0:
                continue
            b = boxes[pos]
            s = cls_scores[pos]

            if use_mmcv:
                bt = torch.from_numpy(b).float()
                st = torch.from_numpy(s).float()
                _, keep = nms_rotated(bt, st, args.nms_thr)
                keep = keep.numpy()
            else:
                keep = np.arange(len(s))

            color = COLORS[cls_name]
            for k in keep:
                cx, cy, w_, h_, ang = b[k]
                draw_rbox(canvas, cx, cy, w_, h_, ang,
                          cls_name, s[k], color)

        save_path = out_dir / f"{rank:02d}_{stem}.jpg"
        cv2.imwrite(str(save_path), canvas)
        print(f"  [{rank+1}/{len(indices)}] {save_path.name}")

    print(f"\n完成！結果儲存在 {out_dir}")


if __name__ == '__main__':
    main()
