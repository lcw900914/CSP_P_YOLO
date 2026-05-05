"""
DOTA 資料集前處理腳本
論文設定：
  - 選取 4 類：plane, large-vehicle, small-vehicle, ship
  - 切片：1024×1024，overlap=200px
  - Train：6049 張，Test：1718 張

DOTA 標註格式（每行）：
  x1 y1 x2 y2 x3 y3 x4 y4 category difficulty

輸出格式（每個切片對應一個 .txt）：
  cx cy w h angle_rad class_id

使用方式：
  python datasets/dota_preprocess.py \
    --src_img  /path/to/DOTA/train/images \
    --src_ann  /path/to/DOTA/train/labelTxt \
    --dst      D:/cspyolo/data/dota/train \
    --split    train

  python datasets/dota_preprocess.py \
    --src_img  /path/to/DOTA/val/images \
    --src_ann  /path/to/DOTA/val/labelTxt \
    --dst      D:/cspyolo/data/dota/val \
    --split    val
"""

import os
import argparse
import math
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple


# 論文選取的 4 個類別
CLASSES = ['plane', 'large-vehicle', 'small-vehicle', 'ship']
CLS2ID  = {c: i for i, c in enumerate(CLASSES)}


# ─────────────────────────────────────────
# 旋轉框轉換
# ─────────────────────────────────────────

def poly2rbox(poly: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    四點多邊形 (8 個值) → 旋轉框 (cx, cy, w, h, angle_rad)
    angle 範圍：[-π/2, π/2)，le90 格式
    """
    pts = poly.reshape(4, 2)
    # 以最小外接旋轉矩形為基準
    rect = cv2_min_area_rect(pts)
    cx, cy = rect[0]
    w,  h  = rect[1]
    angle  = rect[2]  # degree, OpenCV 格式

    # 轉為 le90 格式：long edge first, angle in [-90, 0)
    if w < h:
        w, h  = h, w
        angle = angle + 90
    angle = angle % 180
    if angle >= 90:
        angle -= 180
    # 轉弧度
    return cx, cy, w, h, math.radians(angle)


def cv2_min_area_rect(pts: np.ndarray):
    """用 numpy 實作 cv2.minAreaRect（避免依賴 cv2）"""
    import cv2
    rect = cv2.minAreaRect(pts.astype(np.float32))
    return rect


# ─────────────────────────────────────────
# 切片工具
# ─────────────────────────────────────────

def get_slice_positions(img_w: int, img_h: int,
                        size: int = 1024,
                        overlap: int = 200) -> List[Tuple[int, int]]:
    """計算所有切片的左上角座標 (x, y)"""
    stride = size - overlap
    positions = []
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            positions.append((x, y))
            x += stride
            if x + size > img_w and x < img_w:
                # 貼齊右邊界
                positions.append((img_w - size, y))
                break
        y += stride
        if y + size > img_h and y < img_h:
            # 貼齊下邊界
            x = 0
            while x < img_w:
                positions.append((x, img_h - size))
                x += stride
                if x + size > img_w and x < img_w:
                    positions.append((img_w - size, img_h - size))
                    break
            break
    # 去重
    return list(dict.fromkeys(positions))


def clip_box_to_patch(cx, cy, w, h, angle_rad,
                      px, py, patch_size=1024,
                      min_ratio=0.5) -> bool:
    """
    判斷旋轉框是否與切片有足夠的重疊。
    使用近似法：計算旋轉框的 AABB 是否與切片有重疊，
    且中心點落在切片內（簡化版，與論文實作可能略有差異）。
    """
    # 轉換為切片座標系
    lcx = cx - px
    lcy = cy - py

    # AABB 半寬/半高
    cos_a = abs(math.cos(angle_rad))
    sin_a = abs(math.sin(angle_rad))
    hw = (w * cos_a + h * sin_a) / 2
    hh = (w * sin_a + h * cos_a) / 2

    # AABB 與切片的交集
    ix1 = max(0, lcx - hw);  ix2 = min(patch_size, lcx + hw)
    iy1 = max(0, lcy - hh);  iy2 = min(patch_size, lcy + hh)
    if ix2 <= ix1 or iy2 <= iy1:
        return False, None

    inter_area = (ix2 - ix1) * (iy2 - iy1)
    box_area   = w * h
    if inter_area / (box_area + 1e-6) < min_ratio:
        return False, None

    return True, (lcx, lcy, w, h, angle_rad)


def parse_annotation(ann_path: str) -> List[dict]:
    """解析 DOTA 標註檔"""
    boxes = []
    with open(ann_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('imagesource') or \
               line.startswith('gsd'):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            coords   = list(map(float, parts[:8]))
            category = parts[8].lower()
            if category not in CLS2ID:
                continue
            poly = np.array(coords)
            try:
                cx, cy, w, h, angle = poly2rbox(poly)
            except Exception:
                continue
            if w <= 0 or h <= 0:
                continue
            boxes.append({
                'poly':     poly,
                'cx': cx, 'cy': cy,
                'w':  w,  'h':  h,
                'angle': angle,
                'cls_id': CLS2ID[category],
            })
    return boxes


# ─────────────────────────────────────────
# 主處理流程
# ─────────────────────────────────────────

def process_split(src_img: str, src_ann: str, dst: str,
                  patch_size: int = 1024, overlap: int = 200):
    """處理單個 split（train 或 val）"""
    dst_img = Path(dst) / 'images'
    dst_lbl = Path(dst) / 'labels'
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    img_files = sorted(Path(src_img).glob('*.png')) + \
                sorted(Path(src_img).glob('*.jpg'))

    total_patches = 0
    for img_path in img_files:
        stem     = img_path.stem
        ann_path = Path(src_ann) / f'{stem}.txt'
        if not ann_path.exists():
            continue

        img  = Image.open(img_path).convert('RGB')
        W, H = img.size
        boxes = parse_annotation(str(ann_path))

        positions = get_slice_positions(W, H, patch_size, overlap)

        for (px, py) in positions:
            # 右下角邊界處理
            x2 = min(px + patch_size, W)
            y2 = min(py + patch_size, H)
            pw = x2 - px
            ph = y2 - py

            patch_boxes = []
            for box in boxes:
                ok, local = clip_box_to_patch(
                    box['cx'], box['cy'], box['w'], box['h'], box['angle'],
                    px, py, patch_size
                )
                if ok:
                    lcx, lcy, w, h, angle = local
                    # 過濾中心不在切片內的框
                    if not (0 <= lcx <= patch_size and 0 <= lcy <= patch_size):
                        continue
                    patch_boxes.append(
                        (lcx, lcy, w, h, angle, box['cls_id'])
                    )

            if not patch_boxes:
                continue

            # 儲存切片圖片
            patch_img  = img.crop((px, py, px + patch_size, py + patch_size))
            patch_name = f'{stem}_{px}_{py}'
            patch_img.save(dst_img / f'{patch_name}.jpg', quality=95)

            # 儲存標註（cx cy w h angle_rad class_id，空格分隔）
            with open(dst_lbl / f'{patch_name}.txt', 'w') as f:
                for (cx, cy, w, h, angle, cls_id) in patch_boxes:
                    f.write(f'{cx:.4f} {cy:.4f} {w:.4f} {h:.4f} '
                            f'{angle:.6f} {cls_id}\n')

            total_patches += 1

    print(f'  生成切片數：{total_patches}')
    return total_patches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_img',  required=True, help='原始圖片目錄')
    parser.add_argument('--src_ann',  required=True, help='原始標註目錄')
    parser.add_argument('--dst',      required=True, help='輸出目錄')
    parser.add_argument('--size',     type=int, default=1024)
    parser.add_argument('--overlap',  type=int, default=200)
    args = parser.parse_args()

    print(f'Processing: {args.src_img} -> {args.dst}')
    n = process_split(args.src_img, args.src_ann, args.dst,
                      args.size, args.overlap)
    print(f'Done. Total patches: {n}')


if __name__ == '__main__':
    main()
