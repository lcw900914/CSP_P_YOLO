"""
SODA-A → DOTA 格式預處理
- 讀取 SODA-A JSON 標註（4 角點 polygon）
- 只保留 4 類：airplane→plane(0), small-vehicle(2), large-vehicle(1), ship(3)
- 全圖切成 1024×1024 patch（stride=824, overlap=200）
- 轉換 OBB：polygon → (cx, cy, w, h, angle_rad) LE90
- 輸出到 D:/cspyolo/data/dota/train（與 DOTA 合併）
"""

import json, math, cv2
import numpy as np
from pathlib import Path
from PIL import Image

# ── 路徑設定 ────────────────────────────────────────────────────────
SODA_ROOT  = Path('D:/cspyolo/data/soda_a')
DOTA_TRAIN = Path('D:/cspyolo/data/dota/train')
PATCH_SIZE = 1024
STRIDE     = 824   # overlap = 200px
SPLIT      = 'train'   # 只合併 train split

# ── 類別對照 ────────────────────────────────────────────────────────
# SODA category_id → 我們的 class_id
# airplane=0, helicopter=1, small-vehicle=2, large-vehicle=3,
# ship=4, container=5, storage-tank=6, swimming-pool=7, windmill=8, ignore=9
CLASS_MAP = {
    0: 0,   # airplane  → plane
    2: 2,   # small-vehicle → small-vehicle
    3: 1,   # large-vehicle → large-vehicle
    4: 3,   # ship      → ship
}

# ── 工具函式 ────────────────────────────────────────────────────────
def poly_to_obb_le90(poly):
    """
    poly: [x1,y1,x2,y2,x3,y3,x4,y4]
    回傳 (cx, cy, w, h, angle_rad) LE90 convention: angle ∈ (-π/2, π/2]
    使用 cv2.minAreaRect，它回傳 angle ∈ [-90, 0) 度
    """
    pts = np.array(poly, dtype=np.float32).reshape(4, 2)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle_deg = rect

    # cv2 convention: angle in [-90, 0), width is the "horizontal-ish" side
    # Convert to LE90: angle in (-90°, 0°] → (-π/2, 0] ⊂ (-π/2, π/2]
    angle_rad = math.radians(angle_deg)

    # 確保在 LE90 範圍
    while angle_rad <= -math.pi / 2:
        angle_rad += math.pi
        w, h = h, w
    while angle_rad > math.pi / 2:
        angle_rad -= math.pi
        w, h = h, w

    return cx, cy, w, h, angle_rad


def crop_boxes(boxes, x0, y0, patch_size):
    """
    保留中心點在 patch 內的框，並轉換座標到 patch 座標系
    boxes: list of (cx, cy, w, h, angle_rad, cls_id)
    """
    result = []
    x1, y1 = x0, y0
    x2, y2 = x0 + patch_size, y0 + patch_size
    for cx, cy, w, h, angle, cls_id in boxes:
        if x1 <= cx < x2 and y1 <= cy < y2:
            result.append((cx - x0, cy - y0, w, h, angle, cls_id))
    return result


# ── 主程式 ─────────────────────────────────────────────────────────
ann_dir = SODA_ROOT / 'Annotations' / SPLIT
img_dir = SODA_ROOT / 'Images'

out_img = DOTA_TRAIN / 'images'
out_lbl = DOTA_TRAIN / 'labels'
out_img.mkdir(parents=True, exist_ok=True)
out_lbl.mkdir(parents=True, exist_ok=True)

json_files = sorted(ann_dir.glob('*.json'))
print(f'處理 {len(json_files)} 張 SODA-A {SPLIT} 影像 ...')

total_patches = 0
total_boxes   = 0
skipped_imgs  = 0

for jf in json_files:
    d    = json.load(open(jf))
    meta = d['images']  # dict
    img_file = img_dir / meta['file_name']

    if not img_file.exists():
        print(f'  [WARN] 找不到 {img_file}，跳過')
        skipped_imgs += 1
        continue

    W, H = meta['width'], meta['height']

    # 解析 annotations：只取 4 個目標類別
    boxes_full = []
    for ann in d['annotations']:
        cid = ann['category_id']
        if cid not in CLASS_MAP:
            continue
        poly = ann['poly']
        try:
            cx, cy, w, h, angle = poly_to_obb_le90(poly)
        except Exception as e:
            continue
        if w < 2 or h < 2:
            continue
        boxes_full.append((cx, cy, w, h, angle, CLASS_MAP[cid]))

    if not boxes_full:
        continue  # 此圖無目標類別，跳過

    # 讀取圖片（用 PIL，避免大圖 OOM）
    img = Image.open(img_file).convert('RGB')
    img_np = np.array(img)

    # 切 patch
    stem = jf.stem   # e.g. "00001"
    xs = list(range(0, W - PATCH_SIZE, STRIDE)) + [max(0, W - PATCH_SIZE)]
    ys = list(range(0, H - PATCH_SIZE, STRIDE)) + [max(0, H - PATCH_SIZE)]
    xs = sorted(set(xs))
    ys = sorted(set(ys))

    for y0 in ys:
        for x0 in xs:
            patch_boxes = crop_boxes(boxes_full, x0, y0, PATCH_SIZE)
            if not patch_boxes:
                continue   # 無目標的 patch 不儲存（節省儲存空間）

            # 裁切影像
            patch = img_np[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]
            if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
                # 邊緣補黑
                p = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
                p[:patch.shape[0], :patch.shape[1]] = patch
                patch = p

            patch_name = f'soda_{stem}_{x0}_{y0}'
            Image.fromarray(patch).save(out_img / f'{patch_name}.jpg', quality=95)

            # 寫標註
            with open(out_lbl / f'{patch_name}.txt', 'w') as fw:
                for cx, cy, w, h, angle, cls_id in patch_boxes:
                    fw.write(f'{cx:.4f} {cy:.4f} {w:.4f} {h:.4f} {angle:.6f} {cls_id}\n')

            total_patches += 1
            total_boxes   += len(patch_boxes)

    if (json_files.index(jf) + 1) % 100 == 0:
        print(f'  [{json_files.index(jf)+1}/{len(json_files)}] patches={total_patches} boxes={total_boxes}')

print(f'\n完成！')
print(f'  Patches 生成：{total_patches}')
print(f'  Boxes 總計：{total_boxes}')
print(f'  跳過影像：{skipped_imgs}')
print(f'  輸出目錄：{DOTA_TRAIN}')
