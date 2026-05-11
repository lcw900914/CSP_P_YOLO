"""
DOTA Dataset (PyTorch)
讀取前處理後的切片資料。

標註格式（每行）：cx cy w h angle_rad class_id

v1.2 增強：
  - 30° / 60° 額外隨機旋轉（任意角度補充）
  - Mosaic（4 圖拼接，50% 機率）
  - Scale Jitter + Random Crop（隨機縮放 + 裁切）
"""

import os
import math
import random
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


CLASSES = ['plane', 'large-vehicle', 'small-vehicle', 'ship']


class DOTADataset(Dataset):
    def __init__(self, data_dir: str, img_size: int = 1024,
                 augment: bool = True):
        self.img_dir  = Path(data_dir) / 'images'
        self.lbl_dir  = Path(data_dir) / 'labels'
        self.img_size = img_size
        self.augment  = augment

        self.samples = sorted([
            p.stem for p in self.lbl_dir.glob('*.txt')
            if (self.img_dir / f'{p.stem}.jpg').exists()
        ])
        print(f'[DOTADataset] {data_dir}: {len(self.samples)} samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self.augment and random.random() < 0.5:
            img, boxes, labels = self._mosaic(idx)
        else:
            img, boxes, labels = self._load_sample(idx)

        if self.augment:
            img, boxes, labels = self._augment(img, boxes, labels)

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t,
                             mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
        return img_t, boxes, labels, self.samples[idx]

    # ── 讀取單張 ────────────────────────────────────────────────────
    def _load_sample(self, idx: int):
        stem   = self.samples[idx]
        img    = Image.open(self.img_dir / f'{stem}.jpg').convert('RGB')
        boxes, labels = self._load_label(stem)
        return img, boxes, labels

    def _load_label(self, stem: str):
        path  = self.lbl_dir / f'{stem}.txt'
        boxes, labels = [], []
        with open(path) as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                if len(vals) < 6:
                    continue
                cx, cy, w, h, angle, cls_id = vals
                boxes.append([cx, cy, w, h, angle])
                labels.append(int(cls_id))
        if boxes:
            return (torch.tensor(boxes,  dtype=torch.float32),
                    torch.tensor(labels, dtype=torch.long))
        return (torch.zeros((0, 5), dtype=torch.float32),
                torch.zeros((0,),   dtype=torch.long))

    # ── Mosaic（4 圖拼接）───────────────────────────────────────────
    def _mosaic(self, idx: int):
        S  = self.img_size
        HS = S // 2

        indices = [idx] + random.choices(range(len(self.samples)), k=3)
        canvas  = Image.new('RGB', (S, S), (114, 114, 114))
        all_boxes, all_labels = [], []

        # (x_offset, y_offset) for top-left, top-right, bottom-left, bottom-right
        offsets = [(0, 0), (HS, 0), (0, HS), (HS, HS)]

        for sample_idx, (ox, oy) in zip(indices, offsets):
            img, boxes, labels = self._load_sample(sample_idx)
            img_r = img.resize((HS, HS), Image.BILINEAR)
            canvas.paste(img_r, (ox, oy))

            if boxes.shape[0] > 0:
                scale = HS / S
                b = boxes.clone()
                b[:, 0] = b[:, 0] * scale + ox   # cx
                b[:, 1] = b[:, 1] * scale + oy   # cy
                b[:, 2] = b[:, 2] * scale         # w
                b[:, 3] = b[:, 3] * scale         # h
                # angle 不變

                mask = ((b[:, 0] >= 0) & (b[:, 0] < S) &
                        (b[:, 1] >= 0) & (b[:, 1] < S) &
                        (b[:, 2] > 1)  & (b[:, 3] > 1))
                if mask.any():
                    all_boxes.append(b[mask])
                    all_labels.append(labels[mask])

        if all_boxes:
            return (canvas,
                    torch.cat(all_boxes,  dim=0),
                    torch.cat(all_labels, dim=0))
        return (canvas,
                torch.zeros((0, 5), dtype=torch.float32),
                torch.zeros((0,),   dtype=torch.long))

    # ── 資料增強主流程 ──────────────────────────────────────────────
    def _augment(self, img: Image.Image,
                 boxes: torch.Tensor,
                 labels: torch.Tensor):
        S   = self.img_size
        HPI = math.pi / 2

        def _norm(a: torch.Tensor) -> torch.Tensor:
            """將角度規範至 LE90：(-π/2, π/2]"""
            a[a <= -HPI] += math.pi
            a[a >   HPI] -= math.pi
            return a

        def _rotate_boxes(boxes, deg):
            """
            PIL rotate(deg) = CCW by deg（螢幕座標 y 向下）
            中心座標轉換：
              dx' = dx·cos(α) + dy·sin(α)
              dy' = −dx·sin(α) + dy·cos(α)
            角度轉換：angle' = _norm(angle + α_rad)
            """
            if boxes.shape[0] == 0:
                return boxes
            rad   = math.radians(deg)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            dx = boxes[:, 0] - S / 2
            dy = boxes[:, 1] - S / 2
            boxes[:, 0] = dx * cos_a + dy * sin_a + S / 2
            boxes[:, 1] = -dx * sin_a + dy * cos_a + S / 2
            boxes[:, 4] = _norm(boxes[:, 4] + rad)
            return boxes

        def _filter_boxes(boxes, labels, margin=0):
            """移除中心點超出影像範圍的框"""
            if boxes.shape[0] == 0:
                return boxes, labels
            mask = ((boxes[:, 0] >= margin) & (boxes[:, 0] < S - margin) &
                    (boxes[:, 1] >= margin) & (boxes[:, 1] < S - margin))
            return boxes[mask], labels[mask]

        # ── 水平翻轉 ─────────────────────────────────────────────────
        if random.random() > 0.5:
            img = TF.hflip(img)
            if boxes.shape[0] > 0:
                boxes[:, 0] = S - boxes[:, 0]
                boxes[:, 4] = _norm(-boxes[:, 4])

        # ── 垂直翻轉 ─────────────────────────────────────────────────
        if random.random() > 0.5:
            img = TF.vflip(img)
            if boxes.shape[0] > 0:
                boxes[:, 1] = S - boxes[:, 1]
                boxes[:, 4] = _norm(-boxes[:, 4])

        # ── 90° / 180° / 270° 旋轉 ──────────────────────────────────
        k = random.randint(0, 3)
        if k > 0:
            if boxes.shape[0] > 0:
                cx, cy = boxes[:, 0].clone(), boxes[:, 1].clone()
                if k == 1:      # 90° CW
                    boxes[:, 0] = S - cy
                    boxes[:, 1] = cx
                    boxes[:, 4] = _norm(boxes[:, 4] - HPI)
                elif k == 2:    # 180°
                    boxes[:, 0] = S - cx
                    boxes[:, 1] = S - cy
                else:           # 270° CW
                    boxes[:, 0] = cy
                    boxes[:, 1] = S - cx
                    boxes[:, 4] = _norm(boxes[:, 4] + HPI)
            pil_deg = {1: -90, 2: 180, 3: 90}[k]
            img = img.rotate(pil_deg, expand=False)

        # ── 30° / 60° 額外隨機旋轉（論文增強）──────────────────────
        if random.random() > 0.5:
            deg = random.choice([30, 60, -30, -60])
            boxes = _rotate_boxes(boxes, deg)
            img   = img.rotate(deg, expand=False)
            boxes, labels = _filter_boxes(boxes, labels)

        # ── Scale Jitter + Random Crop ───────────────────────────────
        if random.random() > 0.5:
            scale = random.uniform(0.6, 1.4)
            new_size = int(S * scale)
            img = img.resize((new_size, new_size), Image.BILINEAR)

            if scale >= 1.0:
                # 放大 → 隨機裁切回 S×S
                x0 = random.randint(0, new_size - S)
                y0 = random.randint(0, new_size - S)
                img = img.crop((x0, y0, x0 + S, y0 + S))
                if boxes.shape[0] > 0:
                    boxes[:, 0] = boxes[:, 0] * scale - x0
                    boxes[:, 1] = boxes[:, 1] * scale - y0
                    boxes[:, 2] = boxes[:, 2] * scale
                    boxes[:, 3] = boxes[:, 3] * scale
            else:
                # 縮小 → 貼到黑色 S×S 畫布（隨機位置）
                x0 = random.randint(0, S - new_size)
                y0 = random.randint(0, S - new_size)
                canvas = Image.new('RGB', (S, S), (114, 114, 114))
                canvas.paste(img, (x0, y0))
                img = canvas
                if boxes.shape[0] > 0:
                    boxes[:, 0] = boxes[:, 0] * scale + x0
                    boxes[:, 1] = boxes[:, 1] * scale + y0
                    boxes[:, 2] = boxes[:, 2] * scale
                    boxes[:, 3] = boxes[:, 3] * scale

            boxes, labels = _filter_boxes(boxes, labels)

        # ── HSV 色彩抖動 ─────────────────────────────────────────────
        if random.random() > 0.5:
            img = TF.adjust_brightness(img, random.uniform(0.6, 1.4))
        if random.random() > 0.5:
            img = TF.adjust_contrast(img, random.uniform(0.6, 1.4))
        if random.random() > 0.5:
            img = TF.adjust_saturation(img, random.uniform(0.6, 1.4))
        if random.random() > 0.5:
            img = TF.adjust_hue(img, random.uniform(-0.1, 0.1))

        return img, boxes, labels


def collate_fn(batch):
    imgs, boxes_list, labels_list, stems = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, list(boxes_list), list(labels_list), list(stems)


def build_dataloader(data_dir: str, batch_size: int = 6,
                     augment: bool = True, num_workers: int = 4,
                     img_size: int = 1024) -> DataLoader:
    dataset = DOTADataset(data_dir, img_size=img_size, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=augment,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=augment,
    )
