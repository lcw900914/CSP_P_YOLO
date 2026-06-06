"""
DOTA Dataset (PyTorch)
讀取前處理後的切片資料。

標註格式（每行）：cx cy w h angle_rad class_id

資料增強（對齊論文設定）：
  - 水平翻轉 p=0.5
  - 垂直翻轉 p=0.5
  - 0°/90°/180°/270° 旋轉（等機率）
  - 30°/60° 額外隨機旋轉 p=0.5  ← 論文明確提及
  - HSV Jitter（亮度/對比/飽和/色相，各 p=0.5）
"""

import math
import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset


CLASSES = ['plane', 'large-vehicle', 'small-vehicle', 'ship']


class DOTADataset(Dataset):
    def __init__(self, data_dir: str, img_size: int = 1024,
                 augment: bool = True, exclude_prefix: str = ''):
        self.img_dir  = Path(data_dir) / 'images'
        self.lbl_dir  = Path(data_dir) / 'labels'
        self.img_size = img_size
        self.augment  = augment

        self.samples = sorted([
            p.stem for p in self.lbl_dir.glob('*.txt')
            if (self.img_dir / f'{p.stem}.jpg').exists()
            and (not exclude_prefix or not p.stem.startswith(exclude_prefix))
        ])
        print(f'[DOTADataset] {data_dir}: {len(self.samples)} samples'
              + (f' (excluded prefix="{exclude_prefix}")' if exclude_prefix else ''))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img, boxes, labels = self._load_sample(idx)

        if self.augment:
            img, boxes, labels = self._augment(img, boxes, labels)

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t,
                             mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
        return img_t, boxes, labels, self.samples[idx]

    def _load_sample(self, idx: int):
        stem  = self.samples[idx]
        img   = Image.open(self.img_dir / f'{stem}.jpg').convert('RGB')
        boxes, labels = self._load_label(stem)
        return img, boxes, labels

    def _load_label(self, stem: str):
        path = self.lbl_dir / f'{stem}.txt'
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

    def _mosaic(self, idx: int):
        """4 圖各縮至 512×512，拼成 1024×1024"""
        S = self.img_size
        H = S // 2
        scale = H / S  # 0.5

        indices = [idx] + random.choices(range(len(self.samples)), k=3)
        positions = [(0, 0), (H, 0), (0, H), (H, H)]

        mosaic_img = Image.new('RGB', (S, S), (114, 114, 114))
        all_boxes, all_labels = [], []

        for sample_idx, (x0, y0) in zip(indices, positions):
            img, boxes, labels = self._load_sample(sample_idx)
            img = img.resize((H, H), Image.BILINEAR)
            mosaic_img.paste(img, (x0, y0))

            if boxes.shape[0] > 0:
                b = boxes.clone()
                b[:, 0] = b[:, 0] * scale + x0
                b[:, 1] = b[:, 1] * scale + y0
                b[:, 2] = b[:, 2] * scale
                b[:, 3] = b[:, 3] * scale
                mask = ((b[:, 0] >= 0) & (b[:, 0] < S) &
                        (b[:, 1] >= 0) & (b[:, 1] < S))
                all_boxes.append(b[mask])
                all_labels.append(labels[mask])

        if all_boxes:
            return (mosaic_img,
                    torch.cat(all_boxes, dim=0),
                    torch.cat(all_labels, dim=0))
        return (mosaic_img,
                torch.zeros((0, 5), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long))

    def _augment(self, img: Image.Image,
                 boxes: torch.Tensor,
                 labels: torch.Tensor):
        S   = self.img_size
        HPI = math.pi / 2

        def _norm(a: torch.Tensor) -> torch.Tensor:
            a[a <= -HPI] += math.pi
            a[a >   HPI] -= math.pi
            return a

        # ── 水平翻轉 p=0.5 ──────────────────────────────────────────
        if random.random() > 0.5:
            img = TF.hflip(img)
            if boxes.shape[0] > 0:
                boxes[:, 0] = S - boxes[:, 0]
                boxes[:, 4] = _norm(-boxes[:, 4])

        # ── 垂直翻轉 p=0.5 ──────────────────────────────────────────
        if random.random() > 0.5:
            img = TF.vflip(img)
            if boxes.shape[0] > 0:
                boxes[:, 1] = S - boxes[:, 1]
                boxes[:, 4] = _norm(-boxes[:, 4])

        # ── 90° / 180° / 270° 旋轉（等機率）────────────────────────
        k = random.randint(0, 3)
        if k > 0:
            if boxes.shape[0] > 0:
                cx, cy = boxes[:, 0].clone(), boxes[:, 1].clone()
                if k == 1:
                    boxes[:, 0] = S - cy
                    boxes[:, 1] = cx
                    boxes[:, 4] = _norm(boxes[:, 4] - HPI)
                elif k == 2:
                    boxes[:, 0] = S - cx
                    boxes[:, 1] = S - cy
                else:
                    boxes[:, 0] = cy
                    boxes[:, 1] = S - cx
                    boxes[:, 4] = _norm(boxes[:, 4] + HPI)
            pil_deg = {1: -90, 2: 180, 3: 90}[k]
            img = img.rotate(pil_deg, expand=False)

        # ── 30°/60° 額外隨機旋轉 p=0.5 ─────────────────────────────
        if random.random() > 0.5:
            deg = random.choice([30, 60, -30, -60])
            rad = math.radians(deg)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            if boxes.shape[0] > 0:
                dx = boxes[:, 0] - S / 2
                dy = boxes[:, 1] - S / 2
                boxes[:, 0] = dx * cos_a + dy * sin_a + S / 2
                boxes[:, 1] = -dx * sin_a + dy * cos_a + S / 2
                boxes[:, 4] = _norm(boxes[:, 4] + rad)
                mask = ((boxes[:, 0] >= 0) & (boxes[:, 0] < S) &
                        (boxes[:, 1] >= 0) & (boxes[:, 1] < S))
                boxes  = boxes[mask]
                labels = labels[mask]
            img = img.rotate(deg, expand=False)

        # ── HSV Jitter（各 p=0.5）──────────────────────────────────
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
                     img_size: int = 1024,
                     exclude_prefix: str = '') -> DataLoader:
    dataset = DOTADataset(data_dir, img_size=img_size, augment=augment,
                          exclude_prefix=exclude_prefix)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=augment,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=augment,
    )
