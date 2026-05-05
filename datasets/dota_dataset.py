"""
DOTA Dataset (PyTorch)
讀取前處理後的切片資料。

標註格式（每行）：cx cy w h angle_rad class_id
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
        stem = self.samples[idx]
        img  = Image.open(self.img_dir / f'{stem}.jpg').convert('RGB')
        boxes, labels = self._load_label(stem)

        if self.augment:
            img, boxes = self._augment(img, boxes)

        # PIL → Tensor, normalize to [0,1]
        img_t = TF.to_tensor(img)                        # C,H,W float32
        img_t = TF.normalize(img_t,
                             mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])

        return img_t, boxes, labels, stem

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
            return torch.tensor(boxes, dtype=torch.float32), \
                   torch.tensor(labels, dtype=torch.long)
        return torch.zeros((0, 5), dtype=torch.float32), \
               torch.zeros((0,),   dtype=torch.long)

    def _augment(self, img: Image.Image, boxes: torch.Tensor):
        S = img.width  # 1024, square patch
        HPI = math.pi / 2

        # LE90 convention: angle ∈ (-π/2, π/2].
        # Normalise after any angle perturbation using π-periodicity (no w/h swap).
        def _norm(a):
            a[a <= -HPI] += math.pi
            a[a >   HPI] -= math.pi
            return a

        # 水平翻轉  θ → -θ  (reflect across vertical axis)
        if random.random() > 0.5:
            img = TF.hflip(img)
            if boxes.shape[0] > 0:
                boxes[:, 0] = S - boxes[:, 0]
                boxes[:, 4] = _norm(-boxes[:, 4])

        # 垂直翻轉  θ → -θ  (reflect across horizontal axis)
        if random.random() > 0.5:
            img = TF.vflip(img)
            if boxes.shape[0] > 0:
                boxes[:, 1] = S - boxes[:, 1]
                boxes[:, 4] = _norm(-boxes[:, 4])

        # 隨機 90° / 180° / 270° 旋轉
        k = random.randint(0, 3)  # 0=no-op, 1=90°CW, 2=180°, 3=270°CW
        if k > 0:
            if boxes.shape[0] > 0:
                cx, cy = boxes[:, 0].clone(), boxes[:, 1].clone()
                if k == 1:      # 90° CW : (x,y)→(S-y, x) , θ → θ-π/2
                    boxes[:, 0] = S - cy
                    boxes[:, 1] = cx
                    boxes[:, 4] = _norm(boxes[:, 4] - HPI)
                elif k == 2:    # 180°   : (x,y)→(S-x, S-y), θ unchanged
                    boxes[:, 0] = S - cx
                    boxes[:, 1] = S - cy
                else:           # 270° CW: (x,y)→(y, S-x) , θ → θ+π/2
                    boxes[:, 0] = cy
                    boxes[:, 1] = S - cx
                    boxes[:, 4] = _norm(boxes[:, 4] + HPI)
            if k == 1:
                img = img.rotate(-90, expand=False)
            elif k == 2:
                img = img.rotate(180, expand=False)
            else:
                img = img.rotate(90, expand=False)

        # HSV 色彩抖動（亮度、對比、飽和度、色相）
        if random.random() > 0.5:
            img = TF.adjust_brightness(img, random.uniform(0.6, 1.4))
        if random.random() > 0.5:
            img = TF.adjust_contrast(img, random.uniform(0.6, 1.4))
        if random.random() > 0.5:
            img = TF.adjust_saturation(img, random.uniform(0.6, 1.4))
        if random.random() > 0.5:
            img = TF.adjust_hue(img, random.uniform(-0.1, 0.1))

        return img, boxes


def collate_fn(batch):
    """自訂 collate：boxes 各圖尺寸不同，不能直接 stack"""
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
