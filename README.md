# CSPPartial-YOLO：遙感影像目標偵測重現

> **Paper:** *A Lightweight YOLO-Based Method for Typical Objects Detection in Remote Sensing Images*  
> **Journal:** IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTARS), 2024  
> **Reproduction:** PyTorch from scratch — no official code released

---

## 偵測效果 Demo

| 機場 — Plane | 停機坪 — Plane（多角度） |
|:---:|:---:|
| ![demo1](demo/01_P1854_0_0.jpg) | ![demo2](demo/05_P1088_0_824.jpg) |

| 港灣 — Ship & Small-Vehicle | 停車場 — Small-Vehicle & Large-Vehicle |
|:---:|:---:|
| ![demo3](demo/04_P0761_594_0.jpg) | ![demo4](demo/02_soda_00685_3731_824.jpg) |

> 彩色旋轉框說明：🔵 **Plane**　🟢 **Large-Vehicle**　🟠 **Small-Vehicle**　🟡 **Ship**  
> 灰色細框為 Ground Truth（僅供參考）

---

## 重現結果

### 主要指標對比

| 指標 | 論文報告 | 本次重現 | 差距 |
|------|:-------:|:-------:|:----:|
| **mAP@0.5** | **89.75%** | **81.48%** | −8.27% |
| FLOPs | 16.2 G | 16.1 G | −0.1 G ✓ |
| Latency | 23 ms | 3.5 ms† | — |
| Params | ~6.5 M | 6.52 M | ✓ |

† Latency 以 RTX 4080 量測，論文為嵌入式設備，不具可比性。

### 各類別 AP（DOTA val, 1854 張，IoU@0.5）

| Class | AP |
|-------|---:|
| Plane | **89.79%** |
| Large-Vehicle | **79.24%** |
| Small-Vehicle | **72.18%** |
| Ship | **84.71%** |
| **mAP@0.5** | **81.48%** |

---


## 架構

```
Input (3×1024×1024)
    │
    ▼
CSPPartialNet (Backbone)          ← cp_ratio=0.25, MaxPool Downsample
    │  p3 (128ch, /8)
    │  p4 (256ch, /16)
    │  p5 (512ch, /32)
    ▼
CSPPartialFPN (Neck)              ← Top-down + Bottom-up, PHDC module
    │  p3' / p4' / p5'
    ▼
PPYOLOERHead (Head)               ← DFL reg + 1-dim angle + VFL cls
    │
    ▼
旋轉框輸出 (cx, cy, w, h, θ) × 4 classes
```

- **Params:** 6.52 M　**FLOPs:** 16.1 G

---

## 環境安裝

```bash
# 建立虛擬環境
python -m venv venv && source venv/bin/activate

# 安裝依賴
pip install -r requirements.txt

# 安裝 mmcv（需對應 CUDA 版本）
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3/index.html
```

---

## 資料前處理

```bash
# DOTA v1.0：下載後執行切圖
python make_odp.py \
  --src_img /path/to/DOTA/train/images \
  --src_lbl /path/to/DOTA/train/labelTxt \
  --out     datasets/dota/dota/train \
  --size 512 --stride 256

# 同理處理 val
python make_odp.py \
  --src_img /path/to/DOTA/val/images \
  --src_lbl /path/to/DOTA/val/labelTxt \
  --out     datasets/dota/dota/val \
  --size 512 --stride 256
```

---

## 訓練

```bash
python train.py \
  --train_dir datasets/dota/dota/train \
  --val_dir   datasets/dota/dota/val \
  --output    checkpoints \
  --epochs    300 \
  --batch     28 \
  --lr        0.010 \
  --warmup    10 \
  --workers   4 \
  --dota_only_val
```

---

## 評估

```bash
python eval.py \
  --val_dir    datasets/dota/dota/val \
  --weights    checkpoints/best_model_map.pt \
  --batch      8 \
  --score_thr  0.05 \
  --nms_thr    0.1 \
  --dota_only_val
```

---

## 視覺化

```bash
python visualize.py \
  --weights   checkpoints/best_model_map.pt \
  --val_dir   datasets/dota/dota/val \
  --out       demo \
  --n         8 \
  --score_thr 0.25
```

---

## 最佳權重

`checkpoints/best_model_map.pt`（51 MB）— epoch 210，mAP@0.5 = **81.48%**

---

## 類別說明

| 顏色 | 類別 | AP |
|------|------|----|
| 🔵 藍 | plane | 89.79% |
| 🟢 綠 | large-vehicle | 79.24% |
| 🟠 橙 | small-vehicle | 72.18% |
| 🟡 黃 | ship | 84.71% |
