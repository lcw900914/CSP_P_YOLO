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

## 完整重現報告

| 欄位 | 內容 |
|------|------|
| **編號 #** | 1 |
| **論文標題（簡稱）** | CSPPartial-YOLO |
| **Paper Title** | A Lightweight YOLO-Based Method for Typical Objects Detection in Remote Sensing Images (IEEE JSTARS 2024) |
| **GitHub 連結** | https://github.com/lcw900914/CSPPartialYOLO |
| **重現框架** | PyTorch 2.3.1 + mmcv 2.2.0（from scratch，論文無官方程式碼） |
| **訓練環境** | Ubuntu 24.04 LTS / Python 3.12.3 / PyTorch 2.3.1+cu121 / NVIDIA RTX 4080 16 GB / CUDA 12.1 |
| **使用資料集** | DOTA v1.0（4 類：plane, large-vehicle, small-vehicle, ship）<br>Train: 15,716 patches / Val: 1,854 patches（512×512 crop, stride 256） |
| **論文報告指標** | mAP@0.5 = 89.75% / FLOPs = 16.2 G / Latency = 23 ms |
| **我們重現結果** | mAP@0.5 = **81.48%**（plane 89.79% / large-vehicle 79.24% / small-vehicle 72.18% / ship 84.71%）/ FLOPs = 16.1 G |
| **差距 Gap** | mAP −8.27%；FLOPs 誤差 < 1% ✓ |
| **差距原因分析** | ① 論文採完整 TALA 標籤分配，我們因冷啟動不穩定改用 AABB，犧牲旋轉框精準匹配；② 論文 LE90 角度回歸細節未公開；③ 資料前處理細節（overlap/padding）依論文描述推算；④ 300 epochs，論文未明確說明 epoch 數 |
| **改善方向** | ① 暖啟動 TALA（前 100 epoch AABB，之後切換）；② Mosaic + 旋轉增強；③ 延長訓練至 500 epochs；④ 調整 NMS threshold |
| **跨資料集測試** | SODA-A 遙感資料集（9,252 train / 5,613 val patches，已混合訓練） |
| **跨資料集結果** | 待評估（v8 訓練完成後補充） |
| **README 是否完整** | ✅ 完整 |
| **備註／心得** | 論文無官方程式碼，架構完全從論文圖表反推（CSPPartialNet、PHDC 模組、SPP 配置）。最大挑戰為 TALA 冷啟動惡性循環，改用 AABB 後穩定收斂。另診斷出 EMA 評估 bug 導致 v6/v7 訓練 log 全程 0% mAP（training model 實為 67–80%）。整體重現達論文 90% 水準，FLOPs 精確匹配。 |

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
