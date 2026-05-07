# CSPPartial-YOLO — v1.0 復現結果

> 論文：*CSPPartial-YOLO: A Lightweight YOLO-Based Method for Typical Objects Detection in Remote Sensing Images*  
> IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2024  
> 本版本為 PyTorch 復現，基準版本 tag：`v1.0`

---

## 1. 資料集

| 項目 | 說明 |
|---|---|
| 資料集 | DOTA-v1.0 |
| 選用類別 | plane、large-vehicle、small-vehicle、ship（共 4 類） |
| 切片大小 | 1024 × 1024，overlap = 200 px |
| 訓練集 | 6464 張切片 |
| 驗證集 | 1854 張切片（291 張原始影像） |
| 標注格式 | 旋轉框 LE90（cx, cy, w, h, angle ∈ (−π/2, π/2]） |

---

## 2. 架構

### 2.1 整體架構

```
輸入 (1 × 3 × 1024 × 1024)
    │
    ▼
CSPPartialNet（Backbone）
    ├─ P3：128 × 128 × 128
    ├─ P4：256 ×  64 ×  64
    └─ P5：512 ×  32 ×  32
    │
    ▼
CSPPartialFPN（Neck）
    ├─ Top-down：P5 → P4 → P3
    └─ Bottom-up：P3 → P4 → P5
    │
    ▼
PPYOLOERHead（Head）
    ├─ cls branch（VarifocalLoss）
    ├─ reg branch（ProbIoU + DFL）
    └─ angle branch（回歸，弧度）
```

### 2.2 Backbone：CSPPartialNet

| Stage | 輸入尺寸 | 輸出通道 | PHDC Blocks | CoordAttention |
|---|---|---|---|---|
| Stem | 1024×1024 | 32 | — | — |
| DownStem | 512×512 | 32 | — | — |
| Stage0 | 256×256 | 64 | 1 | ✓ |
| Stage1 (P3) | 128×128 | 128 | 1 | ✓ |
| Stage2 (P4) | 64×64 | 256 | 3 | ✓ |
| Stage3 (P5) | 32×32 | 512 | 1 | ✓ |

### 2.3 PHDC Block（核心模組）

```
Input (C channels)
  ├─ 前 Cp = C × cp_ratio 個通道 → HDC（空洞卷積串聯 d=[1,2,5]）
  └─ 後 C−Cp 個通道 → Identity
  Concat → PW1(1×1) → BN → ReLU → PW2(1×1) → + 殘差
```

- **cp_ratio = 0.5**（本版本，即 50% 通道參與卷積）
- HDC：3 層 3×3 空洞卷積串聯，無 BN/ReLU，空洞率 [1, 2, 5]

### 2.4 Neck：CSPPartialFPN

- 雙向 FPN（Top-down + Bottom-up）
- P5 分支加入 SPP（kernel sizes [5, 9, 13]）
- FPNStage 與 CSPPartialStage 同架構，但不含 CoordAttention

### 2.5 Head：PPYOLOERHead

- 三分支獨立預測：分類 / 框回歸（DFL） / 角度
- Stem：2 × (DWConv 3×3 + BN + SiLU)，各 scale 不共享
- Stride：[8, 16, 32]，reg_max = 16

---

## 3. 訓練設定

| 超參數 | 本版本 | 論文 |
|---|---|---|
| Optimizer | SGD | SGD |
| Momentum | 0.9 | 0.9 |
| Weight Decay | 0.0005 | 0.0005 |
| 初始 LR | 0.006 | 0.006 |
| LR Schedule | CosineDecay + LinearWarmup | CosineDecay |
| Warmup Epochs | 10 | — |
| Epochs | 300 | 300 |
| Batch Size | 6 | 6 |
| Input Size | 1024 × 1024 | 1024 × 1024 |
| AMP (混合精度) | ✓ | — |
| Gradient Clip | max_norm=10.0 | — |

### 3.1 資料增強（線上增強）

| 增強方式 | 設定 |
|---|---|
| 水平翻轉 | p=0.5，θ → −θ（LE90 正確對應） |
| 垂直翻轉 | p=0.5，θ → −θ |
| 90°/180°/270° 旋轉 | 等機率，θ ∓ π/2 後做 π-週期正規化 |
| HSV Jitter | 亮度/對比/飽和/色相，各 p=0.5 |

> **修正說明**：翻轉時不做 w/h 交換；旋轉時角度需加 ∓π/2 並正規化至 (−π/2, π/2]。  
> 原始版本（v0）未正確更新角度，本版本（v1.0）已全部修正。

---

## 4. 評估方式

### 4.1 eval.py（Per-patch）

每個 1024×1024 切片獨立評估，GT 與預測均在切片座標系下計算。

### 4.2 eval_dota.py（DOTA Full-image，論文一致）

1. 推論每個切片
2. 利用 stem 中的 `(px, py)` 將預測框轉回原始影像座標
3. 每張原始影像做全局旋轉框 NMS（去除 overlap 區重複預測）
4. GT 轉回原始座標後做去重 NMS（IoU 閾值 0.5）
5. 計算 smooth integral AP（VOC 2010+）

---

## 5. 評估結果

### 5.1 各類別 AP（eval_dota.py，mAP@IoU=0.5）

| 類別 | AP（本版本） | AP（論文） |
|---|---|---|
| plane | 88.12% | ~89% |
| large-vehicle | 72.77% | ~88% |
| small-vehicle | 61.61% | ~82% |
| ship | 83.04% | ~89% |
| **mAP@0.5** | **76.38%** | **~88.8%** |

### 5.2 效率指標

| 指標 | 本版本 | 論文 |
|---|---|---|
| Params | 9.13 M | — |
| FLOPs | 23.7 G | 16.2 G |
| Latency（GPU） | 28.5 ms | 23 ms |

### 5.3 兩種評估方式比較

| 方式 | mAP@0.5 |
|---|---|
| eval.py（per-patch） | 76.20% |
| eval_dota.py（full-image） | 76.38% |

---

## 6. 與論文差距分析

| 原因 | 說明 |
|---|---|
| **FLOPs 偏高（+46%）** | cp_ratio=0.5，DownSample 用 3×3 Conv；論文為 16.2G |
| **評估集不同** | 本版本用 val set；論文用 test set（官方伺服器評估） |
| **訓練資料** | 本版本從頭訓練 4 類；論文可能使用全 15 類預訓練再 fine-tune |
| **Mosaic 增強缺失** | 本版本未實作 Mosaic；論文系列通常包含此增強 |

---

## 7. 檔案結構

```
CSP_YOLO/
├── models/
│   ├── backbone/
│   │   ├── csp_partial_net.py     # Backbone
│   │   ├── csp_partial_stage.py   # CSPPartialStage + CBN
│   │   └── phdc_block.py          # PHDC Block（PartialConv + HDC）
│   ├── neck/
│   │   └── csp_partial_fpn.py     # 雙向 FPN
│   ├── head/
│   │   └── ppyoloe_r_head.py      # PP-YOLOE-R 旋轉偵測頭
│   ├── layers/
│   │   ├── coord_attention.py     # Coordinate Attention
│   │   └── spp.py                 # SPP
│   └── losses/
│       ├── varifocal_loss.py
│       ├── prob_iou_loss.py
│       └── dfl_loss.py
├── datasets/
│   ├── dota_preprocess.py         # 切片前處理
│   └── dota_dataset.py            # Dataset + 線上增強
├── train.py                       # 訓練腳本（含週期性 val mAP）
├── eval.py                        # Per-patch 評估
├── eval_dota.py                   # Full-image 評估（論文一致）
├── visualize.py                   # 旋轉框視覺化
└── launch_train.py                # 訓練啟動器
```

---

## 8. 執行指令

```bash
# 前處理
python datasets/dota_preprocess.py \
  --src_img /path/to/DOTA/train/images \
  --src_ann /path/to/DOTA/train/labelTxt \
  --dst D:/cspyolo/data/dota/train

# 訓練
python launch_train.py

# 評估（論文版）
python eval_dota.py \
  --weights D:/cspyolo/checkpoints_v2/best_model.pt \
  --val_dir D:/cspyolo/data/dota/val

# 視覺化
python visualize.py \
  --weights D:/cspyolo/checkpoints_v2/best_model.pt \
  --val_dir D:/cspyolo/data/dota/val \
  --out D:/cspyolo/vis
```
