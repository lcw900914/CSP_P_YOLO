# CSPPartial-YOLO 復現進度紀錄

## 專案概述

復現論文：**CSPPartial-YOLO: A Lightweight YOLO-Based Method for Typical Objects Detection in Remote Sensing Images**（IEEE JSTARS 2024）

- 資料集：DOTA（4 類：plane, large-vehicle, small-vehicle, ship）
- 框架：PyTorch（非官方 PaddlePaddle）
- 硬體：RTX 3060 Ti 8GB, Windows 11

---

## 環境設定

```
conda env: D:\cspyolo\env  (Python 3.10)
PyTorch 2.1.2+cu121, CUDA 12.1
mmengine 0.10.3, mmcv 2.1.0
numpy 1.26.4
```

---

## 訓練指令

```bash
cd D:\cspyolo
D:\cspyolo\env\python.exe D:/cspyolo/project/train.py --workers 0
```

**注意事項：**
- Windows 上必須加 `--workers 0`，否則 DataLoader 會 deadlock
- 不要同時啟動多個訓練進程（會共搶 GPU，速度下降 3-4 倍）
- 訓練中打遊戲會導致 GPU 速度大幅下降
- 監控進度：`Get-Content D:\cspyolo\checkpoints\train_log.txt -Wait -Tail 5`

**完整參數（預設值）：**
```
--train_dir  D:/cspyolo/data/dota/train
--val_dir    D:/cspyolo/data/dota/val
--output     D:/cspyolo/checkpoints
--epochs     300
--batch      6
--lr         0.006
--workers    0
```

---

## 評估指令

```bash
cd D:\cspyolo
D:\cspyolo\env\python.exe -u D:/cspyolo/project/eval.py --workers 0 --nms_thr 0.35 --score_thr 0.3
```

---

## Bug 修復紀錄

### Bug 1：dist2rbox 缺少 stride 縮放（嚴重）

**問題：** `rotated_box.py` 的 `dist2rbox` 函數沒有乘以 stride，導致預測框最大只有 16px，無論物體實際多大。

**影響：** 大物體（plane, large-vehicle）幾乎偵測不到，AP 極低。

**修復位置：**
- `models/utils/rotated_box.py`：加入 `stride` 參數，乘到距離計算結果
- `models/head/ppyoloe_r_head.py` decode()：改為逐 scale 呼叫 dist2rbox，傳入正確 stride
- `models/head/ppyoloe_r_head.py` loss()：移除錯誤的 `pos_reg * pos_stride`，改傳 stride 給 dist2rbox

### Bug 2：水平翻轉角度計算錯誤（le90）

**問題：** `dota_dataset.py` 翻轉增強時用 `boxes[:,4] = -boxes[:,4]` 再 clamp，導致非零角度被錯誤歸零（約 50% 訓練樣本角度錯誤）。

**修復：** 正確的 le90 翻轉需要 swap w/h 並減去 π/2：
```python
new_angle = -boxes[:, 4]
mask = new_angle > 0
if mask.any():
    w_tmp = boxes[mask, 2].clone()
    boxes[mask, 2] = boxes[mask, 3]
    boxes[mask, 3] = w_tmp
    new_angle[mask] -= math.pi / 2
boxes[:, 4] = new_angle
```

---

## 訓練結果比較

### v1（有 Bug）

| 項目 | 數值 |
|------|------|
| 訓練 epochs | 300 |
| Best loss | 0.7346 |
| 每 epoch 時間 | ~222s |

| 類別 | AP@0.5 |
|------|--------|
| plane | 26.50% |
| large-vehicle | 12.53% |
| small-vehicle | 47.46% |
| ship | 40.17% |
| **mAP** | **31.67%** |

- FLOPs: 23.7G
- Latency: 26.2ms

### v2（修 Bug 後）

| 項目 | 數值 |
|------|------|
| 訓練 epochs | 300 |
| Best loss | 0.3013 |
| 每 epoch 時間 | ~340s |

| 類別 | AP@0.5 | 相比 v1 |
|------|--------|---------|
| plane | **80.34%** | +53.84% |
| large-vehicle | **49.56%** | +37.03% |
| small-vehicle | **50.82%** | +3.36% |
| ship | **65.26%** | +25.09% |
| **mAP** | **61.50%** | **+29.83%** |

- FLOPs: 23.7G
- Latency: 34.1ms
- 論文目標：89.75% mAP / 16.2G FLOPs / 23ms Latency

### 差距分析

| | v1 | v2 | 論文 |
|--|--|--|--|
| mAP@0.5 | 31.67% | 61.50% | ~89.75% |
| loss（final） | 0.7346 | 0.3013 | — |

v2 vs v1：mAP 幾乎翻倍，主要來自 stride bug 修復（飛機 +54%，大型車輛 +37%）。
與論文差距約 28%，主要原因待分析。

---

## Loss 分量分析（v2）

| Epoch | Total | cls | reg | dfl |
|-------|-------|-----|-----|-----|
| 1 | 1.7417 | 1.162 | 0.481 | 0.098 |
| 50 | 0.7325 | 0.501 | 0.169 | 0.063 |
| 100 | 0.6558 | 0.445 | 0.151 | 0.060 |
| 150 | 0.5558 | 0.367 | 0.132 | 0.056 |
| 200 | 0.4399 | 0.285 | 0.109 | 0.051 |
| 250 | 0.3352 | 0.197 | 0.091 | 0.047 |
| 300 | 0.3013 | 0.168 | 0.086 | 0.047 |

Loss 權重（論文，繼承自 PP-YOLOE-R）：cls × 1.0 + reg × 2.5 + dfl × 0.05

---

## 下一步

### 優先（縮小與論文的差距）

1. **調整 label assignment**：目前使用簡化版 AABB，論文使用 Task-Aligned Label Assignment，可能影響 AP 約 5-15%
2. **增加資料增強**：目前只有 hflip + color jitter，可加 random rotate、mosaic 等
3. **調整 NMS 閾值**：嘗試不同 nms_thr / score_thr 組合進行 eval

### 次要

4. **FLOPs 優化**：目前 23.7G vs 論文 16.2G，可能有通道數設定差異
5. **Latency 優化**：34.1ms vs 論文 23ms
6. **使用全部 DOTA 類別**（16 類）進行完整評估

### 硬體需求

| 用途 | 最低顯存 |
|------|---------|
| 推論 / eval | ~3GB（RTX 2060 可用）|
| 訓練（batch=6） | ~4.5GB（RTX 2060 可用）|

---

## 檔案結構

```
D:\cspyolo\
├── project\                    # 程式碼（本 repo）
│   ├── train.py
│   ├── eval.py
│   ├── plot_loss.py
│   ├── plot_results.py
│   ├── datasets\
│   │   └── dota_dataset.py
│   └── models\
│       ├── csp_partial_yolo.py
│       ├── backbone\
│       │   ├── phdc_block.py
│       │   ├── csp_partial_stage.py
│       │   └── csp_partial_net.py
│       ├── neck\
│       │   └── csp_partial_fpn.py
│       ├── head\
│       │   └── ppyoloe_r_head.py
│       ├── losses\
│       │   ├── varifocal_loss.py
│       │   ├── prob_iou_loss.py
│       │   └── dfl_loss.py
│       ├── layers\
│       │   ├── coord_attention.py
│       │   └── spp.py
│       └── utils\
│           └── rotated_box.py
├── checkpoints\                # 模型權重（不上傳）
├── data\                       # DOTA 資料集（不上傳）
└── env\                        # Python 環境（不上傳）
```
