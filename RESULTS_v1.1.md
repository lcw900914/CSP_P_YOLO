# CSPPartial-YOLO — v1.1 架構優化結果

> 基於 v1.0，針對 FLOPs 過高問題進行架構修正  
> Branch：`exp/reduce-flops`  
> Tag：`v1.1`

---

## 1. 修改內容

### 1.1 修改一：PartialConv cp_ratio 0.5 → 0.25

**檔案**：`models/backbone/phdc_block.py`

| | v1.0 | v1.1 |
|---|---|---|
| cp_ratio | 0.5（50% 通道參與 HDC） | **0.25**（25% 通道參與 HDC） |

**說明**：  
PHDC Block 中，只有前 `Cp = channels × cp_ratio` 個通道會經過三層空洞卷積（HDC），其餘通道直接 Identity 傳遞。  
HDC 的 FLOPs 與 cp_ratio² 成正比，因此從 0.5 降至 0.25 可節省 75% 的 HDC 運算量。

```python
# Before (v1.0)
PartialConv(channels, cp_ratio=0.5)   # Cp = channels/2
PHDCBlock(channels, cp_ratio=0.5)

# After (v1.1)
PartialConv(channels, cp_ratio=0.25)  # Cp = channels/4
PHDCBlock(channels, cp_ratio=0.25)
```

---

### 1.2 修改二：Backbone DownSample 3×3 Conv → MaxPool2d

**檔案**：`models/backbone/csp_partial_net.py`

| | v1.0 | v1.1 |
|---|---|---|
| DownSample | Conv2d(C, C, 3, stride=2) + BN + ReLU | **MaxPool2d(2, 2)** |
| 影響層 | down_stem, down0, down1, down2 | 同左 |

**說明**：  
Backbone 中共有 4 個 DownSample 模組（down_stem / down0 / down1 / down2）負責空間降採樣。  
原本採用 3×3 stride=2 Conv，每層約 1.2G FLOPs，4 層合計 ~4.8G。  
改為 MaxPool2d 後，降採樣 FLOPs 降至接近 0，且不改變通道數。

> **注意**：FPN 頸部（`csp_partial_fpn.py`）的 down_p3 / down_p4 維持原本的 3×3 stride=2 Conv，  
> 因 FPN 底層上升路徑需要可學習的特徵融合，不適合換成 MaxPool。

```python
# Before (v1.0)
class DownSample(nn.Module):
    def __init__(self, channels):
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

# After (v1.1)
class DownSample(nn.Module):
    def __init__(self, channels):
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
```

---

## 2. FLOPs 與參數量對比

| 版本 | FLOPs | Params | 與論文差距 |
|---|---|---|---|
| v1.0（原始） | 23.67 G | 9.13 M | +46% |
| **v1.1（本版本）** | **16.1 G** | **6.52 M** | **−0.6%** |
| 論文目標 | 16.2 G | — | — |

> 兩項修改合計節省 **7.57G FLOPs（−32%）** 及 **2.61M 參數（−28.6%）**

---

## 3. 最終評估結果

> 訓練設定與 v1.0 相同：300 epochs，batch=6，lr=0.006，warmup=10  
> 評估協定：DOTA full-image style（eval_dota.py），IoU threshold=0.5

| 類別 | v1.0 mAP | v1.1 mAP | 變化 |
|---|---|---|---|
| plane | 88.12% | **85.77%** | −2.35% |
| large-vehicle | 72.77% | **69.65%** | −3.12% |
| small-vehicle | 61.61% | **60.93%** | −0.68% |
| ship | 83.04% | **77.43%** | −5.61% |
| **mAP@0.5** | **76.38%** | **73.44%** | **−2.94%** |
| FLOPs | 23.67 G | **16.1 G** | **−32%** |
| Params | 9.13 M | **6.52 M** | **−28.6%** |
| Latency | 28.5 ms | **24.4 ms** | **−14.4%** |

---

## 4. 結果總結

v1.1 以兩項低侵入性的架構修改，在大幅降低計算成本的同時將精度損失控制在合理範圍：

| 目標 | 達成狀況 |
|---|---|
| FLOPs 對齊論文（16.2G） | ✅ 16.1G（差距 −0.6%） |
| Latency 改善 | ✅ 28.5ms → 24.4ms（−14.4%） |
| 參數量減少 | ✅ 9.13M → 6.52M（−28.6%） |
| mAP 損失可控 | ✅ 76.38% → 73.44%（−2.94%） |

**效率提升亮點**：FLOPs 減少 32%、延遲減少 14%，而 mAP 僅下降 2.94%，呈現良好的精度-效率權衡。  
**主要代價**：ship 類別下降最多（−5.61%），推測與 MaxPool 取代可學習 DownSample 後對細粒度特徵的影響有關。

---

## 5. 預期效果分析與實際驗證

| 預測 | 實際結果 |
|---|---|
| cp_ratio 降低緩解過擬合 | 部分成立：small-vehicle 僅降 0.68%，優於預期 |
| MaxPool 缺乏可學習降採樣 | 有影響：整體 mAP 下降，ship 最顯著 |
| 感受野縮小影響精度 | 輕微，大部分類別損失在 3% 以內 |

---

## 6. 訓練與評估指令

```bash
# 訓練
python launch_train.py

# 或直接執行
python train.py \
  --train_dir D:/cspyolo/data/dota/train \
  --val_dir   D:/cspyolo/data/dota/val \
  --output    D:/cspyolo/checkpoints_v3 \
  --epochs 300 --batch 6 --lr 0.006

# 評估（DOTA full-image 協定）
python eval_dota.py \
  --weights D:/cspyolo/checkpoints_v3/best_model_map.pt \
  --val_dir D:/cspyolo/data/dota/val
```
