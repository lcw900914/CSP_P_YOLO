# CSPPartial-YOLO — v1.1 架構優化結果

> 基於 v1.0，針對 FLOPs 過高問題進行架構修正  
> Branch：`exp/reduce-flops`  
> 對應 tag：`v1.1`（訓練完成後建立）

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
| **v1.1（本版本）** | **16.14 G** | **6.52 M** | **−0.4%** |
| 論文目標 | 16.2 G | — | — |

> 兩項修改合計節省 **7.53G FLOPs（-31.8%）** 及 **2.61M 參數（-28.6%）**

---

## 3. 評估結果

> 訓練完成後填入（訓練設定與 v1.0 相同：300 epochs，batch=6，lr=0.006）

| 類別 | v1.0 mAP | v1.1 mAP | 變化 |
|---|---|---|---|
| plane | 88.12% | — | — |
| large-vehicle | 72.77% | — | — |
| small-vehicle | 61.61% | — | — |
| ship | 83.04% | — | — |
| **mAP@0.5** | **76.38%** | **—** | **—** |
| FLOPs | 23.67G | 16.14G | -31.8% |
| Latency | 28.5 ms | — | — |

---

## 4. 預期效果分析

**cp_ratio 降低（0.5 → 0.25）的影響**：
- 每個 PHDC Block 中只有 25% 的通道進行 HDC，感受野覆蓋範圍略有下降
- 但 Identity 路徑（75% 通道）保留完整空間資訊
- 參數量減少，可能緩解 6464 訓練樣本下的過擬合問題

**MaxPool DownSample 的影響**：
- 優點：零額外參數，無梯度噪音，空間降採樣更乾淨
- 潛在缺點：缺少可學習的降採樣特徵轉換
- CSPPartialStage 的入口 CBN 仍是可學習的 1×1 Conv，可彌補特徵轉換能力

---

## 5. 訓練指令

```bash
# 使用 launch_train.py（輸出至 checkpoints_v3）
python launch_train.py

# 或直接執行
python train.py \
  --train_dir D:/cspyolo/data/dota/train \
  --val_dir   D:/cspyolo/data/dota/val \
  --output    D:/cspyolo/checkpoints_v3 \
  --epochs 300 --batch 6 --lr 0.006

# 評估（論文版）
python eval_dota.py \
  --weights D:/cspyolo/checkpoints_v3/best_model.pt \
  --val_dir D:/cspyolo/data/dota/val
```
