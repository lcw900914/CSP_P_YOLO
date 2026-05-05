import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

log_path = 'D:/cspyolo/checkpoints/train_log.txt'

epochs, total, cls_l, reg_l, dfl_l, lr_vals = [], [], [], [], [], []

pattern = re.compile(
    r'Epoch \[(\d+)/\d+\] lr=([\d.e+-]+) loss=([\d.]+) \(cls=([\d.]+) reg=([\d.]+) dfl=([\d.]+)\)'
)
with open(log_path) as f:
    for line in f:
        m = pattern.search(line)
        if m:
            epochs.append(int(m.group(1)))
            lr_vals.append(float(m.group(2)))
            total.append(float(m.group(3)))
            cls_l.append(float(m.group(4)))
            reg_l.append(float(m.group(5)))
            dfl_l.append(float(m.group(6)))

ep = np.array(epochs)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('CSPPartial-YOLO Training Curves (300 Epochs)', fontsize=14, fontweight='bold')

# ── 總 loss ──────────────────────────────────────
ax = axes[0, 0]
ax.plot(ep, total, color='#2196F3', linewidth=1.5)
best_ep = ep[np.argmin(total)]
best_val = min(total)
ax.axvline(best_ep, color='red', linestyle='--', alpha=0.6, label=f'best ep={best_ep} ({best_val:.4f})')
ax.set_title('Total Loss')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(); ax.grid(alpha=0.3)

# ── 各分項 loss ───────────────────────────────────
ax = axes[0, 1]
ax.plot(ep, cls_l, label='cls', color='#E53935', linewidth=1.2)
ax.plot(ep, reg_l, label='reg', color='#43A047', linewidth=1.2)
ax.plot(ep, dfl_l, label='dfl', color='#FB8C00', linewidth=1.2)
ax.set_title('Loss Components')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(); ax.grid(alpha=0.3)

# ── Learning Rate ─────────────────────────────────
ax = axes[1, 0]
ax.plot(ep, lr_vals, color='#8E24AA', linewidth=1.5)
ax.set_title('Learning Rate Schedule')
ax.set_xlabel('Epoch'); ax.set_ylabel('LR')
ax.grid(alpha=0.3)

# ── Loss 下降比例（相對 ep1）────────────────────────
ax = axes[1, 1]
rel = (np.array(total) - total[-1]) / (total[0] - total[-1]) * 100
ax.plot(ep, rel, color='#00ACC1', linewidth=1.5)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Loss Convergence (% remaining gap)')
ax.set_xlabel('Epoch'); ax.set_ylabel('Remaining gap (%)')
ax.grid(alpha=0.3)

plt.tight_layout()
out = 'D:/cspyolo/loss_curves.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'已儲存: {out}')
