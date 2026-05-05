import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── 資料 ────────────────────────────────────────
classes = ['plane', 'large-vehicle', 'small-vehicle', 'ship']
ap_vals = [26.50, 12.53, 47.46, 40.17]
mAP     = 31.67

paper_flops   = 16.2;  our_flops   = 23.7
paper_latency = 23.0;  our_latency = 26.2

# ── 圖 1: AP per class + mAP ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('CSPPartial-YOLO Evaluation Results', fontsize=14, fontweight='bold')

ax = axes[0]
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
bars = ax.bar(classes, ap_vals, color=colors, width=0.5)
ax.axhline(mAP, color='red', linestyle='--', linewidth=1.5, label=f'mAP={mAP:.2f}%')
for bar, v in zip(bars, ap_vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f'{v:.1f}%',
            ha='center', va='bottom', fontsize=10)
ax.set_ylim(0, 65)
ax.set_ylabel('AP (%)')
ax.set_title('AP per Class @ IoU=0.5')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# ── 圖 2: FLOPs 比較 ────────────────────────────
ax = axes[1]
x = np.arange(2)
w = 0.3
b1 = ax.bar(x - w/2, [paper_flops, paper_latency], w, label='Paper Target', color='#4C72B0')
b2 = ax.bar(x + w/2, [our_flops,   our_latency],   w, label='Ours',         color='#DD8452')
ax.set_xticks(x)
ax.set_xticklabels(['FLOPs (G)', 'Latency (ms)'])
ax.set_title('FLOPs & Latency vs Paper')
ax.legend()
ax.grid(axis='y', alpha=0.3)
for bars in [b1, b2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=10)

# ── 圖 3: mAP 雷達圖 ─────────────────────────────
ax = axes[2]
ax.set_aspect('equal')
N = len(classes)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]
vals = ap_vals + ap_vals[:1]

ax2 = plt.subplot(1, 3, 3, polar=True)
ax2.plot(angles, vals, 'o-', linewidth=2, color='#4C72B0')
ax2.fill(angles, vals, alpha=0.25, color='#4C72B0')
ax2.set_thetagrids(np.degrees(angles[:-1]), classes)
ax2.set_ylim(0, 60)
ax2.set_title('AP Radar Chart', pad=15)
ax2.grid(True)

plt.tight_layout()
out = 'D:/cspyolo/results_plot.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'已儲存: {out}')
plt.show()
