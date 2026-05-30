"""
Figure 2: Cross-architecture K-sweep
Data from /tmp/cross_ksweep.log (v100, v101_peak, v103)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# --- Data extracted from /tmp/cross_ksweep.log ---
K_vals = [1, 3, 5, 8, 10]

# OVERALL cell_acc per architecture
data = {
    "v100\n(no compression)":     [0.2283, 0.3777, 0.4008, 0.4074, 0.4071],
    "v101\n(projection waist)":   [0.2585, 0.4303, 0.4667, 0.4754, 0.4757],
    "v103\n(VQ-VAE codebook)":    [0.2386, 0.4166, 0.4543, 0.4689, 0.4661],
}

colors = {
    "v100\n(no compression)":   "#4393c3",
    "v101\n(projection waist)": "#d6604d",
    "v103\n(VQ-VAE codebook)":  "#74add1",
}
markers = {
    "v100\n(no compression)":   "o",
    "v101\n(projection waist)": "s",
    "v103\n(VQ-VAE codebook)":  "^",
}

fig, ax = plt.subplots(figsize=(6.0, 4.2))

for label, acc_list in data.items():
    pct = [100 * a for a in acc_list]
    ax.plot(K_vals, pct, marker=markers[label], color=colors[label],
            lw=2.0, ms=7, label=label, zorder=4)
    # Annotate K=1 vs K=10 gap with a vertical bracket at K=10
    gap = pct[-1] - pct[0]
    ax.annotate("", xy=(10.3, pct[-1]), xytext=(10.3, pct[0]),
                arrowprops=dict(arrowstyle="<->", color=colors[label],
                                lw=1.2, shrinkA=0, shrinkB=0))
    ax.text(10.55, (pct[-1] + pct[0]) / 2, f"+{gap:.1f}pp",
            color=colors[label], fontsize=7.5, va="center")

ax.set_xlabel("Number of breaths $K$", fontsize=11)
ax.set_ylabel("Overall cell accuracy (%)", fontsize=11)
ax.set_title("Compression amplifies breathing's benefit on factor graphs", fontsize=11, pad=8)
ax.set_xlim(-0.5, 11.5)
ax.set_ylim(15, 60)
ax.set_xticks(K_vals)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.legend(fontsize=9, loc="lower right", framealpha=0.9, ncol=1)
ax.grid(True, ls=":", alpha=0.4)

# Shade the K=1 baseline vs K=10 region lightly
ax.axvspan(0.7, 1.3, alpha=0.08, color="gray", label="")

fig.tight_layout()

out_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out_dir, "fig2_cross_arch_ksweep.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(out_dir, "fig2_cross_arch_ksweep.pdf"), bbox_inches="tight")
print("Saved fig2_cross_arch_ksweep.{png,pdf}")
