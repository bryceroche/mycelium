"""
Figure 3: Cell accuracy vs DAG depth at K=10
Data from /tmp/cross_ksweep.log, K=10 sections for v100, v101_peak, v103
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# --- Data from cross_ksweep.log K=10 sections ---
depths = [2, 3, 4, 5, 6, 7]

# v100 K=10
v100_acc = [0.4842, 0.4460, 0.4351, 0.3996, 0.3521, 0.2821]
# v101_peak K=10
v101_acc = [0.5792, 0.5036, 0.4948, 0.4444, 0.4346, 0.3689]
# v103 K=10
v103_acc = [0.5747, 0.4748, 0.4887, 0.4386, 0.4386, 0.3725]

# n per depth (from log, same for all architectures at K=10)
n_per_depth = [107, 125, 111, 96, 82, 79]

data = {
    "v100 (no compression)":   (v100_acc, "#4393c3", "o", "--"),
    "v101 (projection waist)": (v101_acc, "#d6604d", "s", "-"),
    "v103 (VQ-VAE codebook)":  (v103_acc, "#74add1", "^", "-."),
}

fig, ax = plt.subplots(figsize=(6.0, 4.2))

for label, (acc, color, marker, ls) in data.items():
    pct = [100 * a for a in acc]
    ax.plot(depths, pct, marker=marker, color=color, lw=1.8, ms=7,
            linestyle=ls, label=label, zorder=4)

# Annotate n per depth at top
for d, n in zip(depths, n_per_depth):
    ax.text(d, 62.5, f"n={n}", ha="center", va="bottom", fontsize=6.5, color="#555")

ax.set_xlabel("DAG depth (number of operation layers)", fontsize=11)
ax.set_ylabel("Cell accuracy at $K=10$ (%)", fontsize=11)
ax.set_title("DAG depth determines accuracy: empirical signature\nof message-passing inference", fontsize=11, pad=6)
ax.set_xticks(depths)
ax.set_xlim(1.5, 7.5)
ax.set_ylim(20, 68)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
ax.grid(True, ls=":", alpha=0.4)

# Annotate monotonic decline
ax.annotate("Monotonic decline with depth\n= mixing-time signature of BP",
            xy=(6.2, 100 * v101_acc[4]), xytext=(5.0, 35),
            fontsize=7.5, color="#555",
            arrowprops=dict(arrowstyle="->", color="#888", lw=1.0),
            ha="center")

fig.tight_layout()

out_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out_dir, "fig3_depth_correlation.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(out_dir, "fig3_depth_correlation.pdf"), bbox_inches="tight")
print("Saved fig3_depth_correlation.{png,pdf}")
