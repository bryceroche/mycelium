"""
Figure 4: Per-breath CE ladder formation across training
Data from /tmp/v100_continue.log, v100 architecture (topological staging, no compression)
Five representative training step snapshots.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# --- Data parsed from /tmp/v100_continue.log ---
# Each entry: (step, [B0..B9 CE values])
# Parsed via per_breath_ce[B0..B9]: left5 ... right5 format

snapshots = [
    (50,   [0.769, 0.714, 0.699, 0.693, 0.691, 0.690, 0.689, 0.689, 0.690, 0.690]),
    (400,  [0.735, 0.604, 0.551, 0.520, 0.500, 0.489, 0.481, 0.475, 0.472, 0.469]),
    (1000, [1.056, 0.937, 0.891, 0.865, 0.855, 0.851, 0.851, 0.852, 0.854, 0.857]),
    (5000, [1.328, 1.077, 0.981, 0.923, 0.893, 0.877, 0.866, 0.861, 0.858, 0.857]),
    (8500, [1.532, 1.269, 1.123, 1.047, 1.008, 0.986, 0.975, 0.972, 0.972, 0.974]),
]

breath_idx = list(range(10))

# Color map: earlier steps lighter, later darker
cmap = matplotlib.colormaps["Blues"]
n = len(snapshots)
colors = [cmap(0.35 + 0.65 * i / (n - 1)) for i in range(n)]

fig, ax = plt.subplots(figsize=(6.2, 4.2))

for i, (step, vals) in enumerate(snapshots):
    # Compute ladder depth: B0 - B9
    spread = vals[0] - vals[-1]
    label = f"Step {step:,}  (spread {spread:+.2f})"
    ax.plot(breath_idx, vals, "o-", color=colors[i], lw=1.8, ms=5,
            label=label, zorder=4)

ax.set_xlabel("Breath index $k$", fontsize=11)
ax.set_ylabel("Per-breath CE", fontsize=11)
ax.set_title("Iterative refinement emerges during training\n(v100 — topological staging, factor graphs)",
             fontsize=11, pad=6)
ax.set_xticks(breath_idx)
ax.set_xticklabels([f"B{i}" for i in breath_idx], fontsize=8)
ax.legend(fontsize=8.5, loc="lower left", framealpha=0.9,
          title="Training step", title_fontsize=8)
ax.grid(True, ls=":", alpha=0.4)

# Annotate the "flat ladder" vs "deep ladder" arrows
# At step 50, the curve is nearly flat
ax.annotate("Flat: iteration\nproduces no new\ninformation",
            xy=(2, snapshots[0][1][2]), xytext=(4.5, 0.75),
            fontsize=7.5, color=colors[0], ha="center",
            arrowprops=dict(arrowstyle="->", color=colors[0], lw=0.9))

# At step 8500, the curve drops strongly
ax.annotate("Deep ladder:\nlate breaths much\nbetter than B0",
            xy=(8, snapshots[-1][1][8]), xytext=(5.5, 1.55),
            fontsize=7.5, color=colors[-1], ha="center",
            arrowprops=dict(arrowstyle="->", color=colors[-1], lw=0.9))

fig.tight_layout()

out_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out_dir, "fig4_breath_ladder_training.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(out_dir, "fig4_breath_ladder_training.pdf"), bbox_inches="tight")
print("Saved fig4_breath_ladder_training.{png,pdf}")
