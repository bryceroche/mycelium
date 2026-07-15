"""F-7b — vote-entropy basin separation: temperature is orthogonal to truth.

Mean vote entropy by outcome class (n=36 pilot, book-1 gates). Entropy
separates SHALLOW from DEEP almost perfectly — and cannot separate
deep-correct from deep-wrong. One instrument reads basin depth; depth
is not truth; that is why the chain ends at an external key.
"""
import matplotlib.pyplot as plt

import figstyle
from figstyle import PALETTE as C

figstyle.apply_style()

CLASSES = [  # (label, H, color) — ledger-banked pilot, n=36
    ("shallow-correct", 0.846, C["wild"]),
    ("deep-wrong", 0.212, C["kill"]),
    ("refused", 0.116, C["faint"]),
    ("deep-correct", 0.000, C["ok"]),
]

fig, ax = plt.subplots(figsize=(6.4, 3.8))
fig.subplots_adjust(bottom=0.24)
labels = [c[0] for c in CLASSES]
values = [c[1] for c in CLASSES]
colors = [c[2] for c in CLASSES]
ypos = range(len(CLASSES))

ax.barh(ypos, values, height=0.55, color=colors, alpha=0.85)
ax.set_yticks(ypos, labels)
ax.set_xlabel("mean vote entropy H across five renderings")
ax.set_xlim(0, 1.0)
ax.grid(True, axis="x", alpha=0.5)

for y, v in zip(ypos, values):
    ax.text(v + 0.012, y, f"{v:.3f}", va="center", fontsize=8,
            color=C["ink"], family="DejaVu Sans Mono")

# the separation entropy CAN make, and the one it cannot
ax.annotate("entropy separates shallow from deep\n(0.846 vs ≤ 0.212)",
            xy=(0.78, 0.0), xytext=(0.56, 0.85), fontsize=7.5,
            color=C["ink"],
            arrowprops=dict(arrowstyle="->", color=C["faint"], lw=0.9))
ax.annotate("…and cannot separate deep-wrong from deep-correct:\n"
            "the confidently wrong are nearly as quiet as the\n"
            "confidently right — depth is not truth",
            xy=(0.24, 1.15), xytext=(0.36, 2.15), fontsize=7.5,
            color=C["kill"],
            arrowprops=dict(arrowstyle="->", color=C["kill"], lw=0.9))

ax.text(0.985, 0.96, "n = 36 (pilot, book-1 gate items)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=6.8, color=C["faint"])

meta = figstyle.stamp(fig, fixtures=["book1-gates-n36"])
figstyle.save(fig, "f7b_entropy_basins", meta=meta,
              fixtures=["book1-gates-n36"])
