"""F-9a — the saturation curve: the campaign measured its own completion.

Census problems still unread (disjoint read) vs cumulative annotated
unique problems. The curve carries its own §11 scope on the plot: this
is the saturation of ONE distribution's teachable content, not the
closing of the library.
"""
import matplotlib.pyplot as plt

import figstyle
from figstyle import PALETTE as C

figstyle.apply_style()

# (cumulative annotated uniques, census unread, note) — ledger-banked.
POINTS = [
    (0, 81, "baseline (pre-books):\n81 unread"),
    (114, 58, "books 1+2:\n+23 recovered"),
    (188, 61, "book 3 (+74 uniques,\nsame distribution):\n61 — Δ+3, within vote noise"),
]

fig, ax = plt.subplots(figsize=(7.0, 4.4))
fig.subplots_adjust(bottom=0.19)
xs, ys, notes = zip(*POINTS)
ax.plot(xs, ys, "-o", color=C["gate"], markersize=6, zorder=3)

offsets = [(10, 6), (-9, -6), (-6, 16)]
aligns = ["left", "right", "right"]
vas = ["bottom", "top", "bottom"]
for (x, y, note), (dx, dy), ha, va in zip(POINTS, offsets, aligns, vas):
    ax.annotate(note, (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=7.2, color=C["ink"], ha=ha, va=va)

# the marginal-yield-zero segment gets named
ax.annotate("marginal yield ≈ 0\nat fixed distribution",
            xy=(151, 59.5), xytext=(126, 68), fontsize=7.2, color=C["gate"],
            arrowprops=dict(arrowstyle="->", color=C["gate"], lw=0.9))

# the §11 scope, ON the plot — this figure will travel without its caption
ax.axvspan(188, 240, color=C["faint"], alpha=0.10, hatch="//", lw=0)
ax.text(214, 74.5,
        "unmeasured:\nharder strata,\nnew prose registers,\n"
        "post-registry-expansion\nproblems (§11)",
        fontsize=7.0, color=C["faint"], ha="center", va="top")
ax.text(214, 56.5, "the curve saturated;\nthe library did not close",
        fontsize=7.0, color=C["ink"], ha="center", style="italic")

ax.set_xlabel("cumulative annotated unique problems (books 1–3)")
ax.set_ylabel("census problems still unread (disjoint read)")
ax.set_xlim(-8, 240)
ax.set_ylim(52, 86)
ax.grid(True, alpha=0.5)

ax.text(0.01, 0.02,
        "Disjoint reads exclude trained items (pool 100 at x=0, 86 after). "
        "One intermediate read under a stacked-continuation regime was "
        "excluded as a regime artifact (ledger).",
        transform=ax.transAxes, fontsize=6.2, color=C["faint"], va="bottom")

meta = figstyle.stamp(fig, fixtures=["census-100 (disjoint)"])
figstyle.save(fig, "f9a_saturation_curve", meta=meta,
              fixtures=["census-100-disjoint"])
