"""F-7a — the precision-coverage frontier, all rungs + the trajectory.

Two stories on one plot, both on the 1,500-problem fixture:
(1) the vote-threshold ladder at first measurement (3/5, 4/5, 5/5) —
    tightening the rung buys precision at coverage cost;
(2) the certification channel across generations — the reading campaign
    moved the 1.0000 point RIGHT (coverage up at held precision), from
    0.9982 @ 38.1% at first measurement to 1.0000 @ 60.8% at freeze.
"""
import matplotlib.pyplot as plt

import figstyle
from figstyle import PALETTE as C

figstyle.apply_style()

# (coverage %, precision, label) — bigtest-1500, ledger-banked.
LADDER = [
    (51.7, 0.9832, "3/5 majority"),
    (44.3, 0.9925, "4/5"),
    (38.1, 0.9982, "5/5 unanimity\n(570R / 1W)"),
]
CHANNEL = [
    (38.1, 0.9982, "first measurement\n(pre-books)"),
    (57.7, 1.0000, "gen-9b, gate 5/5\n(866/1500)"),
    (55.9, 1.0000, "gen-9b + panel\n(839/1500)"),
    (57.5, 1.0000, "gen-11 + panel\n(862/1500)"),
    (60.8, 1.0000, "FREEZE, gen-14 + panel\n(912/1500)"),
]

fig, ax = plt.subplots(figsize=(7.2, 4.4))

# the vote-threshold ladder
lx, ly, ll = zip(*LADDER)
ax.plot(lx, ly, "-o", color=C["gate"], markersize=5, zorder=3,
        label="vote-threshold ladder (first measurement)")
ladder_off = [(10, -4), (8, -16), (10, -26)]
for (x, y, lab), (dx, dy) in zip(LADDER, ladder_off):
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=7, color=C["gate"], ha="left")

# the certification channel across generations
cx, cy, cl = zip(*CHANNEL)
ax.plot(cx, cy, "--", color=C["ok"], lw=1.2, alpha=0.6, zorder=2)
ax.plot(cx[1:], cy[1:], "s", color=C["ok"], markersize=6, zorder=4,
        label="certification channel across generations")
offsets = {  # staggered with leader arrows for the 1.0000 cluster
    "gen-9b, gate 5/5\n(866/1500)": (30, 42),
    "gen-9b + panel\n(839/1500)": (-52, -40),
    "gen-11 + panel\n(862/1500)": (-72, 28),
    "FREEZE, gen-14 + panel\n(912/1500)": (14, -34),
}
for x, y, lab in CHANNEL[1:]:
    dx, dy = offsets[lab]
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=7, color=C["ok"],
                fontweight="bold" if "FREEZE" in lab else "normal",
                arrowprops=dict(arrowstyle="-", color=C["ok"],
                                lw=0.6, alpha=0.5, shrinkB=4))

# the freeze point gets the ring
ax.scatter([60.8], [1.0000], s=160, facecolors="none",
           edgecolors=C["ok"], linewidths=1.6, zorder=5)

ax.axhline(1.0, color=C["faint"], lw=0.6, ls=":")
ax.text(35.0, 1.00045,
        "measured 1.0000 = zero errors observed —\na bound near a tenth "
        "of a percent,\nnot a demonstration of zero (§11)",
        fontsize=6.8, color=C["faint"], va="bottom")

ax.set_xlabel("coverage (% of 1,500-problem fixture certified)")
ax.set_ylabel("certificate precision")
ax.set_xlim(34, 66)
ax.set_ylim(0.980, 1.0022)
ax.grid(True, axis="both", alpha=0.5)
ax.legend(loc="lower left")

meta = figstyle.stamp(fig, fixtures=["bigtest-1500"])
figstyle.save(fig, "f7a_precision_coverage", meta=meta,
              fixtures=["bigtest-1500"])
