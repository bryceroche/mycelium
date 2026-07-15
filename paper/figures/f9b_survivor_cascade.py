"""F-9b — the survivor cascade: nine registered refutations, one wall priced.

The worked example of §9.2. Each row is a registered hypothesis about
the committed-wrong survivor population, with its measured verdict.
Step 6 is the pivot — the mechanism named, not a kill. The closing box
is the arc's own accounting line from the ledger.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import figstyle
from figstyle import PALETTE as C

figstyle.apply_style()

# (hypothesis, verdict, kind) — kind: kill | pivot | price
STEPS = [
    ("1 · rendering quality (teeth)",
     "uniform across survivors — killed", "kill"),
    ("2 · mention multiplicity",
     "flat — killed", "kill"),
    ("3 · omission blindness",
     "dead — killed", "kill"),
    ("4 · suspicion transplant",
     "flat — killed", "kill"),
    ("5 · binding-field enrichment",
     "runs backwards (de-enriched) — killed", "kill"),
    ("6 · THE MECHANISM NAMED: the routing wall",
     "states 99.6% correctly decodable; the pointer is mis-aimed", "pivot"),
    ("7 · perfect-oracle repair",
     "ceiling 13.9% — even perfect flags cannot re-aim a pointer", "price"),
    ("8 · monitor-gated ratchet",
     "leak found and fixed: 6% — retired", "price"),
    ("9 · input-mark beacon",
     "3.0% at precision 0.165 — does not graduate", "price"),
]

KIND_COLOR = {"kill": C["kill"], "pivot": C["gate"], "price": C["wild"]}
KIND_MARK = {"kill": "✕", "pivot": "→", "price": "$"}

fig, ax = plt.subplots(figsize=(7.6, 7.4))
ax.set_xlim(0, 10)
ax.set_ylim(0.15, 12.4)
ax.axis("off")

TOP, ROW_H = 11.6, 1.06
SPINE_X = 0.55

ax.text(0.15, 12.15, "the committed-wrong survivor population "
        "(every filter passed, still wrong)",
        fontsize=8, color=C["ink"], style="italic")

for i, (hyp, verdict, kind) in enumerate(STEPS):
    y = TOP - i * ROW_H
    col = KIND_COLOR[kind]
    lw = 1.6 if kind == "pivot" else 1.0
    box = FancyBboxPatch((1.05, y - 0.86), 8.6, 0.86,
                         boxstyle="round,pad=0.03,rounding_size=0.08",
                         facecolor=col, alpha=0.16 if kind == "pivot" else 0.07,
                         edgecolor=col, linewidth=lw)
    ax.add_patch(box)
    ax.text(1.3, y - 0.24, hyp, fontsize=8.2, color=C["ink"],
            fontweight="bold" if kind == "pivot" else "normal")
    ax.text(1.3, y - 0.62, verdict, fontsize=7.4, color=col)
    ax.text(SPINE_X, y - 0.43, KIND_MARK[kind], fontsize=11, color=col,
            ha="center", va="center",
            fontweight="bold" if kind == "pivot" else "normal")
    if i < len(STEPS) - 1:
        ax.plot([SPINE_X, SPINE_X], [y - 0.72, y - ROW_H - 0.12],
                color=C["faint"], lw=0.8)

# the closing accounting box — the ledger's own line
y_end = TOP - len(STEPS) * ROW_H - 0.18
box = FancyBboxPatch((1.05, y_end - 1.06), 8.6, 1.06,
                     boxstyle="round,pad=0.03,rounding_size=0.08",
                     facecolor=C["ok"], alpha=0.10,
                     edgecolor=C["ok"], linewidth=1.6)
ax.add_patch(box)
ax.text(5.35, y_end - 0.30,
        "the wall, priced: detect-and-abstain — a measurement, not a "
        "surrender",
        fontsize=8.4, color=C["ok"], ha="center", fontweight="bold")
ax.text(5.35, y_end - 0.72,
        "nine registered refutations · four laws · two retired builds · "
        "one working instrument · one closed population",
        fontsize=6.8, color=C["ink"], ha="center")
ax.plot([SPINE_X, SPINE_X], [TOP - 8 * ROW_H - 0.72, y_end - 0.18],
        color=C["faint"], lw=0.8)
ax.text(SPINE_X, y_end - 0.3, "∎", fontsize=10, color=C["ok"],
        ha="center", va="center")

# legend for the three row kinds
ax.text(1.05, 0.38, "✕ hypothesis killed at its registered bar"
        "     → mechanism named     $ repair priced (and declined)",
        fontsize=7.2, color=C["faint"])

meta = figstyle.stamp(fig, fixtures=["survivor-arc final ledger"])
figstyle.save(fig, "f9b_survivor_cascade", meta=meta,
              fixtures=["survivor-arc-final-ledger"])
