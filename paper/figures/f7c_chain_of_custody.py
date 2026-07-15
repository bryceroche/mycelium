"""F-7c — the chain of custody, drawn (candidate Figure 1).

Four gates, four invariances, five trajectories. Every gate is annotated
with the named specimen it provably catches: [71] dies at the mouth, the
wild stable votes die at the panel, [78] splits the answer dial from the
certify dial at the vote, and the one broken certificate in the fixture
history dies at the key. A clean item runs the whole chain.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import figstyle
from figstyle import PALETTE as C

figstyle.apply_style()

fig, ax = plt.subplots(figsize=(11.5, 5.4))
ax.set_xlim(0, 12.4)
ax.set_ylim(0, 5.8)
ax.axis("off")

# ---- the four gates -------------------------------------------------------
GATES = [
    # (x-center, name, invariance, question, dial, dashed)
    (3.0, "RECOGNITION\nGATE", "REGISTER",
     "the calibrated register?",
     "input-space check,\nupstream of any parse", False),
    (5.6, "5-VIEW\nVOTE", "RENDERING",
     "five retellings, one answer?",
     "unanimity → certify\nmajority → answer", False),
    (8.2, "CROSS-LINEAGE\nPANEL", "LINEAGE",
     "do sibling lineages agree?",
     "912/1500 @ 1.0000\n(at freeze)", False),
    (10.7, "ANSWER\nKEY", "TRUTH",
     "is it true?",
     "measurement only —\ngrades the machinery", True),
]
BOX_W, BOX_BOT, BOX_TOP = 1.25, 1.0, 4.6

for x, name, inv, q, dial, dashed in GATES:
    box = FancyBboxPatch(
        (x - BOX_W / 2, BOX_BOT), BOX_W, BOX_TOP - BOX_BOT,
        boxstyle="round,pad=0.03,rounding_size=0.10",
        facecolor=C["gate"], alpha=0.10, edgecolor=C["gate"],
        linewidth=1.4, linestyle=(0, (4, 2)) if dashed else "solid",
    )
    ax.add_patch(box)
    ax.text(x, 5.58, inv, ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=C["gate"])
    ax.text(x, 5.24, q, ha="center", va="center", fontsize=7.0,
            style="italic", color=C["ink"])
    ax.text(x, BOX_TOP - 0.14, name, ha="center", va="top",
            fontsize=8.5, fontweight="bold", color=C["gate"])
    ax.text(x, 0.68, dial, ha="center", va="top", fontsize=6.8,
            color=C["faint"])

# ---- trajectory helpers ---------------------------------------------------
X0, X1 = 0.85, 11.85


def lane(y, color, x_end=X1, x_start=X0, ls="solid", alpha=1.0, arrow=True):
    if arrow:
        ax.annotate("", xy=(x_end, y), xytext=(x_start, y),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=1.8, ls=ls, alpha=alpha,
                                    shrinkA=0, shrinkB=0))
    else:
        ax.plot([x_start, x_end], [y, y], color=color, lw=1.8,
                ls=ls, alpha=alpha, solid_capstyle="round")


def die(x, y, label, dy=-0.34):
    ax.scatter([x], [y], marker="x", s=110, linewidths=2.6,
               color=C["kill"], zorder=5)
    ax.text(x, y + dy, label, ha="center",
            va="top" if dy < 0 else "bottom",
            fontsize=6.8, color=C["kill"])


def name_lane(y, text, color):
    ax.text(X0 + 0.05, y + 0.11, text, ha="left", va="bottom",
            fontsize=7.4, color=color)


# ---- 1. the certified item: runs the whole chain --------------------------
y = 3.8
lane(y, C["ok"])
name_lane(y, "in-register item", C["ok"])
ax.text(X1 + 0.06, y, "CERTIFIED\n1.0000 measured\n(zero-numerator, §11)",
        ha="left", va="center", fontsize=7.2, color=C["ok"],
        fontweight="bold", clip_on=False)

# ---- 2. the broken certificate: dies at the key ----------------------------
y = 3.25
lane(y, C["ink"], x_end=10.7, arrow=False)
name_lane(y, "the one broken certificate (570R / 1W, first measurement)",
          C["ink"])
die(10.7, y, "caught by the key —\nfed back into the design")

# ---- 3. wild-register prose: slips the mouth, dies at the panel ------------
y = 2.5
lane(y, C["wild"], x_end=8.2, arrow=False)
name_lane(y, "wild-register prose that slips the gate, votes stably",
          C["wild"])
die(8.2, y, "panel dissents 9 / 10\n(the second wall)")

# ---- 4. [78]: 3/5 stable-wrong — answer dial splits from certify dial ------
y = 1.75
lane(y, C["alt"], x_end=6.2, arrow=False)
name_lane(y, "[78]: consistent wrong answer, 3 of 5 views", C["alt"])
ax.plot([6.2, 6.9, 7.3], [y, 0.16, 0.16], color=C["alt"], lw=1.8,
        solid_capstyle="round")
ax.annotate("", xy=(X1, 0.16), xytext=(7.3, 0.16),
            arrowprops=dict(arrowstyle="-|>", color=C["alt"], lw=1.8,
                            shrinkA=0, shrinkB=0))
ax.text(5.6, y + 0.13, "majority, not unanimity", ha="center", va="bottom",
        fontsize=6.8, color=C["alt"])
ax.text(X1 + 0.06, 0.35, "ANSWERED,\nNOT CERTIFIED\n(0.833 precision)",
        ha="left", va="center", fontsize=7.2, color=C["alt"], clip_on=False)

# ---- 5. [71]: dies at the mouth; ghost shows what the vote would have done -
y = 1.2
lane(y, C["kill"], x_end=3.0, arrow=False)
name_lane(y, "[71]: foreign prose", C["kill"])
ax.scatter([3.0], [y], marker="x", s=110, linewidths=2.6,
           color=C["kill"], zorder=5)
ax.text(3.0, y - 0.24, "read as foreign — intercepted upstream",
        ha="center", va="top", fontsize=6.8, color=C["kill"])
lane(y, C["kill"], x_start=3.25, x_end=6.4, ls=(0, (3, 3)), alpha=0.45,
     arrow=False)
ax.text(4.85, y + 0.13, "would vote 5/5 — unanimous, wrong", ha="center",
        va="bottom", fontsize=6.8, color=C["kill"], alpha=0.75,
        style="italic")

meta = figstyle.stamp(fig, fixtures=["bigtest-1500", "census-100"])
figstyle.save(fig, "f7c_chain_of_custody", meta=meta,
              fixtures=["bigtest-1500", "census-100"])
