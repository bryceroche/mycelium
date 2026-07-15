"""F-3a — the binding theorem, by specimen (no schematic spaces).

Four real items, prose beside wiring. Top block: census [45] and [7]
share a frame (rate) and no graph class — surface kinship is not graph
kinship. Bottom block: bigtest[1187] and vtest[116] share one canonical
graph digest and no surface — graph kinship is not surface kinship
(found by the hash audit across fixture boundaries). The concept is the
binding; the caption is the theorem.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch

import figstyle
from figstyle import PALETTE as C

figstyle.apply_style()

fig, ax = plt.subplots(figsize=(11.0, 7.2))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10.4)
ax.axis("off")

PROSE_X, PROSE_W = 0.55, 6.5
GRAPH_X = 7.6


def prose_box(y, h, text, tag, tagcolor):
    box = FancyBboxPatch((PROSE_X, y), PROSE_W, h,
                         boxstyle="round,pad=0.03,rounding_size=0.08",
                         facecolor=tagcolor, alpha=0.06,
                         edgecolor=tagcolor, linewidth=1.0)
    ax.add_patch(box)
    ax.text(PROSE_X + 0.18, y + h - 0.16, text, fontsize=7.6,
            color=C["ink"], va="top", wrap=True)
    ax.text(PROSE_X + PROSE_W - 0.12, y + 0.13, tag, fontsize=6.6,
            color=tagcolor, ha="right", fontweight="bold")


def var(ax, x, y, letter, given=None):
    ax.add_patch(Circle((x, y), 0.16, facecolor="white",
                        edgecolor=C["ink"], lw=1.0, zorder=3))
    ax.text(x, y, letter, ha="center", va="center", fontsize=7, zorder=4)
    if given is not None:
        ax.add_patch(Rectangle((x - 0.15, y + 0.24), 0.30, 0.24,
                               facecolor=C["faint"], alpha=0.30, zorder=3))
        ax.text(x, y + 0.36, given, ha="center", va="center",
                fontsize=5.6, color=C["ink"], zorder=4)


def factor(ax, x, y, op, color):
    ax.add_patch(Rectangle((x - 0.13, y - 0.13), 0.26, 0.26,
                           facecolor=color, alpha=0.85, zorder=3))
    ax.text(x, y, op, ha="center", va="center", fontsize=7.5,
            color="white", zorder=4, fontweight="bold")


def edge(ax, x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color=C["ink"], lw=0.8, zorder=2)


def graph_45(ox, oy):
    """given a, given b, a×b=c — the 3-var chain."""
    var(ax, ox, oy + 0.55, "a", "24")
    var(ax, ox, oy - 0.55, "b", "7")
    factor(ax, ox + 0.85, oy, "×", C["gate"])
    var(ax, ox + 1.7, oy, "c", None)
    edge(ax, ox + 0.16, oy + 0.5, ox + 0.75, oy + 0.1)
    edge(ax, ox + 0.16, oy - 0.5, ox + 0.75, oy - 0.1)
    edge(ax, ox + 0.98, oy, ox + 1.54, oy)
    ax.text(ox + 0.85, oy - 1.05, "3 vars · {given, given, mul}",
            ha="center", fontsize=6.4, color=C["faint"])


def graph_7(ox, oy):
    """given a,b,d; a×b=c; d×e=c — the 5-var double-mul junction."""
    var(ax, ox, oy + 0.55, "a", "3")
    var(ax, ox, oy - 0.55, "b", "60")
    factor(ax, ox + 0.8, oy, "×", C["gate"])
    var(ax, ox + 1.55, oy, "c", None)
    factor(ax, ox + 2.3, oy, "×", C["gate"])
    var(ax, ox + 3.1, oy + 0.55, "d", "4")
    var(ax, ox + 3.1, oy - 0.55, "e", None)
    edge(ax, ox + 0.16, oy + 0.5, ox + 0.7, oy + 0.1)
    edge(ax, ox + 0.16, oy - 0.5, ox + 0.7, oy - 0.1)
    edge(ax, ox + 0.93, oy, ox + 1.39, oy)
    edge(ax, ox + 1.71, oy, ox + 2.17, oy)
    edge(ax, ox + 2.43, oy + 0.1, ox + 2.95, oy + 0.5)
    edge(ax, ox + 2.43, oy - 0.1, ox + 2.95, oy - 0.5)
    ax.text(ox + 1.55, oy - 1.05, "5 vars · {given×3, mul, mul} junction",
            ha="center", fontsize=6.4, color=C["faint"])


def graph_knot(ox, oy, letters, givens, query, sig_dy=-1.05):
    """the shared knot: two operands -> [+] and [−] -> two given results."""
    p, q, r, s = letters
    var(ax, ox, oy + 0.55, p, None)
    var(ax, ox, oy - 0.55, q, None)
    factor(ax, ox + 0.85, oy + 0.55, "+", C["cool"])
    factor(ax, ox + 0.85, oy - 0.55, "−", C["cool"])
    var(ax, ox + 1.7, oy + 0.55, r, givens[0])
    var(ax, ox + 1.7, oy - 0.55, s, givens[1])
    edge(ax, ox + 0.16, oy + 0.55, ox + 0.72, oy + 0.55)
    edge(ax, ox + 0.16, oy - 0.55, ox + 0.72, oy - 0.55)
    edge(ax, ox + 0.13, oy + 0.44, ox + 0.74, oy - 0.46)
    edge(ax, ox + 0.13, oy - 0.44, ox + 0.74, oy + 0.46)
    edge(ax, ox + 0.98, oy + 0.55, ox + 1.54, oy + 0.55)
    edge(ax, ox + 0.98, oy - 0.55, ox + 1.54, oy - 0.55)
    ax.text(ox + 0.85, oy + sig_dy,
            f"4 vars · {{rel+, rel−, given, given}} · query {query}",
            ha="center", fontsize=6.4, color=C["faint"])


# ================= BLOCK A: same frame, different knots =================
ax.text(0.55, 10.05, "Same frame, different knots",
        fontsize=10, fontweight="bold", color=C["ink"])
ax.text(0.55, 9.72, "surface kinship is not graph kinship (the C2 split)",
        fontsize=7.6, color=C["faint"], style="italic")

prose_box(8.1, 1.35,
          "census [45] — “After traveling 50 miles by taxi, Ann is\n"
          "charged a fare of $120. Assuming the taxi fare is directly\n"
          "proportional to distance traveled, how much would Ann be\n"
          "charged if she traveled 70 miles?”",
          "RATE frame", C["wild"])
graph_45(GRAPH_X + 0.3, 8.85)

prose_box(5.95, 1.35,
          "census [7] — “Three faucets fill a 100-gallon tub in\n"
          "6 minutes. How long, in seconds, does it take six faucets\n"
          "to fill a 25-gallon tub? Assume that all faucets dispense\n"
          "water at the same rate.”",
          "RATE frame", C["wild"])
graph_7(GRAPH_X, 6.6)

ax.text(11.6, 7.8, "no shared\ngraph class\nat n = 94", ha="center",
        fontsize=7.2, color=C["kill"], fontweight="bold")
ax.text(3.8, 7.62, "same frame (z = −2.05 at the frozen trunk)",
        ha="center", fontsize=6.6, color=C["wild"])

# ================= BLOCK B: same knot, different surfaces ===============
ax.text(0.55, 4.9, "Same knot, different surfaces",
        fontsize=10, fontweight="bold", color=C["ink"])
ax.text(0.55, 4.57,
        "graph kinship is not surface kinship (the hash audit's "
        "cross-fixture isomorphs)", fontsize=7.6, color=C["faint"],
        style="italic")

prose_box(2.95, 1.35,
          "bigtest[1187] — “Let c, a, d, b be whole numbers. the\n"
          "fourth number has the value 16. The difference between c\n"
          "and a is b. Work carefully through each fact. d has the\n"
          "value 50. d is what you get from c plus a. What is c?”",
          "terse register", C["alt"])
graph_knot(GRAPH_X + 0.3, 3.7, ("c", "a", "d", "b"), ("50", "16"), "c",
           sig_dy=1.0)

prose_box(1.15, 1.6,
          "vtest[116] — “In this problem we are working with the\n"
          "numbers a, b, c, d. If a and b are gathered into one pile,\n"
          "that pile amounts to c. If you look at d, you will find that\n"
          "its value is exactly 16. It turns out that c works out to be\n"
          "50 in the end. If you start from a and take away everything\n"
          "b has, what remains is d. … what value does a end up holding?”",
          "verbose register", C["alt"])
graph_knot(GRAPH_X + 0.3, 1.75, ("a", "b", "c", "d"), ("50", "16"), "a")

ax.text(11.6, 2.75, "one canonical\ndigest:\n468be959…", ha="center",
        fontsize=7.2, color=C["ok"], fontweight="bold",
        family="DejaVu Sans Mono")

ax.text(6.0, 0.28,
        "The concept is the binding between a linguistic frame and a "
        "structural role — recoverable from neither side alone.",
        ha="center", fontsize=8.2, color=C["ink"], style="italic")

meta = figstyle.stamp(fig, fixtures=["book1", "iso_contamination"])
figstyle.save(fig, "f3a_binding_theorem", meta=meta,
              fixtures=["book1.jsonl", "iso_contamination.json",
                        "algebra_nl_bigtest.jsonl", "algv_test_verbose.jsonl"])
