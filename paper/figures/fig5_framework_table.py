"""
Figure 5: 2x2 framework table — rhythm × topology
Conceptual figure rendered as a styled matplotlib table.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

fig, ax = plt.subplots(figsize=(8.0, 4.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

# --- Color palette ---
col_header_bg = "#2c3e50"
col_header_fg = "white"
row_header_bg = "#34495e"
row_header_fg = "white"
success_bg    = "#d4edda"  # light green
failure_bg    = "#f8d7da"  # light red
future_bg     = "#fff3cd"  # light yellow
border_color  = "#aaa"

# Grid layout: rows 0-1 are data rows, col 0 is row header, cols 1-2 are data cols
# Bounding box coordinates: (x0, y0, width, height)
# Table spans x: 0.5-9.5, y: 0.5-5.5

# Column positions
x0 = 0.5
col_widths = [3.0, 3.0, 3.0]
row_heights = [1.2, 1.6, 1.6]

# y positions (top to bottom: header row, row1, row2)
y_top = 5.5

# --- Draw cells ---
def draw_cell(ax, x, y, w, h, bg, text, fontsize=10, bold=False, fg="black",
              valign="center", extra_lines=None):
    rect = FancyBboxPatch((x + 0.04, y - h + 0.04), w - 0.08, h - 0.08,
                           boxstyle="round,pad=0.05",
                           linewidth=1.2, edgecolor=border_color,
                           facecolor=bg, zorder=2)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    cy = y - h / 2
    ax.text(x + w / 2, cy, text, ha="center", va=valign,
            fontsize=fontsize, color=fg, fontweight=weight,
            wrap=False, zorder=3, multialignment="center")
    if extra_lines:
        for eline, esize, ecolor in extra_lines:
            ax.text(x + w / 2, cy - fontsize * 0.038 * (12 - fontsize + 2),
                    eline, ha="center", va="top", fontsize=esize,
                    color=ecolor, zorder=3)

# Row 0 = header
# Corner (empty)
draw_cell(ax, x0, y_top, col_widths[0], row_heights[0],
          col_header_bg, "", fontsize=9, bold=True, fg=col_header_fg)
# Column headers
draw_cell(ax, x0 + col_widths[0], y_top, col_widths[1], row_heights[0],
          col_header_bg, "Cyclic graph\n(Sudoku, n-queens,\ngraph coloring)",
          fontsize=9, bold=True, fg=col_header_fg)
draw_cell(ax, x0 + col_widths[0] + col_widths[1], y_top, col_widths[2], row_heights[0],
          col_header_bg, "Tree graph\n(arithmetic DAGs,\ndependency graphs)",
          fontsize=9, bold=True, fg=col_header_fg)

# Row 1 = Rotational rhythm
y1 = y_top - row_heights[0]
draw_cell(ax, x0, y1, col_widths[0], row_heights[1],
          row_header_bg, "Rotational rhythm\n(π-cycled RoPE)",
          fontsize=9, bold=True, fg=row_header_fg)
draw_cell(ax, x0 + col_widths[0], y1, col_widths[1], row_heights[1],
          success_bg,
          "v98  ✓\n79% puzzle accuracy\nExponential energy decay\n(rate ≈ 0.18 nats/breath)",
          fontsize=8.5)
draw_cell(ax, x0 + col_widths[0] + col_widths[1], y1, col_widths[2], row_heights[1],
          failure_bg,
          "v99  ✗\n9% (chance floor)\nFlat CE ladder\nNo depth correlation",
          fontsize=8.5)

# Row 2 = Topological rhythm
y2 = y1 - row_heights[1]
draw_cell(ax, x0, y2, col_widths[0], row_heights[2],
          row_header_bg, "Topological rhythm\n(staged masks)",
          fontsize=9, bold=True, fg=row_header_fg)
draw_cell(ax, x0 + col_widths[0] + col_widths[1], y2, col_widths[2], row_heights[2],
          success_bg,
          "v100/v101/v103  ✓\n40–48% cell accuracy\n+18 pt K=1→K=10 gap\nDepth-correlated",
          fontsize=8.5)
draw_cell(ax, x0 + col_widths[0], y2, col_widths[1], row_heights[2],
          future_bg,
          "Future work\n(Prediction: fails —\ncyclic graphs have\nno natural DAG order)",
          fontsize=8.5, fg="#7d4e00")

# Title
ax.text(5.0, 5.85, "Rhythm × Topology: which key fits which lock",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#1a1a2e")

# Legend patches
p_success = mpatches.Patch(facecolor=success_bg, edgecolor="#555", label="Success (diagonal)")
p_failure = mpatches.Patch(facecolor=failure_bg, edgecolor="#555", label="Failure (off-diagonal)")
p_future  = mpatches.Patch(facecolor=future_bg, edgecolor="#555", label="Untested prediction")
ax.legend(handles=[p_success, p_failure, p_future], loc="lower center",
          bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=8, framealpha=0.95)

fig.tight_layout(pad=0.3)

out_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out_dir, "fig5_framework_table.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(out_dir, "fig5_framework_table.pdf"), bbox_inches="tight")
print("Saved fig5_framework_table.{png,pdf}")
