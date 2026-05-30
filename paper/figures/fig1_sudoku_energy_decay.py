"""
Figure 1: Constraint energy decay on Sudoku (K-sweep)
Data from /home/bryce/mycelium/.cache/v98_ksweep/K*.log
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
import os

# --- Data extracted from v98_ksweep K*.log ---
# K values tested
K_vals = [1, 3, 5, 8, 12, 15, 18, 20]

# avg_viol (easy) from each log
avg_viol_easy = [20.957, 7.198, 3.476, 1.808, 1.061, 0.856, 0.750, 0.706]

# puzzle accuracy (easy) from each log
puzzle_acc_easy = [0.000, 0.100, 0.335, 0.560, 0.725, 0.750, 0.770, 0.790]

# --- Exponential fit with floor: E(K) = A * exp(-r * K) + C ---
# The data converges to a floor (~0.7) rather than decaying to 0.
# This is physically correct: the residual at K=20 represents the hard unsolved puzzles.
def exp_decay_floor(K, A, r, C):
    return A * np.exp(-r * np.array(K, dtype=float)) + C

popt, _ = curve_fit(exp_decay_floor, K_vals, avg_viol_easy,
                    p0=[22.0, 0.5, 0.7], bounds=([0, 0, 0], [100, 5, 2]))
A_fit, r_fit, C_fit = popt
print(f"Fit: A = {A_fit:.2f}, r = {r_fit:.4f}, floor C = {C_fit:.3f}")

K_fine = np.linspace(0.5, 21, 300)
E_fit = exp_decay_floor(K_fine, A_fit, r_fit, C_fit)

# --- R² in linear space ---
pred = exp_decay_floor(np.array(K_vals), A_fit, r_fit, C_fit)
ss_res = np.sum((np.array(avg_viol_easy) - pred) ** 2)
ss_tot = np.sum((np.array(avg_viol_easy) - np.mean(avg_viol_easy)) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"R² = {r2:.4f}")

# The paper states "rate ~0.18 nats/breath" = log-linear slope over full K=1→20 range
rate_endpoint = np.log(avg_viol_easy[0] / avg_viol_easy[-1]) / (K_vals[-1] - K_vals[0])
print(f"Endpoint-anchored rate (K=1→K=20): {rate_endpoint:.4f} nats/breath")

# --- Plot ---
fig, ax1 = plt.subplots(figsize=(6.0, 4.2))

# Left axis: constraint energy (log scale)
color_energy = "#2166ac"
color_fit    = "#1a1a2e"
ax1.semilogy(K_vals, avg_viol_easy, "o", color=color_energy, ms=6, zorder=5,
             label=r"Mean violation $\bar{E}_K$ (easy, $n$=200)")
ax1.semilogy(K_fine, E_fit, "--", color=color_fit, lw=1.4,
             label=rf"$E(K) = {A_fit:.1f}\cdot e^{{-{r_fit:.2f}K}} + {C_fit:.2f}$  ($R^2$={r2:.3f})")
ax1.set_xlabel("Number of breaths $K$", fontsize=11)
ax1.set_ylabel("Constraint energy (log scale)", color=color_energy, fontsize=11)
ax1.tick_params(axis="y", labelcolor=color_energy)
ax1.set_xlim(0, 21.5)
ax1.set_ylim(0.3, 35)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{x:g}" if x >= 1 else f"{x:.2f}"
))

# Right axis: puzzle accuracy (linear scale)
ax2 = ax1.twinx()
color_acc = "#d73027"
ax2.plot(K_vals, [100 * p for p in puzzle_acc_easy], "s-",
         color=color_acc, ms=5, lw=1.6, label="Puzzle accuracy % (easy)")
ax2.set_ylabel("Puzzle accuracy (%) — easy puzzles", color=color_acc, fontsize=11)
ax2.tick_params(axis="y", labelcolor=color_acc)
ax2.set_ylim(-2, 100)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           fontsize=8.5, loc="lower left", framealpha=0.9)

ax1.set_title("Constraint energy decay — Sudoku K-sweep", fontsize=12, pad=8)

# Light grid on left axis
ax1.grid(True, which="major", axis="y", ls=":", alpha=0.5)
ax1.grid(True, which="minor", axis="y", ls=":", alpha=0.25)
ax1.set_xticks(K_vals)

fig.tight_layout()

out_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out_dir, "fig1_sudoku_energy_decay.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(out_dir, "fig1_sudoku_energy_decay.pdf"), bbox_inches="tight")
print("Saved fig1_sudoku_energy_decay.{png,pdf}")
