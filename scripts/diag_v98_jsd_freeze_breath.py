"""v98 Sudoku K=20 JSD trace — disambiguates uniform-clock vs intrinsic-BP freeze.

Per Bryce's Jun 10 catch on the success-contrast finding: smooth exponential
decay to a fixed point is also what damped BP iteration intrinsically looks
like — the ladder isn't the only candidate author of that curve. The
discriminating test exists in the checkpoint archive: v98 Sudoku trained at
K=20 with the same per-breath weighted CE ladder.

If beliefs freeze at k≈10 (half of K_max, proportional) → convergence clock
is set by supervision schedule. Ladder-squashes-dynamics story confirmed.

If beliefs freeze at absolute k≈4 regardless of K=20 → freeze is intrinsic
BP dynamics. Supervision story mostly dies; need a different reading.

Eval-only on v98_prod_final.safetensors. ~50 puzzles, ~30s per puzzle
unbatched but K=20 instead of K=8 so more breaths per puzzle.

Usage:
  .venv/bin/python scripts/diag_v98_jsd_freeze_breath.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("SUDOKU_TASK", "1")
os.environ.setdefault("SUDOKU_K_MAX", "20")

import numpy as np
from tinygrad import Tensor, dtypes

from mycelium import Config, BreathingTransformer
from mycelium.loader import _load_state, load_breathing
from mycelium.sudoku import attach_sudoku_params, sudoku_breathing_forward
from mycelium.sudoku_data import SudokuLoader
from scripts.sudoku_train import load_ckpt as load_sudoku_ckpt


def main():
    CKPT = os.environ.get(
        "CKPT", ".cache/sudoku_ckpts/v98_prod_final.safetensors"
    )
    VAL = os.environ.get("VAL", ".cache/sudoku_val.jsonl")
    N_PUZZLES = int(os.environ.get("N_PUZZLES", "50"))
    DIFFICULTY = os.environ.get("DIFFICULTY", "medium")
    K = 20
    EPS = 1e-10

    print("=" * 60)
    print("v98 Sudoku JSD freeze-breath probe (K=20)")
    print("=" * 60)
    print(f"  ckpt:        {CKPT}")
    print(f"  val:         {VAL}")
    print(f"  difficulty:  {DIFFICULTY}")
    print(f"  N_puzzles:   {N_PUZZLES}")
    print(f"  K:           {K}")
    print()
    print("  Pre-committed read:")
    print(f"    freeze at k ≈ {K // 2} (half of K_max) → ladder/supervision clock")
    print(f"    freeze at k ≈ 4 (absolute)           → intrinsic BP damping")
    print()

    cfg = Config()
    print("loading Pythia + sudoku params...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    attach_sudoku_params(model, hidden=cfg.hidden, n_heads=cfg.n_heads, k_max=K)
    load_sudoku_ckpt(model, CKPT)

    loader = SudokuLoader(
        VAL, batch_size=1,
        difficulty_filter=DIFFICULTY,
        curriculum=False, seed=42,
    )

    Tensor.training = False

    # Per-puzzle JSD trajectories, split by final-breath correctness
    trajs_correct = []  # each is list of K-1 JSD values, mean over correct cells
    trajs_wrong = []
    trajs_all = []

    t0 = time.time()
    n_seen = 0
    for input_cells, gold, picks in loader.iter_eval(batch_size=1):
        if n_seen >= N_PUZZLES:
            break

        cell_logits_history, _calib_lh = sudoku_breathing_forward(model, input_cells, K=K)
        # Each cell_logits_k: (1, 81, 9) — already covers ALL cells of the 9x9 grid
        # Convert to softmax distributions per breath
        per_breath_probs = []
        for cl_k in cell_logits_history:
            cl_np = cl_k.realize().numpy()           # (1, 81, 9)
            cl_shift = cl_np - cl_np.max(axis=-1, keepdims=True)
            cl_exp = np.exp(cl_shift)
            probs_k = cl_exp / cl_exp.sum(axis=-1, keepdims=True)  # (1, 81, 9)
            per_breath_probs.append(probs_k)

        # Final breath: argmax → predicted digit (1..9, shifted by +1)
        final_probs = per_breath_probs[-1]                       # (1, 81, 9)
        pred_digits = final_probs.argmax(axis=-1) + 1            # (1, 81), in 1..9
        gold_np = gold.numpy()                                    # (1, 81), in 1..9
        ic_np = input_cells.numpy()                               # (1, 81), 0=unknown, 1..9 given
        unobs_mask = (ic_np == 0)                                 # (1, 81), only blanks

        # Per-cell correctness at final breath
        cell_eq = (pred_digits == gold_np)                        # (1, 81)
        unobs_correct = (cell_eq & unobs_mask).flatten()
        unobs_wrong = ((~cell_eq) & unobs_mask).flatten()
        unobs_mask_flat = unobs_mask.flatten()
        n_unobs_correct = int(unobs_correct.sum())
        n_unobs_wrong = int(unobs_wrong.sum())

        # JSD between consecutive breaths, per cell
        jsd_traj_correct = []
        jsd_traj_wrong = []
        jsd_traj_all = []
        for k in range(K - 1):
            p = per_breath_probs[k] + EPS
            q = per_breath_probs[k + 1] + EPS
            p /= p.sum(axis=-1, keepdims=True)
            q /= q.sum(axis=-1, keepdims=True)
            m = 0.5 * (p + q)
            kl_pm = (p * np.log(p / m)).sum(axis=-1)              # (1, 81)
            kl_qm = (q * np.log(q / m)).sum(axis=-1)
            jsd_per_cell_flat = (0.5 * kl_pm + 0.5 * kl_qm).flatten()  # (81,)

            jsd_traj_all.append(
                float(jsd_per_cell_flat[unobs_mask_flat].mean())
                if unobs_mask_flat.any() else 0.0
            )
            if n_unobs_correct > 0:
                jsd_traj_correct.append(
                    float(jsd_per_cell_flat[unobs_correct].mean()))
            if n_unobs_wrong > 0:
                jsd_traj_wrong.append(
                    float(jsd_per_cell_flat[unobs_wrong].mean()))

        trajs_all.append(jsd_traj_all)
        if jsd_traj_correct and len(jsd_traj_correct) == K - 1:
            trajs_correct.append(jsd_traj_correct)
        if jsd_traj_wrong and len(jsd_traj_wrong) == K - 1:
            trajs_wrong.append(jsd_traj_wrong)

        n_seen += 1
        if n_seen % 10 == 0 or n_seen == 1:
            dt = time.time() - t0
            print(f"  [{n_seen:3d}/{N_PUZZLES}] elapsed={dt:.0f}s "
                  f"({dt/n_seen:.1f}s/puzzle)", flush=True)

    print()
    print("=" * 60)
    print(f"Aggregate JSD trajectories (n={n_seen} puzzles, K={K})")
    print("=" * 60)

    def report(label, trajs):
        if not trajs:
            print(f"  {label}: no data")
            return None, None
        means = [sum(t[k] for t in trajs) / len(trajs) for k in range(K - 1)]
        # Find freeze breath: first k where JSD ≤ 0.001 (or 1% of early)
        threshold = max(0.001, means[0] * 0.05)
        freeze_k = next((k for k, v in enumerate(means) if v <= threshold), K - 2)
        early = means[0]
        late = means[-1]
        decay = late / max(early, 1e-9)
        print(f"  {label}: n_puzzles={len(trajs)}")
        # Print full trajectory in chunks of 5
        traj_str_chunks = []
        for start in range(0, K - 1, 5):
            chunk = means[start:start + 5]
            traj_str_chunks.append(" ".join(f"{v:.4f}" for v in chunk))
        print(f"    JSD[B0→B1..B{K-2}→B{K-1}]:")
        for i, chunk in enumerate(traj_str_chunks):
            print(f"      B{i*5}-B{i*5+4}: {chunk}")
        print(f"    freeze breath: k={freeze_k}  (first k where JSD ≤ {threshold:.4f})")
        print(f"    early JSD: {early:.4f}  late JSD: {late:.4f}  decay: {decay:.2%}")
        return freeze_k, means

    fk_all, m_all = report("all unobs cells", trajs_all)
    print()
    fk_corr, m_corr = report("correct cells", trajs_correct)
    print()
    fk_wrong, m_wrong = report("wrong cells", trajs_wrong)
    print()

    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    if fk_all is not None:
        if fk_all <= 6:
            print(f"  Freeze at k={fk_all} (absolute, near v110-step3's k=4):")
            print(f"  → BELIEFS FREEZE INTRINSICALLY in ~4-6 breaths regardless of K_max")
            print(f"  → Supervision-clock hypothesis MOSTLY REFUTED")
            print(f"  → MC-BP-fights-freezing reading stands; cause is BP damping not ladder")
        elif fk_all >= K // 2 - 2:
            print(f"  Freeze at k={fk_all} (≈ K/2 = {K // 2}, proportional):")
            print(f"  → CONVERGENCE CLOCK SET BY SUPERVISION SCHEDULE")
            print(f"  → Ladder-squashes-dynamics CONFIRMED")
            print(f"  → Per-cell adaptive supervision becomes the live direction")
        else:
            print(f"  Freeze at k={fk_all} (between absolute~4 and K/2={K // 2}):")
            print(f"  → MIXED — both factors plausibly contribute")
            print(f"  → Compare correct (k={fk_corr}) vs wrong (k={fk_wrong}) freeze")
            print(f"    if both correct & wrong scale ~K/2 → supervision")
            print(f"    if both freeze near k=4 → intrinsic damping")


if __name__ == "__main__":
    main()
