"""Hard failure-mode bucketing diagnostic (Jun 9).

For 50 hard puzzles, capture per-breath calib trajectory + per-cell predictions
against cont8_step1000 (and optionally v112b_cont1_final). Cluster failures
into calib-signature buckets to decide direction for Phase 2 / pivot:

  STALL (low calib throughout K)        → depth/K — train at K=12 worth it
  CONFIDENT-WRONG (early high calib +   → supervision/landscape — Hessian
    wrong attractor)                       conditioning thread
  PARSE-LEVEL (wrong from breath 0)     → input representation
  NEAR-MISS (1-2 cells wrong, anywhere) → Phase 2 (allocation) may still help
  DISTRIBUTED (many cells wrong,        → structural — Phase 2 is wrong queue
    no near-misses)

Doubles as first real test of calib head's Dopri5 framing — does it carry
diagnostic information that separates stalls from confident-wrong commitments?

Usage:
  CKPT=.cache/fg_v110_step3_ckpts/v110_step3_cont8_step1000.safetensors \
  .venv/bin/python scripts/diag_hard_failure_buckets.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("V110_STEP3_TASK", "1")
os.environ.setdefault("V110_STEP3_K_MAX", "8")
os.environ.setdefault("V110_STEP3_N_DIGITS", "5")
os.environ.setdefault("V110_STEP3_N_MAX", "16")
os.environ.setdefault("V110_STEP3_F_MAX", "8")
os.environ.setdefault("V110_STEP3_WAIST_DIM", "512")
os.environ.setdefault("V110_STEP3_ALTERNATION", "1")
os.environ.setdefault("V110_STEP3_PHASE_SCALE", "1.0")

import numpy as np
from tinygrad import Device, Tensor, dtypes

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.factor_graph_v110_step3 import (
    V110_STEP3_N_MAX, V110_STEP3_F_MAX, V110_STEP3_N_HEADS,
    V110_STEP3_N_DIGITS, V110_STEP3_K_MAX,
    V110_STEP3_WAIST_DIM, V110_STEP3_ALTERNATION, V110_STEP3_PHASE_SCALE,
    V110_STEP3_GATE_PROFILE, V110_STEP3_PHOTON_ALPHA,
    V110_STEP3_CODEBOOK_N, V110_STEP3_IB_CENTROIDS,
    attach_fg_params_v110_step3,
)
from mycelium.factor_graph_v110_step import fg_breathing_forward_v110_step
from scripts.v110_step3_factor_graph_train import load_ckpt_v110_step3, cast_layers_fp32
from mycelium.factor_graph_data_v107 import (
    FactorGraphLoaderV107, _records_to_batch_v107,
)
from mycelium.factor_graph_v108 import bins_to_digits_msd


def _batch_to_tensors(batch_np):
    return (
        Tensor(batch_np["domain_init"]).cast(dtypes.float).contiguous().realize(),
        Tensor(batch_np["node_kinds"]).cast(dtypes.int).contiguous().realize(),
        Tensor(batch_np["staging_mask"]).cast(dtypes.float).contiguous().realize(),
        Tensor(batch_np["head_op_mask"]).cast(dtypes.float).contiguous().realize(),
    )


def classify_failure(per_breath_calib, n_wrong, n_obs, calib_at_breath0_per_cell=None,
                     calib_stall_thresh=0.35, calib_commit_thresh=0.7,
                     near_miss_max=2):
    """Classify a single-puzzle failure into a bucket.

    per_breath_calib: list of K floats (mean calib across the batch, K=8)
    n_wrong: int — wrong cells (unobserved)
    n_obs: int — total unobserved cells
    """
    K = len(per_breath_calib)
    calib_max = max(per_breath_calib)
    calib_final = per_breath_calib[-1]
    # First breath at or above commit threshold
    commit_breath = next(
        (k for k, c in enumerate(per_breath_calib) if c >= calib_commit_thresh),
        -1,
    )

    # NEAR-MISS: 1 or 2 cells wrong regardless of calib pattern
    if n_wrong <= near_miss_max:
        return "near_miss"

    # STALL: calib stays low (max < stall_thresh) — model knows it didn't converge
    if calib_max < calib_stall_thresh:
        return "stall"

    # CONFIDENT-WRONG: calib commits early (k < K/2) at high value, output wrong
    if 0 <= commit_breath < K // 2 and calib_final >= calib_commit_thresh:
        return "confident_wrong"

    # DISTRIBUTED: many cells wrong, no clear calib signature
    if n_wrong >= n_obs // 2:
        return "distributed"

    # OTHER: mid-calib commits, partial failures
    return "other"


def main():
    CKPT = os.environ.get(
        "CKPT",
        ".cache/fg_v110_step3_ckpts/v110_step3_cont8_step1000.safetensors",
    )
    VAL_PATH = os.environ.get("VAL_PATH", ".cache/factor_graph_test.jsonl")
    SEED = int(os.environ.get("SEED", "42"))
    N_PUZZLES = int(os.environ.get("N_PUZZLES", "50"))
    DIFFICULTY = os.environ.get("DIFFICULTY", "hard")

    H = 1024
    K_MAX = V110_STEP3_K_MAX

    print("=== Hard failure-mode bucketing diagnostic ===")
    print(f"  ckpt: {CKPT}")
    print(f"  difficulty: {DIFFICULTY}")
    print(f"  N_puzzles: {N_PUZZLES}")
    print(f"  K_max: {K_MAX}")
    print()

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    cfg = Config()
    print("loading Pythia-410M...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    cast_layers_fp32(model)

    attach_fg_params_v110_step3(
        model, hidden=cfg.hidden,
        n_max=V110_STEP3_N_MAX, f_max=V110_STEP3_F_MAX, k_max=K_MAX,
        n_digits=V110_STEP3_N_DIGITS, n_code=V110_STEP3_CODEBOOK_N,
        ib_centroids_path=V110_STEP3_IB_CENTROIDS,
        waist_dim=V110_STEP3_WAIST_DIM,
    )
    Device[Device.DEFAULT].synchronize()

    print(f"loading ckpt: {CKPT}")
    load_ckpt_v110_step3(model, CKPT)
    print()

    val_loader = FactorGraphLoaderV107(
        VAL_PATH, batch_size=1,
        difficulty_filter=DIFFICULTY, curriculum=False,
        n_max=V110_STEP3_N_MAX, f_max=V110_STEP3_F_MAX, k_max=K_MAX,
        n_heads=V110_STEP3_N_HEADS,
        seed=SEED + 2,
    )
    records = val_loader.records[:N_PUZZLES]
    print(f"  evaluating {len(records)} {DIFFICULTY} puzzles\n", flush=True)

    Tensor.training = False

    # Buckets + per-puzzle records for inspection
    buckets = dict(near_miss=0, stall=0, confident_wrong=0, distributed=0, other=0,
                   correct=0)
    puzzle_logs = []

    t0 = time.time()
    for i, rec in enumerate(records):
        batch_np = _records_to_batch_v107(
            [rec], n_max=V110_STEP3_N_MAX, f_max=V110_STEP3_F_MAX,
            k_max=K_MAX, n_heads=V110_STEP3_N_HEADS,
        )
        di, nk, sm, hm = _batch_to_tensors(batch_np)

        # Eager forward — returns tree_logits_history, var_logits_history,
        # factor_logits_history, calib_history.
        tree_lh, var_lh, factor_lh, calib_lh, _step_mags_lh = fg_breathing_forward_v110_step(
            model, di, nk, sm, hm,
            K=K_MAX, n_max=V110_STEP3_N_MAX, f_max=V110_STEP3_F_MAX,
            n_digits=V110_STEP3_N_DIGITS,
            alternation=V110_STEP3_ALTERNATION,
            phase_scale=V110_STEP3_PHASE_SCALE,
            gate_profile=V110_STEP3_GATE_PROFILE,
            photon_alpha=V110_STEP3_PHOTON_ALPHA,
        )

        # ── Per-breath predictions AND belief distributions across all K breaths ──
        # tree_lh: list of K tensors, each (1, n_max, n_digits, 10)
        # We need both predictions (argmax for cell-flip churn) AND softmax
        # distributions (for JSD between consecutive breaths — the proper
        # measurement of belief motion, per Jun 9 telegraph/HMM reframing).
        per_breath_preds = []
        per_breath_probs = []
        for tree_k in tree_lh:
            tree_np = tree_k.realize().numpy()             # (1, n_max, n_digits, 10)
            preds_k = tree_np.argmax(axis=-1)              # (1, n_max, n_digits)
            # Numerically stable softmax along digit-value axis
            t_shift = tree_np - tree_np.max(axis=-1, keepdims=True)
            t_exp = np.exp(t_shift)
            probs_k = t_exp / t_exp.sum(axis=-1, keepdims=True)
            per_breath_preds.append(preds_k)
            per_breath_probs.append(probs_k)
        pred_digits = per_breath_preds[-1]                # final breath

        gold_digits = bins_to_digits_msd(
            batch_np["gold_bins"].numpy() if hasattr(batch_np["gold_bins"], "numpy")
            else np.array(batch_np["gold_bins"]),
            n_digits=V110_STEP3_N_DIGITS,
        )                                                # (1, n_max, n_digits)
        obs_mask = batch_np["observed_mask"].numpy() if hasattr(batch_np["observed_mask"], "numpy") else np.array(batch_np["observed_mask"])

        # Per-cell correct: all n_digits must match
        cell_eq = (pred_digits == gold_digits).all(axis=-1)  # (1, n_max)
        unobs_mask = (1 - obs_mask).astype(bool)             # (1, n_max)
        n_obs = int(unobs_mask.sum())
        n_correct = int((cell_eq & unobs_mask).sum())
        n_wrong = n_obs - n_correct

        # Per-breath churn: count cells that FLIP between breaths k and k+1.
        # Only count unobserved cells. churn[k] = # flips between B(k) and B(k+1).
        # ALSO: JSD between consecutive breaths' belief distributions per cell.
        # The JSD is the proper measurement of belief motion (telegraph framing).
        # SUCCESS-CONTRAST (Jun 10): split JSD by final-breath per-cell correctness.
        #   correct cells (model got these right at end) vs wrong cells (model got
        #   these wrong at end). Tests whether dynamics differ for solved vs failed
        #   cells WITHIN the same puzzle — controlled comparison.
        churn = []
        jsd_churn = []           # mean across ALL unobserved cells (per-puzzle)
        jsd_churn_correct = []   # mean across unobs+correct cells (per-puzzle)
        jsd_churn_wrong = []     # mean across unobs+wrong cells (per-puzzle)
        per_breath_acc = []
        EPS = 1e-10
        unobs_mask_flat = unobs_mask.flatten()              # (n_max,)
        # cell_eq is based on FINAL-breath predictions
        unobs_correct = (cell_eq & unobs_mask).flatten()    # (n_max,)
        unobs_wrong   = ((~cell_eq) & unobs_mask).flatten() # (n_max,)
        n_unobs_correct = int(unobs_correct.sum())
        n_unobs_wrong   = int(unobs_wrong.sum())
        for k in range(len(per_breath_preds)):
            preds_k = per_breath_preds[k]
            eq_k = (preds_k == gold_digits).all(axis=-1)
            n_correct_k = int((eq_k & unobs_mask).sum())
            per_breath_acc.append(n_correct_k / max(n_obs, 1))
            if k < len(per_breath_preds) - 1:
                preds_next = per_breath_preds[k+1]
                # A cell "flips" if its prediction tuple differs from previous breath
                flip_mask = ~(preds_k == preds_next).all(axis=-1)  # (1, n_max)
                n_flips = int((flip_mask & unobs_mask).sum())
                churn.append(n_flips)

                # JSD between belief distributions at breath k vs k+1.
                # probs_k: (1, n_max, n_digits, 10); compute JSD per digit position,
                # then mean across digit positions and unobserved cells.
                p = per_breath_probs[k] + EPS
                q = per_breath_probs[k+1] + EPS
                p /= p.sum(axis=-1, keepdims=True)
                q /= q.sum(axis=-1, keepdims=True)
                m = 0.5 * (p + q)
                kl_pm = (p * np.log(p / m)).sum(axis=-1)   # (1, n_max, n_digits)
                kl_qm = (q * np.log(q / m)).sum(axis=-1)
                jsd_per_pos = 0.5 * kl_pm + 0.5 * kl_qm    # (1, n_max, n_digits)
                # Mean across digit positions per cell
                jsd_per_cell_flat = jsd_per_pos.mean(axis=-1).flatten()  # (n_max,)
                # Mean across all unobserved cells
                jsd_mean = float(jsd_per_cell_flat[unobs_mask_flat].mean()) \
                            if unobs_mask_flat.any() else 0.0
                jsd_churn.append(jsd_mean)
                # SUCCESS-CONTRAST: split by final-breath correctness
                if n_unobs_correct > 0:
                    jsd_churn_correct.append(
                        float(jsd_per_cell_flat[unobs_correct].mean()))
                else:
                    jsd_churn_correct.append(None)
                if n_unobs_wrong > 0:
                    jsd_churn_wrong.append(
                        float(jsd_per_cell_flat[unobs_wrong].mean()))
                else:
                    jsd_churn_wrong.append(None)

        # Per-breath calib trajectory (mean across batch — batch is 1 here)
        per_breath_calib = [float(c.realize().numpy().mean()) for c in calib_lh]

        if n_wrong == 0:
            bucket = "correct"
        else:
            bucket = classify_failure(per_breath_calib, n_wrong, n_obs)
        buckets[bucket] += 1

        puzzle_logs.append(dict(
            idx=i,
            n_wrong=n_wrong,
            n_obs=n_obs,
            n_unobs_correct=n_unobs_correct,
            n_unobs_wrong=n_unobs_wrong,
            calib_trajectory=per_breath_calib,
            per_breath_acc=per_breath_acc,
            churn=churn,
            jsd_churn=jsd_churn,
            jsd_churn_correct=jsd_churn_correct,
            jsd_churn_wrong=jsd_churn_wrong,
            bucket=bucket,
            difficulty=rec.get("difficulty", DIFFICULTY),
        ))

        if (i + 1) % 10 == 0 or i == 0:
            dt = time.time() - t0
            print(f"  [{i+1:3d}/{len(records)}] bucket={bucket:14s} "
                  f"wrong={n_wrong}/{n_obs} calib[0,K-1]=({per_breath_calib[0]:.2f},"
                  f"{per_breath_calib[-1]:.2f}) ({dt:.0f}s)",
                  flush=True)

    print()
    print("=" * 50)
    print(f"BUCKET COUNTS ({len(records)} puzzles)")
    print("=" * 50)
    total = sum(buckets.values())
    for b in ["correct", "near_miss", "stall", "confident_wrong",
              "distributed", "other"]:
        n = buckets[b]
        print(f"  {b:16s} {n:3d}  ({100*n/total:.1f}%)")

    print()
    print("=" * 50)
    print("CALIB TRAJECTORIES BY BUCKET (mean per breath)")
    print("=" * 50)
    for b in ["correct", "near_miss", "stall", "confident_wrong",
              "distributed", "other"]:
        traj = [log["calib_trajectory"] for log in puzzle_logs if log["bucket"] == b]
        if not traj:
            continue
        means = [sum(t[k] for t in traj) / len(traj) for k in range(K_MAX)]
        mean_str = " ".join(f"{v:.2f}" for v in means)
        print(f"  {b:16s} [B0..B{K_MAX-1}]: {mean_str}  (n={len(traj)})")

    print()
    print("=" * 50)
    print("CHURN BY BUCKET (mean cell-flips between consecutive breaths)")
    print("=" * 50)
    for b in ["near_miss", "stall", "confident_wrong", "distributed", "other"]:
        churns = [log["churn"] for log in puzzle_logs if log["bucket"] == b]
        if not churns:
            continue
        means = [sum(t[k] for t in churns) / len(churns) for k in range(K_MAX - 1)]
        mean_str = " ".join(f"{v:.2f}" for v in means)
        # Late-breath churn k=K-2 (transition into final) is the key signal.
        late_churn = means[-1] if means else 0.0
        print(f"  {b:16s} flips[B0→B1..B{K_MAX-2}→B{K_MAX-1}]: {mean_str}  "
              f"(late_churn={late_churn:.2f}, n={len(churns)})")

    print()
    print("=" * 50)
    print("JSD CHURN BY BUCKET (Jensen-Shannon between consecutive belief dists)")
    print("=" * 50)
    print("  The proper measurement of belief motion. Replaces argmax-flip count.")
    print("  Telegraph framing: HIGH late JSD = beliefs still in motion at k=K-1")
    print("                     (genuine STALL — propagation never converged)")
    print("                     LOW late JSD = beliefs frozen on (wrong) answers")
    print("                     (CONVERGED-TO-GARBAGE — capacity/paradigm needed)")
    print("  Caveat: the per-breath weighted CE ladder TRAINS the readout to")
    print("  not telegraph. Smooth decay here is partly training-shaped.")
    print()
    for b in ["near_miss", "stall", "confident_wrong", "distributed", "other"]:
        jsds = [log["jsd_churn"] for log in puzzle_logs if log["bucket"] == b]
        if not jsds:
            continue
        means = [sum(t[k] for t in jsds) / len(jsds) for k in range(K_MAX - 1)]
        mean_str = " ".join(f"{v:.3f}" for v in means)
        late_jsd = means[-1] if means else 0.0
        early_jsd = means[0] if means else 0.0
        decay = late_jsd / max(early_jsd, 1e-6)
        print(f"  {b:16s} jsd[B0→B1..B{K_MAX-2}→B{K_MAX-1}]: {mean_str}  "
              f"(late={late_jsd:.3f}, early={early_jsd:.3f}, decay={decay:.2%}, n={len(jsds)})")

    print()
    print("=" * 50)
    print("SUCCESS-CONTRAST JSD (per-cell, split by final-breath correctness)")
    print("=" * 50)
    print("  Controlled comparison: within the same puzzle, JSD trajectory on")
    print("  cells the model got RIGHT at final breath vs cells it got WRONG.")
    print("  Pre-committed reads:")
    print("    SAME smooth decay → pure landscape (where you land matters,")
    print("                                          not how you move)")
    print("    WRONG cells faster decay / smaller early → PREMATURE FREEZE")
    print("                                               (model commits early")
    print("                                                on cells it shouldn't)")
    print("    WRONG cells slower decay / larger late → STALL on hard cells only")
    print()
    for diff_filter in [None]:
        for which in ["correct", "wrong"]:
            key = f"jsd_churn_{which}"
            trajs = []
            n_cells_total = 0
            for log in puzzle_logs:
                traj = log[key]
                # Filter out Nones (some puzzles have 0 correct or 0 wrong cells)
                if any(v is None for v in traj):
                    continue
                trajs.append(traj)
                n_cells_total += log[f"n_unobs_{which}"]
            if not trajs:
                print(f"  {which:8s} (no puzzles had any '{which}' cells)")
                continue
            means = [sum(t[k] for t in trajs) / len(trajs) for k in range(K_MAX - 1)]
            mean_str = " ".join(f"{v:.3f}" for v in means)
            late_jsd = means[-1] if means else 0.0
            early_jsd = means[0] if means else 0.0
            decay = late_jsd / max(early_jsd, 1e-6)
            print(f"  {which:8s} cells  jsd[B0..B{K_MAX-2}]: {mean_str}  "
                  f"(late={late_jsd:.4f} early={early_jsd:.4f} decay={decay:.2%} "
                  f"n_puzzles={len(trajs)} n_cells={n_cells_total})")

    # Compute aggregate ratio metrics
    correct_trajs = [log["jsd_churn_correct"] for log in puzzle_logs
                      if not any(v is None for v in log["jsd_churn_correct"])]
    wrong_trajs = [log["jsd_churn_wrong"] for log in puzzle_logs
                    if not any(v is None for v in log["jsd_churn_wrong"])]
    if correct_trajs and wrong_trajs:
        early_corr = sum(t[0] for t in correct_trajs) / len(correct_trajs)
        early_wrong = sum(t[0] for t in wrong_trajs) / len(wrong_trajs)
        late_corr = sum(t[-1] for t in correct_trajs) / len(correct_trajs)
        late_wrong = sum(t[-1] for t in wrong_trajs) / len(wrong_trajs)
        print()
        print(f"  early ratio (wrong/correct): {early_wrong/max(early_corr,1e-6):.2f}")
        print(f"  late  ratio (wrong/correct): {late_wrong/max(late_corr,1e-6):.2f}")
        print()
        if early_wrong < 0.7 * early_corr:
            print("  → wrong cells start with LESS belief motion than correct cells.")
            print("    'Premature freeze' signature: model never had real uncertainty")
            print("    about cells it eventually got wrong. Reads as overconfidence")
            print("    at breath 0 — supervision intervention candidate.")
        elif early_wrong > 1.3 * early_corr:
            print("  → wrong cells start with MORE belief motion than correct cells,")
            print("    then freeze early. 'Floundering then giving up' signature.")
        else:
            print("  → wrong and correct cells move similarly. Dynamics don't")
            print("    separate success from failure → pure landscape question.")

    print()
    print("=" * 50)
    print("PER-BREATH ACCURACY BY BUCKET (cell_acc at each breath)")
    print("=" * 50)
    for b in ["near_miss", "stall", "confident_wrong", "distributed", "other"]:
        accs = [log["per_breath_acc"] for log in puzzle_logs if log["bucket"] == b]
        if not accs:
            continue
        means = [sum(t[k] for t in accs) / len(accs) for k in range(K_MAX)]
        mean_str = " ".join(f"{v:.2f}" for v in means)
        print(f"  {b:16s} acc[B0..B{K_MAX-1}]: {mean_str}  (n={len(accs)})")

    # Distributed-bucket churn interpretation
    dist_churns = [log["churn"] for log in puzzle_logs
                    if log["bucket"] == "distributed"]
    if dist_churns:
        late_churns = [c[-1] for c in dist_churns]
        early_churns = [c[0] for c in dist_churns]
        mean_late = sum(late_churns) / len(late_churns)
        mean_early = sum(early_churns) / len(early_churns)
        print()
        print("=" * 50)
        print(f"DISTRIBUTED-BUCKET CHURN INTERPRETATION (n={len(dist_churns)})")
        print("=" * 50)
        print(f"  early churn (B0→B1): mean={mean_early:.2f} flips/puzzle")
        print(f"  late  churn (BK-2→BK-1): mean={mean_late:.2f} flips/puzzle")
        ratio = mean_late / max(mean_early, 0.01)
        print(f"  late/early ratio: {ratio:.2f}")
        print()
        print(f"  Interpretation:")
        print(f"    late_churn > 1.0 → predictions still flipping at end =")
        print(f"                       STALL (propagation never converged) →")
        print(f"                       depth/K live; train-at-K=12 worth ~3hr")
        print(f"    late_churn < 0.5 → predictions frozen by end =")
        print(f"                       CONVERGED-TO-GARBAGE → capacity/paradigm")
        print(f"                       (Pythia-1B inside paradigm, or v200)")
        print(f"    0.5 ≤ late_churn ≤ 1.0 → partial → ambiguous, both possible")

    # Failure summary
    print()
    print("=" * 50)
    print("ACTION MAPPING (pre-committed Jun 9)")
    print("=" * 50)
    print(f"  Stall-dominated      → depth/K — train at K=12 worth it")
    print(f"  Confident-wrong dom. → supervision/landscape — Hessian thread")
    print(f"  Parse-level dominant → input representation (NOT TESTED HERE)")
    print(f"  Near-miss dominant   → Phase 2 (allocation) may close hard")
    print(f"  Distributed dominant → structural — Phase 2 is wrong direction")

    return buckets, puzzle_logs


if __name__ == "__main__":
    main()
