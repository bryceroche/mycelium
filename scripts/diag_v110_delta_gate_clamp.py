"""v110-step3 inference-time delta_gate clamp — causality test.

Per Bryce's Jun 10 catch: v110-step3's learned delta_gate decay
(B0=0.92 ... B7=0.68) was learned UNDER frozen-belief equilibrium, so
gate decay may be SYMPTOM not CAUSE of the freeze. v98 keeps gate >1
throughout and still freezes at k=5 — the high gate didn't prevent it.

Causality test: clamp v110-step3's delta_gate at B4-B7 to either 1.0
(neutral) or 1.27 (B2 peak — fully unsuppress late breaths) and re-run
JSD freeze-breath trace.

Verdicts:
- JSD stays frozen → gate was epiphenomenal, intrinsic damping fully
  upstream, "fighting both dampings" concern dissolves to fighting one.
- JSD re-mobilizes → gate is causally load-bearing, anti-damping via
  inference-time gate schedule becomes the cheapest mechanism in the
  project.

Eval-only on existing v110_step3_cont8_step1000.safetensors. ~2 min.

Usage:
  CLAMP_MODE=peak (default) | neutral | none
  DIFFICULTY=hard (default)
  N_PUZZLES=50 (default)
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


def main():
    CKPT = ".cache/fg_v110_step3_ckpts/v110_step3_cont8_step1000.safetensors"
    VAL = ".cache/factor_graph_test.jsonl"
    SEED = 42
    N_PUZZLES = int(os.environ.get("N_PUZZLES", "50"))
    DIFFICULTY = os.environ.get("DIFFICULTY", "hard")
    CLAMP_MODE = os.environ.get("CLAMP_MODE", "peak")  # peak | neutral | none

    K = V110_STEP3_K_MAX
    EPS = 1e-10

    print("=" * 60)
    print(f"v110-step3 delta_gate clamp causality test")
    print("=" * 60)
    print(f"  ckpt:       {CKPT}")
    print(f"  difficulty: {DIFFICULTY}")
    print(f"  N_puzzles:  {N_PUZZLES}")
    print(f"  K:          {K}")
    print(f"  clamp_mode: {CLAMP_MODE}")
    print()

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    cfg = Config()
    print("loading Pythia + v110-step3 params...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    cast_layers_fp32(model)

    attach_fg_params_v110_step3(
        model, hidden=cfg.hidden,
        n_max=V110_STEP3_N_MAX, f_max=V110_STEP3_F_MAX, k_max=K,
        n_digits=V110_STEP3_N_DIGITS, n_code=V110_STEP3_CODEBOOK_N,
        ib_centroids_path=V110_STEP3_IB_CENTROIDS,
        waist_dim=V110_STEP3_WAIST_DIM,
    )
    Device[Device.DEFAULT].synchronize()
    load_ckpt_v110_step3(model, CKPT)

    # Print original delta_gate, then clamp
    orig_dg = model.fg_v107_delta_gate.numpy().copy()
    print(f"  original delta_gate: {orig_dg}")
    if CLAMP_MODE == "peak":
        # Hold B4-B7 at B2's peak value 1.27
        clamped = orig_dg.copy()
        clamped[4:] = 1.27
    elif CLAMP_MODE == "neutral":
        # Hold B4-B7 at 1.0 (neutral)
        clamped = orig_dg.copy()
        clamped[4:] = 1.0
    elif CLAMP_MODE == "all_peak":
        # All breaths at 1.27 (max amplification)
        clamped = np.full_like(orig_dg, 1.27, dtype=np.float32)
    elif CLAMP_MODE == "none":
        # No clamp — baseline (sanity check)
        clamped = orig_dg.copy()
    else:
        raise ValueError(f"Unknown CLAMP_MODE: {CLAMP_MODE}")
    print(f"  clamped  delta_gate: {clamped}")
    model.fg_v107_delta_gate.assign(
        Tensor(clamped, dtype=dtypes.float)
    ).realize()
    print()

    val_loader = FactorGraphLoaderV107(
        VAL, batch_size=1,
        difficulty_filter=DIFFICULTY, curriculum=False,
        n_max=V110_STEP3_N_MAX, f_max=V110_STEP3_F_MAX, k_max=K,
        n_heads=V110_STEP3_N_HEADS,
        seed=SEED + 2,
    )
    records = val_loader.records[:N_PUZZLES]

    Tensor.training = False
    trajs_correct = []
    trajs_wrong = []
    trajs_all = []
    n_correct_total = 0
    n_obs_total = 0

    t0 = time.time()
    for i, rec in enumerate(records):
        batch_np = _records_to_batch_v107(
            [rec], n_max=V110_STEP3_N_MAX, f_max=V110_STEP3_F_MAX,
            k_max=K, n_heads=V110_STEP3_N_HEADS,
        )
        di, nk, sm, hm = _batch_to_tensors(batch_np)

        tree_lh, _, _, _, _ = fg_breathing_forward_v110_step(
            model, di, nk, sm, hm,
            K=K, n_max=V110_STEP3_N_MAX, f_max=V110_STEP3_F_MAX,
            n_digits=V110_STEP3_N_DIGITS,
            alternation=V110_STEP3_ALTERNATION,
            phase_scale=V110_STEP3_PHASE_SCALE,
            gate_profile=V110_STEP3_GATE_PROFILE,
            photon_alpha=V110_STEP3_PHOTON_ALPHA,
        )

        per_breath_probs = []
        for tree_k in tree_lh:
            tree_np = tree_k.realize().numpy()
            t_shift = tree_np - tree_np.max(axis=-1, keepdims=True)
            t_exp = np.exp(t_shift)
            per_breath_probs.append(t_exp / t_exp.sum(axis=-1, keepdims=True))

        final_probs = per_breath_probs[-1]
        pred_digits = final_probs.argmax(axis=-1)
        gold_digits = bins_to_digits_msd(
            batch_np["gold_bins"].numpy() if hasattr(batch_np["gold_bins"], "numpy")
            else np.array(batch_np["gold_bins"]),
            n_digits=V110_STEP3_N_DIGITS,
        )
        obs_mask = batch_np["observed_mask"].numpy() if hasattr(batch_np["observed_mask"], "numpy") else np.array(batch_np["observed_mask"])
        cell_eq = (pred_digits == gold_digits).all(axis=-1)
        unobs_mask = (1 - obs_mask).astype(bool)
        unobs_correct = (cell_eq & unobs_mask).flatten()
        unobs_wrong = ((~cell_eq) & unobs_mask).flatten()
        unobs_mask_flat = unobs_mask.flatten()
        n_obs = int(unobs_mask.sum())
        n_correct_total += int((cell_eq & unobs_mask).sum())
        n_obs_total += n_obs

        traj_all, traj_corr, traj_wrong = [], [], []
        for k in range(K - 1):
            p = per_breath_probs[k] + EPS
            q = per_breath_probs[k+1] + EPS
            p /= p.sum(axis=-1, keepdims=True)
            q /= q.sum(axis=-1, keepdims=True)
            m = 0.5 * (p + q)
            kl_pm = (p * np.log(p / m)).sum(axis=-1)
            kl_qm = (q * np.log(q / m)).sum(axis=-1)
            jsd_per_cell = (0.5 * kl_pm + 0.5 * kl_qm).mean(axis=-1).flatten()
            traj_all.append(float(jsd_per_cell[unobs_mask_flat].mean())
                             if unobs_mask_flat.any() else 0.0)
            if unobs_correct.any():
                traj_corr.append(float(jsd_per_cell[unobs_correct].mean()))
            else:
                traj_corr.append(None)
            if unobs_wrong.any():
                traj_wrong.append(float(jsd_per_cell[unobs_wrong].mean()))
            else:
                traj_wrong.append(None)

        trajs_all.append(traj_all)
        if all(v is not None for v in traj_corr):
            trajs_correct.append(traj_corr)
        if all(v is not None for v in traj_wrong):
            trajs_wrong.append(traj_wrong)

        if (i+1) % 10 == 0 or i == 0:
            dt = time.time() - t0
            print(f"  [{i+1:3d}/{len(records)}] dt={dt:.0f}s", flush=True)

    print()
    print("=" * 60)
    print(f"Results (clamp_mode={CLAMP_MODE}, n={len(records)} puzzles)")
    print("=" * 60)
    cell_acc = n_correct_total / max(n_obs_total, 1)
    print(f"  hard cell_acc: {cell_acc:.3f}  ({n_correct_total}/{n_obs_total})")
    print()

    def report(label, trajs):
        if not trajs:
            print(f"  {label}: no data")
            return None
        means = [sum(t[k] for t in trajs) / len(trajs) for k in range(K - 1)]
        mean_str = " ".join(f"{v:.4f}" for v in means)
        threshold = max(0.001, means[0] * 0.05)
        freeze_k = next((k for k, v in enumerate(means) if v <= threshold), K - 2)
        decay = means[-1] / max(means[0], 1e-9)
        print(f"  {label}: JSD[B0..B{K-2}] = {mean_str}")
        print(f"    freeze breath: k={freeze_k}  (threshold {threshold:.4f})")
        print(f"    early={means[0]:.4f}  late={means[-1]:.4f}  decay={decay:.2%}")
        return freeze_k

    fk_all = report("all unobs", trajs_all)
    print()
    fk_corr = report("correct", trajs_correct)
    print()
    fk_wrong = report("wrong", trajs_wrong)


if __name__ == "__main__":
    main()
