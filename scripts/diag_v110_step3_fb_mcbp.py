"""v110-step3 Monte Carlo BP diagnostic.

Hypothesis (Jun 6, after PUCT closed negative): the missing piece in
search-amplification on v110-step3 is INDEPENDENCE between samples,
which PUCT lacks (correlated by tree path). True Monte Carlo aggregates
INDEPENDENT samples — the π-estimate pattern.

This script injects Gaussian noise on h_quant per breath, runs N
independent stochastic BP forwards, averages the tree_logits BEFORE
argmax (continuous estimand from discrete samples), and compares the
ensemble cell_acc to deterministic BP on the same hard subset PUCT was
tested on.

Direct apples-to-apples with the PUCT result (n=50 hard puzzles, same
v110_step3_cont8_step1000 ckpt).

Usage:
  V110_STEP3_TASK=1 .venv/bin/python scripts/diag_v110_step3_mcbp.py

Env overrides:
  CKPT                          (.cache/fg_v110_step3_ckpts/v110_step3_cont8_step1000.safetensors)
  VAL_PATH                      (.cache/factor_graph_test.jsonl)
  V110_S3_MCBP_N_PUZZLES=50
  V110_S3_MCBP_DIFFICULTY=hard
  V110_S3_MCBP_N_SAMPLES=16     samples per puzzle for the MC ensemble
  V110_S3_MCBP_NOISE_SCALES=0.03,0.05,0.10   comma-sep sweep
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
os.environ.setdefault("V110_STEP3_HARD_BREATH_LEVEL", "0")
os.environ.setdefault("V110_STEP_K_MAX",    os.environ["V110_STEP3_K_MAX"])
os.environ.setdefault("V110_STEP_N_DIGITS", os.environ["V110_STEP3_N_DIGITS"])
os.environ.setdefault("V110_STEP_N_MAX",    os.environ["V110_STEP3_N_MAX"])
os.environ.setdefault("V110_STEP_F_MAX",    os.environ["V110_STEP3_F_MAX"])
os.environ.setdefault("V110_STEP_WAIST_DIM", os.environ["V110_STEP3_WAIST_DIM"])
os.environ.setdefault("V110_STEP_ALTERNATION", os.environ["V110_STEP3_ALTERNATION"])
os.environ.setdefault("V110_STEP_PHASE_SCALE", os.environ["V110_STEP3_PHASE_SCALE"])

import numpy as np
from tinygrad import Device, Tensor, dtypes

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.factor_graph_v110_step3 import (
    V110_STEP3_K_MAX, V110_STEP3_N_MAX, V110_STEP3_F_MAX, V110_STEP3_N_HEADS,
    V110_STEP3_N_DIGITS,
    V110_STEP3_WAIST_DIM, V110_STEP3_ALTERNATION, V110_STEP3_PHASE_SCALE,
    V110_STEP3_CODEBOOK_N, V110_STEP3_IB_CENTROIDS,
    V110_STEP3_GATE_PROFILE, V110_STEP3_PHOTON_ALPHA,
)
from mycelium.factor_graph_v108 import digits_to_value_msd
from mycelium.factor_graph_data_v107 import (
    FactorGraphLoaderV107, _records_to_batch_v107,
)
from scripts.v108_factor_graph_train import cast_layers_fp32
from scripts.v110_step3_fb_train import load_ckpt_v110_step3_fb

from mycelium.factor_graph_v110_step3_fb import (
    attach_fg_params_v110_step3_fb,
    compile_jit_mcbp_fb,
)


def _batch_to_tensors(batch_np):
    """Convert _records_to_batch_v107 output to the tensors the forward needs."""
    domain_init  = Tensor(batch_np["domain_init"]).cast(dtypes.half)
    node_kinds   = Tensor(batch_np["node_kinds"]).cast(dtypes.long)
    staging_mask = Tensor(batch_np["staging_mask"]).cast(dtypes.half)
    head_op_mask = Tensor(batch_np["head_op_mask"]).cast(dtypes.half)
    return domain_init, node_kinds, staging_mask, head_op_mask


def _predict_values_from_tree_logits(tree_logits_np):
    """tree_logits_np: (B, n_max, n_digits, 10) → predicted integer values (B, n_max)."""
    digits = tree_logits_np.argmax(axis=-1).astype(np.int64)  # (B, n_max, n_digits)
    return digits_to_value_msd(digits)  # (B, n_max)


def _score_one(predicted_values, batch_np, n_max):
    """Return (cell_correct, cell_total, query_correct, query_total) for one B=1 batch."""
    gold = batch_np["gold_values"][0]
    obs  = batch_np["observed_mask"][0]
    n_total = int(batch_np["n_vars_total"][0])
    query_idx = int(batch_np["query_idx"][0])

    cell_correct = 0
    cell_total = 0
    for vi in range(min(n_total, n_max)):
        if obs[vi] != 0:
            continue
        cell_total += 1
        if int(predicted_values[0, vi]) == int(gold[vi]):
            cell_correct += 1

    query_correct = 0
    query_total = 0
    if 0 <= query_idx < n_max:
        query_total = 1
        if int(predicted_values[0, query_idx]) == int(gold[query_idx]):
            query_correct = 1

    return cell_correct, cell_total, query_correct, query_total


def main():
    CKPT = os.environ.get(
        "CKPT",
        ".cache/fg_v110_step3_ckpts/v110_step3_cont8_step1000.safetensors",
    )
    VAL_PATH    = os.environ.get("VAL_PATH", ".cache/factor_graph_test.jsonl")
    SEED        = int(os.environ.get("SEED", "42"))
    N_PUZZLES   = int(os.environ.get("V110_S3_MCBP_N_PUZZLES", "50"))
    DIFFICULTY  = os.environ.get("V110_S3_MCBP_DIFFICULTY", "hard")
    N_SAMPLES   = int(os.environ.get("V110_S3_MCBP_N_SAMPLES", "16"))
    SCALES_STR  = os.environ.get("V110_S3_MCBP_NOISE_SCALES", "0.03,0.05,0.10")
    NOISE_SCALES = [float(s) for s in SCALES_STR.split(",")]

    H = 1024
    T = V110_STEP3_N_MAX + V110_STEP3_F_MAX
    K_MAX = V110_STEP3_K_MAX

    print("=== v110-step3 Monte Carlo BP diagnostic ===")
    print(f"  ckpt: {CKPT}")
    print(f"  val:  {VAL_PATH}")
    print(f"  difficulty: {DIFFICULTY}")
    print(f"  N_puzzles: {N_PUZZLES}")
    print(f"  N_samples (per puzzle): {N_SAMPLES}")
    print(f"  noise_scales: {NOISE_SCALES}")
    print(f"  K_max: {K_MAX}, n_max: {V110_STEP3_N_MAX}, T: {T}, H: {H}")
    print()

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    cfg = Config()
    print("loading Pythia-410M...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    cast_layers_fp32(model)

    attach_fg_params_v110_step3_fb(
        model, hidden=cfg.hidden,
        n_max=V110_STEP3_N_MAX, f_max=V110_STEP3_F_MAX, k_max=V110_STEP3_K_MAX,
        n_digits=V110_STEP3_N_DIGITS, n_code=V110_STEP3_CODEBOOK_N,
        ib_centroids_path=V110_STEP3_IB_CENTROIDS,
        waist_dim=V110_STEP3_WAIST_DIM,
    )
    Device[Device.DEFAULT].synchronize()

    print(f"loading ckpt: {CKPT}")
    load_ckpt_v110_step3_fb(model, CKPT)
    print(f"  fb_gate value: {float(model.fg_v111_feedback_gate.numpy()[0]):+.4f}")
    print()

    val_loader = FactorGraphLoaderV107(
        VAL_PATH, batch_size=1,
        difficulty_filter=DIFFICULTY, curriculum=False,
        n_max=V110_STEP3_N_MAX, f_max=V110_STEP3_F_MAX, k_max=V110_STEP3_K_MAX,
        n_heads=V110_STEP3_N_HEADS,
        seed=SEED + 2,
    )
    records = val_loader.records[:N_PUZZLES]
    print(f"  will evaluate {len(records)} {DIFFICULTY} puzzles\n", flush=True)

    mcbp_eval = compile_jit_mcbp_fb(
        model, K=K_MAX,
        n_max=V110_STEP3_N_MAX, f_max=V110_STEP3_F_MAX,
        n_digits=V110_STEP3_N_DIGITS,
        alternation=V110_STEP3_ALTERNATION,
        phase_scale=V110_STEP3_PHASE_SCALE,
        gate_profile=V110_STEP3_GATE_PROFILE,
        photon_alpha=V110_STEP3_PHOTON_ALPHA,
    )

    Tensor.training = False
    t_loop_start = time.time()

    # --- Deterministic baseline: one pass per puzzle, noise_scale=0 -----
    print("--- deterministic baseline (noise=0) ---")
    det_cell_corr = 0
    det_cell_tot  = 0
    det_q_corr    = 0
    det_q_tot     = 0
    det_logits    = []   # cache per-puzzle final-breath logits for reuse
    det_batches   = []   # cache batch_np for re-scoring

    noise_zeros_np = np.zeros((K_MAX, 1, T, H), dtype=np.float16)
    ns_zero_t = Tensor(np.array([0.0], dtype=np.float16)).cast(dtypes.half).contiguous().realize()

    t_det = time.perf_counter()
    for i, rec in enumerate(records):
        batch_np = _records_to_batch_v107(
            [rec],
            n_max=V110_STEP3_N_MAX, f_max=V110_STEP3_F_MAX,
            k_max=V110_STEP3_K_MAX, n_heads=V110_STEP3_N_HEADS,
        )
        det_batches.append(batch_np)

        di, nk, sm, hm = _batch_to_tensors(batch_np)
        noise_t = Tensor(noise_zeros_np).cast(dtypes.half).contiguous().realize()
        out = mcbp_eval(di, nk, sm, hm, noise_t, ns_zero_t)
        tl = out.numpy()
        det_logits.append(tl)

        pred = _predict_values_from_tree_logits(tl)
        cc, ct, qc, qt = _score_one(pred, batch_np, V110_STEP3_N_MAX)
        det_cell_corr += cc; det_cell_tot += ct
        det_q_corr    += qc; det_q_tot    += qt

        if (i + 1) % 10 == 0 or i == 0:
            elap = time.time() - t_loop_start
            acc_so_far = det_cell_corr / max(det_cell_tot, 1)
            print(f"  [{i+1:3d}/{len(records)}] det cell={acc_so_far:.3f}  "
                  f"(elapsed={elap:.0f}s)", flush=True)
    det_dt = time.perf_counter() - t_det

    det_cell = det_cell_corr / max(det_cell_tot, 1)
    det_q    = det_q_corr / max(det_q_tot, 1)
    print(f"\n  baseline cell_acc: {det_cell:.4f}  ({det_cell_corr}/{det_cell_tot})")
    print(f"  baseline query_acc: {det_q:.4f}  ({det_q_corr}/{det_q_tot})")
    print(f"  baseline wall-clock: {det_dt:.1f}s ({det_dt/len(records):.2f}s/puzzle)\n")

    # --- MC-BP sweep over noise scales ---------------------------------
    summary_rows = []
    for ns_value in NOISE_SCALES:
        print(f"--- MC-BP sweep: noise_scale={ns_value} ---")
        np.random.seed(SEED + 1000)  # consistent noise across scales (relative to scale)
        ns_t = Tensor(np.array([ns_value], dtype=np.float16)).cast(dtypes.half).contiguous().realize()

        mc_cell_corr = 0
        mc_cell_tot  = 0
        mc_q_corr    = 0
        mc_q_tot     = 0
        flipped_to_correct = 0
        flipped_to_wrong   = 0
        t_mc = time.perf_counter()

        for i, rec in enumerate(records):
            batch_np = det_batches[i]
            di, nk, sm, hm = _batch_to_tensors(batch_np)
            accum = np.zeros_like(det_logits[i], dtype=np.float64)
            for n in range(N_SAMPLES):
                noise_np = np.random.randn(K_MAX, 1, T, H).astype(np.float16)
                noise_t  = Tensor(noise_np).cast(dtypes.half).contiguous().realize()
                out = mcbp_eval(di, nk, sm, hm, noise_t, ns_t)
                accum += out.numpy().astype(np.float64)
            accum /= N_SAMPLES

            pred = _predict_values_from_tree_logits(accum)
            cc, ct, qc, qt = _score_one(pred, batch_np, V110_STEP3_N_MAX)
            mc_cell_corr += cc; mc_cell_tot += ct
            mc_q_corr    += qc; mc_q_tot    += qt

            # Track flips
            det_pred = _predict_values_from_tree_logits(det_logits[i])
            gold = batch_np["gold_values"][0]
            obs  = batch_np["observed_mask"][0]
            n_total = int(batch_np["n_vars_total"][0])
            for vi in range(min(n_total, V110_STEP3_N_MAX)):
                if obs[vi] != 0:
                    continue
                det_p = int(det_pred[0, vi])
                mc_p  = int(pred[0, vi])
                g     = int(gold[vi])
                if det_p != mc_p:
                    if det_p != g and mc_p == g:
                        flipped_to_correct += 1
                    elif det_p == g and mc_p != g:
                        flipped_to_wrong += 1

            if (i + 1) % 10 == 0 or i == 0:
                acc_so_far = mc_cell_corr / max(mc_cell_tot, 1)
                elap = time.time() - t_loop_start
                print(f"  [{i+1:3d}/{len(records)}] mc cell={acc_so_far:.3f}  "
                      f"flips(→correct={flipped_to_correct}, →wrong={flipped_to_wrong})  "
                      f"(elapsed={elap:.0f}s)", flush=True)

        mc_dt = time.perf_counter() - t_mc
        mc_cell = mc_cell_corr / max(mc_cell_tot, 1)
        mc_q    = mc_q_corr / max(mc_q_tot, 1)
        delta_c = mc_cell - det_cell
        delta_q = mc_q - det_q

        print(f"\n  noise={ns_value}  cell={mc_cell:.4f}  Δ={delta_c:+.4f}  "
              f"query={mc_q:.4f}  Δ={delta_q:+.4f}  "
              f"flips →correct/→wrong = {flipped_to_correct}/{flipped_to_wrong}  "
              f"wall={mc_dt:.1f}s ({mc_dt/len(records):.2f}s/puzzle)\n",
              flush=True)
        summary_rows.append((ns_value, mc_cell, delta_c, mc_q, delta_q,
                             flipped_to_correct, flipped_to_wrong,
                             mc_dt / len(records)))

    # --- Final summary -------------------------------------------------
    print("=" * 80)
    print(f"=== SUMMARY (n={len(records)} {DIFFICULTY} puzzles, N_samples={N_SAMPLES}) ===")
    print("=" * 80)
    print(f"  Deterministic BP: cell={det_cell:.4f}  query={det_q:.4f}  "
          f"({det_dt/len(records):.2f}s/puzzle)")
    print()
    print(f"  {'noise':>8} | {'cell':>7} | {'Δcell':>7} | {'query':>7} | {'Δquery':>7} | "
          f"{'→corr':>5} | {'→wrong':>6} | {'wall/p':>7}")
    print(f"  {'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}-+-{'-'*6}-+-{'-'*7}")
    for (s, c, dc, q, dq, fc, fw, wt) in summary_rows:
        print(f"  {s:>8.3f} | {c:>7.4f} | {dc:>+7.4f} | {q:>7.4f} | {dq:>+7.4f} | "
              f"{fc:>5d} | {fw:>6d} | {wt:>6.2f}s")
    print()


if __name__ == "__main__":
    main()
