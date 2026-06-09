"""v200 perceiver-CORE training driver.

Stage 1: cold-start only. No warm-start ckpts available.
Base: SmolLM2-1.7B (LlamaForCausalLM, hidden_size=2048).
Override with LLAMA_WEIGHTS env var for meta-llama/Llama-3.2-1B.

Architecture: Perceiver-CORE
  - 32 learnable latents at 2048d (QR-orthogonal init)
  - Each breath: READ (cross-attn latents→tokens) → THINK (4 Llama layers) → READOUT
  - Tree codebook on first 16 latents (variable beliefs)
  - Per-breath weighted CE ladder + calibration head BCE

Diagnostic logging per step:
  - latent_norm_mean  : mean L2 norm of latents (across B×L)
  - latent_drift      : mean ||latents_k+1 - latents_k||₂ (via per-breath δ norms)
  - per_breath_ce     : CE at each breath (ladder signal)
  - calib_loss        : calibration BCE
  - cell_acc          : accuracy on unobserved cells (last breath)
  - query_acc         : accuracy on query node (last breath, approx)

Env vars (all optional, have defaults):
  V200_TASK=1            — must be set
  V200_K_MAX=8           — breaths
  V200_N_LATENTS=32      — latent count
  V200_N_VAR_LAT=16      — variable-belief latents
  V200_N_DIGITS=5        — tree codebook depth
  V200_N_MAX=16          — max vars per factor graph
  V200_F_MAX=8           — max factors per factor graph
  V200_CALIB_WEIGHT=0.05 — calib loss weight
  BATCH=4                — batch size (verify fits in AMD memory before 8)
  LR=1e-4                — learning rate (cold-start needs higher LR)
  STEPS=1000             — training steps
  CKPT_EVERY=500         — checkpoint interval
  EVAL_EVERY=500         — eval interval
  LOG_EVERY=10           — log interval
  CKPT_LABEL=v200_run    — checkpoint filename prefix
  LLAMA_WEIGHTS=         — path to Llama/SmolLM2 safetensors (optional override)
  DEV=PCI+AMD            — device override
"""
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import safe_save, safe_load

from mycelium.llama_loader import (
    attach_llama_layers, LlamaConfig, SMOLLM2_1_7B_CFG, load_llama_weights,
    llama_layer_parameter_count,
)
from mycelium.factor_graph_v200 import (
    V200_K_MAX, V200_N_LATENTS, V200_N_VAR_LAT, V200_N_DIGITS,
    V200_N_MAX, V200_F_MAX, V200_CALIB_WEIGHT,
    V200_STAGE2A_WAIST, V200_WAIST_DIM,
    attach_fg_params_v200, fg_v200_parameters, fg_v200_state_dict,
    _compile_jit_fg_step_v200, compile_jit_eval_v200,
    fg_accuracy_v200, compute_drift_v200,
)
from mycelium.factor_graph_v108 import bins_to_digits_msd
from mycelium.factor_graph_data_v107 import (
    FactorGraphLoaderV107, DualDataLoaderV107, load_gsm8k_records_v107,
)


DIFFICULTIES = ["easy", "medium", "hard"]


def compute_lr(step: int, warmup_steps: int = 100, lr_max: float = 1e-4, lr_min: float = 1e-5) -> float:
    """Linear LR warmup from lr_min to lr_max over warmup_steps, then constant lr_max."""
    if step <= warmup_steps:
        return lr_min + (lr_max - lr_min) * step / warmup_steps
    return lr_max


# ---------------------------------------------------------------------------
# Simple model container (avoids importing BreathingTransformer for v200)
# ---------------------------------------------------------------------------

class V200Model:
    """Minimal model container for v200 perceiver-CORE.

    No Pythia / BreathingTransformer dependency — v200 is a clean break.
    Attributes are attached dynamically by attach_llama_layers + attach_fg_params_v200.
    """
    pass


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_v200(
    model: V200Model,
    val_loader: FactorGraphLoaderV107,
    K: int,
    max_batches: int,
    eval_fn,
    n_max: int,
    f_max: int,
    n_digits: int,
    n_var_lat: int,
) -> dict:
    """Run eval on val_loader, return per-difficulty results.

    Uses loader.iter_eval() which yields all batches with mixed difficulties,
    aggregates per-difficulty stats from the `picks` records (v107 pattern).
    """
    Tensor.training = False
    agg: dict[str, dict] = {}
    n_batches = 0

    for batch in val_loader.iter_eval(batch_size=val_loader.batch_size):
        domain_init  = batch["domain_init"]
        node_kinds   = batch["node_kinds"]
        gold_bins    = batch["gold_bins"]
        obs_mask     = batch["observed_mask"]
        query_idx_np = batch["query_idx"]
        picks        = batch["picks"]

        gold_bins_np   = gold_bins.numpy()
        gold_digits_np = bins_to_digits_msd(gold_bins_np, n_digits=n_digits)

        outs = eval_fn(domain_init, node_kinds)
        final_logits = outs[0]   # (B, n_var_lat, n_digits, 10)
        pred_digits_np = final_logits.argmax(axis=-1).realize().numpy()

        obs_np = obs_mask.numpy()
        B_local = len(picks)

        for b in range(B_local):
            rec  = picks[b]
            diff = rec.get("difficulty", "easy")
            if diff not in agg:
                agg[diff] = {"n_unobs": 0, "n_correct_unobs": 0,
                             "query_correct": 0, "n_puzzles": 0}
            qi = int(query_idx_np[b])
            nv = int(batch.get("n_vars_total", [n_var_lat]*B_local)[b])
            for vi in range(min(nv, n_var_lat)):
                if obs_np[b, vi] == 0:
                    agg[diff]["n_unobs"] += 1
                    if (pred_digits_np[b, vi] == gold_digits_np[b, vi]).all():
                        agg[diff]["n_correct_unobs"] += 1
            if qi < n_var_lat and (pred_digits_np[b, qi] == gold_digits_np[b, qi]).all():
                agg[diff]["query_correct"] += 1
            agg[diff]["n_puzzles"] += 1

        n_batches += 1
        if n_batches >= max_batches:
            break

    results = {}
    for d, v in agg.items():
        n = v["n_puzzles"]
        if n == 0:
            continue
        results[d] = {
            "cell_acc":  v["n_correct_unobs"] / max(v["n_unobs"], 1),
            "query_acc": v["query_correct"] / n,
            "n_puzzles": n,
        }

    Tensor.training = True
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    V200_TASK_LOCAL = int(getenv("V200_TASK", "0")) > 0
    assert V200_TASK_LOCAL, "V200_TASK=1 must be set"

    K          = int(getenv("V200_K_MAX",      str(V200_K_MAX)))
    BATCH      = int(getenv("BATCH",           "4"))
    STEPS      = int(getenv("STEPS",           "1000"))
    LR         = float(getenv("LR",            "1e-4"))
    CKPT_EVERY = int(getenv("CKPT_EVERY",      "500"))
    EVAL_EVERY = int(getenv("EVAL_EVERY",      "500"))
    LOG_EVERY  = int(getenv("LOG_EVERY",       "10"))
    PER_BREATH_EVERY = int(getenv("PER_BREATH_CE_EVERY", "50"))
    GC_EVERY   = int(getenv("GC_EVERY",        "50"))
    CKPT_LABEL = getenv("CKPT_LABEL",          "v200_run")
    SEED       = int(getenv("SEED",            "42"))
    N_LATENTS  = int(getenv("V200_N_LATENTS",  str(V200_N_LATENTS)))
    N_VAR_LAT  = int(getenv("V200_N_VAR_LAT",  str(V200_N_VAR_LAT)))
    N_DIGITS   = int(getenv("V200_N_DIGITS",   str(V200_N_DIGITS)))
    N_MAX      = int(getenv("V200_N_MAX",      str(V200_N_MAX)))
    F_MAX      = int(getenv("V200_F_MAX",      str(V200_F_MAX)))
    CALIB_W    = float(getenv("V200_CALIB_WEIGHT", str(V200_CALIB_WEIGHT)))
    STAGE2A_WAIST = int(getenv("V200_STAGE2A_WAIST", "1" if V200_STAGE2A_WAIST else "0")) > 0
    WAIST_DIM  = int(getenv("V200_WAIST_DIM",      str(V200_WAIST_DIM)))
    EVAL_BATCHES = int(getenv("EVAL_BATCHES",  "10"))
    EVAL_BATCH   = int(getenv("EVAL_BATCH",    str(BATCH)))
    TRAIN_PATH = getenv("V200_TRAIN",          ".cache/factor_graph_train.jsonl")
    VAL_PATH   = getenv("V200_VAL",            ".cache/factor_graph_test.jsonl")
    GSM8K_PATH = getenv("V200_GSM8K_TRAIN",    ".cache/gsm8k_factor_graphs_train.jsonl")
    GSM8K_RATIO= float(getenv("V200_GSM8K_RATIO", "0.5"))
    N_HEADS    = 16   # v107 factor graph data loader uses 16 heads for masks

    T = N_MAX + F_MAX

    print("=== v200 perceiver-CORE training ===")
    print(f"  K={K}  N_LATENTS={N_LATENTS}  N_VAR_LAT={N_VAR_LAT}  N_DIGITS={N_DIGITS}")
    print(f"  N_MAX={N_MAX}  F_MAX={F_MAX}  T={T}  CALIB_W={CALIB_W}")
    print(f"  device={Device.DEFAULT}  B={BATCH}  STEPS={STEPS}  LR={LR}")
    print(f"  STAGE2A_WAIST={STAGE2A_WAIST}  WAIST_DIM={WAIST_DIM}")
    print()

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    # ---- Build model ----
    model = V200Model()

    print("Loading Llama/SmolLM2 weights...")
    sd = load_llama_weights()
    cfg = SMOLLM2_1_7B_CFG
    attach_llama_layers(model, n_layers=4, sd=sd, cfg=cfg, layer_offset=0)
    del sd
    gc.collect()

    attach_fg_params_v200(
        model,
        n_latents=N_LATENTS,
        n_var_lat=N_VAR_LAT,
        k_max=K,
        n_digits=N_DIGITS,
        n_max=N_MAX,
        f_max=F_MAX,
        stage2a_waist=STAGE2A_WAIST,
        waist_dim=WAIST_DIM,
    )
    Device[Device.DEFAULT].synchronize()

    params   = fg_v200_parameters(model)
    n_params = sum(int(np.prod(t.shape)) for t in params)
    n_llama  = llama_layer_parameter_count(cfg, n_layers=4)
    n_perceiver = n_params - n_llama
    print(f"\n  total trainable params: {n_params/1e6:.1f}M")
    print(f"  Llama L0-L3:           {n_llama/1e6:.1f}M")
    print(f"  perceiver-CORE:        {n_perceiver/1e6:.2f}M")
    print()

    opt = AdamW(params, lr=LR, weight_decay=1e-4)

    # ---- Data loaders ----
    synth_loader = FactorGraphLoaderV107(
        TRAIN_PATH, batch_size=BATCH,
        difficulty_filter=None, curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=N_HEADS, seed=SEED,
    )
    gsm8k_records = load_gsm8k_records_v107(GSM8K_PATH)
    dual_loader   = DualDataLoaderV107(
        synth_loader, gsm8k_records, gsm8k_ratio=GSM8K_RATIO,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=N_HEADS, seed=SEED + 1,
    )
    val_loader = FactorGraphLoaderV107(
        VAL_PATH, batch_size=EVAL_BATCH,
        difficulty_filter=None, curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=N_HEADS, seed=SEED + 2,
    )

    ckpt_dir = ".cache/fg_v200_ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Compile JIT steps ----
    Tensor.training = True
    step_fn = _compile_jit_fg_step_v200(
        model, opt, K=K, B=BATCH,
        n_max=N_MAX, f_max=F_MAX,
        n_var_lat=N_VAR_LAT, n_digits=N_DIGITS,
        calib_weight=CALIB_W,
        grad_clip=1.0,
        stage2a_waist=STAGE2A_WAIST,
    )
    eval_fn = compile_jit_eval_v200(
        model, K=K, B=EVAL_BATCH,
        n_max=N_MAX, f_max=F_MAX,
        n_var_lat=N_VAR_LAT, n_digits=N_DIGITS,
        stage2a_waist=STAGE2A_WAIST,
    )

    print(f"\ntraining (K={K}, N_LATENTS={N_LATENTS}, LR={LR})...\n")
    t0 = time.time()
    log_loss = log_ce = log_calib = log_n = 0.0

    for step in range(1, STEPS + 1):
        # LR warmup: linear from 1e-5 → 1e-4 over first 500 steps, then constant 1e-4
        current_lr = compute_lr(step)
        opt.lr = Tensor([current_lr]).contiguous().realize()

        batch = dual_loader.sample_batch(step=step)

        domain_init  = batch["domain_init"]
        node_kinds   = batch["node_kinds"]
        gold_bins    = batch["gold_bins"]
        obs_mask     = batch["observed_mask"]

        gold_bins_np   = gold_bins.numpy()
        gold_digits_np = bins_to_digits_msd(gold_bins_np, n_digits=N_DIGITS)
        gold_digits_t  = Tensor(
            gold_digits_np.astype(np.int64), dtype=dtypes.int,
        ).contiguous().realize()

        # PADDING BUG FIX (Jun 8): build per-puzzle real-variable mask so the
        # JIT step can exclude padding positions from CE loss and cell_acc.
        # Without this, the model was being rewarded for trivially predicting
        # zero at padding positions where gold_bins=0.
        n_vars_np = np.array(
            [int(batch["n_vars_total"][b]) for b in range(BATCH)],
            dtype=np.int32,
        )  # (B,)
        n_vars_mask_np = (
            np.arange(N_VAR_LAT, dtype=np.int32)[None, :] < n_vars_np[:, None]
        ).astype(np.float32)   # (B, N_VAR_LAT) — 1.0 for real vars
        n_vars_mask_t = Tensor(
            n_vars_mask_np, dtype=dtypes.float,
        ).contiguous().realize()

        outs = step_fn(domain_init, node_kinds, gold_digits_t, obs_mask, n_vars_mask_t)
        total_t, healthy_t, ce_t, calib_t = outs[0], outs[1], outs[2], outs[3]
        cell_acc_t, query_acc_t = outs[4], outs[5]
        pb_ce_ts    = outs[6:6 + K]
        # drift is no longer in the JIT return — computed lazily via compute_drift_v200

        if float(healthy_t.numpy()) < 0.5:
            print(f"[NaN-skip] step {step}", flush=True)
            continue

        log_loss  += float(total_t.numpy())
        log_ce    += float(ce_t.numpy())
        log_calib += float(calib_t.numpy())
        log_n     += 1

        if step % LOG_EVERY == 0:
            dt = time.time() - t0

            # Latent diagnostics (cheap: just the param, not activations)
            lat_np = model.fg_v200_latents.numpy()
            lat_norm_mean = float(np.linalg.norm(lat_np, axis=-1).mean())
            gate_np = model.fg_v200_delta_gate.numpy()
            gate_mean = float(gate_np.mean())

            waist_diag = ""
            if STAGE2A_WAIST:
                wg = float(model.fg_v200_waist_gate.numpy()[0])
                we_norm = float(np.linalg.norm(model.fg_v200_W_expand.numpy()))
                waist_diag = f"  waist_gate={wg:+.4f}  W_expand_norm={we_norm:.4f}"

            print(
                f"[step {step:5d}] loss={log_loss/log_n:.4f} "
                f"ce={log_ce/log_n:.4f} calib={log_calib/log_n:.4f}  "
                f"cell_acc={float(cell_acc_t.numpy()):.3f}  "
                f"lat_norm={lat_norm_mean:.3f}  gate_mean={gate_mean:.3f}"
                f"{waist_diag}  "
                f"lr={current_lr:.2e}  "
                f"({dt:.1f}s, {dt/step:.2f}s/step)",
                flush=True,
            )
            log_loss = log_ce = log_calib = log_n = 0.0

        if step % PER_BREATH_EVERY == 0:
            pb_ce = [float(t.numpy()) for t in pb_ce_ts]
            pb_str = " ".join(f"{v:.3f}" for v in pb_ce)
            ca = float(cell_acc_t.numpy())
            qa = float(query_acc_t.numpy())
            print(f"  per_breath_ce[B0..B{K-1}]: {pb_str}  "
                  f"(cell_acc={ca:.3f} query_acc={qa:.3f})", flush=True)
            if K > 1 and len(pb_ce) >= 2:
                ladder_delta = pb_ce[0] - pb_ce[-1]
                tag = "OK" if ladder_delta > 0.05 else "target > 0.05"
                print(f"  [LADDER] B0-B{K-1} delta = {ladder_delta:.3f} ({tag})",
                      flush=True)
            # Per-breath drift: one extra eager (non-JIT) forward, called only here.
            # Reveals even/odd asymmetry when the alternating waist is active.
            drifts = compute_drift_v200(
                model, domain_init, node_kinds,
                K=K, n_max=N_MAX, f_max=F_MAX,
                stage2a_waist=STAGE2A_WAIST,
            )
            even_drifts = [drifts[i] for i in range(K) if i % 2 == 0]
            odd_drifts  = [drifts[i] for i in range(K) if i % 2 == 1]
            drift_str = " ".join(
                f"B{i}({'E' if i%2==0 else 'O'})={drifts[i]:.4f}"
                for i in range(K)
            )
            even_mean = float(np.mean(even_drifts)) if even_drifts else 0.0
            odd_mean  = float(np.mean(odd_drifts))  if odd_drifts  else 0.0
            ratio_tag = f"even/odd={even_mean/max(odd_mean,1e-9):.2f}"
            asym_tag  = ("EVEN>ODD (waist working)" if even_mean > odd_mean
                         else "UNIFORM (waist not asymmetric)")
            print(f"  [DRIFT]  {drift_str}",    flush=True)
            print(f"  [DRIFT]  even_mean={even_mean:.4f}  odd_mean={odd_mean:.4f}  "
                  f"{ratio_tag}  {asym_tag}", flush=True)

        if step % EVAL_EVERY == 0:
            print(f"  evaluating ({EVAL_BATCHES} batches × B={EVAL_BATCH})...", flush=True)
            results = evaluate_v200(
                model, val_loader, K=K, max_batches=EVAL_BATCHES, eval_fn=eval_fn,
                n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS, n_var_lat=N_VAR_LAT,
            )
            for d in DIFFICULTIES:
                if d not in results:
                    continue
                v = results[d]
                print(f"  val[{d:6s}]: cell={v['cell_acc']:.3f} q={v['query_acc']:.3f} "
                      f"n={v['n_puzzles']}", flush=True)

        if step % CKPT_EVERY == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_step{step}.safetensors")
            safe_save(fg_v200_state_dict(model), ckpt_path)
            print(f"  saved {ckpt_path}", flush=True)

        if step % GC_EVERY == 0:
            gc.collect()

    ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_final.safetensors")
    safe_save(fg_v200_state_dict(model), ckpt_path)
    print(f"\ndone. saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
