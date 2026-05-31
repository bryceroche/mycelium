"""v105 factor graph training driver — digit-by-digit codebook architecture.

Numbers are factor graphs too: value 1234 = digits (1,2,3,4) in place-value
factor graph.  v105 replaces the 100-way domain codebook with a 10-way digit
codebook shared across all digit positions × all variable slots.

Architecture:
  - digit_codebook (10, H): one row per digit 0-9, shared across ALL digit positions
  - digit_pos_embed (N_DIGITS, H): sinusoidal place-value embeddings
  - Sequence length T = N_MAX × N_DIGITS + F_MAX  (vs N_MAX + F_MAX in v100)
  - K iterative-prefill breaths; shared 4-layer transformer
  - Per-digit CE supervision on unobserved variables
  - Expected-value constraint energy (differentiable)
  - No warm-start from v100 (different tensor shapes)

Key advantage over v104:
  - Handles all GSM8K values (no [0,99] filter: ~63% of records were discarded)
  - N_DIGITS=5 covers values 0..99999 with only 10-way classification per digit
  - 10 × 1024 = 10K digit codebook vs 100 × 1024 = 100K domain codebook

Env vars (set in launchers):
  V105_TASK=1
  V105_K_MAX=8
  V105_N_DIGITS=5
  V105_N_MAX=16
  V105_F_MAX=8
  V105_ENERGY_WEIGHT=0.01
  V105_FACTOR_AUX_WEIGHT=0.5
  V105_CALIB_WEIGHT=0.05
  V105_TRAIN=.cache/factor_graph_train.jsonl
  V105_VAL=.cache/factor_graph_test.jsonl
  V105_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl
  V105_GSM8K_RATIO=0.5
  BATCH=8
  STEPS=3000
  LR=3e-5
  CKPT_EVERY=500
  EVAL_EVERY=250
  CKPT_LABEL=v105_smoke
  PYTHIA_INIT=1
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

from mycelium import Config, BreathingTransformer
from mycelium.loader import _load_state, load_breathing
from mycelium.factor_graph_v105 import (
    V105_K_MAX, V105_N_MAX, V105_F_MAX, V105_N_DIGITS, V105_N_HEADS,
    V105_ENERGY_WEIGHT, V105_FACTOR_AUX_WEIGHT, V105_CALIB_WEIGHT,
    attach_fg_params_v105, fg_v105_parameters, fg_v105_state_dict,
    fg_breathing_forward_v105,
    _compile_jit_fg_step_v105, _compile_jit_fg_eval_v105,
    value_to_digits, digits_to_value,
)
from mycelium.factor_graph_data_v105 import (
    FactorGraphLoaderV105, DualDataLoaderV105, load_gsm8k_records_v105,
)

DIFFICULTIES = ["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def cast_layers_fp32(model):
    """Cast L0-L3 + shared weights from fp16 to fp32 for stable training."""
    def _cast(obj, attr):
        t = getattr(obj, attr)
        if t.dtype == dtypes.half:
            setattr(obj, attr, t.cast(dtypes.float).contiguous().realize())
    _cast(model.embed, "weight")
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out"):
        _cast(sw, a)
    for layer in model.block.layers:
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            _cast(layer, a)


def collect_fg_params_v105(model) -> list[Tensor]:
    """Trainable params: shared L0-L3 attn/FFN + v105 fg-specific params."""
    params: list[Tensor] = []
    sw = model.block.shared
    params += [sw.wv, sw.bv, sw.wo, sw.bo, sw.w_out, sw.b_out,
               sw.in_ln_g, sw.in_ln_b, sw.post_ln_g, sw.post_ln_b]
    for layer in model.block.layers:
        params += [layer.wq, layer.bq, layer.wk, layer.bk, layer.w_in, layer.b_in]
    params += [model.ln_f_g, model.ln_f_b]
    params += fg_v105_parameters(model)
    return params


def model_state_dict_v105(model) -> dict:
    sd = {
        "ln_f.g": model.ln_f_g,
        "ln_f.b": model.ln_f_b,
    }
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out",
              "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b"):
        sd[f"shared.{a}"] = getattr(sw, a)
    for i, layer in enumerate(model.block.layers):
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            sd[f"phase{i}.{a}"] = getattr(layer, a)
    sd.update(fg_v105_state_dict(model))
    return sd


def load_ckpt_v105(model, path: str):
    """Load a v105 checkpoint (v105 keys only — no cross-version warm-start)."""
    sd      = safe_load(path)
    targets = model_state_dict_v105(model)
    missing = []
    loaded  = []
    for name, dst in targets.items():
        if name not in sd:
            missing.append(name)
            continue
        src = sd[name].to(dst.device).realize()
        if src.shape != dst.shape:
            # Allow K_max mismatch for breath_embed / delta_gate (extend with existing init)
            if (name in ("fg_v105.breath_embed", "fg_v105.delta_gate")
                    and src.ndim == dst.ndim
                    and src.shape[0] <= dst.shape[0]):
                k_old = int(src.shape[0])
                cur   = dst.numpy()
                cur[:k_old] = src.cast(dst.dtype).numpy()[:k_old]
                from tinygrad import Tensor as _T
                dst.assign(_T(cur, dtype=dst.dtype, device=dst.device).contiguous()).realize()
                loaded.append(name)
                continue
            try:
                src = src.reshape(dst.shape)
            except Exception:
                missing.append(f"{name}(shape mismatch)")
                continue
        if src.dtype != dst.dtype:
            src = src.cast(dst.dtype)
        dst.assign(src).realize()
        loaded.append(name)

    print(f"  loaded {len(loaded)}/{len(targets)} keys from {os.path.basename(path)}")
    if missing:
        print(f"  missing {len(missing)} keys: {missing[:5]}")


def evaluate_v105(
    model, loader: FactorGraphLoaderV105,
    K: int, max_batches: int = 20,
    eval_fn=None,
    n_max: int = V105_N_MAX,
    f_max: int = V105_F_MAX,
    n_digits: int = V105_N_DIGITS,
) -> dict:
    """Run eval on up to max_batches batches. Returns per-difficulty stats."""
    Tensor.training = False
    agg = {}
    n_batches = 0

    for batch in loader.iter_eval(batch_size=loader.batch_size):
        digit_init    = batch["digit_init"]
        node_kinds    = batch["node_kinds"]
        staging_mask  = batch["staging_mask"]
        head_op_mask  = batch["head_op_mask"]
        gold_digits   = batch["gold_digits"]
        obs_mask      = batch["observed_mask"]
        query_idx_np  = batch["query_idx"]
        picks         = batch["picks"]

        if eval_fn is not None:
            pred_dg_t, _cell_acc_t = eval_fn(
                digit_init, node_kinds, staging_mask, head_op_mask,
                gold_digits, obs_mask,
            )
            pred_dg_np = pred_dg_t.numpy()
        else:
            dig_lh, _, _ = fg_breathing_forward_v105(
                model, digit_init, node_kinds, staging_mask, head_op_mask,
                K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            )
            pred_dg_np = dig_lh[-1].argmax(axis=-1).realize().numpy()  # (B, N_MAX, N_DIGITS)

        gold_dg_np = gold_digits.numpy()   # (B, N_MAX, N_DIGITS)
        obs_np     = obs_mask.numpy()

        for b in range(len(picks)):
            rec  = picks[b]
            diff = rec.get("difficulty", "easy")
            if diff not in agg:
                agg[diff] = {"n_unobs": 0, "n_correct_unobs": 0,
                             "query_correct": 0, "n_puzzles": 0}
            qi  = int(query_idx_np[b])
            nv  = int(batch["n_vars_total"][b])
            for vi in range(min(nv, n_max)):
                if obs_np[b, vi] == 0:
                    agg[diff]["n_unobs"] += 1
                    if np.all(pred_dg_np[b, vi] == gold_dg_np[b, vi]):
                        agg[diff]["n_correct_unobs"] += 1
            if qi < n_max and np.all(pred_dg_np[b, qi] == gold_dg_np[b, qi]):
                agg[diff]["query_correct"] += 1
            agg[diff]["n_puzzles"] += 1

        n_batches += 1
        if n_batches >= max_batches:
            break

    out = {}
    for d, v in agg.items():
        n = v["n_puzzles"]
        if n == 0:
            continue
        cell_acc  = v["n_correct_unobs"] / max(v["n_unobs"], 1)
        q_acc     = v["query_correct"] / n
        out[d]    = {"cell_acc": cell_acc, "query_acc": q_acc, "n_puzzles": n}

    Tensor.training = True
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    V105_TASK_LOCAL = int(getenv("V105_TASK", 0)) > 0
    assert V105_TASK_LOCAL, "V105_TASK=1 must be set"

    K        = int(getenv("V105_K_MAX",    str(V105_K_MAX)))
    N_DIGITS = int(getenv("V105_N_DIGITS", str(V105_N_DIGITS)))
    N_MAX    = int(getenv("V105_N_MAX",    str(V105_N_MAX)))
    F_MAX    = int(getenv("V105_F_MAX",    str(V105_F_MAX)))
    T_MAX    = N_MAX * N_DIGITS + F_MAX

    BATCH     = int(getenv("BATCH",       "8"))
    STEPS     = int(getenv("STEPS",       "3000"))
    LR        = float(getenv("LR",        "3e-5"))
    CKPT_EVERY       = int(getenv("CKPT_EVERY",        "500"))
    EVAL_EVERY       = int(getenv("EVAL_EVERY",        "250"))
    LOG_EVERY        = int(getenv("LOG_EVERY",         "10"))
    PER_BREATH_EVERY = int(getenv("PER_BREATH_CE_EVERY","50"))
    GC_EVERY         = int(getenv("GC_EVERY",          "50"))
    CKPT_LABEL       = getenv("CKPT_LABEL",            "v105_smoke")
    RESUME_FROM      = getenv("RESUME_FROM",           "")
    PYTHIA_INIT      = int(getenv("PYTHIA_INIT",       "1")) > 0
    SEED             = int(getenv("SEED",              "42"))
    EVAL_BATCHES     = int(getenv("EVAL_BATCHES",      "20"))
    EVAL_BATCH       = int(getenv("EVAL_BATCH",        str(BATCH)))

    ENERGY_WEIGHT      = float(getenv("V105_ENERGY_WEIGHT",      str(V105_ENERGY_WEIGHT)))
    FACTOR_AUX_WEIGHT  = float(getenv("V105_FACTOR_AUX_WEIGHT",  str(V105_FACTOR_AUX_WEIGHT)))
    CALIB_WEIGHT       = float(getenv("V105_CALIB_WEIGHT",        str(V105_CALIB_WEIGHT)))

    DIFFICULTY_FILTER  = os.environ.get("V105_DIFFICULTY_FILTER", "").strip() or None
    CURRICULUM         = int(getenv("V105_CURRICULUM",        "0")) > 0
    CURRICULUM_ANNEAL  = int(getenv("V105_CURRICULUM_ANNEAL", "1000"))

    TRAIN_PATH  = getenv("V105_TRAIN",       ".cache/factor_graph_train.jsonl")
    VAL_PATH    = getenv("V105_VAL",         ".cache/factor_graph_test.jsonl")
    GSM8K_PATH  = getenv("V105_GSM8K_TRAIN", ".cache/gsm8k_factor_graphs_train.jsonl")
    GSM8K_RATIO = float(getenv("V105_GSM8K_RATIO", "0.5"))

    print("=== v105 factor graph training (digit-by-digit codebook) ===")
    print(f"device={Device.DEFAULT}  B={BATCH}  K={K}  N_DIGITS={N_DIGITS}  steps={STEPS}  lr={LR}")
    print(f"N_MAX={N_MAX}  F_MAX={F_MAX}  T_MAX={T_MAX} (was {N_MAX+F_MAX} in v100)")
    print(f"energy_weight={ENERGY_WEIGHT}  factor_aux_weight={FACTOR_AUX_WEIGHT}  calib_weight={CALIB_WEIGHT}")
    print(f"difficulty_filter={DIFFICULTY_FILTER}  curriculum={CURRICULUM}")
    print(f"train={TRAIN_PATH}  val={VAL_PATH}")
    print(f"gsm8k_train={GSM8K_PATH}  gsm8k_ratio={GSM8K_RATIO}  (NO [0,99] filter)")
    print()

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    cfg = Config()
    print(f"loading Pythia-410M → breathing transformer (PYTHIA_INIT={PYTHIA_INIT})...")
    if PYTHIA_INIT:
        sd    = _load_state()
        model = load_breathing(cfg, sd=sd)
        del sd
    else:
        model = BreathingTransformer(cfg)
    cast_layers_fp32(model)

    attach_fg_params_v105(model, hidden=cfg.hidden,
                          n_digits=N_DIGITS, n_max=N_MAX, f_max=F_MAX, k_max=K)
    Device[Device.DEFAULT].synchronize()

    params   = collect_fg_params_v105(model)
    n_params = sum(int(np.prod(t.shape)) for t in params)
    print(f"  trainable params: {n_params/1e6:.1f}M")

    if RESUME_FROM:
        print(f"resuming from ckpt: {RESUME_FROM}")
        load_ckpt_v105(model, RESUME_FROM)

    opt = AdamW(params, lr=LR, weight_decay=0.0)

    synth_loader = FactorGraphLoaderV105(
        TRAIN_PATH, batch_size=BATCH,
        difficulty_filter=DIFFICULTY_FILTER,
        curriculum=CURRICULUM,
        curriculum_anneal_steps=CURRICULUM_ANNEAL,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V105_N_HEADS,
        n_digits=N_DIGITS, seed=SEED,
    )

    gsm8k_records = load_gsm8k_records_v105(GSM8K_PATH, n_digits=N_DIGITS, n_max=N_MAX, f_max=F_MAX)

    dual_loader = DualDataLoaderV105(
        synth_loader, gsm8k_records,
        gsm8k_ratio=GSM8K_RATIO,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V105_N_HEADS,
        n_digits=N_DIGITS, seed=SEED + 1,
    )

    val_loader = FactorGraphLoaderV105(
        VAL_PATH, batch_size=EVAL_BATCH,
        difficulty_filter=None,
        curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V105_N_HEADS,
        n_digits=N_DIGITS, seed=SEED + 2,
    )

    ckpt_dir = ".cache/fg_v105_ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    Tensor.training = True
    step_fn = _compile_jit_fg_step_v105(
        model, opt, K=K, B=BATCH,
        factor_aux_weight=FACTOR_AUX_WEIGHT,
        calib_weight=CALIB_WEIGHT,
        energy_weight=ENERGY_WEIGHT,
        n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
        grad_clip=1.0,
    )
    eval_fn = _compile_jit_fg_eval_v105(
        model, K=K, B=EVAL_BATCH, n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
    )
    Tensor.training = True

    print(f"\ntraining...\n")
    t0 = time.time()
    log_loss = log_ce = log_calib = log_aux = log_energy = log_n = 0.0

    for step in range(1, STEPS + 1):
        batch = dual_loader.sample_batch(step=step)

        digit_init    = batch["digit_init"]
        node_kinds    = batch["node_kinds"]
        staging_mask  = batch["staging_mask"]
        head_op_mask  = batch["head_op_mask"]
        gold_digits   = batch["gold_digits"]
        obs_mask      = batch["observed_mask"]
        factor_gold_dg = batch["factor_gold_dg"]
        factor_valid  = batch["factor_valid"]
        factor_types  = batch["factor_types"]
        factor_args   = batch["factor_args"]

        outs = step_fn(
            digit_init, node_kinds, staging_mask, head_op_mask,
            gold_digits, obs_mask, factor_gold_dg, factor_valid,
            factor_types, factor_args,
        )
        total_t     = outs[0]
        healthy_t   = outs[1]
        ce_t        = outs[2]
        aux_t       = outs[3]
        calib_t     = outs[4]
        energy_t    = outs[5]
        cell_acc_t  = outs[6]
        query_acc_t = outs[7]
        pb_ce_ts    = outs[8:8 + K]

        if float(healthy_t.numpy()) < 0.5:
            print(f"[NaN-skip] step {step}: CE step skipped", flush=True)

        log_loss   += float(total_t.numpy())
        log_ce     += float(ce_t.numpy())
        log_calib  += float(calib_t.numpy())
        log_aux    += float(aux_t.numpy())
        log_energy += float(energy_t.numpy())
        log_n      += 1

        if step % LOG_EVERY == 0:
            dt = time.time() - t0
            print(
                f"[step {step:5d}] loss={log_loss/log_n:.4f} "
                f"ce={log_ce/log_n:.4f} "
                f"aux={log_aux/log_n:.4f} "
                f"calib={log_calib/log_n:.4f} "
                f"energy={log_energy/log_n:.4f}  "
                f"({dt:.1f}s, {dt/step:.2f}s/step)",
                flush=True,
            )
            log_loss = log_ce = log_calib = log_aux = log_energy = log_n = 0.0

        if step % PER_BREATH_EVERY == 0:
            pb_ce = [float(t.numpy()) for t in pb_ce_ts]
            pb_str = " ".join(f"{v:.3f}" for v in pb_ce)
            ca = float(cell_acc_t.numpy())
            qa = float(query_acc_t.numpy())
            print(
                f"  per_breath_ce[B0..B{K-1}]: {pb_str}  "
                f"(cell_acc={ca:.3f} query_acc={qa:.3f})",
                flush=True,
            )
            if K > 1 and len(pb_ce) >= 2:
                delta = pb_ce[0] - pb_ce[-1]
                status = "LADDER" if delta > 0.1 else "flat"
                print(f"  [{status}] B0-B{K-1} delta = {delta:.3f} (target > 0.1)", flush=True)

            # Per-digit CE breakdown: report B0 and B{K-1} separately for ladder diagnosis
            if len(pb_ce) >= 2:
                print(f"  B0_ce={pb_ce[0]:.4f}  B{K-1}_ce={pb_ce[-1]:.4f}  "
                      f"(negative=inverted; positive=ladder)", flush=True)

        if step % EVAL_EVERY == 0:
            print(f"  evaluating ({EVAL_BATCHES} batches × B={EVAL_BATCH})...", flush=True)
            results = evaluate_v105(
                model, val_loader, K=K,
                max_batches=EVAL_BATCHES,
                eval_fn=eval_fn,
                n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
            )
            for d in DIFFICULTIES:
                if d not in results:
                    continue
                v = results[d]
                print(
                    f"  val[{d:6s}]: cell_acc={v['cell_acc']:.3f} "
                    f"query_acc={v['query_acc']:.3f} n={v['n_puzzles']}",
                    flush=True,
                )

        if step % CKPT_EVERY == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_step{step}.safetensors")
            safe_save(model_state_dict_v105(model), ckpt_path)
            print(f"  saved {ckpt_path}", flush=True)

        if step % GC_EVERY == 0:
            gc.collect()

    # Final save
    ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_final.safetensors")
    safe_save(model_state_dict_v105(model), ckpt_path)
    print(f"\ndone. saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
