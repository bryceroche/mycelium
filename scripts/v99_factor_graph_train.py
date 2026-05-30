"""v99 factor graph training driver.

Iterative-prefill breathing transformer on arithmetic factor graphs.
Reuses BreathingTransformer L0-L3 weights (Pythia-410M init) and the
same training infrastructure as v98 (TinyJit, AdamW, per-breath CE ladder).

Env vars (set in launchers):
  V99_TASK=1                  enable factor-graph params + forward
  V99_K_MAX=20                number of iterative-prefill breaths
  V99_ENERGY_WEIGHT=0.1       constraint energy loss weight
  V99_CALIB_WEIGHT=0.05       calibration loss weight
  V99_N_MAX=16                max variable nodes
  V99_F_MAX=8                 max factor nodes
  V99_TRAIN=.cache/factor_graph_train.jsonl
  V99_VAL=.cache/factor_graph_test.jsonl
  BATCH=8
  STEPS=2000
  LR=3e-5
  CKPT_EVERY=500
  CKPT_LABEL=v99_smoke
  RESUME_FROM=...
  PYTHIA_INIT=1
  V99_CURRICULUM=1
  V99_CURRICULUM_ANNEAL=1000
  V99_DIFFICULTY_FILTER=easy  (smoke mode: easy only)
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
from mycelium.factor_graph import (
    V99_K_MAX, V99_ENERGY_WEIGHT, V99_CALIB_WEIGHT, V99_N_MAX, V99_F_MAX,
    attach_fg_params, fg_parameters, fg_state_dict,
    factor_graph_breathing_forward, factor_graph_constraint_energy,
    _compile_jit_fg_step, _compile_jit_fg_eval,
    build_factor_graph_masks_np,
    factor_graph_accuracy,
)
from mycelium.factor_graph_data import FactorGraphLoader


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


def collect_fg_params(model) -> list[Tensor]:
    """Trainable parameters: shared L0-L3 attn/FFN + factor-graph-specific."""
    params: list[Tensor] = []
    sw = model.block.shared
    params += [sw.wv, sw.bv, sw.wo, sw.bo, sw.w_out, sw.b_out,
               sw.in_ln_g, sw.in_ln_b, sw.post_ln_g, sw.post_ln_b]
    for layer in model.block.layers:
        params += [layer.wq, layer.bq, layer.wk, layer.bk, layer.w_in, layer.b_in]
    params += [model.ln_f_g, model.ln_f_b]
    params += fg_parameters(model)
    return params


def model_state_dict_fg(model) -> dict:
    """Compact state dict for factor-graph training (excludes embed/embed_out/etc)."""
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
    sd.update(fg_state_dict(model))
    return sd


def load_ckpt(model, path: str):
    sd = safe_load(path)
    targets = model_state_dict_fg(model)
    missing = []
    for name, dst in targets.items():
        if name not in sd:
            missing.append(name)
            continue
        src = sd[name].to(dst.device).realize()
        if src.shape != dst.shape:
            # Partial-row copy for per-breath tensors (K expansion)
            if (name in ("fg.breath_embed", "fg.delta_gate")
                    and src.ndim == dst.ndim
                    and src.shape[0] <= dst.shape[0]):
                k_old = int(src.shape[0])
                if src.dtype != dst.dtype:
                    src = src.cast(dst.dtype)
                cur = dst.numpy()
                src_np = src.numpy()
                cur[:k_old] = src_np[:k_old]
                from tinygrad import Tensor as _T
                dst.assign(_T(cur, dtype=dst.dtype, device=dst.device).contiguous()).realize()
                continue
            try:
                src = src.reshape(dst.shape)
            except Exception:
                missing.append(f"{name}(shape mismatch)")
                continue
        if src.dtype != dst.dtype:
            src = src.cast(dst.dtype)
        dst.assign(src).realize()
    if missing:
        print(f"  ckpt missing {len(missing)} fg keys (kept init): {missing[:5]}")


def evaluate(model, loader: FactorGraphLoader, K: int,
             max_batches: int = 20, eval_fn=None,
             n_max: int = V99_N_MAX, f_max: int = V99_F_MAX) -> dict:
    """Run eval on `max_batches` batches. Returns per-difficulty stats."""
    Tensor.training = False
    agg = {}
    n_batches = 0

    for batch in loader.iter_eval(batch_size=loader.batch_size):
        domain_init  = batch["domain_init"]
        node_kinds   = batch["node_kinds"]
        attn_bias    = batch["attn_bias"]
        gold_values  = batch["gold_values"]
        obs_mask     = batch["observed_mask"]
        ft_t         = batch["factor_types"]
        query_idx_np = batch["query_idx"]
        picks        = batch["picks"]

        if eval_fn is not None:
            pred_t, _cell_acc_t = eval_fn(domain_init, node_kinds, attn_bias, ft_t,
                                           gold_values, obs_mask)
            pred_np = pred_t.numpy()
        else:
            var_logits_history, _ = factor_graph_breathing_forward(
                model, domain_init, node_kinds, attn_bias, ft_t, K=K,
                n_max=n_max, f_max=f_max,
            )
            pred_np = var_logits_history[-1].argmax(axis=-1).realize().numpy()

        gold_np = gold_values.numpy()
        obs_np  = obs_mask.numpy()
        B = len(picks)

        for b in range(B):
            rec = picks[b]
            diff = rec.get("difficulty", "easy")
            if diff not in agg:
                agg[diff] = {"n_unobs": 0, "n_correct_unobs": 0,
                             "query_correct": 0, "n_puzzles": 0}
            qi = int(query_idx_np[b])
            # Only count unobserved positions
            for vi in range(n_max):
                if obs_np[b, vi] == 0 and vi < len(rec["gold_values"]):
                    agg[diff]["n_unobs"] += 1
                    if pred_np[b, vi] == gold_np[b, vi]:
                        agg[diff]["n_correct_unobs"] += 1
            if pred_np[b, qi] == gold_np[b, qi]:
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
        cell_acc = v["n_correct_unobs"] / max(v["n_unobs"], 1)
        q_acc = v["query_correct"] / n
        out[d] = {"cell_acc": cell_acc, "query_acc": q_acc, "n_puzzles": n}

    Tensor.training = True
    return out


def main():
    V99_TASK_LOCAL = int(getenv("V99_TASK", 0)) > 0
    assert V99_TASK_LOCAL, "V99_TASK=1 must be set"

    K      = int(getenv("V99_K_MAX",    str(V99_K_MAX)))
    BATCH  = int(getenv("BATCH",        "8"))
    STEPS  = int(getenv("STEPS",        "2000"))
    LR     = float(getenv("LR",         "3e-5"))
    CKPT_EVERY        = int(getenv("CKPT_EVERY",        "500"))
    EVAL_EVERY        = int(getenv("EVAL_EVERY",        "250"))
    LOG_EVERY         = int(getenv("LOG_EVERY",         "10"))
    PER_BREATH_EVERY  = int(getenv("PER_BREATH_CE_EVERY", "50"))
    GC_EVERY          = int(getenv("GC_EVERY",          "50"))
    CKPT_LABEL        = getenv("CKPT_LABEL",            "v99_smoke")
    RESUME_FROM       = getenv("RESUME_FROM",           "")
    PYTHIA_INIT       = int(getenv("PYTHIA_INIT",       "1")) > 0
    SEED              = int(getenv("SEED",              "42"))

    N_MAX  = int(getenv("V99_N_MAX",  str(V99_N_MAX)))
    F_MAX  = int(getenv("V99_F_MAX",  str(V99_F_MAX)))
    T_MAX  = N_MAX + F_MAX

    DIFFICULTY_FILTER = os.environ.get("V99_DIFFICULTY_FILTER", "").strip() or None
    CURRICULUM        = int(getenv("V99_CURRICULUM",            "0")) > 0
    CURRICULUM_ANNEAL = int(getenv("V99_CURRICULUM_ANNEAL",     "1000"))

    TRAIN_PATH  = getenv("V99_TRAIN", ".cache/factor_graph_train.jsonl")
    VAL_PATH    = getenv("V99_VAL",   ".cache/factor_graph_test.jsonl")
    EVAL_BATCHES = int(getenv("EVAL_BATCHES", "20"))
    EVAL_BATCH   = int(getenv("EVAL_BATCH",   str(BATCH)))

    ENERGY_WEIGHT = float(getenv("V99_ENERGY_WEIGHT", str(V99_ENERGY_WEIGHT)))
    CALIB_WEIGHT  = float(getenv("V99_CALIB_WEIGHT",  str(V99_CALIB_WEIGHT)))

    print(f"=== v99 factor graph training ===")
    print(f"device={Device.DEFAULT}  B={BATCH}  K={K}  steps={STEPS}  lr={LR}")
    print(f"N_MAX={N_MAX}  F_MAX={F_MAX}  T_MAX={T_MAX}")
    print(f"energy_weight={ENERGY_WEIGHT}  calib_weight={CALIB_WEIGHT}")
    print(f"difficulty_filter={DIFFICULTY_FILTER}  curriculum={CURRICULUM}")
    print(f"train={TRAIN_PATH}  val={VAL_PATH}")
    print()

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    # Build model
    cfg = Config()
    print(f"loading Pythia-410M -> breathing transformer (PYTHIA_INIT={PYTHIA_INIT})...")
    if PYTHIA_INIT:
        sd = _load_state()
        model = load_breathing(cfg, sd=sd)
        del sd
    else:
        model = BreathingTransformer(cfg)
    cast_layers_fp32(model)
    attach_fg_params(model, hidden=cfg.hidden, n_heads=cfg.n_heads,
                     k_max=K, n_max=N_MAX, f_max=F_MAX)
    Device[Device.DEFAULT].synchronize()
    params = collect_fg_params(model)
    n_params = sum(int(np.prod(t.shape)) for t in params)
    print(f"  trainable params: {n_params/1e6:.1f}M")

    if RESUME_FROM:
        print(f"resuming from ckpt: {RESUME_FROM}")
        load_ckpt(model, RESUME_FROM)
        print("  loaded.")

    opt = AdamW(params, lr=LR, weight_decay=0.0)

    # Data loaders
    train_loader = FactorGraphLoader(
        TRAIN_PATH, batch_size=BATCH,
        difficulty_filter=DIFFICULTY_FILTER,
        curriculum=CURRICULUM,
        curriculum_anneal_steps=CURRICULUM_ANNEAL,
        n_max=N_MAX, f_max=F_MAX,
        seed=SEED,
    )
    val_loader = FactorGraphLoader(
        VAL_PATH, batch_size=EVAL_BATCH,
        difficulty_filter=None,
        curriculum=False,
        n_max=N_MAX, f_max=F_MAX,
        seed=SEED + 1,
    )

    ckpt_dir = ".cache/fg_ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    # JIT compile the train + eval steps
    Tensor.training = True
    step_fn = _compile_jit_fg_step(
        model, opt, K=K, B=BATCH,
        energy_weight=ENERGY_WEIGHT,
        calib_weight=CALIB_WEIGHT,
        n_max=N_MAX, f_max=F_MAX,
        grad_clip=0.0,
    )
    eval_fn = _compile_jit_fg_eval(model, K=K, B=EVAL_BATCH, n_max=N_MAX, f_max=F_MAX)
    Tensor.training = True

    print(f"\ntraining...\n")
    t0 = time.time()
    log_loss = log_ce = log_calib = log_n = 0.0

    for step in range(1, STEPS + 1):
        batch = train_loader.sample_batch(step=step)

        domain_init = batch["domain_init"]
        node_kinds  = batch["node_kinds"]
        attn_bias   = batch["attn_bias"]
        gold_values = batch["gold_values"]
        obs_mask    = batch["observed_mask"]
        ft_t        = batch["factor_types"]
        fa_t        = batch["factor_args"]

        # JIT'd step: kv_bias is built internally from model.fg_op_embed + factor_types
        # Returns: total, healthy, var_ce, calib, cell_acc, query_acc, *pb_ce
        outs = step_fn(domain_init, node_kinds, attn_bias,
                       gold_values, obs_mask, ft_t, fa_t)
        total_t   = outs[0]
        healthy_t = outs[1]
        ce_t      = outs[2]
        calib_t   = outs[3]
        cell_acc_t  = outs[4]
        query_acc_t = outs[5]
        pb_ce_ts  = outs[6:6 + K]

        if float(healthy_t.numpy()) < 0.5:
            print(f"[NaN-skip] step {step}: skipped", flush=True)

        log_loss  += float(total_t.numpy())
        log_ce    += float(ce_t.numpy())
        log_calib += float(calib_t.numpy())
        log_n     += 1

        if step % LOG_EVERY == 0:
            dt = time.time() - t0
            print(f"[step {step:5d}] loss={log_loss/log_n:.4f} "
                  f"ce={log_ce/log_n:.4f} calib={log_calib/log_n:.4f}  "
                  f"({dt:.1f}s, {dt/step:.2f}s/step)", flush=True)
            log_loss = log_ce = log_calib = log_n = 0.0

        if step % PER_BREATH_EVERY == 0:
            pb_ce = [float(t.numpy()) for t in pb_ce_ts]
            if K <= 8:
                pb_str = " ".join(f"{v:.2f}" for v in pb_ce)
            else:
                head = " ".join(f"{v:.2f}" for v in pb_ce[:4])
                tail = " ".join(f"{v:.2f}" for v in pb_ce[-4:])
                pb_str = f"{head} ... {tail}"
            ca = float(cell_acc_t.numpy())
            qa = float(query_acc_t.numpy())
            print(f"  per_breath_ce[B0..B{K-1}]: {pb_str}  "
                  f"(cell_acc={ca:.3f} query_acc={qa:.3f})", flush=True)

        if step % EVAL_EVERY == 0:
            print(f"  evaluating ({EVAL_BATCHES} batches × B={EVAL_BATCH})...", flush=True)
            results = evaluate(model, val_loader, K=K,
                               max_batches=EVAL_BATCHES, eval_fn=eval_fn,
                               n_max=N_MAX, f_max=F_MAX)
            for d in DIFFICULTIES:
                if d not in results:
                    continue
                v = results[d]
                print(f"  val[{d:6s}]: cell_acc={v['cell_acc']:.3f} "
                      f"query_acc={v['query_acc']:.3f} n={v['n_puzzles']}", flush=True)

        if step % CKPT_EVERY == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_step{step}.safetensors")
            safe_save(model_state_dict_fg(model), ckpt_path)
            print(f"  saved {ckpt_path}", flush=True)

        if step % GC_EVERY == 0:
            gc.collect()

    # Final save
    ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_final.safetensors")
    safe_save(model_state_dict_fg(model), ckpt_path)
    print(f"\ndone. saved {ckpt_path}", flush=True)


DIFFICULTIES = ["easy", "medium", "hard"]

if __name__ == "__main__":
    main()
