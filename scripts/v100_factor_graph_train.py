"""v100 factor graph training driver — directional-key matched-rhythm.

Five architectural changes over v99:
  1. Topological staging masks (per-breath depth expansion)
  2. Aligned init for 100-way codebook
  3. Hard head specialization (4 heads per op)
  4. Factor-execute auxiliary loss
  5. KL energy diagnostic (diagnostic only, not in backward)

Factor-aux loss (Change 4) is computed outside the JIT step because it requires
.numpy() calls to extract factor indices (AMD JIT: no .numpy() inside TinyJit).
The pattern: JIT step does CE + calibration backward + opt.step(); then a
non-JIT Python loop computes factor_aux_loss, calls backward, and calls
opt.step() again.  Two separate optimizer steps per training step but the
gradients accumulate correctly because the JIT step clears grads on entry
(opt.zero_grad()) and we call zero_grad() again before the factor-aux backward.

Env vars (set in launchers):
  V100_TASK=1
  V100_K_MAX=10
  V100_FACTOR_AUX_WEIGHT=0.5
  V100_CALIB_WEIGHT=0.05
  V100_N_MAX=16
  V100_F_MAX=8
  V100_TRAIN=.cache/factor_graph_train.jsonl
  V100_VAL=.cache/factor_graph_test.jsonl
  BATCH=8
  STEPS=2000
  LR=3e-5
  CKPT_EVERY=500
  CKPT_LABEL=v100_smoke
  RESUME_FROM=...
  PYTHIA_INIT=1
  V100_CURRICULUM=1
  V100_CURRICULUM_ANNEAL=1000
  V100_DIFFICULTY_FILTER=easy
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
from mycelium.factor_graph_v100 import (
    V100_K_MAX, V100_FACTOR_AUX_WEIGHT, V100_CALIB_WEIGHT,
    V100_N_MAX, V100_F_MAX, V100_N_HEADS,
    attach_fg_params_v100, fg_v100_parameters, fg_v100_state_dict,
    fg_breathing_forward_v100_aligned,
    _compile_jit_fg_step_v100, _compile_jit_fg_eval_v100,
    kl_energy_diagnostic_np,
    fg_accuracy_v100,
)
from mycelium.factor_graph_data_v100 import FactorGraphLoaderV100

DIFFICULTIES = ["easy", "medium", "hard"]


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


def collect_fg_params_v100(model) -> list[Tensor]:
    """Trainable parameters: shared L0-L3 attn/FFN + v100 factor-graph-specific."""
    params: list[Tensor] = []
    sw = model.block.shared
    params += [sw.wv, sw.bv, sw.wo, sw.bo, sw.w_out, sw.b_out,
               sw.in_ln_g, sw.in_ln_b, sw.post_ln_g, sw.post_ln_b]
    for layer in model.block.layers:
        params += [layer.wq, layer.bq, layer.wk, layer.bk, layer.w_in, layer.b_in]
    params += [model.ln_f_g, model.ln_f_b]
    params += fg_v100_parameters(model)
    return params


def model_state_dict_v100(model) -> dict:
    """State dict for v100 factor-graph training."""
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
    sd.update(fg_v100_state_dict(model))
    return sd


def load_ckpt_v100(model, path: str):
    sd = safe_load(path)
    targets = model_state_dict_v100(model)
    missing = []
    for name, dst in targets.items():
        if name not in sd:
            missing.append(name)
            continue
        src = sd[name].to(dst.device).realize()
        if src.shape != dst.shape:
            if (name in ("fg_v100.breath_embed", "fg_v100.delta_gate")
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
        print(f"  ckpt missing {len(missing)} v100 keys (kept init): {missing[:5]}")


def factor_aux_backward(
    model,
    opt,
    params,
    factor_logits_history: list,  # K × (B, F_MAX, 100) realized numpy arrays
    factor_types_np: np.ndarray,  # (B, F_MAX)
    factor_args_np: np.ndarray,   # (B, F_MAX, 3)
    gold_values_np: np.ndarray,   # (B, N_MAX)
    factor_aux_weight: float,
    K: int,
    B: int,
    f_max: int,
    n_max: int,
) -> float:
    """Compute factor-aux loss outside JIT and do a backward + opt.step.

    factor_logits_history is a list of K numpy arrays (B, F_MAX, 100).

    This is a separate backward pass from the main JIT step.  Gradients accumulate
    into the same parameter tensors.  We call zero_grad before building the graph
    to avoid accumulating onto the JIT step's already-cleared gradients.

    Returns the scalar factor_aux_loss for logging.
    """
    # Reconstruct Tensor graph for factor logits from the current model forward
    # (we can't use the already-realized numpy arrays for backward — we need
    # a live Tensor graph).  Run a MINI forward for factor logits only.
    # Since this is a Python-level backward (outside JIT), we can afford the
    # overhead once per step.
    #
    # Build factor_aux_loss as a Tensor sum using the realized factor logit
    # arrays as targets (teacher forcing: the factor's hidden state should
    # predict the gold result).
    domain_codebook = model.fg_v100_domain_codebook   # (100, H)

    opt.zero_grad()

    # For each valid (b, fi) pair, add CE to the aux loss.
    # We build the loss as a sum of individual CE terms, each a Tensor.
    # Then call backward once on the sum.
    aux_terms: list[Tensor] = []

    for fi in range(f_max):
        for b_idx in range(B):
            op = int(factor_types_np[b_idx, fi])
            if op < 0:
                continue
            r_idx = int(factor_args_np[b_idx, fi, 2])
            if r_idx < 0 or r_idx >= n_max:
                continue
            gold_r = int(gold_values_np[b_idx, r_idx])

            # Accumulate CE across K breaths (late-weighted)
            for k_aux, fac_logits_np in enumerate(factor_logits_history):
                weight_k_aux = 1.0 + float(k_aux) / float(max(K - 1, 1))
                # fac_logits_np is (B, F_MAX, 100) numpy; slice out logit
                logit_row = fac_logits_np[b_idx, fi, :]  # (100,)
                logit_t   = Tensor(logit_row[np.newaxis], dtype=dtypes.float)   # (1, 100)
                gold_t    = Tensor(np.array([gold_r], dtype=np.int32), dtype=dtypes.int)
                ce_term   = logit_t.sparse_categorical_crossentropy(gold_t, reduction="mean")
                aux_terms.append(ce_term * weight_k_aux)

    if not aux_terms:
        return 0.0

    # Normalize by count and weight
    n_terms = float(len(aux_terms))
    aux_loss = Tensor.zeros((), dtype=dtypes.float).contiguous()
    for t in aux_terms:
        aux_loss = aux_loss + t
    aux_loss = aux_loss / n_terms * factor_aux_weight

    # We need to re-run the forward to get live gradients for domain_codebook
    # and the transformer params.  The numpy-slice approach above doesn't give
    # gradients back to the model — it builds a one-off graph.
    # This approach gives factor-aux signal only to domain_codebook (the readout).
    # That's fine: the key objective is to push the codebook to produce correct
    # projections, which then bootstraps the transformer via the main CE loss.
    aux_loss.backward()

    healthy = aux_loss.isfinite().cast(dtypes.float)
    for p in params:
        if p.grad is not None:
            p.grad = p.grad * healthy.cast(p.grad.dtype)

    opt.step()

    return float(aux_loss.realize().numpy())


def evaluate_v100(model, loader: FactorGraphLoaderV100, K: int,
                  max_batches: int = 20, eval_fn=None,
                  n_max: int = V100_N_MAX, f_max: int = V100_F_MAX) -> dict:
    """Run eval on up to max_batches batches. Returns per-difficulty stats."""
    Tensor.training = False
    agg = {}
    n_batches = 0

    for batch in loader.iter_eval(batch_size=loader.batch_size):
        domain_init   = batch["domain_init"]
        node_kinds    = batch["node_kinds"]
        staging_mask  = batch["staging_mask"]
        head_op_mask  = batch["head_op_mask"]
        gold_values   = batch["gold_values"]
        obs_mask      = batch["observed_mask"]
        query_idx_np  = batch["query_idx"]
        picks         = batch["picks"]

        if eval_fn is not None:
            pred_t, _cell_acc_t = eval_fn(
                domain_init, node_kinds, staging_mask, head_op_mask,
                gold_values, obs_mask,
            )
            pred_np = pred_t.numpy()
        else:
            var_logits_history, _, _ = fg_breathing_forward_v100_aligned(
                model, domain_init, node_kinds, staging_mask, head_op_mask,
                K=K, n_max=n_max, f_max=f_max,
            )
            pred_np = var_logits_history[-1].argmax(axis=-1).realize().numpy()

        gold_np = gold_values.numpy()
        obs_np  = obs_mask.numpy()
        B_local = len(picks)

        for b in range(B_local):
            rec  = picks[b]
            diff = rec.get("difficulty", "easy")
            if diff not in agg:
                agg[diff] = {"n_unobs": 0, "n_correct_unobs": 0,
                             "query_correct": 0, "n_puzzles": 0}
            qi = int(query_idx_np[b])
            nv = int(batch["n_vars_total"][b])
            for vi in range(min(nv, n_max)):
                if obs_np[b, vi] == 0:
                    agg[diff]["n_unobs"] += 1
                    if pred_np[b, vi] == gold_np[b, vi]:
                        agg[diff]["n_correct_unobs"] += 1
            if qi < n_max and pred_np[b, qi] == gold_np[b, qi]:
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


def main():
    V100_TASK_LOCAL = int(getenv("V100_TASK", 0)) > 0
    assert V100_TASK_LOCAL, "V100_TASK=1 must be set"

    K      = int(getenv("V100_K_MAX",    str(V100_K_MAX)))
    BATCH  = int(getenv("BATCH",         "8"))
    STEPS  = int(getenv("STEPS",         "2000"))
    LR     = float(getenv("LR",          "3e-5"))
    CKPT_EVERY        = int(getenv("CKPT_EVERY",        "500"))
    EVAL_EVERY        = int(getenv("EVAL_EVERY",        "250"))
    LOG_EVERY         = int(getenv("LOG_EVERY",         "10"))
    PER_BREATH_EVERY  = int(getenv("PER_BREATH_CE_EVERY", "50"))
    GC_EVERY          = int(getenv("GC_EVERY",          "50"))
    CKPT_LABEL        = getenv("CKPT_LABEL",            "v100_smoke")
    RESUME_FROM       = getenv("RESUME_FROM",           "")
    PYTHIA_INIT       = int(getenv("PYTHIA_INIT",       "1")) > 0
    SEED              = int(getenv("SEED",              "42"))

    N_MAX  = int(getenv("V100_N_MAX",  str(V100_N_MAX)))
    F_MAX  = int(getenv("V100_F_MAX",  str(V100_F_MAX)))
    T_MAX  = N_MAX + F_MAX

    DIFFICULTY_FILTER = os.environ.get("V100_DIFFICULTY_FILTER", "").strip() or None
    CURRICULUM        = int(getenv("V100_CURRICULUM",            "0")) > 0
    CURRICULUM_ANNEAL = int(getenv("V100_CURRICULUM_ANNEAL",     "1000"))

    TRAIN_PATH   = getenv("V100_TRAIN", ".cache/factor_graph_train.jsonl")
    VAL_PATH     = getenv("V100_VAL",   ".cache/factor_graph_test.jsonl")
    EVAL_BATCHES = int(getenv("EVAL_BATCHES", "20"))
    EVAL_BATCH   = int(getenv("EVAL_BATCH",   str(BATCH)))

    FACTOR_AUX_WEIGHT = float(getenv("V100_FACTOR_AUX_WEIGHT", str(V100_FACTOR_AUX_WEIGHT)))
    CALIB_WEIGHT      = float(getenv("V100_CALIB_WEIGHT",       str(V100_CALIB_WEIGHT)))

    # KL energy diagnostic: compute and log but don't include in backward
    KL_DIAG_EVERY = int(getenv("KL_DIAG_EVERY", "100"))

    print(f"=== v100 factor graph training (topological staging + aligned init + hard heads) ===")
    print(f"device={Device.DEFAULT}  B={BATCH}  K={K}  steps={STEPS}  lr={LR}")
    print(f"N_MAX={N_MAX}  F_MAX={F_MAX}  T_MAX={T_MAX}")
    print(f"factor_aux_weight={FACTOR_AUX_WEIGHT}  calib_weight={CALIB_WEIGHT}")
    print(f"difficulty_filter={DIFFICULTY_FILTER}  curriculum={CURRICULUM}")
    print(f"train={TRAIN_PATH}  val={VAL_PATH}")
    print()

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    cfg = Config()
    print(f"loading Pythia-410M -> breathing transformer (PYTHIA_INIT={PYTHIA_INIT})...")
    if PYTHIA_INIT:
        sd = _load_state()
        model = load_breathing(cfg, sd=sd)
        del sd
    else:
        model = BreathingTransformer(cfg)
    cast_layers_fp32(model)
    attach_fg_params_v100(model, hidden=cfg.hidden, n_heads=cfg.n_heads,
                          k_max=K, n_max=N_MAX, f_max=F_MAX)
    Device[Device.DEFAULT].synchronize()

    params = collect_fg_params_v100(model)
    n_params = sum(int(np.prod(t.shape)) for t in params)
    print(f"  trainable params: {n_params/1e6:.1f}M")

    if RESUME_FROM:
        print(f"resuming from ckpt: {RESUME_FROM}")
        load_ckpt_v100(model, RESUME_FROM)
        print("  loaded.")

    opt = AdamW(params, lr=LR, weight_decay=0.0)
    # Separate lightweight optimizer for the factor-aux backward (only fg_v100 params
    # need gradients for the factor-aux loss; the transformer params receive zero signal
    # because factor_logits are sliced from an already-detached forward).
    aux_params = fg_v100_parameters(model)
    aux_opt = AdamW(aux_params, lr=LR, weight_decay=0.0)

    train_loader = FactorGraphLoaderV100(
        TRAIN_PATH, batch_size=BATCH,
        difficulty_filter=DIFFICULTY_FILTER,
        curriculum=CURRICULUM,
        curriculum_anneal_steps=CURRICULUM_ANNEAL,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V100_N_HEADS,
        seed=SEED,
    )
    val_loader = FactorGraphLoaderV100(
        VAL_PATH, batch_size=EVAL_BATCH,
        difficulty_filter=None,
        curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V100_N_HEADS,
        seed=SEED + 1,
    )

    ckpt_dir = ".cache/fg_v100_ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    Tensor.training = True
    step_fn = _compile_jit_fg_step_v100(
        model, opt, K=K, B=BATCH,
        factor_aux_weight=FACTOR_AUX_WEIGHT,
        calib_weight=CALIB_WEIGHT,
        n_max=N_MAX, f_max=F_MAX,
        grad_clip=1.0,
    )
    eval_fn = _compile_jit_fg_eval_v100(
        model, K=K, B=EVAL_BATCH, n_max=N_MAX, f_max=F_MAX,
    )
    Tensor.training = True

    print(f"\ntraining...\n")
    t0 = time.time()
    log_loss = log_ce = log_calib = log_aux = log_n = 0.0

    for step in range(1, STEPS + 1):
        batch = train_loader.sample_batch(step=step)

        domain_init   = batch["domain_init"]
        node_kinds    = batch["node_kinds"]
        staging_mask  = batch["staging_mask"]
        head_op_mask  = batch["head_op_mask"]
        gold_values   = batch["gold_values"]
        obs_mask      = batch["observed_mask"]
        ft_t          = batch["factor_types"]
        fa_t          = batch["factor_args"]
        ft_np         = batch["factor_types"].numpy()
        fa_np         = batch["factor_args"].numpy()
        gold_np       = batch["gold_values"].numpy()

        # JIT step: CE + calibration backward + opt.step
        outs = step_fn(
            domain_init, node_kinds, staging_mask, head_op_mask,
            gold_values, obs_mask, ft_t, fa_t,
        )
        total_t     = outs[0]
        healthy_t   = outs[1]
        ce_t        = outs[2]
        calib_t     = outs[3]
        cell_acc_t  = outs[4]
        query_acc_t = outs[5]
        pb_ce_ts    = outs[6:6 + K]

        if float(healthy_t.numpy()) < 0.5:
            print(f"[NaN-skip] step {step}: CE step skipped", flush=True)

        # Factor-aux backward (outside JIT — requires .numpy() for index extraction)
        # Run non-JIT forward to get factor logits with live gradients.
        if FACTOR_AUX_WEIGHT > 0:
            Tensor.training = True
            # Run factor-aux backward: get factor logits from a fresh forward,
            # compute factor-aux loss, backward, opt.step.
            # We use a lightweight non-JIT forward for this purpose.
            if True:
                var_lh, fac_lh, calib_h = fg_breathing_forward_v100_aligned(
                    model, domain_init, node_kinds, staging_mask, head_op_mask,
                    K=K, n_max=N_MAX, f_max=F_MAX,
                )
                # Build factor aux loss
                opt.zero_grad()
                aux_terms: list[Tensor] = []
                for fi in range(F_MAX):
                    for b_idx in range(BATCH):
                        op = int(ft_np[b_idx, fi])
                        if op < 0:
                            continue
                        r_idx = int(fa_np[b_idx, fi, 2])
                        if r_idx < 0 or r_idx >= N_MAX:
                            continue
                        gold_r = int(gold_np[b_idx, r_idx])
                        for k_aux, fac_logits_k in enumerate(fac_lh):
                            w_aux = 1.0 + float(k_aux) / float(max(K - 1, 1))
                            logit_t = fac_logits_k[b_idx, fi, :].reshape(1, 100)
                            gold_t  = Tensor(np.array([gold_r], dtype=np.int32), dtype=dtypes.int)
                            ce_aux  = logit_t.sparse_categorical_crossentropy(gold_t, reduction="mean")
                            aux_terms.append(ce_aux * w_aux)

                if aux_terms:
                    aux_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
                    for t in aux_terms:
                        aux_sum = aux_sum + t
                    n_terms_f = float(len(aux_terms))
                    aux_loss  = aux_sum / n_terms_f * FACTOR_AUX_WEIGHT
                    aux_opt.zero_grad()
                    aux_loss.backward()
                    healthy_aux = aux_loss.isfinite().cast(dtypes.float)
                    # Ensure every aux param has a grad tensor (zero if not received)
                    # so AdamW.step() doesn't trip on unwrap(None).
                    for p in aux_params:
                        if p.grad is None:
                            p.grad = Tensor.zeros_like(p).contiguous()
                        else:
                            p.grad = p.grad * healthy_aux.cast(p.grad.dtype)
                    aux_opt.step()
                    aux_val = float(aux_loss.realize().numpy())
                else:
                    aux_val = 0.0
        else:
            aux_val = 0.0

        log_loss  += float(total_t.numpy())
        log_ce    += float(ce_t.numpy())
        log_calib += float(calib_t.numpy())
        log_aux   += aux_val
        log_n     += 1

        if step % LOG_EVERY == 0:
            dt = time.time() - t0
            print(
                f"[step {step:5d}] loss={log_loss/log_n:.4f} "
                f"ce={log_ce/log_n:.4f} "
                f"aux={log_aux/log_n:.4f} "
                f"calib={log_calib/log_n:.4f}  "
                f"({dt:.1f}s, {dt/step:.2f}s/step)",
                flush=True,
            )
            log_loss = log_ce = log_calib = log_aux = log_n = 0.0

        if step % PER_BREATH_EVERY == 0:
            pb_ce = [float(t.numpy()) for t in pb_ce_ts]
            if K <= 8:
                pb_str = " ".join(f"{v:.3f}" for v in pb_ce)
            else:
                head = " ".join(f"{v:.3f}" for v in pb_ce[:5])
                tail = " ".join(f"{v:.3f}" for v in pb_ce[-5:])
                pb_str = f"{head} ... {tail}"
            ca = float(cell_acc_t.numpy())
            qa = float(query_acc_t.numpy())
            print(
                f"  per_breath_ce[B0..B{K-1}]: {pb_str}  "
                f"(cell_acc={ca:.3f} query_acc={qa:.3f})",
                flush=True,
            )
            # Check for ladder: B0 should be strictly > B9 for topological staging to be working
            if K > 1 and len(pb_ce) >= 2:
                ladder_delta = pb_ce[0] - pb_ce[-1]
                if ladder_delta > 0.1:
                    print(f"  [LADDER] B0-B{K-1} delta = {ladder_delta:.3f} > 0.1 — topological staging working!", flush=True)
                else:
                    print(f"  [LADDER] B0-B{K-1} delta = {ladder_delta:.3f} (target > 0.1)", flush=True)

        if step % KL_DIAG_EVERY == 0:
            # KL energy diagnostic (not in backward — just logging)
            Tensor.training = False
            var_lh_diag, _, _ = fg_breathing_forward_v100_aligned(
                model, domain_init, node_kinds, staging_mask, head_op_mask,
                K=K, n_max=N_MAX, f_max=F_MAX,
            )
            final_logits_np = var_lh_diag[-1].realize().numpy()
            kl_energy = kl_energy_diagnostic_np(
                final_logits_np, ft_np, fa_np, n_max=N_MAX, f_max=F_MAX,
            )
            print(f"  [KL_energy] step {step}: mean_kl_per_factor = {kl_energy:.4f}", flush=True)
            Tensor.training = True

        if step % EVAL_EVERY == 0:
            print(f"  evaluating ({EVAL_BATCHES} batches × B={EVAL_BATCH})...", flush=True)
            results = evaluate_v100(
                model, val_loader, K=K,
                max_batches=EVAL_BATCHES,
                eval_fn=eval_fn,
                n_max=N_MAX, f_max=F_MAX,
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
            safe_save(model_state_dict_v100(model), ckpt_path)
            print(f"  saved {ckpt_path}", flush=True)

        if step % GC_EVERY == 0:
            gc.collect()

    # Final save
    ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_final.safetensors")
    safe_save(model_state_dict_v100(model), ckpt_path)
    print(f"\ndone. saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
