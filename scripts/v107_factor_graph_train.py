"""v107 factor graph training driver — hybrid 200-bin codebook architecture.

Architectural pivot from v105 (digit-by-digit, 0% val acc) to v107 (200-way
single-bin prediction, fully correlated). Model predicts ONE bin per variable,
not independent digits. Digits are only derived from bins for MCTS navigation.

Two codebooks:
  domain_codebook  (200, H): value vocabulary (hybrid 100 linear + 50 log + 50 log)
  semantic_codebook (32, H): IB semantic centroids (operator-role vocabulary)

Warm-start from v104:
  - Copy shared.*, phase*.* (transformer backbone)
  - Copy fg_v104.codebook → fg_v107.semantic_codebook
  - Reinitialize fg_v107.domain_codebook (200, H) random orthonormal
  - Skip fg_v100.domain_codebook (100-way, wrong shape)

Training data (50/50 mix):
  - 50K synthetic factor graphs (values [0,99])
  - 4261 GSM8K factor graphs (full value range, no filtering)

Env vars:
  V107_TASK=1
  V107_BIN_COUNT=200
  V107_K_MAX=10
  V107_N_MAX=16
  V107_F_MAX=8
  V107_ENERGY_WEIGHT=0.01
  V107_CALIB_WEIGHT=0.05
  V107_FACTOR_AUX_WEIGHT=0.5
  V107_CODEBOOK_N=32
  V107_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz
  V107_TRAIN=.cache/factor_graph_train.jsonl
  V107_VAL=.cache/factor_graph_test.jsonl
  V107_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl
  V107_GSM8K_RATIO=0.5
  BATCH=8
  STEPS=3000
  LR=3e-5
  CKPT_EVERY=500
  EVAL_EVERY=250
  CKPT_LABEL=v107_smoke
  RESUME_FROM=.cache/fg_v104_ckpts/v104_prod_step3000.safetensors
  PYTHIA_INIT=1
"""
import gc
import json
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
from mycelium.factor_graph_v107 import (
    V107_K_MAX, V107_N_MAX, V107_F_MAX, V107_N_HEADS,
    V107_ENERGY_WEIGHT, V107_CALIB_WEIGHT, V107_FACTOR_AUX_WEIGHT,
    V107_CODEBOOK_N, V107_IB_CENTROIDS,
    attach_fg_params_v107, fg_v107_parameters, fg_v107_state_dict,
    fg_breathing_forward_v107,
    _compile_jit_fg_step_v107, _compile_jit_fg_eval_v107,
    get_bin_values,
)
from mycelium.factor_graph_data_v107 import (
    FactorGraphLoaderV107, DualDataLoaderV107, load_gsm8k_records_v107,
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


def collect_fg_params_v107(model) -> list[Tensor]:
    """Trainable parameters: shared L0-L3 attn/FFN + v107 fg params."""
    params: list[Tensor] = []
    sw = model.block.shared
    params += [sw.wv, sw.bv, sw.wo, sw.bo, sw.w_out, sw.b_out,
               sw.in_ln_g, sw.in_ln_b, sw.post_ln_g, sw.post_ln_b]
    for layer in model.block.layers:
        params += [layer.wq, layer.bq, layer.wk, layer.bk, layer.w_in, layer.b_in]
    params += [model.ln_f_g, model.ln_f_b]
    params += fg_v107_parameters(model)
    return params


def model_state_dict_v107(model) -> dict:
    """State dict for v107 factor-graph training."""
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
    sd.update(fg_v107_state_dict(model))
    return sd


def load_ckpt_v107(model, path: str):
    """Load a v104 or v107 checkpoint into v107 model.

    Key warm-start logic:
      shared.* and phase*.* keys: loaded directly (transformer backbone preserved).
      fg_v104.codebook → fg_v107.semantic_codebook (IB centroids).
      fg_v107.* keys: loaded if present (v107 resume), skipped if missing (v104 start).
      fg_v100.* and fg_v104.* keys other than codebook: silently skipped.
    """
    sd = safe_load(path)
    targets = model_state_dict_v107(model)
    missing = []
    loaded  = []

    for name, dst in targets.items():
        if name in sd:
            src = sd[name].to(dst.device).realize()
            if src.shape != dst.shape:
                # Shape mismatch: v104 domain_codebook is (100, H) — skip
                missing.append(f"{name}(shape {src.shape}!={dst.shape})")
                continue
            if src.dtype != dst.dtype:
                src = src.cast(dst.dtype)
            dst.assign(src).realize()
            loaded.append(name)
        else:
            missing.append(name)

    # Cross-architecture copy: fg_v104.codebook → fg_v107.semantic_codebook
    if "fg_v107.semantic_codebook" not in [k for k in loaded]:
        v104_key = "fg_v104.codebook"
        if v104_key in sd:
            src = sd[v104_key].to(model.fg_v107_semantic_codebook.device).realize()
            dst = model.fg_v107_semantic_codebook
            if src.shape == dst.shape:
                if src.dtype != dst.dtype:
                    src = src.cast(dst.dtype)
                dst.assign(src).realize()
                print(f"  warm-start: {v104_key} → fg_v107.semantic_codebook", flush=True)
            else:
                print(f"  WARN: {v104_key} shape {src.shape} != {dst.shape}; IB init kept",
                      flush=True)

    v107_keys  = [k for k in targets if k.startswith("fg_v107.")]
    loaded_v107 = [k for k in loaded if k.startswith("fg_v107.")]
    backbone_keys = [k for k in loaded if k.startswith("shared.") or k.startswith("phase")]
    print(f"  loaded {len(backbone_keys)} backbone keys, "
          f"{len(loaded_v107)}/{len(v107_keys)} v107 keys",
          flush=True)
    if missing:
        non_trivial = [k for k in missing if not k.startswith("fg_v100.")
                       and not k.startswith("fg_v104.")]
        if non_trivial:
            print(f"  missing {len(non_trivial)} non-trivial keys: {non_trivial[:5]}", flush=True)


def evaluate_v107(model, loader: FactorGraphLoaderV107, K: int,
                  max_batches: int = 20, eval_fn=None,
                  n_max: int = V107_N_MAX, f_max: int = V107_F_MAX,
                  bin_values_np: np.ndarray | None = None) -> dict:
    """Run eval on up to max_batches batches. Returns per-difficulty stats."""
    Tensor.training = False
    agg = {}
    n_batches = 0
    bv = get_bin_values() if bin_values_np is None else bin_values_np

    for batch in loader.iter_eval(batch_size=loader.batch_size):
        domain_init   = batch["domain_init"]
        node_kinds    = batch["node_kinds"]
        staging_mask  = batch["staging_mask"]
        head_op_mask  = batch["head_op_mask"]
        gold_bins     = batch["gold_bins"]
        obs_mask      = batch["observed_mask"]
        query_idx_np  = batch["query_idx"]
        picks         = batch["picks"]

        if eval_fn is not None:
            pred_t, _ = eval_fn(
                domain_init, node_kinds, staging_mask, head_op_mask,
                gold_bins, obs_mask,
            )
            pred_np = pred_t.numpy()
        else:
            var_lh, _, _ = fg_breathing_forward_v107(
                model, domain_init, node_kinds, staging_mask, head_op_mask,
                K=K, n_max=n_max, f_max=f_max,
            )
            pred_np = var_lh[-1].argmax(axis=-1).realize().numpy()

        gold_np = gold_bins.numpy()
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
        cell_acc = v["n_correct_unobs"] / max(v["n_unobs"], 1)
        q_acc    = v["query_correct"] / n
        out[d]   = {"cell_acc": cell_acc, "query_acc": q_acc, "n_puzzles": n}

    Tensor.training = True
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    V107_TASK_LOCAL = int(getenv("V107_TASK", 0)) > 0
    assert V107_TASK_LOCAL, "V107_TASK=1 must be set"

    K      = int(getenv("V107_K_MAX",    str(V107_K_MAX)))
    BATCH  = int(getenv("BATCH",         "8"))
    STEPS  = int(getenv("STEPS",         "3000"))
    LR     = float(getenv("LR",          "3e-5"))
    CKPT_EVERY       = int(getenv("CKPT_EVERY",          "500"))
    EVAL_EVERY       = int(getenv("EVAL_EVERY",          "250"))
    LOG_EVERY        = int(getenv("LOG_EVERY",           "10"))
    PER_BREATH_EVERY = int(getenv("PER_BREATH_CE_EVERY", "50"))
    GC_EVERY         = int(getenv("GC_EVERY",            "50"))
    CKPT_LABEL       = getenv("CKPT_LABEL",              "v107_smoke")
    RESUME_FROM      = getenv("RESUME_FROM",             "")
    PYTHIA_INIT      = int(getenv("PYTHIA_INIT",         "1")) > 0
    SEED             = int(getenv("SEED",                "42"))
    N_CODE           = int(getenv("V107_CODEBOOK_N",     str(V107_CODEBOOK_N)))
    IB_PATH          = getenv("V107_IB_CENTROIDS",       V107_IB_CENTROIDS)
    ENERGY_WEIGHT    = float(getenv("V107_ENERGY_WEIGHT", str(V107_ENERGY_WEIGHT)))
    FACTOR_AUX_WEIGHT = float(getenv("V107_FACTOR_AUX_WEIGHT", str(V107_FACTOR_AUX_WEIGHT)))
    CALIB_WEIGHT     = float(getenv("V107_CALIB_WEIGHT",  str(V107_CALIB_WEIGHT)))

    N_MAX  = int(getenv("V107_N_MAX",   str(V107_N_MAX)))
    F_MAX  = int(getenv("V107_F_MAX",   str(V107_F_MAX)))

    DIFFICULTY_FILTER = (
        os.environ.get("V107_DIFFICULTY_FILTER", "").strip() or None
    )
    CURRICULUM        = int(getenv("V107_CURRICULUM",        "0")) > 0
    CURRICULUM_ANNEAL = int(getenv("V107_CURRICULUM_ANNEAL", "1000"))

    TRAIN_PATH   = getenv("V107_TRAIN",       ".cache/factor_graph_train.jsonl")
    VAL_PATH     = getenv("V107_VAL",         ".cache/factor_graph_test.jsonl")
    GSM8K_PATH   = getenv("V107_GSM8K_TRAIN", ".cache/gsm8k_factor_graphs_train.jsonl")
    GSM8K_RATIO  = float(getenv("V107_GSM8K_RATIO", "0.5"))
    EVAL_BATCHES = int(getenv("EVAL_BATCHES", "20"))
    EVAL_BATCH   = int(getenv("EVAL_BATCH",   str(BATCH)))

    print("=== v107 factor graph training (hybrid 200-bin codebook) ===")
    print(f"device={Device.DEFAULT}  B={BATCH}  K={K}  steps={STEPS}  lr={LR}")
    print(f"N_MAX={N_MAX}  F_MAX={F_MAX}  T_MAX={N_MAX+F_MAX}  n_bins=200")
    print(f"semantic_codebook_n={N_CODE}  ib_centroids={IB_PATH}")
    print(f"factor_aux_weight={FACTOR_AUX_WEIGHT}  calib_weight={CALIB_WEIGHT}  "
          f"energy_weight={ENERGY_WEIGHT}")
    print(f"difficulty_filter={DIFFICULTY_FILTER}  curriculum={CURRICULUM}")
    print(f"train={TRAIN_PATH}  val={VAL_PATH}")
    print(f"gsm8k_train={GSM8K_PATH}  gsm8k_ratio={GSM8K_RATIO}")
    print()

    # Print bin distribution stats
    bv = get_bin_values()
    print(f"[bin_values] linear [0,{bv[99]}] (bins 0-99)  "
          f"log-1 [{bv[100]},{bv[149]}] (bins 100-149)  "
          f"log-2 [{bv[150]},{bv[199]}] (bins 150-199)")
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

    # Attach v107 params (200-way domain codebook + IB semantic codebook)
    attach_fg_params_v107(
        model, hidden=cfg.hidden,
        n_max=N_MAX, f_max=F_MAX, k_max=K,
        n_code=N_CODE, ib_centroids_path=IB_PATH,
    )
    Device[Device.DEFAULT].synchronize()

    params   = collect_fg_params_v107(model)
    n_params = sum(int(np.prod(t.shape)) for t in params)
    print(f"  trainable params: {n_params/1e6:.1f}M")

    if RESUME_FROM:
        print(f"loading ckpt: {RESUME_FROM}")
        load_ckpt_v107(model, RESUME_FROM)
        print("  loaded.")

    opt = AdamW(params, lr=LR, weight_decay=0.0)

    # Bin-values tensor (constant, passed to JIT for energy computation)
    bin_values_t = Tensor(bv.astype(np.float32), dtype=dtypes.float).contiguous().realize()

    # Data loaders
    synth_loader = FactorGraphLoaderV107(
        TRAIN_PATH, batch_size=BATCH,
        difficulty_filter=DIFFICULTY_FILTER,
        curriculum=CURRICULUM,
        curriculum_anneal_steps=CURRICULUM_ANNEAL,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V107_N_HEADS,
        seed=SEED,
    )
    gsm8k_records = load_gsm8k_records_v107(GSM8K_PATH)
    dual_loader   = DualDataLoaderV107(
        synth_loader, gsm8k_records,
        gsm8k_ratio=GSM8K_RATIO,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V107_N_HEADS,
        seed=SEED + 1,
    )
    val_loader = FactorGraphLoaderV107(
        VAL_PATH, batch_size=EVAL_BATCH,
        difficulty_filter=None, curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V107_N_HEADS,
        seed=SEED + 2,
    )

    # --- separate GSM8K eval loader (no difficulty filter) ---
    gsm8k_val_loader = None
    if gsm8k_records:
        from mycelium.factor_graph_data_v107 import _records_to_batch_v107, batch_to_tensors_v107

        class _GsmEvalLoader:
            def __init__(self, records, bs, n_max, f_max, k_max, n_heads):
                self.records  = records
                self.batch_size = bs
                self.n_max = n_max; self.f_max = f_max
                self.k_max = k_max; self.n_heads = n_heads
            def iter_eval(self, batch_size=None):
                bs = batch_size or self.batch_size
                for start in range(0, len(self.records), bs):
                    recs = self.records[start:start+bs]
                    while len(recs) < bs:
                        recs.append(self.records[0])
                    bt = batch_to_tensors_v107(
                        _records_to_batch_v107(recs, self.n_max, self.f_max, self.k_max, self.n_heads)
                    )
                    bt["picks"] = recs
                    yield bt
            def __len__(self):
                return len(self.records)

        gsm8k_val_loader = _GsmEvalLoader(gsm8k_records, EVAL_BATCH, N_MAX, F_MAX, K, V107_N_HEADS)
        print(f"[v107] GSM8K eval loader: {len(gsm8k_records)} records", flush=True)

    ckpt_dir = ".cache/fg_v107_ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    Tensor.training = True
    step_fn = _compile_jit_fg_step_v107(
        model, opt, K=K, B=BATCH,
        factor_aux_weight=FACTOR_AUX_WEIGHT,
        calib_weight=CALIB_WEIGHT,
        energy_weight=ENERGY_WEIGHT,
        n_max=N_MAX, f_max=F_MAX,
        grad_clip=1.0,
    )
    eval_fn = _compile_jit_fg_eval_v107(
        model, K=K, B=EVAL_BATCH, n_max=N_MAX, f_max=F_MAX,
    )
    Tensor.training = True

    print(f"\ntraining...\n")
    t0 = time.time()
    log_loss = log_ce = log_calib = log_aux = log_energy = log_n = 0.0

    for step in range(1, STEPS + 1):
        batch = dual_loader.sample_batch(step=step)

        domain_init   = batch["domain_init"]
        node_kinds    = batch["node_kinds"]
        staging_mask  = batch["staging_mask"]
        head_op_mask  = batch["head_op_mask"]
        gold_bins     = batch["gold_bins"]
        obs_mask      = batch["observed_mask"]
        ft_np         = batch["factor_types"].numpy()
        fa_np         = batch["factor_args"].numpy()
        fgb_t         = batch["factor_gold_bin"]
        fv_t          = batch["factor_valid"]
        factor_types_t = batch["factor_types"]
        factor_args_t  = batch["factor_args"]

        outs = step_fn(
            domain_init, node_kinds, staging_mask, head_op_mask,
            gold_bins, obs_mask,
            fgb_t, fv_t,
            factor_types_t, factor_args_t,
            bin_values_t,
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
            if K > 1 and len(pb_ce) >= 2:
                ladder_delta = pb_ce[0] - pb_ce[-1]
                print(
                    f"  [LADDER] B0-B{K-1} delta = {ladder_delta:.3f} "
                    f"({'OK' if ladder_delta > 0.1 else 'target > 0.1'})",
                    flush=True,
                )

        if step % EVAL_EVERY == 0:
            print(f"  evaluating synthetic ({EVAL_BATCHES} batches × B={EVAL_BATCH})...",
                  flush=True)
            results = evaluate_v107(
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
            # Also eval on GSM8K subset
            if gsm8k_val_loader is not None:
                gsm_results = evaluate_v107(
                    model, gsm8k_val_loader, K=K,
                    max_batches=EVAL_BATCHES,
                    eval_fn=eval_fn,
                    n_max=N_MAX, f_max=F_MAX,
                )
                for d, v in gsm_results.items():
                    print(
                        f"  gsm8k[{d:6s}]: cell_acc={v['cell_acc']:.3f} "
                        f"query_acc={v['query_acc']:.3f} n={v['n_puzzles']}",
                        flush=True,
                    )

        if step % CKPT_EVERY == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_step{step}.safetensors")
            safe_save(model_state_dict_v107(model), ckpt_path)
            print(f"  saved {ckpt_path}", flush=True)

        if step % GC_EVERY == 0:
            gc.collect()

    ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_final.safetensors")
    safe_save(model_state_dict_v107(model), ckpt_path)
    print(f"\ndone. saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
