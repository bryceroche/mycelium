"""Matched eval: v107 vs v105.8 vs v105.10 on the synth val set.

All three ckpts evaluated on the SAME factor_graph_test_loguniform.jsonl
split, per-difficulty (easy / medium / hard), with both cell_acc and
query_acc reported.

  * v107:   per-NUMBER 200-bin codebook (no per-position tokens)
  * v105.8: per-NUMBER 200-bin codebook on per-position-token architecture
  * v105.10 num path:  per-NUMBER 200-bin codebook on dual-readout arch
  * v105.10 pool path: AR digit decoder on dual-readout arch
                       (digit-decomposed prediction from cell_hidden)

Apples-to-apples bin-comparable: v107, v105.8, and v105.10-num all use the
same 200-bin hybrid scheme (build_bin_values), so a cell is "correct" iff
the predicted bin equals the gold bin. Pool path uses the per-digit equality
rule (all valid digit positions match gold).

Usage:
  .venv/bin/python scripts/eval_v107_v105_compare.py
  V107_CKPT=...   V105_8_CKPT=...   V105_10_CKPT=...   \
    .venv/bin/python scripts/eval_v107_v105_compare.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default ckpt paths (overridable via env).
V107_CKPT_DEFAULT    = ".cache/fg_v107_ckpts/v107_prod_step1000.safetensors"
V105_8_CKPT_DEFAULT  = ".cache/fg_v105_5_ckpts/v105_8_prod_step5000.safetensors"
V105_10_CKPT_DEFAULT = ".cache/fg_v105_5_ckpts/v105_10_prod_step5000.safetensors"
VAL_PATH_DEFAULT     = ".cache/factor_graph_test_loguniform.jsonl"


def _print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _print_results(results: dict, label: str, n_check: int) -> None:
    print(f"\n{label}  (n_batches={n_check})")
    for d in ("easy", "medium", "hard"):
        v = results.get(d)
        if v is None:
            continue
        print(
            f"  val[{d:6s}]: cell_acc={v['cell_acc']:.4f}  "
            f"query_acc={v['query_acc']:.4f}  n={v['n_puzzles']}"
        )


# ---------------------------------------------------------------------------
# v107 eval (separate process state — its module-level constants don't share
# globals with v105.5 since the architectures are different)
# ---------------------------------------------------------------------------

def eval_v107(ckpt_path: str, val_path: str, batch: int, max_batches: int,
              seed: int = 42) -> dict | None:
    if not os.path.exists(ckpt_path):
        print(f"[v107] ckpt missing — skipping: {ckpt_path}", flush=True)
        return None

    import numpy as np
    from tinygrad import Device, Tensor, dtypes  # noqa: F401
    from mycelium import Config, BreathingTransformer
    from mycelium.loader import _load_state, load_breathing
    from mycelium.factor_graph_v107 import (
        V107_K_MAX, V107_N_MAX, V107_F_MAX, V107_N_HEADS,
        attach_fg_params_v107, fg_breathing_forward_v107,
        _compile_jit_fg_eval_v107,
    )
    from mycelium.factor_graph_data_v107 import FactorGraphLoaderV107

    _print_header(f"v107 eval — {os.path.basename(ckpt_path)}")

    np.random.seed(seed)
    Tensor.manual_seed(seed)

    cfg = Config()
    sd  = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    # fp32 cast
    for obj_attrs in (
        (model.embed, ["weight"]),
        (model.block.shared, ["wv", "bv", "wo", "bo", "w_out", "b_out"]),
    ):
        obj, attrs = obj_attrs
        for a in attrs:
            t = getattr(obj, a)
            if t.dtype == dtypes.half:
                setattr(obj, a, t.cast(dtypes.float).contiguous().realize())
    for layer in model.block.layers:
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            t = getattr(layer, a)
            if t.dtype == dtypes.half:
                setattr(layer, a, t.cast(dtypes.float).contiguous().realize())

    attach_fg_params_v107(model, hidden=cfg.hidden)
    Device[Device.DEFAULT].synchronize()

    # Load checkpoint
    from tinygrad.nn.state import safe_load
    sd_ck = safe_load(ckpt_path)
    # Standard backbone + v107-specific keys.
    backbone_keys = [("ln_f.g", model.ln_f_g), ("ln_f.b", model.ln_f_b)]
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out",
              "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b"):
        backbone_keys.append((f"shared.{a}", getattr(sw, a)))
    for i, layer in enumerate(model.block.layers):
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            backbone_keys.append((f"phase{i}.{a}", getattr(layer, a)))
    # v107 fg params
    for name, attr in [
        ("fg_v107.domain_codebook",   "fg_v107_domain_codebook"),
        ("fg_v107.var_state_embed",   "fg_v107_var_state_embed"),
        ("fg_v107.var_pos_embed",     "fg_v107_var_pos_embed"),
        ("fg_v107.factor_pos_embed",  "fg_v107_factor_pos_embed"),
        ("fg_v107.node_kind_embed",   "fg_v107_node_kind_embed"),
        ("fg_v107.breath_embed",      "fg_v107_breath_embed"),
        ("fg_v107.delta_gate",        "fg_v107_delta_gate"),
        ("fg_v107.calib_head_w",      "fg_v107_calib_head_w"),
        ("fg_v107.calib_head_b",      "fg_v107_calib_head_b"),
        ("fg_v107.semantic_codebook", "fg_v107_semantic_codebook"),
        ("fg_v107.delta_gate_quant",  "fg_v107_delta_gate_quant"),
        ("fg_v107.temperature",       "fg_v107_temperature"),
    ]:
        if hasattr(model, attr) and name in sd_ck:
            backbone_keys.append((name, getattr(model, attr)))
    loaded = 0
    for name, dst in backbone_keys:
        if name in sd_ck:
            src = sd_ck[name].to(dst.device).realize()
            if src.shape == dst.shape:
                if src.dtype != dst.dtype:
                    src = src.cast(dst.dtype)
                dst.assign(src).realize()
                loaded += 1
    print(f"  loaded {loaded}/{len(backbone_keys)} backbone+v107 keys", flush=True)

    loader = FactorGraphLoaderV107(
        val_path, batch_size=batch, difficulty_filter=None, curriculum=False,
        seed=seed,
    )

    Tensor.training = False
    eval_fn = _compile_jit_fg_eval_v107(
        model, K=V107_K_MAX, B=batch,
        n_max=V107_N_MAX, f_max=V107_F_MAX,
    )

    # Run eval — collect per-difficulty stats.
    agg = {}
    n_batches = 0
    for batch_d in loader.iter_eval(batch_size=batch):
        digit_init   = batch_d["domain_init"]
        node_kinds   = batch_d["node_kinds"]
        staging_mask = batch_d["staging_mask"]
        head_op_mask = batch_d["head_op_mask"]
        gold_bins    = batch_d["gold_bins"]
        obs_mask     = batch_d["observed_mask"]
        query_idx_np = batch_d["query_idx"]
        picks        = batch_d["picks"]

        pred_bin_t, _ = eval_fn(
            digit_init, node_kinds, staging_mask, head_op_mask,
            gold_bins, obs_mask,
        )
        pred_bin_np = pred_bin_t.numpy()
        gold_bin_np = gold_bins.numpy()
        obs_np      = obs_mask.numpy()

        for b in range(len(picks)):
            rec  = picks[b]
            diff = rec.get("difficulty", "easy")
            if diff not in agg:
                agg[diff] = {"n_unobs": 0, "n_correct_unobs": 0,
                             "query_correct": 0, "n_puzzles": 0}
            qi = int(query_idx_np[b])
            nv = int(batch_d["n_vars_total"][b])
            for vi in range(min(nv, V107_N_MAX)):
                if obs_np[b, vi] != 0:
                    continue
                agg[diff]["n_unobs"] += 1
                if int(pred_bin_np[b, vi]) == int(gold_bin_np[b, vi]):
                    agg[diff]["n_correct_unobs"] += 1
            if qi < V107_N_MAX:
                if int(pred_bin_np[b, qi]) == int(gold_bin_np[b, qi]):
                    agg[diff]["query_correct"] += 1
            agg[diff]["n_puzzles"] += 1

        n_batches += 1
        if n_batches >= max_batches:
            break

    out = {}
    for d, v in agg.items():
        out[d] = {
            "cell_acc":  v["n_correct_unobs"] / max(v["n_unobs"], 1),
            "query_acc": v["query_correct"]   / max(v["n_puzzles"], 1),
            "n_puzzles": v["n_puzzles"],
        }
    return out


# ---------------------------------------------------------------------------
# v105.8 / v105.10 eval (share the v105.5 architecture/loader; just env-gate
# which eval JIT runs)
# ---------------------------------------------------------------------------

def eval_v105_5_family(
    ckpt_path: str, val_path: str, batch: int, max_batches: int,
    seed: int, eval_kind: str, label: str,
) -> tuple[dict | None, dict | None]:
    """Eval a v105.5-family ckpt. `eval_kind` selects which readout to use:
      "v8" : 200-bin number readout only
      "v10": BOTH v8 (200-bin) and v9 (pooled-AR digit decoder), side-by-side
    Returns (v8_results, v9_results) where v9_results is None for kind=v8.
    """
    if not os.path.exists(ckpt_path):
        print(f"[{label}] ckpt missing — skipping: {ckpt_path}", flush=True)
        return None, None

    _print_header(f"{label} eval — {os.path.basename(ckpt_path)} (kind={eval_kind})")

    # Configure env vars BEFORE importing v105.5 (module-level constants).
    if eval_kind == "v8":
        os.environ["V105_5_TASK"]                = "1"
        os.environ["V105_8_PER_NUMBER_READOUT"] = "1"
        os.environ["V105_8_N_NUMBER_BINS"]      = "200"
        os.environ["V105_9_AR_DIGIT_DECODER"]   = "0"
        os.environ["V105_10_DUAL_READOUT"]      = "0"
    elif eval_kind == "v10":
        os.environ["V105_5_TASK"]                = "1"
        os.environ["V105_8_PER_NUMBER_READOUT"] = "1"
        os.environ["V105_8_N_NUMBER_BINS"]      = "200"
        os.environ["V105_9_AR_DIGIT_DECODER"]   = "1"
        os.environ["V105_10_DUAL_READOUT"]      = "1"
    else:
        raise ValueError(f"unknown eval_kind={eval_kind}")

    # Import inside the function so env vars take effect on the v105.5 module
    # at first import time within the process. Subsequent calls reuse the
    # already-loaded module (constants are fixed at first import).
    import numpy as np
    from tinygrad import Device, Tensor, dtypes  # noqa: F401
    from mycelium import Config, BreathingTransformer
    from mycelium.loader import _load_state, load_breathing
    from mycelium.factor_graph_v105_5 import (
        V105_5_K_MAX, V105_5_N_MAX, V105_5_F_MAX, V105_5_N_DIGITS,
        V105_5_N_HEADS, V105_5_WAIST, V105_5_CODEBOOK_N,
        attach_fg_params_v105_5, load_ckpt_v105_5,
        _compile_jit_fg_eval_v105_8, _compile_jit_fg_eval_v105_9,
    )
    from mycelium.factor_graph_data_v105_5 import FactorGraphLoaderV105_5

    K        = V105_5_K_MAX
    N_MAX    = V105_5_N_MAX
    F_MAX    = V105_5_F_MAX
    N_DIGITS = V105_5_N_DIGITS

    np.random.seed(seed)
    Tensor.manual_seed(seed)

    cfg = Config()
    sd_p  = _load_state()
    model = load_breathing(cfg, sd=sd_p)
    del sd_p
    # fp32 cast
    for obj_attrs in (
        (model.embed, ["weight"]),
        (model.block.shared, ["wv", "bv", "wo", "bo", "w_out", "b_out"]),
    ):
        obj, attrs = obj_attrs
        for a in attrs:
            t = getattr(obj, a)
            if t.dtype == dtypes.half:
                setattr(obj, a, t.cast(dtypes.float).contiguous().realize())
    for layer in model.block.layers:
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            t = getattr(layer, a)
            if t.dtype == dtypes.half:
                setattr(layer, a, t.cast(dtypes.float).contiguous().realize())

    attach_fg_params_v105_5(
        model, hidden=cfg.hidden,
        n_digits=N_DIGITS, n_max=N_MAX, f_max=F_MAX, k_max=K,
        waist=V105_5_WAIST, n_code=V105_5_CODEBOOK_N,
    )
    Device[Device.DEFAULT].synchronize()
    load_ckpt_v105_5(model, ckpt_path)

    loader = FactorGraphLoaderV105_5(
        val_path, batch_size=batch, difficulty_filter=None, curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V105_5_N_HEADS,
        n_digits=N_DIGITS, seed=seed,
    )

    Tensor.training = False
    eval_fn_v8 = _compile_jit_fg_eval_v105_8(
        model, K=K, B=batch, n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
    )
    eval_fn_v9 = None
    if eval_kind == "v10":
        eval_fn_v9 = _compile_jit_fg_eval_v105_9(
            model, K=K, B=batch, n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
        )

    agg_v8 = {}
    agg_v9 = {}
    n_batches = 0
    for batch_d in loader.iter_eval(batch_size=batch):
        digit_init   = batch_d["digit_init"]
        node_kinds   = batch_d["node_kinds"]
        staging_mask = batch_d["staging_mask"]
        head_op_mask = batch_d["head_op_mask"]
        gold_digits  = batch_d["gold_digits"]
        obs_mask     = batch_d["observed_mask"]
        valid_mask   = batch_d["digit_valid_mask"]
        num_bin_tgt  = batch_d["number_bin_target"]
        query_idx_np = batch_d["query_idx"]
        picks        = batch_d["picks"]

        # v8 (200-bin)
        pred_bin_t, _ = eval_fn_v8(
            digit_init, node_kinds, staging_mask, head_op_mask,
            num_bin_tgt, obs_mask, valid_mask,
        )
        pred_bin_np = pred_bin_t.numpy()
        gold_bin_np = num_bin_tgt.numpy()

        # v9 (pooled-AR digits) — only if eval_kind=v10
        pred_dg_np = None
        gold_dg_np = None
        valid_np   = None
        if eval_fn_v9 is not None:
            pred_dg_t, _ = eval_fn_v9(
                digit_init, node_kinds, staging_mask, head_op_mask,
                gold_digits, obs_mask, valid_mask,
            )
            pred_dg_np = pred_dg_t.numpy()
            gold_dg_np = gold_digits.numpy()
            valid_np   = valid_mask.numpy()

        obs_np  = obs_mask.numpy()
        for b in range(len(picks)):
            rec  = picks[b]
            diff = rec.get("difficulty", "easy")
            for dict_agg in (agg_v8, agg_v9):
                if diff not in dict_agg:
                    dict_agg[diff] = {"n_unobs": 0, "n_correct_unobs": 0,
                                       "query_correct": 0, "n_puzzles": 0}
            qi = int(query_idx_np[b])
            nv = int(batch_d["n_vars_total"][b])
            for vi in range(min(nv, N_MAX)):
                if obs_np[b, vi] != 0:
                    continue
                agg_v8[diff]["n_unobs"] += 1
                if int(pred_bin_np[b, vi]) == int(gold_bin_np[b, vi]):
                    agg_v8[diff]["n_correct_unobs"] += 1
                if pred_dg_np is not None:
                    agg_v9[diff]["n_unobs"] += 1
                    vv = valid_np[b, vi].astype(bool)
                    if vv.any():
                        if (pred_dg_np[b, vi, vv] == gold_dg_np[b, vi, vv]).all():
                            agg_v9[diff]["n_correct_unobs"] += 1
                    else:
                        if (pred_dg_np[b, vi] == gold_dg_np[b, vi]).all():
                            agg_v9[diff]["n_correct_unobs"] += 1
            if qi < N_MAX:
                if int(pred_bin_np[b, qi]) == int(gold_bin_np[b, qi]):
                    agg_v8[diff]["query_correct"] += 1
                if pred_dg_np is not None:
                    qv = valid_np[b, qi].astype(bool)
                    if qv.any():
                        if (pred_dg_np[b, qi, qv] == gold_dg_np[b, qi, qv]).all():
                            agg_v9[diff]["query_correct"] += 1
                    else:
                        if (pred_dg_np[b, qi] == gold_dg_np[b, qi]).all():
                            agg_v9[diff]["query_correct"] += 1
            agg_v8[diff]["n_puzzles"] += 1
            if pred_dg_np is not None:
                agg_v9[diff]["n_puzzles"] += 1

        n_batches += 1
        if n_batches >= max_batches:
            break

    out_v8 = {
        d: {"cell_acc":  v["n_correct_unobs"] / max(v["n_unobs"], 1),
            "query_acc": v["query_correct"]   / max(v["n_puzzles"], 1),
            "n_puzzles": v["n_puzzles"]}
        for d, v in agg_v8.items()
    }
    out_v9 = None
    if eval_kind == "v10":
        out_v9 = {
            d: {"cell_acc":  v["n_correct_unobs"] / max(v["n_unobs"], 1),
                "query_acc": v["query_correct"]   / max(v["n_puzzles"], 1),
                "n_puzzles": v["n_puzzles"]}
            for d, v in agg_v9.items()
        }
    return out_v8, out_v9


# ---------------------------------------------------------------------------
# Main: dispatch each model in its own evaluator. Note v107 and v105.5 share
# the Pythia backbone — we recompile each into a separate model object.
# ---------------------------------------------------------------------------

def main():
    V107_CKPT    = os.environ.get("V107_CKPT",    V107_CKPT_DEFAULT)
    V105_8_CKPT  = os.environ.get("V105_8_CKPT",  V105_8_CKPT_DEFAULT)
    V105_10_CKPT = os.environ.get("V105_10_CKPT", V105_10_CKPT_DEFAULT)
    VAL_PATH     = os.environ.get("VAL_PATH",     VAL_PATH_DEFAULT)
    BATCH        = int(os.environ.get("BATCH", "8"))
    MAX_BATCHES  = int(os.environ.get("MAX_BATCHES", "30"))
    SEED         = int(os.environ.get("SEED", "42"))

    print(f"=== matched eval v107 vs v105.8 vs v105.10 ===")
    print(f"  val:       {VAL_PATH}")
    print(f"  v107:      {V107_CKPT}")
    print(f"  v105.8:    {V105_8_CKPT}")
    print(f"  v105.10:   {V105_10_CKPT}")
    print(f"  batch={BATCH}  max_batches={MAX_BATCHES}")
    print()

    SKIP_V107 = int(os.environ.get("SKIP_V107", "0")) > 0
    SKIP_V105_8 = int(os.environ.get("SKIP_V105_8", "0")) > 0
    SKIP_V105_10 = int(os.environ.get("SKIP_V105_10", "0")) > 0

    # v107 first (sets up its own model object). v105.5 path follows. Both
    # construct independent model objects; the v105.5 module's env-var
    # constants are read once at first import, so we must arrange v8/v10
    # evals to share the same evaluator setup. Since v105.10 needs both
    # eval JITs and v105.8 needs only the v8 JIT, we configure the env
    # for v10 (which is a superset) and run both ckpts under that env.

    t_all = time.time()

    v107_results = None
    if not SKIP_V107:
        v107_results = eval_v107(V107_CKPT, VAL_PATH, BATCH, MAX_BATCHES, SEED)
    else:
        print("[main] SKIP_V107 set — skipping v107 eval", flush=True)

    # For the v105.x ckpts we use eval_kind="v10" so BOTH eval JITs are
    # available; for the v105.8 ckpt the v9 path will produce uninformed
    # output (digit_codebook fresh-init) which we ignore.
    v105_8_v8, v105_8_v9 = (None, None)
    if not SKIP_V105_8:
        v105_8_v8, v105_8_v9 = eval_v105_5_family(
            V105_8_CKPT, VAL_PATH, BATCH, MAX_BATCHES, SEED,
            eval_kind="v10", label="v105.8",
        )
    else:
        print("[main] SKIP_V105_8 set — skipping v105.8 eval", flush=True)

    v105_10_v8, v105_10_v9 = (None, None)
    if not SKIP_V105_10:
        v105_10_v8, v105_10_v9 = eval_v105_5_family(
            V105_10_CKPT, VAL_PATH, BATCH, MAX_BATCHES, SEED,
            eval_kind="v10", label="v105.10",
        )
    else:
        print("[main] SKIP_V105_10 set — skipping v105.10 eval", flush=True)

    _print_header(f"summary (val={os.path.basename(VAL_PATH)})")
    if v107_results is not None:
        _print_results(v107_results, "v107 [200-bin number]", MAX_BATCHES)
    if v105_8_v8 is not None:
        _print_results(v105_8_v8, "v105.8 [200-bin number]", MAX_BATCHES)
    if v105_10_v8 is not None:
        _print_results(v105_10_v8, "v105.10 [num path — 200-bin]", MAX_BATCHES)
    if v105_10_v9 is not None:
        _print_results(v105_10_v9, "v105.10 [pool path — AR digit]", MAX_BATCHES)

    print(f"\n[main] done in {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
