"""v102 factor graph evaluation: per-difficulty accuracy + K-sweep + depth diagnostic.

Usage:
  python scripts/eval_v102_factor_graph.py CKPT_PATH [--test PATH] [--K N] [--batch N]

K-sweep: if --K_sweep is given (e.g. "1,2,5,10"), runs eval at each K value and
         prints a table showing how accuracy improves with more breaths.

DAG-depth diagnostic: accuracy broken out by the topological depth of the query
         variable — shows whether topological staging is matching the BP mixing time.

Codebook diagnostic: printed alongside results — shows temperature value and
         reports mean per-position entropy of the attention weights over the codebook.
         Low entropy → peaked activation (few primitives dominate).
         High entropy → diffuse (many primitives active, weaker inductive bias).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load

from mycelium import Config, BreathingTransformer
from mycelium.loader import _load_state, load_breathing
from mycelium.factor_graph_v100 import (
    V100_N_MAX, V100_F_MAX,
    attach_fg_params_v100, fg_v100_state_dict,
    kl_energy_diagnostic_np,
)
from mycelium.factor_graph_v102 import (
    V102_K_MAX, V102_N_MAX, V102_F_MAX, V102_CODEBOOK_N,
    attach_fg_params_v102, fg_v102_state_dict,
    fg_breathing_forward_v102,
)
from mycelium.factor_graph_data_v100 import FactorGraphLoaderV100, DIFFICULTIES

SHARED_ATTRS = ("wv", "bv", "wo", "bo", "w_out", "b_out",
                "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b")


def cast_layers_fp32(model):
    def _cast(obj, attr):
        t = getattr(obj, attr)
        if t.dtype == dtypes.half:
            setattr(obj, attr, t.cast(dtypes.float).contiguous().realize())
    _cast(model.embed, "weight")
    sw = model.block.shared
    for a in SHARED_ATTRS:
        _cast(sw, a)
    for layer in model.block.layers:
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            _cast(layer, a)


def model_state_dict_v102(model) -> dict:
    sd = {"ln_f.g": model.ln_f_g, "ln_f.b": model.ln_f_b}
    sw = model.block.shared
    for a in SHARED_ATTRS:
        sd[f"shared.{a}"] = getattr(sw, a)
    for i, layer in enumerate(model.block.layers):
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            sd[f"phase{i}.{a}"] = getattr(layer, a)
    sd.update(fg_v100_state_dict(model))
    sd.update(fg_v102_state_dict(model))
    return sd


def load_ckpt_v102(model, path: str):
    sd = safe_load(path)
    targets = model_state_dict_v102(model)
    for name, dst in targets.items():
        if name not in sd:
            continue
        src = sd[name].to(dst.device).realize()
        if src.shape != dst.shape:
            try:
                src = src.reshape(dst.shape)
            except Exception:
                continue
        if src.dtype != dst.dtype:
            src = src.cast(dst.dtype)
        dst.assign(src).realize()

    # Report codebook key loading status
    v102_keys = [k for k in targets if k.startswith("fg_v102.")]
    loaded    = [k for k in v102_keys if k in sd]
    if not loaded:
        print(f"  INFO: no v102 codebook keys in ckpt — using near-identity init")
    else:
        print(f"  loaded {len(loaded)}/{len(v102_keys)} v102 codebook keys from ckpt")


def compute_query_depth(rec) -> int:
    obs = rec["observed_mask"]
    fa  = rec["factor_args"]
    qi  = rec["query_idx"]
    depth: dict = {}
    for i, o in enumerate(obs):
        if o == 1:
            depth[i] = 0
    changed = True
    while changed:
        changed = False
        for a1, a2, res in fa:
            if a1 in depth and a2 in depth and res not in depth:
                depth[res] = max(depth[a1], depth[a2]) + 1
                changed = True
    return depth.get(qi, -1)


def run_eval_at_K(model, loader: FactorGraphLoaderV100, K: int,
                  n_per_diff: int = 500,
                  n_max: int = V102_N_MAX,
                  f_max: int = V102_F_MAX) -> tuple[dict, dict]:
    """Run eval at a specific K value.

    Returns:
      agg_by_diff  : {difficulty: {cell_acc, query_acc, n_puzzles, avg_energy}}
      agg_by_depth : {depth: {cell_acc, query_acc, n_puzzles, avg_energy}}
    """
    agg_diff  = {}
    agg_depth = {}

    for batch in loader.iter_eval(batch_size=loader.batch_size):
        domain_init  = batch["domain_init"]
        node_kinds   = batch["node_kinds"]
        staging_mask = batch["staging_mask"]
        head_op_mask = batch["head_op_mask"]
        gold_values  = batch["gold_values"]
        obs_mask     = batch["observed_mask"]
        ft_t         = batch["factor_types"]
        fa_t         = batch["factor_args"]
        query_idx_np = batch["query_idx"]
        picks        = batch["picks"]

        var_logits_history, _, _ = fg_breathing_forward_v102(
            model, domain_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max,
        )
        final_logits = var_logits_history[-1]
        pred         = final_logits.argmax(axis=-1)

        pred_np  = pred.realize().numpy()
        gold_np  = gold_values.realize().numpy()
        obs_np   = obs_mask.realize().numpy()
        ft_np    = ft_t.numpy()
        fa_np    = fa_t.numpy()

        final_logits_np = final_logits.realize().numpy()
        kl_energy = kl_energy_diagnostic_np(final_logits_np, ft_np, fa_np,
                                             n_max=n_max, f_max=f_max)

        B = len(picks)
        for b in range(B):
            rec  = picks[b]
            diff = rec.get("difficulty", "easy")

            if diff not in agg_diff:
                agg_diff[diff] = {"n_unobs": 0, "n_correct_unobs": 0,
                                  "query_correct": 0, "n_puzzles": 0,
                                  "energy_sum": 0.0}
            if agg_diff[diff]["n_puzzles"] >= n_per_diff:
                continue

            qi = int(query_idx_np[b])
            nv = int(batch["n_vars_total"][b])

            for vi in range(min(nv, n_max)):
                if obs_np[b, vi] == 0:
                    agg_diff[diff]["n_unobs"] += 1
                    if pred_np[b, vi] == gold_np[b, vi]:
                        agg_diff[diff]["n_correct_unobs"] += 1
            if qi < n_max and pred_np[b, qi] == gold_np[b, qi]:
                agg_diff[diff]["query_correct"] += 1
            agg_diff[diff]["n_puzzles"] += 1
            agg_diff[diff]["energy_sum"] += kl_energy

            depth = compute_query_depth(rec)
            if depth not in agg_depth:
                agg_depth[depth] = {"n_unobs": 0, "n_correct_unobs": 0,
                                    "query_correct": 0, "n_puzzles": 0,
                                    "energy_sum": 0.0}
            bd = agg_depth[depth]
            bd["n_puzzles"] += 1
            bd["energy_sum"] += kl_energy
            if qi < n_max and pred_np[b, qi] == gold_np[b, qi]:
                bd["query_correct"] += 1
            for vi in range(min(nv, n_max)):
                if obs_np[b, vi] == 0:
                    bd["n_unobs"] += 1
                    if pred_np[b, vi] == gold_np[b, vi]:
                        bd["n_correct_unobs"] += 1

    out_diff  = {}
    for d, v in agg_diff.items():
        n = v["n_puzzles"]
        if n == 0:
            continue
        out_diff[d] = {
            "cell_acc": v["n_correct_unobs"] / max(v["n_unobs"], 1),
            "query_acc": v["query_correct"] / n,
            "n_puzzles": n,
            "avg_energy": v["energy_sum"] / n,
        }

    out_depth = {}
    for depth, v in agg_depth.items():
        n = v["n_puzzles"]
        if n == 0:
            continue
        out_depth[depth] = {
            "cell_acc": v["n_correct_unobs"] / max(v["n_unobs"], 1),
            "query_acc": v["query_correct"] / n,
            "n_puzzles": n,
            "avg_energy": v["energy_sum"] / n,
        }

    return out_diff, out_depth


def print_codebook_diagnostic(model):
    """Print learned codebook state: temperature and pairwise cosine statistics."""
    cb = model.fg_v102_codebook.realize().numpy()   # (N_CODE, H)
    gq = model.fg_v102_delta_gate_quant.realize().numpy()  # (K_max,)
    tmp = float(model.fg_v102_temperature.realize().numpy().flat[0])

    n_code = cb.shape[0]
    # Pairwise cosine similarity
    norms = np.linalg.norm(cb, axis=1, keepdims=True) + 1e-8
    cb_unit = cb / norms
    cos_sim = cb_unit @ cb_unit.T  # (N_CODE, N_CODE)
    np.fill_diagonal(cos_sim, np.nan)
    mean_off_diag = float(np.nanmean(np.abs(cos_sim)))

    print(f"  codebook: n_code={n_code}  temperature={tmp:.4f}")
    print(f"    mean |off-diag cosine|={mean_off_diag:.4f}  "
          f"(lower = more orthogonal = better separation)")
    gq_nz = np.count_nonzero(np.abs(gq) > 1e-6)
    print(f"    delta_gate_quant: {np.array2string(gq[:5], precision=4)}...  "
          f"({gq_nz}/{len(gq)} nonzero)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="Path to a .safetensors v102 factor-graph ckpt.")
    ap.add_argument("--test",    default=".cache/factor_graph_test.jsonl")
    ap.add_argument("--K",       type=int, default=None,
                    help="Single K to eval at (default = V102_K_MAX env var or ckpt K).")
    ap.add_argument("--K_sweep", type=str, default="1,2,5,10",
                    help="Comma-separated K values for sweep (e.g. '1,2,5,10').")
    ap.add_argument("--k_alloc", type=int, default=None,
                    help="K_max for model allocation. Defaults to max(K_sweep).")
    ap.add_argument("--n",       type=int, default=500, help="Max test records per difficulty.")
    ap.add_argument("--batch",   type=int, default=8)
    ap.add_argument("--n_max",   type=int, default=V102_N_MAX)
    ap.add_argument("--f_max",   type=int, default=V102_F_MAX)
    ap.add_argument("--n_code",  type=int, default=V102_CODEBOOK_N)
    args = ap.parse_args()

    sweep_Ks = [int(x) for x in args.K_sweep.split(",") if x.strip()]
    if args.K is not None:
        sweep_Ks = [args.K]
    k_alloc = args.k_alloc if args.k_alloc is not None else max(sweep_Ks)
    N_MAX, F_MAX = args.n_max, args.f_max

    print(f"=== v102 factor graph eval (v100 + shared codebook compression) ===")
    print(f"ckpt={args.ckpt}  test={args.test}")
    print(f"K_sweep={sweep_Ks}  k_alloc={k_alloc}  N_MAX={N_MAX}  F_MAX={F_MAX}  n_code={args.n_code}")

    cfg = Config()
    sd  = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    cast_layers_fp32(model)
    attach_fg_params_v100(model, hidden=cfg.hidden, n_heads=cfg.n_heads,
                          k_max=k_alloc, n_max=N_MAX, f_max=F_MAX)
    attach_fg_params_v102(model, hidden=cfg.hidden, n_code=args.n_code)
    Device[Device.DEFAULT].synchronize()

    print(f"loading ckpt {args.ckpt}...")
    load_ckpt_v102(model, args.ckpt)

    print_codebook_diagnostic(model)

    loader = FactorGraphLoaderV100(
        args.test, batch_size=args.batch,
        n_max=N_MAX, f_max=F_MAX, k_max=k_alloc, seed=0,
    )
    Tensor.training = False

    sweep_results = {}
    for K_eval in sweep_Ks:
        print(f"\n--- K={K_eval} ---")
        t0 = time.time()
        agg_diff, agg_depth = run_eval_at_K(model, loader, K=K_eval,
                                            n_per_diff=args.n,
                                            n_max=N_MAX, f_max=F_MAX)
        dt = time.time() - t0
        print(f"  ({dt:.1f}s)")

        sweep_results[K_eval] = agg_diff
        rows = []
        for d in DIFFICULTIES:
            if d not in agg_diff:
                continue
            v = agg_diff[d]
            rows.append(v)
            print(f"  [{d:6s}] cell_acc={v['cell_acc']:.4f}  query_acc={v['query_acc']:.4f}  "
                  f"kl_energy={v['avg_energy']:.4f}  n={v['n_puzzles']}")

        if rows:
            n_tot = sum(r["n_puzzles"] for r in rows)
            c_ov  = sum(r["cell_acc"]  * r["n_puzzles"] for r in rows) / n_tot
            q_ov  = sum(r["query_acc"] * r["n_puzzles"] for r in rows) / n_tot
            print(f"  OVERALL cell_acc={c_ov:.4f}  query_acc={q_ov:.4f}")

        if agg_depth:
            print(f"  --- depth diagnostic (K={K_eval}) ---")
            for depth in sorted(agg_depth.keys()):
                bd = agg_depth[depth]
                print(f"    [depth={depth:2d}] cell_acc={bd['cell_acc']:.4f}  "
                      f"query_acc={bd['query_acc']:.4f}  n={bd['n_puzzles']}")

    if len(sweep_Ks) > 1:
        print(f"\n=== K-sweep summary ===")
        print(f"{'K':>5}  {'cell_acc(easy)':>16}  {'query_acc(easy)':>16}  {'cell_acc(hard)':>16}")
        for K_eval, agg_diff in sorted(sweep_results.items()):
            e = agg_diff.get("easy", {})
            h = agg_diff.get("hard", {})
            print(f"{K_eval:>5}  {e.get('cell_acc', float('nan')):>16.4f}  "
                  f"{e.get('query_acc', float('nan')):>16.4f}  "
                  f"{h.get('cell_acc', float('nan')):>16.4f}")


if __name__ == "__main__":
    main()
