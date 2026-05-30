"""v99 factor graph evaluation: per-difficulty accuracy + constraint energy.

Usage:
  python scripts/eval_v99_factor_graph.py CKPT_PATH [--test PATH] [--K N] [--batch N]
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
from mycelium.factor_graph import (
    V99_K_MAX, V99_N_MAX, V99_F_MAX,
    attach_fg_params, fg_state_dict,
    factor_graph_breathing_forward, factor_graph_constraint_energy,
    build_factor_graph_masks_np,
)
from mycelium.factor_graph_data import FactorGraphLoader, DIFFICULTIES

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


def model_state_dict_fg(model) -> dict:
    sd = {"ln_f.g": model.ln_f_g, "ln_f.b": model.ln_f_b}
    sw = model.block.shared
    for a in SHARED_ATTRS:
        sd[f"shared.{a}"] = getattr(sw, a)
    for i, layer in enumerate(model.block.layers):
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            sd[f"phase{i}.{a}"] = getattr(layer, a)
    sd.update(fg_state_dict(model))
    return sd


def load_ckpt(model, path: str):
    sd = safe_load(path)
    targets = model_state_dict_fg(model)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="Path to a .safetensors v99 factor-graph ckpt.")
    ap.add_argument("--test",   default=".cache/factor_graph_test.jsonl")
    ap.add_argument("--K",      type=int, default=int(os.environ.get("V99_K_MAX", V99_K_MAX)))
    ap.add_argument("--k_alloc", type=int, default=None,
                    help="K_max for model allocation (must match ckpt). Defaults to --K.")
    ap.add_argument("--n",      type=int, default=500, help="Max test records per difficulty.")
    ap.add_argument("--batch",  type=int, default=8)
    ap.add_argument("--show",   type=int, default=2, help="Number of sample outputs to print.")
    ap.add_argument("--n_max",  type=int, default=V99_N_MAX)
    ap.add_argument("--f_max",  type=int, default=V99_F_MAX)
    args = ap.parse_args()

    k_alloc = args.k_alloc if args.k_alloc is not None else args.K
    N_MAX = args.n_max
    F_MAX = args.f_max

    print(f"=== v99 factor graph eval ===")
    print(f"ckpt={args.ckpt}  test={args.test}  K={args.K}  k_alloc={k_alloc}")
    print(f"N_MAX={N_MAX}  F_MAX={F_MAX}  n_per_diff={args.n}  batch={args.batch}")

    cfg = Config()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    cast_layers_fp32(model)
    attach_fg_params(model, hidden=cfg.hidden, n_heads=cfg.n_heads,
                     k_max=k_alloc, n_max=N_MAX, f_max=F_MAX)
    Device[Device.DEFAULT].synchronize()

    print(f"loading ckpt {args.ckpt}...")
    load_ckpt(model, args.ckpt)

    loader = FactorGraphLoader(args.test, batch_size=args.batch,
                               n_max=N_MAX, f_max=F_MAX, seed=0)
    Tensor.training = False

    def compute_dag_depth(rec):
        """Longest path from observed leaves to query — the BP mixing time bound."""
        obs = rec["observed_mask"]
        fa = rec["factor_args"]
        qi = rec["query_idx"]
        depth = {i: 0 for i, o in enumerate(obs) if o == 1}
        progress = True
        while progress:
            progress = False
            for a1, a2, res in fa:
                if a1 in depth and a2 in depth and res not in depth:
                    depth[res] = max(depth[a1], depth[a2]) + 1
                    progress = True
        return depth.get(qi, -1)

    agg = {}
    by_depth = {}
    sample_shown = 0

    t0 = time.time()
    n_seen = 0

    for batch in loader.iter_eval(batch_size=args.batch):
        domain_init  = batch["domain_init"]
        node_kinds   = batch["node_kinds"]
        attn_bias    = batch["attn_bias"]
        gold_values  = batch["gold_values"]
        obs_mask     = batch["observed_mask"]
        ft_t         = batch["factor_types"]
        fa_t         = batch["factor_args"]
        query_idx_np = batch["query_idx"]
        picks        = batch["picks"]

        var_logits_history, calib_history = factor_graph_breathing_forward(
            model, domain_init, node_kinds, attn_bias, ft_t, K=args.K,
            n_max=N_MAX, f_max=F_MAX,
        )
        final_logits = var_logits_history[-1]   # (B, N_MAX, 100)
        pred = final_logits.argmax(axis=-1)     # (B, N_MAX)

        # Constraint energy on final breath
        energy_per = factor_graph_constraint_energy(
            final_logits, ft_t, fa_t, n_max=N_MAX, f_max=F_MAX,
        )

        pred_np      = pred.realize().numpy()
        gold_np      = gold_values.realize().numpy()
        obs_np       = obs_mask.realize().numpy()
        energy_np    = energy_per.realize().numpy()
        calib_final  = calib_history[-1].realize().numpy()

        B = len(picks)
        for b in range(B):
            rec  = picks[b]
            diff = rec.get("difficulty", "easy")
            if diff not in agg:
                agg[diff] = {
                    "n_unobs": 0, "n_correct_unobs": 0,
                    "query_correct": 0, "n_puzzles": 0,
                    "energy_sum": 0.0, "calib_sum": 0.0,
                }
            if agg[diff]["n_puzzles"] >= args.n:
                continue

            qi  = int(query_idx_np[b])
            nv  = int(batch["n_vars_total"][b])

            for vi in range(min(nv, N_MAX)):
                if obs_np[b, vi] == 0:
                    agg[diff]["n_unobs"] += 1
                    if pred_np[b, vi] == gold_np[b, vi]:
                        agg[diff]["n_correct_unobs"] += 1

            if qi < N_MAX and pred_np[b, qi] == gold_np[b, qi]:
                agg[diff]["query_correct"] += 1
            agg[diff]["n_puzzles"] += 1
            agg[diff]["energy_sum"] += float(energy_np[b])
            agg[diff]["calib_sum"]  += float(calib_final[b])

            depth = compute_dag_depth(rec)
            if depth not in by_depth:
                by_depth[depth] = {
                    "n_puzzles": 0, "query_correct": 0,
                    "n_unobs": 0, "n_correct_unobs": 0,
                    "energy_sum": 0.0, "calib_sum": 0.0,
                }
            bd = by_depth[depth]
            bd["n_puzzles"] += 1
            if qi < N_MAX and pred_np[b, qi] == gold_np[b, qi]:
                bd["query_correct"] += 1
            for vi in range(min(nv, N_MAX)):
                if obs_np[b, vi] == 0:
                    bd["n_unobs"] += 1
                    if pred_np[b, vi] == gold_np[b, vi]:
                        bd["n_correct_unobs"] += 1
            bd["energy_sum"] += float(energy_np[b])
            bd["calib_sum"]  += float(calib_final[b])

            if sample_shown < args.show:
                gold_list = rec["gold_values"]
                obs_vals  = rec["observed_values"]
                print()
                print(f"--- sample ({diff}) ---")
                print(f"observed: {obs_vals}")
                print(f"gold:     {gold_list}")
                pred_list = [int(pred_np[b, vi]) for vi in range(len(gold_list))]
                print(f"pred:     {pred_list}")
                q_gold = gold_list[qi]
                q_pred = int(pred_np[b, qi])
                print(f"query x{qi}: gold={q_gold}  pred={q_pred}  {'OK' if q_gold==q_pred else 'WRONG'}")
                print(f"constraint energy: {float(energy_np[b]):.3f}")
                sample_shown += 1

        n_seen += B

    dt = time.time() - t0
    print()
    print(f"Eval complete: {n_seen} records in {dt:.1f}s ({n_seen/dt:.1f}/s)")
    print()

    out_rows = []
    for d in DIFFICULTIES:
        if d not in agg:
            continue
        v = agg[d]
        n = v["n_puzzles"]
        if n == 0:
            continue
        cell_acc  = v["n_correct_unobs"] / max(v["n_unobs"], 1)
        query_acc = v["query_correct"] / n
        avg_energy = v["energy_sum"] / n
        avg_calib  = v["calib_sum"] / n
        out_rows.append({
            "difficulty": d, "n_puzzles": n,
            "cell_acc": cell_acc, "query_acc": query_acc,
            "avg_energy": avg_energy, "avg_calib": avg_calib,
        })
        print(f"[{d:6s}] cell_acc={cell_acc:.4f} query_acc={query_acc:.4f} "
              f"avg_energy={avg_energy:.3f} avg_calib={avg_calib:.3f} n={n}")

    if out_rows:
        n_total = sum(r["n_puzzles"] for r in out_rows)
        cell_overall  = sum(r["cell_acc"]  * r["n_puzzles"] for r in out_rows) / n_total
        query_overall = sum(r["query_acc"] * r["n_puzzles"] for r in out_rows) / n_total
        print()
        print(f"OVERALL: cell_acc={cell_overall:.4f}  query_acc={query_overall:.4f}  n={n_total}")

    # DAG-depth diagnostic — BP mixing time vs graph depth.
    # Prediction: convergence K ∝ DAG depth; at K=10, deeper graphs show partial convergence.
    if by_depth:
        print()
        print("=== by DAG depth (BP mixing time diagnostic) ===")
        for depth in sorted(by_depth.keys()):
            bd = by_depth[depth]
            n = bd["n_puzzles"]
            cell_acc = bd["n_correct_unobs"] / max(bd["n_unobs"], 1)
            query_acc = bd["query_correct"] / n
            avg_energy = bd["energy_sum"] / n
            avg_calib = bd["calib_sum"] / n
            print(f"[depth={depth}] cell_acc={cell_acc:.4f} query_acc={query_acc:.4f} "
                  f"avg_energy={avg_energy:.3f} avg_calib={avg_calib:.3f} n={n}")


if __name__ == "__main__":
    main()
