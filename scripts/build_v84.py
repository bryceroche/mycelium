"""Build v84: OUTLINE-TO-ESSAY supervision for the SINGLE-HEAD WaistController.

v84 is the breakthrough architecture after v83. v83 hit 20% parse rate at grad
step 400 (first time post-mask-fix) but 0% accuracy because args content is
random. v84 replaces v82/v83's parallel-diffusion supervision (every breath
emits ALL 3 lists at varying precision) with OUTLINE-TO-ESSAY supervision: each
breath ADDS a new dimension rather than refining all dimensions in parallel.

For K=5, the target at each breath GROWS:

    B0:  ops only                   "2,0"
    B1:  + types                    "2,0 | 0.1.1,0.0.1"
    B2:  + arg magnitudes           "2,0 | 0.1.1,0.0.1 | 50,60,r,10"
    B3:  + arg exact                "2,0 | 0.1.1,0.0.1 | 50,60,-1,12"
    B4:  refinement (same as B3)    "2,0 | 0.1.1,0.0.1 | 50,60,-1,12"

Key properties:
- Monotone length: B0 shortest, B3 longest. Model learns to ADD content per
  breath, not refine.
- No placeholders: B0 just emits ops, no `?` tokens. Each breath is a complete
  partial answer of its own.
- Same canonical input (= L4) and same DAG executor as v82 — the last breath's
  format is identical, so v82_sympy_eval / eval_v82_dag work unchanged after a
  minor wrapper.

For B2's magnitude args we reuse `encode_arg_magnitude` from build_v82.

Edge cases:
  - Shallow leaves (e.g. "DIV.0" — only depth-1 types_path): pad by repeating
    the last index for depth-2 / depth-3 targets ("0.0" / "0.0.0").
  - >2 args per step: REJECTED (we drop the record).
  - Unary ops (1 arg): args list emits the lone arg + '?' as the second arg
    at B3/B4, and the magnitude + '?' at B2. (The '?' at the second-arg slot
    of unary steps will fail SymPy parse — same as v82.)
"""
import os
import sys
import json
import re
import argparse
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import diag_ib_clustering as ibc  # type: ignore

# Reuse v82's helpers.
from build_v82 import (  # type: ignore
    L2_STEP_RE, L4_STEP_RE, OP_TO_IDX,
    parse_l2_nl, parse_l4_args,
    load_tree, assign_leaf,
    types_path_at_depth,
    encode_arg_exact, encode_arg_magnitude,
    encode_args_for_step_exact, encode_args_for_step_mag,
)


# ---- v84 per-breath target builder ----
# K=5: B0..B4. Schedule grows monotonically.

def build_breath_v84(breath_idx, steps, leaf_ids):
    """Build the v84 target string for breath B<breath_idx>.

    Schedule (K=5):
      B0:  ops only
      B1:  ops + types  (depth-3 leaf path)
      B2:  ops + types + args at magnitude precision
      B3:  ops + types + args exact (full essay)
      B4:  same as B3 (refinement pass)
    """
    n_steps = len(steps)
    ops_items = [str(OP_TO_IDX[s["op"]]) for s in steps]
    ops_str = ",".join(ops_items)

    if breath_idx == 0:
        return ops_str

    # Types are emitted from B1 onward at full depth (depth-3 leaf path).
    types_items = [types_path_at_depth(lid, 3) for lid in leaf_ids]
    types_str = ",".join(types_items)

    if breath_idx == 1:
        return f"{ops_str} | {types_str}"

    # Args from B2 onward.
    args_items = []
    for step in steps:
        if breath_idx == 2:
            a1, a2 = encode_args_for_step_mag(step["args"])
        else:  # B3, B4 — exact
            a1, a2 = encode_args_for_step_exact(step["args"])
        args_items.append(a1)
        args_items.append(a2)
    args_str = ",".join(args_items)

    return f"{ops_str} | {types_str} | {args_str}"


# ---- Per-record processing ----

def process_record(rec, tok, embed_w, leaves, centroids, leaf_ops, K=5):
    """Build a v84 record. Returns None to drop the input."""
    v80_layers = rec["layers"]
    l2_text = v80_layers.get("L2", "")
    l4_text = v80_layers.get("L4", "")
    nl_steps = parse_l2_nl(l2_text)
    arg_steps = parse_l4_args(l4_text)
    if not nl_steps or not arg_steps or len(nl_steps) != len(arg_steps):
        return None
    for ns, ar in zip(nl_steps, arg_steps):
        if ns["op"] != ar["op"]:
            return None
    for ar in arg_steps:
        if ar["op"] not in OP_TO_IDX:
            return None
        if len(ar["args"]) == 0 or len(ar["args"]) > 2:
            return None
    leaf_ids = []
    for ns, ar in zip(nl_steps, arg_steps):
        idx = assign_leaf(ns["nl"], ns["op"], tok, embed_w, centroids, leaf_ops)
        leaf_ids.append(leaves[idx]["leaf_id"])
    n_steps = len(arg_steps)
    # Build K layers. The data file always has L0..L<K-1>; V77_N_LAYERS env
    # decides how many to consume.
    layers = {
        f"L{i}": build_breath_v84(i, arg_steps, leaf_ids) for i in range(K)
    }
    return {
        "problem": rec["problem"],
        "answer": rec["answer"],
        "n_steps": n_steps,
        "leaf_ids": leaf_ids,
        "layers": layers,
        **layers,
        "problem_type": rec.get("problem_type", ""),
        "gen_targets": rec.get("gen_targets", []),
        "sympy_value": rec.get("sympy_value"),
        "sympy_matches_gold": rec.get("sympy_matches_gold", True),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=".cache/gsm8k_steps_v80_train.jsonl")
    ap.add_argument("--dst", default=".cache/gsm8k_steps_v84_train.jsonl")
    ap.add_argument("--tree", default=".cache/ib_tree.json")
    ap.add_argument("--centroids", default=".cache/ib_centroids.npz")
    ap.add_argument("--pythia", default=".cache/pythia-410m/model.safetensors")
    ap.add_argument("--num", type=int, default=0, help="limit input records (0=all)")
    ap.add_argument("--K", type=int, default=5, help="number of breaths / layers")
    args = ap.parse_args()

    t0 = time.perf_counter()
    print(f"[1/3] Loading Pythia + IB tree...")
    embed_w = ibc.load_pythia_embed_numpy(args.pythia)
    tok = ibc.load_tokenizer()
    leaves, centroids, leaf_ops, max_depth = load_tree(args.tree, args.centroids)
    print(f"  embed: {embed_w.shape}  leaves: {len(leaves)}  max_depth: {max_depth}  K={args.K}")

    print(f"\n[2/3] Processing {args.src} -> {args.dst}")
    out_path = Path(args.dst)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    kept = 0
    drop_parse = 0
    drop_op = 0
    drop_args = 0
    drop_other = 0
    with open(args.src) as fin, open(out_path, "w") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            if args.num and i >= args.num:
                break
            rec = json.loads(line)
            v80_layers = rec["layers"]
            nl_steps = parse_l2_nl(v80_layers.get("L2", ""))
            arg_steps = parse_l4_args(v80_layers.get("L4", ""))
            drop_reason = None
            if (not nl_steps or not arg_steps
                or len(nl_steps) != len(arg_steps)):
                drop_reason = "parse"
            else:
                for ns, ar in zip(nl_steps, arg_steps):
                    if ns["op"] != ar["op"]:
                        drop_reason = "parse"
                        break
                    if ar["op"] not in OP_TO_IDX:
                        drop_reason = "op"
                        break
                    if len(ar["args"]) == 0 or len(ar["args"]) > 2:
                        drop_reason = "args"
                        break
            if drop_reason is None:
                out_rec = process_record(rec, tok, embed_w, leaves,
                                         centroids, leaf_ops, K=args.K)
                if out_rec is None:
                    drop_other += 1
                    continue
                fout.write(json.dumps(out_rec) + "\n")
                kept += 1
            else:
                if drop_reason == "parse":
                    drop_parse += 1
                elif drop_reason == "op":
                    drop_op += 1
                elif drop_reason == "args":
                    drop_args += 1
            total = kept + drop_parse + drop_op + drop_args + drop_other
            if total % 500 == 0:
                print(f"  [{total}]  kept={kept}  drop[parse/op/args/other]="
                      f"{drop_parse}/{drop_op}/{drop_args}/{drop_other}  "
                      f"({time.perf_counter() - t0:.1f}s)")

    print(f"\n[3/3] Done.  kept={kept}  drop[parse/op/args/other]="
          f"{drop_parse}/{drop_op}/{drop_args}/{drop_other}  out={out_path}")

    # 5 samples — print all K breaths for each.
    print(f"\n=== 5 sample v84 records (verbatim B0..B{args.K - 1}) ===")
    with open(out_path) as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            r = json.loads(line)
            print(f"\n--- sample {i + 1} (n_steps={r['n_steps']}, leaves={r['leaf_ids']}) ---")
            print(f"problem: {r['problem'][:120]}")
            for L in [f"L{k}" for k in range(args.K)]:
                print(f"  {L}: {r[L]}")


if __name__ == "__main__":
    main()
