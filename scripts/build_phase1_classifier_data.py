"""Stage A: Build per-variable classification dataset for the Phase 1 classifier.

Input:
  .cache/gsm8k_factor_graphs_train.jsonl   — 3646 labeled (question, factor-graph) pairs
  .cache/var_descriptions_to_leaf_partial.jsonl — 29218 var→IB-leaf mappings

Output:
  .cache/gsm8k_phase1_classifier_train.jsonl — training split (90%)
  .cache/gsm8k_phase1_classifier_val.jsonl   — validation split (10%)

Each output record:
  {
    "gsm8k_idx": int,
    "question":  str,
    "variables": [
      {
        "var_idx":   int,          # position in var_descriptions list
        "text":      str,          # variable description string
        "leaf_id":   str,          # e.g. "DIV.0.2.1" — 32-way label
        "leaf_int":  int,          # integer index into LEAF_IDS list (0..31)
        "op":        str|null,     # "add"|"sub"|"mul"|"div"|null (null = observed)
        "op_int":    int,          # 0=none,1=add,2=sub,3=mul,4=div
        "observed":  bool,
        "obs_value": float|null    # if observed, the numeric value; else null
      },
      ...
    ],
    "factors": [
      {
        "op":      str,      # "add"|"sub"|"mul"|"div"
        "out_idx": int,      # index of result variable
        "in1_idx": int,      # first input variable
        "in2_idx": int       # second input variable
      },
      ...
    ],
    "query_idx": int,
    "gold_answer": float|null
  }
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / ".cache"

TRAIN_FG   = CACHE_DIR / "gsm8k_factor_graphs_train.jsonl"
VAR_LEAVES = CACHE_DIR / "var_descriptions_to_leaf_partial.jsonl"
OUT_TRAIN  = CACHE_DIR / "gsm8k_phase1_classifier_train.jsonl"
OUT_VAL    = CACHE_DIR / "gsm8k_phase1_classifier_val.jsonl"

# Canonical ordering of the 32 IB leaves (alphabetical = stable across runs)
LEAF_IDS = sorted([
    "ADD.0.0.0", "ADD.0.0.1", "ADD.0.0.2", "ADD.0.1",
    "ADD.1.0",   "ADD.1.1.0", "ADD.1.1.1", "ADD.1.1.2",
    "ADD.1.2.0", "ADD.1.2.1",
    "DIV.0.0.0", "DIV.0.0.1", "DIV.0.0.2",
    "DIV.0.1.0", "DIV.0.1.1", "DIV.0.1.2",
    "DIV.0.2.0", "DIV.0.2.1", "DIV.0.2.2", "DIV.1",
    "MUL.0.0.0", "MUL.0.0.1", "MUL.0.0.2",
    "MUL.0.1.0", "MUL.0.1.1", "MUL.1",
    "SUB.0.0.0", "SUB.0.0.1", "SUB.0.1",
    "SUB.0.2.0", "SUB.0.2.1", "SUB.1",
])
LEAF_TO_INT = {l: i for i, l in enumerate(LEAF_IDS)}

OP_TO_INT   = {"none": 0, "add": 1, "sub": 2, "mul": 3, "div": 4}

VAL_FRAC    = 0.10
SEED        = 42


def load_var_leaves(path: Path) -> dict[int, dict[int, dict]]:
    """Return {gsm8k_idx -> {var_idx -> {leaf_id, op}}}."""
    idx_to_vars: dict[int, dict[int, dict]] = defaultdict(dict)
    with open(path) as f:
        for line in f:
            v = json.loads(line)
            idx_to_vars[v["gsm8k_idx"]][v["var_idx"]] = {
                "text":    v["text"],
                "leaf_id": v["leaf_id"],
                "op":      v.get("op"),  # None for observed vars
            }
    return idx_to_vars


def build_records(fg_path: Path, var_leaves: dict) -> list[dict]:
    """Merge factor-graph records with IB leaf assignments."""
    records = []
    skipped_no_leaves = 0
    skipped_partial   = 0

    with open(fg_path) as f:
        for line in f:
            r = json.loads(line)
            gsm_idx = r["gsm8k_idx"]

            # Skip if no leaf assignments for this problem
            if gsm_idx not in var_leaves:
                skipped_no_leaves += 1
                continue

            var_leaf_map = var_leaves[gsm_idx]

            # Skip if coverage is incomplete
            if len(var_leaf_map) != r["n_vars"]:
                skipped_partial += 1
                continue

            # Build factor_args lookup: out_idx -> (op, in1, in2)
            factor_by_out: dict[int, tuple] = {}
            for ft, fa in zip(r["factor_types"], r["factor_args"]):
                # fa = [out_idx, in1_idx, in2_idx]
                factor_by_out[fa[0]] = (ft, fa[1], fa[2])

            # Build per-variable annotations
            variables = []
            for var_idx in range(r["n_vars"]):
                leaf_entry = var_leaf_map[var_idx]
                is_observed = bool(r["observed_mask"][var_idx])
                obs_value   = r["observed_values"][var_idx] if is_observed else None

                # op comes from the factor that PRODUCES this variable (if any)
                if var_idx in factor_by_out:
                    ft, _, _ = factor_by_out[var_idx]
                    op_str   = ft.lower()     # "add"|"sub"|"mul"|"div"
                    op_int   = OP_TO_INT[op_str]
                else:
                    op_str   = None
                    op_int   = OP_TO_INT["none"]

                # leaf_id from IB assignment
                leaf_id  = leaf_entry["leaf_id"]
                leaf_int = LEAF_TO_INT.get(leaf_id, -1)
                if leaf_int < 0:
                    # Unseen leaf — skip whole problem
                    leaf_int = 0   # fallback; flag below

                variables.append({
                    "var_idx":   var_idx,
                    "text":      leaf_entry["text"],
                    "leaf_id":   leaf_id,
                    "leaf_int":  leaf_int,
                    "op":        op_str,
                    "op_int":    op_int,
                    "observed":  is_observed,
                    "obs_value": float(obs_value) if obs_value is not None else None,
                })

            # Build factor edge list
            factors = []
            for ft, fa in zip(r["factor_types"], r["factor_args"]):
                factors.append({
                    "op":      ft.lower(),
                    "out_idx": fa[0],
                    "in1_idx": fa[1],
                    "in2_idx": fa[2],
                })

            records.append({
                "gsm8k_idx":  gsm_idx,
                "question":   r["question"],
                "variables":  variables,
                "factors":    factors,
                "query_idx":  r["query_idx"],
                "gold_answer": r.get("gold_answer"),
            })

    print(f"Loaded {len(records)} records  "
          f"(skipped: {skipped_no_leaves} no-leaves, {skipped_partial} partial)")
    return records


def print_stats(records: list[dict]) -> None:
    from collections import Counter
    leaf_counts: Counter = Counter()
    op_counts:   Counter = Counter()
    n_vars_list = []

    for r in records:
        n_vars_list.append(len(r["variables"]))
        for v in r["variables"]:
            leaf_counts[v["leaf_id"]] += 1
            if v["op"]:
                op_counts[v["op"]] += 1

    print(f"\n=== Dataset stats ({len(records)} problems) ===")
    print(f"Vars per problem: min={min(n_vars_list)}, max={max(n_vars_list)}, "
          f"avg={sum(n_vars_list)/len(n_vars_list):.1f}")
    print(f"\nLeaf distribution (32 classes):")
    for leaf_id in LEAF_IDS:
        cnt = leaf_counts.get(leaf_id, 0)
        bar = "#" * (cnt // 20)
        print(f"  {leaf_id:20s} {cnt:5d}  {bar}")
    print(f"\nOp distribution: {dict(op_counts)}")


def main() -> None:
    print(f"Loading var-leaf mappings from {VAR_LEAVES}...")
    var_leaves = load_var_leaves(VAR_LEAVES)
    print(f"  Loaded {sum(len(v) for v in var_leaves.values())} var entries "
          f"across {len(var_leaves)} problems")

    print(f"Building records from {TRAIN_FG}...")
    records = build_records(TRAIN_FG, var_leaves)

    print_stats(records)

    # Deterministic shuffle + split
    rng = random.Random(SEED)
    rng.shuffle(records)
    n_val   = max(1, int(len(records) * VAL_FRAC))
    n_train = len(records) - n_val
    train_records = records[:n_train]
    val_records   = records[n_train:]

    print(f"\nSplit: {n_train} train / {n_val} val")

    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TRAIN, "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {n_train} records → {OUT_TRAIN}")

    with open(OUT_VAL, "w") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {n_val} records → {OUT_VAL}")

    # Also save LEAF_IDS for use by model/eval
    leaf_meta = {"leaf_ids": LEAF_IDS, "leaf_to_int": LEAF_TO_INT, "op_to_int": OP_TO_INT}
    meta_path = CACHE_DIR / "phase1_classifier_meta.json"
    with open(meta_path, "w") as f:
        json.dump(leaf_meta, f, indent=2)
    print(f"Wrote leaf/op metadata → {meta_path}")


if __name__ == "__main__":
    main()
