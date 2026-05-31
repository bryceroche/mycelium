"""Stage 1 of the v105.1.2 v2 compositional training plan:
exhaustive single-digit arithmetic factor graphs.

Goal: pre-train single-digit arithmetic INTO the weights so that v105.1.2 v2
does NOT have to rediscover (a, b) -> a +/- b / *  / / from scratch.  Once the
ones digit works, the higher positions can use carry signals.

For each (a, b) in {0..9}^2 and op in {add, sub, mul, div}, create a single
factor graph problem:

  variables: x0 (operand a), x1 (operand b), x2 (result)
  factor:    x2 = op(x0, x1)

  observed: x0, x1
  query:    x2

Validity rules:
  - add: always valid (range [0, 18])
  - sub: only when a >= b (no negatives, range [0, 9])
  - mul: always valid (range [0, 81])
  - div: only when b > 0 AND a % b == 0 (range [0, 9])

We repeat the exhaustive set N_REPEATS times so the model sees the SAME
problems many times across the 300 step run.  This is intentional — at this
stage we WANT memorization of single-digit arithmetic.

Output JSONL follows mycelium/factor_graph_data.py / factor_graph_data_v105_1_2.py
record schema:

  n_vars            : int — number of LEAF (observed) vars (here: 2)
  n_factors         : int
  factor_types      : list[str]
  factor_args       : list[[a, b, r]]
  observed_mask     : list[int]
  observed_values   : list[int|None]
  gold_values       : list[int]
  query_idx         : int
  difficulty        : "easy"

Usage:
  N_REPEATS=50 .venv/bin/python scripts/build_digit_arithmetic_data.py
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Iterator

N_REPEATS_DEFAULT = 50
TRAIN_OUT = ".cache/digit_arith_train.jsonl"
VAL_OUT   = ".cache/digit_arith_val.jsonl"
SEED      = 17


def enumerate_base_problems() -> list[dict]:
    """Return the exhaustive list of (a, b, op, result) for single-digit args."""
    recs: list[dict] = []
    for a in range(10):
        for b in range(10):
            # ADD
            recs.append(_make_record("add", a, b, a + b))

            # SUB — only when result >= 0
            if a - b >= 0:
                recs.append(_make_record("sub", a, b, a - b))

            # MUL — always valid (max 81)
            recs.append(_make_record("mul", a, b, a * b))

            # DIV — only exact integer divisions with b > 0
            if b > 0 and a % b == 0:
                recs.append(_make_record("div", a, b, a // b))
    return recs


def _make_record(op: str, a: int, b: int, result: int) -> dict:
    """Build one v105.1.2-compatible factor graph record."""
    return {
        "n_vars":          2,          # 2 LEAF variables (matches loader convention)
        "n_factors":       1,
        "factor_types":    [op],
        "factor_args":     [[0, 1, 2]],
        "observed_mask":   [1, 1, 0],
        "observed_values": [a, b, None],
        "gold_values":     [a, b, result],
        "query_idx":       2,
        "difficulty":      "easy",
    }


def main():
    n_repeats = int(os.environ.get("N_REPEATS", str(N_REPEATS_DEFAULT)))
    os.makedirs(os.path.dirname(TRAIN_OUT) or ".", exist_ok=True)

    base = enumerate_base_problems()
    print(f"[build_digit_arith] enumerated {len(base)} base problems "
          f"(10x10 grid, 4 ops, with sub/div validity).", flush=True)

    # Op breakdown
    op_counts = defaultdict(int)
    for r in base:
        op_counts[r["factor_types"][0]] += 1
    print(f"[build_digit_arith] op distribution: {dict(op_counts)}", flush=True)

    rng = random.Random(SEED)

    # Train: repeat the exhaustive set N_REPEATS times, then shuffle.
    train_records: list[dict] = []
    for _ in range(n_repeats):
        train_records.extend(base)
    rng.shuffle(train_records)

    # Val: also exhaustive but just one pass (no need to repeat — the loader iterates).
    val_records = list(base)
    rng.shuffle(val_records)

    _write(TRAIN_OUT, train_records)
    _write(VAL_OUT,   val_records)

    print(f"[build_digit_arith] wrote {len(train_records)} train records -> {TRAIN_OUT}",
          flush=True)
    print(f"[build_digit_arith] wrote {len(val_records)} val records   -> {VAL_OUT}",
          flush=True)

    _spot_check(train_records, n=3)


def _write(path: str, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _spot_check(records: list[dict], n: int = 3) -> None:
    rng = random.Random(123)
    samples = rng.sample(records, min(n, len(records)))
    print("\n[build_digit_arith] spot check:", flush=True)
    for i, rec in enumerate(samples):
        op = rec["factor_types"][0]
        a, b = rec["gold_values"][0], rec["gold_values"][1]
        r    = rec["gold_values"][2]
        print(f"  sample {i+1}: x0={a}  {op}  x1={b}  ->  x2={r}", flush=True)


if __name__ == "__main__":
    main()
