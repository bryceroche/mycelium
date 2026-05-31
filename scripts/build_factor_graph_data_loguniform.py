"""Stage 2 of the v105.1.2 v2 compositional training plan:
factor graphs with log-uniform operand values [1, 9999].

The existing .cache/factor_graph_*.jsonl files were generated with operand
values in [1, 20] and final results capped at 99 (VALUE_MAX). That means
~95% of gold values fit in 2 digits — positions 0-2 (ten-thousands /
thousands / hundreds) are almost always leading-zero, providing no learning
signal beyond "predict 0". v105.1.2 v2's MSD-first layout + digit_valid_mask
correctly excludes leading-zero padding from the loss, but the model still
sees mostly-trivial training signal at positions 0-3 because the underlying
gold distribution is degenerate at those positions.

This generator samples leaf values from 10^uniform(0, 4) ≈ log-uniform in
[1, 9999] (rounded to int) and propagates values forward without clamping,
producing gold values that span the full 5-digit range. Every position
0-4 should see >= 15% non-zero coverage.

Validity rules:
  - add / sub: results in [0, 99999] (cap at VALUE_MAX = 99999).  sub must be >= 0.
  - mul: cap product magnitude. We sample one operand log-uniform in [1, 99],
         the other log-uniform in [1, 999], and reject if product > 99999.
  - div: sample dividend, then pick a divisor (1..9 or a divisor of the dividend)
         so result is exact integer.

Output JSONL matches the schema used by
scripts/build_factor_graph_data.py + mycelium/factor_graph_data*.py.

Usage:
  .venv/bin/python scripts/build_factor_graph_data_loguniform.py
  .venv/bin/python scripts/build_factor_graph_data_loguniform.py --train-n 50000 --test-n 5000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_N = 50_000
TEST_N = 5_000
TRAIN_SEED = 142
TEST_SEED = 143

# Same difficulty bands as the original generator.
DIFFICULTY_BANDS = {
    "easy":   (3, 4),
    "medium": (5, 6),
    "hard":   (7, 8),
}

OPS = ("add", "sub", "mul", "div")
VALUE_MAX = 99_999     # 5 digits — matches V105_1_2_N_DIGITS=5
VALUE_MIN = 0
MAX_RETRIES_PER_RECORD = 500

# Operand range. Log-uniform sampling => 10^uniform(LOG_LO, LOG_HI).
LOG_LO = 0.0   # 10^0 = 1
LOG_HI = 4.0   # 10^4 = 10000 (rounded -> typically 1..9999)


# ---------------------------------------------------------------------------
# Operand sampling
# ---------------------------------------------------------------------------

def _sample_loguniform(rng: random.Random, lo: float = LOG_LO, hi: float = LOG_HI) -> int:
    """Sample an integer in [1, 10^hi] log-uniformly. Rounded to int."""
    u = rng.uniform(lo, hi)
    v = int(round(10.0 ** u))
    return max(1, v)


def sample_operand(rng: random.Random) -> int:
    """Default leaf-operand sampler for the log-uniform generator."""
    return _sample_loguniform(rng)


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

def apply_op(op: str, a: int, b: int):
    """Apply op(a, b). Return result int or None if invalid."""
    if op == "add":
        r = a + b
        return r if VALUE_MIN <= r <= VALUE_MAX else None
    if op == "sub":
        r = a - b
        return r if r >= VALUE_MIN else None
    if op == "mul":
        r = a * b
        return r if r <= VALUE_MAX else None
    if op == "div":
        if b == 0 or a % b != 0:
            return None
        r = a // b
        return r if VALUE_MIN <= r <= VALUE_MAX else None
    return None


def sample_valid_pair(rng: random.Random, op: str):
    """Sample (a, b) such that apply_op(op, a, b) is valid.
    Returns (a, b, result) or None after exhausting retries.
    """
    for _ in range(200):
        if op == "add":
            # Both log-uniform; with LO=0 HI=4 the typical result is <= ~20000.
            a = _sample_loguniform(rng)
            b = _sample_loguniform(rng)
        elif op == "sub":
            # Sample a larger than b so result >= 0.
            a = _sample_loguniform(rng)
            b = rng.randint(0, a)
        elif op == "mul":
            # Cap operand magnitudes so the product stays under 99999.
            # Two scales: small (1..99) and medium (1..999). Pick one of each.
            a = _sample_loguniform(rng, 0.0, 2.0)   # 1..99
            b = _sample_loguniform(rng, 0.0, 3.0)   # 1..999
            # Random swap so the order isn't always (small, medium).
            if rng.random() < 0.5:
                a, b = b, a
        elif op == "div":
            # Sample dividend first; pick a divisor of it.
            a = _sample_loguniform(rng)
            # Try a small divisor in 1..9 first (matches the v98 / v105 style).
            if rng.random() < 0.5:
                b = rng.randint(1, 9)
                if a % b != 0:
                    # Round a down to nearest multiple.
                    a = (a // b) * b
                    if a < 1:
                        a = b
            else:
                # Pick from the divisors of a.
                if a < 2:
                    continue
                divisors = [d for d in range(1, min(a + 1, 1000)) if a % d == 0]
                if not divisors:
                    continue
                b = rng.choice(divisors)
        else:
            return None
        r = apply_op(op, a, b)
        if r is not None and VALUE_MIN <= r <= VALUE_MAX:
            return a, b, r
    return None


def _find_valid_a(rng: random.Random, op: str, b: int):
    """Given fixed b, find (a, result) such that op(a, b) is valid."""
    for _ in range(200):
        if op == "add":
            a = _sample_loguniform(rng)
            if a + b > VALUE_MAX:
                continue
        elif op == "sub":
            a = _sample_loguniform(rng)
            if a < b:
                continue
        elif op == "mul":
            if b == 0:
                return None
            max_a = VALUE_MAX // b
            if max_a < 1:
                return None
            # Log-uniform up to log10(max_a)
            la_max = math.log10(max(max_a, 1))
            a = _sample_loguniform(rng, 0.0, la_max)
        elif op == "div":
            if b == 0:
                return None
            max_k_log = math.log10(max(VALUE_MAX // b, 1))
            k = _sample_loguniform(rng, 0.0, max_k_log)
            a = b * k
        else:
            return None
        r = apply_op(op, a, b)
        if r is not None:
            return a, r
    return None


def _find_valid_b(rng: random.Random, op: str, a: int):
    """Given fixed a, find (b, result) such that op(a, b) is valid."""
    for _ in range(200):
        if op == "add":
            b = _sample_loguniform(rng)
            if a + b > VALUE_MAX:
                continue
        elif op == "sub":
            b = rng.randint(0, a)
        elif op == "mul":
            if a == 0:
                return None
            max_b = VALUE_MAX // a
            if max_b < 1:
                return None
            lb_max = math.log10(max(max_b, 1))
            b = _sample_loguniform(rng, 0.0, lb_max)
        elif op == "div":
            if a < 2:
                return None
            # Pick a small divisor or a divisor of a.
            if rng.random() < 0.5:
                b = rng.randint(1, 9)
                if a % b != 0:
                    continue
            else:
                divisors = [d for d in range(1, min(a + 1, 1000)) if a % d == 0]
                if not divisors:
                    return None
                b = rng.choice(divisors)
        else:
            return None
        r = apply_op(op, a, b)
        if r is not None:
            return b, r
    return None


# ---------------------------------------------------------------------------
# Topology helpers (verbatim from build_factor_graph_data.py)
# ---------------------------------------------------------------------------

def topo_sort(n_vars_total: int, factor_args: list):
    in_degree = [0] * n_vars_total
    children = defaultdict(list)
    for fa in factor_args:
        a, b, r = fa
        children[a].append(r)
        children[b].append(r)
        in_degree[r] += 2

    queue = [i for i in range(n_vars_total) if in_degree[i] == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in children[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    return order if len(order) == n_vars_total else None


def make_chain_dag(n_leaves: int):
    n_factors = n_leaves - 1
    n_total = n_leaves + n_factors
    factor_args = []
    for k in range(n_factors):
        a = 0 if k == 0 else n_leaves + k - 1
        b = k + 1
        r = n_leaves + k
        factor_args.append([a, b, r])
    return n_total, factor_args


def make_tree_dag(rng: random.Random, n_leaves: int):
    available = list(range(n_leaves))
    next_idx = n_leaves
    factor_args = []
    rng.shuffle(available)
    while len(available) > 1:
        a = available.pop(0)
        b = available.pop(0)
        r = next_idx
        next_idx += 1
        factor_args.append([a, b, r])
        available.append(r)
        if len(available) > 2:
            rng.shuffle(available)
    return next_idx, factor_args


# ---------------------------------------------------------------------------
# Record generation
# ---------------------------------------------------------------------------

def generate_record(rng: random.Random, n_leaves: int, difficulty: str):
    """Generate one valid factor graph record, or None on failure."""
    use_tree = (n_leaves >= 4) and rng.random() < 0.5
    if use_tree:
        n_total, factor_args = make_tree_dag(rng, n_leaves)
    else:
        n_total, factor_args = make_chain_dag(n_leaves)
    n_factors = len(factor_args)

    topo_order = topo_sort(n_total, factor_args)
    if topo_order is None:
        return None

    result_set = {fa[2] for fa in factor_args}
    leaf_set = set(range(n_total)) - result_set
    result_to_fi = {fa[2]: fi for fi, fa in enumerate(factor_args)}

    factor_types = [rng.choice(OPS) for _ in range(n_factors)]

    gold = [None] * n_total

    # Initial leaf assignment: log-uniform in [1, 9999]. Width depends on op
    # of the first factor that consumes the leaf — but at this stage we just
    # pick something and let _find_valid_X fix it if needed.
    for i in leaf_set:
        gold[i] = sample_operand(rng)

    for node in topo_order:
        if node not in result_to_fi:
            continue
        fi = result_to_fi[node]
        a_idx, b_idx, r_idx = factor_args[fi]
        op = factor_types[fi]
        a_val = gold[a_idx]
        b_val = gold[b_idx]
        if a_val is None or b_val is None:
            return None

        r = apply_op(op, a_val, b_val)
        if r is None:
            a_is_leaf = a_idx in leaf_set
            b_is_leaf = b_idx in leaf_set
            if a_is_leaf and b_is_leaf:
                pair = sample_valid_pair(rng, op)
                if pair is None:
                    return None
                gold[a_idx], gold[b_idx], r = pair
            elif a_is_leaf:
                fix = _find_valid_a(rng, op, b_val)
                if fix is None:
                    return None
                gold[a_idx], r = fix
            elif b_is_leaf:
                fix = _find_valid_b(rng, op, a_val)
                if fix is None:
                    return None
                gold[b_idx], r = fix
            else:
                # Both fixed by upstream factors — can't repair.
                return None

        if r is None or not (VALUE_MIN <= r <= VALUE_MAX):
            return None
        gold[r_idx] = r

    if any(v is None for v in gold):
        return None
    if any(not (VALUE_MIN <= v <= VALUE_MAX) for v in gold):
        return None

    observed_mask = [1 if i in leaf_set else 0 for i in range(n_total)]
    observed_values = [gold[i] if observed_mask[i] == 1 else None for i in range(n_total)]

    result_nodes_topo = [n for n in topo_order if n in result_set]
    if not result_nodes_topo:
        return None
    query_idx = result_nodes_topo[-1]

    # Reject trivial: small graphs where answer = observed
    query_answer = gold[query_idx]
    obs_vals = {v for v in observed_values if v is not None}
    if query_answer in obs_vals and n_leaves <= 3:
        return None

    return {
        "n_vars":         n_leaves,
        "n_factors":      n_factors,
        "factor_types":   factor_types,
        "factor_args":    factor_args,
        "observed_mask":  observed_mask,
        "observed_values": observed_values,
        "gold_values":    gold,
        "query_idx":      query_idx,
        "difficulty":     difficulty,
    }


def generate_record_with_retry(rng: random.Random, n_leaves: int, difficulty: str):
    for _ in range(MAX_RETRIES_PER_RECORD):
        rec = generate_record(rng, n_leaves, difficulty)
        if rec is not None:
            return rec
    return None


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_record(rec: dict) -> bool:
    n_total = len(rec["gold_values"])
    factor_args = rec["factor_args"]
    factor_types = rec["factor_types"]
    observed_mask = rec["observed_mask"]
    observed_values = rec["observed_values"]
    gold_values = rec["gold_values"]

    topo_order = topo_sort(n_total, factor_args)
    if topo_order is None:
        return False

    vals = [None] * n_total
    for i in range(n_total):
        if observed_mask[i] == 1:
            vals[i] = observed_values[i]

    result_to_fi = {fa[2]: fi for fi, fa in enumerate(factor_args)}
    for node in topo_order:
        if node not in result_to_fi:
            continue
        fi = result_to_fi[node]
        a_idx, b_idx, r_idx = factor_args[fi]
        op = factor_types[fi]
        a = vals[a_idx]
        b = vals[b_idx]
        if a is None or b is None:
            return False
        r = apply_op(op, a, b)
        if r is None:
            return False
        vals[r_idx] = r

    for i in range(n_total):
        if vals[i] != gold_values[i]:
            return False
    for i in range(n_total):
        if observed_mask[i] == 1 and observed_values[i] != gold_values[i]:
            return False
    return True


def run_verification(records: list, n_sample: int = 100) -> None:
    if not records:
        raise ValueError("No records to verify")
    sample_size = min(n_sample, len(records))
    rng = random.Random(999)
    sample = rng.sample(records, sample_size)
    failures = []
    for i, rec in enumerate(sample):
        if not verify_record(rec):
            failures.append(i)
    if failures:
        raise AssertionError(
            f"Verification FAILED on {len(failures)}/{sample_size} records. "
            f"First failing index: {failures[0]}"
        )
    print(f"[verify] PASSED: {sample_size} randomly sampled records all correct.",
          flush=True)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(args):
    seed, difficulty, n_target = args
    rng = random.Random(seed)
    lo, hi = DIFFICULTY_BANDS[difficulty]
    results = []
    for _ in range(n_target):
        n_leaves = rng.randint(lo, hi)
        rec = generate_record_with_retry(rng, n_leaves, difficulty)
        if rec is not None:
            results.append(rec)
    return results


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(out_path: str, n_total: int, seed: int, workers: int = 4):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    keys = list(DIFFICULTY_BANDS.keys())
    plan = {}
    remaining = n_total
    for i, k in enumerate(keys):
        if i == len(keys) - 1:
            plan[k] = remaining
        else:
            n = n_total // 3
            plan[k] = n
            remaining -= n

    print(f"[build] plan: {plan} (total {n_total}) -> {out_path}", flush=True)

    tasks = []
    salt = 0
    chunk = max(50, 500 // max(workers, 1))
    for diff, n in plan.items():
        offset = 0
        while offset < n:
            n_this = min(chunk, n - offset)
            tasks.append((seed * 100003 + salt * 7 + abs(hash(diff)) % 1000, diff, n_this))
            salt += 1
            offset += n_this

    random.Random(seed).shuffle(tasks)

    t0 = time.time()
    all_records = []
    counts = defaultdict(int)

    if workers <= 1:
        for task in tasks:
            batch = _worker(task)
            all_records.extend(batch)
            for rec in batch:
                counts[rec["difficulty"]] += 1
            n = len(all_records)
            if n % 5000 < len(batch) or n == n_total:
                print(f"[build] {n}/{n_total} ({time.time()-t0:.1f}s)", flush=True)
    else:
        with Pool(workers) as pool:
            for batch in pool.imap_unordered(_worker, tasks):
                all_records.extend(batch)
                for rec in batch:
                    counts[rec["difficulty"]] += 1
                n = len(all_records)
                if n % 5000 < len(batch):
                    print(f"[build] {n}/{n_total} ({time.time()-t0:.1f}s) {dict(counts)}",
                          flush=True)

    print(f"[build] done: {len(all_records)} records in {time.time()-t0:.1f}s", flush=True)
    return all_records, dict(counts)


def write_records(records: list, out_path: str, seed: int) -> None:
    rng = random.Random(seed + 77)
    rng.shuffle(records)
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Stats + digit coverage diagnostic
# ---------------------------------------------------------------------------

def _n_actual_digits(value: int, n_digits: int = 5) -> int:
    v = max(0, int(value))
    if v == 0:
        return 1
    n = int(math.floor(math.log10(v))) + 1
    return min(n, n_digits)


def print_stats(records: list, label: str, n_digits: int = 5) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total records: {len(records)}")

    diff_counts = defaultdict(int)
    for rec in records:
        diff_counts[rec["difficulty"]] += 1
    print(f"  Difficulty distribution:")
    for d in ["easy", "medium", "hard"]:
        n = diff_counts[d]
        pct = 100 * n / max(len(records), 1)
        print(f"    {d:8s}: {n:6d}  ({pct:.1f}%)")

    op_counts = defaultdict(int)
    for rec in records:
        for op in rec["factor_types"]:
            op_counts[op] += 1
    total_ops = sum(op_counts.values())
    print(f"  Operation distribution ({total_ops} total factors):")
    for op in OPS:
        n = op_counts[op]
        pct = 100 * n / max(total_ops, 1)
        print(f"    {op:5s}: {n:6d}  ({pct:.1f}%)")

    # Digit coverage analysis — across ALL gold values in ALL records.
    #
    # For MSD-first n_digits=5 layout:
    #   position 0 = ten-thousands  (valid when value >= 10000)
    #   position 1 = thousands       (valid when value >= 1000)
    #   position 2 = hundreds        (valid when value >= 100)
    #   position 3 = tens            (valid when value >= 10)
    #   position 4 = ones            (always valid)
    pos_valid_count = np.zeros(n_digits, dtype=np.int64)
    total_values = 0
    actual_digit_sum = 0
    for rec in records:
        for v in rec["gold_values"]:
            ad = _n_actual_digits(int(v), n_digits)
            actual_digit_sum += ad
            total_values += 1
            # MSD: valid positions are the trailing ad positions: [n_digits-ad : n_digits)
            for p in range(n_digits - ad, n_digits):
                pos_valid_count[p] += 1

    mean_ad = actual_digit_sum / max(total_values, 1)
    print(f"  Mean actual_digits per gold value: {mean_ad:.2f}")
    pos_labels = ["10000s", "1000s", "100s", "tens", "ones"]
    print(f"  Per-position valid-digit coverage (over {total_values} gold values):")
    for p in range(n_digits):
        pct = 100 * pos_valid_count[p] / max(total_values, 1)
        print(f"    pos{p} ({pos_labels[p]:>7s}): {int(pos_valid_count[p]):>7d}  "
              f"({pct:5.1f}%)")

    # Per-position coverage on UNOBSERVED-only values (the ones the model is asked
    # to predict) — this is the signal that actually trains the model.
    pos_valid_unobs = np.zeros(n_digits, dtype=np.int64)
    total_unobs = 0
    for rec in records:
        obs = rec["observed_mask"]
        for i, v in enumerate(rec["gold_values"]):
            if obs[i] == 1:
                continue
            ad = _n_actual_digits(int(v), n_digits)
            total_unobs += 1
            for p in range(n_digits - ad, n_digits):
                pos_valid_unobs[p] += 1
    print(f"  Per-position valid-digit coverage (UNOBSERVED only, "
          f"{total_unobs} cells):")
    for p in range(n_digits):
        pct = 100 * pos_valid_unobs[p] / max(total_unobs, 1)
        print(f"    pos{p} ({pos_labels[p]:>7s}): {int(pos_valid_unobs[p]):>7d}  "
              f"({pct:5.1f}%)")

    nv = defaultdict(int)
    for rec in records:
        nv[rec["n_vars"]] += 1
    print(f"  n_vars distribution: {dict(sorted(nv.items()))}")
    print()


def print_samples(records: list, n: int = 3) -> None:
    rng = random.Random(101)
    samples = rng.sample(records, min(n, len(records)))
    print(f"  Spot-check ({n} samples):")
    for i, rec in enumerate(samples):
        print(f"\n  --- Sample {i+1} ---")
        print(f"  difficulty={rec['difficulty']}  n_vars={rec['n_vars']}  "
              f"n_factors={rec['n_factors']}  query_idx={rec['query_idx']}")
        for fi, (fa, ft) in enumerate(zip(rec["factor_args"], rec["factor_types"])):
            a, b, r = fa
            av, bv, rv = rec["gold_values"][a], rec["gold_values"][b], rec["gold_values"][r]
            print(f"    factor {fi}: x{a}={av}  {ft}  x{b}={bv}  ->  x{r}={rv}")
        print(f"  observed: {rec['observed_values']}")
        print(f"  gold:     {rec['gold_values']}")
        print(f"  answer (x{rec['query_idx']}): {rec['gold_values'][rec['query_idx']]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-out",  default=".cache/factor_graph_train_loguniform.jsonl")
    ap.add_argument("--test-out",   default=".cache/factor_graph_test_loguniform.jsonl")
    ap.add_argument("--train-n",    type=int, default=TRAIN_N)
    ap.add_argument("--test-n",     type=int, default=TEST_N)
    ap.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    ap.add_argument("--test-seed",  type=int, default=TEST_SEED)
    ap.add_argument("--workers",    type=int, default=8)
    args = ap.parse_args()

    print(f"[main] v105.1.2 v2 — log-uniform factor graph dataset builder")
    print(f"  train: {args.train_n} records  seed={args.train_seed}  -> {args.train_out}")
    print(f"  test:  {args.test_n} records  seed={args.test_seed}   -> {args.test_out}")
    print(f"  operand range: log-uniform in [10^0, 10^{LOG_HI:.0f}] = [1, 9999]")
    print(f"  value max: {VALUE_MAX} (5 digits)")
    print()

    t_all = time.time()

    train_records, _ = build(args.train_out, args.train_n, args.train_seed, args.workers)
    print(f"\n[verify] Verifying train set (100 samples)...", flush=True)
    run_verification(train_records, n_sample=100)
    print_stats(train_records, f"TRAIN ({args.train_out})")
    print_samples(train_records, n=3)
    write_records(train_records, args.train_out, args.train_seed)
    print(f"\n[main] Train written: {len(train_records)} records -> {args.train_out}",
          flush=True)

    test_records, _ = build(args.test_out, args.test_n, args.test_seed, args.workers)
    print(f"\n[verify] Verifying test set (100 samples)...", flush=True)
    run_verification(test_records, n_sample=100)
    print_stats(test_records, f"TEST ({args.test_out})")
    write_records(test_records, args.test_out, args.test_seed)
    print(f"[main] Test written: {len(test_records)} records -> {args.test_out}",
          flush=True)

    print(f"\n[main] All done in {time.time()-t_all:.1f}s total.", flush=True)


if __name__ == "__main__":
    main()
