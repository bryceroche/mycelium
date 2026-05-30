"""Synthetic arithmetic factor graph generator for Mycelium v99.

Generalizes v98 Sudoku constraint propagation to arbitrary arithmetic DAGs.
Each record is a factor graph where variables are connected by arithmetic
constraints (add/sub/mul/div). The model must infer unobserved values given
the observed leaves.

Factor graph DAG:
  - Variables: x_0 .. x_{n_total-1}
    n_leaves leaf inputs + n_factors intermediate/output results.
  - Factors: factor_args[i] = [arg1_idx, arg2_idx, result_idx]
    computes gold_values[result_idx] = op(gold_values[arg1_idx], gold_values[arg2_idx])
  - Leaf variables (no incoming factor edge) are observed.
  - Result variables are unobserved (to be inferred).

Usage:
  python3 scripts/build_factor_graph_data.py
  python3 scripts/build_factor_graph_data.py --train-n 50000 --test-n 5000

Output:
  .cache/factor_graph_train.jsonl  (50,000 records)
  .cache/factor_graph_test.jsonl   (5,000 records)
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from multiprocessing import Pool

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_N = 50_000
TEST_N = 5_000
TRAIN_SEED = 42
TEST_SEED = 43

# Difficulty: n_leaves = number of input (leaf) variables
DIFFICULTY_BANDS = {
    "easy":   (3, 4),
    "medium": (5, 6),
    "hard":   (7, 8),
}

OPS = ("add", "sub", "mul", "div")
VALUE_MAX = 99
VALUE_MIN = 0
MAX_RETRIES_PER_RECORD = 500


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
    """Sample (a, b) such that apply_op(op, a, b) is not None.
    Returns (a, b, result) or None after exhausting retries.
    """
    for _ in range(200):
        if op == "add":
            a = rng.randint(0, 49)
            b = rng.randint(0, 49)
        elif op == "sub":
            a = rng.randint(1, 99)
            b = rng.randint(0, a)
        elif op == "mul":
            a = rng.randint(1, 9)
            b = rng.randint(1, 9)
        elif op == "div":
            b = rng.randint(1, 9)
            max_k = VALUE_MAX // b
            if max_k < 1:
                continue
            k = rng.randint(1, max_k)
            a = b * k
        else:
            return None
        r = apply_op(op, a, b)
        if r is not None and VALUE_MIN <= r <= VALUE_MAX:
            return a, b, r
    return None


# ---------------------------------------------------------------------------
# DAG construction — build values forward (leaves → results)
# ---------------------------------------------------------------------------

def topo_sort(n_vars_total: int, factor_args: list):
    """Kahn's topo sort. Returns ordered list of var indices, or None if cycle."""
    # result_idx has in-degree=2 (depends on two args)
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
    """Chain DAG: op(x0, x1)->xL, op(xL, x2)->xL+1, ...
    Variables: x0..x_{n_leaves-1} are leaves; x_{n_leaves}.. are results.
    """
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
    """Binary tree DAG built by pairwise merging of available nodes."""
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


def generate_record(rng: random.Random, n_leaves: int, difficulty: str):
    """Generate one valid factor graph record, or None on failure."""
    # Choose topology
    use_tree = (n_leaves >= 4) and rng.random() < 0.5
    if use_tree:
        n_total, factor_args = make_tree_dag(rng, n_leaves)
    else:
        n_total, factor_args = make_chain_dag(n_leaves)
    n_factors = len(factor_args)

    # Topo sort (also validates acyclicity)
    topo_order = topo_sort(n_total, factor_args)
    if topo_order is None:
        return None

    # Identify leaf vs result indices
    result_set = {fa[2] for fa in factor_args}
    leaf_set = set(range(n_total)) - result_set
    result_to_fi = {fa[2]: fi for fi, fa in enumerate(factor_args)}

    # Assign ops
    factor_types = [rng.choice(OPS) for _ in range(n_factors)]

    # Propagate values forward in topological order.
    # For leaf nodes: sample a value in [1, 20] (will be replaced if the
    # factor using it needs something specific).
    # For result nodes: sample (a, b) so that op(a, b) is valid,
    # THEN assign a and b to the appropriate source variables if they are
    # leaves. If either source is already a result node (i.e., set by an
    # earlier factor), we try to find an op that works with the existing value.

    gold = [None] * n_total

    # First pass: assign all leaf nodes a default value
    for i in leaf_set:
        gold[i] = rng.randint(1, 20)

    # Second pass: walk topo order and fill results
    for node in topo_order:
        if node not in result_to_fi:
            continue  # leaf, already set

        fi = result_to_fi[node]
        a_idx, b_idx, r_idx = factor_args[fi]
        op = factor_types[fi]
        a_val = gold[a_idx]
        b_val = gold[b_idx]

        if a_val is None or b_val is None:
            return None  # upstream not yet set (topological ordering bug)

        # Try the current (a, b) pair first
        r = apply_op(op, a_val, b_val)

        if r is None:
            # Current pair is invalid for this op.
            # If BOTH sources are leaves, we can resample both from scratch.
            # If one source is already a result (can't change), try to adjust
            # the leaf source or change nothing and fail fast.
            a_is_leaf = a_idx in leaf_set
            b_is_leaf = b_idx in leaf_set

            if a_is_leaf and b_is_leaf:
                # Full freedom: sample a valid pair
                pair = sample_valid_pair(rng, op)
                if pair is None:
                    return None
                gold[a_idx], gold[b_idx], r = pair
            elif a_is_leaf:
                # b is fixed; adjust a to make op(a, b) valid
                r = _find_valid_a(rng, op, b_val)
                if r is None:
                    return None
                gold[a_idx], r = r
            elif b_is_leaf:
                # a is fixed; adjust b to make op(a, b) valid
                r = _find_valid_b(rng, op, a_val)
                if r is None:
                    return None
                gold[b_idx], r = r
            else:
                # Both are result nodes (already fixed). This op is impossible.
                return None

        if r is None or not (VALUE_MIN <= r <= VALUE_MAX):
            return None
        gold[r_idx] = r

    # Verify all set
    if any(v is None for v in gold):
        return None
    if any(not (VALUE_MIN <= v <= VALUE_MAX) for v in gold):
        return None

    # Build observed/unobserved masks
    observed_mask = [1 if i in leaf_set else 0 for i in range(n_total)]
    observed_values = [gold[i] if observed_mask[i] == 1 else None for i in range(n_total)]

    # query_idx: last result node in topo order
    result_nodes_topo = [n for n in topo_order if n in result_set]
    if not result_nodes_topo:
        return None
    query_idx = result_nodes_topo[-1]

    # Reject trivial: answer equals an observed value (allow for larger graphs)
    query_answer = gold[query_idx]
    obs_vals = {v for v in observed_values if v is not None}
    if query_answer in obs_vals and n_leaves <= 3:
        return None

    return {
        "n_vars": n_leaves,
        "n_factors": n_factors,
        "factor_types": factor_types,
        "factor_args": factor_args,
        "observed_mask": observed_mask,
        "observed_values": observed_values,
        "gold_values": gold,
        "query_idx": query_idx,
        "difficulty": difficulty,
    }


def _find_valid_a(rng: random.Random, op: str, b: int):
    """Given fixed b, find (a, result) such that op(a, b) is valid."""
    for _ in range(100):
        if op == "add":
            a = rng.randint(0, VALUE_MAX - b)
        elif op == "sub":
            a = rng.randint(b, VALUE_MAX)
        elif op == "mul":
            if b == 0:
                return None
            max_a = VALUE_MAX // b
            if max_a < 1:
                return None
            a = rng.randint(1, max_a)
        elif op == "div":
            if b == 0:
                return None
            max_k = VALUE_MAX // b
            if max_k < 1:
                return None
            k = rng.randint(1, max_k)
            a = b * k
        else:
            return None
        r = apply_op(op, a, b)
        if r is not None:
            return a, r
    return None


def _find_valid_b(rng: random.Random, op: str, a: int):
    """Given fixed a, find (b, result) such that op(a, b) is valid."""
    for _ in range(100):
        if op == "add":
            b = rng.randint(0, VALUE_MAX - a)
        elif op == "sub":
            b = rng.randint(0, a)
        elif op == "mul":
            if a == 0:
                return None
            max_b = VALUE_MAX // a
            if max_b < 1:
                return None
            b = rng.randint(1, max_b)
        elif op == "div":
            # Find divisors of a in [1, min(a, 20)]
            divisors = [d for d in range(1, min(a + 1, 21)) if a % d == 0]
            if not divisors:
                return None
            b = rng.choice(divisors)
        else:
            return None
        r = apply_op(op, a, b)
        if r is not None:
            return b, r
    return None


def generate_record_with_retry(rng: random.Random, n_leaves: int, difficulty: str):
    for _ in range(MAX_RETRIES_PER_RECORD):
        rec = generate_record(rng, n_leaves, difficulty)
        if rec is not None:
            return rec
    return None


# ---------------------------------------------------------------------------
# Independent verification (separate code path)
# ---------------------------------------------------------------------------

def verify_record(rec: dict) -> bool:
    """Recompute all gold values from observed leaves. Must match gold exactly."""
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

    # All values must match gold
    for i in range(n_total):
        if vals[i] != gold_values[i]:
            return False

    # Observed consistency
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
    print(f"[verify] PASSED: {sample_size} randomly sampled records all correct.")


# ---------------------------------------------------------------------------
# Worker (multiprocessing)
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

    # Equal split across 3 difficulties
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

    # Shard tasks
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
                    print(f"[build] {n}/{n_total} ({time.time()-t0:.1f}s) {dict(counts)}", flush=True)

    print(f"[build] done: {len(all_records)} records in {time.time()-t0:.1f}s", flush=True)
    return all_records, dict(counts)


def write_records(records: list, out_path: str, seed: int) -> None:
    rng = random.Random(seed + 77)
    rng.shuffle(records)
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Stats + spot-check
# ---------------------------------------------------------------------------

def print_stats(records: list, label: str) -> None:
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
        print(f"  difficulty={rec['difficulty']}  n_vars={rec['n_vars']}  n_factors={rec['n_factors']}  query_idx={rec['query_idx']}")
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
    ap.add_argument("--train-out",  default=".cache/factor_graph_train.jsonl")
    ap.add_argument("--test-out",   default=".cache/factor_graph_test.jsonl")
    ap.add_argument("--train-n",    type=int, default=TRAIN_N)
    ap.add_argument("--test-n",     type=int, default=TEST_N)
    ap.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    ap.add_argument("--test-seed",  type=int, default=TEST_SEED)
    ap.add_argument("--workers",    type=int, default=8)
    args = ap.parse_args()

    print(f"[main] Mycelium v99 — factor graph dataset builder")
    print(f"  train: {args.train_n} records  seed={args.train_seed}  -> {args.train_out}")
    print(f"  test:  {args.test_n} records  seed={args.test_seed}   -> {args.test_out}")
    print()

    t_all = time.time()

    # --- Train ---
    train_records, _ = build(args.train_out, args.train_n, args.train_seed, args.workers)
    print(f"\n[verify] Verifying train set (100 samples)...")
    run_verification(train_records, n_sample=100)
    print_stats(train_records, f"TRAIN ({args.train_out})")
    print_samples(train_records, n=3)
    write_records(train_records, args.train_out, args.train_seed)
    print(f"\n[main] Train written: {len(train_records)} records -> {args.train_out}")

    # --- Test ---
    test_records, _ = build(args.test_out, args.test_n, args.test_seed, args.workers)
    print(f"\n[verify] Verifying test set (100 samples)...")
    run_verification(test_records, n_sample=100)
    print_stats(test_records, f"TEST ({args.test_out})")
    write_records(test_records, args.test_out, args.test_seed)
    print(f"[main] Test written: {len(test_records)} records -> {args.test_out}")

    print(f"\n[main] All done in {time.time()-t_all:.1f}s total.")
    print(f"  {args.train_out}: {len(train_records)} records")
    print(f"  {args.test_out}:  {len(test_records)} records")


if __name__ == "__main__":
    main()
