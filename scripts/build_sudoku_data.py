"""Sudoku puzzle generator for v98 (the breathing transformer's first cold test of
constraint propagation without NL parsing).

Generates puzzles + solutions in JSONL form:
  {"input": [81 ints, 0=unknown 1..9], "solution": [81 ints, 1..9],
   "n_givens": int, "difficulty": "easy"|"medium"|"hard"|"expert"}

Algorithm:
  1. Generate a full valid 9x9 grid via shuffled backtracking (random base perms +
     symmetry-breaking transforms keep distributions varied across runs).
  2. Choose a random cell removal order; remove one at a time, calling the bitset
     solver to check whether the puzzle remains uniquely-solvable (count_solutions
     stops at 2 — we only need to know "exactly one" vs "more than one").
  3. Stop once we reach the target n_givens (or removal would create multiple
     solutions). Tag difficulty by n_givens band.

Uniqueness is enforced. Puzzles failing uniqueness are skipped (the generator
retries with a fresh full grid).

Speed: pure-python bitset solver hits 1-3 puzzles/sec on a single core, so we
shard work across processes (multiprocessing.Pool). Streaming JSONL writes mean
the script is restart-safe.

Usage:
  python scripts/build_sudoku_data.py --out .cache/sudoku_train.jsonl --n 50000
  python scripts/build_sudoku_data.py --out .cache/sudoku_val.jsonl   --n 2000  --seed 17
  python scripts/build_sudoku_data.py --out .cache/sudoku_test.jsonl  --n 5000  --seed 23
"""
import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool


# Difficulty bands by n_givens (more givens = easier). Cells removed = 81 - n_givens.
DIFFICULTY_BANDS = {
    "easy":   (30, 40),   # G in [30, 40]
    "medium": (25, 29),   # G in [25, 29]
    "hard":   (20, 24),   # G in [20, 24]
    "expert": (17, 19),   # G in [17, 19]  (17 = minimum for unique solution)
}

# Default 25/25/25/25 split. Override via env if needed.
DIFFICULTY_WEIGHTS = {"easy": 0.25, "medium": 0.25, "hard": 0.25, "expert": 0.25}


# ---------- bitset solver -----------------------------------------------------

# A 9x9 sudoku is represented as a length-81 list of ints (0 = blank, 1..9 = digit).
# Bitsets track per-row/col/box which digits are USED (bit i = digit i+1 used).

def _idx(r: int, c: int) -> int:
    return r * 9 + c

def _box(r: int, c: int) -> int:
    return (r // 3) * 3 + (c // 3)


def _init_bitsets(grid):
    """Initialize row/col/box used-digit bitsets from a partial grid."""
    rows = [0] * 9
    cols = [0] * 9
    boxes = [0] * 9
    for r in range(9):
        for c in range(9):
            d = grid[_idx(r, c)]
            if d:
                bit = 1 << (d - 1)
                rows[r] |= bit
                cols[c] |= bit
                boxes[_box(r, c)] |= bit
    return rows, cols, boxes


def count_solutions(grid, limit: int = 2) -> int:
    """Count solutions up to `limit`. Returns min(actual_count, limit).
    Uses MRV (minimum-remaining-values) heuristic for speed."""
    g = list(grid)
    rows, cols, boxes = _init_bitsets(g)
    count = [0]

    def solve():
        if count[0] >= limit:
            return
        # MRV: find blank cell with the fewest candidates.
        best_pos = -1
        best_cands = 0
        best_count = 10
        for i in range(81):
            if g[i] == 0:
                r, c = i // 9, i % 9
                used = rows[r] | cols[c] | boxes[_box(r, c)]
                cands = (~used) & 0x1FF  # 9 bits
                # popcount via builtin
                n = bin(cands).count("1")
                if n == 0:
                    return  # dead end
                if n < best_count:
                    best_count = n
                    best_pos = i
                    best_cands = cands
                    if n == 1:
                        break
        if best_pos == -1:
            count[0] += 1
            return
        r, c = best_pos // 9, best_pos % 9
        b = _box(r, c)
        cands = best_cands
        while cands:
            bit = cands & (-cands)  # lowest set bit
            cands &= cands - 1
            d = bit.bit_length()  # 1..9
            g[best_pos] = d
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            solve()
            if count[0] >= limit:
                return
            g[best_pos] = 0
            rows[r] &= ~bit
            cols[c] &= ~bit
            boxes[b] &= ~bit

    solve()
    return count[0]


def solve_unique(grid):
    """If `grid` has exactly one solution, return it (length-81 list). Else None."""
    g = list(grid)
    rows, cols, boxes = _init_bitsets(g)
    solutions = []

    def solve():
        if len(solutions) > 1:
            return
        best_pos = -1
        best_cands = 0
        best_count = 10
        for i in range(81):
            if g[i] == 0:
                r, c = i // 9, i % 9
                used = rows[r] | cols[c] | boxes[_box(r, c)]
                cands = (~used) & 0x1FF
                n = bin(cands).count("1")
                if n == 0:
                    return
                if n < best_count:
                    best_count = n
                    best_pos = i
                    best_cands = cands
                    if n == 1:
                        break
        if best_pos == -1:
            solutions.append(list(g))
            return
        r, c = best_pos // 9, best_pos % 9
        b = _box(r, c)
        cands = best_cands
        while cands:
            bit = cands & (-cands)
            cands &= cands - 1
            d = bit.bit_length()
            g[best_pos] = d
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            solve()
            if len(solutions) > 1:
                return
            g[best_pos] = 0
            rows[r] &= ~bit
            cols[c] &= ~bit
            boxes[b] &= ~bit

    solve()
    return solutions[0] if len(solutions) == 1 else None


# ---------- full-grid generator -----------------------------------------------

def generate_full_grid(rng: random.Random) -> list:
    """Generate a random valid 9x9 grid via shuffled backtracking. Always succeeds."""
    g = [0] * 81
    rows = [0] * 9
    cols = [0] * 9
    boxes = [0] * 9
    order = list(range(81))

    def fill(idx):
        if idx == 81:
            return True
        # We just go cell-by-cell (no MRV); shuffling digits is enough to randomize.
        i = order[idx]
        r, c = i // 9, i % 9
        b = _box(r, c)
        used = rows[r] | cols[c] | boxes[b]
        digits = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        rng.shuffle(digits)
        for d in digits:
            bit = 1 << (d - 1)
            if used & bit:
                continue
            g[i] = d
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            if fill(idx + 1):
                return True
            g[i] = 0
            rows[r] &= ~bit
            cols[c] &= ~bit
            boxes[b] &= ~bit
        return False

    ok = fill(0)
    assert ok, "Backtracking failed to fill 9x9 (should be impossible)"
    return g


# ---------- puzzle generator (remove cells, maintain uniqueness) --------------

def generate_puzzle(rng: random.Random, target_givens: int, max_attempts: int = 5,
                    tolerance: int = 3):
    """Generate one puzzle with ~target_givens cells filled (1..9), uniquely solvable.

    The remove-and-check algorithm sometimes hits a stuck state (no further removal
    preserves uniqueness) above the target. We accept any final count within
    `tolerance` cells of the target. For expert puzzles (target 17-19) we tolerate
    a bit more headroom since reaching 17 is provably the minimum uniquely-solvable.

    Returns (input_grid, solution) or (None, None) if no puzzle could be created
    after `max_attempts` full-grid retries.
    """
    best = None  # (puzzle, solution, filled) with smallest filled across attempts
    for attempt in range(max_attempts):
        solution = generate_full_grid(rng)
        puzzle = list(solution)
        order = list(range(81))
        rng.shuffle(order)
        filled = 81
        for cell in order:
            if filled <= target_givens:
                break
            saved = puzzle[cell]
            puzzle[cell] = 0
            if count_solutions(puzzle, limit=2) != 1:
                puzzle[cell] = saved
            else:
                filled -= 1
        if best is None or filled < best[2]:
            best = (puzzle, solution, filled)
        if filled <= target_givens + tolerance:
            sol_check = solve_unique(puzzle)
            if sol_check is not None and sol_check == solution:
                return puzzle, solution
    # Use best-so-far (largest tolerance) if it's a reasonable puzzle
    if best is not None and best[2] <= 50:
        sol_check = solve_unique(best[0])
        if sol_check is not None and sol_check == best[1]:
            return best[0], best[1]
    return None, None


def difficulty_for_givens(n_givens: int) -> str:
    for label, (lo, hi) in DIFFICULTY_BANDS.items():
        if lo <= n_givens <= hi:
            return label
    if n_givens > 40:
        return "easy"
    return "expert"  # extreme low


def n_givens_for_difficulty(difficulty: str, rng: random.Random) -> int:
    lo, hi = DIFFICULTY_BANDS[difficulty]
    return rng.randint(lo, hi)


# ---------- worker for multiprocessing ----------------------------------------

def _worker(args):
    seed, difficulty, n_per_worker = args
    rng = random.Random(seed)
    results = []
    for _ in range(n_per_worker):
        target = n_givens_for_difficulty(difficulty, rng)
        puzzle, solution = generate_puzzle(rng, target_givens=target)
        if puzzle is None:
            continue
        n_givens = sum(1 for x in puzzle if x != 0)
        results.append({
            "input": puzzle,
            "solution": solution,
            "n_givens": n_givens,
            "difficulty": difficulty_for_givens(n_givens),
        })
    return results


def build(out_path: str, n_total: int, seed: int, workers: int = 4, easy_only: bool = False):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Allocate puzzles per difficulty (or all easy)
    if easy_only:
        plan = {"easy": n_total}
    else:
        plan = {}
        remaining = n_total
        keys = list(DIFFICULTY_WEIGHTS.keys())
        for i, k in enumerate(keys):
            if i == len(keys) - 1:
                plan[k] = remaining
            else:
                n = int(round(DIFFICULTY_WEIGHTS[k] * n_total))
                plan[k] = n
                remaining -= n

    print(f"[build] plan: {plan} (total {n_total}) -> {out_path}", flush=True)

    # Build task list — shard each difficulty's count across workers.
    tasks = []
    salt = 0
    for diff, n in plan.items():
        # Use n_per_worker that's not too small (overhead) or too large (load balance).
        chunk = max(20, (n + workers - 1) // workers)
        offset = 0
        while offset < n:
            n_this = min(chunk, n - offset)
            tasks.append((seed * 100003 + salt, diff, n_this))
            salt += 1
            offset += n_this

    rng = random.Random(seed)
    rng.shuffle(tasks)

    t0 = time.time()
    n_written = 0
    counts = {k: 0 for k in plan.keys()}
    with open(out_path, "w") as f:
        if workers <= 1:
            for task in tasks:
                results = _worker(task)
                for rec in results:
                    f.write(json.dumps(rec) + "\n")
                    n_written += 1
                    counts[rec["difficulty"]] = counts.get(rec["difficulty"], 0) + 1
                print(f"[build] {n_written}/{n_total} ({time.time()-t0:.1f}s)", flush=True)
        else:
            with Pool(workers) as pool:
                for results in pool.imap_unordered(_worker, tasks):
                    for rec in results:
                        f.write(json.dumps(rec) + "\n")
                        n_written += 1
                        counts[rec["difficulty"]] = counts.get(rec["difficulty"], 0) + 1
                    f.flush()
                    if n_written % 500 < len(results):
                        print(f"[build] {n_written}/{n_total} ({time.time()-t0:.1f}s) counts={counts}", flush=True)

    print(f"[build] done. {n_written} puzzles written in {time.time()-t0:.1f}s", flush=True)
    print(f"[build] final counts by difficulty: {counts}", flush=True)
    return n_written, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--easy-only", action="store_true",
                    help="Generate only easy puzzles (overrides difficulty distribution).")
    args = ap.parse_args()
    build(args.out, args.n, args.seed, workers=args.workers, easy_only=args.easy_only)


if __name__ == "__main__":
    main()
