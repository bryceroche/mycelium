"""diag_kenken_cell_difficulty.py — WHERE does the v98 KenKen model fail?

Read-only analysis (NO training). Cross-tabulates the trained model's per-cell
accuracy against each cell's DEDUCTION DEPTH (the propagation round at which the
generator's constraint-propagation solver first pinned that cell to a singleton).

HYPOTHESIS under test: the model solves EASY cells (givens + immediately-forced)
but fails DEEP-DEDUCTION cells (those requiring multi-step constraint chains).

Pipeline:
  1. Build the trained model exactly as scripts/kenken_train.py does for the
     KENKEN_BACKBONE=pythia path:
        _load_state -> load_breathing -> cast_layers_fp32
        -> attach_kenken_params(k_max=K) -> load_ckpt(step1500)
     and run kenken_breathing_forward at K=8 (final-breath argmax) per puzzle.
  2. For EVERY puzzle in .cache/kenken_test_struct.jsonl (1200 puzzles), compute
     per cell:
       (a) MODEL correctness (final-breath argmax vs gold), over VALID cells.
       (b) RESOLUTION ROUND: instrument the generator's propagate() (imported
           from scripts/build_kenken_data.py) to record the round at which each
           cell FIRST became a singleton. givens=round 0; cells forced in the 1st
           pass=round 1; ...; deep cells resolved late=round 5+.
  3. Cross-tabulate model per-cell accuracy by RESOLUTION ROUND (0,1,2,3,4,5+),
     also broken down by puzzle deduction_depth and by N (5/6/7).
  4. Sanity checks: round-0 (givens) acc should be ~1.0; round-1 acc.
  5. The KEY READ: does accuracy MONOTONICALLY DECREASE with resolution round?

Usage:
  KENKEN_TASK=1 KENKEN_K_MAX=8 .venv/bin/python scripts/diag_kenken_cell_difficulty.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv

# ---- import the model machinery exactly as kenken_train.py does --------------
from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.kenken import (
    attach_kenken_params, kenken_breathing_forward,
)
from mycelium.kenken_data import (
    KenKenLoader, load_jsonl, N_MAX, N_CELLS, _rc_to_flat,
)
# kenken_train.py builds the fp32 cast + ckpt load; mirror them by import.
from scripts.kenken_train import cast_layers_fp32, load_ckpt

# ---- import the GENERATOR's constraint-propagation solver (round-tracking) ----
import itertools
from scripts.build_kenken_data import cage_ok, _prod


CKPT = os.environ.get(
    "KENKEN_DIAG_CKPT",
    ".cache/kenken_ckpts/kenken_k8_run/kenken_k8_run_step1500.safetensors")
TEST = os.environ.get("KENKEN_DIAG_TEST", ".cache/kenken_test_struct.jsonl")
K = int(getenv("KENKEN_K_MAX", "8"))
EVAL_BATCH = int(getenv("EVAL_BATCH", "8"))


# =============================================================================
# (1) RESOLUTION-ROUND instrument — adapted from build_kenken_data.propagate().
#     IDENTICAL propagation logic; the ONLY addition is recording, per cell, the
#     round at which |domain|==1 is first reached. Givens (singleton at entry)
#     are round 0; cells pinned in pass i get round i.
# =============================================================================

def propagate_with_rounds(n, cages, clues, max_rounds=64):
    """Run the generator's constraint propagation, returning per-cell resolution
    rounds: a dict {(r,c): round_int}. Givens=0; cells forced on pass i => i.

    The propagation body is COPIED VERBATIM from build_kenken_data.propagate so
    the round labels match the generator's deduction_depth definition exactly.
    A cell is recorded the FIRST time its domain becomes a singleton; unresolved
    cells (shouldn't happen on the uniqueness-verified corpus) are left as -1.
    """
    dom = {(r, c): set(range(1, n + 1)) for r in range(n) for c in range(n)}
    for cage, (typ, tgt) in zip(cages, clues):
        if typ == "given":
            dom[cage[0]] = {tgt}

    resolved_round = {(r, c): -1 for r in range(n) for c in range(n)}
    # givens are singletons at entry -> round 0
    for cell, d in dom.items():
        if len(d) == 1:
            resolved_round[cell] = 0

    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        changed = False
        # --- row/col singleton elimination (VERBATIM from propagate) ---
        for (r, c), d in list(dom.items()):
            if len(d) == 1:
                v = next(iter(d))
                for cc in range(n):
                    if cc != c and v in dom[(r, cc)]:
                        dom[(r, cc)].discard(v); changed = True
                for rr in range(n):
                    if rr != r and v in dom[(rr, c)]:
                        dom[(rr, c)].discard(v); changed = True
        # --- hidden single in a line (VERBATIM) ---
        for line in ([[(r, c) for c in range(n)] for r in range(n)] +
                     [[(r, c) for r in range(n)] for c in range(n)]):
            for v in range(1, n + 1):
                homes = [cell for cell in line if v in dom[cell]]
                if len(homes) == 1 and len(dom[homes[0]]) > 1:
                    dom[homes[0]] = {v}; changed = True
        # --- cage arithmetic support filtering (VERBATIM) ---
        for cage, (typ, tgt) in zip(cages, clues):
            doms = [sorted(dom[cell]) for cell in cage]
            if _prod([len(d) for d in doms]) > 20000:
                continue
            support = [set() for _ in cage]
            for combo in itertools.product(*doms):
                ok = True
                for i in range(len(cage)):
                    for j in range(i + 1, len(cage)):
                        if combo[i] == combo[j] and (
                                cage[i][0] == cage[j][0] or cage[i][1] == cage[j][1]):
                            ok = False; break
                    if not ok:
                        break
                if ok and cage_ok(typ, tgt, combo):
                    for i, v in enumerate(combo):
                        support[i].add(v)
            for i, cell in enumerate(cage):
                if support[i] and (dom[cell] - support[i]):
                    dom[cell] &= support[i]; changed = True
        # --- record cells that became singletons THIS round (first time only) ---
        for cell, d in dom.items():
            if len(d) == 1 and resolved_round[cell] == -1:
                resolved_round[cell] = rounds
        if not changed:
            break

    return resolved_round, rounds


def round_bucket(rnd: int) -> str:
    """Bucket a resolution round into {0,1,2,3,4,5+}. -1 (unresolved) -> 'unres'."""
    if rnd < 0:
        return "unres"
    if rnd >= 5:
        return "5+"
    return str(rnd)


BUCKETS = ["0", "1", "2", "3", "4", "5+"]


# =============================================================================
# (2) Build the trained model exactly as kenken_train.py does (pythia path).
# =============================================================================

def build_model():
    cfg = Config()
    print(f"loading Pythia-410M -> breathing transformer ...", flush=True)
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    cast_layers_fp32(model)
    attach_kenken_params(model, hidden=cfg.hidden, n_heads=cfg.n_heads, k_max=K)
    Device[Device.DEFAULT].synchronize()
    print(f"loading kenken ckpt: {CKPT}", flush=True)
    load_ckpt(model, CKPT)
    Device[Device.DEFAULT].synchronize()
    return model, cfg


# =============================================================================
# (3) Main: run the model over the test set, compute per-cell correctness, and
#     the per-cell resolution rounds (CPU solver), then cross-tabulate.
# =============================================================================

def main():
    assert int(getenv("KENKEN_TASK", "0")) > 0, "KENKEN_TASK=1 must be set"
    t0 = time.time()

    test_recs = load_jsonl(TEST)
    print(f"loaded {len(test_recs)} test puzzles from {TEST}", flush=True)

    # n_cages_max must match the train graph (pinned in measured_config = 30); but
    # for EVAL the loader only needs n_cages_max >= its own corpus max. We use the
    # corpus max here (eval forward retraces fresh each batch — no train JIT graph
    # to match), so any value >= corpus max is byte-equivalent for the forward.
    corpus_cages_max = max(len(r["cages"]) for r in test_recs)
    n_cages_max = int(getenv("KENKEN_N_CAGES_MAX", str(max(corpus_cages_max, 30))))
    print(f"n_cages_max={n_cages_max} (corpus max={corpus_cages_max})", flush=True)

    model, cfg = build_model()

    # --- loader (eval order, padded last batch) ---
    Tensor.training = False
    loader = KenKenLoader(TEST, batch_size=EVAL_BATCH, seed=123,
                          n_cages_max=n_cages_max)

    # --- per-cell resolution rounds for every puzzle (CPU, deterministic) ---
    # Compute ONCE per puzzle, in corpus order. We re-key by the record's identity
    # (we iterate the loader in the SAME order: iter_eval walks self.records in
    # order, and KenKenLoader.records == load_jsonl(path) == test_recs order).
    print("computing per-cell resolution rounds (generator solver)...", flush=True)
    tR = time.time()
    rounds_per_puzzle = []     # list aligned to test_recs: each = (N, flat->round dict)
    depth_check_mismatch = 0
    for rec in test_recs:
        n = int(rec["N"])
        cages = [[tuple(c) for c in cage] for cage in rec["cages"]]
        clues = [(str(cl[0]), int(cl[1])) for cl in rec["clues"]]
        rr, n_rounds = propagate_with_rounds(n, cages, clues)
        # sanity: our reproduced round count should equal the stored deduction_depth.
        if int(rec.get("deduction_depth", -999)) != n_rounds:
            depth_check_mismatch += 1
        # flatten to the N_max grid index space
        flat_round = {}
        for (r, c), rnd in rr.items():
            flat_round[_rc_to_flat(r, c)] = rnd
        rounds_per_puzzle.append((n, flat_round))
    print(f"  rounds done in {time.time()-tR:.1f}s; "
          f"deduction_depth==reproduced_rounds mismatch on "
          f"{depth_check_mismatch}/{len(test_recs)} puzzles", flush=True)

    # --- accumulators ---
    # Per resolution-round bucket: [n_correct, n_cells]. (givens INCLUDED for the
    # sanity check; the model predicts every valid cell.)
    bucket_acc = {b: [0, 0] for b in BUCKETS + ["unres"]}
    # by puzzle deduction_depth: {depth: [n_correct, n_cells]}
    by_depth = {}
    # by N: {N: [n_correct, n_cells]}
    by_N = {}
    # by N x bucket: {(N, bucket): [n_correct, n_cells]}
    by_N_bucket = {}
    # also: bucket restricted to NON-GIVEN cells (the supervised set) -> [c, n]
    bucket_acc_solve = {b: [0, 0] for b in BUCKETS + ["unres"]}

    # puzzle-level
    n_puzzles = 0
    n_puzzles_solved = 0

    print("running model forward over test set...", flush=True)
    puzzle_idx = 0
    n_batches = 0
    for batch in loader.iter_eval(batch_size=EVAL_BATCH):
        cell_logits_history, _ = kenken_breathing_forward(model, batch, K=K)
        final_logits = cell_logits_history[-1]
        pred_np = (final_logits.argmax(axis=-1) + 1).realize().numpy().astype(np.int32)  # (B,49)
        gold_np = batch.gold.realize().numpy().astype(np.int32)
        valid_np = batch.cell_valid.realize().numpy() > 0.5
        given_np = batch.input_cells.realize().numpy().astype(np.int32) > 0  # (B,49)

        Bn = pred_np.shape[0]
        for b in range(Bn):
            # The loader pads the LAST batch with repeats of records[0]; only the
            # first `len(records)` puzzles are real. Stop counting past the end.
            if puzzle_idx >= len(test_recs):
                break
            N_i, flat_round = rounds_per_puzzle[puzzle_idx]
            valid = valid_np[b]
            correct = (pred_np[b] == gold_np[b])
            given = given_np[b]

            puzzle_ok = True
            for f in range(N_CELLS):
                if not valid[f]:
                    continue
                rnd = flat_round.get(f, -1)
                bkt = round_bucket(rnd)
                c = 1 if correct[f] else 0
                bucket_acc[bkt][0] += c
                bucket_acc[bkt][1] += 1
                if not given[f]:
                    bucket_acc_solve[bkt][0] += c
                    bucket_acc_solve[bkt][1] += 1
                    if not correct[f]:
                        puzzle_ok = False
                # by N
                by_N.setdefault(N_i, [0, 0])
                by_N[N_i][0] += c; by_N[N_i][1] += 1
                by_N_bucket.setdefault((N_i, bkt), [0, 0])
                by_N_bucket[(N_i, bkt)][0] += c; by_N_bucket[(N_i, bkt)][1] += 1
            # by puzzle deduction_depth
            dd = int(test_recs[puzzle_idx].get("deduction_depth", 0))
            by_depth.setdefault(dd, [0, 0])
            # count this puzzle's valid-cell correctness toward its depth bucket
            for f in range(N_CELLS):
                if valid[f]:
                    by_depth[dd][0] += (1 if correct[f] else 0)
                    by_depth[dd][1] += 1
            n_puzzles += 1
            n_puzzles_solved += int(puzzle_ok)
            puzzle_idx += 1
        n_batches += 1
        if n_batches % 20 == 0:
            print(f"  ... {puzzle_idx}/{len(test_recs)} puzzles "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # =========================================================================
    # REPORT
    # =========================================================================
    def acc(pair):
        c, n = pair
        return (c / n) if n > 0 else float("nan")

    overall_c = sum(v[0] for v in bucket_acc.values())
    overall_n = sum(v[1] for v in bucket_acc.values())
    solve_c = sum(v[0] for v in bucket_acc_solve.values())
    solve_n = sum(v[1] for v in bucket_acc_solve.values())

    print("\n" + "=" * 72)
    print("KENKEN CELL-DIFFICULTY DIAGNOSIS")
    print(f"  ckpt: {CKPT}")
    print(f"  test: {TEST}  ({n_puzzles} puzzles, K={K})")
    print("=" * 72)

    print(f"\nOVERALL  cell_acc (valid cells, incl givens) = {acc((overall_c, overall_n)):.4f} "
          f"(n={overall_n})")
    print(f"OVERALL  cell_acc (NON-given / supervised)   = {acc((solve_c, solve_n)):.4f} "
          f"(n={solve_n})")
    print(f"PUZZLE   acc (all supervised cells correct)  = {n_puzzles_solved}/{n_puzzles} "
          f"= {n_puzzles_solved/max(n_puzzles,1):.4f}")

    print("\n--- HEADLINE: model cell-accuracy by RESOLUTION ROUND ---")
    print("  (round 0 = givens; round r = forced at the r-th propagation pass)")
    print(f"  {'round':>6} | {'n_cells':>9} | {'all_acc':>8} | "
          f"{'n_solve':>8} | {'solve_acc':>9}")
    print("  " + "-" * 56)
    for b in BUCKETS + ["unres"]:
        if bucket_acc[b][1] == 0 and bucket_acc_solve[b][1] == 0:
            continue
        print(f"  {b:>6} | {bucket_acc[b][1]:>9} | {acc(bucket_acc[b]):>8.4f} | "
              f"{bucket_acc_solve[b][1]:>8} | {acc(bucket_acc_solve[b]):>9.4f}")

    print("\n--- SANITY CHECKS ---")
    g_acc = acc(bucket_acc["0"])
    print(f"  (i)  round-0 (GIVENS) acc = {g_acc:.4f}  "
          f"{'[OK ~1.0]' if g_acc > 0.98 else '[RED FLAG: givens not reproduced!]'}")
    print(f"  (ii) round-1 (immediately-forced) acc = {acc(bucket_acc['1']):.4f} "
          f"(solve-only={acc(bucket_acc_solve['1']):.4f})")

    print("\n--- by PUZZLE deduction_depth (all valid cells) ---")
    print(f"  {'depth':>6} | {'n_cells':>9} | {'cell_acc':>8}")
    print("  " + "-" * 32)
    for dd in sorted(by_depth.keys()):
        print(f"  {dd:>6} | {by_depth[dd][1]:>9} | {acc(by_depth[dd]):>8.4f}")

    print("\n--- by N (all valid cells) ---")
    print(f"  {'N':>3} | {'n_cells':>9} | {'cell_acc':>8}")
    print("  " + "-" * 28)
    for Ni in sorted(by_N.keys()):
        print(f"  {Ni:>3} | {by_N[Ni][1]:>9} | {acc(by_N[Ni]):>8.4f}")

    print("\n--- by N x RESOLUTION ROUND (all valid cells) ---")
    header = "  N   " + " ".join(f"{b:>9}" for b in BUCKETS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for Ni in sorted(by_N.keys()):
        row = f"  {Ni:>3} "
        for b in BUCKETS:
            pr = by_N_bucket.get((Ni, b))
            row += f" {acc(pr):>9.4f}" if pr and pr[1] > 0 else f" {'-':>9}"
        print(row)

    # --- the KEY READ (programmatic monotonicity check) ---
    seq = [(b, acc(bucket_acc_solve[b])) for b in BUCKETS
           if bucket_acc_solve[b][1] >= 30]  # ignore tiny buckets
    print("\n--- KEY READ: monotonic decrease with resolution round? (solve-only) ---")
    print("  " + "  ".join(f"r{b}={a:.3f}" for b, a in seq))
    if len(seq) >= 2:
        vals = [a for _, a in seq]
        strictly_dec = all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1))
        drop = vals[0] - vals[-1]
        spread = max(vals) - min(vals)
        print(f"  monotonic-nonincreasing = {strictly_dec}; "
              f"first->last drop = {drop:+.3f}; spread(max-min) = {spread:.3f}")
        if strictly_dec and drop > 0.15:
            verdict = ("DEEP-DEDUCTION CHAINING is the ceiling: shallow propagation "
                       "works, deep chains fail.")
        elif spread < 0.10:
            verdict = ("FLAT across rounds: the model fails ~uniformly, NOT depth-"
                       "specific. A different fix than depth-targeted self-sup.")
        else:
            verdict = ("PARTIAL depth signal: accuracy degrades with depth but not "
                       "cleanly monotonic. Mixed read.")
        if g_acc <= 0.98:
            verdict = ("READOUT/REPRESENTATION problem: givens (round 0) are NOT "
                       "reproduced at ~1.0 — failure is at the most basic level. " + verdict)
        print(f"\n  VERDICT: {verdict}")

    print(f"\ndone in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
