"""sudoku_neural_ordering.py — does the REAL neural policy capture the 33x ordering gap?

The oracle probe showed: on branching Sudoku, symbolic LCV ~199 decisions, perfect
(oracle) ordering ~6, random ~461. This measures where the ACTUAL v98 Sudoku deducer's
policy lands. The deducer proposes the ordering ONCE at the root (one forward -> per-cell
9-way softmax = the policy); pure symbolic GAC search then disposes, ordered by that
policy (policy_valorder). The iron law holds: the policy ORDERS, GAC commits.

  DEV=AMD .venv/bin/python3 scripts/sudoku_neural_ordering.py --ckpt .cache/sudoku_ckpts/v98_prod_final.safetensors

VERDICT logic: neural decisions near oracle (~6) => the method captures the 33x (a real
Sudoku deducer's ordering approaches optimal) => validated for transfer to bigger prey.
Near symbolic (199) or worse => the policy isn't ordering-useful (mediocre policy can hurt).
"""
import argparse
import json
import random
import statistics
import sys

import numpy as np
from tinygrad import Tensor, dtypes

sys.path.insert(0, ".")
from mycelium import Config                                       # noqa: E402
from mycelium.loader import _load_state, load_breathing          # noqa: E402
from mycelium.sudoku import attach_sudoku_params, sudoku_breathing_forward  # noqa: E402
from mycelium.csp_core import (                                   # noqa: E402
    backtrack_search, gac_propagate, mrv_varorder, lcv_valorder, policy_valorder,
)
from mycelium.csp_domains import problem_from_sudoku, sudoku_registry  # noqa: E402
# Reuse the VALIDATED v98 loader (custom key map model_state_dict_sudoku + fp32 cast).
# A plain get_state_dict load silently matches nothing (ckpt keys are 'sudoku.X', model
# attrs are 'sudoku_X') -> random model -> chance cell-acc. This is the working path.
from scripts.eval_v98_sudoku import load_ckpt, cast_layers_fp32  # noqa: E402

BUDGET = 2_000_000
K = 16          # runtime breaths
K_ALLOC = 20   # ckpt was allocated/trained with k_max=20 (breath_embed shape)


def solve(cells, reg, valorder):
    prob = problem_from_sudoku(cells, n=9, registry=reg)
    return backtrack_search(prob, gac_propagate, mrv_varorder, valorder, budget=BUDGET)


def neural_policy(probs_81x9):
    # probs[v][j] = P(cell v = digit j+1). policy[v][a] indexable by value a in 1..9.
    return [[0.0] + [float(probs_81x9[v][j]) for j in range(9)] for v in range(81)]


def oracle_policy(solution):
    return [[1.0 if a == solution[v] else 0.0 for a in range(10)] for v in range(81)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=".cache/sudoku_ckpts/v98_prod_final.safetensors")
    ap.add_argument("--test", default=".cache/sudoku_test.jsonl")
    ap.add_argument("--scan", type=int, default=2000, help="scan this many for branchers")
    ap.add_argument("--min_dec", type=int, default=8, help="keep puzzles with >= this many symbolic decisions")
    ap.add_argument("--max_keep", type=int, default=60)
    args = ap.parse_args()

    reg = sudoku_registry(9)
    recs = []
    with open(args.test) as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
            if len(recs) >= args.scan:
                break

    # 1) find the BRANCHING puzzles (where ordering can matter at all).
    print(f"scanning {len(recs)} for branchers (symbolic decisions >= {args.min_dec})...", flush=True)
    branchers = []
    for rec in recs:
        r = solve(rec["input"], reg, lcv_valorder)
        d = r.get("decisions", 0)
        if d >= args.min_dec:
            branchers.append((d, rec))
    branchers.sort(key=lambda x: -x[0])
    branchers = branchers[:args.max_keep]
    print(f"  {len(branchers)} branchers (max symbolic decisions {branchers[0][0] if branchers else 0})")
    if not branchers:
        print("  no branching puzzles -> symbolic too strong; nothing to order. STOP.")
        return

    # 2) load the v98 deducer, run ONE forward per brancher -> policy.
    print(f"loading v98 sudoku deducer: {args.ckpt}", flush=True)
    cfg = Config()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    cast_layers_fp32(model)                       # MUST precede the load (eval path order)
    attach_sudoku_params(model, hidden=cfg.hidden, n_heads=cfg.n_heads, k_max=K_ALLOC)
    load_ckpt(model, args.ckpt)

    inp = np.stack([rec["input"] for _, rec in branchers]).astype(np.int32)   # (M, 81)
    ic = Tensor(inp, dtype=dtypes.int).contiguous().realize()
    logits_hist, _ = sudoku_breathing_forward(model, ic, K=K)
    probs = logits_hist[-1].softmax(axis=-1).realize().numpy()                # (M, 81, 9)

    # also the deducer's standalone cell-acc on these (how good is the policy?).
    pred = probs.argmax(-1) + 1                                               # (M,81) digit
    sols = np.stack([rec["solution"] for _, rec in branchers])
    blanks = (inp == 0)
    cell_acc = float((pred[blanks] == sols[blanks]).mean())

    # 3) compare orderings on each brancher.
    rows = []
    for i, (d_sym, rec) in enumerate(branchers):
        cells, sol = rec["input"], rec["solution"]
        d_neu = solve(cells, reg, policy_valorder(neural_policy(probs[i]))).get("decisions", -1)
        d_orc = solve(cells, reg, policy_valorder(oracle_policy(sol))).get("decisions", -1)
        rows.append((d_sym, d_neu, d_orc))

    sym = [r[0] for r in rows]; neu = [r[1] for r in rows]; orc = [r[2] for r in rows]
    print(f"\n=== NEURAL value-ordering vs symbolic vs oracle ({len(rows)} branching puzzles) ===")
    print(f"  deducer cell-acc on these blanks: {cell_acc:.3f}")
    print(f"  symbolic (LCV) decisions : median {statistics.median(sym):.0f}  mean {statistics.mean(sym):.1f}  max {max(sym)}")
    print(f"  NEURAL policy   decisions : median {statistics.median(neu):.0f}  mean {statistics.mean(neu):.1f}  max {max(neu)}")
    print(f"  ORACLE policy   decisions : median {statistics.median(orc):.0f}  mean {statistics.mean(orc):.1f}  max {max(orc)}")
    # how often does neural beat / tie / lose to symbolic?
    win = sum(1 for s, n, _ in rows if n < s); tie = sum(1 for s, n, _ in rows if n == s)
    lose = sum(1 for s, n, _ in rows if n > s)
    print(f"  neural vs symbolic: win {win}  tie {tie}  lose {lose}  (of {len(rows)})")
    frac = (statistics.mean(sym) - statistics.mean(neu)) / max(statistics.mean(sym) - statistics.mean(orc), 1e-9)
    print(f"  neural captured {100*frac:.0f}% of the symbolic->oracle gap (100% = matches oracle)")


if __name__ == "__main__":
    main()
