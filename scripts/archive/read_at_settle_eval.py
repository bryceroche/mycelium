"""read_at_settle_eval.py — The CHEAP control before the cathedral build.

The general-weights multi-task model shows a weak FORGETTING tail: per-domain
accuracy peaks at an INTERIOR breath then drifts down ~1% by the final breath B15
(coloring peak 0.571@B7 -> 0.563@B15; circuit 0.974@B5 -> 0.963@B15; kenken clean).
The CATHEDRAL (cross-breath memory pinning resolved intermediates) is the heavy fix.

The CHEAP fix tested HERE: instead of reading out at the FINAL breath, read each cell
at its OWN gold-free SETTLE-breath (the breath the convergence instrument already
identifies — argmin over k of the per-cell consecutive-breath belief JSD). If reading
at settle recovers the ~1% tail, the "forgetting" was merely reading one breath too
late -> NO cathedral needed. This control also makes any FUTURE cathedral result
attributable (cathedral value = improvement OVER read-at-settle, not over read-at-B15).

THREE READOUT POLICIES compared per domain on cell_acc (+ puzzle_acc / proper-rate):
  (A) READ-AT-FINAL (B15)           — the current default (the ~1% tail).
  (B) READ-AT-PER-CELL-SETTLE       — each cell at its OWN gold-free settle-breath.
                                      (deployable, gold-free — the cheap fix.)
  (C) READ-AT-BEST-GLOBAL-BREATH    — the single breath maximizing whole-test cell_acc
                                      per domain (the ORACLE ceiling; uses gold; NOT
                                      deployable, for reference only).

THE SETTLE-BREATH (gold-free). Per VALID cell c in a puzzle: belief_k(c) = softmax of
that cell's value-logits at breath k. The per-cell consecutive-breath JSD is
jsd_k(c) = JSD(belief_k(c), belief_{k-1}(c)) for k in [1, K-1]. The settle-breath is
k_min(c) = argmin_k jsd_k(c) (the breath where THIS cell's belief stops changing),
reported as the belief index k_min (0-based into the K-length history). This is exactly
the min-based instrument's breath_count_min logic from mycelium.kenken.convergence_
instrument, applied PER CELL instead of mean-over-cells-per-puzzle. GOLD-FREE: gold is
never used to pick the breath (only to SCORE the picked prediction). We REUSE
mycelium.kenken._jsd verbatim (imported, not modified).

NON-INVASIVE. NEVER edits mycelium/factor_graph_engine.py or mycelium/kenken.py. The
model build + per-domain adapter + gated inlet + per-domain head allocation are reused
VERBATIM from scripts/probe_svd_collapse_multitask.py (which itself reuses
scripts/factor_graph_train.py end-to-end), so training/eval match exactly.

THE EXACT VERIFIER (proper-rate), per domain, tensor/record-derived (gold-free except
the gold-match fallbacks noted):
  * coloring : EXACT — for every edge (membership row with 2 ones) the two endpoints'
               PREDICTED colors must differ (a complete proper coloring). NOT a
               gold-match (coloring has many proper solutions).
  * circuit  : EXACT == gold puzzle-acc. The circuit gold IS the unique deterministic
               topological forward-eval given the leaves, so "every gate equals its
               forward-eval" is equivalent to matching gold on every valid cell.
  * kenken   : EXACT — reshape the prediction to the N x N grid and run
               scripts.test_kenken_parity.verify_kenken_solution (cage_ok +
               row/col all-different) on the ORIGINAL records (recovered by mirroring
               KenKenLoader.iter_eval's in-order, last-batch-padded record stream).

SELFTEST (CPU, GPU-free): exercises the per-cell settle-breath picker on synthetic
per-breath belief sequences (a cell that settles at breath j is picked at j; a cell
whose belief flips after an interior peak is read at its pre-flip settle so the
interior-peak prediction is recovered), plus the three-policy aggregation on a tiny
synthetic logits history with a known interior peak.

USAGE:
  CPU selftest (GPU-free):
    SELFTEST_ONLY=1 .venv/bin/python3 scripts/read_at_settle_eval.py
  GPU run on the multi-task ckpt (AMD):
    DEV=AMD .venv/bin/python3 scripts/read_at_settle_eval.py
  One domain:
    DEV=AMD PROBE_ONLY=coloring .venv/bin/python3 scripts/read_at_settle_eval.py

Env vars:
  SELFTEST_ONLY  1 -> CPU selftest only (default 0).
  FG_CKPT        multi-task ckpt path
                 (default .cache/fg_ckpts/fg_multi_fair/fg_multi_fair_final.safetensors).
  PROBE_ONLY     coloring | circuit | kenken | "" (all) — default "".
  FG_MIX         model-build domain set (default coloring,circuit,kenken).
  EVAL_N         instances to score per domain (default 0 = WHOLE held-out test split).
  EVAL_BATCH / BATCH  eval batch size (default 8).
  K              breaths (default 16; the trained K_max).
  TAIL_NOISE     ~tail magnitude for the verdict band, in cell_acc points (default 0.003).
  FG_TRAIN / FG_TEST  kenken corpora (default .cache/kenken_{train,test}.jsonl).
"""
from __future__ import annotations

import ast
import os
import sys

_THIS_FILE = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
sys.path.insert(0, _ROOT)

import numpy as np


# ===========================================================================
# ast.parse gate — always runs, even on CPU
# ===========================================================================

def _ast_parse_ok() -> bool:
    with open(_THIS_FILE) as f:
        src = f.read()
    try:
        ast.parse(src)
        return True
    except SyntaxError as e:
        print(f"[ast.parse] FAILED: {e}", flush=True)
        return False


# ===========================================================================
# The gold-free per-cell settle-breath picker (the cheap fix's engine)
# ===========================================================================

def per_cell_settle_breath(beliefs: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-cell gold-free settle-breath index (the min-based instrument, per cell).

    Replicates mycelium.kenken.convergence_instrument's breath_count_min logic but
    PER CELL rather than mean-over-cells-per-puzzle. Uses mycelium.kenken._jsd verbatim.

    Parameters
    ----------
    beliefs : (K, S, N) float — per-breath softmax belief for every cell.
    valid   : (S,) bool       — valid-cell mask.

    Returns
    -------
    settle : (S,) int — for each cell, the belief-history index k_min in [0, K-1] at
             which that cell's consecutive-breath JSD is minimal (the settle breath).
             For K < 2 (degenerate) every cell settles at 0. Invalid cells -> K-1
             (never read; placeholder).
    """
    from mycelium.kenken import _jsd  # reuse verbatim; do NOT modify kenken.py

    K, S, _N = beliefs.shape
    settle = np.full((S,), K - 1, dtype=np.int64)
    if K < 2:
        return np.zeros((S,), dtype=np.int64)
    # consecutive-breath JSD per cell: jsd[k-1, c] = JSD(belief_k(c), belief_{k-1}(c)).
    jsd = np.empty((K - 1, S), dtype=np.float64)
    for k in range(1, K):
        jsd[k - 1] = _jsd(beliefs[k], beliefs[k - 1], axis=-1)   # (S,)
    # argmin transition index in [0, K-2] -> settle belief index is that transition's k.
    # transition t corresponds to JSD(belief_{t+1}, belief_t); the SETTLED belief is
    # belief_{t+1}, so the settle index is (argmin_t)+1, matching breath_count_min's
    # k_min = argmin+1 (the belief AT the settle breath = pred_by_k[k_min]).
    k_min = np.argmin(jsd, axis=0) + 1                            # (S,) in [1, K-1]
    settle = np.where(valid, k_min, K - 1).astype(np.int64)
    return settle


def softmax_lastaxis(logits: np.ndarray) -> np.ndarray:
    """Stable softmax over the last axis (matches convergence_instrument's belief calc)."""
    l = logits.astype(np.float64)
    l = l - l.max(axis=-1, keepdims=True)
    e = np.exp(l)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


# ===========================================================================
# SELFTEST (CPU) — the settle-picker + three-policy aggregation
# ===========================================================================

def selftest() -> bool:
    print("=== read_at_settle_eval SELFTEST (CPU) ===", flush=True)
    ok = True
    K, S, N = 6, 4, 3

    def onehot(v, sharp=0.90):
        p = np.full((N,), (1 - sharp) / (N - 1)); p[v] = sharp; return p

    # Build a synthetic per-breath belief tensor with KNOWN per-cell settle points. The
    # picker returns argmin_t JSD(belief_{t+1}, belief_t) + 1 (== breath_count_min). We
    # construct each cell so its consecutive-JSD has a UNIQUE strict minimum at a known
    # transition, so the settle index is deterministic.
    #   cell 0: a STRICTLY-SHRINKING JSD that bottoms out at transition 1 (settle idx 2),
    #           then RISES again (a tail-drift U) -> argmin = transition 1 -> settle 2.
    #           Resolved value 0 held through the plateau; tail drifts to value 1.
    #   cell 1: monotone settling — biggest moves early, frozen late. argmin is the first
    #           frozen transition -> a settle index in the frozen plateau; predicts 2.
    #   cell 2: interior peak then FLIP — value 0 for breaths 0..3, flips to value 1 at
    #           4..5. The flip (transition 3) is a LARGE JSD, the frozen plateau gives the
    #           argmin -> picker reads the PRE-FLIP resolved value 0, NOT the tail value 1.
    #   cell 3: invalid (never read).
    beliefs = np.zeros((K, S, N), dtype=np.float64)

    # cell 0: a clean U in consecutive-JSD with a unique min at transition 1 (settle 2).
    seq0 = [onehot(0, 0.40), onehot(0, 0.80), onehot(0, 0.83),   # shrinking moves -> min @ t1
            onehot(0, 0.70), onehot(1, 0.60), onehot(1, 0.90)]   # rising tail (drifts to 1)
    for k in range(K):
        beliefs[k, 0] = seq0[k]
    # cell 1: monotone settle — large early moves, frozen (zero-JSD) from breath 3 on.
    beliefs[0, 1] = onehot(0, 0.40); beliefs[1, 1] = onehot(1, 0.50)
    beliefs[2, 1] = onehot(2, 0.70)
    for k in range(3, K):
        beliefs[k, 1] = onehot(2, 0.95)        # frozen -> first zero transition @ t2 -> settle 3
    # cell 2: resolved value 0 for breaths 0..3 (frozen plateau), flips to value 1 at 4..5.
    for k in range(0, 4):
        beliefs[k, 2] = onehot(0, 0.95)
    beliefs[4, 2] = onehot(1, 0.95); beliefs[5, 2] = onehot(1, 0.95)
    # cell 3: invalid — fill arbitrary.
    for k in range(K):
        beliefs[k, 3] = onehot(2, 0.8)

    valid = np.array([True, True, True, False])
    settle = per_cell_settle_breath(beliefs, valid)
    print(f"  per-cell settle breaths = {settle.tolist()} "
          f"(expect cell0=2, cell1 in [1,5], cell2 in pre-flip plateau <=3)", flush=True)
    cond0 = settle[0] == 2                       # unique U-min at transition 1 -> settle 2
    cond1 = 1 <= settle[1] <= 5                  # in the settling/frozen region (predicts 2)
    cond2 = settle[2] <= 3                       # stable plateau, NOT the post-flip tail
    ok &= bool(cond0 and cond1 and cond2)

    # Value semantics: every cell's settle prediction is the RESOLVED value; cell 2's
    # settle-pred is value 0 (resolved) while read-at-final returns the tail value 1.
    pred_settle_vals = [int(beliefs[settle[c], c].argmax()) for c in range(S)]
    print(f"  settle-preds = {pred_settle_vals[:3]} (expect [0,2,0])", flush=True)
    ok &= (pred_settle_vals[0] == 0 and pred_settle_vals[1] == 2
           and pred_settle_vals[2] == 0)
    pred_final_c2 = int(beliefs[K - 1, 2].argmax())
    print(f"  cell2: settle-pred={pred_settle_vals[2]} (resolved=0)  "
          f"final-pred={pred_final_c2} (the forgotten tail=1)", flush=True)
    ok &= (pred_final_c2 == 1)

    # ---- three-policy aggregation on a synthetic logits history with an interior peak.
    # Two puzzles, S=3 valid cells each, K=4 breaths, gold = [0,1,2] (0-based -> +1).
    # Construct so that: read-at-final is WRONG on one cell (tail drift), read-at-settle
    # RECOVERS it, and read-at-best-global-breath (B=2) is the ceiling.
    Kp, Sp, Np = 4, 3, 3
    gold = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int32)  # 1-based values
    validp = np.ones((2, Sp), dtype=bool)
    # Belief curves per cell: correct at interior breaths 1-2, cell (0,2) drifts wrong at B3.
    bel = np.zeros((2, Kp, Sp, Np), dtype=np.float64)
    for b in range(2):
        for c in range(Sp):
            g = gold[b, c] - 1
            for k in range(Kp):
                bel[b, k, c] = onehot(g, 0.95)
    # puzzle 0, cell 2: stable-correct B0..B2, flips wrong at B3 (the tail).
    for k in range(0, 3):
        bel[0, k, 2] = onehot(2, 0.95)
    bel[0, 3, 2] = onehot(0, 0.95)   # wrong at final

    # Aggregate the three policies.
    res = _aggregate_policies_synthetic(bel, gold, validp, Kp)
    print(f"  synthetic A(final)={res['A']:.3f}  B(settle)={res['B']:.3f}  "
          f"C(oracle)={res['C']:.3f}", flush=True)
    # A misses cell (0,2): 5/6 correct. B + C recover it: 6/6.
    ok &= abs(res["A"] - 5.0 / 6.0) < 1e-9
    ok &= abs(res["B"] - 1.0) < 1e-9
    ok &= abs(res["C"] - 1.0) < 1e-9
    ok &= (res["B"] - res["A"]) > 0.0   # the cheap fix recovers the tail

    print(f"\n  SELFTEST {'PASSED' if ok else 'FAILED'}", flush=True)
    return ok


def _aggregate_policies_synthetic(bel, gold, valid, K):
    """CPU helper for the selftest: A/B/C cell_acc from a (B,K,S,N) belief tensor."""
    B, _K, S, _N = bel.shape
    a_corr = b_corr = 0.0
    n_cells = 0
    # global-breath cell_acc for each k (oracle picks the best).
    per_k_corr = np.zeros(K)
    for bi in range(B):
        v = valid[bi]
        g = gold[bi]
        settle = per_cell_settle_breath(bel[bi], v)
        pred_final = bel[bi, K - 1].argmax(axis=-1) + 1
        pred_settle = np.array([bel[bi, settle[c], c].argmax() + 1 for c in range(S)])
        a_corr += float(((pred_final == g) & v).sum())
        b_corr += float(((pred_settle == g) & v).sum())
        n_cells += int(v.sum())
        for k in range(K):
            pk = bel[bi, k].argmax(axis=-1) + 1
            per_k_corr[k] += float(((pk == g) & v).sum())
    return {"A": a_corr / n_cells, "B": b_corr / n_cells,
            "C": float(per_k_corr.max()) / n_cells}


# ===========================================================================
# Exact verifiers (proper-rate), per domain
# ===========================================================================

def _coloring_proper(pred: np.ndarray, membership: np.ndarray,
                     latent_type: np.ndarray, cell_valid: np.ndarray,
                     edge_gid: int) -> bool:
    """A coloring is proper iff every EDGE's two endpoints have different predicted
    colors. Edges = membership rows whose latent_type == the global coloring-edge id
    and which have exactly two 1s. pred is 1-based color (1..k). Gold-free."""
    L = membership.shape[0]
    for e in range(L):
        if int(latent_type[e]) != edge_gid:
            continue
        ones = np.nonzero(membership[e] > 0.5)[0]
        if ones.shape[0] != 2:
            continue
        u, v = int(ones[0]), int(ones[1])
        if not (cell_valid[u] > 0.5 and cell_valid[v] > 0.5):
            continue
        if int(pred[u]) == int(pred[v]):
            return False
    return True


def _kenken_proper(pred_row_major: list[int], rec: dict) -> bool:
    """Exact KenKen verifier on the ORIGINAL record (cages + clues), reusing
    scripts.test_kenken_parity.verify_kenken_solution. pred_row_major is length N*N
    (row-major, 1-based values)."""
    from scripts.test_kenken_parity import verify_kenken_solution
    n = int(rec["N"])
    cages = [[tuple(cell) for cell in cage] for cage in rec["cages"]]
    clues = [tuple(cl) for cl in rec["clues"]]
    return verify_kenken_solution(n, cages, clues, pred_row_major)


# ===========================================================================
# Per-domain GPU eval (A/B/C policies on the held-out test split)
# ===========================================================================

def _kenken_records_in_eval_order(eval_loader, eval_batch: int) -> list:
    """Recover the ORIGINAL KenKen records in the SAME order + last-batch padding that
    _KKWrap.iter_eval -> KenKenLoader.iter_eval yields, so each scored cell maps back to
    its puzzle's cages/clues for the exact verifier. Mirrors KenKenLoader.iter_eval."""
    base = eval_loader.loader            # the underlying KenKenLoader
    recs = base.records
    n = len(recs)
    ordered = []
    for start in range(0, n, eval_batch):
        batch = list(recs[start:start + eval_batch])
        while len(batch) < eval_batch:
            batch.append(recs[0])        # same padding KenKenLoader uses
        ordered.extend(batch)
    return ordered


def run_domain(domain: str, model, spec, adapter, eval_loader, K: int,
               EVAL_N: int, EVAL_BATCH: int) -> dict:
    """Run K breaths over `domain`'s held-out test split; compute A/B/C cell_acc +
    puzzle_acc + proper-rate. Non-invasive: only calls the engine forward + the
    convergence-instrument JSD logic (imported), never modifies the engine/oracle."""
    from tinygrad import Tensor
    from mycelium.factor_graph_engine import factor_breathing_forward
    from mycelium.factor_inlet import build_generic_factor_inlet, GLOBAL_TYPE_IDS

    print(f"\n{'#'*70}\n#  DOMAIN: {domain}\n{'#'*70}", flush=True)
    Tensor.training = False

    # KenKen needs the original records (cages/clues) for the exact verifier.
    kk_records = None
    kk_cursor = 0
    if domain == "kenken":
        kk_records = _kenken_records_in_eval_order(eval_loader, EVAL_BATCH)

    edge_gid = GLOBAL_TYPE_IDS["coloring_edge"]

    # ---- A/B/C accumulators ----
    a_cell = b_cell = 0.0
    n_cells = 0
    a_puz = b_puz = 0
    n_puz = 0
    # global-breath cell-acc per k (the oracle ceiling reads the argmax over k).
    per_k_cell = np.zeros(K, dtype=np.float64)
    # exact proper-rate counts (policy A and B; C oracle not run through the verifier —
    # the oracle ceiling is reported on cell_acc, its deployability caveat stands).
    a_proper = b_proper = 0
    n_proper = 0
    proper_kind = {"coloring": "exact-edge", "circuit": "exact==gold-puzzle",
                   "kenken": "exact-cage+alldiff"}[domain]

    done = 0
    n_batches = 0
    for native in eval_loader.iter_eval():
        fb = adapter(native)
        fb.factor_inlet = build_generic_factor_inlet(
            model, fb.membership, fb.latent_type, fb.cell_valid,
            op=fb.inlet_op, target=fb.inlet_target, size=fb.inlet_size).realize()

        logits_history, _ = factor_breathing_forward(model, fb, spec, K=K)
        assert len(logits_history) == K, f"expected K={K} logits, got {len(logits_history)}"

        # (K, B, S, N) belief tensor (numpy, float64) — softmax per the instrument.
        logits_np = np.stack([lh.realize().numpy() for lh in logits_history], axis=0)
        beliefs_all = softmax_lastaxis(logits_np)                 # (K, B, S, N)

        cell_valid_np = fb.cell_valid.realize().numpy()           # (B, S)
        gold_np = fb.gold.realize().numpy().astype(np.int32)      # (B, S)
        membership_np = fb.membership.realize().numpy()           # (B, L, S)
        latent_type_np = fb.latent_type.realize().numpy().astype(np.int32)  # (B, L)
        B_cur = cell_valid_np.shape[0]
        real = B_cur if EVAL_N <= 0 else min(B_cur, EVAL_N - done)

        # per-breath argmax (1-based) for every cell: (K, B, S).
        pred_by_k = beliefs_all.argmax(axis=-1).astype(np.int32) + 1

        for bi in range(real):
            valid = cell_valid_np[bi] > 0.5
            nv = int(valid.sum())
            # KenKen record (advance the mirror cursor for EVERY batch row, padded or not).
            rec = None
            if domain == "kenken":
                rec = kk_records[kk_cursor]
            if nv == 0:
                if domain == "kenken":
                    kk_cursor += 1
                continue
            gold_b = gold_np[bi]                                   # (S,)

            # ---- (A) read-at-final (B15) ----
            pred_final = pred_by_k[K - 1, bi]                      # (S,)
            eq_final = (pred_final == gold_b) & valid
            a_cell += float(eq_final.sum())

            # ---- (B) read-at-per-cell-settle (gold-free) ----
            settle = per_cell_settle_breath(beliefs_all[:, bi], valid)   # (S,)
            # each cell c's prediction taken at ITS settle breath: pred_by_k[settle[c], bi, c].
            S_cur = valid.shape[0]
            pred_settle = pred_by_k[settle, bi, np.arange(S_cur)].astype(np.int32)  # (S,)
            eq_settle = (pred_settle == gold_b) & valid
            b_cell += float(eq_settle.sum())

            n_cells += nv
            a_puz += int(np.all(pred_final[valid] == gold_b[valid]))
            b_puz += int(np.all(pred_settle[valid] == gold_b[valid]))
            n_puz += 1

            # ---- (C) per-breath global cell-acc (oracle picks argmax_k later) ----
            for k in range(K):
                per_k_cell[k] += float(((pred_by_k[k, bi] == gold_b) & valid).sum())

            # ---- proper-rate (exact verifier), policies A and B ----
            if domain == "coloring":
                a_ok = _coloring_proper(pred_final, membership_np[bi],
                                        latent_type_np[bi], cell_valid_np[bi], edge_gid)
                b_ok = _coloring_proper(pred_settle, membership_np[bi],
                                        latent_type_np[bi], cell_valid_np[bi], edge_gid)
                a_proper += int(a_ok); b_proper += int(b_ok); n_proper += 1
            elif domain == "circuit":
                # circuit gold IS the unique forward-eval -> proper == gold-puzzle.
                a_proper += int(np.all(pred_final[valid] == gold_b[valid]))
                b_proper += int(np.all(pred_settle[valid] == gold_b[valid]))
                n_proper += 1
            elif domain == "kenken":
                from mycelium.kenken_data import N_MAX
                n = int(rec["N"])
                grid_final = [int(pred_final[r * N_MAX + c])
                              for r in range(n) for c in range(n)]
                grid_settle = [int(pred_settle[r * N_MAX + c])
                               for r in range(n) for c in range(n)]
                a_proper += int(_kenken_proper(grid_final, rec))
                b_proper += int(_kenken_proper(grid_settle, rec))
                n_proper += 1
                kk_cursor += 1

            done += 1
            if EVAL_N > 0 and done >= EVAL_N:
                break
        n_batches += 1
        if EVAL_N > 0 and done >= EVAL_N:
            break

    a_acc = a_cell / max(n_cells, 1)
    b_acc = b_cell / max(n_cells, 1)
    per_k_acc = (per_k_cell / max(n_cells, 1))
    c_best_k = int(np.argmax(per_k_acc))
    c_acc = float(per_k_acc[c_best_k])

    return {
        "domain": domain, "K": K, "n_cells": n_cells, "n_puzzles": n_puz,
        "A_cell": a_acc, "B_cell": b_acc, "C_cell": c_acc, "C_best_k": c_best_k,
        "per_k_acc": per_k_acc.tolist(),
        "A_puz": a_puz / max(n_puz, 1), "B_puz": b_puz / max(n_puz, 1),
        "A_proper": a_proper / max(n_proper, 1), "B_proper": b_proper / max(n_proper, 1),
        "n_proper": n_proper, "proper_kind": proper_kind,
    }


# ===========================================================================
# Reporting + verdict
# ===========================================================================

def report_domain(res: dict, tail_noise: float) -> dict:
    d = res["domain"]; K = res["K"]
    A, Bv, C = res["A_cell"], res["B_cell"], res["C_cell"]
    recovery = Bv - A
    ceiling_gap = C - A
    print(f"\n{'='*72}\n  DOMAIN: {d}   (n_cells={res['n_cells']}, "
          f"n_puzzles={res['n_puzzles']})\n{'='*72}", flush=True)
    print(f"  per-breath GLOBAL cell_acc (B0..B{K-1}):", flush=True)
    pk = res["per_k_acc"]
    for r0 in range(0, K, 8):
        chunk = pk[r0:r0 + 8]
        print("      " + "  ".join(f"B{r0+i}={chunk[i]:.4f}" for i in range(len(chunk))),
              flush=True)
    print(f"\n  (A) READ-AT-FINAL  B{K-1}        cell_acc = {A:.4f}   "
          f"puzzle_acc = {res['A_puz']:.4f}   proper({res['proper_kind']}) = "
          f"{res['A_proper']:.4f}", flush=True)
    print(f"  (B) READ-AT-SETTLE (per-cell) cell_acc = {Bv:.4f}   "
          f"puzzle_acc = {res['B_puz']:.4f}   proper({res['proper_kind']}) = "
          f"{res['B_proper']:.4f}", flush=True)
    print(f"  (C) READ-AT-BEST-BREATH B{res['C_best_k']} (oracle) cell_acc = {C:.4f}   "
          f"[ceiling; uses gold; NOT deployable]", flush=True)
    print(f"\n  (B - A) recovery = {recovery:+.4f}   "
          f"(C - A) ceiling gap = {ceiling_gap:+.4f}   "
          f"tail-noise band = +/-{tail_noise:.4f}", flush=True)

    # Per-domain verdict.
    has_tail = ceiling_gap > tail_noise           # there IS an interior-peak tail to recover
    if not has_tail:
        verdict = ("NO MATERIAL TAIL (oracle ceiling within noise of read-at-final -> "
                   "nothing to recover here; read-at-final is already at the interior peak)")
        fires = False
    elif recovery >= ceiling_gap - tail_noise:
        verdict = (f"CHEAP FIX WORKS (read-at-settle recovers {recovery:+.4f} of the "
                   f"{ceiling_gap:+.4f} ceiling gap -> approaches the oracle; the "
                   f"'forgetting' was readout-timing, NOT lost representation)")
        fires = False
    elif recovery > tail_noise:
        verdict = (f"PARTIAL RECOVERY (read-at-settle recovers {recovery:+.4f} of the "
                   f"{ceiling_gap:+.4f} ceiling gap -> some timing, some representational; "
                   f"cathedral may add the remainder)")
        fires = True
    else:
        verdict = (f"CATHEDRAL JUSTIFIED (read-at-settle recovers only {recovery:+.4f} "
                   f"of the {ceiling_gap:+.4f} ceiling gap -> the resolved belief is "
                   f"genuinely lost in the residual by the settle-breath; degradation is "
                   f"REPRESENTATIONAL, not read-late)")
        fires = True
    print(f"  VERDICT [{d}]: {verdict}", flush=True)
    return {"recovery": recovery, "ceiling_gap": ceiling_gap, "verdict": verdict,
            "cathedral_fires": fires, "has_tail": has_tail}


# ===========================================================================
# Main
# ===========================================================================

def main():
    if int(os.environ.get("SELFTEST_ONLY", "0")) > 0:
        sys.exit(0 if selftest() else 1)

    print("Running CPU selftest before GPU eval...", flush=True)
    if not selftest():
        print("SELFTEST FAILED — aborting GPU eval.", flush=True)
        sys.exit(1)

    from tinygrad import Tensor

    K = int(os.environ.get("K", os.environ.get("FG_K_MAX", "16")))
    BATCH = int(os.environ.get("BATCH", "8"))
    EVAL_BATCH = int(os.environ.get("EVAL_BATCH", str(BATCH)))
    EVAL_N = int(os.environ.get("EVAL_N", "0"))      # 0 = whole held-out split
    SEED = int(os.environ.get("SEED", "42"))
    TAIL_NOISE = float(os.environ.get("TAIL_NOISE", "0.003"))
    CKPT = os.environ.get(
        "FG_CKPT", ".cache/fg_ckpts/fg_multi_fair/fg_multi_fair_final.safetensors")
    FG_TRAIN = os.environ.get("FG_TRAIN", ".cache/kenken_train.jsonl")
    FG_TEST = os.environ.get("FG_TEST", ".cache/kenken_test.jsonl")
    MIX = [m.strip().lower() for m in
           os.environ.get("FG_MIX", "coloring,circuit,kenken").split(",") if m.strip()]
    only = os.environ.get("PROBE_ONLY", "").strip().lower()
    probe_domains = [only] if only else list(MIX)
    for d in probe_domains:
        assert d in MIX, f"PROBE_ONLY={d} not in FG_MIX={MIX}"

    if not os.path.exists(CKPT):
        print(f"[abort] ckpt not found: {CKPT}", flush=True)
        sys.exit(1)

    # Reuse the multi-task model build VERBATIM from probe_svd_collapse_multitask.
    import scripts.probe_svd_collapse_multitask as probe
    import scripts.factor_graph_train as fgt

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    (model, spec, adapters, eval_loaders, n_heads, head_dim, cfg, task) = \
        probe._build_multitask_model(K, MIX, FG_TRAIN, FG_TEST, BATCH, EVAL_BATCH, SEED)

    print(f"\nloading multi-task checkpoint: {CKPT}", flush=True)
    fgt.load_ckpt(model, CKPT)
    Tensor.training = False

    all_res = {}
    all_verd = {}
    for d in probe_domains:
        res = run_domain(d, model, spec, adapters[d], eval_loaders[d], K,
                         EVAL_N, EVAL_BATCH)
        all_res[d] = res
        all_verd[d] = report_domain(res, TAIL_NOISE)

    # ---- overall summary ----
    print(f"\n{'='*72}\n  OVERALL SUMMARY — read-at-final (A) vs read-at-settle (B) vs "
          f"oracle (C)\n{'='*72}", flush=True)
    print(f"  {'domain':<10} {'n_cells':>8} {'A_cell':>8} {'B_cell':>8} {'C_cell':>8} "
          f"{'B-A':>8} {'C-A':>8}  verdict", flush=True)
    any_fires = False
    tot_a = tot_b = tot_cells = 0.0
    for d in probe_domains:
        r = all_res[d]; v = all_verd[d]
        tag = ("CATHEDRAL" if v["cathedral_fires"]
               else ("CHEAP-FIX" if v["has_tail"] else "NO-TAIL"))
        print(f"  {d:<10} {r['n_cells']:>8} {r['A_cell']:>8.4f} {r['B_cell']:>8.4f} "
              f"{r['C_cell']:>8.4f} {v['recovery']:>+8.4f} {v['ceiling_gap']:>+8.4f}  {tag}",
              flush=True)
        any_fires = any_fires or v["cathedral_fires"]
        tot_a += r["A_cell"] * r["n_cells"]; tot_b += r["B_cell"] * r["n_cells"]
        tot_cells += r["n_cells"]
    micro_a = tot_a / max(tot_cells, 1); micro_b = tot_b / max(tot_cells, 1)
    print(f"\n  MICRO-AVG (cell-weighted): A={micro_a:.4f}  B={micro_b:.4f}  "
          f"(B-A)={micro_b - micro_a:+.4f}", flush=True)
    print(f"\n  OVERALL VERDICT: "
          + ("CATHEDRAL JUSTIFIED on >=1 domain (read-at-settle does NOT recover the "
             "tail there -> the degradation is representational)."
             if any_fires else
             "CHEAP FIX WORKS / NO MATERIAL TAIL across domains (read-at-settle recovers "
             "the tail or there is no tail -> NO cathedral needed; switch the default "
             "readout to per-cell settle-breath)."), flush=True)
    print(flush=True)


if __name__ == "__main__":
    ok = _ast_parse_ok()
    print(f"[ast.parse] astparse_ok={ok}", flush=True)
    if not ok:
        sys.exit(1)
    main()
