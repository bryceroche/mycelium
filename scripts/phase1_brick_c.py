"""phase1_brick_c.py — BRICK-C v0: the REAL NACK replaces the oracle.

THE QUESTION: Brick-A proved the loop's mechanism with ORACLE localization (gold-
derived wrong-sentence flags -> fix 0.438 vs shuffled floor 0.360). Brick-C swaps in
the deployable instrument and asks how much of the oracle's localization increment
survives. PRE-REGISTERED RELATIVE KILL (spec §9 post-Brick-A registration #3):

    retention = (fix_addback - fix_shuffled) / (fix_oracle - fix_shuffled)  >= 0.5

THE INSTRUMENT (gold-free end to end):
  1. ADD-BACK SWEEP (symbolic): start from the skeleton (rows/cols implicit in
     problem_from_kenken; NO predicted cages), add the parser's predicted cages one
     at a time in CONFIDENCE order (most-confident first — the trusted graph grows
     as large as possible before the first break); an addition that turns the graph
     UNSAT is BLAMED and left out; the sweep continues (multi-error capable, unlike
     delete-one). O(F) search-tier calls.
  2. SLOT -> SENTENCE, gold-free: the slot's supervised cross-attention argmax over
     sentence indices (the model points at its own source sentence — the span
     supervision made the attention BE the map).
  3. Blamed sentences -> the SAME flag vector + head-level conditioned re-parse
     (the Brick-A flag-dependent checkpoint, unchanged).

ARMS (same model, same graph, same protocol): blank | oracle flags (ceiling
condition) | ADD-BACK flags (the measurement) | shuffled (floor). Read with fix
rate, the flip-rate instrument, solves, and the retention verdict.

USAGE:
  Selftest:      .venv/bin/python3 scripts/phase1_brick_c.py --selftest
  Measurement:   DEV=AMD .venv/bin/python3 scripts/phase1_brick_c.py --eval
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

_THIS_FILE = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(_THIS_FILE))

import numpy as np

from phase1_delta_head import (  # noqa: E402
    T_WINDOW, H_TRUNK, H_WAIST, L_SLOTS, S_CELLS, N_DIGITS, SENT_MAX,
    build_head_params, load_split, decode_slots, _solve_rate_one, OPS, TYPES,
)
from phase1_brick_a import (  # noqa: E402
    COND_DIM, NACK_NPZ, BRICK_A_CKPT, build_encoder_params, head_forward_cond,
    flags_to_token_mask, wrong_slot_mask, shuffle_flags,
)


# ===========================================================================
# THE ADD-BACK SWEEP (symbolic, gold-free, multi-error capable)
# ===========================================================================

def addback_blame(N: int, cages: list, clues: list, conf: list,
                  budget: int = 60_000) -> set:
    """Add predicted cages most-confident-first; blame additions that turn UNSAT.
    Returns the set of blamed indices (into cages/clues). Gold-free."""
    from mycelium.csp_domains import problem_from_kenken
    from mycelium.csp_core import solve_symbolic

    order = sorted(range(len(cages)), key=lambda j: -conf[j])
    accepted, blamed = [], set()
    for j in order:
        trial = accepted + [j]
        try:
            res = solve_symbolic(
                problem_from_kenken(N, [cages[k] for k in trial],
                                    [clues[k] for k in trial]),
                budget=budget, seed=0)
            ok = res["status"] == "solved"
        except Exception:
            ok = False
        if ok:
            accepted.append(j)
        else:
            blamed.add(j)
    return blamed


def slot_confidence(o: dict, j: int) -> float:
    """Gold-free per-slot confidence: presence sigmoid x mean decisive-prob."""
    def sm(x):
        e = np.exp(x - x.max())
        return e / e.sum()
    c = 1.0 / (1.0 + math.exp(-float(o["pres"][j])))
    c *= float(sm(o["op"][j]).max())
    c *= float(np.mean([sm(o["dig"][j][d]).max() for d in range(N_DIGITS)]))
    mem_p = 1.0 / (1.0 + np.exp(-o["mem"][j]))
    c *= float(np.mean(np.maximum(mem_p, 1 - mem_p)))
    return c


def slots_to_flags_via_attn(blamed_slots, attn_mean, sent_row) -> np.ndarray:
    """Gold-free slot->sentence: the slot's attention argmax over sentence ids."""
    flags = np.zeros((COND_DIM,), np.float32)
    for j in blamed_slots:
        tok = int(attn_mean[j].argmax())
        flags[int(sent_row[tok])] = 1.0
    if blamed_slots:
        flags[-1] = 1.0
    return flags


# ===========================================================================
# THE MEASUREMENT
# ===========================================================================

def do_eval(seed: int = 0) -> None:
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load

    samples, states, tokmask, gold, sent = load_split("test")
    zt = np.load(NACK_NPZ.format(split="test"))
    flags_oracle, has_fail = zt["flags"], zt["has_fail"]
    n = len(samples)
    p = build_head_params(0)
    enc = build_encoder_params(1)
    sd = safe_load(BRICK_A_CKPT)
    for d in (p, enc):
        for k in d:
            d[k].assign(sd[k].to(d[k].device).cast(d[k].dtype)).realize()
    rng = np.random.RandomState(seed)

    def run_pass(idx, cond_np):
        outs = {}
        for s0 in range(0, len(idx), 8):
            sl = idx[s0:s0 + 8]
            pad = 8 - len(sl)
            sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
            cond_b = cond_np[s0:s0 + 8]
            if pad:
                cond_b = np.concatenate([cond_b, cond_b[:1].repeat(pad, 0)])
            out = head_forward_cond(
                p, enc,
                Tensor(np.asarray(states[sl_p], dtype=np.float32), dtype=dtypes.float),
                Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
                Tensor(np.ones((1, 1, H_WAIST), np.float32), dtype=dtypes.float),
                Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int),
                Tensor(flags_to_token_mask(cond_b, sent[sl_p]), dtype=dtypes.float),
                Tensor(cond_b[:, -1:].astype(np.float32), dtype=dtypes.float))
            o = {k: out[k].realize().numpy() for k in
                 ("pres", "type", "op", "dig", "mem", "attn_mean")}
            for bi, i in enumerate(sl):
                outs[int(i)] = {k: o[k][bi] for k in o}
        return outs

    fail_i = np.array([i for i in range(n) if has_fail[i]])
    blank = run_pass(fail_i, np.zeros((len(fail_i), COND_DIM), np.float32))

    # THE INSTRUMENT: per failure, decode blank parse -> add-back sweep -> flags
    flags_addback = np.zeros((len(fail_i), COND_DIM), np.float32)
    n_blamed_tot = 0
    for r, i in enumerate(fail_i):
        i = int(i)
        o = blank[i]
        N = int(samples[i]["N"])
        pred = decode_slots(o, N)
        cages, clues, conf, slot_of = [], [], [], []
        pj = 0
        for j in range(L_SLOTS):
            if o["pres"][j] <= 0.0:
                continue
            if TYPES[int(o["type"][j].argmax())] != "cage":
                pj += 1
                continue
            f = pred[pj]; pj += 1
            rc = [[m // 7, m % 7] for m in f["members_flat"]]
            if not rc or any(a >= N or b >= N for (a, b) in rc):
                continue          # malformed: blamed unconditionally below
            cages.append(rc); clues.append([f["op"], f["target"]])
            conf.append(slot_confidence(o, j)); slot_of.append(j)
        blamed = addback_blame(N, cages, clues, conf)
        blamed_slots = [slot_of[k] for k in blamed]
        n_blamed_tot += len(blamed_slots)
        flags_addback[r] = slots_to_flags_via_attn(blamed_slots, o["attn_mean"], sent[i])

    true_p = run_pass(fail_i, flags_oracle[fail_i])
    addb = run_pass(fail_i, flags_addback)
    shuf = run_pass(fail_i, np.stack([shuffle_flags(f, rng)
                                      for f in flags_oracle[fail_i]]))

    stats = {k: [0, 0] for k in ("fix_o", "fix_a", "fix_s")}
    solves = {"blank": 0, "oracle": 0, "addback": 0, "shuf": 0}
    for r, i in enumerate(fail_i):
        i = int(i)
        wrong_b = wrong_slot_mask({k: v[None] for k, v in blank[i].items()}, gold, i, 0)
        flagged = np.where(wrong_b)[0]
        for key, arm in (("fix_o", true_p), ("fix_a", addb), ("fix_s", shuf)):
            wr = wrong_slot_mask({k: v[None] for k, v in arm[i].items()}, gold, i, 0)
            stats[key][0] += int((~wr[flagged]).sum()); stats[key][1] += len(flagged)
        smp = samples[i]
        for key, arm in (("blank", blank), ("oracle", true_p),
                         ("addback", addb), ("shuf", shuf)):
            solves[key] += _solve_rate_one({k: v[None] for k, v in arm[i].items()}, 0, smp)

    fo = stats["fix_o"][0] / max(stats["fix_o"][1], 1)
    fa = stats["fix_a"][0] / max(stats["fix_a"][1], 1)
    fs = stats["fix_s"][0] / max(stats["fix_s"][1], 1)
    retention = (fa - fs) / max(fo - fs, 1e-9)
    print(f"\n[brick-C] organic test failures: {len(fail_i)}; "
          f"add-back blamed {n_blamed_tot} predicted cages total (gold-free)")
    print(f"  flagged-slot FIX: oracle {fo:.3f} | ADD-BACK {fa:.3f} | shuffled {fs:.3f}")
    print(f"  SOLVES: blank {solves['blank']} | oracle {solves['oracle']} | "
          f"ADD-BACK {solves['addback']} | shuffled {solves['shuf']}")
    print(f"  RETENTION of the oracle's localization increment: "
          f"({fa:.3f}-{fs:.3f})/({fo:.3f}-{fs:.3f}) = {retention:.2f}")
    print(f"  PRE-REGISTERED KILL: retention >= 0.5 -> Brick-C v0 LIVES; else the"
          f" instrument (ordering/attn-map/uniqueness-probe) needs work before the"
          f" loop closes gold-free.")
    print(f"  VERDICT: {'LIVES' if retention >= 0.5 else 'KILLED at v0'}")


# ===========================================================================
# SELFTEST (CPU, no GPU: sweep + flag mapping on synthetic parses)
# ===========================================================================

def selftest() -> None:
    # 4x4 puzzle, correct given-cages + ONE wrong cage: sweep must blame exactly it.
    N = 4
    sol = [[1, 2, 3, 4], [2, 1, 4, 3], [3, 4, 1, 2], [4, 3, 2, 1]]
    cages = [[[0, 0]], [[1, 1]], [[0, 1], [0, 2]], [[2, 2], [2, 3]]]
    clues = [["given", 1], ["given", 1], ["add", 5], ["add", 99]]   # last is WRONG
    conf = [0.9, 0.9, 0.8, 0.7]
    blamed = addback_blame(N, cages, clues, conf)
    assert blamed == {3}, blamed
    # all-correct parse: nothing blamed
    clues2 = [["given", 1], ["given", 1], ["add", 5], ["add", 3]]
    assert addback_blame(N, cages, clues2, conf) == set()
    # slot->sentence via attention argmax
    attn = np.zeros((L_SLOTS, T_WINDOW), np.float32); attn[7, 100] = 1.0
    sent_row = np.zeros((T_WINDOW,), np.int8); sent_row[100] = 9
    f = slots_to_flags_via_attn([7], attn, sent_row)
    assert f[9] == 1.0 and f[-1] == 1.0 and f[:9].sum() == 0
    print("[selftest] PASS (sweep blames exactly the wrong cage; attn->sentence map)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
    elif args.eval:
        do_eval()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
