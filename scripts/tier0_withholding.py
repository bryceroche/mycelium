"""tier0_withholding.py — the withholding-cost curve (spec §12, pre-registered).

SEQUENCE PER REGISTRATION: (1) measure the incumbent's PER-FACTOR AUC on the banked
KenKen test failures (the conditional prediction's PRECONDITION: the k=1-2 peak claim
is conditional on AUC >= ~0.7); (2) the curve: withhold the k least-confident
predicted cages, solve the remainder, k in 0..5. COLUMNS: solve-to-GOLD (the
withhold-and-solve THIRD repair channel — no retransmission) and the taxonomy
composition (UNSAT -> multi conversions are DETECTION conversions).

REGISTERED PREDICTIONS: non-monotone with a peak at k=1-2 on KenKen (conditional);
monotone-down = confidence-gating net-harmful on dense domains (tier-0 = flagging
only). CPU-only: banked states + head forward + search-tier solves.

USAGE:  DEV=CPU .venv/bin/python3 scripts/tier0_withholding.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

from phase1_delta_head import (  # noqa: E402
    L_SLOTS, build_head_params, head_forward, load_split, decode_slots,
    _gold_grid_cache, CKPT_PATH, TYPES, H_WAIST,
)
from phase1_brick_a import NACK_NPZ, wrong_slot_mask  # noqa: E402
from phase1_brick_c import slot_confidence  # noqa: E402


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = np.empty(len(allv))
    r[np.argsort(allv)] = np.arange(len(allv))
    return (r[:len(pos)].mean() - (len(pos) - 1) / 2) / len(neg)


def main():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from mycelium.csp_domains import problem_from_kenken
    from mycelium.csp_core import solve_symbolic

    samples, states, tokmask, gold, sent = load_split("test")
    z = np.load(NACK_NPZ.format(split="test"))
    has_fail = z["has_fail"]
    p = build_head_params(0)
    sd = safe_load(CKPT_PATH)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    fail_i = [i for i in range(len(samples)) if has_fail[i]]
    print(f"[withholding] {len(fail_i)} banked KenKen test failures", flush=True)

    # forward once (CPU), collect per-slot outputs + confidences
    per = {}
    for s0 in range(0, len(fail_i), 8):
        sl = np.array(fail_i[s0:s0 + 8])
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        out = head_forward(
            p, Tensor(np.asarray(states[sl_p], dtype=np.float32), dtype=dtypes.float),
            Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
            Tensor(np.ones((1, 1, H_WAIST), np.float32), dtype=dtypes.float),
            Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))
        o = {k: out[k].realize().numpy() for k in ("pres", "type", "op", "dig", "mem")}
        for bi, i in enumerate(sl):
            per[int(i)] = {k: o[k][bi] for k in o}
        print(f"  fwd {min(s0+8, len(fail_i))}/{len(fail_i)}", flush=True)

    # (1) THE PRECONDITION: per-factor incumbent AUC (confidence: correct vs wrong)
    conf_ok, conf_bad = [], []
    for i in fail_i:
        o = per[i]
        wrong = wrong_slot_mask({k: v[None] for k, v in o.items()}, gold, i, 0)
        for j in range(L_SLOTS):
            if gold["presence"][i, j] < 0.5 or o["pres"][j] <= 0:
                continue
            c = slot_confidence(o, j)
            (conf_bad if wrong[j] else conf_ok).append(c)
    a = auc(conf_ok, conf_bad)
    print(f"\n[precondition] incumbent per-factor AUC (correct>wrong): {a:.3f}"
          f"  (the k=1-2 peak prediction is CONDITIONAL on >= ~0.7)")

    # (2) THE CURVE
    print(f"\n  k | solve-to-GOLD | UNSAT | multi/underdet | wrong-grid")
    for k_wh in range(6):
        n_gold = n_unsat = n_multi = n_wrongg = 0
        for i in fail_i:
            o = per[i]
            smp = samples[i]
            N = int(smp["N"])
            pred = decode_slots(o, N)
            cages, clues, confs = [], [], []
            pj = 0
            for j in range(L_SLOTS):
                if o["pres"][j] <= 0:
                    continue
                if TYPES[int(o["type"][j].argmax())] != "cage":
                    pj += 1
                    continue
                f = pred[pj]; pj += 1
                rc = [[m // 7, m % 7] for m in f["members_flat"]]
                if not rc or any(a_ >= N or b_ >= N for (a_, b_) in rc):
                    continue
                cages.append(rc); clues.append([f["op"], f["target"]])
                confs.append(slot_confidence(o, j))
            keep = np.argsort(confs)[k_wh:] if k_wh else np.arange(len(cages))
            c2 = [cages[x] for x in sorted(keep)]
            l2 = [clues[x] for x in sorted(keep)]
            try:
                res = solve_symbolic(problem_from_kenken(N, c2, l2),
                                     budget=100_000, seed=0)
            except Exception:
                n_unsat += 1
                continue
            if res["status"] != "solved":
                n_unsat += 1
                continue
            grid = [[int(res["assignment"][r * N + c]) for c in range(N)]
                    for r in range(N)]
            if grid == _gold_grid_cache(smp)[0]:
                n_gold += 1
                continue
            # solved to a non-gold grid: unique (silent-wrong) or multi (detectable)?
            multi = False
            for r in range(N):
                for c in range(N):
                    p2 = problem_from_kenken(N, c2, l2)
                    p2.domains0[r * N + c].discard(grid[r][c])
                    if not p2.domains0[r * N + c]:
                        continue
                    try:
                        r2 = solve_symbolic(p2, budget=50_000, seed=0)
                    except Exception:
                        continue
                    if r2["status"] == "solved":
                        multi = True
                        break
                if multi:
                    break
            if multi:
                n_multi += 1
            else:
                n_wrongg += 1
        print(f"  {k_wh} |      {n_gold:2d}       |  {n_unsat:3d}  |      {n_multi:3d}       |    {n_wrongg:2d}")
    print(f"\n  READS: solve-to-GOLD vs k = the withhold-and-solve repair channel;"
          f"\n  UNSAT->multi drift = detection conversions. Registered: peak at k=1-2"
          f"\n  (conditional on the AUC above); monotone-down = gating net-harmful.")


if __name__ == "__main__":
    main()
