"""ratchet_nack.py — FLAG-AS-NACK-ON-ACCEPTED, ratchet not re-roll (spec
registration, 2026-07-09, relay adjudication).

THE POLICY: a flagged accepted answer KEEPS its original by default; a NACK
round runs (specialist conditioned on the waist-anomaly slot); the revision
replaces the original ONLY IF IT STRICTLY DOMINATES. The asymmetric hazard it
guards: most flags land on CORRECT answers (precision ~0.4), and a naive
re-roll puts correct answers at risk to chase wrong ones on a population the
specialist was never trained to leave alone.

v0 SCOPE (registered): stage-0 accepts only — the 0.728 detector scores
parses in the PLATEAUED parser's space; later-stage accepts live in the
specialist's space where the centroids are uncalibrated (the per-stage score
table logged here decides whether v1 builds per-space centroids). Same
reasoning defers the anomaly-decrease dominance criterion; v0 DOMINANCE =
(i) revision is FORCED, (ii) revision's UNFLAGGED slots agree factor-exact
with the original (the copy-discipline vote). The break-rate kill criterion
polices whether that is tight enough.

REGISTERED (before firing):
  - BREAK RATE ~0 by construction; ANY break = the dominance test leaks ->
    tighten before deployment.
  - FIX RATE among flagged stage-0 wrongs > 0.346 (the general multi-round
    recovery rate) — relay's prediction, density regime stated: committed-wrong
    parses are near-misses by selection (they passed acceptance), the regime
    where per-round repair is strongest.
  - Deployment earned iff answered-precision rises without net answers
    falling. Floor: 1051/1277 = 0.823 precision, 70.1% end-to-end.
  - STAGE-SPLIT question (direction, not prediction): flagged wrongs
    concentrated in stage-0 = the 90 one-shot invisibles (the routing-wall
    population, first invisible to repair, now visible to the monitor).
  - PER-STAGE CALIBRATION column (free): if repaired-parse scores skew
    anomalous even when correct, the detector needs per-stage calibration
    before v1 trusts it beyond stage 0.
  Operating points: top-10% and top-20% flag rates, both reported; the same
  measurement grades recovery mode (ratcheted replacement) and precision mode
  (flag->abstain on undominated).

USAGE (needs .cache/deploy_audit_bigtest.npz):
  DEV=AMD ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest \
      .venv/bin/python3 scripts/ratchet_nack.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

from phase1_algebra_head import (  # noqa: E402
    T_ALG, L_FAC, build_params, forward, load_alg, decode, ALG_CKPT,
)
from phase1_algebra_nack import (  # noqa: E402
    N_FIELDS, NACK_CKPT, build_cond_params, forward_cond,
)
from waist_abstention_probe import compute_fst, np_heads, slot_kind  # noqa: E402
from repair_replace_swap import solve_forced  # noqa: E402
from survivor_multiplicity import midrank_auc  # noqa: E402


def slot_factor(o, j):
    """Slot-indexed factor tuple (decode()'s logic, one slot) or None."""
    if o["pres"][j] <= 0:
        return None
    res = int(o["res"][j].argmax())
    if o["islit"][j] > 0:
        digs = o["dig"][j].argmax(-1)
        from phase1_algebra_head import N_DIG
        val = int(sum(d * 10 ** (N_DIG - 1 - i) for i, d in enumerate(digs)))
        return ("given", res, val)
    args = tuple(sorted(int(a) for a in np.argsort(-o["args"][j])[:2]))
    op = "add" if o["op"][j].argmax() == 0 else "mul"
    return ("rel", op, args, res)


def main():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load

    aud = np.load(".cache/deploy_audit_bigtest.npz")
    idx = [int(i) for i in aud["idx"]]
    stage = {int(i): int(s) for i, s in zip(aud["idx"], aud["stage"])}
    correct = {int(i): int(c) for i, c in zip(aud["idx"], aud["correct"])}

    samples, states, tokmask, gold, sent = load_alg("test")
    p = build_params(0)
    sd = safe_load(ALG_CKPT)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    p_re = build_params(1)
    c_re = build_cond_params(1)
    sd2 = safe_load(NACK_CKPT)
    for d in (p_re, c_re):
        for k in d:
            d[k].assign(sd2[k].to(d[k].device).cast(d[k].dtype)).realize()
    hd = np_heads(p)

    # centroids from train (same recipe as the probe)
    tr_s, tr_states, tr_tok, _tg, tr_sent = load_alg("train")
    fst_tr = compute_fst(p, tr_states, tr_tok, tr_sent,
                         list(range(len(tr_s))))
    by_kind = {}
    for i in range(len(tr_s)):
        for j in range(L_FAC):
            kind = slot_kind(hd, fst_tr[i, j])
            if kind:
                by_kind.setdefault(kind, []).append(fst_tr[i, j])
    cent = {k: (lambda c: c / np.linalg.norm(c))(np.mean(v, axis=0))
            for k, v in by_kind.items()}

    # scores + worst slot per answered sample (plateaued-parser space)
    fst_te = compute_fst(p, states, tokmask, sent, idx)
    score, worst_slot = {}, {}
    for r, i in enumerate(idx):
        w, wj = 1.0, -1
        for j in range(L_FAC):
            kind = slot_kind(hd, fst_te[r, j])
            if kind is None or kind not in cent:
                continue
            v = fst_te[r, j]
            c = float((v / max(np.linalg.norm(v), 1e-9)) @ cent[kind])
            if c < w:
                w, wj = c, j
        score[i], worst_slot[i] = 1.0 - w, wj

    # per-stage calibration column (free)
    print(f"[ratchet] per-stage detector calibration (mean score):")
    print(f"  stage | n(correct) mean | n(wrong) mean | in-stage AUC")
    for st in sorted(set(stage.values())):
        sc = [score[i] for i in idx if stage[i] == st and correct[i]]
        sw = [score[i] for i in idx if stage[i] == st and not correct[i]]
        auc = (midrank_auc(np.array(sw), np.array(sc))
               if sc and sw else float("nan"))
        print(f"   {st}    | {len(sc):5d} {np.mean(sc) if sc else 0:.3f}   |"
              f" {len(sw):4d} {np.mean(sw) if sw else 0:.3f}  |  {auc:.3f}")

    # blank parses for stage-0 accepted (parse of record)
    blank = {}
    st0 = [i for i in idx if stage[i] == 0]
    for s0 in range(0, len(st0), 8):
        sl = np.array(st0[s0:s0 + 8])
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        out = forward(p, Tensor(states[sl_p].astype(np.float32),
                                dtype=dtypes.float),
                      Tensor(tokmask[sl_p].astype(np.float32),
                             dtype=dtypes.float),
                      Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))
        o = {k: out[k].realize().numpy() for k in
             ("pres", "ftype", "op", "islit", "dig", "args", "res", "query")}
        for bi, i in enumerate(sl):
            blank[int(i)] = {k: o[k][bi] for k in o}

    order = sorted(idx, key=lambda i: -score[i])
    for frac in (0.10, 0.20):
        k = int(len(idx) * frac)
        flagged = set(order[:k])
        fl_wrong = [i for i in flagged if not correct[i]]
        fl_by_stage = {}
        for i in flagged:
            if not correct[i]:
                fl_by_stage[stage[i]] = fl_by_stage.get(stage[i], 0) + 1
        print(f"\n=== FLAG top-{int(frac * 100)}% (n={k}) ===")
        print(f"  flagged wrongs by stage: {sorted(fl_by_stage.items())} "
              f"(stage-0 one-shot invisibles were 90 total)")
        target = [i for i in flagged if stage[i] == 0]
        print(f"  ratchet target (stage-0 flagged): {len(target)} "
              f"({sum(1 for i in target if not correct[i])} wrong)")

        # NACK round on targets, ratcheted replacement
        ffld = np.zeros((len(target), L_FAC, N_FIELDS), np.float32)
        for r, i in enumerate(target):
            if worst_slot[i] >= 0:
                ffld[r, worst_slot[i], :] = 1.0
        revs = {}
        for s0 in range(0, len(target), 8):
            sl = np.array(target[s0:s0 + 8])
            pad = 8 - len(sl)
            sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
            f3 = ffld[s0:s0 + len(sl)]
            f1 = np.zeros((len(sl), T_ALG), np.float32)
            f2 = np.ones((len(sl), 1), np.float32)
            if pad:
                f3 = np.concatenate([f3, f3[:1].repeat(pad, 0)])
                f1 = np.concatenate([f1, f1[:1].repeat(pad, 0)])
                f2 = np.concatenate([f2, f2[:1].repeat(pad, 0)])
            out = forward_cond(
                p_re, c_re,
                Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),
                Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
                Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int),
                Tensor(f1, dtype=dtypes.float), Tensor(f2, dtype=dtypes.float),
                Tensor(f3, dtype=dtypes.float))
            o = {kk: out[kk].realize().numpy() for kk in
                 ("pres", "ftype", "op", "islit", "dig", "args", "res",
                  "query")}
            for bi, i in enumerate(sl):
                revs[int(i)] = {kk: o[kk][bi] for kk in o}

        fixes = breaks = replaced = 0
        undominated = set()
        post_correct = dict(correct)   # correctness after the ratchet
        for i in target:
            ob, rv = blank[i], revs[i]
            facs, q_pred = decode(rv)
            a = solve_forced(facs, q_pred, samples[i])
            agree = all(slot_factor(ob, j) == slot_factor(rv, j)
                        for j in range(L_FAC) if j != worst_slot[i])
            if a is None or not agree:
                undominated.add(i)
                continue
            replaced += 1
            gold_ans = samples[i]["solution"][samples[i]["query_var"]]
            now_right = int(a == gold_ans)
            if now_right and not correct[i]:
                fixes += 1
            elif correct[i] and not now_right:
                breaks += 1
            post_correct[i] = now_right
        net = fixes - breaks
        n_wrong0 = sum(1 for i in target if not correct[i])
        print(f"  replaced {replaced} | undominated {len(undominated)} | "
              f"FIX {fixes}/{n_wrong0} wrong-targets | BREAK {breaks} "
              f"(kill bar: 0)")
        print(f"  RECOVERY MODE: end-to-end {1051 + net}/1500 = "
              f"{(1051 + net) / 1500:.3f} | precision "
              f"{(1051 + net) / 1277:.3f} (floor 0.823)")
        # precision mode: abstain on undominated stage-0 flags + all
        # later-stage flags (v0 cannot ratchet those)
        abstain = undominated | {i for i in flagged if stage[i] != 0}
        ans_pm = 1277 - len(abstain)
        corr_pm = sum(post_correct[i] for i in idx if i not in abstain)
        print(f"  PRECISION MODE (abstain {len(abstain)}): answered {ans_pm} "
              f"| precision {corr_pm / max(ans_pm, 1):.3f} | end-to-end "
              f"{corr_pm}/1500 = {corr_pm / 1500:.3f}")

    print(f"\n  REGISTERED BARS: break=0 (any break -> tighten); fix rate > "
          f"0.346 on flagged stage-0 wrongs; deploy iff precision rises "
          f"without net answers falling.")


if __name__ == "__main__":
    main()
