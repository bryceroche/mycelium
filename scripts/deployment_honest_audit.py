"""deployment_honest_audit.py — THE GOLD-FREE ACCEPTANCE AUDIT (spec
registration, 2026-07-09 — the bug-class arm C exposed).

WHAT ARM C EXPOSED: every acceptance test in the measured stack compared the
forced answer against GOLD. A deployed stack cannot: it accepts ANY
forced-unique answer, at every stage. Three contaminations follow: (1) one-shot
forced-WRONG parses are ACCEPTED WRONG and never reach repair (70 sat hidden
among the 460 survivors alone); (2) measured "recoveries" whose original parse
was forced are PHANTOM (deployment never fires repair on an accepted answer);
(3) withhold/specialist acceptances can be forced-wrong imposters (constraint
removal weakens forcing).

THE AUDIT: replay the full deployed stack (one-shot -> withhold-2 -> 4
field-only specialist rounds) under GOLD-FREE acceptance: any forced answer at
any stage is final. Report per-stage accepted/correct/precision and the
deployment-honest end-to-end (abstentions = wrong).

REGISTERED (before firing):
  P1. One-shot: 797 forced-correct + F_wrong forced-wrong accepted, F_wrong in
      100-180 (70 known survivors + recovered-population forced originals).
  P2. Phantom recoveries exist: >0 of the measured 243 had forced originals.
  P3. Stage precisions DECLINE down the stack (each stage's accept pool is
      more damaged): one-shot ~0.85+, withhold lower, rounds lower still.
  P4. Deployment-honest end-to-end lands BELOW the measured 69.3% (the
      measured number silently credited gold-checked acceptance).
  This number is the MATH-500-relevant one; nothing is quoted without it again.

USAGE: DEV=AMD ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest \
           .venv/bin/python3 scripts/deployment_honest_audit.py
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
from repair_replace_swap import solve_forced  # noqa: E402


def main():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from tier0_incumbent import softmax, sig

    samples, states, tokmask, gold, sent = load_alg("test")
    p_plat = build_params(0)
    sd = safe_load(ALG_CKPT)
    for k in p_plat:
        p_plat[k].assign(sd[k].to(p_plat[k].device).cast(p_plat[k].dtype)).realize()
    p_re = build_params(1)
    c_re = build_cond_params(1)
    sd2 = safe_load(NACK_CKPT)
    for d in (p_re, c_re):
        for k in d:
            d[k].assign(sd2[k].to(d[k].device).cast(d[k].dtype)).realize()
    n = len(samples)

    def slot_conf(o, j):
        cc = float(sig(o["pres"][j]))
        cc *= float(softmax(o["ftype"][j][None])[0].max())
        cc *= float(softmax(o["res"][j][None])[0].max())
        if sig(o["islit"][j]) > 0.5:
            cc *= float(np.mean(softmax(o["dig"][j]).max(-1)))
        else:
            cc *= float(softmax(o["op"][j][None])[0].max())
            cc *= float(np.sort(softmax(o["args"][j][None])[0])[-2:].sum())
        return cc

    def withheld_solve(facs, q_pred, smp, o):
        """Gold-free withhold-2: forced answer after dropping the 2 least-
        confident slots, or None. Returns (answer, withheld-set)."""
        confs = []
        fi = 0
        for j in range(L_FAC):
            if o["pres"][j] <= 0:
                continue
            confs.append((fi, slot_conf(o, j)))
            fi += 1
        order = [x for x, _ in sorted(confs, key=lambda t: t[1])]
        wh = set(order[:2])
        kept = [f for x, f in enumerate(facs) if x not in wh]
        return solve_forced(kept, q_pred, smp), wh

    def run_cond(model_p, cond_c, idx, ffld):
        outs = {}
        for s0 in range(0, len(idx), 8):
            sl = np.array(idx[s0:s0 + 8])
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
                model_p, cond_c,
                Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),
                Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
                Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int),
                Tensor(f1, dtype=dtypes.float), Tensor(f2, dtype=dtypes.float),
                Tensor(f3, dtype=dtypes.float))
            o = {k: out[k].realize().numpy() for k in
                 ("pres", "ftype", "op", "islit", "dig", "args", "res", "query")}
            for bi, i in enumerate(sl):
                outs[int(i)] = {k: o[k][bi] for k in o}
        return outs

    def gold_ans(i):
        return samples[i]["solution"][samples[i]["query_var"]]

    stage_stats = {}
    outcome = {}   # i -> (stage_id, correct) ; absent = abstained

    def accept(stage, i, ans):
        st = stage_stats.setdefault(stage, [0, 0])
        st[0] += 1
        st[1] += int(ans == gold_ans(i))
        outcome[i] = (int(stage[0]), int(ans == gold_ans(i)))

    # STAGE 0: one-shot, gold-free — accept any forced answer
    c_zero = build_cond_params(9)
    blank = run_cond(p_plat, c_zero, list(range(n)),
                     np.zeros((n, L_FAC, N_FIELDS), np.float32))
    pool, flags_pool = [], []
    for i in range(n):
        facs, q_pred = decode(blank[i])
        a = solve_forced(facs, q_pred, samples[i])
        if a is not None:
            accept("0 one-shot", i, a)
            continue
        # STAGE 1: withhold-2, gold-free
        a, wh = withheld_solve(facs, q_pred, samples[i], blank[i])
        if a is not None:
            accept("1 withhold", i, a)
            continue
        ffld_i = np.zeros((L_FAC, N_FIELDS), np.float32)
        fi = 0
        for j in range(L_FAC):
            if blank[i]["pres"][j] <= 0:
                continue
            if fi in wh:
                ffld_i[j, :] = 1.0
            fi += 1
        pool.append(i)
        flags_pool.append(ffld_i)

    # STAGES 2..5: specialist rounds, gold-free acceptance
    for rnd in range(4):
        if not pool:
            break
        re = run_cond(p_re, c_re, pool, np.stack(flags_pool))
        nxt_pool, nxt_flags = [], []
        for i in pool:
            o = re[int(i)]
            facs, q_pred = decode(o)
            a = solve_forced(facs, q_pred, samples[int(i)])
            if a is not None:
                accept(f"{2 + rnd} round-{rnd + 1}", int(i), a)
                continue
            a, wh = withheld_solve(facs, q_pred, samples[int(i)], o)
            if a is not None:
                accept(f"{2 + rnd} round-{rnd + 1}", int(i), a)
                continue
            ffld_i = np.zeros((L_FAC, N_FIELDS), np.float32)
            fi = 0
            for j in range(L_FAC):
                if o["pres"][j] <= 0:
                    continue
                if fi in wh:
                    ffld_i[j, :] = 1.0
                fi += 1
            nxt_pool.append(i)
            nxt_flags.append(ffld_i)
        pool, flags_pool = nxt_pool, nxt_flags

    print(f"[audit] GOLD-FREE acceptance, full deployed stack (n={n})")
    print(f"  stage        | accepted | correct | precision")
    tot_a = tot_c = 0
    for stage in sorted(stage_stats):
        a, c = stage_stats[stage]
        tot_a += a
        tot_c += c
        print(f"  {stage:12s} |   {a:5d}  |  {c:5d}  |   {c / max(a, 1):.3f}")
    print(f"\n  answered {tot_a}/{n} (abstained {n - tot_a}) | "
          f"answered-precision {tot_c / max(tot_a, 1):.3f}")
    print(f"  DEPLOYMENT-HONEST END-TO-END: {tot_c}/{n} = {tot_c / n:.3f}"
          f"  (measured gold-checked headline was 1040/1500 = 0.693)")
    print(f"\n  REGISTERED: P1 one-shot forced-wrong 100-180; P2 phantom "
          f"recoveries >0; P3 precision declines down the stack; P4 honest "
          f"end-to-end < 0.693.")
    idx = np.array(sorted(outcome), np.int32)
    np.savez(".cache/deploy_audit_bigtest.npz", idx=idx,
             stage=np.array([outcome[i][0] for i in idx], np.int32),
             correct=np.array([outcome[i][1] for i in idx], np.int32))
    print(f"  [saved] .cache/deploy_audit_bigtest.npz (answered={len(idx)})")


if __name__ == "__main__":
    main()
