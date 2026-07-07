"""survivor_oracle_ceiling.py — THE ORACLE-FLAG CEILING on the 460 survivors
(spec registration, 2026-07-08 — the encode-side/decode-side partition test).

THE QUESTION: the suspicion-rank probe showed localization is adequate and the
anatomy chain (given-value misbindings, one-directional, swaps marginal) points
at an ENCODE-SIDE wall — every repair round re-decodes the SAME frozen trunk
states, so information mis-committed at encoding time is unrecoverable by any
head-side conditioning. This arm measures the ceiling directly: hand the
specialist PERFECT flags (gold-derived `wrong_fields` masks — which is exactly
its TRAINING regime; deployment's withhold-derived all-fields flags were doubly
OOD) and re-derive them each round. This upper-bounds EVERY possible
flag-quality improvement (tier-0, transplant, all rankers) in one number.

REGISTERED (before measuring): recovery of survivors under oracle flags
  < 10%  -> ENCODE-SIDE CONFIRMED as a measured ceiling: flag quality is
            irrelevant; the frontier is changing the ENCODING (second-view
            re-render with position-aligned suspect marks / deeper prefix),
            not any suspicion or repair improvement.
  > 30%  -> flag QUALITY (incl. the field-pattern mismatch) was the constraint
            after all; P1's flat verdict misled; fix the deployed flag deriver.
  10-30% -> mixed; partition the recovered-by-oracle and re-profile.

Identity from .cache/survivor_profile_bigtest.npz; everything else identical to
the deployed stack (same solve_check incl. withhold-2, same 4 rounds, field-only
arm: span channel zero, fbit ones).

USAGE: DEV=AMD ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest \
           .venv/bin/python3 scripts/survivor_oracle_ceiling.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

from phase1_algebra_head import (  # noqa: E402
    T_ALG, L_FAC, build_params, load_alg, decode, ALG_CKPT,
)
from phase1_algebra_nack import (  # noqa: E402
    N_FIELDS, NACK_CKPT, build_cond_params, forward_cond,
)


def wf_single(o, g, i):
    """Per-sample oracle wrong-mask — phase1_algebra_nack.wrong_fields, unbatched."""
    wf = np.zeros((L_FAC, N_FIELDS), np.float32)
    for j in range(L_FAC):
        pg = g["presence"][i, j] > 0.5
        pp = o["pres"][j] > 0
        if pg != pp:
            wf[j, 0] = 1.0
            continue
        if not pg:
            continue
        wf[j, 1] = float(int(o["ftype"][j].argmax()) != g["ftype"][i, j])
        is_rel = g["ftype"][i, j] == 0
        if is_rel:
            wf[j, 2] = float(int(o["op"][j].argmax()) != g["op"][i, j])
            top2 = set(np.argsort(-o["args"][j])[:2].tolist())
            wf[j, 3] = float(top2 != set(np.where(g["args"][i, j] > 0.5)[0]
                                         .tolist()))
        wf[j, 4] = float(int(o["res"][j].argmax()) != g["res"][i, j])
        if not is_rel:
            wf[j, 5] = float(not bool(
                (o["dig"][j].argmax(-1) == g["digits"][i, j]).all()))
    return wf


def main():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from mycelium.csp_domains import problem_from_algebra
    from mycelium.csp_core import solve_symbolic
    from tier0_incumbent import softmax, sig

    prof = np.load(".cache/survivor_profile_bigtest.npz")
    survivors = sorted(int(i) for i, s in zip(prof["idx"], prof["status"])
                       if s == 2)
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

    def solve_check(facs, q_pred, smp, k_wh, o=None):
        if o is not None and k_wh:
            confs = []
            fi = 0
            for j in range(L_FAC):
                if o["pres"][j] <= 0:
                    continue
                confs.append((fi, slot_conf(o, j)))
                fi += 1
            order = [x for x, _ in sorted(confs, key=lambda t: t[1])]
            wh = set(order[:k_wh])
            facs = [f for x, f in enumerate(facs) if x not in wh]
        rels = [(f["op"], f["args"][0], f["args"][1], f["result"])
                for f in facs if f["ftype"] == "rel"]
        gv = {f["var"]: f["value"] for f in facs if f["ftype"] == "given"}
        gold_ans = smp["solution"][smp["query_var"]]
        try:
            nv = max([smp["n_vars"]] + [v + 1 for f in facs for v in
                     ((list(f["args"]) + [f["result"]]) if f["ftype"] == "rel"
                      else [f["var"]])] + [q_pred + 1])
            res = solve_symbolic(problem_from_algebra(nv, rels, gv, smp["m"]),
                                 budget=200_000, seed=0)
            if res["status"] != "solved":
                return False
            sol = [int(res["assignment"][v]) for v in range(nv)]
            if not (q_pred < len(sol) and sol[q_pred] == gold_ans):
                return False
            p2 = problem_from_algebra(nv, rels, gv, smp["m"])
            p2.domains0[q_pred].discard(sol[q_pred])
            if p2.domains0[q_pred]:
                r2 = solve_symbolic(p2, budget=100_000, seed=0)
                if r2["status"] == "solved":
                    return False
            return True
        except Exception:
            return False

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

    # round-1 oracle flags come from the plateaued parser's blank parse
    c_zero = build_cond_params(9)
    blank = run_cond(p_plat, c_zero, survivors,
                     np.zeros((len(survivors), L_FAC, N_FIELDS), np.float32))
    pool = list(survivors)
    flags = [wf_single(blank[i], gold, i) for i in pool]
    n_flagged0 = float(np.mean([f.any(axis=1).sum() for f in flags]))
    print(f"[oracle] survivors={len(pool)} | mean oracle-flagged slots "
          f"round-1 = {n_flagged0:.2f}")

    rec_idx = []
    for rnd in range(4):
        if not pool:
            break
        re = run_cond(p_re, c_re, pool, np.stack(flags))
        nxt_pool, nxt_flags = [], []
        for i in pool:
            o = re[int(i)]
            facs, q_pred = decode(o)
            if solve_check(facs, q_pred, samples[int(i)], 2, o=o):
                rec_idx.append(int(i))
                continue
            nxt_pool.append(i)
            nxt_flags.append(wf_single(o, gold, int(i)))
        print(f"  ROUND {rnd + 1}: {len(pool) - len(nxt_pool)}/{len(pool)} "
              f"recovered")
        pool, flags = nxt_pool, nxt_flags

    rate = len(rec_idx) / len(survivors)
    print(f"\n[oracle] CEILING: {len(rec_idx)}/{len(survivors)} = {rate:.3f}")
    print(f"  REGISTERED: <10% -> ENCODE-SIDE wall confirmed (flag quality "
          f"irrelevant; frontier = change the encoding). >30% -> deployed flag "
          f"deriver was the constraint. 10-30% -> partition + re-profile.")
    np.savez(".cache/oracle_recovered_bigtest.npz",
             recovered=np.array(sorted(rec_idx), np.int32))

    # ---- OPTION-4 RE-PROFILE (the pinned 10-30% follow-up): the oracle-64 vs
    # the 396 — teeth, multiplicity, error-kind mix from the saved profile.
    from characterize_survivors import sample_teeth
    kinds_names = [str(k) for k in prof["kinds"]]
    by_idx = {int(i): (int(m), prof["counts"][r])
              for r, (i, m) in enumerate(zip(prof["idx"], prof["mult"]))}
    rec_set = set(rec_idx)
    hard = [i for i in survivors if i not in rec_set]

    def prof_group(idx_list):
        t = [sample_teeth(samples[i]) for i in idx_list]
        out = {k: float(np.mean([x[k] for x in t]))
               for k in ("oblique", "shuffled", "irrelevant", "n_vars", "band")}
        out["mult"] = float(np.mean([by_idx[i][0] for i in idx_list]))
        tot = np.sum([by_idx[i][1] for i in idx_list], axis=0).astype(float)
        tot /= max(tot.sum(), 1.0)
        for kn, v in zip(kinds_names, tot):
            out[f"kind_{kn}"] = float(v)
        return out

    po, ph = prof_group(rec_idx), prof_group(hard)
    print(f"\n  OPTION-4 RE-PROFILE: oracle-recovered (n={len(rec_idx)}) vs "
          f"hard remainder (n={len(hard)})")
    for k in sorted(po):
        enr = po[k] / max(ph[k], 1e-9)
        print(f"  {k:16s} | {po[k]:7.3f} | {ph[k]:7.3f} | {enr:.2f}x")


if __name__ == "__main__":
    main()
