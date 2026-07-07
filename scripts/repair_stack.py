"""repair_stack.py — THE COMPOSED REPAIR STACK, measured end to end (spec §12 final
registration): confidence-order -> WITHHOLD-AND-SOLVE -> retransmit only the
survivors -> answer. One pass. The Alternator's production loop, with every
component individually measured and the COMPOSITION the missing number.

KENKEN (--kenken): the full stack on the 57 banked failures.
  Stage 0: blank parse (the Brick-A flag-dependent ckpt's blank pass — the same
           model that retransmits, so stages compose within one weight set).
  Stage 1: WITHHOLD-AND-SOLVE at the measured peak (k=3, confidence-ordered).
           Gold grid -> RECOVERED FREE (Law 7 at the graph level).
  Stage 2: survivors get a NACK: suspect slots = the withheld/blamed set, mapped to
           sentences via the supervised attention (gold-free) -> conditioned
           re-parse -> withhold-and-solve AGAIN on the re-parse (the stack composes
           with itself). Count recoveries per stage.

ALGEBRA (--algebra): the registered SPARSE-DOMAIN FLIP test + the honest partial
  stack. Withholding curve k=0..5 on the big slice's failures (standing prediction:
  peak collapses toward k=0-1 — every equation load-bearing); answer-to-gold as the
  recovery metric; NO retransmission stage (no trained algebra NACK head — stated,
  not hidden).

USAGE:
  DEV=AMD .venv/bin/python3 scripts/repair_stack.py --kenken
  DEV=AMD ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest \
      .venv/bin/python3 scripts/repair_stack.py --algebra
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np


# ===========================================================================
# KENKEN: the full composed stack
# ===========================================================================

def run_kenken():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from mycelium.csp_domains import problem_from_kenken
    from mycelium.csp_core import solve_symbolic
    from phase1_delta_head import (
        L_SLOTS, H_WAIST, build_head_params, load_split, decode_slots,
        _gold_grid_cache, TYPES)
    from phase1_brick_a import (
        NACK_NPZ, BRICK_A_CKPT, build_encoder_params, head_forward_cond,
        flags_to_token_mask, COND_DIM)
    from phase1_brick_c import slot_confidence, slots_to_flags_via_attn

    samples, states, tokmask, gold, sent = load_split("test")
    z = np.load(NACK_NPZ.format(split="test"))
    has_fail = z["has_fail"]
    p = build_head_params(0)
    enc = build_encoder_params(1)
    sd = safe_load(BRICK_A_CKPT)
    for d in (p, enc):
        for k in d:
            d[k].assign(sd[k].to(d[k].device).cast(d[k].dtype)).realize()
    fail_i = [i for i in range(len(samples)) if has_fail[i]]

    def run_pass(idx, cond_np):
        outs = {}
        for s0 in range(0, len(idx), 8):
            sl = np.array(idx[s0:s0 + 8])
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

    def extract(o, N):
        pred = decode_slots(o, N)
        cages, clues, confs, slot_of = [], [], [], []
        pj = 0
        for j in range(L_SLOTS):
            if o["pres"][j] <= 0:
                continue
            if TYPES[int(o["type"][j].argmax())] != "cage":
                pj += 1
                continue
            f = pred[pj]; pj += 1
            rc = [[m // 7, m % 7] for m in f["members_flat"]]
            if not rc or any(a >= N or b >= N for (a, b) in rc):
                continue
            cages.append(rc); clues.append([f["op"], f["target"]])
            confs.append(slot_confidence(o, j)); slot_of.append(j)
        return cages, clues, confs, slot_of

    def withhold_solve(N, cages, clues, confs, k_wh, smp):
        """Returns ('gold'|'multi'|'unsat', withheld_local_indices)."""
        order = np.argsort(confs)
        wh = set(order[:k_wh].tolist())
        keep = [x for x in range(len(cages)) if x not in wh]
        c2 = [cages[x] for x in keep]
        l2 = [clues[x] for x in keep]
        try:
            res = solve_symbolic(problem_from_kenken(N, c2, l2),
                                 budget=100_000, seed=0)
        except Exception:
            return "unsat", wh
        if res["status"] != "solved":
            return "unsat", wh
        grid = [[int(res["assignment"][r * N + c]) for c in range(N)]
                for r in range(N)]
        return ("gold" if grid == _gold_grid_cache(smp)[0] else "multi"), wh

    K_WH = 3   # the measured peak
    blank = run_pass(fail_i, np.zeros((len(fail_i), COND_DIM), np.float32))
    stage1_rec, survivors, flags_rows = [], [], []
    for i in fail_i:
        o = blank[int(i)]
        smp = samples[int(i)]
        N = int(smp["N"])
        cages, clues, confs, slot_of = extract(o, N)
        verdict, wh = withhold_solve(N, cages, clues, confs, K_WH, smp)
        if verdict == "gold":
            stage1_rec.append(i)
        else:
            survivors.append(i)
            sus_slots = [slot_of[x] for x in wh]
            flags_rows.append(slots_to_flags_via_attn(sus_slots, o["attn_mean"],
                                                      sent[int(i)]))
    print(f"[stack:kenken] failures {len(fail_i)} | stage-1 withhold-{K_WH}-and-solve"
          f" RECOVERED {len(stage1_rec)} (free)")

    stage2_rec = 0
    if survivors:
        re = run_pass(survivors, np.stack(flags_rows))
        for i in survivors:
            o = re[int(i)]
            smp = samples[int(i)]
            N = int(smp["N"])
            cages, clues, confs, _s = extract(o, N)
            verdict, _wh = withhold_solve(N, cages, clues, confs, K_WH, smp)
            if verdict == "gold":
                stage2_rec += 1
    total = len(stage1_rec) + stage2_rec
    print(f"[stack:kenken] stage-2 retransmit(+withhold) on {len(survivors)}"
          f" survivors RECOVERED {stage2_rec}")
    print(f"[stack:kenken] COMPOSED RECOVERY: {total}/{len(fail_i)} = "
          f"{total/len(fail_i):.2f}   (components alone: withhold 15, retransmit 8)")


# ===========================================================================
# ALGEBRA: the sparse-domain FLIP test (withhold curve; no retransmit head)
# ===========================================================================

def run_algebra():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from mycelium.csp_domains import problem_from_algebra
    from mycelium.csp_core import solve_symbolic
    from phase1_algebra_head import (
        L_FAC, N_DIG, build_params, forward, load_alg, decode, ALG_CKPT)
    from tier0_incumbent import softmax, sig

    samples, states, tokmask, gold, sent = load_alg("test")
    p = build_params(0)
    sd = safe_load(ALG_CKPT)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    n = len(samples)

    def slot_conf_alg(o, j):
        c = float(sig(o["pres"][j]))
        c *= float(softmax(o["ftype"][j][None])[0].max())
        c *= float(softmax(o["res"][j][None])[0].max())
        if sig(o["islit"][j]) > 0.5:
            c *= float(np.mean(softmax(o["dig"][j]).max(-1)))
        else:
            c *= float(softmax(o["op"][j][None])[0].max())
            c *= float(np.sort(softmax(o["args"][j][None])[0])[-2:].sum())
        return c

    # collect failures (answer wrong) + per-slot data
    fails = []
    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        out = forward(p, Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))
        o = {k: out[k].realize().numpy() for k in
             ("pres", "ftype", "op", "islit", "dig", "args", "res", "query")}
        for bi, i in enumerate(sl):
            i = int(i)
            smp = samples[i]
            onp = {k: o[k][bi] for k in o}
            facs, q_pred = decode(onp)
            rels = [(f["op"], f["args"][0], f["args"][1], f["result"])
                    for f in facs if f["ftype"] == "rel"]
            gv = {f["var"]: f["value"] for f in facs if f["ftype"] == "given"}
            gold_ans = smp["solution"][smp["query_var"]]
            ok = False
            try:
                nv = max([smp["n_vars"]] + [v + 1 for f in facs for v in
                         ((list(f["args"]) + [f["result"]]) if f["ftype"] == "rel"
                          else [f["var"]])])
                res = solve_symbolic(problem_from_algebra(nv, rels, gv, smp["m"]),
                                     budget=200_000, seed=0)
                if res["status"] == "solved":
                    sol = [int(res["assignment"][v]) for v in range(nv)]
                    ok = q_pred < len(sol) and sol[q_pred] == gold_ans
            except Exception:
                pass
            if not ok:
                fails.append((i, onp, facs, q_pred))
    print(f"[stack:algebra] failures: {len(fails)}/{n}")

    print(f"\n  k | answer-recovered | UNSAT | underdet/other")
    for k_wh in range(6):
        rec = n_unsat = n_other = 0
        for i, onp, facs, q_pred in fails:
            smp = samples[i]
            idx_conf = []
            fi = 0
            for j in range(L_FAC):
                if onp["pres"][j] <= 0:
                    continue
                idx_conf.append((fi, slot_conf_alg(onp, j)))
                fi += 1
            order = [x for x, _c in sorted(idx_conf, key=lambda t: t[1])]
            wh = set(order[:k_wh])
            kept = [f for x, f in enumerate(facs) if x not in wh]
            rels = [(f["op"], f["args"][0], f["args"][1], f["result"])
                    for f in kept if f["ftype"] == "rel"]
            gv = {f["var"]: f["value"] for f in kept if f["ftype"] == "given"}
            gold_ans = smp["solution"][smp["query_var"]]
            try:
                nv = max([smp["n_vars"]] + [v + 1 for f in kept for v in
                         ((list(f["args"]) + [f["result"]]) if f["ftype"] == "rel"
                          else [f["var"]])] + [q_pred + 1])
                res = solve_symbolic(problem_from_algebra(nv, rels, gv, smp["m"]),
                                     budget=200_000, seed=0)
                if res["status"] != "solved":
                    n_unsat += 1
                    continue
                sol = [int(res["assignment"][v]) for v in range(nv)]
                if q_pred < len(sol) and sol[q_pred] == gold_ans:
                    # honest check: is it FORCED (unique at the query var)?
                    p2 = problem_from_algebra(nv, rels, gv, smp["m"])
                    p2.domains0[q_pred].discard(sol[q_pred])
                    forced = True
                    if p2.domains0[q_pred]:
                        r2 = solve_symbolic(p2, budget=100_000, seed=0)
                        forced = r2["status"] != "solved"
                    if forced:
                        rec += 1
                    else:
                        n_other += 1
                else:
                    n_other += 1
            except Exception:
                n_unsat += 1
        print(f"  {k_wh} |        {rec:3d}       |  {n_unsat:3d}  |     {n_other:3d}")
    print(f"\n  READ: the registered SPARSE-DOMAIN FLIP — prediction: peak collapses"
          f"\n  toward k=0-1 (coupled systems starve; every equation load-bearing)."
          f"\n  'answer-recovered' requires the query value FORCED (unique), not lucky.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kenken", action="store_true")
    ap.add_argument("--algebra", action="store_true")
    args = ap.parse_args(argv)
    if args.kenken:
        run_kenken()
    if args.algebra:
        run_algebra()


if __name__ == "__main__":
    main()
