"""characterize_survivors.py — profile the 460 (spec registration, 2026-07-08).

THE QUESTION: the multi-round asymptote left a hard-partition remainder. Before the
ledger re-parse is built to serve them, profile them against the RECOVERED
population. REGISTERED PREDICTION: survivors are ENRICHED for referential-binding
stress — oblique mentions, shuffled letters, problem size — the thrice-located
shallow-layer weakness. Uniform-across-teeth = the reading-repair story reworks.

Teeth attribution is post-hoc from the samples themselves (the generator's per-
sample render draws weren't logged — detectable from artifacts):
  oblique  : any mention span longer than 2 chars (letters are single chars)
  shuffled : any used variable i whose mention text != LETTERS[i]
  irrelevant: gen.irrelevant flag (logged)
  size     : n_vars; band: decisions label

Replays the deterministic stack (converged parser -> withhold-2 -> 4 specialist
rounds, field-only) to identify survivors, then prints enrichment ratios.

USAGE: DEV=AMD ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest \
           ROUNDS=4 ARM=field_only .venv/bin/python3 scripts/characterize_survivors.py
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

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def sample_teeth(smp):
    """Post-hoc teeth flags from sample artifacts."""
    oblique = any((b - a) > 2 for spans in smp["mentions"].values()
                  for (a, b) in spans)
    shuffled = False
    for v_str, spans in smp["mentions"].items():
        v = int(v_str)
        for (a, b) in spans:
            if (b - a) == 1 and smp["text"][a:b] != LETTERS[v]:
                shuffled = True
                break
        if shuffled:
            break
    return {"oblique": oblique, "shuffled": shuffled,
            "irrelevant": bool(smp.get("gen", {}).get("irrelevant", False)),
            "n_vars": smp["n_vars"], "band": smp["decisions"]}


def main():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from mycelium.csp_domains import problem_from_algebra
    from mycelium.csp_core import solve_symbolic
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
    K_WH = 2
    ROUNDS = int(os.environ.get("ROUNDS", "4"))

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
        else:
            wh = set()
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
                return False, wh
            sol = [int(res["assignment"][v]) for v in range(nv)]
            if not (q_pred < len(sol) and sol[q_pred] == gold_ans):
                return False, wh
            p2 = problem_from_algebra(nv, rels, gv, smp["m"])
            p2.domains0[q_pred].discard(sol[q_pred])
            if p2.domains0[q_pred]:
                r2 = solve_symbolic(p2, budget=100_000, seed=0)
                if r2["status"] == "solved":
                    return False, wh
            return True, wh
        except Exception:
            return False, wh

    def run_cond(model_p, cond_c, idx, ffld):
        outs = {}
        zt = np.zeros((len(idx), T_ALG), np.float32)
        fb = np.ones((len(idx), 1), np.float32)
        for s0 in range(0, len(idx), 8):
            sl = np.array(idx[s0:s0 + 8])
            pad = 8 - len(sl)
            sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
            rows = slice(s0, s0 + len(sl))
            f3 = ffld[rows]
            f1 = zt[rows]; f2 = fb[rows]
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

    # stage 0/1 with the plateaued parser (zero-cond graph for parity)
    c_zero = build_cond_params(9)
    blank = run_cond(p_plat, c_zero, list(range(n)),
                     np.zeros((n, L_FAC, N_FIELDS), np.float32))
    recovered, pool, flags_pool = [], [], []
    for i in range(n):
        smp = samples[i]
        o = blank[i]
        facs, q_pred = decode(o)
        ok, _ = solve_check(facs, q_pred, smp, 0)
        if ok:
            continue
        ok, wh = solve_check(facs, q_pred, smp, K_WH, o=o)
        if ok:
            recovered.append(i)
            continue
        ffld_i = np.zeros((L_FAC, N_FIELDS), np.float32)
        fi = 0
        for j in range(L_FAC):
            if o["pres"][j] <= 0:
                continue
            if fi in wh:
                ffld_i[j, :] = 1.0
            fi += 1
        pool.append(i)
        flags_pool.append(ffld_i)

    for rnd in range(ROUNDS):
        if not pool:
            break
        re = run_cond(p_re, c_re, pool, np.stack(flags_pool))
        nxt_pool, nxt_flags = [], []
        for r_i, i in enumerate(pool):
            o = re[int(i)]
            facs, q_pred = decode(o)
            ok, wh = solve_check(facs, q_pred, samples[int(i)], K_WH, o=o)
            if ok:
                recovered.append(i)
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

    survivors = pool
    print(f"[survivors] recovered={len(recovered)} survivors={len(survivors)}")

    # THE PROFILE: enrichment of teeth features, survivors vs recovered
    def profile(idx_list):
        t = [sample_teeth(samples[int(i)]) for i in idx_list]
        return {
            "oblique": np.mean([x["oblique"] for x in t]),
            "shuffled": np.mean([x["shuffled"] for x in t]),
            "irrelevant": np.mean([x["irrelevant"] for x in t]),
            "n_vars": np.mean([x["n_vars"] for x in t]),
            "band": np.mean([x["band"] for x in t]),
        }

    ps = profile(survivors)
    pr = profile(recovered)
    print(f"\n  feature     | survivors | recovered | enrichment")
    for k in ("oblique", "shuffled", "irrelevant", "n_vars", "band"):
        enr = ps[k] / max(pr[k], 1e-9)
        print(f"  {k:11s} |   {ps[k]:.3f}   |   {pr[k]:.3f}   |   {enr:.2f}x")
    print(f"\n  REGISTERED PREDICTION: survivors enriched for binding stress"
          f" (oblique/shuffled/size). Uniform = reading-repair story reworks.")


if __name__ == "__main__":
    main()
