"""survivor_multiplicity.py — the CONDITIONAL multiplicity probe (spec registration,
2026-07-08, the relay's sharpening of the refuted teeth profile).

THE TAUTOLOGY RISK IT AVOIDS: "survivors are enriched for high multiplicity" is
partially MECHANICAL — the loop fixes ~1-2 errors/round, so 5-error parses
surviving is arithmetic, not discovery. The informative cuts are CONDITIONAL:

  1. S(m): survivorship rate as a function of initial errors-per-parse, over the
     627 round-entering failures. AUC (midrank — §6 law) of m predicting survival.
  2. THE MECHANICAL MODEL: a parse with multiplicity m recovers by round
     ceil(m/f) for a single fixes-per-round capacity f. Grid f; does any f
     reproduce the observed 123 -> 39 -> 5 -> 0 decay?
  3. THE RESIDUAL: conditional on m (within bins), do survivors differ from
     recovered in FIELD MIX (which heads are wrong) or teeth? Residual structure
     = where decode-degeneracy would live.

REGISTERED DECISION RULE (three futures, thresholds pinned BEFORE measuring):
  A. AUC(m) >= 0.75 AND some f reproduces the decay (each round within ~2x)
     -> the loop is ROUND-BUDGET-LIMITED on the dense tail; the answer is more
     rounds + better suspect-ranking, NOT a re-parse or transplant.
  B. AUC(m) in (0.6, 0.75) or per-bin residual field/teeth structure
     -> multiplicity is real but unsaturated; the residual axis is the frontier.
  C. AUC(m) < 0.6 (multiplicity ~uniform too)
     -> the remainder is plausibly DECODE-DEGENERATE (belief never concentrates);
     rerank toward the deducer-side suspicion transplant.

ERROR TAXONOMY (multiplicity = paired field-errors + unpaired + query flag):
  gold/pred factors canonicalized ("rel", op, sorted(args), result — add/mul are
  commutative by construction; "given", var, value). Unmatched gold factors are
  greedily paired to unmatched pred factors sharing >=2 of 3 rel components (or
  the same var for givens); a pair = ONE field error attributed to the differing
  field (rel_op / rel_args / rel_result / given_value); unpaired = missing /
  phantom; + query_wrong.

USAGE: DEV=AMD ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest \
           .venv/bin/python3 scripts/survivor_multiplicity.py
"""
from __future__ import annotations

import math
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
from characterize_survivors import sample_teeth  # noqa: E402

FIELD_KINDS = ("rel_op", "rel_args", "rel_result", "given_value",
               "missing", "phantom", "query_wrong")


def factor_errors(facs, q_pred, smp):
    """Paired-field error taxonomy vs gold. Returns (multiplicity, kind-counts)."""
    grels, ggivs = [], {}
    for f in smp["factors"]:
        if f["ftype"] == "rel":
            grels.append((f["op"], tuple(sorted(f["args"])), f["result"]))
        else:
            ggivs[f["var"]] = f["value"]
    prels, pgivs = [], {}
    for f in facs:
        if f["ftype"] == "rel":
            prels.append((f["op"], tuple(sorted(f["args"])), f["result"]))
        else:
            pgivs[f["var"]] = f["value"]

    counts = {k: 0 for k in FIELD_KINDS}
    # exact rel matches out first (multiset)
    from collections import Counter
    gc, pc = Counter(grels), Counter(prels)
    gonly = list((gc - pc).elements())
    ponly = list((pc - gc).elements())
    # greedy pair: >=2 of 3 components shared
    used = [False] * len(ponly)
    for g in gonly:
        best, best_n = -1, 1
        for pi, p in enumerate(ponly):
            if used[pi]:
                continue
            nmatch = (g[0] == p[0]) + (g[1] == p[1]) + (g[2] == p[2])
            if nmatch > best_n:
                best, best_n = pi, nmatch
        if best >= 0:
            used[best] = True
            p = ponly[best]
            if g[0] != p[0]:
                counts["rel_op"] += 1
            elif g[1] != p[1]:
                counts["rel_args"] += 1
            else:
                counts["rel_result"] += 1
        else:
            counts["missing"] += 1
    counts["phantom"] += sum(1 for u in used if not u)
    # givens pair by var
    for v, val in ggivs.items():
        if v in pgivs:
            if pgivs[v] != val:
                counts["given_value"] += 1
        else:
            counts["missing"] += 1
    counts["phantom"] += sum(1 for v in pgivs if v not in ggivs)
    if q_pred != smp["query_var"]:
        counts["query_wrong"] = 1
    return sum(counts.values()), counts


def midrank_auc(scores_pos, scores_neg):
    """Mann-Whitney AUC with midranks (discrete scores need midrank — §6)."""
    allv = np.concatenate([scores_pos, scores_neg]).astype(np.float64)
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1, n2 = len(scores_pos), len(scores_neg)
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    return u / (n1 * n2)


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

    # stage 0/1 with the plateaued parser
    c_zero = build_cond_params(9)
    blank = run_cond(p_plat, c_zero, list(range(n)),
                     np.zeros((n, L_FAC, N_FIELDS), np.float32))
    pool, flags_pool = [], []
    stage1, mult, fmix = [], {}, {}
    for i in range(n):
        smp = samples[i]
        o = blank[i]
        facs, q_pred = decode(o)
        ok, _ = solve_check(facs, q_pred, smp, 0)
        if ok:
            continue
        m_i, counts_i = factor_errors(facs, q_pred, smp)
        mult[i], fmix[i] = m_i, counts_i
        ok, wh = solve_check(facs, q_pred, smp, K_WH, o=o)
        if ok:
            stage1.append(i)
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

    pool0 = list(pool)  # the 627 round-entering failures
    round_rec = []      # recovered index sets per round
    for rnd in range(ROUNDS):
        if not pool:
            round_rec.append([])
            continue
        re = run_cond(p_re, c_re, pool, np.stack(flags_pool))
        rec_r, nxt_pool, nxt_flags = [], [], []
        for r_i, i in enumerate(pool):
            o = re[int(i)]
            facs, q_pred = decode(o)
            ok, wh = solve_check(facs, q_pred, samples[int(i)], K_WH, o=o)
            if ok:
                rec_r.append(i)
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
        round_rec.append(rec_r)
        pool, flags_pool = nxt_pool, nxt_flags

    survivors = set(pool)
    recovered_rounds = set(x for r in round_rec for x in r)
    obs_decay = [len(r) for r in round_rec]
    print(f"[mult] stage1={len(stage1)} rounds={obs_decay} "
          f"survivors={len(survivors)} (pool0={len(pool0)})")

    # ---- CUT 1: S(m) + midrank AUC over the round-entering pool ----
    m_surv = np.array([mult[i] for i in pool0 if i in survivors])
    m_rec = np.array([mult[i] for i in pool0 if i in recovered_rounds])
    m_st1 = np.array([mult[i] for i in stage1])
    print(f"\n  mean multiplicity: stage1-recovered {m_st1.mean():.2f} | "
          f"round-recovered {m_rec.mean():.2f} | survivors {m_surv.mean():.2f}")
    print(f"\n  S(m): survivorship by initial errors-per-parse (pool={len(pool0)})")
    print(f"  m     |   n   | survive-rate")
    bins = [(1, "1"), (2, "2"), (3, "3"), (4, "4")]
    binned = {}
    for i in pool0:
        b = min(mult[i], 5)
        binned.setdefault(b, []).append(i)
    for b in sorted(binned):
        idx = binned[b]
        sr = np.mean([i in survivors for i in idx])
        print(f"  {'5+' if b == 5 else b:>2}    | {len(idx):5d} |   {sr:.3f}")
    auc = midrank_auc(m_surv.astype(float), m_rec.astype(float))
    print(f"\n  AUC(multiplicity -> survival, midrank) = {auc:.3f}")

    # ---- CUT 2: the mechanical fixes-per-round model ----
    print(f"\n  mechanical model: recover in round ceil(m/f); observed {obs_decay}")
    ms = np.array([mult[i] for i in pool0])
    for f in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        need = np.ceil(ms / f).astype(int)
        pred = [int((need == r + 1).sum()) for r in range(ROUNDS)]
        pred_surv = int((need > ROUNDS).sum())
        print(f"  f={f:.1f} -> rounds {pred} survivors {pred_surv}")

    # ---- CUT 3: residual structure conditional on m ----
    print(f"\n  residual (within m-bin): field-mix + teeth, survivors vs recovered")
    for b in sorted(binned):
        idx_s = [i for i in binned[b] if i in survivors]
        idx_r = [i for i in binned[b] if i in recovered_rounds]
        if len(idx_s) < 15 or len(idx_r) < 15:
            print(f"  m={'5+' if b == 5 else b}: skipped (n_s={len(idx_s)}, "
                  f"n_r={len(idx_r)} — too thin)")
            continue
        def mix(idx):
            tot = {k: 0 for k in FIELD_KINDS}
            for i in idx:
                for k in FIELD_KINDS:
                    tot[k] += fmix[i][k]
            s = max(sum(tot.values()), 1)
            return {k: tot[k] / s for k in FIELD_KINDS}
        xs, xr = mix(idx_s), mix(idx_r)
        line = " ".join(f"{k}:{xs[k]:.2f}/{xr[k]:.2f}" for k in FIELD_KINDS
                        if xs[k] > 0.01 or xr[k] > 0.01)
        ts = np.mean([sample_teeth(samples[i])["shuffled"] for i in idx_s])
        tr = np.mean([sample_teeth(samples[i])["shuffled"] for i in idx_r])
        print(f"  m={'5+' if b == 5 else b} (n_s={len(idx_s)}, n_r={len(idx_r)}): "
              f"{line} | shuffled {ts:.2f}/{tr:.2f}")

    # the de-enrichment glance: loud-teeth -> low multiplicity?
    sh = [i for i in pool0 if sample_teeth(samples[i])["shuffled"]]
    nsh = [i for i in pool0 if not sample_teeth(samples[i])["shuffled"]]
    print(f"\n  loud-teeth glance: mean m shuffled={np.mean([mult[i] for i in sh]):.2f}"
          f" vs unshuffled={np.mean([mult[i] for i in nsh]):.2f}")

    print(f"\n  REGISTERED RULE: A) AUC>=0.75 + decay reproduced -> round-budget-"
          f"limited; B) 0.6-0.75 or per-bin residual -> residual axis; "
          f"C) AUC<0.6 -> decode-degenerate, transplant reranks.")


if __name__ == "__main__":
    main()
