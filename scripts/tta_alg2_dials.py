"""tta_alg2_dials.py — LATTICE ACCEPTANCE on the expanded domain (2026-07-09):
does the certification channel survive the tranche? Plus the two riding
columns (spec-registered).

ARM D only (sentence permutation — the deployable/lattice arm), K=4 views +
original, tranche head (ALG2=1). Audit-free: view-0 one-shot forced answers
are the baseline population.

REGISTERED (before firing):
  1. CERTIFICATION SURVIVAL (the freeze's question): unanimity 5/5 precision
     >= 0.99 on alg2test. A relation set that degrades this dial FAILS
     acceptance regardless of fac-exact.
  2. PER-KIND COVERAGE (relay — the curriculum's baseline number): unanimity
     coverage split linear-only / has-sel / has-mod(/crt). Prediction:
     systematically LOWER on sel/mod samples = the new relations'
     view-robustness deficit, measured at birth, BEFORE the curriculum
     intervention (baseline before intervention, the house pattern).
  3. SELECTOR SILENT ERRORS (relay, registered at ratification — shortest
     shelf life): selector errors are rare but disproportionately SILENT
     (right graph, right roots, wrong pick — invisible to UNSAT and
     uniqueness); their natural detector is ANSWER-disagreement-despite-
     GRAPH-agreement across views. Column: P(ans-disagree | graph-agree),
     split by has-selector. Enriched on selector samples = the five-seat
     audience is load-bearing for quadratics.

USAGE: DEV=AMD ALG2=1 ALG_CKPT=.cache/phase1_algebra2_head.safetensors \
       ALG_TEST=.cache/algebra2_nl_test.jsonl ALG_TEST_NAME=alg2test \
       .venv/bin/python3 scripts/tta_alg2_dials.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

from phase1_algebra_head import (  # noqa: E402
    T_ALG, build_params, forward, load_alg, decode, sent_indices,
    ALG_CKPT, TOKENIZER_JSON,
)
from tta_views import permuted_view  # noqa: E402
from beacon_closing_arm import recompute_states  # noqa: E402

K_VIEWS = 4


def solve2(facs, q_pred, smp):
    """Gold-free forced answer via the v2 bridge (mod/sel-aware).
    gen-15: macros expand HERE — the solver's doorstep, so every consumer
    (gates, evals, lattice, lanes) inherits the constitutional boundary:
    the solver only ever sees primes."""
    from mycelium.csp_domains import problem_from_algebra3 as problem_from_algebra2
    from mycelium.csp_core import solve_symbolic
    if any(f.get("ftype") == "macro" for f in facs):
        from mycelium.macros import expand_graph
        facs, _ = expand_graph(facs, smp.get("n_vars", 24))
    gv = {f["var"]: f["value"] for f in facs if f["ftype"] == "given"}

    def fvars(f):
        if f["ftype"] in ("rel", "sel"):
            return list(f["args"]) + [f["result"]]
        if f["ftype"] == "pct":
            return list(f["args"])
        if f["ftype"] in ("mod", "fdiv"):
            return [f["var"], f["result"]]
        return [f["var"]]
    try:
        nv = max([smp["n_vars"]] + [v + 1 for f in facs for v in fvars(f)]
                 + [q_pred + 1])
        prob = problem_from_algebra2(nv, facs, gv, smp["m"])
        res = solve_symbolic(prob, budget=200_000, seed=0)
        if res["status"] != "solved":
            return None
        sol = [int(res["assignment"][v]) for v in range(nv)]
        if q_pred >= len(sol):
            return None
        p2 = problem_from_algebra2(nv, facs, gv, smp["m"])
        p2.domains0[q_pred].discard(sol[q_pred])
        if p2.domains0[q_pred]:
            r2 = solve_symbolic(p2, budget=100_000, seed=0)
            if r2["status"] == "solved":
                return None
        return sol[q_pred]
    except Exception:
        return None


def gkey(f):
    if f["ftype"] == "rel":
        return ("rel", f["op"], tuple(sorted(f["args"])), f["result"])
    if f["ftype"] == "given":
        return ("given", f["var"], f["value"])
    if f["ftype"] == "mod":
        return ("mod", f["var"], f["k"], f["result"])
    if f["ftype"] == "pct":
        return ("pct", tuple(f["args"]), f["p"])
    if f["ftype"] == "fdiv":
        return ("fdiv", f["var"], f["k"], f["result"])
    return ("sel", f["sel"], tuple(sorted(f["args"])), f["result"])


def main():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(TOKENIZER_JSON)
    samples, states, tokmask, gold, sent = load_alg("test")
    n = len(samples)
    gold_ans = [s["solution"][s["query_var"]] for s in samples]
    kind = []
    for s in samples:
        fts = {f["ftype"] for f in s["factors"]}
        kind.append("fdiv" if "fdiv" in fts else
                    ("pct" if "pct" in fts else
                     ("sel" if "sel" in fts else
                      ("mod" if "mod" in fts else "linear"))))

    p = build_params(0)
    sd = safe_load(ALG_CKPT)
    assert set(sd.keys()) == set(p.keys()), "eval load must be complete"
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    def parse_all(sts, msk, snt):
        out_a, out_h = {}, {}
        for s0 in range(0, n, 8):
            sl = np.arange(s0, min(s0 + 8, n))
            pad = 8 - len(sl)
            sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
            out = forward(p, Tensor(sts[sl_p].astype(np.float32),
                                    dtype=dtypes.float),
                          Tensor(msk[sl_p].astype(np.float32),
                                 dtype=dtypes.float),
                          Tensor(snt[sl_p].astype(np.int32),
                                 dtype=dtypes.int))
            keys = ("pres", "ftype", "op", "islit", "dig", "args", "res",
                    "query") + (("sel",) if "sel" in out else ())
            o = {k: out[k].realize().numpy() for k in keys}
            for bi, i in enumerate(sl):
                facs, q = decode({k: o[k][bi] for k in o})
                out_a[int(i)] = solve2(facs, q, samples[int(i)])
                out_h[int(i)] = frozenset(gkey(f) for f in facs)
        return out_a, out_h

    print("[tta-alg2] view 0 (original) ...")
    a0, h0 = parse_all(states, tokmask, sent)
    view_a, view_h = [a0], [h0]
    for k in range(1, K_VIEWS + 1):
        texts = [permuted_view(samples[i]["text"], 1000 * k + i)
                 for i in range(n)]
        ids = np.zeros((n, T_ALG), np.int32)
        msk = np.zeros((n, T_ALG), np.float32)
        snt = np.zeros((n, T_ALG), np.int32)
        for i, t in enumerate(texts):
            e = tok.encode(t)
            if len(e.ids) > T_ALG:
                continue
            ids[i, :len(e.ids)] = e.ids
            msk[i, :len(e.ids)] = 1.0
            snt[i] = sent_indices(t, list(e.offsets), msk[i])
        sts = recompute_states(ids)
        a, h = parse_all(sts, msk, snt)
        view_a.append(a)
        view_h.append(h)
        print(f"  view {k}: forced "
              f"{sum(1 for v in a.values() if v is not None)}/{n}")

    # MC-pi gate on the new domain
    wrong0 = [i for i in range(n)
              if a0[i] is not None and a0[i] != gold_ans[i]]
    same = sum(1 for i in wrong0 for k in range(1, K_VIEWS + 1)
               if view_a[k][i] == a0[i])
    print(f"\n  MC-PI GATE: same-wrong "
          f"{same / max(len(wrong0) * K_VIEWS, 1):.3f} (gate <0.30, "
          f"n={len(wrong0)})")

    # ORDINAL-QUERY RIDER (relay cut): failures by query type
    def is_ordinal_q(s):
        qs = s["mentions"].get(str(s["query_var"]), []) or \
            s["mentions"].get(s["query_var"], [])
        return any("term" in s["text"][a0:b0] for a0, b0 in qs)
    for grp, pred in (("ordinal-q", True), ("direct-q", False)):
        ids = [i for i in range(n) if is_ordinal_q(samples[i]) == pred]
        if not ids:
            continue
        fails = sum(1 for i in ids if view_a[0][i] is None
                    or view_a[0][i] != gold_ans[i])
        print(f"  {grp}: n={len(ids)} one-shot-fail {fails / len(ids):.3f}")

    # vote dials + per-kind unanimity coverage
    print(f"\n  VOTE DIALS (K=5):")
    for t in (3, 4, 5):
        r = w = 0
        cov_kind = Counter()
        n_kind = Counter(kind)
        for i in range(n):
            votes = [view_a[k][i] for k in range(K_VIEWS + 1)
                     if view_a[k][i] is not None]
            if not votes:
                continue
            top, cnt = Counter(votes).most_common(1)[0]
            if cnt < t:
                continue
            if top == gold_ans[i]:
                r += 1
            else:
                w += 1
            if t == 5:
                cov_kind[kind[i]] += 1
        print(f"  t={t}/5: accepted {r + w:3d} | precision "
              f"{r / max(r + w, 1):.4f} | coverage {(r + w) / n:.3f}"
              + ("" if t < 5 else "   <- CERTIFICATION (bar >=0.99)"))
        if t == 5:
            print(f"     unanimity coverage by kind: " + " | ".join(
                f"{kd}: {cov_kind[kd]}/{n_kind[kd]} = "
                f"{cov_kind[kd] / max(n_kind[kd], 1):.3f}"
                for kd in ("linear", "mod", "sel", "pct", "fdiv")))

    # persist per-sample outcomes (composition + any-threshold re-votes)
    SENT = -10**9
    va = np.full((n, K_VIEWS + 1), SENT, np.int64)
    for k in range(K_VIEWS + 1):
        for i in range(n):
            if view_a[k][i] is not None:
                va[i, k] = int(view_a[k][i])
    np.savez(".cache/tta_alg2_views.npz", view_ans=va,
             gold=np.array(gold_ans, np.int64))
    print(f"\n  [saved] .cache/tta_alg2_views.npz (per-view answers)")

    # selector silent-error column: ans-disagree despite graph-agree
    stats = {True: [0, 0], False: [0, 0]}   # has_sel -> [graph-agree, +ans-disagree]
    for i in range(n):
        pairs = [(view_a[k][i], view_h[k][i]) for k in range(K_VIEWS + 1)
                 if view_a[k][i] is not None]
        if len(pairs) < 2:
            continue
        for x in range(len(pairs)):
            for y in range(x + 1, len(pairs)):
                if pairs[x][1] == pairs[y][1]:
                    st = stats[kind[i] == "sel"]
                    st[0] += 1
                    st[1] += int(pairs[x][0] != pairs[y][0])
    for has_sel in (False, True):
        ga, ad = stats[has_sel]
        print(f"\n  {'SELECTOR' if has_sel else 'non-selector'} samples: "
              f"graph-agree pairs {ga} | ans-disagree-despite "
              f"{ad} ({ad / max(ga, 1):.4f})")
    print(f"  (relay bar: enrichment on selector samples = the five-seat "
          f"audience is load-bearing for quadratics)")


if __name__ == "__main__":
    main()
