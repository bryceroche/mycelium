"""tier0_incumbent.py — the incumbent confidence signal vs the silent errors
(spec §12 measurement 1, registered form: SEPARATION, not rank).

THE QUESTION: the algebra convergence eval produced 14 SILENT errors (wrong parses
that solve uniquely to wrong answers — invisible to every symbolic jaw). Does the
FREE confidence signal (per-field softmax/ sigmoid margins, zero new parameters —
the entropy null, incumbent since it ordered Brick-C's sweep) SEPARATE them from
correct parses? The registered number is the silents-vs-correct AUC per confidence
definition — 14 low scores mean nothing if correct parses score low too.

CONFIDENCE DEFINITIONS measured (per parse):
  pres    : mean presence-margin |sigmoid-0.5| over decisive slots
  fields  : mean top-prob over ftype/op/islit on present slots
  pointer : mean top-prob of args (top-2) + result pointers on present slots
  digits  : mean top digit prob on literal slots
  query   : query-pointer top-prob
  product : the slot_confidence-style product (the Brick-C incumbent's form)
  min     : the weakest field (the §12 combination-rule candidate)

OUTPUT: per-definition AUC for (a) SILENT vs CORRECT — THE number; (b) all-wrong vs
correct (general discrimination, context); plus the distribution summary (median
confidence per class) so separation is visible, not just scored.

USAGE:  DEV=AMD .venv/bin/python3 scripts/tier0_incumbent.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

from phase1_algebra_head import (  # noqa: E402
    L_FAC, N_DIG, build_params, forward, load_alg, decode, ALG_CKPT,
)


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def parse_confidences(o):
    """Per-parse confidence under each definition. o: per-sample numpy dict."""
    pres_p = sig(o["pres"])
    present = pres_p > 0.5
    if not present.any():
        return {k: 0.0 for k in ("pres", "fields", "pointer", "digits", "query",
                                 "product", "min")}
    idx = np.where(present)[0]
    pres_m = np.abs(pres_p - 0.5).mean() * 2
    ftype_p = softmax(o["ftype"][idx]).max(-1)
    op_p = softmax(o["op"][idx]).max(-1)
    islit_m = np.abs(sig(o["islit"][idx]) - 0.5) * 2
    fields = float(np.mean((ftype_p + op_p + islit_m) / 3))
    args_sm = softmax(o["args"][idx])
    args_p = np.sort(args_sm, -1)[:, -2:].sum(-1)          # mass on the chosen 2
    res_p = softmax(o["res"][idx]).max(-1)
    pointer = float(np.mean((args_p + res_p) / 2))
    lit = sig(o["islit"][idx]) > 0.5
    if lit.any():
        dig_p = softmax(o["dig"][idx][lit]).max(-1).mean()
        digits = float(dig_p)
    else:
        digits = 1.0
    query = float(softmax(o["query"][None])[0].max())
    per_slot = []
    for j in idx:
        c = float(pres_p[j])
        c *= float(softmax(o["ftype"][j][None])[0].max())
        c *= float(softmax(o["res"][j][None])[0].max())
        if sig(o["islit"][j]) > 0.5:
            c *= float(np.mean(softmax(o["dig"][j]).max(-1)))
        else:
            c *= float(softmax(o["op"][j][None])[0].max())
            c *= float(np.sort(softmax(o["args"][j][None])[0])[-2:].sum())
        per_slot.append(c)
    product = float(np.mean(per_slot))
    mn = float(min(pres_m, fields, pointer, digits, query))
    return {"pres": pres_m, "fields": fields, "pointer": pointer,
            "digits": digits, "query": query, "product": product, "min": mn}


def auc(pos, neg):
    """AUC that HIGH confidence predicts membership in `pos` (correct)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = np.empty(len(allv))
    order = np.argsort(allv)
    r[order] = np.arange(len(allv))
    return (r[:len(pos)].mean() - (len(pos) - 1) / 2) / len(neg)


def main():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from mycelium.csp_domains import problem_from_algebra
    from mycelium.csp_core import solve_symbolic

    samples, states, tokmask, gold, sent = load_alg("test")
    p = build_params(0)
    sd = safe_load(ALG_CKPT)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    n = len(samples)

    classes, confs = [], []
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
            confs.append(parse_confidences(onp))
            facs, q_pred = decode(onp)
            rels = [(f["op"], f["args"][0], f["args"][1], f["result"])
                    for f in facs if f["ftype"] == "rel"]
            gv = {f["var"]: f["value"] for f in facs if f["ftype"] == "given"}
            gold_ans = smp["solution"][smp["query_var"]]
            cat = "DETECT"
            try:
                nv = max([smp["n_vars"]] + [v + 1 for f in facs for v in
                         ((list(f["args"]) + [f["result"]]) if f["ftype"] == "rel"
                          else [f["var"]])])
                res = solve_symbolic(problem_from_algebra(nv, rels, gv, smp["m"]),
                                     budget=200_000, seed=0)
                if res["status"] == "solved":
                    sol = [int(res["assignment"][v]) for v in range(nv)]
                    if q_pred < len(sol) and sol[q_pred] == gold_ans:
                        cat = "CORRECT"
                    else:
                        multi = False
                        for v in range(nv):
                            if v in gv:
                                continue
                            p2 = problem_from_algebra(nv, rels, gv, smp["m"])
                            p2.domains0[v].discard(sol[v])
                            if p2.domains0[v]:
                                r2 = solve_symbolic(p2, budget=100_000, seed=0)
                                if r2["status"] == "solved":
                                    multi = True
                                    break
                        cat = "DETECT" if multi else "SILENT"
            except Exception:
                cat = "DETECT"
            classes.append(cat)

    classes = np.array(classes)
    keys = list(confs[0].keys())
    print(f"[tier-0 incumbent] n={n}: CORRECT={int((classes=='CORRECT').sum())} "
          f"SILENT={int((classes=='SILENT').sum())} "
          f"DETECT={int((classes=='DETECT').sum())}")
    print(f"\n  {'definition':9s} | AUC correct-vs-SILENT | AUC correct-vs-ALL-wrong"
          f" | median conf C/S")
    for k in keys:
        v = np.array([c[k] for c in confs])
        a_sil = auc(v[classes == "CORRECT"], v[classes == "SILENT"])
        a_all = auc(v[classes == "CORRECT"], v[classes != "CORRECT"])
        mc = np.median(v[classes == "CORRECT"])
        ms = np.median(v[classes == "SILENT"]) if (classes == "SILENT").any() else float("nan")
        print(f"  {k:9s} |        {a_sil:.3f}         |         {a_all:.3f}        "
              f" | {mc:.3f}/{ms:.3f}")
    print(f"\n  READ (registered): the number is SEPARATION — AUC of correct vs"
          f" silent. ~0.5 = the entropy null is blind to silents (the trained head"
          f" gets its job); >~0.8 = the NACK stack completes with ZERO new params.")


if __name__ == "__main__":
    main()
