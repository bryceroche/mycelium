"""composition_matrix.py — THE COMPOSITION MATRIX (2026-07-31,
registered before firing). Does the gate compose? 36 ordered cells
(A produces derived d, B consumes d) + 6 baselines, N=12/cell,
in-register mint (one lever: composition; register held fixed),
solution-first, solver-verified under the corrected uniqueness guard.
Verdict by the pinned discriminator: novel-vs-co-occurred mean gap
(ALGEBRA <=15 / PHRASEBOOK >30 / MIXED between); chained fdiv->fdiv
reported separately (registry resident). Artifact:
.cache/composition_matrix_v1.json — permanent fixture."""
import sys, os, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
import numpy as np
from collections import Counter
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

MAN = json.load(open(".cache/GENERATION.json"))
CKPT = os.environ.get("CM_CKPT") or MAN["parser_ckpt"]
tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[matrix] gate from manifest: {CKPT}")

L = "abcdefghij"
KINDS = ("add", "mul", "fdiv", "mod", "pct", "sel")

def mk_producer(kind, rng, v0):
    """Emit (sentences, facs, d_var, d_val, next_var). Vars from v0."""
    if kind == "add":
        x, y = int(rng.randint(2, 90)), int(rng.randint(2, 90))
        a, b, d = v0, v0+1, v0+2
        s = [f"{L[a]} is {x}.", f"{L[b]} is {y}.",
             f"{L[a]} plus {L[b]} equals {L[d]}."]
        f = [{"ftype":"given","var":a,"value":x},{"ftype":"given","var":b,"value":y},
             {"ftype":"rel","op":"add","args":[a,b],"result":d}]
        return s, f, d, x+y, v0+3
    if kind == "mul":
        x, y = int(rng.randint(2, 13)), int(rng.randint(2, 13))
        a, b, d = v0, v0+1, v0+2
        s = [f"{L[a]} is {x}.", f"{L[b]} is {y}.",
             f"{L[a]} times {L[b]} equals {L[d]}."]
        f = [{"ftype":"given","var":a,"value":x},{"ftype":"given","var":b,"value":y},
             {"ftype":"rel","op":"mul","args":[a,b],"result":d}]
        return s, f, d, x*y, v0+3
    if kind == "fdiv":
        K = int(rng.choice([2,3,4,5,6,7])); q_ = int(rng.randint(3, 40))
        x = K*q_
        a, d = v0, v0+1
        s = [f"{L[a]} is {x}.",
             f"When {L[a]} is divided by {K}, the quotient is {L[d]}."]
        f = [{"ftype":"given","var":a,"value":x},
             {"ftype":"fdiv","var":a,"k":K,"result":d}]
        return s, f, d, q_, v0+2
    if kind == "mod":
        K = int(rng.choice([3,4,5,6,7])); x = int(rng.randint(10, 290))
        if x % K == 0: x += 1
        a, d = v0, v0+1
        s = [f"{L[a]} is {x}.",
             f"When {L[a]} is divided by {K}, the remainder is {L[d]}."]
        f = [{"ftype":"given","var":a,"value":x},
             {"ftype":"mod","var":a,"k":K,"result":d}]
        return s, f, d, x % K, v0+2
    if kind == "pct":
        P = int(rng.choice([10,20,25,50,75])); base = int(rng.choice([20,40,60,80,120,160,200]))
        a, d = v0, v0+1
        s = [f"{L[a]} is {base}.", f"{L[d]} is {P} percent of {L[a]}."]
        f = [{"ftype":"given","var":a,"value":base},
             {"ftype":"pct","args":[d,a],"p":P}]
        return s, f, d, P*base//100, v0+2
    if kind == "sel":
        x, y = int(rng.randint(2, 140)), int(rng.randint(2, 140))
        if x == y: y += 1
        a, b, d = v0, v0+1, v0+2
        s = [f"{L[a]} is {x}.", f"{L[b]} is {y}.",
             f"{L[d]} is the larger of {L[a]} and {L[b]}."]
        f = [{"ftype":"given","var":a,"value":x},{"ftype":"given","var":b,"value":y},
             {"ftype":"sel","sel":"larger","args":[a,b],"result":d}]
        return s, f, d, max(x,y), v0+3
    raise ValueError(kind)

def mk_consumer(kind, rng, d, dval, v0):
    """Emit (sentences, facs, q_var, q_val, ok) consuming derived d."""
    if kind == "add":
        h = int(rng.randint(2, 90)); e, q = v0, v0+1
        if dval + h > 300: return None
        s = [f"{L[e]} is {h}.", f"{L[d]} plus {L[e]} equals {L[q]}."]
        f = [{"ftype":"given","var":e,"value":h},
             {"ftype":"rel","op":"add","args":[d,e],"result":q}]
        return s, f, q, dval+h
    if kind == "mul":
        h = int(rng.randint(2, 9)); e, q = v0, v0+1
        if dval * h > 300 or dval < 2: return None
        s = [f"{L[e]} is {h}.", f"{L[d]} times {L[e]} equals {L[q]}."]
        f = [{"ftype":"given","var":e,"value":h},
             {"ftype":"rel","op":"mul","args":[d,e],"result":q}]
        return s, f, q, dval*h
    if kind == "fdiv":
        K = int(rng.choice([2,3,4,5,6,7])); q = v0
        if dval < K or dval % K: return None
        s = [f"When {L[d]} is divided by {K}, the quotient is {L[q]}."]
        f = [{"ftype":"fdiv","var":d,"k":K,"result":q}]
        return s, f, q, dval//K
    if kind == "mod":
        K = int(rng.choice([3,4,5,6,7])); q = v0
        if dval <= K: return None
        s = [f"When {L[d]} is divided by {K}, the remainder is {L[q]}."]
        f = [{"ftype":"mod","var":d,"k":K,"result":q}]
        return s, f, q, dval % K
    if kind == "pct":
        P = int(rng.choice([10,20,25,50,75])); q = v0
        if (P*dval) % 100: return None
        s = [f"{L[q]} is {P} percent of {L[d]}."]
        f = [{"ftype":"pct","args":[q,d],"p":P}]
        return s, f, q, P*dval//100
    if kind == "sel":
        h = int(rng.randint(2, 290)); e, q = v0, v0+1
        if h == dval: h += 1
        s = [f"{L[e]} is {h}.", f"{L[q]} is the larger of {L[d]} and {L[e]}."]
        f = [{"ftype":"given","var":e,"value":h},
             {"ftype":"sel","sel":"larger","args":[d,e],"result":q}]
        return s, f, q, max(dval,h)
    raise ValueError(kind)

def mint_cell(A, B, n_target, seed):
    rng = np.random.RandomState(seed)
    rows, tries = [], 0
    while len(rows) < n_target and tries < 600:
        tries += 1
        sp, fp, d, dval, v0 = mk_producer(A, rng, 0)
        if B is None:
            q, qval, sc, fc = d, dval, [], []
        else:
            r = mk_consumer(B, rng, d, dval, v0)
            if r is None: continue
            sc, fc, q, qval = r
        facs = fp + fc
        nv = max(v for f in facs for v in
                 ([f.get("var",-1), f.get("result",-1)] + list(f.get("args",[])))) + 1
        letters = ", ".join(L[:nv])
        text = f"Consider the numbers {letters}. " + " ".join(sp+sc) + f" What is {L[q]}?"
        if any(int(x) > 300 for x in [dval, qval]): continue
        a2 = solve2(facs, q, {"n_vars": 24, "m": 300})
        if a2 != qval: continue
        rows.append({"text": text, "facs": facs, "q": q, "gold": qval})
    return rows

def parse_batch(texts):
    n = len(texts); N = ((n+7)//8)*8
    ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32); snt = np.zeros((N, T_ALG), np.int32)
    for i, t in enumerate(texts):
        e = tok.encode(t); Ln = min(len(e.ids), T_ALG)
        ids[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
        snt[i] = sent_indices(t, list(e.offsets), msk[i])
    st = recompute_states(ids)
    out_r = []
    for s0 in range(0, N, 8):
        out = forward(p, Tensor(st[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(msk[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(snt[s0:s0+8].astype(np.int32), dtype=dtypes.int))
        keys = ("pres","ftype","op","islit","dig","args","res","query") + (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0+bi < n: out_r.append(decode({k: o[k][bi] for k in o}))
    return out_r

# co-occurrence at semantic level from the banked sweep (sub folded to add)
sw = json.load(open(".cache/coverage_sweep.json"))
def sem(k): return k.replace("rel-sub", "rel-add")
mix_pairs = Counter()
for k, v in sw["mix_counts"].items():
    if k.startswith("pair:"):
        a, b = k[5:].split("->")
        mix_pairs[(sem(a).replace("rel-",""), sem(b).replace("rel-",""))] += v
CO_FLOOR = 20

results = {}
cells = [(A, None) for A in KINDS] + [(A, B) for A in KINDS for B in KINDS]
for ci, (A, B) in enumerate(cells):
    name = f"base:{A}" if B is None else f"{A}->{B}"
    rows = mint_cell(A, B, 12, 31000 + ci)
    if not rows:
        results[name] = {"n": 0}; print(f"[{name:12s}] minted 0 — infeasible cell", flush=True)
        continue
    ok = 0
    for j, r in enumerate(rows):
        vt = [r["text"]] + [permuted_view(r["text"], 99000 + 100*ci + 10*j + k) for k in range(1, 5)]
        views = [solve2(f, q, {"n_vars": 24, "m": 300}) for f, q in parse_batch(vt)]
        nn = [a for a in views if a is not None]
        c = Counter(nn).most_common(1); plur, cnt = c[0] if c else (None, 0)
        ok += (cnt >= 3 and plur == r["gold"])
    co = (B is not None) and (mix_pairs.get((A, B), 0) >= CO_FLOOR)
    results[name] = {"n": len(rows), "quorum_correct": ok, "rate": ok/len(rows),
                     "co_occurred": co}
    print(f"[{name:12s}] n={len(rows):2d}  quorum-correct {ok:2d}  rate {ok/len(rows):.2f}"
          f"{'  (co-occurred)' if co else ('' if B is None else '  (NOVEL)')}", flush=True)

# the pinned discriminator (chained fdiv->fdiv excluded from both means)
def cellset(pred):
    return [r for nm, r in results.items() if "->" in nm and nm != "fdiv->fdiv"
            and r.get("n", 0) >= 8 and pred(r)]
co_cells = cellset(lambda r: r["co_occurred"])
nv_cells = cellset(lambda r: not r["co_occurred"])
m_co = float(np.mean([r["rate"] for r in co_cells])) if co_cells else float("nan")
m_nv = float(np.mean([r["rate"] for r in nv_cells])) if nv_cells else float("nan")
gap = (m_co - m_nv) * 100
verdict = ("ALGEBRA (composition transfers)" if gap <= 15 else
           "PHRASEBOOK (compositions must be taught)" if gap > 30 else
           "MIXED (per-cell map stands; no aggregate claim)")
print(f"\n=== THE DISCRIMINATOR: co-occurred mean {m_co:.2f} (n={len(co_cells)} cells) "
      f"vs NOVEL mean {m_nv:.2f} (n={len(nv_cells)} cells)  gap {gap:+.0f} pts ===")
print(f"=== VERDICT (pinned): {verdict} ===")
fd = results.get("fdiv->fdiv", {})
print(f"[registry resident] fdiv->fdiv: {fd} (reported separately, excluded from means)")
json.dump({"ckpt": CKPT, "cells": results, "mean_co": m_co, "mean_novel": m_nv,
           "gap_pts": gap, "verdict": verdict},
          open(".cache/composition_matrix_v1.json", "w"), indent=1)
print("[saved] .cache/composition_matrix_v1.json")
