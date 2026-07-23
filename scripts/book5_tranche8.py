"""book5_tranche8.py — BOOK 5, tranche 8 (2026-07-22): THE FIRST WIDE
TRANCHE under the width law (bench-supply width, annotator floor 0.75,
cures batched by mechanism). Retries carry their mechanism cures:
[36] shortened chain, [40] de-twinned givens, [45] crown DECOMPOSED
per the diet-wall rule (42x10 -> 420 derived; givens never exceed
299). 28 fresh primes + 2 crowns + 5 certificates. Gate =
crown_reader_v4 (FTYPES=8); crowns bank on PRIME TWINS.
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_DUP", "1")
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from mycelium.macros import expand_graph, MACRO_GRAMMAR_VERSION
from hash_audit_iso import canon
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

L = json.load(open(".cache/book5_lanes.json"))
BY = {l["idx"]: l for l in L}
D = "Consider the numbers "

PAGES = [
 # RETRIES (cures assigned at t7 autopsies)
 (36, 300, D+"a, b, c, d, e, f, g. a is 9. b times b equals a. c is 2. b exceeds d by c. d plus d equals e. f is 5. e plus f equals g. What is g?", True, "v2"),
 (40, 300, D+"a, b, c, d. a is 13. b is 12. b times c equals d. d plus c equals a. What is c?", True, "v2"),
 # FRESH
 (58, 300, D+"a, b, c, d, e, f, g, h, i. c is 1. a exceeds d by c. b plus c equals e. f is 3. f times d equals e. a plus c equals g. b exceeds h by c. i is 2. i times g equals h. What is b?", True, "v1"),
 (59, 300, D+"a, b, c, d. a is 3. b is 3. b times b equals c. c plus a equals d. What is d?", True, "v1"),
 (60, 300, D+"a, b, c, d, e, f, g. a is 4. b is 3. c is 27. d times d equals e. a times e equals f. b times d equals g. f exceeds c by g. What is d?", True, "v1"),
 (61, 300, D+"a, b, c. a is 8. b is 7. a times b equals c. What is c?", True, "v1"),
 (62, 300, D+"a, b, c, d, e. a is 8. When a is divided by 4, the quotient is b and the remainder is c. d is 3. b plus d equals e. What is e?", True, "v1"),
 (63, 300, D+"a, b, c, d, e, f, g. b is 12. e is 1. b plus e equals c. a plus e equals d. c times d equals f. It is known that f is 104. a plus b equals g. What is g?", True, "v1"),
 (64, 300, D+"a, b, c. a is 144. b is 16. b times c equals a. What is c?", True, "v1"),
 (65, 300, D+"a, b, c, d, e, f, g, h. a is 3. b is 4. a times a equals c. b times b equals d. c plus d equals e. f times f equals e. g is 10. f plus g equals h. What is h?", True, "v1"),
 (66, 300, D+"a, b, c. a is 3. a plus a equals c. What is c?", True, "v1"),
 (67, 300, D+"a, b, c, d, e. a is 100. b is 72. c is 64. a plus b equals d. d plus c equals e. What is e?", True, "v1"),
 (69, 300, D+"a, b, c. a is 16. a plus a equals c. What is c?", True, "v1"),
 (70, 300, D+"a, b, c. a is 1. b is 14. a plus b equals c. What is c?", True, "v1"),
 (71, 300, D+"a, b, c, d, e, f, g. a is 5. a times a equals b. c is 3. c times c equals d. d plus e equals b. f times f equals e. f plus c equals g. What is g?", True, "v1"),
 (72, 300, D+"a, b, c. a is 5. b is 10. a times b equals c. What is c?", True, "v1"),
 (74, 300, D+"a, b, c, d. a is 10. b is 4. b plus c equals a. What is c?", True, "v1"),
 (75, 300, D+"a, b, c, d, e. a is 3. b is 9. a plus b equals c. d is 2. d times e equals c. What is e?", True, "v1"),
 (77, 300, D+"a, b, c. a is 1. b is 5. a plus b equals c. What is c?", True, "v1"),
 (78, 300, D+"a, b, c. a is 2. When a is divided by 2, the quotient is b and the remainder is c. What is b?", True, "v1"),
 (80, 300, D+"a, b, c, d, e, f, g. b is 11. e is 1. b plus e equals c. a plus e equals d. c times d equals f. It is known that f is 96. a plus b equals g. What is g?", True, "v1"),
 (82, 300, D+"a, b, c. a is 4. b is 1. a plus b equals c. What is c?", True, "v1"),
 (83, 300, D+"a, b, c. a is 6. b is 2. b times c equals a. What is c?", True, "v1"),
 (85, 300, D+"a, b, c, d, e. a is 270. b is 6. c is 3. b plus c equals d. e plus d equals a. What is e?", True, "v1"),
 (86, 300, D+"a, b, c, d, e. a is 42. b is 5. c is 1. b plus c equals d. e plus d equals a. What is e?", True, "v1"),
 (87, 300, D+"a, b, c, d, e. a is 4. b is 12. c is 9. a plus b equals d. d plus c equals e. What is e?", True, "v1"),
 (88, 300, D+"a, b, c, d. a is 23. b is 2. a plus b equals c. d times d equals c. What is d?", True, "v1"),
 (90, 300, D+"a, b, c. a is 107. b is 1. c plus b equals a. What is c?", True, "v1"),
 (91, 300, D+"a, b, c. a is 2. b is 2. a times b equals c. What is c?", True, "v1"),
 (92, 300, D+"a, b, c. a is 96. b is 4. c plus b equals a. What is c?", True, "v1"),
 (93, 500, D+"a, b, c, d, e, f. a is 16. b is 12. a times a equals c. b times b equals d. c plus d equals e. f times f equals e. What is f?", True, "v1"),
]
PAGES = [p for p in PAGES if not p[4].startswith("SKIP")]

MACRO_PAGES = [
 # [45] CROWN RETRY, DECOMPOSED per the diet wall: 42*10=420 derived, +12, /4
 (45, 500,
  D+"a, b, c, d, e, f. a is 42. b is 10. a times b equals c. d is 12. c plus d equals e. When e is divided by 4, the quotient is f. What is f?",
  D+"a, b, c, d, e, f, g. a is 42. b is 10. a times b equals c. d is 12. c plus d equals e. When e is divided by 4, the quotient is f and the remainder is g. What is f?",
  [{"ftype": "given", "var": 0, "value": 42, "spans": []},
   {"ftype": "given", "var": 1, "value": 10, "spans": []},
   {"ftype": "rel", "op": "mul", "args": [0, 1], "result": 2, "spans": []},
   {"ftype": "given", "var": 3, "value": 12, "spans": []},
   {"ftype": "rel", "op": "add", "args": [2, 3], "result": 4, "spans": []},
   {"ftype": "macro", "name": "FRAC_OF", "a": 1, "k": 4, "x": 4, "result": 5}],
  5),
 # [73] equidistant x: (16-4)/4 = 3: sub then FRAC_OF
 (73, 300,
  D+"a, b, c, d. a is 16. b is 4. a exceeds c by b. When c is divided by 4, the quotient is d. What is d?",
  D+"a, b, c, d, e. a is 16. b is 4. a exceeds c by b. When c is divided by 4, the quotient is d and the remainder is e. What is d?",
  [{"ftype": "given", "var": 0, "value": 16, "spans": []},
   {"ftype": "given", "var": 1, "value": 4, "spans": []},
   {"ftype": "rel", "op": "sub", "args": [0, 1], "result": 2, "spans": []},
   {"ftype": "macro", "name": "FRAC_OF", "a": 1, "k": 4, "x": 2, "result": 3}],
  3),
]

REGISTRY = [
 (68, "complex-power-cycle"),
 (79, "geometric-partial-sum"),
 (89, "distinct-value-count"),
 (94, "median-free-choice"),
 (95, "digit-sum-count"),
]

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/crown_reader_v4.safetensors")
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[b5t8] gate = {CKPT} (FTYPES=8) | pages {len(PAGES)} + {len(MACRO_PAGES)} macro | registry {len(REGISTRY)} | WIDE")


def parse_batch(texts):
    n = len(texts)
    N = ((n + 7) // 8) * 8
    ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32)
    snt = np.zeros((N, T_ALG), np.int32)
    for i, t in enumerate(texts):
        e = tok.encode(t)
        Ln = min(len(e.ids), T_ALG)
        ids[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
        snt[i] = sent_indices(t, list(e.offsets), msk[i])
    st = recompute_states(ids)
    res = []
    for s0 in range(0, N, 8):
        out = forward(p, Tensor(st[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(msk[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(snt[s0:s0+8].astype(np.int32), dtype=dtypes.int))
        keys = [k for k in ("pres", "ftype", "op", "islit", "dig", "dig2", "args",
                            "res", "query", "sel", "dup", "y") if k in out]
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0 + bi < n:
                res.append(decode({k: o[k][bi] for k in o}))
    return res


def gate_dialect(text, m, answer, seed0):
    texts = [text] + [permuted_view(text, seed0 + k) for k in range(1, 5)]
    votes, best = [], None
    for facs, q in parse_batch(texts):
        a = solve2(facs, q, {"n_vars": 24, "m": m})
        if a is not None:
            votes.append(a)
            if a == answer and best is None:
                best = (facs, q)
    top, cnt = (Counter(votes).most_common(1)[0] if votes else (None, 0))
    return (cnt >= 3 and top == answer), votes, best


banked, missed = [], []
fresh_attempted = fresh_banked = 0
for li, m, dia, fs, ver in PAGES:
    x = BY[li]
    is_fresh = ver == "v1"
    fresh_attempted += is_fresh
    ok, votes, best = gate_dialect(dia, m, x["answer"], 670000 + 100 * li)
    if ok and best:
        facs, q = best
        fresh_banked += is_fresh
        banked.append(dict(text=x["problem"], factors=facs, query_var=q,
                           n_vars=24, m=m, decisions=[], mentions=[],
                           solution=[0] * 24,
                           gen=dict(src_idx=x["src_idx"], book=5, tranche=8,
                                    floor="prime", fs=fs, dialect=dia,
                                    gate="5view-vote+key", generation="16")))
        print(f"  [page {li}] BANKED (votes {votes})")
    else:
        missed.append((li, votes))
        print(f"  [page {li}] MISS (votes {votes})")

macro_banked = 0
for li, m, mdia, pdia, mfacs, q in MACRO_PAGES:
    x = BY[li]
    pfacs, nv = expand_graph([dict(f) for f in mfacs], 24)
    used = sorted({v for f in pfacs for v in (list(f.get("args", [])) +
                   [f[k] for k in ("result", "var") if k in f])})
    rm = {v: i for i, v in enumerate(used)}
    pf2 = []
    for f in pfacs:
        f = dict(f)
        if "args" in f:
            f["args"] = [rm[v] for v in f["args"]]
        for kk in ("result", "var"):
            if kk in f:
                f[kk] = rm[f[kk]]
        pf2.append(f)
    q_p = rm[q]
    a = solve2(pf2, q_p, {"n_vars": 24, "m": m})
    assert a == x["answer"], (li, a, x["answer"])
    dg_m, _ = canon({"factors": mfacs, "n_vars": 24, "query_var": q})
    dg_p, _ = canon({"factors": pf2, "n_vars": 24, "query_var": q_p})
    assert dg_m == dg_p, (li, dg_m, dg_p)
    ok, votes, best = gate_dialect(pdia, m, x["answer"], 680000 + 100 * li)
    if ok:
        for floor, facs_, dia_, qv in (("macro", mfacs, mdia, q),
                                       ("prime", pf2, pdia, q_p)):
            banked.append(dict(text=x["problem"], factors=facs_, query_var=qv,
                               n_vars=24, m=m, decisions=[], mentions=[],
                               solution=[0] * 24,
                               gen=dict(src_idx=x["src_idx"], book=5, tranche=8,
                                        floor=floor, fs=True, dialect=dia_,
                                        knot=dg_m, grammar=MACRO_GRAMMAR_VERSION,
                                        gate="5view-vote+key(prime-twin)",
                                        generation="16")))
        macro_banked += 1
        print(f"  [MACRO {li}] BANKED both floors, one knot {dg_m[:12]} (votes {votes})")
    else:
        missed.append((li, votes))
        print(f"  [MACRO {li}] prime-twin MISS (votes {votes})")

certs = [dict(lane_idx=li, family=fam, raw=BY[li]["problem"],
              answer=BY[li]["answer"], src_idx=BY[li]["src_idx"],
              book=5, tranche=8) for li, fam in REGISTRY]

with open(".cache/book5_prose_pairs.jsonl", "a") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
json.dump(certs, open(".cache/book5_organ_registry_t8.json", "w"))
rate = fresh_banked / max(fresh_attempted, 1)
print(f"\n[b5t8] banked rows: {len(banked)} ({macro_banked} macro pairs, "
      f"grammar {MACRO_GRAMMAR_VERSION}) | missed: {[m_[0] for m_ in missed]} | "
      f"certificates: {len(certs)}")
print(f"[b5t8] ANNOTATOR FLOOR: fresh first-pass {fresh_banked}/{fresh_attempted}"
      f" = {rate:.2f} (floor 0.75 -> {'HELD' if rate >= 0.75 else 'BREACHED — next tranche narrows'})")
