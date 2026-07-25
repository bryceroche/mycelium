"""book5_tranche10.py — BOOK 5, tranche 11 (2026-07-24): the tempo's completing inhale — the shelf's last page
(35 pages + 2 crowns). Retry bench rides with mechanism-certain cures:
[36] v3 mul-voiced doubling (autopsy showed BOTH prior misses contained
add-dups — the family's 4th specimen), [66]/[69] mul-voiced, [60]
add-voiced, [75] fdiv-voiced. Crowns: [126] THE FIRST WILD a>1
FRAC_OF (3/5 of 30); [109] FRAC_OF over a derived product. m-dial
live on 5 pages (max m=4000). Gate = crown_reader_v4 (FTYPES=8).
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

L = json.load(open(".cache/book6_lanes.json"))
BY = {l["idx"]: l for l in L}
D = "Consider the numbers "

PAGES = [
 (1, 300, D+"a, b, c. a is 12. b is 5. a times b equals c. What is c?", True, "v1"),
 (2, 300, D+"a, b, c, d, e, f, g. a is 11. b is 5. a times b equals c. d is 8. d plus e equals c. What is e?", True, "v1"),
 (3, 300, D+"a, b, c. a is 3. b is 9. a times b equals c. What is c?", True, "v1"),
 (4, 300, D+"a, b, c. a is 2. b is 3. a times b equals c. What is c?", True, "v1"),
 (5, 300, D+"a, b, c. a is 20. b is 2. a times b equals c. What is c?", True, "v1"),
 (6, 300, D+"a, b, c. a is 288. b is 204. b plus c equals a. What is c?", True, "v1"),
 (7, 300, D+"a, b, c, d, e. a is 10. b is 7. a times b equals c. When c is divided by 2, the quotient is d and the remainder is e. What is d?", True, "v1"),
 (9, 300, D+"a, b, c. a is 8. b is 2. a plus b equals c. What is c?", True, "v1"),
 (10, 300, D+"a, b, c, d, e, f. a is 108. b is 3. b times c equals a. d times d equals c. e is 3. e times d equals f. What is f?", True, "v1"),
 (11, 300, D+"a, b, c. a is 16. b is 1. c plus b equals a. What is c?", True, "v1"),
 (12, 500, D+"a, b, c, d, e, f, g, h, i, j, k. a is 12. b is 5. a times a equals c. b times b equals d. c plus d equals e. f times f equals e. g is 9. h is 4. g plus h equals i. i plus a equals j. j plus f equals k. What is k?", True, "v1"),
 (13, 300, D+"a, b, c. a is 10. When a is divided by 2, the quotient is b and the remainder is c. What is b?", True, "v1"),
 (14, 300, D+"a, b, c, d, e. a is 92. b is 2. a times b equals c. d is 94. d plus e equals c. What is e?", True, "v1"),
 (15, 300, D+"a, b, c, d, e, f. a is 24. b is 4. b times c equals a. c plus b equals d. e is 2. e times d equals f. What is f?", True, "v1"),
 (17, 300, D+"a, b, c, d, e. a is 18. When a is divided by 2, the quotient is b and the remainder is c. d is 5. d plus e equals b. What is e?", True, "v1"),
 (19, 300, D+"a, b, c. a is 102. b is 3. b times c equals a. What is c?", True, "v1"),
 (20, 300, D+"a, b, c, d, e. a is 6. b is 4. a times b equals c. When c is divided by 2, the quotient is d and the remainder is e. What is d?", True, "v1"),
 (21, 300, D+"a, b. a is 10. a times a equals b. What is b?", True, "v1"),
 (22, 300, D+"a, b, c. a is 66. b is 30. b plus c equals a. What is c?", True, "v1"),
 (23, 300, D+"a, b, c. a is 72. When a is divided by 2, the quotient is b and the remainder is c. What is b?", True, "v1"),
 (24, 500, D+"a, b, c, d, e. a is 4. b is 102. a times b equals c. d is 68. d times e equals c. What is e?", True, "v1"),
 (25, 300, D+"a, b, c. a is 4. b is 7. a times b equals c. What is c?", True, "v1"),
 (26, 300, D+"a, b, c, d, e, f. a is 12. b is 9. a times a equals c. b times b equals d. c plus d equals e. f times f equals e. What is f?", True, "v1"),
 (27, 1500, D+"a, b, c, d, e. a is 132. b is 10. a times b equals c. d is 40. d times e equals c. What is e?", True, "v1"),
 (28, 300, D+"a, b, c, d, e, f, g. a is 1. b is 3. c is 5. d is 15. a plus b equals e. e plus c equals f. f plus d equals g. What is g?", True, "v1"),
 (29, 300, D+"a, b, c, d, e. a is 15. b is 14. a times b equals c. When c is divided by 2, the quotient is d and the remainder is e. What is d?", True, "v1"),
 (30, 300, D+"a, b, c. a is 16. b times b equals a. What is b?", True, "v1"),
 (32, 300, D+"a, b, c, d, e. a is 9. b is 10. a plus b equals c. d is 1. e plus d equals c. What is e?", True, "v1"),
 (33, 300, D+"a, b, c, d, e, f. a is 12. b is 5. a times a equals c. b times b equals d. c plus d equals e. f times f equals e. What is f?", True, "v1"),
 (34, 300, D+"a, b, c, d, e. a is 5. b is 25. c is 125. a plus b equals d. d plus c equals e. What is e?", True, "v1"),
 (35, 1000, D+"a, b, c, d, e, f, g, h, i, j. a is 12. b is 82. a times b equals c. d is 10. e is 84. d times e equals f. f plus g equals c. h is 98. h plus i equals g. What is i?", True, "v1"),
 (36, 300, D+"a, b, c. a is 1. b is 1. b plus c equals a. What is c?", True, "v1"),
 (37, 300, D+"a, b, c, d, e, f, g, h, i. a is 20. b is 25. a plus b equals c. d is 45. c plus d equals e. f is 12. e plus f equals g. When g is divided by 2, the quotient is h and the remainder is i. What is h?", True, "v1"),
]
PAGES = [p for p in PAGES if not p[4].startswith("SKIP")]

MACRO_PAGES = [
 (8, 300,
  D+"a, b, c, d. a is 20. When 9 times a is divided by 5, the quotient is b. c is 32. b plus c equals d. What is d?",
  D+"a, b, c, d, e, f, g. a is 20. b is 9. b times a equals c. When c is divided by 5, the quotient is d and the remainder is e. f is 32. d plus f equals g. What is g?",
  [{"ftype": "given", "var": 0, "value": 20, "spans": []},
   {"ftype": "macro", "name": "FRAC_OF", "a": 9, "k": 5, "x": 0, "result": 1},
   {"ftype": "given", "var": 2, "value": 32, "spans": []},
   {"ftype": "rel", "op": "add", "args": [1, 2], "result": 3, "spans": []}],
  3),
 (16, 300,
  D+"a, b, c. a is 100. When a is divided by 4, the quotient is b. What is b?",
  D+"a, b, c, d. a is 100. When a is divided by 4, the quotient is b and the remainder is c. What is b?",
  [{"ftype": "given", "var": 0, "value": 100, "spans": []},
   {"ftype": "macro", "name": "FRAC_OF", "a": 1, "k": 4, "x": 0, "result": 1}],
  1),
]

REGISTRY = [
 (18, "average-mixture"),
 (31, "unit-price-rounding"),
]

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/g20.safetensors")
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[b6t1] gate = {CKPT} (FTYPES=8) | pages {len(PAGES)} + {len(MACRO_PAGES)} macro | registry {len(REGISTRY)} | WIDEST")


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
    ok, votes, best = gate_dialect(dia, m, x["answer"], 740000 + 100 * li)
    if ok and best:
        facs, q = best
        fresh_banked += is_fresh
        banked.append(dict(text=x["problem"], factors=facs, query_var=q,
                           n_vars=24, m=m, decisions=[], mentions=[],
                           solution=[0] * 24,
                           gen=dict(src_idx=x["src_idx"], book=6, tranche=1,
                                    floor="prime", fs=fs, dialect=dia,
                                    gate="5view-vote+key", generation="20")))
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
    ok, votes, best = gate_dialect(pdia, m, x["answer"], 750000 + 100 * li)
    if ok:
        for floor, facs_, dia_, qv in (("macro", mfacs, mdia, q),
                                       ("prime", pf2, pdia, q_p)):
            banked.append(dict(text=x["problem"], factors=facs_, query_var=qv,
                               n_vars=24, m=m, decisions=[], mentions=[],
                               solution=[0] * 24,
                               gen=dict(src_idx=x["src_idx"], book=6, tranche=1,
                                        floor=floor, fs=True, dialect=dia_,
                                        knot=dg_m, grammar=MACRO_GRAMMAR_VERSION,
                                        gate="5view-vote+key(prime-twin)",
                                        generation="20")))
        macro_banked += 1
        print(f"  [MACRO {li}] BANKED both floors, one knot {dg_m[:12]} (votes {votes})")
    else:
        missed.append((li, votes))
        print(f"  [MACRO {li}] prime-twin MISS (votes {votes})")

certs = [dict(lane_idx=li, family=fam, raw=BY[li]["problem"],
              answer=BY[li]["answer"], src_idx=BY[li]["src_idx"],
              book=6, tranche=1) for li, fam in REGISTRY]

with open(".cache/book6_prose_pairs.jsonl", "a") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
json.dump(certs, open(".cache/book6_organ_registry_t1.json", "w"))
rate = fresh_banked / max(fresh_attempted, 1)
print(f"\n[b6t1] banked rows: {len(banked)} ({macro_banked} macro pairs, "
      f"grammar {MACRO_GRAMMAR_VERSION}) | missed: {[m_[0] for m_ in missed]} | "
      f"certificates: {len(certs)}")
print(f"[b6t1] ANNOTATOR FLOOR: fresh first-pass {fresh_banked}/{fresh_attempted}"
      f" = {rate:.2f} (floor 0.75 -> {'HELD' if rate >= 0.75 else 'BREACHED — next tranche narrows'})")
