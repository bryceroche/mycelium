"""book5_tranche6.py — BOOK 5, tranche 6 (2026-07-22): the retry bench
first per protocol ([11] cubic corrected, unknown ungifted — the solver
derives a=4 from a^3-3a=52; [15] re-voiced as chained in-cap adds; [13]
re-derived via the PAIRING decomposition 15+180 after the t5 macro page
broke the <=300 cap with 420), then fresh stock idx 16-32: 12 primes +
3 crowns ([17],[22] k>1 OP_APPLY; [18] FRAC_OF-then-sub) + 4
certificates. Gate = crown_reader_v4 (FTYPES=8); crowns bank
floor-paired on PRIME TWINS, one knot.
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
 # [11] RETRY (t5: annotator error, key refused 60) — x+1/x=y, y^3-3y=52; solver finds a=4
 (11, 300, D+"a, b, c, d, e, f. a times a equals b. a times b equals c. d is 3. d times a equals e. c exceeds f by e. It is known that f is 52. What is a?", True, "v2"),
 # [15] RETRY (t5: 2/5 at 275) — ball travel to 4th hit, chained in-cap adds
 (15, 300, D+"a, b, c, d, e, f, g. a is 100. b is 100. a plus b equals c. d is 50. c plus d equals e. f is 25. e plus f equals g. What is g?", True, "v2"),
 # [13] RETRY (t5 macro: prime-twin miss + 420 broke the value cap) — pairing derivation:
 # (2-1)+(4-3)+...+(30-29)=15, remaining evens 32+34+36+38+40=180, 15+180=195
 (13, 300, D+"a, b, c. a is 15. b is 180. a plus b equals c. What is c?", True, "v2"),
 # [19] greatest x with |6x^2-47x+15| prime: (2x-15)(3x-1); 2x-15=1 -> x=8
 (19, 300, D+"a, b, c, d, e. b is 2. e is 15. b times a equals c. c exceeds d by e. It is known that d is 1. What is a?", True, "v1"),
 # [21] revenue price: 130/10 = 13, two-digit divisor voiced as mul-inverse (desk rule)
 (21, 300, D+"a, b, c. a is 130. b is 10. b times c equals a. What is c?", True, "v1"),
 # [23] parabola through (2,3),(4,3): c = 24-16+3 = 11
 (23, 300, D+"a, b, c, d, e. a is 24. b is 16. a exceeds c by b. d is 3. c plus d equals e. What is e?", True, "v1"),
 # [24] AB^2 with origin midpoint: 1^2 + 7^2 = 50 (the isq door)
 (24, 300, D+"a, b, c, d, e. a is 1. b is 7. a times a equals c. b times b equals d. c plus d equals e. What is e?", True, "v1"),
 # [25] 1/x+1/y=1/18 min sum: 30+45 = 75
 (25, 300, D+"a, b, c. a is 30. b is 45. a plus b equals c. What is c?", True, "v1"),
 # [26] 13^{3n}=13^{-(n-24)}: 4n=24 -> n=6
 (26, 300, D+"a, b, c. a is 24. When a is divided by 4, the quotient is b and the remainder is c. What is b?", True, "v1"),
 # [27] floor(146.41) - 144 = 2, sub voiced as add
 (27, 300, D+"a, b, c. a is 146. b is 144. b plus c equals a. What is c?", True, "v1"),
 # [29] n(n-3)/2=9 diagonals: a(a-3)=18 -> a=6, solver derives
 (29, 300, D+"a, b, c, d. b is 3. a exceeds c by b. a times c equals d. It is known that d is 18. What is a?", True, "v1"),
 # [30] vertex 3, intercept -2: other intercept 3+5 = 8
 (30, 300, D+"a, b, c. a is 3. b is 5. a plus b equals c. What is c?", True, "v1"),
 # [32] rationalized (8*sqrt2+4)/7: A+B+C+D = 8+2+4+7 = 21
 (32, 300, D+"a, b, c, d, e, f, g. a is 8. b is 2. c is 4. d is 7. a plus b equals e. e plus c equals f. f plus d equals g. What is g?", True, "v1"),
 # [28] (1,13) on 2y=3f(4x)+5: 3*7=21, +5=26, /2=13, +1=14
 (28, 300, D+"a, b, c, d, e, f, g, h, i. a is 3. b is 7. a times b equals c. d is 5. c plus d equals e. When e is divided by 2, the quotient is f and the remainder is g. h is 1. f plus h equals i. What is i?", True, "v1"),
]
PAGES = [p for p in PAGES if not p[4].startswith("SKIP")]

MACRO_PAGES = [
 # [17] degree f(x^3)*g(x^2) = 3*4 + 2*5 = 22: OP_APPLY add 3x+2y
 (17, 300,
  D+"a, b, c. a is 4. b is 5. 3 times a plus 2 times b equals c. What is c?",
  D+"a, b, c, d, e, f, g. a is 4. b is 5. c is 3. c times a equals d. e is 2. e times b equals f. d plus f equals g. What is g?",
  [{"ftype": "given", "var": 0, "value": 4, "spans": []},
   {"ftype": "given", "var": 1, "value": 5, "spans": []},
   {"ftype": "macro", "name": "OP_APPLY", "op": "add", "k1": 3, "x": 0,
    "k2": 2, "y": 1, "result": 2}],
  2),
 # [22] 2x^2-mx+n, roots sum 6 product 10: m+n = 2*6 + 2*10 = 32
 (22, 300,
  D+"a, b, c. a is 6. b is 10. 2 times a plus 2 times b equals c. What is c?",
  D+"a, b, c, d, e, f, g. a is 6. b is 10. c is 2. c times a equals d. e is 2. e times b equals f. d plus f equals g. What is g?",
  [{"ftype": "given", "var": 0, "value": 6, "spans": []},
   {"ftype": "given", "var": 1, "value": 10, "spans": []},
   {"ftype": "macro", "name": "OP_APPLY", "op": "add", "k1": 2, "x": 0,
    "k2": 2, "y": 1, "result": 2}],
  2),
 # [18] vertex x of parabola through (-1,7),(5,7): half of 6 is 3, 3-1 = 2:
 # FRAC_OF(1,2)(6) then sub
 (18, 300,
  D+"a, b, c, d. a is 6. When a is divided by 2, the quotient is b. c is 1. b exceeds d by c. What is d?",
  D+"a, b, c, d, e. a is 6. When a is divided by 2, the quotient is b and the remainder is c. d is 1. b exceeds e by d. What is e?",
  [{"ftype": "given", "var": 0, "value": 6, "spans": []},
   {"ftype": "macro", "name": "FRAC_OF", "a": 1, "k": 2, "x": 0, "result": 1},
   {"ftype": "given", "var": 2, "value": 1, "spans": []},
   {"ftype": "rel", "op": "sub", "args": [1, 2], "result": 3, "spans": []}],
  3),
]

REGISTRY = [
 (16, "inverse-function-intersection"),
 (20, "piecewise-composition-count"),
 (31, "vieta-k-cancellation"),
 (33, "symmetric-identity"),          # priced x3 in the third-admission docket
]

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/crown_reader_v4.safetensors")
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[b5t6] gate = {CKPT} (FTYPES=8) | pages {len(PAGES)} + {len(MACRO_PAGES)} macro | registry {len(REGISTRY)}")


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
for li, m, dia, fs, ver in PAGES:
    x = BY[li]
    ok, votes, best = gate_dialect(dia, m, x["answer"], 600000 + 100 * li)
    if ok and best:
        facs, q = best
        banked.append(dict(text=x["problem"], factors=facs, query_var=q,
                           n_vars=24, m=m, decisions=[], mentions=[],
                           solution=[0] * 24,
                           gen=dict(src_idx=x["src_idx"], book=5, tranche=6,
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
    ok, votes, best = gate_dialect(pdia, m, x["answer"], 610000 + 100 * li)
    if ok:
        for floor, facs_, dia_, qv in (("macro", mfacs, mdia, q),
                                       ("prime", pf2, pdia, q_p)):
            banked.append(dict(text=x["problem"], factors=facs_, query_var=qv,
                               n_vars=24, m=m, decisions=[], mentions=[],
                               solution=[0] * 24,
                               gen=dict(src_idx=x["src_idx"], book=5, tranche=6,
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
              book=5, tranche=6) for li, fam in REGISTRY]

with open(".cache/book5_prose_pairs.jsonl", "a") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
json.dump(certs, open(".cache/book5_organ_registry_t6.json", "w"))
print(f"\n[b5t6] banked rows: {len(banked)} ({macro_banked} macro pairs, "
      f"grammar {MACRO_GRAMMAR_VERSION}) | missed: {[m_[0] for m_ in missed]} | "
      f"certificates: {len(certs)}")
