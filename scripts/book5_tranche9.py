"""book5_tranche9.py — BOOK 5, tranche 9 (2026-07-23): widest yet
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

L = json.load(open(".cache/book5_lanes.json"))
BY = {l["idx"]: l for l in L}
D = "Consider the numbers "

PAGES = [
 # RETRIES — mechanism-certain cures
 (36, 300, D+"a, b, c, d, e, f, g. a is 9. b times b equals a. c is 2. b exceeds d by c. c times d equals e. f is 5. e plus f equals g. What is g?", True, "v3"),
 (60, 300, D+"a, b, c, d, e, f, g. a is 4. b is 3. c is 27. d times d equals e. a times e equals f. b times d equals g. c plus g equals f. What is d?", True, "v2"),
 (66, 300, D+"a, b, c. a is 3. b is 2. a times b equals c. What is c?", True, "v2"),
 (69, 300, D+"a, b, c. a is 16. b is 2. a times b equals c. What is c?", True, "v2"),
 (75, 300, D+"a, b, c, d, e. a is 3. b is 9. a plus b equals c. When c is divided by 2, the quotient is d and the remainder is e. What is d?", True, "v2"),
 # FRESH
 (96, 300, D+"a, b, c, d, e, f, g, h, i. a is 12. b is 5. c is 2. d is 1. a plus b equals e. e plus c equals f. f plus d equals g. h is 1. i plus h equals g. What is i?", True, "v1"),
 (98, 300, D+"a, b, c. a is 36. b is 27. b plus c equals a. What is c?", True, "v1"),
 (99, 300, D+"a, b, c. a is 10. When a is divided by 2, the quotient is b and the remainder is c. What is b?", True, "v1"),
 (101, 300, D+"a, b, c, d, e, f, g. a is 17. b is 5. b plus c equals a. When c is divided by 2, the quotient is d and the remainder is e. f is 1. d plus f equals g. What is g?", True, "v1"),
 (102, 4000, D+"a, b, c, d, e, f. a is 130. b is 30. a times b equals c. d is 13. d times e equals c. What is e?", True, "v1"),
 (103, 300, D+"a, b, c. a is 2. b is 4. a times b equals c. What is c?", True, "v1"),
 (104, 2000, D+"a, b, c, d, e, f. b is 36. f is 39. a times a equals c. b times b equals d. c plus d equals e. f times f equals e. What is a?", True, "v1"),
 (105, 300, D+"a, b, c, d, e. a is 3. b is 99. a times b equals c. d is 1. c plus d equals e. What is e?", True, "v1"),
 (106, 300, D+"a, b, c, d, e. a is 15. b is 10. b plus c equals a. d is 1. c plus d equals e. What is e?", True, "v1"),
 (107, 300, D+"a, b, c, d, e, f, g. a is 73. b is 3. a plus b equals c. When c is divided by 2, the quotient is d and the remainder is e. f is 5. g plus f equals d. What is g?", True, "v1"),
 (108, 300, D+"a, b, c. a is 100. b is 80. b plus c equals a. What is c?", True, "v1"),
 (110, 300, D+"a, b, c. a is 50. b is 2. a times b equals c. What is c?", True, "v1"),
 (111, 2500, D+"a, b, c, d, e. a is 40. b is 50. a times b equals c. d is 100. d times e equals c. What is e?", True, "v1"),
 (112, 300, D+"a, b, c. a is 15. b is 3. b plus c equals a. What is c?", True, "v1"),
 (113, 300, D+"a, b, c, d. a is 3. a times a equals b. a times b equals c. What is c?", True, "v1"),
 (114, 300, D+"a, b, c. a is 45. b is 4. b plus c equals a. What is c?", True, "v1"),
 (115, 500, D+"a, b, c, d, e. a is 36. b is 10. a times b equals c. d is 18. d times e equals c. What is e?", True, "v1"),
 (116, 300, D+"a, b, c. a is 1. b is 111. a times b equals c. What is c?", True, "v1"),
 (117, 300, D+"a, b, c, d, e. a is 2. b is 2. a times b equals c. d is 1. c plus d equals e. What is e?", True, "v1"),
 (119, 300, D+"a, b, c, d, e, f, g, h. a is 7. b is 23. c exceeds d by b. a times d equals e. f is 4. f times c equals g. h is 10. g plus h equals e. What is c?", True, "v1"),
 (120, 300, D+"a, b, c, d, e. a is 2. b is 2. a times b equals c. c times c equals d. d times d equals e. What is e?", True, "v1"),
 (121, 300, D+"a, b, c. a is 11. a times a equals b. What is b?", True, "v1"),
 (123, 1000, D+"a, b, c, d, e, f, g, h, i, j, k, l. a is 85. b is 9. a times b equals c. d is 75. e is 5. d times e equals f. g is 90. h is 3. g times h equals i. f plus i equals j. j plus k equals c. What is k?", True, "v1"),
 (127, 300, D+"a, b, c, d, e. a is 2. b is 8. c is 1. a plus b equals d. d plus c equals e. What is e?", True, "v1"),
 (128, 300, D+"a, b, c. a is 60. b is 10. b times c equals a. What is c?", True, "v1"),
 (129, 300, D+"a, b, c, d, e, f, g, h, i. a is 54. b is 5. a times b equals c. d is 48. e is 2. d times e equals f. f plus g equals c. h is 3. h times i equals g. What is i?", True, "v1"),
 (130, 300, D+"a, b, c. a is 36. b is 3. a times b equals c. What is c?", True, "v1"),
 (131, 300, D+"a, b, c, d, e. a is 7. b is 8. a times b equals c. When c is divided by 2, the quotient is d and the remainder is e. What is d?", True, "v1"),
 (132, 300, D+"a, b, c, d, e, f. a is 2. b is 2. a times b equals c. c times c equals d. e is 1. f plus e equals d. What is f?", True, "v1"),
 (133, 300, D+"a, b, c, d, e. a is 6. b is 3. a times b equals c. d is 5. c times d equals e. What is e?", True, "v1"),
]
PAGES = [p for p in PAGES if not p[4].startswith("SKIP")]

MACRO_PAGES = [
 # [126] 3:2 mix, 3/5 of 30 = 18 — THE FIRST WILD a>1 FRAC_OF
 (126, 300,
  D+"a, b, c. a is 30. When 3 times a is divided by 5, the quotient is b. What is b?",
  D+"a, b, c, d, e. a is 30. b is 3. b times a equals c. When c is divided by 5, the quotient is d and the remainder is e. What is d?",
  [{"ftype": "given", "var": 0, "value": 30, "spans": []},
   {"ftype": "macro", "name": "FRAC_OF", "a": 3, "k": 5, "x": 0, "result": 1}],
  1),
 # [109] rhombus area: 24*10/2 = 120 — FRAC_OF over a derived product
 (109, 300,
  D+"a, b, c, d, e. a is 24. b is 10. a times b equals c. When c is divided by 2, the quotient is d. What is d?",
  D+"a, b, c, d, e, f. a is 24. b is 10. a times b equals c. When c is divided by 2, the quotient is d and the remainder is e. What is d?",
  [{"ftype": "given", "var": 0, "value": 24, "spans": []},
   {"ftype": "given", "var": 1, "value": 10, "spans": []},
   {"ftype": "rel", "op": "mul", "args": [0, 1], "result": 2, "spans": []},
   {"ftype": "macro", "name": "FRAC_OF", "a": 1, "k": 2, "x": 2, "result": 3}],
  3),
]

REGISTRY = [
 (97, "indistinct-arrangement"),
 (100, "digit-divisibility-count"),
 (118, "permutation-restriction"),
 (122, "digit-reversal-prime"),
 (125, "coin-average-system"),
]

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/crown_reader_v4.safetensors")
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[b5t9] gate = {CKPT} (FTYPES=8) | pages {len(PAGES)} + {len(MACRO_PAGES)} macro | registry {len(REGISTRY)} | WIDEST")


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
    ok, votes, best = gate_dialect(dia, m, x["answer"], 690000 + 100 * li)
    if ok and best:
        facs, q = best
        fresh_banked += is_fresh
        banked.append(dict(text=x["problem"], factors=facs, query_var=q,
                           n_vars=24, m=m, decisions=[], mentions=[],
                           solution=[0] * 24,
                           gen=dict(src_idx=x["src_idx"], book=5, tranche=9,
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
    ok, votes, best = gate_dialect(pdia, m, x["answer"], 695000 + 100 * li)
    if ok:
        for floor, facs_, dia_, qv in (("macro", mfacs, mdia, q),
                                       ("prime", pf2, pdia, q_p)):
            banked.append(dict(text=x["problem"], factors=facs_, query_var=qv,
                               n_vars=24, m=m, decisions=[], mentions=[],
                               solution=[0] * 24,
                               gen=dict(src_idx=x["src_idx"], book=5, tranche=9,
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
              book=5, tranche=9) for li, fam in REGISTRY]

with open(".cache/book5_prose_pairs.jsonl", "a") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
json.dump(certs, open(".cache/book5_organ_registry_t9.json", "w"))
rate = fresh_banked / max(fresh_attempted, 1)
print(f"\n[b5t9] banked rows: {len(banked)} ({macro_banked} macro pairs, "
      f"grammar {MACRO_GRAMMAR_VERSION}) | missed: {[m_[0] for m_ in missed]} | "
      f"certificates: {len(certs)}")
print(f"[b5t9] ANNOTATOR FLOOR: fresh first-pass {fresh_banked}/{fresh_attempted}"
      f" = {rate:.2f} (floor 0.75 -> {'HELD' if rate >= 0.75 else 'BREACHED — next tranche narrows'})")
