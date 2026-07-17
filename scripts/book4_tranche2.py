"""book4_tranche2.py — BOOK 4, tranche 2 (2026-07-17): v2 retry surgery on
the three diagnosed misses + 9 fresh prime pages + 1 macro crown (the
affine k=1 leg's first production page) + 8 registry certificates.

v2 mechanism fixes, per the tranche-1 diagnoses: [26]/[57] swap the fdiv
derived-value digit path (the live autopsy specimen's mechanism) for
MUL-INVERSE phrasing; [10] sheds both remainder chains the same way.
Same gate, same law: 5-view vote >= 3, key disposes, macro rows bank
through prime twins, one knot per pair.
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
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

L = json.load(open(".cache/book4_lanes.json"))
BY = {l["idx"]: l for l in L}
D = "Consider the numbers "

PAGES = [
 # v2 retries (mechanism-fixed)
 (10, 300, D+"a, b, c, d, e, f, g, h, i, j. a is 7. b is 19. a plus b equals c. d is 2. d times e equals c. b exceeds a by f. d times g equals f. e times e equals h. g times g equals i. h plus i equals j. What is j?", True, "v2"),
 (26, 300, D+"a, b, c. a is 27. a times b equals c. It is known that c is 108. What is b?", False, "v2"),
 (57, 300, D+"a, b, c, d, e, f, g. a is 11. a times b equals c. It is known that c is 99. d is 5. b times d equals e. f is 6. e plus f equals g. What is g?", True, "v2"),
 # fresh pages
 (75, 300, D+"a, b, c, d, e. a is 25. When a is divided by 9, the quotient is b and the remainder is c. d is 1. b plus d equals e. What is e?", True, "v1"),
 (86, 300, D+"a, b, c, d. a is 6. a times a equals b. c is 34. b exceeds d by c. What is d?", True, "v1"),
 (100, 300, D+"a, b, c, d, e, f, g, h, i, j, k, l, m, n, o. a is 3. When a is divided by 2, the quotient is b and the remainder is c. d is 1. b plus d equals e. f is 9. When f is divided by 4, the quotient is g and the remainder is h. g plus d equals i. j is 81. When j is divided by 16, the quotient is k and the remainder is l. k plus d equals m. e plus i equals n. n plus m equals o. What is o?", True, "v1"),
 (111, 300, D+"a, b, c, d, e, f, g. a is 8. a times b equals c. d is 9. c plus d equals e. It is known that e is 81. f is 4. b plus f equals g. What is g?", True, "v1"),
 (22, 300, D+"a, b, c, d, e, f. a is 140. When a is divided by 28, the quotient is b and the remainder is c. d is 45. b times d equals e. f times f equals e. What is f?", True, "v1"),
 (29, 300, D+"a, b, c, d, e, f, g, h, i, j, k, l, m. a is 3. b is 7. a times b equals c. d is 5. e is 4. d times e equals f. g is 6. h is 3. g times h equals i. j is 2. f plus i equals k. k plus j equals l. l exceeds m by c. What is m?", True, "v1"),
 (31, 300, D+"a, b, c, d. a is 7. b is 2. a plus b equals c. d times d equals c. What is d?", True, "v1"),
 (33, 300, D+"a, b, c, d, e, f, g, h, i, j, k, l. a is 100. b is 6. b times b equals c. a exceeds c by d. e times e equals d. f is 1. e plus f equals g. g times g equals h. i is 12. i times i equals j. h plus j equals k. l times l equals k. What is l?", True, "v1"),
 (34, 300, D+"a, b, c, d, e. a is 22. b is 9. a exceeds c by b. d is 2. c exceeds e by d. What is e?", True, "v1"),
]

MACRO_PAGES = [
 # [28] n=x^2+2x+17, d=2x+5, quotient x remainder 7 -> x^2+3x=10 -> x=2
 # (fs: the polynomial-division identity explicitated); affine crown k1=1
 (28, 300,
  D+"a, b, c. a times a equals b. b plus 3 times a equals c. It is known that c is 10. What is a?",
  D+"a, b, c, d, e. a times a equals b. c is 3. c times a equals d. b plus d equals e. It is known that e is 10. What is a?",
  [{"ftype": "rel", "op": "mul", "args": [0, 0], "result": 1, "spans": []},
   {"ftype": "macro", "name": "OP_APPLY", "op": "add", "k1": 1, "x": 1,
    "k2": 3, "y": 0, "result": 2},
   {"ftype": "given", "var": 2, "value": 10, "spans": []}],
  0),
]

REGISTRY = [
 (80, "recurrence-relation"),
 (94, "abs-inequality-count"),
 (102, "geometric-divisibility"),
 (103, "vieta-root-sum"),
 (24, "vieta-enumeration"),
 (25, "piecewise-inverse"),
 (32, "value-range"),
 (35, "value-range-noninteger"),
]

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/phase1_gen14_head.safetensors")
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[b4t2] gate = {CKPT} | pages {len(PAGES)} (3 v2) + {len(MACRO_PAGES)} macro | registry {len(REGISTRY)}")


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
        keys = ("pres", "ftype", "op", "islit", "dig", "args", "res", "query") + \
            (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0 + bi < n:
                res.append(decode({k: o[k][bi] for k in o}))
    return res


def gate_dialect(text, m, answer, seed0):
    texts = [text] + [permuted_view(text, seed0 + k) for k in range(1, 5)]
    parses = parse_batch(texts)
    votes, best = [], None
    for facs, q in parses:
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
    ok, votes, best = gate_dialect(dia, m, x["answer"], 140000 + 100 * li)
    if ok and best:
        facs, q = best
        banked.append(dict(text=x["problem"], factors=facs, query_var=q,
                           n_vars=24, m=m, decisions=[], mentions=[],
                           solution=[0] * 24,
                           gen=dict(src_idx=x["src_idx"], book=4, tranche=2,
                                    floor="prime", fs=fs, ver=ver, dialect=dia,
                                    gate="5view-vote+key", generation="14")))
        print(f"  [page {li} {ver}] BANKED (votes {votes})")
    else:
        missed.append((li, votes))
        print(f"  [page {li} {ver}] MISS (votes {votes})")

macro_banked = 0
for li, m, mdia, pdia, mfacs, q in MACRO_PAGES:
    x = BY[li]
    pfacs, nv = expand_graph(mfacs, 24)
    a = solve2(pfacs, q, {"n_vars": 24, "m": m})
    assert a == x["answer"], (li, a, x["answer"])
    dg_m, _ = canon({"factors": mfacs, "n_vars": 24, "query_var": q})
    dg_p, _ = canon({"factors": pfacs, "n_vars": nv, "query_var": q})
    assert dg_m == dg_p, (li, dg_m, dg_p)
    ok, votes, best = gate_dialect(pdia, m, x["answer"], 150000 + 100 * li)
    if ok:
        for floor, facs_, dia_ in (("macro", mfacs, mdia), ("prime", pfacs, pdia)):
            banked.append(dict(text=x["problem"], factors=facs_, query_var=q,
                               n_vars=24, m=m, decisions=[], mentions=[],
                               solution=[0] * 24,
                               gen=dict(src_idx=x["src_idx"], book=4, tranche=2,
                                        floor=floor, fs=True, dialect=dia_,
                                        knot=dg_m, grammar=MACRO_GRAMMAR_VERSION,
                                        gate="5view-vote+key(prime-twin)",
                                        generation="14")))
        macro_banked += 1
        print(f"  [MACRO {li}] BANKED both floors, one knot {dg_m[:12]} (votes {votes})")
    else:
        missed.append((li, votes))
        print(f"  [MACRO {li}] prime-twin MISS (votes {votes})")

certs = [dict(lane_idx=li, family=fam, raw=BY[li]["problem"],
              answer=BY[li]["answer"], src_idx=BY[li]["src_idx"],
              book=4, tranche=2) for li, fam in REGISTRY]

with open(".cache/book4_prose_pairs.jsonl", "a") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
json.dump(certs, open(".cache/book4_organ_registry_t2.json", "w"))
print(f"\n[b4t2] banked rows: {len(banked)} ({macro_banked} macro pairs) | "
      f"missed: {[m_[0] for m_ in missed]} | certificates: {len(certs)}")
print("[b4t2] appended -> .cache/book4_prose_pairs.jsonl")
