"""book4_tranche3.py — BOOK 4, tranche 3 (2026-07-17): the count-cure
bench (3 v2 retries — count casualties get count cures: fewer vars via
mul-inverse and given-path swaps per the digit curve's measured remedy)
+ 4 fresh prime pages (the distance family x3 + vertex-form surgery)
+ 1 macro sub-crown ([36]: sum-of-coefficients = eval-at-1 -> 3*4 - 7*1)
+ 13 registry certificates incl. SECOND certificates for unit-fraction
and vieta families (repeat customers price the next admissions).
Same gate, same law.
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
os.environ.setdefault("ALG_DUP", "1")
from phase1_algebra_head import T_ALG, L_FAC, build_params, forward, decode, sent_indices, TOKENIZER_JSON
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
 # count-cure retries
 (22, 300, D+"a, b, c, d, e, f. a is 28. a times b equals c. It is known that c is 140. d is 45. b times d equals e. f times f equals e. What is f?", True, "v2"),
 (33, 300, D+"a, b, c, d, e, f, g, h, i. a times a equals b. It is known that b is 64. c is 1. a plus c equals d. d times d equals e. f is 12. f times f equals g. e plus g equals h. i times i equals h. What is i?", True, "v2"),
 (100, 300, D+"a, b, c, d, e, f, g, h, i, j, k, l. a is 2. b is 9. When b is divided by 4, the quotient is c and the remainder is d. e is 1. c plus e equals f. g is 81. When g is divided by 16, the quotient is h and the remainder is i. h plus e equals j. a plus f equals k. k plus j equals l. What is l?", True, "v2"),
 # fresh pages
 (9, 300, D+"a, b, c, d, e. a is 6. a times b equals c. It is known that c is 12. d is 5. b times d equals e. What is e?", True, "v1"),
 (43, 300, D+"a, b, c, d, e, f, g, h, i, j. a is 12. b is 5. a times a equals c. b times b equals d. c plus d equals e. f times f equals e. g is 6. h is 3. g plus h equals i. f exceeds j by i. What is j?", True, "v1"),
 (46, 300, D+"a, b, c, d, e, f, g, h. a is 12. b is 5. a times a equals c. b times b equals d. c plus d equals e. f times f equals e. g is 3. f exceeds h by g. What is h?", True, "v1"),
 (4, 300, D+"a, b, c, d, e, f, g. a is 9. a times b equals c. It is known that c is 18. d is 1. b times d equals e. f is 5. f exceeds g by e. What is g?", True, "v1"),
]

MACRO_PAGES = [
 # [36] sum of coefficients = evaluate at x=1: 3*(4) - 7*(1) = 5
 # (fs: eval-at-1 + inner sums explicitated; sign of the second group folded)
 (36, 300,
  D+"a, b, c. a is 4. b is 1. 3 times a minus 7 times b equals c. What is c?",
  D+"a, b, c, d, e, f, g. a is 4. b is 1. c is 3. c times a equals d. e is 7. e times b equals f. d exceeds f by g. What is g?",
  [{"ftype": "given", "var": 0, "value": 4, "spans": []},
   {"ftype": "given", "var": 1, "value": 1, "spans": []},
   {"ftype": "macro", "name": "OP_APPLY", "op": "sub", "k1": 3, "x": 0,
    "k2": 7, "y": 1, "result": 2}],
  2),
]

REGISTRY = [
 (6, "decimal-coefficients"),
 (7, "negative-range-selection"),
 (18, "discriminant-square-enumeration"),
 (19, "decimal-place-value"),
 (37, "radical-rationalize"),
 (41, "exponent-identity"),          # 2nd certificate — family repeat
 (42, "negative-roots"),
 (44, "negative-composition"),
 (45, "unit-fraction-family"),       # 2nd certificate — the autopsy docket's kin
 (47, "prime-roots"),
 (48, "discriminant-enumeration"),
 (49, "rational-function-asymptote"),
 (50, "vieta-root-sum"),             # 2nd certificate — family repeat
]

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/phase1_gen14_head.safetensors")
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[b4t3] gate = {CKPT} | pages {len(PAGES)} (3 v2) + {len(MACRO_PAGES)} macro | registry {len(REGISTRY)}")


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
    ok, votes, best = gate_dialect(dia, m, x["answer"], 180000 + 100 * li)
    if ok and best:
        facs, q = best
        banked.append(dict(text=x["problem"], factors=facs, query_var=q,
                           n_vars=24, m=m, decisions=[], mentions=[],
                           solution=[0] * 24,
                           gen=dict(src_idx=x["src_idx"], book=4, tranche=3,
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
    ok, votes, best = gate_dialect(pdia, m, x["answer"], 190000 + 100 * li)
    if ok:
        for floor, facs_, dia_ in (("macro", mfacs, mdia), ("prime", pfacs, pdia)):
            banked.append(dict(text=x["problem"], factors=facs_, query_var=q,
                               n_vars=24, m=m, decisions=[], mentions=[],
                               solution=[0] * 24,
                               gen=dict(src_idx=x["src_idx"], book=4, tranche=3,
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
              book=4, tranche=3) for li, fam in REGISTRY]

with open(".cache/book4_prose_pairs.jsonl", "a") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
json.dump(certs, open(".cache/book4_organ_registry_t3.json", "w"))
print(f"\n[b4t3] banked rows: {len(banked)} ({macro_banked} macro pairs) | "
      f"missed: {[m_[0] for m_ in missed]} | certificates: {len(certs)}")
