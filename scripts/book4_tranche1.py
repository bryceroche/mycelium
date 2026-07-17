"""book4_tranche1.py — BOOK 4, tranche 1 (2026-07-17): THE FIRST FLOOR-UP
PAGES. 15 hand dialects (13 prime + 2 MACRO CROWNS — the recursion's
first macro-annotated strangers) through the gen-14 gate (5 views,
vote>=3, key disposes), + 12 organ-registry certificates.

MACRO PROTOCOL (charter pin 3), mechanically asserted per macro page:
  expand_graph(macro_gold) solves to the official answer (level
  invariance), canon(macro_row) == canon(prime_row) (one knot, the
  floor-identity protocol), and THE GATE RUNS ON THE PRIME DIALECT —
  the standing trust story byte-unchanged. Banked pairs carry
  floor={"macro"|"prime"}; the pair is arm-(iii)'s corpus by construction.
Frame-strip flags (fs) mark pages whose knowns were lexically
explicitated per the rulebook (unknowns never gifted).
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

# (lane_idx, m, prime_dialect, fs_flag)
PAGES = [
 (0, 300, D+"a, b, c, d. a is 3. d is 8. a times b equals c. b plus d equals c. What is b?", False),
 (5, 300, D+"a, b, c, d, e, f. a is 12. b is 5. a times a equals c. b times b equals d. c plus d equals e. f times f equals e. What is f?", True),
 (10, 300, D+"a, b, c, d, e, f, g, h, i, j, k. a is 7. b is 19. a plus b equals c. When c is divided by 2, the quotient is d and the remainder is e. b exceeds a by f. When f is divided by 2, the quotient is g and the remainder is h. d times d equals i. g times g equals j. i plus j equals k. What is k?", True),
 (11, 300, D+"a, b, c, d, e. a is 30. b is 10. a plus b equals c. c times d equals e. It is known that e is 120. What is d?", True),
 (12, 300, D+"a, b, c, d, e, f, g. a is 1. b is 4. a plus b equals c. d is 2. e is 3. d plus e equals f. c times f equals g. What is g?", True),
 (13, 300, D+"a, b, c, d. a is 2. a times b equals c. d is 5. b plus d equals c. What is b?", False),
 (15, 300, D+"a, b, c, d, e, f, g. b is 3. b times a equals c. It is known that c is 9. a times a equals d. d plus c equals e. g is 2. e exceeds f by g. What is f?", True),
 (26, 300, D+"a, b, c. a is 108. When a is divided by 27, the quotient is b and the remainder is c. What is b?", False),
 (27, 300, D+"a, b, c, d, e. a is 4. b is 16. a plus b equals c. When c is divided by 2, the quotient is d and the remainder is e. What is d?", False),
 (38, 300, D+"a, b, c, d, e. a is 8. b is 6. a times b equals c. When c is divided by 2, the quotient is d and the remainder is e. What is d?", True),
 (40, 300, D+"a, b, c, d. a times a equals b. It is known that b is 225. c is 2. a times c equals d. What is d?", False),
 (57, 300, D+"a, b, c, d, e, f, g. a is 99. When a is divided by 11, the quotient is b and the remainder is c. d is 5. b times d equals e. f is 6. e plus f equals g. What is g?", True),
 (64, 300, D+"a, b, c, d, e, f, g, h. b is 3. b times a equals c. e is 1. c exceeds d by e. It is known that d is 5. a times a equals f. f plus a equals g. g plus e equals h. What is h?", False),
]

# MACRO PAGES: (lane_idx, m, macro_dialect, prime_dialect, macro_factors, query_var)
MACRO_PAGES = [
 # [3] max of -5r^2+40r-12 = 68 at the vertex r=4 (fs: vertex explicitated)
 (3, 300,
  D+"a, b, c, d, e. a is 4. a times a equals b. 40 times a minus 5 times b equals c. d is 12. c exceeds e by d. What is e?",
  D+"a, b, c, d, e, f, g, h, i. a is 4. a times a equals b. c is 40. c times a equals d. e is 5. e times b equals f. d exceeds f by g. h is 12. g exceeds i by h. What is i?",
  [{"ftype": "given", "var": 0, "value": 4, "spans": []},
   {"ftype": "rel", "op": "mul", "args": [0, 0], "result": 1, "spans": []},
   {"ftype": "macro", "name": "OP_APPLY", "op": "sub", "k1": 40, "x": 0,
    "k2": 5, "y": 1, "result": 2},
   {"ftype": "given", "var": 3, "value": 12, "spans": []},
   {"ftype": "rel", "op": "add", "args": [4, 3], "result": 2, "spans": []}],
  4),
 # [20] f(f(x)) points (3,5),(2,1): ab+cd = 3*5+2*1 = 17 (fs: lookups explicitated)
 (20, 300,
  D+"a, b, c. a is 5. b is 1. 3 times a plus 2 times b equals c. What is c?",
  D+"a, b, c, d, e, f, g. a is 5. b is 1. c is 3. c times a equals d. e is 2. e times b equals f. d plus f equals g. What is g?",
  [{"ftype": "given", "var": 0, "value": 5, "spans": []},
   {"ftype": "given", "var": 1, "value": 1, "spans": []},
   {"ftype": "macro", "name": "OP_APPLY", "op": "add", "k1": 3, "x": 0,
    "k2": 2, "y": 1, "result": 2}],
  2),
]

REGISTRY = [
 (1, "rate-noninteger-intermediate"),      # [45]/[7] kin — the frame family
 (2, "consecutive-sum-search"),
 (8, "function-lookup-chain"),
 (14, "diophantine-optimization"),
 (16, "radical-form-answer"),
 (17, "nested-radical"),
 (21, "symbolic-identity"),
 (39, "unit-fraction-family"),             # chained-fdiv kin — autopsy docket
 (54, "floor-abs-lexical"),
 (56, "area-perimeter-diophantine"),
 (59, "functional-iteration"),
 (66, "functional-equation"),
]

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/phase1_gen14_head.safetensors")
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[b4t1] gate = {CKPT} | pages {len(PAGES)} prime + {len(MACRO_PAGES)} macro | registry {len(REGISTRY)}")


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
    """5-view vote on a dialect; returns (banked, votes, gold_parse)."""
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
for li, m, dia, fs in PAGES:
    x = BY[li]
    ok, votes, best = gate_dialect(dia, m, x["answer"], 120000 + 100 * li)
    if ok and best:
        facs, q = best
        banked.append(dict(text=x["problem"], factors=facs, query_var=q,
                           n_vars=24, m=m, decisions=[], mentions=[],
                           solution=[0] * 24,
                           gen=dict(src_idx=x["src_idx"], book=4, tranche=1,
                                    floor="prime", fs=fs, dialect=dia,
                                    gate="5view-vote+key", generation="14")))
        print(f"  [page {li}] BANKED (votes {votes})")
    else:
        missed.append((li, votes))
        print(f"  [page {li}] MISS (votes {votes}) -> retry bench")

macro_banked = 0
for li, m, mdia, pdia, mfacs, q in MACRO_PAGES:
    x = BY[li]
    # mechanical asserts: level invariance + one-knot identity
    pfacs, nv = expand_graph(mfacs, 24)
    a = solve2(pfacs, q, {"n_vars": 24, "m": m})
    assert a == x["answer"], (li, a, x["answer"])
    mrow = {"factors": mfacs, "n_vars": 24, "query_var": q}
    prow = {"factors": pfacs, "n_vars": nv, "query_var": q}
    dg_m, _ = canon(mrow); dg_p, _ = canon(prow)
    assert dg_m == dg_p, (li, dg_m, dg_p)
    ok, votes, best = gate_dialect(pdia, m, x["answer"], 130000 + 100 * li)
    if ok:
        for floor, facs_, dia_ in (("macro", mfacs, mdia), ("prime", pfacs, pdia)):
            banked.append(dict(text=x["problem"], factors=facs_, query_var=q,
                               n_vars=24, m=m, decisions=[], mentions=[],
                               solution=[0] * 24,
                               gen=dict(src_idx=x["src_idx"], book=4, tranche=1,
                                        floor=floor, fs=True, dialect=dia_,
                                        knot=dg_m, grammar=MACRO_GRAMMAR_VERSION,
                                        gate="5view-vote+key(prime-twin)",
                                        generation="14")))
        macro_banked += 1
        print(f"  [MACRO {li}] BANKED both floors, one knot {dg_m[:12]} (votes {votes})")
    else:
        missed.append((li, votes))
        print(f"  [MACRO {li}] prime-twin MISS (votes {votes}) -> retry bench")

certs = [dict(lane_idx=li, family=fam, raw=BY[li]["problem"],
              answer=BY[li]["answer"], src_idx=BY[li]["src_idx"],
              book=4, tranche=1) for li, fam in REGISTRY]

with open(".cache/book4_prose_pairs.jsonl", "w") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
json.dump(certs, open(".cache/book4_organ_registry_t1.json", "w"))
n_rows = len(banked)
print(f"\n[b4t1] banked rows: {n_rows} ({macro_banked} macro pairs) | "
      f"missed -> retry: {[m_[0] for m_ in missed]} | certificates: {len(certs)}")
print(f"[b4t1] -> .cache/book4_prose_pairs.jsonl + book4_organ_registry_t1.json")
