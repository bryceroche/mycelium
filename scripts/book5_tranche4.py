"""book5_tranche4.py — BOOK 5, tranche 4 (2026-07-22): the benched retries
leading — [176] v2 under the NEW DESK RULE (two-digit divisors voice as
mul-inverse) and [185] reworked — + 9 fresh primes + 1 crown ([189]'s
square crown) + 6 certificates. Gate = crown_reader_v4 (FTYPES=8); macro pages gate on
PRIME TWINS per the constitution; crowns bank floor-paired, one knot.
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

L = json.load(open(".cache/book4_lanes.json"))
BY = {l["idx"]: l for l in L}
D = "Consider the numbers "

PAGES = [
 # [176] v2 — THE DESK RULE'S FIRST PAGE: k=10 voiced as mul-inverse
 (176, 300, D+"a, b, c. a is 10. a times b equals c. It is known that c is 120. What is b?", True, "v2"),
 # [185] rework: 49 = 21 + 28 plain
 (185, 300, D+"a, b, c. a is 21. b is 28. a plus b equals c. What is c?", True, "v2"),
 # [187] 1/|m| >= 1/8: m in [-8,8], m!=0: 16 integers
 (187, 300, D+"a, b, c. a is 8. b is 2. a times b equals c. What is c?", True, "v1"),
 # [188] centers dist 17, radii 2+10: closest 17-12 = 5
 (188, 300, D+"a, b, c, d, e. a is 17. b is 2. a exceeds c by b. d is 10. c exceeds e by d. What is e?", True, "v1"),
 # [190] (3,6) -> h: 6^2=36, point (3,36): sum 3+36 = 39
 (189, 300, D+"a, b, c, d. a is 6. a times a equals b. c is 3. b plus c equals d. What is d?", True, "v1"),
 # [190] y = 5+6 = 11 (3-4-5 scaled: dx=8, 10 units -> dy=6)
 (190, 300, D+"a, b, c. a is 5. b is 6. a plus b equals c. What is c?", True, "v1"),
 # [193] degree 7+3+1 = 11
 (193, 300, D+"a, b, c, d, e. a is 7. b is 3. a plus b equals c. d is 1. c plus d equals e. What is e?", True, "v1"),
 # [194] midpoint x-sum = vertex x-sum = 10
 (194, 300, D+"a, b, c. a is 20. When a is divided by 2, the quotient is b and the remainder is c. What is b?", True, "v1"),
 # [196] x^2-2x-8... /(x+2) = x^2... sum of solutions: 4 explicitated 4 = 2+2
 (196, 300, D+"a, b, c. a is 2. b is 2. a plus b equals c. What is c?", True, "v1"),
 # [198] (a+b)^2 = 49, a^2+b^2 = 25... product = (49-25)/2 * ... = 144 = 12^2
 (198, 300, D+"a, b, c. a is 12. a times a equals b. What is b?", True, "v1"),
 # [199] f(-4)=14, f(17.5)... positive difference 21 = 14+7
 (199, 300, D+"a, b, c. a is 14. b is 7. a plus b equals c. What is c?", True, "v1"),
]
PAGES = [p for p in PAGES if not p[4].startswith("SKIP")]

MACRO_PAGES = [
 # [192] largest n for 3x^2+nx+72: n = 3*72+1 = 217 — the factoring-max crown (mg1's shape, second wild instance)
 (192, 300,
  D+"a, b, c. a is 72. b is 1. 3 times a plus b equals c. What is c?",
  D+"a, b, c, d, e. a is 72. b is 1. c is 3. c times a equals d. d plus b equals e. What is e?",
  [{"ftype": "given", "var": 0, "value": 72, "spans": []},
   {"ftype": "given", "var": 1, "value": 1, "spans": []},
   {"ftype": "macro", "name": "OP_APPLY", "op": "add", "k1": 3, "x": 0,
    "k2": 1, "y": 1, "result": 2}],
  2),
]

REGISTRY = [
 (191, "radical-rationalize"),       # 4th
 (195, "factoring-diophantine"),     # 2nd
 (197, "circle-radius-decimal"),
 (172, "ceil-sum-count"),            # carried
 (181, "discriminant-sign"),
 (184, "radical-form-answer"),
]

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/crown_reader_v4.safetensors")
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[b5t4] gate = {CKPT} (FTYPES=8) | pages {len(PAGES)} + {len(MACRO_PAGES)} macro | registry {len(REGISTRY)}")


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
    ok, votes, best = gate_dialect(dia, m, x["answer"], 560000 + 100 * li)
    if ok and best:
        facs, q = best
        banked.append(dict(text=x["problem"], factors=facs, query_var=q,
                           n_vars=24, m=m, decisions=[], mentions=[],
                           solution=[0] * 24,
                           gen=dict(src_idx=x["src_idx"], book=5, tranche=4,
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
    ok, votes, best = gate_dialect(pdia, m, x["answer"], 570000 + 100 * li)
    if ok:
        for floor, facs_, dia_, qv in (("macro", mfacs, mdia, q),
                                       ("prime", pf2, pdia, q_p)):
            banked.append(dict(text=x["problem"], factors=facs_, query_var=qv,
                               n_vars=24, m=m, decisions=[], mentions=[],
                               solution=[0] * 24,
                               gen=dict(src_idx=x["src_idx"], book=5, tranche=4,
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
              book=5, tranche=4) for li, fam in REGISTRY]

with open(".cache/book5_prose_pairs.jsonl", "a") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
json.dump(certs, open(".cache/book5_organ_registry_t4.json", "w"))
print(f"\n[b5t4] banked rows: {len(banked)} ({macro_banked} macro pairs, "
      f"grammar {MACRO_GRAMMAR_VERSION}) | missed: {[m_[0] for m_ in missed]} | "
      f"certificates: {len(certs)}")
