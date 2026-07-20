"""book4_tranche6.py — BOOK 4, tranche 6 (2026-07-20): 9 prime pages +
2 macro crowns ([124]: Vieta sum-of-squares e1^2-2e2 — the identity
family's THIRD; [128]: 2f(3)+3f(-3) add-crown) + 9 certificates incl. a
NEW family (infinite-series) and 2nd/3rd/4th repeats (symmetric-identity,
radical-rationalize, radical-form, lattice-counting). Same gate, same law.
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
 (110, 300, D+"a, b, c, d. a times a equals b. It is known that b is 36. c is 2. a times c equals d. What is d?", True, "v1"),
 (113, 300, D+"a, b, c, d, e. a is 11. b is 4. a plus b equals c. d is 1. c plus d equals e. What is e?", True, "v1"),
 (117, 300, D+"a, b, c, d, e. a is 6. b is 10. a times b equals c. d is 24. c exceeds e by d. What is e?", True, "v1"),
 (118, 300, D+"a, b, c, d. a is 1. b is 12. a plus b equals c. c exceeds d by b. What is d?", True, "v1"),
 (119, 300, D+"a, b, c. b is 3. a plus b equals c. It is known that c is 8. What is a?", True, "v1"),
 (121, 300, D+"a, b, c, d, e. a is 7. a times a equals b. c is 6. c times c equals d. b exceeds d by e. What is e?", True, "v1"),
 (123, 300, D+"a, b, c. a is 16. a times b equals c. It is known that c is 32. What is b?", True, "v1"),
 (125, 300, D+"a, b, c. a is 5. a times b equals c. It is known that c is 25. What is b?", True, "v1"),
 (126, 300, D+"a, b, c. a is 2. b is 4. a times b equals c. What is c?", True, "v1"),
]

MACRO_PAGES = [
 # [124] sum of squares of roots of x^2-13x+4: e1^2 - 2*e2 = 169 - 8 = 161
 (124, 300,
  D+"a, b, c, d. a is 13. a times a equals b. c is 4. b minus 2 times c equals d. What is d?",
  D+"a, b, c, d, e, f. a is 13. a times a equals b. c is 4. d is 2. d times c equals e. b exceeds f by e. What is f?",
  [{"ftype": "given", "var": 0, "value": 13, "spans": []},
   {"ftype": "rel", "op": "mul", "args": [0, 0], "result": 1, "spans": []},
   {"ftype": "given", "var": 2, "value": 4, "spans": []},
   {"ftype": "macro", "name": "OP_APPLY", "op": "sub", "k1": 1, "x": 1,
    "k2": 2, "y": 2, "result": 3}],
  3),
 # [128] 2f(3)+3f(-3) with f evals explicitated: 2*15 + 3*39 = 147
 (128, 300,
  D+"a, b, c. a is 15. b is 39. 2 times a plus 3 times b equals c. What is c?",
  D+"a, b, c, d, e, f, g. a is 15. b is 39. c is 2. c times a equals d. e is 3. e times b equals f. d plus f equals g. What is g?",
  [{"ftype": "given", "var": 0, "value": 15, "spans": []},
   {"ftype": "given", "var": 1, "value": 39, "spans": []},
   {"ftype": "macro", "name": "OP_APPLY", "op": "add", "k1": 2, "x": 0,
    "k2": 3, "y": 1, "result": 2}],
  2),
]

REGISTRY = [
 (108, "solution-count"),
 (109, "radical-form-answer"),       # 4th
 (112, "value-range-system"),
 (114, "lattice-counting"),          # 2nd
 (115, "radical-form-answer"),
 (116, "infinite-series"),           # NEW family
 (120, "radical-rationalize"),       # 3rd
 (122, "discriminant-sign"),
 (127, "symmetric-identity"),        # 2nd
]

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/phase1_gen14_head.safetensors")
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[b4t6] gate = {CKPT} | pages {len(PAGES)} + {len(MACRO_PAGES)} macro | registry {len(REGISTRY)}")


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
    ok, votes, best = gate_dialect(dia, m, x["answer"], 250000 + 100 * li)
    if ok and best:
        facs, q = best
        banked.append(dict(text=x["problem"], factors=facs, query_var=q,
                           n_vars=24, m=m, decisions=[], mentions=[],
                           solution=[0] * 24,
                           gen=dict(src_idx=x["src_idx"], book=4, tranche=6,
                                    floor="prime", fs=fs, ver=ver, dialect=dia,
                                    gate="5view-vote+key", generation="14")))
        print(f"  [page {li}] BANKED (votes {votes})")
    else:
        missed.append((li, votes))
        print(f"  [page {li}] MISS (votes {votes})")

macro_banked = 0
for li, m, mdia, pdia, mfacs, q in MACRO_PAGES:
    x = BY[li]
    pfacs, nv = expand_graph(mfacs, 24)
    a = solve2(pfacs, q, {"n_vars": 24, "m": m})
    assert a == x["answer"], (li, a, x["answer"])
    dg_m, _ = canon({"factors": mfacs, "n_vars": 24, "query_var": q})
    dg_p, _ = canon({"factors": pfacs, "n_vars": nv, "query_var": q})
    assert dg_m == dg_p, (li, dg_m, dg_p)
    ok, votes, best = gate_dialect(pdia, m, x["answer"], 260000 + 100 * li)
    if ok:
        for floor, facs_, dia_ in (("macro", mfacs, mdia), ("prime", pfacs, pdia)):
            banked.append(dict(text=x["problem"], factors=facs_, query_var=q,
                               n_vars=24, m=m, decisions=[], mentions=[],
                               solution=[0] * 24,
                               gen=dict(src_idx=x["src_idx"], book=4, tranche=6,
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
              book=4, tranche=6) for li, fam in REGISTRY]

with open(".cache/book4_prose_pairs.jsonl", "a") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
json.dump(certs, open(".cache/book4_organ_registry_t6.json", "w"))
print(f"\n[b4t6] banked rows: {len(banked)} ({macro_banked} macro pairs) | "
      f"missed: {[m_[0] for m_ in missed]} | certificates: {len(certs)}")
