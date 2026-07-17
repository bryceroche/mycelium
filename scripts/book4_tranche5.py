"""book4_tranche5.py — BOOK 4, tranche 5 (2026-07-17): 11 prime pages
(incl. [85], a 20-var/19-factor ZERO-FDIV WALL PROBE — the clean count
control beside the fdiv-mass markers [100]/[73]) + 2 macro crowns
([82]: the (x-y)^2 identity's SECOND instance — same crown family, new
stranger, the golden-ratio equation; [105]: 5*48+1 factoring-max) + 7
certificates incl. the rate family's THIRD. Same gate, same law.
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
 (81, 300, D+"a, b, c. a is 4. b is 3. a exceeds b by c. What is c?", True, "v1"),
 (84, 300, D+"a, b, c, d, e. a is 18. b is 9. a exceeds c by b. d is 4. c exceeds e by d. What is e?", True, "v1"),
 (87, 300, D+"a, b, c, d, e. a is 216. b is 90. a exceeds c by b. d is 36. c exceeds e by d. What is e?", True, "v1"),
 (90, 300, D+"a, b, c. a is 3. b is 3. a exceeds b by c. What is c?", True, "v1"),
 (91, 300, D+"a, b, c. a is 3. b is 5. a plus b equals c. What is c?", True, "v1"),
 (92, 300, D+"a, b, c, d, e. a is 5. a times b equals c. It is known that c is 5. d is 2. b times d equals e. What is e?", True, "v1"),
 (96, 300, D+"a, b, c, d, e, f, g, h, i, j, k, l, m, n, o. a is 169. b is 25. a exceeds c by b. d times d equals c. e is 7. f is 4. f times e equals g. h is 13. i is 5. h plus i equals j. h exceeds k by i. j plus k equals l. m is 2. m times l equals n. g plus n equals o. What is o?", True, "v1"),
 (99, 300, D+"a, b, c, d, e. a is 4. a times b equals c. It is known that c is 12. d is 10. b times d equals e. What is e?", True, "v1"),
 (104, 300, D+"a, b, c, d, e, f, g, h, i. a is 81. b is 27. a plus b equals c. d is 9. c plus d equals e. f is 3. e plus f equals g. h is 1. g plus h equals i. What is i?", True, "v1"),
 (106, 300, D+"a, b, c, d, e. a is 15. a times a equals b. c is 14. c times c equals d. b exceeds d by e. What is e?", True, "v1"),
 # THE WALL PROBE: 20 vars, 19 factors, ZERO fdivs (clean count control)
 (85, 300, D+"a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t. q is 1. c exceeds d by q. d plus c equals e. f is 3. f times e equals b. f times b equals g. g plus c equals a. h is 2. h times a equals i. j is 5. j times c equals k. l is 7. l times d equals m. i plus g equals n. n plus k equals o. o plus m equals p. It is known that p is 34. a times b equals r. r times c equals s. s times d equals t. What is t?", True, "v1-wallprobe"),
]

MACRO_PAGES = [
 # [82] x^2=x+1: (Phi-phi)^2 = (Phi+phi)^2 - 4*Phi*phi = 1 + 4 = 5
 # (the [51] identity's SECOND instance; signs stripped fs)
 (82, 300,
  D+"a, b, c. a is 1. b is 1. a plus 4 times b equals c. What is c?",
  D+"a, b, c, d, e. a is 1. b is 1. c is 4. c times b equals d. a plus d equals e. What is e?",
  [{"ftype": "given", "var": 0, "value": 1, "spans": []},
   {"ftype": "given", "var": 1, "value": 1, "spans": []},
   {"ftype": "macro", "name": "OP_APPLY", "op": "add", "k1": 1, "x": 0,
    "k2": 4, "y": 1, "result": 2}],
  2),
 # [105] max n with 5x^2+nx+48 factorable: n = 5*48 + 1*1 = 241
 (105, 300,
  D+"a, b, c. a is 48. b is 1. 5 times a plus b equals c. What is c?",
  D+"a, b, c, d, e. a is 48. b is 1. c is 5. c times a equals d. d plus b equals e. What is e?",
  [{"ftype": "given", "var": 0, "value": 48, "spans": []},
   {"ftype": "given", "var": 1, "value": 1, "spans": []},
   {"ftype": "macro", "name": "OP_APPLY", "op": "add", "k1": 5, "x": 0,
    "k2": 1, "y": 1, "result": 2}],
  2),
]

REGISTRY = [
 (83, "radical-rationalize"),        # 2nd
 (89, "discriminant-identity"),
 (95, "inequality-count"),
 (97, "radical-form-answer"),        # 3rd
 (98, "consecutive-root-count"),
 (101, "vieta-transform"),
 (107, "rate-harmonic"),             # the [45]/[7] frame family's THIRD
]

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/phase1_gen14_head.safetensors")
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[b4t5] gate = {CKPT} | pages {len(PAGES)} (incl. wall probe) + {len(MACRO_PAGES)} macro | registry {len(REGISTRY)}")


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
    ok, votes, best = gate_dialect(dia, m, x["answer"], 220000 + 100 * li)
    if ok and best:
        facs, q = best
        banked.append(dict(text=x["problem"], factors=facs, query_var=q,
                           n_vars=24, m=m, decisions=[], mentions=[],
                           solution=[0] * 24,
                           gen=dict(src_idx=x["src_idx"], book=4, tranche=5,
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
    ok, votes, best = gate_dialect(pdia, m, x["answer"], 230000 + 100 * li)
    if ok:
        for floor, facs_, dia_ in (("macro", mfacs, mdia), ("prime", pfacs, pdia)):
            banked.append(dict(text=x["problem"], factors=facs_, query_var=q,
                               n_vars=24, m=m, decisions=[], mentions=[],
                               solution=[0] * 24,
                               gen=dict(src_idx=x["src_idx"], book=4, tranche=5,
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
              book=4, tranche=5) for li, fam in REGISTRY]

with open(".cache/book4_prose_pairs.jsonl", "a") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
json.dump(certs, open(".cache/book4_organ_registry_t5.json", "w"))
print(f"\n[b4t5] banked rows: {len(banked)} ({macro_banked} macro pairs) | "
      f"missed: {[m_[0] for m_ in missed]} | certificates: {len(certs)}")
