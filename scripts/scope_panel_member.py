"""scope_panel_member.py — GUT #26/#29 REGISTERED FOLLOW-UP, one member's
testimony (2026-07-20): THE PANEL EXAM on the manufactured blind-spot
specimens. Env: MEMBER_CKPT, MEMBER_HW, MEMBER_DUP, OUT. Parses the 10
scope pairs (20 texts, both phrasings) x 5 views; banks per-text
majority answers. The join asks Bacon's question: do the narrators
share the deception?
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
os.environ["ALG_HW"] = os.environ.get("MEMBER_HW", "512")
os.environ["ALG_DUP"] = os.environ.get("MEMBER_DUP", "1")
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

CKPT = os.environ["MEMBER_CKPT"]
OUT = os.environ["OUT"]
tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys()), f"key mismatch loading {CKPT}"
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[member] {CKPT} (HW={os.environ['ALG_HW']}, DUP={os.environ['ALG_DUP']})")

VALS = [(7, 4), (9, 5), (8, 3), (11, 6), (12, 7), (13, 4), (10, 3), (15, 8),
        (14, 5), (9, 2)]
D = "Consider the numbers "
texts, meta = [], []
for a_, b_ in VALS:
    texts.append(D + f"a, b, c. a is {a_}. b is {b_}. The difference of the "
                 f"squares of a and b equals c. What is c?")
    meta.append((a_, b_, "dsq", a_*a_ - b_*b_))
    texts.append(D + f"a, b, c. a is {a_}. b is {b_}. The square of the "
                 f"difference of a and b equals c. What is c?")
    meta.append((a_, b_, "sqd", (a_-b_)**2))


def parse_batch(tt):
    n = len(tt)
    N = ((n + 7) // 8) * 8
    ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32)
    snt = np.zeros((N, T_ALG), np.int32)
    for i, t in enumerate(tt):
        e = tok.encode(t)
        L = min(len(e.ids), T_ALG)
        ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0
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


results = []
for ti, t in enumerate(texts):
    views = [t] + [permuted_view(t, 240000 + 10 * ti + k) for k in range(1, 5)]
    votes = []
    for facs, q in parse_batch(views):
        a = solve2(facs, q, {"n_vars": 24, "m": 300})
        if a is not None:
            votes.append(a)
    top, cnt = (Counter(votes).most_common(1)[0] if votes else (None, 0))
    a_, b_, kind, gold = meta[ti]
    results.append({"pair": [a_, b_], "kind": kind, "gold": gold,
                    "votes": votes, "majority": top, "count": cnt,
                    "unanimous": bool(cnt == 5)})
json.dump(results, open(OUT, "w"))
print(f"[member] banked -> {OUT}")
