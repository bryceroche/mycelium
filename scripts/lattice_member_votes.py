"""lattice_member_votes.py — one panel member's votes (2026-07-13, the
lattice probe). Env: MEMBER_CKPT, MEMBER_HW (512/1024), MEMBER_DUP (0/1),
OUT. Emits 5-view vote lists for bigtest + the census pool; the join
(lattice_join.py) computes cert-v2, the autopsy, and the deep-wrong read.
View seeds are SHARED across members (view-matched panel).
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
os.environ["ALG_HW"] = os.environ.get("MEMBER_HW", "512")
os.environ["ALG_DUP"] = os.environ.get("MEMBER_DUP", "0")
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
import re

CKPT = os.environ["MEMBER_CKPT"]
OUT = os.environ["OUT"]
tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys()), f"key mismatch loading {CKPT}"
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[member] {CKPT} (HW={os.environ['ALG_HW']}, DUP={os.environ['ALG_DUP']})")

def parse_batch(texts):
    n = len(texts)
    N = ((n + 7) // 8) * 8
    ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32)
    snt = np.zeros((N, T_ALG), np.int32)
    for i, t in enumerate(texts):
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
        keys = ("pres","ftype","op","islit","dig","args","res","query") + \
            (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0 + bi < n:
                res.append(decode({k: o[k][bi] for k in o}))
    return res

def votes_for(text, gold_m, seed0):
    texts = [text] + [permuted_view(text, seed0 + k) for k in range(1, 5)]
    votes = []
    for facs, q in parse_batch(texts):
        a = solve2(facs, q, {"n_vars": 24, "m": gold_m})
        votes.append(a if a is not None else None)
    return votes

out = {"ckpt": CKPT, "bigtest": [], "census": []}
rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
for i, r in enumerate(rows):
    out["bigtest"].append(votes_for(r["text"], r.get("m", 60), 40000 + 10 * i))
    if (i + 1) % 300 == 0:
        print(f"  bigtest {i+1}/{len(rows)}", flush=True)

h = [json.loads(l) for l in open(".cache/math_harvest_v0.jsonl")]
pool = [x for x in h if x["level"] in ("Level 1", "Level 2", "Level 3")
        and len(x["problem"]) < 300 and "asy]" not in x["problem"]
        and all(int(n) <= 300 for n in re.findall(r"\d+", x["problem"]))][:100]
for xi, x in enumerate(pool):
    out["census"].append(votes_for(x["problem"], 300, 700 * xi))
json.dump(out, open(OUT, "w"))
print(f"[member] votes -> {OUT}")
