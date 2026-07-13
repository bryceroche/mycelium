"""gen11_census.py — census with a DISJOINT-ROWS option (2026-07-13).
Env: GATE_CKPT, GATE_HW (512), GATE_DUP (1), SKIP_IDX (comma pool indices
to exclude — the prose arms' trained items; the disjointness law applied
to the prose side after v0's banked-15 read as training recall).
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
os.environ["ALG_HW"] = os.environ.get("GATE_HW", "512")
os.environ["ALG_DUP"] = os.environ.get("GATE_DUP", "1")
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
import re

skip = set(int(x) for x in os.environ.get("SKIP_IDX", "").split(",") if x)
tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(os.environ["GATE_CKPT"])
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

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

h = [json.loads(l) for l in open(".cache/math_harvest_v0.jsonl")]
pool = [x for x in h if x["level"] in ("Level 1", "Level 2", "Level 3")
        and len(x["problem"]) < 300 and "asy]" not in x["problem"]
        and all(int(n) <= 300 for n in re.findall(r"\d+", x["problem"]))][:100]
banked = near = knotted = 0
for xi, x in enumerate(pool):
    if xi in skip:
        continue
    texts = [x["problem"]] + [permuted_view(x["problem"], 700*xi+k) for k in range(1, 5)]
    votes = []
    for facs, q in parse_batch(texts):
        a = solve2(facs, q, {"n_vars": 24, "m": 300})
        if a is not None:
            votes.append(a)
    top, cnt = (Counter(votes).most_common(1)[0] if votes else (None, 0))
    if cnt >= 3 and top == x["answer"]:
        banked += 1
    elif len(votes) >= 2:
        near += 1
    else:
        knotted += 1
tot = banked + near + knotted
print(f"[census-disjoint] n={tot} (skipped {len(skip)}): "
      f"banked {banked} / near {near} / knotted {knotted}")
