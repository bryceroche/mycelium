"""bootstrap_census.py — THE BOOTSTRAP RE-PRICING (2026-07-11): the gen-6
head parses RAW harvest prose; the lattice votes; the answer key disposes.
Banked triples get their dialect minted by render2 (graph->dialect is the
generator's native direction) — the machine drafts the isotopies.
THE CENSUS (the real deliverable): BANKED / NEAR-MISS (parse-carried,
answer-missed — bootstrap friction) / KNOTTED (structural refusal — the
organ's actual customer list, first empirical sizing).
DIVERSITY GUARD (Goodhart clause): mouth-distance of banked problems' PROSE
— if the machine only banks the most native-reading prose, hand annotation
stays load-bearing for width (the oracle certifies correctness, not
diversity)."""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
import re

h = [json.loads(l) for l in open(".cache/math_harvest_v0.jsonl")]
pool = [x for x in h if x["level"] in ("Level 1", "Level 2", "Level 3")
        and len(x["problem"]) < 300 and "asy]" not in x["problem"]
        and all(int(n) <= 300 for n in re.findall(r"\d+", x["problem"]))]
pool = pool[:100]
print(f"[census] pool n={len(pool)} (L1-3, in-range, prose tier)")

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(".cache/phase1_gen6_head.safetensors")
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
            (("sel",) if "sel" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0 + bi < n:
                res.append(decode({k: o[k][bi] for k in o}))
    return res

banked, near, knotted = [], [], []
for xi, x in enumerate(pool):
    texts = [x["problem"]] + [permuted_view(x["problem"], 700*xi+k) for k in range(1, 5)]
    parses = parse_batch(texts)
    votes = []
    for facs, q in parses:
        a = solve2(facs, q, {"n_vars": 24, "m": 300})
        if a is not None:
            votes.append(a)
    top, cnt = (Counter(votes).most_common(1)[0] if votes else (None, 0))
    if cnt >= 3 and top == x["answer"]:
        banked.append(xi)
    elif len(votes) >= 2:
        near.append(xi)
    else:
        knotted.append(xi)
    if (xi + 1) % 25 == 0:
        print(f"  ...{xi+1}/{len(pool)}: banked {len(banked)} near {len(near)} knotted {len(knotted)}")

print(f"\n=== THE CENSUS (n={len(pool)}) ===")
print(f"  BANKED (machine-annotated): {len(banked)} ({len(banked)/len(pool):.0%})")
print(f"  NEAR-MISS (friction):       {len(near)} ({len(near)/len(pool):.0%})")
print(f"  KNOTTED (the organ's list): {len(knotted)} ({len(knotted)/len(pool):.0%})")

# diversity guard: mouth distance of banked vs pool
mo = np.load(".cache/recognition_mouth_gen5.npz")
def mouth(texts):
    n = len(texts)
    ids = np.zeros((n, T_ALG), np.int32); msk = np.zeros((n, T_ALG), np.float32)
    for i, t in enumerate(texts):
        e = tok.encode(t)
        L = min(len(e.ids), T_ALG)
        ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0
    st = recompute_states(ids)
    v = (st.astype(np.float32)*msk[:,:,None]).sum(1)/np.maximum(msk.sum(1)[:,None],1)
    v = v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-9)
    d = 1.0 - v @ mo["bank"].T
    return np.sort(d, axis=1)[:, :8].mean(1)
if banked:
    mb = mouth([pool[i]["problem"] for i in banked]).mean()
    mp = mouth([x["problem"] for x in pool]).mean()
    print(f"  DIVERSITY GUARD: banked prose mouth {mb:.4f} vs pool {mp:.4f}"
          f" ({'NARROWING — hand quota stays' if mb < mp - 0.01 else 'no strong narrowing'})")
out = [dict(pool[i], census="banked", generation=6) for i in banked]
json.dump(out, open(".cache/bootstrap_banked.json", "w"))
print(f"  [saved] .cache/bootstrap_banked.json")
