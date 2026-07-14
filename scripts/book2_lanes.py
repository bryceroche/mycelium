"""book2_lanes.py — BOOK 2, stage 1 (2026-07-14): lane classification.

Draws candidates from the harvest OUTSIDE the standing census pool (the
pool is a fixture now — charter pin 1), runs the 5-view gate under the
gate ckpt, and emits per-item lane verdicts:
  L1 machine-banked (vote>=3 == key)  -> free substrate entries
  L2 near-miss (>=2 votes, no bank)   -> repair bench (majority parse dumped)
  L3 knotted                          -> hand-surgery candidates
Output: .cache/book2_lanes.json (verdicts, votes, majority-parse factors
for L2) + .cache/book2_lane1.jsonl (banked entries, generation-stamped).
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
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
import re

N_CAND = int(os.environ.get("B2_CAND", "400"))
h = [json.loads(l) for l in open(".cache/math_harvest_v0.jsonl")]
filt = [x for x in h if x["level"] in ("Level 1", "Level 2", "Level 3")
        and len(x["problem"]) < 300 and "asy]" not in x["problem"]
        and all(int(n) <= 300 for n in re.findall(r"\d+", x["problem"]))]
pool_texts = {x["problem"] for x in filt[:100]}          # the fixture — excluded
cands = [x for x in filt[100:] if x["problem"] not in pool_texts][:N_CAND]
print(f"[book2] candidates: {len(cands)} (census pool excluded as fixture)")

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/phase1_gen9b_head.safetensors")
sd = safe_load(CKPT)
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

lanes, lane1 = [], []
for ci, x in enumerate(cands):
    texts = [x["problem"]] + [permuted_view(x["problem"], 90000 + 10*ci + k)
                              for k in range(1, 5)]
    parses = parse_batch(texts)
    votes, best = [], None
    for facs, q in parses:
        a = solve2(facs, q, {"n_vars": 24, "m": 300})
        if a is not None:
            votes.append(a)
            if best is None or a == x["answer"]:
                best = (facs, q)
    top, cnt = (Counter(votes).most_common(1)[0] if votes else (None, 0))
    if cnt >= 3 and top == x["answer"]:
        lane = "L1"
        lane1.append(dict(raw=x["problem"], answer=x["answer"],
                          subject=x.get("subject"), level=x["level"],
                          lane="machine-banked", gate="lattice-vote+answer-key",
                          generation="9b", book=2))
    elif len(votes) >= 2:
        lane = "L2"
    else:
        lane = "L3"
    lanes.append(dict(idx=ci, lane=lane, votes=votes, answer=x["answer"],
                      level=x["level"], subject=x.get("subject"),
                      problem=x["problem"],
                      parse=(best[0] if lane == "L2" and best else None),
                      query=(best[1] if lane == "L2" and best else None)))
    if (ci + 1) % 50 == 0:
        c = Counter(l["lane"] for l in lanes)
        print(f"  ...{ci+1}/{len(cands)}: {dict(c)}", flush=True)

c = Counter(l["lane"] for l in lanes)
print(f"\n[book2-lanes] {dict(c)} of {len(cands)}")
json.dump(lanes, open(".cache/book2_lanes.json", "w"))
with open(".cache/book2_lane1.jsonl", "w") as f:
    for b in lane1:
        f.write(json.dumps(b) + "\n")
print(f"[book2] lane-1 free entries: {len(lane1)} -> .cache/book2_lane1.jsonl")
print("[book2] repair bench + surgery lists -> .cache/book2_lanes.json")
