"""book4_lanes.py — BOOK 4, stage 1 (2026-07-17): lane classification on
the HARDER STRATA (charter pin 2). Draws L4/L5 harvest candidates outside
every prior book's sources and the census fixture, runs the 5-view gate
under GEN-14, and emits per-item lane verdicts:
  L1 machine-banked (vote>=3 == key)  -> free substrate entries
  L2 near-miss (>=2 votes, no bank)   -> repair bench (majority parse dumped)
  L3 knotted                          -> hand-surgery candidates
Rulebook filters stand: length<300 chars, no asy, values<=300 (value-cap
failures counted and routed to value-range certificates, never forced).
Charter prediction P1 in play: L4/5 skews harder to L3 than book 3's.
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

N_CAND = int(os.environ.get("B4_CAND", "200"))
h = [json.loads(l) for l in open(".cache/math_harvest_v0.jsonl")]
used = set()
for b in ("book1", "book2", "book3"):
    for l in open(f".cache/{b}_prose_pairs.jsonl"):
        used.add(json.loads(l)["gen"]["src_idx"])
census_texts = {x["problem"] for x in
                [json.loads(l) for l in open(".cache/math_harvest_v0.jsonl")][:100]}

pool, value_cap_out = [], 0
for i, x in enumerate(h):
    if x["level"] not in ("Level 4", "Level 5") or i in used:
        continue
    if x["problem"] in census_texts or len(x["problem"]) >= 300 or "asy]" in x["problem"]:
        continue
    if not isinstance(x["answer"], int) or abs(x["answer"]) > 300:
        continue
    if any(int(n) > 300 for n in re.findall(r"\d+", x["problem"])):
        value_cap_out += 1
        continue
    pool.append(dict(x, src_idx=i))
cands = pool[:N_CAND]
print(f"[book4] harder-strata pool: {len(pool)} bankable "
      f"(+{value_cap_out} value-cap -> certificates) | running {len(cands)}")

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/phase1_gen14_head.safetensors")
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[book4] gate = {CKPT}")


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
        keys = ("pres", "ftype", "op", "islit", "dig", "args", "res", "query") + \
            (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0 + bi < n:
                res.append(decode({k: o[k][bi] for k in o}))
    return res


lanes, lane1 = [], []
for ci, x in enumerate(cands):
    texts = [x["problem"]] + [permuted_view(x["problem"], 110000 + 10 * ci + k)
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
                          src_idx=x["src_idx"], lane="machine-banked",
                          gate="lattice-vote+answer-key", generation="14", book=4))
    elif len(votes) >= 2:
        lane = "L2"
    else:
        lane = "L3"
    lanes.append(dict(idx=ci, src_idx=x["src_idx"], lane=lane, votes=votes,
                      answer=x["answer"], level=x["level"],
                      subject=x.get("subject"), problem=x["problem"],
                      parse=(best[0] if lane == "L2" and best else None),
                      query=(best[1] if lane == "L2" and best else None)))
    if (ci + 1) % 25 == 0:
        c = Counter(l["lane"] for l in lanes)
        print(f"  ...{ci+1}/{len(cands)}: {dict(c)}", flush=True)

c = Counter(l["lane"] for l in lanes)
print(f"\n[book4-lanes] {dict(c)} of {len(cands)}  "
      f"(P1 watch: book-3 baseline was ~82% L3)")
json.dump(lanes, open(".cache/book4_lanes.json", "w"))
with open(".cache/book4_lane1.jsonl", "w") as f:
    for b in lane1:
        f.write(json.dumps(b) + "\n")
print(f"[book4] lane-1 free entries: {len(lane1)} -> .cache/book4_lane1.jsonl")
print("[book4] repair bench + surgery lists -> .cache/book4_lanes.json")
