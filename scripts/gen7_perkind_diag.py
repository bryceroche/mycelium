"""gen7_perkind_diag.py — attribute gen-7's dag7test 49% by SHAPE KIND
(2026-07-11). Separates undertrained-everywhere from one-kind-poisoning:
if fdiv rows crater while plain rows match gen-6's wild-shape debut, the
fdiv gold (or its 60% overrepresentation) is the defect; if all kinds sit
near 49%, the head is undertrained on the whole distribution (val still
climbing at 16k supports that read). Single-view graph-solve + ANSWER
per row, sliced by kind flags recomputed from the gold factors.
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter, defaultdict
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

DIAG_TEST = os.environ.get("DIAG_TEST", ".cache/dag7_test.jsonl")
DIAG_CKPT = os.environ.get("DIAG_CKPT", ".cache/phase1_gen7_head.safetensors")
rows = [json.loads(l) for l in open(DIAG_TEST)]

def kind_of(r):
    ks = set()
    givens = sum(1 for f in r["factors"] if f["ftype"] == "given")
    adds = sum(1 for f in r["factors"] if f["ftype"] == "rel" and f["op"] == "add")
    for f in r["factors"]:
        if f["ftype"] == "fdiv":
            ks.add("fdiv")
        if f["ftype"] == "rel" and f["args"][0] == f["args"][1]:
            ks.add("sq")
    if givens >= 8 and adds >= 7:
        ks.add("ladder")
    lits = [f["var"] for f in r["factors"] if f["ftype"] == "given"]
    muls = [f for f in r["factors"] if f["ftype"] == "rel" and f["op"] == "mul"]
    if sum(1 for f in muls if f["args"][0] in lits or f["args"][1] in lits) >= 2:
        ks.add("coupled-ish")
    return ks or {"plain"}

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(DIAG_CKPT)
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

acc = defaultdict(lambda: [0, 0])
B = 40
for s0 in range(0, len(rows), B):
    chunk = rows[s0:s0+B]
    parses = parse_batch([r["text"] for r in chunk])
    for r, (facs, q) in zip(chunk, parses):
        gold = r["solution"][r["query_var"]]
        a = solve2(facs, q, {"n_vars": 24, "m": r["m"]})
        hit = (a == gold)
        for kk in kind_of(r):
            acc[kk][0] += hit; acc[kk][1] += 1

print(f"=== PER-KIND (single-view ANSWER) {DIAG_TEST} @ {DIAG_CKPT} ===")
for kk in sorted(acc, key=lambda k: acc[k][0]/max(acc[k][1],1)):
    h, n = acc[kk]
    print(f"  {kk:12s}: {h:3d}/{n:3d} = {h/max(n,1):.3f}")
