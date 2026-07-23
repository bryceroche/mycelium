"""zero_hinge_probe.py — the texture rule's mechanism probe for the
420 transposition (2026-07-22). Bars pinned in the ledger first.

Matched mag-3 given pairs — zero-containing vs zero-free, matched
digit positions — through the banked reader on echo pages (a is N.
b is 1. a times b equals c. What is c?), 5 views each, m=1000.
Reads per-class given accuracy + a transposition census on wrongs.
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
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

ZERO = [420, 530, 704, 810, 205, 609, 130, 902]
FREE = [425, 537, 714, 815, 235, 619, 137, 924]
D = "Consider the numbers "

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/crown_reader_v4.safetensors")
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[zero-hinge] gate = {CKPT} | {len(ZERO)} zero-containing vs {len(FREE)} zero-free")


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


def is_transposition(wrong, true):
    return sorted(str(wrong).zfill(3)) == sorted(str(true).zfill(3)) and wrong != true


def run_class(nums, tag, seed0):
    hits, views_total = 0, 0
    wrongs = []
    for i, n_ in enumerate(nums):
        dia = D + f"a, b, c. a is {n_}. b is 1. a times b equals c. What is c?"
        texts = [dia] + [permuted_view(dia, seed0 + 100 * i + k) for k in range(1, 5)]
        for facs, q in parse_batch(texts):
            a = solve2(facs, q, {"n_vars": 24, "m": 1000})
            views_total += 1
            if a == n_:
                hits += 1
            elif a is not None:
                wrongs.append((n_, a, is_transposition(a, n_)))
    acc = hits / views_total
    print(f"[zero-hinge] {tag}: view-accuracy {acc:.3f} ({hits}/{views_total})"
          f" | wrongs: {wrongs}")
    return acc, wrongs


acc_z, wr_z = run_class(ZERO, "zero-containing", 650000)
acc_f, wr_f = run_class(FREE, "zero-free     ", 660000)

gap = acc_f - acc_z
if gap >= 0.25:
    verdict = "ZERO-HINGE CONFIRMED — decompose rule keeps its scope (zero-containing mag-3)"
elif acc_z >= 0.85 and acc_f >= 0.85:
    verdict = "SPECIMEN-ISOLATED — both classes clear 0.85; the '42' digit-pair hinge hypothesis opens"
else:
    verdict = "MAG-3 BROADLY — both classes trail; decompose rule WIDENS to all mag-3 givens"
n_transp = sum(1 for w in wr_z + wr_f if w[2])
print(f"\n[zero-hinge] gap {gap:+.3f} | transpositions among wrongs: {n_transp}/{len(wr_z)+len(wr_f)}")
print(f"[zero-hinge] VERDICT: {verdict}")
json.dump(dict(zero_acc=acc_z, free_acc=acc_f, gap=gap,
               wrongs_zero=[(a, b, c) for a, b, c in wr_z],
               wrongs_free=[(a, b, c) for a, b, c in wr_f],
               verdict=verdict),
          open(".cache/zero_hinge_probe.json", "w"), indent=1)
print("[zero-hinge] banked -> .cache/zero_hinge_probe.json")
