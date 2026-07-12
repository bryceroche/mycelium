"""alg4_perkind_diff.py — WHICH KIND INSIDE ALG4 DIED (2026-07-12).

The gen-8 verdict's 'mis-targeted ration' attribution is CORRECTED: the
gen-8 kind-ration seed-collided with algebra4_nl_train (2500/3000 exact
duplicates -> an accidental 2x rehearsal upweight of the alg4 register)
and alg4test still moved only +13. A rationed register failing its own
ration reopens the junction FOR ALG4. This instrument attributes: parse
alg4test under gen-6 AND gen-8 heads (single view), slice accuracy by
row features (seq / pct / fdiv / crt / vieta / plain), read the diff.
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import defaultdict
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

rows = [json.loads(l) for l in open(".cache/algebra4_nl_test.jsonl")]

def feats(r):
    fs = set()
    mods_by_var = defaultdict(int)
    for f in r["factors"]:
        if "seq" in f or f.get("role") in ("seq_anchor", "seq_d"):
            fs.add("seq")
        if f["ftype"] == "pct":
            fs.add("pct")
        if f["ftype"] == "fdiv":
            fs.add("fdiv")
        if f["ftype"] == "mod":
            mods_by_var[f["var"]] += 1
        if f.get("role") in ("vieta_sum", "vieta_prod"):
            fs.add("vieta")
    if any(c >= 2 for c in mods_by_var.values()):
        fs.add("crt")
    return fs or {"plain"}

tok = Tokenizer.from_file(TOKENIZER_JSON)

def parse_all(ckpt):
    p = build_params(0)
    sd = safe_load(ckpt)
    assert set(sd.keys()) == set(p.keys())
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    hits = []
    B = 40
    for s0 in range(0, len(rows), B):
        chunk = rows[s0:s0+B]
        n = len(chunk)
        N = ((n + 7) // 8) * 8
        ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32)
        snt = np.zeros((N, T_ALG), np.int32)
        for i, r in enumerate(chunk):
            e = tok.encode(r["text"])
            L = min(len(e.ids), T_ALG)
            ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0
            snt[i] = sent_indices(r["text"], list(e.offsets), msk[i])
        st = recompute_states(ids)
        for b0 in range(0, N, 8):
            out = forward(p, Tensor(st[b0:b0+8].astype(np.float32), dtype=dtypes.float),
                          Tensor(msk[b0:b0+8].astype(np.float32), dtype=dtypes.float),
                          Tensor(snt[b0:b0+8].astype(np.int32), dtype=dtypes.int))
            keys = ("pres","ftype","op","islit","dig","args","res","query") + \
                (("sel",) if "sel" in out else ())
            o = {k: out[k].realize().numpy() for k in keys}
            for bi in range(8):
                gi = s0 + b0 + bi
                if b0 + bi < n:
                    r = chunk[b0 + bi]
                    facs, q = decode({k: o[k][bi] for k in o})
                    a = solve2(facs, q, {"n_vars": 24, "m": r.get("m", 60)})
                    hits.append(a == r["solution"][r["query_var"]])
    return hits

h6 = parse_all(".cache/phase1_gen6_head.safetensors")
h8 = parse_all(".cache/phase1_gen8_head.safetensors")
print(f"[diff] alg4test ANSWER: gen-6 {sum(h6)}/800 | gen-8 {sum(h8)}/800")
acc = defaultdict(lambda: [0, 0, 0])
for r, a6, a8 in zip(rows, h6, h8):
    for f in feats(r):
        acc[f][0] += a6; acc[f][1] += a8; acc[f][2] += 1
print(f"\n=== ALG4TEST PER-KIND: gen-6 -> gen-8 (single-view ANSWER) ===")
for f in sorted(acc, key=lambda k: (acc[k][1]-acc[k][0])/max(acc[k][2],1)):
    a6, a8, n = acc[f]
    print(f"  {f:7s}: {a6/n:.3f} -> {a8/n:.3f}  (delta {(a8-a6)/n:+.3f}, n={n})")
