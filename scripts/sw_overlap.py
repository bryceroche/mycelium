"""sw_overlap.py — door #62's causal meter: given-slot fat-attention
overlap (fat_overlap diagnostic, coupling NEVER) read on an arbitrary
checkpoint over solved-sample / converted-159 / residue-74 populations,
length-controlled. Env: OV_CK (checkpoint), plus the era envs. Prints
pooled deltas and the carrier gate's learned value when present.
"""
import os
import sys
import json

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import numpy as np  # noqa: E402

from phase1_algebra_head import build_params, forward, load_alg  # noqa: E402
from tinygrad import Tensor, dtypes  # noqa: E402
from tinygrad.nn.state import safe_load  # noqa: E402

CK = os.environ["OV_CK"]
samples, states, tokmask, gold, sent = load_alg("test")
census = json.load(open('.cache/miss_census_gen41.json'))
miss = set(int(i) for i in census["miss_idx"])
res = set(r["idx"] for r in
          json.load(open('.cache/residue_census.json'))["rows"])
conv = sorted(miss - res)
rng = np.random.default_rng(41)
solved = sorted(rng.choice([i for i in range(len(samples))
                            if i not in miss], 240, replace=False))
p = build_params(0)
sd = safe_load(CK)
assert set(sd.keys()) == set(p.keys()), \
    f"key mismatch: ckpt-only {set(sd) - set(p)} params-only {set(p) - set(sd)}"
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
if "sw_g" in p:
    print(f"[sw] carrier gate sw_g = {float(p['sw_g'].numpy()[0]):+.5f}",
          flush=True)
gft = gold["ftype"]
gpres = gold["presence"]


def rows_overlap(rows):
    out = {}
    for s0 in range(0, len(rows), 8):
        sl = rows[s0:s0 + 8]
        pad = 8 - len(sl)
        slp = sl + sl[:1] * pad
        o = forward(p, Tensor(states[slp].astype(np.float32),
                              dtype=dtypes.float),
                    Tensor(tokmask[slp].astype(np.float32),
                           dtype=dtypes.float),
                    Tensor(sent[slp].astype(np.int32), dtype=dtypes.int))
        fat = o["fat"].realize().numpy()
        for bi, ri in enumerate(sl):
            gs = [j for j in range(gft.shape[1])
                  if gpres[ri, j] > 0 and gft[ri, j] == 1]
            if len(gs) < 3:
                continue
            F = fat[bi][gs]
            F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
            C = F @ F.T
            m = int(sent[ri][tokmask[ri] > 0].max())
            out[ri] = (m, float(np.mean(
                [C[a, b] for a in range(len(gs))
                 for b in range(a + 1, len(gs))])))
    return out


data = {t: rows_overlap([int(x) for x in r]) for t, r in
        (("solved", solved), ("converted", conv), ("residue", sorted(res)))}
bands = {}
for t, d in data.items():
    for ri, (m, v) in d.items():
        bands.setdefault(m, {}).setdefault(t, []).append(v)
wc, wr = [], []
for m in sorted(bands):
    b = bands[m]
    if len(b.get("solved", [])) >= 8:
        s = np.mean(b["solved"])
        if len(b.get("converted", [])) >= 8:
            wc.append((len(b["converted"]), np.mean(b["converted"]) - s))
        if len(b.get("residue", [])) >= 4:
            wr.append((len(b["residue"]), np.mean(b["residue"]) - s))


def pooled(w):
    n = sum(x[0] for x in w)
    return (sum(x[0] * x[1] for x in w) / max(n, 1), n)


dc, nc = pooled(wc)
dr, nr = pooled(wr)
un = {t: float(np.mean([v for _, v in d.values()])) for t, d in data.items()}
print(f"[sw_overlap {os.path.basename(CK)}] means: solved {un['solved']:.4f}"
      f"  converted {un['converted']:.4f}  residue {un['residue']:.4f}",
      flush=True)
print(f"[sw_overlap {os.path.basename(CK)}] POOLED length-controlled deltas:"
      f" converted-solved {dc:+.4f} (n={nc})  residue-solved {dr:+.4f}"
      f" (n={nr})", flush=True)
json.dump({"ck": CK, "means": un, "d_conv": dc, "d_res": dr},
          open(f".cache/sw_overlap_{os.path.basename(CK).split('.')[0]}.json",
               "w"))
