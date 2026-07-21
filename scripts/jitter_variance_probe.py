"""jitter_variance_probe.py — GUT #38 (2026-07-20): THE PHOTO-BOOTH PROBE.
Distortion as INSTRUMENT, never verdict (the fence: no perturbation enters
any acceptance path — this reads basin curvature, it moves nothing).

On the 37 reflective contests (near-zero fingerpost margin): recompute the
adjudication margin under the original text's 5 permutation views (standing
seeds — gate-historied distortions). Two signatures, pinned:
  FLAT   — |margin| stays ~0 under every retelling -> AMBIGUITY-BY-TEXT
           (the prose refuses; cure = the desk's matching section)
  SCATTER — margins swing/flip across views -> AMBIGUITY-BY-PROJECTION
           (a fold in the trunk's shadow; cure = re-read, jitterable)
BAR: the projection subclass EXISTS if >=20% of the 37 show view-margin
range > 0.0038 (the absorbable class's median). Rider: do scatterers'
view-majority margins re-point to gold?
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
os.environ.setdefault("ALG_DUP", "1")
from phase1_algebra_head import T_ALG, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_alg2_dials import solve2
from tokenizers import Tokenizer

# re-derive contests (standing seeds; the lattice re-runs by design)
from vote_sample_lattice import (fixture, rows, gold, sampled_decode,
                                 view_outs, T_SAMP, K_PER_VIEW, N_VIEWS,
                                 permuted_view)
from fingerpost_v01 import render  # the deterministic writer

refl = set(json.load(open(".cache/reflection_ledger.json"))["reflective_items"])
tok = Tokenizer.from_file(TOKENIZER_JSON)


def pooled(texts):
    ids = np.zeros((len(texts), T_ALG), np.int32)
    msk = np.zeros((len(texts), T_ALG), np.float32)
    for i, t in enumerate(texts):
        e = tok.encode(t)
        Ln = min(len(e.ids), T_ALG)
        ids[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
    st = recompute_states(ids)
    V = (st.astype(np.float32) * msk[:, :, None]).sum(1) / \
        np.maximum(msk.sum(1)[:, None], 1)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


out = []
for i in fixture:
    if i not in refl:
        continue
    cands = {}
    for v in range(N_VIEWS):
        for k in range(K_PER_VIEW):
            rng = np.random.RandomState(70000 + i * 100 + v * 10 + k)
            facs, q = sampled_decode(view_outs[v][i], T_SAMP, rng)
            a = solve2(facs, q, {"n_vars": 24, "m": rows[i].get("m", 60)})
            if a is not None:
                cands.setdefault(a, {"count": 0, "parse": (facs, q)})
                cands[a]["count"] += 1
    top = sorted(cands.items(), key=lambda kv: -kv[1]["count"])[:2]
    if len(top) < 2:
        continue
    (a1, c1), (a2, c2) = top
    d1, d2 = render(*c1["parse"]), render(*c2["parse"])
    texts = [rows[i]["text"]] + [permuted_view(rows[i]["text"], 40000 + 10 * i + v)
                                 for v in range(1, 5)]
    V = pooled(texts + [d1, d2])
    views, r1, r2 = V[:5], V[5], V[6]
    margins = views @ r1 - views @ r2          # margin per retelling
    rng_m = float(margins.max() - margins.min())
    # view-majority re-point: sign of the mean margin
    point = a1 if margins.mean() > 0 else a2
    out.append({"i": int(i), "a1": a1, "a2": a2, "gold": gold[i],
                "margins": [float(m) for m in margins],
                "range": rng_m, "repoint": point,
                "repoint_ok": bool(point == gold[i])})

BAR = 0.0038
scat = [o for o in out if o["range"] > BAR]
flat = [o for o in out if o["range"] <= BAR]
n = len(out)
print(f"=== THE PHOTO-BOOTH PROBE (n={n} reflective contests re-derived) ===")
print(f"  SCATTER (range > {BAR}): {len(scat)} ({len(scat)/max(n,1):.0%})  "
      f"FLAT: {len(flat)}")
exists = len(scat) >= 0.20 * n
print(f"  BAR (projection subclass exists at >=20%): "
      f"{'HOLDS' if exists else 'FAILS — the probe dies cheap'}")
if scat:
    rp = sum(o["repoint_ok"] for o in scat)
    print(f"  RIDER: scatterers' view-majority re-points to gold "
          f"{rp}/{len(scat)} = {rp/len(scat):.2f}")
if flat:
    rp = sum(o["repoint_ok"] for o in flat)
    print(f"  context: flat class re-points {rp}/{len(flat)} = {rp/len(flat):.2f} "
          f"(expected ~coin: the text refuses)")
json.dump({"n": n, "scatter": len(scat), "flat": len(flat),
           "bar_holds": bool(exists), "details": out},
          open(".cache/jitter_variance_probe.json", "w"))
print("[probe] banked -> .cache/jitter_variance_probe.json")
