"""slot_consensus_census.py — GUT #34, READ (a) (2026-07-21): THE
SLOT-CONSENSUS CENSUS — the notebook's customer count. On each fixture
item's distinct solver-consistent candidates (the witnesses), measure
slot-grain consensus: factors shared by ALL candidates (the pinnable
bulk) vs contested loci (the union minus the intersection — where
temperature should be spent). PINNED BARS: pinning has customers if
median shared-fraction >= 0.5 AND median contested loci <= 4 among
items with >=2 candidates; else the notebook dies at its census.
Cautionary prior on record: ~9 contested bindings per top-2 contest.
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from vote_sample_lattice import (fixture, rows, sampled_decode, view_outs,
                                 T_SAMP, K_PER_VIEW, N_VIEWS)
from tta_alg2_dials import solve2


def fkey(f):
    if f["ftype"] == "rel":
        return ("rel", f["op"], tuple(sorted(f["args"])), f["result"])
    if f["ftype"] == "given":
        return ("given", f["var"], f["value"])
    return (f["ftype"], f.get("var"), f.get("k", f.get("p")), f.get("result"))


sh, ct, nc = [], [], 0
for i in fixture:
    cands = {}
    for v in range(N_VIEWS):
        for k in range(K_PER_VIEW):
            rng = np.random.RandomState(70000 + i * 100 + v * 10 + k)
            facs, q = sampled_decode(view_outs[v][i], T_SAMP, rng)
            a = solve2(facs, q, {"n_vars": 24, "m": rows[i].get("m", 60)})
            if a is not None and a not in cands:
                cands[a] = set(map(fkey, facs))
    if len(cands) < 2:
        nc += 1
        continue
    sets = list(cands.values())
    inter = set.intersection(*sets)
    union = set.union(*sets)
    sh.append(len(inter) / max(len(union), 1))
    ct.append(len(union) - len(inter))
sh, ct = np.array(sh), np.array(ct)
n = len(sh)
print(f"=== SLOT-CONSENSUS CENSUS (n={n} multi-candidate items; "
      f"{nc} single/zero-candidate skipped) ===")
print(f"  shared-slot fraction: median {np.median(sh):.2f}  "
      f"IQR [{np.percentile(sh,25):.2f}, {np.percentile(sh,75):.2f}]")
print(f"  contested loci:       median {np.median(ct):.0f}  "
      f"IQR [{np.percentile(ct,25):.0f}, {np.percentile(ct,75):.0f}]")
ok = np.median(sh) >= 0.5 and np.median(ct) <= 4
print(f"  BARS (median shared >=0.5 AND median contested <=4): "
      f"{'HOLD — the notebook has customers; the targeted-sampling spec proceeds' if ok else 'FAIL — the notebook dies at its census'}")
json.dump({"n": int(n), "shared_median": float(np.median(sh)),
           "contested_median": float(np.median(ct)), "bars_hold": bool(ok),
           "shared": sh.tolist(), "contested": ct.tolist()},
          open(".cache/slot_consensus_census.json", "w"))
print("[census] banked -> .cache/slot_consensus_census.json")
