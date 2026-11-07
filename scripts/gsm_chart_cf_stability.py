"""gsm_chart_cf_stability.py -- the Tree charts under the counterfactual fixture (ledger 2026-11-07, read after the CS verdict). Zero GPU.

Pairs every CFNUM / CFNAME item with its ORIG item by gid (same program, same numeral slots, a number or a name changed) and counts the
numerals whose chart (`.cache/gsmb_chart64_rb8.npy`) is unchanged: ST64 (level 0 x 8 + level 1) and level 0 alone (ST8). A role-stable
chart would be ~1.0.  -> printed
"""
import os, sys

os.environ["RD_STATES"] = "gsmb_rb8_states.npy"
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm_breath_reader as BR          # FIRST (the 11-04 import rule)

Dt, states = BR.TR.load(); CH = np.load(BR.GD.V1.C("gsmb_chart64_rb8.npy"), mmap_mode="r")
num = lambda it: [i for i, v in enumerate(it["vals"]) if v is not None]
og = {it["gid"]: it for it in Dt["ORIG"]}
for sp in ("CFNUM", "CFNAME"):
    same = same8 = tot = skip = 0
    for b in Dt[sp]:
        a = og[b["gid"]]; na, nb = num(a), num(b)
        if len(na) != len(nb): skip += 1; continue
        ca = [int(CH[a["st0"] + i]) for i in na]; cb = [int(CH[b["st0"] + i]) for i in nb]
        same += sum(x == y for x, y in zip(ca, cb)); same8 += sum(x // 8 == y // 8 for x, y in zip(ca, cb)); tot += len(na)
    print(f"{sp}: items {len(Dt[sp])} (skipped {skip}), numerals {tot} | same ST64 chart {same / tot:.3f} | same level-0 node {same8 / tot:.3f}")
