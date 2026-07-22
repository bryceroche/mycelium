"""flip_probe.py — GUT #48 (2026-07-22): THE FLIP PROBE. Two narrators
per contest: the KEY stream (trunk paraphrase-preference — the banked
fingerpost margins) and the KNOT stream (structural familiarity — mean
log train-frequency of each candidate's mined subgraph classes). The
2x2: agree-cells vs FLIP-cells. PINNED: fingerpost-wrong contests
concentrate in flip-cells at >=2x the agree rate, or the dual-read dies.
Fence: photographs disagreement; never adjudicates."""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from math import log
from vote_sample_lattice import (fixture, rows, gold, sampled_decode,
                                 view_outs, T_SAMP, K_PER_VIEW, N_VIEWS)
from tta_alg2_dials import solve2
from schema_miner import mine_graph

tc = json.load(open(".cache/train_class_counts.json"))
fp = {d["i"]: d for d in json.load(open(".cache/fingerpost_v01.json"))["details"]}
residue = set(json.load(open(".cache/residue_portrait.json"))["residue_items"])

def knot_score(facs):
    fs = [tc.get(dg, 0) for dg, k, _ in mine_graph(facs) if k <= 4]
    return np.mean([log(1 + f) for f in fs]) if fs else 0.0

cells = {"agree": [], "flip": []}
for i in fixture:
    if i not in fp:
        continue
    cands = {}
    for v in range(N_VIEWS):
        for k in range(K_PER_VIEW):
            rng = np.random.RandomState(70000 + i * 100 + v * 10 + k)
            facs, q = sampled_decode(view_outs[v][i], T_SAMP, rng)
            a = solve2(facs, q, {"n_vars": 24, "m": rows[i].get("m", 60)})
            if a is not None and a not in cands:
                cands[a] = facs
    d = fp[i]
    if d["a1"] not in cands or d["a2"] not in cands:
        continue
    key_pref = d["a1"] if d["s1"] > d["s2"] else d["a2"]
    k1, k2 = knot_score(cands[d["a1"]]), knot_score(cands[d["a2"]])
    knot_pref = d["a1"] if k1 > k2 else d["a2"]
    cell = "agree" if key_pref == knot_pref else "flip"
    cells[cell].append({"i": i, "fp_wrong": not d["ok"],
                        "cold": i in residue,
                        "joint_right": (key_pref == knot_pref == d["gold"])})
for c, v in cells.items():
    n = len(v)
    w = sum(x["fp_wrong"] for x in v)
    cd = sum(x["cold"] for x in v)
    print(f"[flip] {c:6s} n={n:3d}  fingerpost-wrong {w} ({w/max(n,1):.0%})  "
          f"cold-residue members {cd} ({cd/max(n,1):.0%})")
a, f = cells["agree"], cells["flip"]
ra = sum(x["fp_wrong"] for x in a)/max(len(a),1)
rf = sum(x["fp_wrong"] for x in f)/max(len(f),1)
jr = sum(x["joint_right"] for x in a)/max(len(a),1)
print(f"[flip] agree-cell JOINT precision (both narrators on gold): {jr:.0%}")
verdict = ("FLIP-CELLS CONCENTRATE THE UNSEEN — the dual-read lives"
           if rf >= 2*ra and len(f) >= 10 else "the dual-read dies cheap")
print(f"[flip] error rate: agree {ra:.0%} vs flip {rf:.0%} -> {verdict}")
json.dump({"agree": len(a), "flip": len(f), "ra": ra, "rf": rf,
           "joint_precision": jr, "verdict": verdict},
          open(".cache/flip_probe.json", "w"))
print("[flip] banked")
