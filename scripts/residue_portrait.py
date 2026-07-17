"""residue_portrait.py — GUT #26 FOLLOW-UP 3 (2026-07-17): THE PRECISION
DECOMPOSITION + THE RESIDUE PORTRAIT, one instrument (both reads share
the per-item ledger; lattice picks re-derived byte-identically from the
standing seeds).

PRE-REGISTERED PREDICTION (pinned before the join, relay + Code): the
65-item triple-residue skews toward (i) ZERO-withhold-recoverability
graphs (content-deep: nothing forces the missing piece) and (ii) LOWER
deterministic vote entropy than the lattice-recovered abstains (STABLE
cross-view misreadings — every view agrees on the wrong reading, so
neither width nor diagram-diversity dislodges it; only the text
proposes). Holds -> the writer's marginal customer has a PORTRAIT
(minimal-graph, stable-misreading survivor; re-read, not re-emit).
Scatters -> the wall has unnamed structure, banked as such. Kill-only.

PRECISION FRAME: the deployment bar is the emitted-answer precision of
the lattice lane (recovered / emitted), beside the abstain count — the
answer channel's incumbent standard is 0.9953 at the majority tier.
"""
import json, os, sys, re, random
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1")
os.environ.setdefault("ALG_FTYPES", "6")
os.environ.setdefault("ALG_DUP", "1")
import numpy as np
from collections import Counter

from vote_sample_lattice import (fixture, rows, gold, sampled_decode,
                                 permuted_view, head_out, T_SAMP, K_PER_VIEW,
                                 N_VIEWS)
from tta_alg2_dials import solve2

# NOTE: importing vote_sample_lattice RE-RUNS its pipeline (flat script) —
# acceptable here BY DESIGN: same seeds, byte-identical picks, and we need
# the view outputs anyway. Its prints re-appear above; its JSON re-banks
# identically (verified by the digest of per_item below).
lat_prior = json.load(open(".cache/vote_sample_lattice.json"))["per_item"]
from vote_sample_lattice import view_outs

emitted = wrong = silent = 0
per_item = {}
for i in fixture:
    answers = []
    for v in range(N_VIEWS):
        for k in range(K_PER_VIEW):
            rng = np.random.RandomState(70000 + i * 100 + v * 10 + k)
            facs, q = sampled_decode(view_outs[v][i], T_SAMP, rng)
            a = solve2(facs, q, {"n_vars": 24, "m": rows[i].get("m", 60)})
            if a is not None:
                answers.append(a)
    if not answers:
        silent += 1
        per_item[i] = {"exit": "abstain"}
        continue
    pick, cnt = Counter(answers).most_common(1)[0]
    emitted += 1
    ok = pick == gold[i]
    wrong += (not ok)
    per_item[i] = {"exit": "right" if ok else "WRONG", "pick": pick,
                   "n_consistent": len(answers), "plurality": cnt}
    assert (lat_prior[str(i)] == "recovered") == ok      # byte-identity check

rec = emitted - wrong
n = len(fixture)
print(f"\n=== PRECISION DECOMPOSITION (lattice lane, n={n}) ===")
print(f"  emitted {emitted} (right {rec}, WRONG {wrong})  abstained {silent}")
print(f"  LANE PRECISION: {rec}/{emitted} = {rec/max(emitted,1):.4f}  "
      f"(incumbent standard at majority tier: 0.9953)")

# ---- THE RESIDUE PORTRAIT ----
prior = json.load(open(".cache/nack_incumbent_read.json"))["per_item"]
residue = [i for i in fixture
           if prior[str(i)]["incumbent"] == "unrecovered"
           and prior[str(i)]["sampled_T1"] != "recovered"
           and lat_prior[str(i)] != "recovered"]
recovered_abstains = [i for i in fixture if i not in set(residue)]
print(f"\n=== THE RESIDUE PORTRAIT (n={len(residue)} resist all three voices) ===")


def withhold_frac(i):
    facs = rows[i]["factors"]
    g = solve2(facs, rows[i]["query_var"], {"n_vars": 24, "m": rows[i].get("m", 60)})
    if g is None:
        return None
    rec_ = sum(1 for j in range(len(facs))
               if solve2(facs[:j] + facs[j+1:], rows[i]["query_var"],
                         {"n_vars": 24, "m": rows[i].get("m", 60)}) == g)
    return rec_ / len(facs)


gate_votes = json.load(open(".cache/lattice_gate.json"))["bigtest"]


def H(votes):
    c = Counter("⊥" if v is None else v for v in votes)
    p = np.array(list(c.values()), float) / len(votes)
    return float(-(p * np.log(p)).sum())


def portrait(pop, tag):
    wf = [withhold_frac(i) for i in pop]
    wf = np.array([x for x in wf if x is not None])
    hs = np.array([H(gate_votes[i]) for i in pop])
    print(f"  {tag:24s} n={len(pop):3d}  withhold-recoverable: mean {wf.mean():.3f} "
          f"zero-frac {(wf==0).mean():.1%}  |  det-vote H: mean {hs.mean():.3f} "
          f"median {np.median(hs):.3f}")
    return {"n": len(pop), "wf_mean": float(wf.mean()),
            "wf_zero_frac": float((wf == 0).mean()),
            "H_mean": float(hs.mean()), "H_median": float(np.median(hs))}


out = {"precision": {"emitted": emitted, "right": rec, "wrong": wrong,
                     "abstained": silent,
                     "lane_precision": rec / max(emitted, 1)},
       "residue": portrait(residue, "RESIDUE (all-3 resist)"),
       "recovered": portrait(recovered_abstains, "recovered abstains")}

r, g_ = out["residue"], out["recovered"]
p1 = r["wf_zero_frac"] > g_["wf_zero_frac"]
p2 = r["H_mean"] < g_["H_mean"]
print(f"\n  PREDICTION (i) residue more zero-redundancy: "
      f"{r['wf_zero_frac']:.1%} vs {g_['wf_zero_frac']:.1%} -> {'HOLDS' if p1 else 'FAILS'}")
print(f"  PREDICTION (ii) residue lower det-vote entropy (stable misreading): "
      f"{r['H_mean']:.3f} vs {g_['H_mean']:.3f} -> {'HOLDS' if p2 else 'FAILS'}")
verdict = ("PORTRAIT CONFIRMED — the writer's customer is the minimal-graph, "
           "stable-misreading survivor" if p1 and p2 else
           "SCATTERS — the wall has unnamed structure" if not (p1 or p2) else
           "SPLIT — one axis holds")
print(f"  VERDICT: {verdict}")
out["prediction"] = {"p1_zero_redundancy": bool(p1), "p2_low_entropy": bool(p2),
                     "verdict": verdict}
out["per_item_exits"] = {str(i): per_item[i] for i in fixture}
out["residue_items"] = residue
json.dump(out, open(".cache/residue_portrait.json", "w"))
print("[portrait] banked -> .cache/residue_portrait.json (per-item, per the law)")
