"""rel_dsl_depth_verdict.py -- the pinned bars for THE DEPTH SPLIT (ledger 2026-10-28 11:4x). Reads .cache/rel_dsl_{ARM}_depth_s{S}.json
for VANILLA / NOMLP / PHASE / NOTEBOOK; zero training. Train depth <= 3; DEEP = depth 5..8 (n 984), never trained at that depth.
  C0  an arm whose SHALLOW (unseen, depth <= 3) seed mean < 0.95 is VOID for the bars below
  K1  NOTEBOOK - PHASE on DEEP >= +0.03 with both seed pairs positive THE NOTEBOOK CARRIES DEPTH; <= -0.02 HURTS; else EQUIVALENT
  K2  NOMLP - VANILLA on DEEP >= +0.03 with both seed pairs positive BREATHING CHAINS DEEPER; <= -0.02 VANILLA DEEPER; else EQUIVALENT
  K3  the best breathing arm's DEEP - RESULT baseline >= +0.10 CHAINS BEYOND THE TRIVIAL PLAN; else DOES NOT
Reads: PHASE - NOMLP on DEEP; MID (depth 4); accuracy by depth 5..8; paired SE."""
import json, os

import numpy as np

OUT = ".cache/rel_dsl_depth_verdict.txt"
ARMS = ("VANILLA", "NOMLP", "PHASE", "NOTEBOOK")

if __name__ == "__main__":
    assert not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w")

    def say(s):
        print(s, flush=True); out.write(s + "\n")
    J = {(a, s): json.load(open(f".cache/rel_dsl_{a}_depth_s{s}.json")) for a in ARMS for s in (0, 1)}
    res = J["NOMLP", 0]["result"]
    say(f"RESULT baseline shallow {res['shallow']:.3f} mid {res['mid']:.3f} deep {res['deep']:.3f}")
    void = {}
    for a in ARMS:
        js = [J[a, s] for s in (0, 1)]; sh = np.mean([j["shallow"] for j in js]); void[a] = sh < 0.95
        say(f"{a:8s} params {js[0]['params']:8d} | SHALLOW {js[0]['shallow']:.3f}/{js[1]['shallow']:.3f} MID {js[0]['mid']:.3f}/{js[1]['mid']:.3f} "
            f"DEEP {js[0]['deep']:.3f}/{js[1]['deep']:.3f} | by depth " + " ".join(f"d{d} {np.mean([j['deep_by_depth'][str(d)] for j in js]):.3f}" for d in range(5, 9))
            + (" | C0 VOID" if void[a] else ""))

    def pair(a, b):
        d = [J[a, s]["deep"] - J[b, s]["deep"] for s in (0, 1)]
        se = [float(np.std(np.subtract(J[a, s]["deep_per"], J[b, s]["deep_per"]), ddof=1) / np.sqrt(len(J[a, s]["deep_per"]))) for s in (0, 1)]
        say(f"DEEP {a} - {b} {np.mean(d):+.3f} (seeds {d[0]:+.3f} / {d[1]:+.3f}; paired SE {se[0]:.3f} / {se[1]:.3f})")
        return float(np.mean(d)), d
    say("VERDICTS (depth split)")
    g, d = pair("NOTEBOOK", "PHASE")
    say("K1: " + ("VOID (C0)" if void["NOTEBOOK"] or void["PHASE"] else "THE NOTEBOOK CARRIES DEPTH" if g >= 0.03 and min(d) > 0 else "HURTS" if g <= -0.02 else "EQUIVALENT") + f" ({g:+.3f})")
    g, d = pair("NOMLP", "VANILLA")
    say("K2: " + ("VOID (C0)" if void["NOMLP"] or void["VANILLA"] else "BREATHING CHAINS DEEPER" if g >= 0.03 and min(d) > 0 else "VANILLA DEEPER" if g <= -0.02 else "EQUIVALENT") + f" ({g:+.3f})")
    best = max((a for a in ("NOMLP", "PHASE", "NOTEBOOK") if not void[a]), key=lambda a: np.mean([J[a, s]["deep"] for s in (0, 1)]), default=None)
    if best:
        v = np.mean([J[best, s]["deep"] for s in (0, 1)])
        say(f"K3: {'CHAINS BEYOND THE TRIVIAL PLAN' if v - res['deep'] >= 0.10 else 'DOES NOT'} ({best} {v:.3f} - RESULT {res['deep']:.3f} = {v - res['deep']:+.3f})")
    else: say("K3: VOID (every breathing arm C0)")
    pair("PHASE", "NOMLP")
