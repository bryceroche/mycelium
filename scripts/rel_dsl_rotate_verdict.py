"""rel_dsl_rotate_verdict.py -- the pinned bars for ROTATE on THE HARD DEPTH SPLIT (ledger 2026-10-28 19:5x). Reads
.cache/rel_dsl_{ARM}_depth2_s{S}.json for ROTATE / NOTEBOOK / VANILLA; zero training. Train = 372 structures of depth <= 2; DEEP =
depth 5..9 (n 988); NB 10.
  C0  an arm whose SHALLOW seed mean < 0.95 is VOID for the bars below
  T1  ROTATE - NOTEBOOK on DEEP >= +0.03 with both seed pairs positive BREATH ROTATION HELPS; <= -0.02 HURTS; else EQUIVALENT
  T2  NOTEBOOK - VANILLA on DEEP >= +0.10 with both pairs positive THE STACK CHAINS FROM TWO-STEP TRAINING; <= 0 VANILLA; else BETWEEN
Reads: MID (depth 3-4), accuracy by depth 5..9, paired SE."""
import json, os

import numpy as np

OUT = ".cache/rel_dsl_rotate_verdict.txt"
ARMS = ("VANILLA", "NOTEBOOK", "ROTATE")

if __name__ == "__main__":
    assert not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w")

    def say(s):
        print(s, flush=True); out.write(s + "\n")
    J = {(a, s): json.load(open(f".cache/rel_dsl_{a}_depth2_s{s}.json")) for a in ARMS for s in (0, 1)}
    res = J["NOTEBOOK", 0]["result"]
    say(f"RESULT baseline shallow {res['shallow']:.3f} mid {res['mid']:.3f} deep {res['deep']:.3f}")
    void = {}
    for a in ARMS:
        js = [J[a, s] for s in (0, 1)]; void[a] = np.mean([j["shallow"] for j in js]) < 0.95
        bd = lambda d: np.nanmean([j["deep_by_depth"].get(str(d), np.nan) for j in js])
        say(f"{a:8s} params {js[0]['params']:8d} | SHALLOW {js[0]['shallow']:.3f}/{js[1]['shallow']:.3f} MID {js[0]['mid']:.3f}/{js[1]['mid']:.3f} "
            f"DEEP {js[0]['deep']:.3f}/{js[1]['deep']:.3f} | by depth " + " ".join(f"d{d} {bd(d):.3f}" for d in range(5, 10)) + (" | C0 VOID" if void[a] else ""))

    def pair(a, b):
        d = [J[a, s]["deep"] - J[b, s]["deep"] for s in (0, 1)]
        se = [float(np.std(np.subtract(J[a, s]["deep_per"], J[b, s]["deep_per"]), ddof=1) / np.sqrt(len(J[a, s]["deep_per"]))) for s in (0, 1)]
        say(f"DEEP {a} - {b} {np.mean(d):+.3f} (seeds {d[0]:+.3f} / {d[1]:+.3f}; paired SE {se[0]:.3f} / {se[1]:.3f})")
        return float(np.mean(d)), d
    say("VERDICTS (ROTATE, hard depth split)")
    g, d = pair("ROTATE", "NOTEBOOK")
    say("T1: " + ("VOID (C0)" if void["ROTATE"] or void["NOTEBOOK"] else "BREATH ROTATION HELPS" if g >= 0.03 and min(d) > 0 else "HURTS" if g <= -0.02 else "EQUIVALENT") + f" ({g:+.3f})")
    g, d = pair("NOTEBOOK", "VANILLA")
    say("T2: " + ("VOID (C0)" if void["NOTEBOOK"] or void["VANILLA"] else "THE STACK CHAINS FROM TWO-STEP TRAINING" if g >= 0.10 and min(d) > 0 else "VANILLA" if g <= 0 else "BETWEEN") + f" ({g:+.3f})")
