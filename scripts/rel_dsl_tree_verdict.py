"""rel_dsl_tree_verdict.py -- the pinned bars for TREE (organ 3, the snap to SQLite step-type centroids) and ROTQ (query-only breath
rotation) on THE HARD DEPTH SPLIT (ledger 2026-10-29). Zero training. Reads .cache/rel_dsl_{ARM}_depth2_s{S}.json (cov 0.25: NOTEBOOK and
ROTATE banked 10-28) and .cache/rel_dsl_{ARM}_depth2c{LOW}_s{S}.json (LOW = .cache/rel_dsl_tree_low.txt, chosen by the chain's pinned rule:
the larger of 0.0625 / 0.03125 whose NOTEBOOK s0 DEEP <= 0.97, else 0.03125). DEEP = depth 5..9 (n 988); NB 10.
  C0  an arm whose SHALLOW seed mean < 0.95 is VOID for the bars below
  U1  at LOW: TREE - NOTEBOOK on DEEP >= +0.03 with both seed pairs positive THE TREE SNAP HELPS; <= -0.02 HURTS; else EQUIVALENT,
      read CEILING instead when NOTEBOOK's DEEP seed mean >= 0.97 (no headroom)
  U2  at 0.25: TREE - NOTEBOOK on DEEP <= -0.02 HURTS; else NO HARM (NOTEBOOK 0.994 there: HELPS is not testable)
  V1  at LOW: ROTQ - NOTEBOOK, U1's frame (QUERY ROTATION HELPS / HURTS / EQUIVALENT / CEILING)
  V2  at 0.25: ROTQ - NOTEBOOK, U2's frame
Reads: ROTQ - ROTATE at 0.25; MID; by depth 5..9; paired SE; the tree's verified steps per type per window."""
import json, os

import numpy as np

OUT = ".cache/rel_dsl_tree_verdict.txt"

if __name__ == "__main__":
    assert not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w")

    def say(s):
        print(s, flush=True); out.write(s + "\n")
    LOW = open(".cache/rel_dsl_tree_low.txt").read().strip()
    tag = {"0.25": "depth2", LOW: f"depth2c{LOW}"}
    J = {}
    for c, arms in (("0.25", ("NOTEBOOK", "ROTATE", "ROTQ", "TREE")), (LOW, ("NOTEBOOK", "ROTQ", "TREE"))):
        for a in arms:
            for s in (0, 1): J[c, a, s] = json.load(open(f".cache/rel_dsl_{a}_{tag[c]}_s{s}.json"))
    void = {}
    for c in ("0.25", LOW):
        say(f"== cov {c}: train {J[c, 'NOTEBOOK', 0]['n_train']} structures (depth <= 2) | RESULT deep {J[c, 'NOTEBOOK', 0]['result']['deep']:.3f}")
        for a in ("NOTEBOOK", "ROTATE", "ROTQ", "TREE"):
            if (c, a, 0) not in J: continue
            js = [J[c, a, s] for s in (0, 1)]; void[c, a] = np.mean([j["shallow"] for j in js]) < 0.95
            bd = lambda d: np.nanmean([j["deep_by_depth"].get(str(d), np.nan) for j in js])
            say(f"{a:8s} SHALLOW {js[0]['shallow']:.3f}/{js[1]['shallow']:.3f} MID {js[0]['mid']:.3f}/{js[1]['mid']:.3f} DEEP {js[0]['deep']:.3f}/"
                f"{js[1]['deep']:.3f} | by depth " + " ".join(f"d{d} {bd(d):.3f}" for d in range(5, 10)) + (" | C0 VOID" if void[c, a] else ""))
            if a == "TREE":
                for j in js: say(f"   tree s{j['seed']} verified steps per window: " + " ".join(str(sum(w["n"])) for w in j["tree"]))

    def pair(c, a, b):
        d = [J[c, a, s]["deep"] - J[c, b, s]["deep"] for s in (0, 1)]
        se = [float(np.std(np.subtract(J[c, a, s]["deep_per"], J[c, b, s]["deep_per"]), ddof=1) / np.sqrt(len(J[c, a, s]["deep_per"]))) for s in (0, 1)]
        say(f"cov {c} DEEP {a} - {b} {np.mean(d):+.3f} (seeds {d[0]:+.3f} / {d[1]:+.3f}; paired SE {se[0]:.3f} / {se[1]:.3f})")
        return float(np.mean(d)), d

    def low_bar(a, name):
        g, d = pair(LOW, a, "NOTEBOOK"); ceil = np.mean([J[LOW, "NOTEBOOK", s]["deep"] for s in (0, 1)]) >= 0.97
        v = ("VOID (C0)" if void[LOW, a] or void[LOW, "NOTEBOOK"] else f"{name} HELPS" if g >= 0.03 and min(d) > 0 else "HURTS" if g <= -0.02
             else "CEILING" if ceil else "EQUIVALENT")
        return f"{v} ({g:+.3f})"

    def hi_bar(a):
        g, d = pair("0.25", a, "NOTEBOOK")
        return f"{'VOID (C0)' if void['0.25', a] else 'HURTS' if g <= -0.02 else 'NO HARM'} ({g:+.3f})"
    say(f"VERDICTS (TREE / ROTQ, hard depth split; LOW = {LOW})")
    say("U1: " + low_bar("TREE", "THE TREE SNAP")); say("U2: " + hi_bar("TREE"))
    say("V1: " + low_bar("ROTQ", "QUERY ROTATION")); say("V2: " + hi_bar("ROTQ"))
    pair("0.25", "ROTQ", "ROTATE")
