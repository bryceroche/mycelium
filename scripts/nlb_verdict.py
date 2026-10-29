"""nlb_verdict.py -- the pinned bars for THE NL BINDING BENCH (ledger 2026-10-29). Zero training. Reads .cache/nlb_{ARM}_s{S}.json.
  C0  an arm whose IN answer seed mean < 0.80 is VOID for every bar it enters
  H1  ID - VANILLA on HELD >= +0.10 with both seed pairs positive MICRO ID TAGS CARRY UNSEEN NAMES; <= -0.02 HURTS; else EQUIVALENT
  H2  ID - VANILLA on IN <= -0.03 THE ID CHANNEL COSTS IN-DISTRIBUTION; else NO COST
  T1  TREE - ID on HELD >= +0.05 with both pairs positive THE TREE'S MACRO CHANNEL HELPS; <= -0.02 HURTS; else EQUIVALENT
  K1  CLOCK - TREE on LONG >= +0.05 with both pairs positive THE CLOCK CARRIES LENGTH; <= -0.02 HURTS; else EQUIVALENT
  N1  NOMLP - CLOCK, mean of HELD and LONG: <= -0.02 THE MLP IS NEEDED; >= +0.02 NO-MLP BETTER; else EQUIVALENT
  S   NOMLP - VANILLA on HELD >= +0.15 AND on LONG >= +0.05 (both pairs positive on each) THE ORGAN STACK BEATS VANILLA ON BOTH AXES
Reads: owner accuracy on all mentions / pronouns / need-gender pronouns per split; op and role accuracy; paired SE."""
import json, os

import numpy as np

OUT = ".cache/nlb_verdict.txt"
ARMS = ("VANILLA", "ID", "TREE", "CLOCK", "NOMLP")

if __name__ == "__main__":
    assert not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w")

    def say(s):
        print(s, flush=True); out.write(s + "\n")
    J = {(a, s): json.load(open(f".cache/nlb_{a}_s{s}.json")) for a in ARMS for s in (0, 1)}
    void = {a: np.mean([J[a, s]["IN_ans"] for s in (0, 1)]) < 0.80 for a in ARMS}
    for a in ARMS:
        say(f"{a:8s} params {J[a, 0]['params']:7d}" + "".join(
            f" | {k} ans {J[a, 0][k + '_ans']:.3f}/{J[a, 1][k + '_ans']:.3f} own {np.mean([J[a, s][k + '_own'] for s in (0, 1)]):.3f} pron "
            f"{np.mean([J[a, s][k + '_own_pron'] for s in (0, 1)]):.3f} need {np.mean([J[a, s][k + '_own_need'] for s in (0, 1)]):.3f}"
            for k in ("IN", "HELD", "LONG")) + (" | C0 VOID" if void[a] else ""))

    def pair(a, b, k):
        d = [J[a, s][k + "_ans"] - J[b, s][k + "_ans"] for s in (0, 1)]
        se = [float(np.std(np.subtract(J[a, s][k + "_per"], J[b, s][k + "_per"]), ddof=1) / np.sqrt(len(J[a, s][k + "_per"]))) for s in (0, 1)]
        say(f"{k:4s} {a} - {b} {np.mean(d):+.3f} (seeds {d[0]:+.3f} / {d[1]:+.3f}; paired SE {se[0]:.3f} / {se[1]:.3f})")
        return float(np.mean(d)), d
    V = lambda *a: any(void[x] for x in a)
    say("VERDICTS (NL binding bench)")
    g, d = pair("ID", "VANILLA", "HELD")
    say("H1: " + ("VOID (C0)" if V("ID", "VANILLA") else "MICRO ID TAGS CARRY UNSEEN NAMES" if g >= 0.10 and min(d) > 0 else "HURTS" if g <= -0.02 else "EQUIVALENT") + f" ({g:+.3f})")
    g, d = pair("ID", "VANILLA", "IN")
    say("H2: " + ("VOID (C0)" if V("ID", "VANILLA") else "THE ID CHANNEL COSTS IN-DISTRIBUTION" if g <= -0.03 else "NO COST") + f" ({g:+.3f})")
    g, d = pair("TREE", "ID", "HELD")
    say("T1: " + ("VOID (C0)" if V("TREE", "ID") else "THE TREE'S MACRO CHANNEL HELPS" if g >= 0.05 and min(d) > 0 else "HURTS" if g <= -0.02 else "EQUIVALENT") + f" ({g:+.3f})")
    g, d = pair("CLOCK", "TREE", "LONG")
    say("K1: " + ("VOID (C0)" if V("CLOCK", "TREE") else "THE CLOCK CARRIES LENGTH" if g >= 0.05 and min(d) > 0 else "HURTS" if g <= -0.02 else "EQUIVALENT") + f" ({g:+.3f})")
    g1, _ = pair("NOMLP", "CLOCK", "HELD"); g2, _ = pair("NOMLP", "CLOCK", "LONG"); g = (g1 + g2) / 2
    say("N1: " + ("VOID (C0)" if V("NOMLP", "CLOCK") else "THE MLP IS NEEDED" if g <= -0.02 else "NO-MLP BETTER" if g >= 0.02 else "EQUIVALENT") + f" ({g:+.3f})")
    gh, dh = pair("NOMLP", "VANILLA", "HELD"); gl, dl = pair("NOMLP", "VANILLA", "LONG")
    ok = gh >= 0.15 and min(dh) > 0 and gl >= 0.05 and min(dl) > 0
    say("S: " + ("VOID (C0)" if V("NOMLP", "VANILLA") else "THE ORGAN STACK BEATS VANILLA ON BOTH AXES" if ok else "NOT ON BOTH") + f" (HELD {gh:+.3f}, LONG {gl:+.3f})")
