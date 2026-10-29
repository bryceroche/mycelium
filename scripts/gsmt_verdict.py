"""gsmt_verdict.py -- the pinned bars for STEP 0, the MLP-free ladder on frozen Llama states (ledger 2026-10-29). Zero training.
Reads .cache/gsmb_trunk_{ARM}_s{S}.json. Primary read = TEST step_all (the off-floor read; answers sit near 0.02); CF splits by answer.
  C0  an arm whose ORIG step_all seed mean < 0.80 is VOID for every bar it enters
  B1  BASE - VANILLA, TEST step_all: >= +0.02 (both pairs >0) NO-MLP BASE BETTER; <= -0.02 THE MLP CONTROL BETTER; else EQUIVALENT
  I1  ID - BASE, CFNAME answer: >= +0.02 (both pairs >0) ID TAGS CARRY NAME SWAPS; <= -0.02 HURTS; else EQUIVALENT
  T1  TREE - ID, TEST step_all: >= +0.02 (both >0) THE TREE HELPS; <= -0.02 HURTS; else EQUIVALENT
  K1  CLOCK - TREE, TEST step_all: same thresholds (THE CLOCK HELPS / HURTS / EQUIVALENT)
  A1  NOTEBOOK - CLOCK, TEST step_all: >= +0.02 (both >0) THE ALTERNATION HELPS; <= -0.02 HURTS; else EQUIVALENT
  S   NOTEBOOK - VANILLA: TEST step_all >= +0.03 AND CFNAME answer >= +0.02 (all pairs >0) THE MLP-FREE STACK BEATS THE CONTROL
Reads without bars: every split's answer / step_all / first_all per arm; CFNUM answer."""
import json, os

import numpy as np

OUT = ".cache/gsmt_verdict.txt"
ARMS = ("VANILLA", "BASE", "ID", "TREE", "CLOCK", "NOTEBOOK")
if __name__ == "__main__":
    assert not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w")

    def say(s):
        print(s, flush=True); out.write(s + "\n")
    J = {(a, s): json.load(open(f".cache/gsmb_trunk_{a}_s{s}.json")) for a in ARMS for s in (0, 1)}
    m = lambda a, k: np.mean([J[a, s][k] for s in (0, 1)])
    void = {a: m(a, "ORIG_step_all") < 0.80 for a in ARMS}
    for a in ARMS:
        say(f"{a:8s} params {J[a, 0]['params']:7d} | " + " | ".join(f"{sp} ans {J[a, 0][sp + '_ans']:.3f}/{J[a, 1][sp + '_ans']:.3f} step "
            f"{m(a, sp + '_step_all'):.3f}" for sp in ("ORIG", "CFNUM", "CFNAME", "TEST")) + f" | TEST first {m(a, 'TEST_first_all'):.3f}" +
            (" | C0 VOID" if void[a] else ""))

    def gap(a, b, k):
        d = [J[a, s][k] - J[b, s][k] for s in (0, 1)]; say(f"{k:14s} {a} - {b} {np.mean(d):+.3f} (seeds {d[0]:+.3f} / {d[1]:+.3f})"); return np.mean(d), min(d)

    def tri(name, a, b, k, up, down):
        g, lo = gap(a, b, k)
        v = "VOID (C0)" if void[a] or void[b] else up if g >= 0.02 and lo > 0 else down if g <= -0.02 else "EQUIVALENT"
        say(f"{name}: {v} ({g:+.3f})")
    tri("B1", "BASE", "VANILLA", "TEST_step_all", "NO-MLP BASE BETTER", "THE MLP CONTROL BETTER")
    tri("I1", "ID", "BASE", "CFNAME_ans", "ID TAGS CARRY NAME SWAPS", "HURTS")
    tri("T1", "TREE", "ID", "TEST_step_all", "THE TREE HELPS", "HURTS")
    tri("K1", "CLOCK", "TREE", "TEST_step_all", "THE CLOCK HELPS", "HURTS")
    tri("A1", "NOTEBOOK", "CLOCK", "TEST_step_all", "THE ALTERNATION HELPS", "HURTS")
    g1, l1 = gap("NOTEBOOK", "VANILLA", "TEST_step_all"); g2, l2 = gap("NOTEBOOK", "VANILLA", "CFNAME_ans")
    ok = g1 >= 0.03 and g2 >= 0.02 and min(l1, l2) > 0
    say("S: " + ("VOID (C0)" if void["NOTEBOOK"] or void["VANILLA"] else "THE MLP-FREE STACK BEATS THE CONTROL" if ok else "NOT ON BOTH") +
        f" (TEST step {g1:+.3f}, CFNAME ans {g2:+.3f})")
    gap("NOTEBOOK", "CLOCK", "TEST_ans"); gap("NOTEBOOK", "VANILLA", "CFNUM_ans")
