"""rel_dsl_phase_verdict.py -- the pinned bars for the PHASE arm (ledger 2026-10-28 11:0x). Reads .cache/rel_dsl_{ARM}_c{COV}_s{S}.json for
PHASE / NOMLP / VANILLA; zero training. VAL seed means s0/s1, per-seed pairs (same seed, same data).
  Q1  cov 0.03125: PHASE - NOMLP >= +0.02 with both seed pairs positive PHASE BINDING HELPS; <= -0.02 PHASE BINDING HURTS; else EQUIVALENT
  Q2  cov 0.03125: PHASE - VANILLA >= +0.05 with both seed pairs positive PHASE BEATS VANILLA AT LOW DATA; <= -0.02 VANILLA WINS; else EQUIVALENT
Reads (no bar): the same pairs at cov 0.125; wild holdout unseen structures; paired SE."""
import json, os

import numpy as np

OUT = ".cache/rel_dsl_phase_verdict.txt"


def get(a, c, s):
    return json.load(open(f".cache/rel_dsl_{a}_c{c}_s{s}.json"))


if __name__ == "__main__":
    assert not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w")

    def say(s):
        print(s, flush=True); out.write(s + "\n")
    V = {}
    for c in ("0.03125", "0.125"):
        for a in ("PHASE", "NOMLP", "VANILLA"):
            js = [get(a, c, s) for s in (0, 1)]
            say(f"cov {c:7s} {a:7s} VAL {js[0]['val']:.3f} / {js[1]['val']:.3f} | holdout unseen {js[0]['hold_unseen']:.3f} / {js[1]['hold_unseen']:.3f} | params {js[0]['params']}")
        for b in ("NOMLP", "VANILLA"):
            pr = [(get("PHASE", c, s), get(b, c, s)) for s in (0, 1)]
            d = [p["val"] - q["val"] for p, q in pr]
            se = [float(np.std(np.subtract(p["val_per"], q["val_per"]), ddof=1) / np.sqrt(len(p["val_per"]))) for p, q in pr]
            V[c, b] = (float(np.mean(d)), d)
            say(f"cov {c}: PHASE - {b} {np.mean(d):+.3f} (seeds {d[0]:+.3f} / {d[1]:+.3f}; paired SE {se[0]:.3f} / {se[1]:.3f})")
    say("VERDICTS (phase-binding arm)")
    g, d = V["0.03125", "NOMLP"]
    say(f"Q1: {'PHASE BINDING HELPS' if g >= 0.02 and min(d) > 0 else 'PHASE BINDING HURTS' if g <= -0.02 else 'EQUIVALENT'} ({g:+.3f})")
    g, d = V["0.03125", "VANILLA"]
    say(f"Q2: {'PHASE BEATS VANILLA AT LOW DATA' if g >= 0.05 and min(d) > 0 else 'VANILLA WINS' if g <= -0.02 else 'EQUIVALENT'} ({g:+.3f})")
