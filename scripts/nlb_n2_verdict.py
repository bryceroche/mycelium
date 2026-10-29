"""nlb_n2_verdict.py -- the pinned bar for THE WIDTH-MATCHED NOMLP (ledger 2026-10-29, Bryce "word given for both"). Zero training.
NOMLPW = NOMLP at RD_D=208 (851,144 params vs CLOCK 867,336), the N1 confound (no-FF also cut 61 % of params) removed.
  C0  NOMLPW IN answer seed mean < 0.80 -> VOID
  N2  NOMLPW - CLOCK, mean of HELD and LONG: <= -0.02 THE MLP IS NEEDED (not capacity); >= +0.02 NO-MLP BETTER; else EQUIVALENT
      (the N1 deficit was capacity)
  read (no bar): NOMLPW - NOMLP per split = what width bought."""
import json, os

import numpy as np

OUT = ".cache/nlb_n2_verdict.txt"
if __name__ == "__main__":
    assert not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w")

    def say(s):
        print(s, flush=True); out.write(s + "\n")
    J = {(a, s): json.load(open(f".cache/nlb_{a}_s{s}.json")) for a in ("CLOCK", "NOMLP", "NOMLPW") for s in (0, 1)}
    m = lambda a, k: np.mean([J[a, s][k + "_ans"] for s in (0, 1)])
    for a in ("CLOCK", "NOMLP", "NOMLPW"):
        say(f"{a:7s} params {J[a, 0]['params']:7d} | " + " | ".join(f"{k} {J[a, 0][k + '_ans']:.3f}/{J[a, 1][k + '_ans']:.3f}" for k in ("IN", "HELD", "LONG")))
    void = m("NOMLPW", "IN") < 0.80
    d = {k: [J["NOMLPW", s][k + "_ans"] - J["CLOCK", s][k + "_ans"] for s in (0, 1)] for k in ("HELD", "LONG")}
    for k in d:
        say(f"{k:4s} NOMLPW - CLOCK {np.mean(d[k]):+.3f} (seeds {d[k][0]:+.3f} / {d[k][1]:+.3f})")
    g = np.mean([np.mean(d["HELD"]), np.mean(d["LONG"])])
    say("N2: " + ("VOID (C0)" if void else "THE MLP IS NEEDED" if g <= -0.02 else "NO-MLP BETTER" if g >= 0.02 else "EQUIVALENT (the N1 deficit was capacity)") + f" ({g:+.3f})")
    for k in ("IN", "HELD", "LONG"):
        say(f"read: {k:4s} NOMLPW - NOMLP {m('NOMLPW', k) - m('NOMLP', k):+.3f}")
