"""rel_dsl_nostamp_verdict.py -- the pinned bar for THE NOSTAMP ABLATION (ledger 2026-10-29). Zero training. Reads
.cache/rel_dsl_{NOTEBOOK,NOSTAMP}_depth2c{LOW}_s{S}.json, LOW = .cache/rel_dsl_tree_low.txt (the tree chain's pinned headroom rule).
  C0  an arm whose SHALLOW seed mean < 0.95 is VOID
  S1  NOTEBOOK - NOSTAMP on DEEP >= +0.03 with both seed pairs positive THE WRITE-TIME STAMP IS LOAD-BEARING (the notebook needs the
      T6 coupling); <= -0.02 THE STAMP HURTS; else EQUIVALENT (the known bit carries it), read CEILING instead when both arms' DEEP
      seed means are >= 0.97
Reads: MID, by depth 5..9, paired SE."""
import json, os

import numpy as np

OUT = ".cache/rel_dsl_nostamp_verdict.txt"

if __name__ == "__main__":
    assert not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w")

    def say(s):
        print(s, flush=True); out.write(s + "\n")
    LOW = open(".cache/rel_dsl_tree_low.txt").read().strip()
    J = {(a, s): json.load(open(f".cache/rel_dsl_{a}_depth2c{LOW}_s{s}.json")) for a in ("NOTEBOOK", "NOSTAMP") for s in (0, 1)}
    say(f"== cov {LOW}: train {J['NOTEBOOK', 0]['n_train']} structures (depth <= 2) | RESULT deep {J['NOTEBOOK', 0]['result']['deep']:.3f}")
    void, m = {}, {}
    for a in ("NOTEBOOK", "NOSTAMP"):
        js = [J[a, s] for s in (0, 1)]; void[a] = np.mean([j["shallow"] for j in js]) < 0.95; m[a] = np.mean([j["deep"] for j in js])
        bd = lambda d: np.nanmean([j["deep_by_depth"].get(str(d), np.nan) for j in js])
        say(f"{a:8s} SHALLOW {js[0]['shallow']:.3f}/{js[1]['shallow']:.3f} MID {js[0]['mid']:.3f}/{js[1]['mid']:.3f} DEEP {js[0]['deep']:.3f}/"
            f"{js[1]['deep']:.3f} | by depth " + " ".join(f"d{d} {bd(d):.3f}" for d in range(5, 10)) + (" | C0 VOID" if void[a] else ""))
    d = [J["NOTEBOOK", s]["deep"] - J["NOSTAMP", s]["deep"] for s in (0, 1)]; g = float(np.mean(d))
    se = [float(np.std(np.subtract(J["NOTEBOOK", s]["deep_per"], J["NOSTAMP", s]["deep_per"]), ddof=1) / np.sqrt(len(J["NOTEBOOK", s]["deep_per"]))) for s in (0, 1)]
    say(f"DEEP NOTEBOOK - NOSTAMP {g:+.3f} (seeds {d[0]:+.3f} / {d[1]:+.3f}; paired SE {se[0]:.3f} / {se[1]:.3f})")
    v = ("VOID (C0)" if any(void.values()) else "THE WRITE-TIME STAMP IS LOAD-BEARING" if g >= 0.03 and min(d) > 0 else "THE STAMP HURTS" if g <= -0.02
         else "CEILING" if min(m.values()) >= 0.97 else "EQUIVALENT")
    say(f"S1: {v} ({g:+.3f})")
