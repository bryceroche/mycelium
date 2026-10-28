"""rel_dsl_verdict.py -- the pinned bars for the relation-DSL grid (ledger 2026-10-28 09:0x). Reads .cache/rel_dsl_{ARM}_c{COV}_s{S}.json
(rel_dsl_breath.py); zero training. VAL = structure-disjoint train-fixture structures (one item each); seed means over s0, s1.
  R1  LOW DATA (cov 0.125): gap = NOMLP - VANILLA. >= +0.05 with both seed pairs positive BREATHING WINS AT LOW DATA;
      <= -0.02 VANILLA WINS; else EQUIVALENT
  R2  gap(0.125) - gap(1.0) >= +0.03 THE GAP GROWS AS DATA SHRINKS; <= 0 NO GROWTH; else BETWEEN
  R3  NOMLP VAL at 0.125 - the RESULT baseline >= +0.10 LEARNS PROPAGATION; else DOES NOT
Reads (no bar): cov 0.25 and 0.03125; the wild holdout split by structure seen / unseen in train; paired SE per seed."""
import json, os

import numpy as np

OUT = ".cache/rel_dsl_verdict.txt"
COVS = ("1", "0.25", "0.125", "0.03125")


def get(a, c, s):
    p = f".cache/rel_dsl_{a}_c{c}_s{s}.json"
    return json.load(open(p)) if os.path.exists(p) else None


if __name__ == "__main__":
    assert not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w")

    def say(s):
        print(s, flush=True); out.write(s + "\n")
    J = {(a, c, s): get(a, c, s) for a in ("NOMLP", "VANILLA") for c in COVS for s in (0, 1)}
    for (a, c, s), j in sorted(J.items()):
        if j: say(f"{a:7s} cov {c:7s} s{s} n_train {j['n_train']:5d} params {j['params']:8d} | VAL {j['val']:.3f} | holdout {j['hold']:.3f} "
                  f"(seen {j['hold_seen']:.3f} / unseen {j['hold_unseen']:.3f}) | curve " + " ".join(f"{p['val']:.3f}" for p in j["curve"]))
    res = next(j["result_val"] for j in J.values() if j)
    say(f"RESULT baseline VAL {res:.3f}")
    gap = {}
    for c in COVS:
        pr = [(J["NOMLP", c, s], J["VANILLA", c, s]) for s in (0, 1)]
        if not all(n and v for n, v in pr): continue
        d = [n["val"] - v["val"] for n, v in pr]
        se = [float(np.std(np.subtract(n["val_per"], v["val_per"]), ddof=1) / np.sqrt(len(n["val_per"]))) for n, v in pr]
        gap[c] = (float(np.mean(d)), d)
        say(f"cov {c}: NOMLP - VANILLA {np.mean(d):+.3f} (seeds {d[0]:+.3f} / {d[1]:+.3f}; paired SE {se[0]:.3f} / {se[1]:.3f}) | holdout unseen "
            f"NOMLP {np.mean([n['hold_unseen'] for n, _ in pr]):.3f} VANILLA {np.mean([v['hold_unseen'] for _, v in pr]):.3f}")
    say("VERDICTS (relation-DSL)")
    if "0.125" in gap:
        g, d = gap["0.125"]
        say(f"R1: {'BREATHING WINS AT LOW DATA' if g >= 0.05 and min(d) > 0 else 'VANILLA WINS' if g <= -0.02 else 'EQUIVALENT'} ({g:+.3f})")
        if "1" in gap:
            gg = g - gap["1"][0]
            say(f"R2: {'THE GAP GROWS AS DATA SHRINKS' if gg >= 0.03 else 'NO GROWTH' if gg <= 0 else 'BETWEEN'} ({g:+.3f} - {gap['1'][0]:+.3f} = {gg:+.3f})")
        nv = np.mean([J["NOMLP", "0.125", s]["val"] for s in (0, 1)])
        say(f"R3: {'LEARNS PROPAGATION' if nv - res >= 0.10 else 'DOES NOT'} (NOMLP {nv:.3f} - RESULT {res:.3f} = {nv - res:+.3f})")
