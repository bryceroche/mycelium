"""megazord_t3cd_verdict.py -- T3c / T3d pinned bars (ledger 2026-10-26 12:26). Reads banked runs; zero training.
  python megazord_t3cd_verdict.py c          T3c: M2c on .cache/megazord_t3c.json + the reused T3b rows (.cache/megazord_t3b6.json)
  TM_NB=24 python megazord_t3cd_verdict.py d T3d: C24 + S1 on .cache/megazord_t3d.json (gate margins need TM_NB=24)"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import megazord_toy as M
from megazord_t3b_verdict import margin, seeds

MODE = sys.argv[1]
OUT = f".cache/megazord_t3{MODE}_verdict.txt"
BAR = 0.95
SPLITS = ("indist", "held_leaf", "held_h", "joint", "swap", "long", "edge")


def row(name, s, r, tag, arm):
    st, gm = margin(tag, arm, s)
    return (f"{name:9s} s{s}: " + " ".join(f"{k} {r[k]:.3f}" for k in SPLITS) +
            f" | tension AUROC {r['tension_auroc']:.3f} | gate: stays max {st:.3f}, go min {gm:.3f}")


if __name__ == "__main__":
    assert MODE in ("c", "d") and not os.path.exists(OUT), "mode c|d; refusing to overwrite a banked read"
    assert (MODE == "d") == (M.NB == 24), "T3d is scored with TM_NB=24, T3c without"
    out = open(OUT, "w")

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    if MODE == "c":
        R = {a: seeds(".cache/megazord_t3b6.json", a) for a in ("MEGAZORD", "NO_SNAP")}
        R.update({a: seeds(".cache/megazord_t3c.json", a) for a in ("NO_ROT", "MLP_SPLIT", "MLP_MEM")})
        assert all(len(v) == 5 for v in R.values()), {a: len(v) for a, v in R.items()}
        for a, v in R.items():
            tag = "_t3b6" if a in ("MEGAZORD", "NO_SNAP") else "_t3c"
            for s, r in enumerate(v):
                say(row(a, s, r, tag, a))
        say("VERDICTS (T3c; CONTROL = T3b C6 PASS, M1 = T3b K1 PASS, reused)")
        ok = []
        for arm, axis in (("NO_ROT", "held_h"), ("MLP_SPLIT", "held_leaf"), ("MLP_MEM", "swap")):
            d = [x[axis] - y[axis] for x, y in zip(R["MEGAZORD"], R[arm])]
            p = np.mean(d) >= 0.10 and all(x > 0 for x in d); ok.append(p)
            say(f"   M2c {arm:9s} on {axis:9s}: MEGAZORD - arm {np.mean(d):+.3f} (bar 0.10; per seed {[round(x, 3) for x in d]}) -> {'PASS' if p else 'FAIL'}")
        say(f"M2c EVERY ORGAN IS LOAD-BEARING ON ITS AXIS: {'PASS' if all(ok) else 'FAIL'}")
        for a, v in R.items():
            land = [s for s, r in enumerate(v) if r["indist"] >= BAR]
            extra = ""
            if a == "MLP_MEM" and land:
                extra = " | landed: " + "; ".join(f"s{s} in-dist {v[s]['indist']:.3f} held_leaf {v[s]['held_leaf']:.3f} swap {v[s]['swap']:.3f}" for s in land)
            if a == "NO_ROT" and land:
                extra = " | landed held_h " + str([round(v[s]["held_h"], 3) for s in land])
            say(f"READ {a:9s} landed {len(land)}/5 {land}; in-dist {[round(r['indist'], 3) for r in v]}{extra}")
    else:
        R = {a: seeds(".cache/megazord_t3d.json", a) for a in ("MEGAZORD", "NO_SNAP")}
        assert all(len(v) == 5 for v in R.values()), {a: len(v) for a, v in R.items()}
        for a, v in R.items():
            for s, r in enumerate(v):
                say(row(a, s, r, "_t3d", a))
        mz, ns = R["MEGAZORD"], R["NO_SNAP"]
        c24 = all(r["indist"] >= BAR for r in mz)
        d = [x["long"] - y["long"] for x, y in zip(mz, ns)]
        s1 = np.mean(d) >= 0.05 and all(x > 0 for x in d)
        say("VERDICTS (T3d, NB = 24)")
        say(f"C24 CONTROL: {'PASS' if c24 else 'VOID'} (in-dist {[round(r['indist'], 3) for r in mz]})")
        say(f"S1 THE SNAP AT DEPTH: {('PASS' if s1 else 'FAIL') if c24 else 'VOID'} (MEGAZORD - NO_SNAP on LONG {np.mean(d):+.3f}; per seed {[round(x, 3) for x in d]})")
        for a, v in R.items():
            say(f"READ {a:9s} landed {sum(r['indist'] >= BAR for r in v)}/5; EDGE {[round(r['edge'], 3) for r in v]}; HELD_H {[round(r['held_h'], 3) for r in v]}; "
                f"SWAP {[round(r['swap'], 3) for r in v]}")
