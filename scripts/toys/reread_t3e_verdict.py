"""reread_t3e_verdict.py -- T3e pinned bars (ledger 2026-10-26 20:48). Reads .cache/reread_t3e_nb12.json + _nb24.json; zero training.
  CONTROL  REREAD in-dist >= 0.95 on every seed at both depths, else VOID
  R1       REREAD - POOL on HELD_RULE (NB 12) >= 0.10 mean, every seed positive
  R2       REREAD - BUS on HELD_RULE (NB 12): PASS >= 0.05 mean + every seed positive; EQUIVALENT |mean| < 0.02; else BETWEEN
  R3       (REREAD - BUS on LONG at NB 24) - (same at NB 12) >= 0.05"""
import json, os

import numpy as np

OUT = ".cache/reread_t3e_verdict.txt"
SPLITS = ("indist", "held_rule", "held_h", "long", "edge", "swap")


def load(nb):
    return {a["arm"]: a["seeds"] for a in json.load(open(f".cache/reread_t3e_nb{nb}.json"))["arms"]}


if __name__ == "__main__":
    assert not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w")

    def say(s):
        print(s, flush=True); out.write(s + "\n")
    R = {nb: load(nb) for nb in (12, 24)}
    for nb, arms in R.items():
        for a, rs in arms.items():
            assert len(rs) == 5, (nb, a, len(rs))
            say(f"NB {nb} {a:6s} mean: " + " ".join(f"{k} {np.mean([r[k] for r in rs]):.3f}" for k in SPLITS) +
                f" | in-dist per seed {[round(r['indist'], 3) for r in rs]}")
    gap = lambda nb, a, b, k: [x[k] - y[k] for x, y in zip(R[nb][a], R[nb][b])]
    ctrl = all(r["indist"] >= 0.95 for nb in (12, 24) for r in R[nb]["REREAD"])
    say("VERDICTS (T3e)")
    say(f"CONTROL: {'OK' if ctrl else 'VOID'}")
    d1 = gap(12, "REREAD", "POOL", "held_rule"); r1 = np.mean(d1) >= 0.10 and all(x > 0 for x in d1)
    say(f"R1 RE-READ BEATS THE LEARNED PACKAGE: {('PASS' if r1 else 'FAIL') if ctrl else 'VOID'} (REREAD - POOL on HELD_RULE {np.mean(d1):+.3f}; per seed {[round(x, 3) for x in d1]})")
    d2 = gap(12, "REREAD", "BUS", "held_rule"); m2 = np.mean(d2)
    r2 = "PASS" if m2 >= 0.05 and all(x > 0 for x in d2) else ("EQUIVALENT" if abs(m2) < 0.02 else "BETWEEN")
    say(f"R2 RE-READ vs THE BUS: {r2 if ctrl else 'VOID'} (REREAD - BUS on HELD_RULE {m2:+.3f}; per seed {[round(x, 3) for x in d2]})")
    g12, g24 = np.mean(gap(12, "REREAD", "BUS", "long")), np.mean(gap(24, "REREAD", "BUS", "long"))
    say(f"R3 DEPTH: {('PASS' if g24 - g12 >= 0.05 else 'FAIL') if ctrl else 'VOID'} (REREAD - BUS on LONG: NB 12 {g12:+.3f}, NB 24 {g24:+.3f}; growth {g24 - g12:+.3f})")
    for nb in (12, 24):
        say(f"READ NB {nb}: landed (in-dist >= 0.95) " + ", ".join(f"{a} {sum(r['indist'] >= 0.95 for r in rs)}/5" for a, rs in R[nb].items()))
