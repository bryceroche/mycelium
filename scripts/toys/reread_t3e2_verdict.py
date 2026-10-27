"""reread_t3e2_verdict.py -- T3e2 pinned bars (ledger 2026-10-27, the T3e re-registration). Reads .cache/reread_t3e2_nb24.json
(FRESH seeds 5-9: the banked T3e seeds 0-4 are deterministic, a rerun would replay s3's non-landing) + .cache/reread_t3e2_nb12.json
(BUS256 width read at NB 12) + the banked .cache/reread_t3e_nb12.json; zero training. Each depth is gated by ITS OWN control.
  C24   REREAD in-dist >= 0.95 on >= 4 of 5 seeds at NB 24, else VOID; NB 24 bars read on REREAD-landed seeds only
  R2    REREAD - BUS256 on HELD_RULE at NB 24: PASS >= 0.05 mean + every landed seed positive; EQUIVALENT |mean| < 0.02; else BETWEEN
  W1    BUS256 - BUS on HELD_RULE at NB 24 >= +0.05 mean -> the bus's T3e loss was WIDTH (P=64); else not width
  R3    (REREAD - BUS) HELD_RULE gap, NB 24 (landed seeds 5-9) minus NB 12 (banked seeds 0-4) >= +0.05 -> re-read's edge grows with depth"""
import json, os

import numpy as np

OUT = ".cache/reread_t3e2_verdict.txt"
SPLITS = ("indist", "held_rule", "held_h", "long", "edge", "swap")


def load(path):
    return {a["arm"]: a["seeds"] for a in json.load(open(path))["arms"]}


if __name__ == "__main__":
    assert not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w")

    def say(s):
        print(s, flush=True); out.write(s + "\n")
    R24, R12w, R12 = load(".cache/reread_t3e2_nb24.json"), load(".cache/reread_t3e2_nb12.json"), load(".cache/reread_t3e_nb12.json")
    for tag, arms in (("NB 24 (s5-9)", R24), ("NB 12 BUS256 (s0-4)", R12w)):
        for a, rs in arms.items():
            assert len(rs) == 5, (tag, a, len(rs))
            say(f"{tag} {a:6s} mean: " + " ".join(f"{k} {np.mean([r[k] for r in rs]):.3f}" for k in SPLITS) +
                f" | in-dist per seed {[round(r['indist'], 3) for r in rs]}")
    land = [i for i, r in enumerate(R24["REREAD"]) if r["indist"] >= 0.95]; ctrl = len(land) >= 4
    gap = lambda R, a, b, idx: [R[a][i]["held_rule"] - R[b][i]["held_rule"] for i in idx]
    say("VERDICTS (T3e2)")
    say(f"C24 CONTROL: {'OK' if ctrl else 'VOID'} (REREAD landed seeds {[5 + i for i in land]})")
    V = lambda s: s if ctrl else "VOID"
    d2 = gap(R24, "REREAD", "BUS256", land); m2 = np.mean(d2)
    r2 = "PASS" if m2 >= 0.05 and all(x > 0 for x in d2) else ("EQUIVALENT" if abs(m2) < 0.02 else "BETWEEN")
    say(f"R2 RE-READ vs THE WIDE BUS (NB 24): {V(r2)} (REREAD - BUS256 on HELD_RULE {m2:+.3f}; per landed seed {[round(x, 3) for x in d2]})")
    dw = gap(R24, "BUS256", "BUS", range(5)); mw = np.mean(dw)
    say(f"W1 WIDTH: {V('PASS' if mw >= 0.05 else 'FAIL')} (BUS256 - BUS on HELD_RULE at NB 24 {mw:+.3f}; per seed {[round(x, 3) for x in dw]})")
    g24, g12 = np.mean(gap(R24, "REREAD", "BUS", land)), np.mean(gap(R12, "REREAD", "BUS", range(5)))
    say(f"R3 DEPTH: {V('PASS' if g24 - g12 >= 0.05 else 'FAIL')} (REREAD - BUS on HELD_RULE: NB 12 {g12:+.3f}, NB 24 {g24:+.3f}; growth {g24 - g12:+.3f})")
    say(f"READ NB 12 width: BUS256 - BUS on HELD_RULE {np.mean(gap({'A': R12w['BUS256'], 'B': R12['BUS']}, 'A', 'B', range(5))):+.3f} (same seeds 0-4)")
    say("READ landed (in-dist >= 0.95): NB 24 " + ", ".join(f"{a} {sum(r['indist'] >= 0.95 for r in rs)}/5" for a, rs in R24.items()) +
        f"; NB 12 BUS256 {sum(r['indist'] >= 0.95 for r in R12w['BUS256'])}/5")
