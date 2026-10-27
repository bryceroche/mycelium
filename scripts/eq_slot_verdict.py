"""eq_slot_verdict.py -- the eq_slot pinned bars (ledger 2026-10-27). Reads .cache/eq_slot_{BREATH,VANILLA}_{R}_s{0,1}.json
(R = the reader the chain picked, argv[1]); zero training. Holdout = the wild fixture, solver-graded against the custody key.
  E1  BREATH solver >= TEXT + 0.10 on BOTH seeds (TEXT = the identity reading, solver-graded on the same rows)
  E2  BREATH - VANILLA (paired rows, per seed): PASS mean >= +0.020 and both seeds positive; EQUIVALENT |mean| < 0.020;
      VANILLA WINS mean <= -0.020 and both seeds negative; else BETWEEN"""
import json, math, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import eq_slot_grade as EG

R = sys.argv[1]; OUT = f".cache/eq_slot_verdict_{R}.txt"
if __name__ == "__main__":
    assert not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w")

    def say(s):
        print(s, flush=True); out.write(s + "\n")
    J = {(a, s): json.load(open(f".cache/eq_slot_{a}_{R}_s{s}.json")) for a in ("BREATH", "VANILLA") for s in (0, 1)}
    rows = [r for r, _ in J["BREATH", 0]["per_row"]]
    assert all([r for r, _ in j["per_row"]] == rows for j in J.values()), "arms graded different rows"
    T = {t["row"]: t for t in EG.rows("holdout")[0]}
    text = float(np.mean([EG.grade(T[r], EG.text_assign(T[r])) for r in rows]))
    say(f"reader {R} | holdout rows {len(rows)} | TEXT solver {text:.4f} -> E1 bar {text + 0.10:.4f}")
    for (a, s), j in sorted(J.items()):
        say(f"{a:7s} s{s}: solver {j['holdout_solver']:.4f} | exact-order per block {' '.join(f'{x:.3f}' for x in j['holdout_exact'])} | params {j['params']:,}")
    e1 = all(J["BREATH", s]["holdout_solver"] >= text + 0.10 for s in (0, 1))
    say("VERDICTS (eq_slot)")
    say(f"E1 THE BREATHING TRANSFORMER LEARNS THE ORDER TABLE: {'PASS' if e1 else 'FAIL'} (BREATH {[round(J['BREATH', s]['holdout_solver'], 4) for s in (0, 1)]} vs bar {text + 0.10:.4f})")
    ds = []
    for s in (0, 1):
        ob = np.array([o for _, o in J["BREATH", s]["per_row"]], bool); ov = np.array([o for _, o in J["VANILLA", s]["per_row"]], bool)
        b, c, n = int((ob & ~ov).sum()), int((~ob & ov).sum()), len(ob)
        se = math.sqrt((b + c) / n - ((b - c) / n) ** 2) / math.sqrt(n); ds.append(ob.mean() - ov.mean())
        say(f"  seed {s}: BREATH - VANILLA {ds[-1]:+.4f} (discordant {b}/{c}, paired SE {se:.4f})")
    m = float(np.mean(ds))
    e2 = ("PASS" if m >= 0.020 and all(d > 0 for d in ds) else "EQUIVALENT" if abs(m) < 0.020 else
          "VANILLA WINS" if m <= -0.020 and all(d < 0 for d in ds) else "BETWEEN")
    say(f"E2 THE FAIR FIGHT (compute-matched; VANILLA has {J['VANILLA', 0]['params'] / J['BREATH', 0]['params']:.1f}x the params): {e2} (mean {m:+.4f})")
