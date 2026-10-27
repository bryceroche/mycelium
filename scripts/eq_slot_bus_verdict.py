"""eq_slot_bus_verdict.py -- THE BUS ON REAL TEXT pinned bars (ledger 2026-10-27). Reads .cache/eq_slot_{BREATH,BUS}_STU_s{0,1}.json.
  B1  BREATH (re-read every breath) - BUS (one-shot T256 bus, PB=256), paired holdout rows per seed: RE-READ WINS mean >= +0.020 and
      both seeds positive; EQUIVALENT |mean| < 0.020; BUS WINS mean <= -0.020 and both seeds negative; else BETWEEN
  READ the capacity axis: the same difference by text-length tercile (real tokens), and the per-block exact-order curves"""
import json, math, os

import numpy as np

OUT = ".cache/eq_slot_bus_verdict.txt"
if __name__ == "__main__":
    assert not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w")

    def say(s):
        print(s, flush=True); out.write(s + "\n")
    J = {(a, s): json.load(open(f".cache/eq_slot_{a}_STU_s{s}.json")) for a in ("BREATH", "BUS") for s in (0, 1)}
    rows = [r for r, _ in J["BREATH", 0]["per_row"]]; assert all([r for r, _ in j["per_row"]] == rows for j in J.values())
    ntok = np.load(".cache/phase1_alg_states_wildholdSTU.npz")["tokmask"].sum(1)[rows]; cut = np.quantile(ntok, [1 / 3, 2 / 3])
    terc = np.digitize(ntok, cut)
    for (a, s), j in sorted(J.items()):
        say(f"{a:6s} s{s}: solver {j['holdout_solver']:.4f} | exact-order per block {' '.join(f'{x:.3f}' for x in j['holdout_exact'])} | params {j['params']:,}")
    ds = []
    for s in (0, 1):
        ob = np.array([o for _, o in J["BREATH", s]["per_row"]], bool); ou = np.array([o for _, o in J["BUS", s]["per_row"]], bool)
        b, c, n = int((ob & ~ou).sum()), int((~ob & ou).sum()), len(ob); se = math.sqrt((b + c) / n - ((b - c) / n) ** 2) / math.sqrt(n)
        ds.append(ob.mean() - ou.mean()); say(f"  seed {s}: BREATH - BUS {ds[-1]:+.4f} (discordant {b}/{c}, paired SE {se:.4f}) | by length tercile " +
            " ".join(f"T{t} {ob[terc == t].mean() - ou[terc == t].mean():+.3f}" for t in range(3)))
    m = float(np.mean(ds))
    v = ("RE-READ WINS" if m >= 0.020 and all(d > 0 for d in ds) else "EQUIVALENT" if abs(m) < 0.020 else
         "BUS WINS" if m <= -0.020 and all(d < 0 for d in ds) else "BETWEEN")
    say("VERDICTS (the bus on real text)")
    say(f"B1 RE-READ vs THE T256 BUS: {v} (mean {m:+.4f}); tercile token bounds {cut.tolist()}")
