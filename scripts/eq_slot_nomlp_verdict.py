"""eq_slot_nomlp_verdict.py -- the NOMLP arm's pinned bars (ledger 2026-10-27). Bryce's breathing transformer = MHA WITHOUT THE MLP;
the 10-27 BREATH block carried a free feed-forward (the T3d rule's violation). Reads .cache/eq_slot_src_{NOMLP,BREATH}_STU_s{0,1}.json
(eq_slot_source_read.py on the ORIGINAL mix) + .cache/eq_slot_{NOMLP,BREATH}_STU_PD_s{0,1}.json (the capped mix); zero training.
  N1  THE FLASHCARD (original mix, seed means): gap = val-GSM8K-SEEN final exact - holdout final exact. NOMLP gap < 0.08 GONE;
      NOMLP gap <= BREATH gap - 0.15 SHRUNK; else PERSISTS. Guard read: NOMLP holdout exact vs BREATH's (a gap closed by
      losing the holdout too is capacity, not reading)
  N2  READING UNSEEN PROBLEMS (capped mix): NOMLP_PD - BREATH_PD paired holdout solver per seed: NOMLP WINS mean >= +0.020 and
      both positive; EQUIVALENT |mean| < 0.020; MLP WINS <= -0.020 both negative; else BETWEEN
  N3  E1 for NOMLP_PD: >= TEXT + 0.10 on BOTH seeds"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import eq_slot_grade as EG
from eq_slot_pd_verdict import paired

OUT = ".cache/eq_slot_nomlp_verdict.txt"
if __name__ == "__main__":
    assert not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w")

    def say(s):
        print(s, flush=True); out.write(s + "\n")
    SRC = {a: [json.load(open(f".cache/eq_slot_src_{a}_STU_s{s}.json")) for s in (0, 1)] for a in ("NOMLP", "BREATH")}
    PD = {a: [json.load(open(f".cache/eq_slot_{a}_STU_PD_s{s}.json")) for s in (0, 1)] for a in ("NOMLP", "BREATH")}
    g = {}
    for a, js in SRC.items():
        seen = np.mean([j["cells"]["val gsm8k seen"]["exact"][-1] for j in js]); hold = np.mean([j["cells"]["holdout"]["exact"][-1] for j in js])
        g[a] = (seen, hold, seen - hold); say(f"{a:6s} original mix: val GSM8K seen {seen:.3f} | holdout {hold:.3f} | gap {seen - hold:+.3f} "
                                              f"| LOOKUP {js[0]['cells']['val gsm8k seen']['lookup']:.3f}")
    rows = [r for r, _ in PD["NOMLP"][0]["per_row"]]; T = {t["row"]: t for t in EG.rows("holdout")[0]}
    text = float(np.mean([EG.grade(T[r], EG.text_assign(T[r])) for r in rows]))
    for a, js in PD.items():
        for s, j in enumerate(js):
            say(f"{a:6s} capped s{s}: solver {j['holdout_solver']:.4f} | exact-order per block {' '.join(f'{x:.3f}' for x in j['holdout_exact'])} | params {j['params']:,}")
    say("VERDICTS (NOMLP)")
    n1 = "GONE" if g["NOMLP"][2] < 0.08 else "SHRUNK" if g["NOMLP"][2] <= g["BREATH"][2] - 0.15 else "PERSISTS"
    say(f"N1 THE FLASHCARD: {n1} (NOMLP gap {g['NOMLP'][2]:+.3f} vs BREATH {g['BREATH'][2]:+.3f}; guard: holdout NOMLP {g['NOMLP'][1]:.3f} vs BREATH {g['BREATH'][1]:.3f})")
    m, ds = paired(PD["NOMLP"], PD["BREATH"], say, "NOMLP_PD - BREATH_PD")
    say(f"N2 READING UNSEEN PROBLEMS: {'NOMLP WINS' if m >= 0.020 and all(d > 0 for d in ds) else 'EQUIVALENT' if abs(m) < 0.020 else 'MLP WINS' if m <= -0.020 and all(d < 0 for d in ds) else 'BETWEEN'} (mean {m:+.4f})")
    e1 = all(j["holdout_solver"] >= text + 0.10 for j in PD["NOMLP"])
    say(f"N3 E1 FOR NOMLP_PD: {'PASS' if e1 else 'FAIL'} ({[round(j['holdout_solver'], 4) for j in PD['NOMLP']]} vs {text + 0.10:.4f})")
