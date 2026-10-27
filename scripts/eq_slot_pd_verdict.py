"""eq_slot_pd_verdict.py -- the problem-disjoint rerun's pinned bars (ledger 2026-10-27). Reads .cache/eq_slot_{ARM}_STU_PD_s{0,1}.json
(EQ_SPLIT=problem EQ_CAP=1) and the 10-27 runs .cache/eq_slot_{ARM}_STU_s{0,1}.json; zero training. Holdout solver-graded.
  P1  THE CAP: BREATH_PD - BREATH (paired holdout rows, per seed): LIFT mean >= +0.020 and both seeds positive; EQUIVALENT
      |mean| < 0.020; HURT mean <= -0.020 and both negative; else BETWEEN
  P2  E1 on the capped mix: BREATH_PD >= TEXT + 0.10 on BOTH seeds
  P3  E2 on the capped mix: BREATH_PD - VANILLA_PD, same frame as P1 (PASS / EQUIVALENT / VANILLA WINS / BETWEEN)"""
import json, math, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import eq_slot_grade as EG

OUT = ".cache/eq_slot_pd_verdict.txt"


def paired(A, B, say, label):
    ds = []
    for s in (0, 1):
        pa, pb = dict(map(tuple, A[s]["per_row"])), dict(map(tuple, B[s]["per_row"])); rows = sorted(set(pa) & set(pb))
        assert len(rows) == len(pa) == len(pb), (label, s, "arms graded different rows")
        a = np.array([pa[r] for r in rows], bool); b = np.array([pb[r] for r in rows], bool); n = len(rows)
        x, y = int((a & ~b).sum()), int((~a & b).sum()); se = math.sqrt((x + y) / n - ((x - y) / n) ** 2) / math.sqrt(n)
        ds.append(a.mean() - b.mean()); say(f"  {label} seed {s}: {ds[-1]:+.4f} (discordant {x}/{y}, paired SE {se:.4f})")
    return float(np.mean(ds)), ds


if __name__ == "__main__":
    assert not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w")

    def say(s):
        print(s, flush=True); out.write(s + "\n")
    L = lambda a, tag, s: json.load(open(f".cache/eq_slot_{a}_STU{tag}_s{s}.json"))
    PD = {a: [L(a, "_PD", s) for s in (0, 1)] for a in ("BREATH", "VANILLA")}; OG = {"BREATH": [L("BREATH", "", s) for s in (0, 1)]}
    assert all(j["split"] == "problem" and j["cap"] == 1 for v in PD.values() for j in v), "PD runs not problem-split/capped"
    rows = [r for r, _ in PD["BREATH"][0]["per_row"]]; T = {t["row"]: t for t in EG.rows("holdout")[0]}
    text = float(np.mean([EG.grade(T[r], EG.text_assign(T[r])) for r in rows]))
    say(f"holdout rows {len(rows)} | TEXT solver {text:.4f} -> E1 bar {text + 0.10:.4f} | PD train rows {PD['BREATH'][0]['n_train']}")
    for a, v in list(PD.items()) + [("BREATH(10-27)", OG["BREATH"])]:
        for s, j in enumerate(v):
            say(f"{a:13s} s{s}: solver {j['holdout_solver']:.4f} | exact-order per block {' '.join(f'{x:.3f}' for x in j['holdout_exact'])}")
    say("VERDICTS (problem-disjoint rerun)")
    m, ds = paired(PD["BREATH"], OG["BREATH"], say, "BREATH_PD - BREATH")
    say(f"P1 THE CAP: {'LIFT' if m >= 0.020 and all(d > 0 for d in ds) else 'EQUIVALENT' if abs(m) < 0.020 else 'HURT' if m <= -0.020 and all(d < 0 for d in ds) else 'BETWEEN'} (mean {m:+.4f})")
    e1 = all(j["holdout_solver"] >= text + 0.10 for j in PD["BREATH"])
    say(f"P2 E1 ON THE CAPPED MIX: {'PASS' if e1 else 'FAIL'} ({[round(j['holdout_solver'], 4) for j in PD['BREATH']]} vs {text + 0.10:.4f})")
    m, ds = paired(PD["BREATH"], PD["VANILLA"], say, "BREATH_PD - VANILLA_PD")
    say(f"P3 THE FAIR FIGHT, CAPPED: {'PASS' if m >= 0.020 and all(d > 0 for d in ds) else 'EQUIVALENT' if abs(m) < 0.020 else 'VANILLA WINS' if m <= -0.020 and all(d < 0 for d in ds) else 'BETWEEN'} (mean {m:+.4f})")
