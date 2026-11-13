"""step_r.py -- STEP-R (ledger 2026-11-13 15:12, zero-GPU): is the trained control's norm growth linear (steps ~ 1/b) or geometric?
Fits the banked norm census (mean token norm after breaths 1..10) both ways on the four controls -> .cache/step_r.txt"""
import json

import numpy as np

out, wins = [], []
for arm, cov in (("NOMLP", "0.25"), ("NOTEBOOK", "0.0625")):
    for s in (0, 1):
        n = np.array(json.load(open(f".cache/rel_dsl_TG{arm}_depth2c{cov}_s{s}.json"))["norm_census"]["all"]); b = np.arange(1, len(n) + 1)
        r2 = lambda y: 1 - ((y - np.polyval(np.polyfit(b, y, 1), b)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rl, rg = r2(n), r2(np.log(n)); wins.append(rl > rg)
        out.append(f"{arm} s{s}: norms {' '.join(f'{x:.1f}' for x in n)} | increments {' '.join(f'{x:.1f}' for x in np.diff(n))} | "
                   f"ratios {' '.join(f'{x:.3f}' for x in n[1:] / n[:-1])} | R^2 linear {rl:.4f} geometric {rg:.4f}")
out.append(f"STEP-R: {'LINEAR' if all(wins) else 'GEOMETRIC' if not any(wins) else 'MIXED'} (linear wins {sum(wins)}/4)")
txt = "\n".join(out); print(txt); open(".cache/step_r.txt", "w").write(txt + "\n")
