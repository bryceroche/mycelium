"""gsm_tick_verdict.py -- the TICK arm's pinned bars (ledger 2026-10-30): C0, T1 (TICK - banked NOTEBOOK, TEST step_all, paired by seed);
reads vs the NOTEBOOKR re-run (lag-1 accuracy, numeral-for-slot share, CF / TEST answers). Output .cache/gsm_tick_verdict.txt."""
import json

import numpy as np

T = [json.load(open(f".cache/gsmb_trunk_TICK_s{s}.json")) for s in (0, 1)]
N = [json.load(open(f".cache/gsmb_trunk_NOTEBOOK_s{s}.json")) for s in (0, 1)]
NR = [json.load(open(f".cache/gsmb_trunk_NOTEBOOKR_s{s}.json")) for s in (0, 1)]
L = [f"params TICK {T[0]['params']} vs NOTEBOOK {N[0]['params']}"]
for nm, X in (("TICK", T), ("NOTEBOOK", N), ("NOTEBOOKR", NR)):
    L.append(f"{nm:9s} | " + " | ".join(f"{sp} ans {np.mean([x[f'{sp}_ans'] for x in X]):.3f} step {np.mean([x[f'{sp}_step_all'] for x in X]):.3f}"
                                          for sp in ("ORIG", "CFNUM", "CFNAME", "TEST")))
L.append("reproduction: NOTEBOOKR - NOTEBOOK TEST step " + " / ".join(f"{r['TEST_step_all'] - n['TEST_step_all']:+.4f}" for r, n in zip(NR, N)))
c0 = np.mean([x["ORIG_step_all"] for x in T]); d = [x["TEST_step_all"] - n["TEST_step_all"] for x, n in zip(T, N)]
L.append(f"C0 ORIG step_all (TICK) {c0:.3f}" + (" -> VOID" if c0 < 0.80 else " (not void)"))
v = "THE TICK PAYS" if np.mean(d) >= 0.02 and min(d) > 0 else "HURTS" if np.mean(d) <= -0.02 else "EQUIVALENT"
L.append(f"T1 TICK - NOTEBOOK TEST step {np.mean(d):+.4f} (seeds {d[0]:+.4f} / {d[1]:+.4f}): {v}")
for k in ("lag1_ok", "num_for_slot"):
    for sp in ("TEST", "ORIG"):
        a, b = [x[f"{sp}_{k}"] for x in T], [x[f"{sp}_{k}"] for x in NR]
        L.append(f"read: {sp} {k} TICK {np.mean(a):.3f} vs NOTEBOOKR {np.mean(b):.3f} (seeds {a[0] - b[0]:+.3f} / {a[1] - b[1]:+.3f})")
for sp in ("CFNUM", "CFNAME", "TEST"):
    x = [t[f"{sp}_ans"] - n[f"{sp}_ans"] for t, n in zip(T, N)]
    L.append(f"read: TICK - NOTEBOOK {sp} answer {np.mean(x):+.4f} (seeds {x[0]:+.4f} / {x[1]:+.4f})")
print("\n".join(L)); open(".cache/gsm_tick_verdict.txt", "w").write("\n".join(L) + "\n")
