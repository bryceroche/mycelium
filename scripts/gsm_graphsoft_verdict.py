"""gsm_graphsoft_verdict.py -- the GRAPH RESCUE's pinned bars (ledger 2026-10-30 06:51, soft switches): C0 (ORACLE ORIG fit), G1 (ORACLE -
NOTEBOOK), G2 (SOFT - NOTEBOOK), paired by seed on TEST step_all against the step-0 grid's NOTEBOOK; reads OPEN, HARD, owner / event
choice, CF answers. Output .cache/gsm_graphsoft_verdict.txt."""
import json

import numpy as np

G = [json.load(open(f".cache/gsmb_graph_GRAPHSOFT_s{s}.json")) for s in (0, 1)]
N = [json.load(open(f".cache/gsmb_trunk_NOTEBOOK_s{s}.json")) for s in (0, 1)]
L = [f"params GRAPHSOFT {G[0]['params']} vs NOTEBOOK {N[0]['params']}"]
for m in ("SOFT", "ORACLE", "OPEN", "HARD"):
    L.append(f"{m:8s} | " + " | ".join(f"{sp} ans {np.mean([g[f'{sp}_{m}']['ans'] for g in G]):.3f} step {np.mean([g[f'{sp}_{m}']['step_all'] for g in G]):.3f}"
                                         for sp in ("ORIG", "CFNUM", "CFNAME", "TEST")))
L.append("NOTEBOOK | " + " | ".join(f"{sp} ans {np.mean([n[f'{sp}_ans'] for n in N]):.3f} step {np.mean([n[f'{sp}_step_all'] for n in N]):.3f}"
                                     for sp in ("ORIG", "CFNUM", "CFNAME", "TEST")))
c0 = np.mean([g["ORIG_ORACLE"]["step_all"] for g in G])
d = lambda m, sp="TEST", k="step_all", nk="step_all": [g[f"{sp}_{m}"][k] - n[f"{sp}_{nk}"] for g, n in zip(G, N)]
g1, g2 = d("ORACLE"), d("SOFT")
L.append(f"C0 ORIG step_all (ORACLE) {c0:.3f}" + (" -> VOID" if c0 < 0.80 else " (not void)")
         + f" | read: ORIG step_all (SOFT) {np.mean([g['ORIG_SOFT']['step_all'] for g in G]):.3f}")
L.append(f"G1 ORACLE - NOTEBOOK TEST step {np.mean(g1):+.4f} (seeds {g1[0]:+.4f} / {g1[1]:+.4f}): "
         + ("THE TARGET PAYS" if np.mean(g1) >= 0.03 and min(g1) > 0 else "the target does not pay"))
v2 = ("THE QUERY IS LEARNABLE" if np.mean(g2) >= 0.02 and min(g2) > 0 else "HURTS" if np.mean(g2) <= -0.02 else "EQUIVALENT")
L.append(f"G2 SOFT - NOTEBOOK TEST step {np.mean(g2):+.4f} (seeds {g2[0]:+.4f} / {g2[1]:+.4f}): {v2}")
for m in ("OPEN", "HARD"):
    x = d(m); L.append(f"read: {m} - NOTEBOOK TEST step {np.mean(x):+.4f} (seeds {x[0]:+.4f} / {x[1]:+.4f})")
for sp in ("CFNUM", "CFNAME", "TEST"):
    x = d("SOFT", sp, "ans", "ans"); L.append(f"read: SOFT - NOTEBOOK {sp} answer {np.mean(x):+.4f} (seeds {x[0]:+.4f} / {x[1]:+.4f})")
L.append(f"read: TEST owner choice {np.mean([g['TEST_SOFT']['owner_ok'] for g in G]):.3f}, event choice {np.mean([g['TEST_SOFT']['event_ok'] for g in G]):.3f} (SOFT, gold-keyed operands)")
print("\n".join(L)); open(".cache/gsm_graphsoft_verdict.txt", "w").write("\n".join(L) + "\n")
