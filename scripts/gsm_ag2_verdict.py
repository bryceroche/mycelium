"""AG2 THE CROSS-READER AGREEMENT verdict (ledger 2026-11-04, word given): the two pairs BT4M8_sS x BT4RB8_sS (same task, different reader
states = more independent views than two seeds on one reader). Bars pinned in the ledger before the dumps existed. -> .cache/gsm_ag2_verdict.txt"""
import json
import numpy as np

P = [json.load(open(f".cache/gsmb_agree_BT4M8_s{s}_BT4RB8_s{s}.json"))["TEST"] for s in (0, 1)]
d = [p[f"s{i}"]["diff"] for p in P for i in (0, 1)]
sh = float(np.mean([p["s0"]["shared"] for p in P]))
cov = float(np.mean([p["cert"]["coverage"] for p in P])); prec = [p["cert"]["precision"] for p in P]; base = [p["cert"]["base"] for p in P]
V = [f"AG2-1: TEST ans AGREE - BEAM-1 over the four model reads " + " / ".join(f"{x:+.4f}" for x in d) + f" (mean {np.mean(d):+.4f}; bar +0.010, all > 0) -> "
     f"{'PASS' if np.mean(d) >= 0.010 and min(d) > 0 else 'FAIL'}",
     f"AG2-I: TEST shared-value rate across readers {sh:.3f} vs one reader (AG, two seeds) 0.348; bar > 0.348 + 0.03 -> {'PASS' if sh > 0.378 else 'FAIL'}",
     f"AG2-C: TEST certificate precision " + " / ".join(f"{x:.4f}" for x in prec) + " vs 3 x BEAM-1 " + " / ".join(f"{3 * x:.4f}" for x in base)
     + f" (coverage {cov:.3f}; both pairs) -> {'PASS' if all(p >= 3 * b for p, b in zip(prec, base)) else 'FAIL'}"]
print("\n".join(V)); open(".cache/gsm_ag2_verdict.txt", "w").write("\n".join(V) + "\n")
