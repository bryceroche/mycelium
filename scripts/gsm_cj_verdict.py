"""gsm_cj_verdict.py -- the combination-split probe's frame (ledger 2026-11-08 pin). Reads .cache/gsm_cj_{ARM}.npz.
VALID: |re-scored TEST step_all - banked| <= 0.003 per arm, else that arm is VOID.
CJ (skeleton seen - unseen, stratified by program length m; strata with n_seen or n_unseen < 20 dropped):
  CJ-MEM: CJ >= +0.050 on both seeds -> the arm leans on seen skeletons (a template cross join)
  CJ-FREE: CJ <= +0.020 on both seeds -> no skeleton cross join; the TEST wall is elsewhere (text <-> structure binding)
  else UNCLEAR.  CJ-DIFF: BTC2 - BT4RB8 per seed; a difference is named only if >= 0.020 in size with the same sign on both seeds.
-> .cache/gsm_cj_verdict.txt"""
import os

import numpy as np


def cj(d):
    ok, sn, m = d["ok"], d["seen"], d["m"]; num = den = 0.0; rows = []
    for mm in np.unique(m):
        s = (m == mm) & sn; u = (m == mm) & ~sn
        if s.sum() < 20 or u.sum() < 20: continue
        w = s.sum() * u.sum() / (s.sum() + u.sum()); dd = ok[s].mean() - ok[u].mean(); num += w * dd; den += w
        rows.append(f"m={mm}: seen {ok[s].mean():.3f} (n {s.sum()}) unseen {ok[u].mean():.3f} (n {u.sum()})")
    return num / den, rows


out = []; R = {}
for a in ("BT4RB8", "BTC2"):
    for s in (0, 1):
        f = f".cache/gsm_cj_{a}_s{s}.npz"
        if not os.path.exists(f): out.append(f"{a}_s{s}: MISSING"); continue
        d = np.load(f); acc = d["ok"].mean(); dv = abs(acc - float(d["banked"]))
        if dv > 0.003: out.append(f"{a}_s{s}: VOID (re-scored {acc:.4f} vs banked {float(d['banked']):.4f})"); continue
        c, rows = cj(d); R[a, s] = c
        out.append(f"{a}_s{s}: TEST step {acc:.4f} (banked |d| {dv:.4f}) | raw seen {d['ok'][d['seen']].mean():.4f} unseen "
                   f"{d['ok'][~d['seen']].mean():.4f} | CJ (m-stratified) {c:+.4f}")
        out += ["    " + r for r in rows]
for a in ("BT4RB8", "BTC2"):
    if (a, 0) in R and (a, 1) in R:
        c0, c1 = R[a, 0], R[a, 1]
        v = "CJ-MEM" if min(c0, c1) >= 0.050 else "CJ-FREE" if max(c0, c1) <= 0.020 else "UNCLEAR"
        out.append(f"{a}: CJ s0 {c0:+.4f} s1 {c1:+.4f} -> {v}")
if all((a, s) in R for a in ("BT4RB8", "BTC2") for s in (0, 1)):
    d0, d1 = R["BTC2", 0] - R["BT4RB8", 0], R["BTC2", 1] - R["BT4RB8", 1]
    named = min(abs(d0), abs(d1)) >= 0.020 and np.sign(d0) == np.sign(d1)
    out.append(f"CJ-DIFF (BTC2 - BT4RB8): s0 {d0:+.4f} s1 {d1:+.4f} -> {'NAMED' if named else 'none named'}")
txt = "\n".join(out); print(txt); open(".cache/gsm_cj_verdict.txt", "w").write(txt + "\n")
