"""st_organ_diff.py — which ORGAN inside a breath differs between the walker's
segment and the fused loop? Census-arm both, compare per (kb, organ)."""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import step_trainer as ST
H, p, w, fact0 = ST._eq_setup()
# the walker: one segment at a time with the census armed
wal = {}
H._CENSUS = None
w.put(w.b_facts[0], fact0); r = w.s0_fn(); w.waist_bank.assign(r[0]); w.vst_base_bank.assign(r[1]); w.cur_bank[0].assign(r[2]); w.fat_bank.assign(r[3]); w.mask_bank.assign(r[4]); w.Tensor.realize(w.waist_bank, w.vst_base_bank, w.cur_bank[0], w.fat_bank, w.mask_bank)
for k in range(1, w.K_B):
    H._CENSUS = []
    outs = w.fwd_fns[k]()
    for rec in H._CENSUS:
        wal.setdefault((rec[0], rec[1]), rec[2])
    H._CENSUS = None
    # bank exactly as walk_forward does
    w.cur_bank[k].assign(outs[0]); i = 1
    if w.notebook:
        if k == 1: w.nb_bank[0].assign(outs[1]); w.nb_bank[1].assign(outs[2]); i = 3
        else: w.nb_bank[k].assign(outs[1]); i = 2
    if w.garage: w.gar_bank[k - 1].assign(outs[i]); i += 1
    if w.snaps_on:
        for jj in range(4): w.snap_bank[k][jj].assign(outs[i + jj])
        i += 4
        if k >= 2: w.mhp_bank[k].assign(outs[i])
    w.Tensor.realize(w.cur_bank[k], *(w.nb_bank[:k + 1] if w.notebook else []), *(w.gar_bank[:k] if w.garage else []), *(w.snap_bank[k] if w.snaps_on else []))
    w.put(w.b_facts[k], fact0)
# the fused loop
H._CENSUS = []
o_f = ST._fused_out(H, p, w); o_f["res"].realize()
fus = {}
for rec in H._CENSUS:
    fus.setdefault((rec[0], rec[1]), rec[2])
H._CENSUS = None
keys = sorted(set(wal) & set(fus), key=lambda x: (x[0], x[1]))
print(f"[organ-diff] {len(keys)} shared (breath, organ) records; missing on the walker side: {sorted(set(fus) - set(wal))[:8]}")
worst = []
for kb, org in keys:
    a, b = np.asarray(wal[(kb, org)], np.float64), np.asarray(fus[(kb, org)], np.float64)
    if a.shape != b.shape: print(f"  b{kb} {org}: SHAPE {a.shape} vs {b.shape}"); continue
    d = np.abs(a - b).max(); sc = max(1.0, np.abs(b).max()); worst.append((d / sc, kb, org, d, sc))
for rel, kb, org, d, sc in sorted(worst, key=lambda x: (x[1], -x[0])):
    flag = " <--" if rel > 1e-6 else ""
    print(f"  b{kb} {org:14s} rel={rel:.2e} maxabs={d:.2e} scale={sc:.2f}{flag}")
