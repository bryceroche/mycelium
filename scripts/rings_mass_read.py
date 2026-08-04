"""rings_mass_read.py — THE MASS READ (2026-08-04; #133's first
evidence + rung 3's mechanism readout): under the rings ckpt, per-slot
final commitment mass vs slot correctness on adupheld. Questions,
pinned: (a) does mass separate correct from incorrect slots (the
trained when-signal's first calibration)? (b) where does mass sit —
did the pawl engage at all at this dose (init-closed bias -4;
mass ~0.036 at init)?"""
import os, sys, json
os.environ["ALG_BREATH"] = "3"; os.environ["ALG_RINGS"] = "1"
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
os.environ.setdefault("ALG_WIDE", "1")
os.environ.setdefault("ALG_TEST", ".cache/gen17_adup_held.jsonl")
os.environ.setdefault("ALG_TEST_NAME", "adupheld")
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from phase1_algebra_head import (build_params, forward, load_alg,
                                 build_slot_masks, L_FAC)

samples, states, tokmask, gold, sent = load_alg("test")
p = build_params(0)
sd = safe_load(".cache/g24_rings_rings.safetensors")
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
n = len(samples)
mass_ok, mass_bad = [], []
for s0 in range(0, n, 8):
    sl = np.arange(s0, min(s0 + 8, n))
    pad = 8 - len(sl)
    sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    t_tr = Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float)
    t_tk = Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float)
    t_se = Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int)
    o0 = forward(p, t_tr, t_tk, t_se)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(onp0, sent[sl_p])
    o = forward(p, t_tr, t_tk, t_se, slot_mask=Tensor(mk, dtype=dtypes.float))
    m = o["cmt_m"].realize().numpy()
    ft = o["ftype"].realize().numpy().argmax(-1)
    rs = o["res"].realize().numpy().argmax(-1)
    for bi, i in enumerate(sl):
        i = int(i)
        for j in range(L_FAC):
            if gold["presence"][i, j] <= 0:
                continue
            ok = (ft[bi, j] == gold["ftype"][i, j]) and \
                 (rs[bi, j] == gold["res"][i, j])
            (mass_ok if ok else mass_bad).append(float(m[bi, j]))
mo, mb = np.array(mass_ok), np.array(mass_bad)
print(f"[mass] correct slots n={len(mo)}: mean {mo.mean():.4f} "
      f"p50 {np.median(mo):.4f} p90 {np.percentile(mo,90):.4f}")
if len(mb):
    print(f"[mass] wrong slots  n={len(mb)}: mean {mb.mean():.4f} "
          f"p50 {np.median(mb):.4f} p90 {np.percentile(mb,90):.4f}")
    from scipy.stats import mannwhitneyu
    u, pv = mannwhitneyu(mo, mb, alternative="greater")
    auc = u / (len(mo) * len(mb))
    print(f"[mass] separation AUC={auc:.3f} p={pv:.2e} "
          f"(correct-slots carry more mass?)")
else:
    print("[mass] zero wrong slots on this fixture (ceiling) — "
          "separation unreadable here")
json.dump({"n_ok": len(mo), "n_bad": len(mb),
           "mean_ok": float(mo.mean()),
           "mean_bad": float(mb.mean()) if len(mb) else None},
          open(".cache/rings_mass_read.json", "w"), indent=1)
print("[saved] .cache/rings_mass_read.json")
