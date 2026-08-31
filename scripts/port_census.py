"""port_census.py — THE PORT CENSUS (2026-09-01, word given: "let's take
radio interference seriously"). Measures the q_extra antenna per breath:
each organ's injection magnitude (vs the base state), pairwise cosines
between organ injections (alignment/cancellation), and the port SNR.
Requires the _CENSUS hook in phase1_algebra_head.forward (applied by
apply_census_hooks after the running chain exits). Zero training.
Env: PC_CKPT + the organ envs of the artifact being censused.
"""
import os, sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import (build_params, forward, load_alg,
                                 build_slot_masks)
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

vs, vst, vtk, vg, vse = load_alg("test")
p = build_params(0)
sd = safe_load(os.environ["PC_CKPT"])
for k in p:
    if k in sd:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
missing = sorted(set(p) - set(sd))
if missing: print(f"[census] fresh-init: {missing}")

N = 32
acc = {}          # (kb, organ) -> [vecs (H,)], [mags]
base_mag = {}     # kb -> [state rms]
for s0 in range(0, N, 8):
    sl = np.arange(s0, s0 + 8)
    H._CENSUS = []
    ts = Tensor(vst[sl].astype(np.float32), dtype=dtypes.float)
    tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
    o0 = forward(p, ts, tk, se)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(onp0, vse[sl].astype(np.int32))
    H._CENSUS = []
    o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float))
    o["pres"].realize()
    for (kb, organ, arr) in H._CENSUS:
        v = arr.mean(axis=(0, 1)) if arr.ndim == 3 else arr.mean(axis=0)
        m = float(np.sqrt((arr ** 2).mean()))
        acc.setdefault((kb, organ), ([], []))
        acc[(kb, organ)][0].append(v)
        acc[(kb, organ)][1].append(m)
        if organ == "state":
            base_mag.setdefault(kb, []).append(m)
    H._CENSUS = None

organs = sorted({o2 for (_, o2) in acc if o2 != "state"})
kbs = sorted({k for (k, _) in acc})
print(f"[port census] ckpt={os.environ['PC_CKPT'].split('/')[-1]} "
      f"organs={organs}")
print("breath | " + " | ".join(f"{o2}: rms(rel)" for o2 in organs))
for kb in kbs:
    base = np.mean(base_mag.get(kb, [1.0]))
    cells = []
    for o2 in organs:
        if (kb, o2) in acc:
            m = np.mean(acc[(kb, o2)][1])
            cells.append(f"{o2}:{m:.3f}({m / base:.2f}x)")
        else:
            cells.append(f"{o2}:-")
    print(f"  b{kb}:  " + "  ".join(cells))
print("pairwise cos (mean over breaths where both live):")
for i in range(len(organs)):
    for j in range(i + 1, len(organs)):
        cs = []
        for kb in kbs:
            a = acc.get((kb, organs[i])); b = acc.get((kb, organs[j]))
            if a and b:
                va = np.mean(a[0], 0); vb = np.mean(b[0], 0)
                if va.shape != vb.shape:
                    cs = None      # different band (e.g. bank-logit space
                    break          # vs state space) — no in-band interference
                cs.append(float(va @ vb / (np.linalg.norm(va)
                                           * np.linalg.norm(vb) + 1e-9)))
        if cs is None:
            print(f"  {organs[i]} x {organs[j]}: [different band — no "
                  f"in-band interference possible]")
        elif cs:
            print(f"  {organs[i]} x {organs[j]}: {np.mean(cs):+.3f}")
print("[grammar] |cos|<0.2 & small rel-mags = clean spectrum (nulls "
      "stand); cos<-0.3 = measured cancellation (stacked verdicts get "
      "the interference asterisk)")
