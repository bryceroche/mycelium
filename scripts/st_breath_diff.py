"""st_breath_diff.py — where does the walker leave the fused loop? The
walker's cur_bank[k-1] is the state ENTERING breath k; the census hook
records the same state in the fused forward ((kb, "state", cur)).
Per breath: relative max-abs between them. Env: the eq env (ST_CKPT...)."""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import step_trainer as ST
H, p, w, fact0 = ST._eq_setup()
w.walk_forward(fact0)
H._CENSUS = []
o_f = ST._fused_out(H, p, w)
for k in ("res", "args"):
    o_f[k].realize()
cen = {}
for rec in H._CENSUS:
    kb, name = rec[0], rec[1]
    if name == "state" and kb not in cen:
        cen[kb] = rec[2]
    if name == "notebook" and ("nb", kb) not in cen:
        cen[("nb", kb)] = rec[2]
H._CENSUS = None
print(f"[breath-diff] census states at breaths {sorted(k for k in cen if isinstance(k, int))}")
for k in range(1, w.K_B):
    a = w.cur_bank[k - 1].numpy().astype(np.float64); b = cen.get(k)
    if b is None:
        print(f"  breath {k}: no census state"); continue
    b = b.astype(np.float64); d = np.abs(a - b).max(); sc = max(1.0, np.abs(b).max())
    print(f"  state entering breath {k}: rel={d / sc:.3e} maxabs={d:.3e} scale={sc:.2f}")
# the shelf: the walker's banked inks vs the fused loop's (the fused inks are not censused; the read is) — report the walker's ink norms
for j, nb in enumerate(w.nb_bank[:w.K_B]):
    print(f"  walker ink[{j}] norm={float(np.linalg.norm(nb.numpy())):.2f}")
