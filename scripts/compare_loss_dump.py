"""Compare two GC_DUMP npz files (loss + all param grads) under the pinned tolerance.
usage: compare_loss_dump.py before.npz after.npz [loss_rel=1e-6] [grad_rel=1e-5]"""
import sys, numpy as np
a = np.load(sys.argv[1]); b = np.load(sys.argv[2])
lt = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-6; gt = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
keys = sorted(set(a.files) | set(b.files)); worst = []; miss = [k for k in keys if k not in a.files or k not in b.files]
for k in keys:
    if k in miss: continue
    x = a[k].astype(np.float64); y = b[k].astype(np.float64)
    if x.shape != y.shape: worst.append((np.inf, k, "shape")); continue
    rel = np.abs(x - y).max() / (np.abs(x).max() + 1e-30)
    worst.append((rel, k, f"max|x|={np.abs(x).max():.3e}"))
worst.sort(reverse=True)
lk = [w for w in worst if w[1] == "loss"]; gk = [w for w in worst if w[1] != "loss"]
print(f"[compare] {len(keys)} keys ({len(miss)} missing: {miss[:5]})")
if lk: print(f"[compare] loss before={float(a['loss']):.8f} after={float(b['loss']):.8f} rel={lk[0][0]:.2e} (bar {lt:.0e}) -> {'PASS' if lk[0][0] <= lt else 'FAIL'}")
print(f"[compare] grads: worst rel {gk[0][0]:.2e} on {gk[0][1]} ({gk[0][2]}); bar {gt:.0e}; n_over={sum(1 for w in gk if w[0] > gt)}/{len(gk)} -> {'PASS' if gk and gk[0][0] <= gt and not miss else 'FAIL'}")
for w in gk[:6]: print(f"   {w[0]:.2e}  {w[1]}  {w[2]}")
