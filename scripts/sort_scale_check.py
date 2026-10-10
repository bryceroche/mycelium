"""sort_scale_check.py -- THE SORTING ROOM form 2's scale check (2026-10-09, coordinator's
request after the 574.36 gate finding): measures the per-dim std of `waist` (what the bank's
attn_wk/attn_wv actually read) with ALG_SORT unset vs ALG_SORT=4 (post-ReZero-fix) on the SAME
trunk states, to confirm the ReZero zero-door makes the sorting room's output identical to the
pre-sort waist at birth -- not merely similar in scale, bit-for-bit.
Env: the family envs (ALG_SORT=4 required); reads a slice of the tiny64 fixture's precomputed
train states (no warm checkpoint needed -- this probes INIT-time behavior only).
"""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import build_params, load_alg, _sort_room
from tinygrad import Tensor, dtypes

assert int(os.environ.get("ALG_SORT", "0")) > 0, "ALG_SORT=<N> is required"

vs, vst, vtk, vg, vse = load_alg("train")
sl = np.arange(0, 8); B = len(sl)
p = build_params(0)

ts = Tensor(vst[sl].astype(np.float32), dtype=dtypes.float)
tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)

if ts.dtype != dtypes.float:
    ts = ts.cast(dtypes.float)
waist0 = (ts @ p["waist_w"] + p["waist_b"]).gelu() + p["sent_emb"][se]   # forward()'s own recipe, pre-sort
waist1 = _sort_room(p, waist0, tk, B)                                    # the sorting room's output

tm = tk.reshape(B, -1, 1)
w0 = (waist0 * tm).realize().numpy()
w1 = (waist1 * tm).realize().numpy()
mask = (tk.realize().numpy() > 0)

std0 = w0[mask].std()
std1 = w1[mask].std()
maxdiff = np.abs(w0 - w1).max()

print(f"[sort-scale] per-dim std of waist, real tokens only: UNSET(pre-sort)={std0:.6f} "
      f"ALG_SORT={os.environ['ALG_SORT']}(post-sort)={std1:.6f}")
print(f"[sort-scale] max|waist_post - waist_pre| = {maxdiff:.3e} "
      f"({'EXACT (ReZero identity holds)' if maxdiff == 0.0 else 'NONZERO -- the ReZero door is not exact'})")
