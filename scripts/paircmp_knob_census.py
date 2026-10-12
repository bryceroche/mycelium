"""paircmp_knob_census.py -- THE PAIRWISE COMPARATOR ROAD's knob census (2026-10-09; the
pre/post knob law applied after SM_241's autopsy found that term INERT BY SCALE: trained
contribution ~0.2% of the res logit's spread). Prints, for a forward pass over GP_ROWS rows,
the comparator term's std across the 24 candidates (pre-scale raw MLP output and post-scale,
i.e. x ALG_PAIRCMP_SCALE) against the res bilinear's OWN std across candidates, at whichever
breath(s) _heads_of is actually called on this path (eager, no JIT -- the grad probe's
convention; sets the module global _PAIRCMP_CENSUS to a list before calling forward(), which is
the ONLY thing that turns this reporting on -- nothing in do_train's real training path ever
sets it, so the exact same code inside _heads_of is inert there).

BAR (the arm's own, pinned by the coordinator 2026-10-09): the comparator term's POST-SCALE std
at the LAST recorded breath must be >= 10% of the res bilinear's std at that same breath, else
the arm is VOID by the knob law (not a verdict on the form itself).

usage: ALG_PAIRCMP=1 [PC_CKPT=<ckpt>] [GP_ROWS=0:2] .venv/bin/python3 scripts/paircmp_knob_census.py
"""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import build_params, forward, load_alg, ident_build_array, T_ALG
from tinygrad import Tensor, dtypes

assert int(os.environ.get("ALG_PAIRCMP", "0")), "ALG_PAIRCMP=1 is required for this census"

split = os.environ.get("PC_CENSUS_SPLIT", "train")
vs, vst, vtk, vg, vse = load_alg(split)
lo, hi = (int(x) for x in os.environ.get("GP_ROWS", "0:2").split(":"))
sl = np.arange(lo, hi)

p = build_params(0)
for k in ("pc_w1", "pc_b1", "pc_w2", "pc_b2"):
    assert k in p, f"ALG_PAIRCMP did not register {k}"

ckpt = os.environ.get("PC_CKPT", "")
if ckpt:
    from tinygrad.nn.state import safe_load
    sd = safe_load(ckpt)
    _miss = [k for k in p if k not in sd]
    for k in p:
        if k in sd:
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    print(f"[paircmp-knob] warm from {ckpt}: {len(_miss)} params stay at fresh init: {_miss[:6]}", flush=True)
else:
    print("[paircmp-knob] no PC_CKPT given -- fresh init throughout", flush=True)

ts = Tensor(vst[sl].astype(np.float32), dtype=dtypes.float)
tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
_idt = Tensor(ident_build_array([vs[int(i)] for i in sl], T_ALG), dtype=dtypes.int) \
    if (H.ALG_BUSREG or H.ALG_IDKEY) else None

H._PAIRCMP_CENSUS = []
o = forward(p, ts, tk, se, ident=_idt)
o["res"].realize()   # force the final head's graph (and therefore the census append) to run

census = H._PAIRCMP_CENSUS
print(f"[paircmp-knob] split={split} rows={lo}:{hi} ALG_PAIRCMP_SCALE={H.ALG_PAIRCMP_SCALE} "
      f"calls recorded={len(census)}")
assert census, "no _heads_of call recorded a census entry -- the hook never fired"
for i, (raw, post, bil) in enumerate(census):
    ratio_raw = raw / bil if bil else float("nan")
    ratio_post = post / bil if bil else float("nan")
    print(f"[paircmp-knob] call {i}: pc_std_pre_scale={raw:.6f} pc_std_post_scale={post:.6f} "
          f"res_bilinear_std={bil:.6f} ratio_pre={ratio_raw:.4f} ratio_post={ratio_post:.4f}")

last_raw, last_post, last_bil = census[-1]
last_ratio = last_post / last_bil if last_bil else float("nan")
bar_met = last_ratio >= 0.10
print(f"[paircmp-knob] LAST BREATH: ratio_post={last_ratio:.4f} vs bar >= 0.10 -> "
      f"{'PASS' if bar_met else 'FAIL (VOID BY THE KNOB LAW)'}")
