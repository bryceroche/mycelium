"""loop_sensitivity.py — THE LOOP'S OWN RULER (2026-09-11): perturb the
fused forward's trunk states by a relative 1e-6 (fp32's own grain, ~8
ulps) and measure how far the emissions move, per key, as the relative
max-abs the rung-1 harness prints. A segmented walk can never agree with
the fused graph more tightly than the graph agrees with itself under an
ulp; the bar's ruler is this number. Env: LS_CKPT + the family envs."""
import os, sys, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
ckpt = os.environ["LS_CKPT"]; eps = float(os.environ.get("LS_EPS", "1e-6")); n = int(os.environ.get("LS_N", "8"))
vs, vst, vtk, vg, vse = load_alg("test")
p = build_params(0); sd = safe_load(ckpt)
assert set(sd) == set(p); [p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize() for k in p]
sl = np.arange(n); base = vst[sl].astype(np.float32)
rng = np.random.RandomState(0); pert = base * (1.0 + eps * rng.uniform(-1, 1, size=base.shape).astype(np.float32))
tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float); se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
def run(states):
    ts = Tensor(states, dtype=dtypes.float)
    o0 = forward(p, ts, tk, se); onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(onp0, vse[sl].astype(np.int32))
    o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float))
    return {k: o[k].realize().numpy() for k in ("pres", "ftype", "op", "args", "res", "dig", "y", "fat")}
a = run(base); b = run(pert)
rows = []
for k in a:
    d = float(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max()); scale = max(1.0, float(np.abs(a[k]).max())); rows.append((k, d / scale))
print(f"[sensitivity] {os.path.basename(ckpt)} eps={eps:g}: " + " ".join(f"{k}={r:.2e}" for k, r in rows) + f" | loop keys max rel = {max(r for k, r in rows if k != 'fat'):.2e} (fat = breath 0: {dict(rows)['fat']:.2e})")
