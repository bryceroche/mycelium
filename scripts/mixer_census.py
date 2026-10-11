"""mixer_census.py -- THE TYPED MIXER / THE SHOAL MIXER census (2026-10-10): the arm's own
read, required by the build spec ("THE CENSUS: the typed mask's mean density per breath and
the share of rows with at least one edge; the shoal's effective k; a read smoke: keys match,
decodes finite").

Wraps H._mixer_edge_mask (module-level monkeypatch, restored after the one forward() call) to
record every breath's returned edge tensor -- no new _CENSUS tap needed, since the edge mask
is already a single well-defined function call per breath.

Env: MX_ARM (typed|shoal, informational only), MX_CKPT (safetensors to warm-load; empty =
fresh init), MX_SPLIT (train|test, default test), MX_N (row count, default 32), plus the
arm's own family env (ALG_TYPED_MIXER=1 or ALG_SHOAL_MIXER=<k>) exported by the caller.
"""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, ident_build_array, T_ALG
from tinygrad import Tensor, dtypes

ARM = os.environ.get("MX_ARM", "?")
vs, vst, vtk, vg, vse = load_alg(os.environ.get("MX_SPLIT", "test"))
N = int(os.environ.get("MX_N", "32"))
sl = np.arange(min(N, len(vs))); B = len(sl)

p = build_params(0)
ckpt = os.environ.get("MX_CKPT", "")
if ckpt:
    from tinygrad.nn.state import safe_load
    sd = safe_load(ckpt)
    miss = [k for k in p if k not in sd]
    for k in p:
        if k in sd:
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    print(f"[mixer-census] ARM={ARM} warm from {ckpt} ({len(miss)} params at fresh init: {miss[:4]})")
else:
    print(f"[mixer-census] ARM={ARM} fresh init throughout")

ts = Tensor(vst[sl].astype(np.float32), dtype=dtypes.float)
tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
_idt = Tensor(ident_build_array([vs[int(i)] for i in sl], T_ALG), dtype=dtypes.int) \
    if (H.ALG_BUSREG or H.ALG_IDKEY) else None

o0 = H.forward(p, ts, tk, se, ident=_idt)
onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
mk = Tensor(build_slot_masks(onp0, vse[sl].astype(np.int32)), dtype=dtypes.float)

RECORD = []
_orig = H._mixer_edge_mask
def _wrapped(p_, cur, ctx, B_, kb):
    e = _orig(p_, cur, ctx, B_, kb)
    if e is not None:
        RECORD.append((kb, e.detach().numpy()))
    return e
H._mixer_edge_mask = _wrapped
try:
    o = forward(p, ts, tk, se, slot_mask=mk, ident=_idt)
finally:
    H._mixer_edge_mask = _orig

keys = sorted(k for k in o if k != "breaths")
fin = all(np.isfinite(o[k].numpy()).all() for k in ("args", "res", "fat") if hasattr(o[k], "numpy"))
print(f"[mixer-census] ARM={ARM} rows={B} output keys={keys}")
print(f"[mixer-census] ARM={ARM} decode finite (args/res/fat)={fin}")
print(f"[mixer-census] ARM={ARM} edge mask captured at breaths={sorted(set(kb for kb, _ in RECORD))} "
      f"(empty means ALG_TYPED_MIXER/ALG_SHOAL_MIXER were not exported, or both returned None)")

for kb, e in sorted(RECORD):
    L = e.shape[-1]
    has_edge = e > 0.5
    eye = np.eye(L, dtype=bool)[None]
    off = has_edge & ~eye
    density = has_edge.mean()
    frac_rows_with_edge = off.any(-1).mean()
    eff_k = has_edge.sum(-1).mean()
    diag_ok = has_edge[:, np.arange(L), np.arange(L)].all()
    print(f"[mixer-census] breath {kb}: mean density={density:.4f}  "
          f"rows-with->=1-named-edge(excl self)={frac_rows_with_edge:.4f}  "
          f"mean edges/row(incl self)={eff_k:.3f}  self-edge-always-on={diag_ok}")

print(f"[mixer-census] ARM={ARM} SMOKE PASS: {fin and len(RECORD) > 0}")
