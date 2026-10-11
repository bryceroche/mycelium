"""funnel_knob_census.py -- THE FUNNEL GATE's knob census (2026-10-10; the
waist_skip_knob_census.py convention -- the pre/post knob law, extended per-breath). Prints,
for a forward pass over GP_ROWS rows, per breath kb in 1..K_B-1: the per-band mean g_skip
(root/branch/leaf, sigmoid of funnel_skip_theta[kb]) and the rank gate's share of intermediate
(POLAR_D) dims with r < 0.5 (closed) plus its mean/min -- "where does the loss want the funnel
narrow: which breath, which band, how much of the 128-d code?" This IS the arm's own registered
read (ledger 2026-10-10 11:15): (i) skip opens at breaths 0-2 and rank closes at 5-6 = the
funnel is the loss's preference; (ii) shut everywhere = refuted; (iii) rank closes late only =
a late crystallization.

THE TWO-PASS CONVENTION (selfmatch_grad_probe.py's, NOT a one-pass shortcut): _polar_waist lives
inside breath_step, called from forward()'s own iterative breath LOOP -- and that loop never
runs on an UNMASKED call (ledger 2026-09-24 21:05). This script therefore runs pass 1
(unmasked) ONLY to build the slot mask, then the REAL census pass WITH slot_mask=mk, which is
what actually walks all K_B-1 breaths and calls _polar_funnel_rank/_polar_funnel_skip each time.

Armed ONLY by setting the module global H._FUNNEL_CENSUS = {} before the masked forward() call
-- nothing in do_train's real training path ever sets it, so the exact same code inside
_polar_funnel_rank/_polar_funnel_skip is inert there (the waist-skip census's own safety
argument, restated).

usage: ALG_POLAR=1 ALG_POLAR_D=128 ALG_FUNNEL=1 [FG_CKPT=<ckpt>] [GP_ROWS=0:8] \
       .venv/bin/python3 scripts/funnel_knob_census.py
"""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, ident_build_array, T_ALG
from tinygrad import Tensor, dtypes

assert int(os.environ.get("ALG_FUNNEL", "0")), "ALG_FUNNEL=1 is required for this census"

split = os.environ.get("FG_CENSUS_SPLIT", "train")
vs, vst, vtk, vg, vse = load_alg(split)
lo, hi = (int(x) for x in os.environ.get("GP_ROWS", "0:8").split(":"))
sl = np.arange(lo, hi)

p = build_params(0)
assert "funnel_skip_theta" in p and "funnel_rank_theta" in p, \
    "ALG_FUNNEL did not register funnel_skip_theta/funnel_rank_theta"
K_B = int(os.environ.get("ALG_BREATH", "1"))

ckpt = os.environ.get("FG_CKPT", "")
if ckpt:
    from tinygrad.nn.state import safe_load
    sd = safe_load(ckpt)
    _miss = [k for k in p if k not in sd]
    for k in p:
        if k in sd:
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    print(f"[funnel-knob] warm from {ckpt}: {len(_miss)} params stay at fresh init: {_miss[:6]}", flush=True)
else:
    print("[funnel-knob] no FG_CKPT given -- fresh init throughout", flush=True)

ts = Tensor(vst[sl].astype(np.float32), dtype=dtypes.float)
tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
_idt = Tensor(ident_build_array([vs[int(i)] for i in sl], T_ALG), dtype=dtypes.int) \
    if (H.ALG_BUSREG or H.ALG_IDKEY) else None

# PASS 1 (unmasked): builds the slot mask only, exactly as do_train/the grad probes do --
# this call's breath loop never runs (no slot_mask), so it is NOT censused.
o0 = forward(p, ts, tk, se, ident=_idt)
onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
mk = Tensor(build_slot_masks(onp0, vse[sl].astype(np.int32)), dtype=dtypes.float)

# PASS 2 (masked): the REAL census pass -- this is the call that walks all K_B-1 breaths.
H._FUNNEL_CENSUS = {}
o = forward(p, ts, tk, se, slot_mask=mk, ident=_idt)
assert "breaths" in o, "the masked pass did not populate out['breaths'] -- the loop did not run"
o["breaths"][-1]["res"].realize()   # force the final breath's graph (and every breath's census writes) to run

census = H._FUNNEL_CENSUS
print(f"[funnel-knob] split={split} rows={lo}:{hi} theta_init={os.environ.get('ALG_FUNNEL_INIT', '-4.0')} "
      f"breaths recorded={sorted(census.keys())}")
assert census, "no _polar_funnel_rank/_polar_funnel_skip call recorded a census entry -- the hook never fired (ALG_POLAR_D unset?)"
for kb in sorted(census.keys()):
    c = census[kb]
    print(f"[funnel-knob] breath {kb}: g_skip_root={c.get('g_skip_root', float('nan')):.6f} "
          f"g_skip_branch={c.get('g_skip_branch', float('nan')):.6f} "
          f"g_skip_leaf={c.get('g_skip_leaf', float('nan')):.6f} | "
          f"rank_share_closed(r<0.5)={c.get('rank_share_closed', float('nan')):.4f} "
          f"rank_r_mean={c.get('rank_r_mean', float('nan')):.6f} "
          f"rank_r_min={c.get('rank_r_min', float('nan')):.6f}")

_bk = sorted(k for k in census.keys() if k >= 1)
if _bk:
    _early = [k for k in _bk if k <= 2]
    _late = [k for k in _bk if k >= max(_bk) - 1]
    _skip_early = float(np.mean([max(census[k].get("g_skip_root", 0.0),
                                      census[k].get("g_skip_branch", 0.0),
                                      census[k].get("g_skip_leaf", 0.0)) for k in _early])) if _early else float("nan")
    _rank_late = float(np.mean([census[k].get("rank_share_closed", 0.0) for k in _late])) if _late else float("nan")
    print(f"[funnel-knob] BARS READ: skip opens at breaths 0-2 (max band g over {_early})={_skip_early:.4f} "
          f"(birth value ~{1.0/(1.0+np.exp(4.0)):.4f}); rank closes late (breaths {_late}) mean "
          f"share r<0.5={_rank_late:.4f} (birth value ~{1.0/(1.0+np.exp(4.0)):.4f})")
