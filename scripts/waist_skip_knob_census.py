"""waist_skip_knob_census.py -- THE WAIST SKIP, FORM A's knob census (2026-10-09; the pre/post
knob law). Prints, for a forward pass over GP_ROWS rows, per breath: the skipped component
g * (c - P c)'s std across the content dims, vs P c's own std across the content dims, BOTH
pre- and post-keepnorm (the single scalar ratio n0/n1 the real _polar_keepnorm call applies to
the whole content block, replayed exactly against each piece since they sum linearly into it)
-- "where did the loss open the skip: leaf-like dims or everywhere?" -- plus the mean g per
hierarchical band (root/branch/leaf, by PLANE ORDER, the same bands _hier_band_dims names
regardless of whether ALG_HIER_WAIST is set).

THE TWO-PASS CONVENTION (selfmatch_grad_probe.py's, NOT paircmp_knob_census.py's one-pass
shortcut): _polar_waist lives inside breath_step, called from forward()'s own iterative
breath LOOP -- and that loop never runs on an UNMASKED call (ledger 2026-09-24 21:05: "no
slot_mask, so forward()'s breath loop never runs"; pass 1 takes a separate non-loop code path
that still returns fat/args/res via a direct heads_of() call, which is why a paircmp-style
one-pass census can see _heads_of fire but would see NOTHING from inside the loop). This
script therefore runs pass 1 (unmasked) ONLY to build the slot mask, then the REAL census pass
WITH slot_mask=mk, which is what actually walks all K_B breaths and calls _polar_waist each
time.

Armed ONLY by setting the module global H._WAISTSKIP_CENSUS = [] before the masked forward()
call -- nothing in do_train's real training path ever sets it, so the exact same code inside
_polar_waist_skip is inert there (the same safety argument as every `if _CENSUS is not None:`
line elsewhere in the file).

usage: ALG_POLAR=1 ALG_POLAR_D=128 ALG_WAIST_SKIP=1 [WS_CKPT=<ckpt>] [GP_ROWS=0:8] \
       .venv/bin/python3 scripts/waist_skip_knob_census.py
"""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, ident_build_array, T_ALG
from tinygrad import Tensor, dtypes

assert int(os.environ.get("ALG_WAIST_SKIP", "0")), "ALG_WAIST_SKIP=1 is required for this census"

split = os.environ.get("WS_CENSUS_SPLIT", "train")
vs, vst, vtk, vg, vse = load_alg(split)
lo, hi = (int(x) for x in os.environ.get("GP_ROWS", "0:8").split(":"))
sl = np.arange(lo, hi)

p = build_params(0)
assert "waist_skip_theta" in p, "ALG_WAIST_SKIP did not register waist_skip_theta"

ckpt = os.environ.get("WS_CKPT", "")
if ckpt:
    from tinygrad.nn.state import safe_load
    sd = safe_load(ckpt)
    _miss = [k for k in p if k not in sd]
    for k in p:
        if k in sd:
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    print(f"[waistskip-knob] warm from {ckpt}: {len(_miss)} params stay at fresh init: {_miss[:6]}", flush=True)
else:
    print("[waistskip-knob] no WS_CKPT given -- fresh init throughout", flush=True)

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

# PASS 2 (masked): the REAL census pass -- this is the call that walks all K_B breaths.
H._WAISTSKIP_CENSUS = []
o = forward(p, ts, tk, se, slot_mask=mk, ident=_idt)
assert "breaths" in o, "the masked pass did not populate out['breaths'] -- the loop did not run"
o["breaths"][-1]["res"].realize()   # force the final breath's graph (and every breath's census append) to run

census = H._WAISTSKIP_CENSUS
print(f"[waistskip-knob] split={split} rows={lo}:{hi} theta_init={os.environ.get('ALG_WAIST_SKIP_INIT', '-4.0')} "
      f"breaths recorded={len(census)}")
assert census, "no _polar_waist call recorded a census entry -- the hook never fired (ALG_POLAR_D unset?)"
for kb, c in enumerate(census):
    ratio_pre = c["std_skip_pre"] / c["std_pc_pre"] if c["std_pc_pre"] else float("nan")
    ratio_post = c["std_skip_post"] / c["std_pc_post"] if c["std_pc_post"] else float("nan")
    print(f"[waistskip-knob] breath {kb}: std_skip_pre={c['std_skip_pre']:.6f} "
          f"std_pc_pre={c['std_pc_pre']:.6f} ratio_pre={ratio_pre:.4f} | "
          f"std_skip_post={c['std_skip_post']:.6f} std_pc_post={c['std_pc_post']:.6f} "
          f"ratio_post={ratio_post:.4f} | g_mean={c['g_mean']:.6f} "
          f"g_root={c['g_band_root']:.6f} g_branch={c['g_band_branch']:.6f} g_leaf={c['g_band_leaf']:.6f}")

last = census[-1]
print(f"[waistskip-knob] LAST BREATH ({len(census) - 1}): g_mean={last['g_mean']:.6f} "
      f"g_root={last['g_band_root']:.6f} g_branch={last['g_band_branch']:.6f} "
      f"g_leaf={last['g_band_leaf']:.6f} g_max={last['g_max']:.6f}")
print(f"[waistskip-knob] KILL CHECK (g < 0.1 on EVERY dim, the registered bar's own wording): "
      f"{'g_max BELOW 0.1 EVERYWHERE -- KILL CANDIDATE' if last['g_max'] < 0.1 else 'g_max >= 0.1 -- at least one dim live'}")
