"""loop_val.py — THE LOOP-ENGAGED VAL (2026-08-31). _quick_val never runs
the breath loop (no slot_mask — the loop-free-val finding); this reader
computes the SAME fac-exact criterion on the masked two-pass forward, so
organs are measured OPERATING, not just via weight-shaping. Env: LV_CKPT;
mode via ALG_* envs of the caller (SC_EVAL/ALG_SHELF_CIRCLE for seal).
"""
import os, sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from phase1_algebra_head import (build_params, forward, load_alg,
                                 build_slot_masks, L_FAC)
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

vs, vst, vtk, vg, vse = load_alg("test")
p = build_params(0)
sd = safe_load(os.environ["LV_CKPT"])
assert set(sd.keys()) == set(p.keys()), \
    (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
n_ok = n_tot = 0
for s0 in range(0, len(vs), 8):
    sl = np.arange(s0, min(s0 + 8, len(vs)))
    pad = 8 - len(sl)
    sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    ts = Tensor(vst[sl_p].astype(np.float32), dtype=dtypes.float)
    tk = Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
    o0 = forward(p, ts, tk, se)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32))
    fact_t = None
    if int(os.environ.get("ALG_ALT2", "0")):
        # ALTERNATOR V2 fact-fed read (2026-09-01): live facts from this
        # checkpoint's own pass-1 parse — same convention as _quick_val
        from phase1_algebra_head import alt2_fact_buf, K_VARS
        _ka = ("pres", "ftype", "op", "dig") + \
            (("dup",) if "dup" in o0 else ())
        _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
        _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p])
        _ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        fb = alt2_fact_buf(_oa, vse[sl_p].astype(np.int32), _nv, _ma)
        fact_t = Tensor(fb, dtype=dtypes.float)
    o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float),
                fact_buf=fact_t)
    onp = {k: o[k].realize().numpy() for k in
           (("pres", "ftype", "op", "islit", "dig", "args", "res")
            + (("dup",) if "h_dup" in p else ()))}
    for bi, i in enumerate(sl):
        i = int(i)
        for j in range(L_FAC):
            if vg["presence"][i, j] < 0.5:
                continue
            n_tot += 1
            ok = (onp["pres"][bi, j] > 0)
            ok &= int(onp["ftype"][bi, j].argmax()) == vg["ftype"][i, j]
            ok &= int(onp["res"][bi, j].argmax()) == vg["res"][i, j]
            if vg["ftype"][i, j] == 0:
                ok &= int(onp["op"][bi, j].argmax()) == vg["op"][i, j]
                gset = set(np.where(vg["args"][i, j] > .5)[0].tolist())
                if len(gset) == 1 and "dup" in onp:
                    ok &= bool(onp["dup"][bi, j] > 0)
                    ok &= int(np.argmax(onp["args"][bi, j])) in gset
                else:
                    top2 = set(np.argsort(-onp["args"][bi, j])[:2].tolist())
                    ok &= top2 == gset
            else:
                ok &= bool((onp["dig"][bi, j].argmax(-1) ==
                            vg["digits"][i, j]).all())
            n_ok += ok
print(f"[loop-val] {os.environ['LV_CKPT']} mode="
      f"SC={os.environ.get('ALG_SHELF_CIRCLE','0')}/EVAL={os.environ.get('SC_EVAL','-')} "
      f"fac-exact={n_ok / max(n_tot, 1):.4f} (n={n_tot})")
