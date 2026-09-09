"""collider_front_read.py — THE COLLIDER, RUNG 1 (2026-09-09, the word):
per-slot dumps of one reading on a fixture, for the disagreement-front
read. loop_val.read's two-pass masked forward VERBATIM, but instead of
counting it writes, per (row, slot): the gold presence, the fac-exact
correctness flag, and the reading's SIGNATURE [pres, ftype, res, op,
arg0, arg1, dup, dig...] so two readings can be compared slot by slot.
Env: CF_CKPT (the checkpoint), CF_OUT (npz), plus the caller's family
envs (ALG_SEVER=s3 makes reading B). Grader: collider_front_grade.py."""
import os, sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from phase1_algebra_head import (build_params, forward, load_alg,
                                 build_slot_masks, L_FAC)
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from mycelium.jit_read import read_forward as _rf
from loop_val import atlas_tables, _XPV

ckpt = os.environ["CF_CKPT"]; out_path = os.environ["CF_OUT"]
vs, vst, vtk, vg, vse = load_alg("test")
p = build_params(0)
sd = safe_load(ckpt)
assert set(sd.keys()) == set(p.keys()), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
_ATAB, _AIDX, _atl, _acls = atlas_tables(vs)
_jk_open = (("fat", "args", "res")
            + (("pres", "ftype", "op", "dig", "dup")
               if int(os.environ.get("ALG_ALT2", "0")) and not int(os.environ.get("LV_NOFACT", "0")) else ())
            + (("nl0",) if _XPV else ()))
_jk_masked = (("pres", "ftype", "op", "islit", "dig", "args", "res") + (("dup",) if "h_dup" in p else ()))
N = len(vs)
SIGW = 12
present = np.zeros((N, L_FAC), np.int8); correct = np.zeros((N, L_FAC), np.int8)
sig = np.full((N, L_FAC, SIGW), -1, np.int32)
n_ok = n_tot = 0
for s0 in range(0, N, 8):
    sl = np.arange(s0, min(s0 + 8, N)); pad = 8 - len(sl)
    sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    ts = Tensor(vst[sl_p].astype(np.float32), dtype=dtypes.float)
    tk = Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
    o0 = _rf(forward, p, ts, tk, se, keys=_jk_open)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32))
    fact_t = mass_t = None
    if int(os.environ.get("ALG_ALT2", "0")) and not int(os.environ.get("LV_NOFACT", "0")):
        from phase1_algebra_head import alt2_fact_buf, K_VARS
        _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
        _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
        _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p])
        _ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        _mo = (np.zeros((len(sl_p), K_VARS), np.float32) if int(os.environ.get("ALG_MH_MASS", "0")) else None)
        fb = alt2_fact_buf(_oa, vse[sl_p].astype(np.int32), _nv, _ma, mass_out=_mo)
        fact_t = Tensor(fb, dtype=dtypes.float)
        if _mo is not None:
            mass_t = Tensor(np.clip(_mo / 301.0, 0.0, 1.0)[:, :, None].astype(np.float32), dtype=dtypes.float)
    _lvai = _AIDX[sl_p].copy() if _AIDX is not None else None
    if _XPV and _lvai is not None:
        from mycelium.step_atlas import cross_prior
        _xlv, _ = cross_prior(_atl, o0["nl0"].realize().numpy(), return_traj=False)
        _rlv = ((_lvai == len(_acls)) if _XPV == 1 else np.ones(len(_lvai), bool))
        _lvai = np.where(_rlv, _xlv, _lvai)
    _mha_t = (Tensor(_ATAB[_lvai], dtype=dtypes.float) if _ATAB is not None else None)
    o = _rf(forward, p, ts, tk, se, keys=_jk_masked, slot_mask=Tensor(mk, dtype=dtypes.float),
            fact_buf=fact_t, mh_mass=mass_t, mh_atlas_traj=_mha_t)
    onp = {k: o[k].realize().numpy() for k in (("pres", "ftype", "op", "islit", "dig", "args", "res") + (("dup",) if "h_dup" in p else ()))}
    for bi, i in enumerate(sl):
        i = int(i)
        for j in range(L_FAC):
            ft = int(onp["ftype"][bi, j].argmax()); rs = int(onp["res"][bi, j].argmax())
            op_ = int(onp["op"][bi, j].argmax()); pr = int(onp["pres"][bi, j] > 0)
            a2 = sorted(np.argsort(-onp["args"][bi, j])[:2].tolist())
            dp = int(onp["dup"][bi, j] > 0) if "dup" in onp else 0
            dg = onp["dig"][bi, j].argmax(-1).reshape(-1).tolist()[:SIGW - 7]
            row = [pr, ft, rs, op_, a2[0], a2[1], dp] + dg
            sig[i, j, :len(row)] = row
            if vg["presence"][i, j] < 0.5:
                continue
            present[i, j] = 1; n_tot += 1
            ok = (onp["pres"][bi, j] > 0)
            ok &= ft == vg["ftype"][i, j]
            ok &= rs == vg["res"][i, j]
            if vg["ftype"][i, j] == 0:
                ok &= op_ == vg["op"][i, j]
                gset = set(np.where(vg["args"][i, j] > .5)[0].tolist())
                if len(gset) == 1 and "dup" in onp:
                    ok &= bool(onp["dup"][bi, j] > 0)
                    ok &= int(np.argmax(onp["args"][bi, j])) in gset
                else:
                    top2 = set(np.argsort(-onp["args"][bi, j])[:2].tolist())
                    ok &= top2 == gset
            else:
                ok &= bool((onp["dig"][bi, j].argmax(-1) == vg["digits"][i, j]).all())
            correct[i, j] = int(ok); n_ok += int(ok)
np.savez(out_path, present=present, correct=correct, sig=sig)
print(f"[collider-read] {ckpt} sever={os.environ.get('ALG_SEVER','') or '-'} fac-exact={n_ok / max(n_tot, 1):.4f} (n={n_tot}) -> {out_path}")
