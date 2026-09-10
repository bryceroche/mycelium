"""refusal_read.py — THE STEERING WHEEL, read W1 part 1 (GPU): dump every
row's FINAL decoded parse (factor dicts tagged with their slot), the gold
factors, and per-slot correctness, on a fixture, for the refusal grader.
loop_val.read's two-pass masked forward VERBATIM; decode() is the head's
own, called per slot (all other slots' presence masked) so each factor
carries its slot. Env: RR_CKPT, RR_OUT (jsonl) + the family envs."""
import os, sys, json
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from phase1_algebra_head import (build_params, forward, load_alg, build_slot_masks, L_FAC, decode)
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from mycelium.jit_read import read_forward as _rf
from loop_val import atlas_tables, _XPV
ckpt = os.environ["RR_CKPT"]; out_path = os.environ["RR_OUT"]
vs, vst, vtk, vg, vse = load_alg("test")
p = build_params(0); sd = safe_load(ckpt)
assert set(sd.keys()) == set(p.keys()), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
_ATAB, _AIDX, _atl, _acls = atlas_tables(vs)
_jk_open = (("fat", "args", "res") + (("pres", "ftype", "op", "dig", "dup") if int(os.environ.get("ALG_ALT2", "0")) and not int(os.environ.get("LV_NOFACT", "0")) else ()) + (("nl0",) if _XPV else ()))
_jk_masked = ("pres", "ftype", "op", "islit", "dig", "args", "res", "query") + (("sel",) if "h_sel" in p else ()) + (("dup",) if "h_dup" in p else ()) + (("sgn",) if "h_sgn" in p else ()) + (("dargs",) if "W_dargs" in p else ()) + (("dig2",) if "h_dig2" in p else ()) + (("y",) if "W_y" in p else ())
N = len(vs); n_ok = n_tot = 0; fo = open(out_path, "w")
for s0 in range(0, N, 8):
    sl = np.arange(s0, min(s0 + 8, N)); pad = 8 - len(sl)
    sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    ts = Tensor(vst[sl_p].astype(np.float32), dtype=dtypes.float); tk = Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float); se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
    o0 = _rf(forward, p, ts, tk, se, keys=_jk_open)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32))
    fact_t = mass_t = None
    if int(os.environ.get("ALG_ALT2", "0")) and not int(os.environ.get("LV_NOFACT", "0")):
        from phase1_algebra_head import alt2_fact_buf, K_VARS
        _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
        _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
        _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p]); _ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        _mo = (np.zeros((len(sl_p), K_VARS), np.float32) if int(os.environ.get("ALG_MH_MASS", "0")) else None)
        fb = alt2_fact_buf(_oa, vse[sl_p].astype(np.int32), _nv, _ma, mass_out=_mo); fact_t = Tensor(fb, dtype=dtypes.float)
        if _mo is not None: mass_t = Tensor(np.clip(_mo / 301.0, 0.0, 1.0)[:, :, None].astype(np.float32), dtype=dtypes.float)
    _lvai = _AIDX[sl_p].copy() if _AIDX is not None else None
    _mha_t = (Tensor(_ATAB[_lvai], dtype=dtypes.float) if _ATAB is not None else None)
    o = _rf(forward, p, ts, tk, se, keys=_jk_masked, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=fact_t, mh_mass=mass_t, mh_atlas_traj=_mha_t)
    onp = {k: o[k].realize().numpy() for k in _jk_masked if k in o}
    for bi, i in enumerate(sl):
        i = int(i); row = {k: onp[k][bi] for k in onp}
        parse = []
        for j in range(L_FAC):
            if row["pres"][j] <= 0: continue
            rj = dict(row); pr = np.full_like(row["pres"], -1.0); pr[j] = row["pres"][j]; rj["pres"] = pr
            for f in decode(rj): f["_slot"] = j; parse.append(f)
        wrong = []; extra = [int(j) for j in range(L_FAC) if row["pres"][j] > 0 and vg["presence"][i, j] < 0.5]
        for j in range(L_FAC):
            if vg["presence"][i, j] < 0.5: continue
            n_tot += 1
            ok = (row["pres"][j] > 0); ok &= int(row["ftype"][j].argmax()) == vg["ftype"][i, j]; ok &= int(row["res"][j].argmax()) == vg["res"][i, j]
            if vg["ftype"][i, j] == 0:
                ok &= int(row["op"][j].argmax()) == vg["op"][i, j]
                gset = set(np.where(vg["args"][i, j] > .5)[0].tolist())
                if len(gset) == 1 and "dup" in row: ok &= bool(row["dup"][j] > 0); ok &= int(np.argmax(row["args"][j])) in gset
                else: ok &= set(np.argsort(-row["args"][j])[:2].tolist()) == gset
            else: ok &= bool((row["dig"][j].argmax(-1) == vg["digits"][i, j]).all())
            n_ok += int(ok)
            if not ok: wrong.append(int(j))
        fo.write(json.dumps({"i": i, "n_vars": int(vs[i].get("n_vars", 24)), "m": int(vs[i].get("m", 300)), "parse": parse, "gold": vs[i]["factors"], "wrong_slots": wrong, "extra_slots": extra}) + "\n")
fo.close()
print(f"[refusal-read] {ckpt} fac-exact={n_ok / max(n_tot, 1):.4f} (n={n_tot}) rows={N} -> {out_path}")
