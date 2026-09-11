"""wheel_read.py — THE STEERING WHEEL AT READ TIME: loop_val's two-pass read
(verbatim), EAGER (no JIT: the solver runs between breaths), with _WHEEL
armed for the masked pass. Prints fac-exact and the wheel's own stats:
refusals per breath, core sizes, rows turned. Env: WR_CKPT, WR_BETA,
WR_MODE (own|union) + the family envs."""
import os, sys, collections
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, L_FAC, K_VARS
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from loop_val import atlas_tables, _XPV
assert not int(os.environ.get("ALG_JIT_READ", "0")), "the wheel read is eager"
ckpt = os.environ["WR_CKPT"]; beta = float(os.environ.get("WR_BETA", "3")); mode = os.environ.get("WR_MODE", "union")
vs, vst_, vtk, vg, vse = load_alg("test")
p = build_params(0); sd = safe_load(ckpt)
assert set(sd) == set(p), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
_ATAB, _AIDX, _atl, _acls = atlas_tables(vs)
n_ok = n_tot = 0; N = len(vs); ALLST = []; ALLTU = []
for s0 in range(0, N, 8):
    sl = np.arange(s0, min(s0 + 8, N)); pad = 8 - len(sl)
    sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    ts = Tensor(vst_[sl_p].astype(np.float32), dtype=dtypes.float); tk = Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float); se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
    H._WHEEL = None
    o0 = forward(p, ts, tk, se); onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32))
    fact_t = None
    if int(os.environ.get("ALG_ALT2", "0")) and not int(os.environ.get("LV_NOFACT", "0")):
        from phase1_algebra_head import alt2_fact_buf
        _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
        _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
        _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p]); _ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        fb = alt2_fact_buf(_oa, vse[sl_p].astype(np.int32), _nv, _ma, mass_out=None); fact_t = Tensor(fb, dtype=dtypes.float)
    _lvai = _AIDX[sl_p].copy() if _AIDX is not None else None
    _mha_t = (Tensor(_ATAB[_lvai], dtype=dtypes.float) if _ATAB is not None else None)
    H._WHEEL = {"n_vars": [vs[int(i)].get("n_vars", K_VARS) for i in sl_p], "m": [vs[int(i)].get("m", 300) for i in sl_p], "beta": beta, "mode": mode, "stats": [], "turned": []}
    o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=fact_t, mh_mass=None, mh_atlas_traj=_mha_t)
    onp = {k: o[k].realize().numpy() for k in (("pres", "ftype", "op", "islit", "dig", "args", "res") + (("dup",) if "h_dup" in p else ()))}
    ALLST.extend(H._WHEEL["stats"]); ALLTU.extend(H._WHEEL["turned"]); H._WHEEL = None
    for bi, i in enumerate(sl):
        i = int(i)
        for j in range(L_FAC):
            if vg["presence"][i, j] < 0.5: continue
            n_tot += 1
            ok = (onp["pres"][bi, j] > 0); ok &= int(onp["ftype"][bi, j].argmax()) == vg["ftype"][i, j]; ok &= int(onp["res"][bi, j].argmax()) == vg["res"][i, j]
            if vg["ftype"][i, j] == 0:
                ok &= int(onp["op"][bi, j].argmax()) == vg["op"][i, j]
                gset = set(np.where(vg["args"][i, j] > .5)[0].tolist())
                if len(gset) == 1 and "dup" in onp: ok &= bool(onp["dup"][bi, j] > 0); ok &= int(np.argmax(onp["args"][bi, j])) in gset
                else: ok &= set(np.argsort(-onp["args"][bi, j])[:2].tolist()) == gset
            else: ok &= bool((onp["dig"][bi, j].argmax(-1) == vg["digits"][i, j]).all())
            n_ok += int(ok)
ref = collections.Counter((kb, s) for kb, s, _ in ALLST); cores = [c for _, s, c in ALLST if s == "unsat"]
kbs = sorted({k for k, _, _ in ALLST})
per_b = " ".join(f"b{kb}={ref[(kb, 'unsat')]}/{sum(v for (k2, _), v in ref.items() if k2 == kb)}" for kb in kbs)
turned = " ".join(f"b{kb}={sum(t for k2, t, _ in ALLTU if k2 == kb)}" for kb in kbs)
print(f"[wheel-read] {os.path.basename(ckpt)} beta={beta} mode={mode} fac-exact={n_ok / max(n_tot, 1):.4f} (n={n_tot})")
print(f"[wheel-read] refusals per breath (refused/rows): {per_b} | core size mean={np.mean(cores) if cores else 0:.2f} | rows turned per breath: {turned}")
