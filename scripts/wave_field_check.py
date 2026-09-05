"""wave_field_check.py — THE CABLE CHECK (2026-08-31, the patience word).
Before any wave refire: is the determination field ALIVE on real data?
Reads the snapped wiring at each breath on val rows (masked two-pass,
organs operating), computes the field, and reports: determination
coverage per breath, fire rate, feature variance, and — the decisive
cable — FIELD TRUTH: does the snapped field agree with the field computed
from GOLD wiring? A dead or lying field = the unplugged cable.
Env: WF_CKPT (stack artifact with garage; DETWAVE not required — we
recompute the field from the snaps directly).
"""
import os, sys
os.environ.setdefault("DEV", "AMD")
os.environ.setdefault("ALG2", "1")
os.environ.setdefault("ALG_FTYPES", "9")
os.environ.setdefault("ALG_DUP", "1")
os.environ.setdefault("ALG_HW", "512")
os.environ.setdefault("ALG_WIDE", "1")
os.environ.setdefault("ALG_BREATH", "7")
os.environ.setdefault("ALG_NOTEBOOK", "1")
os.environ.setdefault("ALG_SIXWAVE", "1")
os.environ.setdefault("NB_PERSLOT", "1")
os.environ.setdefault("ALG_BINDBUS", "7")
os.environ.setdefault("ALG_BIND_D", "512")
os.environ.setdefault("BIND_CODES", ".cache/bindbus_codes512.npz")
os.environ.setdefault("ALG_BUSGARAGE", "2")
os.environ.setdefault("ALG_MINE_BREATHS", "1")
os.environ.setdefault("ALG_TEST", ".cache/algebra_nl_test.jsonl")
os.environ.setdefault("ALG_TEST_NAME", "test23")
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from phase1_algebra_head import (build_params, forward, load_alg,
                                 build_slot_masks, L_FAC)
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

bz = np.load(".cache/bindbus_codes512.npz")
CB = bz["CB"]; P = CB.shape[1] // 2
CBc = (CB.reshape(32, P, 2)[..., 0] + 1j * CB.reshape(32, P, 2)[..., 1])
ROLE = {r: np.exp(-1j * bz[f"theta_{r}"]) for r in ("arg1", "arg2", "res", "op")}

def snap_ids(w):
    """per-role cleanup ids for a wire (L, D) -> dict of (L,) ids."""
    out = {}
    z0 = w.reshape(w.shape[0], P, 2)
    zc = z0[..., 0] + 1j * z0[..., 1]
    for r in ROLE:
        z = zc * ROLE[r]
        out[r] = np.argmax((z @ np.conj(CBc).T).real, -1)
    return out

def field(a1, a2, rs, gv, pres):
    """determination closure from id arrays (L,) + given flags + presence."""
    det = np.zeros(24, bool)
    for j in np.where(pres & gv)[0]:
        if rs[j] < 24: det[rs[j]] = True
    fired = np.zeros(len(a1), bool)
    for _ in range(3):
        for j in np.where(pres & ~gv)[0]:
            args = [a for a in (a1[j], a2[j]) if a < 24]
            if args and all(det[a] for a in args):
                fired[j] = True
                if rs[j] < 24: det[rs[j]] = True
    return det, fired

vs, vst, vtk, vg, vse = load_alg("test")
p = build_params(0)
sd = safe_load(os.environ.get("WF_CKPT", ".cache/sharp_bind15d.safetensors"))
missing = set(p.keys()) - set(sd.keys())
for k in p:
    if k in sd:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
if missing: print(f"[wf] fresh-init (not in ckpt): {sorted(missing)}")
N = 128
stats = {"cov_snap": [], "cov_gold": [], "fire_snap": [], "fire_gold": [],
         "fire_agree": [], "det_agree": []}
for s0 in range(0, N, 8):
    sl = np.arange(s0, min(s0 + 8, N))
    ts = Tensor(vst[sl].astype(np.float32), dtype=dtypes.float)
    tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
    o0 = forward(p, ts, tk, se)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(onp0, vse[sl].astype(np.int32))
    o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float))
    W = o["bind"].realize().numpy()          # breath-0 wire (the emission)
    # NOTE: the in-loop snaps come from per-breath cur; the emission wire is
    # the closest banked read of snap quality — the field check runs on it.
    for bi, i in enumerate(sl):
        i = int(i)
        pres = vg["presence"][i] > 0.5
        if pres.sum() == 0: continue
        ids = snap_ids(W[bi])
        gvs = ids["op"] == 25                        # snapped 'given'
        d_s, f_s = field(ids["arg1"], ids["arg2"], ids["res"], gvs, pres)
        # gold field
        ga1 = np.zeros(L_FAC, int); ga2 = np.zeros(L_FAC, int)
        for j in range(L_FAC):
            aidx = np.where(vg["args"][i, j] > .5)[0]
            ga1[j] = aidx[0] if len(aidx) else vg["res"][i, j]
            ga2[j] = aidx[1] if len(aidx) > 1 else ga1[j]
        ggv = vg["ftype"][i] == 1
        d_g, f_g = field(ga1, ga2, vg["res"][i].astype(int), ggv, pres)
        stats["cov_snap"].append(d_s.mean()); stats["cov_gold"].append(d_g.mean())
        stats["fire_snap"].append(f_s[pres].mean())
        stats["fire_gold"].append(f_g[pres].mean())
        stats["fire_agree"].append((f_s[pres] == f_g[pres]).mean())
        stats["det_agree"].append((d_s == d_g).mean())
print(f"[wave-field] ckpt={os.environ.get('WF_CKPT')} rows={len(stats['cov_snap'])}")
for k, v in stats.items():
    print(f"  {k:11s} mean {np.mean(v):.3f}")
