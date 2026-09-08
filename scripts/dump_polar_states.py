"""dump_polar_states.py — dump per-slot polar direction states (breaths_u) and
old-coordinate states (breaths_all) for a fixture under the champion, ALG_POLAR=1,
open regime (SC_EVAL=0). Used ONCE to PCA-initialize the content-plane waist
(ALG_POLAR_D) from the champion's own manifold. Read-only. Env: DS_N rows,
DS_TEST mint|wild, DS_CKPT, DS_OUT."""
import os, sys
os.environ.setdefault("DEV", "PCI+AMD")
FX = os.environ.get("DS_TEST", "mint")
ENV = {"ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
       "ALG_BREATH": "7", "ALG_NOTEBOOK": "1", "ALG_SIXWAVE": "1", "NB_PERSLOT": "1",
       "ALG_BINDBUS": "7", "ALG_BIND_D": "512", "BIND_CODES": ".cache/bindbus_codes512.npz",
       "ALG_BUSGARAGE": "2", "ALG_SHELF_CIRCLE": "2", "ALG_ALTMASK": "1", "ALG_ALT21": "1",
       "ALG_ALT2": "1", "ALG_MASKHEAD": "1", "ALG_FED": "1", "ALG_POLAR": "1",
       "ALG_MINE_BREATHS": "1", "SC_EVAL": "0"}
for k, v in ENV.items(): os.environ.setdefault(k, v)
os.environ["ALG_TEST"] = ".cache/algebra_nl_test.jsonl" if FX == "mint" else ".cache/wild_admitted_holdout.jsonl"
os.environ["ALG_TEST_NAME"] = "test23" if FX == "mint" else "wildhold"
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, alt2_fact_buf
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
vs, vst, vtk, vg, vse = load_alg("test")
N = min(int(os.environ.get("DS_N", "256")), len(vs)); B = 8
p = build_params(0); sd = safe_load(os.environ.get("DS_CKPT", ".cache/sharp_fedon242.safetensors"))
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
U, A = [], []
for s0 in range(0, N, B):
    sl = np.arange(s0, min(s0 + B, N)); pad = B - len(sl); sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    ts = Tensor(vst[sl_p].astype(np.float32), dtype=dtypes.float); tk = Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float); se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
    o0 = forward(p, ts, tk, se); onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res", "pres", "ftype", "op", "dig")}
    if "dup" in o0: onp0["dup"] = o0["dup"].realize().numpy()
    mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32))
    nv = np.array([vs[i].get("n_vars", 24) for i in sl_p]); ma = np.array([vs[i].get("m", 0) for i in sl_p])
    fb = alt2_fact_buf(onp0, vse[sl_p].astype(np.int32), nv, ma)
    o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=Tensor(fb, dtype=dtypes.float))
    U.append(np.stack([b.realize().numpy()[:len(sl)] for b in o["breaths_u"]], 1))     # (b, K, L, 512)
    A.append(np.stack([b.realize().numpy()[:len(sl)] for b in o["breaths_all"]], 1))
U = np.concatenate(U); A = np.concatenate(A)
out = os.environ.get("DS_OUT", f".cache/polar_states_{FX}.npz")
np.savez_compressed(out, u=U.astype(np.float32), all=A.astype(np.float32), n=N)
print(f"[dump] {FX}: u {U.shape} all {A.shape} -> {out}")
