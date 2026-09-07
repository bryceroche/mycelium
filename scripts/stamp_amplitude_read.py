"""stamp_amplitude_read.py — THE AMPLITUDE READ (2026-09-07): the
confidence-stamp norms (||_wg4|| per breath x row x slot, the garage
deposits) on a fixed batch of test rows, ALL rows sealed (_PCV = 1),
for a list of checkpoints. Tests the amplitude-collapse hypothesis
behind the dose-0.15 death: the damaged snapshot should show a mass of
near-zero stamps (the stamp is ||w|| + 1e-6, so a collapsed stamp reads
~1e-6) absent from healthy snapshots. Read-only; GPU; no training.
Usage: STAMP_CKPTS=a.safetensors,b.safetensors python stamp_amplitude_read.py"""
import os, sys
os.environ.setdefault("DEV", "PCI+AMD")
ENV = {"ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1", "ALG_HW": "512",
       "ALG_WIDE": "1", "ALG_BREATH": "7", "ALG_NOTEBOOK": "1",
       "ALG_SIXWAVE": "1", "NB_PERSLOT": "1", "ALG_BINDBUS": "7",
       "ALG_BIND_D": "512", "BIND_CODES": ".cache/bindbus_codes512.npz",
       "ALG_BUSGARAGE": "2", "ALG_SHELF_CIRCLE": "2", "ALG_ALTMASK": "1",
       "ALG_ALT21": "1", "ALG_ALT2": "1", "ALG_MASKHEAD": "1",
       "ALG_FED": "1", "ALG_TEST": ".cache/algebra_nl_test.jsonl",
       "ALG_TEST_NAME": "test23", "ALG_PC_MIX": "0.3"}
os.environ.update(ENV); os.environ.pop("SC_EVAL", None)
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as HEAD
from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

vs, vst, vtk, vg, vse = load_alg("test")
B = int(os.environ.get("STAMP_B", "16")); sl = np.arange(B)
p = build_params(0)
ts = Tensor(vst[sl].astype(np.float32), dtype=dtypes.float)
tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
HEAD._PCV = Tensor(np.ones((B, 1, 1), np.float32)).contiguous().realize()   # all rows SEALED

def read(ckpt):
    sd = safe_load(ckpt)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    o0 = forward(p, ts, tk, se)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(onp0, vse[sl].astype(np.int32))
    HEAD._STEP_TAP = {}
    o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float))
    _ = o["pres"].realize()
    garage = HEAD._STEP_TAP["state"]["garage"]; HEAD._STEP_TAP = None
    norms = np.stack([g.realize().numpy().astype(np.float64) for g in garage])   # (breaths, B, L, D)
    n = np.linalg.norm(norms, axis=-1)                                           # (breaths, B, L)
    return n

for ck in os.environ["STAMP_CKPTS"].split(","):
    n = read(ck)
    flat = n.ravel()
    q = np.quantile(flat, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(f"[stamp] {os.path.basename(ck)}: breaths={n.shape[0]} stamps={flat.size} "
          f"<1e-5: {np.mean(flat < 1e-5):.4f}  <1e-3: {np.mean(flat < 1e-3):.4f}  <1e-2: {np.mean(flat < 1e-2):.4f}  "
          f"q01/05/25/50/75/95/99 = {' '.join(f'{x:.3g}' for x in q)}  max={flat.max():.3g}", flush=True)
    per_b = [f"b{i}:{np.median(n[i]):.3g}/{np.mean(n[i] < 1e-3):.2f}" for i in range(n.shape[0])]
    print("         per-breath median/frac<1e-3: " + "  ".join(per_b), flush=True)
