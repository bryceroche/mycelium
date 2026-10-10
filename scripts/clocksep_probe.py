"""THE CLOCK PROBE for ALG_CLOCK_SEP (2026-10-09; CPU, zero GPU, zero training).

At INIT (build_params(0), no checkpoint, random init): can breath time be read from the
REGISTER (state["clk"], surfaced as out["clk_all"] -- scripts/phase1_algebra_head.py's
breath_step/forward, the same ALG_MINE_BREATHS tap convention as breaths_u/breaths_r) the
same way clock_read.ridge_probe reads it from the bus's clock planes today ("the clock (no
gain, probe 1.0)", ledger 2026-09-08/09)? BAR (the build's own spec): probe >= 0.95 at init
-- the register's turn is the SAME deterministic six-wave code at full width, so it should
be at least as readable as today's narrower in-bus band, with no training and no gain.

Run plain for the POSITIVE read; run with ALG_SEVER=clockband (already set by the caller,
this script never touches ALG_SEVER itself) for the SEVER CHECK -- the register zeroed
leaving each breath -- which must read a LOWER probe (the register is where breath time
lives now; if the sever did not move it, nothing material happened). The chain/gate script
runs this twice (plain, then with ALG_SEVER=clockband in the environment) and compares.

Usage:  .venv/bin/python3 scripts/clocksep_probe.py
        ALG_SEVER=clockband .venv/bin/python3 scripts/clocksep_probe.py
"""
import os
import sys

sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
os.environ["DEV"] = "CPU"                 # HARD: this probe never touches the GPU
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
os.environ.setdefault("ALG_SHELF_CIRCLE", "2")
os.environ.setdefault("ALG_ALTMASK", "1")
os.environ.setdefault("ALG_ALT21", "1")
os.environ.setdefault("ALG_ALT2", "1")
os.environ.setdefault("ALG_MASKHEAD", "1")
os.environ.setdefault("ALG_FED", "1")
os.environ.setdefault("ALG_TEST", ".cache/algebra_nl_test.jsonl")
os.environ.setdefault("ALG_TEST_NAME", "test23")
os.environ["ALG_POLAR"] = "1"             # THE SEPARATED CLOCK needs its bus
os.environ["ALG_CLOCK_SEP"] = "1"
os.environ["ALG_MINE_BREATHS"] = "1"      # arms the clk_all tap
os.environ.setdefault("SC_EVAL", "0")

import numpy as np                                               # noqa: E402
from tinygrad import Tensor, dtypes                              # noqa: E402
import phase1_algebra_head as M                                  # noqa: E402
import clock_read as CR                                          # noqa: E402

B = 8
vs, vst, vtk, vg, vse = M.load_alg("test")
sl = np.arange(B)
TS = Tensor(vst[sl].astype(np.float32), dtype=dtypes.float)
TK = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
SE = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
SENT = vse[sl].astype(np.int32)

P = M.build_params(0)
_ckpt = os.environ.get("CSEP_CKPT", "")
if _ckpt:
    # THE CLOCK PROBE ON THE TRAINED BODY: load a real checkpoint (e.g. CS_241's own) in
    # place of the fresh-init read above -- does breath time survive 48k steps of training
    # in the register, or does it collapse/get overwritten? STRICT key match (the checkpoint
    # must have been trained under this exact ALG_CLOCK_SEP=1 build).
    from tinygrad.nn.state import safe_load
    _sd = safe_load(_ckpt)
    assert set(_sd.keys()) == set(P.keys()), (
        "STRICT key mismatch loading CSEP_CKPT", sorted(set(_sd) - set(P))[:6],
        sorted(set(P) - set(_sd))[:6])
    for _k in P:
        P[_k].assign(_sd[_k].to(P[_k].device).cast(P[_k].dtype)).realize()
    print(f"[clocksep-probe] loaded CSEP_CKPT={_ckpt} (strict key match)")
o0 = M.forward(P, TS, TK, SE)
onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
mk = M.build_slot_masks(onp0, SENT)
o = M.forward(P, TS, TK, SE, slot_mask=Tensor(mk, dtype=dtypes.float))
assert "clk_all" in o, "ALG_CLOCK_SEP did not surface out['clk_all'] -- check the build"
clk_all = [b.realize().numpy() for b in o["clk_all"]]
K = len(clk_all)
assert K == 6, f"expected 6 register pages (breaths 1..6, no breath-0 register); got {K}"
for j, a in enumerate(clk_all):
    assert np.isfinite(a).all(), f"clk_all[{j}] non-finite"
D = clk_all[0].shape[-1]
assert D == 128, f"register width {D} != 128"
X = np.stack([a.reshape(-1, D) for a in clk_all], axis=0)        # (K, N, D), per-slot samples
pr = CR.ridge_probe(X, 242)
_sev = os.environ.get("ALG_SEVER", "") or "(none)"
_tag = "ON THE TRAINED BODY" if _ckpt else "AT BIRTH"
print(f"[clocksep-probe] ALG_SEVER={_sev} REGISTER breath probe {_tag}: "
      f"acc={pr['acc']:.4f} norm-only control={pr['norm_acc']:.4f} "
      f"(K={K} N={X.shape[1]} D={D})")
