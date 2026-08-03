"""sign_terminal_smoke.py — THE SIGN-TERMINAL SMOKE (2026-08-03; its
own smoke per Bryce's rider — Fire-0's inert-patch shape: a new pathway
whose gradient can be structurally absent while everything compiles.
One deliberately-signed row through the loss; gradient asserted NONZERO
at the sign terminal, BEFORE any corpus mints)."""
import os, sys
os.environ["ALG_WIDE"] = "1"
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from tinygrad import Tensor, dtypes
from phase1_algebra_head import (build_params, forward, loss_fn, decode,
                                 N_DIG, L_FAC, K_VARS, T_ALG, ALG_WIDE)

assert ALG_WIDE == 1 and N_DIG == 7, (ALG_WIDE, N_DIG)
print(f"[smoke] ALG_WIDE=1: N_DIG={N_DIG}")

rng = np.random.default_rng(103800)
p = build_params(0)
assert "h_sgn" in p, "sign terminal missing from params"
B = 1
st = Tensor(rng.standard_normal((B, T_ALG, 2048)).astype(np.float32))
msk = Tensor(np.ones((B, T_ALG), np.float32))
snt = Tensor(np.zeros((B, T_ALG), np.int32), dtype=dtypes.int)
for t in p.values():
    t.requires_grad = True
o = forward(p, st, msk, snt)
assert "sgn" in o, "sgn missing from forward emission"
print("[smoke] forward emits sgn:", tuple(o["sgn"].shape))

# gold: slot 0 = given a = -7 (sign 1, digits |7| wide); slot 1 = given b = 987654
digits = np.zeros((B, L_FAC, N_DIG), np.int32)
digits[0, 0] = [0, 0, 0, 0, 0, 0, 7]
digits[0, 1] = [0, 9, 8, 7, 6, 5, 4]
sign = np.zeros((B, L_FAC), np.float32); sign[0, 0] = 1.0
is_lit = np.zeros((B, L_FAC), np.float32); is_lit[0, :2] = 1.0
pres = np.zeros((B, L_FAC), np.float32); pres[0, :2] = 1.0
g = {"presence": Tensor(pres),
     "is_lit_f": Tensor(is_lit),
     "is_mod": Tensor(np.zeros((B, L_FAC), np.float32)),
     "is_sel": Tensor(np.zeros((B, L_FAC), np.float32)),
     "is_pct": Tensor(np.zeros((B, L_FAC), np.float32)),
     "is_fdiv": Tensor(np.zeros((B, L_FAC), np.float32)),
     "args": Tensor(np.zeros((B, L_FAC, K_VARS), np.float32)),
     "fspan": Tensor(np.ones((B, L_FAC, T_ALG), np.float32)),
     "vspan": Tensor(np.ones((B, K_VARS, T_ALG), np.float32)),
     "ftype": Tensor(np.full((B, L_FAC), 1, np.int32), dtype=dtypes.int),
     "op": Tensor(np.zeros((B, L_FAC), np.int32), dtype=dtypes.int),
     "res": Tensor(np.zeros((B, L_FAC), np.int32), dtype=dtypes.int),
     "digits": Tensor(digits, dtype=dtypes.int),
     "sign": Tensor(sign),
     "query": Tensor(np.zeros((B,), np.int32), dtype=dtypes.int)}
l = loss_fn(o, g)
lv = float(l.numpy())
assert np.isfinite(lv), lv
l.backward()
gmax = float(p["h_sgn"].grad.abs().max().numpy())
gdig = float(p["h_dig"].grad.abs().max().numpy())
print(f"[smoke] loss={lv:.4f}  |h_sgn.grad|max={gmax:.3e}  |h_dig.grad|max={gdig:.3e}")
assert gmax > 0, "SIGN TERMINAL INERT — grad is zero (the Fire-0 patch shape)"
assert gdig > 0

# decode roundtrip with sign
onp = {"pres": np.full((L_FAC,), -9., np.float32),
       "ftype": np.zeros((L_FAC, 8), np.float32),
       "op": np.zeros((L_FAC, 2), np.float32),
       "islit": np.full((L_FAC,), 9., np.float32),
       "dig": np.zeros((L_FAC, N_DIG, 10), np.float32),
       "sgn": np.full((L_FAC,), -9., np.float32),
       "args": np.full((L_FAC, K_VARS), -9., np.float32),
       "res": np.zeros((L_FAC, K_VARS), np.float32),
       "query": np.zeros((K_VARS,), np.float32)}
onp["pres"][0] = 9.; onp["ftype"][0, 1] = 9.; onp["res"][0, 3] = 9.
for d_, dig_ in enumerate([0, 0, 0, 0, 0, 0, 7]): onp["dig"][0, d_, dig_] = 9.
onp["sgn"][0] = 9.
facs, q = decode(onp)
assert facs[0] == {"ftype": "given", "var": 3, "value": -7}, facs[0]
print("[smoke] decode roundtrip: given var=d value=-7 ✓")
print("SIGN-TERMINAL SMOKE: PASS — the terminal is live, both feeds wired")
