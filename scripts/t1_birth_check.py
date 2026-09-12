"""T1 birth equivalence (zero GPU, CPU): the forward with ALG_T1=K and fresh
(identity) T1 params must equal the ALG_T1=0 forward BITWISE on the same
random params/input. Runs itself twice (the door is module-level).
usage: t1_birth_check.py  (family env in the environment; DEV=CPU)"""
import os, sys, subprocess, numpy as np
if len(sys.argv) == 1:
    outs = []
    for k in ("0", "5"):
        env = dict(os.environ, ALG_T1=k, DEV="CPU"); o = f".cache/t1_birth_{k}.npz"
        subprocess.run([sys.executable, __file__, o], env=env, check=True); outs.append(o)
    a, b = np.load(outs[0]), np.load(outs[1]); worst = 0.0; n = 0
    for key in a.files:
        x, y = a[key], b[key]; eq = np.array_equal(x, y); n += 1
        d = float(np.abs(x - y).max()) if x.shape == y.shape else float("inf"); worst = max(worst, d)
        if not eq: print(f"  {key}: NOT bitwise, maxabs {d:.3e}")
    print(f"[t1-birth] {n} outputs compared; worst maxabs {worst:.3e} -> {'PASS (bitwise)' if worst == 0.0 else 'FAIL'}")
    sys.exit(0 if worst == 0.0 else 1)
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import phase1_algebra_head as H
from tinygrad import Tensor, dtypes
p = H.build_params(0)
if H.ALG_T1:
    print(f"[t1-birth] ALG_T1={H.ALG_T1}: t1 params {sum(int(np.prod(p[k].shape)) for k in p if k.startswith('t1_'))}")
rng = np.random.RandomState(3); B, T = 2, H.T_ALG
tr = Tensor((rng.randn(B, T, H.H_TRUNK) * 0.1).astype(np.float32))
tk = np.zeros((B, T), np.float32); tk[0, :23] = 1; tk[1, :31] = 1
se = np.zeros((B, T), np.int32); se[0, 10:23] = 1; se[1, 12:31] = 1; se[1, 25:31] = 2
o = H.forward(p, tr, Tensor(tk), Tensor(se), slot_mask=Tensor(np.ones((B, H.L_FAC, H.L_FAC), np.float32)),
              fact_buf=Tensor(np.zeros((B, H.K_VARS, 4), np.float32)))
np.savez(sys.argv[1], **{k: v.numpy() for k, v in o.items() if hasattr(v, "numpy")})
