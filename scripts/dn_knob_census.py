"""THE STATE-SPACE DENOISER'S KNOB CENSUS (2026-10-12, the pre/post knob law): prints, per
breath, the READING stream's injected Gumbel-noise std vs the bank's own pre-bias score spread
(reusing the existing _CERT_CENSUS_SC readback in bank(), armed directly rather than through
ALG_CERT_CENSUS/ALG_WHEEL_CERT's own one-shot call), and the MATH stream's sigma_k (already a
closed-form fraction of the content norm — no probe needed for the ratio itself, only for a
sanity readout of the content std it is a fraction OF, from the dn_whip_content_pre _CENSUS tap).

Usage (DEV=CPU; the gate's own env plus):
  ALG_DN_READ=<tau_max> ALG_DN_WHIP=<sigma_max> [ALG_SEED_RNG=<seed>] \
  .venv/bin/python3 scripts/dn_knob_census.py [n_rows]
"""
import os, sys, math
os.environ["DEV"] = "CPU"
sys.path.insert(0, "scripts")
import numpy as np
from tinygrad import Tensor, dtypes
import phase1_algebra_head as H

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
samples, states, tokmask, gold, sent = H.load_alg("test")
n = min(N, len(samples))
sl = np.arange(n)
p = H.build_params(0)
_ckpt = os.environ.get("DN_KNOB_CKPT")
if _ckpt:
    from tinygrad.nn.state import safe_load
    sd = safe_load(_ckpt)
    for k in p:
        if k in sd:
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    print(f"[dn-knob] loaded {_ckpt} ({sum(1 for k in p if k in sd)}/{len(p)} keys)")
K_B = int(os.environ.get("ALG_BREATH", "7"))
T = H.T_ALG
GUMBEL_STD = math.pi / math.sqrt(6.0)   # std of a standard Gumbel(0,1) draw

gfqsched_t = None
if H.ALG_DN_READ:
    gfqsched = H.seed_gumbel_sched_build_array(int(os.environ.get("ALG_SEED_RNG", "777")), sl, T, K_B)
    gfqsched_t = Tensor(gfqsched.astype(np.float32), dtype=dtypes.float)

H.__dict__["_CERT_CENSUS_SC"] = {}
if H.ALG_DN_WHIP:
    H.__dict__["_CENSUS"] = []

mask = np.ones((n, H.L_FAC, H.L_FAC), np.float32)
out = H.forward(p, Tensor(states[sl].astype(np.float32), dtype=dtypes.float),
                 Tensor(tokmask[sl].astype(np.float32), dtype=dtypes.float),
                 Tensor(sent[sl].astype(np.int32), dtype=dtypes.int),
                 slot_mask=Tensor(mask, dtype=dtypes.float),
                 seed_fq_sched=gfqsched_t)
out["pres"].realize()

print(f"[dn-knob] N={n} K_B={K_B} ALG_DN_READ={H.ALG_DN_READ!r} ALG_DN_WHIP={H.ALG_DN_WHIP!r}")

if H.ALG_DN_READ:
    sc = H.__dict__["_CERT_CENSUS_SC"]
    for kb in range(K_B):
        tau = (H.ALG_SEED_TEMP * (math.cos(kb * math.pi / (2.0 * (K_B - 1))) ** 2)
               if K_B > 1 else H.ALG_SEED_TEMP)
        inj_std = tau * GUMBEL_STD
        vals = sc.get(kb)
        port = (sum(vals) / len(vals)) if vals else None
        ratio = (inj_std / port) if port else None
        inert = "UNKNOWN" if ratio is None else ("YES" if ratio < 0.10 else "NO")
        print(f"[dn-knob-read] kb={kb} tau_k={tau!r} inj_std={inj_std!r} "
              f"port_spread={port!r} ratio={ratio!r} INERT_BY_CONSTRUCTION={inert}")

if H.ALG_DN_WHIP:
    census = H.__dict__.get("_CENSUS") or []
    pre = {kb: a for (kb, name, a) in census if name == "dn_whip_content_pre"}
    for kb in range(1, K_B):
        sigma = H._dn_whip_sigma(kb, K_B)
        pre_a = pre.get(kb)
        content_std = float(pre_a.std()) if pre_a is not None else None
        inert = "YES" if sigma < 0.10 else "NO"
        print(f"[dn-knob-whip] kb={kb} sigma_k={sigma!r} content_std_sample={content_std!r} "
              f"(sigma_k IS the injected fraction of the content NORM -- the ratio itself) "
              f"INERT_BY_CONSTRUCTION={inert}")

print("[dn-knob] DONE")
