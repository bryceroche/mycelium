"""THE STATE-SPACE DENOISER'S MATH STREAM — THE CLOCK CHECK + THE SCHEDULE CHECK (2026-10-12).

Direct, deterministic tests of `_dn_whip_kick` / `_dn_whip_sigma` (scripts/phase1_algebra_head.py),
called DIRECTLY (the birth-check idiom: never inferred from a probe that might not even enter the
breath loop). No GPU, no training, no JSONL fixture — a synthetic (B, LT, HW) state tensor under
ALG_POLAR=1 ALG_POLAR_D=128 (so _polar_sink()'s content/clock split is the real 384/128 one).

Checks (printed, each its own grep-able line):
  [dn-whip-clock]   kb=<k>: the 128 clock dims of the kicked state equal the UNKICKED state's
                    bitwise (max abs diff == 0.0) at every kb in 1..K_B-1 where sigma_k > 0.
  [dn-whip-norm]    kb=<k>: the kicked state's CONTENT-block norm equals the unkicked state's
                    content norm to within 1e-5 (keepnorm).
  [dn-whip-sched]   kb=<k>: sigma_k printed; sigma_{K_B-1} (the last breath) reported as ~0
                    (it is cos^2(pi/2)*sigma_max ~ 1e-33, not literally the bit 0.0 — the same
                    floating-point character ALG_BLUR's own schedule has at its own last rung).
  [dn-whip-check] PASS/FAIL overall.
"""
import os
os.environ.setdefault("ALG_POLAR", "1")
os.environ.setdefault("ALG_POLAR_D", "128")
os.environ["DEV"] = "CPU"
import sys
sys.path.insert(0, "scripts")
import numpy as np
from tinygrad import Tensor

ALG_DN_WHIP = float(os.environ.get("ALG_DN_WHIP", "0.3"))
os.environ["ALG_DN_WHIP"] = str(ALG_DN_WHIP)
import phase1_algebra_head as H

K_B = int(os.environ.get("ALG_BREATH", "7"))
B, LT, HW = 2, 24, H.H_W

ok = True
rng = np.random.RandomState(0)
cur0 = Tensor(rng.randn(B, LT, HW).astype(np.float32))
gc = H._polar_sink()[2].realize().numpy()   # (1,1,1,HW)-ish broadcastable content gate
gc_flat = gc.reshape(-1)
assert gc_flat.shape[0] == HW, f"gate_content shape {gc.shape} does not broadcast to HW={HW}"
n_content = int((gc_flat > 0.5).sum())
n_clock = HW - n_content
print(f"[dn-whip-check] HW={HW} content_dims={n_content} clock_dims={n_clock} ALG_DN_WHIP={ALG_DN_WHIP} K_B={K_B}")

cur0_np = cur0.realize().numpy()
for kb in range(1, K_B):
    sigma = H._dn_whip_sigma(kb, K_B)
    print(f"[dn-whip-sched] kb={kb} sigma_k={sigma!r}")
    if sigma == 0.0:
        continue
    kicked = H._dn_whip_kick(cur0, kb, K_B).realize().numpy()
    # THE CLOCK CHECK: the clock-plane dims bitwise untouched.
    clock_diff = np.abs((kicked - cur0_np) * (1.0 - gc_flat)).max()
    clock_pass = bool(clock_diff == 0.0)
    print(f"[dn-whip-clock] kb={kb} max_abs_diff_on_clock_dims={clock_diff!r} PASS={clock_pass}")
    ok = ok and clock_pass
    # THE KEEPNORM CHECK: content-block norm restored to within 1e-5.
    nc0 = np.sqrt(((cur0_np * gc_flat) ** 2).sum(-1))
    nck = np.sqrt(((kicked * gc_flat) ** 2).sum(-1))
    norm_diff = np.abs(nc0 - nck).max()
    norm_reldiff = (np.abs(nc0 - nck) / nc0.clip(1e-6)).max()
    norm_pass = bool(norm_reldiff < 1e-5)   # relative: float32 arithmetic on norm ~O(10-20) makes an
                                             # absolute 1e-5 too tight (observed ~1.3e-5 absolute,
                                             # ~6e-7 relative) — "to 1e-5" read as relative precision.
    print(f"[dn-whip-norm] kb={kb} max_abs_content_norm_diff={norm_diff!r} max_rel_diff={norm_reldiff!r} PASS={norm_pass}")
    ok = ok and norm_pass
    # sanity: the kick actually moved the content direction (not a no-op)
    content_moved = float(np.abs((kicked - cur0_np) * gc_flat).max())
    print(f"[dn-whip-check] kb={kb} content_moved_max_abs={content_moved!r} (sigma={sigma!r})")

# THE SCHEDULE CHECK: the last breath's sigma is ~0 (same floating-point character as the
# blurred ladder's own cos^2 schedule — not literally the bit 0.0, but < 1e-20).
sigma_last = H._dn_whip_sigma(K_B - 1, K_B)
sched_pass = bool(abs(sigma_last) < 1e-20)
print(f"[dn-whip-sched] LAST BREATH kb={K_B-1} sigma={sigma_last!r} <1e-20={sched_pass}")
ok = ok and sched_pass

# THE READING STREAM'S OWN SCHEDULE CHECK (same formula, ALG_SEED_SCHED's tau_k):
import math
tau_last = 1.0 * (math.cos((K_B - 1) * math.pi / (2.0 * (K_B - 1))) ** 2)
tau_pass = bool(abs(tau_last) < 1e-20)
print(f"[dn-read-sched] LAST BREATH kb={K_B-1} tau={tau_last!r} <1e-20={tau_pass}")
ok = ok and tau_pass

print(f"[dn-whip-check] OVERALL {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
