"""impulse_response.py — THE IMPULSE RESPONSE OF DELIBERATION
(2026-09-01, word given; the mature domain: systems identification).
Kick the breath loop's state at breath k0 with a small pattern; read the
response at every later breath. Gain g(lag) = ||delta(k0+lag)|| /
||delta(k0)||, averaged over rows. Two probes: NOISE (gaussian) and
FACT-SHAPED (a role-bound codebook wire). PINNED: (1) the loop is a
CONTRACTION — noise gain g(lag>=3) < 0.5 (settle exists, causally);
(2) fact-shaped probes damp SLOWER than noise (the loop keeps what fits
its modes — truth resonates, structurally). Amplification (g growing)
would flag instability: none expected, worth knowing.
Env: IR_CKPT (default sharp_bind14a).
"""
import os, sys
for _k, _v in {"DEV": "PCI+AMD", "ALG2": "1", "ALG_FTYPES": "9",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_BREATH": "7", "ALG_NOTEBOOK": "1", "ALG_SIXWAVE": "1",
                   "NB_PERSLOT": "1", "ALG_BINDBUS": "7", "ALG_BIND_D": "512",
                   "BIND_CODES": ".cache/bindbus_codes512.npz",
                   "ALG_BUSGARAGE": "2", "ALG_MINE_BREATHS": "1",
                   "ALG_TEST": ".cache/algebra_nl_test.jsonl",
                   "ALG_TEST_NAME": "test23"}.items():
    os.environ.setdefault(_k, _v)   # IMPULSE v2 (2026-09-18): the caller's family env wins (the bind14a defaults were an older era)
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import (build_params, forward, load_alg, L_TOT,
                                 build_slot_masks)
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

vs, vst, vtk, vg, vse = load_alg("test")
p = build_params(0)
sd = safe_load(os.environ.get("IR_CKPT", ".cache/sharp_bind14a.safetensors"))
for k in p:
    if k in sd:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
N, K_B = 64, 7
rng = np.random.default_rng(3)
bz = np.load(".cache/bindbus_codes512.npz")


def run(sl, imp=None):
    H._IMP = imp
    ts = Tensor(vst[sl].astype(np.float32), dtype=dtypes.float)
    tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
    o0 = forward(p, ts, tk, se)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(onp0, vse[sl].astype(np.int32))
    o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float))
    br = [b.realize().numpy() for b in o["breaths_all"]]
    H._IMP = None
    return br


# state scale from a baseline batch
sl0 = np.arange(8)
b0 = run(sl0)
rms = float(np.sqrt(np.mean(np.square(b0[3]))))
EPS = 0.1 * rms
print(f"[impulse] state rms {rms:.3f}; kick eps {EPS:.3f} "
      f"(ckpt={os.environ.get('IR_CKPT', 'sharp_bind14a')})")

# probes: gaussian noise vs fact-shaped (role-bound codebook wire)
noise = rng.standard_normal((1, L_TOT, 512)).astype(np.float32)
noise = noise / np.linalg.norm(noise) * EPS * np.sqrt(L_TOT)
th = bz["theta_res"]
code = bz["CB"][7].reshape(256, 2)
c, s_ = np.cos(th), np.sin(th)
fact1 = np.stack([code[:, 0] * c - code[:, 1] * s_,
                  code[:, 0] * s_ + code[:, 1] * c], -1).reshape(512)
fact = np.tile(fact1, (1, L_TOT, 1)).astype(np.float32)
fact = fact / np.linalg.norm(fact) * EPS * np.sqrt(L_TOT)

for pname, pat in (("noise", noise), ("fact", fact)):
    print(f"probe={pname}:  k0 \\ gain at lag 1..")
    for k0 in (1, 2, 3, 4):
        gains = []
        for s0 in range(0, N, 8):
            sl = np.arange(s0, s0 + 8)
            base = run(sl)
            pert = run(sl, imp=(k0, Tensor(np.broadcast_to(
                pat, (8, L_TOT, 512)).copy(), dtype=dtypes.float)))
            d = [float(np.linalg.norm(pert[k] - base[k]) / 8) for k in range(K_B)]
            d0 = d[k0] + 1e-9
            gains.append([d[k] / d0 for k in range(k0, K_B)])
        g = np.mean(gains, 0)
        print(f"  k0={k0}:  " + " ".join(f"{x:.3f}" for x in g))
print("[pinned] contraction iff noise gain(lag>=3) < 0.5; "
      "fact damps slower than noise = resonance")

# IMPULSE v2 (2026-09-18, collected): tau-fit + the linearity sweep + the sextet projection, on the noise probe at k0=1
def _gain_curve(pat_, k0=1):
    gains = []
    for s0 in range(0, N, 8):
        sl = np.arange(s0, s0 + 8); base = run(sl)
        pert = run(sl, imp=(k0, Tensor(np.broadcast_to(pat_, (8, L_TOT, 512)).copy(), dtype=dtypes.float)))
        d = [float(np.linalg.norm(pert[k] - base[k]) / 8) for k in range(K_B)]; d0 = d[k0] + 1e-9
        gains.append([d[k] / d0 for k in range(k0, K_B)])
    return np.mean(gains, 0)
g1 = _gain_curve(noise); lags = np.arange(len(g1))
m = lags[1:] if len(lags) > 1 else lags; y = np.log(np.maximum(g1[1:], 1e-6)) if len(lags) > 1 else np.log(g1)
A = np.vstack([m, np.ones_like(m)]).T; slope, icpt = np.linalg.lstsq(A, y, rcond=None)[0]; tau = -1.0 / slope if slope < 0 else float("inf")
resid = y - (slope * m + icpt); sext = float(np.dot(resid, np.cos(2 * np.pi * m / 6.0))) / max(len(m), 1)
g_half = _gain_curve(noise * 0.5); g_dbl = _gain_curve(noise * 2.0)
lin = float(np.mean(np.abs(g_half[1:] - g1[1:]) + np.abs(g_dbl[1:] - g1[1:])) / max(float(np.mean(g1[1:])), 1e-6)) if len(g1) > 1 else float("nan")
print(f"[impulse-v2] noise k0=1 gains {' '.join(f'{x:.3f}' for x in g1)} | tau-fit {tau:.2f} breaths (log-linear lags 1..{len(m)}) | sextet projection of the residual {sext:+.4f} (6 lags: a projection, not a detection) | linearity: mean |gain(0.5x, 2x) - gain(1x)| / gain(1x) = {lin:.3f} (0 = linear)", flush=True)
