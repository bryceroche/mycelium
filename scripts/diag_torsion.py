"""Torsion diagnostic (v300 seed measurement tier — predictions pinned first).

Discrete torsion of latent trajectories across breaths, from persisted z
bundles (pure numpy; no model, no GPU). For each (sample, latent) trajectory
z_0..z_K: deltas D_k = z_{k+1} - z_k; consecutive osculating planes
span(D_k, D_{k+1}) and span(D_{k+1}, D_{k+2}) share D_{k+1}; the dihedral
angle about the shared line is tau_k:
    u = D_k    - (D_k    . e) e,  e = D_{k+1}/|D_{k+1}|
    w = D_{k+2} - (D_{k+2} . e) e
    tau_k = angle(u, w)
Estimator floor (pinned): mask tau_k where sin(theta) of EITHER flanking
delta vs the shared delta < 0.05 (~3 deg of parallel) — the ridge regime,
where the angle is numerically meaningless. Curvature reported alongside.

Pinned predictions: (1) tau > 0 above floor; (2) parity periodicity (even
vs odd k); (3) #238 twist-direction overlap with carrier dims (proxy for
the memory subspace; slot-subspace overlap is the diag's v1.1) exceeds the
control's.

Usage:
  .venv/bin/python scripts/diag_torsion.py            # both runs, all bundles
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)

PERSIST = ".cache/v200_smoke/persistence"
OUT     = ".cache/v200_smoke/torsion_diag.json"
FLOOR   = 0.05   # pinned estimator floor (sin of flanking-vs-shared angle)
# Carrier dims measured at #237 step-500 (source-attribution check) — the
# persistent-subspace proxy for prediction 3:
CARRIER_DIMS = [146, 176, 302, 594, 916, 1139, 1369, 1554, 1642, 1644]


def torsion_for_bundle(z: np.ndarray) -> dict:
    """z: (K+1, B, L, H) float. Returns per-k torsion stats over (B, L)."""
    D = np.diff(z.astype(np.float64), axis=0)            # (K, B, L, H)
    K = D.shape[0]
    taus, curls, masks, twist_carrier = [], [], [], []
    for k in range(K - 2):
        d0, d1, d2 = D[k], D[k + 1], D[k + 2]            # (B, L, H)
        e = d1 / (np.linalg.norm(d1, axis=-1, keepdims=True) + 1e-12)
        u = d0 - (d0 * e).sum(-1, keepdims=True) * e
        w = d2 - (d2 * e).sum(-1, keepdims=True) * e
        un = np.linalg.norm(u, axis=-1)
        wn = np.linalg.norm(w, axis=-1)
        sin_u = un / (np.linalg.norm(d0, axis=-1) + 1e-12)
        sin_w = wn / (np.linalg.norm(d2, axis=-1) + 1e-12)
        valid = (sin_u >= FLOOR) & (sin_w >= FLOOR)      # (B, L)
        cosang = (u * w).sum(-1) / (un * wn + 1e-12)
        tau = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        taus.append(float(tau[valid].mean()) if valid.any() else float("nan"))
        curls.append(float(np.minimum(sin_u, sin_w)[valid].mean()) if valid.any() else float("nan"))
        masks.append(float(valid.mean()))
        # twist direction = out-of-plane component of d2 (w orthogonal to u too)
        w_hat = w / (wn[..., None] + 1e-12)
        u_hat = u / (un[..., None] + 1e-12)
        t = w_hat - (w_hat * u_hat).sum(-1, keepdims=True) * u_hat
        tn = np.linalg.norm(t, axis=-1, keepdims=True)
        t = t / (tn + 1e-12)
        frac = (t[..., CARRIER_DIMS] ** 2).sum(-1)       # energy of twist dir in carrier dims
        twist_carrier.append(float(frac[valid].mean()) if valid.any() else float("nan"))
    return {"tau_deg_per_k": taus, "curvature_per_k": curls,
            "valid_frac_per_k": masks, "twist_carrier_frac_per_k": twist_carrier}


def main() -> None:
    results = {}
    for run, pat in (("237_5", f"{PERSIST}/step*_z_237_5.npz"),
                     ("238",   f"{PERSIST}/step*_z_238.npz")):
        entries = {}
        for p in sorted(glob.glob(pat),
                        key=lambda q: int(re.search(r"step(\d+)_", q).group(1))):
            step = int(re.search(r"step(\d+)_", p).group(1))
            z = np.load(p)["data"].astype(np.float32)   # (K+1, B, L, H)
            r = torsion_for_bundle(z)
            entries[step] = r
            t = r["tau_deg_per_k"]
            even = [t[i] for i in range(len(t)) if i % 2 == 0 and np.isfinite(t[i])]
            odd  = [t[i] for i in range(len(t)) if i % 2 == 1 and np.isfinite(t[i])]
            print(f"[{run} step {step:5d}] tau(deg)={['%.1f' % v for v in t]}  "
                  f"parity even={np.mean(even):.1f} odd={np.mean(odd):.1f}  "
                  f"valid={['%.2f' % v for v in r['valid_frac_per_k']]}  "
                  f"carrier-frac={['%.3f' % v for v in r['twist_carrier_frac_per_k']]}",
                  flush=True)
        results[run] = entries
    chance = len(CARRIER_DIMS) / 2048
    print(f"(carrier-frac chance level = {chance:.4f})")
    with open(OUT, "w") as f:
        json.dump({"floor": FLOOR, "carrier_dims": CARRIER_DIMS,
                   "carrier_frac_chance": chance, "results": results}, f, indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
