"""polar_waist_init.py — THE CONTENT-PLANE WAIST's PCA BIRTH (CPU, numpy
only, zero GPU, zero training; 2026-09-08).

ALG_POLAR_D (scripts/apply_polar_sink.py, deliverable A) collapses and
expands the CONTENT dims of the polar direction u once per breath:

    c' = c @ W_down @ W_up      W_down (384, d), W_up (d, 384)

with the CLOCK planes untouched (the bottleneck must not be able to fight
the rotation) and the content block's norm restored afterwards (so ‖u‖ = 1
still holds and the clock planes stay BITWISE what the sextet left).

THIS SCRIPT BUILDS THE BIRTH. A random W_down/W_up would be a fresh 98k-
parameter organ dropped into a warm continuation — the refuted-cell map's
"blur, not compute". Instead the birth is the ORTHOGONAL PROJECTION onto
the CHAMPION'S OWN top-d content subspace: the top-d principal directions
V of the content dims, pooled over both fixtures, all seven breaths and
all 24 slots of the dumped fedon242 states (.cache/polar_states_mint.npz,
.cache/polar_states_wild.npz, key `u` = (256, 7, 24, 512) breaths_u under
ALG_POLAR=1, open regime).  W_down = V, W_up = V^T.

The head LOADS this file when ALG_POLAR_D > 0 (loud error if absent — no
silent random init; ALG_POLAR_D_INIT=path overrides the default name).

Content planes come from the SAME band file the head reads
(.cache/polar_bands.json v2): plane p is CONTENT iff no wheel claims it;
plane p owns dims 2p and 2p+1.  Nothing here is a diagnostic that enters
a loss: it is an initialization, computed once, banked, and hashed.

Run from the repo root:
  .venv/bin/python3 scripts/polar_waist_init.py            # d = 64,128,256
  PWI_DIMS=128 .venv/bin/python3 scripts/polar_waist_init.py
  PWI_MODE=uncentered PWI_DIMS=128 .venv/bin/python3 scripts/polar_waist_init.py
"""
import hashlib
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)

BANDS = os.environ.get("ALG_POLAR_BANDS", ".cache/polar_bands.json")
FIX = {"mint": ".cache/polar_states_mint.npz",
       "wild": ".cache/polar_states_wild.npz"}
DIMS = [int(x) for x in os.environ.get("PWI_DIMS", "64,128,256").split(",")]
# THE BASIS (lead's ruling 2026-09-08, decision 2): the map the head runs is
#   c -> c @ W_down @ W_up   —  LINEAR, no bias, NO CENTERING.
# "centered" takes the eigenvectors of the covariance (classical PCA: the
# spec's wording); "uncentered" takes the eigenvectors of the SECOND MOMENT
# E[c c^T], which is the least-squares-optimal rank-d subspace FOR THE MAP
# ACTUALLY RUN. The two differ because the content mean is not small: it
# carries ~0.16 of the per-row content energy and only ~0.89 of it lies
# inside the centered top-128. Uncentered files carry the suffix "u".
PWI_MODE = os.environ.get("PWI_MODE", "centered")
assert PWI_MODE in ("centered", "uncentered"), \
    f"PWI_MODE={PWI_MODE} (centered|uncentered)"
SUF = "" if PWI_MODE == "centered" else "u"
WAIST = 512


def content_dims(bands_path):
    """The CONTENT dims, from the head's own band file. Plane p owns dims
    2p, 2p+1; a plane is content iff no wheel claims it."""
    bj = json.load(open(bands_path))
    assert bj["waist"] == WAIST and bj["n_planes"] == WAIST // 2, bands_path
    claimed = set()
    for w in bj["wheels"]:
        for p in w["planes"]:
            assert p not in claimed, f"plane {p} claimed twice"
            claimed.add(int(p))
    assert len(claimed) == int(bj["n_clocked"]), "band file self-inconsistent"
    cp = np.array(sorted(set(range(WAIST // 2)) - claimed), np.int64)
    assert len(cp) == int(bj["n_content"]), "content count != n_content"
    dims = np.empty(2 * len(cp), np.int64)
    dims[0::2] = 2 * cp
    dims[1::2] = 2 * cp + 1
    return cp, dims, bj


def main():
    cp, cdim, bj = content_dims(BANDS)
    print(f"[pwi] bands={BANDS} v{bj['version']}: {bj['n_clocked']} clocked "
          f"planes / {len(cp)} content planes -> {len(cdim)} content dims",
          flush=True)

    blocks, meta = [], []
    for name, path in sorted(FIX.items()):
        assert os.path.exists(path), f"{path} missing (the champion's dump)"
        z = np.load(path)
        u = z["u"]
        assert u.ndim == 4 and u.shape[-1] == WAIST, u.shape
        # ‖u‖ = 1 is the dump's own contract — assert it before trusting
        nrm = np.linalg.norm(u.reshape(-1, WAIST), axis=-1)
        assert float(np.abs(nrm - 1.0).max()) < 1e-3, \
            f"{path}: breaths_u is not unit ({float(np.abs(nrm-1).max()):.2e})"
        c = u.reshape(-1, WAIST)[:, cdim].astype(np.float64)
        blocks.append(c)
        meta.append((name, path, u.shape, c.shape[0]))
        print(f"[pwi]   {name:5s} {path} u{u.shape} -> {c.shape[0]} rows "
              f"x {c.shape[1]} content dims", flush=True)
    C = np.concatenate(blocks, 0)
    n, D = C.shape
    print(f"[pwi] pooled {n} rows x {D} dims (both fixtures, all breaths, "
          f"all slots)", flush=True)

    mu = C.mean(0)
    if PWI_MODE == "centered":
        Xc = C - mu
        M = (Xc.T @ Xc) / (n - 1)        # covariance: classical PCA
        what = "centered variance"
    else:
        M = (C.T @ C) / n                # second moment: the map's own metric
        what = "uncentered energy"
    ev, V = np.linalg.eigh(M)            # ascending
    ev = ev[::-1]
    V = V[:, ::-1]                       # (D, D) columns = PCs, descending
    tot = float(ev.sum())
    print(f"[pwi] mode={PWI_MODE} ({what}) total = {tot:.6g}; "
          f"content-block energy per row = "
          f"{float((C * C).sum(1).mean()):.6g} "
          f"(mean ‖mu‖^2 share = {float(mu @ mu):.6g})", flush=True)

    rows = []
    for d in DIMS:
        assert 0 < d <= D, f"d={d} out of range (D={D})"
        Vd = np.ascontiguousarray(V[:, :d].astype(np.float32))   # (D, d)
        kept = float(ev[:d].sum() / tot)
        # RECONSTRUCTION on the RAW (uncentered) content vectors — the map
        # the head actually runs is c -> c V V^T, no bias, no centering.
        P = Vd.astype(np.float64)
        R = (C @ P) @ P.T
        num = np.linalg.norm(R - C, axis=1)
        den = np.linalg.norm(C, axis=1)
        rel = float(np.mean(num / np.maximum(den, 1e-12)))
        rel_g = float(np.linalg.norm(R - C) / np.linalg.norm(C))
        # AFTER the head's norm-restoring rescale (the content block keeps
        # its own norm so ‖u‖ = 1 and the clock stays bitwise)
        Rn = R * (den / np.maximum(np.linalg.norm(R, axis=1), 1e-12))[:, None]
        rel_n = float(np.mean(np.linalg.norm(Rn - C, axis=1)
                              / np.maximum(den, 1e-12)))
        orth = float(np.abs(P.T @ P - np.eye(d)).max())
        assert orth < 1e-4, f"V_{d} is not orthonormal ({orth:.2e})"
        out = f".cache/polar_waist_init_d{d}{SUF}.npz"
        np.savez(out, W_down=Vd, W_up=np.ascontiguousarray(Vd.T),
                 mean=mu.astype(np.float32),
                 content_dims=cdim.astype(np.int64),
                 evr=(ev[:d] / tot).astype(np.float64),
                 var_kept=np.float64(kept), d=np.int64(d),
                 n_rows=np.int64(n), waist=np.int64(WAIST),
                 bands=np.array(BANDS), version=np.int64(1),
                 basis=np.array(PWI_MODE))
        sha = hashlib.sha256(open(out, "rb").read()).hexdigest()[:16]
        rows.append((d, kept, rel, rel_g, rel_n, out, sha,
                     2 * d * D))
        print(f"[pwi] d={d:4d}  var_kept={kept:.4f}  recon rel(mean/row)="
              f"{rel:.4f}  rel(global)={rel_g:.4f}  after-rescale={rel_n:.4f}"
              f"  params={2 * d * D}  -> {out} sha16={sha}", flush=True)

    print(f"[pwi] --- THE PCA TABLE ({PWI_MODE}, pooled) ---")
    print("[pwi]    d | var kept | recon rel err | after rescale | params")
    for d, kept, rel, rel_g, rel_n, out, sha, np_ in rows:
        print(f"[pwi] {d:4d} |  {kept:.4f}  |    {rel:.4f}     |   "
              f"{rel_n:.4f}      | {np_}")
    print(f"[pwi] fixtures: " + "; ".join(
        f"{m[0]}={m[3]} rows" for m in meta), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
