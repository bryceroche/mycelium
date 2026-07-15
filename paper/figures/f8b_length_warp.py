"""F-8b — the length warp, before and after (real per-item native reads).

Reproduces the recalibration's native fit exactly from banked artifacts
(dag8test pooled states + the gen-9b mouth bank; numpy only, no GPU):
distance ~ a + b/length, raw within-register correlation r = -0.825,
residual r = -0.024. Asserts reproduction before saving.
"""
import numpy as np
import matplotlib.pyplot as plt

import figstyle
from figstyle import PALETTE as C

figstyle.apply_style()

import os, sys
os.environ.setdefault("ALG2", "1")
os.environ.setdefault("ALG_FTYPES", "6")
os.environ.setdefault("ALG_DUP", "1")
sys.path.insert(0, "../../"); sys.path.insert(0, "../../scripts")
from phase1_algebra_head import STATES_NPZ, STATES_NPY  # noqa: E402

bank = np.load("../../.cache/recognition_mouth_gen9b.npz")["bank"]
z = np.load(STATES_NPZ.format(split="dag8test").replace(".cache", "../../.cache"))
st = np.load(STATES_NPY.format(split="dag8test").replace(".cache", "../../.cache"),
             mmap_mode="r")
tk = z["tokmask"].astype(np.float32)
V = np.zeros((st.shape[0], st.shape[-1]), np.float32)
for i in range(st.shape[0]):
    m = tk[i][:, None]
    V[i] = (np.asarray(st[i], np.float32) * m).sum(0) / max(m.sum(), 1)
V /= np.linalg.norm(V, axis=1, keepdims=True)
dn = np.sort(1.0 - V @ bank.T, axis=1)[:, :8].mean(1)
Ln = tk.sum(1)

X = np.stack([np.ones_like(Ln), 1.0 / Ln], 1)
coef, *_ = np.linalg.lstsq(X, dn, rcond=None)
res = dn - X @ coef
r_raw = np.corrcoef(Ln, dn)[0, 1]
r_res = np.corrcoef(Ln, res)[0, 1]
thr = float(np.percentile(res, 99))
print(f"[reproduce] raw r={r_raw:+.3f} residual r={r_res:+.3f} thr={thr:.4f}")
assert abs(r_raw - (-0.825)) < 0.02 and abs(r_res) < 0.05, "does not reproduce"

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharex=True)

ax = axes[0]
ax.scatter(Ln, dn, s=8, alpha=0.35, color=C["gate"], edgecolors="none")
xs = np.linspace(Ln.min(), Ln.max(), 200)
ax.plot(xs, coef[0] + coef[1] / xs, color=C["kill"], lw=1.6,
        label=f"fit  d = {coef[0]:.4f} + {coef[1]:.2f}/len")
ax.set_title(f"raw distance — r = {r_raw:+.3f} with length", fontsize=9)
ax.set_xlabel("problem length (trunk tokens)")
ax.set_ylabel("recognition-gate distance (kNN)")
ax.legend(loc="upper right")
ax.text(0.97, 0.72, "\"is this distance,\nor is this n?\"",
        transform=ax.transAxes, ha="right", fontsize=7.2,
        color=C["faint"], style="italic")
ax.grid(True, alpha=0.4)

ax = axes[1]
ax.scatter(Ln, res, s=8, alpha=0.35, color=C["ok"], edgecolors="none")
ax.axhline(thr, color=C["kill"], lw=1.2, ls="--")
ax.text(Ln.max(), thr + 0.001, f"deployment threshold {thr:.4f}\n(99th pct native residual)",
        ha="right", va="bottom", fontsize=7, color=C["kill"])
ax.axhline(0, color=C["faint"], lw=0.7)
ax.set_title(f"length-controlled residual — r = {r_res:+.3f}", fontsize=9)
ax.set_xlabel("problem length (trunk tokens)")
ax.set_ylabel("residual distance")
ax.text(0.97, 0.06, "the ruler, straightened:\nforeign refusal held at 100%\non the corrected read (§8.3)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2,
        color=C["ink"], style="italic")
ax.grid(True, alpha=0.4)

fig.tight_layout(rect=(0, 0.05, 1, 1))
meta = figstyle.stamp(fig, fixtures=["dag8test states", "mouth_gen9b bank"])
figstyle.save(fig, "f8b_length_warp", meta=meta,
              fixtures=["dag8test-states", "recognition_mouth_gen9b.npz"])
