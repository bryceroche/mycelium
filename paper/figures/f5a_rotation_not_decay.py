"""F-5a — drift is rotation, not decay (three panels, real constellations).

Loads the ACTUAL gen-5 and gen-9b monitor centroids and reproduces the
latent audit's Procrustes exactly (same SVD, same cosines). Panel A:
raw overlay — the constellations look unrelated (mean cos ~0.59).
Panel B: after orthogonal Procrustes — near-identity (mean cos ~0.988).
Panel C: what did NOT align — per-kind residuals, shown honestly.
"""
import numpy as np
import matplotlib.pyplot as plt

import figstyle
from figstyle import PALETTE as C

figstyle.apply_style()

g5 = np.load("../../.cache/monitor_centroids_gen5.npz")
g9 = np.load("../../.cache/monitor_centroids_gen9b.npz")
kinds = sorted(set(g5.files) & set(g9.files))
A = np.stack([g5[k] for k in kinds])   # gen-5
B = np.stack([g9[k] for k in kinds])   # gen-9b

# — the audit's exact computation (latent_audit.py, probe A) —
U, _, Vt = np.linalg.svd(A.T @ B)
R = U @ Vt
A_rot = A @ R
raw_cos = np.array([a @ b for a, b in zip(A, B)])
rot_cos = np.array([a @ b for a, b in zip(A_rot, B)])
res = float(np.linalg.norm(A_rot - B) / np.linalg.norm(B))
print(f"[reproduce] raw {raw_cos.mean():.3f} aligned {rot_cos.mean():.3f} "
      f"residual {res:.3f}")
assert abs(rot_cos.mean() - 0.988) < 0.01, "does not reproduce the ledger"

def project2d(*sets):
    """One shared PCA basis per panel, fit on everything shown in it."""
    X = np.vstack(sets)
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return [(s - X.mean(0)) @ Vt[:2].T for s in sets]

fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))

def constellation(ax, P, Q, labels, title, subtitle):
    for (x5, y5), (x9, y9), k in zip(P, Q, labels):
        ax.plot([x5, x9], [y5, y9], color=C["faint"], lw=0.6, alpha=0.6)
    ax.scatter(P[:, 0], P[:, 1], marker="o", s=46, facecolors="none",
               edgecolors=C["gate"], linewidths=1.4, label="gen-5")
    ax.scatter(Q[:, 0], Q[:, 1], marker="s", s=40, color=C["ok"],
               alpha=0.85, label="gen-9b")
    for (x, y), k in zip(Q, labels):
        ax.annotate(k, (x, y), textcoords="offset points", xytext=(5, 5),
                    fontsize=6.8, color=C["ink"])
    ax.set_title(title, fontsize=9)
    ax.text(0.5, 0.03, subtitle, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=7, color=C["faint"], style="italic")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")

Pa, Qa = project2d(A, B)
constellation(axes[0], Pa, Qa, kinds,
              f"raw coordinates — mean cos {raw_cos.mean():.2f}",
              "\"the constellations look unrelated\"")
axes[0].legend(loc="lower right", fontsize=7)

Pb, Qb = project2d(A_rot, B)
constellation(axes[1], Pb, Qb, kinds,
              f"after orthogonal Procrustes — mean cos {rot_cos.mean():.3f}",
              "the shape was intact; the space had rotated")

ax = axes[2]
order = np.argsort(rot_cos)
ax.barh(range(len(kinds)), 1.0 - rot_cos[order], height=0.55,
        color=C["kill"], alpha=0.8)
ax.set_yticks(range(len(kinds)), [kinds[i] for i in order])
for i, j in enumerate(order):
    ax.text(1.0 - rot_cos[j] + 0.0012, i, f"{rot_cos[j]:.3f}",
            va="center", fontsize=7, family="DejaVu Sans Mono",
            color=C["ink"])
ax.set_xlabel("1 − aligned cosine (per kind)")
ax.set_title(f"what did not align — residual {res:.3f}", fontsize=9)
ax.text(0.97, 0.93, "rotation explains most, not all;\nthe rest is reported",
        transform=ax.transAxes, ha="right", va="top", fontsize=7,
        color=C["faint"], style="italic")
ax.grid(True, axis="x", alpha=0.5)

fig.tight_layout(rect=(0, 0.04, 1, 1))
meta = figstyle.stamp(fig, fixtures=["monitor_centroids_gen5/gen9b"])
figstyle.save(fig, "f5a_rotation_not_decay", meta=meta,
              fixtures=["monitor_centroids_gen5.npz", "monitor_centroids_gen9b.npz"])
