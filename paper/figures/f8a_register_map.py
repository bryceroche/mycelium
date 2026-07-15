"""F-8a — the register map: where every population sits on one ruler.

Raw (uncorrected) gen-9b-mouth vintage throughout, stated on the plot:
the native fixture's distance distribution (banked states, numpy kNN),
the 100 census problems as real per-item dots (banked join), and the
foreign benchmark's banked band — with the wall annotated.
"""
import json
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
thr_raw = float(np.percentile(dn, 99))

census = json.load(open("../../.cache/census_mouth_join.json"))
dc = np.array([e["mouth_d"] for e in census])

# banked summary (ledger): MATH-500 raw band + native/foreign verdicts
M5_LO, M5_HI = 0.236, 0.273

fig, ax = plt.subplots(figsize=(8.6, 3.6))

# native distribution as a filled histogram
ax.hist(dn, bins=48, density=True, color=C["ok"], alpha=0.45,
        label=f"native fixture (n={len(dn)}, generated register)")
ax.axvline(thr_raw, color=C["kill"], lw=1.4, ls="--")
ax.text(thr_raw + 0.004, ax.get_ylim()[1] * 0.93,
        f"gate threshold (99th pct native)\nforeign refused 100.0% at 1% "
        f"native false-refusal;\nall 160 anchor false-certificates refused",
        fontsize=7.2, color=C["kill"], va="top")

# census harvest prose: real per-item dots
rng_y = np.full_like(dc, ax.get_ylim()[1] * 0.28)
jitter = (np.arange(len(dc)) % 7 - 3) * (ax.get_ylim()[1] * 0.012)
ax.scatter(dc, rng_y + jitter, s=10, color=C["wild"], alpha=0.7,
           label=f"harvest prose, census pool (n={len(dc)}, per-item)")

# MATH-500 band (banked summary)
ax.axvspan(M5_LO, M5_HI, ymin=0.55, ymax=0.72, color=C["alt"], alpha=0.35)
ax.text((M5_LO + M5_HI) / 2, ax.get_ylim()[1] * 0.75,
        "MATH-500 (banked band 0.236–0.273:\neverything a different forest)",
        ha="center", fontsize=7.2, color=C["alt"])

ax.set_xlabel("recognition-gate distance (raw kNN, gen-9b mouth vintage — "
              "corrected reads in Fig. 8-b)")
ax.set_ylabel("density")
ax.set_xlim(0, 0.5)
ax.legend(loc="center right", fontsize=7)
ax.grid(True, alpha=0.4)

fig.tight_layout(rect=(0, 0.05, 1, 1))
meta = figstyle.stamp(fig, fixtures=["dag8test states", "census_mouth_join",
                                     "mouth_gen9b bank"])
figstyle.save(fig, "f8a_register_map", meta=meta,
              fixtures=["dag8test-states", "census_mouth_join.json",
                        "recognition_mouth_gen9b.npz"])
