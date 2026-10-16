"""reading_order_restage.py -- stage the READING-ORDER splits (2026-10-16; word given for the run). The trunk states are a
function of the TEXT only and the reading-order fixtures keep every row's text and row order, so the BP states are
REUSED (a symlink); only the gold arrays are rebuilt (build_gold on the _RO jsonl) and the npz is stamped with the _RO
mix sha (the SHA-FENCE then binds the split to its content).
THE GATE (before anything is written): build_gold on the ORIGINAL jsonl under this process's env must reproduce the
banked {src}BP npz's g_* arrays BIT-IDENTICALLY (every key) -- proof the env here is the env the arrays were built under.
usage: (the run's family env) reading_order_restage.py SRC_SPLIT ORIG.jsonl RO.jsonl OUT_SPLIT
"""
import os
import sys

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np  # noqa: E402

import phase1_algebra_head as PH  # noqa: E402
from mycelium.era import mix_sha16  # noqa: E402

src, orig, ro, out = sys.argv[1:5]
z = np.load(PH.STATES_NPZ.format(split=src))
if "mix_sha" in z.files:
    assert str(z["mix_sha"]) == mix_sha16(orig), f"{src} is not staged from {orig}"
# THE GATE
s0, _, m0, o0 = PH.tokenize(orig)
g0 = PH.build_gold(s0, o0)
banked = {k[2:]: z[k] for k in z.files if k.startswith("g_")}
# the trainer's conditional losses key on PRESENCE ("aspan" in g ...): the staged split must carry EXACTLY the banked key
# set (today's build_gold emits later-added keys the banked split never had; dropping them keeps the run's losses = SRK3's)
assert set(banked) <= set(g0), f"banked keys build_gold no longer emits: {sorted(set(banked) - set(g0))}"
_extra = sorted(set(g0) - set(banked))
if _extra:
    print(f"[restage] keys build_gold emits that {src} never carried (DROPPED to mirror it): {_extra}", flush=True)
g0 = {k: g0[k] for k in banked}
for k in g0:
    assert g0[k].shape == banked[k].shape and np.array_equal(g0[k], banked[k]), f"GATE FAIL: g_{k} does not reproduce"
assert np.array_equal(m0.astype(np.uint8), z["tokmask"].astype(np.uint8)), "GATE FAIL: tokmask"
print(f"[restage] GATE PASS: {len(g0)} gold arrays of {src} reproduce bit-identically from {orig}", flush=True)
# THE READING-ORDER SPLIT
s1, _, m1, o1 = PH.tokenize(ro)
assert len(s1) == len(s0) and all(a["text"] == b["text"] for a, b in zip(s0, s1)), "RO rows/texts differ from the original"
assert np.array_equal(m1, m0), "RO tokmask differs"
g1 = PH.build_gold(s1, o1)
g1 = {k: g1[k] for k in banked}
changed = {k: int((g1[k] != g0[k]).reshape(len(s0), -1).any(1).sum()) for k in g0}
dst_npz = PH.STATES_NPZ.format(split=out); dst_npy = PH.STATES_NPY.format(split=out)
assert not os.path.exists(dst_npz) and not os.path.exists(dst_npy), f"{out} exists: never overwrite a staged split"
np.savez(dst_npz, tokmask=z["tokmask"], sent=z["sent"], mix_sha=mix_sha16(ro), **{f"g_{k}": v for k, v in g1.items()},
         **{k: z[k] for k in z.files if not k.startswith("g_") and k not in ("tokmask", "sent", "mix_sha")})
os.symlink(os.path.abspath(PH.STATES_NPY.format(split=src)), dst_npy)
print(f"[restage] {out}: gold rebuilt from {ro} (rows whose array changed: "
      + ", ".join(f"{k} {v}" for k, v in sorted(changed.items()) if v) + f"); states -> symlink {src}", flush=True)
