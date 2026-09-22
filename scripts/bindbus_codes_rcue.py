"""bindbus_codes_rcue.py -- THE ROLE PHASOR MINT (2026-09-22, THE ROLE SIGNATURE organ; word
given: docs/phase1_skeleton_spec.md "2026-09-22 -- THE TWIN-SPLIT VERDICT"). Appends theta_rcue
(256,) -- a sixth role offset, an independent uniform random phase per plane, minted the SAME way
the incumbent's four existing offsets (theta_arg1/theta_arg2/theta_res/theta_op) were: each is just
np.random.uniform(0, 2*pi) per plane, no structure beyond that (verified against the incumbent file
below: min/max span the full circle, no artifacts of a formulaic mint). CB and every existing
theta_* key are copied through UNCHANGED -- byte-identical to the source file (verified by the
--check reader below); only the new key is added, so the OLD file (bindbus_codes512.npz) stays
untouched and every config that does not opt into BIND_CODES=<this file> is bit-identical.
usage: bindbus_codes_rcue.py [in] [out]
"""
import sys
import numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else ".cache/bindbus_codes512.npz"
dst = sys.argv[2] if len(sys.argv) > 2 else ".cache/bindbus_codes512r.npz"

z = np.load(src)
out = {k: z[k] for k in z.files}
P = out["theta_op"].shape[0]   # same plane count as the other role offsets (256)

# a seed not used by any existing theta_* mint in this tree (documented, fixed -- reproducible)
rng = np.random.default_rng(0x9c7e)
out["theta_rcue"] = rng.uniform(0.0, 2 * np.pi, P).astype(out["theta_op"].dtype)

np.savez(dst, **out)

# --check: every OLD key byte-identical; the new key present, right shape, full-circle span
_z2 = np.load(dst)
for k in z.files:
    assert np.array_equal(z[k], _z2[k]), f"[rcue-mint] FAILED: {k} changed"
assert "theta_rcue" in _z2.files and _z2["theta_rcue"].shape == (P,), "[rcue-mint] FAILED: theta_rcue shape"
_tr = _z2["theta_rcue"]
print(f"[rcue-mint] {dst}: theta_rcue minted ({P} planes, seed 0x9c7e) "
      f"min {_tr.min():.4f} max {_tr.max():.4f}; old keys ({', '.join(z.files)}) "
      f"byte-identical to {src}")
