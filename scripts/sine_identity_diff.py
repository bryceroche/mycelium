"""sine_identity_diff.py -- pairwise diff of two scripts/depth_identity_probe.py MODE=dump
outputs (.cache/depth_identity_outputs_<TAG>.npz), for THE SINE BREATH's own identity bar:
"the pre-update dump diff sine vs depth4 = 0.0 and anti vs depth4 = 0.0". depth_identity_probe
itself only ever diffs its own hardcoded on/off pair; this script takes two arbitrary TAGs so
the SAME dump convention compares depth4 against sine4/anti4/seal directly -- at ReZero birth
every ALG_DEPTH block's output is EXACTLY zero regardless of width or schedule, so a plain
depth4 run and a sine/anti/seal run (same WARM_FROM, same rows, ALG_DEPTH=4 in both) must
produce bit-identical forward() outputs even though their dp{i}_* parameter SHAPES differ.
Usage: sine_identity_diff.py <TAG_A> <TAG_B>
"""
import sys
import numpy as np

A, B = sys.argv[1], sys.argv[2]
a = np.load(f".cache/depth_identity_outputs_{A}.npz")
b = np.load(f".cache/depth_identity_outputs_{B}.npz")
common = sorted(set(a.files) & set(b.files))
only_a = sorted(set(a.files) - set(b.files))
only_b = sorted(set(b.files) - set(a.files))
maxd = 0.0
worst = None
for k in common:
    d = float(np.abs(a[k].astype("float64") - b[k].astype("float64")).max()) if a[k].size else 0.0
    if d > maxd:
        maxd = d
        worst = k
print(f"[sine-identity-diff] {A} vs {B}: maxdiff={maxd:.3e} over {len(common)} common keys "
      f"({A} {len(a.files)} keys, {B} {len(b.files)} keys; only-in-{A}={only_a}; "
      f"only-in-{B}={only_b}; worst key={worst})")
