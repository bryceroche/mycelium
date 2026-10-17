"""photobooth_stage.py -- stamp a photobooth view split (2026-10-16). precompute_masked.py --split SRC --jsonl VIEW --out OUT
already asserted the view's tokmask = SRC's and copied SRC's npz. THE GATE here: tokenize(VIEW) reproduces SRC's tokmask AND
sentence indices, and build_gold(VIEW) reproduces every SRC g_* array BIT-IDENTICALLY (the views change names only); then
the OUT npz is re-saved stamped with mix_sha16(VIEW), so the SHA-FENCE binds the split to the view's content.
usage: (the family env) photobooth_stage.py SRC_SPLIT VIEW.jsonl OUT_SPLIT"""
import sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np  # noqa: E402
import phase1_algebra_head as PH  # noqa: E402
from mycelium.era import mix_sha16  # noqa: E402

src, view, out = sys.argv[1:4]
z = np.load(PH.STATES_NPZ.format(split=src)); arr = {k: z[k] for k in z.files}
s1, _, m1, o1 = PH.tokenize(view)
assert np.array_equal(m1.astype(np.uint8), arr["tokmask"].astype(np.uint8)), "GATE FAIL: tokmask"
g1 = PH.build_gold(s1, o1)
banked = [k for k in arr if k.startswith("g_")]
assert set(k[2:] for k in banked) <= set(g1), "GATE FAIL: build_gold lacks a banked key"
for k in banked:
    assert np.array_equal(g1[k[2:]], arr[k]), f"GATE FAIL: {k} differs"
zo = np.load(PH.STATES_NPZ.format(split=out)); assert np.array_equal(zo["tokmask"], arr["tokmask"])
assert np.array_equal(zo["sent"], arr["sent"]), "the copied npz's sent differs"
sv = np.stack([PH.sent_indices(r["text"], o1[i], m1[i]) for i, r in enumerate(s1)])
assert np.array_equal(sv.astype(arr["sent"].dtype), arr["sent"]), "GATE FAIL: the view's sentence indices differ"
arr["mix_sha"] = mix_sha16(view)
np.savez(PH.STATES_NPZ.format(split=out), **arr)
print(f"[pb-stage] GATE PASS {out}: tokmask + {len(banked)} gold arrays bit-identical to {src}; stamped {arr['mix_sha']}", flush=True)
