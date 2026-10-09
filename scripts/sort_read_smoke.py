"""sort_read_smoke.py -- THE SORTING ROOM form 2's read-path smoke (2026-10-09, zero-GPU, CPU-only;
the selfmatch_read_smoke.py convention). ALG_SORT/ALG_SORT_KEY need no new read-time PORT beyond
what build_params/forward already carry (the sorting room lives entirely inside forward(), between
the waist projection and _make_bank; the owner-key shuffle lives entirely inside breath_step, reading
ctx["tok_key"] forward() already built) -- this smoke proves that claim by loading the gate's own
trained checkpoint and running loop_val.py's REAL `read()` function, under ALG_JIT_READ=1, on the
first 2 rows of the fixture's own test split (sliced host-side from the already-precomputed .cache
states so no new precompute pass is needed): if the checkpoint's extra sort* keys loaded cleanly
(read()'s own `assert set(sd.keys()) == set(p.keys())`) and the JIT'd read ran to a finite fac-exact
without error, the road threads train-to-read with zero new wiring.
usage: ALG_SORT=<N> [ALG_SORT_KEY=1] ALG_JIT_READ=1 ALG_TEST=... ALG_TEST_NAME=testtiny64 \
       LV_CKPT=<ckpt> .venv/bin/python3 scripts/sort_read_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

assert int(os.environ.get("ALG_SORT", "0")) > 0, "ALG_SORT=<N> is required for this smoke"
assert int(os.environ.get("ALG_JIT_READ", "0")), "ALG_JIT_READ=1 is required for this smoke"

from phase1_algebra_head import load_alg, build_params
import loop_val

N_ROWS = 2
ckpt = os.environ["LV_CKPT"]

vs, vst, vtk, vg, vse = load_alg("test")
sl = slice(0, N_ROWS)
data2 = (vs[sl], vst[sl], vtk[sl], {k: v[sl] for k, v in vg.items()}, vse[sl])
print(f"[sort-read-smoke] sliced {N_ROWS} rows from the precomputed test split "
      f"(full split has {len(vs)} rows)")

p = build_params(0)
assert "sort0_wq" in p, "ALG_SORT did not register the sorting room's own params"
n_ok, n_tot = loop_val.read(ckpt, data=data2, p=p)
print(f"[sort-read-smoke] PASS: n_ok={n_ok} n_tot={n_tot} fac-exact={n_ok / max(n_tot, 1):.4f} "
      f"(checkpoint keys matched p's keys exactly -- read()'s own hard-error assert did not fire; "
      f"the sorting room's output ran under ALG_JIT_READ=1 with no new port)")
