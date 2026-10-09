"""paircmp_read_smoke.py -- THE PAIRWISE COMPARATOR ROAD's read-path smoke (2026-10-09, zero-GPU,
CPU-only; the selfmatch_read_smoke.py convention). ALG_PAIRCMP needs no new read-time port
(in-head: it consumes s/vst, both already present inside _heads_of at every call site, train or
read) -- this smoke proves that claim by loading the gate's own trained paircmp checkpoint and
running loop_val.py's REAL `read()` function, under ALG_JIT_READ=1, on just the first 2 rows of
the champion fixture's own test split (sliced host-side from the already-precomputed .cache
states so no new precompute pass is needed): if the checkpoint's extra pc_w1/pc_b1/pc_w2/pc_b2
keys loaded cleanly (read()'s own `assert set(sd.keys()) == set(p.keys())`) and the JIT'd read
ran to a finite fac-exact without error, the road threads train-to-read with zero new wiring.
usage: ALG_PAIRCMP=1 ALG_JIT_READ=1 ALG_TEST=... ALG_TEST_NAME=testtiny64 LV_CKPT=<ckpt>
       .venv/bin/python3 scripts/paircmp_read_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

assert int(os.environ.get("ALG_PAIRCMP", "0")), "ALG_PAIRCMP=1 is required for this smoke"
assert int(os.environ.get("ALG_JIT_READ", "0")), "ALG_JIT_READ=1 is required for this smoke"

from phase1_algebra_head import load_alg, build_params
import loop_val

N_ROWS = 2
ckpt = os.environ["LV_CKPT"]

vs, vst, vtk, vg, vse = load_alg("test")
sl = slice(0, N_ROWS)
data2 = (vs[sl], vst[sl], vtk[sl], {k: v[sl] for k, v in vg.items()}, vse[sl])
print(f"[paircmp-read-smoke] sliced {N_ROWS} rows from the precomputed test split "
      f"(full split has {len(vs)} rows)")

p = build_params(0)
for k in ("pc_w1", "pc_b1", "pc_w2", "pc_b2"):
    assert k in p, f"ALG_PAIRCMP did not register {k}"
n_ok, n_tot = loop_val.read(ckpt, data=data2, p=p)
print(f"[paircmp-read-smoke] PASS: n_ok={n_ok} n_tot={n_tot} fac-exact={n_ok / max(n_tot, 1):.4f} "
      f"(checkpoint keys matched p's keys exactly -- read()'s own hard-error assert did not "
      f"fire; the predicted pc_w1/pc_b1/pc_w2/pc_b2-driven res bias ran under ALG_JIT_READ=1 "
      f"with no new port)")
