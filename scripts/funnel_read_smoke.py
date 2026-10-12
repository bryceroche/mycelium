"""funnel_read_smoke.py -- THE FUNNEL GATE's read-path smoke (2026-10-10, zero-GPU, CPU-only).
ALG_FUNNEL needs no new read-time port (in-head: _polar_waist consumes p, state and kb, all
already present inside breath_step at every call site, train or read) -- this smoke proves that
claim by loading the gate's own trained funnel checkpoint and running loop_val.py's REAL
`read()` function, under ALG_JIT_READ=1, on just the first 2 rows of the champion fixture's own
test split (sliced host-side from the already-precomputed .cache states so no new precompute
pass is needed): if the checkpoint's extra funnel_skip_theta/funnel_rank_theta keys loaded
cleanly (read()'s own `assert set(sd.keys()) == set(p.keys())`) and the JIT'd read ran to a
finite fac-exact without error, the road threads train-to-read with zero new wiring, exactly as
claimed (the waist_skip_read_smoke.py precedent).
usage: ALG_FUNNEL=1 ALG_JIT_READ=1 ALG_TEST=... ALG_TEST_NAME=testtiny64 LV_CKPT=<ckpt>
       .venv/bin/python3 scripts/funnel_read_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

assert int(os.environ.get("ALG_FUNNEL", "0")), "ALG_FUNNEL=1 is required for this smoke"
assert int(os.environ.get("ALG_JIT_READ", "0")), "ALG_JIT_READ=1 is required for this smoke"

from phase1_algebra_head import load_alg, build_params
import loop_val

N_ROWS = 2
ckpt = os.environ["LV_CKPT"]

vs, vst, vtk, vg, vse = load_alg("test")
sl = slice(0, N_ROWS)
data2 = (vs[sl], vst[sl], vtk[sl], {k: v[sl] for k, v in vg.items()}, vse[sl])
print(f"[funnel-read-smoke] sliced {N_ROWS} rows from the precomputed test split "
      f"(full split has {len(vs)} rows)")

p = build_params(0)
assert "funnel_skip_theta" in p and "funnel_rank_theta" in p, \
    "ALG_FUNNEL did not register funnel_skip_theta/funnel_rank_theta"
n_ok, n_tot = loop_val.read(ckpt, data=data2, p=p)
print(f"[funnel-read-smoke] PASS: n_ok={n_ok} n_tot={n_tot} fac-exact={n_ok / max(n_tot, 1):.4f} "
      f"(checkpoint keys matched p's keys exactly -- read()'s own hard-error assert did not fire; "
      f"the predicted funnel-gated content waist ran under ALG_JIT_READ=1 with no new port)")
