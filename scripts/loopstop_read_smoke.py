"""loopstop_read_smoke.py -- THE LOOP STOP's read-path smoke (2026-10-11, zero/CPU-cheap). ALG_LOOP_STOP
needs no new read-time port -- it only changes how many breaths forward()'s own loop runs before
heads_of() reads whatever `breaths[-1]` happens to be at that point, and every downstream consumer
(loop_val's read(), chain_acc, membrane_rack) already calls forward() without an explicit stop_after,
so the env override picked up inside forward() is the only wiring needed. This smoke proves that claim
two ways in one process: load a (tiny, CPU-trained) checkpoint and run loop_val.py's REAL `read()`
function, under ALG_JIT_READ=1, on just the first N_ROWS rows of the fixture's own test split (sliced
host-side from the already-precomputed .cache states -- no new precompute pass), once under whatever
ALG_LOOP_STOP the environment already has (the training-time truncation), and once with
ALG_LOOP_STOP_READ set to override it at just this call (the K3-vs-K7 exposure read) -- toggling the env
var between the two calls in-process, the same pattern _quick_val's own ALG_JIT_READ toggle already uses
inside phase1_algebra_head.py.

usage: ALG_LOOP_STOP=3 ALG_JIT_READ=1 ALG_TEST=... ALG_TEST_NAME=... LV_CKPT=<ckpt> \
       LSR_READ_OVERRIDE=7 .venv/bin/python3 scripts/loopstop_read_smoke.py
  (LSR_READ_OVERRIDE unset/empty = only the first read runs, reporting under the ambient
  ALG_LOOP_STOP/ALG_LOOP_STOP_READ as given)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

assert int(os.environ.get("ALG_JIT_READ", "0")), "ALG_JIT_READ=1 is required for this smoke"

from phase1_algebra_head import load_alg, build_params
import loop_val

N_ROWS = 2
ckpt = os.environ["LV_CKPT"]

vs, vst, vtk, vg, vse = load_alg("test")
sl = slice(0, N_ROWS)
data2 = (vs[sl], vst[sl], vtk[sl], {k: v[sl] for k, v in vg.items()}, vse[sl])
print(f"[loopstop-read-smoke] sliced {N_ROWS} rows from the precomputed test split "
      f"(full split has {len(vs)} rows)")

p = build_params(0)

_ls = os.environ.get("ALG_LOOP_STOP", "0")
_lsr = os.environ.get("ALG_LOOP_STOP_READ", "")
n_ok, n_tot = loop_val.read(ckpt, data=data2, p=p)
print(f"[loopstop-read-smoke] PASS pass1 (ALG_LOOP_STOP={_ls} ALG_LOOP_STOP_READ={_lsr or '(unset)'}): "
      f"n_ok={n_ok} n_tot={n_tot} fac-exact={n_ok / max(n_tot, 1):.4f}")

_override = os.environ.get("LSR_READ_OVERRIDE", "")
if _override:
    _prev = os.environ.get("ALG_LOOP_STOP_READ")
    os.environ["ALG_LOOP_STOP_READ"] = _override
    try:
        n_ok2, n_tot2 = loop_val.read(ckpt, data=data2, p=p)
        print(f"[loopstop-read-smoke] PASS pass2 (ALG_LOOP_STOP_READ={_override} override): "
              f"n_ok={n_ok2} n_tot={n_tot2} fac-exact={n_ok2 / max(n_tot2, 1):.4f}")
    finally:
        if _prev is None:
            os.environ.pop("ALG_LOOP_STOP_READ", None)
        else:
            os.environ["ALG_LOOP_STOP_READ"] = _prev
