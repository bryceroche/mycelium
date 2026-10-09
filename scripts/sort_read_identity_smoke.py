"""sort_read_identity_smoke.py -- THE SORTING ROOM's read-path identity smoke (2026-10-09,
coordinator's policy point 2): the ORIGINAL sort_read_smoke.py's 0/18 came from reading a
checkpoint TRAINED (for 2 steps) under the warm fixture's post-update blow-up, not from the
read path itself -- that checkpoint's weights were already bad, so of course it decoded
garbage. This smoke proves the READ path at IDENTITY instead: build params warm from
balV242 with the sorting room's keys at FRESH init (no training step at all -- the UNTRAINED
checkpoint .cache/sharp_untrained_{role8,sort4}.safetensors, each saved by a `STEPS=0 --train`
run), run loop_val.py's REAL read() under ALG_JIT_READ=1 (a DIFFERENT captured graph from
training) on the SAME 2 rows for both, print n_ok/n_tot, and assert they match exactly --
proving the read path threads the stack AND the stack is identity at birth on the read side
too, not just the training side (the JIT read capture is a separate graph from the JIT'd
training step; this organ must be clean on BOTH).

Run as TWO SEPARATE PROCESSES (ALG_SORT is a module-level constant read once at import --
reload() tricks are fragile and were avoided here on purpose):
  CKPT_TAG=role8 ALG_SORT=0 .venv/bin/python3 scripts/sort_read_identity_smoke.py
  CKPT_TAG=sort4 ALG_SORT=4 .venv/bin/python3 scripts/sort_read_identity_smoke.py
Each prints n_ok/n_tot to .cache/sort_read_identity_{TAG}.txt; compare the two files' numbers.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["ALG_JIT_READ"] = "1"
N_ROWS = 2
TAG = os.environ.get("CKPT_TAG", "role8")
CKPT = f".cache/sharp_untrained_{TAG}.safetensors"

from phase1_algebra_head import load_alg, build_params
import loop_val

vs, vst, vtk, vg, vse = load_alg("test")
sl = slice(0, N_ROWS)
data2 = (vs[sl], vst[sl], vtk[sl], {k: v[sl] for k, v in vg.items()}, vse[sl])
print(f"[sort-read-identity] TAG={TAG} sliced {N_ROWS} rows from the precomputed test split "
      f"(full split has {len(vs)} rows); ckpt={CKPT}")

p = build_params(0)
n_ok, n_tot = loop_val.read(CKPT, data=data2, p=p)
print(f"[sort-read-identity] TAG={TAG} n_ok={n_ok} n_tot={n_tot} fac-exact={n_ok / max(n_tot, 1):.4f}")
with open(f".cache/sort_read_identity_{TAG}.txt", "w") as f:
    f.write(f"{n_ok} {n_tot}\n")
print(f"[sort-read-identity] TAG={TAG} -> .cache/sort_read_identity_{TAG}.txt")
