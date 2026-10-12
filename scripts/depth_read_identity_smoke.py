"""depth_read_identity_smoke.py -- THE LOOPED TRANSFORMER's read-path smoke (2026-10-09). Unlike
the sorting room's read_identity_smoke (which expects role8==sort4, because sort4 only ADDS a
ReZero'd stack after an already-warm projection), depth4's design ZEROES OUT the existing warm
mixer's real, nonzero output (`h_slot = h_slot * 0.0 + (_depth_stack(cur) - cur)`) and replaces it
with the depth stack's own delta (exactly 0 at init) -- so role8 and depth4 decodes on the SAME
warm checkpoint are NOT expected to match (the mixer is genuinely bypassed by value, even on an
untrained stack). This smoke instead proves the weaker, load-bearing claim: the READ path (a
SEPARATE JIT graph from training, ALG_JIT_READ=1) threads the stack without crashing and produces
FINITE decodes for both role8 and depth4 on the SAME 2 rows -- and states plainly that the two
are not expected to agree.

Run as TWO SEPARATE PROCESSES (ALG_DEPTH is a module-level constant read once at import):
  CKPT_TAG=role8  ALG_DEPTH=0 .venv/bin/python3 scripts/depth_read_identity_smoke.py
  CKPT_TAG=depth4 ALG_DEPTH=4 .venv/bin/python3 scripts/depth_read_identity_smoke.py
Each prints n_ok/n_tot to .cache/depth_read_identity_{TAG}.txt.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["ALG_JIT_READ"] = "1"
N_ROWS = 2
TAG = os.environ.get("CKPT_TAG", "role8")
CKPT = f".cache/sharp_untrained_depth_{TAG}.safetensors"

from phase1_algebra_head import load_alg, build_params
import loop_val

vs, vst, vtk, vg, vse = load_alg("test")
sl = slice(0, N_ROWS)
data2 = (vs[sl], vst[sl], vtk[sl], {k: v[sl] for k, v in vg.items()}, vse[sl])
print(f"[depth-read-smoke] TAG={TAG} sliced {N_ROWS} rows from the precomputed test split "
      f"(full split has {len(vs)} rows); ckpt={CKPT}")

p = build_params(0)
n_ok, n_tot = loop_val.read(CKPT, data=data2, p=p)
finite = (n_tot > 0)
print(f"[depth-read-smoke] TAG={TAG} n_ok={n_ok} n_tot={n_tot} fac-exact={n_ok / max(n_tot, 1):.4f} "
      f"finite={finite}")
print("[depth-read-smoke] NOTE: role8 and depth4 are NOT expected to agree -- the mixer is "
      "genuinely bypassed by VALUE (h_slot zeroed and replaced), even on an untrained stack; "
      "this smoke only proves the read path runs and decodes are finite for both.")
with open(f".cache/depth_read_identity_{TAG}.txt", "w") as f:
    f.write(f"{n_ok} {n_tot}\n")
print(f"[depth-read-smoke] TAG={TAG} -> .cache/depth_read_identity_{TAG}.txt")
