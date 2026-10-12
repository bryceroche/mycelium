"""waist_skip_read_identity_smoke.py -- THE WAIST SKIP, FORM A's read-path identity smoke
(2026-10-09, the sort_read_identity_smoke.py precedent): proves the flag-off path and the g=0
path AGREE on read -- the instrument a road that modifies state needs. Build params warm from
balV242 with waist_skip_theta at FRESH init only (no training step at all -- the UNTRAINED
checkpoint .cache/sharp_untrained_waistskip_{role8,ginf}.safetensors, each saved by a
`STEPS=0 --train` run), run loop_val.py's REAL read() under ALG_JIT_READ=1 (a DIFFERENT
captured graph from training) on the SAME 2 rows for both, print n_ok/n_tot, and assert they
match exactly:
  - role8: ALG_WAIST_SKIP=0 (waist_skip_theta never allocated -- the flag-off path)
  - ginf:  ALG_WAIST_SKIP=1 ALG_WAIST_SKIP_INIT=-10000 (sigmoid(-10000) underflows to EXACTLY
           0.0 in float32 -- exp(10000) overflows to inf, so 1/(1+inf) = 0.0, not merely small
           -- the g=0 path)
Run as TWO SEPARATE PROCESSES (ALG_WAIST_SKIP is a module-level constant read once at import):
  WS_IDTAG=role8 ALG_WAIST_SKIP=0 .venv/bin/python3 scripts/waist_skip_read_identity_smoke.py
  WS_IDTAG=ginf ALG_WAIST_SKIP=1 ALG_WAIST_SKIP_INIT=-10000 \
      .venv/bin/python3 scripts/waist_skip_read_identity_smoke.py
Each prints n_ok/n_tot to .cache/waistskip_read_identity_{TAG}.txt; compare the two files' numbers.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["ALG_JIT_READ"] = "1"
N_ROWS = 2
TAG = os.environ.get("WS_IDTAG", "role8")
CKPT = f".cache/sharp_untrained_waistskip_{TAG}.safetensors"

from phase1_algebra_head import load_alg, build_params
import loop_val

vs, vst, vtk, vg, vse = load_alg("test")
sl = slice(0, N_ROWS)
data2 = (vs[sl], vst[sl], vtk[sl], {k: v[sl] for k, v in vg.items()}, vse[sl])
print(f"[waistskip-read-identity] TAG={TAG} sliced {N_ROWS} rows from the precomputed test split "
      f"(full split has {len(vs)} rows); ckpt={CKPT}")

p = build_params(0)
if TAG == "ginf":
    assert "waist_skip_theta" in p, "ALG_WAIST_SKIP=1 did not register waist_skip_theta"
    _g0 = p["waist_skip_theta"].sigmoid().numpy()   # tinygrad's OWN float32 sigmoid -- the
                                                      # exact quantity the forward pass uses
    assert float(_g0.max()) == 0.0, \
        f"ALG_WAIST_SKIP_INIT must drive g to EXACTLY 0.0 (tinygrad float32) for this smoke " \
        f"(saw max g={_g0.max():.3e})"
else:
    assert "waist_skip_theta" not in p, "ALG_WAIST_SKIP=0 must never allocate waist_skip_theta"

n_ok, n_tot = loop_val.read(CKPT, data=data2, p=p)
print(f"[waistskip-read-identity] TAG={TAG} n_ok={n_ok} n_tot={n_tot} fac-exact={n_ok / max(n_tot, 1):.4f}")
with open(f".cache/waistskip_read_identity_{TAG}.txt", "w") as f:
    f.write(f"{n_ok} {n_tot}\n")
print(f"[waistskip-read-identity] TAG={TAG} -> .cache/waistskip_read_identity_{TAG}.txt")
