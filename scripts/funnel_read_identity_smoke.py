"""funnel_read_identity_smoke.py -- THE FUNNEL GATE's read-path identity smoke (2026-10-10, the
waist_skip_read_identity_smoke.py precedent): proves the flag-off path and the
g_skip=0/rank_r=1 path AGREE on read -- the instrument a road that modifies state needs. Build
params warm from balV242 with funnel_skip_theta/funnel_rank_theta at FRESH init only (no
training step at all -- the UNTRAINED checkpoint
.cache/sharp_untrained_funnel_{role8,ginf}.safetensors, each saved by a `STEPS=0 --train` run),
run loop_val.py's REAL read() under ALG_JIT_READ=1 (a DIFFERENT captured graph from training)
on the SAME 2 rows for both, print n_ok/n_tot, and assert they match exactly:
  - role8: ALG_FUNNEL=0 (funnel_skip_theta/funnel_rank_theta never allocated -- the flag-off
           path, THE WAIST SKIP gate's own role8 chassis unchanged)
  - ginf:  ALG_FUNNEL=1 ALG_FUNNEL_INIT=-10000 (sigmoid(-10000) underflows to EXACTLY 0.0 in
           float32 -- exp(10000) overflows to inf, so 1/(1+inf) = 0.0, not merely small --
           g_skip = 0.0 EXACTLY (the bypass fully closed, c' = P c unmodified) AND rank r =
           1 - 0.0 = 1.0 EXACTLY (the rank gate fully open, lat unmodified): the funnel path
           reduces exactly to the plain waist's P c computation)
Run as TWO SEPARATE PROCESSES (ALG_FUNNEL is a module-level constant read once at import):
  FG_IDTAG=role8 ALG_FUNNEL=0 .venv/bin/python3 scripts/funnel_read_identity_smoke.py
  FG_IDTAG=ginf ALG_FUNNEL=1 ALG_FUNNEL_INIT=-10000 \
      .venv/bin/python3 scripts/funnel_read_identity_smoke.py
Each prints n_ok/n_tot to .cache/funnel_read_identity_{TAG}.txt; compare the two files' numbers.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["ALG_JIT_READ"] = "1"
N_ROWS = 2
TAG = os.environ.get("FG_IDTAG", "role8")
CKPT = f".cache/sharp_untrained_funnel_{TAG}.safetensors"

from phase1_algebra_head import load_alg, build_params
import loop_val

vs, vst, vtk, vg, vse = load_alg("test")
sl = slice(0, N_ROWS)
data2 = (vs[sl], vst[sl], vtk[sl], {k: v[sl] for k, v in vg.items()}, vse[sl])
print(f"[funnel-read-identity] TAG={TAG} sliced {N_ROWS} rows from the precomputed test split "
      f"(full split has {len(vs)} rows); ckpt={CKPT}")

p = build_params(0)
if TAG == "ginf":
    assert "funnel_skip_theta" in p and "funnel_rank_theta" in p, \
        "ALG_FUNNEL=1 did not register funnel_skip_theta/funnel_rank_theta"
    _gs0 = p["funnel_skip_theta"].sigmoid().numpy()        # tinygrad's OWN float32 sigmoid --
    _gr0 = p["funnel_rank_theta"].sigmoid().numpy()        # the exact quantities the forward pass uses
    assert float(_gs0.max()) == 0.0, \
        f"ALG_FUNNEL_INIT must drive g_skip to EXACTLY 0.0 (tinygrad float32) for this smoke (saw max={_gs0.max():.3e})"
    assert float(_gr0.max()) == 0.0, \
        f"ALG_FUNNEL_INIT must drive the rank gate's raw sigmoid to EXACTLY 0.0 (tinygrad float32), " \
        f"i.e. r = 1 - 0.0 = 1.0 EXACTLY (saw max={_gr0.max():.3e})"
else:
    assert "funnel_skip_theta" not in p and "funnel_rank_theta" not in p, \
        "ALG_FUNNEL=0 must never allocate funnel_skip_theta/funnel_rank_theta"

n_ok, n_tot = loop_val.read(CKPT, data=data2, p=p)
print(f"[funnel-read-identity] TAG={TAG} n_ok={n_ok} n_tot={n_tot} fac-exact={n_ok / max(n_tot, 1):.4f}")
with open(f".cache/funnel_read_identity_{TAG}.txt", "w") as f:
    f.write(f"{n_ok} {n_tot}\n")
print(f"[funnel-read-identity] TAG={TAG} -> .cache/funnel_read_identity_{TAG}.txt")
