"""depth_knob_census.py -- THE LOOPED TRANSFORMER's knob census (the pre/post knob law: census
every organ's injection pre- and post-gain per breath before any claim). For each loop breath and
each of the ALG_DEPTH blocks, prints the std of the block's net residual contribution
(block_out - block_in) against the std of the state it met (block_in) -- the SAME ratio grammar
the port census uses for every other organ, specialized to this organ's own (kb, depth{i}_in /
depth{i}_out) tags. BAR (the arm's own registration): every block >= 5% of the state (std ratio)
at the LAST breath, else the block is inert and this census says so.
Env: PC_CKPT (required), PC_N (rows, default 32), PC_TEST/PC_TEST_NAME (default train split).
"""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

assert int(os.environ.get("ALG_DEPTH", "0")) > 0, "ALG_DEPTH=<N> required"
N_BLOCKS = int(os.environ["ALG_DEPTH"])
K_B = int(os.environ.get("ALG_BREATH", "7"))

vs, vst, vtk, vg, vse = load_alg(os.environ.get("PC_SPLIT", "train"))
p = build_params(0)
ckpt = os.environ.get("PC_CKPT")
assert ckpt, "PC_CKPT required"
sd = safe_load(ckpt)
for k in p:
    if k in sd:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
missing = sorted(set(p) - set(sd))
if missing: print(f"[depth-census] fresh-init (not in ckpt): {missing[:8]}{'...' if len(missing) > 8 else ''}")

N = int(os.environ.get("PC_N", "32"))
N = min(N, len(vs))
# acc[(kb, i)] -> list of (std_in, std_delta) per batch
acc = {}
for s0 in range(0, N, 8):
    sl = np.arange(s0, min(s0 + 8, len(vs)))
    pad = 8 - len(sl)
    sl = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    ts = Tensor(vst[sl].astype(np.float32), dtype=dtypes.float)
    tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
    H._CENSUS = []
    o0 = forward(p, ts, tk, se)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(onp0, vse[sl].astype(np.int32))
    H._CENSUS = []
    o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float))
    o["pres"].realize()
    by_kb_i = {}
    for (kb, name, arr) in H._CENSUS:
        if not (isinstance(name, str) and name.startswith("depth") and (name.endswith("_in") or name.endswith("_out"))):
            continue
        i = int(name[len("depth"):-len("_in")] if name.endswith("_in") else name[len("depth"):-len("_out")])
        by_kb_i.setdefault((kb, i), {})[name.rsplit("_", 1)[1]] = arr
    for (kb, i), d in by_kb_i.items():
        if "in" not in d or "out" not in d:
            continue
        std_in = float(d["in"].std())
        std_delta = float((d["out"] - d["in"]).std())
        acc.setdefault((kb, i), []).append((std_in, std_delta))
    H._CENSUS = None

print(f"[depth-census] ALG_DEPTH={N_BLOCKS} K_B={K_B} rows={N} ckpt={ckpt}")
print("breath  block  std(in)   std(delta)   rel=delta/in")
last_kb = max(kb for (kb, i) in acc) if acc else None
bar_fail = []
for (kb, i) in sorted(acc):
    vals = acc[(kb, i)]
    std_in = float(np.mean([v[0] for v in vals]))
    std_delta = float(np.mean([v[1] for v in vals]))
    rel = std_delta / max(std_in, 1e-12)
    flag = ""
    if kb == last_kb:
        flag = "  <= LAST BREATH" + ("  INERT (< 5%)" if rel < 0.05 else "  live")
        if rel < 0.05:
            bar_fail.append((kb, i))
    print(f"{kb:>6}  dp{i:<4}  {std_in:8.5f}  {std_delta:10.5f}   {rel:8.4f}{flag}")

print(f"[depth-census] KNOB CENSUS BAR (every block >= 5% of the state at the last breath {last_kb}): "
      f"{'PASS' if not bar_fail else 'FAIL'} ({'all live' if not bar_fail else f'inert blocks: {bar_fail}'})")
