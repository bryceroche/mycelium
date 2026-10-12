"""sort_feed_knob_census.py -- THE SORTING ROOM'S FEEDBACK FORM's knob census (SRF_241,
2026-10-10; the pre/post knob law: census every organ's injection pre- and post-gain per breath
before any claim). For each loop breath kb = 1..K_B-1, prints the std of the WHOLE re-run's net
change to the token tensor (srf_tok_out - srf_tok_in, over real tokens only, tokmask-respecting)
against the std of the token tensor it met (srf_tok_in) -- the depth_knob_census.py ratio
grammar, specialized to this organ's own (kb, srf_tok_in / srf_tok_out) tags (a whole-stack tap,
not a per-block one -- the arm's own registered bar is on the RE-RUN's net effect, not any one
internal block). BAR (the arm's own registration): the re-run's change >= 10% of the token
tensor's own std at the LAST breath, else the KNOB CENSUS says the cross-attention branch was
voted down from birth (the token tensor does not read the slots) -- one of the two KILL legs.
Env: PC_CKPT (required), PC_N (rows, default 32), PC_SPLIT (default train).
"""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

assert int(os.environ.get("ALG_SORT", "0")) == 4, "ALG_SORT=4 required"
assert int(os.environ.get("ALG_SORT_FEED", "0")) == 1, "ALG_SORT_FEED=1 required"
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
if missing:
    print(f"[sort-feed-census] fresh-init (not in ckpt): {missing[:8]}{'...' if len(missing) > 8 else ''}")

N = min(int(os.environ.get("PC_N", "32")), len(vs))
acc = {}   # kb -> list of (std_in, std_delta)
for s0 in range(0, N, 8):
    sl = np.arange(s0, min(s0 + 8, N))
    pad = 8 - len(sl)
    sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    ts = Tensor(vst[sl_p].astype(np.float32), dtype=dtypes.float)
    tk = Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
    H._CENSUS = []
    o0 = forward(p, ts, tk, se)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32))
    H._CENSUS = []   # discard the open pass's own entries, keep only the real (masked) pass's
    o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float))
    o["pres"].realize()
    by_kb = {}
    for (kb, name, arr) in H._CENSUS:
        if name in ("srf_tok_in", "srf_tok_out"):
            by_kb.setdefault(kb, {})[name] = arr
    H._CENSUS = None
    tkm = vtk[sl_p] > 0.5   # (8, T) real-token mask, shared across H_W
    for kb, d in by_kb.items():
        if "srf_tok_in" not in d or "srf_tok_out" not in d:
            continue
        tin = d["srf_tok_in"][tkm]
        tout = d["srf_tok_out"][tkm]
        acc.setdefault(kb, []).append((float(tin.std()), float((tout - tin).std())))

print(f"[sort-feed-census] ALG_SORT_FEED=1 K_B={K_B} rows={N} ckpt={ckpt}")
print("breath  std(tok_in)   std(delta)   rel=delta/in")
last_kb = max(acc) if acc else None
bar_fail = []
for kb in sorted(acc):
    vals = acc[kb]
    std_in = float(np.mean([v[0] for v in vals]))
    std_delta = float(np.mean([v[1] for v in vals]))
    rel = std_delta / max(std_in, 1e-12)
    flag = ""
    if kb == last_kb:
        flag = "  <= LAST BREATH" + ("  INERT (< 10%)" if rel < 0.10 else "  LIVE (>= 10%)")
        if rel < 0.10:
            bar_fail.append(kb)
    print(f"{kb:>6}  {std_in:10.5f}  {std_delta:10.5f}   {rel:8.4f}{flag}")
print(f"[sort-feed-census] KNOB CENSUS BAR (the re-run's change at the LAST breath >= 10% of the "
      f"token tensor's own std): {'PASS (live)' if (last_kb is not None and last_kb not in bar_fail) else 'FAIL (inert -- the tokens do not read the slots)'}")
