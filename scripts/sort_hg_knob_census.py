"""sort_hg_knob_census.py -- THE HOURGLASS SORTING ROOM form 2b's knob census (the pre/post knob
law: census every organ's injection pre- and post-gain per breath before any claim). _sort_room_hg
is a single-shot organ (no breath index; it runs once, before the loop, replacing `waist`), so the
census taps THREE arrays it appends to the generic H._CENSUS hook under kb=-1:
  sorthg_tok1 -- block0's own output, the token path block3 would otherwise meet unchanged
  sorthg_Uc   -- the clause-resolution path's own contribution to block3's input (U_c(clause))
  sorthg_Us   -- the sentence-resolution path's own contribution (U_s(sentence))
Prints std(U_c + U_s) against std(tok1) over REAL tokens only (tokmask-respecting) -- the SAME
ratio grammar every other organ's knob census uses. BAR (the arm's own registration): the
coarse path's contribution >= 10% of the token path's at block3's input, else the hourglass's
coarse path was voted down from birth (the residual-seal law's own warning, read as a measurement).
Env: PC_CKPT (required), PC_N (rows, default 32), PC_SPLIT (default train).
"""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, T_ALG
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

assert int(os.environ.get("ALG_SORT", "0")) == 4, "ALG_SORT=4 required"
assert int(os.environ.get("ALG_SORT_HG", "0")) in (1, 2), "ALG_SORT_HG=1 or 2 required"

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
    print(f"[sort-hg-census] fresh-init (not in ckpt): {missing[:8]}{'...' if len(missing) > 8 else ''}")

N = min(int(os.environ.get("PC_N", "32")), len(vs))
stds_tok1, stds_uc, stds_us, stds_sum = [], [], [], []
for s0 in range(0, N, 8):
    sl = np.arange(s0, min(s0 + 8, N))
    pad = 8 - len(sl)
    sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    ts = Tensor(vst[sl_p].astype(np.float32), dtype=dtypes.float)
    tk = Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
    tree_t = Tensor(np.stack([H.tree_row_ids(vs[int(i)]["text"], vtk[i], vse[i], T_ALG)
                              for i in sl_p]).astype(np.int32), dtype=dtypes.int)
    H._CENSUS = []
    o0 = forward(p, ts, tk, se, tree=tree_t)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32))
    H._CENSUS = []   # the depth-census convention: discard the open pass's own entries, keep only the real (masked) pass's
    o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), tree=tree_t)
    o["pres"].realize()
    by_name = {}
    for (kb, name, arr) in H._CENSUS:
        if name in ("sorthg_tok1", "sorthg_Uc", "sorthg_Us"):
            by_name[name] = arr
    H._CENSUS = None
    if not all(k in by_name for k in ("sorthg_tok1", "sorthg_Uc", "sorthg_Us")):
        continue
    tkm = vtk[sl_p] > 0.5   # (8, T) real-token mask, shared across H_W
    tok1 = by_name["sorthg_tok1"][tkm]     # (n_real, H_W)
    uc = by_name["sorthg_Uc"][tkm]
    us = by_name["sorthg_Us"][tkm]
    stds_tok1.append(float(tok1.std()))
    stds_uc.append(float(uc.std()))
    stds_us.append(float(us.std()))
    stds_sum.append(float((uc + us).std()))

std_tok1 = float(np.mean(stds_tok1))
std_uc = float(np.mean(stds_uc))
std_us = float(np.mean(stds_us))
std_sum = float(np.mean(stds_sum))
rel = std_sum / max(std_tok1, 1e-12)
print(f"[sort-hg-census] ALG_SORT_HG={os.environ['ALG_SORT_HG']} rows={N} ckpt={ckpt}")
print(f"[sort-hg-census] std(tok1)={std_tok1:.6f} std(U_c)={std_uc:.6f} std(U_s)={std_us:.6f} std(U_c+U_s)={std_sum:.6f} rel=std(U_c+U_s)/std(tok1)={rel:.4f}")
print(f"[sort-hg-census] KNOB CENSUS BAR (the coarse path's contribution at block3's/the final block's input "
      f">= 10% of the token path's): {'PASS (live)' if rel >= 0.10 else 'FAIL (inert -- the coarse path was voted down)'}")
