"""notebook_census.py -- THE NOTEBOOK's read/write census (2026-10-10): HUB_241's bonus metric
("the notebook's read/write census up"). Reports, per breath, the mean L2 norm of the
notebook's WRITE (new ink this breath, tag "notebook_write") and READ (the shelf's query-
weighted readout actually added to q_extra, tag "notebook"; existing tap, unchanged) -- the two
halves of "read/write" for this organ. Dark unless the forward pass is run with _CENSUS armed
(the eyes_autopsy.py / mixer_census.py convention); no new loss, no training.

Usage: notebook_census.py <TAG> [N_ROWS]
  TAG      a body whose checkpoint is .cache/sharp_<TAG>.safetensors (e.g. HUB_241, PMS8_241)
  N_ROWS   row count from the wild split (default 32)

Env: the caller exports the arm's own family env (FAM + SURF8 + the arm's own flag, e.g.
ALG_MIXER_OFF=1 for HUB_241) exactly as its own chain trained it -- this script does not
default ALG_NOTEBOOK or anything else itself (ALG_NOTEBOOK=1 is already in every FAM this
campaign uses; a missing export fails loud via the assert below, not a silent skip).
"""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np

if len(sys.argv) < 2:
    raise SystemExit("usage: notebook_census.py <TAG> [N_ROWS] -- caller must export the arm's own env first")
TAG = sys.argv[1]
N_ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 32
os.environ.setdefault("DEV", "CPU")

import phase1_algebra_head as H
from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, ident_build_array, T_ALG
from tinygrad.nn.state import safe_load
from tinygrad import Tensor, dtypes

assert H.ALG_NOTEBOOK, "ALG_NOTEBOOK must be set (it is part of every FAM this campaign trains with) -- otherwise there is no notebook to census"

CKPT = f".cache/sharp_{TAG}.safetensors"
vs, vst, vtk, vg, vse = load_alg(os.environ.get("MX_SPLIT", "test"))
sl = np.arange(min(N_ROWS, len(vs))); B = len(sl)

p = build_params(0)
sd = safe_load(CKPT)
miss = [k for k in p if k not in sd]
for k in p:
    if k in sd:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[notebook-census] TAG={TAG} ckpt={CKPT} ({len(miss)} params at fresh init: {miss[:4]}) rows={B}")

ts = Tensor(vst[sl].astype(np.float32), dtype=dtypes.float)
tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
_idt = Tensor(ident_build_array([vs[int(i)] for i in sl], T_ALG), dtype=dtypes.int) \
    if (H.ALG_BUSREG or H.ALG_IDKEY) else None

o0 = forward(p, ts, tk, se, ident=_idt)
onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
mk = Tensor(build_slot_masks(onp0, vse[sl].astype(np.int32)), dtype=dtypes.float)

H._CENSUS = []
o = forward(p, ts, tk, se, slot_mask=mk, ident=_idt)
census = H._CENSUS
H._CENSUS = None

reads = {}; writes = {}
for kb, tag, arr in census:
    if tag == "notebook":
        reads.setdefault(kb, []).append(arr)
    elif tag == "notebook_write":
        writes.setdefault(kb, []).append(arr)

print(f"[notebook-census] TAG={TAG} breaths with a read tap: {sorted(reads)}; write tap: {sorted(writes)}")
print(f"[notebook-census] TAG={TAG} breath  read-norm  write-norm")
read_norms = []; write_norms = []
for kb in sorted(set(reads) | set(writes)):
    rn = float(np.linalg.norm(reads[kb][-1], axis=-1).mean()) if kb in reads else float("nan")
    wn = float(np.linalg.norm(writes[kb][-1], axis=-1).mean()) if kb in writes else float("nan")
    read_norms.append(rn); write_norms.append(wn)
    print(f"[notebook-census] TAG={TAG}   b{kb}    {rn:9.4f}  {wn:9.4f}")
rmean = float(np.nanmean(read_norms)) if read_norms else float("nan")
wmean = float(np.nanmean(write_norms)) if write_norms else float("nan")
print(f"[notebook-census] TAG={TAG} MEAN read-norm={rmean:.4f} MEAN write-norm={wmean:.4f}")
