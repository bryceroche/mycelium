"""grad_cosine_census.py -- THE GRADIENT-COSINE CENSUS (2026-09-24; THE NESTED LADDER's mechanism bar,
the blog "One job, six resolutions": nested targets should make the rungs' gradients on the SHARED
weights agree instead of fighting). Eager, no JIT, a few training rows, the head's own two-pass forward
(open pass -> slot masks -> masked pass), then ONE backward per rung: rung k's own _loss_single (at the
rung's level under ALG_NEST, at level 3 otherwise) over the shared parameters -> a flattened gradient
vector; the pairwise cosine matrix between rungs and its mean off-diagonal. Run twice — ALG_NEST unset
and set — on the same checkpoint and rows: the nested run's mean cosine must exceed the flat run's
(the bar), and the per-rung norms are printed beside (which rungs pull hardest).
Env: GC_CKPT (the checkpoint; missing keys stay at init, stated), GC_ROWS (default 0:8) + the family envs
(and ALG_NEST for the nested read). Output: a line per rung + the matrix; GC_OUT appends the summary."""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, _loss_single, _NEST_LEVELS, ident_build_array, T_ALG
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

ckpt = os.environ["GC_CKPT"]
vs, vst, vtk, vg, vse = load_alg("train")
lo, hi = (int(x) for x in os.environ.get("GC_ROWS", "0:8").split(":")); sl = np.arange(lo, hi); B = len(sl)
p = build_params(0); sd = safe_load(ckpt)
_miss = [k for k in p if k not in sd]
for k in p:
    if k in sd: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
if _miss: print(f"[grad-cos] {len(_miss)} params not in the checkpoint stay at init: {_miss[:6]}", flush=True)
ts = Tensor(vst[sl].astype(np.float32), dtype=dtypes.float); tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float); se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
_idt = Tensor(ident_build_array([vs[int(i)] for i in sl], T_ALG), dtype=dtypes.int) if (H.ALG_BUSREG or H.ALG_IDKEY) else None
o0 = forward(p, ts, tk, se, ident=_idt); onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
mk = Tensor(build_slot_masks(onp0, vse[sl].astype(np.int32)), dtype=dtypes.float)
g = {}
for k in vg:
    a = vg[k][sl]; g[k] = Tensor(a.astype(np.float32) if a.dtype.kind == "f" else a.astype(np.int32), dtype=dtypes.float if a.dtype.kind == "f" else dtypes.int)
if "is_lit_f" not in g and "is_lit" in g: g["is_lit_f"] = g["is_lit"]
shared = [k for k in sorted(p) if k not in H._FRZ_NAMES()] if hasattr(H, "_FRZ_NAMES") else sorted(p)
K = int(os.environ.get("ALG_BREATH", "7"))
vecs, norms, losses = [], [], []
for kb in range(K):
    o = forward(p, ts, tk, se, slot_mask=mk, ident=_idt)   # a fresh graph per rung (a backward consumes it)
    assert "breaths" in o and len(o["breaths"]) == K, ("breaths", len(o.get("breaths", [])), K)
    full = dict(o, **o["breaths"][kb])
    lvl = (_NEST_LEVELS[kb] if (_NEST_LEVELS is not None and kb < len(_NEST_LEVELS)) else 3)
    loss = _loss_single(full, g, level=lvl)
    for t in p.values(): t.grad = None
    loss.backward()
    v = np.concatenate([(p[k].grad.detach().numpy().reshape(-1) if p[k].grad is not None else np.zeros(int(np.prod(p[k].shape)), np.float32)) for k in shared]).astype(np.float64)
    vecs.append(v); norms.append(float(np.linalg.norm(v))); losses.append(float(loss.numpy()))
    print(f"[grad-cos] rung {kb} level {lvl}: loss {losses[-1]:.4f} |grad| {norms[-1]:.4f}", flush=True)
V = np.stack(vecs); N = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
C = (V / N) @ (V / N).T
# PER PARAMETER GROUP (Opus's check 3): an average can hide a fight inside one head — the args head,
# the digit head, the ftype head, the router's own weights, and everything else, each its own cosine.
_off = 0; _spans = {}
for k in shared:
    n_ = int(np.prod(p[k].shape)); _spans[k] = (_off, _off + n_); _off += n_
GROUPS = {"args-head": ["W_args"], "dig-head": ["h_dig", "h_dig_b"], "ftype-head": ["h_ftype", "h_ftype_b"],
          "router": [k for k in shared if k in ("W_rk2", "W_rq2", "theta_given", "W_role", "w_prec")]}
GROUPS["rest"] = [k for k in shared if not any(k in v for v in GROUPS.values())]
for gname, keys in GROUPS.items():
    idx = np.concatenate([np.arange(*_spans[k]) for k in keys if k in _spans]) if keys else np.zeros(0, int)
    if len(idx) == 0:
        continue
    Vg = V[:, idx]; Ng = np.linalg.norm(Vg, axis=1, keepdims=True) + 1e-12; Cg = (Vg / Ng) @ (Vg / Ng).T
    og = Cg[~np.eye(K, dtype=bool)]
    print(f"[grad-cos]   group {gname:11s} ({len(idx):9d} params): mean off-diag cosine {og.mean():+.4f} | min {og.min():+.4f} | negatives {int((og < 0).sum())}/{len(og)}")
off = C[~np.eye(K, dtype=bool)]
np.set_printoptions(precision=3, suppress=True, linewidth=140)
print("[grad-cos] pairwise cosine between rungs' gradients on the shared weights:"); print(C)
adj = float(np.mean([C[i, i + 1] for i in range(K - 1)]))
summary = (f"[grad-cos] {os.path.basename(ckpt)} ALG_NEST={os.environ.get('ALG_NEST', '') or 'unset'} rows {lo}:{hi} | mean off-diagonal cosine "
           f"{off.mean():.4f} | min {off.min():.4f} | adjacent-rung mean {adj:.4f} | rung 0 vs rung {K-1} {C[0, K-1]:.4f}")
print(summary)
if os.environ.get("GC_OUT"):
    open(os.environ["GC_OUT"], "a").write(summary + "\n")
