"""grad_census.py — THE GRADIENT CENSUS (2026-09-10): how much of the real
training loss's gradient reaches the state entering each breath. Eager
(no JIT), 8 training rows, the head's own two-pass forward (open pass ->
slot masks -> masked pass under the taps), _loss_single, backward.
Reports ||dL/d(state entering breath kb)|| per breath (mean over rows and
slots), normalized by the readout's own state gradient, and the notebook
read's attention over the shelf per breath (whose ink is read). Env:
GC_CKPT + the family envs; GC_ROWS (default 0:8)."""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
if os.environ.get("GC_HEAD_DIR"): sys.path.insert(0, os.environ["GC_HEAD_DIR"])   # a reference copy of the head (e.g. git HEAD) for before/after rulings
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, L_FAC, L_TOT, _loss_single
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
ckpt = os.environ["GC_CKPT"]
vs, vst, vtk, vg, vse = load_alg("train")
lo, hi = (int(x) for x in os.environ.get("GC_ROWS", "0:8").split(":")); sl = np.arange(lo, hi); B = len(sl)
p = build_params(0); sd = safe_load(ckpt)
assert set(sd) == set(p), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
ts = Tensor(vst[sl].astype(np.float32), dtype=dtypes.float); tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float); se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
o0 = forward(p, ts, tk, se); onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
mk = Tensor(build_slot_masks(onp0, vse[sl].astype(np.int32)), dtype=dtypes.float)
K = int(os.environ.get("ALG_BREATH", "7"))
taps = {kb: Tensor(np.zeros((B, L_TOT, 512), np.float32), requires_grad=True) for kb in range(1, K)}
taps["final"] = Tensor(np.zeros((B, L_TOT, 512), np.float32), requires_grad=True)
H._GTAP = taps
o = forward(p, ts, tk, se, slot_mask=mk)
g = {}
for k in vg:
    a = vg[k][sl]; g[k] = Tensor(a.astype(np.float32) if a.dtype.kind == "f" else a.astype(np.int32), dtype=dtypes.float if a.dtype.kind == "f" else dtypes.int)
if "is_lit_f" not in g and "is_lit" in g: g["is_lit_f"] = g["is_lit"]
loss = _loss_single(o, g)
for t in p.values(): t.grad = None
loss.backward()
lv = float(loss.numpy())
if os.environ.get("GC_DUMP"):                       # the loss + every param grad, for a before/after ruling
    np.savez(os.environ["GC_DUMP"], loss=np.array(lv), **{("g_" + k): (p[k].grad.detach().numpy() if p[k].grad is not None else np.zeros(1)) for k in p})
    print(f"[grad-census] dumped loss + {len(p)} param grads -> {os.environ['GC_DUMP']}")
def gn(t):
    return 0.0 if t.grad is None else float((t.grad.detach() ** 2).sum(-1).sqrt().mean().numpy())
fin = gn(taps["final"])
prof = [gn(taps[kb]) for kb in range(1, K)]
print(f"[grad-census] {os.path.basename(ckpt)} loss={lv:.4f} ||dL/dstate|| entering breath 1..{K-1}: " + " ".join(f"b{kb}={v:.4f}" for kb, v in zip(range(1, K), prof)) + f" | readout state={fin:.4f}")
print(f"[grad-census] {os.path.basename(ckpt)} relative to the readout: " + " ".join(f"b{kb}={v / max(fin, 1e-12):.3f}" for kb, v in zip(range(1, K), prof)) + f" | THE FARMER (b1/readout) = {prof[0] / max(fin, 1e-12):.3f}, b1/b{K-1} = {prof[0] / max(prof[-1], 1e-12):.3f}")
ats = H._GTAP.get("at", [])
for kb, at in ats:
    a = at.numpy(); ent = float(-(a * np.log(a + 1e-12)).sum(-1).mean()); w = a.mean((0, 1))
    print(f"[grad-census] shelf read at breath {kb}: entries={a.shape[-1]} mean attention over inks [b0..] = {np.round(w, 3).tolist()} entropy={ent:.3f} nats (uniform={np.log(a.shape[-1]):.3f})")
H._GTAP = None
