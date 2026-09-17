"""match_loss_probe.py — WHICH GOLD FIELD-GROUP'S PERMUTATION RAISES THE LOSS (2026-09-17). CPU: the head's plain
forward on the smoke rows (balV242), the loss on the original gold vs the gold permuted by match_gold, then the
permutation applied ONE field-group at a time. The loss is the audit. Env: family env, ALG_TEST=the smoke split."""
import os, sys, json, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from phase1_algebra_head import build_params, forward, load_alg, loss_fn, L_FAC, K_VARS, build_slot_masks, alt2_fact_buf
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
import match_gold as MG
vs, vst, vtk, vg, vse = load_alg("test"); B = int(os.environ.get("MP_B", "8")); idx = np.arange(B)
p = build_params(0); sd = safe_load(os.environ["PD_CKPT"]); assert set(sd) == set(p)
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in idx]); ma = np.array([vs[int(i)].get("m", 0) for i in idx])
ts = Tensor(np.ascontiguousarray(vst[idx]), dtype=dtypes.half); tk = Tensor(vtk[idx].astype(np.float32)); se = Tensor(vse[idx].astype(np.int32), dtype=dtypes.int)
o0 = forward(p, ts, tk, se); onp0 = {k: o0[k].numpy() for k in ("fat", "args", "res")}
mk = build_slot_masks(onp0, se.numpy()); _oa = {**onp0, **{k: o0[k].numpy() for k in ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())}}
o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=Tensor(alt2_fact_buf(_oa, se.numpy(), nv, ma), dtype=dtypes.float))
KEYS = ["pres", "ftype", "op", "dig", "args", "res"] + (["dup"] if "dup" in o else [])
onp = {k: o[k].numpy() for k in KEYS}
onp["pres"] = onp["pres"].squeeze(-1) if onp["pres"].ndim == 3 else onp["pres"]
if "dup" in onp: onp["dup"] = onp["dup"].squeeze(-1) if onp["dup"].ndim == 3 else onp["dup"]
preds = MG.preds_from_decode(onp)
feed = {k: np.array(vg[k][idx]) for k in vg if hasattr(vg[k], "shape") and len(vg[k]) >= B}; feed["is_lit_f"] = feed.pop("is_lit")
if "arg_dup" not in feed: feed["arg_dup"] = np.zeros_like(feed["is_rel"])
def tens(f): return {k: Tensor(np.ascontiguousarray(v), dtype=(dtypes.int if v.dtype.kind in "iu" else dtypes.float)) for k, v in f.items()}
def L(f): return float(loss_fn(o, tens(f)).numpy())
import copy
base = L(feed); print(f"[probe] loss on the original gold: {base:.4f}")
raw = {k: onp[k] for k in ("pres", "ftype", "op", "dig", "args", "res")}; raw["pres"] = raw["pres"].reshape(B, L_FAC, 1)
full = copy.deepcopy(feed); n = MG.match_feed(full, preds, raw=raw); print(f"[probe] permuted {n[0]}/{n[1]} rows; loss on the fully permuted gold: {L(full):.4f} (delta {L(full)-base:+.4f})")
# per-row: which rows changed
rows = [i for i in range(B) if any((full[k][i] != feed[k][i]).any() for k in feed)]
print("[probe] rows permuted:", rows)
GROUPS = {"slot fields (presence/ftype/op/digits/is_* /fspan/sel/arg_dup)": [k for k in MG.SLOT_KEYS if k in feed and k != "bind_ids"], "args": ["args"], "res+y+query": ["res", "y", "query"], "vspan": ["vspan"], "bind_ids": ["bind_ids"] if "bind_ids" in feed else []}
for name, keys in GROUPS.items():
    if not keys: continue
    part = copy.deepcopy(feed)
    for k in keys: part[k] = full[k].copy()
    print(f"[probe]   only {name:<60} -> {L(part):.4f} (delta {L(part)-base:+.4f})")
for i in rows:
    one = copy.deepcopy(feed)
    for k in feed: one[k][i] = full[k][i]
    print(f"[probe]   only row {i} permuted -> {L(one):.4f} (delta {L(one)-base:+.4f}); sigma head: {MG.assign(feed, i, {k: v[i] for k, v in preds.items()})[:int(feed['presence'][i].sum())].tolist()}")
