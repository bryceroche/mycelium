"""sort_bisect_replay.py -- THE SORTING ROOM bisect, point (A)+(C) (2026-10-09, the
coordinator's method): loads .cache/sortroom_bisect_inputs_<TAG>.npz (every kwarg
step()'s own forward() call actually passed, dumped verbatim from inside do_train
via ALG_SORT_DEBUG=1 SORT_DEBUG_DUMP_STEP=<s> SORT_DEBUG_TAG=<TAG>), rebuilds p the
SAME way do_train does (build_params + WARM_FROM), calls forward() with the EXACT
same kwarg list as step()'s own call site (line ~9533), then loss_fn(o, bg) -- this
MUST reproduce the CLI's own printed step-0 loss. If it does not, the inputs are not
yet faithful (print each b_* sum and compare to the CLI's own debug line).
Point (C): dumps every tensor in `o` (top-level keys) and every per-breath dict to
.cache/sortroom_bisect_outputs_<TAG>.npz for a later key-by-key diff.
Usage (same family env as the original run, PLUS ALG_SORT=<N> or unset to match):
  SORT_DEBUG_TAG=role8 .venv/bin/python3 scripts/sort_bisect_replay.py
"""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import build_params, forward, loss_fn
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

TAG = os.environ.get("SORT_DEBUG_TAG", "unknown")
IN_PATH = f".cache/sortroom_bisect_inputs_{TAG}.npz"
OUT_PATH = f".cache/sortroom_bisect_outputs_{TAG}.npz"
z = np.load(IN_PATH, allow_pickle=True)


def _t(arr):
    if arr.dtype.kind in ("U", "S") and str(arr) == "__NONE__":
        return None
    if arr.dtype.kind == "f":
        return Tensor(arr.astype(np.float32), dtype=dtypes.float)
    if arr.dtype.kind in ("i", "u", "b"):
        return Tensor(arr.astype(np.int32), dtype=dtypes.int)
    raise ValueError(f"unhandled dtype {arr.dtype} shape {arr.shape}")


KW_NAMES = ("s_tr", "b_tk", "b_se", "b_mask", "b_tail", "drop", "lsent", "reg",
            "fact_buf", "mh_mass", "mh_atlas_traj", "xcorr", "res_map", "hud",
            "tree", "ident", "dircue", "busreg_ramp", "valfact", "facts3", "facts5",
            "cert3", "cert5", "rack3", "rack5", "chalk3", "chalk5", "dirgold")
kw = {k: _t(z[k]) for k in KW_NAMES}
print(f"[sort-bisect] loaded {IN_PATH}: " + " ".join(
    f"{k}={'None' if v is None else tuple(v.shape)}" for k, v in kw.items()))

bg = {k[4:]: _t(z[k]) for k in z.files if k.startswith("bg__")}
print(f"[sort-bisect] bg keys: {sorted(bg.keys())}")

p = build_params(int(os.environ.get("SEED", "0")))   # do_train's own build_params(seed) call --
                                                       # the gate's SEED=242 leaves W_rq2/W_rk2/r_gain
                                                       # (never warm-loaded) at a DIFFERENT random draw
                                                       # than build_params(0) would -- must match exactly
ckpt = os.environ.get("WARM_FROM", ".cache/sharp_balV242.safetensors")
sd = safe_load(ckpt)
n_load = 0
for k in p:
    if k in sd and tuple(sd[k].shape) == tuple(p[k].shape):
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
        n_load += 1
print(f"[sort-bisect] warm from {ckpt}: {n_load}/{len(p)} keys")

o = forward(p, kw["s_tr"], kw["b_tk"], kw["b_se"], slot_mask=kw["b_mask"],
            tail=kw["b_tail"], drop=kw["drop"], lsent=kw["lsent"], reg=kw["reg"],
            fact_buf=kw["fact_buf"], mh_mass=kw["mh_mass"],
            mh_atlas_traj=kw["mh_atlas_traj"], xcorr=kw["xcorr"],
            res_map=kw["res_map"], hud=kw["hud"], tree=kw["tree"], ident=kw["ident"],
            dircue=kw["dircue"], busreg_ramp=kw["busreg_ramp"], valfact=kw["valfact"],
            facts3=kw["facts3"], facts5=kw["facts5"], cert3=kw["cert3"],
            cert5=kw["cert5"], rack3=kw["rack3"], rack5=kw["rack5"],
            chalk3=kw["chalk3"], chalk5=kw["chalk5"], dirgold=kw["dirgold"])
l = loss_fn(o, bg)
if int(os.environ.get("SORT_BISECT_STEP", "0")):
    # Does step()'s own backward()+opt.step(), happening BEFORE `return l.realize()`,
    # change what a LATER .realize() of the SAME lazy `l` reads back (tinygrad's
    # laziness re-evaluating against the now-updated, in-place-assigned weights)?
    # NOTE: calling .numpy() on `l` BEFORE backward() severs the graph (found the
    # hard way) -- backward() must be the FIRST read, exactly as step() orders it.
    from tinygrad.nn.optim import AdamW
    Tensor.training = True
    _frz = {"r_gain"}
    opt = AdamW([v for k, v in p.items() if k not in _frz], lr=1e-4, weight_decay=0.01)
    opt.zero_grad()
    l.backward()
    opt.step()
    _l_post = float(l.realize().numpy())
    print(f"[sort-bisect] TAG={TAG} loss_fn(o,bg) POST-backward+opt.step() "
          f"(the SAME tensor object, re-realized; step()'s own order)={_l_post:.6f}")
else:
    print(f"[sort-bisect] TAG={TAG} loss_fn(o,bg) PRE-backward={float(l.numpy()):.6f}")

if not int(os.environ.get("SORT_BISECT_STEP", "0")):
    # POINT (C): dump every top-level tensor in o, and every per-breath dict, in
    # forward order, for a later key-by-key diff against the other config's dump.
    # Only when SORT_BISECT_STEP=0 -- backward()+opt.step() mutates the param
    # buffers in place, and `o`'s tensors are still lazy at this point, so reading
    # them AFTER the update would silently dump POST-update values instead of the
    # ones this forward() call was actually defined against.
    _dump = {}
    for k, v in o.items():
        if k == "breaths":
            continue
        if hasattr(v, "numpy"):
            _dump[k] = v.numpy()
    if "breaths" in o:
        for bi, ob in enumerate(o["breaths"]):
            for k, v in ob.items():
                if hasattr(v, "numpy"):
                    _dump[f"breath{bi}__{k}"] = v.numpy()
    np.savez(OUT_PATH, **_dump)
    print(f"[sort-bisect] dumped {len(_dump)} tensors -> {OUT_PATH}")
