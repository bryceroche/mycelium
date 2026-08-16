"""splitgrad.py — the majority-veto read (pinned 2026-08-15): d(loss)/d(sw_g)
at the warm-start point, easy vs crowded training rows. grad>0 pushes the
gate NEGATIVE (push-off); grad<0 pulls it ON. Signs split = VETOED;
both push off = DECLINED. Zero training."""
import os, sys
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import (build_params, forward, load_alg, loss_fn,
                                 K_VARS, L_FAC, T_ALG)
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

samples, states, tokmask, gold, sent = load_alg("train")
p = build_params(0)
sd = safe_load(".cache/g23v5.safetensors")
warm = 0
for k in p:
    if k in sd and tuple(sd[k].shape) == tuple(p[k].shape):
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize(); warm += 1
print(f"[splitgrad] warmed {warm}/{len(p)} keys; sw_g init {float(p['sw_g'].numpy()[0]):.4f}", flush=True)

nfac = gold["presence"].sum(1)
q1, q3 = np.quantile(nfac, [0.25, 0.75])
easy = np.where(nfac <= q1)[0]; crowd = np.where(nfac >= q3)[0]
print(f"[splitgrad] easy n={len(easy)} (nfac<={q1:.0f})  crowded n={len(crowd)} (nfac>={q3:.0f})", flush=True)

FEED_F = ["presence","args","fspan","vspan","is_rel","is_mod","is_sel","is_pct",
          "is_fdiv","arg_dup","is_macro","is_frac","is_chain","sign"]
FEED_I = ["ftype","op","res","digits","sel","digits2","y","query"]
def gt(idx):
    g = {}
    g["is_lit_f"] = Tensor(gold["is_lit"][idx].astype(np.float32), dtype=dtypes.float)
    for k in FEED_F:
        if k in gold: g[k] = Tensor(gold[k][idx].astype(np.float32), dtype=dtypes.float)
    for k in FEED_I:
        if k in gold: g[k] = Tensor(gold[k][idx].astype(np.int32), dtype=dtypes.int)
    return g

def pop_grad(pool, tag, nb=48, seed=0):
    rng = np.random.default_rng(seed)
    gs = []; ls = []
    for b in range(nb):
        idx = rng.choice(pool, 8, replace=False)
        o = forward(p, Tensor(states[idx].astype(np.float32), dtype=dtypes.float),
                    Tensor(tokmask[idx].astype(np.float32), dtype=dtypes.float),
                    Tensor(sent[idx].astype(np.int32), dtype=dtypes.int))
        l = loss_fn(o, gt(idx))
        for k in p: p[k].grad = None
        l.backward()
        gs.append(float(p["sw_g"].grad.numpy()[0])); ls.append(float(l.numpy()))
    gs = np.array(gs); ls = np.array(ls)
    sem = gs.std()/np.sqrt(len(gs))
    print(f"[splitgrad {tag:8s}] grad(sw_g) mean {gs.mean():+.5f} (sem {sem:.5f})  "
          f"median {np.median(gs):+.5f}  {(gs>0).sum()}/{len(gs)} positive  loss {ls.mean():.4f}", flush=True)
    return gs.mean()

ge = pop_grad(easy, "easy", seed=1)
gc = pop_grad(crowd, "crowded", seed=2)
verdict = ("VETOED (signs split: easy pushes off, crowded pulls on)"
           if ge > 0 and gc < 0 else
           "DECLINED (both push off)" if ge > 0 and gc > 0 else
           "INVERTED-VETO (easy pulls on, crowded pushes off)" if ge < 0 and gc < 0 else
           "WANTED-BY-EASY-ONLY (unexpected)")
print(f"[splitgrad] VERDICT: {verdict}", flush=True)
