"""pos_probe.py — POSITION CHANNEL A0 (2026-08-24, word given): is
graph-position ABSENT from slot states or merely NONLINEAR? Sweep as
the ladder survey; train a small MLP (512->64->6) on depth labels over
frozen gsb227 states; held accuracy vs chance 0.47 decides the organ's
form: MLP >> chance = readout-head organ; ~chance = position must be
COMPUTED/INJECTED architecturally. Terminality probed alongside
(balanced accuracy this time).
"""
import os, sys, json
os.environ.setdefault("ALG_MINE_BREATHS", "1")
os.environ.setdefault("ALG_BREATH", "7")
os.environ.setdefault("ALG_NOTEBOOK", "1")
os.environ.setdefault("ALG_SIXWAVE", "1")
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "9",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_TRAIN": ".cache/form_mix8.jsonl",
                   "ALG_TRAIN_NAME": "form8",
                   "ALG_ALLOW_PEN_TRAIN": "1",
                   "ALG_TEST": ".cache/algebra_nl_test.jsonl",
                   "ALG_TEST_NAME": "test23"})
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from phase1_algebra_head import (build_params, forward, load_alg,
                                 build_slot_masks, L_FAC)
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from dialect_ladder import fac_meta

samples, states, tokmask, gold, sent = load_alg("train")
p = build_params(0)
sd = safe_load('.cache/gsb227_real.safetensors')
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
CAP = 600
rng = np.random.RandomState(11)
rows = rng.choice(states.shape[0], CAP, replace=False)
Xs = []; Yd = []; Yt = []
for s0 in range(0, CAP, 8):
    sl = [int(r) for r in rows[s0:s0 + 8]]
    pad = 8 - len(sl); slp = sl + sl[:1] * pad
    ts = Tensor(states[slp].astype(np.float32), dtype=dtypes.float)
    tk = Tensor(tokmask[slp].astype(np.float32), dtype=dtypes.float)
    se = Tensor(sent[slp].astype(np.int32), dtype=dtypes.int)
    o0 = forward(p, ts, tk, se)
    o0n = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(o0n, sent[slp])
    o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float))
    B = [b.realize().numpy() for b in o["breaths_all"]]
    for bi, ri in enumerate(sl):
        r = samples[ri]; meta = fac_meta(r)
        for j in range(min(L_FAC, len(r["factors"]))):
            if gold["presence"][ri, j] <= 0: continue
            v = np.concatenate([B[0][bi, j], B[-1][bi, j]])  # early+late
            Xs.append(v / (np.linalg.norm(v) + 1e-9))
            d, t = meta[j] if j < len(meta) else (0, 0)
            Yd.append(d); Yt.append(t)
X = np.stack(Xs).astype(np.float32)
Yd = np.array(Yd); Yt = np.array(Yt)
print(f"[pos] {len(X)} instances, dim {X.shape[1]}", flush=True)
idx = np.random.RandomState(3).permutation(len(X))
tr, te = idx[:int(.7 * len(X))], idx[int(.7 * len(X)):]

def mlp_train(Y, ncls, epochs=60, lr=0.05):
    rs = np.random.RandomState(9)
    W1 = rs.randn(X.shape[1], 64).astype(np.float32) * 0.05
    b1 = np.zeros(64, np.float32)
    W2 = rs.randn(64, ncls).astype(np.float32) * 0.05
    b2 = np.zeros(ncls, np.float32)
    for ep in range(epochs):
        perm = rs.permutation(len(tr))
        for s0 in range(0, len(tr), 256):
            bidx = tr[perm[s0:s0 + 256]]
            xb, yb = X[bidx], Y[bidx]
            h = np.maximum(xb @ W1 + b1, 0)
            z = h @ W2 + b2
            z -= z.max(1, keepdims=True)
            pr = np.exp(z); pr /= pr.sum(1, keepdims=True)
            g = pr; g[np.arange(len(yb)), yb] -= 1; g /= len(yb)
            gW2 = h.T @ g; gb2 = g.sum(0)
            gh = g @ W2.T; gh[h <= 0] = 0
            gW1 = xb.T @ gh; gb1 = gh.sum(0)
            W2 -= lr * gW2; b2 -= lr * gb2
            W1 -= lr * gW1; b1 -= lr * gb1
    h = np.maximum(X[te] @ W1 + b1, 0)
    pred = (h @ W2 + b2).argmax(1)
    acc = float((pred == Y[te]).mean())
    # balanced accuracy
    bacc = float(np.mean([np.mean(pred[Y[te] == c] == c)
                          for c in set(Y[te].tolist())
                          if (Y[te] == c).sum() >= 5]))
    return acc, bacc

for name, Y, ncls in (("depth", Yd, int(Yd.max()) + 1),
                      ("term", Yt, 2)):
    acc, bacc = mlp_train(Y, ncls)
    ch = float(np.bincount(Y[tr]).max() / len(tr))
    print(f"[pos] {name}: MLP acc {acc:.3f} balanced {bacc:.3f} "
          f"(majority chance {ch:.3f})", flush=True)
print("[pos] VERDICT: MLP >> chance = readout-head organ; "
      "~chance = position must be computed/injected", flush=True)
