"""probe_train.py — THE THREE-ARM TOKEN-GRAIN COUNT PROBE (2026-08-26,
word given; bars corrected by the constant reckoning). Per class c a
token-contribution direction w_c; sufficient stats reduce the arms to:
  A (intensive control): features s_c = w_c . P            (known-dead law)
  B (extensive):         features S_c = w_c . (P * L)      (the law's form)
  C (winding):           features (cos S_c, sin S_c)       (rotation form)
Each arm: per-class 8-bin count logits from its features; joint CE;
trained on form8 (96k), graded on the 143 wild golds.
BARS (pinned, the reckoning's): op-only exact > 17/143 AND F1 > 0.637
(the best-constant). A expected ~constant (the control that confirms
the intensive law); B vs C separates extensivity from rotation.
"""
import os, sys, json, glob
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from collections import Counter

OPC = ["add","sub","mul","div","sq","opa","fr","given","mod","sel","pct","fdiv"]
OPS = ("add","sub","mul","div","sq","opa","fr")
NC, NB = 12, 8
rng = np.random.default_rng(0)

P = np.load('.cache/probe_P_form8.npy')
L = np.load('.cache/probe_L_form8.npy')
G = np.load('.cache/phase1_alg_states_form8.npz')['g_opc']
n, D = P.shape
print(f"[probe] train rows {n}, D {D}", flush=True)
Pg = np.load('.cache/probe_P_gold143.npy')
Lg = np.load('.cache/probe_L_gold143.npy')

def gold143_counts():
    byid = {}
    for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl')):
        for l in open(f): r = json.loads(l); byid[r["src_idx"]] = r
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l); byid[r["src_idx"]] = r
    sk = set(json.load(open('.cache/book12_anchor_skips.json')))
    rows = [v for k, v in sorted(byid.items()) if k not in sk]
    out = np.zeros((len(rows), NC), np.int64)
    for i, r in enumerate(rows):
        c = Counter()
        for f in r["factors"]:
            if f["ftype"] == "rel":
                if f.get("op") == "mul" and len(set(f.get("args", []))) == 1:
                    c["sq"] += 1
                else: c[f.get("op", "add")] += 1
            elif f["ftype"] == "macro":
                c["opa" if f.get("name") == "OP_APPLY" else "fr"] += 1
            elif f["ftype"] == "frac": c["fr"] += 1
            else: c[f["ftype"]] += 1
        for ci, k in enumerate(OPC):
            out[i, ci] = min(c.get(k, 0), NB - 1)
    return out
Gg = gold143_counts()

mu, sd = P.mean(0), P.std(0) + 1e-6
Pn = (P - mu) / sd
Pgn = (Pg - mu) / sd
Ln = L / 100.0; Lgn = Lg / 100.0

def train_arm(arm, epochs=6, bs=512, lr=0.05):
    W = rng.standard_normal((D, NC)).astype(np.float32) * 0.02
    nf = 2 if arm == "C" else 1
    V = rng.standard_normal((NC, nf, NB)).astype(np.float32) * 0.1
    b = np.zeros((NC, NB), np.float32)
    Yt = np.minimum(G, NB - 1).astype(np.int64)
    idx_all = np.arange(n)
    for ep in range(epochs):
        rng.shuffle(idx_all)
        tot = 0.0
        for s0 in range(0, n, bs):
            idx = idx_all[s0:s0 + bs]
            X = Pn[idx]; l = Ln[idx][:, None]
            S = X @ W                      # (B, NC)
            if arm == "A": F = S[:, :, None]
            elif arm == "B": F = (S * l)[:, :, None]
            else:
                th = S * l
                F = np.stack([np.cos(th), np.sin(th)], axis=2)
            Z = np.einsum('bcf,cfk->bck', F, V) + b
            Z -= Z.max(-1, keepdims=True)
            E = np.exp(Z); Pr = E / E.sum(-1, keepdims=True)
            y = Yt[idx]
            oh = np.zeros_like(Pr); np.put_along_axis(oh, y[:, :, None], 1.0, 2)
            tot += -np.log(np.take_along_axis(Pr, y[:, :, None], 2) + 1e-9).mean()
            dZ = (Pr - oh) / len(idx)
            dV = np.einsum('bcf,bck->cfk', F, dZ)
            db = dZ.sum(0)
            dF = np.einsum('bck,cfk->bcf', dZ, V)
            if arm == "A": dS = dF[:, :, 0]
            elif arm == "B": dS = dF[:, :, 0] * l
            else:
                dS = (-np.sin(th) * dF[:, :, 0] + np.cos(th) * dF[:, :, 1]) * l
            dW = X.T @ dS
            W -= lr * dW; V -= lr * dV; b -= lr * db
        print(f"[probe {arm}] epoch {ep} loss {tot/(n//bs):.4f}", flush=True)
    # eval on 143
    S = Pgn @ W; l = Lgn[:, None]
    if arm == "A": F = S[:, :, None]
    elif arm == "B": F = (S * l)[:, :, None]
    else:
        th = S * l
        F = np.stack([np.cos(th), np.sin(th)], axis=2)
    Z = np.einsum('bcf,cfk->bck', F, V) + b
    pred = Z.argmax(-1)
    exf = exo = 0; f1s = []; distinct = set()
    for i in range(len(Gg)):
        d = Counter({OPC[c]: int(pred[i, c]) for c in range(NC) if pred[i, c] > 0})
        g = Counter({OPC[c]: int(Gg[i, c]) for c in range(NC) if Gg[i, c] > 0})
        distinct.add(tuple(sorted(d.items())))
        if d == g: exf += 1
        if Counter({k: v for k, v in d.items() if k in OPS}) == \
           Counter({k: v for k, v in g.items() if k in OPS}): exo += 1
        inter = sum((d & g).values())
        f1s.append(2 * inter / max(sum(d.values()) + sum(g.values()), 1))
    print(f"[probe {arm}] GOLD143: full {exf}/143  OP-ONLY {exo}/143  "
          f"F1 {np.mean(f1s):.3f}  distinct-preds {len(distinct)} "
          f"(bars: >17, >0.637)", flush=True)

for arm in ("A", "B", "C"):
    train_arm(arm)
