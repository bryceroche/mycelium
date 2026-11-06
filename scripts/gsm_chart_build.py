"""gsm_chart_build.py -- THE ZOOMING QUERY's charts (ledger 2026-11-06 11:59, Bryce's word for the whole queue).

The POINTER-CHART ORACLE's nested Tree (RB8 states at numerals, L2-normalized, k-means 8 per node x 3 levels, fit on TRAIN numerals, seed 0;
the grow() below is the oracle's, verbatim) -> the ST64 node (level-0 x 8 + level-1) of every numeral in every split, as an int16 array laid
out as the states memmap; 64 = no chart (every non-numeral). TRAIN numerals take their fit labels (written last: ORIG is inside TRAIN); the
other splits descend the tree by predict.  -> .cache/gsmb_chart64_rb8.npy
"""
import os, sys

os.environ["RD_STATES"] = "gsmb_rb8_states.npy"
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm_breath_reader as BR          # FIRST (the 11-04 import rule)
from sklearn.cluster import KMeans

Dt, states = BR.TR.load()
NB, NONE = 8, 64
keys = {sp: [(g, i) for g, it in enumerate(Dt[sp]) for i, v in enumerate(it["vals"]) if v is not None] for sp in Dt}


def feats(sp):
    X = np.stack([states[Dt[sp][g]["st0"] + i] for g, i in keys[sp]]).astype(np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)


XT = feats("TRAIN"); pT = np.zeros((len(XT), 3), np.int64); node = {(): None}


def grow(idx, pre, lvl):
    if lvl == 3: return
    km = KMeans(NB, n_init=1, random_state=0).fit(XT[idx]) if len(idx) >= NB else None
    node[pre] = km
    lab = km.labels_ if km else np.zeros(len(idx), np.int64)
    pT[idx, lvl] = lab
    for c in range(NB): grow(idx[lab == c], pre + (c,), lvl + 1)


grow(np.arange(len(XT)), (), 0)
CH = np.full(states.shape[0], NONE, np.int16)
for sp in [s for s in Dt if s != "TRAIN"] + ["TRAIN"]:
    if sp == "TRAIN": p = pT
    else:
        X = feats(sp); p = np.zeros((len(X), 2), np.int64)
        for lvl in range(2):   # descend by groups of the same prefix
            pre = [tuple(r[:lvl]) for r in p]
            for q in set(pre):
                idx = np.array([j for j, r in enumerate(pre) if r == q]); km = node.get(q)
                p[idx, lvl] = 0 if km is None else km.predict(X[idx])
    for j, (g, i) in enumerate(keys[sp]): CH[Dt[sp][g]["st0"] + i] = p[j, 0] * NB + p[j, 1]
    print(f"[chart] {sp}: {len(keys[sp])} numerals, {len(set(p[:, 0] * NB + p[:, 1]))} charts used", flush=True)
np.save(BR.GD.V1.C("gsmb_chart64_rb8.npy"), CH)
print(f"[chart] -> gsmb_chart64_rb8.npy rows {len(CH)}, numerals {(CH < NONE).sum()}")
