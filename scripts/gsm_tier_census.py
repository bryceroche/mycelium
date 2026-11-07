"""gsm_tier_census.py -- THE TIER CENSUS (ledger 2026-11-07 06:21, Bryce's 3-tier charts; zero GPU, torch CPU).

Per candidate representation of numerals: the POINTER-CHART ORACLE's Tree recipe (k-means 8 x 3 levels on TRAIN numerals, seed 0,
L2-normalized; other splits descend by predict), then STAB = share of ORIG numerals keeping their ST8 / ST64 chart in the CFNUM / CFNAME
twin (paired by gid), and ISO = over the oracle's token-operand rows, the share where some valid candidate's ST64 chart differs from every
non-valid candidate's (rows with no non-valid candidate skipped). Candidates: A raw RB8 states, B role feats (14-d), C BT4RB8 s0 xq,
D the gold role label (7 classes, used directly as the chart: the ceiling).  Bar T2: some of A-C has STAB64 >= 0.90 on both edits and
ISO TEST >= A's.  -> .cache/gsm_tier_census.json + _verdict.txt
"""
import json, os, sys

for k_, v_ in dict(BT_D="512", BT_NH="8", BT_DB="64", BT_CYCLOCK="1", BT_ZOOM="1", BT_ROLE="1", BT_T6OFF="1",
                   RD_STATES="gsmb_rb8_states.npy", BT_ROLEF="gsmb_role_feats_rb8.npy").items():
    os.environ[k_] = v_                                                         # BT4RB8's own flags
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm_breath_reader as BR          # FIRST (the 11-04 import rule)
import torch
from sklearn.cluster import KMeans
from gsm_role_tree import roles

torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
Dt, states = BR.TR.load(); tree = BR.TR.load_tree()
BR.RF = np.load(BR.GD.V1.C(BR.rolef_name()), mmap_mode="r")
SPL = ("TRAIN", "ORIG", "CFNUM", "CFNAME", "TEST"); NB = 8
RL = ["*", "+", "-A", "-B", "/A", "/B", "none"]
keys = {sp: [(g, i) for g, it in enumerate(Dt[sp]) for i, v in enumerate(it["vals"]) if v is not None] for sp in SPL}
at = {sp: {gi: j for j, gi in enumerate(keys[sp])} for sp in SPL}


def xq_read(sp):
    net = BR.make_net(); net.load_state_dict(torch.load(".cache/gsmb_breath_BT4RB8_s0.pt")); net.eval(); its = Dt[sp]; xs = {}
    with torch.no_grad():
        for s0 in range(0, len(its), 128):
            part = its[s0:s0 + 128]; t, E = BR.batch(part, states, tree, np.random.default_rng(s0)); xq = net.read(t)
            for b, it in enumerate(part):
                for i, v in enumerate(it["vals"]):
                    if v is not None: xs[s0 + b, i] = xq[b, i].numpy()
    return np.stack([xs[gi] for gi in keys[sp]]).astype(np.float32)


def feats(name, sp):
    if name == "A": return np.stack([states[Dt[sp][g]["st0"] + i] for g, i in keys[sp]]).astype(np.float32)
    if name == "B": return np.stack([BR.RF[Dt[sp][g]["st0"] + i] for g, i in keys[sp]]).astype(np.float32)
    return xq_read(sp)


def charts(name):
    """-> {split: (n, 2) array of [level 0, level 1]}"""
    if name == "D":
        out = {}
        for sp in SPL:
            R = {g: roles(Dt[sp][g]) for g in range(len(Dt[sp]))}
            lab = np.array([RL.index(R[g][i]) for g, i in keys[sp]]); out[sp] = np.stack([lab, np.zeros_like(lab)], 1)
        return out
    F = {}
    for sp in SPL:
        X = feats(name, sp); F[sp] = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
    node = {}

    def grow(idx, pre, lvl):
        if lvl == 2: return
        km = KMeans(NB, n_init=1, random_state=0).fit(F["TRAIN"][idx]) if len(idx) >= NB else None
        node[pre] = km; lab = km.labels_ if km else np.zeros(len(idx), np.int64)
        for c in range(NB): grow(idx[lab == c], pre + (c,), lvl + 1)
    grow(np.arange(len(F["TRAIN"])), (), 0)
    out = {}
    for sp in SPL:
        p = np.zeros((len(F[sp]), 2), np.int64)
        for lvl in range(2):
            pre = [tuple(r[:lvl]) for r in p]
            for q in set(pre):
                idx = np.array([j for j, r in enumerate(pre) if r == q]); km = node.get(q)
                p[idx, lvl] = 0 if km is None else km.predict(F[sp][idx])
        out[sp] = p
    return out


def rows(it):
    num = [i for i, v in enumerate(it["vals"]) if v is not None]; cons, out = set(), []
    for k, (op, Aa, Bb) in enumerate(it["prog"]):
        for r, X_ in enumerate((Aa, Bb)):
            if BR.TR.first(X_)[0] == "t": out.append((set(X_["t"]), [i for i in num if i not in cons]))
        for X_ in (Aa, Bb):
            f = BR.TR.first(X_)
            if f[0] == "t": cons.add(f[1])
    return out


o = {}
og = {it["gid"]: g for g, it in enumerate(Dt["ORIG"])}
for name in ("A", "B", "C", "D"):
    P = charts(name); c64 = {sp: P[sp][:, 0] * NB + P[sp][:, 1] for sp in SPL}; r = {}
    for sp in ("CFNUM", "CFNAME"):
        s8 = s64 = n = 0
        for gb, b in enumerate(Dt[sp]):
            ga = og[b["gid"]]; na = [i for i, v in enumerate(Dt["ORIG"][ga]["vals"]) if v is not None]
            nb_ = [i for i, v in enumerate(b["vals"]) if v is not None]
            if len(na) != len(nb_): continue
            for i, j in zip(na, nb_):
                x, y = at["ORIG"][ga, i], at[sp][gb, j]; n += 1
                s8 += P["ORIG"][x, 0] == P[sp][y, 0]; s64 += c64["ORIG"][x] == c64[sp][y]
        r[f"stab8_{sp}"], r[f"stab64_{sp}"] = s8 / n, s64 / n
    for sp in ("ORIG", "CFNUM", "CFNAME", "TEST"):
        iso = []
        for g, it in enumerate(Dt[sp]):
            for valid, cand in rows(it):
                V = [i for i in cand if i in valid]; W = [i for i in cand if i not in valid]
                if not V or not W: continue
                wc = {c64[sp][at[sp][g, i]] for i in W}; iso.append(any(c64[sp][at[sp][g, i]] not in wc for i in V))
        r[f"iso_{sp}"] = float(np.mean(iso)); r[f"n_iso_{sp}"] = len(iso)
    o[name] = r; print(f"[tier] {name} {r}", flush=True)
json.dump(o, open(".cache/gsm_tier_census.json", "w"), indent=1)
NM = dict(A="raw RB8 states", B="role feats", C="BT4RB8 xq", D="gold role (ceiling)")
L = [f"{n} {NM[n]:20s} STAB8 CFNUM {o[n]['stab8_CFNUM']:.3f} CFNAME {o[n]['stab8_CFNAME']:.3f} | STAB64 CFNUM {o[n]['stab64_CFNUM']:.3f} "
     f"CFNAME {o[n]['stab64_CFNAME']:.3f} | ISO ORIG {o[n]['iso_ORIG']:.3f} CFNUM {o[n]['iso_CFNUM']:.3f} CFNAME {o[n]['iso_CFNAME']:.3f} "
     f"TEST {o[n]['iso_TEST']:.3f}" for n in "ABCD"]
win = [n for n in "ABC" if min(o[n]["stab64_CFNUM"], o[n]["stab64_CFNAME"]) >= 0.90 and o[n]["iso_TEST"] >= o["A"]["iso_TEST"]]
L.append(f"T2 (some of A-C: STAB64 >= 0.90 both edits AND ISO TEST >= A's {o['A']['iso_TEST']:.3f}): {'PASS ' + ','.join(win) if win else 'FAIL'}")
print("\n".join(L)); open(".cache/gsm_tier_census_verdict.txt", "w").write("\n".join(L) + "\n")
