"""gsm_keychart_probe.py -- THE KEY-CHART PROBE (ledger 2026-11-06 10:43, Bryce's word; zero GPU, banked BT4RB8 s0 / s1, torch CPU).

The POINTER-CHART ORACLE's frame (rows, candidates, consumed rule; the nested Tree of RB8 states at numerals, k-means 8 x 3 levels, seed 0,
copied verbatim) and the READER PER-POINTER read's protocol (gold slots < k committed, cond = gold types, cycle floor(k NCYC / K)).
KP-1: multinomial logistic probe of the ST64 chart, fit on TRAIN numerals, scored on TEST numerals; r = acc(xq) / acc(raw state), seed mean
(CARRIED >= 0.90 / PARTIAL / DROPPED < 0.70). KP-E: of the wrong RESTR picks on TEST rows with a valid candidate, the share whose ST64 chart
equals some valid candidate's chart (>= 0.50 ADDRESS errors dominate / < 0.50 CHART errors dominate).  -> .cache/gsm_keychart_probe.json + _verdict.txt
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(int(os.environ.get("TOY_THREADS", "6")))
Dt, states = BR.TR.load(); tree = BR.TR.load_tree()
BR.RF = np.load(BR.GD.V1.C(BR.rolef_name()), mmap_mode="r")
SPL = ("TRAIN", "TEST"); K, NCYC = BR.K, BR.NCYC; NB = 8

# ---- the nested Tree (as the POINTER-CHART ORACLE)
keys = {sp: [(g, i) for g, it in enumerate(Dt[sp]) for i, v in enumerate(it["vals"]) if v is not None] for sp in SPL}
X = {sp: np.stack([states[Dt[sp][g]["st0"] + i] for g, i in keys[sp]]).astype(np.float32) for sp in SPL}
for sp in SPL: X[sp] /= np.linalg.norm(X[sp], axis=1, keepdims=True) + 1e-6
path = {sp: np.zeros((len(keys[sp]), 3), np.int64) for sp in SPL}
node = {(): None}


def grow(idx, pre, lvl):
    if lvl == 3: return
    km = KMeans(NB, n_init=1, random_state=0).fit(X["TRAIN"][idx]) if len(idx) >= NB else None
    node[pre] = km
    lab = km.labels_ if km else np.zeros(len(idx), np.int64)
    path["TRAIN"][idx, lvl] = lab
    for c in range(NB): grow(idx[lab == c], pre + (c,), lvl + 1)


grow(np.arange(len(keys["TRAIN"])), (), 0)
for j in range(len(keys["TEST"])):
    pre = ()
    for lvl in range(3):
        km = node.get(pre); c = 0 if km is None else int(km.predict(X["TEST"][j:j + 1])[0]); path["TEST"][j, lvl] = c; pre += (c,)
chart = {sp: {gi: tuple(path[sp][j]) for j, gi in enumerate(keys[sp])} for sp in SPL}
lab64 = {sp: np.array([p[0] * NB + p[1] for p in path[sp]]) for sp in SPL}
lab8 = {sp: path[sp][:, 0] for sp in SPL}


def rows(it):
    """as the oracle: per token operand (k, role, valid set, candidates)"""
    num = [i for i, v in enumerate(it["vals"]) if v is not None]; cons, out = set(), []
    for k, (op, Aa, Bb) in enumerate(it["prog"]):
        for r, X_ in enumerate((Aa, Bb)):
            if BR.TR.first(X_)[0] == "t": out.append((k, r, set(X_["t"]), [i for i in num if i not in cons]))
        for X_ in (Aa, Bb):
            f = BR.TR.first(X_)
            if f[0] == "t": cons.add(f[1])
    return out


def read(net, sp):
    """-> xq at every numeral (keys order), and per row (g, valid, cand, RESTR pick)"""
    its = Dt[sp]; xs = {}; picks = []
    with torch.no_grad():
        for s0 in range(0, len(its), 128):
            part = its[s0:s0 + 128]
            t, E = BR.batch(part, states, tree, np.random.default_rng(s0)); L = t["pad"].shape[1]; B = len(part)
            for b, it in enumerate(part): E[b]["vals"] = it["vals"]
            progs, gt = BR.gold(E); xq = net.read(t); gtt = torch.as_tensor(gt)
            for b, it in enumerate(part):
                for i, v in enumerate(it["vals"]):
                    if v is not None: xs[s0 + b, i] = xq[b, i].numpy()
            R_ = [rows(it) for it in part]; need = {k for rr in R_ for k, *_ in rr}; P = {}
            for k in sorted(need):
                com = np.zeros((B, K), bool); com[:, :k] = True
                nb = BR.nb_tensors(BR.notebook(E, L, progs, com, gt, np.ones((B, K), np.float32)))
                _, pa, pb, _, _ = net.breath(t, xq, nb, gtt, torch.full((B,), min(k * NCYC // K, NCYC - 1)))
                P[k] = (pa[:, k].numpy(), pb[:, k].numpy())
            for b, rr in enumerate(R_):
                for k, r, valid, cand in rr:
                    lg = P[k][r][b]
                    picks.append((s0 + b, valid, cand, max(cand, key=lambda i: (lg[i], -i)) if cand else None))
    return np.stack([xs[gi] for gi in keys[sp]]).astype(np.float32), picks


def probe(Ftr, Fte, ytr, yte):
    sc = StandardScaler().fit(Ftr)
    m = LogisticRegression(max_iter=300, C=1.0).fit(sc.transform(Ftr), ytr)
    return float((m.predict(sc.transform(Fte)) == yte).mean())


def kinds(picks):
    """KP-E: wrong picks on rows with a valid candidate -> share in a valid candidate's chart (ST64, ST512) + the random-wrong base rate"""
    n = w64 = w512 = 0; base = []
    for g, valid, cand, p in picks:
        V = [i for i in cand if i in valid]
        if not V or p in valid: continue
        n += 1; ch = chart["TEST"]; vc64 = {ch[g, i][:2] for i in V}; vc512 = {ch[g, i] for i in V}
        w64 += ch[g, p][:2] in vc64; w512 += ch[g, p] in vc512
        wr = [i for i in cand if i not in valid]; base.append(np.mean([ch[g, i][:2] in vc64 for i in wr]))
    return dict(n_wrong=n, addr64=w64 / n, addr512=w512 / n, base64=float(np.mean(base)))


o = {}
rng = np.random.default_rng(0); RP = rng.standard_normal((X["TRAIN"].shape[1], 512)).astype(np.float32) / np.sqrt(512)
o["raw_ST64"] = probe(X["TRAIN"], X["TEST"], lab64["TRAIN"], lab64["TEST"]); o["raw_ST8"] = probe(X["TRAIN"], X["TEST"], lab8["TRAIN"], lab8["TEST"])
o["rp512_ST64"] = probe(X["TRAIN"] @ RP, X["TEST"] @ RP, lab64["TRAIN"], lab64["TEST"])
print(f"[kp] raw ST64 {o['raw_ST64']:.4f} ST8 {o['raw_ST8']:.4f} | random-512 ST64 {o['rp512_ST64']:.4f}", flush=True)
for s in (0, 1):
    net = BR.make_net(); net.load_state_dict(torch.load(f".cache/gsmb_breath_BT4RB8_s{s}.pt")); net.eval()
    (Ftr, _), (Fte, pk) = read(net, "TRAIN"), read(net, "TEST")
    o[f"s{s}"] = dict(xq_ST64=probe(Ftr, Fte, lab64["TRAIN"], lab64["TEST"]), xq_ST8=probe(Ftr, Fte, lab8["TRAIN"], lab8["TEST"]), **kinds(pk))
    print(f"[kp] s{s} {o[f's{s}']}", flush=True)
json.dump(o, open(".cache/gsm_keychart_probe.json", "w"), indent=1)

r = np.mean([o[f"s{s}"]["xq_ST64"] for s in (0, 1)]) / o["raw_ST64"]
band = "CARRIED" if r >= 0.90 else "DROPPED" if r < 0.70 else "PARTIAL"
e = np.mean([o[f"s{s}"]["addr64"] for s in (0, 1)])
f2 = lambda k: " / ".join(f"{o[f's{s}'][k]:.4f}" for s in (0, 1))
L = [f"numerals TRAIN {len(keys['TRAIN'])} TEST {len(keys['TEST'])} | probe TEST acc ST64: raw {o['raw_ST64']:.4f} | random-512 {o['rp512_ST64']:.4f} "
     f"| xq {f2('xq_ST64')} | ST8: raw {o['raw_ST8']:.4f} xq {f2('xq_ST8')}",
     f"KP-1: r = xq / raw ST64 = {r:.4f} -> {band} (CARRIED >= 0.90 / PARTIAL / DROPPED < 0.70)",
     f"KP-E: wrong picks (n {f2('n_wrong')}) in a valid candidate's ST64 chart {f2('addr64')} (mean {e:.4f}) -> "
     f"{'ADDRESS errors dominate' if e >= 0.50 else 'CHART errors dominate'} (bar 0.50)",
     f"  beside: random-wrong base rate {f2('base64')} | at ST512 {f2('addr512')}"]
print("\n".join(L)); open(".cache/gsm_keychart_probe_verdict.txt", "w").write("\n".join(L) + "\n")
