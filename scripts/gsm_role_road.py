"""gsm_role_road.py -- THE ROLE TIER'S ROAD (ledger 2026-11-07 12:52, Bryce's word for the queue; zero GPU, banked BT4RB8 s0 / s1, torch CPU).

The TWO-TIER ORACLE's frame verbatim (rows, candidates, per-pointer protocol). Probe: logistic regression xq -> gold role (gsm_role_tree.roles),
fit on TRAIN numerals per seed. Arms: BASE / RR-HARD (candidates whose predicted role = the needed role (gold op, side); all if none) /
RR-SOFT (log softmax pointer + log p_probe(needed role)). Bars RR-S / RR-A / RR-1 / RR-CF.  -> .cache/gsm_role_road.json + _verdict.txt
"""
import json, os, sys

for k_, v_ in dict(BT_D="512", BT_NH="8", BT_DB="64", BT_CYCLOCK="1", BT_ZOOM="1", BT_ROLE="1", BT_T6OFF="1",
                   RD_STATES="gsmb_rb8_states.npy", BT_ROLEF="gsmb_role_feats_rb8.npy").items():
    os.environ[k_] = v_                                                         # BT4RB8's own flags
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm_breath_reader as BR          # FIRST (the 11-04 import rule)
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from gsm_role_tree import roles

torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
Dt, states = BR.TR.load(); tree = BR.TR.load_tree()
BR.RF = np.load(BR.GD.V1.C(BR.rolef_name()), mmap_mode="r")
SPL = ("ORIG", "CFNUM", "CFNAME", "TEST"); K, NCYC = BR.K, BR.NCYC
ROLES = ("*", "+", "/A", "/B", "-A", "-B", "none")


def rows(it):
    """verbatim gsm_two_tier_oracle.rows (that script runs on import): per token operand (k, role, valid set, candidates)"""
    num = [i for i, v in enumerate(it["vals"]) if v is not None]; cons, out = set(), []
    for k, (op, Aa, Bb) in enumerate(it["prog"]):
        for r, X_ in enumerate((Aa, Bb)):
            if BR.TR.first(X_)[0] == "t": out.append((k, r, set(X_["t"]), [i for i in num if i not in cons]))
        for X_ in (Aa, Bb):
            f = BR.TR.first(X_)
            if f[0] == "t": cons.add(f[1])
    return out


def need(op, r):
    return f"{op}{'AB'[r]}" if op in "-/" else op


def read(net, sp, pointers=True):
    """-> numeral xq (N, D), their gold roles (N,), and per row (problem-local numeral rows, logits on candidates, cand, valid, needed role)"""
    its = Dt[sp]; X, Y, R = [], [], []
    with torch.no_grad():
        for s0 in range(0, len(its), 128):
            part = its[s0:s0 + 128]
            t, E = BR.batch(part, states, tree, np.random.default_rng(s0)); L = t["pad"].shape[1]; B = len(part)
            for b, it in enumerate(part): E[b]["vals"] = it["vals"]
            progs, gt = BR.gold(E); xq = net.read(t); gtt = torch.as_tensor(gt)
            base = len(X); loc = []
            for b, it in enumerate(part):
                ro = roles(it); m = {}
                for i in sorted(ro): m[i] = len(X); X.append(xq[b, i].numpy()); Y.append(ROLES.index(ro[i]))
                loc.append((m, ro))
            if not pointers: continue
            R_ = [rows(it) for it in part]; P = {}
            for k in sorted({k for rr in R_ for k, *_ in rr}):
                com = np.zeros((B, K), bool); com[:, :k] = True
                nb = BR.nb_tensors(BR.notebook(E, L, progs, com, gt, np.ones((B, K), np.float32)))
                _, pa, pb, _, _ = net.breath(t, xq, nb, gtt, torch.full((B,), min(k * NCYC // K, NCYC - 1)))
                P[k] = (pa[:, k].numpy(), pb[:, k].numpy())
            for b, rr in enumerate(R_):
                m, ro = loc[b]
                for k, r, valid, cand in rr:
                    V = [i for i in cand if i in valid]
                    if not V: continue
                    nd = need(part[b]["prog"][k][0], r)
                    R.append(dict(lg=P[k][r][b][cand].astype(np.float64), cand=cand, valid=valid, need=nd, idx=[m[i] for i in cand],
                                  free=nd in {ro[v] for v in V}))
    return np.stack(X).astype(np.float32), np.array(Y), R


def arms(R, lp):
    """lp (N, 7) the probe's log-probs per numeral -> hit rates"""
    h = dict(BASE=[], HARD=[], SOFT=[], S=[])
    for row in R:
        c, lg, ri = row["cand"], row["lg"], ROLES.index(row["need"]) if row["need"] in ROLES else None
        pick = lambda sc: c[max(range(len(c)), key=lambda j: (sc[j], -c[j]))]
        h["BASE"].append(pick(lg) in row["valid"]); h["S"].append(row["free"])
        if ri is None: h["HARD"].append(h["BASE"][-1]); h["SOFT"].append(h["BASE"][-1]); continue
        pr = lp[row["idx"]]; cut = [j for j in range(len(c)) if pr[j].argmax() == ri]
        h["HARD"].append(pick([lg[j] if j in cut else -np.inf for j in range(len(c))]) in row["valid"] if cut else h["BASE"][-1])
        ls = lg - np.logaddexp.reduce(lg)
        h["SOFT"].append(pick(ls + pr[:, ri]) in row["valid"])
    return {k: float(np.mean(v)) for k, v in h.items()} | dict(n=len(R))


o = {}
for s in (0, 1):
    net = BR.make_net(); net.load_state_dict(torch.load(f".cache/gsmb_breath_BT4RB8_s{s}.pt")); net.eval()
    Xtr, Ytr, _ = read(net, "TRAIN", pointers=False)
    sc = StandardScaler().fit(Xtr); pm = LogisticRegression(max_iter=500, C=1.0).fit(sc.transform(Xtr), Ytr)
    assert list(pm.classes_) == list(range(len(ROLES))), pm.classes_
    o[f"s{s}_TRAIN"] = dict(acc=float((pm.predict(sc.transform(Xtr)) == Ytr).mean()))
    for sp in SPL:
        X, Y, R = read(net, sp); lp = pm.predict_log_proba(sc.transform(X))
        o[f"s{s}_{sp}"] = arms(R, lp) | dict(acc=float((lp.argmax(1) == Y).mean()), prior=float((Y == np.bincount(Ytr).argmax()).mean()))
        print(f"[rr] s{s} {sp} {o[f's{s}_{sp}']}", flush=True)
json.dump(o, open(".cache/gsm_role_road.json", "w"), indent=1)

g = lambda s, sp, m: o[f"s{s}_{sp}"][m]
f2 = lambda sp, m: f"{g(0, sp, m):.4f} / {g(1, sp, m):.4f}"
rs = min(g(s, sp, "S") for s in (0, 1) for sp in SPL)
ra = np.mean([g(s, "TEST", "acc") for s in (0, 1)])
d1 = [g(s, "TEST", "HARD") - g(s, "TEST", "BASE") for s in (0, 1)]
dcf = [g(s, cf, "HARD") - g(s, "ORIG", "HARD") for s in (0, 1) for cf in ("CFNUM", "CFNAME")]
L = ["           " + "  ".join(f"{sp:>17s}" for sp in SPL)]
for m in ("BASE", "HARD", "SOFT", "acc", "prior", "S"):
    L.append(f"{m:10s} " + "  ".join(f2(sp, m).rjust(17) for sp in SPL))
L.append(f"  probe TRAIN acc {g(0, 'TRAIN', 'acc'):.4f} / {g(1, 'TRAIN', 'acc'):.4f} | rows " + " ".join(f"{sp} {g(0, sp, 'n')}" for sp in SPL))
L.append(f"RR-S needed role = gold operand's role, min over splits/seeds {rs:.4f} (>= 0.95) -> {'PASS' if rs >= 0.95 else 'FAIL'}")
L.append(f"RR-A probe TEST role accuracy seed mean {ra:.4f} (>= 0.80) -> {'PASS' if ra >= 0.80 else 'FAIL'}")
L.append(f"RR-1 RR-HARD - BASE on TEST {d1[0]:+.4f} / {d1[1]:+.4f} (>= +0.020 both) -> {'PASS' if min(d1) >= 0.02 else 'FAIL'}")
L.append("RR-CF RR-HARD CF - ORIG (s0 CFNUM, s0 CFNAME, s1 CFNUM, s1 CFNAME) " + " / ".join(f"{x:+.4f}" for x in dcf)
         + f" (>= -0.030 all) -> {'PASS' if min(dcf) >= -0.03 else 'FAIL'}")
L.append(f"  beside RR-SOFT - BASE on TEST {g(0, 'TEST', 'SOFT') - g(0, 'TEST', 'BASE'):+.4f} / {g(1, 'TEST', 'SOFT') - g(1, 'TEST', 'BASE'):+.4f}")
print("\n".join(L)); open(".cache/gsm_role_road_verdict.txt", "w").write("\n".join(L) + "\n")
