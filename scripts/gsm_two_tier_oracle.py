"""gsm_two_tier_oracle.py -- THE TWO-TIER ORACLE (ledger 2026-11-07 06:35, Bryce's word; zero GPU, banked BT4RB8 s0 / s1, torch CPU).

The READER PER-POINTER protocol (as gsm_keychart_probe.read) on ORIG / CFNUM / CFNAME / TEST. Per token-operand row with a valid candidate
among the unconsumed numerals: BASE = RESTR pick (argmax, ties to the lower position); the business tier given = the gold operand's role
(gsm_role_tree.roles, every numeral's role known) cuts the candidates to the same-role numerals; T2-QK = the pointer's argmax among them;
T2-ORD = the first same-role numeral. Bars TT-1 / TT-CF / TT-O.  -> .cache/gsm_two_tier_oracle.json + _verdict.txt
"""
import json, os, sys

for k_, v_ in dict(BT_D="512", BT_NH="8", BT_DB="64", BT_CYCLOCK="1", BT_ZOOM="1", BT_ROLE="1", BT_T6OFF="1",
                   RD_STATES="gsmb_rb8_states.npy", BT_ROLEF="gsmb_role_feats_rb8.npy").items():
    os.environ[k_] = v_                                                         # BT4RB8's own flags
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm_breath_reader as BR          # FIRST (the 11-04 import rule)
import torch
from gsm_role_tree import roles

torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
Dt, states = BR.TR.load(); tree = BR.TR.load_tree()
BR.RF = np.load(BR.GD.V1.C(BR.rolef_name()), mmap_mode="r")
SPL = ("ORIG", "CFNUM", "CFNAME", "TEST"); K, NCYC = BR.K, BR.NCYC


def rows(it):
    """as the POINTER-CHART ORACLE: per token operand (k, role, valid set, candidates)"""
    num = [i for i, v in enumerate(it["vals"]) if v is not None]; cons, out = set(), []
    for k, (op, Aa, Bb) in enumerate(it["prog"]):
        for r, X_ in enumerate((Aa, Bb)):
            if BR.TR.first(X_)[0] == "t": out.append((k, r, set(X_["t"]), [i for i in num if i not in cons]))
        for X_ in (Aa, Bb):
            f = BR.TR.first(X_)
            if f[0] == "t": cons.add(f[1])
    return out


def read(net, sp):
    its = Dt[sp]; hit = dict(BASE=[], T2QK=[], T2ORD=[]); ncut = []
    with torch.no_grad():
        for s0 in range(0, len(its), 128):
            part = its[s0:s0 + 128]
            t, E = BR.batch(part, states, tree, np.random.default_rng(s0)); L = t["pad"].shape[1]; B = len(part)
            for b, it in enumerate(part): E[b]["vals"] = it["vals"]
            progs, gt = BR.gold(E); xq = net.read(t); gtt = torch.as_tensor(gt)
            R_ = [rows(it) for it in part]; P = {}
            for k in sorted({k for rr in R_ for k, *_ in rr}):
                com = np.zeros((B, K), bool); com[:, :k] = True
                nb = BR.nb_tensors(BR.notebook(E, L, progs, com, gt, np.ones((B, K), np.float32)))
                _, pa, pb, _, _ = net.breath(t, xq, nb, gtt, torch.full((B,), min(k * NCYC // K, NCYC - 1)))
                P[k] = (pa[:, k].numpy(), pb[:, k].numpy())
            for b, rr in enumerate(R_):
                ro = roles(part[b])
                for k, r, valid, cand in rr:
                    V = [i for i in cand if i in valid]
                    if not V: continue
                    lg = P[k][r][b]; pick = lambda c: max(c, key=lambda i: (lg[i], -i))
                    cut = [i for i in cand if ro[i] in {ro[v] for v in V}]; ncut.append(len(cut))
                    hit["BASE"].append(pick(cand) in valid); hit["T2QK"].append(pick(cut) in valid); hit["T2ORD"].append(min(cut) in valid)
    return {m: float(np.mean(v)) for m, v in hit.items()} | dict(n=len(ncut), mean_cut=float(np.mean(ncut)), mean_cand=None)


o = {}
for s in (0, 1):
    net = BR.make_net(); net.load_state_dict(torch.load(f".cache/gsmb_breath_BT4RB8_s{s}.pt")); net.eval()
    for sp in SPL:
        o[f"s{s}_{sp}"] = read(net, sp); print(f"[tt] s{s} {sp} {o[f's{s}_{sp}']}", flush=True)
json.dump(o, open(".cache/gsm_two_tier_oracle.json", "w"), indent=1)
g = lambda s, sp, m: o[f"s{s}_{sp}"][m]
d1 = [g(s, "TEST", "T2QK") - g(s, "TEST", "BASE") for s in (0, 1)]
dcf = [g(s, cf, "T2QK") - g(s, "ORIG", "T2QK") for s in (0, 1) for cf in ("CFNUM", "CFNAME")]
L = ["           " + "  ".join(f"{sp:>22s}" for sp in SPL)]
for m in ("BASE", "T2QK", "T2ORD"):
    L.append(f"{m:10s} " + "  ".join(f"{g(0, sp, m):.4f} / {g(1, sp, m):.4f}".rjust(22) for sp in SPL))
L.append("rows / mean same-role cut " + "  ".join(f"{sp} {o[f's0_{sp}']['n']} / {o[f's0_{sp}']['mean_cut']:.2f}" for sp in SPL))
L.append(f"TT-1 T2-QK - BASE on TEST {d1[0]:+.4f} / {d1[1]:+.4f} (>= +0.050 both) -> {'PASS' if min(d1) >= 0.05 else 'FAIL'}")
L.append(f"TT-CF T2-QK CF - ORIG (s0 CFNUM, s0 CFNAME, s1 CFNUM, s1 CFNAME) " + " / ".join(f"{x:+.4f}" for x in dcf)
         + f" (>= -0.020 all) -> {'PASS' if min(dcf) >= -0.02 else 'FAIL'}")
L.append(f"TT-O T2-ORD on TEST {g(0, 'TEST', 'T2ORD'):.4f} (>= 0.80) -> {'PASS' if g(0, 'TEST', 'T2ORD') >= 0.80 else 'FAIL'}")
print("\n".join(L)); open(".cache/gsm_two_tier_oracle_verdict.txt", "w").write("\n".join(L) + "\n")
