"""gsm_zq_census.py -- THE ZOOMING QUERY's census (ledger 2026-11-06, the pre/post knob law; banked with the ZQ verdict). Zero GPU.

Banked ZQ s0 / s1 (BT_ZQ=1, BT4RB8's flags) on TRAIN[:256] and TEST[:256] with the READER PER-POINTER protocol (as gsm_pp_census.py). Exact
ablation: the centred chart term is 0 at zck = 0, so base = zck zeroed, ZQ = full - base. Over the token-operand rows (the oracle's rows(),
numerals the kind mask leaves open): the std of base and of ZQ, and the ZQ margin = mean ZQ on valid candidates - mean ZQ on the other open
candidates (nats; > 0 = the chart term favours the gold), beside the same margin for base.  -> .cache/gsm_zq_census.json + printed table
"""
import json, math, os, sys

for k_, v_ in dict(BT_D="512", BT_NH="8", BT_DB="64", BT_CYCLOCK="1", BT_ZOOM="1", BT_ROLE="1", BT_T6OFF="1", BT_ZQ="1",
                   RD_STATES="gsmb_rb8_states.npy", BT_ROLEF="gsmb_role_feats_rb8.npy").items():
    os.environ[k_] = v_
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm_breath_reader as BR          # FIRST (the 11-04 import rule)
import torch

torch.set_num_threads(int(os.environ.get("TOY_THREADS", "6")))
Dt, states = BR.TR.load(); tree = BR.TR.load_tree()
BR.RF = np.load(BR.GD.V1.C(BR.rolef_name()), mmap_mode="r")
K, NCYC = BR.K, BR.NCYC; OPEN = -1e4 / math.sqrt(BR.D) + 1


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


def logits(net, t, E, xq, progs, gt, k, B, L):
    com = np.zeros((B, K), bool); com[:, :k] = True
    nb = BR.nb_tensors(BR.notebook(E, L, progs, com, gt, np.ones((B, K), np.float32)))
    _, pa, pb, _, _ = net.breath(t, xq, nb, torch.as_tensor(gt), torch.full((B,), min(k * NCYC // K, NCYC - 1)))
    return pa[:, k].numpy(), pb[:, k].numpy()


o = {}
for s in (0, 1):
    net = BR.make_net(); net.load_state_dict(torch.load(f".cache/gsmb_breath_ZQ_s{s}.pt")); net.eval(); z0 = net.zck.detach().clone()
    for sp in ("TRAIN", "TEST"):
        A = dict(base=[], zq=[]); M = dict(base=[], zq=[])
        with torch.no_grad():
            for s0 in range(0, 256, 128):
                part = Dt[sp][s0:s0 + 128]
                t, E = BR.batch(part, states, tree, np.random.default_rng(s0)); L = t["pad"].shape[1]; B = len(part)
                for b, it in enumerate(part): E[b]["vals"] = it["vals"]
                progs, gt = BR.gold(E); xq = net.read(t)
                R_ = [rows(it) for it in part]
                for k in sorted({k for rr in R_ for k, *_ in rr}):
                    net.zck.copy_(z0); full = logits(net, t, E, xq, progs, gt, k, B, L)
                    net.zck.zero_(); base = logits(net, t, E, xq, progs, gt, k, B, L); net.zck.copy_(z0)
                    for b, rr in enumerate(R_):
                        for kk, r, valid, cand in rr:
                            if kk != k: continue
                            c = [i for i in cand if base[r][b, i] > OPEN]; V = [i for i in c if i in valid]; W = [i for i in c if i not in valid]
                            bz = {"base": base[r][b], "zq": full[r][b] - base[r][b]}
                            for n_, v_ in bz.items():
                                A[n_].extend(v_[c].tolist())
                                if V and W: M[n_].append(float(v_[V].mean() - v_[W].mean()))
        o[f"s{s}_{sp}"] = {f"std_{n_}": float(np.std(A[n_])) for n_ in A} | {f"margin_{n_}": float(np.mean(M[n_])) for n_ in M} | dict(n_rows=len(M["zq"]))
        print(f"[zqc] s{s} {sp}: {o[f's{s}_{sp}']}", flush=True)
json.dump(o, open(".cache/gsm_zq_census.json", "w"), indent=1)
