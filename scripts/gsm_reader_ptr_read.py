"""gsm_reader_ptr_read.py -- THE READER PER-POINTER READ (ledger 2026-11-06 06:25, Bryce's word; zero GPU, banked weights, torch CPU).

Banked BT4RB8 s0 / s1 on the POINTER-CHART ORACLE's rows (every gold token operand; rows() as in scripts/gsm_relquery_oracle.py). Per gold
step k: the notebook holds the gold slots < k committed (the teacher's first operands: the oracle's consumed rule), cond = the gold types,
the cycle stamp floor(k NCYC / K); slot k's pa / pb. RESTR = argmax over the oracle's candidates; FREE = argmax over the full domain, right
iff a valid token. Frame RP-W (seed-mean RESTR TEST: BELOW < 0.5467 <= BETWEEN < 0.6876 <= NEAR CEILING), RP-G (RESTR TRAIN - TEST >= 0.15
both seeds). Overlap with R-ALL from .cache/gsm_relquery_rows.json (RQ_DUMP=1).  -> .cache/gsm_reader_ptr_read.json + _verdict.txt
"""
import json, os, sys

for k_, v_ in dict(BT_D="512", BT_NH="8", BT_DB="64", BT_CYCLOCK="1", BT_ZOOM="1", BT_ROLE="1", BT_T6OFF="1",
                   RD_STATES="gsmb_rb8_states.npy", BT_ROLEF="gsmb_role_feats_rb8.npy").items():
    os.environ[k_] = v_                                                         # BT4RB8's own flags (the banked run's B4 + ZOOM)
ARMT = os.environ.get("RP_TAG", "BT4RB8")                                      # (ledger 11-06 PP) the weights read: .cache/gsmb_breath_{ARMT}_s{0,1}.pt
if os.environ.get("RP_PHPTR") == "1": os.environ["BT_PHPTR"] = "1"             # (PP) the phase-bound pointers' flag
OUT = "" if ARMT == "BT4RB8" else f"_{ARMT}"                                   # the banked read keeps its file names
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm_breath_reader as BR          # FIRST (the 11-04 import rule)
import torch

torch.set_num_threads(int(os.environ.get("TOY_THREADS", "6")))
Dt, states = BR.TR.load(); tree = BR.TR.load_tree()
BR.RF = np.load(BR.GD.V1.C(BR.rolef_name()), mmap_mode="r")
SPL = ("TRAIN", "TEST"); K, NCYC = BR.K, BR.NCYC


def rows(it):
    """as the oracle: per token operand (k, role, valid set, candidates)"""
    num = [i for i, v in enumerate(it["vals"]) if v is not None]; cons, out = set(), []
    for k, (op, Aa, Bb) in enumerate(it["prog"]):
        for r, X in enumerate((Aa, Bb)):
            if BR.TR.first(X)[0] == "t": out.append((k, r, set(X["t"]), [i for i in num if i not in cons]))
        for X in (Aa, Bb):
            f = BR.TR.first(X)
            if f[0] == "t": cons.add(f[1])
    return out


def read(net, its):
    res = {"RESTR": [], "FREE": []}
    with torch.no_grad():
        for s0 in range(0, len(its), 128):
            part = its[s0:s0 + 128]
            t, E = BR.batch(part, states, tree, np.random.default_rng(s0)); L = t["pad"].shape[1]; B = len(part)
            for b, it in enumerate(part): E[b]["vals"] = it["vals"]
            progs, gt = BR.gold(E); xq = net.read(t); gtt = torch.as_tensor(gt)
            R_ = [rows(it) for it in part]; need = {k for rr in R_ for k, *_ in rr}
            P = {}
            for k in sorted(need):
                com = np.zeros((B, K), bool); com[:, :k] = True
                nb = BR.nb_tensors(BR.notebook(E, L, progs, com, gt, np.ones((B, K), np.float32)))
                _, pa, pb, _, _ = net.breath(t, xq, nb, gtt, torch.full((B,), min(k * NCYC // K, NCYC - 1)))
                P[k] = (pa[:, k].numpy(), pb[:, k].numpy())
            for b, rr in enumerate(R_):
                for k, r, valid, cand in rr:
                    lg = P[k][r][b]
                    res["RESTR"].append(bool(cand) and max(cand, key=lambda i: (lg[i], -i)) in valid)
                    res["FREE"].append(int(np.argmax(lg)) in valid)
    return {m: np.asarray(v) for m, v in res.items()}


o, rq = {}, json.load(open(".cache/gsm_relquery_rows.json"))
for s in (0, 1):
    net = BR.make_net(); net.load_state_dict(torch.load(f".cache/gsmb_breath_{ARMT}_s{s}.pt")); net.eval()
    for sp in SPL:
        r = read(net, Dt[sp]); q = np.asarray(rq[sp]); assert len(q) == len(r["RESTR"]), (len(q), len(r["RESTR"]))
        o[f"s{s}_{sp}"] = dict(n=len(q), RESTR=float(r["RESTR"].mean()), FREE=float(r["FREE"].mean()), RALL=float(q.mean()),
                               both=float((r["RESTR"] & q).mean()), reader_only=float((r["RESTR"] & ~q).mean()),
                               rall_only=float((~r["RESTR"] & q).mean()))
        o[f"s{s}_{sp}_rows"] = r["RESTR"].tolist()
        print(f"[ptr] s{s} {sp} {o[f's{s}_{sp}']}", flush=True)
a0, a1 = (np.asarray(o[f"s{s}_TEST_rows"]) for s in (0, 1)); o["seed_agree_TEST"] = float((a0 == a1).mean())
json.dump(o, open(f".cache/gsm_reader_ptr_read{OUT}.json", "w"))

m = (o["s0_TEST"]["RESTR"] + o["s1_TEST"]["RESTR"]) / 2
band = "BELOW raw cosine" if m < 0.5467 else "BETWEEN" if m < 0.6876 else "NEAR CEILING"
gaps = [o[f"s{s}_TRAIN"]["RESTR"] - o[f"s{s}_TEST"]["RESTR"] for s in (0, 1)]
f2 = lambda sp, k: " / ".join(f"{o[f's{s}_{sp}'][k]:.4f}" for s in (0, 1))
L = [f"rows TRAIN {o['s0_TRAIN']['n']} TEST {o['s0_TEST']['n']} | R-ALL TEST {o['s0_TEST']['RALL']:.4f} | ceiling 0.852"]
L += [f"  {sp}: RESTR {f2(sp, 'RESTR')} | FREE {f2(sp, 'FREE')} | both {f2(sp, 'both')} | reader only {f2(sp, 'reader_only')} | "
      f"R-ALL only {f2(sp, 'rall_only')}" for sp in SPL]
L.append(f"RP-W: RESTR TEST seed mean {m:.4f} -> {band} (BELOW < 0.5467 <= BETWEEN < 0.6876 <= NEAR CEILING)")
L.append(f"RP-G: RESTR TRAIN - TEST {gaps[0]:+.4f} / {gaps[1]:+.4f} (bar >= 0.15 both) -> {'PASS' if min(gaps) >= 0.15 else 'FAIL'}")
L.append(f"  beside: the seeds agree on {o['seed_agree_TEST']:.4f} of TEST rows")
print("\n".join(L)); open(f".cache/gsm_reader_ptr_read{OUT}_verdict.txt", "w").write("\n".join(L) + "\n")
