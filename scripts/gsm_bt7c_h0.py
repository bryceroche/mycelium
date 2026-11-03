"""BT7c-H0 THE EXECUTOR HEADROOM READ (ledger 2026-11-03): a zero-training read on banked BT weights, before BT7c is built.

The headroom corollary: measure what an executor factor COULD catch before building it. Per committed step (gold positions
k < m, the score's own units), the executor runs the predicted program left to right and flags the step's own value:
  hard  the step cannot execute (a None operand, /0, a reference past the computed slots)    -> the UNSAT kind (crosses hard)
  neg   value < 0                                                                            -> soft
  frac  value not an integer                                                                 -> soft
Gold false-flag rates come from executing the GOLD programs (TRAIN[:600], never a scored split). Commit rule: the BT6 gate, rho0
fit on TRAIN[:600] over the BT6 grid (as gsm_bt6_read). Run with the weights' own BT_* env + BT_LOAD -> .cache/gsmb_bt7c_h0_{TAG}.json
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import gsm_breath_reader as BR

GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
SMOKE = int(os.environ.get("H0_SMOKE", "0"))   # >0: that many items, two grid points, no JSON
FLAGS = ("hard", "neg", "frac")
TAG = os.path.basename(BR.LOAD).replace("gsmb_breath_", "").replace(".pt", "")
assert BR.LOAD and os.path.exists(BR.LOAD) and not BR.GATE and not BR.ORACLE
torch.set_num_threads(int(os.environ.get("TOY_THREADS", "3")))
Dt, states = BR.TR.load(); tree = BR.TR.load_tree()
if BR.ROLE: BR.RF = np.load(BR.GD.V1.C(os.environ["BT_ROLEF"]), mmap_mode="r")
net = BR.make_net(); net.load_state_dict(torch.load(BR.LOAD)); net.eval()
K, STOP, NCYC = BR.K, BR.STOP, BR.NCYC


def step_flags(prog, vals):
    """prog [(op, a, b)] -> per step dict(hard, neg, frac) on the step's own value (the executor run left to right)"""
    out, fl = [], []
    for op, a, b in prog:
        try:
            av, bv = [vals[i] if k == "t" else (out[i] if k == "s" else BR.GD.CONSTS[i]) for k, i in (a, b)]
            v = None if av is None or bv is None else {"+": av + bv, "-": av - bv, "*": av * bv, "/": av / bv}[op]
        except (ZeroDivisionError, IndexError):
            v = None
        out.append(v)
        fl.append(dict(hard=v is None, neg=v is not None and v < -1e-9, frac=v is not None and abs(v - round(v)) > 1e-6))
    return fl


def score(its, rho):
    """the BT6-gate commit loop (gsm_bt7_read.score, bp False) -> per gold step: right (step_all) + the three flags"""
    res = dict(right=[], **{f: [] for f in FLAGS}, gold={f: [] for f in FLAGS})
    with torch.no_grad():
        for s0 in range(0, len(its), 128):
            part = its[s0:s0 + 128]
            t, E = BR.batch(part, states, tree, np.random.default_rng(s0)); L = t["pad"].shape[1]; B = len(part)
            for b, it in enumerate(part): E[b]["vals"] = it["vals"]
            xq = net.read(t)
            com = np.zeros((B, K), bool); typ = np.zeros((B, K), np.int64); rad = np.zeros((B, K), np.float32)
            IA = np.zeros((B, K), np.int64); IB = np.zeros((B, K), np.int64)
            progs = [[(".", None, None)] * K for _ in range(B)]
            dec = lambda i: ("t", int(i)) if i < L else (("s", int(i - L)) if i < L + K else ("c", int(i - L - K)))
            sp = None
            for c in range(NCYC):
                nb = BR.nb_tensors(BR.notebook(E, L, progs, com, typ, rad))
                logit, pa, pb, _, p = net.breath(t, xq, nb, None, torch.full((B,), c), sp)
                if BR.SOFT and c + 1 < NCYC: sp = (logit / BR.TAU(torch.full((B,), c + 1))[:, None, None]).softmax(-1)
                ty = logit.argmax(-1).numpy(); pt = p.max(-1).values.numpy()
                ia, ib = pa.argmax(-1).numpy(), pb.argmax(-1).numpy()
                ca = pa.softmax(-1).max(-1).values.numpy(); cb = pb.softmax(-1).max(-1).values.numpy()
                conf = np.where(ty == STOP, pt, pt * ca * cb)
                for b in range(B):
                    for k in [k for k in np.argsort(-conf[b]) if not com[b, k] and conf[b, k] >= rho * (1 - c / (NCYC - 1))]:
                        com[b, k] = True; typ[b, k] = ty[b, k]; rad[b, k] = conf[b, k]; IA[b, k], IB[b, k] = ia[b, k], ib[b, k]
                        op = BR.TYPES[ty[b, k]][0] if ty[b, k] != STOP else "."
                        progs[b][k] = (op, dec(ia[b, k]), dec(ib[b, k])) if op != "." else (".", None, None)
            gprogs, gt = BR.gold(E)
            for b, (it, e) in enumerate(zip(part, E)):
                vals = it["vals"] + [None] * (L - len(it["vals"]))
                prog = []
                for st in progs[b]:
                    if st[0] == ".": break
                    prog.append(st)
                fl = step_flags(prog, vals); gfl = step_flags(gprogs[b][:e["m"]], vals)
                for k in range(e["m"]):
                    o = progs[b][k][0] == gprogs[b][k][0]
                    res["right"].append(bool(o and t["va"][b, k, IA[b, k]] and t["vb"][b, k, IB[b, k]]))
                    for f in FLAGS:
                        res[f].append(bool(k < len(fl) and fl[k][f])); res["gold"][f].append(bool(gfl[k][f]))
    return res


fit = {r: float(np.mean(score(Dt["TRAIN"][:SMOKE or 600], r)["right"])) for r in (GRID[-3:-1] if SMOKE else GRID)}
rho = max(fit, key=lambda r: fit[r])
G = score(Dt["TRAIN"][:SMOKE or 600], rho)["gold"]
out = dict(tag=TAG, load=BR.LOAD, rho0=rho, fit=fit, gold_rate_train={f: float(np.mean(G[f])) for f in FLAGS})
print(f"[h0] {TAG} rho0 {rho} | gold flag rate TRAIN[:600] " + " ".join(f"{f} {out['gold_rate_train'][f]:.4f}" for f in FLAGS), flush=True)
for split in ("ORIG", "TEST"):
    R = score(Dt[split][:SMOKE or None], rho); right = np.array(R["right"]); wrong = ~right
    o = out[split] = dict(n=int(len(right)), step=float(right.mean()))
    for f in FLAGS:
        x = np.array(R[f])
        o[f] = dict(catch=float(x[wrong].mean()), false=float(x[right].mean()), gold=float(np.mean(R["gold"][f])))
    anyf = np.array(R["neg"]) | np.array(R["frac"]) | np.array(R["hard"])
    o["any"] = dict(catch=float(anyf[wrong].mean()), false=float(anyf[right].mean()))
    print(f"[h0] {TAG} {split} n {o['n']} step {o['step']:.4f} | " + " | ".join(
        f"{f} catch {o[f]['catch']:.3f} false {o[f]['false']:.3f}" + (f" gold {o[f]['gold']:.3f}" if f != "any" else "") for f in FLAGS + ("any",)), flush=True)
if not SMOKE: json.dump(out, open(f".cache/gsmb_bt7c_h0_{TAG}.json", "w"), indent=1)
