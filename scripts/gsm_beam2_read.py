"""BEAM-2 THE ALTERNATION PROPER (ledger 2026-11-03, word given for the whole queue): a zero-training read on banked BT weights.

Round 0 = BEAM-1 (gsm_beam_read: the BT6-gate breaths, commit-time candidates, beam + circuit breaker + solver rerank with BEAM-1's
own TRAIN-fit rho0 / lambda from .cache/gsmb_beam_{TAG}.json). Each further round is the alternation:
  WRITE-BACK   the solver's chosen program is written into the notebook (every slot committed, the executor's certificates)
  RE-READ      for each slot k: k masked, every other slot committed to the winner (the training distribution: masked diffusion
               with the rest committed), one breath at the last cycle's stamp -> slot k's fresh top-2 x 2 x 2 candidates
               + the winner's own step scored under the fresh distribution
  DISPOSE      beam + circuit breaker over the fresh candidates; the solver reranks (lambda2 fit on TRAIN[:600], never a scored split)
Rounds: 1 pinned, 2 read beside. -> .cache/gsmb_beam2_{TAG}.json
"""
import json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import gsm_beam_read as BM

BR, net, K, STOP, NCYC = BM.BR, BM.net, BM.K, BM.STOP, BM.NCYC
ROUNDS = 2
B1 = json.load(open(f".cache/gsmb_beam_{BM.TAG}.json")); RHO = B1["rho0"]; LAM1 = tuple(B1["lam"])


def choose(fin, li, ln):
    """the solver's pick among finished programs -> (prog, value) | None"""
    if not fin: return None
    sc = [lp - li * (abs(v - round(v)) > 1e-6) - ln * (v < -1e-9) for lp, _, v in fin]
    i = int(np.argmax(sc)); return fin[i][1], fin[i][2]


def enc(st, L):
    k, i = st
    return i if k == "t" else (L + i if k == "s" else L + K + i)


def rebreath(t, E, xq, L, winners, dec):
    """winners: per item a program (list of steps) -> per item, per slot fresh candidates (the winner's step included)"""
    B = len(E)
    progs = [list(w) + [(".", None, None)] * (K - len(w)) for w in winners]
    typ = np.array([[BR.step_type(*st) if st[0] != "." else STOP for st in p] for p in progs])
    cands = [[None] * K for _ in range(B)]
    with torch.no_grad():
        for k in range(K):
            com = np.ones((B, K), bool); com[:, k] = False
            nb = BR.nb_tensors(BR.notebook(E, L, progs, com, typ, np.ones((B, K), np.float32)))
            logit, _, _, _, _ = net.breath(t, xq, nb, None, torch.full((B,), NCYC - 1), None)
            tv, ti, alt = BM.top_alts(logit)
            lpt = logit.log_softmax(-1).numpy()
            qa, qb = (q.log_softmax(-1).numpy() for q in net.ptrs(torch.as_tensor(typ)))
            for b in range(B):
                cl = BM.slot_cands(b, k, dec, tv, ti, alt); st = progs[b][k]
                if st[0] == ".": w = (float(lpt[b, k, STOP]), None)
                else: w = (float(lpt[b, k, typ[b, k]] + qa[b, k, enc(st[1], L)] + qb[b, k, enc(st[2], L)]), st)
                if math.isfinite(w[0]) and all(c[1] != w[1] for c in cl): cl.append(w)
                cands[b][k] = cl
    return cands


def collect(its):
    """per item: greedy ans, BEAM-1 ans, and per round the finished programs [(logp, value, right)]"""
    res = []
    for s0 in range(0, len(its), 128):
        part = its[s0:s0 + 128]
        t, E, xq, L, progs, cands = BM.gate_batch(part, RHO, s0)
        dec = lambda i: ("t", int(i)) if i < L else (("s", int(i - L)) if i < L + K else ("c", int(i - L - K)))
        vals = [it["vals"] + [None] * (L - len(it["vals"])) for it in part]
        greedy = []
        for b in range(len(part)):
            prog = []
            for st in progs[b]:
                if st[0] == ".": break
                prog.append(st)
            greedy.append(prog)
        fins = [[BM.beam(cands[b], vals[b]) for b in range(len(part))]]
        for r in range(ROUNDS):
            win = []
            for b in range(len(part)):
                ch = choose(fins[-1][b], *LAM1); win.append(ch[0] if ch else greedy[b])
            nc = rebreath(t, E, xq, L, win, dec)
            fins.append([BM.beam(nc[b], vals[b]) for b in range(len(part))])
        for b, it in enumerate(part):
            g = BR.GD.execute(greedy[b], vals[b])
            res.append(dict(greedy=g is not None and BR.GD.V1.near(g, it["final"]),
                            rounds=[[(lp, v, BR.GD.V1.near(v, it["final"])) for lp, _, v in f[b]] for f in fins]))
    return res


def pick(R, r, li, ln):
    out = []
    for x in R:
        f = x["rounds"][r]
        if not f: out.append(x["greedy"]); continue
        sc = [lp - li * (abs(v - round(v)) > 1e-6) - ln * (v < -1e-9) for lp, v, _ in f]
        out.append(f[int(np.argmax(sc))][2])
    return np.array(out, float)


if __name__ == "__main__":
    SM = BM.SMOKE
    RT = collect(BM.Dt["TRAIN"][:SM or 600])
    lfit = {(li, ln): float(pick(RT, 1, li, ln).mean()) for li in BM.LAMS for ln in BM.LAMS}
    lam2 = max(lfit, key=lambda x: (lfit[x], -x[0] - x[1]))
    out = dict(tag=BM.TAG, rho0=RHO, lam1=LAM1, lam2=lam2, lam2_fit=lfit[lam2], rounds=ROUNDS)
    print(f"[beam2] {BM.TAG} rho0 {RHO} lam1 {LAM1} | lam2 {lam2} TRAIN ans r1 {lfit[lam2]:.4f} (BEAM-1 on TRAIN {pick(RT, 0, *LAM1).mean():.4f})", flush=True)
    for split in ("ORIG", "TEST"):
        R = collect(BM.Dt[split][:SM or None])
        g = np.array([x["greedy"] for x in R], float); b1 = pick(R, 0, *LAM1)
        rr = [pick(R, r, *lam2) for r in range(1, ROUNDS + 1)]
        orc = [np.array([any(y[2] for y in x["rounds"][r]) for x in R], float).mean() for r in range(ROUNDS + 1)]
        d = rr[0] - b1
        o = out[split] = dict(n=len(R), greedy=g.mean(), beam1=b1.mean(), **{f"round{r + 1}": x.mean() for r, x in enumerate(rr)},
                              diff=d.mean(), se=d.std(ddof=1) / math.sqrt(len(d)), oracle_by_round=orc)
        print(f"[beam2] {BM.TAG} {split} n {o['n']} ans greedy {o['greedy']:.4f} | BEAM-1 {o['beam1']:.4f} | round 1 {o['round1']:.4f} "
              f"({o['diff']:+.4f}, SE {o['se']:.4f}) | round 2 {o['round2']:.4f} | oracle@8 by round " + " ".join(f"{x:.4f}" for x in orc), flush=True)
    if not SM: json.dump(out, open(f".cache/gsmb_beam2_{BM.TAG}.json", "w"), indent=1, default=float)
