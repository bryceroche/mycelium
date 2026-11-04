"""BEAM THE PARSE-SIDE BEAM + CIRCUIT BREAKER (ledger 2026-11-03, word given): a zero-training read on banked BT weights.

The BT runs as always (the BT6 gate commits, the notebook carries the greedy choices). At each slot's COMMIT cycle its candidates are
stored: the top-2 step types x (under each) the top-2 a-pointers x top-2 b-pointers, scored by log p(type) + log p(a) + log p(b)
(STOP = one candidate). Then a left-to-right beam (width BW) over the slots builds whole programs, and the solver acts:
  circuit breaker  a prefix that cannot execute (None operand, /0, a dangling slot ref) is PRUNED -- a proof, crosses hard
  soft factors     the final value: - lam_i * [not an integer] - lam_n * [negative]; (lam_i, lam_n) fit on TRAIN[:600] (max ans)
Approximation (stated): the beam is post-hoc over commit-time distributions; the notebook saw the greedy choices.
Reads: greedy ans (the BT6 gate), beam ans (solver-reranked), oracle@BW (any beam program right) -> .cache/gsmb_beam_{TAG}.json
"""
import itertools, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import gsm_breath_reader as BR

GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
LAMS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
BW = int(os.environ.get("BEAM_W", "8"))
SMOKE = int(os.environ.get("BEAM_SMOKE", "0"))   # >0: that many items per split, two grid points, no JSON
TAG = os.path.basename(BR.LOAD).replace("gsmb_breath_", "").replace(".pt", "")
assert BR.LOAD and os.path.exists(BR.LOAD) and not BR.GATE and not BR.ORACLE
torch.set_num_threads(int(os.environ.get("TOY_THREADS", "3")))
Dt, states = BR.TR.load(); tree = BR.TR.load_tree()
if BR.ROLE: BR.RF = np.load(BR.GD.V1.C(BR.rolef_name()), mmap_mode="r")
net = BR.make_net(); net.load_state_dict(torch.load(BR.LOAD)); net.eval()
K, STOP, NCYC = BR.K, BR.STOP, BR.NCYC
OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b, "/": lambda a, b: a / b}


def run(prog, vals):
    """-> list of step values (None = cannot execute)"""
    out = []
    for op, a, b in prog:
        try:
            av, bv = [vals[i] if k == "t" else (out[i] if k == "s" else BR.GD.CONSTS[i]) for k, i in (a, b)]
            out.append(None if av is None or bv is None else OPS[op](av, bv))
        except (ZeroDivisionError, IndexError):
            out.append(None)
    return out


def beam(cands, vals):
    """cands: per slot [(logp, step)] (step None = STOP) -> up to BW finished (logp, prog, final value); UNSAT prefixes pruned"""
    live, done = [(0.0, [], [])], []
    for k in range(K):
        nxt = []
        for lp, prog, out in live:
            for clp, st in cands[k]:
                if st is None:
                    if prog and out[-1] is not None: done.append((lp + clp, prog, out[-1]))
                    continue
                o = run(prog + [st], vals)[-1:]
                if o[0] is None: continue                                   # the circuit breaker: a proof
                nxt.append((lp + clp, prog + [st], out + o))
        live = sorted(nxt, key=lambda x: -x[0])[:BW]
    done += [(lp, prog, out[-1]) for lp, prog, out in live]                 # programs that used every slot
    return sorted(done, key=lambda x: -x[0])[:BW]


def collect(its, rho):
    """the BT6-gate commit loop; per item: greedy prog/ans, the beam's finished programs, gold final"""
    res = []
    for s0 in range(0, len(its), 128):
        part = its[s0:s0 + 128]
        t, E, xq, L, progs, cands = gate_batch(part, rho, s0)
        for b, it in enumerate(part):
            vals = it["vals"] + [None] * (L - len(it["vals"]))
            prog = []
            for st in progs[b]:
                if st[0] == ".": break
                prog.append(st)
            g = BR.GD.execute(prog, vals)
            fin = beam(cands[b], vals)
            res.append(dict(greedy=g is not None and BR.GD.V1.near(g, it["final"]),
                            beam=[(lp, v, BR.GD.V1.near(v, it["final"])) for lp, _, v in fin]))
    return res


def slot_cands(b, k, dec, tv, ti, alt):
    """top-2 types x top-2 a x top-2 b for slot (b, k) from precomputed (tv, ti, alt) -> [(logp, step | None)]"""
    cl = []
    for r in range(2):
        tr = int(ti[b, k, r])
        if tr == STOP: cl.append((float(tv[b, k, r]), None)); continue
        va, xa, vb, xb = alt[r]
        for i, j in itertools.product(range(2), range(2)):
            s = float(tv[b, k, r] + va[b, k, i] + vb[b, k, j])
            if math.isfinite(s): cl.append((s, (BR.TYPES[tr][0], dec(xa[b, k, i]), dec(xb[b, k, j]))))
    return cl


def top_alts(logit):
    """the top-2 types and, under each, the top-2 a / b pointers (log-probs) for every slot"""
    lpt = logit.log_softmax(-1); tv, ti = lpt.topk(2, -1)
    alt = []
    for r in range(2):
        qa, qb = net.ptrs(ti[..., r])
        va, xa = qa.log_softmax(-1).topk(2, -1); vb, xb = qb.log_softmax(-1).topk(2, -1)
        alt.append((va.numpy(), xa.numpy(), vb.numpy(), xb.numpy()))
    return tv.numpy(), ti.numpy(), alt


def gate_batch(part, rho, s0):
    """one batch through the BT6-gate loop -> t, E, xq, L, greedy progs, per-slot candidates at each slot's commit cycle"""
    with torch.no_grad():
        if True:
            t, E = BR.batch(part, states, tree, np.random.default_rng(s0)); L = t["pad"].shape[1]; B = len(part)
            for b, it in enumerate(part): E[b]["vals"] = it["vals"]
            xq = net.read(t)
            com = np.zeros((B, K), bool); typ = np.zeros((B, K), np.int64); rad = np.zeros((B, K), np.float32)
            progs = [[(".", None, None)] * K for _ in range(B)]
            cands = [[None] * K for _ in range(B)]
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
                tv, ti, alt = top_alts(logit)                                                    # (B, K, 2) the top-2 types
                for b in range(B):
                    for k in [k for k in np.argsort(-conf[b]) if not com[b, k] and conf[b, k] >= rho * (1 - c / (NCYC - 1))]:
                        com[b, k] = True; typ[b, k] = ty[b, k]; rad[b, k] = conf[b, k]
                        op = BR.TYPES[ty[b, k]][0] if ty[b, k] != STOP else "."
                        progs[b][k] = (op, dec(ia[b, k]), dec(ib[b, k])) if op != "." else (".", None, None)
                        cands[b][k] = slot_cands(b, k, dec, tv, ti, alt)
    return t, E, xq, L, progs, cands


def pick(R, li, ln):
    """solver-reranked beam answer per item (fallback: greedy when the breaker pruned everything)"""
    out = []
    for r in R:
        if not r["beam"]: out.append(r["greedy"]); continue
        sc = [lp - li * (abs(v - round(v)) > 1e-6) - ln * (v < -1e-9) for lp, v, _ in r["beam"]]
        out.append(r["beam"][int(np.argmax(sc))][2])
    return np.array(out)


if __name__ == "__main__":
  fit = {}
  for rho in (GRID[-3:-1] if SMOKE else GRID):
    fit[rho] = float(np.mean([r["greedy"] for r in collect(Dt["TRAIN"][:SMOKE or 600], rho)]))
  rho = max(fit, key=lambda r: fit[r])
  RT = collect(Dt["TRAIN"][:SMOKE or 600], rho)
  lfit = {(li, ln): float(pick(RT, li, ln).mean()) for li in LAMS for ln in LAMS}
  lam = max(lfit, key=lambda x: (lfit[x], -x[0] - x[1]))                    # ties -> the smaller penalties
  out = dict(tag=TAG, load=BR.LOAD, bw=BW, rho0=rho, fit=fit, lam=lam, lam_fit_best=lfit[lam], lam_fit_zero=lfit[(0.0, 0.0)])
  print(f"[beam] {TAG} rho0 {rho} | lam (int, neg) {lam} TRAIN ans {lfit[lam]:.4f} (lam 0: {lfit[(0.0, 0.0)]:.4f})", flush=True)
  for split in ("ORIG", "TEST"):
      R = collect(Dt[split][:SMOKE or None], rho)
      g = np.array([r["greedy"] for r in R], float); bm = pick(R, *lam).astype(float); b0 = pick(R, 0.0, 0.0).astype(float)
      orc = np.array([any(x[2] for x in r["beam"]) or r["greedy"] for r in R], float)
      d = bm - g; d0 = b0 - g
      o = out[split] = dict(n=len(R), greedy=g.mean(), beam=bm.mean(), breaker_only=b0.mean(), oracle=orc.mean(),
                            diff=d.mean(), se=d.std(ddof=1) / math.sqrt(len(d)), diff_breaker=d0.mean(),
                            pruned_all=float(np.mean([not r["beam"] for r in R])), mean_finished=float(np.mean([len(r["beam"]) for r in R])))
      print(f"[beam] {TAG} {split} n {o['n']} ans greedy {o['greedy']:.4f} | breaker only {o['breaker_only']:.4f} | beam+solver {o['beam']:.4f} "
            f"({o['diff']:+.4f}, SE {o['se']:.4f}) | oracle@{BW} {o['oracle']:.4f} | all pruned {o['pruned_all']:.3f} | finished {o['mean_finished']:.1f}", flush=True)
  if not SMOKE: json.dump(out, open(f".cache/gsmb_beam_{TAG}.json", "w"), indent=1, default=float)
