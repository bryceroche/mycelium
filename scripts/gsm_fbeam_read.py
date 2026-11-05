"""FB THE FACTORED BEAM (ledger 2026-11-04 16:3x, word given): move the conditioning OUT of the classifier and INTO the search.

A zero-training read on banked BT weights. BEAM-1 (scripts/gsm_beam_read.py) keeps the top-2 step types per slot at its commit cycle; here
the slot keeps the top-NT types (FB_NT, default 8) of the BT's own type distribution, each with its top-2 a x top-2 b pointers, and the
left-to-right beam + circuit breaker + soft factors choose the combination (neural proposes, symbolic disposes). For the factored arms
(BT_FACT=1 BT8 / =2 BT8b) the type distribution IS log p(op) + log p(ka) + log p(kb) (no cross join trained), so the top 8 ~ the 2 x 2 x 2
factor grid; the joint arm (BT4M8) at the same NT is the width-matched placebo. rho0 and the lams are fit on TRAIN[:600] per arm per NT,
exactly as BEAM-1 fits them. Beside: oracle@BW (any finished beam
program right).
-> .cache/gsmb_fbeam_{TAG}_nt{NT}.json
"""
import json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gsm_breath_reader as BR          # FIRST (the 11-04 import rule): it sets RD_ARM=TICK, which gsm_trunk_reader reads at import
import gsm_beam_read as GB

assert BR.TR.ARM == "TICK" and BR.TR.TICK and BR.TR.NOTE, f"trunk reader arm {BR.TR.ARM}"
assert BR.CYCLOCK and BR.T6OFF, "the T6 clock must be on (BT_CYCLOCK=1 BT_T6OFF=1)"
NTY = int(os.environ.get("FB_NT", "8"))
STOP = GB.STOP


def top_alts(logit):
    """the top-NTY types and, under each, the top-2 a / b pointers (log-probs) for every slot"""
    lpt = logit.log_softmax(-1); tv, ti = lpt.topk(NTY, -1)
    alt = []
    for r in range(NTY):
        qa, qb = GB.net.ptrs(ti[..., r])
        va, xa = qa.log_softmax(-1).topk(2, -1); vb, xb = qb.log_softmax(-1).topk(2, -1)
        alt.append((va.numpy(), xa.numpy(), vb.numpy(), xb.numpy()))
    return tv.numpy(), ti.numpy(), alt


def slot_cands(b, k, dec, tv, ti, alt):
    cl = []
    for r in range(NTY):
        tr = int(ti[b, k, r])
        if tr == STOP: cl.append((float(tv[b, k, r]), None)); continue
        va, xa, vb, xb = alt[r]
        for i in range(2):
            for j in range(2):
                s = float(tv[b, k, r] + va[b, k, i] + vb[b, k, j])
                if math.isfinite(s): cl.append((s, (BR.TYPES[tr][0], dec(xa[b, k, i]), dec(xb[b, k, j]))))
    return cl


GB.top_alts, GB.slot_cands = top_alts, slot_cands       # the only change: the candidate list's type width


SM = GB.SMOKE or None
if os.environ.get("FB_FIX"):                             # gate only: "rho,li,ln" fixed (no fit) to reproduce a banked BEAM-1 read
    rho, li, ln = map(float, os.environ["FB_FIX"].split(",")); lam = (li, ln); lfit = {lam: float("nan")}
else:
    fit = {rho: float(np.mean([r["greedy"] for r in GB.collect(GB.Dt["TRAIN"][:SM or 600], rho)])) for rho in (GB.GRID[-3:-1] if SM else GB.GRID)}
    rho = max(fit, key=lambda r: fit[r])
    RT = GB.collect(GB.Dt["TRAIN"][:SM or 600], rho)
    lfit = {(li, ln): float(GB.pick(RT, li, ln).mean()) for li in GB.LAMS for ln in GB.LAMS}
    lam = max(lfit, key=lambda x: (lfit[x], -x[0] - x[1]))
out = dict(tag=GB.TAG, load=BR.LOAD, fact=os.environ.get("BT_FACT", "0"), nt=NTY, bw=GB.BW, rho0=rho, lam=lam, lam_fit_best=lfit[lam])
print(f"[fb] {GB.TAG} NT {NTY} rho0 {rho} lam {lam} TRAIN ans {lfit[lam]:.4f}", flush=True)
for split in ("ORIG", "TEST"):
    R = GB.collect(GB.Dt[split][:SM], rho)
    g = np.array([r["greedy"] for r in R], float); bm = GB.pick(R, *lam).astype(float)
    orc = np.array([any(x[2] for x in r["beam"]) or r["greedy"] for r in R], float)
    o = out[split] = dict(n=len(R), greedy=g.mean(), beam=bm.mean(), oracle=orc.mean(), per_item=bm.tolist())
    if os.environ.get("FB_FIX"): print("[fb] per item", "".join(str(int(x)) for x in bm), flush=True)
    print(f"[fb] {GB.TAG} NT {NTY} {split} n {o['n']} ans greedy {o['greedy']:.4f} | beam+solver {o['beam']:.4f} | oracle@{GB.BW} {o['oracle']:.4f}", flush=True)
json.dump(out, open(f".cache/gsmb_fbeam_{GB.TAG}_nt{NTY}{'_smoke' if SM else ''}.json", "w"), default=float)
