"""AG part 1, THE BEAM DUMP (ledger 2026-11-04, word given): a zero-training read on banked BT weights (BR.LOAD).

Runs the BEAM-1 read (scripts/gsm_beam_read.py: the BT6-gate commit loop, top-2 x 2 x 2 candidates per slot at its commit cycle, the
left-to-right beam with the circuit breaker) and DUMPS per item what a cross-view selector needs: the greedy answer value and every
finished beam program's (log p, final value, right?). rho0 and the soft-factor lams are fit on TRAIN[:600] exactly as BEAM-1 fits them.
The selectors themselves live in scripts/gsm_agree_read.py.
-> .cache/gsmb_agdump_{TAG}.json
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gsm_beam_read as GB

BR, Dt = GB.BR, GB.Dt


def dump(its, rho):
    """per item: greedy value (None = no program executes), beam [(logp, value, right?)], gold final"""
    res = []
    for s0 in range(0, len(its), 128):
        part = its[s0:s0 + 128]
        t, E, xq, L, progs, cands = GB.gate_batch(part, rho, s0)
        for b, it in enumerate(part):
            vals = it["vals"] + [None] * (L - len(it["vals"]))
            prog = []
            for st in progs[b]:
                if st[0] == ".": break
                prog.append(st)
            g = BR.GD.execute(prog, vals)
            res.append(dict(g=None if g is None else float(g), g_ok=bool(g is not None and BR.GD.V1.near(g, it["final"])),
                            beam=[(float(lp), float(v), bool(BR.GD.V1.near(v, it["final"]))) for lp, _, v in GB.beam(cands[b], vals)]))
    return res


SM = GB.SMOKE or None                                                     # BEAM_SMOKE=n: n items per split, two grid points
fit = {rho: float(np.mean([r["greedy"] for r in GB.collect(Dt["TRAIN"][:SM or 600], rho)])) for rho in (GB.GRID[-3:-1] if SM else GB.GRID)}
rho = max(fit, key=lambda r: fit[r])
RT = GB.collect(Dt["TRAIN"][:SM or 600], rho)
lfit = {(li, ln): float(GB.pick(RT, li, ln).mean()) for li in GB.LAMS for ln in GB.LAMS}
lam = max(lfit, key=lambda x: (lfit[x], -x[0] - x[1]))
out = dict(tag=GB.TAG, load=BR.LOAD, bw=GB.BW, rho0=rho, lam=lam)
print(f"[agdump] {GB.TAG} rho0 {rho} lam {lam} TRAIN ans BEAM-1 {lfit[lam]:.4f}", flush=True)
for split in ("ORIG", "TEST"):
    out[split] = dump(Dt[split][:SM], rho)
    print(f"[agdump] {GB.TAG} {split} n {len(out[split])} greedy {np.mean([r['g_ok'] for r in out[split]]):.4f}", flush=True)
json.dump(out, open(f".cache/gsmb_agdump_{GB.TAG}{'_smoke' if SM else ''}.json", "w"))
