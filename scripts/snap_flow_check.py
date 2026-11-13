"""snap_flow_check.py -- SNAP-FLOW (ledger 2026-11-13, a derivation check, not an experiment). The stack's shrinkage step
p <- (c + kappa p) / (n + kappa) is one step of the contracting flow x(t) = m + e^{-t}(x - m) toward the row's empirical m = c / n,
stopped at t = log(1 + n / kappa): the evidence sets the stopping time. Checked on every admitted edge row of the banked stack stores
(.cache/tree_stack_{local,full}_k{0..3}.db), descending root -> row exactly as the stack does; PASS iff max |shrinkage - flow| < 1e-12.
-> .cache/snap_flow_check.txt"""
import os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mycelium.stack_db import load_stack

KAPPA = 0.5


def shrink(p, c): return (c + KAPPA * p) / (c.sum() + KAPPA)


def flow(p, c):
    n = c.sum(); t = np.log1p(n / KAPPA); return c / n + np.exp(-t) * (p - c / n)


out, worst = [], 0.0
for src in ("local", "full"):
    for k in range(4):
        _, maps, E, _ = load_stack(f".cache/tree_stack_{src}_k{k}.db", f"rb8-{src}-k{k}")
        l2of = {int(c): int(maps[1][c]) for c in maps[0]}
        l3of2 = {int(maps[1][c]): int(maps[2][c]) for c in maps[0]}
        up = {1: lambda x: l2of[x], 2: lambda x: l3of2[x]}
        nops = len(next(iter(E.values()))); m, steps, nref = 0.0, 0, 0
        for key in E:
            l, kd = key[0], key[1]
            if l == 4: continue
            chain = [key]
            while chain[-1][0] < 3 and not any(isinstance(x, tuple) for x in chain[-1][2:]):   # a result-class side: its parent row needs the
                # producer's coarser inputs, so that row descends root -> itself (same algebra)
                kk = chain[-1]; chain.append((kk[0] + 1, kd) + tuple(None if x is None else up[kk[0]](x) for x in kk[2:]))
            if chain[-1][0] < 3: nref += 1
            chain.append((4, kd))
            p = q = np.ones(nops) / nops
            for kk in reversed(chain):
                if kk in E and E[kk].sum() > 0:
                    p, q = shrink(p, E[kk]), flow(q, E[kk]); m = max(m, float(np.abs(p - q).max())); steps += 1
        worst = max(worst, m); out.append(f"{src} k{k}: rows {len(E)}, descent steps {steps} (rows with a result-class side, root -> row: {nref}), max |shrinkage - flow| {m:.3e}")
out.append(f"SNAP-FLOW: max {worst:.3e} -> {'PASS' if worst < 1e-12 else 'FAIL'} (shrinkage n/(n+kappa) = the flow e^-t, t = log(1 + n/kappa))")
txt = "\n".join(out); print(txt); open(".cache/snap_flow_check.txt", "w").write(txt + "\n")
