"""pj_capacity_read.py — PJ (ledger 2026-11-12): junction-bus capacity. Register mode: exact junction recovery over all TEST
junctions per P. Problem mode: n_items = 6*n_steps (6 = 2 roles x 3 levels); consecutive steps grouped, remainder dropped;
reports step-exact rate and per-item rate (class correct at each role x level, presence correct per role). Output .cache/pj/capacity.txt."""
import ast, numpy as np
from mycelium.junction_bus import roundtrip

J = [ast.literal_eval(s) for s in np.load('.cache/pj/dump_local_k0_BEAM4+CMOTIF.npz', allow_pickle=True)['Jte']]
def ok(a, b): return a == b
def items(a, b):
    n = 0
    for x, y in zip(a, b):
        if x is None or y is None: n += (x is None and y is None) * 4; continue
        n += 1 + sum(x[k] == y[k] for k in (1, 2, 3))
    return n
L = ["PJ capacity read, n=%d TEST junctions" % len(J), "register mode (step-exact recovery):", "P     seed0   seed1"]
for P in (32, 64, 128, 256, 512):
    L.append(f"{P:<5} " + "  ".join(f"{np.mean([ok(a, b) for a, b in zip(J, roundtrip(J, P, 'register', s))]):.4f}" for s in (0, 1)))
L += ["", "problem mode (step-exact / per-item), cols=seed0, seed1", "P     n_items(steps)  " + "seed0 exact/item   seed1 exact/item"]
for P in (32, 64, 128, 256, 512):
    for n in (6, 12, 24, 48, 72):
        ns = n // 6; m = len(J) // ns * ns; Jm = J[:m]; row = []
        for s in (0, 1):
            R = roundtrip(Jm, P, 'problem', s, ns)
            row.append(f"{np.mean([ok(a, b) for a, b in zip(Jm, R)]):.4f}/{sum(items(a, b) for a, b in zip(Jm, R)) / (8 * m):.4f}")
        L.append(f"{P:<5} {n:>3}({ns:>2})        " + "   ".join(row))
open('.cache/pj/capacity.txt', 'w').write("\n".join(L) + "\n"); print("\n".join(L))
