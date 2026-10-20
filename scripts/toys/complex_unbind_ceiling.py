# The analytic ceiling of rung 1T's oracle-write arm: exact owner attention, exact unbind, nearest scene entity. No training.
import sys; sys.path.insert(0, "scripts/toys"); sys.argv = [sys.argv[0], "--smoke"]
import numpy as np, complex_unbind_toy as T
import os; rng = np.random.default_rng(0); M = int(os.environ.get("CEIL_M", "48"))
E = np.exp(1j * rng.uniform(0, 2 * np.pi, (T.N_ENT, M))); P = np.exp(1j * rng.uniform(0, 2 * np.pi, (T.N_ROLE, M)))
for test in (False, True):
    fa, q, t = (x.numpy() for x in T.make_batch(2000, T.held_table(0), test, rng))
    ok = {1: [], 2: [], 3: []}
    for b in range(len(t)):
        F_ = fa[b]; mem = E[F_[:, 0]] + E[F_[:, 2]] * P[F_[:, 1] // 3]; scene = np.unique(F_[:, 0])
        cur = q[b, 0]
        for h in range(q[b, 4]):
            r = mem[F_[:, 0] == cur].mean(0) * np.conj(P[q[b, 1 + h] // 3])
            cur = scene[np.argmax((np.conj(E[scene]) @ r).real)]           # cleanup: snap to the nearest scene entity
        ok[q[b, 4]].append(cur == t[b])
    print("heldout" if test else "val    ", " ".join(f"h{d} {np.mean(v):.3f}" for d, v in ok.items()))
