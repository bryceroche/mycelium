# The analytic ceiling of rung 1T's oracle-write arm: exact owner attention, exact unbind, nearest scene entity. No training.
# Env: CEIL_M planes (48), CEIL_MAXHOP chain length (3; 5 = the depth-generalization read, 10-20), UT_K facts per record,
# CEIL_CODES=<path.npz> to replace the random entity codes with saved ones (key 'ent': (N_ENT, 2M) [re|im]; the overlap read).
import os, sys; sys.path.insert(0, "scripts/toys"); H = int(os.environ.get("CEIL_MAXHOP", "3")); os.environ["UT_MAXHOP"] = str(H)
sys.argv = [sys.argv[0], "--smoke"]
import numpy as np, complex_unbind_toy as T
rng = np.random.default_rng(0); M = int(os.environ.get("CEIL_M", "48"))
E = np.exp(1j * rng.uniform(0, 2 * np.pi, (T.N_ENT, M))); P = np.exp(1j * rng.uniform(0, 2 * np.pi, (T.N_ROLE, M)))
if os.environ.get("CEIL_CODES"):
    z = np.load(os.environ["CEIL_CODES"])["ent"]; M = z.shape[1] // 2; E = z[:, :M] + 1j * z[:, M:]
    P = np.exp(1j * rng.uniform(0, 2 * np.pi, (T.N_ROLE, M)))
for test in (False, True):
    fa, q, t = (x.numpy() for x in T.make_batch(2000, T.held_table(0), test, rng, 1, H))
    ok = {d: [] for d in range(1, H + 1)}
    for b in range(len(t)):
        F_ = fa[b]; mem = E[F_[:, 0]] + E[F_[:, 2]] * P[F_[:, 1] // 3]; scene = np.unique(F_[:, 0])
        cur = q[b, 0]
        for h in range(q[b, H + 1]):
            r = mem[F_[:, 0] == cur].mean(0) * np.conj(P[q[b, 1 + h] // 3])
            num = (np.conj(E[scene]) @ r).real / (np.linalg.norm(E[scene], axis=1) + 1e-9)
            cur = scene[np.argmax(num)]                                          # cleanup: snap to the nearest scene entity (cosine)
        ok[q[b, H + 1]].append(cur == t[b])
    print(f"M {M} K {T.KF}", "heldout" if test else "val    ", " ".join(f"h{d} {np.mean(v):.3f}" for d, v in ok.items()), flush=True)
