"""Layout A under COHERENT noise (Opus #2) and the read policies (Opus #3). A real tree: 64 top codes, 8 children per
parent, 8 grandchildren per child = 4,096 valid addresses (of 64^3 = 262,144 code combinations). The encoder's error at
each level leans toward a plausible SIBLING (top: another top code; lower: a sibling under the same parent) with strength
alpha ~ U(0, amax), independently per level, plus 0.5 rad random phase noise."""
import numpy as np
rng = np.random.default_rng(0); D, N, B, T = 256, 64, 8, 2000
C = [np.exp(1j * rng.uniform(0, 2 * np.pi, (N, D))) for _ in range(3)]
kid1 = np.array([rng.choice(N, B, replace=False) for _ in range(N)])            # children of each top code (level-2 codes)
kid2 = {(a, b): rng.choice(N, B, replace=False) for a in range(N) for b in kid1[a]}
valid = np.array([(a, b, c) for a in range(N) for b in kid1[a] for c in kid2[(a, b)]])
def enc(l, true, sib, al):
    v = (1 - al) * C[l][true] + al * C[l][sib]; v = v / np.abs(v).clip(1e-9)
    return v * np.exp(1j * rng.normal(0, 0.5, D))
for amax in (0.4, 0.6, 0.8, 1.0):
    R = dict(free=0, greedy=0, beam2=0, joint=0)
    for _ in range(T):
        a, b, c = valid[rng.integers(len(valid))]
        sa = rng.choice([x for x in range(N) if x != a]); sb = rng.choice([x for x in kid1[a] if x != b])
        sc = rng.choice([x for x in kid2[(a, b)] if x != c])
        X = [enc(0, a, sa, rng.uniform(0, amax)), enc(1, b, sb, rng.uniform(0, amax)), enc(2, c, sc, rng.uniform(0, amax))]
        S = [(C[l].conj() @ X[l]).real / D for l in range(3)]                       # every register read at once
        R["free"] += [int(np.argmax(s)) for s in S] == [a, b, c]                     # independent snaps, no tree
        g0 = int(np.argmax(S[0])); g1 = kid1[g0][np.argmax(S[1][kid1[g0]])]           # greedy descent (the tree's death)
        g2 = kid2[(g0, g1)][np.argmax(S[2][kid2[(g0, g1)]])]; R["greedy"] += (g0, g1, g2) == (a, b, c)
        best = None                                                                  # beam-2 per level, joint score
        for t0 in np.argsort(-S[0])[:2]:
            for t1 in kid1[t0][np.argsort(-S[1][kid1[t0]])[:2]]:
                for t2 in kid2[(t0, t1)][np.argsort(-S[2][kid2[(t0, t1)]])[:2]]:
                    sc_ = S[0][t0] + S[1][t1] + S[2][t2]
                    if best is None or sc_ > best[0]: best = (sc_, (t0, t1, t2))
        R["beam2"] += best[1] == (a, b, c)
        J = S[0][valid[:, 0]] + S[1][valid[:, 1]] + S[2][valid[:, 2]]                 # the full joint read over valid paths
        R["joint"] += tuple(valid[np.argmax(J)]) == (a, b, c)
    print(f"sibling lean alpha ~ U(0, {amax}) | " + " | ".join(f"{k} {v / T:.3f}" for k, v in R.items()))
