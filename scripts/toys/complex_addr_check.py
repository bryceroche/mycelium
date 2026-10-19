"""complex_addr_check.py -- Opus's CHECK REGISTER on the sibling-lean table (2026-10-19; zero GPU). The same 64x8x8 tree
and lean model as complex_addr_sibling.py; a 4th register holds k = (a + 7b + 13c) mod 64 (a checksum digit). Arms on
IDENTICAL draws: no check | check with INDEPENDENT lean (Opus's model) | check with COHERENT lean (the encoder writes the
checksum of the path it believes: k(true) blended toward k(sibling path) with the largest level lean) | REPEAT-TOP (the
4th register = a second copy of level 1, leaning independently; the same width, a repetition code). Reads: beam-2 joint
and the full joint over valid paths; plus how many levels flipped in each trial (sibling score > true score)."""
import collections
import numpy as np
rng = np.random.default_rng(0); D, N, B, T = 256, 64, 8, 2000
C = [np.exp(1j * rng.uniform(0, 2 * np.pi, (N, D))) for _ in range(4)]
kid1 = np.array([rng.choice(N, B, replace=False) for _ in range(N)])
kid2 = {(a, b): rng.choice(N, B, replace=False) for a in range(N) for b in kid1[a]}
valid = np.array([(a, b, c) for a in range(N) for b in kid1[a] for c in kid2[(a, b)]])
chk = lambda a, b, c: (a + 7 * b + 13 * c) % N
K = chk(valid[:, 0], valid[:, 1], valid[:, 2])
def enc(l, true, sib, al, noise):
    v = (1 - al) * C[l][true] + al * C[l][sib]; v = v / np.abs(v).clip(1e-9)
    return v * np.exp(1j * noise)
def read(S, S3, idx3):
    J = S[0][valid[:, 0]] + S[1][valid[:, 1]] + S[2][valid[:, 2]] + (S3[idx3] if S3 is not None else 0)
    return tuple(valid[np.argmax(J)])
for amax in (0.6, 0.8, 1.0):
    R = collections.Counter(); flips = collections.Counter(); fail_by_flips = collections.Counter()
    for _ in range(T):
        a, b, c = valid[rng.integers(len(valid))]
        sa = rng.choice([x for x in range(N) if x != a]); sb = rng.choice([x for x in kid1[a] if x != b])
        sc = rng.choice([x for x in kid2[(a, b)] if x != c])
        al = rng.uniform(0, amax, 3); al3 = rng.uniform(0, amax); nz = rng.normal(0, 0.5, (5, D))
        X = [enc(0, a, sa, al[0], nz[0]), enc(1, b, sb, al[1], nz[1]), enc(2, c, sc, al[2], nz[2])]
        S = [(C[l].conj() @ X[l]).real / D for l in range(3)]
        nf = sum(S[l][t] < S[l][s] for l, t, s in ((0, a, sa), (1, b, sb), (2, c, sc))); flips[nf] += 1
        # the believed path: the sibling wherever the lean passed halfway
        bel = tuple(s if w > 0.5 else t for t, s, w in ((a, sa, al[0]), (b, sb, al[1]), (c, sc, al[2])))
        k, kb = chk(a, b, c), chk(*bel)
        k_sib_ind = rng.choice([x for x in range(N) if x != k])
        S3i = (C[3].conj() @ enc(3, k, k_sib_ind, al3, nz[3])).real / D
        S3c = (C[3].conj() @ (enc(3, k, kb, al.max(), nz[3]) if kb != k else enc(3, k, k_sib_ind, al3, nz[3]))).real / D
        Srep = (C[0].conj() @ enc(0, a, rng.choice([x for x in range(N) if x != a]), al3, nz[4])).real / D
        truth = (a, b, c)
        res = {"no check": read(S, None, None) == truth,
               "check, independent lean": read(S, S3i, K) == truth,
               "check, COHERENT lean": read(S, S3c, K) == truth,
               "repeat-top (same width)": read(S, Srep, valid[:, 0]) == truth}
        for kk, v in res.items():
            R[kk] += v
        fail_by_flips[(nf, "no check")] += not res["no check"]; fail_by_flips[(nf, "indep")] += not res["check, independent lean"]
    print(f"lean cap {amax}: " + " | ".join(f"{k} {v / T:.3f}" for k, v in R.items()))
    print(f"   levels flipped per trial: {dict(sorted(flips.items()))} | failures by flips (no check / check indep): "
          + ", ".join(f"{n} flips {fail_by_flips[(n, 'no check')]}/{fail_by_flips[(n, 'indep')]}" for n in sorted(flips)))
