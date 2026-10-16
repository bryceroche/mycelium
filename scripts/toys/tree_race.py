"""tree_race.py -- THE TREE RACE (2026-10-16; registered + bars pinned in the ledger before this ran). Zero GPU, numpy.

Bryce's register tree: routers inside, leaves that store, exact phase bind/unbind as the routing primitive. Which CODE
should a tree use? D = 256 unit phasors, branching b = 4, depth 6 = 4096 leaves. Arms:
  PRODUCT  leaf = prod_k c(node_k)                      (exact path; siblings unrelated)
  PREFIX   leaf = sum_k 2^-(k-1) c(node_k)              (coarse levels dominate similarity)
  PATHSUM  leaf = sum_k 2^-(k-1) prod_{j<=k} c(node_j)  (both: hierarchical similarity + exact path products)
  FLAT     leaf = an independent random phasor          (the no-tree baseline; needs 4096 stored codes)
Two codebook FORMS:
  PERNODE  (pinned) every tree node has its own random code (5460 codes). Note found at build time: under this form
           PREFIX and PATHSUM are statistically identical (each node's code is already unique and random).
  SHARED   (declared addition) one 4-code book per LEVEL (24 codes name 4096 leaves -- Bryce's 'small set of router
           nodes'); here PREFIX's terms collide across branches and PATHSUM's prefix products do not.
Measures A1..A7 as registered (ledger 2026-10-16, 'THE TREE RACE + THE (WHO, WHEN) CEILING').
usage: tree_race.py [--seeds 3] [--queries 400]
"""
import argparse
import itertools
import numpy as np

D, B, L = 256, 4, 6
SCHED = {"half": [2.0 ** -k for k in range(L)],      # the PINNED schedule
         "0.7": [0.7 ** k for k in range(L)],         # declared additions (build-time finding: under 'half' the leaf
         "equal": [1.0] * L}                          # level weighs 1/32, below the crosstalk floor at D = 256)
W = np.array(SCHED["half"])
PATHS = np.array(list(itertools.product(range(B), repeat=L)))          # (4096, 6), leaf index = base-4 number


def phasors(rng, n):
    return np.exp(1j * rng.uniform(0, 2 * np.pi, (n, D)))


def build(rng, form):
    """-> dict arm -> (4096, D) leaf codes; plus 'partial' arm -> list over k of (4096, D) prefix-partials for descent."""
    if form == "PERNODE":
        node = {}
        for k in range(1, L + 1):
            for p in itertools.product(range(B), repeat=k):
                node[p] = None
        keys = list(node); C = phasors(rng, len(keys)); node = dict(zip(keys, C))
        term = np.stack([np.stack([node[tuple(pp[:k + 1])] for pp in PATHS]) for k in range(L)], 1)  # (4096, L, D)
    else:
        book = phasors(rng, L * B).reshape(L, B, D)
        term = np.stack([book[k, PATHS[:, k]] for k in range(L)], 1)
    prod = np.cumprod(term, axis=1)                                     # prefix products (4096, L, D)
    codes = {
        "PRODUCT": prod[:, -1],
        "PREFIX": (W[None, :, None] * term).sum(1),
        "PATHSUM": (W[None, :, None] * prod).sum(1),
        "FLAT": phasors(rng, len(PATHS)),
    }
    partial = {  # the code of the level-k prefix node, for greedy/beam descent
        "PRODUCT": [prod[:, k] for k in range(L)],
        "PREFIX": [(W[None, :k + 1, None] * term[:, :k + 1]).sum(1) for k in range(L)],
        "PATHSUM": [(W[None, :k + 1, None] * prod[:, :k + 1]).sum(1) for k in range(L)],
    }
    return codes, partial, term, prod


def cos(Q, K):
    Qn = Q / np.linalg.norm(Q, axis=-1, keepdims=True); Kn = K / np.linalg.norm(K, axis=-1, keepdims=True)
    return np.real(Qn @ Kn.conj().T)


def noisy(rng, X, sigma):
    rms = np.sqrt((np.abs(X) ** 2).mean(-1, keepdims=True))
    return X + sigma * rms * (rng.normal(size=X.shape) + 1j * rng.normal(size=X.shape)) / np.sqrt(2)


def a1_sibling(codes):
    out = {}
    for arm, C in codes.items():
        S = cos(C[:256], C[:256])   # leaves 0..255 = the subtree (0,0,*,*,*,*)
        by = {}
        for i in range(256):
            for j in range(i + 1, 256):
                shared = next((k for k in range(L) if PATHS[i, k] != PATHS[j, k]), L)
                by.setdefault(shared, []).append(S[i, j])
        out[arm] = {s: float(np.mean(v)) for s, v in sorted(by.items())}
    return out   # shared-prefix depth -> mean cos (5 = siblings)


def a2_exact(rng, codes, nq, sigmas=(0.5, 1.0, 2.0, 3.0)):
    idx = rng.choice(len(PATHS), nq, replace=False)
    return {arm: {s: float((cos(noisy(rng, C[idx], s), C).argmax(1) == idx).mean()) for s in sigmas}
            for arm, C in codes.items()}


def a3_coarse(rng, codes, term, prod, nq):
    """only level 1 known: the query is the arm's own level-1 term; branch recall = top-1 leaf in the right level-1
    subtree (chance 0.25). Clean and sigma=1."""
    idx = rng.choice(len(PATHS), nq, replace=False)
    q = {"PRODUCT": prod[idx, 0], "PREFIX": term[idx, 0], "PATHSUM": prod[idx, 0], "FLAT": None}
    out = {}
    for arm, C in codes.items():
        if q[arm] is None:
            out[arm] = {"clean": float("nan"), "s1": float("nan")}; continue
        r = {}
        for tag, Q in (("clean", q[arm]), ("s1", noisy(rng, q[arm], 1.0))):
            top = cos(Q, C).argmax(1)
            r[tag] = float((PATHS[top, 0] == PATHS[idx, 0]).mean())
        out[arm] = r
    return out


def unit(X):
    return X / np.abs(X)


def a5_capacity(rng, codes, Ns=(5, 10, 15, 20, 25, 30, 40, 60), nv=1000):
    vals = phasors(rng, nv)
    out = {}
    for arm, C in codes.items():
        for keyset in ("random", "siblings"):
            r = {}
            for N in Ns:
                ki = rng.choice(len(PATHS), N, replace=False) if keyset == "random" else np.arange(N)
                K = unit(C[ki]); vi = rng.choice(nv, N, replace=False)
                M = (K * vals[vi]).sum(0)
                got = cos(M[None] * K.conj(), vals).argmax(1)
                r[N] = float((got == vi).mean())
            out[f"{arm}/{keyset}"] = r
    return out


def a6_registers(rng, codes, Ns=(10, 50, 200, 500), nv=1000):
    vals = phasors(rng, nv)
    C = unit(codes["PRODUCT"])
    out = {}
    for N in Ns:
        ki = rng.choice(len(PATHS), N, replace=False); vi = rng.choice(nv, N, replace=True)
        K = C[ki]
        one = (K * vals[vi]).sum(0)
        r1 = float((cos(one[None] * K.conj(), vals).argmax(1) == vi).mean())
        reg = PATHS[ki, 0] * B + PATHS[ki, 1]
        R = np.zeros((B * B, D), complex)
        np.add.at(R, reg, K * vals[vi])
        r16 = float((cos(R[reg] * K.conj(), vals).argmax(1) == vi).mean())
        out[N] = {"one_vector": r1, "16_registers": r16, "max_load": int(np.bincount(reg, minlength=16).max())}
    return out


def a7_ambiguity(rng, codes, partial, term, prod, nq):
    """query = 0.5 * (true level-1 term) + 0.5 * (another top branch's level-1 term) + the true levels 2..6."""
    out = {}
    idx = rng.choice(len(PATHS), nq, replace=False)
    for arm in ("PRODUCT", "PREFIX", "PATHSUM"):
        T = prod if arm in ("PRODUCT", "PATHSUM") else term
        alt_b = (PATHS[idx, 0] + rng.integers(1, B, nq)) % B
        alt_rows = [np.where((PATHS[:, 0] == a) & (PATHS[:, 1:] == PATHS[i, 1:]).all(1))[0][0] for a, i in zip(alt_b, idx)]
        if arm == "PRODUCT":
            Q = 0.5 * prod[idx, -1] + 0.5 * prod[alt_rows, -1]
        else:
            Q = W[0] * (0.5 * T[idx, 0] + 0.5 * T[alt_rows, 0]) + (W[None, 1:, None] * T[idx, 1:]).sum(1)
        flat = float((cos(Q, codes[arm]).argmax(1) == idx).mean())
        top = cos(Q, codes[arm]).argmax(1)
        res = {"flat_argmax": flat, "flat_top_ok": float((PATHS[top, 0] == PATHS[idx, 0]).mean())}
        for beam in (1, 2):
            right = 0; top_ok = 0
            for qi in range(nq):
                cand = [()]
                for k in range(L):
                    kids = [c + (b,) for c in cand for b in range(B)]
                    rows = [int(np.ravel_multi_index(kd + (0,) * (L - 1 - k), (B,) * L)) for kd in kids]
                    sc = cos(Q[qi][None], partial[arm][k][rows])[0]
                    cand = [kids[j] for j in np.argsort(-sc)[:beam]]
                right += cand[0] == tuple(PATHS[idx[qi]]); top_ok += cand[0][0] == PATHS[idx[qi], 0]
            res[f"beam{beam}"] = right / nq; res[f"beam{beam}_top_ok"] = top_ok / nq
        out[arm] = res
    return out


def main():
    global W
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--queries", type=int, default=400)
    a = ap.parse_args()
    for sched, form in [(s_, f_) for s_ in SCHED for f_ in ("PERNODE", "SHARED")]:
        W = np.array(SCHED[sched])
        acc = {}
        for seed in range(a.seeds):
            rng = np.random.default_rng(seed)
            codes, partial, term, prod = build(rng, form)
            res = {"A1": a1_sibling(codes), "A2": a2_exact(rng, codes, a.queries),
                   "A3": a3_coarse(rng, codes, term, prod, a.queries), "A5": a5_capacity(rng, codes),
                   "A6": a6_registers(rng, codes), "A7": a7_ambiguity(rng, codes, partial, term, prod, min(a.queries, 200))}
            acc.setdefault("runs", []).append(res)
        print(f"\n######## WEIGHTS {sched}{' (PINNED)' if sched == 'half' else ''} / FORM {form} (mean over {a.seeds} seeds) ########")
        runs = acc["runs"]

        def mean(path):
            vs = []
            for r in runs:
                x = r
                for p in path: x = x[p]
                vs.append(x)
            return float(np.nanmean(vs))
        print("A1 mean cos by shared-prefix depth (0 = different at the top ... 5 = siblings):")
        for arm in runs[0]["A1"]:
            print(f"   {arm:8s} " + " ".join(f"d{s}={mean(['A1', arm, s]):+.3f}" for s in runs[0]['A1'][arm]))
        print("A2 exact-leaf recall from a noisy full query (argmax over 4096):")
        for arm in runs[0]["A2"]:
            print(f"   {arm:8s} " + " ".join(f"s{s}={mean(['A2', arm, s]):.3f}" for s in runs[0]['A2'][arm]))
        print("A3 coarse-only query (level 1 known): top-1 leaf in the right top branch (chance 0.25):")
        for arm in runs[0]["A3"]:
            print(f"   {arm:8s} clean={mean(['A3', arm, 'clean']):.3f} sigma1={mean(['A3', arm, 's1']):.3f}")
        print("A5 single-vector capacity: recall of N key(x)value pairs (cleanup vs 1000 values):")
        for k in runs[0]["A5"]:
            print(f"   {k:18s} " + " ".join(f"N{N}={mean(['A5', k, N]):.2f}" for N in runs[0]['A5'][k]))
        print("A6 one vector vs 16 registers routed by the key's 2-level prefix (PRODUCT keys):")
        for N in runs[0]["A6"]:
            print(f"   N={N:4d} one={mean(['A6', N, 'one_vector']):.3f} 16reg={mean(['A6', N, '16_registers']):.3f} "
                  f"max_load={mean(['A6', N, 'max_load']):.0f}")
        print("A7 top-split ambiguity (50/50 top branch + true lower path): exact-leaf recall:")
        for arm in runs[0]["A7"]:
            print(f"   {arm:8s} " + " ".join(f"{k}={mean(['A7', arm, k]):.3f}" for k in runs[0]['A7'][arm]))


if __name__ == "__main__":
    main()
