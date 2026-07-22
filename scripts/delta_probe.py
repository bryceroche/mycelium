"""The delta-probe — the atlas's gate-2 instrument (gut #53).

Reads the banked kind-centroid constellation and asks whether it knows a
tree. Kill-only; the registered lean is FLAT-ISH at two-floor vintage.
Bars pinned in the ledger BEFORE this ran. Zero GPU; single vintage only
(never mix generations' coordinates).

Ground-truth sapling (mg2 expansion edges):
    root -> {frac, macro, given, sel, mod, pct}
    frac -> {rel_mul, fdiv}
    macro -> {rel_add}
"""
import itertools, json, os, random
import numpy as np

CENTROIDS = os.environ.get("DELTA_CENTROIDS", ".cache/monitor_centroids_gen16.npz")
OUT = os.environ.get("DELTA_OUT", ".cache/delta_probe_gen16.json")

PARENT = {  # child -> parent in the sapling; absent = child of root
    "rel_mul": "frac", "fdiv": "frac", "rel_add": "macro",
}
CROWNS = ["frac", "macro"]


def tree_dist(a, b):
    def path_to_root(k):
        p = [k]
        while p[-1] in PARENT:
            p.append(PARENT[p[-1]])
        p.append("root")
        return p
    pa, pb = path_to_root(a), path_to_root(b)
    for i, anc in enumerate(pa):
        if anc in pb:
            return i + pb.index(anc)
    raise AssertionError


def gromov_delta(D):
    """Max four-point-condition defect over all quadruples."""
    n = D.shape[0]
    delta = 0.0
    for i, j, k, l in itertools.combinations(range(n), 4):
        s = sorted([D[i, j] + D[k, l], D[i, k] + D[j, l], D[i, l] + D[j, k]])
        delta = max(delta, (s[2] - s[1]) / 2.0)
    return delta


def main():
    z = np.load(CENTROIDS, allow_pickle=True)
    kinds = list(z.files)
    X = np.stack([z[k].astype(np.float64) for k in kinds])
    n = len(kinds)
    assert n == 9, f"expected 9 kinds, got {n}"

    # centered-cosine distances (the house's standard read on this space)
    Xc = X - X.mean(0, keepdims=True)
    Xn = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
    D = 1.0 - Xn @ Xn.T
    np.fill_diagonal(D, 0.0)

    # (i) Gromov delta / diameter
    delta = gromov_delta(D)
    diam = D.max()
    ratio = delta / diam

    # (ii) cophenetic correlation vs sapling, label-permutation null
    T = np.array([[tree_dist(a, b) if a != b else 0 for b in kinds] for a in kinds], float)
    iu = np.triu_indices(n, 1)
    obs = np.corrcoef(D[iu], T[iu])[0, 1]
    rng = random.Random(53)
    null = []
    for _ in range(5000):
        perm = list(range(n)); rng.shuffle(perm)
        Tp = T[np.ix_(perm, perm)]
        null.append(np.corrcoef(D[iu], Tp[iu])[0, 1])
    pct = 100.0 * sum(1 for v in null if v < obs) / len(null)

    # (iii) parenthood rank read
    ranks = {}
    for crown, children in [("frac", ["rel_mul", "fdiv"]), ("macro", ["rel_add"])]:
        ci = kinds.index(crown)
        order = sorted((j for j in range(n) if j != ci), key=lambda j: D[ci, j])
        ranks[crown] = {kinds[j]: r + 1 for r, j in enumerate(order) if kinds[j] in children}

    # (iv) radius footnote — raw norms, reported only
    norms = {k: float(np.linalg.norm(z[k].astype(np.float64))) for k in kinds}
    crown_mean = float(np.mean([norms[k] for k in CROWNS]))
    prime_mean = float(np.mean([norms[k] for k in kinds if k not in CROWNS]))

    frac_ok = set(ranks["frac"].values()) <= {1, 2}
    macro_ok = ranks["macro"].get("rel_add") == 1
    surprise = pct >= 95.0 and frac_ok and macro_ok

    print(f"[delta] substrate {CENTROIDS} | kinds {kinds}")
    print(f"[delta] (i) Gromov delta {delta:.4f} / diam {diam:.4f} = {ratio:.3f}  (lean bar: >~0.25 flat-ish)")
    print(f"[delta] (ii) cophenetic corr {obs:.3f} | permutation percentile {pct:.1f}  (surprise bar: >=95)")
    print(f"[delta] (iii) parenthood ranks: frac children {ranks['frac']} (top-2? {frac_ok}) | macro child {ranks['macro']} (top-1? {macro_ok})")
    print(f"[delta] (iv) radius footnote: crown mean {crown_mean:.3f} vs prime mean {prime_mean:.3f} (reported only)")
    if surprise:
        print("[delta] VERDICT: THE SAPLING KNOWS ITS PARENTS — flagged accrual on the gate ledger (NOT a gate opening)")
    else:
        print("[delta] VERDICT: FLAT-ISH AS PINNED — baseline banked; the instrument accrues per entourage")

    json.dump({"substrate": CENTROIDS, "kinds": kinds, "delta": delta, "diam": diam,
               "delta_over_diam": ratio, "cophenetic": obs, "percentile": pct,
               "ranks": ranks, "norms": norms, "crown_mean_norm": crown_mean,
               "prime_mean_norm": prime_mean, "surprise": bool(surprise)},
              open(OUT, "w"), indent=1)
    print(f"[delta] banked -> {OUT}")


if __name__ == "__main__":
    main()
