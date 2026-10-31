"""gsm_role_tree.py -- G-TREE (ledger 2026-10-31, Bryce: "building the database from GSM8K ... aggressively memorizing and overfitting
the data in the Tree"). Zero-GPU. The tree memorizes CERTIFIED facts: for every numeral word (vals not None) of every TRAIN problem,
the ROLE the gold program gives it (the op and side of its first use, or 'none' = a distractor), Welford counts keyed by text context:
  L0 (tok, next)  L1 (prev, tok, next)  L2 (prev, tok, next, next2)   PLACEBO L1 = (tok, next, a random word of the same problem)
Backoff L2 > L1 > L0 (N >= 2), else the prior. TRAIN rows are LEAVE-ONE-OUT (the problem's own counts removed); TEST plain;
CFNUM/CFNAME read as memorization transfer (their source problems are in the tree by design: overfit the tree, the CF answer is
recomputed from new values). Writes .cache/gsm_role_tree.txt."""
import collections, json, os, sys, zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import gsm_bench_data as GD

MINN = 2


def roles(it):
    """-> {numeral word index: role} for every numeral word"""
    out = {}
    for i, v in enumerate(it["vals"]):
        if v is None: continue
        r = "none"
        for op, A, B in it["prog"]:
            if i in A["t"]: r = f"{op}A" if op in "-/" else f"{op}"; break
            if i in B["t"]: r = f"{op}B" if op in "-/" else f"{op}"; break
        out[i] = r
    return out


REL = os.environ.get("GT_KEYS") == "REL"   # G-TREE2: the RELATIONAL key (the numeral's unit x the question's target noun)


def qnoun(it):
    """the word after 'how many' / 'how much' in the last sentence ('#' if none)"""
    t, sid = it["toks"], it["sid"]; last = max(sid)
    for j in range(len(t) - 2):
        if sid[j] == last and t[j] == "how" and t[j + 1] in ("many", "much"): return t[j + 2]
    return "#"


def keys(it, i, rng):
    t = it["toks"]; g = lambda j: t[j] if 0 <= j < len(t) else "#"
    if REL:
        q = qnoun(it)
        return {"L0": (g(i), g(i + 1)), "L1": (g(i), g(i + 1), q), "L2": (g(i - 1), g(i), g(i + 1), q),
                "L1P": (g(i), g(i + 1), t[int(rng.integers(len(t)))])}
    return {"L0": (g(i), g(i + 1)), "L1": (g(i - 1), g(i), g(i + 1)), "L2": (g(i - 1), g(i), g(i + 1), g(i + 2)),
            "L1P": (g(i), g(i + 1), t[int(rng.integers(len(t)))])}


if __name__ == "__main__":
    D = json.load(open(GD.OUT))
    C = {lv: collections.defaultdict(collections.Counter) for lv in ("L0", "L1", "L2", "L1P")}; prior = collections.Counter()
    rows = {}
    for sp in D:
        rows[sp] = []
        for it in D[sp]:
            R = roles(it); rows[sp].append([(keys(it, i, np.random.default_rng(zlib.crc32(f"{sp}|{it['gid']}|{i}".encode()))), r) for i, r in R.items()])
    for prob in rows["TRAIN"]:
        for k, r in prob:
            prior[r] += 1
            for lv in C: C[lv][k[lv]][r] += 1
    top = prior.most_common(1)[0][0]
    L = [f"G-TREE: TRAIN {len(D['TRAIN'])} problems, {sum(prior.values())} numeral words; roles {dict(prior.most_common())}; nodes "
         + " ".join(f"{lv}:{len(C[lv])}" for lv in C)]

    def pred(k, own, levels):
        best = None
        for lv in levels:
            c = C[lv].get(k[lv])
            if c is None: continue
            c = c - own[lv] if own else c                                   # leave-one-out: the problem's own counts removed
            if sum(c.values()) >= MINN: best = c.most_common(1)[0][0]
        return best or top

    res = {}
    for sp in ("TRAIN", "TEST", "CFNUM", "CFNAME"):
        acc = collections.defaultdict(list); hit = []
        for prob in rows[sp]:
            own = None
            if sp == "TRAIN":
                own = {lv: collections.defaultdict(collections.Counter) for lv in C}
                for k, r in prob:
                    for lv in C: own[lv][k[lv]][r] += 1
            for k, r in prob:
                ow = {lv: own[lv][k[lv]] for lv in C} if own else None
                for name, lv in (("PRIOR", ()), ("L0", ("L0",)), ("L1", ("L0", "L1")), ("L1P", ("L0", "L1P")), ("BACKOFF", ("L0", "L1", "L2"))):
                    acc[name].append(pred(k, ow, lv) == r)
                hit.append(k["L2"] in C["L2"])
        res[sp] = {n: float(np.mean(v)) for n, v in acc.items()}
        L.append(f"  {sp:6s} " + " ".join(f"{n} {v:.3f}" for n, v in res[sp].items()) + f" | L2 node seen {np.mean(hit):.3f}"
                 + (" (LOO)" if sp == "TRAIN" else ""))
    t = res["TEST"]
    T1 = t["BACKOFF"] - t["PRIOR"] >= 0.10; T2 = t["BACKOFF"] - t["L0"] >= 0.03 and t["L1"] - t["L1P"] >= 0.03
    L += [f"T1 THE TREE KNOWS ROLES: TEST backoff {t['BACKOFF']:.3f} - prior {t['PRIOR']:.3f} = {t['BACKOFF'] - t['PRIOR']:+.3f} (bar +0.10) -> {'PASS' if T1 else 'FAIL'}",
          f"T2 CONTEXT PAYS: backoff - L0 {t['BACKOFF'] - t['L0']:+.3f} (bar +0.03); L1 - placebo {t['L1'] - t['L1P']:+.3f} (bar +0.03) -> {'PASS' if T2 else 'FAIL'}"]
    print("\n".join(L)); open(f".cache/gsm_role_tree{'_rel' if REL else ''}.txt", "w").write("\n".join(L) + "\n")
