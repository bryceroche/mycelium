"""match_gold.py — THE MATCHING LOSS's assignment (2026-09-17, word given; the first alternation arm).
The head's loss is positional: gold factor k is graded against slot k. A correct graph in another order
scores zero (the matched read: order was all of the positional diet's holdout loss). This module, on the
HOST between the walker's forward chain and its backward chain, assigns each gold factor to the predicted
slot that carries it (givens by value, relations by op + pointers under the correspondence the assignment
induces) and PERMUTES the gold feed so the loss grades content, not serialization. Only rows obeying the
positional law (var k introduced at slot k — every prose row) are permuted; mint rows keep their order.
At the identity assignment the feed is unchanged (the gate)."""
import numpy as np

SLOT_KEYS = ("presence", "is_lit_f", "fspan", "ftype", "op", "digits", "sel", "is_rel", "is_mod", "is_sel", "is_pct", "is_fdiv", "arg_dup", "is_macro", "digits2", "is_frac", "is_chain", "bind_ids")   # bind_ids: (L, 4) bind codes per SLOT
VAR_VALUE_KEYS = ("res", "y")          # per-slot ints that ARE variable indices -> remapped by V

def law_holds(feed, i):
    """positional law on row i of the feed: the var-set of slot k minus the vars seen before is exactly {k}"""
    L = feed["presence"].shape[1]; seen = set()
    n = int(feed["presence"][i].sum())
    for k in range(n):   # the gold encoding: a given's variable is res[k] (args empty); a relation's are args + res
        vs = set(np.where(feed["args"][i, k] > 0.5)[0].tolist()); vs.add(int(feed["res"][i, k]))
        if vs - seen != {k}: return False
        seen |= vs
    return True

def assign(feed, i, pred):
    """sigma: gold slot -> predicted slot, a full permutation of L. pred: dict of per-slot arrays for row i
    (pres (L,), ftype (L,), op (L,), dig (L, ND), args (L, K) scores, res (L,), dup (L,))."""
    L = feed["presence"].shape[1]; n = int(feed["presence"][i].sum()); used = set(); sigma = {}; vmap = {}
    gtype = feed["ftype"][i]; gres = feed["res"][i]; gdig = feed["digits"][i]; gargs = feed["args"][i]; gop = feed["op"][i]; grel = feed["is_rel"][i]
    ppres = pred["pres"] > 0; pft = pred["ftype"]; pdig = pred["dig"]; pop = pred["op"]; pres_ = pred["res"]; ptop2 = np.argsort(-pred["args"], axis=-1)[:, :2]; pdup = pred.get("dup", np.zeros(L, bool)) > 0
    # givens first (any order): match by digits among predicted given slots
    for k in range(n):
        if grel[k] > 0.5: continue
        cands = [p for p in range(L) if p not in used and ppres[p] and pft[p] == gtype[k] and (pdig[p] == gdig[k]).all()]
        if not cands: cands = [p for p in range(L) if p not in used and ppres[p] and pft[p] == gtype[k]]
        if not cands: continue
        p = min(cands, key=lambda p: abs(p - k)); used.add(p); sigma[k] = p; vmap[k] = p
    # relations in gold slot order: the new variable k maps to the chosen slot
    for k in range(n):
        if grel[k] < 0.5: continue
        ga = set(np.where(gargs[k] > 0.5)[0].tolist()); gr = int(gres[k]); best = None
        for p in range(L):
            if p in used or not ppres[p] or pft[p] != gtype[k]: continue
            vm = dict(vmap); vm[k] = p
            m_args = all(a in vm for a in ga) and ((len(ga) == 1 and pdup[p] and ptop2[p, 0] == vm[next(iter(ga))]) or (len(ga) == 2 and set(ptop2[p].tolist()) == {vm[a] for a in ga}))
            m_res = gr in vm and int(pres_[p]) == vm[gr]; m_op = int(pop[p]) == int(gop[k])
            sc = (int(m_op) + int(m_args) + int(m_res), -abs(p - k))
            if best is None or sc > best[0]: best = (sc, p)
        if best is None: continue
        p = best[1]; used.add(p); sigma[k] = p; vmap[k] = p
    # complete to a permutation: unmatched gold factors and absent slots take the nearest free slots
    free = [p for p in range(L) if p not in used]
    for k in range(L):
        if k in sigma: continue
        p = min(free, key=lambda p: abs(p - k)); free.remove(p); sigma[k] = p
    return np.array([sigma[k] for k in range(L)])

def hits(feed, i, pred, sigma):
    """how many present gold factors are fully carried by the slot sigma assigns them (the read's fac-exact rule)"""
    n = int(feed["presence"][i].sum()); V = sigma; h = 0
    for k in range(n):
        p = int(sigma[k])
        if not (pred["pres"][p] > 0) or int(pred["ftype"][p]) != int(feed["ftype"][i, k]): continue
        if int(pred["res"][p]) != int(V[int(feed["res"][i, k])]): continue
        if feed["is_rel"][i, k] > 0.5:
            ga = {int(V[a]) for a in np.where(feed["args"][i, k] > 0.5)[0]}; top2 = np.argsort(-pred["args"][p])[:2].tolist()
            if int(pred["op"][p]) != int(feed["op"][i, k]): continue
            if not ((len(ga) == 1 and pred.get("dup", np.zeros(len(sigma)))[p] > 0 and top2[0] in ga) or (len(ga) == 2 and set(top2) == ga)): continue
        else:
            if not (pred["dig"][p] == feed["digits"][i, k]).all(): continue
        h += 1
    return h

def permute_row(feed, i, sigma):
    """gold slot k -> slot sigma[k]; variable k -> sigma[k] (the law: var k lives at slot k); in place on row i"""
    L = len(sigma); V = sigma   # the var permutation IS sigma under the law (K == L == 24)
    for key in SLOT_KEYS:
        if key in feed: a = feed[key][i]; new = np.empty_like(a); new[sigma] = a; feed[key][i] = new
    a = feed["args"][i]; new = np.empty_like(a); new[np.ix_(sigma, V)] = a; feed["args"][i] = new
    for key in VAR_VALUE_KEYS:
        if key in feed: a = feed[key][i]; new = np.empty_like(a); new[sigma] = V[a.astype(int)]; feed[key][i] = new
    a = feed["vspan"][i]; new = np.empty_like(a); new[V] = a; feed["vspan"][i] = new
    feed["query"][i] = V[int(feed["query"][i])]
    if "bind_ids" in feed:   # (L, 4) per slot: (arg1, arg2, res, 24+ftype) — the first three are VARIABLE indices; moved by sigma above, remapped by V here
        b = feed["bind_ids"][i]; b[:, :3] = V[b[:, :3].astype(int)]; feed["bind_ids"][i] = b

def match_feed(feed, preds, identity=False):
    """permute every law-abiding row of the feed toward its predictions; returns (n_permuted, n_law)"""
    B = feed["presence"].shape[0]; n_perm = 0; n_law = 0
    for i in range(B):
        if not law_holds(feed, i): continue
        n_law += 1
        if identity: continue
        pr = {k: v[i] for k, v in preds.items()}; sigma = assign(feed, i, pr); ident = np.arange(len(sigma))
        # accept the assignment only if it carries MORE gold factors than the identity (the read's own rule): a
        # permutation can never make the graded content worse than the positional gold
        if (sigma != ident).any() and hits(feed, i, pr, sigma) > hits(feed, i, pr, ident): permute_row(feed, i, sigma); n_perm += 1
    return n_perm, n_law

if __name__ == "__main__":   # CPU self-test: a shuffled gold matched against its own unshuffled predictions returns to the original
    rng = np.random.default_rng(0); L = K = 24; T = 8; ND = 3
    def row():
        f = {"presence": np.zeros((1, L)), "is_lit_f": np.zeros((1, L)), "args": np.zeros((1, L, K)), "fspan": np.zeros((1, L, T)), "vspan": np.zeros((1, K, T)), "ftype": np.zeros((1, L), int), "op": np.zeros((1, L), int),
             "res": np.zeros((1, L), int), "digits": np.zeros((1, L, ND), int), "query": np.zeros((1,), int), "sel": np.zeros((1, L), int), "is_rel": np.zeros((1, L)), "is_mod": np.zeros((1, L)), "is_sel": np.zeros((1, L)), "is_pct": np.zeros((1, L)), "is_fdiv": np.zeros((1, L)), "arg_dup": np.zeros((1, L)), "y": np.zeros((1, L), int)}
        # g0=5, g1=7, r2 = add(0,1)->2, g3 = 9, r4 = mul(2,3)->4 ; query 4
        vals = [5, 7, None, 9, None]
        for k, v in enumerate(vals):
            f["presence"][0, k] = 1
            if v is not None: f["ftype"][0, k] = 1; f["is_lit_f"][0, k] = 1; f["res"][0, k] = k; f["digits"][0, k] = [0, 0, v]; f["vspan"][0, k, k] = 1; f["fspan"][0, k, k] = 1
        f["is_rel"][0, 2] = 1; f["args"][0, 2, [0, 1]] = 1; f["res"][0, 2] = 2; f["op"][0, 2] = 0
        f["is_rel"][0, 4] = 1; f["args"][0, 4, [2, 3]] = 1; f["res"][0, 4] = 4; f["op"][0, 4] = 1
        f["query"][0] = 4; return f
    g = row(); assert law_holds(g, 0)
    # predictions in a DIFFERENT order: givens first then relations (g0 g1 g3 r2 r4), numbering by slot: 5,7,9 at 0,1,2; add(0,1)->3 at 3; mul(3,2)->4 at 4
    pr = {"pres": np.array([1, 1, 1, 1, 1] + [0] * 19, float), "ftype": np.array([1, 1, 1, 0, 0] + [0] * 19), "op": np.array([0, 0, 0, 0, 1] + [0] * 19), "dig": np.zeros((L, ND), int), "args": np.zeros((L, K)), "res": np.zeros(L, int), "dup": np.zeros(L)}
    pr["dig"][0] = [0, 0, 5]; pr["dig"][1] = [0, 0, 7]; pr["dig"][2] = [0, 0, 9]; pr["args"][3, [0, 1]] = 1; pr["res"][3] = 3; pr["args"][4, [3, 2]] = 1; pr["res"][4] = 4
    sigma = assign(g, 0, pr); print("sigma (gold slot -> pred slot):", sigma[:5].tolist())
    assert sigma[:5].tolist() == [0, 1, 3, 2, 4], sigma[:5]
    import copy; g2 = copy.deepcopy(g); permute_row(g2, 0, sigma)
    assert g2["ftype"][0, :5].tolist() == [1, 1, 1, 0, 0] and g2["res"][0, :5].tolist() == [0, 1, 2, 3, 4] and set(np.where(g2["args"][0, 4] > 0)[0]) == {3, 2} and g2["query"][0] == 4 and g2["digits"][0, 2].tolist() == [0, 0, 9]
    g3 = copy.deepcopy(g); n = match_feed(g3, {k: v[None] for k, v in pr.items()}); assert n == (1, 1)
    g4 = copy.deepcopy(g); n = match_feed(g4, {k: v[None] for k, v in pr.items()}, identity=True); assert all((g4[k] == g[k]).all() for k in g)
    print("[match_gold] self-test PASS: permuted gold matches the predictions' order; identity leaves the feed untouched")


def preds_from_decode(onp):
    """the walker's decode pull (raw heads, batch-first) -> the matcher's fields, with the read's conventions
    (loop_val: ftype/op/res argmax, dig argmax per digit, pres > 0, dup > 0, args = scores)"""
    out = {"pres": onp["pres"], "ftype": onp["ftype"].argmax(-1), "op": onp["op"].argmax(-1), "dig": onp["dig"].argmax(-1), "args": onp["args"], "res": onp["res"].argmax(-1)}
    if "dup" in onp: out["dup"] = onp["dup"]
    return out
