"""arg_kind_read.py -- WHERE DO THE WRONG ARGS LIVE? (2026-10-16, zero GPU; follows planning_drop_read.py.)
Wild gold introduces exactly ONE var per factor at its own slot (1040/1042 relations; ops are add/mul only), so the
var bank already IS the 'introduced at step j' address. This read grades each gold relation ARGUMENT (per-arg, under
matched_read's value-induced var map) by its KIND in the dump's own frame: GIVEN / PRODUCED (an earlier relation's
result) / SEEN-ARG (introduced as an arg of an earlier relation) / FIRST-HERE (introduced by this relation itself).
usage: arg_kind_read.py TAG [TAG ...]  (reads .cache/dump_wild_TAG.pkl)"""
import collections, pickle, sys

def rows_of(path):
    by = collections.defaultdict(list)
    for t in pickle.load(open(path, "rb")): by[t[0]].append(t[1:])
    return {i: sorted(s, key=lambda s: s[0]) for i, s in by.items()}

def match(slots):
    gold_g = [s for s in slots if s[1] != 0]; gold_r = [s for s in slots if s[1] == 0]
    pred_g = [s for s in slots if s[6] != 0 and s[11]]; pred_r = [s for s in slots if s[6] == 0 and s[11]]
    vmap, used, chosen = {}, set(), {}
    for g in gold_g:
        c = [p for p in pred_g if p[0] not in used and p[10] == g[5]] or [p for p in pred_g if p[0] not in used]
        if c:
            p = min(c, key=lambda p: abs(p[0] - g[0])); used.add(p[0]); vmap[g[0]] = p[0]
    for g in gold_r:
        k, gop, gargs, gres = g[0], g[2], g[3], g[4]; best = None
        for p in pred_r:
            if p[0] in used: continue
            vm = dict(vmap); vm[k] = p[0]
            m_args = all(a in vm for a in gargs) and ((len(gargs) == 1 and p[12] and p[8][0] == vm[gargs[0]]) or (len(gargs) == 2 and set(p[8]) == {vm[a] for a in gargs}))
            sc = ((p[7] == gop) + m_args + (gres in vm and p[9] == vm[gres]), -abs(p[0] - k))
            if best is None or sc > best[0]: best = (sc, p)
        if best is None: continue
        p = best[1]; used.add(p[0]); vmap[k] = p[0]; chosen[k] = p
    return vmap, chosen

for tag in sys.argv[1:]:
    C = collections.Counter()
    for i, slots in rows_of(f".cache/dump_wild_{tag}.pkl").items():
        intro = {}
        for s in slots:
            for v in ([s[4]] if s[1] != 0 else list(s[3]) + [s[4]]): intro.setdefault(v, s)
        vmap, chosen = match(slots)
        for g in slots:
            if g[1] != 0: continue
            p = chosen.get(g[0])
            for a in g[3]:
                src = intro[a]
                kind = ("GIVEN" if src[1] != 0 else "FIRST-HERE" if src[0] == g[0] else
                        "PRODUCED" if src[4] == a else "SEEN-ARG")
                hit = p is not None and a in vmap and vmap[a] in p[8]
                C[kind] += 1; C[kind + "_ok"] += hit
    print(f"== {tag}: per-arg hit by kind | " + " | ".join(f"{k} {C[k + '_ok'] / max(C[k], 1):.3f} (n {C[k]})" for k in ("GIVEN", "PRODUCED", "SEEN-ARG", "FIRST-HERE")))
