"""matched_read.py — THE MATCHED READ (2026-09-17, word given). The positional read scores a slot right only
if the machine put the RIGHT factor in the SAME slot as the gold; a correct graph in another order scores 0.
This read matches predicted factors to gold factors by CONTENT: givens by value (digits), relations in gold
slot order by (op, pointers under the variable correspondence the matching induces — gold var k is the
variable introduced at gold slot k; the machine's var p is the one introduced at its slot p). Reports
matched fac-exact beside positional, per field. Input: the LV_DUMP pickle (one tuple per gold slot).
usage: matched_read.py dump.pkl [dump2.pkl ...]"""
import pickle, sys, collections

def match_row(slots):
    # slots: list of (j, gft, gop, gargs, gres, gdig, pft, pop, pargs, pres_, pdig, ppres, pdup) for one row, sorted by j
    gold_g = [s for s in slots if s[1] != 0]; gold_r = [s for s in slots if s[1] == 0]
    pred_g = [s for s in slots if s[6] != 0 and s[11]]; pred_r = [s for s in slots if s[6] == 0 and s[11]]
    vmap = {}; used = set(); ok = {}; fields = collections.Counter()
    # givens by value (exact digits first, then leftovers by slot distance)
    for g in gold_g:
        cands = [p for p in pred_g if p[0] not in used and p[10] == g[5]]
        if not cands: cands = [p for p in pred_g if p[0] not in used]
        if not cands: ok[g[0]] = False; continue
        p = min(cands, key=lambda p: abs(p[0] - g[0])); used.add(p[0]); vmap[g[0]] = p[0]
        hit = p[6] == g[1] and p[10] == g[5]; ok[g[0]] = hit; fields["dig"] += p[10] == g[5]; fields["dig_n"] += 1
    # relations in gold slot order; the new variable of gold slot k maps to the chosen predicted slot
    for g in gold_r:
        k, gop, gargs, gres = g[0], g[2], g[3], g[4]; best = None
        for p in pred_r:
            if p[0] in used: continue
            vm = dict(vmap); vm[k] = p[0]
            m_args = all(a in vm for a in gargs) and (((len(gargs) == 1) and p[12] and p[8][0] == vm[gargs[0]]) or (len(gargs) == 2 and set(p[8]) == {vm[a] for a in gargs}))
            m_res = gres in vm and p[9] == vm[gres]; m_op = p[7] == gop
            score = (m_op + m_args + m_res, -abs(p[0] - k))
            if best is None or score > best[0]: best = (score, p, m_op, m_args, m_res)
        if best is None: ok[k] = False; continue
        _, p, m_op, m_args, m_res = best; used.add(p[0]); vmap[k] = p[0]
        ok[k] = m_op and m_args and m_res; fields["op"] += m_op; fields["args"] += m_args; fields["res"] += m_res; fields["rel_n"] += 1
    return ok, fields

def positional(slots):
    n = 0
    for (j, gft, gop, gargs, gres, gdig, pft, pop, pargs, pres_, pdig, ppres, pdup) in slots:
        hit = ppres and pft == gft and pres_ == gres
        if gft == 0: hit = hit and pop == gop and ((len(gargs) == 1 and pdup and pargs[0] == gargs[0]) or (len(gargs) == 2 and set(pargs) == set(gargs)))
        else: hit = hit and pdig == gdig
        n += hit
    return n

if __name__ == "__main__":
    for path in sys.argv[1:]:
        D = pickle.load(open(path, "rb")); rows = collections.defaultdict(list)
        for t in D: rows[t[0]].append(t[1:])
        pos = 0; mat = 0; tot = 0; F = collections.Counter()
        for i, slots in rows.items():
            slots.sort(key=lambda s: s[0]); tot += len(slots); pos += positional(slots)
            ok, f = match_row(slots); mat += sum(ok.values()); F.update(f)
        print(f"[matched] {path}: gold slots {tot} | positional fac-exact {pos/tot:.4f} | MATCHED fac-exact {mat/tot:.4f} (order's share of the gap: +{(mat-pos)/tot:.4f}) | matched fields: dig {F['dig']/max(F['dig_n'],1):.3f} op {F['op']/max(F['rel_n'],1):.3f} args {F['args']/max(F['rel_n'],1):.3f} res {F['res']/max(F['rel_n'],1):.3f}")
