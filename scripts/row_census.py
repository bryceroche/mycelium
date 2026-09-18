"""row_census.py — THE PER-ROW ERROR CENSUS (2026-09-18): which factor breaks the row. From an LV_DUMP pickle
(per gold slot: gold fields + the machine's argmaxes), per row: the number of wrong slots and which FIELD
broke each (presence / ftype / op / args / res / digits, first failing in the read's order). Reports the
distribution of rows by wrong-slot count (rows at 0 are the solvable-by-parse rows), and the first-break
field histogram over rows with exactly one wrong slot (the rows one fix away).
usage: row_census.py dump.pkl [dump2.pkl ...]"""
import pickle, sys, collections

def slot_ok(t):
    (j, gft, gop, gargs, gres, gdig, pft, pop, pargs, pres_, pdig, ppres, pdup) = t
    if not ppres: return "presence"
    if pft != gft: return "ftype"
    if pres_ != gres: return "res"
    if gft == 0:
        if pop != gop: return "op"
        ok = (len(gargs) == 1 and pdup and pargs[0] == gargs[0]) or (len(gargs) == 2 and set(pargs) == set(gargs))
        if not ok: return "args"
    else:
        if pdig != gdig: return "digits"
    return None

for path in sys.argv[1:]:
    D = pickle.load(open(path, "rb")); rows = collections.defaultdict(list)
    for t in D: rows[t[0]].append(t[1:])
    by_wrong = collections.Counter(); first_break_1 = collections.Counter(); field_all = collections.Counter(); n_slots = 0
    for i, slots in rows.items():
        breaks = [slot_ok(s) for s in slots]; wrong = [b for b in breaks if b]; by_wrong[len(wrong)] += 1; n_slots += len(slots)
        for b in wrong: field_all[b] += 1
        if len(wrong) == 1: first_break_1[wrong[0]] += 1
    n = len(rows); dist = " ".join(f"{k}:{by_wrong[k]}" for k in sorted(by_wrong))
    print(f"[row-census] {path.split('/')[-1]}: rows {n}, slots {n_slots} | rows by wrong-slot count {dist} | rows fully right {by_wrong[0]} ({by_wrong[0]/n:.1%}), one away {by_wrong[1]} ({by_wrong[1]/n:.1%}), <=2 away {(by_wrong[0]+by_wrong[1]+by_wrong[2])/n:.1%}")
    print(f"              wrong slots by field (all rows): {dict(field_all.most_common())} | the one-away rows break on: {dict(first_break_1.most_common())}")
