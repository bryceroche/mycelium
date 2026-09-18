"""violation_census.py — THE DECODE DOOR's read (2026-09-18): the rulebook's structural rules over the machine's
decoded slots (from an LV_DUMP: the graded gold slots' predictions), per checkpoint — how often a decoded graph
breaks single_intro (a variable introduced twice), pointers, value_legal — before any second mask is built.
usage: violation_census.py dump.pkl split.jsonl [dump2.pkl ...]"""
import pickle, sys, json, collections
sys.path.insert(0, "."); from mycelium.rulebook import violations
split = sys.argv[2]; rows = [json.loads(l) for l in open(split)]; texts = {i: r["text"] for i, r in enumerate(rows)}
for path in [sys.argv[1]] + sys.argv[3:]:
    D = pickle.load(open(path, "rb")); per_row = collections.defaultdict(list)
    for (i, j, gft, gop, gargs, gres, gdig, pft, pop, pargs, pres_, pdig, ppres, pdup) in D:
        if not ppres: continue
        if pft != 0: per_row[i].append({"ftype": "given", "var": pres_, "value": int("".join(map(str, pdig)))})
        else: per_row[i].append({"ftype": "rel", "op": "add" if pop == 0 else "mul", "args": (pargs[:1] * 2 if pdup else pargs[:2]), "result": pres_})
    tot = collections.Counter(); rows_hit = collections.Counter()
    for i, parse in per_row.items():
        v = violations(parse, texts[i], register="prose")
        for k, c in v.items(): tot[k] += c; rows_hit[k] += 1
    n = len(per_row)
    print(f"[violations] {path.split('/')[-1]}: rows {n} | rows breaking a rule: " + ", ".join(f"{k} {rows_hit[k]} ({rows_hit[k]/n:.0%})" for k in ("value_legal", "single_intro", "pointers", "pointer_order") if k in rows_hit) + f" | slot counts {dict(tot)}")
