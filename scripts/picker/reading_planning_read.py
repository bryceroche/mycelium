"""reading_planning_read.py -- STEP 3 of the reading-order line (2026-10-16, zero GPU): re-score banked wild dumps
(LV_DUMP pickles, one tuple per gold slot) into READING vs PLANNING, and read the machine's own ORDER HABIT.
  READING   per gold given: its value is among the machine's predicted given values in the row (multiset, order-free).
  ROW READ  every gold given read, no extra predicted givens (the reading block a reading-order frame would grade).
  PLANNING  matched_read.py's per-slot verdict on RELATION slots (op + pointers under the value-induced variable
            map), overall and on rows whose reading is fully right (planning with the reading held fixed).
  ORDER HABIT on the rows the re-serialization REORDERED (gold derivation order != text order): per given position
            k, does the machine's k-th predicted given carry the derivation-order value (the current gold) or the
            text-order value (the reading-order gold)? (positions where the two orders agree are skipped)
usage: reading_planning_read.py TAG [TAG ...]   (reads .cache/dump_wild_TAG.pkl; the RO fixture for the orders)"""
import collections, json, pickle, sys
sys.path.insert(0, "scripts")
import matched_read as M

def val(d): return int("".join(str(int(x)) for x in d))

old = [json.loads(l) for l in open(".cache/wild_admitted_holdout.jsonl")]
ro = [json.loads(l) for l in open(".cache/wild_admitted_holdout_RO.jsonl")]
gold_seq = [[int(f["value"]) for f in r["factors"] if f["ftype"] == "given"] for r in old]
text_seq = [[int(f["value"]) for f in r["factors"] if f["ftype"] == "given"] for r in ro]
for tag in sys.argv[1:]:
    D = pickle.load(open(f".cache/dump_wild_{tag}.pkl", "rb"))
    by = collections.defaultdict(list)
    for t in D: by[t[0]].append(t[1:])
    C = collections.Counter()
    for i, slots in by.items():
        slots = sorted(slots, key=lambda s: s[0])
        ok, _ = M.match_row(slots)
        pg = [val(s[10]) for s in slots if s[6] == 1 and s[11]]
        gg = [val(s[5]) for s in slots if s[1] == 1]
        rem = collections.Counter(pg); hit = 0
        for v in gg:
            if rem[v] > 0: rem[v] -= 1; hit += 1
        C["giv"] += len(gg); C["giv_read"] += hit
        row_read = hit == len(gg) and len(pg) == len(gg)
        C["rows"] += 1; C["rows_read"] += row_read
        rel = [s[0] for s in slots if s[1] != 1]
        C["rel"] += len(rel); C["rel_ok"] += sum(bool(ok.get(j)) for j in rel)
        if row_read:
            C["rel_rr"] += len(rel); C["rel_rr_ok"] += sum(bool(ok.get(j)) for j in rel)
        C["slots"] += len(slots); C["matched_ok"] += sum(bool(v) for v in ok.values())
        if gold_seq[i] != text_seq[i]:
            C["reord_rows"] += 1
            for k in range(min(len(pg), len(gg))):
                g, t = gold_seq[i][k], text_seq[i][k]
                if g == t: continue
                C["pos"] += 1; C["pos_gold"] += pg[k] == g; C["pos_text"] += pg[k] == t
    f = lambda a, b: C[a] / max(C[b], 1)
    print(f"== {tag}: READING per-given {f('giv_read','giv'):.3f} (n={C['giv']}) | rows fully read {f('rows_read','rows'):.3f} "
          f"({C['rows_read']}/{C['rows']}) | PLANNING relation slots (matched) {f('rel_ok','rel'):.3f} (n={C['rel']}), on fully-read "
          f"rows {f('rel_rr_ok','rel_rr'):.3f} (n={C['rel_rr']}) | matched fac-exact {f('matched_ok','slots'):.4f}")
    print(f"   ORDER HABIT on {C['reord_rows']} reordered rows, {C['pos']} positions where the orders differ: the machine's k-th given "
          f"= derivation-order gold {f('pos_gold','pos'):.3f} | = text-order {f('pos_text','pos'):.3f} | neither {1 - f('pos_gold','pos') - f('pos_text','pos'):.3f}")
