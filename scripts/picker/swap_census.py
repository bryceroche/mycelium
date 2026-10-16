"""swap_census.py -- THE SWAP CENSUS (2026-10-16, zero GPU). Usage: swap_census.py TAG (reads .cache/dump_wild_TAG.pkl + ps_legal_wild_TAG.npz).
Swap census: of the wrong given picks, how many choose ANOTHER gold given's value (misplaced: right
number, wrong slot) vs a value no gold given carries (content). Crossed with the governor census's owner
classes and with matched_read's per-slot verdict."""
import sys, os, json, pickle, collections
sys.path.insert(0, "/home/bryce/mycelium/scripts")
os.chdir("/home/bryce/mycelium")
import numpy as np, spacy
import governor_census as G
import matched_read as M
tag = sys.argv[1]
NLP = spacy.load("en_core_web_sm")
rows = [json.loads(l) for l in open(G.HOLDOUT)]
D = pickle.load(open(f".cache/dump_wild_{tag}.pkl", "rb"))
by_row = collections.defaultdict(list)
for t in D: by_row[t[0]].append(t[1:])
z = np.load(f".cache/ps_legal_wild_{tag}.npz")
npz_ok = {(int(r), int(s)): bool(o) for r, s, o in zip(z["rows"], z["slots"], z["ok"])}
C = collections.Counter(); cyc = collections.Counter()
for i, row in enumerate(rows):
    doc = NLP(row["text"]); sl = list(doc.sents); cands = G.numeral_candidates(doc)
    gl = [(f["var"], f["value"]) for f in row["factors"] if f["ftype"] == "given"]
    bound = G.bind_given_occurrences(doc, gl, cands)
    chain = {v: (G.governor_chain(doc, sl, t) if t is not None else None) for v, t in bound.items()}
    slots = sorted(by_row[i], key=lambda s: s[0])
    mok, _ = M.match_row(slots)
    gold_vals = collections.Counter(G.digits_to_int(s[5]) for s in slots if s[1] == 1)
    gval_slot = {s[0]: G.digits_to_int(s[5]) for s in slots if s[1] == 1}
    pval_slot = {s[0]: G.digits_to_int(s[10]) for s in slots if s[1] == 1}
    for s in slots:
        j, gft, gop, gargs, gres, gdig, pft, pop, pargs, pres_, pdig, ppres, pdup = s
        if gft != 1: continue
        ok = npz_ok.get((i, j))
        if ok is None or ok: continue
        gv, cv = G.digits_to_int(gdig), G.digits_to_int(pdig)
        if gv == cv: C["not-a-value-error"] += 1; continue
        gch = chain.get(gres)
        ctok = G.nearest_occurrence(cands, cv, None, bound[gres].idx if bound.get(gres) is not None else 0)
        cch = G.governor_chain(doc, sl, ctok) if ctok is not None else None
        cls = "NO-CHAIN" if gch is None or cch is None else ("SAME" if gch["owner_key"] == cch["owner_key"] else "DIFF")
        mis = "MISPLACED" if cv in gold_vals else "CONTENT"
        # a 2-cycle: some other given slot k has gold cv and predicted gv
        two = any(gval_slot[k] == cv and pval_slot.get(k) == gv for k in gval_slot if k != j)
        C[(cls, mis)] += 1; C[cls] += 1; C[mis] += 1
        if two: cyc[cls] += 1; C["2cycle"] += 1
        if mok.get(j): C[(cls, "matched_right")] += 1; C["matched_right"] += 1
n = sum(C[k] for k in ("SAME", "DIFF", "NO-CHAIN"))
print(f"== {tag}: classified wrong given picks {n} (+{C['not-a-value-error']} not-a-value-error)")
for cls in ("SAME", "DIFF", "NO-CHAIN"):
    m = C[cls]
    print(f"  {cls:9s} {m:4d} | misplaced {C[(cls,'MISPLACED')]:4d} ({C[(cls,'MISPLACED')]/max(m,1):.3f}) "
          f"| 2-cycle swaps {cyc[cls]:4d} ({cyc[cls]/max(m,1):.3f}) | right under matched read {C[(cls,'matched_right')]:4d} ({C[(cls,'matched_right')]/max(m,1):.3f})")
print(f"  ALL       {n:4d} | misplaced {C['MISPLACED']} ({C['MISPLACED']/n:.3f}) | 2-cycle {C['2cycle']} ({C['2cycle']/n:.3f}) | matched-right {C['matched_right']} ({C['matched_right']/n:.3f})")
