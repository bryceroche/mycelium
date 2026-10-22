"""cascade_read.py -- THE CASCADE TEST + THE HARD-ROW CROSS-TAB (2026-10-21, zero GPU; ledger same day).
Do relation slots fail BECAUSE their parent slots failed (cascade), or because the whole row is hard? The placebo swaps
the parents for random non-parent slots of the same row. -> .cache/cascade_read.txt"""
import collections, json, os, pickle
import numpy as np
OUT = ".cache/cascade_read.txt"
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
rows_txt = [json.loads(l) for l in open(".cache/wild_admitted_holdout.jsonl")]
L = []
for tag in os.environ.get("CR_TAGS", "CLK3_241,SRK3_241,K3_242").split(","):
    z = np.load(f".cache/ps_legal_wild_{tag}.npz")
    ok = {(int(r), int(s)): bool(o) for r, s, o in zip(z["rows"], z["slots"], z["ok"])}
    D = pickle.load(open(f".cache/dump_wild_{tag}.pkl", "rb"))
    R = collections.defaultdict(list)
    for t in D:
        R[t[0]].append(dict(j=t[1], ft=t[2], args=list(t[4]), res=t[5], ok=ok[(t[0], t[1])]))
    rng = np.random.default_rng(0)
    par = {"ok": [], "bad": []}; pla = {"ok": [], "bad": []}; noparent_args = collections.defaultdict(lambda: [0, 0])
    for i, sl in R.items():
        for s in sl:
            if s["ft"] != 0:
                continue
            ps = [u for u in sl if u["j"] != s["j"] and u["res"] in s["args"]]
            for a in s["args"]:
                noparent_args[i][1] += 1
                noparent_args[i][0] += not any(u["j"] != s["j"] and u["res"] == a for u in sl)
            if not ps:
                continue
            par["ok" if all(u["ok"] for u in ps) else "bad"].append(not s["ok"])
            others = [u for u in sl if u["j"] != s["j"] and u not in ps]
            if len(others) >= len(ps):
                pick = [others[k] for k in rng.choice(len(others), len(ps), replace=False)]
                pla["ok" if all(u["ok"] for u in pick) else "bad"].append(not s["ok"])
    m = lambda v: float(np.mean(v)) if v else float("nan")
    gp, gq = m(par["bad"]) - m(par["ok"]), m(pla["bad"]) - m(pla["ok"])
    c1 = m(par["ok"]) <= 0.50 and m(par["bad"]) >= 0.85
    c2 = gp - gq >= 0.10
    L.append(f"== {tag}")
    L.append(f"   PARENTS: P(wrong | all ok) {m(par['ok']):.3f} (n {len(par['ok'])}) | P(wrong | a parent wrong) {m(par['bad']):.3f} (n {len(par['bad'])}) | gap {gp:+.3f}")
    L.append(f"   PLACEBO: P(wrong | all ok) {m(pla['ok']):.3f} (n {len(pla['ok'])}) | P(wrong | one wrong) {m(pla['bad']):.3f} (n {len(pla['bad'])}) | gap {gq:+.3f}")
    L.append(f"   (C1) " + ("CASCADE EXPLAINS IT" if c1 else "CASCADE DOES NOT EXPLAIN IT") + f"; (C2) " + ("BEYOND ROW HARDNESS" if c2 else "NOT BEYOND ROW HARDNESS") + f" (parent gap − placebo gap {gp - gq:+.3f})")
    acc = {i: np.mean([s["ok"] for s in sl]) for i, sl in R.items()}
    qs = np.quantile(list(acc.values()), [1 / 3, 2 / 3])
    for name, lo, hi in (("HARD", -1, qs[0]), ("MID", qs[0], qs[1]), ("EASY", qs[1], 2)):
        ids = [i for i, a in acc.items() if (lo < a <= hi if name != "HARD" else a <= hi)]
        g = [s["ok"] for i in ids for s in R[i] if s["ft"] == 1]; r = [s["ok"] for i in ids for s in R[i] if s["ft"] == 0]
        L.append(f"   {name:4s} rows {len(ids):3d}: acc {np.mean([acc[i] for i in ids]):.3f} | slots/row {np.mean([len(R[i]) for i in ids]):.2f} | text {np.mean([len(rows_txt[i]['text']) for i in ids]):.0f} ch"
                 f" | relation share {np.mean([s['ft'] == 0 for i in ids for s in R[i]]):.3f} | no-parent arg share {sum(noparent_args[i][0] for i in ids) / max(1, sum(noparent_args[i][1] for i in ids)):.3f}"
                 f" | given acc {np.mean(g):.3f} | relation acc {np.mean(r):.3f}")
print("\n".join(L)); open(OUT, "w").write("\n".join(L) + "\n")
