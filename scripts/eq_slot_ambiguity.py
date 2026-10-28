"""eq_slot_ambiguity.py -- IS THE EQ_SLOT TARGET DETERMINED? (ledger 2026-10-28; zero GPU). For each fixture row with <= 5 stated
givens, grade EVERY assignment of mentions to slots through the solver (eq_slot_grade.grade: the custody key decides). Reports the
share of rows where ONLY the gold order solves vs several orders solve, and the mean count of solving orders -- on the holdout and
on a GSM8K-train sample (one row per problem). Writes .cache/eq_slot_ambiguity.txt."""
import itertools, json, os, random, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import eq_slot_grade as EG
from eq_dsl import problem_key

L = []
say = lambda s: (print(s, flush=True), L.append(s))
for split, lim in (("holdout", None), ("train", 6000)):
    T, _ = EG.rows(split, lim); seen = set(); rows = []
    for t in T:
        if split == "train":
            g = t["smp"].get("gen")
            if not (isinstance(g, dict) and g.get("src") == "gsm8k"): continue
            k = problem_key(t["smp"])
            if k in seen: continue
            seen.add(k)
        st = [s for s in t["slots"] if s[3] is not None]
        if 2 <= len(st) <= 5 and EG.grade(t, EG.gold_assign(t)): rows.append(t)
    rows = rows[:300] if split == "train" else rows
    cnt, ident_ok = [], []
    for t in rows:
        st = [s for s in t["slots"] if s[3] is not None]
        ok = [EG.grade(t, {s[0]: c for s, c in zip(st, p)}) for p in itertools.permutations(t["cands"])]
        cnt.append(sum(ok)); ident_ok.append(ok[0])   # permutations(cands) starts with text order = identity
    cnt = np.array(cnt)
    say(f"{split}: rows {len(rows)} (2-5 stated givens, gold solves) | ONLY ONE order solves {np.mean(cnt == 1):.3f} | mean solving orders "
        f"{cnt.mean():.2f} | text order solves {np.mean(ident_ok):.3f} | by n: " +
        " ".join(f"n{n} {np.mean(cnt[[len([s for s in t['slots'] if s[3] is not None]) == n for t in rows]] == 1):.2f}"
                 for n in range(2, 6) if any(len([s for s in t['slots'] if s[3] is not None]) == n for t in rows)))
open(".cache/eq_slot_ambiguity.txt", "w").write("\n".join(L) + "\n")
