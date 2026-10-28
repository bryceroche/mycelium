"""eq_dsl_pairs.py -- the DSL census, local level (ledger 2026-10-28; zero GPU). Whole-problem DSL skeletons are near-unique
(eq_dsl_census.txt), so generalization must come from LOCAL structure. For every text-ordered mention pair (i < j) the label is
INVERTED (the derivation uses j's given before i's). A pair-skeleton = (codes of i, codes of j, same-ent, same-obj, j-i distance).
Majority vote by pair-skeleton learned on GSM8K TRAIN (one row per problem), read on the HOLDOUT's pairs vs 'never invert'.
Coarsening ladder shows which fields carry the order. Writes .cache/eq_dsl_pairs.txt."""
import collections, json

import numpy as np

D = json.load(open(".cache/eq_dsl.json"))
gs = {}
for i, r in sorted(D["train"].items(), key=lambda x: int(x[0])):
    if r["src"] == "gsm8k" and r["key"] not in gs: gs[r["key"]] = r
pos = lambda r: {t: k for k, t in enumerate(r["tgt"])}   # text rank -> derivation rank
F = {"full": ("ent", "obj", "verb", "cues", "form", "q"), "no_verb": ("ent", "obj", "cues", "form", "q"), "roles": ("ent", "obj", "q"),
     "cues_form": ("cues", "form", "q"), "q_only": ("q",)}
L = []
say = lambda s: (print(s, flush=True), L.append(s))


def pairs(r, fields):
    p = pos(r); m = r["m"]
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            a = tuple(tuple(m[i][f]) if isinstance(m[i][f], list) else m[i][f] for f in fields)
            b = tuple(tuple(m[j][f]) if isinstance(m[j][f], list) else m[j][f] for f in fields)
            yield (a, b, m[i]["ent"] == m[j]["ent"], m[i]["obj"] == m[j]["obj"], min(j - i, 3)), int(p[j] < p[i])


base_tr = [y for r in gs.values() for _, y in pairs(r, ())]; base_ho = [y for r in D["holdout"].values() for _, y in pairs(r, ())]
say(f"pairs: train {len(base_tr)} (inverted {np.mean(base_tr):.3f}) | holdout {len(base_ho)} (inverted {np.mean(base_ho):.3f}) | 'never invert' holdout acc {1 - np.mean(base_ho):.3f}")
for name, fields in F.items():
    tab = collections.defaultdict(collections.Counter)
    for r in gs.values():
        for k, y in pairs(r, fields): tab[k][y] += 1
    hits = cov = n = 0
    for r in D["holdout"].values():
        for k, y in pairs(r, fields):
            n += 1
            if k in tab and sum(tab[k].values()) >= 3: cov += 1; pred = tab[k].most_common(1)[0][0]
            else: pred = 0
            hits += pred == y
    say(f"{name:9s}: pair-skeletons {len(tab):6d} | holdout pairs covered (>=3 train) {cov / n:.3f} | majority-vote acc {hits / n:.3f}")
open(".cache/eq_dsl_pairs.txt", "w").write("\n".join(L) + "\n")
