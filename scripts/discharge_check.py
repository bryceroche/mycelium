"""discharge_check.py — the discharge ledger's meter (gut #54).

Walks the campaign's standing accumulators, reads their charge from
banked artifacts, and reports each against its PINNED threshold. A
breach OPENS A REVIEW — it never fires watts, writes manifests, or
holds a pen (thresholds convene, humans fire). Re-run any time; rides
the entourage duty roster.

Pinned zeners (gut #54, 2026-07-22):
  wild_crown_mass  >= 25 unique banked crown knots
                   -> opens the next major-fire registration review
                      (band-restart arm rides it per the registered lean)
  admission_family >= 6 certificates in one registry family
                   -> the rung test convenes for that family
                      (the admission review holds the pen)
  macro_of_macro   >= 5 independent instruments (hand-updated; ledger)
                   -> the macro-of-macro charter review convenes
"""
import glob, json
from collections import Counter

THRESHOLDS = {"wild_crown_mass": 25, "admission_family": 6, "macro_of_macro": 5}
MACRO_OF_MACRO_INSTRUMENTS = 3  # width gradient 51%, curve saturation, [73]'s crack


def main():
    fams = Counter()
    for f in sorted(glob.glob(".cache/*organ_registry*.json")):
        for c in json.load(open(f)):
            fams[c["family"]] += 1

    knots = set()
    for path in sorted(glob.glob(".cache/book*_prose_pairs.jsonl")):
        for ln in open(path):
            r = json.loads(ln)
            g = r.get("gen", {})
            if g.get("floor") == "macro" and g.get("knot"):
                knots.add(g["knot"])

    breaches = []
    print(f"[discharge] wild_crown_mass: {len(knots)} / {THRESHOLDS['wild_crown_mass']}"
          f" {'** BREACH -> major-fire registration review opens **' if len(knots) >= THRESHOLDS['wild_crown_mass'] else '(accruing)'}")
    if len(knots) >= THRESHOLDS["wild_crown_mass"]:
        breaches.append("wild_crown_mass")

    K = THRESHOLDS["admission_family"]
    for fam, n in fams.most_common():
        if n >= K:
            print(f"[discharge] admission_family {fam}: {n} / {K} ** BREACH -> rung test convenes **")
            breaches.append(f"admission_family:{fam}")
    nxt = [(f, n) for f, n in fams.most_common() if n < K][:3]
    print(f"[discharge] admission next-in-line: {nxt}")

    m = MACRO_OF_MACRO_INSTRUMENTS
    print(f"[discharge] macro_of_macro: {m} / {THRESHOLDS['macro_of_macro']}"
          f" {'** BREACH -> charter review convenes **' if m >= THRESHOLDS['macro_of_macro'] else '(accruing)'}")
    if m >= THRESHOLDS["macro_of_macro"]:
        breaches.append("macro_of_macro")

    json.dump({"thresholds": THRESHOLDS, "wild_crown_mass": len(knots),
               "families": dict(fams), "macro_of_macro": m,
               "breaches": breaches},
              open(".cache/discharge_ledger.json", "w"), indent=1)
    print(f"[discharge] {len(breaches)} breach(es) -> .cache/discharge_ledger.json"
          " | breaches OPEN REVIEWS; they never fire watts")


if __name__ == "__main__":
    main()
