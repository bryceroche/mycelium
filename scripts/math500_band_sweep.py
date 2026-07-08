"""math500_band_sweep.py — THE SHOPPING LIST: band-sweep over MATH-500 itself
(spec registration, 2026-07-09 — the registry-expansion chapter's first move,
a measurement before any relation gets built).

WHAT IT MEASURES (transparent regex classifier over problem+solution text —
auditable, no learned components, no API calls; MATH-500 is MEASURED here,
never trained on):
  1. subject x level distribution (the dataset's own labels)
  2. ANSWER-TYPE distribution — the domain question: what fraction of answers
     are plain integers (the current value-codebook world), fractions,
     radicals, expressions, intervals, tuples (the finite-integer expiry
     number, measured not feared)
  3. RELATION-NEED categories per problem (keyword profiles)
  4. GREEDY SET-COVER shopping list: relations ranked by MARGINAL coverage —
     a problem is covered when every category it needs is built (linear
     arithmetic assumed as the existing base)
  5. THE ARCHITECTURE QUESTION (relay): inequality share. <10% -> later
     chapter; ~30% -> the csp_core three-valued-predicate/interval
     conversation happens now.

REGISTERED EXPECTATIONS (before running): plain-integer answers 50-60%;
inequality-needing <10% (MATH-500 skews checkable computation over
inequality proofs) -> LATER-chapter prediction; algebra+prealgebra ~40%;
geometry ~20%. Hand-audit sample printed (n=20) — classifier precision is
eyeballed before the list is trusted.

USAGE: .venv/bin/python3 scripts/math500_band_sweep.py
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

ROWS = [json.loads(l) for l in open(".cache/math500_test.jsonl")]

# ---------------- answer types ----------------
def answer_type(a: str) -> str:
    s = a.strip().replace("\\!", "").replace("\\,", "").replace(" ", "")
    if re.fullmatch(r"-?\d+", s):
        return "integer"
    if re.fullmatch(r"-?\d*\.\d+", s):
        return "decimal"
    if "\\text" in s and not re.search(r"\d", s):
        return "text"
    if s.count(",") >= 1 and (s.startswith("(") or s.startswith("\\left(")):
        return "tuple/point"
    if "," in s:
        return "multi-value"
    if "\\in" in s or ("[" in s and "]" in s) or "\\cup" in s:
        return "interval/set"
    if "\\frac" in s or re.fullmatch(r"-?\d+/\d+", s):
        return ("fraction+" if re.search(r"\\sqrt|\\pi|[a-z]", s.replace(
            "\\frac", "")) else "fraction")
    if "\\sqrt" in s:
        return "radical"
    if "\\pi" in s:
        return "pi-expr"
    if "^" in s and "circ" in s:
        return "degrees"
    if re.search(r"[a-wyz]", re.sub(r"\\[a-z]+", "", s)) or "x" in re.sub(
            r"\\[a-z]+", "", s):
        return "expression"
    if "%" in s or "\\%" in s:
        return "percent"
    return "other"

# ---------------- relation-need categories ----------------
CATS = {
    "quadratic/poly": r"x\^2|x\^\{2\}|quadratic|polynomial|\broots?\b|"
                      r"factor(?:ed|ing)?\b|discriminant|x\^3",
    "inequality": r"\\le\b|\\ge\b|\\leq|\\geq|inequalit|greatest.*less than|"
                  r"smallest.*greater than",
    "modular/divis": r"\bmod\b|modulo|remainder|divisible|divisor|multiple of|"
                     r"\bprime\b|\bgcd\b|\blcm\b|greatest common|least common",
    "geometry": r"triangle|circle|\bangle\b|\barea\b|perimeter|volume|"
                r"rectangle|square(?!\s*root)|polygon|cylinder|sphere|"
                r"diagonal|radius|diameter|parallel|perpendicular",
    "combinatorics": r"probabilit|how many ways|\bchoose\b|arrange|"
                     r"permutation|combination|\bdice\b|\bcoin|\bdeck\b|"
                     r"random|expected value",
    "sequence/series": r"sequence|arithmetic series|geometric series|"
                       r"\bterm\b|fibonacci|recursion|recurrence",
    "function-alg": r"f\(x\)|g\(x\)|\bfunction\b|\bdomain\b|\brange\b|"
                    r"inverse|composition|f\^\{-1\}",
    "trig/precalc": r"\bsin\b|\bcos\b|\btan\b|polar|complex number|"
                    r"\bvector|matrix|\bdeterminant|\\theta|radian",
    "exp/log": r"\blog|\bln\b|exponent(?:ial)?|\b\d+\^x|x\^\{?[a-z]",
    "ratio/percent": r"percent|\bratio\b|proportion|\brate\b|per hour|"
                     r"per minute|average|\bmean\b",
    "base-repr": r"base\s*\d+|base-\d+|binary|hexadecimal|units digit|"
                 r"tens digit|digit sum|digits of",
    "abs-floor": r"\babsolute value|\|x|\\lfloor|\\rfloor|\\lceil|floor|"
                 r"ceiling",
}

def categorize(row):
    text = (row["problem"] + " " + row["solution"]).lower()
    text_raw = row["problem"] + " " + row["solution"]
    cats = set()
    for k, pat in CATS.items():
        if re.search(pat, text if k != "abs-floor" else text_raw,
                     re.IGNORECASE):
            cats.add(k)
    return cats or {"linear-arith-only"}


def main():
    print(f"[sweep] MATH-500: n={len(ROWS)} (measured, never trained on)")

    subj = Counter((r["subject"], r["level"]) for r in ROWS)
    by_subj = Counter(r["subject"] for r in ROWS)
    print(f"\n  SUBJECT x LEVEL (dataset labels):")
    for s, n in by_subj.most_common():
        lv = [subj.get((s, l), 0) for l in range(1, 6)]
        print(f"  {s:24s} {n:4d}  levels 1-5: {lv}")

    at = Counter(answer_type(r["answer"]) for r in ROWS)
    print(f"\n  ANSWER TYPES (the domain question):")
    for k, n in at.most_common():
        print(f"  {k:14s} {n:4d}  ({n / len(ROWS):.1%})")
    ints = at.get("integer", 0)
    print(f"  -> plain-integer share: {ints / len(ROWS):.1%} "
          f"(registered expectation 50-60%)")

    cats_per = [categorize(r) for r in ROWS]
    cat_count = Counter(c for cc in cats_per for c in cc)
    lvl_by_cat = defaultdict(list)
    for r, cc in zip(ROWS, cats_per):
        for c in cc:
            lvl_by_cat[c].append(r["level"])
    print(f"\n  RELATION-NEED CATEGORIES (problem may need several):")
    for k, n in cat_count.most_common():
        ml = sum(lvl_by_cat[k]) / len(lvl_by_cat[k])
        print(f"  {k:18s} {n:4d}  ({n / len(ROWS):.1%})  mean level {ml:.1f}")
    ineq = cat_count.get("inequality", 0)
    print(f"  -> INEQUALITY share: {ineq / len(ROWS):.1%} "
          f"(<10% later chapter / ~30% predicate conversation now)")

    # greedy set-cover shopping list
    built = set()
    remaining = list(range(len(ROWS)))
    print(f"\n  GREEDY SET-COVER SHOPPING LIST (marginal coverage; "
          f"linear-arith base assumed):")
    covered0 = [i for i in remaining
                if cats_per[i] == {"linear-arith-only"}]
    print(f"  (base)  linear-arith-only     covers {len(covered0):4d}  "
          f"cum {len(covered0) / len(ROWS):.1%}")
    built.add("linear-arith-only")
    cum = len(covered0)
    for _ in range(len(CATS)):
        best, best_gain = None, -1
        for k in CATS:
            if k in built:
                continue
            trial = built | {k}
            gain = sum(1 for i in remaining
                       if cats_per[i] <= trial) - cum
            if gain > best_gain:
                best, best_gain = k, gain
        if best is None or best_gain <= 0:
            break
        built.add(best)
        cum += best_gain
        print(f"  +{best:22s} marginal {best_gain:4d}  "
              f"cum {cum / len(ROWS):.1%}")

    # hand-audit sample
    import random
    rng = random.Random(0)
    print(f"\n  HAND-AUDIT SAMPLE (n=20) — eyeball before trusting the list:")
    for i in rng.sample(range(len(ROWS)), 20):
        r = ROWS[i]
        print(f"  [{r['subject'][:12]:12s} L{r['level']}] "
              f"{sorted(cats_per[i])} | ans={answer_type(r['answer']):10s} | "
              f"{r['problem'][:90].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
