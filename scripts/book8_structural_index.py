"""book8_structural_index.py — gut #92's two payments (2026-07-28).

1. THE DRAFTABILITY SCREEN: coarse features over source text predict
   skip-likelihood before the pen spends a draft. The 45+ registered
   skips (three-category taxonomy) are the labeled data. THE FENCE IS
   MANDATORY: the screen ROUTES, never excludes — flagged candidates go
   to the needs-primitive queue feeding the ladder docket; nothing is
   deleted from eligibility (migration law: books move mass across
   zones, not away from hard ones).

2. THE MARINARA: structural search retrieving, for each starved
   construction on the diet queue, the eligible harvest candidates that
   would exercise it — the mint queue becomes a DRAWING ORDER.

Outputs: .cache/book8_draftability_screen.json (per-candidate flags +
needs-primitive queue) and .cache/diet_drawing_order.json (per-customer
candidate lists).
"""
import json, re

h = [json.loads(l) for l in open(".cache/math_harvest_v0.jsonl")]
c = json.load(open(".cache/book8_candidates.json"))
# eligible = current census minus tranche1 (already processed);
# regenerate the live eligible view from the census script's own output
eligible = [x for x in c.get("tranche1", [])]  # tranche1 key holds the restored t1 record
# the true remaining pool: recompute quickly with the same filter the census uses
import subprocess
# Instead of re-running, read remaining from the freshest tranche file's census metadata
t7 = json.load(open(".cache/book8_candidates_t7.json"))
# Rebuild remaining eligible: all candidates the census would emit past t7.
# Simplest faithful source: run the census script logic inline (mirrors book8_candidates.py).
import glob, os
def int_answer(a):
    s = str(a).strip().replace("$", "").replace(",", "")
    return int(s) if re.fullmatch(r"-?\d+", s) else None
filt_idx = [i for i, x in enumerate(h)
            if x["level"] in ("Level 1", "Level 2", "Level 3")
            and len(x["problem"]) < 300 and "asy]" not in x["problem"]
            and all(int(n) <= 300 for n in re.findall(r"\d+", x["problem"]))]
fixture = set(filt_idx[:100])
used_texts = set()
for f in sorted(glob.glob(".cache/book*_prose_pairs*.jsonl")):
    for line in open(f):
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except json.JSONDecodeError: continue
        if "text" in r: used_texts.add(r["text"].strip())
SKIPS = {189,206,230,233,234,237,254,256,277,324,358,371,378,380,404,412,419,431,473,
         489,499,504,539,544,549,553,577,525,589,621,625,643,652,672,636,
         702,707,712,715,716,725,739,746,774,779,782,788,802,807,814,818,838,867,
         869,877,880,888,896,899,914,935,941}
pool = []
for i in filt_idx:
    if i in fixture or i in SKIPS or h[i]["problem"].strip() in used_texts:
        continue
    a = int_answer(h[i]["answer"])
    if a is None or not (0 <= a <= 300): continue
    pool.append({"src_idx": i, "problem": h[i]["problem"], "answer": a,
                 "level": h[i]["level"], "subject": h[i].get("subject","")})
print(f"[index] live eligible pool: {len(pool)}")

# ---- 1. DRAFTABILITY SCREEN (features distilled from the skip registry's reasons)
RISK = {
 "floor-ceil":      r"\\lfloor|\\lceil|\\rfloor|\\rceil|floor|ceiling",
 "complex-domain":  r"\bi\^|\$i\$|imaginary|complex number",
 "exponent-search": r"\^\{?[a-z]\}?|\^\{[a-z0-9+\-]{2,}\}|\blog_",
 "inequality":      r"\\ge|\\le|\\geq|\\leq|inequalit|at least|at most|no more than|greater than or equal",
 "counting":        r"how many (?:integer|value|solution|positive|number)s?|number of (?:integer|solution|value)s?",
 "decimals":        r"\d\.\d",
 "radical-inexact": r"\\sqrt\[|\\sqrt\{[^{}]*[a-z]",
 "primality":       r"\bprime\b",
}
screen = []
needs_primitive = []
for cand in pool:
    p = cand["problem"]
    flags = [k for k, pat in RISK.items() if re.search(pat, p)]
    entry = {**{k: cand[k] for k in ("src_idx","answer","level")}, "flags": flags}
    screen.append(entry)
    if flags:
        needs_primitive.append(entry)
n_flag = len(needs_primitive)
print(f"[screen] flagged {n_flag}/{len(pool)} as skip-risk (ROUTED to needs-primitive queue, NOT excluded)")
from collections import Counter
fc = Counter(f for e in needs_primitive for f in e["flags"])
print(f"[screen] flag distribution: {dict(fc.most_common())}")
json.dump({"pool_size": len(pool), "flagged": n_flag,
           "flag_distribution": dict(fc.most_common()),
           "fence": "ROUTES-NEVER-EXCLUDES: flagged rows stay eligible; queue feeds the ladder docket",
           "needs_primitive_queue": needs_primitive,
           "screen": screen},
          open(".cache/book8_draftability_screen.json","w"), indent=1)

# ---- 2. THE MARINARA: drawing order for the diet customers
CUSTOMERS = {
 "add-dup (X plus X)":        r"twice|double[ds]?\b|two equal|same number.*(?:sum|add)|sum of .* and itself",
 "pct pointers":              r"percent|\\%|%",
 "two-free-variable systems": r"two (?:numbers|integers)|sum of two|product of two|their sum|their product",
 "parallel inversions":       r"(?:=\s*\d+.*){2,}.*(?:product|multiply)|[a-z][a-z]\s*=\s*\d+.*[a-z][a-z]\s*=\s*\d+",
 "subtrahend distractors":    r"less than|fewer than|difference between|exceeds",
 "repeated-constant chains":  r"each (?:day|month|time|hour|stop|step)|every (?:day|time|month)|doubles|halves|triples",
}
order = {}
for name, pat in CUSTOMERS.items():
    hits = [c_["src_idx"] for c_ in pool if re.search(pat, c_["problem"], re.I)]
    order[name] = hits
    print(f"[marinara] {name}: {len(hits)} candidates  {hits[:8]}{'...' if len(hits)>8 else ''}")
json.dump({"note": "the mint queue as a DRAWING ORDER — per starved construction, the eligible "
                   "harvest candidates that would exercise it; the road feeds the gate's "
                   "starvation from its own remaining candidates",
           "drawing_order": order},
          open(".cache/diet_drawing_order.json","w"), indent=1)
print("[index] wrote book8_draftability_screen.json + diet_drawing_order.json")
