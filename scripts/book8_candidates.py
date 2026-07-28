"""book8_candidates.py — BOOK 8, stage 0 (2026-07-28): candidate census.

The road at delegation tempo, under the two-axis gate (ledger 2026-07-28,
THE WORD). Book 7 hunted the zero-class; book 8 resumes the GENERAL road:
next tranche of in-reach L1-3 harvest problems not yet annotated by any
book and outside the census fixture.

Filter (the standing in-reach definition, book2_lanes.py lineage):
  level in {L1,L2,L3}, len(problem) < 300, no 'asy]',
  every integer literal <= 300,
  answer parses as an integer in [0, 300]  (dialect value law),
  NOT in the census fixture (first 100 in-reach — measurement fixture),
  NOT already consumed by books 1-7 (src_idx across all prose_pairs files,
  book7 draft included).

Output: .cache/book8_candidates.json — [{src_idx, problem, answer, level,
subject}], ordered by src_idx. Tranche 1 = first B8_TRANCHE (default 40).
"""
import json, re, glob, os

B8_TRANCHE = int(os.environ.get("B8_TRANCHE", "40"))

h = [json.loads(l) for l in open(".cache/math_harvest_v0.jsonl")]

def int_answer(a):
    s = str(a).strip().replace("$", "").replace(",", "")
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    return None

filt_idx = [i for i, x in enumerate(h)
            if x["level"] in ("Level 1", "Level 2", "Level 3")
            and len(x["problem"]) < 300 and "asy]" not in x["problem"]
            and all(int(n) <= 300 for n in re.findall(r"\d+", x["problem"]))]
fixture = set(filt_idx[:100])  # the census pool — never a substrate source

# CONSUMPTION IS TRACKED BY SOURCE TEXT, NOT src_idx: the src_idx field is
# candidate-relative in book 2, a third space in book 3, and harvest-global
# only from book 4 onward (verified 2026-07-28 — book2 src 13 == its own
# candidate list's [13], not harvest[13]). The text field of every
# prose_pairs row is the source problem verbatim; text identity is the one
# key that is index-space-invariant.
used_texts = set()
for f in sorted(glob.glob(".cache/book*_prose_pairs*.jsonl")):
    for line in open(f):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "text" in r:
            used_texts.add(r["text"].strip())

# Registered honest skips from tranche 1 (2026-07-28, book8_draft_report.md):
# inexpressible cores — floor(), prime factorization, table lookup,
# exponent-law magnitudes over the cap, discriminant-sign, inequality
# counting, inexact roots. Held out of later tranches; a skip is a verdict.
TRANCHE1_SKIPS = {189, 206, 230, 233, 234, 237, 254, 256, 277, 324, 358}
# Tranche 2 skips (2026-07-28, book8_t2_draft_report.md): discrete-log,
# fractional exponents, interval bracketing, primality reasoning,
# comparison counting, exponent bookkeeping over the cap.
TRANCHE2_SKIPS = {371, 378, 380, 404, 412, 419, 431, 473}
# Tranche 3 skips (2026-07-28): decimal clearing over cap, inequality/
# bracket-search, unknown exponents, combinatorial argmax. Plus [525],
# bench-WITHDRAWN post-certification (coincidental correctness — the
# ordering lives over a domain the dialect cannot hold).
TRANCHE3_SKIPS = {489, 499, 504, 539, 544, 549, 553, 577, 525}
# Tranche 4 skips (2026-07-28): discrete-exponent family x4, counting
# primitives x2. CATEGORY NOTE (bench 2026-07-28): operation-shaped skips
# may earn primitives via the #49 ladder; DOMAIN-BOUNDARY skips (e.g.
# t5's [746] complex numbers) are a different category — a number system
# is not an operation and never argues a rung. Plus [636], bench-WITHDRAWN post-certification
# (coincidental correctness in canonicality's clothes — the q=1
# remainder constraint has no in-graph witness).
TRANCHE4_SKIPS = {589, 621, 625, 643, 652, 672, 636}
TRANCHE1_SKIPS |= TRANCHE2_SKIPS | TRANCHE3_SKIPS | TRANCHE4_SKIPS

cands = []
for i in filt_idx:
    if i in fixture or i in TRANCHE1_SKIPS or h[i]["problem"].strip() in used_texts:
        continue
    a = int_answer(h[i]["answer"])
    if a is None or not (0 <= a <= 300):
        continue
    cands.append({"src_idx": i, "problem": h[i]["problem"],
                  "answer": a, "level": h[i]["level"],
                  "subject": h[i].get("subject", "")})

print(f"[book8] in-reach total: {len(filt_idx)}  fixture: {len(fixture)}  "
      f"consumed texts across books: {len(used_texts)}")
print(f"[book8] eligible candidates: {len(cands)}  tranche 1 size: "
      f"{min(B8_TRANCHE, len(cands))}")
out = {"filter": "L1-3, len<300, no asy], literals<=300, int answer in [0,300], "
                 "fixture+used excluded",
       "eligible_total": len(cands),
       "tranche1": cands[:B8_TRANCHE],
       "remaining": len(cands) - min(B8_TRANCHE, len(cands))}
json.dump(out, open(".cache/book8_candidates.json", "w"), indent=1)
print("[book8] wrote .cache/book8_candidates.json")
