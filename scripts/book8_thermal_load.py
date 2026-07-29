"""book8_thermal_load.py — the book's thermal output (gut #89, 2026-07-28).

Every BTU cooked into the kitchen is a BTU the AC pays to remove:
accommodation marks (workarounds taught instead of constructions),
MEDIUM provenance grades (rows a future census re-examines), and skip
entries (work deferred to primitives/audits) counted per tranche and
reported as one load figure.

Reading discipline: load trending UP while certification stays clean =
the pipeline buying certification with deferred work — a trade to SEE,
not infer.
"""
import json, glob, os, re

TRANCHE_FILES = {
    1: ".cache/book8_prose_pairs_draft.jsonl",
    2: ".cache/book8_t2_prose_pairs_draft.jsonl",
    3: ".cache/book8_t3_prose_pairs_draft.jsonl",
    4: ".cache/book8_t4_prose_pairs_draft.jsonl",
    5: ".cache/book8_t5_prose_pairs_draft.jsonl",
    6: ".cache/book8_t6_prose_pairs_draft.jsonl",
    7: ".cache/book8_t7_prose_pairs_draft.jsonl",
}
# skip counts per tranche from the registered skip sets (book8_candidates.py
# is the registry; mirrored here as data)
SKIPS = {1: 11, 2: 8, 3: 9, 4: 7, 5: 9, 6: 8}

CERT_FILES = {
    1: ".cache/book8_certification.json",
    2: ".cache/book8_t2_certification.json",
    3: ".cache/book8_t3_certification.json",
    4: ".cache/book8_t4_certification.json",
    5: ".cache/book8_t5_certification.json",
    6: ".cache/book8_t6_certification.json",
    7: ".cache/book8_t7_certification.json",
}

# TWO LINES, NOT ONE (bench refinement 2026-07-28): the skip floor is
# HONEST COST — work the dialect genuinely cannot hold, paid openly;
# its only lever is the primitive ladder (each admitted rung lowers it
# permanently). Accommodation + MEDIUM are DEFERRED COST — work the
# dialect CAN hold but this row routed around or entered with a
# demerit; the lever is the mint diet. One blended number would
# eventually hide the exact signal the meter was built to catch: skip
# floor dropping as primitives admit while accommodations rise keeps
# the total flat while the pipeline quietly buys certification with
# deferred work. Split, both trends stay legible.
out = {}
print(f"{'tranche':>7} {'drafted':>7} {'HONEST(skips)':>13} "
      f"{'DEFERRED(acc+MED)':>17}  {'cert':>4} {'abst':>4} {'wrong':>5}")
for t, path in sorted(TRANCHE_FILES.items()):
    if not os.path.exists(path):
        continue
    rows = [json.loads(l) for l in open(path) if l.strip()]
    acc = sum(1 for r in rows if "accommodation" in r.get("gen", {}))
    med = sum(1 for r in rows
              if "MEDIUM" in str(r.get("gen", {}).get("provenance", "")))
    skp = SKIPS.get(t, 0)
    deferred = acc + med
    cert = abst = wrong = "-"
    cf = CERT_FILES.get(t)
    if cf and os.path.exists(cf):
        c = json.load(open(cf))
        cert, abst, wrong = (len(c.get("certified", [])),
                             len(c.get("abstain", [])),
                             len(c.get("wrong", [])))
    out[t] = {"drafted": len(rows),
              "honest_cost_skips": skp,
              "deferred_cost": deferred,
              "deferred_detail": {"accommodation": acc, "medium": med},
              "certified": cert, "abstain": abst, "wrong": wrong}
    print(f"{t:>7} {len(rows):>7} {skp:>13} {deferred:>17}  "
          f"{cert:>4} {abst:>4} {wrong:>5}")

json.dump(out, open(".cache/book8_thermal_load.json", "w"), indent=1)
print("\n[thermal] wrote .cache/book8_thermal_load.json")
print("[thermal] HONEST lever = primitive ladder; DEFERRED lever = mint diet.")
print("[thermal] deferred-marks note: acc/MEDIUM marks exist only from the t4 "
      "sitting — deferred trend reads honestly from t4 (regime-tag law).")
