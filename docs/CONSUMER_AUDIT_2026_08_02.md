# THE CONSUMER AUDIT — disjoint/consumed keys (2026-08-02)

*(Delegated audit, run before the framing session per Bryce's ruling.
Question: which consumers of consumed/disjoint framing used which key,
and which banked claims need scope lines. Full table below; the ledger
entry summarizes.)*

## Headline verdicts

1. **BOOKS SCALE DOES NOT INHERIT THE BAD KEY.** The disjoint-86
   census family (Book 2 slope +8, Book 3, gen-13/14 reads, the
   paper's citations) uses gen11_census.py's SKIP_IDX mechanism,
   confirmed correctly populated ("n=86 skipped 14") in battery.log,
   b3battery.log, gen11_chain.log, gen14.log. Sound and unaffected.
2. **A SECOND, INDEPENDENT BUG: the entourage CENSUS_DISJOINT no-op.**
   entourage14.py..entourage22.py pass CENSUS_DISJOINT=1 — a variable
   gen11_census.py NEVER READS (only SKIP_IDX excludes; never passed).
   Every entourage "disjoint census" print since gen-13 was a silent
   FULL-100-POOL read ("n=100 (skipped 0)" in e13/e14/e15/e16/
   entourage20b/entourage21/entourage22 logs). Entourage-13/14 banked
   honest "full-pool" caveats same-day; the caveat was DROPPED by
   entourage-16 and stated flatly as "disjoint census" by 20/21/22 —
   a slow regression of the same discipline that caught the wild-
   ledger bug. Predates and is independent of the candidates-glob
   defect. Stale-key family member.

## Scope lines (applied or referenced)

- docs/NEXT_SESSION.md:39 stale "DISJOINT 0.75" → CORRECTED IN PLACE
  (cold-start doc; the 08-02 seal carries the full correction).
- spec.md ~19747 (deep-clean-2's "DISJOINT n=978 @ 0.75"): SUPERSEDED
  2026-08-02 — widened the glob (66→426) but kept the wrong key
  CONCEPT (waiting-room membership ≠ mix membership). Dead.
- spec.md ~17896 ("THE WILD LEDGER READS," 2026-07-31): the DISJOINT
  column used the pre-widened key (66/1404 flagged — effectively
  near-ALL). The MIXED verdict and the residual/quorum findings are
  ALL-population reads and STAND; the DISJOINT quartile numbers carry
  no evidentiary weight as a disjoint read.
- spec.md ~10525 (ENTOURAGE-20) and ~17750 (ENTOURAGE-22) "disjoint
  census" figures: mislabeled full-pool reads (CENSUS_DISJOINT no-op;
  run logs confirm skipped 0). Compare entourage-13/14's own honest
  caveat at ~4806/~7150.
- The pooled 0.63: the NUMBER is sound (unkeyed, tier==answered); the
  INTERPRETATION (≈ open-admission precision on unseen text) was the
  error, already struck with leaf B.

## Sound consumers (no action)

- gen11_census.py SKIP_IDX family (books verdicts, paper cites).
- book2/5/6_lanes.py + book8_candidates.py candidate-exclusion
  (src_idx / prose_pairs text identity) — answers "already selected
  for annotation," its own question, used consistently.
- wild_settle_v2/pricing, flip_corrected/peritem, silhouette_distance_
  read, binding_invariance_read — filter on tier/correct/mouth only;
  never touched the consumed field.
- census_altitude_prep.py — implements the CORRECTED key (sha ∩
  gen22_mix); the template for the sidecar fix.
- All unrelated English uses of "consumed"/"disjoint" (slot
  consumption, knot disjointness, MATH train/test split, etc.).

## Dockets produced

1. wild_ledger consumption SIDECAR: repoint the consumed computation
   at deployed-mix text identity on the next re-cut (pattern already
   live in census_altitude_prep.py); never rewrite v1 in place.
2. Entourage census semantics: decide what entourage's census read
   SHOULD be now that trained-verbatim ∩ census-pool is checkable
   mechanically (sha ∩ mix); fix rides the next entourage, never
   mid-campaign silently (the permuted_view precedent).
