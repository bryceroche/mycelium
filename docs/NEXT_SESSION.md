# NEXT SESSION — start here (handoff, 2026-07-08 night)

Cold-start entry point. Read this first; it points to everything else.

## Where we are (one paragraph)
**The survivor anatomy is COMPLETE and the wall is measured.** End-to-end on
teeth-hardened algebra: **1,040/1,500 = 69.3%** gold-free forced-correct (797
one-shot + 243 multi-round repair). The multi-round remainder (460 survivors) was
profiled through SIX registered probes in one arc — five refutations (teeth
uniform; multiplicity flat AUC 0.524; omission-blindness dead; suspicion-rank flat
0.518; rel_args binding DE-enriched 0.78x) — converging on a named partition:
**DECODE-side errors (drained by the repair stack, always front-loaded
123→39→5→0) vs ENCODE-side casualties (info mis-committed in the frozen
precomputed trunk states; no head-side conditioning can recover it — every repair
round re-decodes the SAME encoding)**. The capstone ORACLE-FLAG CEILING: perfect
gold-derived per-field flags, re-derived each round × 4 rounds, recover only
**64/460 = 13.9%** — the other **396 (26.4% of the corpus) are unrecoverable by
ANY flag-quality or repair improvement**. Full chain + numbers: spec ledger
(`docs/phase1_skeleton_spec.md`, 2026-07-08 entries).

## THE PENDING DECISION (relay + Bryce — this is the seam)
The frontier is now a BUILD choice, not a probe. Options on the table:
1. **Change the encoding between rounds (second-view re-render).** Re-render the
   problem text with POSITION-ALIGNED suspect marks (features, NOT prose — the §6
   structure law; the old text-NACK arm was content-blind and this is why) and
   recompute trunk states for round-2+ on failures only (~700/1500). Cost: per-round
   trunk forward for failures; residency fine (2.9GB total, 21GB headroom).
2. **Deeper prefix (L0–L7).** Richer encoding; attacks the same wall from the
   capacity side. Cost: recompute the whole trunk-state memmap + retrain heads;
   measure whether binding survives deeper before committing (a probe can gate this).
3. **Capture the flag-quality dividend (small, deployable, cheap).** The deployed
   flag deriver is doubly OOD vs the specialist's training (withhold-derived,
   all-fields pattern vs gold per-field masks). Oracle bounds the win at ≤64 cases
   (≤ +4.3 pts end-to-end); a gold-free per-field confidence deriver might capture
   part. Worth doing regardless of 1/2; do not oversell — it cannot touch the 396.
4. **Registered but unfired:** partition + re-profile the 64 oracle-recovered vs
   the 396 (the pinned 10–30% rule's follow-up; zero GPU — profile npz exists:
   `.cache/survivor_profile_bigtest.npz`).

DEAD ENDS, measured (do not revive): ledger re-parse as reading-repair (premise
refuted — teeth uniform); deducer-suspicion transplant as ranker (localization
already adequate — P1 flat); joint-decode-for-swaps (3x ratio but 4% mass);
more rounds/better ranking (oracle ceiling 13.9%).

## Ledger lessons promoted this arc
- **An enrichment bar without a MASS bar is a trap** (CUT 3b: 3x ratio on a 4%
  slice passed the pinned bar; the conclusion still couldn't fire). Register the
  mass a mechanism must explain, not just the ratio.
- **Real causes have jurisdictions** (relay; 1 sighting from §6): binding weakness
  is thrice-real for ERRORS yet governs neither survivorship (teeth uniform) nor
  unfixability (rel_args 0.78x). Locating a real cause of X does not license it
  as the cause of adjacent-X.
- Front-loaded repair decay (4 independent sightings today) = a population
  boundary, never a difficulty gradient. Expect it; read it as a partition.

## Standing state (unchanged)
- Parser ckpt: `.cache/phase1_algebra_head.safetensors` (best-by-val 0.783);
  specialist: `.cache/phase1_algebra_nack.safetensors` (pure curriculum, NACK_SPLIT
  mining). Corpora: bigtest 1500 (measurement), repair 3000 seed 21 (mining).
- Equivalence class ~17% = design parameter (16.6/17.2 across independent draws);
  forced-answer is the uniform honest metric everywhere.
- KenKen composed stack 47% gold-free; Job-B GSM8K calculator band 184/186.
- Scripts of the arc: `grade_equivalence.py`, `characterize_survivors.py`,
  `survivor_multiplicity.py`, `survivor_suspicion_rank.py` (CUTs 2/3/3b),
  `survivor_oracle_ceiling.py`. All zero-GPU-retrain; all replay-based.
- Deferred, still answerable: algebra-side teeth check; registry expansion toward
  MATH-500 relations; equivalence-aware fac metrics; purity-dip disambiguation.
