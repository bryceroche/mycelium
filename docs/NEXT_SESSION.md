# NEXT SESSION — start here (handoff, 2026-07-09)

Cold-start entry point. Read this first; it points to everything else.

## HEADLINE UPDATE (2026-07-09, supersedes the numbers below)
**The quotable number: 1051/1500 = 70.1% DEPLOYMENT-HONEST end-to-end**
(gold-free acceptance at every stage; answered-precision 0.823, abstention
14.9%). The gold-checked 69.3% is superseded. Arm C (replace-and-solve) FAILED
its soundness gate (55/60 accepted-wrong — forced-unique is not a correctness
certificate on multi-error graphs) but exposed the acceptance bug-class; the
audit then found a second bug (rounds only ever evaluated the withheld
variant, discarding correct repairs). Both in the spec ledger, 2026-07-09.
70 forced-WRONG survivors are deployment-INVISIBLE (the m=1 routing-wall
population presents as success); 226 committed-wrong answers = the measured
customer for a calibrated ABSTENTION signal — the waist-interpolation probe
(parse-side waist ONLY; the tap has no decoder) is queued as its first arm.
New law (2 sightings in 2 days): acceptance criteria are mechanisms — audit
them for imposter rate at deployed error density; never assume soundness.

**WAIST PROBE (2026-07-09, both halves landed):** parse-side waist is SMOOTH
within kind (interpolation coherent: 0.940 sharpness ratio, 0.843
endpoint-match) — KL/VAE machinery PARKED, no deficiency. The abstention
signal WORKS: dense AUC 0.728 on the 226 committed-wrong (clears dense-ranker
AND rare-flag bars; precision@10% 0.417, top-20% recall 0.451). The first
instrument that consults neither the solver nor the emission heads.
Pinned policy: blind abstention loses accuracy — the paying use is
flag-as-NACK-on-accepted-answers (the §8.5 session-monitor role, arrived by
measurement). PENDING: relay adjudicates the flagged-accept second-look
policy; cheap next cut = stage-split of flagged wrongs (are they the 90
one-shot invisibles?). Script: `scripts/waist_abstention_probe.py`.

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
**UPDATE 2026-07-08 late night: the pre-build discriminators RAN and redirected
the build.** The depth probe returned a decisive **ROUTING wall**: a fresh probe
reads the gold value off the CURRENT L4 states at the gold span at **0.996**
digit-exact ON THE 396's own misbound givens (base 0.998; L8 slightly worse
everywhere). The information is fully present; the trained head's pointer
circuit reads the wrong location (§6 attention-bootstrap law). Deeper prefix
RETIRED with a number. "Encode-side" corrected to **ROUTING-side** (operationally
the same: no conditioning fixes it; mechanistically different: the states
contain the answer). Option-4 re-profile: flat — the oracle-64 are same-population
flag stragglers.

THE FORK TO ADJUDICATE (spec ledger has the full entry):
- **(A) span-restricted structural read (probe-as-repair, cheapest):** flagged
  given slot -> pool L4 states over the suspect var's predicted mention span ->
  probe-decodes value (0.996 with gold spans) -> substitute -> re-solve. Zero
  retrain/re-render. Unknown: gold-free span prediction quality on survivors.
  Scope: given_value class (~0.36 of survivor error mass).
- **(B) marker-token re-render v0 (the Alternator build, mechanism = attention
  BEACON):** reserved vocab tokens at suspect spans, forward-only re-encode (no
  trunk backprop — AM hazard stays dissolved). Wrinkle to measure: deployment
  places markers via the model's own routing (marks where it LOOKS).
- **(C) both, sequenced:** A first (immediate bite at the 396), B as the general
  build with A's recovery as its baseline to beat.

DEAD ENDS, measured (do not revive): ledger re-parse as reading-repair (teeth
uniform); suspicion transplant as ranker (P1 flat 0.518); joint-decode-for-swaps
(3x ratio, 4% mass); more rounds/better ranking (oracle ceiling 13.9%);
**deeper prefix (L8 <= L4 on all groups)**. Flag-quality dividend still on the
table (≤64 cases bounded) but cannot touch the 396.

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
