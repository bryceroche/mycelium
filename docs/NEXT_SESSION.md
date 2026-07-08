# NEXT SESSION — start here (handoff, 2026-07-09)

## THE TRANCHE IS TRAINED (2026-07-09 late — acceptance in progress)
Registry (MOD+SEL, zero core edits), generator (Vieta+selector+CRT,
symmetry-aware gates), head (ALG2=1 geometry, legacy byte-compatible), and
training (mixed 4500, warm-start, best-by-val **0.809**) are ALL DONE —
spec ledger has the four-seam build log. New ckpt:
`.cache/phase1_algebra2_head.safetensors` (envs: ALG2=1 ALG_CKPT=... ALG_TRAIN/
ALG_TEST/ALG_TRAIN_NAME/ALG_TEST_NAME). **alg2test per-band: 480/800 = 60%
one-shot ANSWER** (legacy was 53.5% on the OLD corpus); fac 0.84-0.92 through
band 4; graph-solve conservative on v2 by design (Vieta root swaps).
ACCEPTANCE REMAINING (the lattice table): (1) old-bigtest regression of the
new head (fired; if fac/ANSWER hold near legacy = no forgetting); (2) TTA-D
vote + certification dials on alg2test with the new ckpt (sentence permutation
is corpus-agnostic; adapt tta_views envs); (3) waist-monitor centroids rebuilt
in the NEW head's fst space; (4) specialist/NACK retrain for the new corpus
(the deployed stack's repair rounds are legacy-trained — the composed stack
number waits on this); (5) the curriculum ablation (one extra run:
coarse->fine teeth anneal vs all-teeth-from-birth, graded on VIEW-ROBUSTNESS
via the TTA harness). Selector silent-error prediction (relay, registered):
check answer-disagreement-despite-graph-agreement on selector samples.

Cold-start entry point. Read this first; it points to everything else.

## NEW HEADLINE (2026-07-09 late): TTA COMPOSITION = 71.5% / 0.833
**COMPOSE 1 (TTA-D vote -> deployed stack): 1072/1500 = 71.5%, precision
0.833, answered 1287 — all dials beat the 70.1%/0.823 floor; gold-free,
deployable (sentence-permutation views + majority 3/5).** TTA-D also: 0.983
precision channel when voting alone; 33/460 + 8/90 routing-wall recoveries
(beats beacon+ratchet combined, zero training); agreement-AUC 0.840 = the
board's best anomaly signal. MC-pi second clause measured: darts must be
independent AND land on the board (oracle re-renders decorrelate best but
parse at ~21% forced — competence, not gold access, is the binding
resource). Progressive-resizing curriculum still registered for the registry
tranche. Next: the registry tranche build (quadratic + modular first per the
band-sweep), TTA voting as standing final stage.

**PORTFOLIO + THRESHOLD (2026-07-09 night):** certification channel CONFIRMED
— unanimity 5/5 = **0.9982 precision @ 38.1% coverage** (570R/1W); 4/5 =
0.9925 @ 44.3%. Portfolio: combo AUC bar failed (0.833 < 0.840,
Spearman 0.464) but the combo WINS at every abstention operating point
(kept-precision 0.862 vs 0.846 @ 10%) — AUC was the wrong instrument for a
tail decision (§6 4th sighting, my own registration). Deployed abstention =
rank-sum combo at the tail. K>5 sweep priced (~7 min/view), deferred.
Curriculum metric reframed: view-robustness, graded by the TTA harness from
each relation's first training run.

## THE SURVIVOR CHAPTER IS CLOSED (2026-07-09 — read this first)
**Beacon verdict: 14/460 = 3.0% — the population is DETECT-AND-ABSTAIN ONLY
under current machinery.** Every repair arm is now measured: conditioning dead
(oracle ceiling 13.9%), symbolic replace unsound (92% imposters), ratchet fix
6%, beacon 3% with 71 wrong per 14 right. Footnote: 11 beacon recoveries were
in the hard-396 (input perturbation occasionally moves what conditioning
can't — existence result, not mechanism). The working instrument: the waist
monitor (AUC 0.728, zero params — the §8.5 session-monitor role, filled).
Quotable: **70.1% deployment-honest / 0.823 precision**, precision dial to
0.880 at 0.615 coverage-accuracy. **NEXT CHAPTER: registry expansion toward
MATH-500 relations** (quadratics, inequalities — band-sweep decides), with the
week's design constraint: every new relation's pointer gets candidate
restriction + span supervision FROM BIRTH, or it grows its own 396. Deferred
until a flaggable-AND-fixable population exists: v1 ratchet (per-space
centroids + anomaly-decrease dominance), score-gated beacon.

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
measurement). Script: `scripts/waist_abstention_probe.py`.

**RATCHET-NACK v0 (2026-07-09, ran):** kill bar FIRED (break=1 at top-20% —
dominance leaks; anomaly-decrease criterion is load-bearing, needs per-space
centroids for v1). Fix rate ~6% vs 0.346 bar — refuted with mechanism: the
flagged stage-0 wrongs ARE the routing-wall invisibles (near-misses BECAUSE
unfixable). New lesson: selection effects have jurisdictions too. Detection
and repair are separate capabilities: monitor sees (0.728), specialist can't
fix what it sees. Dials: recovery mode flat (+1); precision mode real (0.880
precision at 0.615 end-to-end, top-20%) — for wrong-costs-more deployments,
not MATH-500. Stage-1 detector dead in-stage (0.532); correct parses drift
anomalous with stage — per-stage calibration before any v1. NEXT DECISIONS:
(a) beacon arm on the flagged population (flipped prediction standing —
failure = detect-and-abstain-only verdict); (b) v1 ratchet only if per-space
centroids + anomaly-decrease close the leak. Script: `scripts/ratchet_nack.py`.

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

## BRICK-P IS IN FLIGHT (2026-07-09 — the parser learns to breathe)
Bryce's gut ("we're not breathing right") diagnosed + built same day: the
parser was a ONE-SHOT parallel decoder inside a settling project. BRICK-P =
v98 recipe transplanted to the slot banks with the relay's two amendments:
(1) SPLIT BARS — collisions (swaps/inconsistent pairs; breathing's
jurisdiction, should move) vs LONE misbindings (internally consistent;
pointer law says may not move) counted SEPARATELY or the result is mush;
(2) MASKED slot-to-slot attention (evidence-sharing topology from the
model's own breath-0 parse: same-sentence OR shared-var; mean degree
10.9/24) — free-form is the perceiver trap, not built. Deltas via zero-init
W_bo + init-closed gates (breath-0 == incumbent at init). ALG_BREATH=K env;
ckpt .cache/phase1_breath_head.safetensors; K=2, 8000 steps, warm from
tranche head (~0.86s/step). RUNNING: train + per-band evals both domains.
NEXT SESSION MUST: (a) write the SPLIT-BAR eval (collision-rate vs
lone-misbinding-rate vs invisible count, breath vs tranche incumbent, both
domains — the registered verdict); (b) if bars move: K sweep + TTA-compose
on the breathing head; (c) still queued: NACK hygiene (3-cause confound
registered), TTA-compose dump, monitor rebuild + drift, curriculum ablation.
