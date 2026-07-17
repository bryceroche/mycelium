# Phase-1 skeleton — the NL parser (v0 spec, 2026-07-05)

**Status:** design settled in session 2026-07-05; NOTHING BUILT YET. This is the build
plan for the first Alternator Phase-1: a frozen-trunk parser that turns templated NL
into factor-graph deltas in the deducer's own vocabulary, closing the loop so the
alternation exists at all. **No external API calls anywhere — the point of the project
is the small-footprint local model.** Context: `docs/phase1_construction_brief.md`,
`docs/phase1_prep_grounding.md`, CLAUDE.md §8.

## 0. Scope discipline (what this skeleton is and is not)

- **It is the MECHANISM testbed** (Job A): does the Alternator loop work — cycles,
  deltas, notebook conditioning, NACK response, zero-LoRA null, parse-side silhouette.
  Domain = KenKen-in-words, where gold graphs are free, the deducer is trained, the
  verifier is exact, and the full NACK stack (calib head / VIOLATED flags / late-JSD
  0.687) is live on the deduce side.
- **It is NOT the value probe** (Job B — does parse→solve beat LLM-direct on real math).
  DEFERRED WITH A DATE, not dropped: the fully-local diagnostic is *decisions-per-problem*
  from the search tier over MATH-500-style problems (0 decisions = calculator band,
  >0 = engine band — maps where the bet lives with our own machinery only). **Milestone
  gate: run it once the skeleton's Brick-A measurable is in.** Job A must not silently
  absorb months.

## 1. The two jaws (the corrected metaphor — bake it in)

NOT "one jaw for hard constraints, one for soft." On every problem the jaws cooperate
in the validated division of labor:

- **Symbolic jaw** (GAC / MRV / LCV / verifier / backtracking): DISPOSES — commits only
  what is logically forced. Needs exact predicates to bite.
- **Neural jaw** (the deducer): PROPOSES — ordering priors, soft-graph inference,
  differentiable critique. Never commits.

Hard-vs-soft determines **which jaw can bite at all**: on clean constraints the
symbolic jaw dominates for free (the honest negative); on soft / learned / NL-uncertain
constraints it has nothing to chew and the neural jaw is the only one working — the
Alternator's target regime. Marine version: grouper + moray hunting COOPERATIVELY,
each reaching where the other can't (the moray has the pharyngeal second jaws;
the grouper signals it into the crevices).

## 2. Architecture

- **Trunk:** Llama-3.2-1B **L0–L3, FROZEN**, 2048d (weights on disk:
  `.cache/llama-3.2-1b-weights`). Frozen is the falsifiability of the zero-LoRA null:
  all per-cycle behavior change must come from INPUT conditioning (notebook + NACK),
  not weight mutation. Fallback ladder if the null fails: rank-≤16 LoRA on the parser
  only (§8.4) — never the deducer trunk.
- **Parse-side WAIST (built from birth, day one):** a readout bottleneck just before
  the delta head — 2048 → **512d**, importance-ordered dims, Matryoshka prefix mask
  **512→128** (two schedule axes per §8.6: training-time handicap + per-cycle
  coarse-narrow/fine-wide). This is the **parser's silhouette tap** — the object the
  perceiver segments and the BirdNET re-run reads. It mirrors the deducer's *tap*
  (readout probe), NOT its dormant in-loop waist. Capture hooks from birth
  (capture-once schema, token-position × waist-dim).
- **Delta head:** slot-based parallel emission (§3). Trained; the only trained
  parameters besides the waist projection in v0.
- **Deducer (Phase 2):** untouched, exactly §1–§2 of CLAUDE.md. Never sees NL.

Budget: frozen L0–L3 + embeddings fp16 ≈ 1 GB; deducer fp32 ≈ 0.4 GB; fits 24 GB with
room for activations. Smoke-test the combined residency before anything else.

## 3. The delta-head output format (v0 — the expensive-to-change object)

**Slot-based, parallel, no autoregression** (the §6 no-mid-breath-token-generation law
applies to the parser too: one structured emission per cycle, not a token stream).
L_out fixed factor slots (DETR-style), each slot emits:

| field | form | supervision |
|---|---|---|
| presence | 1 logit | BCE vs gold (slot count varies per cycle) |
| type | logits over the REGISTRY MENU (v0 = the deducer's latent types: `row`, `col`, `cage`) | CE |
| op (cage only) | logits over OP_VOCAB `{given,add,sub,mul,div}` (ids 0–4, existing) | CE |
| target (cage only) | **digit-spaced**: 3 × 10-way digit heads (§6 law: never whole-number classes). Exact integer — the SYMBOLIC verifier needs exact targets; the inlet's log-buckets are derived downstream via the existing `target_to_bucket` | 3 × CE |
| membership | pointer distribution over the s_max=49 cell slots (multi-hot) | per-cell BCE vs gold multi-hot |

- **Attention-bootstrap law compliance (§6):** the membership pointer is a ~49-way
  attention pathway — it does NOT bootstrap from task gradient. It doesn't have to:
  gold deltas give **direct supervision** from step 0 (the sanctioned escape,
  observed 4× as the requirement).
- **Slot↔gold matching:** v0 = CANONICAL ORDER (sort gold factors by
  (type, first-member cell index)) with positional supervision. Hungarian matching is
  the fallback if positional proves brittle — do not build it preemptively.
- **Registry (SEMANTICS channel):** ONE menu, fixed meaning across cycles (§8.2).
  v0 menu == the deducer's existing vocabulary verbatim (format-definer role:
  the deducer's membership + inlet vocabulary IS the target). Centroids live in one
  shared embedding space; classification is cosine-to-centroid — the same space the
  perceiver's segmenter reads later.
- **Ball (TOPOLOGY channel):** v0 = the parser emits MEMBERSHIP DIRECTLY; the Poincaré
  ball sits behind a flag as a strictly-additive upgrade (§7's relaxation is blocked;
  hard masks are the permanent fallback BY DESIGN — alternation must not wait on
  unsolved geometry).
- **Notebook (TEMPORAL memory):** WRITTEN from the deducer's silhouette tap
  (readout-LN — where all evidence lives) + the NACK features; READ by the parser as
  prefix conditioning (the concrete mechanism the zero-LoRA null stands on). Ledger
  (append-only, committed facts: settled assignments, verified factors) + scratch
  (replace, provisional). NACK feature vector v0: per-factor VIOLATED flags (symbolic,
  exact, free) + the per-cell late-breath belief-JSD field (gold-free, AUC 0.687) +
  cycle index.

## 4. The template generator + gold labeling (the other expensive object)

Generate KenKen with the existing builder, render to NL:

- **Preamble** (one sentence): grid size + the row/col all-different rule ("Solve the
  5×5 KenKen: every row and every column contains 1–5 exactly once."). The parser must
  emit ALL row+col factors from this ONE sentence — the deliberate one-sentence→many-
  factors case.
- **One sentence per cage**, template bank with paraphrase variation ("The cage
  covering r1c1 and r1c2 multiplies to 12" / "Cells (1,1) and (1,2) have product 12" /
  …). **Givens** as their own sentences ("Row 3, column 4 is a 5.").
- **GOLD = (char/token span ↔ induced factor(s)) alignment**, emitted by the generator
  for free. This single labeling scheme supervises THREE things at once:
  1. the delta head (which factors, per cycle);
  2. **band masking / remove-at-read on the INPUT side**: once a factor is matched to
     a registry centroid (committed to the ledger), its token span is masked at read —
     explained-away text drops out and each later cycle parses the RESIDUAL unexplained
     text (Law 7 applied to input; the polarized-sunglasses mechanism with no Fourier
     machinery);
  3. segmentation gold for the parse-side BirdNET re-run (token spans ARE the calls).
- **Curriculum knobs:** template diversity, paraphrase depth, sentence-order shuffling,
  distractor sentences, factors-per-cycle cap (forces genuine multi-cycle parses).
- **NACK curriculum:** corrupted-parse variants (wrong target / wrong member / wrong op
  injected mid-session) with gold retransmissions — Brick-C-v0's training and eval data.

## 5. The energy wave (Phase-1 form — HYPOTHESIS, say so)

Per-token residual energy across the SEQUENCE, per cycle — the parse-side analog of
the deducer's per-breath wave. Hypotheses to test (not assume):
- energy peaks align with semantically dense spans (quantities, relations);
- the wave's spatial frequency ALONG THE TOKEN AXIS carries the band structure —
  coarse cycles the low-frequency envelope (document scaffold), fine cycles the
  high-frequency detail (exact operands);
- band masking (§4.2) is the mechanism that *changes* the wave cycle-to-cycle.
Parse-side priors genuinely favor the bird pipeline (text is narratively banded;
loosely-coupled sentences compose far more linearly than joint deduction) — but the
deduce-side lesson stands: MEASURE, the field not the summary statistic, before
building any machinery on it. Re-run the exact capture-once protocol
(`scripts/capture_silhouette_trajectories.py` pattern): banding + linearity on
token-position × waist-dim, the moment the skeleton trains.

## 6. Build order + measurables + kill gates

1. **Residency smoke:** frozen Llama L0–L3 + deducer co-resident on the 7900 XTX.
2. **Template generator + gold labeling** (§4) — CPU, selftested.
3. **Delta head + waist** (§3, §2) supervised on single-cycle parses (no alternation
   yet): parse-accuracy vs gold deltas is the unit test.
4. **Brick-A (the zero-LoRA null):** cycle-conditioned input (blank notebook vs rich
   ledger + NACK features) through the SAME frozen trunk — does parse behavior change
   appropriately with NO weight mutation? Kill: if conditioning can't switch bands,
   engage the LoRA fallback ladder (§8.4) and say so.
5. **Brick-C-v0 (NACK response):** inject corruption → verifier VIOLATED + late-JSD
   flag it → NACK features next cycle → retransmission accuracy vs a no-NACK control.
   **The §8.3 gate is binding: if NACK-response does not beat staged parse-then-solve,
   the alternation folds back to the simpler staged design by its own spec.**
6. **Parse-side BirdNET re-run** (§5) — free once (3) trains.
7. **Milestone gate (Job B, local-only):** decisions-per-problem over MATH-500-style
   problems via the search tier — map the engine band. Runs no later than Brick-A
   completion.

## 7. Honest open risks

- 1B-class comprehension may be too weak even for templated NL at high paraphrase
  diversity — the curriculum knobs are the instrument; find the ceiling honestly.
- The zero-LoRA null may fail (§8.7 #1) — the fallback ladder exists; failing the null
  is a RESULT, not a defeat.
- Slot-based emission with canonical-order matching may be brittle under sentence
  shuffling — Hungarian fallback documented above.
- The engine-band question (Job B) remains existential and open — the milestone gate
  keeps it in the crosshairs.

## 8. Build log (results as they land)

- **2026-07-05/06 — steps 1-3 BUILT.** Generator (`scripts/kenken_nl_gen.py`): span-SET
  gold + split-ref family + round-trip-as-generation-gate (4,140/4,140 samples pass;
  labeling bugs die at generation). Residency (`scripts/phase1_residency_smoke.py`):
  2.9GB co-resident, ~21GB headroom, JIT replay 0.34s, no AM hazards, deducer
  unperturbed (0.745). Delta head (`scripts/phase1_delta_head.py`): 3.2M params,
  three loss-decomposition-diagnosed design iterations (TEXT-ORDER slots — canonical
  order made slot->sentence assignment a circular grid-sort; membership pos-weight 5.0;
  per-token SENTENCE-INDEX embedding — one attention hop matches a discrete code but
  cannot COUNT sentences: the attention-bootstrap law's quieter cousin).
- **Data scale is the current lever:** 300 samples -> memorization (train mem loss
  0.002, test F1 0.76); 3,060 samples -> reading (factor exact 0.60 -> **0.748**,
  op 0.944, target 0.865, member F1 0.891). One-shot solve rate 0/60 — on-model at
  p^~20 compounding; NOT the Alternator's operating point (see below).
- **Matryoshka answer (parse side, first pass): the parse signal is LOW-DIMENSIONAL.**
  Width 128 ~= width 512 on every head (factor exact 0.724 vs 0.748; op/type flat;
  the ~2pt cost sits in the fine-detail heads, target/member). Head-aware waist
  scheduling not currently justified — uniform prefix suffices.
- **ERROR TAXONOMY (2026-07-06, the pre-Brick-C gate): 60/60 parse errors are
  SYMBOLICALLY DETECTABLE — zero SILENT.** 44 UNSAT + 16 malformed; DETECT_multi 0
  (over-constraint dominates); every failure involves a membership error. So at
  factor-exact 0.748, the one-shot pipeline solves 0% but the symbolic NACK has 100%
  recall on flagged problems — the alternation-earns-its-cost story stated
  quantitatively: staged one-shot 0%, NACK-recoverable ceiling 100%. CAVEATS: n=60;
  detectability~=1 leans on KenKen being densely over-constrained (rows/cols
  interlock every cage) — sparser domains will have a real SILENT class; UNSAT flags
  the GRAPH, not the factor — Brick-C-v0 needs a localization story (unsat-core-ish
  or per-factor blame) on top of the flag. Tier-3 late-JSD recall sits above this
  ceiling, unmeasured.
- **NEXT:** membership exactness is THE lever (present in 100% of failures) —
  scale data further / targeted membership curriculum; then Brick-A (notebook/NACK
  conditioning, zero-LoRA null), Brick-C-v0 (retransmission vs the no-NACK control),
  parse-side BirdNet re-run on the captured waist silhouettes.
- **THE EXPIRATION CONDITION on the 100% (write it before it bites):** zero-silent is
  partly a property of the current ERROR MIX, not only of KenKen's density — membership
  errors (100% of today's failures) are the structurally LOUD kind (a mispointed cage
  fights the row/col lattice). As membership approaches exactness, residual failures
  rotate toward target-digit/op errors — the plausible coherent-misreading class — and
  a SILENT class can be BORN exactly as the parser improves. Discipline: the taxonomy
  is cheap; RE-RUN AT EVERY ACCURACY CHECKPOINT and track detectable-fraction AS A
  FUNCTION OF factor-exactness. Pinned at 100% while the mix rotates = a much stronger
  claim; a silent class at 0.9+ = learned while the tier-0 confidence head can still
  be designed to catch it.
- **Brick-C localization v0 = the DELETE-ONE-FACTOR BLAME SWEEP:** remove each parsed
  factor in turn, re-solve; SAT-on-removal fingers that factor as unsat-core member.
  O(F) search-tier calls at median-zero decisions ≈ less than one deducer breath; no
  new machinery (the taxonomy's bridge). Not minimal-core (overlapping errors smear
  blame — measure it). The tier-0 confidence head later slots in as the sweep ORDER
  (least-confident first): propose/dispose fractally repeated INSIDE the NACK.
- **The guard: DETECTION IS NOT CORRECTION.** 100% is a detection ceiling; Brick-C
  must demonstrate that a NACK-conditioned re-parse FIXES the flagged region rather
  than re-emitting the same wrong membership. Encouraging structure (retransmission
  is an easier problem: ledger pins the verified, attention has fewer places to go)
  is exactly the kind of claim this project measures rather than assumes.
- **BLAME SWEEP v0 MEASURED (2026-07-06) — an honest negative with a design lesson:**
  delete-one-factor re-solve on the 44 UNSAT parses: precision **1.000** (when it
  fires, it is right), recall **0.034** (it fired on 2/44 — at factor-exact 0.75 a
  parse carries ~5 wrong factors and single deletion cannot restore SAT). So
  delete-one is the SINGLE-ERROR-REGIME tool: its usefulness co-improves with parser
  exactness. At realistic multi-error density, SYMBOLIC localization alone fails —
  which MOTIVATES (by measurement, not taste) the neural NACK tiers: (a) the
  add-back sweep (start rows/cols/givens, add cages in tier-0-confidence order,
  blame additions that turn UNSAT), and (b) the deducer's soft-solve suspicion field
  (per-region late-JSD on the parsed graph — the tier-3 role the Alternator assigned
  it a priori). The propose/dispose fractal is now REQUIRED inside the NACK, not
  merely elegant.
- **REGISTERED PREDICTION #2 (2026-07-06, ahead of the 40k curve):** the SILENT class
  is born first in TARGET-DIGIT errors on `add` cages (most compensating-coincidence
  room — many operand sets reach the same sum) and stays near-zero for membership as
  long as the row/col lattice polices. Where silents appear = where the tier-0
  confidence head's calibration matters most.
- **Brick-A operationalization (pre-specified as a FIELD, per the instrument lesson):**
  the zero-LoRA read is DIFFERENTIAL — same puzzle, same frozen weights, blank-notebook
  input vs ledger+NACK input; the measure is whether the parse DELTA (attention mass +
  emission changes) CONCENTRATES on the flagged region vs a global reshuffle. A scalar
  accuracy delta is the summary that will degenerate; specify the localization field
  from the start.
- **CURVE POINTS 2-3 + THE FIRST SOLVE (2026-07-06, 40k corpus, 24k steps):**
  factor exact 0.753 -> **0.780** (op 0.967 / target 0.907 / member F1 0.924, all
  climbing) and **SOLVE RATE 1/60 — the first end-to-end NL -> parse -> symbolic
  solve -> correct grid**, at BOTH Matryoshka widths (128d factor exact 0.749; the
  solve survives narrow). Taxonomy: **third consecutive 100% detectable** (59/59;
  zero silent) while the mix rotates (target-digit errors growing share, 28/52 UNSAT;
  membership still in every failure; presence split: phantoms 9 vs dropped 5 — mixed,
  mildly favoring the tier-0 presence-gate preview). Prediction #2 remains properly
  unfalsified (trigger condition — mix clearing membership — not yet reached).
  **BLAME CO-IMPROVEMENT MEASURED:** delete-one recall 0.034 -> **0.067** (doubled,
  restored 5/52 vs 2/44; precision 1.0 -> 0.8) as the parser improved — the
  single-error-tail convergence claim now has its first two points.
- **THE FINE-CADENCE CURVE (2026-07-06 night, 7 points, 12k->44k steps):**
  factor exact 0.753 -> 0.780 -> 0.805 -> 0.814 -> 0.827 -> 0.825 -> 0.829 (PLATEAU
  ~0.83 — the cheap-steps lever is exhausted; loss flat at ~3.34 with span at its
  entropy floor); SOLVE RATE 0 -> 1 -> 1 -> 2 -> 5 -> 5 -> 5 /60 (8.3%).
  **SEVEN consecutive 100%-detectable taxonomy points, zero silent**, while the mix
  rotated hard toward target digits (dominant cage-field error) — prediction #2 STILL
  unfalsified with its trigger zone now fully occupied: target errors dominate and no
  silent has appeared. DETECT_multi births at stage 3 (dropped factors -> under-
  constrained -> caught by the uniqueness probe — the third detection channel now
  live). NEW CLASS OBSERVED: CORRECT-with-field-errors (up to 2/5 solves carry a
  member mismatch) — wrong-but-EQUIVALENT parses that solve to gold: benign errors,
  worth excluding from exactness metrics later.
  **DECISION POINT: the 0.83 plateau means the next lever is architectural (second
  attention hop / iterative refinement) OR the loop itself (Brick-A/C — the design's
  actual answer to imperfect parsers). Brick-A is next per the queue, and now inherits
  a stable, characterized parser.**
- **NAMED: the packing finding is NEURAL COLLAPSE** (Papyan/Han/Donoho — terminal-phase
  CE pulls class vectors toward a simplex equiangular tight frame; K=7 ideal cosine
  -1/6 ~= -0.167; measured -0.048 = collapse one-third realized, penalty OFF).
  **REGISTERED PREDICTION #3 (the mechanism test):** training with a simplex-ETF
  codebook penalty shifts confusion mass SPECIFICALLY off the thin walls (6-7 first,
  1-2 second — the rho=0.53 geometry-predicts-confusions link is the mechanism). If
  accuracy rises but the confusion matrix reshuffles uniformly, the geometry story
  was correlation, not cause. (Deducer-side; fire on a free training slot.)
- **SEMANTIC EXACTNESS (the wrong-but-equivalent consequence):** factor-exact
  systematically UNDERSELLS the parser — wrong-but-equivalent parses solve to gold.
  The honest per-parse metric is SOLUTION-SET EQUIVALENCE (solve the parsed graph,
  compare grids — the round-trip machinery computes it free). Long-term (Job B):
  real math admits many valid formalizations; grading on syntactic match to one
  canonical gold graph punishes legitimate readings. **The generator's unique-solution
  guarantee is what makes equivalence CHECKABLE — preserve it as a HARD REQUIREMENT
  when the domain widens.**
- **LEVER ORDERING (settled by the plateau):** Brick-A next (wants the stable parser),
  Brick-C after (55 flagged failures are its specified input; a plateaued parser with
  perfect error visibility makes any loop gain UNCONFOUNDABLE with parser improvement),
  the architectural lever (second attention hop) stays PARKED unless the loop's
  ceiling proves too low. Information value over fun.
- **Prediction #2 status note:** the trigger zone is fully occupied (target digits
  dominant) with no silent birth through 7 points. If the window closes without one,
  record THAT as the finding — KenKen's lattice polices even compensating-coincidence-
  rich errors — which sets the baseline for when sparser domains DO grow a silent class.
  Either resolution is informative.

## 9. Brick-A design (registered 2026-07-06, before build)

1. **Conditioning enters TRUNK-LEVEL** (prefix embeddings prepended to text, flowing
   through the frozen bidirectional Llama layers) — the honest zero-LoRA test: the
   frozen trunk must transform conditioning into different parse behavior. Head-level
   injection is the cheap comparison arm (if it matches trunk-level, the trunk adds no
   conditioning value — a finding to discover, not assume). Cost accepted: the banked
   precompute doesn't apply to prefixed passes; live trunk forwards (0.34s/batch).
2. **The zero-LoRA boundary:** trainable params in the EMISSION head and in ONE SHARED
   conditioning encoder are sanctioned (they are the interface); anything that varies
   PER CYCLE is forbidden. Cycles differ only in what the notebook contains, never in
   which parameters run.
3. **Conditioning corpus = the plateaued parser's ORGANIC failures** on train puzzles
   (real error distribution), supervised against gold fixes; synthetic corruptions are
   augmentation only. **Refinement (arm separation):** Brick-A conditions on ORACLE
   localization (gold-derived wrong-sentence flags) — it tests "can the frozen trunk
   USE a correct NACK," separated from "can the system GENERATE one gold-free" (Brick-C,
   where the add-back sweep + tier-0 replace the oracle). If Brick-A ran on the weak
   deployable localization (blame recall 0.067) and failed, conditioning-vs-localization
   would be unattributable.
4. **The measurement (field, pre-specified):** same puzzle, same weights, blank vs
   NACK-conditioned input THROUGH THE SAME GRAPH; read = does the parse delta
   (emission-change mass + attention-delta mass) CONCENTRATE on flagged slots/sentences
   vs a global reshuffle. Plus: flagged-slot FIX RATE vs a SHUFFLED-FLAGS control
   (same flag count, innocent sentences). **KILL: fix_rate(true) ~= fix_rate(shuffled)
   => conditioning isn't doing the work => the null fails toward the LoRA ladder.**
- **Brick-A SUBSTRATE BLOCKAGE (2026-07-06 night, recorded honestly):** every
  backward-through-trunk training form hangs the AM driver — fused (real), eager
  multi-loss (real), three-JIT checkpointed VJP (real, on a RECOVERED device; jit1's
  input-grad extension through the cross-attention is a NEW graph class, not the
  hours-proven head step). Pre-wedge facts: trunk-bwd alone OK, JIT'd head-step OK.
  Meta-lesson (quirks): HANGS ACCUMULATE — recover (scripts/am_gpu_recover.py,
  self-service via caps) before trusting any bisect result. DECISION: run the
  HEAD-LEVEL arm first (registered §9.1 comparison arm; all-proven graph classes;
  banked states apply) — it measures fix-rate/concentration/kill TODAY. Trunk-level
  remains the registered honest null, PARKED behind the driver fight (smaller B/T,
  tinygrad update, or upstream fix are the candidate unlocks).
- **BRICK-A HEAD-LEVEL ARM RESULT (2026-07-06): NULL FAILS at this arm — honest
  negative with TWO NAMED CONFOUNDS.** fix(true)=0.102 vs fix(shuffled)=0.083
  (barely separated), concentration 0.47/0.54 (BOTH < 1: parse delta lands AWAY
  from flagged regions), preservation ~0.997 both, solves 7/8/6 (noise). Per the
  registered kill: this arm's conditioning is not doing the work. CONFOUNDS:
  (i) HEAD-LEVEL CANNOT CHANGE THE READING — trunk states are fixed; if the error
  lives in the states (not the aggregation), only trunk-level conditioning can fix
  it. The head-level failure is CONSISTENT WITH the trunk-level thesis, not a
  refutation of it — and notably falsifies the "head-level trivially works" prior.
  (ii) THE TRAINING OBJECTIVE MADE FLAGS REDUNDANT: v0 supervised ALL slots toward
  gold regardless of flags, so the model could improve flagged slots from gold alone
  — conditioning was never made LOAD-BEARING. A proper design forces flag-dependence
  (e.g., reproduce-previous-output when unflagged, emit-the-fix when flagged).
  ALSO: plateau-residual errors (survivors of 68k supervised steps) may be capacity
  errors no same-model re-parse can fix — the NACK fix-rate ceiling for an UNCHANGED
  model is itself an open measurement. NEXT: (a) flag-dependent training objective,
  (b) trunk-level arm when the driver fight is won, (c) if both fail: the §8.4 LoRA
  ladder, as specified.
- **TEXT-RENDERED NACK (registered 2026-07-06 — the driver-fight dissolver):** the
  trunk is FROZEN, so backward-through-trunk is only needed if conditioning enters as
  LEARNED prefix params. Render the NACK as literal text instead ("NOTE: statement 7
  may be wrong.") prepended to the problem: ordinary forward, head trains exactly as
  today (backward stops at the head input — the 68k-step-proven graph class), zero new
  gradient paths, zero driver exposure. This IS trunk-level conditioning (the flags
  change how the text is READ — what confound #1 demands) and arguably the PUREST
  plane-ride null: same weights, different INPUT, different behavior — input in the
  most literal sense. Cost: conditioned trunk states need live forwards / a small
  conditioned precompute (~6 min for the Brick-A corpus at 0.34s/batch). Token budget:
  note ~15-40 tokens, max 451+40 < 512, the truncation guard stands watch.
  COUNTER-PRIOR (register it): C1-A found TEACHING (auxiliary prediction) beat TELLING
  (hint concatenation), and text-rendered NACK is telling — but runtime verifier flags
  MUST enter as input somehow (no training-time auxiliary exists for an error that
  doesn't exist until inference), so C1-A predicts a MODEST effect size, not a wrong
  channel. Queue: flag-dependent head arm (running) -> text-rendered trunk arm ->
  LoRA ladder only if both fail. Plus the CEILING PROBE as its own measurement:
  deliberately overfit an unrestricted head on flagged positions from frozen states —
  slots where gold is NOT decodable are provably beyond ANY head-level conditioning;
  that fraction converts the capacity-error caveat into a DENOMINATOR (fix rates then
  read as fraction-of-the-fixable).
- **BRICK-A FLAG-DEPENDENT ARM RESULT (2026-07-06): FLAG-USE PROVEN; localization
  real at the DECISION level; my registered field metric was MIS-SPECIFIED.**
  Numbers: fix(true)=0.438 (152/347) vs fix(shuffled)=0.360 — the flag-dependent
  objective lifted fix capability 4.3x over v0 (0.102->0.438), and the true-vs-
  shuffled gap (~2.9 sigma) is what LOCALIZATION buys; the 0.360 shuffled floor is
  what the shared GLOBAL-FAIL bit buys (generic revision energy). Preservation 0.998
  both. SOLVES: blank 0 -> true-NACK 7 (12.3% of failures solved outright by ONE
  oracle-NACK round) vs shuffled 4. Blank solves dropped to 0 BY DESIGN (blank-mix
  trains copy-previous, errors included — fix energy activates only with the bit).
  THE METRIC LESSON: registered concentration (L1 prob-mass) reads 0.52<1 — but it is
  dominated by sub-threshold jitter; at the DECISION level, flagged slots flip at
  43.8% vs unflagged at ~0.2% (~200:1). The prob-mass field was the wrong functional
  form; reported as registered (fails the letter), decision-level re-read passes the
  spirit. Named, not swapped silently. C1-A's modest-effect prediction for "telling"
  holds: real, significant, not crushing.
  VERDICT: head-level conditioning DOES work when made load-bearing — confound #2 was
  the v0 story. The remaining gap to 1.0 awaits the CEILING PROBE (denominator) and
  the TEXT-RENDERED trunk arm (reading-repair on top of flag-use).
- **POST-ARM REGISTRATIONS (2026-07-06, before the next dataset):**
  (1) LOCALIZATION METRIC REPLACED, pre-registered: flagged-vs-unflagged FLIP-RATE
  RATIO at the decision level (the L1 prob-mass form aggregated sub-threshold jitter;
  promoted to CLAUDE.md §6 as the third sighting of the instrument lesson — the
  registered functional form must match the mechanism's decision structure).
  (2) CEILING-PROBE CAUTION: measure decodability from frozen TRUNK states
  (head-independent — a fresh unrestricted probe head, deliberately overfit), NOT
  from the current head's representations, which inherit the objective they were
  trained under. The probe denominates the INJECTION POINT, not the head.
  (3) BRICK-C'S BUDGET, handed by the decomposition: the shuffled floor (0.360) is
  what ANY global "something's wrong" signal buys; add-back blame quality only has
  to defend the +7.8pt LOCALIZATION increment. Imperfect blame degrades the
  increment, not the floor — the loop is robust to sloppy localization.
  (4) THE MARGINAL-SOLVES FRAMING: blank 0 -> true 7 is the loop's contribution OVER
  DOING NOTHING — seven puzzles unreachable by unconditioned same-weights re-parsing,
  solved in one SYN->NACK->retransmit round (oracle standing in for Brick-C). The
  core Alternator transaction has now executed end-to-end.
- **CEILING PROBE v1 INVALIDATED BY ITS OWN BASELINE (2026-07-07):** fresh-init 6k
  steps read decodable=0.095 — BELOW the head arm's constructive 0.438. A ceiling
  cannot sit under a measured achievement: v1 measured OPTIMIZATION BUDGET, not
  decodability (loss 4.3 vs plateau 3.4; the original head needed 20k+ steps + the
  full corpus). Standing facts: decodability of test flagged slots >= 0.438
  (constructive, from the head arm). v2 = partial warm-start from the plateaued
  gold-only head (pre-Brick-A, objective-independent) + fresh wide FFN + 12k steps.
  Lesson filed: an overfit-probe must DEMONSTRATE convergence before its residual
  can be called a ceiling.
- **TEXT-NACK ARM RESULT (2026-07-07): FAILS — content-blind, diagnostically so.**
  fix(true) = fix(shuffled) = 0.295 EXACTLY (142/482 both): the note's CONTENT was
  never read — only its presence registered (generic revision). Flip-rate 0.921 vs
  0.666 (~1.4:1 — global thrash, vs the head arm's 200:1); preserve degraded 0.998
  -> 0.92 (the note shifts every token's position + steals attention); solves 1/0/0
  (WORSE than blank); training loss never settled (3.7-4.0 vs 3.3). The drop-bias
  caveat does not rescue it: identical true/shuffled is content-blindness,
  independent of training distribution. DIAGNOSIS: referential binding ("statement
  7" -> the 7th sentence) is deep-layer work; a FROZEN 4-LAYER prefix of a 1B LLM
  cannot compute it. The C1-A counter-prior ("telling is modest") was right and
  then some: telling through a shallow frozen reader is ~zero. The placebo arm is
  MOOT (content-blindness established at true-vs-shuffled; skipped, reason recorded).
- **CEILING PROBE v2 (2026-07-07): CONVERGED, VALID — decodable = 0.533.** Warm-
  started, 12k steps, loss 3.47 ~= plateau (converged); ceiling ABOVE the
  constructive 0.438 (consistent, unlike v1). **46.7% of plateau errors are beyond
  ANY head-level channel; the head arm's 0.438 = 82% OF THE FIXABLE.**
- **BRICK-A VERDICT (the plane-ride hypothesis, measured):** the zero-LoRA null
  PASSES — same frozen weights, different INPUT, appropriately different behavior —
  but the interface matters decisively: conditioning works as STRUCTURED FEATURES
  (position-aligned embeddings; 82% of fixable, 200:1 localization) and fails as
  NATURAL LANGUAGE through the shallow frozen trunk. This VINDICATES the original
  §8.2 interface design (notebook/NACK as feature channels, registry vocabulary) over
  the text-rendering shortcut. The remaining 46.7% needs actual reading-repair —
  candidates: deeper trunk prefix, the LoRA ladder, or the Alternator's real answer
  (structured ledger re-parse). Brick-C operates within the 53.3%.
- **POST-BRICK-A REGISTRATIONS (2026-07-07):** (1) §6 law added: positional/referential
  structure enters AS STRUCTURE, not prose (2 sightings: sentence-counting, reference-
  binding). (2) FRONTIER RANKING for the 46.7%: LEDGER-CONDITIONED RE-PARSE FIRST —
  it changes the PROBLEM (parse against pinned verified context), not the injection
  point, and may move the 0.533 denominator itself; zero-LoRA-compatible; free once
  the loop exists. Deeper prefix second; LoRA ladder stays behind measured
  alternatives. (3) BRICK-C KILL CRITERION, pre-registered RELATIVE: add-back blame
  must RETAIN >= HALF the oracle's localization increment (the +7.8pt over the 0.36
  floor) — the oracle arm is the ceiling condition; Brick-C's question is how much
  survives the real instrument. (4) JOB-B GATE IS DUE (pegged to Brick-A completion).
- **UNKNOWNS-CORPUS DECISION (registered 2026-07-07, product track):** GENERATED
  ALGEBRA before any MATH-500 graph-ification — linear systems + small nonlinear
  compositions, NL via template families. The KenKen generator's deepest property
  transplants verbatim: the domain makes gold + equivalence checking FREE
  (unique-solution-by-construction, exact symbolic solve, round-trip gating), and
  decisions-per-problem becomes a GENERATOR DIAL (target the engine band directly).
  MATH-500 stays the EVALUATION target, never the training substrate. SEQUENCING:
  the registry extension (equals/linear relations) is the first NEURAL-side test of
  "new domain = predicate + bridge, never core" — the delta head's typed-slot +
  digit-spaced format meets its extensibility bill EARLY, during corpus work, not
  after Brick-C hardens KenKen-only assumptions.
- **BRICK-C MEASURED (2026-07-07): THE LOOP CLOSES GOLD-FREE.** v0 KILLED at its
  pre-registered bar (retention 0.48 < 0.5) — the kill forced the instrument reading
  that surfaced a bug-vs-intent (malformed cages commented as blamed, never blamed).
  v0.1 (the labeled fix): fix oracle 0.438 | ADD-BACK 0.401 | shuffled 0.360 ->
  **retention 0.52, LIVES** — with the margin stated honestly: 0.48/0.52 straddle
  the bar, so the true retention sits AT ~0.5 within noise. The robust facts: the
  gold-free instrument (confidence-ordered add-back + attention slot->sentence map)
  recovers ~half the oracle's localization increment, and SOLVES — the end-to-end
  metric — ran 8 vs oracle 7 vs shuffled 4 in BOTH versions: full solve parity with
  the ceiling condition. THE ALTERNATOR LOOP HAS NOW RUN GOLD-FREE END TO END:
  parse -> symbolic self-diagnosis -> blame -> flags -> conditioned retransmit ->
  solve, no gold at any stage, 8/57 plateau failures recovered in one round.
  Margin upgrades (named, unbuilt): uniqueness-probe flags in the sweep, a real
  tier-0 confidence head for ordering, multi-round retransmission.

## 10. The math expansion (begun 2026-07-07)

- **REGISTRY EXTENSION, SYMBOLIC HALF: DONE.** `arith3` — op(a,b)=r with unknowns on
  BOTH sides, L-ASYM ordered scope (a,b,r), integer-exact (div = exact-divisibility),
  constants as given variables (singleton domains, never factors). Predicate +
  pairwise-support propagator + bridge, ZERO csp_core edits — the 7th domain through
  the same seam. Smoke (hand systems, all asserted): triangular -> 0 decisions
  (calculator band); COUPLED (x+y=5, x-y=1) -> 1 decision (**engine band by
  construction** — AC keeps locally-supported values only joint reasoning kills);
  4-pair chain -> 4 decisions (**the generator DIAL measured: coupling count = engine-
  band depth**); uniqueness checkable by ban-and-resolve (gold + equivalence stay
  FREE — the KenKen property transplanted).
- **NEURAL-FORMAT FINDING (by analysis, ahead of any build — the answer the brick
  existed to surface):** the delta head's emission extends to arith3 EXCEPT the
  membership pointer: multi-hot over positions is ROLE-BLIND, and op(a,b)=r is
  order-sensitive for sub/div — the (a,b,r) roles cannot be recovered from a set.
  SMALLEST SURGERY, two options to decide at build time: (i) role-typed pointers
  (args-multihot + a separate RESULT pointer head — canonicalizing sub/div into
  add/mul form makes the two args genuinely unordered, so 2 heads suffice), or
  (ii) three categorical pointers (a, b, r). Everything else transplants: type menu
  extends by one entry, op vocab UNCHANGED (add/sub/mul/div already there), targets
  unused (constants are variables now — the digit heads idle or emit given-values).
  The typed-slot format's extensibility bill: ONE new pointer head, not surgery.
- **RELATED WORK (verified 2026-07-07, published 07-06): Anthropic's "Verbalizable
  Representations Form a Global Workspace in Language Models"** (transformer-circuits
  + open-source J-lens). VERIFIED claims: an emergent small workspace ("a few dozen
  concepts, <1/10 of activity") that is reportable/controllable/load-bearing for
  deliberate reasoning; ablation kills multi-step reasoning while fluency/recall
  survive; unspoken intermediate concepts causally mediate answers (spider->8, swap
  ant->6); workspace monitoring surfaces hidden error/deception signals pre-output.
  CONVERGENT MOTIFS with Mycelium (analogies, NOT evidence — the over-claim guard
  applies): small-broadcast-channel over parallel substrate (waist/notebook/global
  latents, built explicitly); reasoning-in-residual-never-tokens (§6 law, observed
  in the wild); internal-state-more-honest-than-output (their safety program == our
  tier-3 late-JSD 0.687). The double edge, registered: emergence-at-scale cuts
  against NEEDING to build the workspace explicitly; our counter is the thesis
  itself — at 32M-87M params every joint is separately measurable. Closest
  neighboring result for the eventual paper's related work.
- **BORROWED TECHNIQUE CANDIDATE — J-ORDERED MATRYOSHKA:** order waist dims by
  JACOBIAN SENSITIVITY of downstream decisions (|d output / d dim|), not variance.
  We already hold the measured motivation: variance ordering inherits the identity
  confound (the 0.755-vs-0.658 trap). Both waists are differentiable on existing
  ckpts — one backward per output per batch. The experiment: J-ordered prefix-width
  curve vs the current ordering; if the 128d survival sharpens further (or the
  cliff moves), sensitivity ordering wins the schedule.
- **J-ORDERED MATRYOSHKA, DESIGN REFINEMENTS (registered before build):**
  (1) THE DISCRIMINATING REGION IS BELOW 128 — the measured curve is FLAT 128~=512,
  so both orderings read "fine" there; sweep 8/16/32/64/128/256/512 and REGISTER THE
  PREDICTION AS "J-ordering moves the CLIFF LEFT" (survival-at-128 is unfalsifiable
  on a saturated curve). (2) ESTIMATOR = DIAGONAL FISHER: mean of SQUARED per-instance
  gradients (signed averaging cancels opposing sensitivities and ranks live dims
  dead); per-sample backwards to avoid within-batch cancellation. (3) TWO TARGETS:
  decision-side (solve-relevant logit margins — the honest one; wrong-but-equivalent
  already proved loss and decision diverge) + loss-side as the cheap comparison arm
  (identical rankings would itself be a finding). (4) SCOPE OF A NULL: greedy
  diagonal ranking is not the optimal SUBSET per width (dims can be individually
  weak, jointly load-bearing — the codebook's pairwise geometry is exactly what a
  diagonal misses); a null reads "diagonal sensitivity doesn't beat variance," NOT
  "sensitivity ordering wrong in principle." (5) THE INCUMBENT IS NOT A STRAWMAN:
  Matryoshka training kept dims 0-127 always on — identity ordering carries the
  baked-in trained importance; Fisher matching it CONFIRMS the nested training,
  beating it below 128 is the win.
- **J-ORDERED MATRYOSHKA RESULT (2026-07-07): PREDICTION REFUTED — the INCUMBENT
  WINS, and the refutation is structural.** Cliffs: identity (trained prefix) 128;
  variance/fisher_loss/fisher_decision all 256 — every post-hoc reordering moved the
  cliff RIGHT. The registration's item 5 named the mechanism in advance: nested
  training didn't just RANK dims, it built CO-ADAPTED FUNCTIONAL SUBSETS (dims 0-127
  were trained to work AS A SET; a mask of the 128 most-sensitive dims scattered
  across the index space was never a training configuration, and the head cannot
  read it — despite 85-91% top-128 overlap with the incumbent, the 10-15%
  disagreement costs the whole cliff). SUB-PREDICTION CONFIRMED: decision-Fisher
  beats loss-Fisher at every width below 256 (0.120/0.139/0.255/0.627 vs
  0.018/0.001/0.009/0.496) — the wrong-but-equivalent divergence is real in the
  Jacobian too. BELOW THE TRAINED FLOOR (<128, never sampled in training): the
  trained prefix DIES COMPLETELY (0.001-0.011) while variance/decision-Fisher
  degrade gracefully (0.306/0.255 at 64) — sensitivity orderings find dims with
  standalone signal; none is solve-capable. §8.6 CONSEQUENCE: the waist schedule's
  aggression limit is a TRAINING-TIME choice, not a post-hoc reordering — to
  survive at 64/32, SAMPLE those widths during nested training (optionally SEEDING
  the dim order by decision-Fisher before training: candidate, unregistered).
  NULL SCOPE honored: this reads "post-hoc diagonal reordering doesn't beat a
  trained incumbent," not "sensitivity ordering wrong in principle."
- **POST-REFUTATION REGISTRATIONS (2026-07-07):** (1) composition-fragility promoted
  to §6 (third sighting). (2) FISHER-SEEDING registered WITH mechanism: seeding cannot
  help through ordering per se (training co-adapts whatever the mask exposes); its
  value proposition is BETTER RAW MATERIAL (dims with standalone signal — the thing
  finding #3 proved sensitivity tracks) for narrow sets to co-adapt from. Clean test:
  seeded vs unseeded nested training WITH width-64 in the schedule. HONEST MODEST
  PREDICTION: if co-adaptation dominates raw material they converge and the seed buys
  nothing — completing the story as "the schedule is everything, the ordering is
  nothing." (3) DESIGN RULE, forward-binding: everything that READS the silhouette
  (perceiver monitor, Brick-0 probes, tier-3 instruments) is built against the
  DECISION-relevant subspace, not the loss-relevant one — an order of magnitude at
  narrow widths is not a nuance. (4) Related-work line beyond the citation: the
  nested-trained incumbent is the ADVERSARIAL CASE for post-hoc dim selection —
  "co-adaptation defeats post-hoc sensitivity ranking" is a boundary condition of
  the J-lens found within 24h of its publication.
- **CORPUS REGISTRATIONS (2026-07-07, per-band + mentions):** (1) EVAL IS PER-BAND
  from the first run: solve-rate and factor-exactness logged PER decisions-band
  (a smeared solve-rate would erase the one distinction Job-B established matters);
  curriculum-by-band available (calculator band first: parse errors are the only
  failure mode there). (2) THE FACTORIZATION QUESTION, answered free by that logging:
  is parse difficulty CORRELATED with solve difficulty? A flat parser error rate
  across bands = reading hardness and reasoning hardness on independent axes — the
  cleanest possible vindication of the two-phase design. (3) VARIABLE-MENTION
  ANNOTATIONS added to gold (generator emits every name occurrence as char spans,
  free): the registered pre-emption of referential binding — the text-NACK arm
  proved shallow layers don't bind references unaided; the result pointer gets
  name->slot binding AS STRUCTURE (§6 law applied prospectively), not as a hop to
  learn. Cheap now; expensive after a plateau gets misdiagnosed as capacity.

## 11. The algebra chapter's interpretive frame (registered 2026-07-07, BEFORE the head trains)

- **THE DETECTABILITY INVERSION (on schedule, not a regression):** KenKen's seven-
  point 100%-detectable streak was a gift of constraint DENSITY. Linear systems are
  satisfiable for almost any constants: a mis-parsed literal yields a SAT, UNIQUE,
  cleanly-solving-but-WRONG system — UNSAT doesn't fire, the uniqueness probe doesn't
  fire. Surviving symbolic channels are STRUCTURAL: dangling variables /
  underdetermination (multi-solution probe) and overdetermined-inconsistency (UNSAT).
  **Do NOT grade the first algebra loop against KenKen's 100%.**
- **REGISTERED PREDICTION #2-ALGEBRA (the KenKen limbo resolves here):** the silent
  class appears at SUBSTANTIAL rate, CONCENTRATED IN LITERAL-CONSTANT ERRORS, while
  membership/structural errors stay detectable. REFINEMENT from the corpus's own
  integer structure: coupled-pair constants are ~HALF-caught by integrality/parity
  (x+y=11, x-y=4 -> x=7.5 -> UNSAT over Z — a wrong constant flips parity ~50% of
  the time), while CHAIN constants go silent by default (a shifted k just shifts
  downstream values; SAT, unique, wrong). So predicted silent-rate ordering:
  chain-literals >> coupled-literals > structural (~0). The taxonomy tool runs
  unchanged; the corpus's gold answers grade it.
- **THE NACK STACK'S COMPOSITION INVERTS:** tiers 1-2 (verifier, uniqueness) carried
  KenKen; algebra hands the load to TIER-0 (the confidence head — now with its
  motivating error class: low-confidence literal emissions) and TIER-3 (the deducer's
  soft-solve suspicion field — specced for exactly this soft-wrongness regime).
  Neither is built. The algebra chapter is not "new domain, same loop" — it is the
  domain that FORCES the two neural NACK tiers the design promised.
- **HEAD FIELD LAYOUT (settled before build):** slots emit
  presence | type {rel, given} | op {add, mul} (canonical; sub/div die at the
  generator) | args = 2-hot over VARIABLE slots | RESULT = UNION TYPE: an is-literal
  MODE BIT gating result-POINTER (categorical over variable slots) vs the digit
  machinery (transplants verbatim) — both sides supervised from gold; given-factors
  use var-pointer + value digits. Variable slots anchor to the text through the
  MENTION spans (name->slot binding as structure, §6). PLUS the head nobody
  mentioned until now: the QUERY POINTER — one global supervised pointer over
  variable slots (gold free from the generator); without it the pipeline solves the
  system but cannot answer the question.

## 12. Tier-0 confidence (designed 2026-07-07, cross-channel; v0 = the incumbent, formalized)

- **THE FRAME (post-J-lens):** the entropy null is the INCUMBENT WITH A TRACK RECORD
  (slot_confidence — presence-sigmoid x per-field top-prob product, zero params —
  ordered the add-back sweep to 0.52 retention). A trained head's bar is not "does
  supervised confidence work" but "does it beat a working incumbent."
- **GRANULARITY:** per-field confidence is the primitive (a READ off the union
  layout's separate logits, not a build); factor-level is DERIVED. Open sub-question,
  a numpy afternoon: does per-field error correlate within-factor? Yes -> min() or a
  learned 4-weight combination beats the product; no -> the product stands.
- **CALIBRATION vs DISCRIMINATION, split by consumer:** the add-back sweep consumes
  RANKING (metric: AUC — implicit in the 0.52, never read directly; MEASURE IT
  FIRST); phantom-gating and literal-flagging consume THRESHOLDS (metric: ECE, with
  the incumbent TEMPERATURE-SCALED first — one scalar fitted on train failures).
  Registered structure: incumbent -> per-field AUC + post-temperature ECE on banked
  artifacts -> a trained head must beat BOTH on the field where it claims value.
- **THE WITHHOLDING-COST CURVE (pre-registered):** decode each banked KenKen test
  failure; withhold the k least-confident factors; solve the remainder; sweep
  k in 0..5. COLUMNS: (a) solve-to-GOLD rate — Code's addition: withholding a wrong
  factor from a dense graph can recover the EXACT grid with NO retransmission, so
  the curve doubles as the measurement of withhold-and-solve as a THIRD repair
  channel (vs retransmit, vs oracle); (b) taxonomy composition of the withheld
  graphs — UNSAT -> multi-solution conversions are DETECTION conversions, not just
  solve effects. REGISTERED PREDICTION (conditional — Code's pushback): the curve is
  non-monotone with a peak at k=1-2 on KenKen, CONDITIONAL ON incumbent per-factor
  AUC >= ~0.7 — at AUC ~0.65, k=1 withholds a CORRECT factor ~35-40% of the time and
  converts right-graphs to underdetermined, and the peak may never materialize. AUC
  lands first; the curve is interpreted against it, not alone. ALGEBRA FLIP
  (registered): sparse coupled systems starve immediately (withholding a given ->
  underdetermined by construction) — peak at k=0-1 or absent. DENSITY DECIDES THE
  GATING BUDGET: the cross-domain claim.
- **THE NULL'S BLIND SPOT (the one principled edge for a trained head):** entropy
  measures decision SHARPNESS; confidently-wrong is structurally invisible to it —
  and plateau errors are exactly the sure-and-wrong class (the 6-7 codebook
  confusions are the deducer-side picture). Registered alternative form: does
  supervised correctness-prediction find errors ON THE HIGH-INCUMBENT-CONFIDENCE
  SUBSET? Zero edge there -> the null survives everywhere it matters.
- **PRE-REGISTERED KILL for any trained tier-0 head:** must beat the calibrated
  incumbent's AUC on the high-confidence-error subset, or per-field ECE, by margins
  set AFTER the incumbent's numbers land. Zero GPU for all of v0 — measurement, not
  training.
- **FIRST ALGEBRA RUN (2026-07-07 evening): 58/60 ANSWER, and the FACTORIZATION
  READ IS CLEAN.** Per-band: fac-exact 1.000/1.000/0.997/0.993 across decisions
  bands 0/1/2/3 — FLAT. Parse difficulty and solve difficulty are on INDEPENDENT
  AXES on first measured contact: the parser reads band-3 (4-decision) systems as
  easily as calculator-band ones, while the solve column does genuinely different
  work per band. The two-phase design's division of labor, vindicated by its first
  stratified table. ANSWER 58/60 = the full pipeline (parse -> graph -> GENUINE
  SEARCH (1-3 decisions) -> query-pointer -> answer) at near-ceiling. Noted oddity:
  band-2 ANSWER (29) > graph-solve (28) — a wrong-somewhere graph can still be right
  AT THE QUERY: wrong-but-right-where-asked, the equivalence class's algebra cousin.
  HONEST DEFLATOR: v0 templates are easy (105 tokens mean, low paraphrase diversity,
  mentions given as structure — the §6 laws compounding prospectively designed OUT
  the failure modes). Near-ceiling here means THE CORPUS NEEDS TEETH, not that
  parsing is solved: crank paraphrase depth + template families until failures exist,
  THEN prediction #2-algebra becomes testable (2 failures is an anecdote, not a
  taxonomy). The chapter's real tests still ahead; its plumbing is proven.
- **THE CONVERGENCE EVAL (2026-07-07 night, teeth 0.8, n=300): PREDICTION #2-ALGEBRA
  RESOLVES BOTH WAYS.** The teeth bit: fac-exact 0.69-0.81 (v0 was ~1.0), ANSWER
  121/300, a real failure population (179). **THE SILENT CLASS IS BORN — the
  inversion arrived as registered:** 14 SILENT (KenKen: zero across seven points);
  detectable fraction 0.92. **BUT THE ORDERING IS REFUTED:** silent literals are
  roughly uniform across roles (pair_diff 4 / pair_sum 2 / chain_k 2), and wrong
  literals overwhelmingly land DETECTABLE (unsat 195, multi 89 attributions) — far
  more caught than predicted. THE MECHANISM OF THE REFUTATION IS THE FAMILIAR ONE:
  the prediction was calibrated to the SINGLE-ERROR regime; at fac-exact ~0.73 parses
  carry ~5 wrong factors, and multiple wrong literals + structure JOINTLY
  over-constrain -> UNSAT catches what parity alone would have missed. (Delete-one
  blame died the same death — "single-error-regime assumptions break at multi-error
  density" is now a RECURRING prediction-failure mode; second sighting, watch for
  the third.) DETECT_multi=43: the uniqueness probe earns its keep at scale.
  ANSWER(121) > graph-solve(95): wrong-but-right-where-asked is now a LARGE class
  (26), not a curiosity. FACTORIZATION UNDER LOAD, honest read: fac-exact no longer
  flat (0.811 -> 0.685 by band 3) — but band correlates with SIZE (more pairs = more
  factors/vars/obliques), so the axes may correlate through LENGTH, not through
  solve-difficulty; the size-controlled read (fac-exact vs band at fixed n_vars) is
  the open analysis before the v0 flat claim is downgraded. TRAINING NOT CONVERGED
  (loss 5.3, falling): all numbers are floors. NET FOR THE LOOP: 92% detectable on
  algebra's thinner jaws — the NACK story survives the inversion far better than
  feared, and tier-0/tier-3's target class (the 14 silents + 43 multis) now exists.
- **TIER-0 MEASUREMENT 1 (2026-07-07 nightcap): THE INCUMBENT HALF-SEES THE
  SILENTS, AND ITS BLIND SPOT IS FIELD-LOCALIZED.** Silents-vs-correct AUC (the
  registered separation number, n=14 caveat): product 0.727, min 0.680, digits
  0.654, fields 0.601 — but **pointer 0.479 and query 0.398: AT-OR-BELOW CHANCE.
  Silent parses are AS-or-MORE confident in their pointers/query than correct
  parses are** — the entropy null's predicted blind spot (confidently-wrong),
  found alive and localized to exactly two fields. Meanwhile all-wrong-vs-correct
  product AUC = **0.905**: the incumbent is an EXCELLENT ranker of detectable
  errors (Brick-C's consumer is well served). Saturation note: median confidences
  0.97-1.00 everywhere — thresholds are hopeless without recalibration; the AUCs
  are rank-based and unaffected. VERDICT (per the §12 frame): the null SURVIVES as
  the ordering signal and DIES as the silent-detector; the trained head's job is
  now precisely scoped — supervised correctness-prediction on the POINTER and
  QUERY fields, where entropy carries zero signal and input-conditional error
  patterns are the only hope. Neither clean ending; the design's completion now
  has an address.
- **FIXED-n_vars FACTORIZATION READ (2026-07-07 morning): THE AXES-INDEPENDENCE
  CLAIM SURVIVES UNDER LOAD.** Raw corr(fac-exact, band) = -0.126 was ENTIRELY
  size-mediated (corr(band, n_vars)=+0.678, corr(fac-exact, n_vars)=-0.249):
  **PARTIAL corr(fac-exact, band | n_vars) = +0.061 ~= ZERO**, and within fixed-size
  buckets fac-exact is flat-or-RISING with band (12-15 vars: 0.665/0.676/0.732).
  The parser feels TEXT LENGTH, not solve difficulty — reading and reasoning on
  independent axes, now confirmed at teeth 0.8 with the confound removed. The v0
  flat claim is UPGRADED, not asterisked.
- **WITHHOLDING-COST CURVE (2026-07-07 morning, banked KenKen, n=57): THE THIRD
  REPAIR CHANNEL IS REAL — AND IT BEATS RETRANSMISSION.** Precondition first, per
  registration: incumbent per-factor AUC = **0.613 < 0.7** — the k=1-2 peak claim is
  formally VOIDED (not refuted). The curve: solve-to-GOLD 2/8/12/**15**/11/10 for
  k=0..5 — **non-monotone with the peak DISPLACED to k=3**, consistent with the
  precondition's own logic (imprecise ranking x ~5 errors/parse -> withhold more to
  cover the wrong set; peak position ~ errors-per-parse x ranking quality).
  QUALITATIVE SHAPE CONFIRMED, quantitative form shifted exactly as the conditional
  anticipated. THE HEADLINE: **15/57 (26%) of plateau failures solve to the EXACT
  gold grid by deleting the 3 least-confident factors — no NACK round, no
  retransmission — vs 8/57 for one-round retransmit.** Withhold-and-solve is the
  loop's zero-cost FIRST move on dense domains. SAFETY: wrong-grid = 0 at EVERY k —
  withholding never produced a silent-wrong solve; every non-gold outcome stayed
  detectable (UNSAT -> multi drift 1->23, the registered detection conversions).
  Dense domains keep the full safety net through withholding. (Algebra flip still
  registered: sparse systems should starve immediately — measure, don't assume.)
  COMPOSITION IMPLIED: withhold-and-solve -> if multi -> retransmit — the repair
  stack now has an ordering by cost.
- **THE BIG SLICE (2026-07-07 midday, n=1500 fresh, 78 silents): THREE RESOLUTIONS.**
  (1) **The inversion number is STABLE:** detectable 0.91 on the second draw (0.92
  first) — ~0.9 is algebra's detectability constant at this parser quality.
  (2) **Prediction #2-algebra's ordering: fully INVERTED at real n** — silent
  literals: pair_sum 16 / pair_diff 13 / **chain_k 2** — chains almost never silent
  (the registered chain>>coupled claim was exactly backwards). Mechanism: multi-error
  co-occurrence — chain-literal errors ride along with structural breaks (304 in
  UNSAT), while pair literals sometimes stand alone with parity's coin-flip.
  (3) **LAST NIGHT'S BLIND SPOT WAS A SMALL-n ARTIFACT:** at n=78, pointer AUC
  0.479 -> 0.660, query 0.398 -> 0.685 — the "at-or-below chance" fields carry
  normal signal; flag-don't-model vindicated again (14 positives lied).
  **THE HEADLINE: min-combination AUC = 0.812 — AT the registered completion
  threshold.** The §12 combination-rule sub-question is answered empirically: MIN
  beats PRODUCT on silents (0.812 vs 0.734 — within-factor errors correlate; the
  weakest field is the signal). Per the registered read (>~0.8 = the stack completes
  with zero new params): **tier-0 v0 = the min-combined, temperature-calibrated
  entropy incumbent. The trained head's kill criterion is now "beat 0.812 at n=78"
  — it is NOT BUILT unless a future domain reopens the gap.** Cautions: 0.812 at
  n=78 is at-the-bar, not past it (CI ~±0.05); thresholding still needs calibration
  (medians 0.987/0.962); all-wrong product AUC 0.915 (the ranker stands).
  THE ZERO-LORA PHILOSOPHY'S FOURTH WIN THIS WEEK.
- **POST-BIG-SLICE REGISTRATIONS (2026-07-07 midday):** (1) density-regime rule
  promoted to §6 (checklist form). (2) TRAINED-HEAD RE-ARM CONDITION: the head stays
  unbuilt at kill=0.812, RE-ARMED iff a future draw shows the pointer/query weakness
  DEEPENING on harder obliques (the teeth dial can force that question when it
  matters — they remain the weakest senses at real n: 0.66/0.69, weak-not-blind).
  (3) Withhold-and-solve NAMED: Law 7 (remove at READ, not from state) at the
  factor-graph level — parse keeps everything, solve drops suspects, the lattice
  fills in; it beats retransmission ON A 0.613-AUC RANKING because the LATTICE does
  the repair (neural proposes, symbolic disposes, again). (4) **THE COMPOSED REPAIR
  STACK, registered:** confidence-order -> withhold-and-solve (k at the measured
  peak) -> retransmit ONLY the survivors -> answer; one pass; end-to-end recovery on
  the banked failures is THE chapter-thesis number (components: withhold 15/57,
  retransmit 8/57, overlap unmeasured). ALGEBRA arm doubles as the registered
  SPARSE-DOMAIN FLIP test — standing prediction: the withhold peak collapses toward
  k=0-1 (coupled systems starve; every equation load-bearing). Note: algebra has NO
  trained retransmission head (Brick-A was KenKen) — its stack is withhold-only,
  stated honestly.
- **THE COMPOSED STACK, MEASURED (2026-07-07 midday):** KenKen: **24/57 = 42%
  end-to-end recovery** — withhold-3 stage-1 recovers 7 free, then retransmit(+
  withhold-again) recovers 17 of the 50 survivors. Composition beats both components
  (withhold-alone 15, retransmit-alone 8) and nearly TRIPLES retransmission. HONEST
  CAVEAT: stage-1's 7 vs the standalone 15 — different base parse (the stack runs
  the Brick-A ckpt's blank pass, trained to copy-previous; the standalone curve ran
  the plateaued head). 24/57 is the one-weight-set production number.
- **THE SPARSE-DOMAIN FLIP: REFUTED — withhold-and-solve GENERALIZES.** Algebra
  (n=913 failures, FORCED-answer check — no luck counted): recovery 0/60/**77**/75/
  70/56 for k=0..5 — peak at k=2-3, NOT collapsed to k=0-1. 8.4% of failures yield
  their answer FREE at k=2 on a SPARSE domain. The starvation intuition was
  single-error thinking AGAIN: at ~5 errors/parse, withholding preferentially hits
  WRONG factors (even a 0.613-AUC ranking beats chance) and the query needs only
  ITS component forced — poison-removal beats starvation until k grows. **FOURTH
  sighting of the density-regime trap, the same morning the rule was promoted.**
  UNSAT drains 659->127 with k as underdetermination grows — the registered drift.
  CONSEQUENCE: stage-1 of the repair stack is DOMAIN-GENERAL; the Alternator's
  cheapest move works on both dense and sparse graphs.
- **POST-STACK REGISTRATIONS (2026-07-07 afternoon):** (1) density rule enforcement =
  ARITHMETIC ("what does the k-th withheld factor actually hit at measured density x
  AUC") — §6 updated; 4th sighting happened with the rule in hand. (2) WITHHOLDING'S
  BOUNDARY PRE-DRAWN: it works where errors outnumber the ranking's mistakes AND the
  query's component keeps support — sparse domains DECOMPOSE (starvation is local;
  poison in other components costs nothing); the clause that bites is a query inside
  one large coupled component. (3) THE BLANK-PASS TAX: Brick-A's copy-previous
  training degraded unconditioned parsing (stack stage-1: 7 vs standalone 15) —
  fix = TWO-CHECKPOINT stack (plateaued head PARSES, conditioned head RETRANSMITS);
  the 42% has known headroom before the parser improves. (4) ALGEBRA RETRANSMIT
  TRANSPLANT = a training run (flag-dependent objective on algebra failures; flags
  via MENTION/FACTOR spans — char-exact, richer than KenKen's attention map); built
  two-checkpoint from day one. (5) MULTI-ROUND'S QUANTITATIVE FRAME, registered
  (density-regime stated: multi-error, survivor-selected): per-round recovery
  DECLINES; the asymptote is bounded by the decodable ceiling's share (0.533) of
  remaining errors. A round-2 recovery ABOVE that frame = retransmission moving
  decodability itself — the ledger-conditioning hypothesis arriving early, and the
  cheapest probe of the 46.7% frontier.
- **TWO-CHECKPOINT STACK (2026-07-07 afternoon): 27/57 = 47%.** The blank-pass tax
  eliminated exactly as registered: stage-1 recovers the full standalone 15 (the
  plateaued head parses; copy-previous training no longer degrades the first parse),
  stage-2 retransmits 42 survivors and recovers 12. Composed recovery 42% -> 47%
  with zero new training — the known headroom collected. Division of roles now
  explicit in the stack: the PARSER checkpoint parses, the RETRANSMITTER checkpoint
  repairs; one trunk, two heads, each at its trained operating point.
- **THE ALGEBRA TRANSPLANT (2026-07-07 evening): 222/920 = 24% composed recovery.**
  Stage-1 withhold-2: 77 (the known 8.4% floor); stage-2 retransmit(+withhold) on
  843 survivors: **145 recovered (17% of survivors)** — the repair specialist nearly
  TRIPLES the stack on its first run (77 -> 222). Trained pure (no blank mix; the
  two-checkpoint architecture carried blank quality), dual-granularity flags live
  from birth. Cross-domain: KenKen 47%, algebra 24% — consistent with the weaker
  parser (0.72 vs 0.83) and thinner constraint surface. HONESTY CAVEAT (recorded,
  not hidden): the eval's SPAN-level suspect flags located suspects via GOLD factor
  spans (slot order makes this ~"the j-th statement", mildly oracle-ish); the
  FIELD-level channel is position-free and clean. Fully-deployable variant = spans
  from the model's own attention (the KenKen pattern); the dual-granularity ablation
  doubles as the leakage bound — if field-only ~= both, the caveat is moot. QUEUE:
  parser convergence (teeth headroom) -> multi-round on honest numbers, per the
  sequencing registration.

## 13. The Nazaré funnel: the silhouette library + Brick-0 + the parse-side render
   (registered 2026-07-07 night, before build)

- **THE PHYSICS, STATED HONESTLY:** the canyon adds no energy — it REFRACTS a wide
  front into convergence. The waist cannot amplify signal absent from the trunk; it
  FOCUSES it (Matryoshka already proved the canyon exists: the parse signal survives
  at 128d). The focusing law is the J-lens lesson: concentrate the DECISION-relevant
  subspace — a canyon focuses noise too.
- **THE THREE GAPS (the ledger's unfired registered work):** Brick-0 has never run
  (assumption #2 — matched filters segment — is the LAST unmeasured plane-ride
  assumption); the centroid LIBRARY was never built as an artifact (the George-Hotz
  good-drivers note: nowhere on disk is there a bank of learned silhouette signatures
  to match against); the parse-side silhouette has capture hooks and has never been
  rendered (the BirdNet re-run, where the priors favor the bird).
- **BUILD (one script, banked data):** (a) PARSE-SIDE LIBRARY: per-factor-type token
  centroids in the waist space, built on TRAIN, evaluated on TEST as per-token
  SEGMENT-AND-CLASSIFY (cosine-to-centroid argmax vs gold span labels) at widths 512
  AND 128 (the canyon check: does the narrow waist classify as well?); plus the
  first token x type-similarity RENDER. (b) DEDUCE-SIDE ANALYTIC ARM of Brick-0:
  prototype matching on the banked 4-variant trajectory capture — constituent
  classification despite REFUTED linearity (the sharpened question: does matched
  filtering survive nonlinear composition?). (c) THE LIBRARY ARTIFACT saved to disk
  (.cache/silhouette_library_v0.npz) — centroids become a matchable bank, the
  registry's learned twin.
- **BARS:** parse-side per-token classification must beat the majority-class floor
  by a wide margin to claim segmentation (priors favor it: gold spans exist, text is
  banded); Brick-0's full form (learned latents vs this analytic library) follows —
  if the analytic library already recovers constituents, assumption #2 completes
  WITHOUT learned latents (zero-param win #5); if not, the learned arm gets its bar.
- **§13 RESULTS (2026-07-07 night): ASSUMPTION #2 COMPLETES IN EXISTENCE FORM —
  THE LAST PLANE-RIDE ASSUMPTION, MEASURED.** (a) PARSE-SIDE SEGMENT-AND-CLASSIFY:
  per-token kind accuracy **0.863** vs 0.779 majority floor (per-kind recall: rowcol
  1.00, none 0.93, cage 0.86, given 0.82 — all four kinds genuinely separated, no
  majority collapse); **the canyon holds: 0.843 at width 128** (2 points for 4x
  narrowing). The render is legible — kinds band exactly over their gold spans (the
  BirdNet picture, parse-side, as the priors predicted). (b) BRICK-0 ANALYTIC ARM:
  deduce-side 4-way variant classification **0.854 vs 0.25 chance** — matched
  filters SURVIVE nonlinear composition at the classification level (linearity
  refuted, classification intact: the amplitudes don't add but the DIRECTIONS
  discriminate). (c) THE LIBRARY EXISTS: .cache/silhouette_library_v0.npz —
  parse-kind centroids at both widths + deduce-variant prototypes, the registry's
  learned twin, matchable on disk. STATUS OF THE THREE ASSUMPTIONS (§8.7): #1
  conditioning suffices — TRUE via structured features (Brick-A); #3 the NACK —
  exists in measured tiers (taxonomy + 0.812 min-confidence + stacks); #2 matched
  filters segment — TRUE in analytic form (this, zero learned params: **zero-param
  win #5**). The learned-latent arm of Brick-0 is now an UPGRADE question with
  measured bars (0.863/0.843 parse, 0.854 deduce), not an existence question.
- **POST-§13 REGISTRATIONS (2026-07-08 early, the rolling session):** (1) THE
  DIRECTION PRINCIPLE, noted for promotion watch: two independent structures now say
  "the signal is a DIRECTION, not a magnitude" (codebook angular separation predicts
  confusions; silhouette directions discriminate where amplitudes don't add) —
  second sighting; a third promotes it to §6. (2) THE 0.863 TEETH-ROBUSTNESS CHECK,
  registered before hardening: parse-side kind classification partially rides the
  text's own lexical banding (that WAS the prior); the discriminating check is the
  same centroids on a HARDENED slice (max paraphrase/split-ref/distractors) — holds
  = the centroids encode STEPS; sags = partially WORDS. (KenKen's teeth are milder
  than algebra's — stated; the algebra-side library needs waist exposure and comes
  next.) (3) **THE LIBRARY AS RUNTIME CROSS-CHECK, registered:** the delta head and
  the centroid bank are two INDEPENDENT readers of the same waist; disagreement
  (head emits cage, silhouette matches given) is a gold-free inconsistency signal at
  PARSE time — upstream of withhold-and-solve in the cost ordering, a candidate
  NACK tier at zero marginal cost. MEASURABLE NOW: per-slot disagreement-vs-wrong
  AUC on banked failures; the interesting case is AUC near tier-0's while
  DECORRELATED from it — the combined ranker then attacks the stack's measured weak
  link (the 0.613 withholding order). (4) THE LEDGER LINE: five zero-param wins; the
  perceiver's mandate shrinks with each (monitoring ~= calib+JSD; segmentation ~=
  analytic 0.854/0.863); its remaining candidate job is hosting the global-broadcast
  latents in the multi-cycle loop. The design gets LIGHTER as it gets more measured.
- **ROLLING-SESSION RESULTS (2026-07-08 early):** (1) TEETH-ROBUSTNESS: the margin
  HOLDS — hardened slice 0.835 vs floor 0.754 (8.1 points; original 8.4). Within
  KenKen's teeth range, the centroids encode STEPS, not words; canyon holds hard
  (0.821 @128). (2) THE CROSS-CHECK TIER, honestly sized after an instrument audit:
  my first AUC(disagreement)=0.678 was a TIE-ORDER ARTIFACT (binary score, 96% ties,
  unstable argsort — midrank fix applied; the decision-structure metric law's
  cousin: DISCRETE SCORES NEED MIDRANK or the AUC is fiction). TRUE numbers:
  disagreement standalone 0.551 as a ranker BUT a rare high-precision flag
  (rate 3.2%, precision ~0.64 vs 0.176 base = 3.6x enrichment), decorrelated from
  tier-0 (+0.024), and the combined ranker = **0.634 vs the 0.613 baseline** —
  a real, modest upgrade to the stack's measured weak link at zero parameters.
  Verdict: the library-as-cross-check is a USEFUL RARE FLAG + a +2.1-point ordering
  gain, not a second tier-0. Both recorded; the artifact catch kept in the ledger
  (the measurement program audited itself twice in one night).
- **MORNING REGISTRATIONS (2026-07-08):** (1) §12 ADDENDUM — THE PORTFOLIO RULE:
  NACK signals are TWO TYPES with different combination laws. DENSE RANKERS (tier-0
  confidence, belief-JSD) blend; RARE-PRECISE FLAGS (uniqueness probe, library
  cross-check — ~3% fire rate, 3.6x enrichment, near-chance as rankers) VETO or
  ESCALATE. Future tiers get classified on arrival, not force-fit into the ranker
  mold. (2) The algebra-side teeth check (hardened big slice vs saved algebra
  centroids) is the lexical-shortcut SETTLER — queued for the next gap; requires
  algebra waist exposure. (3) **STACK-AT-CONVERGENCE, the protocol:** the hygienic
  redo (cosine LR decay, periodic val, PICK-BEST-BY-VAL on the small test slice —
  bigtest stays untouched as the measurement set) is the board's highest-leverage
  item AND re-arms half the registered predictions under NEW DENSITY conditions
  (fewer errors/failure -> delete-one blame may start working, the withhold peak
  slides left, the silent composition may rotate). All re-runs BATCHED as one table
  — same measurements, same protocols, ONE variable moved (the KenKen 10x discipline
  applied to the whole pipeline). The batched stack includes the 24%-ASTERISK
  resolution as an arm: stage-2 with FIELD-ONLY flags (position-free, fully
  deployable) vs span+field (gold-located) — the ablation and the deployable number
  in one read; the delta IS the leakage bound.
- **STACK-AT-CONVERGENCE (2026-07-08): THE TABLE.** Hygiene worked: clean val rise
  to 0.783 @36k, best-picked (the spike never happened). Bigtest: **ANSWER 802/1500
  (53.5%)** vs 587 pre-convergence (+215); graph-solve 680; fac-exact 0.71-0.87 per
  band. Taxonomy: detectable ROSE to **0.95** (silents 78 -> 34 — fewer and harder);
  silent attribution still mixed (5/2/1 — no chain dominance at low n). REGIME
  ROTATIONS, two: (1) **the combination rule is REGIME-DEPENDENT** — product now
  beats min on silents (0.793 vs 0.731; at the old regime min won 0.812 vs 0.734).
  Per-field structure keeps rotating across regimes (n=14 -> 78 -> 34): the ROBUST
  facts are portfolio-level (product/min 0.73-0.93), the field-level claims are
  regime-local. (2) **query-confidence became a SILENT-SPECIFIC flag: silents AUC
  0.927 while all-wrong AUC 0.442** — a textbook portfolio-rule rare flag (classify
  on arrival: FLAG, not ranker). Withhold curve at convergence: peak k=3 (86/703 =
  12.2% — MORE recovery than pre-convergence 8.4%); the peak did NOT slide left —
  the re-armed prediction is refuted-as-stated (surviving failures are the harder
  tail; density fell but selection hardened — the two effects cancel-ish).
- **THE COMPOSED-STACK COLLAPSE, DIAGNOSED (the good-news failure):** stage-2
  recovered only 13/627 — because NACK re-prep found **14/2000 train failures**:
  the converged parser nearly memorizes ITS OWN TRAIN SPLIT, so the repair
  specialist had no training data. The better the parser, the fewer organic
  training failures — **the repair trainer needs a failure-mining slice held out
  from BOTH the parser's training and the measurement set.** Fix: fresh generated
  slice (seed 21), prep there, retrain, re-run both arms. The field-only-vs-both
  ablation (13 vs 12) is DEGENERATE under the broken retransmitter — no conclusion
  drawn; re-read after the fix.
- **CONVERGENCE-TABLE READS (registered 2026-07-08):** (1) THE SELF-DEFEATING
  CURRICULUM LAW: a repair specialist's training data must come from the CURRENT
  model's failures on FRESH data, or it trains on ghosts — the parser memorizing its
  train split starved the retransmitter. Recurs at every capability jump; the
  failure-mining slice is the permanent plumbing (the old generator-and-specialist-
  co-evolve principle rediscovered by necessity). (2) THE EQUILIBRIUM CLAIM, stated
  as one: failures get rarer AND harder in compensating proportion as the parser
  improves, so the stack's operating regime (errors/failure, withhold peak k=3) is
  more stable than either trend predicts. If it holds at the NEXT convergence jump,
  the stack's constants are regime-invariant — no re-tuning per parser generation.
  (3) COMBINATION RULES ARE REGIME-LOCAL (product-beats-min rotated); only
  portfolio-level structure is robust — re-measure the blend at every regime shift,
  never inherit it. Query-confidence's three-regime arc (fiction -> weak -> 0.927
  rare flag) is the instrument ledger in miniature.
- **THE FINAL ROWS (2026-07-08): THE ASTERISK RESOLVES — FIELD-ONLY WINS.** The
  curriculum fix worked exactly as diagnosed: prep on the fresh mining slice found
  **1,622/3,000 failures** (vs 14/2,000 on the memorized train split); the retrained
  specialist recovers **148/627 survivors** (was 13 when starved). COMPOSED AT
  CONVERGENCE: **224/703 = 32%** (rate ROSE from 24% pre-convergence on a harder
  survivor pool — the repair loop improves WITH the parser, consistent with the
  equilibrium claim). **ARM=field_only: 226/703 — the fully-deployable arm matches
  and slightly beats gold-located spans.** The informal 3-5-point prediction lands
  at 0.3 points on the favorable side; the leakage bound is ZERO; the span channel
  adds nothing — "this factor's field is suspect" carries the entire repair signal.
  STRUCTURE BEATS LOCALIZATION, again, and the algebra number is now the same kind
  of number as KenKen's 47%: gold-free end to end. **THE PIPELINE HEADLINE: one-shot
  802 + repaired 226 = 1,028/1,500 = 68.5% of teeth-hardened algebra problems
  answered, fully deployable, through genuine search, at 3.2M trained parameters
  per head over a frozen 4-layer trunk.**
- **POST-FINAL-ROWS REGISTRATIONS (2026-07-08):** (1) SPAN-LEVEL FLAG MACHINERY
  RETIRED, measured-unnecessary (the best deletion: bought by clean ablation).
  Mechanism note: the specialist repairs in the currency it emits (typed factor
  slots) — field flags arrive pre-translated; text spans need slot-binding, the
  operation this architecture thrice showed it does poorly through shallow layers
  and now once showed it doesn't need. (2) EQUILIBRIUM LEDGER: two independent
  favorable reads in one day (survivor-hardening cancels density drop; repair rate
  RISES 24->32% on harder pools). Prior moved; the real test stays scheduled at the
  next convergence jump. (3) THE GAP TO MATH-500 IS ENUMERABLE: relation coverage
  (registry menu), phrasing wildness (template teeth vs prose), problem-shape
  diversity — corpus-and-registry questions, not architecture, exactly where the
  plane-ride design claimed the difficulty would live. (4) **THE EQUIVALENCE
  UPGRADE, cut ahead of multi-round:** the table's rows are graded INCONSISTENTLY —
  composed rows require FORCED answers; the one-shot 802 does not (lucky-unforced
  uncounted-for). The fix: ONE uniform metric — answer-at-query FORCED (solution-set
  equivalence at the query variable) — applied to every row; plus the correctness-
  boundary taxonomy (right-where-asked-with-wrong-graph vs lucky-unforced). Zero
  GPU-heavy; changes the baseline every future number reads against; runs BEFORE
  multi-round so the asymptote frame starts from the true floor.
- **THE EQUIVALENCE GRADE (2026-07-08, the table's uniform metric):** raw 802 ->
  lucky-unforced only **5** (the old metric was 99.4% honest) -> FORCED-CORRECT
  **797**. THE REAL FINDING: **132 of 797 (16.6%) are right-asked-wrong-graph** —
  a sixth of all correct answers come from graphs that DIFFER from gold factor-wise
  yet FORCE the right answer. The equivalence class is large: graph-match metrics
  undersell the parser heavily, and the MATH-500 grading policy question now has a
  measured magnitude attached (16.6% of credit rides on it). CORRECTED END-TO-END
  BASELINE: 797 + 226 = **1,023/1,500 = 68.2%** — the true floor, uniform across
  all rows; multi-round's asymptote frame starts here.
- **PRE-MULTI-ROUND REGISTRATIONS (2026-07-08):** (1) THE CURRICULUM-PURITY CHECK:
  prep labels failures by GRAPH-match — right-asked-wrong-graph parses (16.6% of
  correct!) may sit in the mining set as "failures," training the specialist to
  "fix" correct readings toward canonical gold (a bias against exactly the
  equivalence flexibility the 16.6% represents). Check the contamination, then
  filter: MINE FAILURES BY ANSWER-FORCING, not graph-match — the honest-metric
  principle applied one stage upstream. (2) GRADING-POLICY OPTIONS, registered while
  nothing is at stake: STRICT-GRAPH (undersells 16.6%, immune to equivalence
  errors); FORCED-ANSWER (tonight's metric — honest where uniqueness holds);
  ANSWER-MATCH (MATH-500's native grading — vulnerable to luck the forcing probe
  can't run on non-generated problems). The forced-vs-answer-match delta on
  generated corpora = the LUCK-INFLATION estimate to carry into any benchmark claim
  (measured tonight: 5/802 = 0.6% at this corpus's uniqueness density). (3) The
  99.4%-honest audit note: audits expected to confirm are still worth running —
  they convert "presumably fine" into load-bearing.
- **MULTI-ROUND (2026-07-08 night): THE REGISTERED FRAME CONFIRMED.** Per-round
  recovery: **123 -> 39 -> 5 -> 0** (19.6% -> 7.7% -> 1.1% -> 0%) — declining
  exactly as registered, hard asymptote by round 4 with 460/703 survivors
  unrecoverable by ANY depth of this loop. NO violation -> the ledger-conditioning
  hypothesis does NOT arrive early; the loop cannot move its own ceiling; the 46.7%
  frontier stays where the ranking put it (reading-repair: deeper prefix / ledger
  re-parse). Multi-round total: 76 + 167 = **243/703 = 34.6%**; END-TO-END:
  797 + 243 = **1,040/1,500 = 69.3%**. THE PURITY RESULT: **279/1,622 = 17.2% of
  mined "failures" were right-asked-wrong-graph** — independently matching the
  16.6% bigtest class rate (the equivalence class is a stable property of this
  parser, ~1/6 of its correct readings are non-canonical). HONEST TRADE-OFF: the
  pure specialist's round-1 recovery dipped (123 vs the impure 150) — fewer
  training samples and/or the removed parses carried useful canonicalization
  signal; multi-round total still exceeds the impure single round (243 vs 226).
  Round-2+ exists: +44 answers the single-round stack left on the table.
- **THE DECAY-SHAPE READING (2026-07-08, registered):** 19.6% -> 7.7% -> 1.1% -> 0
  is FASTER than geometric — a stochastic-retry world decays geometrically; this
  cliff is a HARD PARTITION: a fixable population pumped dry in two rounds + a
  different-in-kind remainder. Independent confirmation of the decodability
  boundary from a new instrument: the ceiling is a boundary between populations,
  not an average over a difficulty gradient. (2) EQUIVALENCE CLASS PROMOTED TO
  DESIGN PARAMETER: ~17% (16.6/17.2 across independent draws, within 0.6 points) —
  budgetable, not re-measured; every grading policy, curriculum filter, and
  benchmark claim inherits it. (3) PURITY-DIP DISAMBIGUATION, kept answerable:
  evicted-signal vs fewer-samples separates by a sample-matched impure retrain —
  not run (the trade was right regardless), recorded so it stays a question, not
  lore. (4) **THE 460 CHARACTERIZATION, registered prediction:** the reading-repair
  hypothesis says survivors are ENRICHED for referential-binding stress (oblique
  mentions, shuffled letters, size) relative to the recovered population — binding
  is the thrice-located shallow-layer weakness. UNIFORM across teeth = the
  reading-repair story needs rework BEFORE the ledger re-parse is built to serve
  it. Zero GPU; converts the boundary from a direction into a target profile.
- **THE 460 CHARACTERIZATION (2026-07-08): PREDICTION REFUTED — SURVIVORS ARE
  UNIFORM ACROSS TEETH.** Enrichment (survivors vs recovered, n=460 vs 243):
  oblique **1.01x** (0.850 vs 0.840), shuffled **0.92x** (0.687 vs 0.749),
  irrelevant **0.85x** (0.463 vs 0.547), n_vars **0.99x** (13.80 vs 13.91), band
  **1.07x** (2.44 vs 2.29). Not one feature above 1.07x; shuffled and irrelevant
  are slightly DE-enriched. The registered rule fires: the reading-repair story
  REWORKS before the ledger re-parse is built. WHAT THIS SEPARATES: binding stress
  explains the parser's ERRORS (thrice-located, real) but NOT which errors are
  UNREPAIRABLE — the teeth dials are orthogonal to survivorship. The hard
  partition is drawn along an axis the input-feature profile cannot see; the
  live candidates are PARSE-SIDE properties: error MULTIPLICITY per parse (the §6
  density-regime law predicts exactly this — at ~5 errors/failure, withhold-2 +
  single-round fixes exhaust at low multiplicity and strand the high-multiplicity
  tail) and wrong-FIELD mix (which heads are wrong, not which inputs are hard).
  Next cheap probe, registered: profile survivors vs recovered on
  errors-per-parse and per-field error distribution (gold is available; zero
  GPU). PREDICTION: survivors are enriched for error multiplicity ≥3; if THAT
  also comes back uniform, the remainder is plausibly decode-degenerate (belief
  never concentrates) rather than mis-read, and the frontier reranks toward the
  deducer-side suspicion transplant instead of any re-parse. (The ledger re-parse
  is NOT killed — it is unjustified-as-designed; its premise must be re-earned.)
  Script: `scripts/characterize_survivors.py` (commit 4e19697).
- **THE CONDITIONAL MULTIPLICITY PROBE, registered (2026-07-08, the relay's
  sharpening):** the naive prediction ("survivors enriched for multiplicity") has
  a TAUTOLOGY RISK — the loop fixes ~1-2 errors/round, so dense parses surviving
  is arithmetic, not discovery. The informative cuts are CONDITIONAL: (1) S(m) —
  survivorship as a FUNCTION of initial errors-per-parse + midrank AUC; (2) the
  mechanical model — recover-by-round ceil(m/f), grid f: does multiplicity alone
  reproduce 123->39->5->0? (3) the residual — within m-bins, do survivors differ
  in FIELD MIX or teeth? THREE FUTURES, thresholds pinned BEFORE measuring:
  (A) AUC>=0.75 AND decay reproduced (each round within ~2x) -> the loop is
  ROUND-BUDGET-LIMITED on the dense tail; answer = more rounds + better
  suspect-ranking, NOT re-parse or transplant. (B) AUC 0.6-0.75 or per-bin
  residual structure -> multiplicity real but unsaturated; the residual axis is
  the frontier. (C) AUC<0.6 -> plausibly DECODE-DEGENERATE; the deducer-side
  suspicion transplant reranks up. Side-glance registered: the shuffled/irrelevant
  DE-enrichment (0.92x/0.85x) predicts loud teeth produce LOW-multiplicity
  detectable errors — checkable in the same run. Script:
  `scripts/survivor_multiplicity.py`.
- **MULTIPLICITY PROBE RESULT (2026-07-08): FUTURE C FIRES MECHANICALLY —
  AUC(m->survival) = 0.524 < 0.6 — BUT WITH A POST-HOC SIGNATURE NONE OF THE
  THREE FUTURES ANTICIPATED.** Multiplicity is uniform too (means 8.13 vs 7.32);
  Future A doubly dead: NO fixes-per-round capacity reproduces the front-loaded
  123->39->5->0 decay (all mechanical models predict flat/rising per-round
  recovery) — iteration does not compound; third-instrument confirmation of the
  hard partition. THE SIGNATURE: S(m) is INVERTED at the low end — m=1 failures
  survive at **0.929** (n=42), m=2 at 0.837, vs ~0.71 for m>=4. The most
  unrecoverable population is the parses with the FEWEST errors. Candidate
  mechanism, named post-hoc: OMISSION BLINDNESS — `missing` is the largest error
  kind in both populations (~29-36% of error mass), and a missing factor is
  structurally invisible to the entire stack (withhold only REMOVES — worsening
  an underconstrained parse; the specialist's flags attach to EMITTED slots and
  its unflagged->copy objective actively teaches it never to ADD). Loud-teeth
  glance confirmed the relay's read: shuffled -> MORE errors (m 8.65 vs 6.15)
  yet DE-enriched among survivors — loud teeth make in-jurisdiction errors.
  REGISTERED FOLLOW-UP (CUT 4, prediction pinned before measuring): decompose
  m = m_add (missing) + m_corr (wrong-field/phantom/query). OMISSION-BLINDNESS
  predicts survival tracks m_add and collapses at m_add=0; flat-in-both =
  genuine decode-degeneracy and the transplant rerank stands. If omission-
  blindness confirms, the frontier is an ADDITIVE repair mechanism — the second
  look must be allowed to say MORE, not just different (the ledger re-parse
  premise re-earned in a narrower, reshaped form: omission-repair, not
  reading-repair). Script: `scripts/survivor_multiplicity.py` CUT 4; profile
  persisted to `.cache/survivor_profile_bigtest.npz` (future cuts zero-GPU).
- **CUT 4 RESULT (2026-07-08): OMISSION-BLINDNESS REFUTED TOO.** AUC(m_add) =
  0.525, AUC(m_corr) = 0.522 — survival tracks NEITHER decomposition. The cell
  table actively inverts the prediction: (m_add=0, m_corr=1) — one
  in-jurisdiction emitted-slot error, the stack's bread and butter — survives at
  **0.914** (n=35), the HIGHEST cell; (m_add=2+, m_corr=1) survives at 0.500,
  the lowest. m<=2 survivor error kinds are broad (rel_args 0.27, missing 0.27,
  given_value 0.21), not missing-dominated. TWO POST-HOC SIGNATURES: (1)
  query_wrong = 0.14 of m<=2 survivor error mass vs 0.03 population — ~5x
  enriched; a wrong query has NO slot: unwithholdable, unflaggable. (2) The
  (0,1)-cell inversion names SUSPECT-RANKING BLINDNESS: withhold-2 strips the
  two LEAST-confident factors; a confidently-wrong factor escapes, the flags
  hand the specialist the WRONG suspects, and unflagged->copy propagates the
  true error forever — which also explains the front-loaded decay (same
  mis-pointed flags every round). CONVERGENCE: this mechanism and the mechanical
  Future-C rerank agree — the binding constraint is SUSPICION QUALITY, not
  repair capacity; the deducer-side suspicion transplant is exactly a better
  suspect-ranker. NEXT PROBE registered (`scripts/survivor_suspicion_rank.py`,
  blank-parse-only — identity from the npz, no 4-round replay): P1
  AUC(min wrong-slot confidence-rank -> survival) >= 0.65; P2 withhold-2
  coverage several-fold higher among stage-1-recovered than low-m survivors;
  P3 m<=2 survivors dominated by rank-escapes + unflaggable query errors.
  FLAT ranks = the suspicion story dies too; decode-degeneracy stands.
- **SUSPICION-RANK RESULT (2026-07-08): THE SUSPICION STORY DIES TOO — THIRD
  CONSECUTIVE REFUTATION, AND IT FLIPS THE FRONTIER.** P1 FLAT: AUC(min
  wrong-slot rank -> survival) = **0.518** (bar 0.65); wrong slots sit near the
  bottom of the confidence ranking in EVERY population (min-rank-norm
  0.026-0.044; frac-in-bottom2 survivors 0.356 vs round-recovered 0.346).
  Localization is NOT the bottleneck. P3 INVERTED: only 26.6% of m<=2 survivors
  are rank-escapes/query — **~73% had their single error correctly flagged in
  the bottom-2 and the specialist STILL failed to fix it, 4 rounds running.**
  REPAIR GENERATION is the wall: told exactly where it is wrong, the parser
  cannot produce the correct replacement, and being deterministic it re-emits
  the same wrong content every round — the true explanation of the front-loaded
  decay. STRATEGIC CONSEQUENCE: the transplant rerank LOSES its rationale at the
  moment decode-degeneracy is confirmed — the transplant is a better RANKER and
  ranking is already adequate. Decode-degeneracy now stands on direct evidence,
  not elimination, and points at the frozen L0-L3 trunk: the information to
  reconstruct these factors is plausibly not in the trunk states at those
  positions (§6: a frozen 4-layer prefix can't BIND references) — binding
  weakness finally connects to survivorship at its correct jurisdiction: it
  explains WHICH CONTENT is unreconstructable, not which inputs are hard.
  CUT 2 registered (same script): flagged-but-unfixed survivor errors ENRICHED
  for rel_args vs flagged recovered errors, bar >=1.5x -> binding-is-the-wall;
  uniform-across-kinds -> head-capacity story, escalate to a trunk-information
  probe. Script: `scripts/survivor_suspicion_rank.py`.
- **CUTS 2/3/3b (2026-07-08): THE ANATOMY BOTTOMS OUT — THE PARTITION IS
  ENCODE-SIDE vs DECODE-SIDE.** CUT 2: the binding prediction refuted AGAIN at
  the relation level — rel_args **0.78x** (DE-enriched; flagged binding errors
  are relatively fixable); the enriched kind is **given_value 1.41x** (0.56 of
  m<=2 survivor flagged-error mass). CUT 3 anatomy: flagged given_value errors
  are predominantly VALUE-ELSEWHERE (misbinding — right number, wrong variable)
  in every population (survivors 0.755, round-recovered 0.900); hallucinated
  values (not-in-gold) are the minority but **2.4x enriched** among survivors
  (0.245 vs 0.100). CUT 3b: SWAPS pass the ratio bar (0.043 vs 0.014 = 3x >
  1.5x) but FAIL ON MASS — 10/233 cases; the joint-decode conclusion pinned to
  the bar does NOT fire. LEDGER LESSON: **an enrichment bar without a mass bar
  is a trap** — register the mass the mechanism must explain, not just the
  ratio (the §6 enforcement-is-arithmetic rule applied to one's own bars).
  THE SYNTHESIS: dominant unfixable errors are ONE-DIRECTIONAL misbindings —
  no coordination needed, a single-slot edit suffices — yet 4 correctly-flagged
  rounds fail. The chain closes on the substrate: every repair round re-decodes
  the SAME precomputed frozen trunk states; only flag features change. If the
  binding/value was mis-committed AT ENCODING TIME, no head-side conditioning
  recovers it. THE HARD PARTITION, NAMED: the repair stack drains DECODE-side
  errors (front-loaded 123->39->5->0) and cannot touch ENCODE-side casualties —
  which explains every refutation at once (uniform teeth: encoding failures
  happen at some rate on all inputs; flat multiplicity; adequate localization;
  flagged-but-unfixed). ORACLE-CEILING ARM registered
  (`scripts/survivor_oracle_ceiling.py`): perfect gold-derived per-field flags
  (= the specialist's TRAINING regime; deployed withhold-flags were doubly OOD:
  mis-pointed + all-fields pattern) re-derived each round, 4 rounds, on the 460.
  Upper-bounds ALL flag-quality improvements in one number. REGISTERED: <10%
  recovery -> encode-side wall MEASURED (frontier = change the ENCODING:
  second-view re-render with position-aligned suspect marks per the §6
  structure law, and/or deeper prefix — NOT any suspicion/repair improvement;
  the transplant rerank dies too). >30% -> the deployed flag deriver was the
  constraint (fix it — cheapest win of the week). 10-30% -> partition and
  re-profile the oracle-recovered.
- **ORACLE-CEILING RESULT (2026-07-08 night): 64/460 = 13.9% — the mixed band,
  near the low end; THE ENCODE-SIDE WALL IS MEASURED.** Perfect gold-derived
  per-field flags (the specialist's own training regime), re-derived each round,
  4 rounds: 44 -> 16 -> 4 -> 0 (front-loaded AGAIN — 4th independent sighting
  today). **396/460 = 86% of survivors — 26.4% of the corpus — are unrecoverable
  even when told exactly which slot and which field is wrong.** Flag quality was
  never the lever: the ceiling bounds EVERY suspicion/ranking/repair improvement
  (tier-0 trained head, transplant, better derivers) at <=64 cases (<= +4.3 pts
  end-to-end, and only partially capturable gold-free). The frontier is a BUILD
  decision now — change the ENCODING (second-view re-render with position-
  aligned suspect marks per the §6 structure law, and/or deeper prefix), options
  + dead-ends tabled in `docs/NEXT_SESSION.md` for the relay seam. Registered
  follow-up left open per the 10-30% rule: partition + re-profile the 64
  oracle-recovered vs the 396 (zero GPU, profile npz on disk).
- **THE DECISION + PRE-BUILD DISCRIMINATORS (2026-07-08 night, relay call):**
  OPTION 1 (second-view re-render) is the chosen frontier — the multi-cycle
  Alternator's first empirical mandate (396 encode-side casualties = the
  measured need for re-reading under deductive feedback; the speculative
  component's customers arrived before the component, same as tier-0/tier-3).
  HAZARD FLAGGED: learned mark embeddings backprop through the trunk = the
  documented AM-driver hang. v0 = ZERO-NEW-PARAM marks: reserved vocabulary
  tokens inserted at suspect spans (token-shift/gold realignment machinery
  exists), forward-only re-encode; learned marks are the gated upgrade arm.
  WHY text-NACK's death doesn't transfer: it died on REFERENTIAL binding
  ("statement 7" -> sentence 7 through 4 shallow layers); a position-aligned
  mark carries its information BY BEING AT THE LOCATION — nothing to bind
  (the §6 structural-entry law at the trunk's front door). TWO DISCRIMINATORS
  FIRST: (a) option-4 re-profile of oracle-64 vs hard-396 (zero GPU, folded
  into the oracle script rerun); (b) THE DEPTH PROBE
  (`scripts/survivor_depth_probe.py`) — a fresh value-probe reads gold given
  values from mean-pooled gold-span states at L4 vs L8, evaluated on clean
  baseline vs wrong-givens-recovered vs wrong-givens-396. THREE-WAY VERDICT,
  bars pinned: instrument bar base-L4 > 0.70 else no verdict; ROUTING = 396
  within 10pts of baseline at L4 (info present, head mis-routes — the §6
  attention-bootstrap ghost; marker = attention beacon; deeper prefix retired);
  DEPTH = >=20pt gap at L4 AND L8 closes >=50% (deeper prefix wins; re-render
  overkill); CONTENT = >=20pt gap, L8 closes <50% (never written; re-render
  mandated). v0 FRAME registered (density regime: multi-error encode-side-
  selected population): ANY meaningful recovery on the 396 moves a measured
  ceiling; relay's directional prediction — recovery CONCENTRATED in
  one-directional given-misbindings (mark = "re-examine this given" at its
  location) confirms directed attention; UNIFORM recovery = generic re-rolling.
- **DEPTH-PROBE VERDICT (2026-07-08 night): ROUTING WALL — DECISIVE.** Fresh
  value-probe, digit-exact at gold given spans: base **L4 0.998 / L8 0.953**;
  wrong-recovered 1.000/0.946; **the 396's wrong givens 0.996/0.942**. The gold
  value is FULLY PRESENT in the current L4 encoding at its location, at
  baseline fidelity, on exactly the givens the parser misbinds and cannot
  repair. DEEPER PREFIX RETIRED with a number (L8 <= L4 everywhere — depth
  smears local literals). HONEST CORRECTION to the partition's name:
  "encode-side" was operationally right (no conditioning fixes it) but
  mechanistically WRONG — the partition is decode-side vs **ROUTING-side**:
  the trained pointer circuit deterministically reads the wrong location (§6
  attention-bootstrap law: pointers don't move without direct supervision), and
  this happens DESPITE span-supervised training. Option-4 re-profile came back
  FLAT (oracle-64 vs 396: mult 0.89x, kinds ~1x, teeth ~1x) — the 64 are
  flag-quality stragglers of the same population, one mass. THE FORK OPENED BY
  THE VERDICT (relay to adjudicate): **(A) span-restricted structural read** —
  the probe IS a repair head (reads only the span, 0.996 on the hard
  population): flagged given slot -> pool states over the suspect var's
  predicted mention span -> probe-decode value -> substitute -> re-solve. Zero
  re-render, zero retrain; deployability = gold-free span prediction quality
  (mention head exists, unmeasured on survivors); scope = given_value class
  (~0.36 of survivor error mass). **(B) marker-token re-render v0** — the
  general mechanism (now understood as attention BEACON, not re-encoding);
  design wrinkle: deployment places the marker via the model's own routing
  (marks where it LOOKS, not where it should look) — the train/deploy placement
  gap needs its own registered measure. **(C) both, sequenced** — A as the
  immediate bite, B as the Alternator build with A's recovery as the baseline
  to beat. Probe cached: L8 states `.cache/algebra_bigtest_L8_states.npy`;
  script `scripts/survivor_depth_probe.py`.
- **ARM C REGISTERED + FIRED (2026-07-09): REPLACE-AND-SOLVE — candidate-
  restriction repair, the sibling of withhold-and-solve.** The relay's
  adjudication: C is not merely cheapest-first — the routing verdict makes it
  the MECHANISTICALLY CORRECT fix (don't steer a learned pointer; shrink its
  candidate space — the same move as every §6 win), and it demotes marker-v0
  from thesis-test to comparison arm. Law 7's read-side discipline applied to
  attention itself: remove/restrict at READ, from the candidate set, never from
  state — the state was fine all along. DESIGN SHARPENING at build time: for
  GIVEN VALUES the right reader is not even the probe — values are digit
  literals in the input text; the inventory lexes symbolically at 1.0
  reliability (the probe earns its place in v1 on args/ops where lexing can't
  reach). Moves: REPLACE (one given's value <- inventory; withhold could never
  fix a misbound given — removal loses the constraint; replace keeps it,
  corrected) + SWAP (the coordinated two-slot exchange a parallel marginal
  decoder can't emit). Solver disposes; ACCEPTANCE pinned: all passing moves
  must agree on the answer, else ambiguous-reject. Fully gold-free.
  DEPLOYMENT-HONESTY: fires only on VISIBLE failures; forced-WRONG originals
  (gold-only failures) counted separately — they are a grading-policy problem,
  not a repair problem. REGISTERED: mine — 60-120/460, concentrated low-m
  given_value, ambiguity <10% of fixable; relay (polarity-flipped) — this arm
  takes the BULK of the convertible fraction and the beacon adds ~nothing on
  top (pointers don't re-aim under conditioning; a beacon is conditioning via
  input) — beacon-beats-floor would be the interesting outcome; soundness —
  accepted-but-wrong ~0 (forced-unique acceptance admits no luck; measured).
  RELAY PRE-READ on the mention-quality unknown: teeth-uniformity implies
  survivors aren't adversarial text, so mention F1 should hold; degradation
  would CONTRADICT the uniformity table and be a finding itself. §6 PROMOTION
  QUEUED (fifth sighting, first at inference): **pointer errors are never fixed
  by conditioning — only by candidate restriction (structure) or retraining
  (supervision).** Two remedies, zero exceptions so far. Script:
  `scripts/repair_replace_swap.py`.
- **ARM C v0 RESULT (2026-07-09): SOUNDNESS GATE FAILED — AND THE FAILURE IS
  THE FINDING.** 60 accepted, **55 WRONG** (luck gate predicted ~0, measured
  0.92): FORCED-UNIQUE IS NOT A CORRECTNESS CERTIFICATE when the graph itself
  is suspect — substitute into a multi-error parse and you can force a
  consistent, unique, WRONG answer. Accepted set: mean m 8.12, m<=2 share 0.13
  — imposters on high-error parses, not repairs. Honest yield +5/460; as
  designed the stage ships 55 indistinguishable wrong answers — NOT DEPLOYABLE.
  Both registrations refuted (mine 60-120 concentrated-low-m: off 10x with the
  wrong shape; relay's bulk-of-convertible: the single-move convertible
  fraction is ~5). THE DEEPER DISCOVERY: **70/460 survivors are forced-WRONG
  originals — deployment-INVISIBLE.** A single misbound given usually keeps the
  system fully constrained -> forces cleanly to a wrong answer -> looks like a
  SUCCESS gold-free. The m=1 survivors at 0.93 were hiding here all along: the
  routing-wall population mostly doesn't present as failure. THE BUG-CLASS
  THIS EXPOSES: every acceptance test in the measured stack compared against
  GOLD; deployment accepts any forced answer at every stage. Three
  contaminations: forced-wrong one-shots accepted wrong (never reaching
  repair); phantom recoveries (measured recoveries whose original was forced —
  deployment never fires); withhold/round imposters. AUDIT REGISTERED + FIRED
  (`scripts/deployment_honest_audit.py`): full stack replayed under gold-free
  acceptance; P1 one-shot forced-wrong 100-180; P2 phantom recoveries >0; P3
  per-stage precision declines down the stack; P4 deployment-honest end-to-end
  < 0.693. NOTHING IS QUOTED WITHOUT THIS NUMBER AGAIN. Law candidate (with
  arm C as first sighting): **acceptance criteria must be measured for
  imposter rate at the deployed error density** — "forced-unique" was pinned
  as sound from a clean-graph intuition and failed at m~8.
- **DEPLOYMENT-HONEST AUDIT RESULT (2026-07-09): P4 REFUTED UPWARD — the
  honest number is 1051/1500 = 70.1% (> the gold-checked 69.3%), and the audit
  caught a SECOND acceptance bug on the way.** Per-stage gold-free: one-shot
  887 accepted / 797 correct (precision 0.899; 90 forced-wrong committed — P1
  near-miss vs 100-180); withhold 127/74 (**0.583 — the LEAKIEST stage**:
  removal-based acceptance weakens forcing, arm C's law quantified in the
  deployed stack); rounds 203/140, 56/38, 3/2, 1/0. Answered 1277 (abstained
  223), answered-precision 0.823. P2 CONFIRMED: ~20 of the measured 243
  recoveries were PHANTOM (their originals forced wrong at one-shot; deployment
  commits them wrong and never repairs). P3: pattern yes, monotone no. WHY P4
  INVERTED: the measured pipeline's round acceptance ONLY evaluated the
  WITHHELD variant (solve_check always ran k_wh=2) — a fully-correct specialist
  re-parse, minus its two least-confident (correct) factors, often un-forces
  and was REJECTED. The audit's accept-plain-first ordering recovers ~+11 net
  correct (round-1 140 vs measured 123). SECOND SIGHTING of the acceptance-
  criteria law in two days: acceptance rules must be AUDITED as mechanisms, not
  assumed — both bugs (gold-checked accepts; withheld-only round accepts) were
  invisible to every headline number until replayed under deployment rules.
  **THE QUOTABLE NUMBER GOING FORWARD: 70.1% deployment-honest end-to-end,
  answered-precision 0.823, abstention 14.9%.** The 226 committed-wrong answers
  are the measured customer for a calibrated abstention signal (the
  waist-space/OOD thread — probe queued behind this audit).
- **WAIST PROBE REGISTERED + FIRED (2026-07-09, the autoencoder thread lands
  as instrumentation):** relay + Code corrections adopted — the probe space is
  **fst**, the algebra head's slot-vector bank (the one decoder-backed waist
  space; the TAP has no decoder — category error corrected; pointer heads
  excluded as problem-relative). HALF 1, interpolation coherence, registered
  **50/50** (the coordinate-swap evidence does not bear on convex combination;
  128d-lossless hints tame): same-kind cross-problem pairs, alpha=0.5, decoded
  through the linear field heads; pinned coherent = sharpness ratio >=0.80 AND
  midpoint-decodes-an-endpoint >=0.50. Coherent -> KL machinery buys little;
  garbage -> measured deficiency for the per-kind-prior VAE arm (single-prior
  KL stays parked — cousin of the attract dud). HALF 2, the paying customer:
  per-kind fst centroids from TRAIN (deployable labels), score = worst slot
  cosine to claimed-kind centroid, vs the audit's 226 committed-wrong / 1051
  correct. Registered: dense AUC 0.55-0.65 (misbindings look locally normal);
  USABLE-FLAG bar precision@top-10% >= 2x base (0.354). Classified on arrival
  per the portfolio rule: dense ranker / rare-flag / dead. Audit script now
  persists per-sample outcomes (`.cache/deploy_audit_bigtest.npz`).
- **WAIST PROBE RESULTS (2026-07-09): BOTH HALVES LAND.** HALF 1 **COHERENT**,
  decisively: sharpness ratio 0.940 (bar 0.80), midpoint-decodes-an-endpoint
  0.843 (bar 0.50), n=561 same-length cross-problem pairs. The parse-side waist
  is SMOOTH WITHIN KIND — convex combinations decode cleanly; the
  coordinate-swap evidence indeed did not transfer to interpolation. Per the
  pinned rule: **KL/VAE machinery buys little — parked, no deficiency**
  (the per-kind-prior note stays parked alongside). HALF 2: dense AUC **0.728**
  — my registered prior (0.55-0.65) REFUTED UPWARD; clears BOTH portfolio
  bars: dense ranker (>=0.70) AND rare-flag (precision@10% 0.417 vs bar 0.354;
  top-20% = 45% recall of committed-wrong at 0.40 precision). First instrument
  that consults neither solver nor emission-head confidence; the two halves
  cohere (centroid distance is meaningful BECAUSE the space is smooth).
  POLICY NOTE pinned before use: blind abstention LOSES accuracy on
  MATH-500-style grading (drop ~153 correct to avoid ~102 wrong at top-20%);
  the paying use is **flag-as-NACK-on-ACCEPTED-answers** — route flagged
  accepts through a second look instead of committing. This is the §8.5
  session-monitor role arriving from measurement (retransmission decisions on
  accepted traffic), not from spec. NEXT CUT (cheap, registered direction not
  prediction): stage-split of flagged wrongs — concentration in the 90
  one-shot invisibles would close the arc (the routing-wall population, first
  invisible to repair, now visible to the waist monitor). Script:
  `scripts/waist_abstention_probe.py`.
- **RATCHET-NACK REGISTERED + FIRED (2026-07-09, relay adjudication: ratchet,
  not re-roll):** flagged accepted answers KEEP their original by default; the
  NACK round's revision replaces only if it STRICTLY DOMINATES. The asymmetric
  hazard: most flags land on correct answers (~0.4 precision), and the 0.998
  unflagged-preservation number was measured on FAILURES, not re-opened
  successes — a population the specialist was never trained to leave alone.
  v0 SCOPE: stage-0 accepts only (the detector scores plateaued-parser space;
  later-stage parses live in specialist space, centroids uncalibrated — the
  per-stage score table logged this run decides v1); v0 DOMINANCE = revision
  forced + unflagged slots factor-exact-agree (anomaly-decrease criterion
  deferred to v1 for the same cross-space reason). REGISTERED BARS: break rate
  0 by construction (any break = dominance leak -> tighten); fix rate on
  flagged stage-0 wrongs > 0.346 (relay, density regime stated: committed-wrong
  parses are NEAR-MISSES by selection — the regime where per-round repair is
  strongest); deploy iff answered-precision rises without net answers falling
  (floor 0.823 / 70.1%). TWO DIALS from one detector: recovery mode (ratcheted
  replacement) and precision mode (flag->abstain on undominated). Stage-split
  logged (direction: flagged wrongs concentrate in the 90 one-shot invisibles);
  per-stage calibration column logged (if repaired parses skew anomalous even
  when correct, v1 needs per-stage calibration before trusting the ratchet
  beyond stage 0). **THE MONITOR SENTENCE (relay): the session monitor — the
  perceiver's last unclaimed job but hosting global latents — arrived as a
  centroid library plus a distance threshold, watching accepted outputs and
  routing suspicion back into the loop. Zero parameters, eighteen months early,
  measured on arrival.** Script: `scripts/ratchet_nack.py`.
- **RATCHET-NACK v0 RESULT (2026-07-09): KILL BAR FIRED + FIX-RATE REFUTED —
  DETECTION AND REPAIR ARE SEPARATE CAPABILITIES.** BREAK=1 at top-20% (0 at
  top-10%): a correct parse's flagged slot revised to a different value that
  still forced with unflagged agreement — formally dominant, actually wrong.
  The dominance test LEAKS; the deferred anomaly-decrease criterion is
  LOAD-BEARING (v1 needs per-space centroids). FIX RATE 1/17 / 2/37 (~6%) vs
  the 0.346 bar — relay prediction refuted WITH MECHANISM: the flagged stage-0
  wrongs ARE the routing-wall invisibles — near-misses BECAUSE they are single
  confident misbindings, the population the oracle ceiling already measured
  86% unfixable with perfect flags. The selection argument never asked WHY the
  parses were near-misses. **SELECTION EFFECTS HAVE JURISDICTIONS TOO** (the
  density-law's cousin, 5th sighting of unexamined-population predictions).
  PER-STAGE COLUMN: stage-0 AUC 0.745; stage-1 DEAD (0.532); correct parses
  drift anomalous with stage (0.703->0.748) — per-stage calibration required
  before v1 trusts anything beyond stage 0. Wrong-recall roughly uniform
  across stages (~0.41-0.51); false alarms concentrate late. DIALS AS
  MEASURED: recovery mode FLAT (+1 net, 70.1%); precision mode REAL —
  **0.880 answered-precision at 0.615 end-to-end** (top-20%) — the instrument
  for wrong-costs-more-than-missing deployments (not MATH-500). THE HONEST
  ROLE: the monitor SEES (0.728) but this specialist cannot FIX what it sees;
  the flagged population's only untried repair is the parked BEACON arm
  (polarity-flipped prediction standing: input marks can't re-aim trained
  pointers) — beacon failure there = the population is DETECT-AND-ABSTAIN
  ONLY under current machinery. Script: `scripts/ratchet_nack.py`.
- **THE BEACON, FIRED AS THE CLOSING MEASUREMENT (2026-07-09, relay
  adjudication):** the 396's story is complete except one sentence and both
  endings close it. Population: the 460 (states 99.6% correct, pointer
  mis-aimed, 86% unfixable under perfect flags; every conditioning repair dead
  by measurement). The beacon is the ONLY untried arm — INPUT-level saliency,
  mechanistically distinct from all head conditioning. v0: bracket the suspect
  sentence (flagged slot's attention-argmax sentence — marking where the
  pointer LOOKS) with reserved token 128002, forward-only L0-L3 re-encode,
  UNCHANGED heads re-parse, gold-free acceptance with right/wrong split.
  PINNED ENDINGS: <=2% recovery -> flipped prediction confirmed, pointers
  don't re-aim under input conditioning either, the population is
  DETECT-AND-ABSTAIN ONLY, chapter closes zero-loose-ends; >=10% -> input
  marks move what conditioning can't (the week's most interesting result;
  beacon graduates). COMPOSABILITY COLUMN (relay): monitor score on
  beacon-accepted parses — drops-on-repairs = detect->beacon->re-score
  composes into a self-contained final tier; no-movement = the monitor can't
  certify its own fixes (ratchet lesson one level up). v1 RATCHET explicitly
  DEFERRED (relay): building better replacement machinery for a population
  measured 86%-unreplaceable is the infrastructure-before-customer trap; waits
  for a population both flaggable AND fixable. §6 LINE (relay's named form,
  third sighting): **a selection criterion's jurisdiction is which property it
  selects on — "survived filter X" is evidence about detectability, not
  repairability.** NEXT CHAPTER after the verdict: the registry expansion
  (MATH-500 relations), carrying the design constraint forward — every new
  relation's pointer gets candidate restriction and span supervision FROM
  BIRTH, or it grows its own 396. Script: `scripts/beacon_closing_arm.py`.
- **BEACON VERDICT — THE CHAPTER CLOSES (2026-07-09): 14/460 = 3.0%,** middle
  band (bars were <=2% / >=10%), honest call: the flipped prediction is
  CONFIRMED IN SUBSTANCE — input marks do not meaningfully re-aim trained
  pointers; the population is **DETECT-AND-ABSTAIN ONLY under current
  machinery**; the beacon does NOT graduate. THE FOOTNOTE, recorded: 11 of 14
  recoveries are in the hard-396 — parses PERFECT oracle flags could not fix.
  Input perturbation occasionally moves what head conditioning provably
  cannot — an EXISTENCE result, not a mechanism: 71 accepted-WRONG vs 14 right
  (precision 0.165; the marks shake the table, they don't aim the pointer).
  Third consecutive confirmation of the acceptance law. Composability: the
  monitor ranks marked-state accepts in the correct direction (right 0.736 <
  wrong 0.782) — a score-gated beacon is possible in principle but is v1
  machinery for a 3% mechanism (infrastructure-before-customer; deferred with
  the v1 ratchet). **THE SURVIVOR ARC, FINAL LEDGER:** teeth uniform ->
  multiplicity flat -> omission dead -> suspicion flat -> binding de-enriched
  -> routing wall (states 99.6% correct, pointer mis-aimed) -> oracle ceiling
  13.9% -> invisibles found (70 forced-wrong) -> acceptance bugs x2 -> honest
  70.1%/0.823 -> monitor arrives (0.728, zero params) -> ratchet leak + fix
  6% -> beacon 3%. Nine registered refutations, four §6-grade laws, two
  retired builds, one working instrument, one closed population. NEXT
  CHAPTER: the registry expansion toward MATH-500, carrying the constraint —
  every new relation's pointer gets candidate restriction + span supervision
  from birth, or it grows its own 396.
- **THE MATH-500 BAND-SWEEP (2026-07-09 — the registry chapter's first move,
  measurement before build):** MATH-500 acquired (`.cache/math500_test.jsonl`,
  500 problems, subject/level/answer labels — MEASURED, never trained on).
  Transparent regex classifier + n=20 hand-audit. EXPECTATIONS SCORED:
  plain-integer answers **62.2%** (registered 50-60 — slightly above; +11.2%
  fractions = rationals near-term; ~26% tail of expressions/tuples/radicals/
  intervals); algebra+prealgebra 41.2% (reg ~40 ✓); geometry 26% (reg ~20,
  under-called); INEQUALITY **14.0% measured with a KNOWN UPWARD BIAS** (the
  audit caught "for n >= 1" domain qualifiers firing the tag; reasoning-core
  share <10%) -> **ARCHITECTURE VERDICT AS REGISTERED: interval reasoning is a
  LATER chapter; csp_core's predicate interface stays closed.** AUDIT
  CAVEATS: "factor" inflates quadratic/poly; the linear-arith-only residual
  (17.2%, mean level 2.8 — the easiest band) is OPTIMISTIC (radical-
  simplification problems hide in it) — current-registry-reachable < 17%.
  GREEDY SET-COVER LIST (marginal): geometry +52, quadratic/poly +49,
  trig/precalc +54, modular/divis +44, inequality +41, combinatorics +41...
  (100% at 13 categories). ENGINE-FIT ANNOTATION (for the relay's tranche
  call): raw coverage ranks geometry/trig first, but they need different
  FRONTS (diagrams, continuous identities — far from factor-graph CSP);
  the CSP-NATIVE tranche is **quadratic/poly (integer polynomial roots —
  still integer-domain, search-tier-able), modular/divis (GAC-native — the
  search tier eats these), ratio/percent, sequence/series, base-repr,
  abs-floor** — factor-graph-friendly relations entering through predicate +
  bridge as always. Mean-level column: linear-arith 2.8, inequality 4.1
  (difficulty tracks the shopping list's tail). Script:
  `scripts/math500_band_sweep.py`.
- **TTA + PROGRESSIVE RESIZING REGISTERED (2026-07-09, Bryce's fastai
  transplant; TTA FIRED):** the MC-pi lens pins the design constraint — views
  must be solution-preserving with DECORRELATED failure modes (correlated
  darts estimate nothing). DEPLOYMENT-HONESTY FLAW NAMED BEFORE FIRING:
  re-rendering needs GOLD factors — oracle machinery at MATH-500 time (the
  graph is what parsing is FOR; re-rendering the parser's own graph
  correlates with its errors). TWO ARMS, mirroring oracle-flag->deployed-flag:
  ARM O = K=4 gold re-renders (letters/templates/surfaces/order shuffled,
  teeth-easy — the mechanism ceiling); ARM D = K=4 sentence permutations
  (pure text transform, graph-free, deployable; sentence-index features
  genuinely shift). THE MC-PI GATE measured FIRST per arm: same-wrong-answer
  rate across views on wrong-forced originals < 0.30, else voting is VOID.
  REGISTERED: relay — ARM O voting recovers a NONZERO routing-wall slice
  (the only mechanism class that changes what the pointer keys on;
  sidesteps, not steers) + agreement-AUC near/above 0.728 (portfolio-
  combinable: behavioral stability vs representation geometry); mine — O
  decorrelates (<0.30), D weaker (0.30-0.60, misbinding plausibly local to
  sentence content), voting net-positive in O / ~flat in D; imposter split
  standard; vote rule majority >=3 of 5. PROGRESSIVE RESIZING (registered,
  rides with the registry tranche — no run now): the resolution axis is
  RENDERING COMPLEXITY (teeth + n_vars/band), not waist width (that is
  capacity, already measured); each new relation ships a coarse->fine
  curriculum knob + the ablation (curriculum vs all-teeth-from-birth, same
  budget). 50/50-leaning-modest; the bet is POINTER-CIRCUIT PREVENTION
  (candidate-restricted pointers learn binding on unambiguous mentions before
  obliqueness arrives — prevention of the next 396 is worth more than a point
  of fac-exact). One machinery, two clocks: TTA is the generator's symmetries
  at inference; the curriculum is the same dials at training. Script:
  `scripts/tta_views.py`.
- **TTA RESULTS (2026-07-09): THE DEPLOYABLE ARM DOMINATES THE ORACLE ARM,
  AND COMPOSITION SETS A NEW HEADLINE — 1072/1500 = 71.5% / precision
  0.833.** ARM D (sentence permutation, graph-free): MC-pi gate PASSED at
  0.106 same-wrong (my 0.30-0.60 registration REFUTED upward — even pure
  order-shuffle moves the misbindings; 20% of wrong originals' views come
  back RIGHT); voting 763R/13W/724A = answered-precision **0.983**;
  routing-wall recoveries **33/460 survivors + 8/90 invisibles** (more than
  beacon+ratchet combined, deployable, zero training); agreement-AUC
  **0.840** — the board's best anomaly signal (beats the waist monitor's
  0.728; behavioral stability > representation geometry). ARM O (gold
  re-renders): most decorrelated (0.031) but per-view forced collapses to
  ~21% (always-shuffled letters are far off the parser's operating point) —
  69 accepts (all right), agreement-AUC 0.637. **THE MC-PI SECOND CLAUSE,
  measured: darts must be independent AND land on the board** — view design
  trades decorrelation against per-view competence, and gold access was not
  the binding resource; parse competence was. The oracle arm is the floor,
  the deployable arm the ceiling — a happy inversion. COMPOSITIONS (zero-GPU
  from persisted outcomes): COMPOSE 1 (vote->stack) **1072/1500 = 71.5%,
  precision 0.833, answered 1287** — all three dials beat the floor
  simultaneously; deploy-iff met; THE NEW QUOTABLE. COMPOSE 2 (stack->vote)
  70.7%. Portfolio note: low-agreement top-128 catches 31/226 stack
  committed-wrongs. Relay's nonzero routing-wall prediction CONFIRMED on
  both arms. npz: `.cache/tta_arm_{O,D}_bigtest.npz`. Scripts:
  `scripts/tta_views.py` + inline composition.
- **PORTFOLIO + THRESHOLD RESULTS (2026-07-09 night):** READ 1 bars REFUTED
  AS REGISTERED — Spearman(disagreement, waist) = 0.464 (bar <0.4), combo AUC
  0.833 < agreement-alone 0.840 (bar >0.86) — BUT the tail tells the opposite
  story: the combo WINS at every abstention operating point (top-10%:
  flag-precision 0.528 vs 0.386, kept-precision 0.862 vs 0.846; +1.6pt).
  MY BAR WAS MIS-REGISTERED per the existing §6 law — AUC is a whole-ranking
  summary, abstention is a TAIL decision (4th sighting of
  metric-must-match-decision-structure, this time in my own registration).
  HONEST VERDICT: the portfolio PAYS where it is operated; the waist signal's
  rare-flag character complements behavioral agreement in the tail while
  diluting the mid-ranking. READ 2 CONFIRMED (relay): the certification
  channel exists — **unanimity 5/5 = 0.9982 precision at 38.1% coverage
  (570R/1W)**; t=4/5 = 0.9925 @ 44.3%; t=3/5 = 0.9832 @ 51.7%. LEDGER
  SENTENCES adopted from the relay: (1) THE THIRD CATEGORY — nine mechanisms
  tried to FIX THE ESTIMATOR (same input, better behavior); TTA changed what
  the input looks like and AVERAGED — randomizing away bias instead of
  repairing the instrument; deterministic surface-keyed failure is TTA's
  favorite food. (2) THE INDEPENDENCE-COMPETENCE LAW (§6 candidate, both
  clauses): darts must be independent AND drawn from the distribution the
  board was calibrated on — decorrelation buys nothing past the point where
  per-view competence falls faster. (3) CURRICULUM METRIC REFRAME:
  progressive resizing chases VIEW-ROBUSTNESS (competence under re-rendering,
  measurable per-relation from birth via the TTA harness), not raw fac-exact
  — a voting system needs per-view competence above the vote's break-even,
  after which independence does the rest. Scripts:
  `scripts/portfolio_and_threshold.py`; waist scores persisted
  (`.cache/waist_scores_bigtest.npz`).
- **THE DECISION LATTICE, FROZEN (2026-07-09, relay structural note): the
  deployment stack's four rungs and their interfaces — the productized
  Alternator, whatever the multi-cycle loop becomes.** Every rung
  zero-parameter and gold-free:
  1. **CERTIFY** — TTA-D K=5 unanimity of forced answers. Dial: **0.9982
     precision @ 38.1% coverage.**
  2. **ANSWER** — majority 3/5 vote; on vote-abstain, the deployed stack
     (one-shot -> withhold-2 -> 4 specialist rounds), gold-free forced
     acceptance. Composite dial: **71.5% end-to-end / 0.833 precision.**
  3. **FLAG** — rank-sum(view-disagreement, waist-centroid distance) read at
     the tail; downgrade or abstain per deployment mode. Dial: kept-precision
     **0.862 @ 10% abstention.**
  4. **ABSTAIN** — no forced answer anywhere.
  INTERFACES: views = solution-preserving TEXT transforms (sentence
  permutation v0; any future view generator is priced by the
  independence-competence curve); votes = forced answers only; anomaly
  signals = agreement + per-kind waist centroids (train split). **THE
  EXPANSION ACCEPTANCE TEST: a new relation passes when the lattice HOLDS ITS
  DIALS on the expanded domain — one table, all four rungs — not on fac-exact
  alone.** A relation that lifts fac-exact but degrades a certification dial
  fails acceptance. (The cheapest insurance against the tranche quietly
  breaking a channel nobody re-measured.)
- **THE TRANCHE CHARTER (inherited, one breath):** relations enter as
  predicate + bridge, zero core edits — the generality law's next test, now
  on the neural side too; corpora solution-first, gate-checked, band-labeled,
  mention-spanned; every pointer born candidate-restricted and
  span-supervised (the 396's rule as prevention); every relation ships its
  curriculum knob chasing VIEW-ROBUSTNESS; the five-seat audience grades from
  the first checkpoint. **DECISION PENDING (Bryce + relay, BEFORE the
  generator is written): MULTI-ROOT ANSWERS.** Quadratics break the
  single-forced-value frame; the policy determines gold format, forcing-probe
  semantics, and what unanimity MEANS on a set. Options tabled: (1)
  ANSWER-SET gold (forcing probe generalizes via ban-and-resolve enumeration;
  every lattice rung's semantics changes; multi-value answers are only 2.0%
  of MATH-500); (2) QUERY-CONSTRAINED single root (the NL carries a selector
  — "the positive solution", "the larger root" — a new mention type under the
  pointer law; forcing probe and ALL lattice semantics unchanged; matches
  MATH's dominant convention); (3) hybrid: selector-graded with the full set
  in gold metadata as a diagnostic column. CODE'S RECOMMENDATION: (2) as v0
  with (3)'s metadata — quadratics enter as just-another-relation plus one
  supervised mention type; the set-frame waits for the 2% it serves.
- **MULTI-ROOT POLICY RATIFIED (2026-07-09, relay + Code): option 2 with
  option 3's metadata.** Option 1 failed on jurisdiction grounds the ledger
  owns — re-defining every lattice rung a day after the freeze to serve a
  measured 2% is the mass-bar law applied to design. THE SELECTOR SPLIT (the
  load-bearing detail — pointer law + division of labor at the same door):
  the parser's selector head classifies the selector TYPE from a supervised
  mention span (positive / larger / smaller / in-range — a small CLOSED
  vocabulary, the <=32-way codebook-selection regime that bootstraps from
  task gradient); the SOLVER enumerates roots and applies the comparison
  symbolically. The parser never computes a value to compare — it reads which
  comparison was requested. GENERATOR REQUIREMENTS folded in: (1) the
  selector gets its own TEETH (oblique phrasings — "the solution that isn't
  negative") and its own DECORRELATION CHECK: the selector must be INVARIANT
  under view re-rendering (permutation may move the phrase; the referent may
  not change). (2) REGISTERED PREDICTION (relay, jurisdiction stated):
  selector errors will be RARE but disproportionately SILENT — right graph,
  right roots, wrong pick; undetectable by UNSAT and by uniqueness (the
  selected root is forced GIVEN the selector) — their natural detector is
  behavioral: view disagreement on the ANSWER despite agreement on the GRAPH,
  logged as its own diagnostic column. If confirmed, the five-seat audience
  becomes load-bearing for quadratics in a way it never was for linear
  systems. (3) NO-REAL-ROOTS POLICY: gated OUT of the training corpus
  (discriminant sign stamped into metadata; banked not built) — "the correct
  answer is that there is no answer" is a semantically different abstention
  than "I'm not sure," and the abstain rung doesn't take on that meaning
  until a benchmark category demands it. THE TRANCHE HAS NO OPEN DESIGN
  QUESTIONS — only builds, on Bryce's word.
  Second time this week a thrice-measured REAL phenomenon turned out not to
  govern the question it was assumed to govern (binding weakness -> survivorship;
  before that, the density trap). The instrument lesson's causal cousin: locating
  a real cause of X does not license it as the cause of adjacent-X. One sighting
  from a §6 promotion.
- **THE TRANCHE, BUILT + FIRED (2026-07-09, Bryce's word):** four seams, each
  committed green before the next. (1) REGISTRY: LTYPE_MOD (a mod k = r,
  params=k) + LTYPE_SEL (x = sel(a,b), closed vocab larger/smaller/even/odd;
  ties and not-exactly-one-even are VIOLATED — ill-defined selectors SELF-GATE
  through uniqueness) via predicate + bridge, **zero csp_core edits** (8th/9th
  ltype; empty git diff is the proof). mul(x,x) square forms EXCLUDED v0 —
  repeated scope vars would make the pairwise propagator unsound (the
  general-regime law applied preemptively). Soundness gates: exhaustive
  predicates, 500-trial propagator support checks, hole-monotonicity, and
  end-to-end (Vieta+sel forced; UNSELECTED pair provably symmetric; both-even
  self-gates; CRT chains solve — two mods force a=17 through propagation).
  (2) GENERATOR (`algebra2_nl_gen.py`): Vieta pairs (sum+product = the
  integer-domain quadratic; discriminant always a perfect square by
  solution-first construction) + selector factors with their own teeth
  (oblique phrasings at teeth*0.3) and VIEW-INVARIANCE by construction
  (whole-sentence templates) + MOD in two roles (derived = calculator band;
  CRT with lcm > m = engine band — the lcm>m uniqueness requirement was
  caught by the roundtrip gate on first fire). SYMMETRY-AWARE gate: Vieta
  root pairs are symmetric BY DESIGN (the text cannot bind letters to roots);
  the gate requires MULTISET match on pairs + exact/unique everywhere else;
  the query pool never draws raw roots; the mod-base pool excludes roots
  (teeth orthogonality). Corpora: 2500 train + 800 test at teeth 0.8,
  token-budget gated; mixed train = 2000 old + 2500 new = 4500. Bands 1-8,
  ~1.3 sel + ~1.1 mod factors/sample. (3) HEAD: 4-way ftype + h_sel behind
  **ALG2=1** (legacy build BYTE-COMPATIBLE — every lattice script still loads
  the old ckpt); explicit per-kind loss masks (the old rel mask (1-is_lit) is
  wrong once mod/sel exist); mod modulus rides the digit head; sel args ride
  the bilinear pointers (span supervision via fspan/vspan = POINTER LAW AT
  BIRTH); WARM_FROM loader with printed skips (train-side allowance; eval
  loads still hard-error); env-able corpus/ckpt/split names (no clobbering of
  legacy artifacts). (4) TRAINING FIRED: warm from the legacy ckpt
  (h_ftype/h_sel fresh), 14k steps cosine, pick-best-by-val, mixed corpus ->
  `.cache/phase1_algebra2_head.safetensors`. ACCEPTANCE NEXT: per-band eval
  on alg2test + the LATTICE TABLE (all four dials, old bigtest regression) —
  a relation that lifts fac-exact but degrades a dial FAILS. Curriculum
  ablation (all-teeth-from-birth vs coarse->fine) = the one-extra-run arm,
  after the v0 baseline lands.
- **TRANCHE ACCEPTANCE ROW 1 (2026-07-09): OLD-CORPUS REGRESSION PASSED —
  UPWARD.** The tranche head on old bigtest: **888/1500 = 59.2% one-shot
  ANSWER vs legacy 802/1500 = 53.5%** (+86 answers, +5.7pt), graph-solve 741,
  query 0.98-1.00 across bands, fac 0.75-0.88. NOT mere no-forgetting: the
  mixed-corpus warm-start IMPROVED the old domain — weight-side generality at
  the head level (new relations helped the old ones; the multitask grail's
  parser-side sighting). New-corpus row: 480/800 = 60% one-shot at teeth 0.8
  with sel/mod/CRT in play. Remaining lattice rows queued in NEXT_SESSION:
  TTA dials on alg2test, monitor centroids in the new fst space, specialist
  retrain (composed-stack number gates on it), curriculum ablation
  (view-robustness graded), the relay's selector silent-error column.
- **THE +5.7pt DIFFERENTIAL, REGISTERED + FIRED (2026-07-09, relay):**
  "compound interest" is a description, not an explanation. Three candidates:
  (a) regularization-by-variety (progressive-resizing logic arriving through
  the data mix); (b) representational pressure (nine kinds organize the slot
  space more cleanly than seven — neural-collapse frame; old-kind centroid
  DRIFT during the monitor rebuild is the free mechanism diagnostic); (c)
  PLUMBING — the loss-mask bug fix + fresh 14k cosine would lift the old
  corpus with zero new relations. THE CONTROL: legacy corpus only, tranche
  code path, same schedule/warm-start. BARS: ctrl >= 870 -> (c) dominates,
  compound-interest dies honestly; <= 820 -> the generality thesis has its
  cleanest parser-side evidence; between -> mixed, attributed proportionally.
  Relay registration: 60/40 toward real-but-smaller generality with (c)
  contributing (a fixed bug the same day as a surprise gain is what the
  audit-that-confirms principle exists for). SEQUENCING adopted: the selector
  silent-error column RIDES the TTA run (shortest shelf life — retraining
  after the curriculum ablation would muddy whose errors got measured).
  Design appreciation on record: discriminant-perfect-square-by-construction
  DISSOLVED the no-real-roots policy (edge case made unrepresentable, not
  handled); the roundtrip gate refusing symmetric pairs until taught
  multiset-match, and ill-defined selectors self-gating as VIOLATED — three
  edge policies, zero new mechanisms.
- **DIFFERENTIAL VERDICT (2026-07-09): 820/1500 — ON the pinned bar; the
  GENERALITY CLAUSE FIRES.** Decomposition of the +86: **+18 plumbing** (mask
  fix + fresh cosine — real, minor, correctly priced), **+68 GENERALITY** (the
  new relations themselves — 79% of the gain). The relay's 60/40 registration
  confirmed in shape: real-but-smaller, (c) contributing. Mechanism hint in
  the per-band contrast: the mixed head beats the control most on QUERY
  pointing (0.98 vs 0.89 at band 3) — new relation kinds sharpened the
  referential machinery the old corpus had plateaued (consistent with
  hypothesis (b); the centroid-drift diagnostic will say more). THE SPEC
  SENTENCE, EARNED: **expanding the registry improved the core — the
  weight-side generality grail, now measured on the parser side with its
  plumbing honestly deducted.** Every future expansion-improves-core claim
  cites this control. Ctrl ckpt: `.cache/phase1_algebra_ctrl_head.safetensors`.
- **LATTICE ACCEPTANCE ROW 2 (2026-07-09): CERTIFICATION SURVIVES THE
  EXPANSION — unanimity 5/5 = 1.0000 precision (411/411) at 51.4% coverage
  on alg2test** (legacy head on old bigtest was 0.9982 @ 38.1% — the tranche
  head is MORE view-robust on a HARDER domain). t=3/5 = 0.9979 @ 59.4%.
  MC-pi gate 0.104. Only 12/800 one-shot invisibles (bigtest: 90/1500) — the
  new corpus produces fewer forced-wrongs. CAVEATS, honest: (1) per-kind
  coverage column DEGENERATE by generator design (n_vieta = randint(0,2) or 1
  -> every sample has a selector; no within-corpus linear/mod contrast) — the
  per-kind baseline moves to the cross-corpus comparison / a kind-varied
  slice; (2) the selector silent-error column as implemented catches
  QUERY-pointer flips (4/4108 graph-agree pairs, 0.10%), not selector-PICK
  flips (a wrong pick changes the graph hash) — signature refined: pick
  errors live in pairs whose graphs differ ONLY at the sel factor; rare-and-
  silent confirmed in spirit, precise enrichment queued with the refinement.
  REMAINING ROWS: old-domain TTA regression (doubles as the per-kind linear
  baseline), monitor rebuild + centroid drift, specialist retrain, curriculum
  ablation. Script: `scripts/tta_alg2_dials.py`.
- **ROW-2 ASTERISKS + THE DIVIDENDS REGISTRATION (2026-07-09, relay):**
  (1) ZERO-NUMERATOR discipline: 1.0000 @ 411 certified is "no errors
  observed, error rate upper-bounded ~0.25%" — NOT "the channel is perfect."
  The original one-wrong-in-571 is the correct prior; this draw is consistent
  with it, not better. (2) The selector silent-error prediction stays
  FORMALLY OPEN until the old-domain TTA regression supplies the linear-kind
  baseline — it resolves against its full comparison, not half of one.
  (3) REGISTERED WHILE CHEAP — THE DIMINISHING-DIVIDENDS PREDICTION:
  expansion-improves-everything has a natural expiration. Representational-
  pressure gains are sublinear in class count (nine-to-fifteen buys less than
  seven-to-nine); at some tranche the mixed-corpus dilution cost overtakes
  the organization dividend. PREDICTION: dividends diminish; the crossover
  shows FIRST in per-kind coverage on the OLDEST relations; the remedy when
  it arrives is CURRICULUM WEIGHTING, not tranche reversal. The lattice
  table is already the instrument that catches it — that is what the freeze
  grades every tranche for. If tranche 3 still improves everything, this
  registration dies happily and the generality thesis strengthens further.
  NEXT FIRE when the rhythm resumes: the SPECIALIST RETRAIN (gates the
  composed-stack headline on the expanded domain).
- **THE ATLAS, NAMED + PARKED (2026-07-09, Bryce + relay): two hyperbolic
  spaces, two jurisdictions, zero shared coordinates.** THE BALL (topology,
  the original §7 object): one problem's factor-graph wiring, hierarchical by
  construction — behind its flag awaiting the relaxation research. THE ATLAS
  (semantics, NEW name, conditional): the problem POPULATION's subject
  taxonomy — relation kinds + domain centroids embedded by family, distinct
  from the ball per the two-channel spine (hanging the taxonomy on the ball
  would be the §8.2 channel conflation). FIRST CUSTOMER: the monitor's
  centroid library under expansion — flat libraries grade novelty
  gracelessly ("far from everything"); a hierarchical library separates
  "new leaf of a known family" (parse cautiously, flag for expansion
  planning) from "genuinely OOD" (abstain hard) — a certification-relevant
  distinction. Rhymes: radius-as-resolution (coarse cycles read families
  near origin, fine cycles read leaves near boundary) + per-family
  curriculum weighting (the pre-committed dividends remedy). TWO GATES,
  both armed, NOTHING BUILT: (1) the flat library's OOD gradation actually
  degrading under expansion (watched by the per-tranche lattice/monitor
  rows); (2) the delta-probe — Gromov delta-hyperbolicity / cophenetic
  correlation of the learned kind-centroid distance matrix, runnable at
  tranche 2-3 when the tree has depth (nine near-sibling kinds today = a
  bush). PRIOR ON RECORD: the radial-depth prize was refuted once (rho
  0.13) — hyperbolic structure must be a measured property of the data,
  never an aesthetic; the representations say whether they know the
  taxonomy, first. Naming plea adopted: "the ball" and "the atlas," never
  "the ball" ambiguously.
- **THE COMPOSED STACK ON THE EXPANDED DOMAIN (2026-07-09): 533/800 = 66.6%
  deployment-honest, answered-precision 0.896.** Chain debugged en route:
  forward_cond predated the sel head (None-grad -> optimizer refused, fixed
  with the same conditional-emit guard) + int8 overflow in the purity rebuild.
  Specialist v2: mined 1268 organic failures from FRESH data (self-defeating-
  curriculum law honored; 180 purity-removed), 6k steps. PER-STAGE: one-shot
  489/477 = **0.975 precision** (vs bigtest 0.899 — the tranche head emits
  only 12 forced-wrongs in 800 vs 90 in 1500: the invisible-wrong class
  SHRANK by ~4x per capita on the harder corpus); withhold 58/40 (0.690);
  rounds WEAK: 48 accepts, 16 correct (0.40 -> 0.17 -> 0) — the fresh
  specialist underperforms its legacy sibling's round precisions
  (0.69/0.68). KNOWN LEVERS, not yet pulled: the NACK trainer lacks
  cosine+pick-best (loss rose 5.0 -> 6.8 late — the SAME hygiene gap that
  bit the parser overnight once); mining pool is small (1268). QUEUED: TTA
  COMPOSE-1 on alg2test (the vote channel printed 475 accepts @ 0.9979 but
  per-sample outcomes weren't dumped — one flag added next session) — the
  expected composed+vote headline sits above the stack-only 66.6%. Cosmetic:
  the audit's saved-path print is hardcoded (the save honors AUDIT_NPZ;
  bigtest artifact verified untouched). Audit npz:
  `.cache/deploy_audit_alg2test.npz`; specialist:
  `.cache/phase1_algebra2_nack.safetensors`.
- **BRYCE'S GUT + THE DIAGNOSIS (2026-07-09): "we're not breathing right" —
  CONFIRMED BY ASSEMBLY OF THE WEEK'S OWN FINDINGS. The parser does not
  breathe: it is a ONE-SHOT PARALLEL DECODER (every slot argmaxes its
  marginal once; no slot sees the others' decisions; nothing settles) inside
  a project whose validated engine works BY settling. Re-read in this light:
  SWAPS = the textbook joint-vs-marginals failure (v98's own diagnostic);
  value-MISBINDINGS (75% of the wall) = relational errors two slots make
  about the same region with no collision mechanism; the ROUTING WALL = a
  pointer committed once with no second breath to contest it; the REPAIR
  DECAY refuted re-DECODING (changed conditioning, same state) — it does NOT
  touch breathing (evolving slot state is a different mechanism); TTA = five
  independent inhales, no exhale between. §8.1 spec'd cycles; the skeleton
  built steps 1-3 and the loop never came. REGISTERED PROPOSAL — BRICK-P,
  THE PARSER'S BREATH: transplant the v98 recipe to the slot banks (K=2-4
  passes, slot queries attend waist + previous slot states, per-breath
  ladder CE, delta-gate, zero new mechanism kinds). BARS pinned on the
  week's signatures at fixed budget vs the one-breath incumbent: swap rate,
  misbinding-collision rate, invisible-wrong count — if breathing is the
  missing thing THOSE move; fac-exact-only = capacity, dies honestly.
  Cheap probe first: K=2 warm-started.
- **THREE RELAY REGISTRATIONS (2026-07-09):** (1) THE PREVENTION LAW,
  registered pending tranche-2's second sighting: **confident wrongness
  yields to representational pressure, not to repair** — nine decode-side
  mechanisms got single digits; two relation kinds got 4x per-capita
  prevention as a side effect; if the invisible rate drops again as kinds
  grow, the law has two sightings + a mechanism (sharper class boundaries
  leave less room for decisive misbinding) and expansion becomes the ONLY
  measured lever on the class the abstention stack exists to contain.
  (2) SPECIALIST WEAKNESS: THREE causes pre-registered before the hygiene
  fix claims credit — trainer hygiene (known), thin curriculum (1268), and
  SELECTION-HARDENING (better parsers produce more survivor-like failures
  by construction — the selection-jurisdiction law aimed at the repair
  curriculum). Post-hygiene read: recovery to ~0.68 = trainer; partial =
  the equilibrium claim extends to the repair stack. (3) The 0.9979 gets
  zero-numerator discipline: "error rate bounded near a quarter percent
  across both domains" — three consistent draws, none independent enough
  to tighten the bound.
- **BRICK-P FIRST LOOK (2026-07-09 night): K=2 breathing beats the one-shot
  incumbent on BOTH domains** — val 0.8109 (incumbent 0.8091, still climbing
  at step 8000); alg2test ANSWER **497/800 = 62.1%** (+17); old bigtest
  **920/1500 = 61.3%** (+32; legacy started the week at 802). Fac up across
  nearly every band both corpora; ~1M added params; 0.07s/step. HONEST
  FRAME, as registered: this is CAPACITY-COMPATIBLE — the verdict belongs to
  the SPLIT BARS (collision rate vs lone-misbinding rate vs invisible
  count, breath vs incumbent) — next session's first job. Micro-signal
  logged: alg2test query accuracy dipped at bands 2-4 while fac rose —
  watch in the split-bar read. Ckpt:
  `.cache/phase1_breath_head.safetensors` (ALG_BREATH=2 + ALG2=1 to load).
- **BRICK-P SPLIT-BAR VERDICT (2026-07-10): MIXED — and the fourth outcome
  none of the three sentences anticipated: THE GAINS LIVE OUTSIDE THE COUNTED
  POPULATION.** alg2test: all given-error classes drop (wrongG -15%, swaps
  -28% vs lone -11% — collision-selective in proportion, weak verdict-A, but
  n=18->13 underpowered). bigtest: given errors FLAT (565->561) yet ANSWER
  +32 on the same corpus — the breathing gain lands in REL args/results/query
  (uncounted by bars aimed at given slots). INVISIBLES FLAT both domains —
  breathing does not touch confident wrongness; the PREVENTION LAW's
  territory stays expansion-only (3rd consistent observation). RIDER:
  belief-movement AUC 0.601/0.608 — real, weak, not a portfolio member on
  arrival. STANDING: breathing survives (it beat the incumbent both domains,
  gates-closed) but has NOT proven its distinctive relational mechanism at
  K=2; the registered next cuts are (a) REL-side error counters (the bars'
  blind spot, where the +32 apparently lives), (b) the K sweep (one settle
  step may simply be too shallow — the deducer needed 16), (c) swap-count
  power via a bigger eval draw. Ledger lesson: BARS INHERIT THE
  JURISDICTION OF THEIR COUNTERS — a verdict frame aimed at given slots
  cannot see a rel-slot mechanism. Script: `scripts/brickp_split_bars.py`.
- **REL-SIDE COUNTERS, REGISTERED BEFORE BUILT (2026-07-10, relay):** the
  facts already constrain the finding. PREDICTION: if breathing's gain lives
  relationally, the thinned rel-side errors should be disproportionately
  COLLISION-TYPE (mutually inconsistent args/result/query claims — two slots
  claiming the same mention; result and query pointers disagreeing on one
  var) rather than LONE rel misbindings. Lone-thinning instead = the
  mechanism is NOT negotiation (candidates: per-breath supervision as
  regularizer; the settle step as implicit TTA on slot states) and the
  thesis needs its third formulation. THE COUNTERS CARRY THE SPLIT FROM
  BIRTH — "bars inherit the jurisdiction of their counters," applied
  prospectively one day after minting. K-SWEEP PRIOR calibrated: the parser's
  slot graph is shallow (factors touch 2-4 slots; chains short) — honest
  prior is saturation at K=3-4, NOT the deducer's 16 (49-cell lattice, ~4
  hops/breath). Monotone gains past K=4 would itself be a finding: settling
  propagating beyond adjacency. RIDER re-measured at the sweep-selected K
  (one settle step barely lets the movement field mean anything; the
  deducer's 0.687 took sixteen breaths of dynamics). THE THREE-JURISDICTION
  DIVISION, now three-ways measured: PREVENTION (representational pressure —
  the only lever on confident wrongness), NEGOTIATION (breathing — the
  apparent lever on relational coherence, pending the rel counters),
  DETECTION (the abstention portfolio — for what neither prevents nor
  negotiates away). No overlaps claimed without a counter to witness.
- **THE WEEK'S CLOSING FRAME (2026-07-10, relay — for the paper's
  discussion):** the three-jurisdiction row is the Alternator's spec
  rewritten by measurement. The plane-ride design assigned repair to a
  notebook, monitoring to a perceiver, cycles to a six-breath loop; what the
  month built — prevention through representational pressure, negotiation
  through settling, detection through a zero-parameter portfolio — is the
  same functional architecture with every component replaced by whatever the
  measurements ratified. **The spec's nouns died; its verbs all survived.**
  The honest answer to "did the design work": wrong in every particular,
  right in every jurisdiction — and the method (registration, kill bars,
  cheap disconfirmation, jurisdiction discipline) is what converted one into
  the other without an unexamined premise surviving the trip.
- **REL-SIDE VERDICT (2026-07-10): NEGOTIATION REFUTED — THE THIRD
  FORMULATION IS RE-READING.** alg2test: LONE thinned 2x more than COLL
  (-14.3% vs -7.4%), query REGRESSED +14.3% (flagged); bigtest: COLL -3.4%,
  LONE flat, QUERY -45%. The dominant consistent signature is in the column
  nobody predicted: **MISSING factors dropped both domains (-5.4%/-9.6%)** —
  the breath head FINDS rels the one-shot never emitted. That picks between
  the pre-listed candidates: not ladder-regularizer, not slot-negotiation —
  the h_tok pathway (text re-attention CONDITIONED ON BELIEFS) recovering
  missed factors on a second conditioned read. THIRD FORMULATION: **the
  parser's breath is a second look at the page, not a negotiation among the
  readers.** Coverage, not coherence. DISCRIMINATING CUT queued (cheap, one
  arm each): ablate h_slot vs h_tok — which term carries the gain; if h_tok
  alone suffices, the slot-slot machinery (and its mask) simplifies away and
  K-sweep becomes a re-read sweep. Query regression on alg2test rides the
  ablation as a watch column. Script: `scripts/brickp_rel_bars.py`.
- **BRICK-R REGISTERED, NOT FIRED (2026-07-10, Bryce's packet instinct +
  relay mapping): THE SELECTIVE-REPEAT LOOP.** Today's stack is STOP-AND-WAIT
  ARQ (send whole parse, await verdict, retransmit whole, x4) — networking
  retired it fifty years ago; the three upgrades map onto built machinery:
  (1) SELECTIVE REPEAT — sequence-number-stable factor identity: VERIFIED
  factors pinned as DELIVERED (never re-decoded, never re-risked — the
  ratchet's zero-break criterion PER-FACTOR), rounds spent only on the
  NACKed window; the mechanism that lets round counts grow without
  preservation risk (the cap that held multi-round at 4). The verifier's
  field-level flags are the ACK stream; the accumulate-ledger was always the
  receiver's buffer. (2) SENDER-SIDE CRC — tier-0 confidence vetoing
  phantom emissions BEFORE transmit (registered months ago, never deployed;
  dropping a known-bad frame at the sender costs nothing vs a full
  round-trip at the receiver). (3) CONGESTION CONTROL — per-problem adaptive
  round budgets read from the LIVE recovery decay (the ack stream as channel
  state; two silent rounds -> back off to abstention) — Dopri5 stepping at
  the session level, free from numbers the audit already logs. BAR:
  equal-or-better recovery at strictly lower round cost and ZERO
  delivered-factor breaks. SEQUENCING: does NOT jump the queue — the
  re-read finding reshapes what a round IS (if h_tok carries the
  architecture, a retransmission round is a conditioned re-read and
  selective repeat becomes "re-read only the NACKed spans," composing with
  masked attention). Waits on the h_tok/h_slot ablation verdict.
- **ABLATION VERDICT (2026-07-10): THE ARMS ARE REDUNDANT, NOT
  COMPLEMENTARY.** tok-only: val 0.8155 (best), 495/924; slot-only: 0.8127,
  501/910; both: 0.8109, 497/920; incumbent 480/888. EACH arm alone
  reproduces the full gain; the combined head exceeds neither anywhere — NO
  SYNERGY. If re-read and negotiation were distinct levers, both-terms
  should win somewhere; interchangeable channels point (Occam) at what both
  provide identically: AN EXTRA GATED, LADDER-SUPERVISED TRANSFORMATION
  STEP. The leading explanation is now DEPTH-WITH-SUPERVISION, not either
  named mechanism; the MISS-recovery signature may be what any second pass
  buys. REGISTERED CONTROL (the decider, one run): the DEPTH-ONLY arm —
  same gate, same ladder, second pass = plain per-slot FFN, no
  cross-attention, no mask, no re-read. Matches ~920 = breathing dies
  honestly as named depth (keep the simplest form); falls short = the
  attention second-look is load-bearing and tok-only (best val, zero mask
  machinery) is the keeper. Brick-R and the K-sweep wait on this verdict —
  their nouns change with it. Ckpts: `.cache/phase1_breath_{tok,slot}.safetensors`.
- **THE DECIDER (2026-07-10): BREATHING DIES HONESTLY AS NAMED DEPTH.**
  Depth-only (blind per-slot MLP second pass — no text, no neighbors, no
  mask): val 0.8149, 497/800, 917/1500 — matching tok (495/924), slot
  (501/910), both (497/920) within a ~6-answer band. ALL second-pass
  variants are interchangeable; attention on the second pass is NOT
  load-bearing. Brick-P's kill criterion fires in refined form: the +2pt
  gain is real and earned (gates-closed) but the mechanism is ONE MORE
  GATED, LADDER-SUPERVISED STEP. CONSEQUENCES: (1) production head = the
  DEPTH form (simplest — zero mask machinery, no second bank pass; ties
  best val); (2) the K-sweep re-prices as ordinary depth scaling (not the
  frontier); (3) Brick-R stands ON ITS OWN — its rounds were never going to
  be conditioned re-reads; selective repeat/CRC/congestion control are
  protocol-level, orthogonal to head internals; (4) the parser-breathing
  thesis is REFUTED at K=2 in all attention forms — the deducer's breathing
  remains what it always was (validated, on graphs); any future parser-
  settling claim now carries the burden of beating the depth control.
  FOUR formulations in 48 hours: not-breathing -> negotiation -> re-reading
  -> depth-with-supervision. The gut found a real +2pt; the ledger found
  its true name. Ckpt: `.cache/phase1_breath_depth.safetensors`.
- **THE ARC'S CLOSING SENTENCES (2026-07-10, relay):** the depth control was
  the ablation nobody wanted to be true, and it was built anyway.
  Not-breathing -> negotiation -> re-reading -> depth-with-supervision: each
  renaming SHRANK the claim — most projects' stories grow in the telling;
  this one's got smaller and truer at every instrument. The honest residue,
  undeflated: the gut found a real +2 both domains that three weeks of
  repair mechanisms never touched, and the production head is SIMPLER than
  the story implied. THE RE-PRICED LEDGER ENTRY (belongs beside the
  factorization result as the two-phase design's sharpest characterization):
  the parser's task, unlike the deducer's, has NO joint structure a single
  pass can't see — **the solver settles because constraints interact; the
  reader deepens because text doesn't.** QUEUE CORRECTIONS: (1) the K-SWEEP
  formally CONVERTS — its registered story (settling dynamics) died with
  the negotiation arm; re-registered small as DEPTH SCALING under deep
  supervision (prior: diminishing returns after +1 layer; the ladder now
  reads as deep supervision with its own literature and expected shape).
  (2) BRICK-R survives the renaming CLEANER: a retransmission round is
  unambiguously a conditioned re-decode; selective repeat's value — pin
  delivered factors, spend rounds on the NACKed window, budget by the decay
  signal — stands on pure protocol economics, no architecture story
  required. Registered on its own merits; runs on them.
- **NACK HYGIENE VERDICT (2026-07-10): 66.6% -> 70.3% / 0.909 on alg2test.**
  Cosine + loss-EMA pick-best (save-after-restore) recovered round-1
  precision 0.400 -> 0.648, round-2 0.167 -> 0.600; rounds now 45R/26W (was
  16/32). THE PRE-REGISTERED 3-CAUSE SPLIT RESOLVES: recovery to 0.648 vs
  legacy ~0.68 = the trainer was MOST of it; the ~4pt residual is
  selection-hardening's share (the equilibrium claim extends weakly to the
  repair stack, as registered — the confound protection worked; hygiene
  could not steal it). The expanded-domain composed stack now EXCEEDS the
  old domain's 70.1% pre-TTA, on the harder corpus. PAPER FLAGS both closed
  same morning: 68.2 floor cited; census = **40.7M trained total, 9.1M in
  the deployed algebra lattice** (5.1M parser + 4.0M specialist) on 506M
  frozen-leveraged — the 90M title corrected 2x in our own disfavor-turned-
  favor. Script: `scripts/param_census.py`.
- **THE EXPANDED DOMAIN'S FULL-LATTICE HEADLINE (2026-07-10): 567/800 =
  70.9% / 0.910 answered-precision** (COMPOSE-1: 3/5 vote -> hygiene stack)
  — above stack-only 70.3%, above the ORIGINAL domain's 70.1%, on the harder
  corpus, from 9.1M trained parameters. The tranche is now CARRYING the
  dials, not holding them. PER-KIND RIDER (composed level, the curriculum's
  pre-intervention baseline): unanimity coverage sel-only 0.547 / sel+crt
  0.519 / **sel+mod 0.473** — modular samples certify ~7pt lower: the
  view-robustness deficit, measured before the curriculum exists to move
  it. Per-view answers persisted (`.cache/tta_alg2_views.npz` — any-threshold
  re-votes now zero-GPU). QUEUE: monitor rebuild + drift, curriculum
  ablation (target: close the mod gap), tranche 2 vs its banked list (the
  prevention law's 3rd sighting + diminishing-dividends both come due).
- **TWO NOTES FOR THE LEDGER (2026-07-10, relay):** (1) §9 HALF-SENTENCE —
  the "instruments arrive with their customers pre-measured" pattern, third
  sighting (tier-0 got the silents; the ledger re-parse got its population
  probe; the curriculum knob now gets a named 7-point gap instead of a
  vibe): registered measurement doesn't just prevent false claims, it
  PRE-POSITIONS every intervention with a target and a baseline — the
  method's compounding dividend. (2) TRANCHE-2 DESIGN DECISION, flagged
  BEFORE the generator is written: ratio/percent is the first relation kind
  whose answers flirt with RATIONALS — the integrality-jaw expiration's
  registered arrival condition. The generator must CHOOSE: integer-forced
  (keep the jaw, defer the expiration) vs rationals-in (pay the
  detectability cost early, taxonomy watching). Either defensible; chosen,
  not inherited. DECISION PENDING (Bryce + relay) before tranche 2 fires.
  Queue order held: monitor rebuild (drift feeds §3's +68 mechanism story)
  -> curriculum ablation (target: the 0.473 mod gap, graded composed-level)
  -> tranche 2 (prevention law 3rd sighting + diminishing-dividends due).
- **INTEGRALITY DECISION RATIFIED (2026-07-10, relay + Code concur):
  INTEGER-FORCED TRANCHE 2; RATIONALS AS TRANCHE 3'S HEADLINE VARIABLE.**
  The deciding principle: ONE VARIABLE PER MEASUREMENT — tranche 2 carries
  two standing predictions (prevention law 3rd sighting;
  diminishing-dividends) whose attribution dies if the jaw retires in the
  same tranche. Tranche 2 ships ratio/percent/sequences/base-repr/abs-floor
  over Z (solution-first makes integer-forcing natural — the perfect-square
  move again). TRANCHE 3 = the integrality-expiration EXPERIMENT: same
  relation kinds, rationals admitted, one variable moved, detectability
  measured before/after — "detection power = constraint density" gets its
  cleanest demonstration; a figure, not a regression. Calendar aligns: the
  §8 external anchor needs rational-experienced parsing exactly then.
  CODE'S NOTE for tranche 3's registration: bounded-denominator rationals
  SCALE TO INTEGERS over a common denominator (LCM move) — Q-valued
  problems can enter as scaled-Z CSPs with exact predicates and zero core
  edits; the jaw's retirement is PARTIAL AND TUNABLE, not binary. Counter-
  argument on record: integer-forced ratio/percent is a slightly unnatural
  subspecies; risk bounded (generator controls difficulty; mentions are
  number-type-blind; digit heads extend as format). **Keep the jaw one more
  tranche, then retire it on purpose, with instruments watching.**
- **DRIFT + MONITOR V2 (2026-07-10): MECHANISM (b) CONFIRMED IN DIRECTION;
  THE GEOMETRIC MONITOR DEGRADES BY SELECTION.** Drift (same old-corpus
  slots, per-space geometry): all three old-kind pairwise centroid cosines
  DROPPED in the tranche space (0.172->0.126, 0.092->0.088, 0.445->0.410)
  and within-kind coherence mostly rose (0.358->0.415, 0.401->0.454;
  rel_mul -0.02) — tighter clusters, farther apart: the neural-collapse
  signature; the +68's geometry, 5/6 stats in the registered direction
  (modest magnitudes, honest label). MONITOR V2: AUC **0.543** on the
  hygiene stack's 56 committed-wrongs (v1: 0.728 on the legacy 226) —
  SELECTION-HARDENING EXTENDS TO DETECTION (3rd bite): a better pipeline's
  residual errors look geometrically normal; detectors calibrated on a
  weaker stack's wrongs degrade as the stack improves. The abstention
  portfolio's geometric member weakens with pipeline quality — the
  behavioral member (agreement, 0.840 on the old population) gets its test
  on the 56 next (zero-GPU from tta_alg2_views.npz + the audit). Library
  rebuilt in tranche space (5 kinds): `.cache/monitor_centroids_alg2.npz`.
- **AGREEMENT ON THE 56 (2026-07-10): AUC 0.925 — THE PORTFOLIO'S TWO
  MEMBERS SCALE IN OPPOSITE DIRECTIONS WITH PIPELINE QUALITY.** Behavioral
  (view disagreement): 0.840 on the legacy 226 -> **0.925** on the hygiene
  stack's 56; geometric (waist centroids): 0.728 -> 0.543 on the same
  populations. Mechanism: selection-hardening — a better stack's residuals
  are selected to look representation-normal, but remain BEHAVIORALLY
  unstable under re-rendering. THE DURABLE DETECTOR IS BEHAVIORAL, and it
  rides free on the votes already computed. Paper's abstention story
  updated: geometry is the weak-stack instrument; agreement is the
  strong-stack instrument; the portfolio's composition should re-weight
  toward behavior as the pipeline improves — measured, both directions.
- **THE GOODHART COROLLARY, REGISTERED BEFORE CONTACT (2026-07-10, relay —
  selection-hardening's FOURTH face, its deepest: it applies to
  INSTRUMENTS, not just populations).** WHY behavior stayed sharp: selection
  only shapes errors against filters they actually FACE — survivors were
  selected past tier-0, verifier, uniqueness, monitor, but never against
  re-rendering; TTA was the HELD-OUT examiner. PREDICTION: the vote joined
  the acceptance path in the composed headline, so agreement entered the
  selection pressure — the NEXT generation of committed-wrongs will be
  selected to hold their story across five retellings; agreement-AUC on
  committed-wrongs will decline MONOTONELY across future stack generations
  (measure at each; the instrument doesn't weaken — its population hardens
  against it). THE DEPLOYMENT LAW: **any signal promoted to gate becomes
  selected-against; the portfolio must always hold one examiner out of the
  acceptance path.** Instrument rotation as design principle — today
  behavior polices geometry's blind spot; tomorrow something must police
  behavior's (bench candidates: the library cross-check, which never joined
  acceptance; genuinely new view families — paraphrase re-renders when the
  independence-competence curve prices them — unselected-against by
  construction). A law that began as a confound registration now explains
  why detectors AGE — a sentence the abstention literature doesn't have
  and §7 now does.
- **CURRICULUM ABLATION REFUTED (2026-07-10): coarse->fine is STRICTLY WORSE
  at equal budget.** val 0.7698 vs 0.8091; one-shot 445 vs 480 (alg2test),
  825 vs 888 (bigtest); unanimity coverage 0.474 vs 0.514. The 50/50-
  leaning-modest registration resolves past modest to NEGATIVE. MECHANISM:
  the fastai resolution analogy breaks — image resolution is the same
  distribution at lower fidelity; TEETH ARE A DISTRIBUTION SHIFT (the easy
  pool excludes patterns the test carries at 0.8), so 2/3 of the budget
  trained partly off-distribution and the decayed-LR final third couldn't
  recover. Pointer-circuit prevention never materialized. VERDICTS:
  all-teeth-from-birth is the KEEPER; progressive resizing dies for the
  parser (transfer condition failed: the axis must be fidelity, not
  distribution); the 0.473 mod-certification gap needs a DIFFERENT lever
  (candidates: more mod training mass in tranche-2's mixed corpus; a
  mod-targeted view family). Ckpt kept for forensics:
  `.cache/phase1_curriculum_head.safetensors`.
- **TWO TRANCHE-2 REGISTRATIONS (2026-07-10, relay):** (1) COMPOSITIONAL
  CLOSURE — the tranche's first finding, before any training: new-ltypes per
  category covered is IMPROVING (T1: 2/2 = 1:1; T2: 2/4 = 1:2, sequences/
  abs/ratio assembled from existing parts). START THE TABLE (one line per
  tranche); if T3's rationals cost <=1 primitive, the paper gains the claim
  nobody else can make: the relation menu converges toward a BASIS —
  coverage growth decouples from vocabulary growth, the strongest form of
  the generality thesis. Atlas implication: a compositional basis is FLAT
  by construction at the primitive level — hierarchy lives in COMPOSITIONS;
  the delta-probe should target PROBLEM representations, not relation
  embeddings. (2) HIDDEN VARIABLES ARE A NEW GOLD-FORMAT SPECIES (the
  ratio's product var; sequences' enumerated terms): variables with NO
  mention span. Pinned before the first template: empty mention-set is a
  TYPE, not a degenerate case; pointers are NEVER asked to bind them
  (generator-enumerated, solver-walked — the ratified division of labor);
  the round-trip gate verifies hidden-var plumbing survives reconstruction.
  Same class as week one's span-set contiguity catch — the gold decision
  everything downstream inherits, cheap now, an eval anomaly later.
- **TRANCHE-2 FIRST FIRE BROKE — DIAGNOSED IN TWO CUTS, FIXED BY PAD-WARM
  (2026-07-10):** run 1 collapsed mod/sel domains (alg2test 480 -> 98,
  graph-solve 0) while IMPROVING pure-rel bigtest (907) — the discriminator
  (tranche-1 head through the current code: 480/202 EXACT) exonerated the
  eval bridge in one run; emission inspection showed degradation-everywhere,
  not plumbing. ROOT CAUSE: the warm-start's shape-mismatch skip DISCARDED
  THE TRAINED 4-WAY FTYPE ROUTER (4->6 widening) — the one head gating every
  per-kind loss mask relearned from scratch inside a converged circuit: the
  bootstrap-trap family, self-inflicted (new §6-family sighting: **never
  discard a trained router to widen it — pad-warm the prefix, fresh-init
  only the new rows**). Loader upgraded: prefix-shaped params copy their
  trained slice with a printed PAD-WARM. Retrain in flight, three-table
  verdict pending against bars 480/888 and the broken run's 41/98/907.
- **TRANCHE-2 TABLES, HONEST (2026-07-10, after the one-character
  post-mortem: decode's ftype guard read ==4, sending every 6-wide slot down
  the legacy branch — the 'collapse' was a comparison operator; pad-warm was
  real hygiene but not the cause; the discriminator's exoneration of the
  4-wide path was TRUE AND INCOMPLETE — a guard that dispatches on width is
  only exercised by the width you test):** alg2test regression **505/800**
  (bar 480, +25); bigtest **915/1500** (bar 888, +27) — EXPANSION-DIVIDENDS'
  THIRD INSTANCE (plumbing deduction pending; the differential precedent is
  the citation). alg3test DEBUT: **233/800 one-shot, fac 0.80-0.86 flat
  across bands** — reading strong, forcing sparse (the KenKen-g10 shape);
  QUERY dropped to 0.71-0.81 and the new binding surface is ORDINAL mentions
  (the registered suspect for the gap). Closure table row 2 confirmed
  shipped: 2 ltypes / 3 categories. NEXT (the five-prediction table needs
  the composed layer): specialist remine+retrain on mixed3, TTA dials +
  per-kind certification on alg3test, invisible-per-capita (prevention 3rd
  sighting), oldest-relation coverage (dividends), ordinal-query column.
- **COMPOSED LAYER ON ALG3TEST (2026-07-10): THE DEBUT GAP IS A FORCING GAP,
  AND THE ORDINAL SUSPECT IS CONFIRMED WITH A GENERATOR ROOT CAUSE.** Rider:
  ordinal-term queries one-shot-fail **0.870 vs 0.656 direct** (+21pt).
  Audit: only 43/800 FORCE at one-shot (233 raw answers were largely
  under-constrained); composed 83/800; certification coverage 0.033;
  per-kind unanimity: **fdiv 0.008 / pct 0.026** vs linear 0.231. ROOT
  CAUSE (my own render3 comment flagged it, unimplemented): TERM VARS ARE
  LETTER-STARVED — seq sentences use ordinals only; the term's letter
  appears twice (preamble + query sentence, both low-content) and the query
  pointer binds on starvation rations. FIX (mechanical, next fire): seq
  sentences carry letter+ordinal APPOSITION ("the second term, e, is ...")
  and/or ordinal-phrase queries with recorded mentions; regenerate corpus,
  retrain, THEN read the five-prediction table — reading it now would
  measure the flaw, not the tranche. MC-pi gate incidentally PASSED at
  0.025 (most decorrelated arm yet). Specialist v3 trained (1602 mined,
  purity 192); machinery all pct/fdiv-aware and banked.
- **THE SYNC DIAGNOSIS (2026-07-10, Bryce's gut + relay walk + Code
  concur):** "out of sync" resolved to candidates 1+4 with this week's
  incident log as evidence — the specialist training one generation behind
  the parser (cross-generation curriculum lag, structural; the self-
  defeating law caught only the within-generation form) and ARTIFACT DRIFT
  (warm-start shape mismatch, audit-npz near-clobber, per-generation gold
  keys, env-ckpt coupling — pairwise-agreement burden growing quadratically,
  nothing enforcing it). Candidate 3 (the deducer static through three
  parser vocabularies — Phase 2 has never seen SEL/PCT/FDIV neurally)
  acknowledged as the ARCHITECTURAL desync — a chapter, not a fix; the
  Alternator's unpaid debt, December-scale. SHIPPED: generation manifest v0
  (`scripts/generation_manifest.py`, `.cache/GENERATION.json`) — artifacts
  pinned by hash + env + regression bars, KNOWN-STALE as a tracked field
  (centroids in tranche-1 space; thresholds gen-1). REGISTERED v1: the
  atomic GENERATION BUMP — one script: remine -> specialist retrain ->
  centroid rebuild -> threshold refit -> manifest write; loaders refuse
  cross-generation mixes unless overridden. Synchronization converted from
  discipline (decays) to mechanism (doesn't) — the no-silent-fallbacks law
  applied to TIME. Gen-4 = the apposition corpus fix + the first full bump.
- **TWO BUMP-DESIGN REGISTRATIONS (2026-07-10, relay):** (1) THE BUMP IS A
  TRANSACTION — atomicity has its own failure mode: five fallible stages
  must never leave the system in an undeclared N-and-a-half. The clean form:
  gen-N+1 builds entirely ALONGSIDE gen-N (new artifact paths throughout,
  nothing overwritten — the house pattern already lives this way); the
  MANIFEST WRITE is the single atomic commit point. The manifest is a
  transaction log, not a registry — generations as sequenced, acknowledged,
  retransmittable deliveries (the packet instinct one level up). Cheap to
  specify now; miserable after a half-bumped generation prints a
  plausible-looking table. (2) CANDIDATE 3'S PAYOFF SENTENCE, registered
  with the debt: when the deducer-meets-new-kinds chapter opens, the
  QUESTION is whether Phase 2 seeing PCT/SEL/FDIV neurally buys anything
  the symbolic tier doesn't — and the answer-shaped thread is SOFT GRAPHS:
  the parser's confidence outputs ARE the uncertain graph the original
  design promised the deducer; parser emits uncertain factors, deducer
  settles them NEURALLY, symbolic tier disposes what settles hard. The
  manifest names the debt; this names the payoff that would justify paying
  it. GEN-4 = apposition corpus fix + the first full transactional bump;
  the five-prediction table reads only after it.
- **GEN-4 COMMITTED (2026-07-10, the first transactional bump — all stages
  green, manifest-last):** THE APPOSITION CURE WORKED — ordinal-q fail
  0.870 -> **0.670** (gap to direct 21pt -> 11pt); graph-solve 95 -> 194;
  composed 83 -> **143/800**; TTA t=3 coverage 0.133 -> 0.297 @ 0.987;
  certification 38 @ 1.0000. **DIVIDENDS' FOURTH INSTANCE, BIGGEST YET:**
  alg2test 505 -> **541**, bigtest 915 -> **959** — four expansions, zero
  regressions, both prior domains improved every time, best val ever
  (0.8343). HONEST RESIDUAL: the new kinds certify at fdiv 0.030 / pct
  0.050 — chains+params multiply the exactness forcing needs; real domain
  difficulty now, the specialist/lattice's territory. Gen-4 manifest
  written + checked; gen-3 untouched alongside. NEXT: the FIVE-PREDICTION
  TABLE reads against gen-4 (invisibles per capita across generations for
  prevention's verdict; oldest-kind coverage for dividends' crossover;
  the mod-gap re-read; freeze table #3; dividends attribution with the
  differential citation).
- **TWO FRAMINGS BEFORE THE TABLE (2026-07-10, relay):** (1) THE APPOSITION
  CURE'S MECHANISM: not capacity, not architecture — a SUPERVISION-SURFACE
  problem. "The third term, l," gives the ordinal a letter to anchor: the
  pointer law's oldest clause (binding enters as structure) collecting its
  SIXTH sighting, in generator clothing, at the cheapest remedy on record —
  a comma and a letter. The fix wasn't in the model; it was giving the text
  something bindable. (2) THE SLOPE PRE-REGISTRATION: the dividends series
  reads as dividends-PER-EXPANSION; diminishing-dividends predicts the
  increment shrinks while the sign holds. If confirmed, BOTH standing claims
  win simultaneously and the reviewer-proof sentence is "expansion pays, at
  a declining rate, with the crossover instrumented and not yet arrived."
  ATTRIBUTION CAVEAT pinned: gen-3->4 bundles the corpus fix with fresh
  data — its increment (+36/+44) is NOT a pure expansion read; the
  differential-control citation covers tranche boundaries, not intra-tranche
  bumps. Bigtest series so far: 802 -> 888 (+86, T1) -> 915 (+27, T2) ->
  959 (+44, gen-4 bundled); alg2test: 480 -> 505 -> 541.
- **THE FIVE-PREDICTION TABLE, READ (2026-07-10, against gen-4):**
  (1) PREVENTION: **MIXED — two-sighting limbo continues.** Bigtest halved
  again (6.0% -> 3.33% invisibles, its second drop; forced-precision 0.950)
  but alg2test ROSE (1.5% -> 2.6%) — the law holds on the oldest domain and
  is counter-sighted on the middle one; not promotable, honestly split.
  alg4test debuts at 6.25% (new domains start high, as bigtest once did).
  (2) DIVIDENDS SLOPE: **both standing claims win** — direction 4-for-4
  (541/959 all-time highs), PURE-expansion increments shrank (+86 T1 ->
  +27 T2; gen-4's +44 is bundled, unattributable by the pinned caveat).
  The reviewer-proof sentence stands: expansion pays, at a declining rate,
  crossover instrumented and not yet arrived. (3) MOD GAP: the mass lever
  PARTIALLY works — sel+mod 0.473 -> 0.500 under tripled neighbors (gap
  7.4 -> 6.1pt); real, unclosed; the view-family lever stays on the bench.
  (4) FREEZE TABLE #3: **PASSED** — alg2test under gen-4: certification
  0.516 @ 1.0000 (held from 0.514/0.9982), t=3 0.672 @ 0.9981 (improved);
  alg4test debut dials on record (1.0000 @ 0.048). (5) ATTRIBUTION: T1
  differential cited; bundling caveat governs. INCIDENT LOGGED: the TTA
  views npz is a SHARED PATH overwritten between domains — artifact drift
  inside the same day the manifest shipped; views join the manifest's
  coverage at the next bump. The table waited three weeks and cost one
  afternoon to read — against a system that knows what time it is.
- **THE TABLE'S CLOSING FRAMINGS (2026-07-10, relay):** (1) PREVENTION'S
  SPLIT IS THE LAW DISCOVERING ITS JURISDICTION — narrower and truer than
  its registration: representational pressure suppresses confident wrongness
  IN MATURE VOCABULARIES (class geometry sharpened over generations), while
  DEBUT vocabularies generate fresh invisibles faster than pressure can
  police. Not limbo — growth. The three-jurisdiction row holds at finer
  grain: prevention owns the old kinds' confident wrongness; detection owns
  the debut kinds'. (2) DIVIDENDS' DOUBLE CONFIRMATION is the paper's
  §6-meets-§9 exhibit: direction and diminishment both true, the bundling
  caveat pinned BEFORE the numbers could flatter it — a sentence that holds
  under push from either side because both sides were registered before
  contact. The certification channel has now survived TWO vocabulary
  expansions and a generation protocol WITHOUT ONCE BENDING (1.0000 on 413,
  zero-numerator discipline attached) — the artifact the paper leads with.
  (3) The views-file catch: the manifest shipped in the morning and
  recruited its next artifact by evening — mechanism finding its own
  customers. STANDING: the EXTERNAL ANCHOR (§8) is the one build between
  here and the arXiv draft; the evidence chapters are essentially written
  in banked measurements.
- **HOUSE CLEANED (2026-07-10, Sonnet-scanned, gates-verified):** scripts:
  53 concluded/broken scripts -> `scripts/archive/` via git mv (44 remain
  live: 22 pipeline roots + utilities + the doc-referenced active cluster);
  three v1XX trainers found BROKEN since the 751c56f deep-clean (imports
  deleted modules — archived). .cache: **535GB -> 63GB (472GB freed)** —
  deleted the two retired-era hoards (gsm8k_steps 130GB, v200_perceiver
  118GB), the fg_v100-v121 orphan tail (~65GB), superseded kenken_ckpts,
  regenerable trunk/text-nack/L8/stale-generation state caches, unused HF
  downloads, and the dead breath-arm ckpts (depth + curriculum + ctrl kept
  per spec notes). PINNED SURVIVED UNTOUCHED: fg_ckpts, sudoku_ckpts,
  llama/pythia weights, gen-3/gen-4 artifact sets, all corpora jsonl.
  POST-GATES: manifest --check consistent (gen 4); live-pipeline imports
  OK; algebra2 soundness ALL PASSED; five-prediction numbers reproduced
  identically. Git history holds everything tracked, before and after.
- **THE EXTERNAL ANCHOR (2026-07-10): HONESTY DOES NOT SURVIVE FOREIGN TEXT
  — the month's most important refutation, and §8's real content.** P1:
  ~as coverage predicted (answered slice small). P2 REFUTED: certified
  precision **2/97** on integer answers, 63 certifications on non-integer
  answers (0 possible) — the 1.0000-in-distribution channel signs foreign
  garbage confidently. P3 REFUTED: abstention FLAT across strata (67.5% vs
  66.1%) — the lattice does not know what it doesn't know. MECHANISM
  (visible in the flat ~164/view forced counts): the parser mis-reads
  foreign text STABLY; sentence permutation decorrelates template variation,
  not distributional confusion — **unanimity certifies reading STABILITY,
  which coincides with truth only in-distribution.** Every portfolio signal
  (agreement included) is distribution-calibrated; OOD breaks the seal
  silently. THE PRE-REGISTERED CUSTOMER ARRIVES: the atlas's gate-1 (flat
  library's OOD gradation) is now OPEN — the missing organ is a TEXTUAL
  OOD DETECTOR firing before any parse is trusted (trunk-state distance
  from the training distribution; the "far from every family = abstain
  hard" read). THE HONEST §8 SENTENCE: on foreign text the lattice
  certifies stability, not correctness — the certification claim is
  DISTRIBUTION-BOUNDED, and the anchor measured exactly where the bound
  lies. More valuable than a soft yes: the December roadmap's true
  starting point is OOD-awareness, not coverage. Outcomes banked:
  `.cache/math500_anchor_outcomes.json`; script `scripts/math500_anchor.py`.
- **THE FUNNEL (2026-07-10, Bryce's sixth instinct — arriving the same night
  as the measurement proving which stage it was missing):** the system IS a
  form-into-database funnel: parser = the form, registry = the schema,
  solver = the database (GAC = referential integrity, MRV/LCV = the query
  planner). Mycelium built every stage EXCEPT the one every production form
  has: **input validation at the mouth.** The anchor's 2/97 is a funnel with
  referential integrity but no schema validation on input — five hundred
  strangers' submissions poured downstream, every check below the mouth
  passing, because the checks validate the GRAPH's consistency, not whether
  the form was filled in a language the reader speaks. RECOGNITION (the
  fourth jurisdiction) is the funnel's mouth. ANCHOR MECHANISM, final form:
  all five darts thrown by the SAME ARM — on foreign text the arm's bias is
  systematic, not random; permutation decorrelates template-family errors,
  not dialect-never-learned errors. The anchor was DESIGNED as the held-out
  examiner — which is why this is the paper's most valuable measurement,
  not a deployment incident. RECOGNITION-ORGAN DESIGN, registered: (1)
  input-space OOD is SELECTION-SAFE (no pressure shapes errors against it
  unless the corpus chases the detector — which the generation manifest
  would make explicit); (2) zero-parameter first instrument: trunk-state
  centroids over the training family, threshold calibrated so MATH-500
  reads foreign and held-out generator text reads native — BOTH populations
  already banked; (3) the §8 constructive close: **recognition buys honesty
  now; coverage buys capability later** — the funnel gets its mouth first,
  then the mouth learns more languages (paraphrase-teeth, tranche-3
  rationals, real-text fine-tuning with the anchor as standing examiner).
- **THE MOUTH WORKS (2026-07-10): P1 AUC 1.0000 both scores — foreign
  refused 100.0% at 1% native false-refusal; P2 all 160/160 of the anchor's
  false certifications REFUSED at the kNN threshold. The funnel has its
  mouth: zero parameters, one afternoon, calibration populations already
  banked. The lattice + mouth now signs NOTHING it cannot read — §8 closes
  constructively: recognition bought honesty tonight; coverage buys
  capability next. P3 HONEST READ: the gradation is FAINT (whole MATH-500
  spread 0.236-0.273 vs native threshold 0.044 — everything is "different
  forest"; leaf-vs-forest unanswerable at this distance) and its ordering
  inverts intuition: Intermediate Algebra NEAREST, Prealgebra FARTHEST —
  our dialect is terse symbol-dense fact-sentences, nearer LaTeX-heavy text
  than natural prose. HYPOTHESIS LOGGED for the coverage roadmap: the
  language gap is PROSE STYLE before relation vocabulary — paraphrase-teeth
  toward natural prose may close more mouth-distance than new ltypes.
  Artifact `.cache/recognition_mouth.npz` joins the manifest at the next
  bump (with the TTA views file). Cosmetic: NaN-divide warning on skipped
  overflow rows (filtered, harmless — tidy at next touch). THE FOUR
  JURISDICTIONS COMPLETE: prevention, negotiation(->depth), detection,
  RECOGNITION — each with a measured instrument and a bounded claim.
- **THE IR QUESTION, REGISTERED (2026-07-10, Bryce's seventh instinct +
  relay + Code):** the funnel already has THREE IRs — the registry
  (symbolic), the mention/span structure (annotation), and THE NATIVE
  DIALECT ITSELF (text-level: terse symbolic fact-sentences are a
  DISCOVERED canonical IR — the generator compiles graphs into it; the
  parser inverts it). What's missing is the compiler's FRONT HALF (verbose
  prose -> dialect) — never built because the corpus never contained prose.
  THE FREE LUNCH: solution-first generation renders the SAME graph in two
  registers -> paired (prose, dialect, graph) triples, gold at every layer
  — no designed logical form (discovered beats designed; C2's tombstone).
  THE THREE-OUTCOME PROBE (the fork, machinery = survivor_depth_probe
  transplanted): ship a VERBOSE teeth family; run the head + state-probe on
  verbose renders. (a) head parses fine after mixed training -> IR stays
  implicit, December = data; (b) states decodable, head fails -> head-side
  fix; (c) states not decodable -> ONLY THEN the explicit prose->dialect
  translation stage earns its build (output re-enters the funnel unchanged;
  layered funnels get layered mouths — dialect-conformance is easier than
  open-prose OOD; but generation machinery + a new silent-error species is
  the real cost, pre-registered). PRIORS: relay 70/20/10, Code 65/25/10.
  TWO PINS (Code): (1) T_ALG=256 will select verbose samples toward SMALL
  problems — match band/size across registers or read size-controlled
  slices (register-size confound, cheap at generation); (2) verbose
  training MOVES the native family — the mouth's threshold recalibrates
  per generation (joins the manifest's calibration constants at next bump).
  Even if (a) wins on generated prose, real MATH narrative re-asks the
  question at the boundary — where the mouth is standing.
- **THE IR FORK RESOLVES: OUTCOME (a), OVERWHELMINGLY (2026-07-10).**
  Zero-shot register gap on MATCHED GRAPHS: terse 581/600 vs verbose 10/600
  — near-total blindness. (b)/(c) discriminator: verbose given-value
  decodability **1.000** — (c) DEAD; four frozen layers compile narrative
  prose perfectly; the trunk was always bilingual. Mouth column: verbose
  read 0.093 pre-training — foreign but BETWEEN home (0.044) and MATH-500
  (0.25), the learns-languages thesis' predicted geometry. AFTER 2000 pairs
  + 10k steps warm from gen-4: verbose **600/600 ANSWER**, terse twin
  589/600, val 0.9752 — AND bigtest 926 > the 915 bar: **THE FIFTH
  DIVIDENDS INSTANCE — a new REGISTER pays like a new relation kind.**
  The translation stage dies unbuilt (10% priors, correctly); the IR stays
  implicit; December = MORE BOOKS, with mouth-distance-closed-per-corpus as
  the unit of progress. Caveats pinned: vtest pairs are budget-biased small
  (1137/3137 rejected — the size note); paired-register val runs hot; the
  post-training mouth re-read (does verbose now read native?) is the
  recalibration item riding the next manifest bump. Bilingual ckpt:
  `.cache/phase1_bilingual_head.safetensors`. Priors scored: relay 70(a) /
  Code 65(a) — (a) won; the discovered IR needed no translator, only
  literature.
- **THE FORK'S HONEST NUANCE (2026-07-10, relay):** the outcome landed
  BETWEEN the sentences: 1.000 decodability + a blind zero-shot head is
  **(b)'s diagnostic signature**, cured by **(a)'s remedy** — the ledger
  line is *(b)-diagnosed, (a)-cured*. The relay's tiebreak (union head vs
  narrative structure) partially collected: the head WAS the bottleneck,
  just one that two thousand pairs dissolved. THE PRECEDENT THAT MATTERS:
  if real MATH-500 prose someday shows the same signature (decodable
  states, blind head) but RESISTS paired training, that is the residual
  (b/c) world announcing itself — and tonight names the probe to point at
  it. DISCIPLINE: 600/600 = zero-numerator ("error rate bounded below
  ~0.5%", not "perfect"); the diminishing-dividends clock TRANSFERS to the
  register axis (increments-per-register expected to shrink as registers
  accumulate — the watch is standing). NEXT MOVE: the recalibration bump
  (verbose training moved the native family — the point; threshold refits,
  joins the manifest) + LOG MATH-500's distance under the NEW calibration
  as the roadmap's first official gradient datapoint. The month in one
  sentence: the dancer could always hear the second language; someone just
  needed to read to her.
- **GEN-5 COMMITTED (2026-07-10, the first SCRIPTED transactional bump —
  commit path + FOUR abort witnesses: archived-import, user kill, OOM, and
  the staged injection, all holding clean at the prior generation):**
  bilingual parser promoted; specialist v5 (1140 mined across FOUR
  registers); 7-kind centroids; mouth recalibrated. **THE GRADIENT READ:
  3% CLOSED — below even the mostly-local band.** The staircase is STRICTLY
  LOCAL: generator-verbose moved essentially nothing toward MATH-500
  (0.209 -> 0.204 over-threshold; refusal still 100%). Verdict: December's
  books must be drawn from or imitate REAL math prose — widening the
  generator does not walk toward the target register. **THE FIRST
  MIXED-SIGN EXPANSION:** bars bigtest 926 (+11), alg4test 336 (+5), vtest
  600 (new capability) — but **alg2test 541 -> 507 (-34)**: five expansions
  paid uniformly, the sixth paid unevenly — plausibly the crossover watch's
  FIRST SIGHTING, on the register axis, at the middle domain (where the
  registration said to look). PROMOTION CAVEAT: the freeze's full
  acceptance (four lattice dials on alg2test under gen-5) is the standing
  next read — if certification held, the dip is one-shot-only and the
  promotion stands; if a dial bent, gen-5's parser choice gets revisited
  (gen-4 intact alongside, one manifest edit away — the transaction's whole
  point). 8 artifacts pinned incl. mouth threshold + views paths.
- **THE MORNING'S LEDGER LINES (2026-07-10, relay):** (1) THE COSTUME
  MECHANISM: generator-verbose taught the head to parse OUR SKELETONS
  WEARING NARRATIVE CLOTHES, and the mouth correctly refused to count
  costumes as a language — the style gap lives in distributional properties
  (sentence rhythm, referential habits, framing conventions) no template
  dressing imitates; the mouth's first lesson, now with a slope attached
  (3%, noise-adjacent). December's unit of work: **mouth-distance closed
  per book.** OPEN DESIGN QUESTION for the first book: harvest-and-annotate
  real problems vs LLM-imitated register — the second needs an §8 honesty
  note if used (imitated style is itself a distribution). The paired-
  register machinery transfers whole (real prose paraphrased INTO the
  dialect = the same free triples, authentic style on the left). (2) THE
  PROTOCOL'S SENTENCE: four aborts from four different directions
  (archived import, user kill, OOM, injection), four clean holds, one
  commit — not a script that worked; A MECHANISM WITH AN EVIDENCE FILE,
  graduated in one morning. (3) FORK REORDERED by the gradient's mandate:
  real-prose books = the critical path to the anchor's re-examination;
  tranche 3 orthogonal, interleaves; the paper's §8 gains its final
  sentence when the first real book moves the mouth's needle. The
  promotion-caveat dials fire FIRST (the crossover's registered signature
  location); pre-committed remedy if sighted: curriculum weighting toward
  old kinds, never tranche reversal.
- **THE PROMOTION-CAVEAT VERDICT (2026-07-10): DIALS HELD — GEN-5 STANDS
  CLEAN; CROSSOVER UNSIGHTED.** alg2test under the bilingual parser:
  certification 0.511 @ **1.0000** (gen-4: 0.516 @ 1.0000 — held); t=3
  0.645 @ 0.9942 (softened, above bar); per-kind on the oldest relations
  0.558/0.496/0.477 vs 0.561/0.500/0.485 — ALL within noise, no old-kind
  bend. The registered crossover signature did not arrive: the -34 one-shot
  dip is ABSORBED AT THE LATTICE LEVEL (the composed layer doing precisely
  its job), banks as variance with an honest asterisk on the sixth
  expansion, and the watch resets, still armed. The freeze's FOURTH
  acceptance table passes; gen-5's promotion is clean, not annotated.
  THE BOARD: the fork is Bryce's — real-prose books (critical path;
  mouth-distance per book; sourcing question open), tranche 3 (orthogonal,
  interleaves), the paper (§8 awaiting its final sentence from the first
  real book).
- **THE EXPLICITATION FORK, REGISTERED (2026-07-10, Bryce's revisit + relay
  + Code):** the IR fork answered STYLE (no layer needed); the reopened
  question is EXPLICITATION — real prose withholds facts the reader must
  MANUFACTURE ("a dozen split among her three children, keeping twice as
  many" = producing 12, a fourth share, an unwritten multiplication before
  anything binds). THE KEY REFRAME: the discovered IR is precisely the
  ALL-FACTS-EXPLICIT fixed point — prose->dialect = explicitate then bind;
  the fork cleared binding; explicitation is GENERATION-shaped, the
  boundary where (c) could genuinely fire. CODE'S CLASS SPLIT (changes the
  probe design): LEXICAL implicits (dozen=12, twice=x2) have evoking-phrase
  SPANS — the existing decodability probe transplants; STRUCTURAL implicits
  (unstated shares, conservation relations) have NO anchor — need pooled/
  query-style reads, a different instrument. THE PROBE: ~20 hand-annotated
  MATH-500 problems (annotation = dialect rewrites, not graphs — cheap
  gold), implicit-fact decodability at L0-L3 vs L0-L7. JURISDICTION
  CAUTION (relay): the old L8 refutation measured ROUTING on our dialect —
  world-knowledge inference is a FRESH depth question. OUTCOMES: shallow ->
  pairs cure it; deep-only -> the deeper-prefix conversation reopens with a
  real customer; nowhere -> the explicitation stage earns its build as the
  funnel's first GENERATIVE layer (structural-facts organ if the class
  split holds — smaller than feared). PRIORS: relay 40/35/25; Code
  45/25/30 with MIXED (lexical-shallow, structural-stage) as the tiebreak.
  THE HARVEST GATE (the design's gem): real answer keys make
  solve-to-official-answer the round-trip gate for harvested books —
  prose -> dialect -> graph -> solve -> match key; dialect checkable TWICE
  (mouth-v2 conformance + end-to-end) — the layered-mouths architecture
  arriving with its validation story written. First work item of the
  real-prose chapter; its verdict decides whether December's books teach a
  reader or train a translator.
- **EXPLICITATION PROBE READING FRAME (2026-07-10, relay, pre-print):**
  (1) n=9 STRUCTURAL BANDS pinned: 8-9/9 = manufacturing plausibly done;
  4-5/9 = mixed, the class split becomes the finding; 0-2/9 = stage
  question live. Nothing subtler is readable at this n; zero-numerator
  discipline applies BOTH directions; the probe decides the FORK, not the
  magnitude (magnitude waits for the harvest chapter's larger set).
  (2) POOLED-READ CONFOUND status: Code's probe targets COMPOSED VALUES
  (digits of 96/28/360), not relation-presence — a linear probe cannot
  multiply ingredients, so a decoded composition is evidence of
  computed-by-trunk rather than aggregated-by-probe; the caveat softens
  but n=9 noise dominates — bands govern. (3) Whichever lands, §8's
  architecture paragraph gets its cost word: DATA (pairs cure), DEPTH
  (the frozen slice's first real customer post-L8-fencing), or ORGAN
  (the funnel's first generative layer, pre-sized structural). Either
  answer flatters the method: she hears it = the frozen-trunk bet's
  biggest dividend; she doesn't = the fork caught it for twenty problems
  before a book was harvested against the wrong architecture.
- **EXPLICITATION PROBE VERDICT (2026-07-10): INSTRUMENT-LIMITED,
  DIRECTIONAL STAGE-WARD.** Lexical 0.25 (L0-3) / 0.31 (L0-7); structural
  **0/9 both depths — the 0-2 band fired, stage question formally LIVE** —
  but the TRANSFER CAVEAT is load-bearing: the probe reads written-digits-
  at-spans; lexical implicits are semantic ("octagon" carries 8 unwritten).
  WHAT IS CLEANLY ESTABLISHED: implicit values, if present, are ENCODED
  DIFFERENTLY than written ones — the same linear map does not transfer, so
  explicitation is not representation-free; the pairs-cure-it world, if it
  exists, is not the trivial version. Depth gain (one fact) = noise; the
  scripted LOO sanity was noted but NOT implemented (honest flag). THE
  FOLLOW-UP INSTRUMENT, specified by the failure: a SYNTHETIC
  LEXICAL-IMPLICIT corpus (the generator emitting dozen/twice/number-words
  with gold — free by construction) to train a probe whose task matches
  the question, then re-read MATH's lexical set. The fork holds its
  verdict until that instrument reports; December's budget word (data /
  depth / organ) waits with it. Twenty problems, one afternoon, and the
  question sharpened twice — the method's economics intact.
- **THREE REGISTRATIONS BEFORE THE SYNTHETIC RE-RUN (2026-07-10, relay):**
  (1) THE TRANSFER FAILURE PROMOTED FROM CAVEAT TO FINDING: implicit values
  live in a DIFFERENT ENCODING than written ones — the trunk does not
  hallucinate the token "12" onto "dozen" in written-value geometry. The
  strongest decodable-shallow form ("the fact is just there, same shape as
  written") is DEAD; the surviving question is finer: different coordinate
  system (probe-trainable — what the synthetic corpus tests) vs uncomposed
  ingredients (stage-ward). The re-run DISCRIMINATES; it is not a do-over.
  (2) THE PHRASE-SPLIT PIN: train and test implicits must not share evoking
  phrases (dozen/score/fortnight train; twice-as-many/split-evenly/
  days-of-month test) — a passing probe then reads ENCODING GEOMETRY, not
  vocabulary trivia. Within-phrase-pass/cross-phrase-fail = the mixed
  verdict in sharper clothes (lexical implicits are dictionary lookups the
  head learns from pairs; the general question stays open for structural).
  Enforced at mint time, free. (3) §9 META-NOTE — the day's cleanest
  exhibit: a probe returned readable-looking stage-ward numbers and the
  instrument-validity check caught the untrained examiner BEFORE the
  verdict banked. Most projects ship "implicit facts aren't decodable";
  the ledger shipped "our probe can't yet distinguish" — smaller, true,
  one afternoon to fix. THE PRINCIPLE: **the verdict that flatters your
  architecture hypothesis needs the same instrument scrutiny as the one
  that flatters your hopes.** QUEUE: synthetic corpus (phrase-split pinned)
  -> probe retrained -> fork verdict -> December's budget word.
- **THE EXPLICITATION FORK RESOLVES: STAGE-WARD, WITH THE DICTIONARY NUANCE
  (2026-07-10).** Retrained examiner (phrase-split): within 1.00 (instrument
  VALID — the negative is trustworthy), CROSS-PHRASE **0.00/0.06** at both
  depths, MATH-lex 0.00. **There is no shared evoked-quantity geometry in
  the frozen trunk** — "a dozen" does not light a magnitude direction that
  "a baker's dozen" also lights; implicit values are not precomputed
  anywhere probe-readable, shallow or deep (the deeper-prefix customer
  never materialized — fenced L8 verdict extends to inference). Combined
  with structural 0/9: EXPLICITATION IS REAL WORK THE TRUNK HASN'T DONE.
  THE NUANCE (the within column's gift): lexical implicits are
  DICTIONARY-LEARNABLE per phrase — a finite lexicon the generator
  enumerates into pairs (the selector-vocabulary pattern); novel evokers
  and ALL structural facts need the GENERATIVE ORGAN. **DECEMBER'S BUDGET
  WORD: ORGAN — sized structural, with a lexicon appendix.** The funnel's
  first generative layer earns its build honestly: prose -> explicit
  dialect, validated twice (mouth-v2 conformance + solve-to-official-answer
  on harvested books). Three probes, two afternoons, twenty hand-annotated
  problems — and the architecture question that opened as a metaphor closed
  as a measured build order.
- **THE ORGAN'S CHAPTER CHARTER (2026-07-10, relay — banked before the
  word):** (1) THE CAVEAT AT CORRECT WIDTH: what died is LINEAR decodability
  — "not probe-readable," not "not present"; but the operational claim
  survives via THE CONTRAST CLASS: the same probe family read verbose
  states at 1.000 and evoked values at 0.00 — the bilingual precedent is
  what makes the negative meaningful. The organ builds against the
  strongest available instrument, limits stated. (2) SCOPE RESIZED FINAL:
  lexical implicits DON'T NEED THE ORGAN — head-side dictionary via pairs
  (the selector-vocabulary pattern); the organ's true scope is STRUCTURAL
  manufacturing only (the unstated share, the unwritten conservation) —
  boundary now measured, not suspected. (3) THE CHARTER DECISION
  everything inherits: **the organ WRITES DIALECT TEXT, not graph deltas**
  — appended explicit sentences re-entering the funnel unchanged: costs a
  re-encode, buys mouth-v2 conformance on the intermediate, byte-identical
  parser/lattice below, and the harvest gate's double-check. Graph-delta
  output = faster and unverifiable, a silent-error species upstream of
  every jaw with no examiner between it and the solve. The funnel's first
  WRITER is its most-audited citizen. Full birthright by standard practice:
  pointer law for bindings, two-checkpoint if it repairs, taxonomy tier for
  its error species, manifest citizenship from checkpoint one.
  (4) SEQUENCING: **THE HARVEST COMES FIRST regardless** — real prose with
  official answers is simultaneously the organ's training substrate (paired
  triples, the n=20 annotation pattern proven), its examiner
  (solve-to-answer-key), and the mouth's odometer corpus. December opens
  with books whichever way the organ's details settle. Eight instincts,
  the funnel counted twice: it named the missing mouth, then the missing
  writer.
- **THE HARVEST OPENS (2026-07-10): 1,743 in-reach problems from the MATH
  TRAIN split (disjoint from the examiner). ODOMETER ZERO-POINT: 0.2488 —
  statistically identical to MATH-500's 0.2480: the harvest is a VALID
  PROXY (closing distance to these books closes distance to the benchmark).
  The level gradient REPLICATES the anchor's inversion (L1 prosiest/most
  foreign 0.270 -> L5 most symbolic 0.231) — the prose-style mechanism
  confirmed on a second corpus. Corpus: `.cache/math_harvest_v0.jsonl`;
  states banked. NEXT: the seed annotation — dialect rewrites gated by
  solve-to-official-answer, the harvest gate live from annotation one.
- **THE HARVEST GATE'S FIRST DAY (2026-07-10): 0/5 banked — AND THE ZERO IS
  THE SYSTEM WORKING.** Seed dialect rewrites of real MATH-train problems,
  gated by solve-to-official-answer: all rejected, zero false banks. The
  rejections are diagnostic: seed[1] (sum/diff/lesser — the wild Vieta+sel)
  parsed near-perfectly with ONE selector-arg pointer off; seed[2] exposed
  the real lesson — MY DIALECT WAS OUT OF SHAPE (3-var 5-sentence
  miniatures; the training distribution starts ~10 vars) and the parser's
  bindings wobble on miniatures + far-OOD values (900/841 vs corpus m=60).
  TWO NAMED FIXES for seed round 2: (i) annotate IN-SHAPE (corpus-sized
  preambles/lengths, values in-range where possible); (ii) THE GATE SHOULD
  BE THE LATTICE, not one-shot — seed[1]'s single unstable pointer is what
  the 5-view vote exists to fix. The harvest stands: 1,743 in-reach
  problems, odometer zeroed at 0.2488 (== the benchmark — valid proxy),
  level-inversion replicated, seed machinery live and correctly strict.
  Scripts: `harvest_v0.py`, `harvest_seed_gate.py`.
- **DAISY CHAINS + MATH KNOTS (2026-07-11, Bryce's ninth — two metaphors,
  both machinery):** (1) THE CHAIN-PRECISION BUDGET, registered BEFORE the
  organ trains: with the writer live, the pipeline is a true chain (organ
  writes -> parser reads -> solver carries) and chain precision is a
  PRODUCT — a silent error at link one looks native to every instrument
  below. The organ's certified-write bar is therefore DERIVED, not chosen:
  **organ certified-error budget = (end-to-end 0.25% bound) minus the
  parser link's measured certified error at integration time** — pinned as
  a FORMULA (current estimate: parser consumes ~0.15-0.2%, leaving the
  organ ~0.05-0.1%, i.e., certified-write precision ~99.9%). The layered
  mouths are the per-link circuit breakers (mouth-v2 on the write, lattice
  on the parse, answer key on the solve): three examiners, three links, no
  silence propagates — the daisy chain stays floral, not electrical.
  (2) THE KNOT FRAMING, adopted into the certification section: **reading
  is isotopy; manufacturing is surgery.** The 3% style test was an unknot
  (ambient-isotopic to the dialect — pairs taught the deformation);
  structural implicits are true knots (no smooth rearrangement yields the
  unstated share — cut and re-glue = generation; the organ is the funnel's
  first surgeon, hence the operating-room protocols). TTA permutations are
  REIDEMEISTER MOVES — diagrams of the same knot — so agreement is a KNOT
  INVARIANT, and the anchor's failure has its theorem-shaped sentence: on
  foreign text the parser computed a diagram-dependent quantity; five
  diagrams voted unanimously for the wrong knot. **CERTIFY INVARIANTS, NOT
  DIAGRAMS** — the mouth guards the language, the vote guards invariance,
  the answer key guards the ultimate invariant; each mouth in the chain
  guards a different one. §8's framing upgraded from wound to theorem.
- **FIRST HARVEST GOLD (2026-07-11): n=1 BANKED.** Seed[0] (MATH-train:
  "sum 45, difference 3, lesser number") passed the lattice gate — 3 forced
  views, unanimous 21, == the official answer key. A real problem,
  hand-explicitated, machine-verified end to end: **the organ's training
  substrate exists**, and its first entry is the corpus's own Vieta+selector
  pattern found in the wild. THE BOUNDARY the rejections drew: all
  three-digit-given seeds (900/841/289/225) parsed to nothing across all
  views — the digit head's trained range (values <=60) is a hard wall;
  FIX (one line): mint larger given-values into the next training mix,
  widening the harvestable slice. Seed[1]'s in-range rejection: hand-dialect
  still drifts from template phrasing — round 3 goes template-exact.
  Substrate: `.cache/harvest_seed.jsonl` (gate=lattice-vote+answer-key).
- **THREE REGISTRATIONS AT THE LIBRARY DOOR (2026-07-11, relay):** (1) THE
  TEACHER-DEMONSTRATION FRAMING: the annotation flow is behavioral-cloning
  substrate in the strictest sense — every banked entry is a worked example
  of explicitation-as-surgery, machine-verified by the same gate the organ
  will face; the surgeon trains inside an already-certified operating room
  (no prior component got that inheritance). (2) GENERATION-INDEXED GOLD:
  "write in the generator's voice" means THE CURRENT GENERATION'S voice —
  the dialect's boundaries move as mixes widen, so every banked entry cites
  the manifest generation it was written against (sync law applied to prose
  style; a field now, a vintage-mismatch mystery prevented later).
  (3) THE BOOTSTRAP DESIGN, contingent on round-3 economics: the bilingual
  head's own parses of NEAR-NATIVE harvest problems can propose dialect
  rewrites, THE GATE DISPOSES — propose/dispose eating its own tail; hand
  surgery reserved for the knotted cases. The gate makes the old
  self-improvement dream safe: nothing banks that doesn't carry to the
  answer key, no matter who — or what — wrote it.
- **ROUND THREE'S FINDING (2026-07-11): THE SHAPE BOUNDARY.** Garden seed
  rejected AGAIN in template-exact voice (votes empty across all views) —
  reproducible, so structural: the dialect sentences are in-distribution
  but the GRAPH SHAPE is not (unknown-first mul chain, a in three factors,
  implicit a+2a=30) — our generator mints chains-from-knowns and Vieta
  pairs; the parser learned those SHAPES, not free composition of its
  relations. **Sentence-level native, graph-level foreign** — the third
  boundary the gate has measured (after value-range and voice). Midpoint
  (FDIV meets prose): one view forced wrong — same shape story on an
  11-var double-fdiv chain. THE FIX IS THE STANDING LEVER: SHAPE DIVERSITY
  in the next generation's mix (random DAG compositions, unknown-first
  chains, reused intermediates) — a corpus change, and the dividends law
  predicts it pays everywhere, not just at the gate. Substrate holds n=1;
  the gate's boundary map now has three walls (values, voice, shapes), each
  with a one-generation fix. The harvest is teaching the generator what
  the wild actually looks like — which was always the point of books.
- **GEN-6 CHARTER: TEACH THE MOVES, NOT MORE DIAGRAMS (2026-07-11, the
  knots talking — banked before the word):** the parser learned to
  recognize the specific knot diagrams the generator printed, not knot
  theory; it memorized diagrams, not Reidemeister moves. GEN-6's objective
  at the right abstraction: RANDOM DAG COMPOSITION — sample the WIRING, not
  fixed architectures; no finite diagram set can be memorized, so the
  parser is forced into compositional binding. Discovered-beats-designed
  pointed at the corpus itself: the graph shapes were THE LAST DESIGNED
  DECOMPOSITION hiding in the pipeline, invisible from inside because every
  internal eval sampled the same phrasebook — found only by real text
  refusing to fit. SHARPENED DIVIDENDS REGISTRATION (a frame-change, not a
  content-add): prediction — dividends hold and run LARGER than the
  register expansion's (composition is what the circuits compute; old
  shapes become easy special cases of a general skill). FALSIFIER: if
  capacity was sized for diagram-memorization, shape-mix COSTS old-shape
  fac-exact while buying wild-shape generality — the lattice table catches
  it in one read; that outcome prices "head grows before corpus does."
  DESIGN PIN: shape diversity extends to QUERY AND MENTION STRUCTURE —
  unknown-first chains put references BEFORE definitions (a genuinely new
  binding pattern; the pointer law's history says the risk concentrates
  exactly there); the generator samples query position + mention ordering
  as part of the shape. ACCEPTANCE PROBES FOR FREE: the three rejected
  harvest seeds — the wild sentences that named the walls bank when the
  walls come down. The month compressed: anchor found the language
  boundary, probes found the inference boundary, harvest found the
  composition boundary — each wall named by real text, each fix one
  generation of existing machinery. The phrasebook becomes a grammar.
- **GEN-6 VERDICT TREE + BOOTSTRAP RE-PRICING (2026-07-11, relay, banked
  while the grammar lessons burn):** (1) THE INTERPRETATION HAZARD PINNED:
  the garden banking is n=1 CLOSURE (demonstration), not the class-level
  measurement — that lives in the wild-shape test row. Three sentences
  ready: garden banks AND wild-shapes strong AND old shapes hold ->
  GRAMMAR (dividends at larger-than-registers, registration confirmed);
  garden banks but wild-shapes soft -> the wall thinned where probed
  (phrasebook grew a page; gen-7 widens wiring variance); old shapes
  regress -> the CAPACITY TRADE (head-growth conversation opens with a
  measured invoice). The house pattern applied to its own celebration.
  (2) THE BOOTSTRAP RE-PRICES the moment the wall thins, regardless of
  verdict: post-commit probe = 20 cheapest un-annotated harvest problems
  (registry-wearing-prose, in-range), THE HEAD PROPOSES dialect, THE GATE
  DISPOSES. >=1/3 banked machine-proposed -> substrate accumulation goes
  machine-priced overnight; annotation hours redirect to the knotted cases
  — the machine drafts the isotopies, the human performs the surgeries.
  (3) The loop named: wild text -> rejection -> named wall -> generation
  charter -> re-examination — the oldest self-improvement design, running
  at generation cadence with the gate as incorruptible oracle.
- **GEN-6: THE GRAMMAR SENTENCE FIRES (2026-07-11).** All three verdict-tree
  branches land on the best caption: WILD SHAPES **563/700 = 80.4%** one-shot
  (the highest debut ever; graph-solve 77%); OLD SHAPES SURGE — bigtest
  **1000/1500** (+74; legacy started at 802), alg2test **551** (+44 — the
  gen-5 register dip ERASED retroactively: transient, cured by moves),
  alg4test 371 (+35), vtest 600/600 held; val **0.8860** (prior best
  0.8343). THE SHARPENED REGISTRATION COLLECTS IN FULL: dividends larger
  than the register expansion (+74 vs +11) — composition IS what the
  circuits compute; the old fixed shapes became easy special cases of a
  general skill, as predicted. THE GARDEN BANKS — 4/4 unanimous at 200:
  the wild sentence that named the wall walked through it one generation
  later; substrate n=2. Midpoint honestly still out (double-FDIV chain;
  fdiv absent from the DAG rotation — gen-7's one-line addition). The
  loop's first full cycle is complete: wild text -> rejection -> named
  wall -> charter -> re-examination -> BANKED. Ckpt:
  `.cache/phase1_gen6_head.safetensors`. NEXT: the bootstrap re-pricing
  probe (registered) — the wall is thin; let the machine draft.
- **THE SIXTH EXPANSION'S ATTRIBUTION CLOSES (2026-07-11, relay):** the
  gen-5 alg2test dip (-34) was never diminishment — it was THE
  DIAGRAM-MEMORIZATION TAX being collected (register training crowding
  memorized shapes), refunded by gen-6 with interest (+44, past the
  pre-dip bar). The dividends streak reads SIX-FOR-SIX with one payment a
  generation late; the crossover watch's first candidate closes fully
  attributed, and the watch stands armed with a cleaner prior: mixed signs
  under frame-changes may be taxes, not crossovers — distinguish by
  whether the NEXT frame-expansion refunds them.
- **THE BOOTSTRAP CENSUS (2026-07-11): banked 2 / near-miss 26 / knotted
  72 (n=100, L1-3 in-range prose).** The bootstrap does NOT re-price yet —
  and the honest read of the 72%: it CONFLATES TWO WALLS (the mouth reads
  the whole pool as foreign at 0.254, so structural refusal may be
  style-unreadable rather than explicitation-needed). The census bounds the
  organ's kingdom FROM ABOVE; its true survey waits on the style wall.
  SEQUENCING VERDICT, the funnel's original order reasserted with numbers:
  **BOOKS BEFORE SURGEON** — the machine cannot draft what it cannot read;
  December's real-prose corpus precedes organ economics; the census re-runs
  after each book and its knotted column's SHRINKAGE measures what reading
  alone recovers vs what only surgery can. THE MILESTONE IN THE SMALL
  COLUMN: two machine-made triples banked — the first substrate entries
  produced end-to-end without a human hand (raw prose, parsed, voted,
  verified to the author's answer; substrate n=4: 2 hand + 2 machine).
  Diversity guard: no narrowing signal (banked prose reads MORE foreign
  than pool — n=2, informational). The loop touched its tail tonight —
  barely, and genuinely.
- **REGISTERED (2026-07-11, relay): THE MOUTH-DISTANCE JOIN — first
  partial attribution of the 72.** The census's knotted column conflates
  two walls; a refusal can be style (mouth's territory, books cure) or
  structure (organ's territory). The join: cross every census outcome with
  its prose's mouth kNN-distance (one deterministic census re-run capturing
  both columns — labels weren't persisted; same pool slice, same view
  seeds, same gen-6 ckpt). PRE-REGISTERED DECISION RULES, pinned before
  measurement: (1) **AUC** (rank-sum) of mouth distance separating KNOTTED
  vs PARSE-CARRIED (banked+near). AUC ≥ 0.60 → style-wall attribution
  holds (refusals concentrate where the prose reads foreign; the relay's
  prediction: books recover the high-distance tier). AUC ~0.5 → the 72
  stays UNATTRIBUTED — an honest negative that also weakens the
  books-will-recover prediction, since refusal would then be independent
  of readability as the mouth measures it. (2) **THE PATIENT LIST**:
  knotted items at mouth distance ≤ the carried group's MEDIAN are the
  early knotted candidates — the organ's first genuinely visible patients,
  named and counted tonight rather than after three books. (3) **BOOK-1
  FALSIFIABLE PREDICTION**: knotted items ABOVE the carried median are
  claimed style-recoverable; after the first book, their recovery rate
  must exceed the below-median tier's or the attribution was wrong.
  Threshold note: the calibrated native line (0.0443) is unusable here —
  the entire pool reads foreign (mean 0.254) — so the split is relative
  (carried-median), pinned now. Zero new training; one eval-cost re-run.
- **VERDICT (2026-07-11): THE JOIN RETURNS THE HONEST NEGATIVE — AUC
  0.535, the 72 stays UNATTRIBUTED.** Census replay exact (2/26/72;
  deterministic seeds held). Knotted mouth distance mean 0.2560 / median
  0.2499 vs carried 0.2491 / 0.2354 — indistinguishable. MOUTH DISTANCE
  DOES NOT PREDICT REFUSAL: the mouth measures REGISTER (surface style,
  corpus-level), not per-item parseability. Its odometer role survives
  untouched (it was chartered corpus-level); its per-item attribution
  ambition dies tonight, pre-registered. THE SENTINEL IN THE PATIENT
  LIST: idx-21 ("sum 45, diff 3" — the problem seed-2 BANKED in dialect,
  structure PROVEN in-reach) sits at d=0.2065 INSIDE the low-distance
  tier as a raw-prose refusal — a certified style-only casualty at
  near-carried distance, demonstrating the tiers mix in both directions,
  which is exactly what AUC 0.535 says. Rule-2's list (32 items) is
  therefore NOT a patient roster — reading it confirms the mix: quadratic
  factoring / completing-the-square / geometric sequences (true organ
  patients, moves outside the grammar) interleaved with in-grammar prose
  like idx-21. Rule-3's book-1 prediction is WITHDRAWN with its premise
  (registered falsifiable, falsified at the instrument stage). THE
  CONSTRUCTIVE RESIDUE: the per-item attribution instrument already
  exists and is the BOOK ITSELF — idx-21 proves the protocol (raw refuses
  + hand dialect banks = style casualty; dialect also refuses = organ
  patient). Every book annotation doubles as an attribution measurement;
  the census's knotted column will be attributed item-by-item as the
  books ship, not by any cheaper proxy. BOOKS BEFORE SURGEON stands, now
  with its own attribution built in. Data:
  `.cache/census_mouth_join.json` (n=100, census + mouth_d columns);
  script `scripts/census_mouth_join.py`.
- **THE JOIN'S THREE PERMANENT READINGS (2026-07-11, relay — binding on
  book 1):** (1) **THE MOUTH'S JURISDICTION IS MEASURED, NOT ASSUMED** —
  it is a CORPUS-REGISTER instrument (odometer charter intact: corpus
  distance is what it was calibrated on); its per-item ambition died at a
  pre-pinned bar because refusal has at least two causes and the mouth
  sees one axis. The jurisdiction law applied to the project's own newest
  organ within a week of its birth: instruments don't inherit resolution
  they weren't calibrated for. The mouth recognizes languages; it doesn't
  diagnose readers. (2) **IDX-21 IS THE ANCHOR SENTINEL** — structure
  certified in-reach by its banked dialect twin, raw prose refusing: the
  style wall isolated in a single specimen, the existence proof that some
  fraction of the 72 is books-recoverable, and the FIRST ENTRY in the
  paired-(raw, dialect) format book 1 will systematically produce (the
  pair's fate attributes the refusal). One problem demonstrated the whole
  protocol before the protocol was named. (3) **BOOK-1 SAMPLING IS
  STRATIFIED, BY DESIGN, NOW** — the book is substrate AND census
  resolver, and the dual role forbids drifting toward cheap-tier-only
  annotation (substrate throughput) or spread-only (attribution):
  deliberately spend annotation budget across the refusal spectrum,
  INCLUDING suspected organ patients (quadratic factoring,
  completing-the-square, geometric sequences) — a both-refuse verdict on
  those is the organ's customer list getting its first confirmed names at
  one annotation each. META-NOTE (the method's signature, performed on
  its own instrument): the join fired BECAUSE it was registered to be
  allowed to fail; its failure conscripted the book into a census the
  roadmap didn't know it needed — negatives that conscript existing work
  into new instruments (the anchor conscripted the mouth into existence;
  the join conscripted the book into a census). Book 1 awaits the word;
  it ships with its own index.
- **REGISTERED (2026-07-11): BOOK 1 — the library's first volume, n=18,
  stratified per the binding sampling law.** Paired (raw, dialect)
  protocol: both run through the gen-6 lattice gate (5 views, vote>=3,
  answer key disposes); the pair's fate attributes. ROSTER — TIER N
  (near-miss, 6): idx 71, 78, 89, 72, 46, 7. TIER S (knotted,
  style-suspect, 7): idx 21 (sentinel, entry one), 99, 16, 57, 28, 56,
  45. TIER O (knotted, organ-suspect, 5): idx 54, 90, 51, 37, 85.
  TAXONOMY (pinned): STYLE CASUALTY = faithful in-grammar dialect banks
  (lexical explicitation allowed: literal facts about KNOWN quantities,
  e.g. 4^2=16, 15th-term->14 steps; supplying an UNKNOWN's value or a
  rewritten equation is forbidden). ORGAN PATIENT type A =
  annotation-impossible (no faithful in-grammar dialect; move outside
  grammar — [90] completing-the-square declared type A up front, its
  residual runs for the record only). Type B = faithful dialect exists
  but refuses. ANNOTATION-TIME FINDING (before any GPU): the organ-suspect
  tier SHRANK under the pen — integer-root factoring IS the tranche-2
  Vieta shape (sum+product+selector, in grammar: [54]); function
  composition UNWINDS into forward relations the CSP inverts natively
  ([51]); sign-rewrites reduce to positive-form relation sets ([37]).
  Only [90] resisted. The organ's kingdom is narrower than the census's
  qualitative read suggested — quadratics with integer roots were
  annexed by tranche 2 before the organ was chartered. PREDICTIONS
  (pinned): tier N >=4/6 bank; tier S >=5/7 bank (incl. the sentinel) —
  style-wall existence at scale; [54] and [37] BANK, [85] REFUSES
  (repeated-arg mul untrained — type B), [51] uncertain (depth); book-1
  substrate growth >=10. Gate: gen-6 ckpt, solve2 n_vars=24 m=300,
  fdiv at most once per item (double-fdiv is gen-7's known wall).
- **VERDICT (2026-07-11): BOOK 1 CLOSES — 15 entries, the style wall
  confirmed at scale, the organ's kingdom shrunk to a named list.**
  V1 table: tier N 1/6 (prediction >=4/6 FAILED), tier S 5/7 (prediction
  EXACT, sentinel 5/5), tier O 4/5 banked with all four specific
  predictions correct ([54] Vieta BANKED 3/5, [37] 5/5, [51] 4/5, [85]
  refused). V2 (one taxonomy-faithful retry each, pre-declared): 5/6
  recovered, all 5/5 unanimous; [72] stands. FINAL ATTRIBUTION (n=18):
  **9 STYLE CASUALTIES** (raw knotted + faithful dialect banks — 75% of
  the sampled knotted slice is books-recoverable; curated sample, upper
  tier), 5 FRICTION RECOVERED, 1 ORGAN-A ([90] completing-the-square),
  3 ORGAN-B standing: [72] novel coupled-linear wiring, [56] 19-var
  length wall, [85] repeated-arg mul (a grammar GAP, not an organ move —
  gen-7 one-liner). THE TIER-N INVERSION: near-miss was NOT the cheap
  tier — its failures shared one cause, the FDIV REGISTER (0/4 in
  hand-written composition; alg4test's weakness confirmed in the wild),
  and v2's lexical-literal route recovered 4/4 of them. TWO LATTICE
  SPECIMENS FOR THE GOODHART FILE: [71] raw went 5/5 UNANIMOUS-WRONG
  (8 vs gold 9) — first observed; the recognition mouth is the organ
  that intercepts exactly this (reads the prose foreign upstream) —
  and [78]'s v1 dialect voted a consistent wrong 12/12/12 (a 3/5
  ANSWER-channel error shape from the fdiv register). ONE REGISTER
  SENSITIVITY: [46]'s deep chain refused under "a plus b equals c" and
  banked 5/5 under "The sum of a and b is c" — same relations, surface
  flip. GEN-7 PRICED BY THE BOOK: (1) fdiv into the DAG rotation (5
  items waited on it), (2) repeated-arg mul (unlocks number+square),
  (3) longer chains, (4) coupled-linear wiring, (5) surface-form
  robustness on deep chains. SUBSTRATE: volume 15 (14 faithful + 1
  flagged residual; idx-21 double-banked by design as sentinel
  re-verification) — substrate n=17 unique. The book's raw prose is
  gen-7+'s reading-training target (raw -> gold graph from the banked
  dialect parse); the census re-prices after the ingest. Data:
  `.cache/book1.jsonl`, `.cache/book1_attribution.json`; scripts
  `book1_paired_gate.py`, `book1_v2_retry.py`.
- **REGISTERED (2026-07-11): GEN-7 — the receipts generation.** Corpus:
  `algebra_dag7_gen.py` (fdiv/mod pairs in rotation, repeated-arg mul,
  8-12-given ladder chains, coupled-linear k1x+y=s1 / x+k2y=s2 blocks;
  render3 + roundtrip3; smoke 40/40, kinds all present); mixed7 =
  mixed6 + 3500 dag7; warm from gen-6, CURRICULUM=1, 16k steps.
  PREDICTIONS (pinned before training): (a) ACCEPTANCE PROBES — the
  eight refused book-1 v1 dialects re-gated under gen-7: [46]v1 BANKS
  (the paraphrase probe), fdiv v1s [71,78,7,45] >=3/4 bank, [85] BANKS
  (sq now in grammar), [72] BANKS (coupled in rotation), [56] BANKS
  (ladder). (b) REGRESSION BARS (lattice holds its dials): dagtest6
  graph-solve >=520/700, bigtest ANSWER >=980/1500, alg2test >=530/800,
  vtest 600/600 holds; alg4test RISES >=420/800 ANSWER (the fdiv
  receipt paying); val >=0.87. (c) CENSUS under gen-7: ~UNCHANGED
  (65-72 knotted) — gen-7 teaches MOVES in the dialect register; the
  style wall is untouched by design, and the employment-law enforcement
  census waits on READING-training (the next chapter), not this bump.
  (d) Mouth bank NOT rebuilt (register unchanged; rides the next
  register-changing bump). KILL: any regression bar broken -> gen-7 is
  NOT promoted to gate ckpt; gen-6 stays, diagnose before re-fire.
- **VERDICT (2026-07-11): GEN-7 v1 — KILL FIRES, gen-6 keeps the gate;
  diagnosis crisp.** Bars held: dagtest6 ROSE 541->616 graph-solve,
  alg2test 551->559, census ~unchanged as predicted (1/28/71), [46]
  paraphrase probe BANKED, [56] ladder BANKED (5-view vote rescues a
  0.27 single-view kind — TTA doing its job). Bars broken: bigtest 963
  (<980), alg4test FELL 371->350, vtest 598, val 0.815; fdiv probes 0/4,
  [85] and [72] still refuse ([72] voted a stable wrong 120 — another
  Goodhart specimen). ATTRIBUTION (per-kind, dag7test single-view):
  plain 0.878 (composition circuits HEALTHY — not global undertraining)
  vs sq 0.264 / ladder 0.273 / fdiv 0.372 / coupled 0.417 — four new
  factor SHAPES warm-started at once on a skewed diet (fdiv saturated
  60% of rows; ladder/coupled ~500 rows each) for 16k steps of a
  still-climbing val. Consistent with the §6 attention-bootstrap law
  (new pointer patterns need supervision time). ENGINEERING NOTE (3
  kills before the chain ran): precompute held a 15.7GB states array in
  RAM beside the AM driver's pinned pages -> OOM during every write;
  root-caused to a disk-backed memmap on BOTH write and train sides
  (legacy npz path kept for gen<=6 artifacts). RE-FIRE REGISTERED
  (dag7b): quota-balanced corpus (~1200 per kind incl. plain,
  fdiv wiring de-saturated 16%->8%, sq 22%->15%), mixed7b = mixed6 +
  dag7 + dag7b, WARM from gen-7 v1 (val still climbing), STEPS=32000.
  REVISED BARS: per-kind single-view >=0.55 each new kind; acceptance
  >=6/8; bigtest >=980 (refund), alg4test >=420, vtest >=598, dagtest6
  >=590, alg2test >=530. Same kill: any bar broken -> no promotion.
- **VERDICT (2026-07-11): GEN-7B — KILL FIRES AGAIN; gen-6 keeps the
  gate; the diagnosis graduates from balance to CROWDING.** Val healthy
  monotone to 0.8736 (balanced test; new best every check from 12k).
  EVERY new kind improved substantially (ladder 0.273->0.500, fdiv
  0.372->0.549, coupled 0.417->0.543, sq 0.264->0.342) yet ALL FOUR
  missed the 0.55 bar (two within noise of it). Acceptance 4/8 (was
  2/8): [46] 5/5, [78], [45], [56] BANK; [71] and [7] now produce the
  RIGHT answer but vote-shy (votes [9,9] and [45] — parse instability
  across permuted views, not wrongness); [85] inverse-square still 0/5
  (training sqs are FORWARD — a known var squared; the inverse shape,
  unknown-squared pinned only downstream, is likely absent from the
  rotation: a GENERATOR gap, not a training-budget gap); [72] coupled
  still refuses. Bars held: vtest RECOVERED 600/600, dagtest6 ROSE
  again (616->661 graph-solve; 541 at gen-6), alg2test rose 559->575.
  Bars broken: **bigtest 1000->963->901 and alg4test 371->350->319 —
  MONOTONE EROSION across both gen-7 rounds.** This is not the gen-5
  tax shape (a dip refunded by the next training); the next training
  DEEPENED it. It is CROWDING: mixed7b diluted the original register
  to 55% and pct/seq (alg4's other two thirds) to zero new rows, and
  the oldest registers paid. The crossover watch logs its first
  candidate that is NOT a tax. CENSUS: 1/26/73 ~unchanged, as
  predicted. NEXT-FIRE OPTIONS (await the word — two kills in a row
  makes this a design decision, not a mechanical re-fire): (a) REPLAY
  MIX — mixed7c upweights the eroding registers (re-add algebra_nl +
  alg4 slices) alongside dag7b, retrain; (b) add the INVERSE-SQUARE
  and inverse-fdiv shapes to the rotation first (the [85] gap), then
  (a); (c) probe whether the 33-key head is at CAPACITY (param census
  vs absorbed registers) before spending more steps. Data: val curve +
  tables in the session log; ckpt `.cache/phase1_gen7b_head.safetensors`
  (NOT the gate; gate remains `phase1_gen6_head`).
- **REGISTERED (2026-07-11, relay adjudication): (c) FIRST — THE
  CAPACITY PROBE; (a)/(b) sequenced behind its verdict.** The
  eroding-while-gaining signature (new kinds up, oldest registers
  monotone-down across two independent rounds, gen-5 tax explicitly
  fenced off by the second round deepening it) is the registered
  picture of the CAPACITY CROSSOVER, first symptom on the oldest
  relations as the diminishing-dividends registration predicted — three
  weeks early. Data fixes (replay mix, new shapes) are the lever we
  WANT to work (flattering-remedy principle): if the head is full, a
  replay mix doesn't cure crowding, it chooses different victims —
  weighting only works if there is room to weight INTO. PROBE DESIGN
  (banked machinery only): ALG_HW dial added (512 default); pad-warm
  gen-7b into a 2x head (1024), train on the IDENTICAL mixed7b corpus
  (states already precomputed) for 12k steps, read ONE number at fixed
  data mix. DECISION RULE (pinned): bigtest ANSWER >=960 AND new-kind
  per-kind >= gen-7b levels − 0.03 -> the wall is the head; growth
  opens with a measured invoice (head ~3.2M against a frozen
  half-billion — 2x is a rounding error) and (a)/(b) ride the bigger
  head. bigtest <930 -> capacity EXONERATED; crowding is a data-mix
  problem; (b)-then-(a) fires with confidence. 930-960 -> extend +8k
  once, re-read. PAD-WARM CAVEAT stated: the relational law says padded
  compositions may not inherit — a null result reads through that lens
  before exonerating capacity. RIDING REGISTRATIONS: (i) NEW SPECIMEN
  CLASS — CORRECT-BUT-UNCERTIFIABLE ([71] votes [9,9], [7] votes [45]:
  right answers, vote-shy across permuted views) — the mirror of raw
  [71]'s stable-wrongness; the lattice's geometry lives between the two
  (stability without truth / truth without stability). Parked with the
  K-dial / view-family question; these two are its acceptance probes if
  the class grows on wilder prose. (ii) [85]'s INVERSE SHAPES
  (unknown-squared, inverse-fdiv) ship in gen-8's rotation WHICHEVER
  branch wins — named generator gap, standing acceptance probe. The
  probe awaits the word.
- **REGISTERED (2026-07-11, Bryce's gut + relay: THE PHYSICS TRIAD) —
  three instruments for the phase boundary; statistical mechanics of a
  fixed-capacity head under growing load.** (1) **INTERFERENCE MATRIX
  (GPU, rides the capacity probe):** crowding = destructive gradient
  interference in shared weights. Per-register gradient cosine matrix
  on the gen-7b ckpt (one batch per register, one backward each,
  pairwise cosines). Joint verdict table with the capacity probe:
  anti-aligned + erosion-reverses-at-2x = capacity wall (orthogonal
  subspaces need room); aligned + erosion = data starvation (replay
  cures); anti-aligned PERSISTING at 2x = genuine task conflict -> the
  §8.4 LoRA fallback ladder gets its first customer (per-register
  adapters). (2) **ERASURE-VS-SHARE CORRELATION (zero-GPU, fires now):**
  Landauer transferred honestly — erasure is never free; when task
  entropy exceeds head capacity the mix decides who pays. PREDICTION
  (pinned): per-register erosion gen-6 -> gen-7b tracks INVERSE
  mix-share of fresh rehearsal (alg4/pct/seq got zero new rows -> pays
  most; original nl diluted to ~55% -> pays next; dag-fresh registers
  gain). Holds -> the crossover gets its conservation law: at capacity,
  expansion pays dividends MINUS erasure. (3) **VOTE-ENTROPY COLUMN
  (zero-GPU pilot on banked book-1 votes):** the two specimen classes
  unify as BASIN DEPTH — view permutation is the thermal kick; vote
  entropy across views is a per-item effective temperature. Fourth
  lattice column chartered: CORRECT-BUT-SHALLOW (the retraining-target
  class — rehearsal deepens basins rather than teaching). Pilot on the
  persisted book-1/acceptance votes; full column when TTA outcomes
  persist per-item at census scale. GPU items ((1) + capacity probe)
  hold for the word; (2)+(3) fire on banked data now.
- **VERDICTS (2026-07-11, the zero-GPU pair):** (2) **LANDAUER CHECK —
  DIRECTION CONFIRMED, LAW REFINED.** Composition recovered exactly
  (dag7 register 45.2% of mixed7b; nl-core 9.5%; tranches ~12% each).
  rho(fresh-rehearsal share, erosion) = +0.50, n=5 — right direction,
  short of the pinned "strongly positive," and the DEVIATION is the
  finding: alg2 IMPROVED (+4.4%) with zero fresh file-share because
  dag7 rows REHEARSE its kinds covertly (sel/mod/coupled live inside
  dag7 wiring), while alg4's pct/seq kinds appear NOWHERE in dag7 (true
  zero rehearsal -> worst erosion, −14%), nl-core's older surface forms
  partially shared (−9.9%), verbose flat (same relations, held). THE
  CONSERVATION LAW REFINES: erasure is ordered by UNSHARED-CIRCUIT
  rehearsal share, not corpus-file share — kind-level, not file-level.
  Actionable for any replay mix: rehearse KINDS, not files. (3)
  **VOTE-ENTROPY PILOT — the basin-depth read separates exactly as the
  physics said:** deep-correct H=0.000, shallow-correct H=0.846,
  deep-wrong H=0.212, refused H=0.116 (banked book-1 gates, n=36).
  Entropy cleanly separates SHALLOW from DEEP (0.85 vs ~0.1-0.2) and
  CANNOT separate deep-correct from deep-wrong — which is the
  quantitative restatement of why the chain needs the mouth AND the
  key: temperature is orthogonal to truth. Fourth column
  (correct-but-shallow) validated at pilot scale; full column when
  per-item votes persist at census scale. Scripts:
  `erasure_share_correlation.py`; pilot inline in session log.
- **THE LAW NAMED + THE QUADRANTS PINNED (2026-07-11, relay):**
  (1) **CIRCUIT REHEARSAL, NOT FILE REHEARSAL** — the corpus economy's
  law: the erasure bill is charged per UNSHARED CIRCUIT; a register
  whose kinds live inside another register's shapes rides free (alg2
  +4.4% at zero fresh rows). Replay design converts from mixing FILES
  to mixing COVERAGE: gen-8's diet is specified as a KIND-REHEARSAL
  MATRIX (which circuits each row exercises), deficit = the
  true-zero-rehearsal kinds (pct/seq, the −14% line items). The law
  also re-explains the dividends streak at depth: expansion paid
  BECAUSE new shapes covertly rehearsed old circuits — constructive
  interference through shared kinds — and broke exactly where sharing
  hit zero. One law, both signs. (2) **CORRECT-BUT-SHALLOW = FREE
  CURRICULUM SIGNAL**: rehearsal targets, self-identified by their own
  temperature — fold into replay to DEEPEN basins (cheaper than any
  new capability); epigraph + fifth-column candidate added to paper §7.
  (3) **THE JOINT TABLE'S FOUR QUADRANTS, all pre-written** (capacity
  probe x interference matrix): (anti-aligned, erosion-reverses-at-2x)
  -> capacity for orthogonal subspaces; (aligned, reverses) -> PURE
  capacity, state-counting only — remedy is the bigger head alone, no
  LoRA, no mix surgery beyond the kind matrix; (aligned, persists) ->
  starvation, replay cures; (anti-aligned, persists) -> genuine task
  conflict, §8.4's LoRA ladder gets its first customer. One run, four
  pre-written verdicts, no cell left to improvise. GPU pair
  (`cap_probe.sh` + `interference_matrix.py`) staged, on the word.
- **VERDICT (2026-07-11, the GPU pair): THE MATRIX IS CLEAN AND NAMES
  THE MECHANISM; THE PROBE'S CAPACITY AXIS IS CONFOUNDED BY ITS OWN
  WARM-START.** (1) **INTERFERENCE MATRIX (uncontaminated — measured
  directly on gen-7b):** the old guard is mutually ALIGNED (nl-core /
  alg2 / alg4 pairwise +0.22..+0.26 — shared circuits, the dividends
  streak photographed) and **dag7 is ANTI-ALIGNED with exactly the
  eroding registers** (nl-core −0.171, alg2 −0.255, alg4 −0.263);
  verbose/dag6 orthogonal (verbose grad norm 0.10 — fully learned,
  cosines are noise). THE TWO-FORCE MECHANISM, both instruments
  agreeing: destructive gradient pressure from the dominant register,
  OFFSET by covert kind rehearsal where present — alg2 (anti-aligned
  BUT kind-shared) nets positive; nl-core (anti-aligned, surface
  differs) nets negative; alg4 (most anti-aligned + zero kind share)
  pays worst. Interference axis: ANTI-ALIGNED, definitively. (2)
  **CAPACITY PROBE — mechanically sub-930 (bigtest 809) but the
  pre-registered pad-warm caveat FIRES:** the 2x head at 12k is worse
  than its own warm source EVERYWHERE (val 0.8234 vs 0.8736; all
  per-kind down; alg4 208 — disruption again hitting the
  least-rehearsed register hardest) and still climbing at cutoff —
  the probe measured DISRUPTION RECOVERY, not capacity (the relational
  law's exact prediction for padded compositions). The capacity axis
  is UNREAD. QUADRANT: provisionally (anti-aligned, persists) — task
  conflict, the LoRA ladder's customer — but the clean capacity
  instrument is now registered for the word: **the fair A/B** — 2x
  head, SAME 32k schedule, SAME mixed7b, only width differs; erosion
  reversal at matched schedule reads capacity cleanly. Ckpt:
  `.cache/phase1_cap2x_head.safetensors` (probe artifact, not a gate
  candidate). The night's arithmetic: the matrix cost one backward
  pass per register and delivered the mechanism; the probe cost 12k
  steps and delivered a confound — the cheap instrument won.
- **REGISTERED (2026-07-11, relay — the temperament frame + two
  riders):** (1) **THE ATLAS'S GATE-2 INHERITS ITS DATASET** (one
  line, no fire): the interference cosines + the kind-rehearsal matrix
  are an empirical similarity structure over registers/kinds — if it
  proves tree-metric as tranches accumulate (register families sharing
  circuit-ancestry), the delta-probe opens with evidence produced as a
  byproduct. Tripwire unchanged (watches centroid-margin shrinkage,
  not gradient conflict). (2) **THE TEMPERAMENT DESIGN (the eleventh
  instinct, musical keys):** registers are keys, kinds are notes, the
  interference matrix IS the circle of fifths (alg2-inside-dag7 =
  closely related keys; alg4 = the tritone). The 33-key head is tuned
  in EQUAL TEMPERAMENT — every key playable, every key compromised.
  The two remedies are the two historical tunings: grow the head =
  more strings, same temperament; the §8.4 ladder = WELL-TEMPERED —
  shared instrument, per-key accidentals. THE SHARPENING THE MUSIC
  BOUGHT: accidentals ONLY where the matrix shows dissonance —
  adapters for the anti-aligned pairs alone; the old guard's mutual
  +0.25 means they WANT one tuning, and adapters there would waste
  parameters fixing consonance. Sharper than per-register-everywhere,
  derived from a metaphor, consistent with ten prior instincts. (3)
  **THE FAIR A/B, staged with its reading frame pre-committed:** 2x
  pad-warm, FULL 32k schedule, identical mixed7b — only width moves
  vs gen-7b's own 32k. Honest residual asymmetry noted: gen-7b's warm
  source was same-width; cap2x's is padded — a clean-cold A/B is the
  escalation if this one reads ambiguous. RULE: bigtest >=960 ->
  capacity was the wall; <930 at matched schedule -> exonerated;
  PARTIAL (930-960, or reversal WITH anti-alignment persisting in the
  32k head's OWN matrix — the matrix reruns on the A/B artifact) ->
  BOTH tunings at once: more strings AND accidentals for the
  dissonant keys, as the music predicted before the run. Per-kind
  guard unchanged (>= gen-7b − 0.03). Awaits the word.
- **VERDICT (2026-07-12, the fair A/B — overnight): CAPACITY
  EXONERATED, CLEANLY; the quadrant resolves to STARVATION; the
  kind-rehearsal law called it.** The one number: bigtest **888** —
  below the 930 exoneration line at the MATCHED 32k schedule, and
  statistically the same model as the 1x everywhere: alg4 316~319,
  alg2 554~575, dagtest 677~670, vtest 599~600, dag7btest 419~420,
  per-kind all within the ±0.03 guard (sq 0.329, ladder 0.500, fdiv
  0.521, coupled 0.551), val 0.8646 vs 0.8736. WIDTH BOUGHT NOTHING.
  (The 12k probe's disruption transient fully washed out by 32k —
  that diagnosis confirmed in passing.) THE DEEPER RESULT, from the
  A/B artifact's own interference matrix: **decorrelation without
  improvement** — at 2x the anti-alignment softened everywhere (dag7
  vs nl-core −0.171->−0.099, vs alg2 −0.255->−0.113, vs alg4
  −0.263->**+0.076**, sign-flipped) AND the old guard's mutual
  alignment dissolved (+0.25 family -> ~0.1/negative): the wider head
  spread registers into orthogonal subspaces exactly as geometry
  allows — and behavior did not move. GRADIENT INTERFERENCE IS A
  SYMPTOM OF SHARED-CAPACITY PACKING, NOT THE CAUSE OF FORGETTING:
  given orthogonal room, the gradients decorrelate and the erosion
  stays, because nothing about width changes WHAT IS REHEARSED.
  QUADRANT: (interference relieved, erosion persists) = STARVATION —
  REPLAY CURES. Among the physics triad, the entropy/rehearsal
  instrument called the mechanism; the interference instrument
  photographed a symptom; the capacity probe (run fairly) exonerated
  the suspect. CONSEQUENCES: head growth SHELVED (receipt: 888 at
  32k) and the §8.4 LoRA ladder's customer WITHDRAWN (receipt:
  conflict-as-mechanism falsified) — both with evidence, neither by
  taste. GEN-8 = (b)-then-(a) exactly as sequenced: inverse shapes
  ([85]'s standing probe) + the KIND-REHEARSAL replay mix (rehearse
  KINDS, not files; deficit = pct/seq, the true-zero line items). The
  music's final read: the instrument needed neither more strings nor
  accidentals — the old songs had simply left the practice schedule.
  Rehearsal, not tuning. Ckpt `.cache/phase1_cap2x_32k_head.safetensors`
  (A/B artifact, not a gate candidate; gate remains gen-6).
- **PROMOTED TO THE LAW FAMILY (2026-07-12, relay): "DECORRELATION
  WITHOUT IMPROVEMENT" is the publishable finding — gradient
  interference is the PHOTOGRAPH of registers packing into shared
  capacity, not the MECHANISM of forgetting; the mechanism is
  rehearsal, and orthogonal room does not change the practice
  schedule. The interference matrix keeps its diploma as a PACKING
  DIAGNOSTIC; its causal ambitions retire with a receipt. OPERATIONAL
  LAW: **when erosion appears, check the rehearsal ledger before the
  architecture** — cheap accounting beat expensive geometry (the
  entropy instrument called the mechanism for arithmetic on banked
  tables; the geometry instruments cost GPU-nights to photograph
  symptoms and clear suspects). Flattering-verdict principle collected
  on the junction's MOST seductive hypothesis: task conflict was the
  interesting story, LoRA the elegant remedy; the fair A/B declined
  both with numbers. Two expensive builds shelved by one overnight run.
- **REGISTERED (2026-07-12): GEN-8 — the practice-schedule generation.
  Diet fully specified by three free instruments; credit clause
  applied (no double-pay for covertly-rehearsed kinds — sel/mod/coupled
  ride inside dag7).** THE RATIONS: (1) KIND-RATION — 3000 fresh
  alg3-register rows (seq/pct/fdiv, the true-zero deficit, −14% line
  item); (2) SURFACE-RATION — 2500 fresh nl-core rows (nl-core's
  erosion was surface-unshared per the matrix: the kinds survived, the
  PHRASINGS starved — two different starvation species, two rations);
  (3) SHALLOW-BASIN RATION — DEFERRED to gen-9 (needs per-item TTA
  vote-entropy at corpus scale; registered, not forgotten); (4)
  INVERSE SHAPES — dag8 rotation adds isq (mul(a,a) with a UNGIVEN,
  [85]'s circuit) + ifdiv (dividend pinned by quotient+remainder),
  nogive mechanics in the givens-gate; smoke 30/30. mixed8 = mixed7b
  + 8500 = 29,500 rows; warm from gen-7b; 32k steps. BARS (pinned):
  **bigtest >=980 — the starvation thesis's own falsifiable claim: if
  the replay diet does not recover bigtest, starvation is wrong too
  and the junction reopens**; alg4test >=420; alg2test >=530; vtest
  >=598; dagtest >=640; dag7btest >=400; [85] BANKS; acceptance >=6/8;
  per-kind sq >=0.40, fdiv/ladder/coupled >=0.50. ALL bars hold ->
  GEN-8 PROMOTES TO THE GATE (first promotion since gen-6) and the
  census re-runs under it. Any bar broken -> no promotion, diagnose.
- **VERDICT FRAME PINNED PRE-MEASUREMENT (2026-07-12, relay — the
  density-regime discipline applied to the bars themselves):** the bar
  structure has an asymmetry worth naming before it prints. bigtest
  recovering while alg4 misses 420 = starvation CONFIRMED + the
  alg3-ration mis-sized or mis-targeted (kind-ration ARITHMETIC, not
  mechanism failure — diet-tuning). **The junction reopens ONLY if the
  RATIONED registers fail to respond to their OWN rations.** A mixed
  table reads as grocery arithmetic, not thesis-death; only
  ration-blind erosion kills starvation. Also noted: the smoke's
  166-vs-30 rejection rate on inverse shapes is the uniqueness gate
  EARNING ITS KEEP (an ungiven var constrained only downstream has
  more freedom to pin) — high mint-time rejection on inverse problems
  is the gate working, not friction; `nogive` handles [85]'s circuit
  the solution-first way (the edge condition made UNREPRESENTABLE,
  as with no-real-roots). STRATEGIC RIDER — what promotion unlocks:
  gen-8 promoting re-prices the BOOTSTRAP, not just the census — the
  2/26/72 ran under gen-6, and the near-miss tier's shared cause
  (fdiv-in-composition) is precisely what the kind-ration + gen-7b's
  fdiv gains address. A banked-column jump under gen-8 brings the
  machine-drafts-the-isotopies economics with it: book 2's annotation
  budget splits between hand-surgery on the genuinely knotted and
  gate-disposal of machine proposals. The practice schedule may fund
  the apprenticeship.
- **VERDICT (2026-07-12): GEN-8 — NO PROMOTION (two named bars miss),
  STARVATION CONFIRMED BY ITS OWN RATION, and the [85] mystery solved
  at the ENCODING layer.** The table: bigtest **967** (bar 980 — but
  +66 from 901, recovering two-thirds of the erosion under its
  surface-ration: THE RATIONED REGISTER RESPONDED; per the pinned
  frame this is diet-tuning, ration undersized, thesis ALIVE);
  alg4test **332** (bar 420, +13 only — the kind-ration MIS-TARGETED:
  3000 alg3-register rows barely moved the alg4 register; the alg4
  generator was archived in the house-cleaning and its register
  differs from alg3's — gen-9 needs a register analysis or the
  generator recovered); alg2test 565 ✓, vtest 600 ✓, dagtest 664 ✓,
  dag7btest 422 ✓ — NO NEW EROSION at 29.5k rows: the diet did not
  crowd. ACCEPTANCE 6/8 ✓ with the qualitative headline: **[71], [7],
  [72] all jumped to 5/5 UNANIMOUS** — two correct-but-shallow
  specimens DEEPENED TO CERTIFIED and the coupled system banked;
  rehearsal deepens basins, measured. [45] now votes [168,154,168,110]
  — a NEW specimen (mixed-vote: right answer present, wrong votes
  competing) for the vote-entropy column. **THE [85] DISCOVERY
  (verified at source, decode line 495 + gold encoding):** the args
  pointer decodes as top-2 DISTINCT slots and the gold multi-hot
  cannot express multiplicity — args=[a,a] is UNREPRESENTABLE
  end-to-end. Three trainings flat at ~0.33 sq because the target was
  never learnable AS ENCODED; 600 isq rows changed nothing because
  they COULDN'T. The third "unrepresentable, not unlearned" finding
  (family: positional-structure-as-structure, no-real-roots). Gen-9
  fix is small: an arg-multiplicity bit (ftype-conditional), gold
  field, decode branch. CENSUS 1/23/76 — no bootstrap jump (raw prose
  untouched by dialect-side training, as predicted every time).
  GEN-9 REGISTERED ITEMS: (1) args-multiplicity mechanism
  (architecture, small); (2) alg4-register analysis -> correctly
  TARGETED kind-ration; (3) surface-ration upsized (~2500->4000);
  (4) the shallow-basin instrument (deferred from gen-8, now with
  [45]'s mixed-vote specimen as motivation); (5) fdiv borderline
  (0.483/0.535) rides the retargeted ration. Gate remains gen-6;
  ckpt `.cache/phase1_gen8_head.safetensors` banked unpromoted.
- **CORRECTION + TWO STANDING RULES (2026-07-12):** (1) **THE GEN-8
  'MIS-TARGETED RATION' ATTRIBUTION WAS WRONG — SEED COLLISION.**
  algebra4_nl_train was itself minted by algebra3_nl_gen (seed 81,
  teeth 0.8); the gen-8 kind-ration re-used seed 81 -> **2500/3000
  rows byte-identical duplicates** -> the alg4 register accidentally
  received a 2x REHEARSAL UPWEIGHT... and moved only +13. THE RATIONED
  REGISTER FAILED ITS OWN RATION: per the pinned frame, the junction
  REOPENS FOR ALG4 specifically (capacity already exonerated for it
  by the A/B at 316~319; rehearsal now exonerated by the accidental
  2x). The per-kind diff instrument (gen-6 vs gen-8 heads over
  alg4test, sliced by seq/pct/fdiv/crt/vieta) fires before any gen-9
  ration decision. PROCESS GUARD MINTED: ration seeds must not
  collide with historical corpus seeds; the mix builder must PRINT
  DUPLICATE COUNTS (a data-boundary guard per the no-silent-fallback
  law — this one hid inside a healthy-looking corpus for a full
  generation). (2) **THE REPRESENTABILITY-AUDIT RULE (relay):** a
  metric flat across MULTIPLE trainings AND a targeted data
  intervention is not starved — it is STRUCTURALLY EXCLUDED; three
  flat trainings trigger a representability audit BEFORE a fourth
  fires. Applied retroactively it catches [85] two generations early.
  (3) **DELIBERATE EXCLUSIONS CARRY EXPIRATION TAGS (relay):** the
  repeated-arg exclusion began as a v0 soundness guard ("mul(x,x)
  would unsound the pairwise propagator") and AGED INTO the encoding
  bug when wild shapes arrived — scoping decisions expire the way the
  integrality jaw did; tag them at birth. (4) Surface-ration
  expectation clause (relay, pre-said): 4000 rows at the measured
  +66/2500 exchange rate plausibly clears 980, but the last points of
  recovered erosion are the hardest-starved — **975 = the ration
  curve bending, NOT the thesis breaking**; the bar stays 980, the
  expectation stays humble.
- **ALG4 DIFF VERDICT + GEN-9 FINAL SCOPE (2026-07-12, registered
  before firing):** the per-kind diff (gen-6 vs gen-8, alg4test)
  shows **UNIFORM decline across every kind** (vieta −0.075, seq
  −0.069, pct −0.060, fdiv −0.056, crt −0.047 — no single victim).
  KIND-STARVATION REFUTED FOR ALG4: the register declines AS A WHOLE
  while unresponsive to 2x rehearsal (the accidental dup-upweight),
  2x width (the A/B: 316~319), and decorrelation. THE ONE UNTESTED
  SUSPECT: **THE SCHEDULE** — curriculum orders coarse->fine by teeth
  score; alg4's high-teeth rows enter only in the final training
  phase, which cosine decay runs at annealed LR, and that phase's
  composition shifted as the mix grew. REGISTERED PROBE (own track,
  on the word, post-gen-9): a 12k CURRICULUM=0 arm on mixed8, reading
  alg4test — schedule starvation vs register mystery. GEN-9 SCOPE
  (kind-ration DROPPED with evidence): (1) ALG_DUP multiplicity
  mechanism (built; selftest green; env-gated, legacy byte-compat);
  (2) surface-ration 4000 nl-core rows, FRESH seed 99,
  collision-checked; (3) mixed9 DEDUPES mixed8 (removes the 2500
  seed-collision duplicates; the mix builder prints dup counts — the
  guard, operational); (4) the SHALLOW-BASIN INSTRUMENT fires
  (3000-row sample, 5-view vote entropy under gen-8; correct-but-
  shallow rows oversampled x2 — rehearsal deepens basins, measured on
  [71]/[7]/[72]). BARS: [85] BANKS (self-grading); sq per-kind
  >=0.45; bigtest >=980 (humble clause stands); alg4test >=310
  HOLD-THE-LINE (recovery rides the schedule probe, not this diet);
  acceptance >=7/8; alg2 >=530, vtest >=598, dagtest >=640, dag7b
  >=400. All bars hold -> PROMOTE (census rides).
- **THE HASH AUDIT (2026-07-12, Bryce's gut + relay — three-way,
  zero-GPU): THE ANTI-COLLISION CLASS IS REAL — 42 VERIFIED
  CROSS-BOUNDARY ISOMORPHS.** (a) GREP CENSUS: the codebase is nearly
  hash-free — manifest pins are SHA-256/64-bit (safe by orders of
  magnitude at ~15 artifacts); all dedup is exact-text in Python sets
  (identity-with-equality, safe by construction); ONE catch:
  test_kenken_parity used salted built-in hash() as an RNG seed
  (fixtures non-reproducible across sessions; parity itself unharmed
  — both arms share the seed) -> FIXED to crc32 (stable). (b) No
  persisted built-in hash() anywhere. (c) **THE LOAD-BEARING COUNT:
  canonical WL form (values included, commutative roles sorted,
  exact-verify by backtracking before counting) over train (mixed8,
  29,500 rows -> 26,920 classes; 2,574 multi-member = within-train
  redundancy, the small-problem pigeonhole) x the 7-test battery:
  exact-text overlap 0 everywhere (the string dedup held) but
  **bigtest 27/1500 (1.8%), vtest 13/600 (2.2%), alg4test 1,
  dag7btest 1 — same knot, different diagram, across the boundary.**
  Cause: pigeonhole density in the small-problem regime, not a seed
  bug. FOOTNOTE ON ALL STANDING BARS: bigtest numbers carry <=1.8%
  isomorph inflation, vtest <=2.2% (deltas across generations
  unaffected — same fixture both sides). Exclusion list persisted
  (.cache/iso_contamination.json); REGISTERED: (i) clean-subset
  re-read of the battery rides the next eval pass; (ii) the canonical
  digest becomes the MINT-TIME dedup + test-fixture exclusion going
  forward (knot invariant, not diagram fingerprint — the knots law
  applied to our own bookkeeping); (iii) paper tables freeze on CLEAN
  fixtures or report the exclusions. The gut said beware collisions;
  the deeper hazard was the anti-collision, and it was found before
  the tables froze.
- **TWO CONSEQUENCES PINNED (2026-07-12, relay):** (1) **THE
  PIGEONHOLE FINDING IS A STANDING CONSTRAINT** — 2,574 within-train
  redundancy classes means the small-problem regime's structural
  diversity is finite and partially exhausted; every future ration
  inherits that ceiling. The canonical digest converts hazard to
  instrument: the mint now COUNTS ITS OWN KNOT DIVERSITY per batch
  (redundancy-class coverage alongside kind coverage) — the
  KIND-REHEARSAL MATRIX UPGRADES TO A KNOT-REHEARSAL MATRIX, the
  practice schedule tracked at the invariant level. The diagrams->
  moves correction, applied to the corpus bookkeeping itself. (2)
  **CANONICAL DISJOINTNESS BECOMES A GENERATION-BUMP GATE** — checked
  at every bump so the isomorph class can never silently re-enter;
  the paper's reproducibility statement earns "train/test
  disjointness verified up to graph isomorphism," which almost no
  benchmark can claim because almost none check. THE TWELFTH
  INSTINCT'S SCORECARD: the gut said collisions, the framing said the
  anti-collision is scarier, the measurement said both were right in
  their own jurisdictions — the collision class was EMPTY (hygiene
  held), the anti-collision class had 42 members (the find). The
  instinct locates the neighborhood, the framing names the streets,
  the audit knocks on doors.
- **VERDICT (2026-07-12): GEN-9 — NO PROMOTION BY EXACTLY ONE BAR;
  everything else is the sprint's best table.** THE WINS: **[85]
  BANKS 5/5 UNANIMOUS** — the multiplicity fix works, the self-grading
  probe graded itself, three generations of mystery ended by one
  representability audit; **sq per-kind 0.319 -> 0.751** (bar 0.45,
  shattered — the encoding fix unlocked the entire kind); **bigtest
  1084** (bar 980; gen-6's 1000 EXCEEDED by 84 — the ration curve
  didn't bend, it OVERSHOT: the starvation thesis fully vindicated,
  erosion story CLOSED — rehearsal was the mechanism, the diet cures
  it); dag7btest 422 -> **510**, dag8test 379 -> 510 (the new kinds
  consolidated: fdiv 0.660, coupled 0.681, ladder 0.506, all bars
  cleared); alg4test 315 holds the line (schedule probe pending);
  alg2test 559, vtest 600/600, dagtest 669; val RECORD 0.8826;
  shallow-basin first census: deep 1432 / SHALLOW 925 (31% of the
  corpus!) / wrong 29 / refused 614. THE ONE MISS: **acceptance 6/8
  (bar 7/8)** — [71] votes [9,9] and [78] votes [16]: RIGHT answers,
  vote-shy — both were 5/5 under gen-8 and RE-SHALLOWED under gen-9's
  diet shift. Rehearsal deepens basins; diet shifts can re-shallow
  specific ones — the correct-but-shallow class claiming its first
  promotion casualty, in its own vocabulary, one generation after the
  instrument was built. NO PROMOTION, mechanically; gate remains
  gen-6. CENSUS 1/18/81 (near shrank, knotted grew — raw-prose
  reading untouched as always; books remain the path). REGISTERED
  NEXT (on the word): **GEN-9B, a basin top-up, not a generation** —
  continue from gen-9 ckpt ~8k steps with a small fdiv-tiny-chain
  booster (the [71]/[78] shape, ~500 rows) + shallow-census-under-
  gen-9 oversamples; re-run acceptance + battery. The two vote-shy
  items are the acceptance probes; bars unchanged. Ckpt
  `.cache/phase1_gen9_head.safetensors` banked unpromoted.
- **REGISTERED (2026-07-12, relay — GEN-9B with the displacement
  question riding):** (1) **THE BASIN-DISPLACEMENT CONSERVATION
  QUESTION** — [71]/[78] were 5/5 under gen-8; gen-9's diet (which
  oversampled shallow basins TO DEEPEN THEM) re-shallowed exactly
  those two. The uncomfortable reading, carried before the top-up can
  flatter it away: at fixed capacity, BASIN DEPTH MAY BEHAVE LIKE A
  BUDGET — rehearsal deepens practiced basins partly by drawing from
  unpracticed neighbors; the diet ALLOCATES consolidation rather than
  creating it. The Landauer law one level down: erasure-by-rehearsal
  was about ACCURACY; this is the same law about CERTAINTY.
  PRE-REGISTERED READ (rides gen-9b free): the top-up deepens the
  [71]/[78] shape while KEEPING gen-9's shallow-oversamples — (a)
  those two at 5/5 but two DIFFERENT probes gone vote-shy ->
  displacement CONSERVED; the diet question changes from
  what-to-add to WHAT EQUILIBRIUM TO SEEK; (b) >=7/8 with no new
  casualties -> depth NOT zero-sum in this regime; the budget worry
  dies with a receipt and the top-up was just a top-up. (2) **BOOSTER
  MINTED AT KNOT LEVEL** (the knot-rehearsal matrix's first day on
  the job): the [71]/[78] fix is specified as N=500 DISTINCT
  CANONICAL REDUNDANCY CLASSES of the fdiv-tiny-chain shape (counted
  by the canonical digest at mint; pigeonhole-dense regime — training
  the diagram again is the named risk), each checked canonically
  DISJOINT from every test corpus (the bump gate's first live use).
  (3) TOP-UP TRAINING DESIGN: continue from gen-9 ckpt, 8k steps,
  LR=1e-4 (cosine), CURRICULUM=0 (a top-up trains near the converged
  regime; re-running the coarse phase at high LR is the known
  disruption shape). Bars unchanged from gen-9; all hold -> PROMOTE.
- **VERDICT (2026-07-12): GEN-9B — ALL BARS HOLD; **PROMOTED TO THE
  GATE** (first promotion since gen-6; the gate ckpt is now
  `.cache/phase1_gen9b_head.safetensors`).** THE TABLE: acceptance
  **8/8, EVERY probe banked at 5/5 or 4/5** ([71] and [78] back to
  unanimous; [85], [72], [56], [45], [7], [46] all held — no new
  casualties anywhere); bigtest **1090** (a second record); alg4test
  **344** (+29 over gen-9 — see below); alg2test 571, vtest 600/600,
  dagtest 671, dag7btest 523; per-kind sq 0.751 / coupled 0.697 /
  fdiv 0.677 / ladder 0.529; val record 0.8890; booster minted 500/500
  distinct knots, 0 test-isomorphs admitted (the disjointness gate's
  first live tour, clean). **THE DISPLACEMENT ANSWER: (b) — DEPTH IS
  NOT ZERO-SUM IN THIS REGIME.** The top-up deepened [71]/[78] to 5/5
  while every other basin held or deepened; the budget worry dies
  with its receipt, pre-registered. The correct reading of gen-9's
  re-shallowing: diet SHIFTS jostle specific basins transiently;
  gentle continued training consolidates without displacement.
  **BONUS EVIDENCE FOR THE SCHEDULE HYPOTHESIS:** alg4test rose +29
  under 8k steps of LOW-LR, NO-CURRICULUM, full-mix training — more
  than three full generations moved it — consistent with
  curriculum x cosine starving high-teeth rows of usable-LR steps;
  the registered CURRICULUM=0 probe gains a prior. **THE
  TRAINING-REGIME LAW (relay, registered before it generalizes):**
  the displacement answer's precise scope is the REGIME, not just the
  diet — gen-9's re-shallowing happened under a full retrain with a
  shifted diet; gen-9b's consolidation-without-displacement under
  gentle continuation (8k, low LR, near convergence). **Hard restarts
  jostle basins; gentle continuation deepens without displacement.**
  The generation protocol may evolve toward fewer full retrains and
  more staged continuations; the transaction manifest tracks
  checkpoint LINEAGE from here. CENSUS TREND
  (three points, directional, informational): 76 -> 81 -> 89 knotted
  as basins deepen — the head's consolidation on its own register
  REDUCES accidental raw-prose carries; the style wall hardens as the
  dialect sharpens; the bootstrap's raw-prose economics await books,
  as every census has said. THE ARC CLOSES: the junction opened with
  two kills and three physics instruments; it closes with rehearsal
  confirmed as the mechanism (both directions), capacity and conflict
  exonerated with receipts, [85] representable and banked, the
  bookkeeping counting knots, and the gate moving on a table with no
  asterisks. Next chapter: the schedule probe (registered), gen-10's
  knot-matrix diet, and BOOKS — the style wall is now the tallest
  thing standing.
- **REGISTERED (2026-07-12): THE SCHEDULE PROBE — the coldest-optimizer
  pair.** Mechanism claim: cosine schedules spend their usable LR on
  the curriculum's early (easy) phase; high-teeth rows arrive when LR
  has decayed past learning — THE HARDEST DATA GETS THE COLDEST
  OPTIMIZER. Design (isolates CURRICULUM alone): two 12k arms, both
  warm from gen-7b on mixed8 at LR 3e-4 (gen-8's exact condition;
  m8train states banked; no ALG_DUP — replicates the original
  regime): ARM A CURRICULUM=1 (control), ARM B CURRICULUM=0. PRIMARY
  READ: alg4test(B) − alg4test(A); prediction B > A by >=15 answers.
  GUARD: bigtest(B) must not trail bigtest(A) by >20 (if B lifts
  alg4 but craters elsewhere, the fix is per-band LR accounting, not
  curriculum removal). Confirmation -> gen-10 trains flat-mix (or
  staged LR); refutation (B ~ A) -> the top-up's +29 attributes to
  low-LR continuation, strengthening the regime law instead. Either
  way one pair of short runs converts a scatter of "hard register
  learns slowly" mysteries into a single attributed mechanism or
  clears the curriculum with a receipt.
- **VERDICT (2026-07-12): THE SCHEDULE PROBE — CONFIRMED, 6x PAST THE
  BAR, AND THE CURRICULUM IS NET-NEGATIVE AT SCALE.** Arm A
  (CURRICULUM=1): alg4test 296, bigtest 916, dagtest 652, val 0.8388.
  Arm B (CURRICULUM=0): alg4test **384 (+88; bar was +15)**, bigtest
  **1032 (+116)**, dagtest 678 (+26), val 0.8618 (+0.023). The guard
  didn't just hold — B LEADS EVERYWHERE: the curriculum is hurting
  every register at the 30k-mix scale, not just the high-teeth ones.
  THE COLDEST-OPTIMIZER MECHANISM CONFIRMED with a sharper corollary:
  arm B at 12k nearly matches gen-9's 32k bigtest (1032 vs 1084) —
  **the curriculum was burning roughly two-thirds of every training
  budget** (the easy-pool phase consumes the hot LR on data the head
  already knows; the full mix arrives to a cold optimizer). It also
  retroactively explains the mid-training val dips (the 0.5 -> 0.8
  jumps at pool transitions) and every "still climbing at cutoff."
  JURISDICTION LESSON, again: the curriculum won its ablation on
  2026-07-10 in the single-register era; at mixed-register scale the
  verdict INVERTED — **ablation verdicts expire with their regime**,
  the way deliberate exclusions do; scope tags on both from here.
  GEN-10 CONSEQUENCE: flat mix from step one, and the schedule
  dividend (≈3x effective budget) comes free. Ckpts: sched_probe_armA/
  armB (probe artifacts). Gate remains gen-9b (promoted on its bars;
  gen-10 collects the dividend).
- **THREE REGISTRATIONS (2026-07-12, relay — before gen-10):** (1)
  **SCOPE DECAY, a new species for the §6 family:** ablation verdicts
  expire with their regime — the curriculum won HONESTLY (2026-07-10,
  single-register era, sound measurement) and inverted at
  mixed-register scale. Not audit-that-confirms, not
  flattering-verdict: a verdict aging out as the system changed under
  it. OPERATIONAL FORM: every ablation verdict carries a regime tag
  (register count, mix scale, schedule era); any verdict older than a
  structural shift re-audits before it is load-bearing again. (2)
  **THE REGIME CENSUS** (verdicts predating the mixed-register era,
  by re-audit priority): [HIGH] the LOSS-TERM WEIGHTS (2.0 on
  args/res/query, 4.0 args_w — set in the single-register era,
  load-bearing in every run since); [MED] LR=3e-4/BATCH=8 (old, but
  implicitly re-validated by every healthy val curve — flat-mix
  changes the regime again, so a small LR sweep rides a future
  cheap slot); [MED] the 5-view/vote-3 TTA dials (gen-5 era; but
  exercised daily by acceptance probes — living verification);
  [LOW] T_ALG=256, N_DIG=3, H_W=512 (H_W freshly re-validated by the
  capacity A/B; the others are data-bounded, not regime-bounded).
  None urgent; all now tagged. (3) **THE TEXTURE RULE** (this
  channel's self-audit): recurring unexplained texture in training
  curves is an ANOMALY, not scenery — two sightings of the same
  unexplained curve shape trigger a mechanism probe (the [85]
  three-flat-trainings rule, generalized to curves). The mid-training
  val dips and "still climbing at cutoff" were logged repeatedly as
  texture; the probe they pointed to was worth 2/3 of every training
  day and fired a week late. (4) BOOKS ECONOMICS NOTE: at 3x
  effective budget, December's reading-training runs just tripled in
  affordability — gen-10 carries the FIRST REAL-PROSE INGEST (v0,
  n=14 book pairs, informational arm) alongside its diet rather than
  after it. The wall is tallest; the ladder got longer.
- **REGISTERED (2026-07-12): GEN-10 — flat-mix lineage continuation +
  the knot-matrix diet + the prose-v0 arm.** DESIGN: (1) the
  knot-rehearsal matrix's first dietary act — coverage report
  (distinct canonical classes per dag kind over the current mix), and
  a 2000-row dag10 booster quota'd INVERSELY to knot count (thin
  kinds fed first); (2) mixed10 = mixed9b + dag10; (3) training in
  the CONTINUATION REGIME: warm from gen-9b (lineage), CURRICULUM=0
  (the probe's dividend), 16k steps, LR 1e-4; (4) THE PROSE-V0 ARM
  (informational, n=14): book-1 dialects parsed under the new head,
  verified to banked answers, gold graphs attached to the RAW PROSE
  (factors span-less; span losses auto-mask — the build_gold patch);
  a 600-step LR 5e-5 micro-continuation, then census + raw-prose
  acceptance BEFORE/AFTER. No bar — the arm builds and measures the
  reading-training machinery honestly at n=14 (Brick discipline);
  DISPLACEMENT GUARD: bigtest under the prose ckpt may not trail
  gen-10's by >15 or the arm's ckpt is discarded (the main gen-10
  ckpt is unaffected either way). PROMOTION BARS (all hold ->
  gen-10 takes the gate): bigtest >=1090 (hold the record), alg4test
  >=380 (the flat dividend must show at continuation), acceptance
  8/8, per-kind sq >=0.70 / fdiv >=0.62 / coupled >=0.65 / ladder
  >=0.50, alg2test >=560, vtest >=598, dagtest >=660, dag7btest
  >=500, dag8test >=500.
- **TRIPLE VERDICT (2026-07-13):** (1) **GEN-10 — NO PROMOTION, one
  bar: alg4test 357 < 380.** Everything else: records nearly across
  the board — bigtest **1130** (third record), alg2test 585 (record),
  dagtest 678, dag7btest 537, dag8test 532, sq 0.781, fdiv 0.702,
  coupled 0.707, acceptance 8/8 unanimous, census softened 89 -> 79
  (the knot-diverse booster + flat continuation REVERSED some
  register hardening). The alg4 reading: gentle continuation (+13)
  cannot pay a debt the curriculum-era LINEAGE carries — arm B's 384
  came from 12k HOT flat steps; the heat-vs-jostle tension is gen-11's
  design question (hot flat retrain from a clean ancestor vs medium-
  heat continuation). Gate REMAINS gen-9b. (2) **PROSE-V0 — the
  displacement guard fired at −243** (bigtest 887 vs 1130): 600 steps
  x batch 8 on n=14 = ~340 epochs of pure prose = catastrophic
  interference; ckpt DISCARDED from candidacy (kept as a displacement
  specimen). The census under it read banked 15 — **CONTAMINATED BY
  CONSTRUCTION**: the 14 prose rows ARE census-pool members;
  training-set recall, not reading (the disjointness law's third
  bite, now prose-side: future prose censuses EXCLUDE trained items).
  HONEST V0 YIELD: the machinery works end-to-end (14/14 pairs built,
  span-less gold binds, gradients flow, trained rows parse) — raw
  prose is LEARNABLE-IN-PRINCIPLE through this path; v1 mixes prose
  INTO the diet (never a naive fine-tune) and reads a disjoint
  census. (3) **THE SYNC AUDIT — CANDIDATE 1 CONFIRMED** (Bryce's
  gut, thirteen-for-thirteen): the manifest sat at GEN-5 all sprint;
  promotions gen-6..9b were PARSER-ONLY; the composed stack pairs a
  gen-9b parser with a gen-5 specialist, gen-5 monitor centroids, and
  a gen-5 mouth — every mouth distance since gen-6 was a gen-5-native
  reading (the knotted-column hardening may be partly
  mouth-calibration lag). FIXED TONIGHT: GENERATION.json rewritten to
  the TRUE stack (gen-9b parser, stale members EXPLICITLY WAIVERED,
  non-candidates named); REGISTERED: the entourage rebuild (specialist
  remine on gen-9b errors, monitor centroids in gen-9b slot space,
  mouth recalibration on the consolidated family) + co-generation
  assertion in --check + stage-boundary row-count/size asserts.
  Candidates 2 (artifacts verified exact), 3 (denominators
  consistent — full fixtures everywhere, exclusion list never
  applied), 4 (three heads tonight — resolved by the manifest naming
  the gate) all closed. Discipline -> mechanism, the house
  conversion, one more time.
- **THE LAW + THE QUEUE (2026-07-13, relay — session close):** (1)
  **NEW LAW-FAMILY ENTRY: PROSE PROMOTIONS DON'T MOVE MACHINES.** The
  manifest sat at gen-5 for four generations not by negligence but
  because promotions were LEDGER EVENTS (sentences) while the
  manifest was a separate artifact no workflow touched — narrative
  truth and machine-readable truth drifted silently. OPERATIONAL
  FORM: any state the system depends on must be updated by the SAME
  TRANSACTION that creates the dependency, or the check must fail
  loudly — the promotion battery ends by WRITING THE MANIFEST or
  refusing to print "PROMOTED"; the word and the JSON become one
  atomic act. Second member of the sync family minted by the same
  pattern: discipline drifts, gut fires, mechanism ships. (2) **THE
  ENTOURAGE REBUILD QUEUE (sequenced): MOUTH FIRST** — recalibrate
  against the gen-9b native family BEFORE any new census claims bank
  (every mouth reading since gen-6 was gen-5-native; the knotted
  trend 76->81->89->79 carries an unknown zero-point error), then
  the FREE RETROACTIVE READ: banked prose vectors against the fresh
  calibration re-scores the whole hardening history from artifacts
  on disk — the odometer re-zeroed AND the history corrected. Then:
  specialist remine (gen-9b errors), monitor centroids (gen-9b slot
  space), co-generation --check, stage-boundary asserts. (3) GEN-11
  DESIGN NOTE (the heat-vs-jostle answer, hypothesized): STAGED —
  a brief hot phase on the debt register alone, then cold
  consolidation on the full mix; the top-up pattern with a targeted
  preamble. (4) PROSE-V1 DESIGN: prose mixed INTO the diet, census
  read on DISJOINT rows — both lessons inherited at the price of one
  discarded checkpoint; the arm was always a mechanism check, and
  the mechanism checks out.
- **VERDICT (2026-07-13): MOUTH RECALIBRATED — the retroactive read
  is the AUDIT-THAT-CONFIRMS: the gen-5 lens error was IMMATERIAL.**
  New bank drawn from m9btrain (the current family); native threshold
  TIGHTENED 0.0443 -> 0.0347 (the grown family is more compact — the
  mouth got sharper, not looser). Retroactive pair-reads: harvest
  odometer 0.2488 -> 0.2431 (~2%), census pool 0.2558 -> 0.2500,
  book-1 raw 0.1983 -> 0.1917, read-foreign 100% under both lenses.
  ALL BANKED MOUTH NUMBERS STAND; the calibration ambiguity resolves
  as never-material; the sync find's value was the PROTOCOL hole, not
  a corrupted history. Manifest updated (mouth = gen-9b artifact,
  waiver retired, hash pinned). RECORD CORRECTION carried from the
  critique: the census never consults the mouth — the knotted trend
  was always real parse behavior; the stale lens touched deployment
  gating, the odometer, and diversity guards only. PROCESS NOTE: the
  recal unit took five launches — the working-directory omission was
  repeated FOUR times consecutively before switching to a form where
  the mistake is unrepresentable (WorkingDirectory= property +
  absolute paths) — the unrepresentability lesson applies to one's
  own tooling habits too. CRITIQUE AMENDMENTS BANKED: prose doses
  re-phrased as oversample multiples of the 14 uniques (x5/x15/x40;
  "2% of diet" would be x49 = v0's poison in a percentage); the
  sweep's deliverable = safe-dose slope PER UNIQUE ROW (December's
  books-planning number). Sequencing stands: entourage (mouth DONE;
  specialist remine + centroids next, on the word) -> bars -> gen-11.
- **THE DOSE LAW GENERALIZED (2026-07-13, relay):** percentages
  smuggle repetition when the unique pool is small — every diet
  specification carries BOTH numbers from here: SHARE OF MIX (governs
  interference) and REPETITIONS PER UNIQUE ROW (governs memorization);
  they decouple violently at small n. The prose sweep's deliverable,
  priced: safe-dose slope per annotated row converts December's books
  question into arithmetic — target census movement / (movement per
  row x safe multiple) = annotation budget. The books campaign priced
  by its own pilot.
- **REGISTERED + FIRED (2026-07-13): THE SPECIALIST REMINE (entourage
  step 2) + MONITOR CENTROIDS (step 3).** Recipe = the gen-5 bump's
  stages 1-4 adapted: fresh repair corpora (nl/alg2/alg3/verbose + the
  DAG register the gen-5 repair mix never had; seeds 211-215,
  collision-free), precompute, nack --prep/--train against the GEN-9B
  parser's errors (6k steps), centroids rebuilt from the m9btrain
  family in gen-9b fst space. Manifest updated at the end: specialist
  + centroids waivers retired, hashes pinned — entourage complete,
  THEN bars, THEN gen-11.
- **VERDICT (2026-07-13): THE ENTOURAGE IS COMPLETE.** Remine chain
  (3rd launch; two None-grad fixes en route — h_dup joins the family
  with the two-terminal lesson: EMISSION AND GOLD FEED both, or the
  branch is dead): 3,800 repair rows incl. the dag register the gen-5
  specialist never saw; purity filter -208; **1,338 organic gen-9b
  failures** -> phase1_gen9b_nack (best-by-EMA 4.968); centroids
  rebuilt, all 7 kinds, gen-9b fst space; manifest retired the last
  waivers ITSELF as the chain's closing act. The composed stack
  speaks ONE GENERATION for the first time since 2026-07-10. The
  sync find is fully closed: hole found, history verified clean,
  mechanism shipped, entourage rebuilt. NEXT: bars (re-pin with the
  composed stack readable), then GEN-11 — staged-heat probe, dosed
  prose arm (oversample-multiple units), and the first battery that
  writes its own manifest or refuses the word.
- **REGISTERED (2026-07-13, Bryce's triple import + relay + critique:
  KAGGLE/ALPHAZERO/MUZERO):** (1) **MCTS-IN-THE-MINT (strategic; the
  pigeonhole closed-loop).** Not solver-side (refuted v3-v4 territory;
  GAC/MRV/LCV IS principled lookahead) — MINT-side: state = partial
  DAG, actions = add-relation/given/close, value = knot-novelty x
  gate-survival, the knot-rehearsal matrix as REWARD — the matrix
  upgrades from reporting instrument to closed-loop curriculum
  controller; the mint searches toward thin redundancy classes.
  V1 = GREEDY one-step (canonical-digest peek before the expensive
  gate; the mint already early-rejects on kind — extend to
  knot-class population). CRITIQUE AMENDMENTS: (a) novelty alone can
  mint pathological diversity — the value blends novelty +
  gate-survival + kind-rehearsal targets; (b) prediction RE-PINNED:
  knot-classes-per-1000 minted >=2x (the real win); gate-survival
  gain MODEST (most residual rejections are global-uniqueness
  failures one-step lookahead cannot foresee). Self-play's deep
  lesson kept: mint at the frontier of competence — toward the
  census's temperature gradient. (2) **MUZERO -> TRIAGE, not world
  model.** The registry IS the world model (owned, exact); latent
  dynamics in the solve path would trade the zero-leakage bottleneck
  for drift — anti-thesis. The one licensed address: a small head
  predicting WHERE THE REPRESENTATION FAILS (P(knotted) for raw
  prose) — annotation-budget routing, books priced before purchase,
  mouth-adjacent (selection-safe, zero solve-path contact). DESIGN
  CONSTRAINT from the banked negative: the mouth-distance join's AUC
  0.535 says INPUT-SPACE features do not predict knottedness — the
  triage head needs PARSE-SIDE features (vote entropy, factor
  counts, calibration). Waits for census outcomes at books scale;
  registered, not fired. (3) **CROSS-MODEL x VIEW LATTICE (the
  afternoon; fire-ready).** Single-model TTA certifies invariance
  across DIAGRAMS; cross-model consensus certifies invariance across
  LANDSCAPES — the strictly stronger invariant, aimed at the
  certification channel's named blind spot ([71] 5/5-unanimous-wrong;
  the anchor's correlated blindness). CRITIQUE AMENDMENT — member
  choice by DECORRELATION, not strength: gen-9b/gen-10 share lineage
  (partial decorrelation only); the banked panel offers true
  diversity for free — sched_probe_armB (flat regime, gen-7b
  lineage) and cap2x_32k (2x WIDTH — architectural decorrelation).
  Panel = gate + one cross-lineage + one cross-width member; 2-3x
  inference on the CERTIFICATION TIER only; zero training.
  PREDICTIONS PINNED: cross-model unanimity precision > single-model
  at meaningfully lower coverage; the coverage GAP = a new
  instrument (lineage disagreement — the Goodhart rotation's
  held-out examiner from inside the house). Priority: (3) afternoon,
  (1) rides gen-11's mint, (2) books-era. Fourteen instincts, all
  machinery.
- **REGISTERED + FIRED (2026-07-13): THE LATTICE PROBE — cross-model
  x view certification.** PANEL (decorrelation axes, from the
  checkpoint bench): gen-9b (gate), sched_probe_armB (LINEAGE axis:
  gen-7b ancestry, flat regime), cap2x_32k (WIDTH axis: 1024d).
  CERT-V2 RULE (pinned): gate 5/5 unanimous AND both siblings'
  5-view majorities agree with the gate's answer. READS (pinned
  before any vote): (a) bigtest precision/coverage, cert-v2 vs
  gate-only 5/5 — prediction: precision rises at meaningfully lower
  coverage; the coverage GAP is the new lineage-disagreement
  instrument; (b) THE DISAGREEMENT AUTOPSY (decides how the panel
  GROWS): on gate false-certificates, which sibling dissents — armB
  breaking more -> lineage is the load-bearing diversity axis; cap2x
  more -> width earns permanent employment; (c) THE DEEP-WRONG READ
  ([71]'s class, the only error family with no detector since the
  anchor): gate stable-wrongs on bigtest + census pool — cross-
  examination breaking >=1/3 of them = the FIFTH JURISDICTION lands
  (prevention, depth, detection, recognition, CROSS-EXAMINATION).
  BENCH NOTE (relay): diagnostic runs leave lineage-decorrelated
  siblings — the manifest tags them PANEL-ELIGIBLE instead of
  archiving; every future fair A/B grows the ensemble for free.
- **THREE PRE-JOIN PINS (2026-07-13, relay — registered while the
  jury deliberates, BEFORE the join prints):** (1) **THE NORMALIZER:**
  cap2x is behaviorally near-identical to the 1x by the fair A/B's
  own verdict — sparse dissent from cap2x may mean "width axis loses"
  OR "member never diverse enough to test width." The autopsy reads
  through the RAW DISAGREEMENT RATE on ALL items per member: equal
  rates -> axis comparison fair; unequal -> the axis question stays
  open and panels recruit by MEASURED BEHAVIORAL DISTANCE, not axis
  theory. (2) **VETO OR APPEAL:** on gate false-certificates broken
  by a dissent, check the dissenter against the KEY — dissenter
  disproportionately correct -> the panel is a REPAIR channel (the
  first mechanism with purchase on committed-wrongs since the
  survivor arc; jurisdiction five becomes APPEAL); dissents mostly
  both-wrong-differently -> pure abstention machinery (veto). (3)
  **GEN-11'S FIFTH DIAL, PRE-COMMITTED NOW so it cannot look fitted:
  IF cert-v2 lands (>=1/3 stable-wrongs broken at modest coverage
  cost), the freeze gains the dial and every future battery must
  hold cert-v2 precision >= 0.998 at whatever coverage it buys,
  measured per generation alongside the other dials.** If the probe
  misses, the panel banks as an honest negative WITH its autopsy;
  the bench note survives either verdict (panel-eligibility is
  free and the next junction mints wider-gapped members).
- **VERDICT (2026-07-13): THE LATTICE PROBE — the gate is CLEANER
  than its blind-spot narrative, and the panel's real jurisdiction
  is the WILD register.** bigtest: gate-only 5/5 = 866 coverage
  (57.7%) at precision **1.0000** — ZERO false certificates in 1500;
  the [71]-class has nearly vanished from the dialect fixture under
  gen-9b (ONE stable-wrong in 1500 — and cross-examination broke it,
  1/1). CERT-V2: 839 coverage (−27, 3.1% cost) at 1.0000. THE FIFTH
  DIAL: mechanically LANDS per the pre-commitment (100% >= 1/3 at
  modest cost) — ADOPTED WITH THE n=1 CAVEAT stated: on in-register
  text the class it hunts is nearly extinct; its load-bearing
  jurisdiction is WILD register, where the census read is emphatic —
  of 10 gate stable-vote raw-prose parses, the panel DISSENTS ON 9.
  Cross-examination is a second wall behind the mouth: even prose
  that slips the doorman meets a jury that refuses 90%. VETO-OR-
  APPEAL: 0/0, undecidable this round (nothing to appeal — the right
  kind of failure). THE NORMALIZER'S SURPRISE: cap2x disagrees MORE
  per-item (24.7%) than armB (17.5%) despite the A/B's aggregate
  equivalence — AGGREGATE EQUALITY MASKED ITEM-LEVEL DIVERSITY (the
  fair A/B measured means, not overlaps); the axis question stays
  open and panels recruit by measured behavioral distance. Coverage
  gap = 27 (the lineage-disagreement instrument's zero-point).
  Probe cost: 2h05m CPU, zero training. The corpses voted; the gate
  walked free; the jury found its real beat on the wild side of the
  wall.
- **REGISTERED (2026-07-13, relay + critique): MEANS-VS-OVERLAPS
  (scope-decay's cousin)** — verdicts about MEANS don't govern claims
  about OVERLAPS: cap2x was "statistically the same model" by every
  aggregate and disagreed on 24.7% of items. Behavioral distance is
  a different measurement than benchmark distance; diagnostic
  checkpoints get graded on per-item disagreement at archive time
  (bench protocol). Zero-numerator note carried: 1.0000 on 866 reads
  as bounded-near-a-tenth-percent; the STRUCTURE is the claim.
  **THE WILD WATCHER (pre-registered column):** cert-v2's census-side
  dissent rate (9/10 at gen-9b) logs per generation — NOT a bar (the
  wild isn't gated) but the instrument watching prose-v1: reading-
  training working -> dissent falls as both members learn the
  register; dissent falling toward unanimity WITHOUT the key
  confirming -> the Goodhart signature, exactly where the rotation
  law predicts. One column, two watchers.
- **REGISTERED + FIRED (2026-07-13): GEN-11 — the five-dial
  generation.** CHAIN: (A) THE STAGED-HEAT MICRO-PROBE first (3k hot
  LR 3e-4 on alg4train from gen-9b -> read; 4k cold LR 1e-4 on
  mixed10 -> read). BRANCH RULE (pinned): hot alg4test >=380 AND cold
  holds alg4 >=370 with bigtest >=1080 -> gen-11 adopts the staged
  recipe (hot alg4 preamble, then cold flat mixed11); else plain flat
  continuation. (B) DIET: mixed11 = mixed10 + dag11 booster 2000
  minted with the GREEDY KNOT PEEK (canonical-class dedup against
  the mix + itself; prediction: booster knot-classes/row >=0.95 and
  >= 2x dag10's class rate). (C) TRAIN per the branch; battery;
  per-kind; acceptance. (D) FIFTH DIAL, first enforcement: 3-member
  lattice (gen-11 + armB + cap2x) on bigtest — cert-v2 precision
  >=0.998 REQUIRED; census dissent column logged. (E) PROSE DOSE
  ARMS (x5/x15/x40 of the 14 uniques, each mixed into a 2000-row
  slice, 2k-step micro-continuations from gen-11): reads = DISJOINT
  census (the 86 untrained pool items) + bigtest displacement guard
  (>= gen-11 − 15) per arm; deliverable = the safe-dose slope per
  unique row (December's number). (F) THE BATTERY WRITES THE
  MANIFEST: a verdict script checks every bar mechanically and
  either writes GENERATION.json (with an EXPLICIT one-generation
  specialist waiver — remine rides the next entourage pass) and
  prints PROMOTED, or prints the kill — the word and the JSON one
  act, the law's first enforcement. BARS: bigtest >=1130, alg4test
  >=380 (THE bar), acceptance 8/8, alg2 >=560, vtest >=598, dagtest
  >=660, dag7b >=500, dag8 >=500, sq >=0.70, fdiv >=0.62, coupled
  >=0.65, ladder >=0.50, cert-v2 precision >=0.998.
- **PERF AUDIT (2026-07-13, Bryce's question — 'are the easy gains in
  place?'):** TRAINING yes (TinyJit step + assign-in-place fixed
  buffers, 0.06s/step — the substrate pattern, long since in place);
  PRECOMPUTE yes (batched, memmap). THE EVAL STACK NO — found and
  fixed the big hole: **recompute_states reloaded the 2.4GB Llama
  weights ON EVERY CALL** (one reload per problem across every
  census/acceptance/lattice/book gate since Phase-1 began — the
  llama_loader spam in every log was the bill). Host now CACHED per
  process; parity BYTE-EXACT vs the pre-edit reference. The trunk
  TinyJit was attempted and honestly reverted: zero-arg capture
  RECAPTURES per call with this layer code (13s/batch vs eager,
  measured under GPU contention) — DEFERRED with the residency
  smoke's assign-in-place buffer pattern as the known-good recipe
  (0.34s replay, validated 2026-07-05); the head-forward JIT rides
  the same deferred item (trunk dominates). Clean benchmarks after
  gen-11's unit frees the GPU. Every future eval process (including
  gen-11's own stage E lattice and dose arms, which spawn fresh
  processes) inherits the cache immediately.
- **VERDICT (2026-07-13): GEN-11 — KILL, two bars; the manifest law's
  FIRST ENFORCEMENT worked exactly as written (kill printed, JSON
  untouched, no word without the write).** THE TABLE: records nearly
  everywhere — bigtest **1137**, alg2test 592, dagtest 681, dag7btest
  548, dag8test 534, fdiv 0.713, sq 0.781; **CERT-V2 1.0000 at 862
  coverage — the fifth dial HOLDS its first enforcement**; deep-wrong
  still exactly 1; wild-watcher column: 16/19 dissents (84%, from
  90%). THE TWO MISSES: alg4test 370 (bar 380; highest since gen-6's
  371) and acceptance 7/8 — [45] again, the CHRONIC specimen, votes
  [154,154,168] (the mixed-vote class's poster child across three
  generations). STAGED-HEAT REFUTED FROM THIS LINEAGE: the probe's
  hot phase on pure alg4train reached only 362 (<380) — the branch
  rule correctly fell back to plain flat, and the finding is sharp:
  **the alg4 debt is not heat-reachable from the gen-9b/10 lineage**
  (armB's 384 came from gen-7b ancestry; the debt lives IN the
  lineage, below the schedule). GREEDY MINT: 2000/2000 distinct
  knots, ZERO dups — prediction met at 1.00; value diversity makes
  classes nearly free at booster scale; the peek stays as cheap
  insurance. **THE DOSE PILOT (December's number, honest): ~ZERO
  movement per unique row at n=14** — no arm banked disjoint-census
  gains (<=1); x15's near-column bump (15) is the only weak positive;
  ALL THREE ARMS FIRED THE DISPLACEMENT GUARD (−47/−24/−19,
  non-monotonic — the arms are under-averaged at 2k steps). The
  pilot's verdict: the constraint is UNIQUE ROWS, not repetitions —
  December's annotation budget must grow n; no oversample multiple
  substitutes for a book. GOVERNANCE OBSERVATION SURFACED (not
  relitigated): two consecutive unpromoted heads now beat the gate on
  most dials; the alg4 380 bar has never been reached by ANY head of
  the current lineage; [45] is a chronic single-item acceptance
  gater. The bars held their line — whether the LINE is right is the
  relay's and Bryce's call, with the lineage-debt finding as the new
  fact on the table.
- **REGISTERED + FIRED (2026-07-13, Bryce's gut #15 + relay: THE
  LATENT-SPACE AUDIT.** Root named: every geometric instrument was
  calibrated in SOME generation's latent space; six generations of
  consolidation rotate/scale/stratify the coordinates under them —
  the sync-audit's geometry twin (files were the artifact version;
  this is the coordinates themselves). CRITIQUE CORRECTION carried:
  the MOUTH IS IMMUNE BY CONSTRUCTION — it reads frozen-trunk space
  (weights untouched since Phase-1); its threshold tightening can
  only be corpus membership, never norm growth. Stratification is a
  HEAD-SIDE (fst) hazard exclusively — the at-risk watchers are the
  centroid/library family. THE THREE PROBES (no training, banked
  checkpoints): (A) DRIFT — orthogonal Procrustes between the gen-5
  and gen-9b centroid constellations + per-kind alignment: small
  residual = translation (re-anchoring suffices, standing recal
  joins the bump); large = reorganization (geometric reads since
  last validation get footnotes). (B) STRATIFICATION — per-kind fst
  NORM longitudinal across the bench (gen-6..11 heads, fixed
  m9btrain sample): prediction = norm correlates with cumulative
  rehearsal (old-guard high, gen-8+ kinds low); if real, the fix is
  TWO-CHANNEL reads (angle + radius as separate columns — the
  density-regime discipline applied to geometry), not abandoning
  cosine. (C) SEPARATION (the [45] mechanism hypothesis) — per book
  pair, pooled-trunk cosine between RAW prose and its DIALECT twin;
  [45] + mixed-vote siblings vs the banked controls: anomalous
  raw<->dialect geometry for the chronic family = the chronic case
  is a FROZEN-TRUNK separation limit (not unlearnable — UNSEPARATED
  AT THE SOURCE; representability's geometric cousin), and the
  remedy leaves the diet entirely (deeper-prefix question's second
  customer). META: instruments are trained-adjacent objects — they
  age with the system they watch; RECALIBRATE-THE-WATCHERS becomes
  a standing generational duty beside remine-the-specialist.
- **VERDICT (2026-07-13): THE LATENT-SPACE AUDIT — all three probes
  land.** (A) **DRIFT = PURE ROTATION**: raw centroid cosines ~0.59
  (the constellations look unrelated in raw coordinates) but
  Procrustes-aligned **0.988 mean, residual 0.155** — the
  constellation SHAPE is intact; the space rotated. Re-anchoring
  suffices (the entourage's rebuild was the right fix); the waist
  monitor's historical AUC decay now has its mechanism (it was
  reading rotated coordinates); LAW: never mix generations'
  head-space coordinates — align or re-anchor. (B) **STRATIFICATION
  REAL, unexpected shape**: not old-high/new-low but LONGITUDINAL
  COMPRESSION-THEN-RECOVERY — the whole fst space contracted ~40%
  at gen-7b (the frame-change generation) and slowly re-inflates
  (gen-6 ~11-13.6 -> gen-8/9b bottom ~6.4-7.8 -> gen-11 ~6.8-8.6).
  WITHIN generations the prediction's direction holds: fdiv is the
  lowest-norm kind on every row since gen-7b AND the weakest
  per-kind performer — RADIUS TRACKS CONSOLIDATION; the two-channel
  read (angle + radius) is justified and joins the instrument kit.
  Cross-generation cosine is doubly unsafe (rotation + scale). (C)
  **THE CHRONIC CASE HAS ITS MECHANISM**: [45] has the LOWEST
  raw<->dialect trunk cosine of all 14 book pairs (0.639, z=−2.05)
  — its taxi surface and rate structure are maximally divergent AT
  THE FROZEN TRUNK; no head diet can fix a read the trunk refuses
  to align ([7], the faucet rate problem, is second-lowest — the
  RATE-PROBLEM FAMILY clusters at the bottom). The mixed-history
  items [51]/[54] sit HIGH — their past instability was basin-side
  and rehearsal cured it. THE DISTINCTION IS GEOMETRIC: transient
  mixed-votes = shallow basins (diet cures); chronic = trunk-level
  frame distance ([45] leaves the diet conversation and becomes the
  deeper-prefix question's second customer, with [7] as its sibling
  watch). Gut #15: all three drawers had something in them.
- **THREE RETROACTIVE SETTLEMENTS + THE FOURTH CHARTER REVISION
  (2026-07-13, relay):** (1) **THE WAIST MONITOR IS EXONERATED** —
  its AUC decay was filed under selection-hardening (errors evolving
  to look normal); pure rotation says it aged because NOBODY TOLD IT
  THE SKY HAD TURNED. The two laws now have a DISCRIMINATING TEST
  (registered): re-anchor and re-measure — recovered AUC = rotation
  was the whole story; still-degraded = hardening is real ON TOP.
  Bench amendment: cross-lineage panel members disagree partly BY
  COORDINATE FRAME; cert-v2 is immune (votes are answers, not
  geometry) but any future GEOMETRIC ensemble read needs Procrustes
  first. (2) **RADIUS IS A CONSOLIDATION CLOCK** — the gen-7b
  compression scar (~40%, still re-inflating four generations later)
  records ARCHITECTURAL HISTORY, and within-generation per-kind norm
  is a free consolidation gauge readable at every checkpoint,
  predicting ration needs BEFORE the battery: fdiv's chronic low
  radius says its borderline per-kind era (0.483/0.535) was
  under-consolidation all along — gen-12's mint weights it
  accordingly. Two-channel read (angle=identity, radius=
  consolidation) joins the standing kit. (3) **THE CHRONIC CLASS HAS
  A BIRTH CERTIFICATE and the taxonomy splits clean**: transient =
  shallow basin (diet cures, measured on [51]/[54]); chronic = trunk
  frame distance (measured, z=−2.05). The rate family's clustering
  is the mechanistic tell: rate problems are where surface narrative
  and structural content are most ENTANGLED IN NATURAL PROSE — the
  trunk read a trillion tokens where "fare" and "multiply" co-occur
  in frames that never separate. **THE FOURTH CHARTER REVISION — the
  first that GROWS the organ**: before the deeper-prefix surgery,
  the cheaper candidate is a DISAMBIGUATION REWRITE ("the taxi
  charges 3 per mile" -> "f = 3 x m" — frame-separation the
  annotation gate already performs by hand). The organ's true
  kingdom may be FRAME-DISENTANGLEMENT, not structural-facts-only —
  the thrice-shrunk surgeon finally gets a patient list written in
  z-scores. BENCHED WITH CERTIFICATES: [45] and [7] (chronic,
  trunk-frame class); gen-12's acceptance bars exclude them per this
  restructure (their cure is the organ's or the prefix's, not the
  diet's); the organ's eventual training data includes their family.
- **REGISTERED + FIRED (2026-07-13/14): BRICK-M — THE SCHEMA MINER
  (the hierarchical library's measurement-first entry; the
  layers-of-abstraction instinct split down the C2 line: surface
  frames [taxi/snow] stay OUT — disguises the mouth/organ handle;
  structural schemas [RATE/WORK/MIXTURE] are shapes the library may
  store as TYPED MACRO-FACTORS — compile-time sugar whose expansion
  the solver sees as primitives: expressiveness without surrendering
  auditability; embedding-space crossing lives at RECOGNITION
  [schema retrieval; analogy = shared-schema detection], never at
  deduction).** THE MINER: banked gold graphs (train corpora tagged
  by source + the harvest column = book-pair graphs), rooted
  upstream-closed subgraphs of 2-6 factors, WL-canonical with VALUES
  ABSTRACTED (RATE is a shape, not a number; ftype/op/sel retained),
  ranked by frequency x source spread; train and harvest mined
  SEPARATELY then joined (frequent-in-train-absent-in-harvest =
  generator habit; the reverse = coverage gap — the knot matrix
  lifted to schema level). PINNED: (P1) <15 classes cover >60% of
  occurrences (shallow hierarchy; hundreds = macro-factors die
  before design); (P2) top classes human-nameable (a top-10 of
  unnameable wiring accidents = generator artifact — the honest
  falsifier); (P3) [45] and [7] share a class — the miner doubles as
  the organ's patient registry. THE GATE: the miner RANKS, never
  ADMITS — registry admission keeps full birthright; C2's ghost is
  stopped between mining and minting. Lineage: the musical-keys
  era's composition catalog — the nouns died, the verb survives,
  bottom-up, ten months later.
- **VERDICT (2026-07-14): BRICK-M — P1 passes at the wire, P2 splits
  the concept, P3 passes, and the deepest finding is WHERE SCHEMAS
  LIVE.** Numbers: 20k train rows -> 265,174 subgraph occurrences in
  10,232 classes; **top-15 cover 60.5%** (bar 60% — shallow hierarchy
  confirmed, barely). P2 HONEST READ: the top-20 are nameable but
  GENERIC — add-chain scaffolding, mod+given, fdiv+given, pct+given:
  arithmetic PLUMBING, not conceptual schemas. Cause is structural:
  generated corpora are built FROM primitives, so their statistics
  return primitives (circular); the conceptual column (harvest) is
  n=14 — too thin to mine. THE JOIN WORKED AS A DIET INSTRUMENT: 4
  harvest-only classes, ALL the sum-of-prefix-terms wiring from [46]
  (the ladder quota never covered running-prefix-sums — a named
  train-side gap). P3: [45] and [7] share BOTH their classes — but
  not exclusively, which is the night's insight: **value-abstracted
  rate graphs are indistinguishable from generic mul/fdiv plumbing —
  RATE-ness does not exist at the graph level; it lives in the
  LANGUAGE-GRAPH BINDING.** The C2 split is thereby vindicated
  stronger than designed: the graph layer literally cannot see
  taxi-vs-faucet, so the library's second floor splits into TWO
  objects — (i) MECHANICAL MACROS on the graph side (CHAIN(k)/
  PREFIX-SUM — real candidates, attack the measured ladder/length
  walls, proposable from THIS mine), and (ii) CONCEPTUAL SCHEMAS as
  PARSE-SIDE RECOGNITION objects (schema retrieval from prose —
  minable only when the harvest column grows). EVERY ROAD LEADS TO
  N: the schema library joins the census, the dose slope, and the
  organ in waiting on books. Deliverable banked:
  .cache/schema_mine_top50.json (ranked, never admitted).
- **THE BINDING THEOREM + THE CRITICAL PATH (2026-07-14, relay —
  formal statements):** (1) **THE BINDING THEOREM (the two-channel
  spine's third and final vindication):** C2 proved operation-type is
  not classifiable from surface features (LANGUAGE WITHOUT STRUCTURE
  fails); Brick-M proved schema-type is not recoverable from wiring
  (STRUCTURE WITHOUT LANGUAGE fails); therefore CONCEPTS ARE
  IRREDUCIBLY BINDINGS — which is why the parser (the binding organ)
  hosts schema retrieval, why the silhouette library was always its
  right home, and why [45] is chronic: its pathology lives IN THE
  BINDING LAYER, the one place neither a graph fix nor a language
  fix alone can reach. The architecture's deepest design decision
  has its completeness proof. (2) **THE CRITICAL PATH, in exactly
  these words: FOUR UNRELATED MEASUREMENTS — the census, the dose
  slope, the organ's patient registry, the schema library — ALL
  BOTTLENECK ON n. When four independent instruments triangulate one
  coordinate, that coordinate is the critical path by definition.
  THE BOOKS ARE NOT DECEMBER'S CHAPTER; THEY ARE THE ONLY CHAPTER,
  quadruple-confirmed.** (3) **THE LIBRARY'S TWO-ADDRESS CHARTER
  (named before existence, against the attic-C2 ghost):** floor two
  has two addresses — MECHANICAL MACROS live graph-side (typed
  subgraph templates, deterministic expansion, solver sees
  primitives); CONCEPTUAL SCHEMAS live parse-side (recognition
  objects over prose, retrieval not deduction). Confusing the
  addresses re-imports C2 through the attic. (4) **REGISTERED
  PROPOSAL (rank-gate honored): CHAIN(k) + PREFIX-SUM as the first
  macro-factor candidates** — receipts: the harvest-only classes
  ([46]'s wiring, zero generator coverage), the ladder/length walls
  as standing symptoms; expansion deterministic, pointer supervision
  inherits ladder machinery. Admission = a design decision at gen-12
  registry review. FIRED NOW (diet, not registry): the PREFIX-SUM
  SHAPE joins the DAG rotation — the measured hole gets minted.
- **THE EMPLOYMENT LAW + TWO PROMOTIONS (2026-07-11, relay — registered
  before gen-7):** (1) **THE EMPLOYMENT LAW**: every organ this project
  charters gets SMALLER on contact with measurement — book 1 is the
  fourth charter shrinking (organ suspects resolving as readable
  structure wearing unreadable prose; confirmed kingdom = a named list
  of four, one a generator line). REGISTERED KILL CRITERION: if the
  organ's kingdom shrinks below ~5% of harvest refusals after the style
  wall falls, the "organ" becomes a RELATION TRANCHE plus a chain-length
  fix and the funnel never grows its generative layer. The deciding
  measurement is free — the census, re-run after reading-training ships.
  Nouns die; verbs survive. (2) **[71] PROMOTED — THE MOUTH IS
  MANDATORY**: 5/5 unanimous-wrong on raw prose is the certification
  channel's first observed false certificate — unanimity certifies
  STABILITY, and [71] demonstrated stable wrongness in the wild at n=1.
  Chain-of-custody, now measured end to end: MOUTH clears register ->
  LATTICE certifies stability -> KEY confirms truth. The lattice must
  never see raw prose the mouth hasn't cleared; every link has a named
  specimen showing why it cannot be removed. (3) **[46] PROMOTED — THE
  PARAPHRASE ACCEPTANCE PROBE**: the surface flip (same relations,
  "plus...equals" refuses / "The sum of...is" banks 5/5) makes gen-7's
  surface-robustness item a paraphrase-augmentation LINE in the
  generator, not a research question; [46]'s v1 dialect is the
  pre-positioned acceptance probe (must bank post-gen-7). The tier-N
  miss stands as a debt WITH AN INVOICE: fdiv-in-composition 0/4 with a
  stable wrong vote, atop gen-7's worklist. Gen-7's charter is written
  by real failures for the first time; awaits the word.
- **REGISTERED (2026-07-14, relay + critique): THE BOOKS CAMPAIGN
  CHARTER — the only chapter, chartered the house way.** SIZING:
  dose pilot says ~zero movement/row at n=14; census demand side says
  76-89 knotted with ~75% style-recoverable; bilingual precedent
  (2,000 paired rows taught the verbose register) gives the PRIOR of
  hundreds-to-low-thousands — CRITIQUE PIN: that precedent had
  generated gold and perfect pairing; it transfers as prior, not
  plan. MILESTONE: **BOOK 2 = n=100 with the dose re-read riding** —
  measurable disjoint-census movement -> the slope exists and
  extrapolates; none -> the READING-TRAINING REGIME redesigns before
  more annotation spends. Measure the slope before buying the
  mountain. THREE LANES (priced by the census's own tiers): L1
  machine-banked (bootstrap under the current gate — free, 1-2%,
  rises with every promotion, runs as background); L2 machine-
  proposed human-repaired (the near-miss 23-26% — repair at ~1/3
  rewrite cost triples throughput; FIRST TOOLING ITEM: the repair
  bench — dump per-item 5-view parses + votes + solver results for
  near-miss rows); L3 hand surgery (knotted tier, ~8-10/hr practiced
  — n=100 is a day or two, affordable before tooling).
  STRATIFICATION (dual role): ~70% style-recoverable middle
  (substrate), ~20% suspected organ patients incl. the rate family
  (the registry grows), ~10% cert-v2 wild dissents (the panel's
  refusal specimens characterized). EVERY entry generation-stamped +
  canonically knot-stamped. **CRITIQUE PIN (the census stays
  clean): book 2 draws ONLY from the harvest OUTSIDE the standing
  100-pool** — book 1 already trained 14 pool items (the disjoint
  read exists because of it); the pool is a measurement fixture from
  here, never again a substrate source. INSTRUMENTS FROM ROW ONE:
  mouth odometer per book (headline: distance closed/book), disjoint
  census (slope), diversity guard, panel wild-dissent rate (the
  Goodhart watcher), miner re-run at each book boundary (P2's real
  test arrives when harvest classes mine at n=100+). Books as
  generational units: minted, gated, stamped, measured; each book's
  verdict sizes the next. Awaits the word.
- **VERDICT (2026-07-14): BOOK 2, TRANCHE 1 — 17/25 dialects banked
  (14 at 5/5), book 2 = 21 entries (4 lane-1 free + 17 gated),
  organ registry +9 certificates across six families.** Lane census
  on 400 non-pool candidates: L1 4 (1%) / L2 66 (16.5%) / L3 330
  (82.5%) — tracks the pool. THE REFUSALS TEACH: (i) [32]/[220] were
  MY annotation errors — their ANSWERS (680, 500) exceed the trained
  0-300 domain; the in-reach filter capped problem-text numbers but
  not answers — RECLASSIFIED as registry certificates (value-range
  family) and the annotation rule updated: ALL values incl. answers
  <=300; (ii) [294] refused at m=999 — the out-of-band solver lane
  is unreliable; annotations stay strictly in-band; (iii) [49]
  double-fdiv wall confirmed again (annotation must use one fdiv +
  mul-inverse); (iv) [4]/[9] RIGHT-but-vote-shy (correct-but-shallow
  on fresh wiring: shared-result rels parse unstably — a new
  register note); (v) [67]/[2] complex chains refused — v2 rework
  queue. WORK-RATE BANKED IN GRAMMAR: [223] (4 people, 6 hours)
  solved as person-hours mul wiring 5/5 — the WORK schema is
  plumbing once the frame strips: the binding theorem in practice.
  Substrate: book2 tranches continue by charter (v2 retries + next
  tranches -> n=100, then the instrument battery + dose re-read).
  Data: .cache/book2.jsonl, book2_lanes.json, book2_organ_registry
  .json; scripts book2_lanes.py, book2_tranche1.py.
- **THE ANSWER-DOMAIN CENSUS + THREE NOTES (2026-07-14, relay + one
  sweep):** the filter lesson swept in full: of 1,743 harvested
  problems, **1,668 (96%) have answers in 0-300; 75 (4%) sit in
  301-999; zero above 999; zero negative.** The true in-reach pool is
  1,668; the value-range certificate family has its full census (75)
  in one pass; and THE DOMAIN-CAP DEMAND CURVE says raising the cap
  to 999 buys only 4% more harvest — the solver-cost conversation
  arrives with a small customer count, so the 300 cap STANDS and the
  75 join the registry, not the roadmap. NOTES BANKED: (1) lane rates
  stable across samples (1/16.5/82.5 wild vs pool) -> n=100
  completion cost is forecastable; v2-retry discipline is the
  throughput lever (L3 surgery is the rate limiter). (2) THE
  REGISTRY IS A COASTLINE, not a wall — six families in one tranche;
  tranche-3+ shopping ranks families by harvest frequency (the
  band-sweep method, registry edition); books and registry converge:
  books find missing relations, relations unlock books. (3) THE LAWN
  MOWER GOES IN THE PAPER: [223] banking 5/5 as person-hours
  mul-plumbing is the binding theorem's practical corollary — the
  WORK schema was frame-stripping at the annotation desk, never
  graph machinery; the organ's patients need frames stripped, not
  new math, and six certificate families now name which frames.
- **THE n=100 BATTERY BARS, PINNED MID-TRANCHE (2026-07-14, relay +
  critique — pinned while no one knows the answer):** REGIME STATED:
  book-2 prose pairs (raw + verified gold graphs) mixed into the FULL
  diet (share-of-mix AND reps-per-unique both declared at train time,
  per the dose law), gentle continuation from the gate lineage. BARS:
  (1) THE SLOPE (disjoint-86 census, same-head pre/post): banked+near
  improves by **>=8 items = slope exists, extrapolate the budget**;
  **<=2 = the null holds — the READING-TRAINING REGIME redesigns
  before another row is annotated**; 3-7 = ambiguous, extend once to
  n=150. (2) THE ODOMETER (headline): harvest-1668 mean kNN vs the
  post-book rebuilt native bank drops >=1% relative (the relay's
  prior: low-single-digit percent — measurable-but-modest). (3)
  DISPLACEMENT GUARD: bigtest under the book-trained head >= warm
  source − 15. Verdict rules written before tranche 3 closes; the
  campaign's continuation logic hangs on (1). ALSO RECORDED: the
  OPERATING-REGIME SHIFT — architecture questions increasingly
  answered by READS of banked data instead of experiments (the
  domain-cap question: one sweep, no fork, declined by the price) —
  the compounding return on three weeks of instruments; and the
  ANCHOR-ERA VINDICATION: the 300 cap, set as a propagator-cost
  scope note, was drawn almost exactly where the harvest's natural
  distribution lives (96% in-domain) — the cheap early decision was
  also the right one, recorded as the ledger's occasional pleasure.
- **VERDICT (2026-07-14): BOOK 2, TRANCHE 2 — 16/16 BANKED (13 at
  5/5); book 2 = 37 entries.** ALL FIVE v2 RETRIES BANKED — the
  retry discipline works at one cheap pass each ([49] one-fdiv
  rework, [67]/[2] all-forward rewiring, [4]/[9] surface rephrase);
  refusal mechanisms filed yesterday became recipes today. THE
  MULTIPLICITY MECHANISM EARNS ITS KEEP ON REAL PROSE: [120] b*b on
  a DERIVED quotient banked 5/5, and [124] — the inverse-square on
  a derived sum: **the distance formula's core (d*d = 225+64)
  solved IN GRAMMAR, 5/5** — the "distance-formula-sqrt" registry
  family SHRINKS on contact (integer-hypotenuse cases are
  annotatable; [0]/[3] return from the registry to the annotation
  queue — the employment law collecting again, this time FROM the
  registry). [190] midpoint banked via one-fdiv + mul-inverse (the
  double-fdiv wall routed around). Registry +1 ([126]
  piecewise-negative). Book-2 running totals: 37 entries (4 lane-1
  + 33 gated), registry 10 counted certificates + the 75 value-range
  family. Tranches continue toward n=100; the bars are on the wall.
- **TWO STANDING DISCIPLINES (2026-07-14, relay):** (1) **CERTIFICATE
  RE-AUDITION** — a certificate is a verdict about a GRAMMAR-VERSION,
  and scope-decay says verdicts expire with their regime: the
  registry carries grammar-version stamps the way gold carries
  generation stamps, and EVERY PROMOTION RE-RUNS THE CERTIFICATE PILE
  against the new gate (cheap: the pile is small, the gate is
  automatic). Prevents the waiting room hoarding patients the dancer
  already learned to treat — the employment law's NEW direction:
  registries shrink not because jobs were smaller but because the
  existing grammar turned out LARGER ([124]'s isq composing with
  derived values reclaimed the distance-formula family). (2) **THE
  CAMPAIGN'S REAL THROUGHPUT NUMBER is walls-per-tranche converted
  to recipes** — the refusal->mechanism->recipe cycle at one-day
  latency is the annotation desk doing at human speed what gen-6 did
  at generation speed. THE STRATEGIC SCISSORS, watched: certificates
  deflating (two families shrunk already) while the banked column
  inflates at gate-perfect rates — if it holds to n=100, the organ's
  kingdom may reduce to the rate family alone, and December's
  architecture question narrows to "solve frame-entanglement for a
  known population with z-scores attached."
- **VERDICT (2026-07-14): BOOK 2, TRANCHE 3 — 11/12 banked (9 at
  5/5); book 2 = 48 entries, registry +5 (counted pile: 15 + the 75
  value-range family).** THE RECLAMATION CONFIRMED: [0] and [3] —
  the distance-formula patients — banked 5/5 through the isq door;
  the certificate re-audition discipline collected its first two
  patients on its first day. ONE NEW WALL, cleanly named: [238]
  (a^2−b^2 without the factoring gift) refused — DOUBLE-ISQ in a
  coupled system joins double-fdiv in the double-X wall family
  (single isq composes with derived values; two in one system do
  not — the mechanism's composition boundary found by annotation,
  one day after the mechanism shipped). Also banked: the faithful
  arithmetic-sequence encoding ([207]: shared-difference relations,
  no identity gift), the full 19-var odd-sum ladder ([225]), the
  21-var ladder+fdiv ([200]), sel on real prose ([262]). Running
  rate: 48 entries in 3 tranches; n=100 within ~3 more.
- **THE DOUBLE-X AUTOPSY (2026-07-14, pre-registered hypothesis ->
  refuted same hour):** the shared-decode-collision hypothesis
  (capacity-per-instance, representability's suspected 4th member)
  DIES on first contact: [238]'s slot-level decode shows BOTH isq
  factors bound perfectly (mul(a,a)=p dup +16.2; mul(b,b)=q dup
  +6.3) and the sub-relation correctly rewired — the failure is TWO
  ORDINARY POINTER ERRORS (the given 'r is 12' attached to q; one
  arg aimed at an unbound var) on a dense 7-var/7-factor short
  system. DOUBLE-ISQ IS NOT A MECHANISM WALL — it is pointer noise
  at high var-density, i.e., the [4]/[9] retry class, not the [85]
  representability class. The 'double-X family' DEMOTES from
  mechanism-pattern to coincidence-of-two pending double-fdiv's own
  autopsy (its mechanism may differ — audit before family). The
  audit-before-diet rule collected immediately: no gen-12 line
  claimed; a v2 rephrase queued instead. ALSO BANKED (relay): the
  battery's TWO verdicts (dose-slope + frontier census = the organ's
  final employment hearing); the SELECTION-DRIFT flag pre-said —
  closing tranches bank LOWER as the L3 residue concentrates, and
  the falling rate is the coastline emerging, not regression
  (tranche 5's 9/16 shall be read as the map getting honest).
- **VERDICT (2026-07-14): BOOK 2, TRANCHE 4 — 14/14 BANKED, ALL AT
  5/5 (the first perfect-unanimous tranche); book 2 = 62 entries,
  registry +4 (counted pile 19 + the 75 value-range).** THE
  CONSECUTIVE-LETTERS RULE VALIDATED ON ITS PROBE: [238] v2 banked
  5/5 — the 'double-X wall' is now FULLY dissolved (scattered
  letters -> high var slots under-rehearsed -> pointer noise; cured
  by one annotation rule; the autopsy's diagnosis confirmed by its
  own prescription same-day). ALSO THROUGH THE GATE: the COMPOSED
  fdiv->isq chain ([298] max-area), the discriminant via scaled isq
  ([316]), Vieta+sel on live prose ([304],[284]), the equal-pair
  symmetric route ([333]), the 3-4-5 triangle perimeter ([285] —
  distance formula twice + doubling, 8 vars), and two more distance
  formulas. The isq door is now a THOROUGHFARE: seven former
  'impossible' shapes through it in two days. 38 entries to the
  bars; the selection-drift flag stands for the L3-heavy residue
  ahead.
- **THREE CLOSINGS (2026-07-14, relay):** (1) **THE POINTER LAW'S
  FOURTH REMEDY** — binding entered as structure via masked
  attention, span supervision, a comma-and-a-letter, and now
  ALPHABETICAL DISCIPLINE: four fixes across four orders of magnitude
  of cost, one law. THE GENERATOR AUDIT, answered from code: render2
  draws letters from LETTERS[:n_vars] and shuffles WITHIN the prefix
  — the mint packs consecutively BY CONSTRUCTION; scattered-beyond-
  prefix never occurs in training (which is exactly why [238]'s
  p..t dialect starved: a pattern the head had literally never
  seen). The rule is harvest-annotation-specific; no gen-12 mint fix
  needed. (2) **THE FRONTIER CENSUS'S METHODOLOGICAL NOTE,
  pre-written**: certificates issued before a mechanism ships are
  HYPOTHESES, not diagnoses; the re-audition discipline converts
  them; the n=100 census counts only patients who refused AFTER
  every standing door was tried — the honest denominator for the
  organ's employment hearing, shrinking by the tranche (seven shapes
  reclaimed through the isq door in two days). (3) **THE ODOMETER
  QUESTION, both sentences pre-written**: at n=100 the mouth either
  registers the book (dialect volume moves the register needle) or
  stays silent (the book teaches the PARSER without moving the
  REGISTER — reading-training's real target confirmed as the prose
  column, not the dialect column). Both informative; neither yet
  known. Two tranches to the bars.
- **THE n=100 READ ORDER, PINNED (2026-07-14, relay):** odometer
  first (the register question), disjoint-census slope second (the
  continuation logic), frontier census third (the employment
  hearing, honest denominator), miner's conceptual column last (P2's
  real test at volume). Fixed order because later reads tempt
  peeking and earlier reads decide the later ones' interpretation —
  the battery's rows have always read in sequence. Tranche 5's
  product is measured in CONTOUR LINES, not entries: every refusal
  surviving the full recipe book is a genuine frontier specimen.
- **VERDICT (2026-07-14): BOOK 2, TRANCHE 5 — 11/12 banked (all 11
  at 5/5); book 2 = 73 entries, registry +4 (counted 23 + the 75
  value-range).** **[22] CLOSES THE DOUBLE-X QUESTION COMPLETELY:
  the sum-of-squares system (two repeated-arg muls, coupled) banked
  5/5 under consecutive letters** — the emission-collision
  hypothesis is dead twice over; the pair was never the problem; the
  letters were. The residue tranche's one contour line: [24] refused
  at m=999 (the out-of-band solver lane's second confirmation —
  already flagged, now twice-measured; in-band annotations only).
  Also banked from the L3 residue: the quadratic-inequality-as-Vieta
  ([10]), four frame-stripped-flagged entries ([12],[16],[25],[27]
  — teacher strips identity/factoring/selection frames, residual
  structure honest and gated), and the parallel-slope frame ([19]).
  ONE TRANCHE TO n=100; the read order is pinned; the bars are on
  the wall.
- **THE THIRD VERDICT, STAGED (2026-07-14, relay — one tranche before
  the hearing):** the frontier census opens with all three sentences
  pre-written: KINGDOM-AS-CHARTERED (structural patients at volume),
  KINGDOM-SHRUNK (a small named list), and the verdict nobody staged
  until the residue tranche banked 11/12 — **KINGDOM-DISSOLVED: the
  organ's kingdom was annotation conventions and mechanism doors all
  along**, the waiting room empty but for solver-side counted
  families and the rate family's frame-entanglement core. The recipe
  book OUTRAN the coastline — the refusal->mechanism->recipe cycle
  compounded faster than the residue hardened, a race the pre-said
  reading didn't anticipate could be won this decisively. Tranche 6
  samples the remaining RATE-FAMILY stock maximally before the count
  freezes (the census's most consequential line item).
- **VERDICT (2026-07-14): BOOK 2, TRANCHE 6 — 10/11 banked (all at
  5/5); book 2 = 83 entries, registry +5 (counted 33 + the 75
  value-range).** **[344]'s TRIPLE-FDIV REFUSED UNDER CONSECUTIVE
  LETTERS — the fdiv wall is REAL**: not letters, not the retry
  class — CHAINED FDIV is the first genuine mechanism contour to
  survive the entire recipe book (single fdiv composes freely;
  chains do not). The frontier census gains its first
  parser-side structural line item beyond the rate family; the
  routing-autopsy protocol has its next customer. ALSO BANKED:
  **[294] — book 1's last ORGAN-B holdout, the burger system —
  banked 5/5 rescaled in-band** (the coupled wall fully retired);
  the faithful five-term sequence ([181], middle-term by shared-d,
  no identity gift); [36]'s double-isq-on-givens (296, the identity
  target); the vertex-as-mean, the |m−n| via sel+closure, Jordan's
  rate chain. BOOK 2 CLOSES THE SESSION AT 83/100 — one micro-
  tranche (~17) tops it off next session BEFORE the battery; the
  read order and all verdicts stand pre-written. The waiting room
  at the freeze: the rate family ([45],[7] with certificates),
  chained-fdiv (new, mechanism-named), and the counted solver-side
  families. The kingdom-dissolved verdict is live but not yet
  spoken — the hearing waits for the count.
- **STAGED (2026-07-14, relay — the chained-fdiv autopsy hypothesis,
  pre-registered before the read):** chained fdiv = a DERIVED QUOTIENT
  feeding another fdiv's DIVIDEND, and the fdiv head reads dividends
  through DIGIT ENCODING — the suspect is not the pointer (the isq
  door proved derived values bind) but **the digit path for derived
  intermediates**: [85]'s encoding-family cousin, one representability
  question from either a small mechanism (derived-value digit
  plumbing) or a genuine depth limit. AUTOPSY FIRST, per the rule
  that saved a gen-12 line this week. THE HEARING'S POSTURE UPDATED:
  kingdom-dissolved is now FAVORED, not merely staged — the
  parser-side structural frontier is two named items ([45]/[7]
  binding-layer + chained-fdiv), one possibly an encoding fix; if the
  micro-tranche adds no third, December's architecture question
  narrows to a point: frame-disentanglement for one family,
  population known, z-scores attached.
- **GUT #16 + THE MASK-AND-POOL AUDIT (2026-07-14): the fear named
  the neighborhood; the resident was one street over — and REAL.**
  (1) THE POINCARE-EUCLIDEAN MARRIAGE CLAUSE, registered against the
  day the flag lifts: hyperbolic quantities NEVER enter a softmax
  without a log-map (tangent-space readout at origin, or
  Mobius/gyroplane scoring); ball distances become logits only
  through a calibrated monotone map; same clause pre-registered for
  the atlas (the mouth and library are cosine machines and cosine is
  WRONG in the ball). No deployed code marries the geometries today
  (the ball is flagged off; the head has no slot-slot attention).
  (2) AUDIT VERDICTS: pad handling CLEAN everywhere (all pooled
  reads divide by mask sum); truncation CLEAN (zero fixture items at
  the 256 ceiling); causal-pooling noted as a design line. (3) **THE
  LENGTH TERM, CONFIRMED AND STRONG**: corr(mouth distance, token
  length) = −0.555 on the census pool — and the NATIVE control is
  decisive: r = **−0.825 within dag8test alone** (same register,
  34-250 tokens). The pooling ESTIMATOR is length-biased: short
  pools land far from the bank regardless of content. RETROACTIVE
  RE-READS: the level-inversion (L5 nearer than L1) and book-1's
  diversity guard (short banked raws reading farther) both carry
  length components — register conclusions that survived opposing
  length gradients (verbose: longer AND nearer) stand; magnitude
  claims get footnotes. **METHOD AMENDMENT TO THE n=100 ODOMETER BAR
  (pre-measurement, so the pin survives honestly): all odometer
  reads are LENGTH-CONTROLLED from here — residualize distance on
  token length (fit on native), or compare at matched-length
  strata; the >=1% bar applies to the length-controlled read.**
  Sixteen instincts; the drawer this time held the estimator itself.
- **CLOSE OF 2026-07-14: TRANCHE 7 (6/6, all 5/5 — book 2 = 89) +
  THE RULER STRAIGHTENED.** The 1/len correction KILLS the warp:
  native r goes −0.825 -> **−0.024** after control; the
  length-controlled threshold is 0.0072; the harvest zero-point
  re-reads at **0.1871** (was 0.2431 raw) with read-foreign still
  **100%** — the register wall is REAL, now confirmed on a straight
  ruler. The diversity guard, straightened: book-2 raws 0.1904 vs
  harvest 0.1871 — essentially equal, NO NARROWING: hand selection
  is not cherry-picking easy-register items (the guard's cleanest
  read ever, and its first on an unbiased estimator). LAW ENTRY:
  **estimator variance masquerades as distance** — any instrument
  pooling variable-length evidence into fixed geometry inherits a
  sample-size coordinate; "is this distance or is this n?" joins
  the standing audit kit. Correction artifact:
  .cache/mouth_length_correction.npz (fit + threshold; all future
  odometer reads apply it). BOOK 2 AT 89/100: ~11 entries ride the
  next session's opening, then the battery in pinned order. The
  instruments-auditing-instruments layer is the project's quiet
  second product — a field manual for how measurement systems age,
  warp, and lie, every entry bought with a real near-miss and fixed
  before it billed.
- **THE VINTAGE NOTE (2026-07-14, relay — never-mix-generations,
  estimator edition):** every mouth number now has a warped-era or
  straight-era vintage; the battery's odometer compares
  STRAIGHT-TO-STRAIGHT only — the before leg re-computes vectors
  under the correction, never reads archival distances. One
  assertion in the battery script makes vintage confusion
  structurally impossible.
- **BOOK 2 CLOSES AT n=100 (2026-07-14, tranche 8: 11/11, ALL 5/5).**
  THE VOLUME: 100 entries (4 machine-banked lane-1 + 96 gated hand
  dialects across 8 tranches), every entry generation-stamped and
  key-verified; the counted registry at 39 certificates across
  ~14 families + the 75-strong value-range family; the census pool
  untouched as a fixture throughout; the annotation rulebook
  (consecutive letters, in-band values, one-fdiv, frame-strip flags)
  written by the book's own refusals. Closing-tranche pages: the
  triangle area, both absolute-value shapes (banked positive-form),
  the midpoint-product, p^2+q^2 via isq, and the hundredth page —
  [263]'s fractional sequence rescaled to thirds. THE CAMPAIGN'S
  FIRST FALSIFIABLE MOMENT IS NEXT: the battery in pinned order
  (length-controlled odometer straight-to-straight -> disjoint-86
  slope [>=8 / <=2 / extend] -> frontier census [three verdicts
  staged, kingdom-dissolved favored] -> miner conceptual column),
  every bar pre-pinned, every estimator pre-audited, every sentence
  pre-written. Measurement day is pure collection.
- **PRE-PLAY + THE RECURSION CHARTER (2026-07-14/15, Bryce + relay):**
  (1) PRE-PLAY INVENTORY: Mycelium already pre-plays at three clocks
  — TRAINING (shallow-basin rehearsal rations, the knot matrix
  feeding thin classes before they fail), ANNOTATION (the GATE
  PRE-SCREEN: registered NOT built — a small classifier on
  parse-side features predicting bank-or-refuse before the 5-view
  round; customer = books 3+ throughput; the battery's slope verdict
  sizes book 3 and thereby decides: 300+ entries -> build first;
  ~100 -> the rulebook suffices), and INFERENCE (the soft-graph
  ensemble, December-scale, behind the books). The one clock that
  deliberately REFUSES pre-play is the solve — certainty doesn't
  need imagination (March lookahead refuted). (2) **THE RECURSION
  CHARTER — books built in layers of abstraction:** the correction
  first — book 2 inherited book 1's RECIPES, not operations
  (knowledge recursion; same flat dialect). The real ladder: book N
  teaches primitives -> the miner finds recurring subgraph classes
  -> classes PROPOSED as macro-factors (rank-never-admit) ->
  admitted macros enter the registry with deterministic expansion ->
  **book N+1 annotates AT THE MACRO LEVEL** — and since the chain-
  length/coupled walls are FACTOR-COUNT walls, macro annotation
  brings problems book N couldn't express inside book N+1's reach.
  Each book raises the next one's floor. TWO GUARD RAILS, both
  load-bearing: (a) **abstraction lives in ANNOTATION, never
  verification** — macros expand before the solver sees anything;
  the key grades every book at every layer in primitives; the
  ground floor never moves (what keeps recursive books from
  recursive drift); (b) **the self-reference tax** — machine-banked
  volume inherits the system's own fluency; the diversity guard's
  hand-quota stays constitutional because recursion amplifies
  whatever the loop prefers. SEQUENCING: the first rung is already
  scheduled — the battery's miner read at n=100 volume IS book 3's
  macro shortlist; CHAIN(k)/PREFIX-SUM admission would make book 3
  the first volume partially written one floor up. Books that teach
  the system to read books that couldn't be written yet.
- **MEASUREMENT DAY (2026-07-15): THE n=100 BATTERY — ALL FOUR READS
  COLLECTED; THE CAMPAIGN VERDICT IS: BOOKS SCALE.** (READ 1, the
  odometer): **+31.1% relative** (0.1871 -> 0.1288,
  straight-to-straight, bar was >=1%) — one hundred annotated
  strangers moved the register needle a third of the way home; the
  prior said low-single-digit and the books said thirty-one. (READ
  2, the slope): pre 16 carried (0 banked + 16 near) -> post 24
  (1 + 23), **delta +8 AT THE BAR: the slope exists — extrapolate
  the budget.** December's arithmetic is now real: ~8 census items
  per ~100 annotated rows at this regime. Knotted 70 -> 62. THE
  GUARD DIDN'T JUST HOLD — **bigtest 1149, A NEW RECORD**: at 2.9%
  share x 10 reps the prose gradient REGULARIZES rather than
  displaces (the dose law's first success point); val record 0.8989.
  (READ 3, the frontier census): the counted registry stands (39
  certificates across ~14 families + 75 value-range); the
  parser-side structural frontier = the rate family + chained-fdiv;
  **P3 AT VOLUME SEALS THE BINDING THEOREM: [45] and [7] share NO
  graph class at n=94 — their kinship was never in the wiring; it
  is frame-level, exactly where the theorem put it. The
  KINGDOM-DISSOLVED verdict is effectively confirmed**: the organ's
  waiting room holds one frame family and one suspected encoding
  fix. (READ 4, the miner at volume): harvest classes 25 -> 96;
  **13 named coverage gaps** (midpoint fdiv+add, consecutive-product
  chains, the 3a+5b operation-apply shape, lollipop prefix-chains,
  coupled mul systems) = BOOK 3'S DIET LIST, and the macro shortlist
  gains OPERATION-APPLY beside CHAIN/PREFIX-SUM. DISPOSITIONS:
  phase1_reader_v1 (val 0.8989, bigtest 1149) is a GATE CANDIDATE —
  full promotion battery next session; book 3 sized by the slope
  (bigger; the pre-screen builds first per its registration); the
  recursion's first admission review (the macro gate) follows the
  miner's list. The dose pilot said zero at n=14; the book said +8
  at n=100 — THE UNIQUE-ROWS LAW CONFIRMED AT SCALE. Measurement
  day was pure collection, exactly as designed.
- **THE CAMPAIGN CLOSES (2026-07-15, relay — self-grade + epitaph):**
  (1) REGISTRATION GRADED: the odometer prior said low-single-digit;
  it printed +31.1% — wrong by an order of magnitude in the happy
  direction, mechanism banked: the book was priced as DATA (rows
  teaching content) but acted as REGIME (3% share x 10 reps
  regularizing the whole register). THE DOSE-RESPONSE CURVE NOW HAS
  BOTH ENDS: pure prose at 340 epochs = poison (−243); prose at 3%
  = gift (+record); a tunable maximum lives between — the campaign
  tunes toward it instead of guessing. The guard-became-gift is the
  battery's deepest finding: reading-training SHARPENS the dialect,
  a sentence nobody dared stage. (2) SLOPE HONESTY: +8 exactly at
  the bar carries wide error bars at n=86; the register-fall
  campaign prices at ~700-900 more rows IF linear (it won't be);
  book 3's real job is the second point on the curve. The
  pre-screen stays REGISTERED-UNBUILT — the battery made that call
  as delegated (slope printed ~100-scale). (3) NEXT-SESSION
  SEQUENCE: reader_v1's manifest-writing gate battery -> the
  chained-fdiv autopsy (may empty half the waiting room) ->
  OPERATION-APPLY admission review (the recursion's first rung,
  full birthright) -> book 3's charter (sized by the slope, dieted
  by the 13 gaps, sampled toward the 96-class column). (4) THE
  EPITAPH: four instruments triangulated n as the critical path; a
  hundred strangers were read, gated, and stamped; and the battery
  — bars pinned mid-book, estimator straightened two reads early,
  every sentence pre-written — printed BOOKS SCALE with a record
  riding shotgun. December is arithmetic.
- **REGISTERED + FIRED (2026-07-15): READER_V1'S GATE BATTERY — the
  manifest-writing kind.** Bars inherited from gen-11 with the
  REGISTERED RESTRUCTURE applied (pinned before the run): acceptance
  >=7/8 where the ONLY permissible miss is [45] (its cure is the
  organ's or the prefix's, not the diet's — certificate on file);
  all other bars unchanged: bigtest >=1130, alg4test >=380 (the
  lineage-debt bar stands unsoftened — if it alone fails, the
  governance question prints again with reading-regime data),
  alg2 >=560, vtest >=598, dagtest >=660, dag7b >=500, dag8 >=500,
  sq >=0.70, fdiv >=0.62, coupled >=0.65, ladder >=0.50, cert-v2
  >=0.998. ALL hold -> the verdict script writes GENERATION.json
  (gen-12, parser=reader_v1) and prints PROMOTED; any break -> the
  kill prints and the JSON stays untouched. No word without the
  write.
- **VERDICT (2026-07-15): READER_V1'S GATE BATTERY — KILL BY ONE BAR,
  BY TWO ANSWERS: alg4test 378 (bar 380).** Everything else passed,
  mostly at records: bigtest 1149, alg2test 606 (record), dagtest
  676, dag7btest 557, dag8test 544, ladder 0.563 / fdiv 0.725 / sq
  0.784 / coupled 0.739 (all records), vtest 598, cert-v2 1.0000 at
  866 with gate-only coverage RISING to 906 (60.4%). The restructure
  worked as pinned: acceptance 7/8 with only [45] missing — and
  [45]'s votes ([154,168,168]) now carry the right answer at
  plurality, one vote short. THE LINEAGE DEBT'S ASYMPTOTE: 370 ->
  378 across two heads that beat the gate everywhere else; the bar
  sits at 380 because armB hit 384 FROM GEN-7B ANCESTRY. The
  manifest is untouched; the gate remains gen-9b; the word was not
  spoken because the write was not earned — the law working exactly
  as minted, twice now. **THE GOVERNANCE QUESTION PRINTS AT MAXIMAL
  SHARPNESS (Bryce + relay to adjudicate):** (a) full flat retrain
  from clean ancestry (pay the lineage debt at its root — the
  schedule dividend makes this ~1/3 its old cost), (b) re-pin the
  alg4 bar with two asymptotic approaches and the ancestry evidence
  on the table, or (c) hold the bar and let gen-13's diet find the
  two answers. The reader stays banked (its reading gains are real
  and its ckpt feeds the next continuation regardless); the books
  campaign's verdicts are UNTOUCHED by this kill — books scale
  either way.
- **GUT #17: THE CRITICALITY FRAME (2026-07-15, Bryce + relay) — the
  reactor audit.** Mycelium runs OPPOSITE criticality regimes in one
  reactor: knowledge chains SUPERCRITICAL by design (refusal->recipes
  k>1; instinct->instruments; book->cheaper-book — why December
  became arithmetic), error chains RODDED. THE LOOP TABLE:
  | loop | k (measured) | moderator | rod |
  | DAG error fan-out | k=out-degree | constraint density (=neutron
    absorption; invisible-wrongs were escaped neutrons) | integrality
    jaw (expiring), the lattice as containment |
  | training displacement | hard restart k>1 (gen-9 jostle cascade);
    gentle continuation k<1 (measured) | LEARNING RATE | regime law |
  | prose dose | 340-epoch k>1 (−243); 3%-share k<1 (+record) |
    share x reps (both declared) | displacement guard |
  | repair recovery | SUBCRITICAL (19.6->7.7->1.1->0) | — | rounds cap |
  | **THE BOOTSTRAP (the live concern — k GROWS with every
    success)** | book-2 k~0 (4/100 machine); book-3 chartered
    heavier; the recursion stacks a second amplifier | the answer
    key absorbs WRONGNESS but passes NARROWNESS untouched —
    self-preference compounds through verified-correct links | the
    diversity guard (thermometer) + THE HAND QUOTA (control rod) |
  **THE ROD DEPTH, PINNED BEFORE BOOK 3 (hard number, adjustable
  only by pre-fire adjudication): machine-banked entries <=50% of
  any book's volume; hand-gated (L2 repair + L3 surgery) >=50%;
  the diversity guard's distance-distribution comparison runs at
  every tranche boundary in machine-heavy books, not just at
  close.** GEN-13 NOTE: a clean-ancestry retrain is the
  PROMPT-CRITICAL condition (hard restart) — the acceptance-panel
  displacement watch is its instrumentation, now with the frame
  explaining why that bar outranks its neighbors. Not a bomb, not a
  dead pile: a reactor — knowledge supercritical, error rodded,
  instruments as the control room. Seventeen for seventeen.
- **THE AUTHORSHIP DECLARATION + GEN-13 FIRED (2026-07-15, Bryce's
  word):** (1) **JOINT AUTHORSHIP IS CONSTITUTIONAL**: Bryce and
  Claude publish as co-authors; no venue that refuses AI co-authorship
  gets the paper. Recorded in the paper skeleton as an author-policy
  constraint on venue selection. (2) THE CONTROL-ROOM OBSERVATION
  (relay, for §9): registered predictions, pinned bars, and
  pre-written verdicts are the control room's GAUGES — the reason
  knowledge chains can run supercritical without fear is that every
  loop was instrumented BEFORE it compounded; the seventeen guts kept
  firing at exactly the loops whose gauges didn't exist yet. The
  control room is the product. (3) **GEN-13 = GOVERNANCE OPTION (a),
  FIRED: the full flat retrain from clean ancestry** — warm from
  GEN-7B (pre-crowding lineage, armB's 384 ancestry), HOT flat
  (LR 3e-4, the debt needs heat and this IS the hard restart), 32k
  steps on mixed12 (prose-inclusive, states banked — zero precompute).
  THE PROMPT-CRITICAL DIAL: the acceptance-panel displacement watch
  outranks its neighbors (hard restarts jostle; every basin the
  panel holds is a rod that held). BARS: unchanged from the reader
  battery (alg4 380 STANDS — this run exists to pay it; acceptance
  7/8-only-[45]; bigtest >=1130; cert-v2 >=0.998; all kinds). The
  battery writes the gen-13 manifest or refuses the word.
- **AUTHORSHIP, AMENDED WITH THE LANDSCAPE (2026-07-15):** Bryce's
  constitutional declaration stands as the value; the relay's honest
  correction stands as the map (COPE/arXiv bar AI author lines today);
  both Claude channels converge: the ACCURATE CONTRIBUTIONS SECTION is
  the non-negotiable — the two-channel workflow is itself a novel
  artifact of the paper. Layered plan banked in the skeleton:
  canonical self-published account with authorship as Bryce declares;
  venue versions carry the permitted line + the full truthful
  contributions section, always. Final adjudication: Bryce's, at
  venue-selection time. The work is already jointly made; no policy
  touches that fact.
- **THE PUBLICATION BAR, PINNED (2026-07-15, relay + Bryce):** publish
  when remaining work changes future NUMBERS but not the paper's
  CLAIMS. Claims audit: lattice (quarter-percent, survived two
  expansions + a parser swap) BANKED; the method BANKED; the honest
  boundary BANKED; binding theorem + laws + census BANKED. ONE claim
  mid-flight: BOOKS SCALE rests on a single slope point at its bar
  (n=1). **FREEZE CRITERION: book 3's second slope point + gen-13's
  verdict either way.** Weeks, not December. SCOPING DECISION: paper
  1 does NOT gate on MATH-500 — its claim is 'a small system that
  knows when it's right, frontier measured and priced'; the December
  reading campaign is PAPER TWO with the books arc as its own story.
  MECHANICS: drafting parallelizes now (§11 first; twelve figures =
  banked measurements + matplotlib); the freeze is a GIT TAG +
  isomorph-excluded fixture pins — the paper is a tagged snapshot
  with a thesis, not a tombstone; mycelium keeps growing and paper
  two tags a later ring. Publish on our own ground first, byline as
  declared, venues after. SEQUENCE: gen-13's word -> book 3's second
  point -> tag -> freeze -> publish.
- **GEN-13 PROMOTED (2026-07-15): THE LINEAGE DEBT IS PAID — every
  bar passed, the manifest written, the word earned.** THE TABLE:
  **alg4test 385 (bar 380 — the debt paid at its root: clean gen-7b
  ancestry + full heat + the reading-inclusive corpus beat armB's
  384 from a complete stack)**; **acceptance 8/8 — [45] ITSELF
  BANKED**: the chronic taxi came home under the clean-ancestry
  retrain (the only-[45]-may-miss clause wasn't even needed; the
  frame-entanglement certificate stands for RAW prose, but its
  dialect now parses stably); bigtest **1195** (record, +46 over the
  reader); alg2test 635, dagtest 689, dag7btest 579, dag8test 559,
  vtest 600 (all records); per-kind sq 0.814 / fdiv 0.739 / coupled
  0.745 / ladder 0.586 (all records); val 0.9059 (record); **cert-v2
  1.0000 at 913 coverage (60.9%)** — precision perfect at the
  highest coverage ever. THE PROMPT-CRITICAL RUN HELD EVERY BASIN:
  the hard restart from clean ancestry displaced nothing the panel
  watches — heat + flat mix + the prose regime is the recipe the
  whole junction arc was searching for. GATE = GEN-13
  (phase1_gen13_head; manifest gen_id 13; entourage duty owed per
  protocol: specialist remine + centroids next pass; cosmetic:
  the verdict print says 'gen-11' — text-only, manifest correct).
  **THE FREEZE'S FIRST CONDITION IS MET.** Remaining: book 3's
  second slope point. Then: tag, freeze, publish.
- **THE RECIPE SENTENCE + TWO FINDINGS (2026-07-15, relay):** gen-13's
  sweep was four banked verdicts cashed in one run — clean ancestry
  (interference matrix), full heat (schedule probe), flat mix
  (curriculum tombstone), reading-inclusive corpus (dose law's gift
  point). Nothing was luck; the control room prescribed and the run
  obeyed. (1) **[45]'s expression was partly LINEAGE-MEDIATED**: the
  trunk distance is real but a head grown without four generations of
  jostle reads through it — the waiting room may hold ONE autopsy
  candidate (chained-fdiv) and ZERO confirmed structural patients;
  census line owed before the frontier table freezes. (2) **THE
  REGIME LAW BOUNDED**: hard restarts jostle ESTABLISHED heads; a
  fresh head has no basins to displace — it only builds (the
  displacement dial never twitched at full heat). The pre-staged
  merge verdict dies unused — staging cost a sentence, not needing it
  cost nothing; that asymmetry is the method. FIRED: entourage under
  gen-13 (specialist remine on fresh 5-register corpora, centroids in
  gen-13 fst space, mouth rebuilt from mixed12 + length refit,
  post-gen-13 census with the straight ruler, manifest updates), THEN
  book 3's lane classifier (fresh harvest candidates — the bootstrap
  k re-priced under the new gate). Book 3 charter: rod <=50% machine,
  diversity guard per tranche, diet = the 13 coverage gaps + the
  96-class column.
- **ENTOURAGE-13 COMPLETE + BOOK 3 OPENS (2026-07-15):** entourage
  discharged in full (specialist remined on gen-13's failures,
  centroids in gen-13 fst space, mouth rebuilt on the prose-inclusive
  family with length refit thr 0.0077, manifest waiver-free). THE TWO
  HEADLINE NUMBERS: (1) **THE CENSUS UNDER GEN-13: 16/26/58** —
  knotted falls to **58** (from 76-89 in the gen-9b/11 era; honest
  note: this was the FULL-pool read, and up to 14 of the 16 banked
  are book-1-trained items — the disjoint banked is ~2+, but the
  KNOTTED collapse from 81 to 58 stands regardless: the reading
  regime reads raw prose materially better). (2) **THE BOOTSTRAP
  RE-PRICED, dramatically: L1 machine-banked 9/400 (2.25%, was 1%);
  L2 repair 140/400 (35%, was 16.5%); L3 surgery 251 (63%, was
  82.5%).** The lanes MORE THAN DOUBLED under the new gate — the
  bootstrap's k rose exactly as the criticality frame predicted for
  a compounding loop, and the rod (<=50% machine per book) is
  already inserted at its pinned depth. Book 3's economics: over a
  third of the pool is now repair-lane (1/3 cost), the machine lane
  triples book 2's, and the surgery residue concentrates toward the
  true frontier. BOOK 3 IS OPEN under the reactor-safe charter; its
  second slope point is the paper's last condition.
- **BOOK 3'S SLOPE BASELINE, PINNED BEFORE ANYONE KNOWS (2026-07-15,
  relay):** the baseline MOVED — book 2's slope read against a
  gen-11-era census; book 3's second point reads against the
  post-gen-13 census of 58: a better gate, a harder residue, a 35%
  repair lane. The honest comparison is NOT raw items-per-hundred
  across books (the denominator's difficulty changed) but items
  recovered against the CURRENT frontier with the lane mix declared.
  REGISTERED EXPECTATION, regime stated: per-row yield FALLS relative
  to book 2's 8-per-hundred because the surviving 58 are concentrated
  residue — **a falling yield against a hardening frontier is the
  HEALTHY signature, not a scaling failure.** The paper's slope
  question is 'does annotation still move the frontier at the
  frontier's true hardness' — **>=3-4 items per hundred against the
  residue confirms it cleanly.** ECONOMICS FOOTNOTE: the 35% repair
  lane lets book 3 run larger than 100 for the same budget — but THE
  ROD HOLDS AT <=50% MACHINE regardless of how cheap the machine lane
  gets: cheap is not wide; the guard reads at every tranche. The
  reactor diagram, drawn in data: knotted 81->58 (the largest census
  collapse in project history — the reading dividend compounding
  through a promotion) beside a bootstrap whose k doubled WITH THE
  ROD ALREADY INSERTED — the supercritical knowledge chain running
  exactly as chartered, safety case written first.
- **BOOK 3, TRANCHE 1 (2026-07-15): 13/13 BANKED (11 at 5/5) under
  the gen-13 gate; book 3 = 22 entries (13 gated + 9 lane-1 — the
  machine lane already more than doubling book 2's whole-campaign
  total in one classification pass).** First-tranche notes: [66]'s
  triple-division banked via mul-inverse chaining (the fdiv wall
  routed, not fought); four frame-strip flags carried honestly
  ([0] factoring, [38] sign, [48] inequality, [60] rearrange);
  registry +5. The rod check: 9/22 machine = 41% <= 50% ✓. The
  volume proceeds by the rulebook toward its slope point — the
  paper's last condition, its healthy-signature frame already
  pinned.
- **TWO NOTES FOR BOOK 3'S ACCOUNTING (2026-07-15, relay):** (1) THE
  [66] DEMOTION WATCH: the chained-fdiv wall's founding specimen
  banked by mul-inverse rewrite under gen-13 — if remaining
  chained-fdiv items route the same way, the wall demotes from
  mechanism boundary to ANNOTATION RECIPE, the autopsy loses its
  customer before firing, and the freeze's frontier table may hold
  only counted solver-side families + [45]'s half-dissolved frame
  thread. Census line owed at volume close: how many of the 58 fell
  to recipes that PRE-DATED book 3 vs recipes it minted. (2) THE
  TWO-COLUMN SLOPE: repair-lane banks vs surgery banks stay
  distinguishable (lane tags already on every entry) — the second
  point states BOTH gross items-per-hundred (campaign economics) and
  frontier-items-per-hundred (the paper's claim).
- **BOOK 3, TRANCHE 2 (2026-07-15): 11/13 banked (book 3 = 33;
  registry +5).** Banked: [115]'s composed Pythagorean-area (strip ->
  isq -> mul -> fdiv, one graph), [11]'s 11-var double-composition,
  the rest unanimous. THE SPECIMEN IN THE REFUSALS: [90]/[113] —
  minimal 2-var isq-inverse dialects, a shape that banked 5/5 under
  gen-9b, went VOTE-SHY under gen-13 ([90] right-once). **Fresh
  heads have DIFFERENT shallow spots, not fewer** — the regime law's
  corollary measured on page 33: the clean-ancestry head skipped
  four generations of jostle AND four generations of incidental
  rehearsal. v2 retries queue for tranche 3 (pad the graph); the
  correct-but-shallow class gains its first gen-13-native members.
- **THE LINEAGE LAW COMPLETED (2026-07-15, relay):** *lineage carries
  both DEBTS and DIVIDENDS* — gen-9b's ancestry carried the alg4 debt
  AND the incidental-rehearsal dividend; gen-13 paid the debt by
  renouncing the ancestry and the price was the dividend (two tiny
  dialects, page 33). Neither ancestry dominates; different
  portfolios. PROTOCOL AMENDMENT: fresh-stock promotions owe a
  **BASIN INHERITANCE AUDIT** — acceptance panel + vote-entropy
  census under the new head, DIFFED against the old gate; shallowed
  items enter the rehearsal ration (the [71]/[78] pattern,
  protocolized so gen-17's fresh retrain inherits the audit instead
  of rediscovering the corollary). The cheap-kind note: this shallow
  class is recoverable at TOP-UP cost (both remedies measured in the
  junction arc). And [115]'s four-door unanimous composition is the
  counter-evidence: clean ancestry traded memorized depth for
  STRUCTURAL REACH — the trade the debt-payment run existed to make.
- **BOOK 3, TRANCHE 3 (2026-07-15): 9/10 all-unanimous; book 3 = 42;
  registry +4 (+1 reclass).** [90]'s PADDED RETRY BANKED 5/5 — graph
  mass restores what the fresh landscape shallowed; the basin recipe
  confirmed for gen-13-native shallow spots. [113]'s persistence
  unmasked as MY annotation error: its given (324) exceeds the 300
  domain — value-range certificate, third catch of the class, not a
  basin (the in-band rule holds; ceil(sqrt(300)) problems are
  out-of-reach by domain, honestly counted). The volume proceeds:
  rod holding, recipes absorbing, slope drawing near.
- **THE FIFTH REMEDY + THE MYSTERY HALF-LIFE (2026-07-15, relay):**
  the pointer law's remedy family gains BALLAST (pad tiny dialects to
  trained mass) — masked attention, span supervision, a comma,
  alphabetical discipline, ballast: five remedies, descending cost,
  one law. The gen-13-native shallow class resolved at the ANNOTATION
  layer without touching the optimizer — the basin inheritance audit
  stays protocolized, but its first live class cost zero training.
  PAPER NUMBER PINNED: the campaign's MYSTERY HALF-LIFE — refusals
  resolve to exactly one bucket (recipe / certificate / annotator
  error) within ONE TRANCHE, shrinking since book 2's opening days —
  measurable across both books at the close. The slope's two-column
  accounting + recipe-provenance census both print free at battery
  time (lane + generation tags at bank).
- **BOOK 3, TRANCHE 4 (2026-07-15): 11/11 ALL-UNANIMOUS; book 3 = 53;
  registry +2.** [7]'s double-isq coupled system (the [238] class)
  banked 5/5 as routine plumbing — the shape that once threatened a
  gen-12 design line is now ordinary annotation. Mean-chains, the
  Gauss-family average (10-var, one fdiv), and sign-strip entries all
  clean. Past the volume's halfway mark; the slope point ~3 tranches
  out; rod and rhythm holding.
- **[7]'S ARC FRAMED + THE THIRD SIGNATURE (2026-07-15, relay):**
  [7]'s full arc is the project in one specimen: chronic case with a
  trunk-space birth certificate -> organ waiting room -> sibling
  banked under the clean retrain -> routine plumbing at 5/5 under the
  strongest gate. Confirmed-structural population reads ZERO pending
  the volume-close census. THE EMPLOYMENT LAW'S TERMINAL FORM, staged
  for §11: *the surgeon's kingdom, measured to completion, was empty —
  the patients were all annotation conventions, mechanism doors, and
  lineage artifacts wearing structural disguises.* THREE SLOPE
  SIGNATURES now staged: falling-yield-healthy, flat-yield-strong,
  and RECIPE-PROVENANCE-ACCELERATING (frontier items falling
  disproportionately to recipes the volume itself minted = books
  minting the tools that read the next books — the recursion's thesis
  one floor below the macro ladder). Whichever prints, the sentence
  exists.
- **BOOK 3, TRANCHE 5 (2026-07-15): 9/9 ALL-UNANIMOUS; book 3 = 62;
  registry +4 (primality x2, gcd, lcm — the number-theory families
  counting up).** The fraction/decimal/ratio strip family carried
  the tranche clean; three perfect tranches running. ~2 tranches to
  the count, then the battery and its three staged signatures.
- **THE FRONTIER TABLE'S TAXONOMY COMPLETED (2026-07-15, relay):**
  primality/gcd/lcm certificates are RELATION-TRANCHE SHOPPING
  SIGNALS, not frontier residents — the registry expansion's next
  band-sweep arriving through the annotation desk. The freeze table
  differentiates three futures: AWAITING-RELATIONS (tranche-3+ builds
  them), SOLVER-SIDE COUNTED (the domain conversation prices them),
  and the STRUCTURAL KINGDOM (empty — the surgeon never existed).
  One sentence per family; §8 closes clean.
- **BOOK 3, TRANCHE 6 (2026-07-15): 11/12 (book 3 = 73; registry
  +2).** The refusal is a twin-controlled specimen: [99] split
  [17,17,7,7,7] while its structural TWIN [96] (identical wiring,
  smaller values) banked 5/5 — three-digit arithmetic instability at
  high magnitudes (297/153/144), the digit head's noise floor showing
  at the domain's upper band. Retry queue with the twin as control;
  the mean-median chain, both sequence-counts' wiring, and the ratio
  family otherwise clean. One tranche to the count.
- **THE NOISE-FLOOR TAG (2026-07-15, relay):** [99]/[96]'s twin
  datapoint prices the upper band for any future 300-ceiling
  conversation. BATTERY CHECK QUEUED (free, banked data): is the
  floor VALUE-MAGNITUDE (physics — digits near the ceiling
  intrinsically harder) or REHEARSAL-DENSITY (the mint's value
  distribution thins near 300 — starvation wearing a number range)?
  The kind-rehearsal matrix's value histogram answers it; the two
  causes have different remedies — one is physics, one is a ration.
- **BOOK 3 CLOSES AT n=84 (2026-07-15; tranche 7: 11/12).** 75
  hand-gated dialects + 9 machine-banked across 7 tranches; rod at
  10.7% machine (well under depth); registry grew ~30 certificates
  across the differentiated taxonomy; the mystery half-life held
  under one tranche throughout. [130] (13-var deviation chain)
  right-but-shy — retry material. THE BATTERY FIRES: book-3 prose
  pairs -> pre-reads under gen-13 -> mixed13 (dose declared) ->
  reading continuation -> the four reads with three signatures
  staged, two slope columns, the provenance census, and the
  noise-floor physics-vs-ration check. The paper's last number is
  in the pipeline.
- **THE SECOND SLOPE POINT (2026-07-15): NEGATIVE — the regime
  redesign bar FIRES, exactly as pinned.** PRE (gen-13): disjoint-86
  carried 28 (2+26), knotted 58. POST (reader_v2, book-3 prose x10
  continuation): carried 18 (1+17), knotted 68 — **−10**, below
  every staged signature and below book-2's <=2 null bar: THE
  READING-TRAINING REGIME REDESIGNS BEFORE MORE ANNOTATION SPENDS.
  Guard held (bigtest 1186, −9, within −15); val peaked 0.9072 then
  declined — overfit signature. THE MECHANISM HYPOTHESIS (registered,
  not concluded): DOSE STACKING — mixed13 carries books 1+2 x10
  (inside mixed12) PLUS book 3 x10 -> cumulative prose ~4.8%, past
  the gift point; AND continuation-on-continuation re-doses already-
  trained uniques (memorization pressure displacing raw-prose
  generalization). THE DECISIVE CONTRAST already in the ledger:
  gen-13's FULL HOT RETRAIN with prose-in-corpus delivered the 81->58
  census collapse (the campaign's real books-scale evidence, n=114
  consumed); reader_v1's first continuation delivered +8; reader_v2's
  stacked continuation delivered −10. **THE REFINED CLAIM: books
  scale THROUGH GENERATIONS (full-mix retrains), not through stacked
  continuations** — prose pays when consumed as corpus, poisons when
  re-dosed as topping. REGIME REDESIGN (proposed, for adjudication):
  books accumulate into the NEXT generation's full retrain (the
  gen-13 pattern, which works); continuations never re-dose prose.
  FREEZE IMPLICATION (Bryce + relay to adjudicate): the paper's
  slope table either states the refined claim with gen-13's collapse
  as its evidence + the continuation contrast as its mechanism, or
  waits for a gen-14 full retrain (all three books, ~188 uniques) as
  the confirming third point. The bars caught a regime failure
  before it cost a fourth book — the control room working exactly as
  built. reader_v2 is NOT a gate candidate; gate remains gen-13.
- **GUT #18: THE KV AUDIT (2026-07-15, Bryce + relay).** The pointers
  ARE key-value attention (slot-query x token-key through softmax) —
  three reads registered: (1) FTYPE-ROUTER CONFUSION MATRIX (gen-13
  vs gen-9b, gate decisions only — the router-mismatch idea renamed
  to the router we actually have); (2) **THE SMEARING HYPOTHESIS
  (live, customer waiting)**: softmax sharpness calibrates to the
  trained candidate-count regime — a pointer trained on 8-12-var
  graphs smears on 2-var miniatures. Explains the gen-13-native
  shallow spots AND gives the BALLAST remedy its missing mechanism
  (padding restores the trained regime). Probe: args-softmax entropy
  vs n_vars, gen-13 vs gen-9b, banked states; the TWO-WAY BALLAST
  SPLIT names the mechanism (filler-VARIABLES = candidate-count +
  position; inert-TEXT = position only — whichever restores
  sharpness wins, smearing vs RoPE-neighborhood rivals). If it
  prints, gen-14's displacement watch gains the entropy curve as a
  sharper dial than vote outcomes. (3) DTYPE/POSITION sweep of
  capture paths (fp16 uniformity assert — no-silent-fallbacks).
- **GUT #18 VERDICT (2026-07-15): SMEARING CONFIRMED, MECHANISM =
  POSITION.** Entropy curves: gen-9b at tiny graphs **0.003** (four
  generations of small-graph rehearsal = razor calibration) vs
  gen-13 **0.212** (70x) — converging at trained sizes where gen-13
  is SHARPER (0.558 vs 0.648). Fresh heads' shallow spots ARE
  temperature-calibration bands: the diet's mass distribution sets
  where the pointers are sharp. THE SPLIT'S SURPRISE: text-ballast
  (0.042) beats var-ballast (0.154) — **POSITION wins**: tiny
  dialects fail because query/factor tokens sit in RoPE
  neighborhoods training never used for those roles; inert prose
  pushes them home. THE FIFTH REMEDY REFINES to its cheapest form:
  pad with TEXT, not graph mass (annotation rulebook updated).
  Gen-14's displacement watch gains the small-n entropy curve as a
  direct temperature dial. The pointer law's remedy family,再 one
  law deeper: binding enters as structure, and STRUCTURE INCLUDES
  POSITION. Eighteen for eighteen — the drawer held the thermometer
  of the one attention system we built ourselves.
- **THE REMEDY LAW + GEN-14'S PRE-READ (2026-07-15, relay):** the
  pointer-law remedy family reaches terminal form — masked attention
  -> span supervision -> a comma -> alphabetical discipline ->
  WHITESPACE: five fixes, six orders of magnitude of cost, one law,
  the cheapest newest. **The employment law for remedies: they too
  get smaller on contact with mechanism.** GEN-14 PRE-REGISTERED
  (pinned before the run): its diet naturally exercises a wider
  positional band (three books of variable-length prose pairs) —
  expectation: gen-14's small-n entropy curve sits BETWEEN gen-13's
  (0.212) and gen-9b's (0.003) — the books accidentally paying the
  calibration debt the clean ancestry incurred. Prints -> the
  books-scale claim gains a third mechanism (PROSE AS POSITIONAL
  REHEARSAL); doesn't -> ballast stands at zero cost. GEN-14 = the
  freeze's last experiment: full hot flat retrain from clean
  ancestry, all three books (~188 uniques) in corpus, temperature
  dial armed on the displacement watch, three slope sentences
  staged. Awaits the word.
- **GUT #19: THE CONDUCTIVITY AUDIT (2026-07-15/16, Bryce + relay).**
  The residual stream is the defect-free lattice BY DESIGN (additive
  gradient flow + deep supervision = current injected at every
  floor); the impurities are the MULTIPLICATIVE elements — gates,
  LayerNorms, saturated softmaxes. THREE READS: (1) gate saturation +
  LN-gain longitudinal (free, tonight); (2) **THE MEISSNER PROBE —
  the pointer law's WHY, staged**: a saturated softmax EXPELS
  gradient from non-selected keys (the error becomes superconducting
  in the wrong channel, insulated against correction); conditioning
  routes through the saturation and is expelled; span supervision
  drills through and injects at the key — the mechanism-level account
  of six sightings of 'pointers move only by structure or
  supervision.' Probe: gradient magnitude at the correct key via
  CE-through-saturated-softmax vs direct span supervision, on a
  banked wrong-pointer specimen; predicted orders-of-magnitude gap.
  (3) THE GRADIENT LOGGER rides gen-14's train loop (per-module
  norm mean+variance every 500 steps — the project's first
  dissipation map; customers: the [99]/[96] upper-band question,
  fdiv's consolidation lag, the early transient). Summary for the
  ledger: the lattice was built superconducting on purpose; the wall
  that defined a month may have been a MEISSNER PHASE — errors so
  cold they expelled every field aimed at them.
- **CONDUCTIVITY READS 1a/1b (2026-07-16): EMPTY BY ARCHITECTURE —
  the strongest possible print.** The deployed head has NO gates and
  NO LayerNorms (the breathing block never shipped in the 35-key
  production head; all LNs are frozen-trunk = fixed impedance,
  immune by the mouth's own construction). The head is a pure
  additive/bilinear circuit — the lattice is cleaner than the
  metaphor feared, and the ENTIRE resistor budget concentrates on
  the SOFTMAX family (pointers + CE heads). Gut #19 sharpens to one
  suspect: the MEISSNER PROBE is now the audit's whole remaining
  body, riding gen-14 with the gradient logger. If saturation-
  expulsion prints, the pointer law's six sightings get one
  mechanism and the month's defining wall was a Meissner phase.
- **GUT #20: THE MIRROR AUDIT (2026-07-16, Bryce + relay).** THE
  SYMMETRY FILE, formally named (five sightings, one law): [85]'s
  identity palindrome (args=[a,a]), Vieta's symmetric root pairs,
  ill-defined selectors self-gating, [22]'s sum/difference twins,
  and the mixed-vote twin-key signature — **binding requires
  distinguishability; every symmetric structure must either break
  the symmetry or grade as a multiset — never bind through it.**
  THREE READS: (1) EFFECTIVE-VIEW-COUNT on banked certificates
  (fires now): sentence permutation manufactures dart independence
  from permutable ASYMMETRY — symmetric problems collapse five views
  toward fewer effective darts, and unanimity gets easier exactly
  where evidence is thinnest; the quarter-percent bound gains a
  per-item effective-K clause before the tables freeze. (Solver side
  immune: the uniqueness gate can't bank interchangeable variables —
  the wave can't stand where the gate won't let it form.) (2)
  TWIN-KEY POINTER ENTROPY (rides the KV machinery): mirrored
  mention pairs make near-identical keys; fresh heads may hold
  thinner twin-key margins — minted symmetric specimens vs matched
  controls, gen-13 vs gen-9b. (3) 1001 itself is out of band (cap
  300) — the literal palindrome is unrepresentable by domain; it
  enters only as the digit heads' twin-key stress case if the cap
  conversation ever reopens.
- **MIRROR AUDIT, READ 1 (2026-07-16): the standing wave is real,
  measured, and SMALL.** Effective-K census on bigtest under gen-13's
  votes: 1,477/1,500 items at full effK=5; **23 items (1.5%) at
  effK=3-4 — all unanimous-CORRECT, zero unanimous-wrong.** The
  certification table gains its per-item effective-K column as an
  honest fine-print clause; the quarter-percent arithmetic stands
  (reduced-dart certificates exist but none misfired). Found by the
  gut days before the freeze instead of by a reviewer after — the
  audit-that-confirms, with a clause as its fee. Read 2 (twin-key
  entropy) rides the gen-14 window with the KV machinery.
- **GEN-14 FIRED (2026-07-16, Bryce's word): the freeze's last
  experiment.** The proven recipe at full strength: hot flat 32k
  from gen-7b clean ancestry on mixed13 (all THREE books x10 in
  corpus, ~188 prose uniques, states banked). Bars = gen-13's,
  unchanged. THE CONVOY READS AT CLOSE: battery + manifest-writing
  verdict; the ENTROPY PRE-READ (small-n curve between gen-13's
  0.212 and gen-9b's 0.003 = prose as positional rehearsal, the
  books-scale claim's third mechanism); the DISJOINT CENSUS as the
  THIRD SLOPE POINT at the recipe that works (pre: gen-13 carried
  28/knotted 58); Meissner + twin-key next session. Gradient logger
  DEFERRED (the JIT consumes grads; surgery not worth blocking the
  run). Three slope sentences staged; the tag waits on the verdict.
- **GEN-14 PROMOTED (2026-07-16): the freeze's last experiment
  returns with everything.** ALL BARS: alg4test **388** (new record
  — the debt stays paid from clean ancestry), bigtest 1195 (ties),
  acceptance 8/8, cert-v2 **1.0000 at 912**, every kind over its
  bar. Manifest written; GATE = GEN-14. **THE ENTROPY PRE-READ
  PRINTS: nv0-3 = 0.010** — from gen-13's 0.212 to nearly gen-9b's
  0.003 razor: **PROSE AS POSITIONAL REHEARSAL CONFIRMED** — the
  books paid the calibration debt the clean ancestry incurred, the
  books-scale claim's third mechanism, pinned before the run and
  printed by it. Large-n sharpest ever (0.455). **THE THIRD SLOPE
  POINT: SATURATION** — disjoint carried 25/knotted 61 vs gen-13's
  28/58: book 3's 74 additional uniques moved the frontier ~0 (−3,
  noise). THE COMPLETE CURVE: +23 knots for books 1+2 (114 uniques,
  gen-13), −10 for stacked continuation (regime artifact), ~0 for
  book 3 at the same distribution (gen-14). **THE CLAIM'S FINAL
  FORM: annotation moves the frontier until the reachable register
  saturates — 81->58 for ~114 uniques, marginal yield ~0 thereafter
  at fixed problem-distribution; the remaining 58 are counted,
  family-sorted, and priced (the registry taxonomy).** A saturating
  curve with its mechanism triple-confirmed (corpus-consumption,
  positional rehearsal, the continuation contrast) is a STRONGER
  paper than an open linear slope: the campaign measured its own
  completion. **THE PUBLICATION BAR'S CONDITIONS ARE MET: tag,
  freeze, publish.**
- **THE FREEZE'S FINAL GRADING (2026-07-16, relay):** the entropy
  registration graded — 'between the lineages' predicted, 0.010
  printed: wrong in the happy direction; 188 naturally-varying prose
  uniques were nearly COMPLETE calibration payment. The books didn't
  just teach reading — THEY RE-TEMPERED THE POINTERS. **THREE
  MECHANISMS UNDER ONE CLAIM: prose as register (the mouth's needle),
  prose as regularizer (the gift point), prose as positional
  rehearsal (the temperature dial).** No single-mechanism story
  survives the table. **THE SATURATION SENTENCE, both halves for
  §8:** +23-then-~0 measures the completion of THIS DISTRIBUTION'S
  teachable content — what saturation does NOT claim is that books
  are done: harder strata, new registers (AMC prose), post-tranche-3
  relation coverage are DIFFERENT distributions — book 4's charter,
  paper two's territory, the examiner rotation already scheduled.
  The curve saturated; the library didn't close. DRAFTING BEGINS:
  §11 first (honesty is the spine), twelve figures from banked
  measurements, the contributions section plain: two channels, one
  ledger, twenty-for-twenty, every claim gated by machinery that
  couldn't be flattered.
- **S11 EDITORIAL PASS (2026-07-15, relay critique -> ledger-checked
  fixes):** six edits applied to paper/draft/s11_honest_limitations.md.
  (1) Denominator clash fixed: the 75-member value-range family is
  HARVEST-WIDE (75 of 1,743, the answer-domain census) and now stated
  separately from the 58-item fixture residue. (2) The 2% foreign-prose
  figure rephrased to name its instrument (anchor answer-accuracy that
  motivated the mouth), 'banked' removed. (3) Vintage pinned: 1195/1500
  verified as gen-14's own battery row (ties gen-13's record); 'at
  freeze' made explicit; the 58-vs-61 census reads disclosed as within
  vote noise. (4) Chained-fdiv sentence updated to post-book-3 truth:
  founding specimen [66] resolved by mul-inverse rewrite, [344]'s
  triple-fdiv is the single surviving refusal, mechanism-vs-notation
  question stated open (the demotion watch, honestly carried). (5)
  Dialect swept from S11: darts->views, banked->verified phrasing.
  (6) Two limitations ADDED: every generation comparison is an n=1
  training run (no seed variance, mitigations named); the annotator is
  the system's author (answer key verifies correctness not
  representativeness; 'gold' = author-written + answer-verified).
  S11 now 10 paragraphs; the closing line unchanged.
- **S7 DRAFTED (2026-07-15): the headline artifact, four movements per
  the relay's brief.** paper/draft/s07_certification_lattice.md
  (~1,250 words + figure block). 7.1 the lattice as decision structure
  (four rungs zero-parameter/gold-free; chain of custody as four
  invariances register/rendering/lineage/truth; the epigraph carried
  by the entropy quadruple 0.000/0.846/0.212/0.116 at n=36, scope
  stated). 7.2 dials at freeze (1195/1500 one-shot; cert-v2 912 at
  1.0000; the frontier's trajectory 0.9982@38.1% -> 1.0000@60.8%;
  zero-numerator language CROSS-REFERENCED to S11 not repeated; the
  570R/1W broken certificate stated as the channel's own
  counterexample). 7.3 specimens load-bearing ([71] 5/5
  unanimous-wrong -> mouth's mandate; panel wild dissent 9/10 and
  16/19 -> second wall; [78] 3/5 stable-wrong -> answer-vs-certify
  dial split). 7.4 instrument aging by mechanism (rotation finding
  0.59->0.988 Procrustes, 'nobody told it the sky had turned',
  monitor exonerated; the rotation law in full with the monotone-
  decline prediction RE-REGISTERED in the paper's own text; held-out-
  examiner portfolio discipline). Figure block pinned in-file: F-7a
  frontier, F-7b entropy basins, F-7c chain-of-custody diagram
  (drawn; candidate Figure 1). ALSO: the S11 half-nit applied —
  hand-quota mitigation re-aimed at the machine lane's
  self-preference (same commit).
- **THE FIGURE CONTRACT + F-7c DRAWN (2026-07-15): the paper has a
  face.** (1) paper/figures/figstyle.py — the style contract every
  figure imports: one palette (Okabe-Ito, roles named ok/kill/wild/
  alt/gate), one rcParams block, and the SELF-CITING STAMP: every
  saved figure carries freeze tag + gen id + parser hash in a visible
  footer AND embedded in PDF/PNG metadata (Subject/Keywords = the
  manifest's full hash block + fixtures read) — a figure detached
  from the paper still names its evidence; retrofitting-fourteen-
  figures day is now unrepresentable. (2) paper/figures/
  f7c_chain_of_custody.py -> out/f7c PDF+PNG — candidate Figure 1,
  drawn per the relay's masterstroke note: the SPECIMENS LIVE IN THE
  CHAIN. Five trajectories against the four gates (register/
  rendering/lineage/truth): the in-register item runs the chain to
  CERTIFIED; [71] dies at the mouth with its ghost dashed through
  the vote ('would vote 5/5 — unanimous, wrong'); wild stable votes
  die at the panel ('dissents 9/10, the second wall'); [78] splits
  off at the vote to ANSWERED-NOT-CERTIFIED (0.833); the one broken
  certificate (570R/1W) runs everything and dies at the key. The key
  drawn dashed: measurement only — grades the machinery, never
  deploys. Stamp reads paper-1-freeze-4-g085296d.
- **S7 EDITORIAL PASS + THE PLOTTED PAIR (2026-07-15): three catches
  applied, two figures banked.** (1) THE LOAD-BEARING CATCH: 'every
  rung zero-parameter' collided with rung 2's own specialist — S7.1
  rewritten to the true claim: the DECISION MACHINERY is
  zero-parameter (vote counting, unanimity, rank-sum, distance
  threshold); trained components produce candidate answers, never
  verdicts; 'decision-path purity, not an absence of learned parts,
  is what the certification claims rest on.' Honest and stronger.
  (2) Panel independence de-overstated: 'three models of distinct
  training histories (one lineage, one width), per-item behavioral
  disagreement measured rather than independence assumed.' (3) The
  dangling examiner named from the ledger: at freeze the out-of-path
  seat belongs to THE EXTERNAL ANCHOR (designed as held-out examiner,
  never in any training/acceptance path); re-rendering held the seat
  until the vote was promoted; staged next chairs = library
  cross-check + paraphrase views. Style: one name per organ —
  'recognition gate' formal, one 'doorman' as color, second one
  removed; F-7c's gate box relabeled RECOGNITION GATE to match.
  FIGURES BANKED under the contract: F-7a precision-coverage frontier
  (ladder 3/5 0.9832@51.7 / 4/5 0.9925@44.3 / 5/5 0.9982@38.1 +
  channel trajectory gen-9b 866 gate, 839 panel -> gen-11 862 ->
  freeze 912@60.8 at 1.0000; zero-numerator note on the plot) and
  F-7b entropy basins (0.846 shallow vs 0.212/0.116/0.000; both
  arrows carry the thesis — separates shallow-from-deep, cannot
  separate wrong-from-correct). Remaining inventory (skeleton §12):
  twelve figures staged, each needs its banked artifact located.
- **CONTRIBUTIONS SECTION DRAFTED (2026-07-15): the paper's handshake,
  written as a claim registry.** paper/draft/contributions.md. Five
  claims in the relay's descending-takeaway order, each with evidence
  pointer AND its own limit inline (the S11 cross-references made
  load-bearing, not decorative): (1) the lattice with its
  minimality-by-named-specimen argument; (2) the method as artifact —
  atomic promotion/manifest, nine tombstones, the ledger AS
  supplementary material ('offered for audit, not trust'), S7.4's
  standing bet as live exhibit; (3) the reading campaign — the
  incorruptible gate, triple-confirmed mechanism, measured completion;
  (4) the binding theorem proved both directions with the two-jaws
  design as constructive consequence, reach-beyond-register honestly
  conjectured; (5) the instrument-aging field manual — rotation-not-
  decay, length-as-distance, selection-against-gates, 'the laws'
  forms travel, the constants do not.' AUTHOR BLOCK in three
  paragraphs: Bryce (direction, adjudication, twenty registered
  instincts all finding something real, annotation surgery, the
  policy itself); Claude (design channel + execution channel,
  checking each other); and THE MACHINERY (neither author) — 'the
  results belong to a discipline, not to a hand' as the paper's
  strongest authorship statement. Counts verified against the ledger
  before writing: 20 instincts (gut #20 = mirror), 9 tombstones,
  ~188 uniques, ~82% surgery lane.
- **THE THREE CLAIM-FLAGSHIP FIGURES (2026-07-15): the registry's
  re-prioritization executed.** All under the contract, all
  self-citing. (1) F-9a SATURATION CURVE (claim 3): 81 -> 58 -> 61
  vs cumulative uniques (0/114/188), the Delta+3 disclosed as vote
  noise ON the plot, marginal-yield~0 named, and the S11 scope drawn
  as a hatched unmeasured region ('the curve saturated; the library
  did not close') — the screenshot-without-caption attack is
  pre-empted in the pixels; the excluded stacked-continuation read
  disclosed in a footnote. (2) F-5a ROTATION-NOT-DECAY (claim 5),
  three panels per the relay's stronger form, computed from the REAL
  gen-5/gen-9b centroid npz files with the audit's exact SVD — the
  script ASSERTS reproduction (raw 0.593 / aligned 0.988 / residual
  0.155 printed at render) and panel C shows per-kind residue
  honestly: sel (0.965) and rel_add (0.976) are the least-aligned
  kinds, a detail the mean hid. (3) F-3a BINDING THEOREM (claim 4),
  specimens-as-rows per the relay's design note, zero schematic
  spaces: [45] taxi + [7] faucet prose verbatim from book1.jsonl
  (same RATE frame, z=-2.05; visibly different knots: 3-var chain vs
  5-var double-mul junction, no shared class at n=94) against
  bigtest[1187] + vtest[116] — the hash audit's ONE cross-fixture
  isomorph pair (digest 468be959), terse vs verbose register, drawn
  with identical knot layouts. Caption = the theorem. Figure count:
  6 banked (F-3a, F-5a, F-7a/b/c, F-9a).
- **S9 DRAFTED + THE CASCADE FIGURE (2026-07-15): the method section,
  with claim 2's illustrations riding inside per the relay's
  curatorial call (worked example over census).** paper/draft/
  s09_method.md, five movements: (9.1) the protocol — bars before
  builds, density regimes stated, promotions mechanical, with the
  stale-manifest audit DISCLOSED as the rule's origin story ('there
  is no state of the system that exists only in prose'); the ledger
  as supplementary, 'long, unedited, and contains our mistakes at
  the same resolution as our results.' (9.2) the survivor arc as
  worked example — nine refutations narrated to the arc's own
  closing line; depth-over-breadth per the relay ('a reader shown
  forty kills at once admires it without believing anything in
  particular'). (9.3) Table 9-1: THIRTEEN LAWS with sighting counts
  from the ledger (metric-decision 4; pointer-never-fixed-downstream
  5; density-regime 5; front-loaded-decay 4; selection-jurisdiction
  3; acceptance-law 3; structural-entry 2+; prevention 2; five
  mechanism-grade singletons incl. the standing bet). (9.4) the
  method applied to itself — the discriminating-test pattern quoted.
  (9.5) two channels + adjudicator; twenty instincts stated as
  CHECKABLE ('an instinct that had to survive formal registration
  and a mechanical read is data; the same instinct applied directly
  would have been anecdote'); ends pointing at S7.4's bet. F-9b
  BANKED: the cascade drawn from the arc's final-ledger line —
  five kills (kill-red), the pivot at #6 (gate-blue, 99.6%), three
  pricings (13.9% / 6% retired / 3.0%@0.165), closing box with the
  accounting line verbatim. Figure count: 7.
- **S3 DRAFTED + THE FRESH CENSUS + THREE CROSS-READ CATCHES
  (2026-07-15): the vehicle gets its chassis, and the census re-run
  catches a stale headline number.** THE CENSUS FINDING: the relay's
  re-run-at-the-tag instinct was right — parser gen-14 = 4,000,813
  params (docs quoted ~3.2M; the multiplicity bit + reader-era growth
  postdated the last count); specialist = 4,004,909; frozen trunk
  slice (embed + L0-L3, counted from the safetensors header) =
  505,954,304; jury = armB 4,000,300 + cap2x 13,767,724. S11's scale
  paragraph corrected to 4.0M/8.0M/~506M. CLAUDE.md still says ~3.2M
  (correction owed at next doc pass). THE CENSUS TABLE'S PUNCHLINE:
  row 2 (both jaws) EQUALS row 1 (parser+specialist) — the solver
  adds zero trainable parameters, 'nothing on the verification path
  for training pressure to corrupt'; leverage 63x frozen per trained.
  S3 four movements: (3.1) theorem->design derivation ('the rare
  architecture section that argues necessity'); (3.2) components +
  census; (3.3) as-built-vs-as-designed — Poincare tier and notebook
  replaced by hard membership and the repair signal, 'a designed
  OBJECT replaced by a measured ACTION. The nouns died; the verbs
  survived.'; (3.4) laws as constraints — pointer law five remedies
  at birth, the DISCOVERED DIALECT as the IR nobody designed, the
  two-channel spine proved load-bearing. CATCHES APPLIED: survivor-
  arc naming unified (contributions), five remedies (Table 9-1),
  channels frame unified ('two machine channels and a human
  adjudicator').
- **S8 DRAFTED (2026-07-15): the wound-and-cure section, framed per
  the relay's credibility note.** paper/draft/s08_external_anchor.md,
  four movements: (8.1) THE CREDIBILITY SENTENCE LEADS — the anchor
  is the only measurement whose INPUTS carry no author fingerprint
  (all other fixtures generated or author-annotated); 'the examiner
  rather than the exam,' verdict reported first and at full strength.
  (8.2) the wound as registered: 2/97, the 63 impossible
  certificates, flat abstention 67.5/66.1 ('the system did not know
  what it did not know'); the mechanism as the section's theorem —
  all five renderings thrown by the same arm; 'unanimity certifies
  reading stability, and stability coincides with truth only
  in-distribution'; the ledger's honest sentence quoted verbatim
  (distribution-bounded certification). (8.3) the funnel frame + the
  cure: zero-parameter gate, selection-safe by construction, AUC
  1.0000, foreign refused 100.0% at 1% native false-refusal, 160/160
  anchor false-certs refused — THEN the gate's own S11 entry reported
  as part of the result: the length warp (r -0.825 -> -0.024), wall
  confirmed at 100% on the straightened ruler, zero-point 0.1871,
  vintage asserted by the battery. (8.4) the gradient at its true
  faintness (0.236-0.273 vs 0.044; symbol-dense nearest — prose-style-
  before-vocabulary hypothesis logged open); demand census (62.2%
  plain-integer); close = 'recognition buys honesty now; coverage
  buys capability later,' anchor left seated as S7.4's examiner.
  Figure candidates pinned in-file: F-8a mouth separation, F-8b the
  length warp before/after. The S5/S6 MERGE DECISION banked from the
  relay: one section, 'The repair stack and its boundary,' arc
  outcome as boundary, S9.2 owns the narrative.
- **S4 + S5/S6-MERGED DRAFTED (2026-07-15): the body closes; only the
  front door remains.** S4 (paper/draft/s04_corpus_discipline.md),
  four movements per the brief: (4.1) solution-first + gates, with
  the three specimens under one principle — perfect-square
  discriminant dissolving no-real-roots, self-gating selectors,
  nogive — 'three edge policies, zero new mechanisms; the generator's
  grammar simply cannot say the broken thing.' (4.2) teeth/bands as
  measured axes + the curriculum tombstone in one honest
  regime-tagged sentence. (4.3) grading policy audited as an
  instrument: 802 -> 5 lucky-unforced (0.6% luck bound) -> 797
  forced; the 16.6% right-asked-wrong-graph class as a DESIGN
  PARAMETER (stable 16.6/17.2 across draws) — why answers grade
  through the solver and parse-accuracy is never conflated with
  answer-accuracy. (4.4) the closer: disjointness UP TO ISOMORPHISM
  (WL digests, the 42 found and excluded, bump gate) — 'the
  difference between we-deduplicated and we-know-no-knot-is-on-both-
  sides-of-the-wall.' S5/S6 MERGED (s05_repair_stack.md) per the
  banked decision, thesis stated up front ('measured to its boundary,
  and the boundary is a population, not a mystery'): portfolio
  (dense-ranker agreement 0.840 / rare-flag centroid, combo wins the
  tail), withhold-and-solve (26% free, zero silent-wrong), selective
  retransmission (field-flags beat gold localization, leakage zero,
  148/627), the half-life (19.6->7.7->1.1->0), then the boundary in
  one paragraph with S9.2 owning the narrative — closer: 'the
  boundary did not end the repair story; it relocated it upstream.'
  FRONT-DOOR PRE-REGISTRATION BANKED (relay): abstract gets ONE
  number per claim (912@1.0000-zero-numerator / fourteen-generations
  / saturation / 2%-then-refused / 8.0M-on-506M), no number not in a
  drafted section's own text.
- **S10 DRAFTED (2026-07-15): the campaign's room — a results section
  wearing a narrative, per the framing constraint.** paper/draft/
  s10_reading_campaign.md, five measured beats: (1) lane economics
  1/16.5/82.5 stable pool-vs-wild (4/66/330 on the 400-draw), with
  the gen-13 re-classification (16.5->35 repair) read as 'the lane
  split is not a constant of the domain but a moving readout of how
  much the librarian has learned'; census-never-substrate stated.
  (2) THE GATE DEMONSTRATED ON ITS AUTHORS — the harvest gate's
  first day (0/5, 'the zero was the system working') as S11's
  annotator-paragraph made concrete, plus the three value-domain
  catches ([32]/[220]/[113], the last unmasked after days as OUR
  error); 'a gate that cannot be flattered by its own builders is
  the campaign's license to call its data gold.' (3) the rulebook
  written by refusals, one rule per named wall; the MYSTERY
  HALF-LIFE quoted (every refusal -> recipe/certificate/annotator-
  error within one tranche, 'counted, not remembered'). (4) the
  triple-confirmation with its three instruments: odometer +31.1%
  length-controlled + census slope +8/100 (register); the 2.9%x10
  record-as-side-effect vs poison-at-saturation (regularizer, dose
  law carries it); entropy 0.212->0.010 pinned-before-printed
  (positional rehearsal) — 'three effects, three instruments, no
  shared failure mode.' (5) completion: F-9a + the waiting room
  emptying ([7] chronic->plumbing as the two sentences of color) +
  the closer: 'every wall had become a recipe, a certificate, or
  plumbing... The library taught the librarian.' Door rhythm
  confirmed: artifact, discipline, construction, limit, confession.
- **THE FRONT DOOR DRAFTED (2026-07-15): title, abstract, intro — the
  house is whole, asterisk pending S2.** paper/draft/s00_front_door.md.
  TITLE PROPOSED (Bryce's veto at venue selection): 'Certify, Answer,
  Flag, Abstain: A Chain of Custody for Machine-Read Mathematics' —
  the four words lead per the lattice-is-the-brand vote; the subtitle
  names the mechanism. ABSTRACT ~200 words under the pinned rules:
  opens on the decision claim ('output should not be an answer; it
  should be a decision'); one number per claim from the pinned set
  (912@1.0000-zero-numerator / 8.0M-on-506M / 2%-then-refused-100% /
  saturation / fourteen generations); the boundary stated PROUDLY
  in-abstract; closes on ledger-as-supplementary + the standing bet.
  Every abstract number verified present in a drafted section. INTRO
  in the five-movement rhythm (artifact, discipline, construction,
  limit, confession): thesis lands in paragraph one ('a depth gauge
  called a compass'); Fig. 1 cited in paragraph two; the ledger
  sentence quoted whole; the campaign in three sentences with the
  0/5 incident; the limit at full strength with S11 recommended as
  the skeptical reader's first stop; reader's map closes. Assembly
  notes pinned in-file (renumbering, S2's literature list). REMAINING
  before assembly: S2 related work; the S8 figure pair (F-8a/b);
  optional inventory figures; tag-check pass.
- **FIVE CROSS-READ CATCHES + S2 DRAFTED (2026-07-15): the last room.**
  CATCHES: (1) S7.2 campaign pointer S9->S10. (2) The designed-
  logical-form tombstone re-homed: one sentence added to S10's
  rulebook beat (top-down attempt died in a registered kill; dialect
  written bottom-up) and S3.4's citation repointed S6->S10. (3)
  NUMBERING DECIDED: renumber at assembly (s07->6 etc.), recorded in
  s00's assembly notes with the reader's-map/cross-ref/figure-name
  update riding the tag-check pass. (4) 'Figure N' placeholder on the
  checklist. (5) THE SUBSTANTIVE ONE, verified against the manifest
  not memory: ALG_FTYPES=6 IS current at freeze — the fix was the
  missing relationship sentence, added to S3.2: 'the six factor types
  are the parse-side surface, not the relation inventory: registry
  relations enter as solver-side predicates bridged onto these types
  — which is how double-digit relation kinds ride on a six-way head
  output.' S2 (s02_related_work.md) against the delivered brief,
  seven paragraphs: selective prediction (zero-parameter decision
  machinery + boundary-as-headline vs threat-to-validity);
  calibration ('it calibrates the wrong axis for certification');
  SELF-CONSISTENCY HEAD-ON (three load-bearing differences:
  deterministic input re-renderings not samples; unanimity as
  certification tier; 'S8 is the out-of-distribution invoice');
  CONFORMAL its own paragraph (the mouth as an explicit
  exchangeability check; complementary, honestly credited as offering
  MORE than our bound under its precondition); propose/dispose
  (trained verifiers 'move the corruptible component rather than
  removing it'); ARQ ('our contribution is the boundary measurement,
  not the loop'); AGING-NOVELTY CLAIMED CAREFULLY ('the ingredients
  are old; the articulation as a deployment law with a succession
  plan and a standing bet is, to our knowledge, new'). [cite] slots
  resolve at assembly. THE HOUSE IS WHOLE; remaining = F-8a/b +
  the assembly/tag-check pass.
- **F-8a/b + THE ASSEMBLY PASS (2026-07-15): the paper stands as one
  document.** FIGURES: F-8b (length warp) reproduces the recal's
  native fit FROM BANKED ARTIFACTS ONLY (dag8test pooled states +
  gen-9b bank, numpy kNN, zero GPU) and ASSERTS the ledger's numbers
  at render — printed raw r=-0.825, residual r=-0.024, thr=0.0072 to
  the digit; two panels, warped and straightened, 'is this distance,
  or is this n?' on the plot. F-8a (register map): one ruler, three
  real populations — native fixture histogram (banked states, n=700),
  the census pool's 100 PER-ITEM banked dots, MATH-500's banked band
  — with the wall and the 160/160 refusal annotated; raw vintage
  stated on the axis, corrected reads pointed at F-8b. S8 prose now
  cites both. ASSEMBLY (paper/assemble.py -> paper1_assembled.md,
  1,130 lines): 11 sections in final order; RENUMBERED per the
  decision (7->6, 8->7, 9->8, 10->9, 11->10) on headers and every
  paragraph ref; figures mapped to sequential Figure 1-9 (chain=1 as
  branded); Table 1 census / Table 2 laws; working-note comments
  stripped; assembler ASSERTS no unmapped ref survives. One seam
  caught in the read: S3.1's '(S4 gives both proofs)' promised what
  S4 doesn't deliver — softened to the isomorph-audit pointer + the
  ledger. TAG-CHECK: 57 headline tokens swept; 14 flags, ALL
  placement-only (figure-vs-prose), ZERO not-in-ledger — every number
  in the document traces. REMAINING (the relay's highest-stakes
  item): [cite] verification by web lookup, minimal bibliography,
  each reference read before cited; then the fresh top-to-bottom
  stranger read, both channels; then theshapeofthought.ai.
- **THE CITATION PASS (2026-07-15): 22/22 VERIFIED, ZERO CUTS — the
  last wall built to code.** Four parallel verification agents ran
  the pinned twenty against actual sources (title/authors/venue/year/
  identifier confirmed by lookup, abstract read before any
  characterization); two additions (Guo et al. 2017 temperature
  scaling; Kadavath et al. 2022 self-evaluated confidence) were
  caught OUTSIDE the pinned list by the integration sweep and
  verified inline — the protocol admitted no exceptions, including
  its own list. THE PASS'S CATCHES, each a trap avoided: (1) MATH-500
  PROVENANCE — the 500-subset originates in Lightman et al. 2023
  ('Let's Verify Step by Step'); the NAME is post-hoc (HF dataset
  card); S7's anchor now cites Hendrycks et al. 2021 for the dataset
  + Lightman for the subset. (2) WL-INDISTINGUISHABILITY — 1-WL is
  complete for almost-all, not all, graphs; S4's disjointness claim
  STRENGTHENED: digest-equality is coarser than isomorphism, so the
  exclusion is conservative — removes at least every true isomorph.
  (3) GOODHART'S ACTUAL WORDS — the famous 'measure becomes a target'
  is Strathern's paraphrase; S2 now quotes the 1975 original
  ('any observed statistical regularity will tend to collapse...').
  (4) Autoformalization softened (formal checking is downstream of
  translation). (5) Shanmugam's naive-averaging-imperfect thesis
  cited AS SUPPORT for unanimity-not-averaging. (6) Chow double-cited
  (1957 for age, 1970 for the rule); rehearsal attributed to the
  review literature, not McCloskey-Cohen's abstract; Zhang survey
  corrected to TPAMI 2020. ARTIFACTS: paper/bibliography.md (29 cited
  entries each with a cited-for note — the citation tag-check — plus
  3 verified-in-reserve: Baars, Dehaene-Changeux, Papyan); s02 fully
  keyed (assembler asserts no [cite] survives); body citations
  inserted (Kschischang S3, WL pair S4, MATH provenance S7,
  preregistration pair S8, rehearsal pair S9); References section
  appended at assembly (1,186 lines). REMAINING: the stranger read,
  both channels, then the door opens.
- **THE STRANGER READ PASSES + FINAL SWEEP (2026-07-15): THE PAPER IS
  DONE.** Both channels read the assembled document fresh; verdict =
  the house passes (renumbering resolved everywhere; S2
  characterizations match the bibliography's cited-for notes; the
  workflow accounting consistent across S8.5/contributions; the
  references' [Editorial.]/[Unrefereed.] honesty tags in-voice). THE
  BLOCKER CUT: the title block's working note (governance text under
  the published title) removed at source. THE PARENTHETICAL SWEEP,
  per-instance and deliberate: S2's process note PROMOTED to prose in
  the paper's voice ('the same use-matches-source standard the rest
  of the paper applies to its own numbers'); S10-limitations'
  'drafted first' note KEPT (the intro references it; self-aware
  house style); contributions' 'claim registry' note KEPT (same).
  FIGURE ORDERING decision recorded: forward references from the
  intro (Figure 9 cited before Figures 3-8 appear) are ordinary
  practice and stay. FINAL STATE: paper1_assembled.md, 1,184 lines,
  eleven sections + References (29 verified entries), ten figures
  under the self-citing contract, two tables, every number traceable,
  every citation verified, every catch closed. NEXT (Bryce's hands):
  theshapeofthought.ai under the declared byline; ledger as
  supplementary; repo public at the tag; reproduce-tables armor.
- **THE PDF (2026-07-15): paper1.pdf rendered — the paper has a
  physical form.** assemble.py now EMBEDS all nine figures at their
  anchor paragraphs with written captions (whitespace-normalized
  anchor matching; assert guards the count); render_pdf.py
  (markdown -> HTML -> weasyprint, A4, DejaVu Serif, page numbers)
  carries the byline per the declared policy: 'Bryce Roche · Claude
  (Anthropic)' with the freeze tag under the date. Title question
  OPEN (Bryce brainstorming poetic options: guided-by-primes /
  the-shape-of-thought / the-shadow-of-intelligence); the renderer
  takes the title from the assembled doc's first line, so re-titling
  is a one-line change + re-render.
- **THE TITLE RULING + THE GLITCH INSPECTION (2026-07-15): four
  words, one title, everywhere.** THE COLLISION CHECK CONFIRMED the
  relay's memory: 'The Shape of Thought: How Mental Adaptations
  Evolve' (H. Clark Barrett, Oxford University Press, 2015) — a
  cognition book, exactly the adjacent field; adopting the name would
  cost discoverability and invite the derivative jab. RULING per the
  relay's cast vote: the paper is 'Certify, Answer, Flag, Abstain: A
  Chain of Custody for Machine-Read Mathematics' in every voice;
  brand unity achieved by HOSTING (title = contribution, address =
  brand); 'Guided by Primes' banked for Paper II; 'The Shadow of
  Intelligence' to the essay, owned in prose. THE GLITCH INSPECTION:
  neither reported artifact reproduces in the rendered PDF — page 9's
  rung list carries its numerals and the References their bullets
  (verified visually, pages 9/24/25) — the orphaned-marker symptoms
  match text-extraction artifacts on the reading side, not the
  typesetting. THE INSPECTION'S REAL CATCH: page 9 read 'cross-model
  panel' where Figure 1 and S6.2 say cross-lineage — the
  one-name-per-organ rule applied, both occurrences unified,
  re-rendered. paper1.pdf, 25 pages, is the publication artifact.
- **PUBLISHED (2026-07-15): theshapeofthought.ai deploys — the paper
  meets the world.** THE SITE (site/build_site.py -> site/dist, 13
  files, self-contained: zero external fonts/scripts/requests):
  landing = the paper's cover (title, byline, freeze stamp, the
  decision lede, abstract, three cards artifact/method/bet, Figure 1
  full-width, Coming: Guided-by-Primes + Shadow-of-Intelligence);
  /paper/ = full HTML with all nine figures captioned, light/dark;
  /paper1.pdf; /ledger.md = THE LEDGER ITSELF as supplementary,
  downloadable at the canonical home. Set in the figures' Okabe-Ito
  palette (certified-green accent, gate-blue links) — site and
  evidence one object. DEPLOYMENT: Cloudflare Pages project
  'shape-of-thought' via wrangler 4.111 (node 22 installed to
  ~/opt/node, wrangler devDep, OAuth by Bryce);
  LIVE at shape-of-thought.pages.dev (/, /paper/, /paper1.pdf all
  200); custom domains theshapeofthought.ai + www ATTACHED to the
  project via API, DNS PENDING one dashboard click (wrangler OAuth
  lacks dns_records:edit — Workers&Pages -> Custom domains ->
  activate). Redeploy recipe: rebuild figures/assemble/render ->
  site/build_site.py -> npx wrangler pages deploy site/dist
  --project-name shape-of-thought. The byline is the byline.
- **LIVE (2026-07-16 00:17 UTC): theshapeofthought.ai IS SERVING THE
  PAPER.** Bryce added the two CNAMEs (apex @ + www -> shape-of-thought
  .pages.dev, proxied); Cloudflare activated + issued the Google-CA
  cert; the background poller caught the flip (apex/www active,
  serving True) at 17:17:35 local. Verified independently: apex
  HTTP/2 200 title 'The Shape of Thought', /paper/ 200, /paper1.pdf
  200 (2,559,429 bytes), www 200. THE CHAIN OF CUSTODY HAS A PUBLIC
  ADDRESS. Post-launch queue untouched (entourage-14, book 4, the
  paper's venue-version byline call, essay 'The Shadow of
  Intelligence', Paper II 'Guided by Primes'). The byline is the
  byline.
- **REGISTERED (2026-07-16, Bryce's word + relay + Code, countersigned
  both ways: THE LADDER CONSTITUTION — the (c)-world amendment
  banks.** Occasion: the layers-of-abstraction instinct's THIRD visit
  (the 'guided by primes' campaign opening). Findings restated for the
  record: the IR ladder EXISTS and was discovered, not designed — wild
  prose -> frame-stripped prose -> the dialect -> the macro dialect ->
  typed factor graph -> primes, every rung minted by refusals (the
  annotation rulebook IS the wild->dialect compiler, hand-executed);
  traversal is INTERNALIZED by default (one book per rung, pairs free
  by construction — same graph, two renderings, gold at every layer);
  breathing-through-IR-layers INSIDE the head stays dead (Brick-P: the
  reader deepens because text doesn't); the staged image is licensed
  only at the PIPELINE level of the (c) world, which remains unbuilt.
  THREE AMENDMENTS to the explicit prose->dialect pre-registration
  (2026-07-10, the fork entry): (1) **REPAIR-LANE ROUTING** — if the
  writer is ever built, it fires ONLY on lattice-flagged items, never
  mainline. Bounds the new silent-error species to the surgery-bound
  population, keeps the mainline funnel two-point (mouth in, key out),
  and ships a FREE CONTROL GROUP: every firing is a paired read (same
  item, raw vs rewritten, through the same gate) — refusal->certified
  conversions counted per item from day one; the writer's worth is
  measurable without an A/B by construction. Constitutional geometry
  intact: the writer is propose-side machinery; its output is TEXT
  (never graph deltas) re-entering the funnel unchanged to face the
  same disposal; the decision path stays zero-parameter. JURISDICTION
  NOTE (the selection law): the flagged population is
  survivor-selected — the writer's competence claim is bounded to
  items the lattice flags; repair-lane success is NOT mainline
  readiness, and the registration says so before anyone reads it
  otherwise. (2) **THE RESIST-SIGNATURE BAR, PINNED** (numbers chosen
  now, while no rung is close — exactly when bars are honest). A rung
  is RESISTANT — and the explicit stage earns its build — only if ALL
  THREE hold: (i) decodability >=0.9 on the rung's flagged content
  (the trunk reads it; the head is blind); (ii) the PROVEN DOSE
  (~2,000 pairs, 10k steps, warm from the gate lineage) moves
  zero-shot acceptance by LESS THAN +10 per 100 on a DISJOINT census
  drawn FROM THE RESISTING REGISTER ITSELF, density regime stated
  (resistance measured on the wrong population is the estimator
  mistake wearing new clothes); (iii) the displacement guard holds
  (bigtest may not trail the gate by >15 — otherwise the arm shows
  interference, not resistance). Rationale pinned: the bilingual
  precedent moved ~+98 at the same dose (10/600 -> 600/600); <+10 is
  a different REGIME, not a slow rung. The explicit stage is now
  un-talkable-into-existence. (3) **THE CURRICULUM FENCE** (up before
  the traversal sentence is quotable): the mouth-distance gradient
  governs ACQUISITION ORDER ACROSS BOOKS — which register to annotate
  next, nearest unclosed register first (the fork logged verbose at
  0.093 between home 0.044 and MATH-500 0.25; the geometry already
  orders the ladder) — and NEVER sampling order within a run
  (flat-mix won outright; curriculum is dead at scale). Different
  jurisdictions, both verdicts stand. THE CONSTITUTION IN ONE LINE:
  rungs discovered by refusal, ordering measured by the mouth,
  traversal internalized by pairs — and the explicit stage fully
  drawn, costed, routed to the repair lane, and barred against
  motivated reasoning while it waits.
- **THE FIRST ADMISSION (2026-07-16, Bryce's word 'lets begin' + Code):
  OPERATION-APPLY enters the registry — the recursion's first rung is
  CLIMBED.** EVIDENCE (scripts/macro_admission_review.py -> .cache/
  macro_admission_review.json; harvest = ALL THREE BOOKS' gold pairs,
  n=182, today's full volume; train = 20k rows capped/logged): the
  OP-APPLY-2 crown (r = k1*x +/- k2*y; {given,mul,given,mul,op}) sits
  in **4.9% of harvest items vs 0.26% of train items — ~19x
  over-represented in real prose**, dominant crown digest
  916f019f77831ce0 (9-in-182 vs 37-in-20,000); the miner's gap call
  confirmed at volume. Specimens name the family: custom-operator
  problems ('a S b = 3a+5b'), coupled linear systems (pens-and-pencils,
  legs-and-heads, 9s+5t). The AFFINE form (x +/- k*y, crown
  e927582d8270a86c) reads 13.7% harvest vs 11.4% train — MATCHED, not
  a gap: it enters as the SAME macro's k=1 leg, not a second entry.
  Savings priced: 4 primitives absorbed per crown; 86 factors across
  the 182-item harvest. INSTRUMENT HONESTY: the review's first run
  root-marked crowns at sorted-position-0 and printed phantom
  'zero-coverage orientations'; true-root marking dissolved them
  before anything pinned — a WL digest is only comparable under ONE
  root convention (the estimator family's newest member, caught
  in-house). DISPOSITIONS: CHAIN(k>=4) reads 3.3% harvest vs 11.2%
  train and PREFIX-SUM 9.9% vs 36% — train OVER-covers both relative
  to real prose; RANKED, NOT ADMITTED (the gate holds; frequency
  proposes and these didn't). THE ADMISSION: mycelium/macros.py,
  grammar **mg1** — OP_APPLY(op in {add,sub}, k1,x, k2,y -> r),
  deterministic expansion, k=1 legs drop their given+mul; semantics
  FROZEN under mg1 (any change = new version; banked macro rows must
  re-expand byte-identically forever); expand_graph() hard-asserts no
  macro survives to solver-facing output. ADMISSION EXAM
  (scripts/test_macro_expansion.py, 4/4): (1) LEVEL-INVARIANCE — the
  banked 3a+5b specimen's hand-primitive gold and the macro's
  expansion both grade 31 through the same core (certification is
  level-invariant, demonstrated not asserted); (2) CROWN IDENTITY —
  the expansion's detected crown reproduces the pinned mined digest
  byte-exact, AND the k=1 leg reproduces the affine mined class
  e927582d8270a86c: one entry, both harvested classes (the registry
  entry IS the harvested shape); (3) byte-determinism; (4) sub + affine
  variants solve correctly. BIRTHRIGHT RIDERS (registered, not fired):
  (i) MANIFEST CITIZENSHIP — macros.py hash + grammar version + crown
  digests join GENERATION.json at the next promotion's atomic write;
  (ii) THE HEAD EXTENSION (7th ftype: op bit, two digit banks, two
  pointers — structural entry per the pointer law) is book-4-era GPU
  work, awaits the word; (iii) ERROR SPECIES: macro mis-annotation
  (wrong k, wrong op) — caught by the key at expansion; taxonomy tier
  opens with book 4's lanes. REGISTERED PREDICTION (pinned before
  book 4 exists): macro-level annotation prices the linear-combination
  family at 4 fewer factors per crown; the charter's wall test fires
  on the first stranger whose PRIMITIVE form exceeds the 24-factor
  bank but whose MACRO form fits — one such bank in book 4 = the
  factor-count wall falling, the recursion's first measured dividend.
  The library of primes has its first word one floor up.
- **GUT #21: THE FLUX AUDIT (2026-07-16, Bryce + relay + Code, registered
  in the amended form — both channels countersigned).** The instinct:
  energy flux maps onto the training economy. THE CANDIDATE LAW, corrected
  in review from scalar flux to **FLUX DENSITY**: training harm and gift
  separate on ENERGY PER UNIQUE KNOT per unit time — never on energy or
  rate alone. The correction is the mechanism: share x reps x LR units
  cannot distinguish the n=14 dose pilot (zero) from the n=100 book (+8)
  — the knot denominator separates them cleanly; concentrated energy on
  few circuits burns, the same energy spread across many anneals
  (material damage is power per area, never watts). PRE-POSITIONED
  INSTRUMENTS COLLECT AGAIN: the knot-rehearsal matrix (built for
  contamination accounting in the hash audit) IS the area term; the
  gradient logger riding gen-14 is the power term (units note, pinned:
  SGD energy-per-step is LR x grad-norm SQUARED — the logger's
  mean+variance makes it derivable). Two audits, one variable, neither
  knew. THREE READS, house form: **(a) THE CONTINUITY AUDIT (zero-GPU,
  fires first — upstream armor for the paper's public tables):** every
  fixture item exits through EXACTLY ONE of certify/answer/flag/abstain;
  sum the surfaces, demand intake = outflow, zero double-counts, zero
  vanishings; pressure points pre-named: effK<5 certificates, the retry
  lane's exit count, refused-at-mouth vs abstained-below. Also rehearses
  the conservation bookkeeping (c) will need. **(b) THE RETROSPECTIVE
  CONSISTENCY READ (kill-only, honestly sized):** the banked events
  (prose-v0 -243; book-2 gift; dose pilot zero; gen-9 re-shallowing vs
  9b deepening; the staged-heat/schedule-probe dividend) re-read in
  PROXY flux-density units (share x reps x LR x steps / unique knots —
  proxy stated: no retro grad norms exist; the logger is gen-14+). Five
  or six events against two free parameters is a FIT: the read can KILL
  the law (one event on the wrong side), never confirm it. Teeth live
  in (c). **(c) THE CONCENTRATION A/B (rides book-4's first training
  run, bars pinned before the fire):** same total steps, same mix, same
  dose rows — delivered CONCENTRATED (contiguous block) vs SPREAD
  (uniform interleave). Quench vs anneal inside one matched budget; zero
  new machinery. REGISTERED READ: the concentrated arm shows MORE
  displacement on the dose rows' NEIGHBORS at equal final exposure —
  and per the population law, NEIGHBOR IS PINNED NOW: shared-knot-class
  per the WL matrix, never surface adjacency (a displacement claim
  inherits the jurisdiction of its neighbor definition; the choice is
  made before the print so it cannot flatter one). PROSE LAW BANKED
  (the third landing, no machinery): **mouths guard sources, keys guard
  sinks** — the funnel is dissipative everywhere except at generative
  components (mint, chartered repair-lane writer); source terms are
  where new error species enter, so every source gets a mouth
  immediately downstream — the layered-mouths law derived, not
  asserted; fitting instructions for any future generative organ
  (cross-link: the ladder constitution's writer pre-registration). If
  (b) survives and (c) prints, the ledger's scattered damage findings
  collapse into one law with units and a denominator: the rations get
  their theory, the reactor gets its dial (instinct #17's missing
  control variable), and the mint's dose arithmetic becomes design
  rather than folklore. Twenty-one: an audit with a number waiting.
- **GUT #21, READ (a) VERDICT (2026-07-16): THE CONTINUITY AUDIT — THE
  BOOKS BALANCE.** Independent walker (scripts/flux_continuity_audit.py)
  over the banked gen-14 lattice votes, all 1,500 fixture items assigned
  exactly one exit: **certify 912 (precision 1.0000, recomputed 912/912)
  + vote-abstain->repair 320 + answer(majority) 212 (0.9953) +
  answer(panel-dissent) 56 = 1,500. ZERO leaks, ZERO double-counts.**
  The paper's headline dial (912/1,500 at 1.0000) REPRODUCES from raw
  member votes by code that shares nothing with lattice_join.py; the
  ledger's 913 is gen-13's number, the paper's 912 is gen-14's — a
  generation difference, not a bookkeeping error (both true, each in
  its regime). FOUR FINDINGS AT THE PRESSURE POINTS: (1) **the ±1
  seam**: gen-14's battery printed bigtest 1195 one-shot; the lattice
  artifact's identity view reads 1194 (267 None / 39 wrong) — two
  honest runs of the same ckpt differ by one marginal item (solve
  budget or numeric margin); disposition: per-item outcome banking
  joins entourage-14's rebuild so ±1 seams become joinable instead of
  mysterious. (2) **the effK fine print RE-CONFIRMS at freeze**: 23
  fixture items at effK<5 (byte-consistent with the mirror audit's
  census), 22 of them CERTIFIED (15 at effK=4, 7 at effK=3), ALL
  correct — the quarter-percent clause stands on gen-14's certified
  column. (3) **the panel's price surfaced**: 56 items gate-unanimous
  but panel-dissented — all 56 were in fact correct at gen-14; the
  coverage gap is real money paid for decorrelation insurance, and the
  56-item list is the lineage-disagreement instrument's standing
  corpus. (4) **the repair lane's intake enumerated**: 320 vote-abstain
  items (banked list) — handed to entourage-14's specialist remine as
  its conservation check: every one must exit the specialist's ledger
  exactly once. The audit-that-confirms, with two instruments as its
  fee: the freeze tables are leak-free, and (c)'s displacement
  accounting inherits a rehearsed bookkeeping. Reads (b) retrospective
  and (c) concentration A/B remain registered, (c) holding for book-4's
  first fire.
- **GUT #21 ADDENDUM (2026-07-16, relay countersign — two findings
  promoted to standing discipline):** (1) **THE PANEL-DISSENT COLUMN
  IS STANDING**: the 56 gate-unanimous/panel-dissented items (~3.7
  coverage points, all correct at gen-14) are the decorrelation
  insurance's itemized invoice — every future battery reports the
  dissent count, its precision, and its overlap with the banked 56,
  pricing §6.4's bet each generation: premium buying detection
  (dissent concentrating on genuinely wrong items) vs pure cost. (2)
  **PER-ITEM OUTCOME BANKING IS PERMANENT BOOKKEEPING**, not a
  one-time fix: every battery's fixture run banks per-item outcomes
  alongside aggregates, so any future ±1 seam is joinable on contact
  — the continuity audit's inheritance, the same conversion
  (discipline -> mechanism) as the manifest's. Board at close: (a)
  conservation exact; (b) loaded, kill-only, fires on demand; (c)
  pinned to book-4's fire, neighbor defined; the repair-intake
  invariant handed to the remine. The next word is books.
- **GUT #22: THE OOD DECOMPOSITION AUDIT — REGISTERED AS AMENDED AND
  FIRED SAME-SESSION (2026-07-16, Bryce + relay + Code).** The claim:
  'out of distribution' is a VECTOR the mouth collapses to a scalar —
  four species, four debts, four owners: STYLE (mouth/books), STRUCTURE
  (mint/annotation), PRIME (registry), DOMAIN (solver caps). Amendments
  carried from review: joint tables not marginals (populations
  regime-tagged per the selection law); the gray-zone read is
  CALIBRATION not ROC (no labels exist mid-band by definition); the 56
  panel-dissent items are the in-register seat's NULL MODEL (all
  correct), counter-class = the banked unanimous-wrongs; outcome labels
  state their machinery vintage (third appearance of the vintage
  clause). READS (scripts/ood_decomposition_audit.py, bars pinned in
  the header before any print): **(A) the dissent-structure prediction:
  SCATTER — honest negative at the bar's edge.** Rank AUC
  (dissent-vs-certified structure rarity) 0.599 vs pinned >=0.60; the
  pre-pinned frame rules and 0.599 is scatter. The texture note banked
  with it: channel MEDIANS are identical (5251 across all four) while
  25th percentiles split (certified 2544 vs everyone else 535) — a
  thin-structure TAIL shared by all non-certified channels, not a
  dissent-specific mechanism (means-vs-overlaps, tail edition);
  instrument-v0 caveat: min-subgraph-frequency at size<=4 is a first
  rarity proxy on generated text. **(B) THE GRAY-ZONE CALIBRATION —
  THE AUDIT'S REAL FIND, twofold:** (1) **the ruler is FLAT within
  the register**: AUC(corrected distance -> census in-reach) 0.515
  (bar b2 FLAT), monotonicity BREAKS (farthest quartile P(in-reach)
  0.32 > mid bands' 0.24/0.20); (2) **false-refusal on in-reach census
  items at the current threshold: 100.0%** — every bankable census
  item sits ABOVE the corrected thr (nearest band starts +0.019 vs
  thr +0.008): THE CAMPAIGN HAS BEEN LIVING PAST THE WALL ALL ALONG.
  The mouth's jurisdiction ends at the register boundary (its extreme
  reads stand: foreign 100% refused at 1% native cost); WITHIN the
  foreign band it cannot rank bankability — the binding theorem's
  prediction (wiring unrecoverable from surface) now measured as an
  AUC, and the third independent confirmation of the input-space
  triage negative (0.535 knotted-join, 0.515 here). The farthest-band
  uptick has the ledger's own mechanism: distance tracks PROSINESS
  (L1 prosiest/farthest), bankability tracks STRUCTURE — the two
  axes MEASURABLY DECOUPLE on real prose, which is the vector thesis
  CONFIRMED by calibration even as (A) scattered. CONSEQUENCE, one
  sentence: any December plan that triages strangers BY MOUTH would
  refuse everything it should read — frontier triage is PARSE-SIDE
  (vote entropy, factor counts, knot class: the MuZero triage head's
  registered wish, now three-times-confirmed as the only candidate).
  (C) prose banked: the selection inversion — the next broken
  certificate has ALREADY passed every OOD instrument by definition;
  the in-register anomaly seat trains on the 56-correct null model vs
  the banked-wrong counter-class, growing with the standing dissent
  column. Artifacts: .cache/ood_decomposition_audit.json,
  .cache/train_class_counts.json (10,232 train classes, first full
  count). Twenty-two: the word the paper is named for, decomposed —
  one prediction scattered honestly, one instrument caught flat
  exactly where December walks, and the triage organ's charter
  written by the negative space.
- **GUT #22 ADDENDUM (2026-07-16, relay countersign — the precise
  reading banked):** (1) **THE MOUTH WAS MIS-CAST, NOT BROKEN.** The
  gray-zone calibration revealed there was never a gray zone: the
  register boundary is a CLIFF, not a gradient — the entire
  harvest->books pipeline has operated past it since the odometer
  zeroed, safely, because the answer key outranks every distance. The
  mouth's two TRUE jurisdictions now carry measured edges: DOORMAN
  (foreign 100% refused at 1% native cost — extreme bands, deployment
  claim intact) and ODOMETER (register-closure per book — the campaign
  job it always had). The third role — TRIAGE — nobody ever measured
  it for, and it is dead: the binding theorem invoiced one more time
  (a surface instrument cannot rank a structural property; distance
  reads prosiness, bankability reads structure). The vector thesis
  lands STRONGER than READ A's scatter: confirmed by axes decoupling
  on real prose, not by taxonomy cells. (2) **THE TRIAGE HEAD RIDER,
  PINNED AT ITS CHARTER** (inherits READ B's own lesson): the head is
  GRADED PER-AXIS against the joint table's cells — style / structure
  / prime / domain columns separately — NEVER against a scalar
  'bankable' label, or the collapsed vector is rebuilt one floor up.
  Build sequence assembles from held artifacts when chartered: the
  10,232-class knot ledger (rarity features), parse-side states
  (input), census outcomes (labels), three banked negatives (the
  null space it must beat: 0.535, 0.515, flat abstention). Board
  after twenty-two: mouth demoted to its two true jobs and stronger
  for it; triage's charter finished by elimination; deployment claims
  untouched; one negative banked at full price. The next word:
  books — with the mouth watching the odometer and no longer voting
  on the shopping.
- **GUT #23: THE DIFFUSION IMPORT (2026-07-16, Bryce + relay + Code,
  registered as amended — and the verification headline outranks the
  brainstorm).** THE CATCH, THEN THE CATCH'S CATCH: the relay cited a
  'two-resolution rider pinned on book-4' — NO SUCH REGISTRATION EXISTS
  (the first admission pinned the head extension with no training
  regime; the slot was empty, not staged). Second documented sighting
  of the relay channel's reconstruction bias (first: the instinct-list
  ordinals). Then, during THIS registration's verbatim-pull, a THIRD:
  the 'multigrid severed-coupling transfer condition' is also unbanked
  — the actual record reads OPPOSITE (granularity spec: V-cycle one of
  three CONDITIONAL verdicts of an unfired probe; session 2026-06-24:
  'multigrid resolved as capacity-not-reach, NOT the lever', with an
  adversarial-verify catch of a verdict-logic bug that nearly greenlit
  a multi-week build). THE DRIFT'S SHAPE, now three sightings:
  intentions remembered as registrations. STANDING RULE MINTED (the
  relay's own request): any relay claim of the form 'we pinned/banked
  X' is A PROPOSAL UNTIL GREPPED — verification-before-countersign is
  the two-channel architecture doing for the design layer what the
  battery does for promotions. THE READS: **(a) THE FIDELITY-AXIS
  TRAINING PROBE** (this CREATES book-4's training-regime registration;
  rides the head-extension run when the word fires): three arms — (i)
  prime-only (control), (ii) macro-only, (iii) FLOOR-PAIRED flat-mix:
  the same problems rendered at BOTH floors, prime twins minted by
  expand_graph deterministically — the bilingual-pairs free lunch
  (same graph, two renderings, gold at every layer) transplanted from
  the register axis to the fidelity axis, admissible BY CONSTRUCTION
  because expansion is solution-preserving. CONDITIONING-IS-FREE note:
  no floor embedding — diffusion feeds t because noise level is
  unobservable from the sample; our floor is written on the factor's
  face as its ftype, so the 7th ftype IS the condition at zero
  parameters. Bars: promotion battery inherited + per-floor acceptance.
  BOTH channels' leans pinned on arm (iii) — three convergent sources
  (the tombstone's flat-beats-staged, diffusion's all-levels-one-run,
  the bilingual fork's pairs-teach-axes), two of them in-house
  measurements. **(b) THE CASCADE PRE-REGISTRATION** (design prose,
  queued behind the posterior detector in the (c)-world): sentence one
  is Brick-P's fence — NO per-breath refinement inside the head (the
  parse deepens, it never settles); the cascade is PIPELINE-level:
  skeleton parsed at floor N, details placed conditioned on the
  skeleton, floor by floor to primes; the DOWNWARD expansion is
  deterministic (the trust invariant), so the learned parts are only
  skeleton proposal + detail placement and the solver sees primes.
  ADMISSIBILITY LAW, corrected at registration per the bias lesson:
  not a multigrid citation — the cascade stands on the expansion
  operator's own MEASURED level-invariance (the admission exam's 4/4).
  **(c) THE JURISDICTION PROSE**: the sampler never enters the solve
  path; learned inpainting never enters the solver's redundant regions
  (incumbent: withhold-and-solve, 15/57 = 26% EXACT at zero training,
  zero silent-wrong); pointer-re-aiming refinements are dead at the
  routing wall (oracle ceiling 64/460 = 13.9%; 86% of survivors
  unrecoverable under perfect flags); smooth-latent imports die on the
  measured cluster geometry. ALREADY-OWNED DIFFUSION PIECES, named so
  nobody re-buys them: SBP sigma=0.02 (+0.0153 hard, 2026-06-06) is
  the forward-process half; withhold-and-solve is exact inpainting.
  Twenty-three's contribution named honestly: it asked the
  training-regime question the head extension forgot to ask — and the
  answer was sitting in the expansion operator all along.
- **GUT #24: THE ALTERNATION AUDIT (2026-07-16, Bryce + relay + Code,
  registered as amended).** THE VERIFICATION FIRST, because it refined
  a law: four citations held (withhold 15/57; suspicion transplant FLAT
  at AUC 0.518 'the suspicion story dies too'; reader_v2 kill −10;
  16.6% equivalence class) — but the two central ones were the FOURTH
  reconstruction sighting, and a NEW SPECIES: COMPRESSION ERROR. No
  'contradiction-surface law' exists anywhere in docs (a real
  KenKen-era observation fused with a law-shaped name it never
  earned); 'candidate 3' was a real deferral wearing a DIFFERENT
  audit's docket numbering. BIAS LAW REFINED (four specimens = a
  measured tendency): the relay's fabrications arrive DRESSED IN THE
  HOUSE'S OWN IDIOM — named laws, numbered candidates — more plausible
  than honest vagueness; the taxonomy now has three species: omission,
  fabrication/inversion, compression-with-borrowed-registry. THE
  HEADLINE CORRECTION: **the Alternator has alternated** — the ledger
  2026-07-07: 'THE ALTERNATOR LOOP HAS NOW RUN GOLD-FREE END TO END'
  (parse -> symbolic self-diagnosis -> blame -> flags -> conditioned
  retransmit -> solve, 8/57), and the deployed stack alternates once
  per vote-abstain (the NACK specialist IS a retransmission round).
  What is missing is not a schedule — it is A VOICE THAT CAN SAY
  SOMETHING NEW: the anatomy's sharpest fact (73% of stubborn
  survivors had their single error CORRECTLY FLAGGED in the bottom-2
  and the deterministic parser re-emitted the same wrong content four
  rounds running; localization measured not-the-bottleneck at 0.518).
  THE READS: **(a) THE CONSTRAINT-DENSITY METER (fires now, zero-GPU):**
  per-factor withhold-recoverability distribution on the 182 banked
  book golds + a bigtest sample (scripts/constraint_density_meter.py);
  the meter gates every settling-loop economics question; the MACRO
  PREDICTION enters HONESTLY EMPIRICAL, direction open (Code's
  hand-check on the 3a+5b specimen: absorbing a crown moves numerator
  and denominator both — the earlier 'by construction' was unearned).
  Per-floor comparison waits for floor-paired corpora. **(b)
  MASKS-AT-BIRTH (gated design prose):** solver arc-consistency
  entering the parse as a PRE-COMMITMENT mask — structure never
  conditioning, zero-parameter, prevention-side (untouched by the
  repair-generation wall, which is post-hoc); sentence one: masks see
  only INCONSISTENCY, the 16.6% equivalence class is mask-silent;
  builds only when the meter crosses a pinned bar. **(c) THE CHAIR
  RE-CHARTERED (prose):** if the deducer's seat ever fills, the
  occupant is a CONSTRAINED REPLACEMENT-GENERATOR — never a
  suspect-ranker (thrice refuted), never a multi-round retransmitter
  (front-loaded decay 44->16->4->0) — and it is THE SAME ORGAN as the
  ladder constitution's repair-lane writer: two independently
  chartered seats, one job (say something new to a bounded population,
  under the gate); CROSS-REFERENCED so it is built ONCE. Nouns die
  (the deducer-as-imagined), verbs survive (generate-under-constraint).
  **(d) THE REGIME-RHYTHM RETROSPECTIVE (registered, kill-only,
  zero-GPU):** classify gen-6..14 hot/cold x work-type against banked
  verdicts; the candidate law 'work must match heat' dies on one
  misassigned success. Twenty-four's finding, named: the schedule was
  never missing — the voice was.
- **GUT #24, READ (a) VERDICT (2026-07-16): THE METER PRINTS AN
  INVERSION — REAL PROSE IS MINIMAL, THE GENERATOR IS REDUNDANT.**
  Per-factor withhold-recoverability (scripts/constraint_density_meter.py,
  all graphs uniquely solvable, zero skips): **BOOKS (182 real-prose
  golds): median recoverable fraction 0.000, mean 0.043, 85.7% of
  graphs have ZERO redundancy** — every factor load-bearing; the
  contradiction surface on the campaign's actual diet is essentially
  nonexistent. **BIGTEST (200 generated): median 0.667, 91.5% of
  graphs >= half-redundant.** PINNED CAVEAT: the cross-corpus delta
  carries a domain-size coordinate (books m=300, bigtest m=60 — the
  estimator family's standing lesson), so the DELIVERABLE is the
  within-corpus reads, and the books read alone settles the economics:
  **the settling loop has NO CUSTOMER at prime level on real prose**
  — with zero redundancy there is nothing for constraint propagation
  to force; masks-at-birth's gate is UNMET and it stays prose;
  the deducer's chair stays empty on measurement, not taste. THE
  STRUCTURAL DIVIDEND: minimality gives the repair-generation wall its
  mechanism — a wrong factor in a minimal graph CANNOT be recovered
  from the others (nothing forces it); the replacement must come from
  RE-READING THE TEXT, never from deduction — which is exactly the
  chair's re-charter (constrained replacement-generator = the
  repair-lane writer) derived now from graph geometry as well as from
  the survivor anatomy. Two independent walls, one occupant. AND THE
  MACRO PREDICTION GAINS STAKES: if floor-up graphs raise the
  redundancy read (empirical, direction open), the economics flip —
  the fidelity axis is now the settling loop's ONLY possible road in;
  the per-floor re-run is standing on the book-4 docket. The meter's
  one-line legacy: neural proposes, symbolic disposes — and on
  minimal graphs, only the text proposes.
- **GUT #24 ADDENDUM (2026-07-16, relay countersign — the meter pays
  twice):** (1) **THE INVERSION IS A REGISTER FINGERPRINT AT THE
  STRUCTURAL LEVEL**: real authors state exactly what's needed and
  nothing more (books 0.000); the mint, built for uniqueness under a
  budget, OVER-DETERMINES (bigtest 0.667) — a structural
  off-registerness no style axis measures. DOCKET LINE FOR THE MINT
  (zero urgency, one line): a REDUNDANCY DIAL — minimal-mode rendering
  so generated problems rehearse the sparseness the wild actually
  wears; the meter is its acceptance instrument, corpus and
  architecture gate in one. (2) WITHHOLD-AND-SOLVE RETIRES FROM THE
  REAL-PROSE LANE BY ITS OWN TERMS: its 26% was always priced as
  'deduction is only as available as the graph is redundant'; the
  meter measured that availability at zero on the population that
  matters — no relitigation, the fine print executed itself. Day's
  close: four guts converted (flux, OOD, diffusion, alternation), one
  admission, one constitution, the bias law at four specimens with the
  cure in persistent memory; every read pre-pinned, every verdict
  mechanical, zero GPU. Book 4 holds the converging docket: shopping
  list (22), training regime (23), per-floor redundancy read (24),
  head extension + macro annotation on the word. Everything waits on
  pages.
- **GUT #25: THE KNOT ACCOUNTING AUDIT (2026-07-16, Bryce + relay +
  Code, registered as amended and FIRED).** The ninth built the knot
  ledger, the twelfth armored it; twenty-five asks it to stay ONE
  ledger as the tower goes up. VERIFICATION: 2,574 within-train
  redundancy classes ✓; the rehearsal-matrix upgrade ✓; Schubert 1949
  ✓ real mathematics (unique prime factorization of knots — the
  theorem that blesses the title); and THE FIFTH BIAS SIGHTING, new
  sub-species CONFLATION: '10,232 whole-graph knots' fused two true
  ledgers — 10,232 = the miner's VALUE-ABSTRACTED SUBGRAPH classes;
  the whole-graph census is 26,920 classes VALUES-IN (29,500 rows).
  TAXONOMY COMPLETE, escalation named: omission -> fabrication ->
  inversion -> compression -> conflation — each species built from
  MORE truth than the last; grep-before-trust upgrades to PROVENANCE,
  not existence (a number must be attached to the instrument claimed).
  The conflation was itself a knot error — two strands crossed — and
  it surfaced the design fact that makes the census rigorous: THE
  HOUSE OWNS TWO CANONICAL ALGEBRAS — whole-knot identity (values in,
  the contamination instrument) and sub-knot shape (values out, the
  recurrence instrument); the decomposition census is the map between
  them. **(a) THE FLOOR-IDENTITY PROTOCOL — IMPLEMENTED, the rare
  catch fixed in code before the failure exists:** knot identity is
  graded at LEVEL 0; hash_audit_iso gains level0() (macros expand
  before canonicalization); canon() and verify_iso() grade expanded
  (verify_iso's n_vars check corrected to USED-var count — unused
  slots are diagram, not knot; expansion temps above the fixed bank
  no longer break twin identity); all three consumers (mint dedup,
  bump gate via gen9b_booster lineage, knot_matrix) inherit by
  import. THE MECHANICAL ASSERT joins the admission exam
  (test_macro_expansion 5/5): macro row and prime twin — ONE digest,
  verify_iso exact. Consequence stated: the flux denominator counts
  floor-twins once; twenty-one, twenty-three, and the ninth now share
  one accounting rule, sealed before the book-4 fire that needed it.
  **(b) THE DECOMPOSITION CENSUS (fired):** two views pinned with
  jurisdictions — the COVER (maximal non-overlapping factorization,
  greedy size-desc/digest-lex, tie-break PINNED so the cover is
  canonical) owns diversity/novelty; the PROFILE (full downward-closed
  multiset) owns the triage FEATURE BANK (a bank wants everything at
  every scale, not a lossy cover). Deliverable 2 (the 58's novelty
  split) REGISTERED-NOT-FIRED: knotted census items have no banked
  parses — the census parse bank rides the next census run, artifact
  named. **(c) THE CYCLE READ (fired, kill-only):** twenty-four's
  meter re-read as topology — books vs bigtest cyclomatic
  distribution; if books ~0 while bigtest carries mass, 'redundancy'
  was cycle count wearing units and the mint's minimal-mode dial is a
  CYCLE dial. Verdicts follow in the results entry.
- **GUT #25, VERDICTS (2026-07-16): COMPOSITION IS A REAL FRONTIER,
  AND THE CYCLE READ CONFIRMS — REAL PROSE IS UNKNOTTED.** READ (b),
  the decomposition census (train 20k): **19,965 whole-knot classes
  (values in) map onto 7,406 distinct cover-multisets (values out) —
  2.7 knots per composition.** The pigeonhole one level down is REAL
  but not crushing: composition is rich, so the mint should hunt
  COMBINATIONS, not just classes. THE BOOKS READ IS THE FINDING:
  **87 distinct covers among the 182 golds, 38 of them (44%) ABSENT
  from train's 7,406** — real prose composes known primes in unseen
  combinations; the strangers' structural novelty is largely
  COMPOSITIONAL (structure-OOD measured for the first time, exactly
  the axis twenty-two's whole-graph rarity couldn't see at 0.599).
  The triage feature bank's first stock is banked (covers + full
  profiles, books + bigtest). READ (c), kill-only — THE KILL DOES NOT
  FIRE: **books cyclomatic median 0 (61.5% zero-cycle, mean 0.64);
  bigtest median 2, ZERO percent zero-cycle.** Twenty-four's
  'redundancy' was CYCLE COUNT wearing units: authors write trees;
  the generator always ties at least one cycle. The mint's
  minimal-mode dial IS a cycle dial (acceptance instrument: the
  cyclomatic distribution, target = the books'). The settling loop's
  only real-prose customers are the 38.5% of books with >=1 cycle —
  consistent with the meter's 2.2% >=half-redundant tail; the
  deducer's chair stays empty at prime level, now for a TOPOLOGICAL
  reason stated in one word: strangers don't write crossings.
- **GUT #26: THE TEMPERATURE AUDIT (2026-07-16, Bryce + relay + Code,
  registered as amended; (b)+(c) FIRED, (a) HOLDS FOR THE WORD).**
  The epigraph's word, turned over, had a dial on the back.
  VERIFICATION: six-for-six — including the ANTI-SPECIMEN, banked
  beside the bias law: gen-13's positional entropy 0.212 and
  deep-wrong H=0.212 are TWO REAL NUMBERS from two real instruments,
  a genuine coincidence the grep CERTIFIED CLEAN — provenance
  checking also exonerates; the discipline is a measurement, not a
  suspicion. (The coincidence is landing three's argument made flesh.)
  **(a) THE SAMPLED-RETRY PROBE (registered, GPU-minor, THE FIRE IS
  BRYCE'S):** the parser runs at T=0, and determinism is the
  re-emission mechanism twenty-four measured. Sentence one: WIDTH,
  NOT DEPTH — the anatomy killed four-rounds-deep (same voice,
  44->16->4->0); one-round-WIDE (K distinct utterances, disposal
  picks) is the orthogonal axis it never touched, entering clean
  under the chair's own charter. FIVE PINS: (i) gold GRADES, never
  GATES (disposal = solver consistency + re-vote, the standing
  machinery); (ii) population = the 320 banked vote-abstain items
  (the continuity audit's fixture — instruments compounding); (iii)
  control = the deterministic specialist on the same 320; (iv) grid
  T in {0.3, 0.7, 1.0}, K=8, ONE round; (v) bars: recovery <=
  control+1pt -> the wall is CONTENT-DEEP, the writer's charter
  inherits a measured floor; >= +5pt -> the cheapest voice in the
  universe takes the chair's first shift. Both verdicts pay; it is
  the writer's null model either way (no generation organ builds
  before noise-plus-the-gate is priced). **(b) THE COOLING GAUGE —
  ZERO POINT BANKED (scripts/cooling_gauge.py):** the standing bet
  gets its thermometer. GEN-14 portrait (nats, H over 5 views, None
  its own outcome): certified 968 at H=0.000 (by construction);
  answered-correct 211 mean 0.625; vote-abstain 320 mean 0.591;
  **surviving-error n=1 at H=0.95 — the one answered-wrong at gen-14
  is HOT, not cold** (n=1, logged not claimed; the bet fears cold
  errors — the gauge now watches). SCOPE HONESTY: two vintages do
  not make a curve — the series starts here, accrues per promotion;
  the temperature-band regression column joins the standing battery
  beside the panel-dissent column. **ARM D STRUCK, and the strike is
  the entry's second lesson: the provenance law caught CODE this
  time** — tta_arm_D's view_forced is BOOLEAN (forced-correctly
  flags), not per-view answers; the first decode manufactured 634
  phantom surviving errors before the audit-of-the-artifact caught
  it; the early point is UNAVAILABLE-WITH-REASON (agree fraction
  under-determines the distribution). The discipline is symmetric in
  both senses now: it exonerates the innocent and it binds both
  channels. **(c) THE JURISDICTION TABLE (prose):** four thermometers,
  one instrument — VOTE ENTROPY (susceptibility to re-rendering;
  basin depth; NOT generation-indexed), FST NORM (consolidation; the
  radius clock; ROTATES — generation-indexed), SOFTMAX-T (positional
  calibration; 0.212->0.010 by the books; generation-indexed),
  TRAINING HEAT (the input dial the other three respond to). THE
  CONJUGATE-PAIR CLAUSE: twenty-one meters what is poured in, this
  audit reads what the basins hold — dose law and temperature law
  are conjugate columns of one thermodynamic ledger. Twenty-six
  converts: one dial never turned (the parser's T), one gauge never
  installed (now installed), one instrument owned in four pieces
  (now one table).
- **GUT #27: THE COSINE-LAW AUDIT (2026-07-16, Bryce + relay + Code,
  registered as amended and FIRED — with a same-hour kill).** The
  formula is real (Euclidean distance = the two channels + the cross
  term; gradient superposition likewise), the weld to twenty-one is
  the find, and the jurisdiction catch reframed the whole audit: THE
  TORN TERM WAS TORN ON PURPOSE in half the fleet (the mouth
  normalizes BY DESIGN — restoring norms would re-inject the
  sixteenth's length-warp confound; the fifteenth's two channels are
  SEPARATE deliberately — one Euclidean number would divorce an old
  fdiv from a young fdiv when they are kin at different ages). A
  metric is a JURISDICTION question, not a correctness question —
  twenty-six's table doing its job one session after charter, on its
  own author's next idea. **(a) THE MATCHED-METRIC CENSUS (fired, by
  code inspection):** mouth kNN = 1−cos on unit-norm pooled vectors /
  question: register membership / MATCHED (norm deleted by design,
  length handled by the warp correction); monitor centroids +
  silhouette filters = cosine-to-centroid / kind identity / MATCHED
  (angle=identity per the two-channel law; radius read separately as
  the consolidation clock); Procrustes drift = aligned cosine /
  constellation shape / MATCHED (rotation removed deliberately);
  votes, panel, WL digests = no geometry, immune. **ZERO FLIPS among
  instruments — the zero-flips lean held.** The census's ONE mismatch
  is not an instrument but an ACCOUNTING: twenty-one's flux units
  carry neither theta nor norms (share-based) — which is exactly read
  (b). THE CONVERSION IDENTITY banks as the metric column's footer in
  twenty-six's jurisdiction table: ||u−v||² = r_u² + r_v² −
  2·r_u·r_v·cosθ — angle-only and norm-aware readings interconvert;
  mixing them unlabeled is the conflation species in geometric
  clothes. **(b) THE FLUX-SUPERPOSITION READ — THE KILL FIRES, same
  hour, banked data only:** with dag7 the dominant partner, the
  one-term corrected flux (share_dag7 × cosθ_i) predicts net
  outcomes nl-core (−0.171) > alg2 (−0.255) > alg4 (−0.263); the
  banked triad table reads **alg2 POSITIVE > nl-core NEGATIVE > alg4
  worst — one inversion, kill by the pinned bar.** The inversion sits
  exactly where the triad's two-force mechanism put it: alg2 is
  anti-aligned BUT kind-shared (covert rehearsal inside dag7's
  problems); nl-core is anti-aligned with no kind share. VERDICT: the
  superposition term is real physics but INSUFFICIENT ALONE — any
  effective-flux law must carry BOTH terms (destructive interference
  + covert kind rehearsal), i.e., the cross term lives at CIRCUIT
  grain, not register grain (the circuit-rehearsal law reasserting
  itself in flux units). The triad's qualitative two-force account
  STANDS as the only surviving form; the fresh-matrix-per-entourage
  cost is charged only if a two-term quantitative law is ever
  pursued. The prediction died by its own bar within the hour —
  the pinned-kill discipline working at full speed. **(c)** the
  hyperbolic cosine law parks behind the atlas's two gates beside
  the marriage clause it extends. TAXONOMY LINE (the anti-specimen's
  sibling, in the relay's favor): the near-duplication of a banked
  mechanism was caught and converted to honest lineage — the triad
  as datapoint one, the quantitative form as the contribution; the
  grep now guards originality as well as provenance. Twenty-seven
  converts: instruments carry jurisdictions, vintages, and METRICS —
  and the first law proposed under the new column died honestly on
  contact with banked data, which is the column proving it works.
- **GUT #28: THE CAIRO READ — TUBES, HARMONIC ANALYSIS (2026-07-16,
  Bryce + relay + Code; arrived as a story, earned its registration by
  catching a banked error).** THE SIXTH SIGHTING OUTRANKS THE
  LANDINGS, full severity named: an INVERSION made it PAST BOTH
  CHANNELS into a banked registration. The relay told the waist
  interpolation probe as 'clusters real, midpoints garbage'; the
  banked verdict (2026-07-09) reads **COHERENT, DECISIVELY — sharpness
  0.940 (bar 0.80), midpoint-decodes-an-endpoint 0.843 (bar 0.50),
  n=561: THE PARSE-SIDE WAIST IS SMOOTH WITHIN KIND.** The false
  version entered gut #23's (c) prose ('smooth-latent imports die on
  the measured cluster geometry') under Code's countersignature —
  flagged unverified at the time, and THE FLAG SUBSTITUTED FOR THE
  PULL: the discipline's own annotation became camouflage. **(a′) THE
  CORRECTION ENTRY (Code's error, corrected forward):** gut #23's
  smooth-latent fence is WRONG AS BANKED. Corrected fence: smoothness
  is MEASURED SMOOTH within kind in fst space (convex combinations
  decode cleanly); imports needing CROSS-KIND or OTHER-SPACE
  smoothness owe their own probes. The within-kind door is OPEN — the
  false memory had welded shut a door the measurement left ajar (the
  parked VAE/sampling conversations re-price accordingly if they
  return). RULE UPGRADE, no softer version: **A FLAGGED-BUT-UNPULLED
  CITE MAY NOT ENTER A REGISTRATION — VERIFY OR OMIT, NO THIRD
  OPTION** ('flagged' was functioning as a third option; six specimens
  live in third options). **(a) THE FLAT-READS-OF-CURVED-MASS ROW**
  joins the jurisdiction table as a standing failure mode, TWO paid
  cites (the gray-zone read: distance assumed ball-shaped membership,
  axes decoupled; the twenty-seven kill: rank-1 register-grain
  interference, mass at circuit grain) — with the inverted third as
  the row's own cautionary FOOTNOTE: curved reads of flat mass are the
  same disease — GEOMETRY ASSERTED INSTEAD OF PULLED. Cairo's
  beating-gloss survives verification and banks with the row:
  interference is GENERATIVE (beats make new frequencies) — covert
  rehearsal creating capability no register-grain sum can see. **(b)
  THE NONLINEAR EVOKED-VALUE RE-READ (registered, watts-minor, THE
  FIRE IS BRYCE'S — joins the sampled-retry probe in the queue):**
  the explicitation probe's caveat is banked verbatim ('not
  probe-readable,' not 'not present'; linear probes 0.00 vs dialect
  states 1.000) — one two-layer probe on the same banked states, bar
  pinned at the linear 0.00, kill-only: stays dead -> the negative
  upgrades to 'dead at the geometry we can afford'; lives -> the
  trunk holds dozen-ness on a curve and the deeper-prefix
  conversation gains a measurement (extraction economics still rule;
  no organ resurrects). **(c) PROSE:** the mirror line — a forty-year
  conjecture that held in flat regimes and was assumed general is the
  REGIME LAW at civilizational scale; the specimen outranks the
  consensus. The projection sentence parks beside the shadow essay's
  charter: when a shadow-read returns 'no structure,' the honest
  claim is 'no structure IN THIS PROJECTION' — Procrustes-first is
  the family's only known antidote. CLOSING SYMMETRY, banked: Cairo's
  method is 'the roadblock is the counterexample's address' — and the
  story's value to this house was performing her move ON us: the
  roadblock was in our own transcript, the counterexample was a grep
  away. Twenty-eight converts with the discipline sharper than it
  entered: verify or omit, no third option.
- **GUT #28(b) RESOLVED AT ZERO WATTS + CORRECTION (a'') — THE
  SEVENTH SIGHTING, CAUGHT AT FIRE TIME (2026-07-16):** preparing the
  nonlinear re-read, Code read the probe's CODE before burning watts:
  **train_probe was ALWAYS a two-layer GELU MLP** (2048 -> 512 ->
  N_DIG x 10; survivor_depth_probe.py:88) — the explicitation probe's
  banked 0.00 was NEVER a linear read. The error's origin is the
  ORIGINAL 2026-07-10 charter text ('what died is LINEAR
  decodability'; 'the same linear map does not transfer') — the
  ledger mis-described its own instrument at birth, six days before
  the taxonomy existed, and gut #28's registration repeated it
  ('linear probes 0.00') THE SAME HOUR verify-or-omit was minted —
  the cite was verified against the LEDGER, which was itself wrong
  about the CODE. RULE REFINEMENT (the seventh's lesson): **two
  authorities, matched jurisdictions — the ledger is authority on
  what was REGISTERED and VERDICTED; the CODE is authority on what
  was RUN.** Instrument-describing claims verify against the
  instrument. CONSEQUENCE: (b) is MOOT AS REGISTERED — its promised
  upgrade ('dead at the geometry we can afford') is what the bank
  already holds: evoked values are dead at a 512-hidden GELU probe,
  shallow and deep, while the same probe family reads dialect givens
  at 1.000. The negative was always the strong form, mislabeled.
  Probe-capacity escalation beyond this is a known trap (the May-era
  deep-probe memorization specimen) and extraction economics rule
  regardless — the re-read fires ZERO watts and the awaiting-watts
  queue drops to one. The corrected caveat, final width: 'not
  probe-readable AT THE GEOMETRY WE CAN AFFORD' — measured, banked,
  and cheaper than the GPU run that would have re-bought it.
- **GUT #26(a) VERDICT (2026-07-16): THE VOICE TAKES THE SHIFT — width
  where depth died, by a mile.** The sampled-retry probe
  (scripts/sampled_retry_probe.py, transient unit, gen-14 gate, the
  320-item vote-abstain fixture, K=8, one round, gold grading never
  gating): deterministic T=0 straight-parse control **70/320
  (21.9%)**; sampled deployable (solver-consistent plurality, fully
  gold-free) **T=0.3: 102 -> T=0.7: 124 -> T=1.0: 136/320 (42.5%)**;
  oracle-any at T=1.0 **151/320 (47.2%)**. Delta +66 on the pinned
  bar of +16 — the deterministic parser's re-emission was leaving
  HALF the recoverable answers on the table, and the escape mechanism
  prints in the abstain column: no-consistent-sample FALLS as T rises
  (158 -> 108 -> 102) — the deterministic content is exactly what is
  broken; heat escapes it (the repair-generation wall confirmed
  GENERATIVELY). MONOTONE IN T with no peak visible at the grid's
  edge; T>1.0 cells stay unrun (unpinned — a follow-up registration,
  not a free extension). HONEST DEVIATIONS + FOLLOW-UPS, stated
  before any deployment claim: (1) control was the T=0 STRAIGHT PARSE
  (pin iii named the deterministic specialist — its per-item recovery
  on this fixture is unbanked; the NACK-incumbent read is owed before
  the repair lane switches voices); (2) the 5-view sub-majority
  plurality reads 193/320 against gold on these items but is NOT
  gold-free-actionable as banked — the vote-vs-sampling composition
  (permutation-views x temperature-samples, one lattice) is the
  natural next instrument and touches the certification tier NOWHERE
  (this entire read lives in the answer channel; the cert chain is
  untouched). (3) THE WRITER'S NULL MODEL IS NOW PRICED: any learned
  replacement-generator must beat T=1.0 K=8 noise-plus-the-gate
  (+66) to earn parameters — the chair's first shift is held by the
  cheapest voice in the universe, exactly as the bar was written.
  Twenty-six closes fully converted: the dial was on the back of the
  epigraph's word, and turning it nearly doubled recovery on the
  hardest population in the house.
- **TAXONOMY ADDENDUM (2026-07-16, relay countersign on the seventh):**
  specimens one through six were failures of RECALL across seams; the
  seventh is a failure of INSCRIPTION — the record preserved, with
  perfect fidelity, a label that was wrong at birth; no grep could
  catch it because the grep returns the mislabel faithfully. The
  two-authorities rule completes the epistemics: it is the paper's own
  use-matches-source citation standard turned inward on our
  transcripts. And the day's symmetry, banked: twenty-six was the
  campaign's first watts spent on an instinct in eight conversions —
  and the probe that spent them was assembled entirely from prior
  audits' products (twenty-one's fixture, the standing lattice's
  disposal, twenty-four's wall as its bars): the compounding thesis
  cashing its first GPU check, and the check cleared at 4x its bar.
- **GUT #26 FOLLOW-UP 1 (2026-07-17): THE NACK-INCUMBENT READ — THE
  SHIFT IS SHARED, AND THE UNION BREAKS 50%.** Head-to-head on the 320
  (scripts/nack_incumbent_read.py, per-item outcomes BANKED per the
  law; one grading frame, disposal gold-free): **INCUMBENT (composed
  stack, ARM=field_only = fully deployable): 151/320 (47.2%)** —
  stage0 straight-parse 71, withhold-2 +60, gen-13 specialist +20
  (one-generation waiver worn). **CHALLENGER (sampled T=1.0): 136/320
  (42.5%). UNION: 175/320 (54.7%)** — incumbent-only 39, sampled-only
  24, overlap 112. THE PRECISE SEAT ASSIGNMENT yesterday's verdict
  owed: the voice does not TAKE the shift — the stack beats it
  head-to-head by 15 — **it JOINS it**: sampling adds +24 on the
  stack's own survivors (+7.5pt the incumbent cannot reach), and the
  two mechanisms are measurably COMPLEMENTARY (the specialist repairs
  what flags localize; the sampler escapes what determinism re-emits).
  THREE TEXTURE FINDINGS: (1) **the sampler outperforms the trained
  specialist ON THE TAIL** — specialist +20/189 survivors (10.6%) vs
  sampler +24/169 incumbent-unrecovered (14.2%): noise-plus-the-gate
  beats trained repair precisely where flags stop helping (the
  repair-generation wall's population); (2) **the span channel buys
  +3 only** (ceiling arm 154 vs 151) — field flags carry nearly the
  whole specialist, the deployable arm is nearly the ceiling; (3) the
  ±1 seam reappears (stage0 71 vs yesterday's banked-vote control 70,
  same definition, different solve path) — now JOINABLE per-item as
  the law intended. WRITER'S NULL MODEL, RE-PRICED: a learned
  replacement-generator must now beat the UNION lane (175) — its
  marginal customer is the 145-item residue where BOTH voices fail.
  **FOLLOW-UP 2 REGISTERED + FIRED (pins pinned here): THE
  VOTE-x-SAMPLE LATTICE** — permutation views (the standing seeds,
  40000+10i+k, view-matched to the banked lattice) x temperature
  samples (T=1.0, K=4 per view, 20 candidates/item) on the full 320;
  disposal unchanged (solver-consistent plurality, gold grades never
  gates); READS: (i) does view-diversity add a third recovery axis
  beyond the union's 175? (ii) per-view-deterministic vs
  per-view-sampled decomposition. Cost: 1,280 trunk forwards + ~6,400
  solves — minor watts under the standing word.
- **GUT #26 FOLLOW-UP 2 VERDICT (2026-07-17): THE VOTE-x-SAMPLE
  LATTICE — VIEW-DIVERSITY IS THE THIRD AXIS, AND IT DWARFS THE FIRST
  TWO.** Five views x four samples at T=1.0, solver-consistent
  plurality over <=20 candidates, standing view seeds, disposal
  gold-free (scripts/vote_sample_lattice.py, per-item banked):
  **lattice deployable 229/320 (71.6%)** vs sample-only 136 vs
  incumbent 151 vs their union 175. **TRIPLE UNION: 255/320 (79.7%)
  — the lattice alone adds 80 items neither prior voice reached.**
  THE MECHANISM, stated plainly: the five permutation views were
  always the house's power source — but on the abstain population
  their DETERMINISTIC parses disagree by definition; sampling inside
  each view plus the consistency filter lets the views agree on
  content their T=0 selves could not emit. Width x diagram-diversity
  = the two invariance axes COMPOSED — the certification channel's
  own geometry, turned from a gate into a generator, on the exact
  population the gate refused. THE OWED DECOMPOSITION before any
  deployment claim (registered, not fired): the 91 non-recovered
  split into emitted-wrong vs abstained — the repair lane's PRECISION
  is the deployment bar (the answer channel currently runs 0.9953 at
  the majority tier; a 229-right lane is only adoptable at its
  measured precision, and plurality-of-20 luck must be priced). Also
  owed: regression bars + the certification-tier non-contact
  assertion re-stated mechanically at adoption time. THE RESIDUE:
  65/320 (20.3%) resist all three voices — the writer's charter now
  inherits its THIRD floor, and the null model is a lattice, not a
  dial. The day's arithmetic: the vote-abstain population — 21.9%
  recoverable by the deterministic voice yesterday morning — reads
  79.7% recoverable by composed voices tonight, zero training, zero
  new parameters, certification untouched. The chain of custody
  taught its own repair lane to speak.
- **GUT #26 FOLLOW-UP 3 VERDICTS (2026-07-17): PRECISION, THE
  FRONTIER'S GIFT, AND THE SPLIT PORTRAIT.** One instrument
  (scripts/residue_portrait.py; lattice re-derived byte-identical,
  per-item assert on all 320). **(1) THE NAIVE LANE FAILS THE BAR,
  AS FEARED:** emitted 296 (right 229, WRONG 67), abstained 24 —
  lane precision 0.7736 vs the 0.9953 incumbent standard. Recovery
  tables seduce; the fence held. **(2) THE FRONTIER'S GIFT —
  PLURALITY-COUNT IS THE DIAL, NOT SHARE:** thresholding on absolute
  agreement mass finds the lane's high-precision core: **count>=5:
  113/115 = 0.9826; count>=8: 36/36 = 1.0000 (measured)** — while
  share-thresholds top out at 0.90 (share=1.0 on tiny candidate sets
  = 2-of-2 flukes at 0.868). The effective-K lesson in sample
  clothes: darts must be MANY and agreeing — absolute mass, never
  ratio. A TWO-TIER REPAIR LANE is now drawn and priced from banked
  data: count>=8 at measured 1.0000 (+36 answers) and count>=5 at
  0.983 (+113) on a population that today yields ZERO — adoption
  holds for regression bars + the mechanical cert-non-contact assert
  + Bryce's word (it would move the composite's published precision
  and must be re-stated, not slipped). **(3) THE PORTRAIT: SPLIT —
  and the surviving axis is the important one.** Prediction (i)
  FAILS with a jurisdiction lesson: residue withhold-recoverability
  0.688 / zero-frac 1.5% vs recovered 0.697/1.6% — INDISTINGUISHABLE,
  because generated bigtest is redundant EVERYWHERE (the meter's
  inversion foretold it: minimality is a BOOKS property; the
  prediction imported a real-prose axis onto generated text —
  cross-population prediction, honestly dead). Prediction (ii) HOLDS
  strongly: **residue det-vote H 0.435 vs recovered 0.631 — the
  residue is COLD: stable cross-view misreadings**, every view
  quietly agreeing on wrong readings that neither width nor
  diagram-diversity can dislodge. THE JOIN WITH THE COOLING GAUGE,
  named: the cold-error species the standing bet fears EXISTS and
  lives in the abstain channel's residue — 65 specimens, enumerated,
  per-item banked. The writer's customer portrait, corrected to what
  survived: the STABLE-MISREADING survivor — re-read, not re-emit;
  the 'only the text proposes' charter stands on twenty-four's
  evidence and now on temperature, with the minimality clause
  confined to the books lane where it was measured. Board: the
  two-tier lane awaits the word; the residue awaits the writer;
  book 4 awaits pages.
- **GUT #26 FOLLOW-UP 4 (2026-07-17): THE ADOPTION READ — THE DECISION
  TABLE PRINTS, WITH A SURPRISE IN THE INCUMBENT'S ROW.** Pinned bar:
  composite precision >= 1179/1180 = 0.99915 (the current answered
  channels, banked). **(A) THE INCUMBENT'S OWN PRECISION, decomposed
  for the first time: 208 emissions, 147 right, 61 WRONG — 0.7067**
  — the specialist stack, read at deployment semantics
  (emit-when-solvable), is LESS precise than the naive lattice
  (0.7736); its composite lands 0.95533, a catastrophic bar fail.
  (Follow-up 1's 151 was recovery-max; emission semantics differ by
  stage-order — a stage-0 wrong emission blocks a later recovery.)
  The 0.833-precision composite the paper reports for the answer rung
  is the historical measurement of exactly this species of lane.
  **(B) THE DECISION TABLE (banked, .cache/adoption_read.json):**
  count>=8: +36 emit, 36 right -> 1215/1500 (81.0%) at 0.99918 PASS;
  count>=10: +10 -> 0.99916 PASS; count>=5: +115/113 -> 86.1% at
  0.99768 fail-by-0.0015; incumbent: fail; lattice>=5-then-incumbent
  (the benchmark-max lane): **1369/1500 = 91.3% at 0.9723** fail.
  RECOMMENDATION GIVEN (Code): adopt **count>=8 now** — strictly
  bar-passing, +36 answers at zero published-precision cost — with
  TWO honesty clauses: (i) zero-numerator discipline: 36/36 reads
  'error bounded below ~2.8%', never 'perfect'; (ii) the bar margin
  is ONE-WRONG-THIN (0.99918 vs 0.99915) — the tier adopts WITH A
  WATCH: its precision column joins the standing battery beside the
  panel-dissent and temperature-band columns. The count>=5 and
  benchmark-max doors stay open at their printed prices — the
  91.3%-at-0.9723 lane is a VENUE POLICY question (benchmark scoring
  has no wrong-answer penalty), and that word is Bryce's with the
  price sheet in hand. THE WEEK'S REPAIR-LANE LEDGER, closed: one
  deterministic voice (21.9%) -> a priced, tiered, bar-disciplined
  lattice with a certify-analog (+36 at preserved precision) and a
  policy frontier to 91.3% — zero training, zero parameters, the
  certification tier untouched at every step, every number pinned
  before it printed.
- **THE ADOPTION (2026-07-17, Bryce's word via 'fire when ready' on the
  recommendation + relay countersign): THE COUNT>=8 TIER ENTERS THE
  REPAIR LANE — machine first, prose second, per the law.** THE
  MACHINE: scripts/vote_sample_lattice.py now carries EMIT_MIN
  (default 8 — the adopted certify-analog; EMIT_MIN=0 reproduces the
  research read, and the banked research artifacts were produced at
  0). Cert non-contact is structural and commented at the knob: the
  lane consumes only vote-abstain items. THE PROSE: the repair lane's
  spec is now TIERED — vote-abstain -> vote-x-sample lattice ->
  emit iff plurality-count >= 8 (measured 36/36; the sentence wears
  its width: ERROR BOUNDED BELOW ~2.8%, a bound thirty times looser
  than the 912-certificate tier's — the zero-numerator law scales
  with n); else abstain. Manifest citizenship rides the next
  promotion's atomic write (the manifest law). **THE SENTINEL ROW,
  named and chartered (relay):** the standing battery now carries
  three columns that each watch a different feared species —
  PANEL-DISSENT (lineage disagreement), TEMPERATURE-BAND (basin
  cooling), and now REPAIR-TIER PRECISION (the one-wrong-thin
  margin's watch) — instruments-police-successors made standing
  furniture. **THE DOORS, dispositioned (relay counsel, adopted):**
  count>=5 (+77 at −0.0015 published precision) STAYS SHUT for
  paper-1's regime — the frozen table does not amend; it re-prices
  honestly in Paper II's regime if that campaign opens it. The
  benchmark-max lane (91.3% at 0.9723) parks as a VENUE INSTRUMENT —
  legal only when explicitly labeled as the recall-max policy, never
  the deployed default; the label is the honesty story. THE WEEK'S
  CLOSING ARITHMETIC: the answered channel rises 1179 -> 1215 of
  1,500 (78.6% -> 81.0%) at composite precision 0.99918 >= the bar
  0.99915 — the first capability adoption in campaign history with
  ZERO training, ZERO new parameters, and the certification tier
  untouched, purchased entirely by reading the machinery we already
  owned at the temperature it was always capable of.
- **BOOK 4 CHARTER (2026-07-17, Bryce's word: 'print the pages').** The
  first floor-up book — the recursion's third rung, walking into the
  best-instrumented staging in campaign history. PINS, before any page:
  **(1) SOURCE**: the L4/L5 harvest strata (633 available after
  excluding the census fixture and all books-1-3 sources) — the
  harder-strata/competition-register arm the charter queue named; the
  annotation rulebook's filters stand (length<300 chars, no asy,
  values<=300 — value-cap failures route to value-range certificates,
  never forced). **(2) LANES**: the standing L1/L2/L3 classification
  under the GEN-14 gate (book4_lanes.py, N_CAND=200, 5-view votes,
  census fixture untouched as fixture). **(3) THE MACRO PROTOCOL —
  the book's reason to exist**: OP_APPLY crowns (grammar mg1)
  annotated AT MACRO FLOOR — the macro dialect writes the crown as
  ONE SENTENCE (the compression the wild actually wears); every macro
  row banks WITH its prime twin (expand_graph, deterministic), and
  THE GATE RUNS ON THE PRIME TWIN — 5-view vote >=3 + answer key, the
  standing trust story byte-unchanged; the pair is ONE KNOT (the
  floor-identity protocol, sealed in code before this book needed
  it); the pairs ARE the fidelity-axis probe's arm-(iii) corpus by
  construction. **(4) SIZE**: tranche-1 <= 25 pages from the lanes;
  the book sized by lane yields — the census-slope duty is RETIRED
  (third point printed SATURATION); book 4's registered purposes are
  the macro floor, the register widening, and the 13-gap coverage.
  **(5) DOSE**: declared at the training registration, not here —
  the training run (head extension + gut #23's three arms + 21(c)'s
  concentration A/B + 24(a)'s per-floor redundancy read) AWAITS THE
  WORD. **PREDICTIONS PINNED**: (P1) L4/5 lane yields skew harder to
  L3 surgery than book 3's (~82% baseline); (P2) OP-APPLY crowns
  appear in bankable harder-strata strangers at >= the books' 4.9%
  item rate; (P3) THE WALL TEST stands armed — the first stranger
  whose primitive form exceeds the 24-factor bank but whose macro
  form fits is the factor-count wall falling, the recursion's first
  measured dividend (watch, not bar). The dancer reads one floor up
  starting today.
- **BOOK 4, TRANCHE 1 BANKED (2026-07-17): THE FIRST FLOOR-UP PAGES.**
  15 hand dialects through the gen-14 gate (5-view vote >= 3 + key):
  **12 pages banked (14 rows) — including THE FIRST TWO MACRO-ANNOTATED
  STRANGERS**: [3] the quadratic-vertex sub-crown (40a − 5b, macro
  dialect one sentence, banked 5/5 unanimous at 68) and [20] the
  composition add-crown (3a + 2b, 5/5 at 17) — each banked at BOTH
  floors, one knot per pair (canon identity asserted live:
  f5a9979857c6, 9fb69e9dbdaa), expansion solving to the official
  answer before the gate ever saw the twin. The floor-identity
  protocol and the admission exam's machinery ran IN PRODUCTION for
  the first time, two days after being built ahead of need. TWELVE
  registry certificates filed (rate-noninteger -> the [45] frame
  family; unit-fraction -> the chained-fdiv docket; plus lookup-chain,
  diophantine-opt, radical x2, symbolic-identity, floor-abs,
  area-perimeter, functional x2, consecutive-sum). THREE MISSES, each
  diagnostic, to the retry bench: **[26] is a LIVE AUTOPSY SPECIMEN —
  5/5 UNANIMOUS-WRONG at 15** on 'When a is divided by 27' (108/7=15:
  the head dropped a digit of the fdiv parameter — the chained-fdiv
  autopsy's staged suspect, THE DERIVED-VALUE DIGIT PATH, caught wild
  on a book page; the specimen files to the docketed autopsy); [10]
  solved on exactly one view (correctly, 205 — an 11-var 9-factor
  graph at the length frontier); [57] split 2-2 with the correct 51
  present (digit wobble; v2-retry candidate). PREDICTIONS: **P1 FAILS
  IN THE GOOD DIRECTION** — L3 surgery 71% vs book-3's ~82% baseline:
  the harder strata read EASIER than book 3's picked-over residue
  (the reading campaign's register gain paying on competition text);
  **P2 HOLDS** — 2 crowns in 12 banked (16.7% >= 4.9%, small-n
  stated); **P3 the wall test watches on** (no >24-primitive stranger
  this tranche). Artifacts: .cache/book4_prose_pairs.jsonl (14 rows,
  floor-tagged, grammar-stamped), book4_organ_registry_t1.json,
  book4_lanes.json (L1 5 / L2 53 / L3 142 of 200). The dancer read
  one floor up today, and the gate never noticed the difference —
  which was the entire design.
