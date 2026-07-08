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
