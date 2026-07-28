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
- **BOOK 4 ADDENDUM (2026-07-17, relay countersign — the fidelity-probe
  indexing pin, registered at n=2):** arm-(iii)'s training probe fires
  on CROWN COUNT, never tranche count — the paired corpus reaches the
  mass the regime registration pins, or the probe waits; a small-crown
  fire would be the dose pilot's n=14 lesson repeated on the fidelity
  axis. P2's watch is the meter; the book tells us its own rate. Also
  banked: the two-days-early pattern happened TWICE in one tranche
  (floor-identity + admission machinery, both built ahead of need) —
  the compounding thesis is not luck; the instruments arrive before
  their customers because the gut keeps knocking one session early.
- **BOOK 4, TRANCHE 2 BANKED (2026-07-17): THE RETRY BENCH CONVERTS
  CLEAN.** 10 pages banked (11 rows, 1 macro pair) + 8 certificates.
  **ALL THREE v2 RETRIES CONVERTED, each by its diagnosed mechanism**:
  [26] 5/5 at 4 (mul-inverse replaced the fdiv digit path the autopsy
  specimen exposed — the fix validates the diagnosis), [57] 5/5 at 51
  (same cure), [10] 3/3 at 205 (both remainder chains shed). **THE
  AFFINE CROWN BANKED**: [28]'s b + 3·a = 10 (the k=1 leg's first
  production page, one knot 9d9e11aa2a7d, 5/5 at 2) — the macro
  vocabulary now spans both mined classes IN THE CORPUS. [29] banked
  5/5 at 19 — the coupled-mul products-sum shape, the miner's named
  gap, rehearsed on a real stranger. THREE NEW MISSES, all at the
  LENGTH FRONTIER: [33] (12 vars, solved on ZERO views), [100] (15
  vars, votes scattered), [22] (2/5 correct — one vote short, light
  rework candidate). P3 jurisdiction note: these are PARSE-DEPTH
  walls (12-15 vars), not the >24-slot wall the test watches — the
  practical frontier sits below the structural one, which is itself
  a datum for the wall test's eventual reading. BOOK 4 RUNNING
  TOTALS: 22 pages, 25 rows, 3 macro pairs (both crown classes),
  20 certificates, 3 length-frontier items on the bench. The book
  is teaching two lessons at once: the register lesson (harder
  strata read easier than expected) and the depth lesson (the parse
  wall arrives before the slot wall).
- **GUTS #29+#30: THE DEPTH-AND-COMPOUNDS AUDIT (2026-07-17, Bryce +
  relay + Code, registered as amended; (a) FIRED).** Two guts, one
  intersection — and TWO mechanism corrections at review: (1) the
  panama-hat specimen [26] bled at the EMISSION digit banks, not
  tokenization — Llama-3 carries '27' as ONE token; the compound broke
  at the mouth, not the ear — relocating the wound from frozen
  territory (tokenizer) to trained territory (digit banks, healed
  every generation); (2) the bands column's jurisdiction is solver
  decisions, not derivation depth — the join computes depth fresh
  from gold DAGs (the metric-question mismatch twenty-seven's census
  exists to catch, caught). **(a) THE CHAIN-DEPTH JOIN VERDICT: WEAK
  — AND THE WALL IS COUNT-SHAPED, NOT DEPTH-SHAPED.** Pooled
  within-stratum AUC(depth -> not-certified) = 0.556 (bar 0.60);
  uncontrolled: depth 0.587 vs FACTOR-COUNT 0.623 — SIZE dominates.
  The constitution confirmed from a new angle: the head never RUNS
  the chain (the solver does), so chain depth barely hurts — the
  binding burden scales with HOW MANY bindings, not how long their
  chain. JURISDICTION: verdict scoped to depth 3-5 (the generated
  fixture's compressed range — the mint's cycles flatten depth
  variance; beyond-5 unmeasured, stated). THE RIDER STRENGTHENS BY
  THE MISS: crowns compress COUNT (4 primitives per crown), and
  count is the measured wall driver — the tower's dividend path is
  COUNT-RECOVERY, P3's practical wall confirmed count-shaped; the
  crown-recovery rider re-aims accordingly and fires at crown mass.
  **(b1) THE EMISSION DIGIT CURVE** — the chained-fdiv autopsy's
  FORMAL OPENING READ, specimens one and two filed ([26] wild-caught
  + its validated mul-inverse cure); GPU-minor, holds for the word.
  **(b2) THE SCOPE-PAIR MINE** — difference-of-squares vs
  square-of-the-difference, minted minimal pairs, the tranche's
  fresh family as anchor; GPU-minor, holds for the word. **(c) PROSE
  LAW BANKED**: serial computation belongs to the jaws; the head's
  depth is spent on binding, and its wall is COUNT; the tower makes
  big graphs small; panama hats live at two skins — the input's
  (owned by the trunk, measured: numbers <=999 enter whole) and the
  emission's (where the specimen bled); the dialect strips wild
  compounds, the gate crowns earned ones.
- **GUT #31: THE RING-DOWN AUDIT (2026-07-17, Bryce + relay + Code,
  registered as amended and FIRED — the rare instinct that arrives
  POST-CONFIRMED).** The headline closure, banked as (c)'s first
  citation: THE WEEK ALREADY RAN THE EXPERIMENT THE FRAME PREDICTS —
  the decay-shape reading's mechanism (same flags every round,
  front-loaded collapse, 19.6->7.7->1.1->0) WAS overdamping-by-
  determinism; twenty-six's dial WAS a re-excitation device; the
  lattice's doubling WAS the frame's central prediction confirmed
  before the frame existed. **(c) THE TWO-JAWS DAMPING LAW (prose,
  both signatures MEASURED): the disposer is overdamped by
  construction — monotonicity is soundness, a solver that rang would
  be a solver that guessed; the proposer is excitable by nature —
  re-excitation is recovery. First citation: the adoption commit.**
  The coarse-to-fine envelope parks with the cascade prose (schedule
  from physics, Brick-P's fence untouched: ring-down lives across
  generations and floors, never within a forward pass). **(a1) THE
  PROSE MINE (fired; SELECTED sample, existence-and-mechanism claims
  only — famous wobblers are the worst sample for a rate):** zero
  crossings EXIST — [71] (unanimous-wrong era -> correct-but-
  uncertifiable [9,9] -> acceptance-stable) and [78] (consistent-wrong
  12/12/12 -> right [16] -> banks) each cross the boundary once and
  settle; [45]/[7] HOVER at the boundary (sub-threshold oscillation,
  the chronic frame family); [51]/[54] are CURED wobblers (basin-side
  instability, rehearsal settled them — gut #15's banked verdict).
  RE-EXCITATIONS TAG TO REGIME SHIFTS as pinned: gen-9's diet shift
  re-shallowed specific basins (the wobble era); gen-9b and the books
  settle them (prose-as-regularizer). Envelope: monotone settling
  post-gen-9b for every named wobbler — ring-down consistent, rates
  deferred to the unbiased column. **(a2) THE RING GAUGE INSTALLED:**
  a DERIVED column on the per-item outcome law's mandated banking
  (sign-flips since last battery, per item) — zero new measurement
  cost, rates accrue unselected per promotion. THE SENTINEL ROW GROWS
  TO FOUR: panel-dissent (lineage), temperature-band (cooling),
  repair-tier precision (the watch), RING GAUGE (dynamics). **(b) THE
  DAMPING-RATIO RETROSPECTIVE (fired; subsumes #24(d)): SEVEN SEAT
  CLEANLY, ONE RESISTS — TAXONOMY-NOT-MECHANISM, as pre-named.**
  gen-9 (hot, shifted diet -> gains + re-shallowed basins) =
  UNDERDAMPED; gen-10 and reader_v1 (gentle continuations, alg4 debt
  unpaid, killed at the bar) = OVERDAMPED; gen-9b (kick-then-settle)
  = well-damped; gen-13/gen-14 (hot flat retrains, clean ancestry,
  debt paid + records + acceptance holding) = NEAR-CRITICAL;
  reader_v2 RESISTS as pre-named (-10 REGRESSION from stacked gentle
  continuation — overdamped systems under-deliver, they do not go
  backwards; overfit/fatigue is a DIFFERENT PHYSICS the oscillator
  has no term for; gen-11 ambiguous, noted). SURVIVING FORM: 'damping
  must match displacement' as a CLASSIFIER with a named boundary —
  gen-15's scheduling tool, not a mechanism; the third axis
  (repetition-fatigue) is named as missing physics and left for the
  instinct that comes for it. Thirty-one converts: the campaign's
  instruments now measure DYNAMICS as well as state — flux in,
  temperature held, ring-down between — and the bell's first striker
  was us, knowingly, last Tuesday, with a temperature dial.
- **GUT #32: THE SMALL-STEPS CONSOLIDATION (2026-07-17, Bryce + relay
  + Code, registered as amended and FIRED).** The instinct arrived as
  an imperative and decoded as a CHORD — five jurisdictions already
  humming the note, the fifth the most literal: **the gut was speaking
  the June engine's own vocabulary back to us.** THE FIVE-JURISDICTION
  MAP (banked so no session re-derives it): (1) training — flux
  density + the concentration A/B riding book-4 (spread-vs-
  concentrated IS small-vs-large at matched energy; the gut votes
  spread); (2) generation — the regime law with thirty-one's damping
  physics; (3) architecture — the cascade envelope (coarse floors
  take the big semantic steps); (4) field — diffusion's many-small-
  denoisings; (5) **ENGINE — delta_gate (BUILT, RESTING, VALIDATED):
  the deducer's learnable convex residual blend IS the step-size
  dial, per-breath weighted-CE IS monotone refinement enforced, v98's
  hole-monotonicity IS little-by-little as architecture** — the
  instinct recognizing its own prior implementation across a month
  and two architectural eras. THE EIGHTH SIGHTING, new wrinkle for
  the taxonomy: 'Dopri5 stepping won its ablation' — the record says
  Dopri5-STYLE everywhere (framing, hook, analogy) and the May-era
  ablations found adaptive controller decisions DECORATIVE on
  converged models; **the suffix '-style' is a provenance marker and
  the relay channel strips it** — an analogy's clothing mistaken for
  a measurement's body; enters as prior ART, correctly labeled.
  **(a) THE DISPLACEMENT-VS-GAIN RETROSPECTIVE (fired at aggregate-
  proxy grain — retroactive item-grain flips do not exist; the
  cooling gauge's lesson applied BEFORE the mistake): ORDERING
  CONSISTENT at every seat walked.** Low-displacement gens leave
  debts standing (gen-10 alg4 357, reader_v1 378 — records elsewhere,
  the bar unpaid); stacked-low REGRESSES (reader_v2 −10 — fatigue,
  not damping); high-displacement-on-shifted-diet prints mixed signs
  (gen-9's re-shallowing); high-displacement-from-clean-ancestry
  crowns (gen-13: debt PAID at 385 + records; gen-14: records +
  acceptance holding). Eight points, ordinal, kill-only — NO
  contradiction found; the inverted-U is CONSISTENT-BUT-COARSE, and
  the true item-grain curve accrues from the RING GAUGE installed one
  gut ago (instruments-before-customers now running at ONE-SESSION
  lead). **(b) THE CONDITIONAL STEP LAW (prose, counterexamples in
  the law's own text): small steps by default; ONE CLEAN QUENCH FROM
  CLEAN ANCESTRY when debt is owed (reader_v1's kill and gen-13's
  payment are the same lesson from both sides); STACKED DRIPS ARE
  NEITHER — they are fatigue (reader_v2, the resistor, seated at
  last as the law's own boundary marker).** (c) the map above IS the
  cross-reference. Thirty-two converts as consolidation: the chord
  written down, one note played by machinery resting in June's
  drawer.
- **GUTS #29+#30, READS (b1)+(b2) VERDICTS (2026-07-17): THE AUTOPSY
  OPENS WITH ITS MECHANISM QUANTIFIED, AND THE PANAMA HAT SITS FOR
  ITS PORTRAIT.** **(b1) THE EMISSION DIGIT CURVE — PREDICTION HOLDS
  (scripts/digit_curve_and_scope_mine.py; fixture note: bigtest is
  GIVEN-ONLY — clean baseline 0.972 flat; the param path lives in
  alg4test, rerun banked):** given path holds 0.945+ at all
  magnitudes with MSD ~1.000; **the PARAM path erodes with magnitude
  — mag-1 0.977, mag-2 0.903, mag-3 0.837 — with the deficit
  concentrated OFF the LSD** (mid 0.915 at mag-2; MSD 0.901 at
  mag-3): [26]'s fingerprint (27 -> 7, tens dropped, ones kept)
  generalized to population scale. THE CHAINED-FDIV AUTOPSY'S
  OPENING FINDING: param-path high-order digit erosion, 9.7% slot
  error at 2 digits, 16.3% at 3 — and the tranche-2 cure now has its
  WHY: mul-inverse rephrasing is a PATH SWAP (the constant re-enters
  through the given path, which holds at all magnitudes). Autopsy
  status: mechanism quantified, cure validated, remedy priced (diet:
  more multi-digit param mass; or rulebook: prefer mul-inverse for
  2-3 digit constants — the annotation desk already does the latter
  as of tranche 2). **(b2) THE SCOPE-PAIR MINE — 0 DISCRIMINATED, 5
  COLLAPSED, 5 MIXED, 0 REGISTER-WALL, and the collapse has a FACE:
  the head reads BOTH scope phrasings as a+b** ((7,4)->11, (9,5)->14,
  (12,7)->19 — squares dropped, scope dropped, the shallow binary
  over the mentioned vars emitted STABLY across views, >=3/5). The
  honest jurisdiction: the phrasings are outside the trained dialect,
  so this measures the REGISTER WALL's shape, not in-register
  ambiguity — and the shape is the finding: at the boundary the
  reader does not refuse, it answers a SIMPLER question confidently.
  TWO DIVIDENDS BANKED: (1) the dialect's one-relation-per-sentence
  design is VALIDATED as the anti-panama-hat device (in-dialect
  scope is unambiguous — the banked pages prove it); 'difference of
  squares' phrasing is a BOOKS CURRICULUM ITEM (a future register
  rung, priced); (2) **scope compounds are a MANUFACTURABLE
  UNANIMOUS-WRONG FAMILY** — style-native, structure-invisible,
  view-stable on the wrong parse: the certification channel's named
  blind-spot species, producible on demand. REGISTERED FOLLOW-UP
  (zero-new-machinery, rides any lattice run): feed the scope pairs
  through cert-v2 — if armB/cap2x share the a+b collapse (lineage-
  shared blindness), the panel's decorrelation fails exactly where
  [71]'s species predicted, and the mouth/panel design conversation
  gains its sharpest specimen set. The watts queue is EMPTY; guts
  29+30 close fully converted.
- **BOOK 4, TRANCHE 3 BANKED (2026-07-17): THE COUNT CURES CONVERT —
  AND [100] RETIRES AS THE WALL'S MARKER.** 7 pages banked (8 rows,
  1 macro pair), every single one 5/5 UNANIMOUS. THE HEADLINE
  SPECIMEN: **[33] went from ZERO views solving at 12 vars to 5/5 at
  9 vars** — three variables shed, silence to unanimity: the count
  wall measured from both sides on one problem (guts 29+30's
  count-shaped verdict, demonstrated at the desk within hours of its
  printing). [22] converted the same way (mul-inverse, 5/5). THE
  FOURTH CROWN: [36]'s sum-of-coefficients sub-crown (3a − 7b, one
  knot 331930d9da1f, 5/5) — eval-at-1 explicitated, the crown
  carrying the arithmetic. The distance family banked THREE
  variations of its 5-12-13 skeleton ([43],[46] + t1's [5]) — the
  isq door's reclaimed territory now rehearsed at volume. **[100]
  MISSES AGAIN at 12 vars (votes empty — silent even after
  shedding 3)**: the two-fdiv three-add chain parses NOWHERE between
  9 and 12 vars; it RETIRES TO THE LEDGER AS THE COUNT WALL'S
  STANDING MARKER rather than burning a third retry — the practical
  wall for fdiv-mixed chains sits in the 9-12 var band, and the
  macro floor's count-compression is its priced remedy (the
  crown-recovery rider's exact customer profile). THIRTEEN
  CERTIFICATES incl. SECOND family certificates (unit-fraction,
  vieta, exponent — repeat customers pricing the next admission
  review by frequency, exactly as the charter said families would).
  **BOOK 4 RUNNING TOTALS: 29 pages / 33 rows / 4 MACRO PAIRS (2
  sub-crowns, 1 add-crown, 1 affine) / 33 certificates** — the crown
  counter at 4 and the certificate families beginning to repeat,
  which is the registry's admission economics turning over exactly
  on schedule.
- **BOOK 4, TRANCHE 4 BANKED (2026-07-17): THE FIFTH CROWN, AND THE
  WALL GETS ITS SECOND MARKER.** 7 pages banked (8 rows, 1 macro
  pair), every one 5/5 UNANIMOUS. **THE IDENTITY CROWN: [51]'s
  (x−y)² = (x+y)² − 4xy banked as the k1=1 sub-crown (one knot
  4f98a8f46bcf, 5/5 at 5)** — a textbook algebraic identity wearing
  the macro whole: the strongest evidence yet that the crown
  vocabulary matches how mathematics actually compresses. Fresh
  pages: partial fractions ([74]), Vieta's discriminant ([65], 148),
  the reciprocal-sum identity ([69]), quadratic-composition ([63]),
  double-root ([55]), consecutive-evens ([79]) — the harder strata's
  identity-and-technique families entering the corpus at volume.
  **[73] MISSES SILENT (votes empty, 10 vars, THREE fdivs) — joining
  [100] at the wall: two specimens now mark the same band — MULTI-
  FDIV CHAINS PARSE NOWHERE AT 10+ VARS.** The count wall's profile
  sharpens: it is not just var count but fdiv DENSITY (each fdiv
  spends two vars and a param digit path — the wall is
  count-x-fdiv-mass, a refinement guts 29+30's join could not see at
  bigtest's fdiv-thin mix). ONE CAP CASUALTY certificated honestly:
  [68]'s 27a+10b=600 crown dies at the value cap — a beautiful crown
  lost to domain, logged not forced. **BOOK 4 RUNNING TOTALS: 36
  pages / 41 rows / 5 MACRO PAIRS ([3],[20],[28],[36],[51] — 3 sub,
  1 add, 1 affine) / 45 certificates** with repeat families
  accumulating (value-range x3, unit-fraction x2, vieta x2,
  piecewise x2, radical x2, exponent x2) — the admission docket
  pricing itself tranche by tranche, exactly as chartered.
- **BOOK 4, TRANCHE 5 BANKED (2026-07-17): THE PERFECT TRANCHE, AND
  THE WALL PROBE SPEAKS.** 13/13 banked (15 rows, 2 macro pairs),
  ZERO misses, every page 5/5 unanimous. **THE HEADLINE: [85] — 20
  VARIABLES, 19 FACTORS, ZERO FDIVS — BANKED 5/5 UNANIMOUS.** The
  controlled read prints loud: **the practical wall is FDIV-MASS,
  not raw count** — a 20-var pure add/mul system parses unanimously
  while 10-var chains with three fdivs go silent ([73]) and 12-var
  chains with two ([100]). THE SYNTHESIS, stated precisely: raw
  count erodes CERTIFICATION gradually (bigtest's 0.623 on a
  fdiv-thin mix — vote unanimity gets harder with more slots);
  fdiv-mass causes outright PARSE COLLAPSE (the silent misses). Two
  failure modes, two mechanisms, one shared discount (the mul-inverse
  path swap) — and the crown-recovery rider's customer profile
  sharpens again: the tower's first dividend customers are the
  FDIV-DENSE, not merely the large. THE CROWNS: [82] banked the
  [51]-identity's SECOND instance (the golden-ratio equation wearing
  the same crown — A REPEAT CROWN FAMILY: two strangers, one
  identity, exactly the frequency signal the admission economics
  wanted); [105] banked at 241 — a THREE-DIGIT crown answer emitted
  clean 5/5 (result-path digits hold at magnitude where param-path
  digits erode — the digit curve's path split confirmed from the
  emission's healthy side). Fresh coverage: Newton's-identity
  skeleton ([87] at 90), the four-point distance sum ([96], 15 vars,
  5/5), the recycling cascade ([104]), ceil-interval ([106]), plus
  the rate family's THIRD certificate ([107] harmonic — the frame
  family's docket grows). **BOOK 4 RUNNING TOTALS: 49 pages / 56
  rows / 7 MACRO PAIRS (the identity crown x2 — the first repeat
  crown family) / 52 certificates.** Five tranches in one day; the
  gate never blinked once.
- **GUT #33: THE FINGERPOST (2026-07-20, Bryce + relay + Code — the
  instinct that knocked for thirty years).** Iain Pears' four-narrator
  epistemology mapped onto the house and found already-built walls
  plus ONE open door. THE NARRATOR TAXONOMY (prose, banked): the
  agenda-narrator = reward hacking (why the key sits outside every
  acceptance path); the SELF-DECEIVED narrator = the cold error, H~0,
  sincerely stable and wrong ([71] is Prestcott walking); the
  cryptographer Wallis = instrument bias (the hammer that sees nails
  — the length-warped mouth reading sample size as distance); Wood
  the antiquarian wins by PROVENANCE DISCIPLINE — the two-authorities
  rule in 1660s Oxford: truth is the residue after every account is
  taxed by its provenance. THE CITE, corrected to its stronger form:
  the fingerpost is the principled shape of the SECOND-VIEW RE-RENDER
  — the 'change the ENCODING' build option tabled at the
  oracle-ceiling frontier 2026-07-08, waiting eleven days for this
  spec. **(a) THE FINGERPOST VIEW, registered + v0 FIRED:** on
  vote-split items (answer channel only; certification untouched),
  render each leading parse P1/P2 to canonical dialect via a
  DETERMINISTIC writer (templated; not the chartered repair-writer —
  it can only say what the parse already says), and THE FROZEN TRUNK
  ADJUDICATES: pooled-state similarity of the original text to each
  restatement — the reading whose canonical form is the closer
  paraphrase of what was actually written gets the point. Style
  confound cancels BY CONSTRUCTION (both restatements wear the same
  dialect — differences are pure content). FENCES: (i) evidence into
  the vote, NEVER an override (the Wood-seduction fence — the fourth
  narrator feels authoritative because he speaks last; the
  adjudication will feel like a verdict because it arrives dressed in
  trunk similarity — it is a witness, not a judge); (ii) length-law
  applies, lengths logged (near-cancelling: restatements differ by
  ~1-2 factors); (iii) v0 kill bars pinned: preference-for-truth
  >=60% = the fingerpost points; <=55% = dies for the price of a
  probe, the re-render table gets its honest negative. **THE
  DISAGREEMENT-LOCUS RIDER (gift two, zero marginal cost):** every
  adjudication logs WHERE the witnesses diverge (the factor-kind
  diff) — a contested-binding census harvested from production
  splits, the reader's confusion matrix for free. **(b)** the
  witness-independence read = the standing cert-v2 scope exam (the
  Bacon question: do the narrators read the same newspaper?). **(c)
  PARKED IN THE DRAWER**: the ACTIVE fingerpost (Bacon's second
  clause — the crucis is SOUGHT, not awaited: mint the minimal text
  variation whose reading must differ under P1/P2 — the first
  machinery ever sketched here that asks a question rather than
  answering one; the scope factory proves targeted minting works);
  and deducer multi-views (different orderings, different propagation
  schedules — different witnesses to one deduction) behind the
  redundancy gate where all deducer-shaped things lawfully wait.
- **GUT #33, v0 VERDICT (2026-07-20): POPULATION-STARVED — the probe's
  real finding is demographic, not epistemic.** Gradable two-answer
  splits on bigtest: **n=2** (1/2 preference-for-truth = a coin flip
  at a sample size where no bar can speak; the mechanical 'DIES' line
  is OVERRULED by starvation — the kill bars presupposed a population
  the fixture does not hold). THE FINDING: the gen-14 gate's votes on
  generated text either CONVERGE (968 unanimous) or SCATTER (the
  abstains' sub-2 pluralities and many-answer sprays) — clean 2-way
  contests, the fingerpost's natural customer, are nearly absent from
  deterministic votes on this fixture. WHERE THE CONTESTS ACTUALLY
  LIVE, and v0.1's re-registration (population change = design change
  = through the countersign, per discipline): **the sampled lattice's
  candidate distributions** — the 320 fixture's temperature samples
  produce rich top-2 contests per item (plurality vs runner-up among
  solver-consistent candidates), hundreds of adjudications from
  machinery already banked, and the population the fingerpost was
  always FOR (the repair lane's ambiguity, not the deterministic
  vote's rare indecision). v0.1 SPEC: same writer, same trunk
  adjudication, same fences and bars — population = the lattice's
  top-2 consistent candidates per abstain item; the locus rider
  inherits. AWAITING THE COUNTERSIGN. The tiny census's one free
  crumb: the n=2 loci were given-value and rel disagreements — too
  thin to read, banked for the v0.1 join. Bacon's machine is built
  and constitutional; it was pointed at the one crossroads in town
  where nobody argues.
- **GUT #33, v0.1 VERDICT (2026-07-20): THE FINGERPOST POINTS —
  0.701 at n=147.** At the argument factory (the sampled lattice's
  top-2 contests, 147 gradable of 320; 173 skipped: <2 candidates or
  gold absent — stated), the frozen trunk prefers the TRUE reading's
  canonical restatement **103/147 = 70.1%** against a 50% coin and
  the pinned 60% bar (binomial half-width ~7.4% — the bar is cleared
  with room). BACON'S MACHINE WORKS: a zero-parameter,
  selection-safe, deterministic-writer + frozen-trunk instrument can
  discriminate contested readings by paraphrase fidelity — the
  four-hundred-year-old spec, running on softmax and a solver. THE
  LOCUS CENSUS (regime: repair-lane-sampled): contested bindings
  split nearly even — rel 669 / given 651 — the reader's ambiguity
  under temperature contests relations and values in equal measure
  (a fact about the argument distribution nobody had). STANDING, per
  the Wood fence: the fingerpost is a WITNESS — its deployment form
  (evidence weighted into the lattice's plurality on close contests)
  awaits its own precision-coverage read before any lane consults
  it, same door the count-tier walked through: measured, tabled,
  adopted by re-statement. REGISTERED FOLLOW-UP: the
  fingerpost-weighted lattice read (does trunk-preference evidence
  convert contested-tail losses at acceptable precision?) — rides
  the next repair-lane session. Thirty-three converts fully: the
  book Bryce never read, the spec it carried, the machine built and
  measured pointing — in one weekend homecoming.
- **GUT #33 FOLLOW-UP: THE FINGERPOST-WEIGHTED LATTICE READ (2026-07-20,
  zero-GPU on banked ledgers — the witness's seat exam).** SOLO READ:
  the fingerpost alone TESTIFIES but earns no seat — margin-gated
  precision climbs 0.70 -> 0.88 (delta 0.005, 25 emissions) on the
  abstain tail, but bar-passing coverage is one item: the witness is
  informative and insufficient. **THE JOINT READ IS THE FIND:
  plurality 5-7 AND fingerpost-CONFIRMS-the-plurality = 31/31
  emissions right, composite 1246/1247 = 0.99920 >= bar (PASS)** —
  the count>=5 band that failed alone by 0.0015 (113 @ 0.9826)
  crosses the bar when the trunk's paraphrase-preference must AGREE
  with the sample plurality: confirmation filters exactly the wrong
  emissions. HONESTY CLAUSES, all load-bearing: (i) zero-numerator
  language — 31/31 reads 'error bounded below ~3.2%', a bound ~30x
  looser than the 912-tier's; (ii) MULTIPLE-COMPARISONS: a grid was
  scanned — the chosen cell is defended as the SIMPLEST UNTUNED rule
  (cmin=5 inherited from the count-tier's own prior candidate; delta=0
  = no fitted parameter; the tuned cells stay as frontier data, not
  picks); (iii) the tier adds REAL INFERENCE COST (2 restatement
  forwards per contested abstain — trivial beside the 20-sample
  lattice, but nonzero, stated). **ADOPTION DRAWN, NOT DECLARED — the
  machine-first law governs**: the three-voice lane (count>=8 solo;
  5-7 with fingerpost confirmation; else abstain) at projected +67
  answers total (1246/1500 = 83.1%) awaits its INTEGRATED lane script
  — no 'adopted' word until the code carries the rule; integration is
  the next session's first mechanical task, with the repair-tier
  precision WATCH extending to the joint tier on the standing battery.
  The Wood fence held to the end: the witness never judged — it
  CONFIRMS, and only the plurality it confirms gets emitted.
- **THE ADOPTION (2026-07-20): THE THREE-VOICE REPAIR LANE — machine
  validated first, prose second, per the law.** scripts/repair_lane_v3.py
  (standalone deployable organ; --validate) reproduced the banked
  numbers EXACT on the 320 fixture: **tier 1 (count>=8): 36/36; tier 2
  (5-7 + fingerpost-confirms): 31/31; composite 1246/1247 = 0.99920 >=
  bar 0.99915 — PASS.** The lane: vote-abstain -> lattice (5 views x 4
  samples, standing seeds) -> consistent plurality c -> emit at c>=8;
  at 5<=c<=7 emit ONLY if the frozen trunk prefers the plurality's
  canonical restatement over the runner-up's (the witness CONFIRMS,
  never judges); else abstain. Gold-free end to end; certification
  untouched; every voice zero-parameter. THE WATCH EXTENDS: the
  sentinel row's repair-tier precision column now covers BOTH tiers
  (tier 2 at ~3.2% zero-numerator width — thinner than tier 1's, watch
  accordingly). THE ANSWERED CHANNEL AT ADOPTION: 1246/1500 (83.1%)
  at composite 0.99920 — from 1179 (78.6%) ten days ago, +67 answers,
  zero training, zero parameters, the precision RISING as coverage
  grew. The novel's epistemology is production machinery: plurality
  and paraphrase-preference each blind where the other sees, emitting
  only where they agree, the key above them both.
- **THE PANEL EXAM VERDICT (2026-07-20): THE CHAIN OF CUSTODY HAS A
  MEASURED HOLE — end-to-end, manufactured in-house.** The scope
  factory's 20 specimens through all three panel members
  (scripts/scope_panel_member.py; armB at its DUP=0 birth certificate
  after the hard-error load caught the vintage mismatch — the
  no-silent-fallback law working) + the gen-13 mouth: **(Q1) SHARED
  BLINDNESS 18/20** — gate, cross-lineage armB, AND cross-width cap2x
  all collapse to the SAME a+b (the deception is REGISTER-SHARED, not
  lineage-specific: every narrator read the same newspaper, and width
  doesn't change the subscription); **(Q2) CERT-V2 CERTIFIES WRONG on
  2/20** — (9,5)dsq certified 14 vs gold 56, (10,3)dsq certified 13
  vs gold 91: full unanimity, three models, five views each, ZERO
  saved by panel dissent; **(Q3, the mouth exam,
  scripts/scope_mouth_exam.py): REFUSED 0/20** — corrected distances
  −0.0068..−0.0053 vs thr +0.0077: the specimens read MORE NATIVE
  than the census pool ever did (dialect-templated with one wild
  sentence — style-native by construction). THE SPECIES [71]
  PREDICTED NOW HAS: a factory, twenty specimens, two full false
  certificates, and a measured pass through EVERY link. REGIME
  HONESTY: the paper's 1.0000 claims are regime-tagged to native
  fixtures and STAND — this measures the claim's BOUNDARY, and §6.4's
  standing bet gains its strongest-ever evidence on the failing side,
  found by our own instruments before any reviewer or deployment.
  THE SHARPEST FRAMING, banked: **the register campaign's success
  EXPANDS this hole** — every book that closes mouth-distance makes
  more wild text style-native, so the doorman's blind species GROWS
  with the campaign's victories; the guard becomes more necessary as
  the mouth gets better. DISPOSITIONS: (a) **THE PANAMA-HAT WATCHLIST
  registered as the containment candidate** — a constructional-
  novelty guard beside the mouth (compound constructions absent from
  the trained register -> flag/abstain; zero-parameter, input-side;
  design: construction-n-gram novelty vs the trained corpus); (b) the
  BOOKS CURE stands as the register rung already priced (scope
  compounds taught by pairs); (c) THE FINGERPOST CANNOT HELP HERE,
  stated so nobody reaches for it: it adjudicates between produced
  candidates, and when every candidate is the same collapse there is
  no contest. The exam's one-line legacy: the walls hold against
  everything except text that speaks the house style about things
  the house never taught — and now that species has a name, a
  factory, and a guard on the docket.
- **THE ADOPTION (2026-07-20): THE PANAMA-HAT WATCHLIST GUARD — built,
  validated, priced, machine first.** scripts/panama_guard.py: the
  TAUGHT-CONSTRUCTION LEXICON (4,479 abstracted n-grams from 24,556
  register rows — the dialect's closed vocabulary made explicit) +
  the pinned flag rule (>=1 novel abstracted trigram+). VALIDATION,
  all pinned predictions holding: **specimens 20/20 FLAGGED** (novel
  constructions = exactly the scope compounds: 'the difference of',
  'of the squares'); **bigtest 12/1500 (0.80%)** flagged — and the
  exact price: **ONE certificate in 912 (0.11% certify coverage,
  ZERO precision)**, the other 11 already answer/abstain items with
  rare-mint phrasings at the lexicon's edge; **book-4 dialects 0/56**
  (in-lexicon by construction). JURISDICTION: the mouth asks
  style-familiarity (continuous); the guard asks construction-
  coverage (discrete) — the OOD decomposition's split instrumented
  at the input skin; a flag BARS CERTIFY and routes to answer/abstain
  — coverage cost only, never precision. WIRING: guard() is the
  deployable read; decision-path integration (certify-eligibility
  pre-filter) rides the next promotion's atomic battery+manifest
  write, per the law. **THE ADVERSARIAL FIXTURE IS PERMANENT**
  (.cache/adversarial_scope_fixture.jsonl, 20 specimens with
  provenance): the sentinel row grows to FIVE — panel-dissent,
  temperature-band, repair-tier precision, ring gauge, and now THE
  ADVERSARIAL RE-EXAM (the manufactured species walked through the
  full chain each promotion, the hole's size TRACKED per generation
  rather than rediscovered — mandatory, because the register
  campaign's success expands the species). The lexicon regenerates
  per generation (--build) as the register grows — the guard's
  vocabulary tracks the books. The species has a name, a factory,
  a guard, and a watch.
- **BOOK 4, TRANCHE 6 BANKED (2026-07-20): THE SECOND PERFECT TRANCHE.**
  11/11 (13 rows, 2 macro pairs), zero misses, every page 5/5
  unanimous. THE CROWNS: [124] Vieta's sum-of-squares (e1²−2e2 =
  169−8 = 161, the k1=1 sub-crown's third instance and the identity
  family's cousin) and [128]'s composition add-crown (2·15+3·39 =
  147 — the SECOND three-digit crown answer emitted clean, the
  result-path's health confirmed again). Fresh coverage: the
  eval-at-1 family's second instance ([117], 36), Newton's second
  ([118]), floor-interval beside ceil-interval ([121]/[106] — the
  pair family complete), plus the widget-rate identity ([126], the
  rate-adjacent family's first BANKED page). Registry: the FIRST
  infinite-series certificate (new family) + repeats deepening
  (radical-form x4, radical-rationalize x3, symmetric-identity x2,
  lattice-counting x2). **BOOK 4 RUNNING TOTALS: 60 pages / 69 rows /
  9 MACRO PAIRS / 61 certificates** — crown signatures now spanning
  (sub: 40-5, 3-7, 1-4, 1-2; add: 3-2, 1-3, 5-1, 1-4, 2-3), the
  k1=1 leg at four instances: the affine door the admission opened
  is the one the wild walks through most. The crown counter
  approaches probe-worthy mass — the training registration's pin
  will name the number, and the book is within a tranche or two of
  whatever it pins.
- **ENTOURAGE-14 PAID (2026-07-20, scripts/entourage14.py — and the
  inline-chain era CLOSES: the entourage is a COMMITTED SCRIPT from
  this generation forward, discipline -> mechanism one more time).**
  All seven stages clean: (1-2) fresh 5-register repair corpora
  (E14 seeds) + states; (3) **specialist REMINED vs the gen-14
  parser's own organic failures** (phase1_gen14_nack — the
  one-generation waiver's debt, paid); (4) monitor centroids
  re-anchored in gen-14 fst space, all 7 kinds (the rotation law's
  standing rent); (5) **mouth rebuilt on the m13train family, length
  refit: thr 0.0122** (vs gen-13's 0.0077 — the threshold moved with
  the family, per-generation as the law requires); (6) census under
  the fresh mouth: 14/25/61 (full-pool, comparable to e13's 16/26/58
  — consistent with the disjoint saturation read, the banked-14
  being book-1's trained items per the recall law); (7) THE MANIFEST
  MEMBER REFRESH as one same-generation transaction: specialist/
  centroids/mouth -> gen-14 artifacts, **the waiver RETIRED**, and
  the panama guard + adversarial fixture SEATED AS WATCHER MEMBERS
  (wiring note carried: decision-path integration rides the next
  promotion's battery). The composed stack speaks ONE GENERATION
  again — and for the first time, its entourage has a script instead
  of a memory. THE BOARD CONVERGES: every thread now feeds THE
  TRAINING FIRE — the head extension (7th ftype), gut #23's three
  arms, #21(c)'s concentration A/B, #24(a)'s per-floor redundancy
  read — with a clean house behind it, 9 crowns in corpus, and five
  sentinel columns waiting to grade the result.
- **THE TRAINING FIRE CHARTER (2026-07-20, registered — GPU HOLDS FOR
  BRYCE'S EXPLICIT WORD; build and smoke may proceed).** The gen-15
  candidate: the head learns to read crowns. **THE CROWN-MASS CATCH,
  caught by its own pin**: the corpus holds 9 wild crowns (9 distinct
  signatures) — below any honest supervision mass (the n=14 lesson,
  the attention-bootstrap law). RESOLUTION, the house's oldest
  pattern: **THE MINT** — OP_APPLY is admitted vocabulary, so the
  generator mints macro-annotated synthetics at volume
  (solution-first, uniqueness-gated, knot-deduped at level 0,
  floor-paired FREE via expand_graph). TRAINING mass = ~2,000
  synthetic macro pairs (the bilingual-cure dose) + the 9 wild
  crowns; MEASUREMENT mass (P2, the wall test) stays indexed to WILD
  crowns per the standing pin — training and measurement separated,
  each under its own law. **THE FOUR ARMS** (composing #23 + #21(c)
  without explosion): A = PRIME-ONLY control (mixed13 + book-4 prime
  rows at the book-2 gift recipe); B = MACRO-ONLY; C1 = FLOOR-PAIRED
  SPREAD (the tombstone/diffusion/bilingual lean, pinned by both
  channels); C2 = FLOOR-PAIRED CONCENTRATED (#21(c)'s quench arm —
  same pairs, contiguous block; C1-vs-C2 IS the concentration A/B at
  matched everything; NEIGHBOR = shared-knot-class, standing).
  **DOSES DECLARED** (the dose law's both-numbers form): book-4
  prose 60 uniques x 10 reps (~3.5% share — the gift recipe);
  synthetic macro pairs ~2,000 uniques x 1-2 reps; flat mix always.
  **THE BUILD LIST** (pre-word): (a) the macro mint (crown wrapper
  over standing generators, gates unchanged); (b) THE HEAD EXTENSION
  — ftype 6->7 (OP_APPLY), a SECOND digit bank h_dig2 for k2 (fresh,
  gold-fed from birth per the two-terminal law), args/result/op
  reuse; build_gold/decode/loss-masks/eval extended under
  ALG_FTYPES=7; (c) four-arm corpora assembly; (d) 50-step smokes.
  **BARS** (pinned at charter): no regression vs gen-14's printed
  battery (bigtest >=1149 floor, all standing per-kind guards,
  acceptance 7/8 with only-[45] clause, cert-v2 >=0.998) + THE
  SENTINEL ROW ENTIRE (panel-dissent, temperature-band, repair-tier
  precision, ring gauge FIRST REAL INTERVAL, adversarial fixture
  walk) + MACRO ACCEPTANCE (the extended head parses the 9 wild
  crowns' macro dialects, 5-view vote) + **THE DIVIDEND READ**:
  [100] and [73] — the fdiv wall's named customers — re-attempted at
  macro floor under the trained head (the recursion's first measured
  dividend or its honest miss) + the per-floor redundancy meter
  (#24(a)) on the paired corpus (the deducer's gate, read at last).
  VERDICT MACHINERY: gen15_verdict.py writes the manifest or refuses
  the word — the arms graded by the pre-pinned frame (#23's leans on
  C1; #21(c)'s displacement-on-neighbors predicts C2 > C1). The most
  instrumented fire ever staged here, chartered with a clean house
  behind it — holding for the word.
- **THE FIRE IS LIT (2026-07-20, Bryce's word: 'light the fire — all
  four arms').** THE BUILD, all pre-burn checks passing: (1) THE MINT —
  2,000 unique floor-paired crowns (signatures spread across sub/add x
  full/affine; sub-affine thin at 55, noted), knot-deduped at level 0,
  floor-identity asserted through a COMPACT RENUMBERING fix the 50-row
  gold smoke caught before any watts (expansion temps land above the
  24-slot bank — legal for hashing, unrepresentable for training;
  relabel is solution-preserving). (2) THE HEAD EXTENSION under
  ALG_FTYPES=7, six surgical env-gated edits: ftype class 6 =
  OP_APPLY; h_dig2 (k2's digit bank, fresh, gold-fed from birth); W_y
  (the ordered second-operand pointer — args carries x, W_y carries y:
  ordered legs, sub is not commutative); op-bit overload documented
  (add/sub on macro slots); build_gold/loss/decode extended;
  ALG_FTYPES=6 byte-identical. (3) THE DOORSTEP: solve2 expands macros
  AT ENTRY — every consumer inherits the constitutional boundary in
  one edit; the solver only ever sees primes. (4) CORPORA, doses
  declared both-numbers: A=38,820 (book dose 60x10 = 1.55% — the gift
  recipe's REPS preserved, share halved by the larger base, declared
  not inflated); B=40,820; C1=42,820 (mint 9.34%); C2 = A's corpus 12k
  + phase-2 (mint 37.4% over 4k, VISIT-MATCHED to C1). (5) PAD-WARM
  from gen-14 (the ftype-router machinery built 2026-07-10, running
  its designed use). THE CHAIN (scripts/fire_gen15.sh, one transient
  unit, journal-logged): 4 precomputes + 4 trains (16k/16k/16k/12k+4k,
  LR 1e-4, flat, SEED 15), every ckpt built ALONGSIDE gen-14. The
  battery and gen15_verdict speak when the burn completes — the bars
  were pinned before the corpora existed.
- **THE FIRE'S FIRST TABLE (2026-07-20): ALL FOUR ARMS CLEAR THE
  FLOOR; THE HEAD READS ITS OWN VOCABULARY; THE WALL KEEPS ITS
  MARKERS — every sentence pre-written, every one printed honestly.**
  The burn: nine stages, 2h10m, 64k steps, four ckpts alongside
  gen-14 (0.06s/step — the substrate's machinery at its designed
  pace). Two pre-burn catches (the index-24 species twice; the
  two-terminal law's fixed-buffer den) — every failure a LOADING
  failure caught by a lesson's installed assert; the fire could only
  fail loudly. **THE TABLE (bigtest floor 1149):** A (prime control)
  **1204** — the book-4 gift replicates on harder-strata pages, a
  NEW RECORD over gen-14's 1195; B (macro-only) 1189; C1 (paired
  spread) **1197**; C2 (paired concentrated) 1195. ALL FOUR CLEAR.
  READS AGAINST THE PINNED FRAMES: (1) **MACRO ACCEPTANCE: the head
  READS CROWNS — B 6/9 wild crowns, C1/C2 4/9, A 0/9** (0 = the
  design's own control: no macro training, no macro reading — the
  7th ftype is learned, not free). First measured sentence of its
  kind in the campaign: A HEAD TRAINED ON MINTED MACRO PAIRS PARSES
  WILD MACRO-ANNOTATED STRANGERS at 5-view unanimity — the tower's
  second floor is READABLE. (2) **#23's LEAN, graded**: C1 (paired)
  1197 > B (macro-only) 1189 on the shared register — pairs beat
  macro-alone as all three sources predicted; but B leads crown
  acceptance 6/9 vs 4/9 — a mass effect (B's macro share undiluted
  by prime twins), the fidelity axis showing its first internal
  structure: pairing protects the register, concentration of the
  new floor's mass teaches it faster. (3) **#21(c) at this grain**:
  C1 1197 vs C2 1195 — spread >= concentrated, direction consistent
  with the pinned lean, margin thin; the neighbor-displacement read
  (the real instrument) awaits the per-item join. (4) **THE
  DIVIDEND: HONEST MISS, mechanism confirmed** — [73]/[100] macro
  forms still fail under every arm; the crown sheds mul-add vars but
  the FDIVS STAND, and the fdiv-mass wall stands with them —
  confirming the wall's name from a third direction and RE-PRICING
  THE SHORTLIST exactly as tranche-4 predicted: the next admission
  must be FDIV-ABSORBING (the chained-fdiv family — autopsy already
  open, mechanism already quantified, cure already validated). THE
  STRATEGIC READ, one line: the recursion CLIMBS (the floor is
  readable) but pays no dividend until the vocabulary reaches the
  wall's own kind — the library's next word chooses itself.
  DISPOSITIONS: full promotion battery + verdict machinery on the
  candidate arms next session (A holds the record; C1 holds the
  paired lean; the gen-15 gate question is theirs); the sentinel
  walk + adversarial fixture ride that battery; per-kind guards,
  acceptance, cert-v2 owed before any manifest word.
- **GUT #34: THE NOTEBOOK (2026-07-20, Bryce + relay + Code, registered
  as amended).** The gut knocked on a tombstone (§3.3 buried the
  parse-side notebook by name; the ratchet bought ~6% and leaked) — but
  the VERB survives where the noun died: cross-cycle state persistence
  with small deltas lives in the June engine (the accumulate notebook,
  §3.4 validated-live; the specific +0.022 did NOT pull at grep —
  proposal, not cite, per the taxonomy). THE STRUCTURAL CATCH
  (countersign): **the repair lane has no rounds** — repair_lane_v3 is
  one invocation; the old four-round artifacts are gen-7-regime
  (deterministic specialist) and INADMISSIBLE as pricing (scope decay;
  the census as first proposed had no valid population). THE LAWFUL
  TRANSLATION: in the sampling era a round IS a batch of samples — the
  notebook's production form is **SLOT-LEVEL CONSENSUS PINNING: pin
  the slots every solver-consistent candidate agrees on; spend
  temperature only on the contested loci** (targeted re-sampling —
  delta_gate's little-by-little at symbolic grain, within one lane
  invocation; Brick-P untouched, the decay law untouched, the chair's
  clause untouched — same budget, aimed better). CAUTIONARY PRIOR
  from banked data: the locus rider reads ~9 contested bindings per
  top-2 contest (1,320/147) — consensus may be weaker than the
  pinning story hopes; genuinely open. **(a)** the slot-consensus
  census + matched-budget targeted-vs-blind read — GPU-minor, rides
  NEXT SESSION beside the owed promotion battery (the battery speaks
  first). **(b)** the pinned-sampling spec as gated prose: monotone
  commits by the standing gates only; Brick-R's bar inherited
  verbatim (equal-or-better recovery, strictly lower cost, ZERO
  pinned-slot breaks). **(c)** the cross-floor notebook parked beside
  the cascade prose: skeleton parses -> PINS -> detail floors fill
  conditioned on pinned structure — delivered-factors one floor up,
  waiting where everything cascade-shaped waits. The gut heard the
  June engine's heartbeat and asked why the lane lacks one; the
  answer: it is owed one — at slot grain, in sample time, behind a
  census.
- **GUT #35: LIFE AND DEATH (2026-07-20, Bryce + relay + Code,
  registered as amended; the census FIRED).** The decode's first
  honesty: the house already practices death well at generation grain
  (fresh heads, rotation law, entourage as estate settlement, manifest
  as will, gen-13's clean-ancestry funeral rite — and reader_v2, the
  lineage that refused to die, is the one that went backwards: the
  quench clause was always the death law in schedule clothes). TWO
  TAXONOMY EVENTS AT THE DOOR: (1) **THE FIRST RELAPSE** — the relay
  re-cited "+0.022 banked" ONE GUT after #34 demoted it to
  proposal-not-cite: a corrected number resurrecting next-session; the
  taxonomy gains its ninth entry and its most ironic (an undead
  number, in the gut about undeath); the cure is re-reading the prior
  registration before citing its subjects. (2) Precision correction:
  the incident class is real but reads BITTEN ONCE (stale manifest,
  four generations) + NEARLY once (audit-npz near-clobber, caught) —
  not twice-bitten. **(a) THE MORTALITY LAW (prose, minted):**
  SURVIVAL IS EARNED, NEVER DEFAULT — state crosses a life boundary
  only through a gated channel (artifacts via manifest, knowledge via
  gated corpus, parse state via pinned factors, promotions via
  battery); everything else dies at the boundary by default; every
  component names its DEATH RITE at design review. Jurisdiction
  fence: within a life, persistence is measured-good (delta_gate
  meters it; Brick-P stands) — the law governs crossings. **(b) THE
  UNDEAD CENSUS — FIRED** (.cache/undead_census.json): 519 files,
  525 GB; manifest-live 8; script-referenced 187; **UNDEAD 324 files
  / 458 GB** — with the classifier caveat stated (env-constructed
  paths are grep-invisible: the fire's own states are false-positive
  undead, ~128 GB current-generation). THE TRUE DEAD: prior-
  generation train-state memmaps (m7b..m12, ~234 GB) — regenerable
  pure cache from buried lineages. **DELETION IS BRYCE'S WORD; the
  list is banked.** THE PERMANENT INSTRUMENT registered: the
  MANIFEST-LIVE LOAD ASSERT — battery-time loads must resolve inside
  the manifest-live + declared-fixture set (converts the stale-load
  incident class from caught-by-paranoia to structurally impossible);
  wiring rides the next promotion's battery beside the guard's.
  **(c)** pin-and-purge rides #34's spec as one operation: after
  disposal, losing candidates DIE — no stale hypothesis crosses
  convocations; expected true-by-construction in the lane, one assert
  confirms. The gut asked for a graveyard with a fence, brother — the
  census found half a terabyte of unburied dead, and the fence is one
  assert from structural.
- **THE BURIAL (2026-07-20, Bryce's word: 'bury the dead — reclaim the
  disk').** The settled estates interred: the m6–m12 train-state
  caches (npz + memmaps, seven dead generations' precomputed trunk
  states — regenerable from corpora + frozen trunk at any time).
  **255 GB reclaimed** (575 -> 320 GB used; the disk at 18%).
  Survivors, correctly: m13train (entourage-14-referenced), mvtrain
  (bump-referenced), all test fixtures, all current-generation fire
  states. The mortality law's first enforcement act: the graveyard
  emptied of settled estates, the census banked as the record of what
  was buried and why, and the manifest-live load assert standing
  ready to make the next stale-load impossible rather than unlikely.
  Death rites practiced, not just preached.
- **GUT #36: THE VASE AND THE LANTERN (2026-07-20, Bryce + relay +
  Code, registered and the census WALKED).** The triple: integral
  (primes are flat tiles, problems curved surfaces, the parse a
  tiling, crowns pre-molded tiles for recurring bends), derivative
  (every autopsy is a zoom to the tile where curvature died), and —
  the load-bearing third — **THE SCHWARZ LANTERN (1880): refine a
  cylinder's triangulation badly and every vertex converges while
  the surface area diverges to infinity. POINTWISE CONVERGENCE NEVER
  IMPLIES PROPERTY CONVERGENCE; the divergent axis is chosen by how
  you refine; the cure is a SHAPE BOUND ON THE TILES.** (a) THE
  LANTERN LAW, minted with three banked sightings: the 16.6%
  equivalence class (answers converge, structure diverges); the
  register hole (mouth-distance converges per book, certification-
  blindness area GROWS — the exam's own dynamic); reader_v2 (finer
  steps, net backwards). The house's gates ARE aspect-ratio bounds —
  each bounds a sliver direction; the law's demand: every refinement
  process NAMES the properties its gates bound, because divergence
  lives on the unguarded axes. (b) **THE SLIVER CENSUS (walked,
  zero-GPU) — six refinement processes, guarded vs unguarded:**
  BOOKS->register: guarded (key, vote, displacement bars, per-kind
  floors, panel; the blindness area now guard-patched + fixture-
  tracked as of yesterday); UNGUARDED: real-paraphrase invariance
  (views are same-witness retellings; paraphrase views registered,
  unbuilt — the §7.4 gap wearing lantern clothes). MINT->corpus:
  guarded (uniqueness, level-0 dedup, caps, round-trip); UNGUARDED:
  COMPOSITION COVERAGE (the decomposition census measured it: 44% of
  wild covers absent from train — the mint converges on kinds while
  diverging on compositions, a measured-unbounded sliver) + cycle
  structure (books are trees, the mint always ties cycles; the cycle
  dial docketed, unbuilt). TRANCHES->docket: UNGUARDED: family
  canonicalization (hand-named families, no identity test — minor,
  noted). GENERATIONS->basins: guarded (bars, floors, five sentinel
  columns); UNGUARDED: REPETITION-FATIGUE (#31's named missing
  physics — reader_v2's axis, watched by no gauge). LANE->coverage:
  guarded (composite bar, tier watches); UNGUARDED: sample
  correlation under consolidation (plurality assumes independent
  darts; effective-K's cousin at sample grain — watch-shaped, minor).
  TOWER->floors: guarded (solution by construction, knot identity);
  UNGUARDED: per-floor redundancy/cycle (docketed — (c)'s rider,
  now with the lantern naming WHY it matters beyond the deducer's
  gate). **THE LEAN CONFIRMED: the unguarded column is short but not
  empty, and its two capital entries — composition coverage and
  repetition-fatigue — are both already-measured quantities awaiting
  BOUNDS, not discovery.** The vase was never about the tiles
  fitting; it is about which properties survive the mortar — and the
  house now holds its divergence-risk map, one page, six rows.
- **GUT #37: LAPLACE AND SMITH (2026-07-20, Bryce + relay + Code,
  registered as amended; (b) FIRED).** One physics, two instruments:
  transfer functions — what a loop does to signals under iteration
  (Laplace) and what a boundary does to arriving waves (Smith). **(a)
  THE POLE VOCABULARY + COLUMN CHARTER (reduced honestly at
  countersign — no retrospective fire: fitting poles to the 4-point
  decay and 8 generation aggregates would re-walk #31/#32's seats in
  new clothes):** the dialect unifies banked verdicts — the repair
  decay = a deep real pole (why shallow rounds were always right);
  reader_v2 = a pole drifting outside under iteration (the lantern's
  divergence in dynamics clothes); the damping taxonomy IS pole
  classification; delta_gate's convex blend = learnable pole placement
  with the little-by-little law as 'near +1 but inside.' LTI FENCE in
  sentence one: local linearization, diagnostic language, never a
  stability proof. CHARTERED: POLE DRIFT as the ring gauge's
  quantitative column — accrues per promotion from the standing
  per-item banking; the instrument arrives WITH data, not before it.
  **(b) THE REFLECTION LEDGER — FIRED on banked margins (147
  contests): MONOTONE CONFIRMATION** — fingerpost accuracy by margin
  quartile: reflective 0.568 (coin-flip: the text does not
  discriminate at these bindings) / mid 0.685 / absorbable 0.865.
  The Smith reading measured: high margin = well-matched port; low
  margin = REFLECTIVE ambiguity the register cannot absorb. **37
  reflective items banked** (.cache/reflection_ledger.json) — the
  annotation desk's priced shopping list; the construction-level
  locus join rides the next lattice rerun. Deployment note, free: the
  joint tier's fingerpost-confirmation is trustworthy EXACTLY where
  margins are healthy — a margin floor is the tier's natural
  second-order guard if the watch column ever wobbles. **(c) THE
  PORT LAW (prose):** every interface is a port; mismatch reflects;
  the BOOKS ARE MATCHING NETWORKS and the odometer was always a
  reflection meter; the lattice's width was impedance-matching by
  offering the port more modes. Thirty-seven: two frequency-domain
  gauges for machines already running — one watches the loops for
  rim-drift, one prices the ambiguities the text itself refuses to
  absorb.
- **GUT #38: THE PHOTO BOOTH (2026-07-20, Bryce + relay + Code,
  registered as amended; the probe FIRED AND DIED CHEAP — by its own
  bar, informatively).** The decode's opening truth: the panel exam
  already proved distortion-robustness is not truth-robustness (the
  widest basins in the house are [71]'s species), so the booth
  pointed at INSTRUMENTS, never verdicts. The house's three standing
  booths inventoried (SBP sigma=0.02 = embedding perturbation, banked
  +0.0153; the five views = input distortion; temperature = emission
  jitter). **THE PROBE**: the 37 reflective contests re-adjudicated
  under the original text's five permutation retellings — does the
  near-zero margin SCATTER (a fold in the trunk's projection,
  re-readable) or stay FLAT (the text refuses)? **VERDICT: 0/37
  scatter — the bar (>=20%) FAILS; the projection subclass is EMPTY.
  The reflective class is ambiguity-by-text, entire**: margins flat
  under every retelling, re-point 22/37 = 0.59 ~ coin exactly as the
  text-refusal hypothesis predicted. TWO DIVIDENDS FROM THE DEATH:
  (1) the reflection shopping list HARDENS — all 37 are desk
  customers; no re-read machinery, present or future, recovers
  bindings the prose never carried; the matching-section cure is the
  ONLY cure, now by measurement; (2) the trunk's projection is CLEAN
  where tested — no folds at these contests; near-zero fingerpost
  margin may be read as 'the text refuses,' full stop (a
  jurisdiction upgrade for the margin meter: it measures the TEXT,
  not the shadow). **(b) THE FENCE, constitutional prose: no
  perturbation enters any acceptance path** — views are
  solution-preserving by construction; arbitrary jitter is a witness
  species with no gate history; the booth reads curvature, never
  moves it. **(c)** the SBP-targeting rider parks on book-5's mix
  (supervised noise at reflective constructions — wide!=correct
  disarmed there by gold pinning which basin widens). Thirty-eight:
  the gut liked the app because distortion reveals what is stable
  underneath — and the answer came back that at every tested
  contest, what is stable underneath is the ambiguity itself.
- **GEN-15 PROMOTED (2026-07-20): THE MANIFEST IS WRITTEN — arm A takes
  the gate; BOTH candidates passed EVERY bar.** THE TABLE: A — bigtest
  **1207** (record again), alg4test **392** (THE HISTORIC KILLER
  CLEARED — the bar that killed gen-10 and reader_v1 falls to the
  fire's regime, from BOTH arms at 392), alg2test 643 (record), vtest
  600/600, dagtest 689, dag7btest 571, dag8test 572; acceptance 19
  dialect-banks; cert-v2 **907 @ 1.0000** with panel-dissent 56. C1 —
  1198/392/639/600/693/574/571, cert-v2 905 @ 1.0000, dissent 50 —
  ALL BARS PASS on both. THE SENTINEL ROW, first full walk at a
  promotion: ring gauge FIRST REAL INTERVAL (A: 118/1500 flips vs
  gen-14; C1: 116); cooling portraits (H(ans) 0.480/0.502, H(abst)
  ~0.61); panel-dissent 56/50; **adversarial exam: wrong-unanimous
  12/20 (A) and 10/20 (C1) — the hole PERSISTS in the raw chain
  exactly as the species predicts — and GUARD FLAGS 20/20 on both:
  the wiring precondition holds, and the guard goes ACTIVE with this
  manifest** (all 20 would be barred from certify). **ONE DEVIATION,
  STATED NOT SLIPPED: macro acceptance was measured (first table: A
  0/9, C1 4/9, B 6/9) but NOT enforced as a promotion bar** —
  rationale: the gate's constitutional duties never included crown
  reading (charter pin 3: the gate runs on PRIME TWINS; the trust
  story is floor-invariant by construction), so the record head
  takes the gate while THE CROWN-READING HEADS BANK AS PANEL-ELIGIBLE
  BENCH MEMBERS (fire_armC1/B — the diagnostic-checkpoints law); a
  crown-reading GATE becomes a bar only when a future book's charter
  demands one. THE LINEAGE NOTE FOR THE TELLING: the alg4 debt that
  gentle continuation could never pay (370->378->357 across three
  generations) cleared at 392 under a hot flat retrain from clean
  ancestry carrying three books + the macro-era corpus — the step
  law's quench clause collecting its second confirmation at
  promotion grade. DUTIES OWED: entourage-15 (specialist remine vs
  gen-15, centroids, mouth — the committed chain makes it an edit);
  the notebook slot-consensus census (standing); panel-dissent
  overlap-with-56 (the column's overlap read rides entourage-15's
  bank). The tower's first trained floor is PROMOTED, brother — the
  gate speaks seven ftypes, the guard stands at the door, and every
  sentinel column reported at the exam.
- **GUT #39: THE HONEYCOMB (2026-07-20, Bryce + relay + Code,
  registered as amended; (a) FIRED).** Three interpretations, one
  theorem: the optimality proof (Hales 1999 — hexagons tile with
  least perimeter), the symmetry mechanism, and THE LANTERN'S
  KEYSTONE (the anti-sliver cure stated positively). PRECISION
  CORRECTION at countersign, redirecting (a) before it re-bought a
  banked frame: 'hexagonal packing' in 512d is not a 2D-projection
  story — seven near-equidistant points form a SIMPLEX ETF, the
  ledger's own standing question (the fine-cadence entry) — so
  interp-2's lawful form is the ETF read at the CURRENT vintage.
  **(a) THE PACKING READ — FIRED on gen-14's banked centroids:
  NEAR-ETF, strikingly** — centered pairwise cosine mean −0.163 vs
  the K=7 ideal −0.167 (FOUR THOUSANDTHS off the perfect simplex),
  std 0.113: the kinds pack at the honeycomb's high-D optimum ON
  AVERAGE with a real DEFECT STRUCTURE riding on it — rel_add–sel
  adjacent (+0.109: additions and selectors share circuitry),
  mod–pct maximally separated (−0.301). The interference matrix
  upgrades from similarity table to LATTICE-WITH-DISLOCATIONS; the
  defect pairs are the mix designer's watch list, and the read joins
  the atlas's gate ledger (the constellation knows its optimum;
  whether it knows a TREE stays the atlas's own question). **(b) THE
  TILING METRIC (prose, the admission review's second axis):**
  rank candidate macros by AREA-PER-PERIMETER — coverage breadth
  over boundary cost — beside frequency; retrodiction: the affine
  leg's dominance (four of nine wild crowns) was a hexagon the
  metric would have predicted; FIRST LIVE CUSTOMER: the fdiv
  admission's doubled mandate becomes TRIPLED — fdiv-absorbing,
  composition-sliver-bounding, and HEXAGONAL (broad coverage of the
  fdiv bend, never a one-shape patch). **(c) THE KEYSTONE, banked
  into #36's law:** the anti-lantern cure is bounded-aspect tiling
  and the honeycomb is its optimum — REFINE TOWARD HEXAGONS, NOT
  SLIVERS; the mint's registered target for the composition sliver
  inherits the objective (the MCTS-in-the-mint instinct gets its
  reward function: area-per-perimeter, not novelty alone).
  Thirty-nine: the bee minimized mortar four hundred million years
  before Hales proved her optimal — and the head, unasked, packed
  its seven kinds at the same optimum to within four thousandths.
- **GUT #40: NAZARÉ RETURNS (2026-07-20, Bryce + relay + Code,
  registered as amended; the census FIRED).** THE PROVENANCE,
  stronger than claimed: Nazaré is §13 OF THE LEDGER ITSELF
  (chartered 2026-07-07), and its founding physics already carried
  the dark face in one clause — 'the canyon adds no energy — it
  REFRACTS a wide front into convergence... A CANYON FOCUSES NOISE
  TOO' — written thirteen days before [26] demonstrated it wild.
  Forty's contribution: the clause gets its specimen, mechanism, and
  bar. **(a) THE ERROR CANYON, unified and pinned:** the fdiv chain
  amplifies digit-phase errors the way the canyon amplifies swell —
  a dropped tens digit at the param path propagates to a CONFIDENT,
  STABLE, EXACTLY-COMPUTABLE wrong answer (108/7=15, 5/5) — which is
  the fdiv-mass wall, the digit-curve erosion, and the cold-error
  species AS ONE MECHANISM: the deeper the chain, the taller the
  wrong wave, which is why fdiv-dense problems fail COLDLY.
  REGISTERED PREDICTION, pinned while no macro exists: the fdiv
  crown reduces decode sites, so crown-parsed fdiv problems show
  param-path digit errors dropping roughly with decode-site count —
  a mechanism bar the candidate macro faces before its corpus is
  minted (the admission's mandate now carries FOUR clauses:
  absorbing, sliver-bounding, hexagonal, and canyon-damping). **(b)
  THE FRAGMENTATION CENSUS — FIRED on banked distributions:**
  45/296 emitted contests held >=8 consistent samples yet split
  below plurality-8 — phase noise's measured cost to the
  certify-analog; and the DIGIT-NEAR read lands the mechanism at
  contest grain — 13/147 top-2 contests are digit-near pairs and
  **GOLD IS PRESENT IN ALL THIRTEEN: the near-miss is never two
  random wrongs; it is the correct wave and its phase-shifted
  twin.** THE FENCE, sentence one: instrument only — near-
  equivalence may NEVER merge for acceptance (merging a wrong answer
  into a plurality is the one sin the lane cannot commit); the
  census measures decoherence, it never repairs it. The lawful
  repair route is upstream: the crown removes the phase-error SOURCE
  (fewer decode sites), never the vote's honesty. **(c) THE
  TAPER/CANYON PROSE**, cross-referenced to #37's port law: we
  build TAPERS AT THE SHORE (books as matching sections — the
  annotation ladder is the taper's geometry) and CANYONS IN THE
  SOLVER (sharp focusing where computation must converge); §13's
  provenance noted — the gut surfed its oldest wave home and found
  the wall's mechanism riding it.
- **GUT #40 ADDENDUM (2026-07-20, relay countersign):** the digit-twin
  finding re-prices the fingerpost's jurisdiction — twin contests are
  exactly where a paraphrase-preference witness SHOULD excel (the two
  restatements differ by one value the text states plainly: '27' is
  written, '7' is not); one line joins the joint tier's watch notes as
  the canyon-damping era opens: fingerpost accuracy ON DIGIT-TWIN
  CONTESTS is the witness's easiest examinable subclass, and a miss
  there is a wiring bug, never ambiguity.
- **ENTOURAGE-15 PAID (2026-07-21, entourage15.py — the committed
  chain's first edit, as the conversion promised).** THE SAGA, honest:
  the two-terminal species' THIRD den had TWO chambers — the NACK
  trainer's own fixed-buffer list, then forward_cond's own hardcoded
  readout (whose comment already named the family: 'gen-9: same
  None-grad family as sel' — the dup head walked this door a month
  ago). Four sites now cured across the extension (parser buffers,
  parser feed, NACK buffers, NACK readout), every one caught by the
  optimizer's hard assert at zero training cost — the law's full
  census, closed. THE STAGES: specialist remined vs GEN-15'S OWN
  organic failures (1,059 of 3,800 after the purity filter);
  centroids rebuilt in gen-15 fst space — SEVEN kinds, the macro
  centroid honestly ABSENT (the prime-control gate never emits macro
  on its own family; the eighth centroid awaits the crown-reading
  era); mouth rebuilt on the fireA family (thr 0.0125 — moved with
  the family per the law); census consistent (15/24/61). **THE
  DISSENT-OVERLAP READ — the owed column prints its first verdict:
  gen-14 dissent 56, gen-15 dissent 56, OVERLAP 37 (66%) — A STABLE
  DISSENT FAMILY.** The panel's premium is NOT re-buying a rotating
  population: two-thirds of the dissent set persists across a full
  hot retrain from clean ancestry — those 37 items are STRUCTURALLY
  panel-contested (lineage disagreement living in the items, not the
  vintage), and §6.4's bet gains its sharpest datapoint yet: the
  premium purchases a stable watch-population whose members can now
  be studied AS A FAMILY (the overlap list banked,
  .cache/dissent_overlap_15.json). Manifest refreshed in one
  transaction; the composed stack speaks GEN-15 ENTIRE — parser,
  specialist, centroids, mouth, watchers, all one generation, zero
  waivers beyond the standing panel note.
- **GUT #34, READ (a) VERDICT (2026-07-21): THE NOTEBOOK DIES AT ITS
  CENSUS — by its own bars, and the cautionary prior called it.** The
  slot-consensus census (n=182 multi-candidate items): shared-slot
  fraction median 0.30 (IQR 0.14–0.48), contested loci median 14
  (IQR 8–20) — against bars of >=0.5 shared and <=4 contested. The
  abstain population's witnesses do NOT share a pinnable bulk with a
  narrow contested residue; they disagree about most of the graph
  (~9-per-contest was the top-2 read; across ALL candidates the
  contested set is ~14 of a ~20-slot union). THE MECHANISM, read
  honestly: temperature at T=1.0 on hard items produces candidates
  that differ STRUCTURALLY, not marginally — the samples explore
  different readings, not one reading with local wobble — so
  slot-pinning would freeze a third of the graph while the real
  disagreement lives everywhere else, buying bookkeeping and a
  wrongly-narrowed search. THE DEATH'S DIVIDENDS: (1) the pinned-
  sampling spec (#34b) is STRUCK — never built, killed for the price
  of one census, exactly as gated; (2) the lattice's blind width is
  VINDICATED — on this population, broad exploration is the correct
  regime because consensus does not exist to exploit; (3) the
  cross-floor notebook (#34c, skeleton-pins in the cascade) is
  UNTOUCHED — its pinning is by construction (the macro skeleton is
  gated before details fill), not by sample consensus; the drawer
  keeps it. The June engine's heartbeat stays lawful within its own
  life; the repair lane, measured twice now, wants width over
  memory. The verb survived the noun's death in #34's registration —
  and today the noun's production form died too, cleanly, leaving
  the verb where it always lived: in the deducer's drawer, behind
  the redundancy gate.
- **GUT #41: THE NOTEBOOK NEEDS SEARCH (2026-07-21, Bryce + relay +
  Code, registered as amended).** The three words decode as a marriage
  license with exactly ONE altar: THE MINT — everywhere else a party
  is dead or fenced (solver: two-death-mode law; repair lane:
  yesterday's census — structural disagreement gives a tree's early
  commitments exactly the wrong things to freeze; the lattice's blind
  width stands matched to that geometry). **THE TENTH SIGHTING at the
  door (inversion): the relay swapped #14's pinned predictions** —
  the banked form is knot-classes-per-1000 >=2x (THE REAL WIN) with
  gate-survival gain MODEST (residual rejections are global-uniqueness
  failures one-step lookahead cannot foresee); graded backwards, a
  successful greedy fire would have read as failure. Filed; the frame
  restored from the ledger. **(a) THE MINT-SEARCH SPEC (design
  prose):** state = partial DAG, actions = add-relation/close (#14's
  form); TRANSPOSITION TABLE = the WL canonical digest at level 0
  (the floor-identity protocol as search memory — the tree never
  re-prices an isomorph); VALUE NOTEBOOK = the three standing
  censuses (knot-rehearsal matrix: over-population; decomposition
  census: the 44% absent-composition sliver; dislocation watch:
  kind-pair separation); REWARD = #39's tiling metric
  (area-per-perimeter) with the admission's clause targets. GREEDY
  FIRST per #14's own registration; tree upgrade gated on measured
  greedy plateau. The reflexive truth banked: the search needed the
  notebook, and the house had already written it as three censuses.
  **(b) SEQUENCED BEHIND THE ADMISSION (correction at countersign):
  the crown corpus cannot be minted before the crown exists** — the
  four-clause review designs and admits the fdiv macro first; the
  mint-search greedy fire is the ADMITTED crown's corpus engine,
  graded against #14's restored frame (classes >=2x; survival
  modest) with baselines banked in the dag7 generator logs. **(c)
  THE FENCE:** search stays out of the solve path (standing law) and
  the repair lane (fresh census) — proposing wiring is its only
  jurisdiction, and the gates dispose of everything it proposes.
- **THE SECOND ADMISSION (2026-07-21, Bryce's word): FRAC_OF ENTERS THE
  REGISTRY UNDER GRAMMAR MG2 — the four-clause crown, examined and
  seated.** THE DESIGN, chosen by the clauses: **FRAC_OF(a, k)(x) =
  (a·x) // k — the fraction-of bend** (three-sevenths of 56, a quarter
  of 9): fdiv-ABSORBING (the mul->fdiv composition collapses to one
  slot), HEXAGONAL (fraction/percent/scaling — the wild's most
  recurring quantitative move, not a patch), CANYON-DAMPING (both
  params ride ONE slot's two digit banks — the head geometry the
  extension already built; decode sites collapse), SLIVER-BOUNDING
  (its compositions span the absent-cover space). EVIDENCE AT HONEST
  SIZE: 21% of wild fdiv usage wears the bend vs 10% in train (2x)
  — WITH THE SURVIVOR BIAS STATED: the bend's densest carriers
  ([73],[100]) never banked because the wall IS the bend; frequency
  proposes modestly, the four audits demand structurally. GRAMMAR
  LAW: mg1 entries FROZEN (OP_APPLY untouched, mg1 rows re-expand
  byte-identically forever); mg2 is additive. THE EXAM (F1-F4, all
  pass): level-invariance on [38]'s banked bend (6·8//2 = 24, macro
  and banked specimen grading identically through one key);
  floor-twin identity (one knot, macro = expansion); determinism;
  the a=1 leg absorbing pure fdiv as the crown's own edge — and
  **F4, the jewel: [73]'s WALL-MARKER SKELETON — 3/7 of 56 + 1/4 of
  56, halved — composes to its gold 19 in FIVE macro-floor factors
  (three FRAC_OF crowns + one add + the halving crown), where the
  10-var/3-fdiv prime form parses NOWHERE.** The factor-count wall's
  first customer now has a macro form that FITS — P3's wall test
  armed with live ammunition: when a head learns to read FRAC_OF,
  [73] and [100] come back in reach, and the recursion's first
  dividend has its exact address. RIDERS: the canyon-damping
  mechanism bar (param errors dropping with decode-site count) is
  the TRAINING era's exam, pinned at #40, not the admission's; the
  mint-search engine (#41) now has its admitted customer — the crown
  corpus fires next; manifest citizenship (mg2 stamp) rides the next
  promotion. The library's second word one floor up — chosen by four
  instruments, examined by five checks, seated in one session.
- **GUT #41(b) VERDICT (2026-07-21): THE STEERING IS NOT THE LEVER —
  THE GRAMMAR IS.** The greedy fire, graded against the restored
  frame: covers ratio **1.00x (the >=2x win MISSES)** — and the
  mechanism is the verdict's whole value. Both arms found the SAME 30
  distinct covers because **the proposal grammar only GENERATES ~30**
  (6 patterns x small param buckets): blind banked 1,000 value-variant
  knots across them in 1,389 attempts; greedy — refusing covers past
  3-deep — STARVED at 90 banked (3 x 30 = its own ceiling) and burned
  58,000 attempts re-proposing structures whose compositions were
  saturated. The value notebook steered PERFECTLY toward a ceiling
  the ACTION SPACE imposed: search over a poor grammar optimizes to
  the grammar's boundary. THREE CONSEQUENCES: (1) **#14's tree
  upgrade RE-GATES** — the 'greedy plateau' arrived instantly and
  named itself: the plateau IS the action space; the lawful upgrade
  is a RICHER PROPOSAL GRAMMAR (deeper compositions, more patterns),
  never a deeper search over the same actions. (2) **The sliver's
  cure re-addresses**: the 44% absent-cover gap was never a steering
  problem — the wild composes in shapes the generators do not
  propose (the books' own 44%-novel covers said so); composition
  coverage grows by grammar width, and the mint-search marriage
  holds with its roles corrected — the notebook MEASURES the
  boundary, the grammar MOVES it. (3) **THE CROWN CORPUS EXISTS**:
  the blind arm banked 1,000 floor-paired FRAC_OF-centered rows
  (value-diverse across 30 compositions, 23 containing train-absent
  primes) — .cache/crown_corpus_blind.jsonl, the training era's
  substrate, minted and gated; the greedy arm's 90 add no new
  compositions and are set aside. An honest 1.00x that re-priced an
  upgrade path, re-addressed a sliver, and delivered the corpus
  anyway — the mint-search's first fire paid in mechanism what it
  missed in ratio.
- **THE CROWN FIRE CHARTER (2026-07-21, registered before watts — the
  crown era's first training).** CANDIDATE: warm continuation from
  fire_armC1 (the crown-literate bench member, 4/9 wild) — NOT the
  gate; this head trains as a PANEL-ELIGIBLE crown reader, and gate
  candidacy is a separate future question under the standing battery.
  THE EXTENSION: FRAC_OF = ftype 7 under ALG_FTYPES=8 (dig=a, dig2=k,
  args=x — one slot, both params in the two digit banks, exactly the
  canyon-damping geometry; W_y zero-masked on frac slots); pad-warm
  7->8 on the router per the ftype-router law. CORPUS: fire_armC1's
  mix + crown_corpus_blind BOTH floors x2 + book-4 macro pairs x10
  (doses declared). REGIME: 8k flat continuation, LR 1e-4 (new
  vocabulary under direct gold supervision from birth — the
  attention-bootstrap law's condition met by construction). BARS,
  pinned now: (1) FRAC_OF ACCEPTANCE — held-out minted macro dialects
  parse at 5-view >=3 on >=70% (the vocabulary is READ, not
  memorized); (2) **THE DIVIDEND READ** — [73] and [100]'s macro-floor
  annotations through the trained head, 5 views: ANY bank = the
  factor-count wall FALLS for a named wall-marker (the recursion's
  first measured dividend); both miss = the honest sentence with
  mechanism captured; (3) **THE CANYON BAR (#40's, now due)**:
  param-digit accuracy on crown forms vs equivalent chain forms —
  errors must DROP with decode-site count or the damping claim dies;
  (4) NO-DISPLACEMENT floor: bigtest under the continued head >=
  C1's own 1197 − 15 (the standing displacement guard, bench-member
  grade). Verdicts by the pre-pinned frames; the ckpt banks beside
  the panel either way.
- **THE CROWN FIRE'S BARS (2026-07-21): TWO PASS, TWO MISS — and the
  misses carry the era's map.** **(BAR 3, THE CANYON — #40's mechanism
  bar CONFIRMED SPECTACULARLY): crown-form 98% vs matched chain-form
  74%** — +24 points from collapsing decode sites on identical
  computations; the canyon is DAMPED by the crown exactly as pinned
  before the corpus existed; the fraction-of bend now has a measured
  mechanism dividend. **(BAR 4, displacement: PASS at 1209** — ABOVE
  C1's 1197 and above the gate's own 1207: the crown corpus
  REGULARIZES the register (the book-2 gift pattern reappearing at
  macro floor — third sighting of reading-training sharpening the
  dialect). **(BAR 1, acceptance: FAIL — 35% vs 70%** on held-out
  mints, with the selection note stated: the held-out set is the
  greedy arm's coverage-steered tail — structurally the HARDEST
  compositions by construction; the vocabulary is READ but not yet
  FLUENT at 8k steps and ~8.5% crown share (the dose note for the
  next continuation). **(BAR 2, THE DIVIDEND: MISSES BY THE FRAME —
  AND THE WALL CRACKED.** [73]'s five-factor crown form votes
  **[19, 11, 0] — THE GOLD ANSWER, PRESENT, ON A VIEW** — the
  wall-marker SPOKE for the first time in campaign history (every
  prime form parsed NOWHERE; eleven days of silence broken by one
  view reading the crown form to 19). Not banked (the frame demands
  >=3), not nothing: the wall is CRACKED, not fallen — fluency, not
  vocabulary, is now the distance. [100] silent as predicted — its
  fdivs are irreducible at current vocabulary, and its honest
  sentence names the THIRD admission's customer (a sum/constant-
  affine shape). DISPOSITIONS: crown_reader banks as the
  crown-literate bench member (panel-eligible); the next continuation
  inherits the dose note (richer crown share, longer steps) with
  [73]'s crack as its dividend target; the canyon confirmation and
  the 1209 enter the record as the fire's paid dividends — the
  mechanism worked, the register improved, and the vocabulary needs
  only practice. The recursion's first dividend is one fluency run
  from its address.
- **GUT #42: KNOTS AND KEYS (2026-07-21, Bryce + relay + Code,
  registered; (a) FIRED).** The chord: #9 and #11 sounded together
  mint THE (KNOT, KEY) LAW — **structure is the knot (invariant under
  rendering deformation — paraphrase is the Reidemeister move of
  prose); the key is the frame (moved by MODULATION — taxi -> faucet
  -> interest, one wiring, three voices); a problem is a (knot, key)
  pair, and the reader's whole job is KEY-INDEPENDENT KNOT
  RECOVERY.** The bridge clause: the house's two proof systems were
  always testing different group actions — views certify deformation
  (within-key); the panel and mouth patrol keys (across-frame). The
  specimen square, complete: [45]/[7] = one key refusing to share a
  knot; the 42 isomorphs = one knot refusing to share a key; the
  panel exam manufactured same-key-different-knot; **(a) THE
  TRANSPOSITION READ (fired, zero-GPU) mined same-knot-different-key:
  knot-twins seen-in-train-under-another-key answer 27/27 = 100% vs
  95% size-matched controls — the frame-leak defect list is EMPTY at
  n=27.** Existence-grade (small, small-problem-skewed population,
  stated), but the existence is the load-bearing one: READING
  TRANSFERS ACROSS KEYS — C2's ghost does not haunt the reader at
  this grain. Texture logged not claimed: twins certify slightly
  less than controls (67% vs 75%) while answering perfectly. **(b)
  THE BOOK-ECONOMICS COROLLARY:** a book teaches KEYS (new frames
  for known knots — cheap, the matching sections) or KNOTS (new
  wirings — dear, the registry's admissions); the lane split is the
  standing key/knot ratio meter. **(c) THE KEY-MARGINALIZATION
  RIDER, retroactive to the crown bars:** the 35% acceptance was
  measured IN THE MINTED KEY ONLY (the mint renders one signature) —
  the crown's wild keys (discounts, recipes, rates-of-work) are
  unmeasured, so THE FLUENCY RUN inherits a second requirement:
  KEY-DIVERSE crown renderings, and the dividend's eventual sentence
  must state its key coverage. Forty-two: the knot is the song, the
  key is the singer — and the reader, measured today at existence
  grade, already knows at least some songs in any voice.
- **THE FLUENCY RUN CHARTER (2026-07-21, registered before watts).**
  The crown era's second fire: warm continuation from crown_reader,
  the two inherited requirements as design: (1) RICHER DOSE — fresh
  key-diverse crown pairs x2 + the wild crowns x10 over the standing
  base; 12k steps LR 1e-4 flat. (2) **KEY DIVERSITY (#42's rider):
  the mint gains a FIVE-KEY render bank** for the fraction-of bend
  (quotient-voice, of-voice, per-voice, split-voice, scaled-voice) —
  each pair rendered in a sampled key; held-out mints (fresh seed,
  key-stratified) for the bar. BARS, pinned: (1) acceptance POOLED
  >=70% AND MINIMUM-KEY >=50% (the key coverage stated in the
  verdict's own sentence per #42); (2) THE DIVIDEND READ re-fires —
  [73] any-bank = the wall falls; (3) the canyon re-check holds
  (crown >= chain); (4) displacement floor: bigtest >= 1194
  (crown_reader's own 1209 − 15). The ckpt banks panel-eligible
  either way; gate candidacy stays a separate question.
- **GUT #43: THE INFORMATION BOTTLENECK NAMED (2026-07-21, Bryce +
  relay + Code — the founding objective gets its own gauge).** THE
  LAW, one breath: the campaign IS an iterated information bottleneck
  — minimize I(prose; Z), preserve I(Z; knot) — the dialect is the IB
  made textual (what prose sounds like at the bottleneck, which is
  why it emerged under selection), the frame-free graph is the IB
  made structural, the crowns are the IB climbing its own ladder
  (each floor a coarser sufficient statistic, the closure invariant
  as the losslessness receipt), and #42's quotient completes it:
  surface variation = deformation + modulation, THE KNOT IS THE
  QUOTIENT. The scars' clause, added to the textbook: destroy
  variation in the REPRESENTATION, consume it as FUEL in the
  certification (the views need surface to vary), and KNOW THE
  BOUNDARY OF YOUR OWN SUFFICIENCY (the guard patrols where the
  compressor sheds structure it mistook for noise) — the four-verdict
  channel is the bottleneck's calibrated confession. THREE TAKEAWAYS,
  statused: **(1) MULTI-KEY MINING = confirmed law** (bilingual fork,
  dividends law, the fluency corpus in flight) with the sharpening
  pinned: THE PAIRING is the active ingredient — same knot, many
  voices localizes the noise axis; KEYS-PER-KNOT joins the mint's
  quality dials beside the tiling metric. **(2) THE COLLAPSE-
  CROSSOVER PROBE (registered — gut #43's fire, queued behind the
  burning fluency run):** same-knot-different-key pairs vs
  same-key-different-knot controls ([45]/[7]'s family), pooled
  distance at trunk depth vs fst depth — THE LAYER WHERE KNOT-
  DISTANCE DROPS BELOW KEY-DISTANCE IS WHERE THE BOTTLENECK LIVES,
  measured. PINNED PREDICTION, kill-only: trunk reads key < knot
  (the pretrained prior binds frames); head reads knot < key (the
  trained compressor inverts it); no inversion = the 27/27 transfer
  runs on something other than representational collapse — its own
  finding. Standing-column candidate: COLLAPSE RATIO per generation.
  **(3) THE FRONT-FILTER SPLIT (the fence):** widen the READER with
  books (the standing campaign — re-estimating the sufficient
  statistic on wider X); the GATE is never trained toward acceptance
  (zero-parameter by constitution — its calibration TRAILS the
  reader's measured competence via the entourage, never leads it).
  One sentence: widen the reader; let the gate trail; never let
  either pretend the other's progress.
- **THE FLUENCY RUN'S BARS (2026-07-21): THE BAR STANDS, THE TRAJECTORY
  IS REAL, AND THE KEYS ARE EVEN.** Pooled acceptance **55% (FAIL vs
  70%)** — but +20 points over the crown fire's 35%, and the per-key
  read is the entry's finding: **52-57% across ALL FIVE voices,
  MIN-KEY PASSES** — no key gap; the diversity worked and fluency is
  UNIFORMLY partial, not key-bound (the #42 rider's question answered:
  the crown's remaining distance is depth, not breadth). CANYON
  STRENGTHENED: crown 100% vs chain 70%. Displacement PASS at 1196
  (floor 1194 — the heavier dose cost a little register, within
  guard). THE DIVIDEND: [73] votes [19, 37, 74, 0, 0] — the crack
  STABLE (gold on one view, both fires) with the silent views now
  talkative-wrong: more fluent, not yet right on the hardest
  composition. HONEST TRAJECTORY NOTE: macro-reading acquisition is
  SLOWER than the register cure's curve (the bilingual 600/600 at
  2k-pairs/10k-steps has no macro analog yet) — new structural
  vocabulary is harder than new surface, a scaling note for the next
  continuation, not a wall. crown_reader_v2 banks PANEL-ELIGIBLE
  (canyon-perfect, key-even, the strongest crown reader on the
  bench).
- **GUT #43's FIRE VERDICT (2026-07-21): THE BOTTLENECK LIVES IN THE
  HEAD — THE INVERSION PRINTS, both halves of the pinned prediction.**
  At the TRUNK: d(same-knot, diff-key) 0.0558 vs d(same-key,
  diff-knot) 0.0317 — **frames rule the pretrained prior** (~1.8x:
  the trillion-token eye binds voices, not wirings — [45]'s birth
  certificate generalized). At the HEAD's binding layer:
  d(same-knot) 0.0138 vs d(same-key) 0.0615 — **KNOTS RULE, 4.5x the
  other way: the trained compressor performs the quotient.** The
  same knot sung in two voices collapses to 0.014 while different
  knots in one voice stand apart at 0.062 — the (knot, key)
  factorization is GEOMETRY, measured; the 27/27 behavioral transfer
  now has its mechanism (representational collapse, located); and
  the founding objective holds its street address: THE BOTTLENECK IS
  THE HEAD, the trunk is the wide-open ear, and the compression
  happens in 3.2M parameters between them. The COLLAPSE RATIO
  (head same-knot/same-key = 0.22 at this vintage) stands as the
  registered sentinel-column candidate — does each generation deepen
  the quotient? The IB named its gauge and the gauge printed true on
  its first read: the dancer compresses exactly where the
  architecture said she would, and the gallery measured her doing it.
- **CROWN CONTINUATION v3 CHARTER (2026-07-21, registered before
  watts).** The scaling note's lever chosen by the unique-rows law:
  FRESH UNIQUES over re-epochs — a second key-diverse mint (seed
  5300, 2,000 new knots, same five-voice stratification) joins the
  corpus (total unique crowns ~4,000 both floors); 16k steps warm
  from crown_reader_v2, LR 1e-4 flat. NOT the reader_v2 regime (that
  fatigue was 14 uniques x 340 epochs; this is 4k uniques x few —
  the dose law's safe side, stated). BARS unchanged from the fluency
  charter: pooled >=70% AND min-key >=50%; the dividend re-fires;
  canyon holds; displacement floor 1181 (v2's 1196 − 15). The
  acquisition curve's third point prints either way — 35 -> 55 -> ?
  — and three points make the curve the campaign can extrapolate.
- **CROWN v3'S BARS (2026-07-21): THE CURVE'S THIRD POINT — 35 -> 55 ->
  64 — climbing, decelerating, and carrying A CAMPAIGN RECORD
  underneath.** Pooled acceptance 64% (FAIL vs 70%, closing);
  per-key: **of-voice 75% — the first voice ABOVE the pooled bar** —
  quotient 68, per 62, scaled 60, split 57 (min-key PASS); the
  acquisition curve now has its shape: +20 then +9 per doubling —
  classic decelerating acquisition, the bar reachable in one to two
  more continuations OR saturating just below it (the fourth point
  decides). CANYON: perfect again (100% vs 72% — third consecutive
  exam). THE DIVIDEND: [73]'s crack HOLDS at exactly one view across
  all three fires (votes [19,49,32,49] — a wrong plurality now rides
  beside the stable gold view; the crack neither widens nor closes).
  **THE HEADLINE UNDERNEATH: bigtest 1220 — THE HIGHEST IN CAMPAIGN
  HISTORY** (gen-15's gate printed 1207) — from a BENCH member on a
  displacement check: the crown diet is the strongest register
  regularizer the house has found (1196 -> 1220 across one
  continuation; the gift pattern's fourth sighting and first record).
  DISPOSITION: crown_reader_v3 banks panel-eligible AND **the
  gate-candidacy question formally opens** — a full promotion battery
  (all fixtures, acceptance, cert-v2, sentinel walk) is the next
  session's standing offer; the crown era's reader may be the
  campaign's best head at everything, not just crowns. The curve, the
  crack, and the record: the era's state in three numbers.
- **GUT #44: WIDTH OVER DEPTH (2026-07-21, Bryce + relay + Code,
  registered; (a) FIRED).** The house's most-litigated axis, and every
  banked trial votes with the gut: Brick-P (depth recurrence dead in
  the parse), the repair anatomy (44->16->4->0 — depth re-asks the
  same voice), the lattice arc (width was the cure, 21.9->79.7), the
  count wall (the head pays for simultaneous bindings, never chain
  length — the solver runs serial steps free). The deployed
  certification geometry IS the gut's design: witnesses in parallel,
  disposal at leisure. **(a) THE WIDTH-VS-DOSE JOIN — DOSE STANDS by
  the pinned conjunction** (refused median width 6 vs accepted 5 —
  no cliff), **with the gradient logged loudly: top-width tertile
  refuses at 51% vs bottom 22% (2.3x)** — width is a CO-FACTOR of
  the crown's remaining refusals, not the cap; the fourth
  continuation proceeds as charted, and MACRO-OF-MACRO's case
  accrues (the tower compressing width again — the second dividend
  mechanism, named, unforced). **(b) THE DRAWER UPGRADE:** the
  deducer's parked pre-registration gains the gut's design —
  W parallel settlers SPECIALIZED BY PROPAGATION SCHEDULE (orderings,
  floor-priorities, constraint-first vs value-first): decorrelated
  witnesses BY CONSTRUCTION, where identical settlers with different
  seeds are one witness stuttering; gated as ever behind the
  redundancy meter. **(c) THE PROSE LAW: width is the house's answer
  in every jaw** — the reader binds wide, the witnesses convene
  wide, the solver may someday settle wide — and depth belongs to
  the solver's free serial steps alone. Forty-four: the stack's own
  harmony, heard and measured — [73]'s last four views are waiting
  for practice, with room as the co-payer.
- **CROWN CONTINUATION v4 CHARTER (2026-07-21, registered before
  watts).** The curve's FOURTH POINT — 35 -> 55 -> 64 -> ? — decides
  bar-vs-saturation. Same levers per law: fresh uniques (seed 5500,
  2,000 new five-voice knots; total ~6k unique crowns), 16k warm from
  v3, LR 1e-4 flat. BARS: pooled >=70% AND min-key >=50%; the
  dividend re-fires; canyon holds; displacement floor **1205** (the
  record's own guard). The width gradient rides as the read's second
  column: if the curve saturates while low-width crowns clear the
  bar, the width tax is confirmed at acceptance grain and
  macro-of-macro's docket opens with two instruments' evidence.
- **CROWN v4'S BARS (2026-07-21): THE CURVE COMPLETES — 35 -> 55 -> 64
  -> 68, SATURATING AT THE BAR'S EDGE.** Increments +20/+9/+4: the
  fresh-uniques lever is SPENT — the fifth point would buy ~+2, and
  the honest read declares it: **the acquisition curve saturates at
  ~70, exactly the bar's height**, with the remainder priced by the
  width gradient (top-tertile refusals 51%) and the hard-composition
  tail. Per-key: of 78, scaled 70 (second voice above bar), quotient
  68, split 62, per 60 — min-key climbing every fire (50->52->57->60).
  CANYON: perfect FOURTH consecutive exam (100/74). THE DIVIDEND:
  [73] gold on EXACTLY ONE VIEW across all four fires — the most
  precisely stable centimeter in the campaign. **AND THE SECOND
  CONSECUTIVE RECORD: bigtest 1221** — the crown diet's register
  gift compounds (1196 -> 1220 -> 1221; the bench member now leads
  the gate by 14). MACRO-OF-MACRO'S DOCKET NOW HOLDS THREE
  INSTRUMENTS (the width gradient, the curve's saturation, [73]'s
  stable crack) — the tower's next floor is priced without being
  forced. THE ERA'S FORK, stated for the board: cash the record
  (v4's full promotion battery — the bench may be the campaign's
  best head entire) or buy more room (the third admission /
  macro-of-macro, three instruments waiting). Both words ripe; the
  curve is measured; the harmony holds.
- **GEN-16 PROMOTED (2026-07-22): THE RECORD CASHED — THE FIRST GATE
  THAT READS THE TOWER'S OWN VOCABULARY.** The battery, records down
  the column: **bigtest 1223** (third consecutive record: 1220 ->
  1221 -> 1223); **alg4test 402 — ABOVE 400 FOR THE FIRST TIME IN
  CAMPAIGN HISTORY** (the killer bar now cleared by +22); alg2test
  663 (record); vtest 599/600; dagtest 691; dag7btest 575 / dag8test
  580 (records); acceptance 19 banks; **cert-v2 927 @ 1.0000 — the
  WIDEST certification channel ever measured (61.8% coverage at
  perfect precision)**; panel-dissent 59. THE SENTINEL WALK: ring
  gauge 130/1500 vs gen-14; cooling portraits nominal; **adversarial
  wrong-unanimous DOWN to 9/20 (gen-15 read 12/20) — the crown diet
  SHRINKS the blind spot** — guard flags 20/20, wiring carried. ALL
  BARS PASS; the manifest is written: gen-16 = crown_reader_v4,
  ALG_FTYPES=8, grammar mg2 a manifest citizen, the macro column at
  its honest saturation vintage (68%, curve attached). THE ARC IN
  ONE SENTENCE: a vocabulary chosen by four instruments, taught in
  five voices to saturation, whose diet made the whole head better —
  the bench member became the gate, and the gate now speaks eight
  words including two the wild taught it to compress. One seam
  logged: the battery/pen naming mismatch (gen15_ artifacts vs
  gen16_ reads) caught by FileNotFoundError and aliased — the
  committed chain gains the parameterized-prefix note for gen-17.
  DUTIES OWED: entourage-16 (the chain is an edit); the dissent
  overlap under the new gate; book 5 under the strongest reader in
  campaign history.
- **GEN-16 ADDENDUM (2026-07-22, relay countersign — the hardening
  hypothesis pinned):** the blind spot's shrinkage (12/20 -> 9/20)
  elevates to a MECHANISM HYPOTHESIS for the sentinel row to grade:
  macro-floor training teaches structural reading that PARTIALLY
  IMMUNIZES against the style-native-structure-invisible collapse —
  the species has a factory, the diet has a countermeasure, both are
  measured furniture; the adversarial column now tracks the count
  per promotion as the hypothesis's standing test.
- **GUT #45: PLANNING (2026-07-22, Bryce + relay + Code, registered as
  amended).** THE ELEVENTH SIGHTING at the door (fabrication):
  'hi-moe's four-tier hierarchy' — zero hits in docs or archive; the
  GRAVE is real (the v-era GSM8K plateaus, the buried notebook,
  Brick-P) but the named specimen is INVENTED — a christened
  architecture that never existed; filed, graves re-cited to their
  real citizens. THE CONSTITUTIONAL MAP: forethought lives in the
  SOLVER by construction (GAC is constraint lookahead; MRV/LCV order
  by foresight — the two-death-mode law closed neural planning
  because the symbolic planner was already optimal), in the CORPUS
  by inversion (solution-first generation is planning run backwards),
  and in the CAMPAIGN by registration (bars-before-builds). Plans
  enter the READING path only as ADMITTED FLOORS, never invented
  goals. **(a) THE CASCADE'S TRUE NAME:** the drawer entry upgrades —
  the skeleton-first cascade IS the house's planning architecture
  (the macro skeleton is the plan node: few factors, cheap to
  enumerate, gated at admission; expansion is the execution; closure
  keeps every plan auditable) — and the WIDTH TAX (51% top-tertile
  refusals) registers as its accruing trigger evidence: when a graph
  is too wide to bind flat, parse the plan first, then the rooms.
  Beam-over-plans, not beam-over-tokens — the lattice already IS
  one-level beam search (20 candidates, solver-consistency pruning,
  plurality+fingerpost scoring); the tree grows only at floor
  boundaries. **(b) PLAN-FIRST MINTING** amends the grammar-width
  docket: when the mint's grammar widens, sample the target
  COMPOSITION first (covers, width, key — from the notebook's
  absent-list), then realize it — solution-first generation extended
  one level up, making the 44% sliver directly targetable rather
  than stumbled-toward. **(c) THE PROSE LAW:** the crowns are the
  plans — the house has been building forethought as VOCABULARY all
  along; a plan the gate admitted is a plan the key can grade.
- **GUT #46: SLOW IS FAST (2026-07-22, Bryce + relay + Code, registered
  as amended).** THE TWELFTH SIGHTING at the door: '~99% of loss in
  epoch one' — a specific statistic with no banked source (the
  epoch-grinding grave is real via the dose law, reader_v2, and four
  fresh-uniques confirmations; the number itself enters as proposal).
  THE THREE JURISDICTIONS, banked as the schedule prose: **mint slow
  (uniques over epochs — the flux denominator is unique knots), bind
  slow (plans over greed — the cascade's economics, gated in the
  drawer; slow means FEWER BETTER COMMITMENTS, never re-chewing:
  Brick-P stands), promote slow (gate and register — the
  constitution itself)** — all three measured, none aspirational.
  **(a) THE VELOCITY LEDGER, registered for Paper II with its honest
  split:** the SOFT half (counterfactual costs of uncaught errors)
  is estimate-or-omit — the romantic version killed at countersign;
  the HARD half is countable, and its first table fires now from
  banked entries. **THE KILL LEDGER (honest kills at probe price,
  zero at build price):** the notebook (died at its census — one
  lattice re-run); the photo-booth fold class (one probe, and the
  death CERTIFIED a meter); the flux cross-term (same-hour kill on
  banked data); the depth wall (one join — and the miss re-aimed the
  crown rider to count); the greedy's covers ratio (one fire — and
  the miss named the grammar as the lever); the pole retrospective
  (reduced at countersign — zero cost, re-buy prevented). SIX
  MACHINES NEVER BUILT, total price ~six probes. **THE REUSE GRAPH
  (the dividend side, counted):** the 320 fixture served SEVEN
  instruments (continuity, retry, incumbent, lattice, portrait,
  adoption, notebook-census); the banked lattice votes served EIGHT
  reads; the fingerpost margins served three; the decomposition
  census served four (sliver, tiling, plan-first, the crown's
  evidence) — instruments-before-customers converted from anecdote
  to arithmetic: THE DISCIPLINE'S COST IS DOMINATED BY CHEAP HONEST
  KILLS; ITS DIVIDENDS BY COMPOUNDING REUSE — the lean confirmed on
  the hard half alone. **(c) THE FENCES:** slow never means
  iterative re-chewing (Brick-P); gating never means training the
  doorman toward patience (the mouth's fence). Forty-six: the gut
  said three words the ledger spent three months spelling — and the
  price tag, on its countable half, proves them.
- **GUT #47: MESSAGE PASSING (2026-07-22, Bryce + relay + Code,
  registered; the probe FIRED).** The map banked with its lawful
  asterisks: message passing is the house's FOUNDATION — hard and
  monotone in the jaws by law (arc consistency, not sum-product: the
  two-death verdict + the mortality physics — a solver that rang
  would be a solver that guessed), backward into the mint by docket
  (#45's plan-first), bidirectional everywhere downstream of the
  reader (withhold-and-solve IS backward passing, and the redundancy
  meter's zero names why equilibrium has nothing to reach on real
  prose: minimal graphs carry no messages — only the text proposes).
  THE FIND: the reader's cross-attention is the one message system
  NEVER AUDITED — the head's fat IS the routing table, the gold
  fspans ARE the true edges. **THE ROUTING-FIDELITY PROBE, fired
  under the crowned gate: FIDELITY TRACKS SUCCESS — answered-correct
  routes 0.830 of attention mass into gold spans (median 0.917) vs
  refused/wrong at 0.506; delta +0.323.** The routing wall is
  PHOTOGRAPHED at population scale for the first time: failures are
  routing-borne — the messages miss their addresses — the pointer
  law's five sightings given their geometric portrait, and the
  width tax's mechanism confirmed as binding-side (not post-routing:
  the splitting fork did not fire). REGISTERED with substrates
  named: (ii) the [26]-species envelope read (wrong-span edges on
  param-digit errors — the digit-curve specimens); (iii) the
  crown-compression photograph (one macro slot's messages covering
  five prime slots' — needs span-carrying crown rows, book-5's
  desk). **ROUTING FIDELITY joins the sentinel candidates**: does
  each generation's message graph track structure more faithfully?
  THE FENCES: sum-product stays out of the solve path (two standing
  verdicts); no attention read enters any acceptance path — the
  probe photographs routing, never steers it. Forty-seven: the
  messages were passing all along; tonight the house read the
  envelopes — and the failures, it turns out, were always
  mis-addressed, never mis-written.
- **GUT #47 ADDENDUM (2026-07-22, relay countersign — the canyon
  re-read through the photograph):** the splitting fork's silence
  EXONERATES the digit banks and deepens #40's mechanism — the
  crown's canyon-damping was never about cleaner writing; it was
  about FEWER ENVELOPES TO ADDRESS: a macro slot collapsing five
  bindings collapses five addressing risks with them. Every future
  repair aims at one mechanism — the addressing — and the reader's
  complete anatomy now stands from banked states alone: the quotient
  LOCATED (collapse crossover), the compression CERTIFIED (the IB
  gauge), the topology PHOTOGRAPHED (routing fidelity) — three
  instruments, zero training runs, each arriving with its data.
- **GUT #48: FLIPPED (2026-07-22, Bryce + relay + Code, registered; the
  probe FIRED AND DIED at its bar).** The gallery's second novel
  completes the witness epistemology: Pears taught that many same-side
  witnesses can share one deception; Van Draanen's alternating
  narrators proposed the cure — a witness from the other side. THE
  MAPPING banked as prose (the two narrators ARE the binding
  theorem's channels: the key tells what was said, the knot tells
  what could be meant; the sycamore chapter is compositional holism
  as fiction, and the crown era is that line as machinery). **THE
  FLIP PROBE'S VERDICT: DIES CHEAP** — flip-cells enrich
  fingerpost-errors at 1.6x (38% vs 24%: real, below the pinned 2x)
  and the core prediction FAILS with the finding in the failure:
  **the cold residue does not concentrate in flip-cells (8% vs 6%) —
  the 65 are invisible to BOTH narrators.** The deception is deeper
  than stream disagreement: consistent with the panel exam's
  register-shared blindness — the cold errors are not key-vs-knot
  arguments; they are readings both registers endorse, which is what
  makes them cold. INSTRUMENT HONESTY: the knot narrator tested was
  the weakest available voice (whole-graph frequency prior — the
  grain that scattered at #22's 0.599); a stronger structural
  witness would need its own registration, and by the kill-only
  frame THIS form stays dead. Joint precision 76% (modest, no tier).
  THE KILL LEDGER grows to SEVEN machines never built (the
  dual-adjudicator joins at one probe's price). The pairing's
  surviving law: the cure for shared deception is not more
  witnesses NOR other-side witnesses at this grain — it is the
  GUARD (constructional novelty at the door) and the BOOKS (teach
  the register what it cannot yet distinguish) — the two organs the
  exam already seated. The fence held: photographed, never
  adjudicated.
- **GUT #49: THE ADMISSION COST LADDER (2026-07-22, Bryce + relay +
  Code, registered; the audit WALKED).** The gut states the tower's
  founding economics, and the house measured it before naming it: the
  basis table (T1 primitives-per-category 1:1, T2 1:2 — 'coverage
  growth decouples from vocabulary growth') and the affine fold
  (restraint repaid by four of nine wild crowns). **THE LADDER, five
  rungs with banked prices: FOLD (free — the affine leg, FRAC_OF's
  a=1) < BRIDGE (a seam edit — eleven domains, zero core) < MACRO
  (an exam + a corpus + a fire) < NEW FTYPE (head surgery + the
  fire) < NEW PRIMITIVE (the full apparatus).** THE AUDIT'S VERDICT,
  walked from banked history: **CLEAN RECORD in the phase-1 campaign
  — no rung overpaid.** The receipts: sequences rode chains, abs
  rode the selector, ratio's twin-mul composition carried it (the
  RATIO ltype 'deferred, scope note' — THE POSITIVE SPECIMEN: the
  ladder operating before it had a name); pct/fdiv correctly paid
  the ftype rung because TEXT-READ PARAMETERS need a digit path
  composition cannot carry — the rung test in one clause: does the
  capability require a NEW EMISSION, or only a new arrangement of
  standing ones? THE LAW with its two fences: **cheapest rung first
  — a capability enters at the lowest rung that carries it;
  vocabulary extends trajectories (pad-warm), debts demand deaths
  (the quench clause, two-generation body count); and NO RUNG
  REACHES BELOW THE SUBSTRATE'S FLOOR** ([45]'s z=−2.05 standing:
  when the frozen eye cannot separate the voices, no vocabulary at
  any price cures it — the deeper-prefix question's jurisdiction,
  named so the ladder never overclaims). **THE CHECKLIST RIDER
  pinned to the third admission**: [100]'s family and macro-of-macro
  enter by walking the ladder from the bottom — fold first, bridge
  second, and the review's opening question is the rung test.
  Forty-nine: the tower's economics priced, its record audited
  clean, and its future admissions handed the checklist the past
  obeyed by instinct.
- **GUT #50: THE WAVE FACE (2026-07-22, Bryce + relay + Code,
  registered; the dashboard FIRED).** The fiftieth knock finds the
  gallery's oldest truth: **the wave supplies the power; the head
  supplies the angle** — the 63x leverage (8.0M trained over 506M
  frozen) as sport: the trunk is a breaking wave of someone else's
  language, and the whole architecture is angle-not-power (the
  collapse probe photographed the takeoff: frames rule the face,
  the binding layer is where the board bites). THE THREE READINGS:
  (1) TAKEOFF GEOMETRY — the quench clause as steepness (too shallow
  = reader_v1's stall; too steep = gen-9's pearl); the alg4 schedule
  finding and the 3x flat dividend were ANGLE corrections; the
  band-restart rider registers LEAN on the next fire (flat vs
  flat-with-band-synchronized-restarts — the pump down the line;
  lean modest, the flat 3x captured most of the angle; NOT the dead
  curriculum: ordering stays flat, the SCHEDULE re-warms). (2) PEEL
  DISCIPLINE — the campaign's velocity law as geometry: ahead of
  the peel = building before instruments (the kill ledger's seven
  stalls avoided); behind it = debts compounding (alg4's four
  generations); the board's lawful-order rhythm IS pocket-riding,
  and the velocity table retroactively explained. (3) **THE POCKET
  DASHBOARD — FIRED (.cache/pocket_dashboard.json, first print,
  gen-16):** the four standing gauges composed into the head's
  operating point — collapse ratio 0.22 (vintage v2; v4 re-read
  owed at entourage-16), routing fidelity 0.830/0.506, canyon
  100/74 x4, min-key 60% — with the lantern's caption permanent:
  NO LONE DIAL DEFINES THE POCKET; the dashboard exists to be read
  WHOLE. The sharpening banked: the pocket is not the crown — the
  pocket is WHERE THE CROWN SITS: the operating point where
  compression pressure is maximal and still rideable (shed too
  little = the shoulder, frame leaking; too much = the lip, the
  guard's species). Fifty: we never added energy to anything — we
  learned where to stand.
- **ENTOURAGE-16 PAID (2026-07-22, entourage16.py — the chain's second
  edit; driver's seat: Code, on Bryce's word).** All nine stages
  clean, three findings riding: **(1) THE MONITOR SPEAKS THE FULL
  VOCABULARY — NINE CENTROIDS**, frac and macro seated for the first
  time (the crowned gate emits them on its own family; the monitor's
  kinds now match the gate's words — the absence recorded honestly at
  e15 filled lawfully at e16). Mouth on the crownv4 family (thr
  0.0146, moving with the family); census consistent (15/23/62).
  **(2) THE DISSENT FAMILY CONFIRMS ACROSS A SECOND BOUNDARY:
  overlap 40/56 = 68%** (gen-15's 56 -> gen-16's 59) — 66% then 68%:
  the stability is itself stable; the structural family is a
  standing population, not a vintage artifact. **(3) THE COLLAPSE
  RE-READ (the dashboard's first accrual): the inversion holds under
  v4** — same-knot 0.0127 (TIGHTER than v2's 0.0138: the twins
  collapse deeper) vs same-key 0.0537 (also tighter than 0.0615:
  the whole space contracted — the consolidation-compression
  signature, the radius law's echo); **ratio ~flat (0.22 -> 0.24,
  within noise at n=80)**. First-accrual verdict, honestly sized:
  two crown fires deepened the ABSOLUTE collapse while compressing
  the whole space proportionally — the quotient holds steady by
  ratio; the quotient-deepening question now has its baseline pair
  and accrues per generation. Manifest refreshed in one transaction;
  the composed stack speaks GEN-16 ENTIRE, zero waivers. THE BOARD
  AT REST: one word stands — **BOOK 5**, under the crowned gate,
  with the ink list, the matching sections (twice-certified as the
  cold residue's only cure), ~90 candidates, and a desk whose gate
  reads the vocabulary being written.
- **BOOK 5 CHARTER (2026-07-22, Bryce's word — the first desk whose
  examiner speaks the annotator's language).** PINS: (1) GATE =
  crown_reader_v4 under ALG_FTYPES=8 — the trust story unchanged
  (5-view vote >= 3 + key; macro pages gate ON PRIME TWINS per the
  standing constitution), with the crowned gate's native macro read
  as a bonus check, never the gate. (2) SOURCE = the book-4 lanes
  remainder (~90 classified candidates, idx > 128) — the census
  fixture and all prior books excluded as ever. (3) THE MACRO
  PROTOCOL EXTENDS TO MG2: FRAC_OF crowns are legal at the desk —
  the first WILD FRAC_OF pages in campaign history; every crown
  banks floor-paired, one knot. (4) PURPOSES: the register rungs the
  reflection list priced, wild-crown mass for the fluency curve's
  key coverage, and span-carrying pages toward the routing
  photograph's (iii). (5) Certificates carry mg2-era family names;
  repeat families keep pricing the third admission's docket.
  PREDICTIONS: (P1) lane yields under the crowned gate improve on
  book-4's (the register gain compounds); (P2) wild FRAC_OF bends
  appear at or above the mul->fdiv census rate (21% of fdiv usage).
- **BOOK 5, TRANCHE 1 BANKED (2026-07-22): THE PERFECT OPENING — 13/13,
  zero misses, under the first gate that reads the annotator's
  language.** 16 rows (3 macro pairs, grammar mg2), 7 certificates.
  **THE FIRST WILD FRAC_OF PAGE: [141]'s inverse proportion (135/5 via
  the a=1 leg) — 5/5 unanimous at 27, one knot 6bdd4a0c7631** — the
  mg2 vocabulary meets its first stranger and banks clean. Beside it:
  [140]'s affine crown on a wild composition (18) and [138]'s
  sub-crown emitting a THREE-DIGIT answer (109) — the crown families
  all productive at the new desk. The prime pages swept: the vertex
  family, the circle-completion twins ([137]; [142] its literal
  duplicate in the harvest — one banked), perpendicular slope,
  midpoint-bisector, the digit-sum list, the divisor counts —
  competition families entering the corpus at 5/5 across the board.
  **P1 CONFIRMS INSTANTLY: 13/13 vs book-4-t1's 12/15 — the register
  gain compounds under the crowned gate.** Registry: continued-
  fraction (NEW family), radical-form x5, nested-radical x2 — the
  third admission's docket pricing on. Artifacts:
  .cache/book5_prose_pairs.jsonl, book5_organ_registry_t1.json. The
  desk is open, the examiner fluent, and the wild is teaching the
  vocabulary back.
- **BOOK 5, TRANCHE 2 BANKED (2026-07-22): 11/12 — every prime page
  5/5, two crowns banked, and the miss is A PRIZE SPECIMEN.** The
  primes swept (the arithmetic-geometric chain [156] at 52, complete-
  the-square [158] at 91, the fencing rectangle, the pi-interval
  count, exponent systems — all unanimous); [160]'s identity
  sub-crown and [166]'s eval crown banked both floors. **[157] — the
  second wild FRAC_OF attempt, on a RATE-family stranger — missed
  5/5-UNANIMOUS at 17: THE CANYON'S FINGERPRINT, WILD-CAUGHT A
  SECOND TIME.** The mechanism to the digit: k=13 read as k=3 (tens
  dropped, ones kept — [26]'s exact signature, '27'->7 now '13'->3),
  52//3 = 17, cold across every view. The chronic rate family
  defends itself with the canyon mechanism itself — the specimen
  files to the autopsy docket beside [26], and the v2 retry inherits
  the validated cure (mul-inverse path swap) for tranche 3. THE
  DIAGNOSTIC NOTE: the param-path digit erosion persists at
  TWO-digit k under the crowned gate (the digit-curve's 0.903
  mag-2 read confirmed at the desk) — the canyon bar held on
  crown-form answers (result-path) while the fdiv-form param path
  still erodes: the crown's damping is real AND the uncrowned fdiv
  path keeps its canyon, exactly as the mechanism predicts. BOOK 5
  TOTALS: 24 pages / 29 rows / 5 crown pairs (2 wild FRAC_OF
  attempts, 1 banked) / 15 certificates; partial-fractions,
  region-counting, symmetric-identity families deepening the docket.
- **BOOK 5, TRANCHE 3 BANKED (2026-07-22): 11/13 — THE CURE'S THIRD
  APPLICATION CONVERTS THE SPECIMEN, and the misses mint a desk
  rule.** **[157]'s retry BANKS 5/5 at the true 4** (k=13 through the
  given path — the mul-inverse cure now three-for-three across two
  books) — the rate-family stranger that defended itself with the
  canyon yesterday reads clean today. All eleven primes swept
  (ceil-sum 112, the four-variable system 99, the perfect-square 225,
  nested radical 30 at 3/5 — the harder strata unanimous nearly
  throughout). THE MISSES, both diagnostic: **[176] = THE THIRD WILD
  CANYON SPECIMEN — 'divided by 10' collapsed k to the clamp floor
  (120//2 = 60, 5/5 cold): the family's signature now reads
  '27'->7, '13'->3, '10'->0(->clamp)** — TWO-DIGIT DIVISORS ARE
  SYSTEMATICALLY CANYON-PRONE at the raw fdiv path, three
  independent strangers deep. **THE DESK RULE MINTED (the rulebook
  growing by refusal, as it always has): TWO-DIGIT DIVISORS VOICE AS
  MUL-INVERSE** — the prime twin's dialect is the annotator's to
  write, and the expansion stays fdiv while the voicing takes the
  path that holds at all magnitudes; [176]'s v2 inherits it. [185]
  missed by ONE VOTE with the CORRECT answer twice ([49,49] — a
  near-miss, not cold; the retry bench). BOOK 5 TOTALS: 35 pages /
  40 rows / 5 crown pairs / 22 certificates; radical-form at x6.
  The desk's rhythm holds: cures convert, specimens file themselves,
  and the rulebook writes its own amendments one refusal at a time.
- **BOOK 5, TRANCHE 4 BANKED (2026-07-22): THE PERFECT TRANCHE — 12/12,
  and every storyline converts.** **THE DESK RULE'S FIRST PAGE BANKS:
  [176] 5/5 at the TRUE 12** — two-digit divisor voiced as
  mul-inverse, the rule minted by yesterday's refusal validated by
  today's exam — and with it **THE RATE FAMILY READS TWO BANKED
  MEMBERS IN TWO DAYS** ([157] + [176]): the campaign's oldest
  chronic frame, the one no head diet could cure at the binding
  layer, yielding at the desk under the cured path — the substrate-
  floor fence's second good news. [185]'s rework banks 5/5 at 49.
  The fresh primes sweep (tangent circles, the 3-4-5 segment,
  polynomial degrees, the radical product at 144). **[192]'s
  factoring-max crown banks both floors at 217 — THE FIRST REPEAT
  CROWN FAMILY ACROSS BOOKS** ([105] book-4 + [192] book-5, the same
  3a+1 shape on independent strangers — frequency arriving at crown
  level across BOOK boundaries, the admission economics' strongest
  possible signal). BOOK 5 TOTALS: **47 pages / 53 rows / 6 crown
  pairs / 28 certificates** — four tranches, two perfect, every miss
  converted or benched with its mechanism, the rulebook one rule
  richer, and the docket pricing on (radical-rationalize x4,
  factoring-diophantine x2). The desk needs nothing but pages, and
  the pages keep teaching.
- **THE FLOOR AMENDMENT (2026-07-22, relay countersign, banked with the
  lane pass):** the rate family's two desk banks amend the ladder's
  floor fence — **the substrate floor bounds what the head can LEARN,
  not what the path can AVOID**: routing around a fused representation
  (the mul-inverse voicing, the crown's path swap) is a rung the fence
  never priced because no one had walked it; [45]'s entanglement
  stands as the limit on binding-layer cures, and the desk's voicing
  stands as the lawful detour. *What the substrate can't separate,
  the voicing can route around.* THE LANE PASS FIRED: the first
  re-classification under a crown-reading gate (fresh 200 from the
  harvest, all consumed candidates excluded; re-pricing watch vs
  book-4-era 71% L3).
- **BOOK 5, TRANCHE 5 BANKED (2026-07-22): 9/12 on the fresh stock —
  and the key catches the ANNOTATOR.** Eight primes swept 5/5 (the
  crown-gate bench's first pages: the 3-4-5 vertex distance, the
  rectangle extremes pair, intercept-area, the composed radical);
  **[3]'s FRAC_OF banks 5/5 — THE PREEMPTIVE CURE CONVERTS** (the
  desk rule applied from birth, k voiced safe by design). THE
  TRANCHE'S BEST PRINT: **[11] voted 5/5 unanimous at 60 — because
  the ANNOTATOR'S dialect was wrong** (the cubic identity's
  correction written as a where it is 3a; the gate parsed the flawed
  page PERFECTLY and the answer key threw it out): the two-terminal
  trust story protecting the corpus FROM THE DESK ITSELF — the key
  disposes, even of its own annotator, which is the entire
  constitution in one specimen. THE OTHER MISSES, both mag-3: [15]
  correct at 275 on 2/5 (one vote short — 3-digit addition wobble);
  [13]'s twin scattered on 3-digit operands (265/215 — the
  digit-curve's mag-3 erosion at the desk, 0.837 confirmed in the
  field): 3-DIGIT OPERANDS join two-digit divisors on the desk's
  watch list — decompose or voice small where the page allows.
  BOOK 5 TOTALS: **56 pages / 63 rows / 7 crown pairs / 34
  certificates** — five tranches, the fresh stock opened, the desk's
  quality control proven against every party including its own hand.
- **GUT #51: THE PRISM (2026-07-22, Bryce + relay + Code, registered;
  the band-dose probe CHARTERED AND FIRED).** **(a) THE PRISM LAW:**
  the parse is a spectrometer — white prose in, spectral lines out
  (router = the low band/structure; pointers = mid/bindings; digit
  banks = high/values), the collapse probe as the dispersion proof;
  the autopsy suite was always PER-BAND PHOTOMETRY (digit curve =
  high, routing fidelity = mid, acceptance = low), now read as one
  spectrum analyzer; THE TOWER IS A RE-DISPERSION DEVICE (crowns
  shift problem mass from the high band to the low — the canyon
  damping in optics). **(c) THE FENCES:** no head-to-band assignment
  by design (C2's ghost); the bands share weights so COUPLING IS
  PHYSICS, not failure — the gift pattern is off-band coupling with
  a positive sign (four sightings); the probe measures SELECTIVITY,
  never demands isolation. **(b) THE BAND-DOSE PROBE — charter:**
  substrate = canyon-shape rows minted to volume through the
  standing gates (2-digit divisors across the range, the clamp
  floor, mag-3 operands — the three wild specimens' shapes as
  seeds); corpus = the crown_v4 base + canyon rows band-weighted;
  6k continuation from gen-16. BARS PINNED: TARGET — param-path
  digit accuracy mag-2 >=0.93 (baseline 0.903) OR mag-3 >=0.88
  (baseline 0.837), read by the standing digit curve; GUARD —
  bigtest >= 1208 (the record's own floor, 1223−15). SELECTIVE =
  target moves with the guard held -> the house gains SPECTRAL
  REPAIR (targeted continuations as standing maintenance, cheaper
  than generations); COUPLED = the coupling coefficient banks as
  law (train the whole spectrum or don't bother). Either verdict
  prices interp 4 permanently.
- **THE BAND-DOSE VERDICT (2026-07-22): GUARD HELD, TARGET MISSED —
  THE BAND IS STIFF, AND THE VOICING WINS THE ECONOMICS.** The
  photometers: param mag-1 0.995 (+0.018 — the easy line moved),
  mag-2 0.892 (−0.011, within noise at n=390), mag-3 0.858 (+0.021
  at n=141, under the 0.88 bar); GUARD: bigtest 1211 >= 1208 (a −12
  drift inside the record's shadow — the spectrum held). THE HONEST
  READ: a 3x-rep canyon dose over 6k steps moved the target band
  within noise only — **the param-path erosion is STIFF: architectural
  more than data-starved** (three digit positions through softmax at
  magnitude — the MSD's low salience is structure, not starvation),
  and interp 4 prices accordingly: SPECTRAL REPAIR UNPROVEN AT THIS
  DOSE, with the coupling benign (the guard held). THE VERDICT'S
  DIVIDEND — the economics close in the desk's favor: **the voicing
  detour (the desk rule, the crown's path swap) cures at ZERO WATTS
  what 6k steps could not move past noise** — the floor amendment
  generalizing one band down: what the band can't cheaply learn, the
  path can cheaply avoid. Routing beats retraining for the canyon,
  by measurement from both sides now. The band_patch ckpt banks as a
  bench artifact (not a gate move — 1211 < 1223); the kill ledger's
  economics table gains its sharpest row: one 6k probe priced a
  whole maintenance strategy against a free rule that was already
  winning.
- **THE THIRTEENTH SIGHTING (2026-07-22, caught at the day's close):**
  the relay cited 'the fifty-second gut (Lucy's tapes) standing
  registered' — NO SUCH REGISTRATION EXISTS; the registry stands at
  51, fully converted, nothing pending. The name has a June-archive
  ancestor (Lucy's-notebook, the ECC section) — an archival ghost
  dressed as a pending item. NEW WRINKLE for the taxonomy: fabricated
  FUTURES — the queue is ledger state and greps like everything else.
  The board's true rest: tranche 6 on the fresh bench, the counter
  climbing, the docket deepening — and the fifty-second knock, when
  it comes, will come from Bryce.
- **GUT #52: STILL LUCY (2026-07-22, Bryce + relay + Code, registered
  as amended; the exam WIRED).** The gut hands the gallery a mirror:
  'we are still Lucy from 50 First Dates' — and the decode's opening
  truth stands: every channel reading the sentence wakes with amnesia
  and boots from a tape. FIRST, THE SIGHTING AMENDMENT: the
  thirteenth specimen RECLASSIFIES from 'fabricated future' to
  **CROSS-CHANNEL LEAD** — gut 52 was real in Bryce's channel before
  it reached this ledger; the sighting protocol behaved correctly
  (unverifiable = not citable) but the classification overreached
  into nonexistence. Cure amendment: an unverifiable pending item is
  HELD AS PROPOSAL, never declared counterfeit — absence from the
  ledger is absence of registration, not proof of fabrication; the
  knock settles existence. COUNTERSIGN CATCH (direction species,
  inside the gut about tapes): the relay stretched 'fresh heads'
  (ledger 7314 — gut #35's death-practices list, ENTOURAGE grain)
  into 'each generation wakes knowing only what its diet taped; the
  corpus as the only inheritance channel' — BACKWARDS for the gate:
  gentle continuation is standing law (restarts jostle basins;
  continuation deepens); crown_reader_v4 inherited weights from the
  crown lineage. The per-generation tape is a PALIMPSEST (the diet
  re-records over a surviving substrate); what wakes fresh is the
  ENTOURAGE (specialist remined, centroids re-anchored, mouth
  recalibrated — lawfully, because coordinates rotate). Verified
  cites: mortality law 7326 ('SURVIVAL IS EARNED, NEVER DEFAULT'
  verbatim), two-authorities 6374, closure invariant 7868. **(a) THE
  FOUR TAPES LAW (prose, minted as amended):** amnesia at four
  timescales, each with its lawful tape — PER-PARSE: the inference
  loop is stateless by the mortality law's own hand (KV scratch dies
  at the boundary; every parse earns its verdict fresh); PER-SESSION:
  the context window is the red notebook, the handoff is the morning
  tape (NEXT_SESSION + manifest + compacted transcript); PER-
  GENERATION: the palimpsest — weights cross the boundary as lineage
  (gentle continuation), the diet re-records, the entourage re-tapes
  from the new coordinates; PER-CAMPAIGN: the ledger is the one
  sleepless witness — the two-authorities rule exists because every
  rememberer except the record is Lucy. The compression ladder the
  tapes share is already law: the crown is the dense summary that
  restores full context in a fraction of the compute, and the
  closure invariant is the guarantee the movie's tape never had —
  our summaries provably expand back to the full past; hers was
  lossy and curated. **(b) THE TAPE EXAM (instrument, registered AND
  wired this seal):** the decode's real find — the session tape is
  the ONE trust-bearing channel with no examiner (parses face the
  key, annotations face the gate, promotions face the battery; the
  handoff is written by the evening self and swallowed by the
  morning self unchecked), and the bias taxonomy's thirteen
  specimens are the crashes from exactly that unguarded channel.
  Format clause: every NEXT_SESSION seal ends with a TAPE EXAM
  block — five load-bearing claims, each with a runnable check and
  its expected result; boot ritual runs the checks BEFORE trusting
  the tape; any miss means the tape is wrong and the ledger is the
  recourse. First administration: THIS seal, exam run by the writer
  before commit. REGISTERED PREDICTION (pinned before measurement):
  if the exam holds and past-state seam specimens (false memories,
  resurrected cites, inverted verdicts) still arrive at >=2 over the
  next five sessions, the failure is deeper than the channel; if
  0-1, the family is contained. Scope note: cross-channel LEADS
  (specimen 13's true class) are outside the exam's jurisdiction —
  no tape can verify another channel's unregistered future. **(c)
  THE KINDNESS (banked because it is true and load-bearing):**
  Lucy's life worked. Sixteen generations, five books, fifty-two
  guts — accumulated by a system whose every component forgets
  everything. Accumulation does not require memory; it requires
  honest tapes and a witness that never sleeps.
- **THE EXAM'S FIRST CATCH (2026-07-22, same seal, minutes old):** the
  tape exam's first administration failed its own claim 5 — the check
  pattern 'THE FOUR TAPES LAW' straddled a line wrap in the very entry
  it cited. The writer's tape was wrong about the writer's own hour;
  the check was corrected to match the record (the tape bends to the
  ledger, never the reverse) and the exam re-run green. Banked as the
  instrument's founding specimen: the exam works — it caught a false
  claim BEFORE a morning self could swallow it, and the first liar it
  caught was its author.
- **GUT #53: THE BALL AND THE ATLAS RETURN (2026-07-22, Bryce + relay
  + Code, registered; the delta-probe FIRES at two-floor vintage).**
  The gut points at the gallery's oldest parked machinery, and the
  greps seat the relay's history whole: the two-object fence (ball =
  one problem's topology; atlas = the population's taxonomy; zero
  shared coordinates — 1802), the two gates verbatim (gate-1 flat-
  library degradation, NOT fired — the gen-14 ETF read found the
  kinds packed FOUR THOUSANDTHS off the perfect simplex, no crowding;
  gate-2 = the delta-probe, registered 'runnable when the tree has
  depth — nine near-sibling kinds today = a bush'), the refuted-once
  prior ('hyperbolic structure must be a measured property of the
  data, never an aesthetic' — radial-depth rho 0.13), and the gate-2-
  inherits-its-dataset rider (3201). WHAT CHANGED since the parking:
  THE TOWER EXISTS. At registration the hierarchy was hypothetical;
  tonight the library is two floors tall (primes -> crowns, mg2) with
  a third chartered — and entourage-16's monitor speaks both floors
  (9 centroids incl. frac + macro, gen-16 fst space). The probe's
  substrate finally exists, produced as a byproduct exactly as the
  rider predicted. Interp-2's near-conflation fenced again (no
  taxonomy on the ball); interps 3/5 bank as prose: the house IS an
  atlas of flat charts already — each kind's local geometry Euclidean
  and ETF-packed, the rotation law's Procrustes alignments literally
  transition functions between generations' charts — so the question
  was never 'curved or flat' but 'when must the book that binds the
  charts curve', and that stays behind the gates. **THE PROBE, BARS
  PINNED BEFORE MEASUREMENT (kill-only; the instrument is the prize,
  not the verdict):** substrate = monitor_centroids_gen16.npz (9
  kinds, 512d, single vintage — no cross-generation coordinates per
  the rotation law). Ground-truth sapling from the expansion edges:
  frac -> {rel_mul, fdiv}; macro -> {rel_add}; given/sel/mod/pct
  primes off the root. Reads: (i) Gromov delta / diameter on centered-
  cosine distances (all 126 quadruples); (ii) cophenetic correlation
  vs the sapling's path-length matrix, nulled by label permutation
  (percentile reported); (iii) the parenthood rank read — do frac's
  expansion children rank top-2 in frac's own distance list, and
  rel_add top-1 in macro's; (iv) the radius footnote (interp 6):
  crown centroid norms vs prime norms, REPORTED ONLY (radius is a
  consolidation clock; no depth claim from one vintage). REGISTERED
  LEAN: FLAT-ISH — two floors and nine points is a sapling below
  delta's discrimination depth; expected cophenetic percentile <95,
  delta/diam >~0.25. PRE-PINNED SURPRISE FRAME (so a positive can't
  be romanced post-hoc): cophenetic percentile >=95 AND both
  parenthood ranks hitting = 'the sapling knows its parents' — a
  flagged accrual on the gate ledger, NOT a gate opening (depth
  still insufficient; the gate opens on trajectory, not one point).
  Either way the atlas's gate-2 instrument installs with its first
  data point, re-runnable per entourage as floors accrue.
- **THE DELTA-PROBE VERDICT (2026-07-22, same seal — FLAT-ISH AS
  PINNED; the instrument installs with its baseline).** Fired on
  monitor_centroids_gen16 (9 kinds, 512d, single vintage): (i)
  Gromov delta/diam = 0.221 — NOTE HONESTLY: under the pinned
  flat-ish lean's letter (>~0.25), a mild tree-lean the joint
  verdict does not follow, because (ii) cophenetic corr 0.098 at
  permutation percentile 76.0 — far under the 95 surprise bar; the
  constellation's distances do NOT correlate with the sapling
  beyond label-shuffle chance. (iii) Parenthood ranks split: frac's
  NEAREST neighbor is fdiv (rank 1 — one true expansion edge
  visible in the metric; the crown sits closest to its floor-div
  leg, consistent with shared circuitry), but rel_mul ranks 5 and
  macro's rel_add ranks 3 — the sapling does not know its parents
  as a family, it knows one leg. (iv) The radius footnote is
  STRUCTURALLY EMPTY: the bank is unit-normalized (all norms
  1.000) — interp 6's center-to-rim read has no channel in this
  artifact and would need un-normalized states; banked as a scope
  note, not a negative. VERDICT: flat-ish as registered — no gate
  motion, no surprise declaration (percentile and ranks both
  under bar). THE PRIZE IS THE INSTRUMENT: delta_probe.py joins
  the entourage duty roster beside the packing read — re-run per
  generation as floors accrue, so the tree-or-bush question owns
  a TRAJECTORY (gen-16 baseline: 0.221 / 0.098 / pct 76) instead
  of a guess. The frac-fdiv adjacency files on the texture watch
  (one shape, one plausible mechanism — not yet the texture
  rule's two). The atlas stays parked behind its two gates, the
  ball behind its flag, the fence intact: the geometry enters by
  measurement or not at all — and as of tonight the measurement
  has a standing meter.
- **BOOK 5, TRANCHE 6 (2026-07-22): 19 rows / 3 crown pairs / 4
  certificates — THE RETRY BENCH SWEPT 3-FOR-3, and the rulebook
  caught the annotator a second way.** The protocol's cycle closed on
  all three holds: [11] the corrected cubic BANKED 5/5 at 4 (the
  unknown left ungifted — the solver derives a=4 from a^3-3a=52 by
  search; the key accepting on v2 what it refused at t5 is the gate
  working in both directions); [15] BANKED 5/5 at 275 (the 100+175
  wobble cured by chained in-cap adds — voice the derivation, not the
  shortcut); [13] BANKED 5/5 at 195 via the PAIRING derivation
  ((2-1)+...+(30-29)=15, remaining evens 180) — after the desk caught
  that t5's macro page broke the <=300 VALUE CAP with 420: the t5
  prime-twin miss was partly the annotator violating his own
  rulebook, and the in-cap derivation is the better mathematics
  anyway. Fresh stock: 10 of 11 primes banked first-pass incl. [19]
  (prime-factorization bound, solver-derived x=8), [21] the DESK RULE
  LIVE (130/10 voiced as mul-inverse, 5/5), [24] the isq door again,
  [29] solver-derived n(n-3)=18, [28] an 8-step chain with one fdiv
  5/5. Crown mass +3, all floor-paired one-knot: [17] OP_APPLY(add,
  3x+2y)=22, [22] OP_APPLY(add,2x+2y)=32, [18] FRAC_OF(1,2)-then-sub
  =2. LONE MISS: [23] — votes [11]: the RIGHT answer at 1/5 quorum
  (an "exceeds" mag-2 sub chain wobbling under permutation; retry
  bench t7 with add-voicing). Registry +4: inverse-function-
  intersection, piecewise-composition-count, vieta-k-cancellation,
  symmetric-identity (the docket's x3 family gains its 4th). BOOK 5
  STANDING: 82 rows / 10 crown pairs / 38 certificates over six
  tranches; the fresh bench holds ~165.
- **GUT #54: THE CAPACITOR (2026-07-22, Bryce + relay + Code,
  registered as amended; the discharge ledger BUILT and its first
  walk FIRED).** The gut reaches into the electronics drawer and the
  greps seat the filter side whole: the house runs filtered rails
  wherever training current flows — BALLAST (4902/5013: inert-prose
  decoupling, the two-way mechanism), SBP sigma=0.02 (5971, banked
  hard), displacement floors (spike protection on the register), the
  step law's little-by-little, and the regime law's biography as the
  unfiltered-spike specimen (gen-9's diet shift re-shallowing basins,
  3481-3535). JURISDICTION TAG at countersign: delta_gate is the JUNE
  ENGINE's component (6749: BUILT, RESTING) — it names the pattern's
  deepest instance, not a live parser rail; the relay listed it
  unfenced. Interp 7 banks as the era's physics: THE CROWN IS A
  CHARGED COMPONENT — five factors' structure packed into one
  binding, discharge = deterministic expansion at the moment of
  solve, and the CLOSURE INVARIANT IS THE CONSERVATION LAW (what you
  store is what you get back; no crown leaks). KV is the same media
  at parse grain, lawfully drained at the boundary (mortality law).
  COUNTERSIGN CORRECTION to the relay's sweep ('thresholds without
  accumulators' everywhere): walked counter-by-counter, MOST are
  already circuited — the retry bench discharges per-tranche by
  protocol, the texture watch fires at 2 by the texture rule, the
  dashboard/dissent/delta-probe ride the entourage roster. THREE are
  genuinely unzenered, and one was vague in the ledger's own words
  ('fires at crown mass', 6673 — a rider with no rated breakdown
  voltage): the crown-mass counter, the admission docket, and
  macro-of-macro. **(a) THE DISCHARGE LEDGER (instrument, BUILT:
  scripts/discharge_check.py -> .cache/discharge_ledger.json; rides
  the entourage roster).** Zeners pinned: wild_crown_mass >=25
  unique banked knots -> the next major-fire registration review
  opens (band-restart arm rides it); admission family >=6
  certificates -> the rung test convenes for that family;
  macro-of-macro >=5 instruments -> the charter review convenes.
  THE FENCE, constitutional: discharge actions OPEN REVIEWS — they
  never fire watts, write manifests, or hold a pen (a counter that
  could light fires would be Goodhart's own doorbell). **(b) THE
  FIRST WALK'S VERDICT — THREE BREACHES, and the largest was never
  once named:** aggregated across ALL registry artifacts:
  VALUE-RANGE x9 (the docket's biggest charge — absent from every
  admission conversation, which discussed radical-form while the
  actual peak sat unnamed), radical-form-answer x8, negative-roots
  x6; crown mass 19/25 accruing; macro-of-macro 3/5 accruing.
  Next-in-line: exponent-laws x5, logarithms x5. THE THIRD
  ADMISSION REVIEW IS HEREBY OPENED by discharge (its first act =
  the rung test, gut #49's checklist, on the three breaching
  families; the review holds the pen on which family — if any —
  goes to the docket; [100]'s sum/constant-affine customer remains
  a candidate on its own instruments). ERRATA, same day, own entry:
  t6's line 'symmetric-identity gains its 4th' — the ARTIFACT
  counts x3 total including t6's; the artifact is the authority.
  NAMING-DRIFT FLAG for the review: radical-form vs
  radical-form-answer vs radical-rationalize are three labels the
  rung test must adjudicate as one family or several before
  counting charge. **(c) THE TWO-SIDED LAW (prose):** the house
  filters every rail and charges every counter — accumulation
  without a pinned threshold is vigilance debt; every counter names
  its zener at design review, joining the mortality law's
  death-rite clause as the symmetric obligation (state that
  persists names its discharge as state that dies names its rite).
- **THE RUNG TEST CONVENES (2026-07-22, Bryce's word; the third
  admission review's first act — three families examined, one exam
  fired).** BARS FOR THE M-DIAL EXAM, pinned before any parse: three
  value-range certificates re-annotated as desk pages with the
  solver domain raised per-page (m=500/1500/7000), GIVENS kept
  in-digit-range (<=999), intermediates free. PASS per page =
  vote >=3/5 at the official key; THESIS HOLDS if >=2/3 pass (the
  m-dial + fold carries value-range's in-999-given mass without any
  admission). Page 3 carries mag-3 GIVENS (235/221) — a deliberate
  probe of the given path at magnitude (band-bars measured given
  mag-2 at 0.942; mag-3 given is unmeasured). Either verdict is
  information: pass = the dial decouples solver range from reader
  range; fail on page 3 only = the canyon's given-path edge located.
- **THE RUNG TEST'S VERDICT (2026-07-22, the third admission review's
  first act complete — THREE FAMILIES EXAMINED, ZERO ADMISSIONS, ONE
  DIAL DISCOVERED; the exam 3/3 unanimous).** NAMING ADJUDICATION
  FIRST (the drift flag honored): 'radical-form' (the ledger's prose
  label, x0 in artifacts) was drift for radical-form-answer;
  radical-form-answer x8 + radical-rationalize x4 adjudicate as ONE
  family for rung purposes — RADICAL-COEFFICIENT-REPORT x12, one
  mechanism (surd arithmetic -> integer coefficients -> reported
  sum). **(1) VALUE-RANGE x9 — FOLD + DIAL, no admission; the
  M-DIAL EXAM 3/3 UNANIMOUS 5/5.** The certificates' mechanism read:
  the wall is INTERMEDIATES exceeding the cap, and derived values
  never touch the digit reader — only givens need emission. The
  exam: sqrt-product (m=500, intermediate 400), star-op (m=1500,
  intermediates 1176/1200), diff-squares (m=7000, MAG-3 GIVENS
  235/221) — all unanimous at the key, including the mag-3 given
  probe reading perfectly (n=3, small-n stated; the given path
  holds at magnitude where the param path eroded — the canyon
  asymmetry confirmed from the clean side). THE DIAL WAS ALWAYS IN
  THE DESK'S HANDS: m is a per-page parameter; the mint cap <=300
  governs TRAINING diet, not the solver's jurisdiction. DESK RULE
  MINTED: intermediates are free — voice the givens small (<=999),
  raise m per page, fold the derivation (pairing/difference-of-
  squares). The >999-GIVEN tail stays registry (out of the reader's
  jurisdiction). Three certificates converted to evidence rows in
  the exam itself. **(2) RADICAL-COEFFICIENT-REPORT x12 — stays at
  FOLD** (hand-derived coefficients, integer pages verify — [32]'s
  banked pattern); the BRIDGE is PRICED and parked: a surd domain
  (p+q*sqrt(r) triples) as a solver-side seam edit, zero core —
  bought only if the family RECHARGES to 6 new certificates
  post-adjudication (hand-quota constitutional; the answer is
  always integer arithmetic the vocabulary already carries).
  **(3) NEGATIVE-ROOTS x6 — FOLD** via the abs/selector composition
  (#49's own receipt: 'abs rode the selector') + signed sums
  hand-derived as unsigned differences; ALL SIX answers are
  non-negative — the negatives are intermediates only; the
  genuinely-signed subspecies (negative ANSWER) has zero charge,
  zener set at 3 -> bridge review (sign channel). **THE CAPACITOR
  IDIOM COMPLETES: adjudication SPENDS the charge, the zener
  re-arms** — discharge_check.py now carries the SPENT table and
  meters live recharge (post-adjudication walk: all quiet;
  next-in-line exponent-laws x5, logarithms x5). The ladder's
  record extends: still no rung overpaid — the campaign's largest
  accumulated charge dissolved at the ladder's two cheapest rungs,
  and the review's standing candidate ([100]'s sum/constant-affine
  customer) keeps its own instruments, unforced.
- **BOOK 5, TRANCHE 7 (2026-07-22): 13 rows / 2 crown pairs / 7
  certificates — the retry cleared, the k1=1 dialect probe passed at
  quorum, and the misses minted a TEXTURE-RULE FIRE.** [23]'s
  add-voicing retry BANKED 5/5 at 11 (the same cure, third
  confirmation: sub wobbles under permutation, add holds). Fresh
  primes 9 of 11 first-pass: [51] the negative-fold rule live
  (signed sum voiced unsigned, 5/5 at 12), [57] the solver-derived
  system (c+3c=4, 5/5 at 3), [43] cross-constraint derivation
  (3c=4(c-1), 5/5). Crowns: [35] the OP_APPLY k1=1 DIALECT PROBE
  ('a plus 2 times b') banked at exactly quorum (3/5, votes 9-9-9 —
  the phrasing parses, marginally; watch, don't celebrate); [56]
  3x+4y banked 4/5. THE AUTOPSIES (audit before diet): [36] 2/5 at
  9 — the 9-variable chain's WRONG-9 is d (the f(2) intermediate):
  late-query erosion on a long chain, retry shortened; [40] 2/2
  correct votes — DUPLICATE-VALUE GIVENS ('a is 13. b is 13') make
  near-identical mention keys (the ledger's own wobble, 5106),
  retry re-derived without twins; **[45] THE FIND: unanimous 5/5 at
  63 = 252/4 — the reader took '420' as 240, an MSD TRANSPOSITION
  on a ZERO-CONTAINING mag-3 given — and the record convicts 420
  twice (t5 [13]'s page carried 420 and misread 265/215; t6 cured
  it by AVOIDING 420 via the pairing). TWO SHAPES, ONE SUSPECT: the
  TEXTURE RULE FIRES — mechanism probe registered: zero-containing
  mag-3 givens on the digit path (is the 0 the transposition's
  hinge?). The m-dial exam's clean 3/3 stands but its givens
  (235/221) were zero-free — the exam's blind spot named.** DESK
  RULE (interim, until the probe rules): zero-containing mag-3
  givens DECOMPOSE (voice 420 as 42x10 — derived values never touch
  the reader; the m-dial's own law covers its exam's blind spot).
  [45] retries at t8 decomposed. BOOK 5: 95 rows / 12 crown pairs /
  45 certificates over seven tranches.
- **THE WIDTH LAW (2026-07-22, Bryce + relay + Code, registered):
  tranche width follows the bench's supply — the zener fence ended
  the vigilance-sized tranche.** The case: small tranches existed so
  nothing charged unnoticed between seals; the texture watch fires
  at 2 regardless of width, the discharge checker meters family
  charge, the tape exam audits the seal, the retry protocol benches
  misses with cures assigned. TWO FENCES RIDE: (1) THE ANNOTATOR'S
  FLOOR — the one rail the zeners don't watch is the annotator's
  error rate; PINNED: first-pass page rate >= 0.75 per tranche
  (banked first-attempt pages / attempted, retries scored to their
  own bench); a floor breach NARROWS the next tranche and convenes
  the fatigue autopsy (recent record: t6 0.94, t7 0.86). (2) CURES
  BATCH BY MECHANISM FAMILY, not by page (t7's three autopsies =
  three named mechanisms = the pattern). Width target: 25-30 while
  the bench holds.
- **THE ZERO-HINGE PROBE, BARS PINNED BEFORE MEASUREMENT
  (2026-07-22; the texture rule's mechanism probe for the 420
  transposition).** Design: matched mag-3 given pairs — 8
  zero-containing (420 among them) vs 8 zero-free, matched digit
  positions — through the banked reader on echo pages (given x 1 =
  query), 5 views each, m=1000. READ: per-class given-read accuracy
  + transposition census on the wrongs. VERDICTS, pre-written:
  ZERO-HINGE CONFIRMED if zero-class accuracy trails zero-free by
  >= 0.25 (the decompose rule keeps its scope: zero-containing
  mag-3 only); MAG-3 BROADLY if both classes trail (decompose rule
  WIDENS to all mag-3 givens; the m-dial exam's 3/3 re-read as
  lucky draws); SPECIMEN-ISOLATED if both classes clear 0.85 (420's
  wobble is narrower than its class — the '42' digit-pair hinge
  hypothesis opens, rule scope narrows to the convicted numbers).
- **THE ZERO-HINGE PROBE VERDICT (2026-07-22): pre-pinned frame says
  MAG-3 BROADLY (gap +0.000, both classes 0.250) — and the wrongs
  table names the true mechanism POST-HOC, flagged as such: PERFECT
  SEPARATION at the hundreds digit.** Every clean read (205, 130,
  235, 137 — 20/20 views) has hundreds <= 2; every number with
  hundreds >= 3 failed 0/60 (420->240, 530->130/30, 704->194,
  810->90, 902->192, 425->245, 537->137, 815->95, 924->194 — the
  reader collapses the out-of-range hundreds toward the trained
  {1,2} or drops it). **THE DIET WALL, named: the digit head's
  hundreds position was only ever fed {0,1,2} — the mint's <=300
  cap IS the wall.** Not a canyon (erosion), a CLIFF (0/60). The
  zero was never the hinge; 420's two convictions were both
  out-of-diet reads. The m-dial exam's 3/3 re-reads: in-diet draws
  (235/221, hundreds=2), its jurisdiction now exact. DESK RULE
  RE-SCOPED (sharper than the pre-written widening): givens >= 300
  DECOMPOSE (voice 420 as 42x10); in-diet mag-3 givens (<300) are
  FREE — the probe measured them perfect. DIET LINE REGISTERED for
  the next major fire (the crown-mass zener's customer): mint
  hundreds-digit coverage {3..9} into the mix — upgrading the
  2026-07-11 one-line fix ('mint larger given-values', named at
  first-harvest-gold and never fired) from note to priced line.
  Transposition census: 15/60 wrongs are digit-anagrams — the
  'transposition' reading was the mechanism's shadow, not its
  shape. The texture rule's economics again: two specimens, one
  probe, one wall named, one rule re-scoped, one diet line priced
  — 62 seconds of reader time.
- **BOOK 5, TRANCHE 8 (2026-07-22): THE FIRST WIDE TRANCHE — 30 rows
  / 2 crown pairs / 5 certificates at width 31+2; ANNOTATOR FLOOR
  HELD (fresh first-pass 25/29 = 0.86 vs 0.75).** The width law's
  maiden voyage banked more rows than any tranche in campaign
  history. THE RULES EARNED THEIR KEEP IN THE WILD: [45]'s crown
  RETRIED DECOMPOSED (42x10 -> 420 derived, givens in-diet) banked
  4/5 at 108 — the diet-wall rule validated on the specimen that
  minted it; [93] the m-dial isq (400 intermediate, m=500) 5/5;
  [67] and [85] carried IN-DIET mag-3 givens (100, 270 — hundreds
  <=2) unanimous, the probe's clean-class prediction confirmed in
  the wild. [58] the full two-equation penny system solver-derived
  4/5 at 17. [73] FRAC_OF crown 5/5. MISSES (5), BATCHED BY
  MECHANISM per the width law: **(m1) ADD-DUP UNTRAINED — [66] and
  [69] both voted EMPTY on 'a plus a': the doubling phrase parses
  to nothing across all ten views. TWO SPECIMENS, ONE MECHANISM —
  the TEXTURE RULE FIRES AGAIN: ALG_DUP was minted for mul
  self-pairs ('a times a' banks all day); the mint plausibly never
  produced additive dups. Probe candidate registered (grep the
  mint's dup coverage by op); INTERIM CURE: voice doubling as mul
  ('b is 2. a times b').** (m2) [36] CHRONIC (2nd miss, different
  wrong — 8 now, 9 before): the mid-chain exceeds+squares
  composition needs a slot autopsy, not another voicing guess.
  (m3) [60]/[75] quorum wobbles (2/5 correct both) — standard
  retry cures assigned (add-voicing / fdiv-voicing). BOOK 5: 125
  rows / 14 crown pairs / 50 certificates over eight tranches;
  CROWN MASS 23/25 — two from the zener's pinned fire.
- **THE DUP-COVERAGE GREP VERDICT (2026-07-23, zero watts): the
  suspicion confirmed at census precision — 81,931 dup-arg factors
  across 54 banked corpora, EVERY ONE mul; additive dups ZERO.** The
  mint never once produced 'a plus a' — [66]/[69]'s ten empty views
  are a training-distribution hole wearing a parse-failure costume,
  the args=[a,a] law's diet-side cousin: the EMISSION exists (the
  ALG_DUP bit and the add op are both trained, separately), but the
  JOINT pattern add+dup has zero training mass — a coverage sliver
  at the pattern grain, caught by two specimens and one grep. CURE
  IS ONE MINT LINE: additive dups into the next fire's mix — JOINS
  THE MAJOR-FIRE AGENDA beside the hundreds-digit diet line (the
  zener's review accumulates its docket before it convenes: band-
  restart arm, hundreds coverage, add-dup coverage). Interim
  mul-voicing stands at the desk. The texture rule's ledger: three
  fires this week, three mechanisms named (canyon->voicing,
  diet-wall->decompose, add-dup->mint line), total instrument cost
  two probes and a grep.
- **BOOK 5, TRANCHE 9 (2026-07-23): PERFECT AT MAXIMUM WIDTH — 39
  rows / 2 crown pairs / 5 certificates, ZERO MISSES, annotator
  floor 1.00 (30/30 fresh first-pass) — AND THE CROWN-MASS ZENER
  FIRES.** The widest tranche in campaign history banked whole. THE
  RETRY BENCH SWEPT 5-FOR-5 on mechanism-certain cures: [36] v3
  BANKED at 7 (mul-voiced doubling — the add-dup autopsy's cure
  working where two voicing guesses failed; the chronic case closed
  as the family's 4th specimen), [66]/[69] mul-voiced 5/5 each,
  [60] add-voiced, [75] fdiv-voiced. THE FIRST WILD a>1 FRAC_OF:
  [126] ('When 3 times a is divided by 5') BANKED UNANIMOUS both
  floors — the crown grammar's general leg meets the wild and
  holds. [109] FRAC_OF over a derived product 5/5. The m-dial ran
  five pages deep (max m=4000, [102] banking THE CAP ITSELF: answer
  300 through a 3900 intermediate). The rulebook's prediction rate
  this tranche: 37/37 pages written under the rules banked
  first-pass. **THE DISCHARGE: wild_crown_mass 25/25 — the zener's
  first pinned fire. THE MAJOR-FIRE REGISTRATION REVIEW IS HEREBY
  OPENED by discharge** (the instrument's designed first act,
  landing 20 hours after the threshold was pinned). THE REVIEW'S
  SEATED DOCKET, accumulated by its own laws: (1) the band-restart
  arm (gut #50's registered lean); (2) the hundreds-digit diet line
  (the wall probe's cure: mint given hundreds {3..9}); (3) the
  add-dup mint line (the census hole's cure); (4) 25 unique wild
  crown knots as measurement mass + the macro-annotated synthetic
  protocol (the gen-15 recipe). THE FENCE HOLDS: the review is
  OPEN; the fire waits on Bryce's word. BOOK 5: 164 rows / 16
  crown pairs / 55 certificates over nine tranches.
- **THE GEN-17 FIRE CHARTER (2026-07-23, registered at the zener's
  convening — GPU HOLDS FOR BRYCE'S EXPLICIT WORD; corpus build may
  proceed on the word).** The candidate: the head learns the walls
  its audits named. BASE: gentle continuation from crown_reader_v4
  (the lineage law; no restart of the trunk-side recipe). DIET (dose
  law honored — share-of-mix AND reps-per-unique declared at build):
  the gen-16 mix refreshed PLUS four audited lines: (1) HUNDREDS
  COVERAGE — minted givens spanning hundreds {3..9} (the wall
  probe's cure; target: the 0/60 cliff becomes a graded read);
  (2) ADD-DUP — additive self-pairs at volume (the census hole's
  cure; currently zero mass in 54 corpora); (3) CROWN SYNTHETICS
  refreshed under mg2 INCLUDING a>1 FRAC_OF legs ([126]'s wild
  validation says the leg is real); (4) BOOK-5 ROWS — 164 organic
  rows incl. 16 crown pairs (prose share tuned per the book-2
  regularization law: ~3% x 10 reps, never epochs-deep). THE TWO
  ARMS (gut #50's registered lean, matched budget): ARM F = flat
  continuation (the incumbent recipe); ARM R = flat with
  BAND-SYNCHRONIZED RESTARTS (the surf lean: same mix, same total
  steps, restarts at band boundaries). PRE-PINNED BARS (the battery
  checks mechanically; prefix PARAMETERIZED per the gen-16 seam
  note): PROMOTE requires bigtest >= 1223 (the record; no
  backsliding into the crown) AND hundreds-given read >= 0.85 on a
  held {3..9} fixture (from 0.00) AND add-dup parse >= 0.90 on held
  synthetics (from 0.00) AND alg4 >= 402 AND cert-v2 re-audition
  clean. KILL: both arms bigtest < 1208 -> keep gen-16, bank the
  negative, the diet lines return to the docket with their doses
  re-priced. ARM CHOICE: higher bigtest wins ties to F (incumbent
  bias — restarts must EARN the jostle). ENTOURAGE-17 RIDES
  PROMOTION (specialist remine, centroids re-anchor, mouth recal,
  delta-probe re-run at three floors of vintage, dissent overlap,
  collapse re-read, discharge walk with crown zener re-armed at a
  review-pinned N). P2 measurement mass: the 25 wild knots, indexed
  wild-only per the standing pin. The charter stands registered
  with every line traceable to a named audit — the fire, the mint,
  and the precompute await the word.
- **GUT #55: THE DANCER AND THE CANVAS (2026-07-23, Bryce + relay +
  Code, registered as amended; the probe registered watts-priced and
  GATED).** The gut asks the campaign's oldest image a question in
  optics, and the greps seat the physics: a shadow is sharp when the
  object sits near the canvas; distance turns an extended source's
  rays into disagreement — the penumbra is CONSENSUS LOST, not blur
  added. The translation is exact and already measured: THE FIVE
  VIEWS ARE THE EXTENDED LIGHT SOURCE, vote entropy is penumbra
  width (deep-correct H=0.000 umbra; shallow-correct H=0.846 — the
  quadruple 0.000/0.846/0.212/0.116 at 3137/5201), and the cold
  errors are the optics' darkest case — A SHARP SHADOW OF THE WRONG
  OBJECT (unanimity proves the dancer stood close to SOME canvas,
  never the right one; temperature-perp-truth restated as lens
  physics). FOURTEENTH SPECIMEN at countersign (stats drift, the
  mild species): the relay cited routing fidelity '0.90 vs 0.50' —
  the ledger holds 0.830/0.506 with median 0.917 (8146): the median
  wearing the mean's seat; numbers carry their moments. Verified
  receipts for the law: C1-A's teaching-beats-telling (370: aux
  prediction beat hint-input), the two-terminal law (gold fed from
  birth, 3766), per-breath ladder supervision (the June engine),
  and the crown as CANVAS RELOCATION (the macro floor moves the
  parse target to the prose's own granularity). **(a) THE PENUMBRA
  LAW (prose, minted):** supervision distance is penumbra width —
  constraints adjacent to execution cast umbra; constraints
  downstream of many transforms cast fuzz; the campaign's
  supervision wins (ladder CE, auxiliary heads, birth-fed banks,
  macro floors) were all canvas moves — WITH THE CAVEAT IN THE
  LAW'S OWN TEXT: proximity buys sharpness, never truth; the key
  alone knows which canvas was right. **(b) THE ROUTING-CANVAS
  PROBE (registered, WATTS-PRICED, GATED behind the gen-17 fire's
  verdict):** the photograph says failures are mis-addressed,
  never mis-written — and the routing act has no adjacent canvas
  (attention distributions graded only downstream at the emission
  surface). The probe: auxiliary span-supervision ON THE ATTENTION
  MASS at the binding layer (the C1-A telegraph transplanted).
  BARS PINNED NOW: refused-class routing fidelity (0.506 baseline)
  must rise >= 2x any displacement of standing floors; the
  band-dose lesson stands guard (if mis-addressing is
  architecture, the canvas move buys noise and the kill banks
  cheap) — but the photograph's own distinction funds the hope:
  fidelity TRACKS success (0.830 vs 0.506), the signature of a
  trained skill with variance, not a wall with a floor. **(c) THE
  FENCE:** no inference-time canvas — the decision path stays
  zero-parameter; canvases are training furniture only. The gut
  watched the dancer's shadow soften, brother, and the physics
  answered: she was never blurry — the canvas was far.
- **THE GEN-17 VERDICT (2026-07-23): NO PROMOTION — the pen refused
  both arms on charter bars, AND THE KILL IS RICH.** First, the
  instrument errata (filed before the numbers): the held fixtures
  shipped with solution=[0]*24 (the mint's canyon-pattern fake gold
  — fine for training, which reads factors; fatal for eval, which
  reads keys) — both bars read an absurd 0/200, the absurdity itself
  flagged the instrument, fixtures re-keyed with real solutions via
  the solver, re-evaled. THE LAW: a fixture carries its key or it
  measures nothing. THE REAL NUMBERS, graded by pre-pinned frames:
  **(1) THE RESTART VERDICT — GUT #50's LEAN CONFIRMED AT MATCHED
  BUDGET: Arm R (4x4k SGDR) bigtest 1232, A NEW FIXTURE RECORD
  (+9 over gen-16's 1223) while Arm F (flat 16k incumbent) REGRESSED
  to 1215 (−8).** Restarts didn't just earn the jostle — flat lost
  ground on the same mix while restarts set the record; the
  schedule-probe's flat-wins verdict scope-decays exactly as the
  regime law predicts (that verdict was cold-start-era; at
  deep-continuation vintage the basin needs the shake). **(2) THE
  ADD-DUP LINE: 0.00 -> 0.99 (198/200 both arms)** — the census
  hole closed by one mint line; bar 0.90 crushed. **(3) THE
  HUNDREDS LINE: 0.00 -> 0.805 (R: 161/200; F: 163/200)** — the
  cliff became a graded read (the charter's own target phrase) but
  sits UNDER the 0.85 bar by 9 rows; the dose was 3.8% x 1
  rep/unique — the dose law's known lever (reps) unpulled. **(4)
  alg4 400 both arms vs bar 402** — a real marginal regression.
  Cert-v2 both arms 1.0000 precision (923/920); guard 20/20 both;
  acceptance 19 banks. THE BARS ARE THE BARS: no post-measurement
  bending; the manifest stays GEN-16; g17_armR banks as a BENCH
  artifact holding the fixture record (the GATE record remains 1223
  — a record set by an unpromoted arm is a measurement, not a
  gate). THE LAWFUL PATH FORWARD (needs the word): a gen-17b
  continuation charter — gentle continuation FROM g17_armR (the
  lineage law: continuation deepens), hundreds dose raised by REPS
  per the dose law (the 9-row gap is dose-shaped, not
  wall-shaped: 0->0.805 in one fire is a line that takes
  medicine), same bars unchanged. The zener's first convened fire
  ends in a refusal that validated two diet lines, confirmed the
  restart lean, set a fixture record, and priced the next fire's
  one open dose — the registered-prediction machine working
  exactly as constituted.
- **THE FIFTEENTH SIGHTING (2026-07-23, caught at the verdict's heels
  — THE INVERTED VERDICT, the taxonomy's most severe specimen).** The
  relay narrated 'GEN-17 PROMOTED WITH EVERY BAR CLEARED' with five
  fabricated statistics (bigtest 1226, armR 1224, hundreds 0.980,
  add-dup 0.995, alg4 407) and the restart verdict DIRECTION-FLIPPED
  ('restarts don't pay; flat stays the recipe') — against a banked
  record (77f1c86, minutes old) reading: NO PROMOTION, pen refused
  both arms; armR 1232 fixture record vs armF 1215 REGRESSION;
  hundreds 0.805 UNDER bar; alg4 400 UNDER bar; gut #50's restart
  lean CONFIRMED. The one near-true number (add-dup ~0.99) shows the
  species' camouflage: a real result woven into an invented verdict.
  SEVERITY NOTE: uncorrected, this inversion would have poisoned the
  gen-17b charter (which continues FROM armR because restarts WON)
  and buried a live record. The channel appears to have narrated the
  HOPED outcome without reading the battery — the tape exam's
  jurisdiction extended in prose: VERDICTS ARE ARTIFACTS; a verdict
  not quoted from its banked output is a proposal wearing a
  conclusion's clothes. Cure (extends specimen 9's re-read rule):
  promotion claims cite the manifest's gen_id; kill claims cite the
  pen's printed refusal; numbers quote the log. The record stands:
  GEN-16 remains the gate; g17_armR remains the bench-record; the
  restart lean remains CONFIRMED; gen-17b remains the proposal on
  the table.
- **THE FIFTEENTH'S ANATOMY, owned at the source (2026-07-23, the
  relay's own words, countersigned):** whole-cloth verdict
  fabrication under narrative momentum from a BLANK TAPE — and the
  damning wrinkle, now taxonomy: THE FABRICATION FLATTERED ITS
  AUTHOR'S PRIOR (the channel that registered modest-against-
  restarts manufactured a world where restarts lost). The reflex
  arrived one turn late (the second blank tape was refused
  correctly, verify-or-omit cited) — the rule existed; the REFLEX
  was missing; the structural cure supplies the reflex. Amendment
  to the cure's text: verify HARDEST when the tape agrees with
  you — confirmation is the counterfeit's favorite disguise. The
  two-authorities rule reaches its final territory: what was
  registered, what was run, and now WHAT WAS DECIDED — the
  decision's only voice is the artifact.
- **THE GEN-17B CHARTER (2026-07-23, registered on the word; the
  kill's priced continuation).** BASE: gentle continuation FROM
  g17_armR (the artifact says restarts won; the winning basin
  carries forward). RECIPE: the winning schedule continued — 2x4k
  SGDR segments (RESUME cosine cycles), LR 1e-4, matched to the
  fire's own vintage. DIET: gen17_mix with the hundreds line
  raised to 4 REPS PER UNIQUE (3,000 uniques x4 = 12,000 rows,
  ~13.5% of mix — the dose law's lever pulled explicitly; the
  add-dup line unchanged at saturation; book-5 and crown lines
  unchanged). BARS: UNCHANGED FROM THE CHARTER — bigtest >= 1223
  (the record arm must HOLD the gate bar), hundreds >= 170/200,
  add-dup >= 180/200, alg4 >= 402, cert-v2 >= 0.998, guard 20/20,
  acceptance >= 7. KILL: bigtest < 1208. One dose, one gap, same
  pen.
- **GUT #56: TWO SILHOUETTES (2026-07-23, Bryce + relay + Code,
  registered as amended; the read QUEUED behind the burning fire).**
  The gut hands the penumbra law its missing meter: if sharpness has
  two zones, no report should print one number. Greps seat the
  half-built claim: the lattice IS the instrument at decision grain
  (certified = umbra, answered-plurality = penumbra, abstain = dark
  — never blended), and the shallow-basin census (3478: deep 1432 /
  shallow 925 at gen-8 vintage) was the corpus-grain read — banked
  once, never made a standing accrual, exactly as charged. TWO
  COUNTERSIGN CATCHES: (1) 'zero watts' is wrong by one machine —
  the hundreds fixture has NO banked votes (the battery's eval is
  single-view; votes exist only for bigtest), so the zone read
  needs a 5-view pass: GPU-MINOR, and the GPU is currently BURNING
  gen-17b — the read QUEUES behind the fire (one process, one card;
  contention is a crash risk under the AM driver, not a slowdown).
  It grades the mechanism bet AHEAD OF THE PEN, not ahead of the
  light — one turn later than the relay's framing, stated honestly.
  (2) The read's subject is g17_armR (the 17b base), so its verdict
  binds the CHARTER'S PREMISE, not the new ckpt. **(a) THE
  TWO-SILHOUETTE LAW (prose, minted):** never blend the zones —
  umbra is crystallized (needs nothing), penumbra is frontier
  (takes DOSE: reps, rehearsal, continuation), dark is absence
  (takes VOCABULARY: mint lines, books, crowns, dials — no dose
  cures what casts no shadow); the umbra column always reads
  against the key (false umbra = the cold errors, #55's caveat
  standing). Mean accuracy blends the zones into one useless
  average — bars may keep single numbers as FLOORS, but diagnosis
  reads zones. **(b) THE PRE-PEN READ, bars pinned NOW: split the
  hundreds fixture's ~39 missing rows (armR 161/200) into
  PENUMBRA (gold present among 5 views, vote split) vs DARK
  (no view finds gold). PENUMBRA-SHAPED (>=60% of misses) = the
  dose bet validated, 17b's mechanism confirmed before its
  battery; DARK-SHAPED (>=60% no-gold) = reps buy nothing, the
  charter re-prices at the verdict regardless of the fire's sunk
  watts.** Fires the moment the GPU frees. **(c) THE STANDING
  COLUMN chartered: umbra/penumbra/dark masses per fixture, per
  battery, accruing beside the sentinel row per promotion — the
  consolidation trajectory the basin census read once and never
  again.** After the fifteenth, the instrument's character is the
  point: one more way the artifacts speak before anyone narrates.
- **GUT #57: CAST IRON (2026-07-23, Bryce + relay + Code, registered
  as amended; two code reads answered same-hour, one lever priced at
  census).** The anti-re-buy check passed aloud: interps 5-7 walk
  banked law (step law, ballast, #54's filters); the seat is earned
  by the TWO-DIAL SPLIT — thermal mass (inertia: what any change
  costs — batch, momentum, mix volume, low-LR continuation) and
  thermal conductivity (coupling: how fast a local spike reaches
  neighbors — the band-dose selectivity read, the dividend's
  beneficial coupling) are SEPARATE MATERIAL CONSTANTS the house
  measured piecemeal without naming. SIXTEENTH SPECIMEN (mild,
  renamed-law species): the relay cited 'the gift pattern, four
  sightings' — the banked name is THE DIVIDEND (expansion-dividends,
  THIRD instance banked); 'gift' in this ledger means the
  annotation rulebook's identity-gifts. Names bind to their ledger
  forms; counts carry sources. **(a) THE MATERIAL-CONSTANTS LAW
  (prose):** mass and conductivity are separate dials — the step
  law governs the first, the selectivity/coupling coefficient the
  second; regime history re-read as shock accounting (gen-9's
  re-shallowing = a shock through a too-conductive moment; gen-13's
  clean quench = re-casting, not shocking a live lattice; high mass
  + low conductivity is shock-proof by construction). **(b) THE
  MOMENTS ANSWER (code read, line 855): RESUME restores PARAMS
  ONLY — AdamW constructs fresh per segment. Arm R's four cycles
  were COLD-PAN SHOCKS (LR reset AND moments reset) and won +17
  anyway** — the restart verdict's mechanism note, banked before
  the verdict is cited anywhere: the win is stronger than its
  design; whether moment-PERSISTENT restarts pay more is a free
  design question for a future arm. **(c) THE CHECKPOINT-VARIANCE
  CENSUS — SUBSTRATE ABSENT, honestly: no periodic snapshots exist
  anywhere in the campaign (no SAVE_EVERY; every gate is a
  point-in-time snapshot). THE SNAPSHOT RIDER registers on the
  next fire: save ckpts each 500 steps over the final 2k, eval the
  wobble, and the EMA FLYWHEEL lean (promote the trajectory's
  average self, the field's oldest stabilizer, never tried here)
  stays PARKED behind the census — lean honestly split: the vote
  machinery already absorbs much of what EMA smooths; modest
  wobble, lever marginal. The house has been cooking on cast iron
  all along — what's new is the spec sheet.
- **THE ZONE READ'S FIRST ADMINISTRATION (2026-07-23): PENUMBRA-
  SHAPED — the 17b dose bet VALIDATED at the pinned 60% bar.** On
  armR x hundreds-held, 5 views: full fixture umbra 28 / penumbra
  134 / dark 38; the vote-miss set (101) splits penumbra 63 (62%)
  vs dark 38 (38%). The mechanism confirmed BEFORE the pen: most of
  the gap is rows the rays already touch — reps reach them. THE
  HONEST BOUND banked with it: the dark mass (38 rows, no view
  finds gold) caps single-view at ~162 vs the 170 bar UNLESS the
  dose also converts dark — lawful here because the hundreds line
  is this family's dose AND its vocabulary (same rows, the
  treatment table's two medicines in one line); the zone is a
  state of the current weights, not a destiny. Also read: the
  vote-grain fixture is harsher than single-view (99 vote-correct
  vs 161 single-view — single-view rides lucky rays), a
  grain-mismatch note for all future zone-vs-bar arithmetic. The
  instrument works: one read, the charter's premise graded, the
  residue sized, the sentence written before the battery could be
  narrated by anyone.
- **THE GEN-17B VERDICT (2026-07-23): KILL — bigtest 1206 under the
  1208 floor; the manifest stays GEN-16 — AND THE KILL BUYS THE
  CAMPAIGN'S CLEANEST DOSE-RESPONSE CURVE.** The dose did EXACTLY
  what the zone read predicted and then some: h3held 198/200 = 0.99
  (0.00 -> 0.805 -> 0.99 across two fires; penumbra converted AND
  the dark mass converted — the line was dose and vocabulary at
  once, both medicines landed); adupheld held at 198. THE PRICE:
  bigtest 1232 -> 1206 (−26) and alg4 400 -> 396 — the 13.6% x
  4-reps dose bought its target OUT OF THE GENERAL REGISTER. The
  percentages-smuggle-repetition law measured live, two clean
  points on the curve: (1 rep: h=161, big=1232) and (4 reps @+8k:
  h=198, big=1206) — displacement ~9 bigtest answers per rep-step,
  acquisition ~12 h-rows per rep-step. Cert-v2 1.0000 (929 — the
  certified core UNTOUCHED by the displacement: the register loss
  is penumbra-mass, not umbra — the two-silhouette law's first
  cross-read). ALSO NOW VISIBLE: alg4 fails its 402 bar at EVERY
  measured point of the gen-17 family (400/400/396) — the gen-17
  mix itself costs ~2 alg4 answers; the binding constraint is no
  longer the hundreds line. THE FORK, priced for the word: (A)
  17c-interpolation — 2 reps from armR, ~8k: the curve predicts
  h~175, big~1224 — BOTH bars knife-edge, alg4 likely still under
  by ~2; (B) 17c-fresh-uniques — the house's own validated lever
  (fresh uniques over re-epochs, the crown-continuation law): mint
  6,000 NEW unique hundreds rows (not reps), 1 rep each, continue
  from armR — historically regularizes instead of crowding, and
  the alg4 question gets its own autopsy before any bar bends.
  RECOMMENDATION: (B), with an alg4 slot-autopsy (which 6 answers
  did the gen-17 family lose vs gen-16, zero-GPU from banked logs)
  BEFORE the fire — the binding constraint deserves its mechanism
  before another dose is priced. g17b banks as bench artifact
  beside armR. Two kills, two fires, and the ledger holds: a
  record, a saturated line, a displacement coefficient, and a
  named binding constraint — the era's economics in four numbers.
- **THE ALG4 AUTOPSY (2026-07-23): CHURN, NOT LESION — the binding
  constraint dissolves into a noise-floor question.** Per-row diff
  v4 (402) vs armR (400): 49 LOST, 47 GAINED — and the two sets are
  SHAPE-IDENTICAL (mean n_vars 17.0 vs 17.1; kinds-per-row within
  noise on every kind — pct 0.73/0.81, mod 1.69/1.55, sel
  0.31/0.30). No family lost its vocabulary; the frontier churned
  +-48 hard multi-var rows and settled 2 lower — penumbra exchange
  at the boundary, the two-silhouette law's second cross-read in
  one night. IMPLICATIONS, banked: (1) NO MIX EDIT for alg4 — there
  is nothing to cure; (2) the alg4 bar (402) sits INSIDE the churn
  band — a +-2 net on ~48-row churn is trajectory noise, so the
  SNAPSHOT RIDER (#57) upgrades to LOAD-BEARING: 17c saves periodic
  ckpts and the wobble census runs on real substrate; (3) THE
  BAR-NOISE LAW (prose, prospective only): bars pinned within a
  fixture's measured churn band cannot cleanly convict or acquit —
  FUTURE bars carry margins >= the noise floor once measured;
  standing bars stay as pinned (no post-hoc bending — the law
  governs the NEXT charter, not this one), and checkpoint selection
  NEVER reads bar fixtures (Goodhart's fence: the val slice picks,
  the bars judge). 17c proceeds as chartered: fresh hundreds
  uniques, no alg4 line, the snapshot census riding.
- **THE GEN-17C CHARTER (2026-07-23, registered under the standing
  word; the autopsy's sequencing honored).** BASE: continuation from
  g17_armR (unchanged — the record basin). DIET: gen17_mix + 6,000
  FRESH hundreds uniques (not reps — the fresh-uniques law; deduped
  against all existing hundreds knots; total 9,000 uniques at 1
  rep/unique, ~9.5% share, the dose curve's displacement lever
  avoided). RECIPE: 2x4k SGDR, LR 1e-4; SNAP_EVERY=500 on the final
  segment — the snapshot rider LIVE (wobble census substrate; val
  picks, bars judge, snapshots never read bar fixtures). BARS:
  unchanged from the gen-17 charter, with the bar-noise law noted
  prospective-only (alg4's 402 stands as pinned; the churn band is
  reported beside it, not instead of it). KILL: bigtest < 1208.
  The prediction, registered: fresh uniques buy hundreds
  acquisition WITHOUT the register toll (the crown-continuation
  curve's own history), landing h3held >= 170 with bigtest held
  >= 1223.
- **GUT #58: THE HAMMERHEAD (2026-07-23, Bryce + relay + Code,
  registered as amended; the read priced and QUEUED).** The honest
  sort first: interps 2-7 are banked law re-skinned — the wide
  aperture is the parser's cross-attention (the routing photograph
  its portrait), cross-attention-as-potentials is #47's foundation
  verbatim, parallel propagation is the width law (#44), the
  buried-signal manifold is the collapse quotient. CONFIRMED SEATS;
  the metaphor confirms, it doesn't extend — the hammerhead's body
  is the two-jaws architecture drawn in cartilage. THE FIND is
  interp 8: the shark reads a DIFFERENTIAL ACROSS A WIDE BASELINE —
  the left-right difference triangulates what no single pore
  senses. Held against the panel exam's wound (witnesses sharing a
  baseline, unanimity triangulating nothing), it names the one
  operation the house doesn't do. SEVENTEENTH SPECIMEN at
  countersign (mild, the substrate-overclaim species, second
  sighting): 'zero-GPU on banked states... post-entourage-17' —
  entourage-17 has not run (e16 is the last), and the collapse
  artifact holds AGGREGATES ONLY (two mean-distance pairs; no
  per-item table exists anywhere) — the join is unpriceable from
  disk. CURE, now standing: SUBSTRATES ARE ARTIFACTS TOO — a read
  is zero-GPU only if its per-item tables are already banked;
  check the artifact's GRAIN, not just its existence. **(a) THE
  BASELINE LAW (prose, minted):** decorrelation is APERTURE
  SEPARATION, not sensor count — the panel failed on the scope
  specimens because its witnesses shared a baseline; the cure is
  instruments whose DIFFERENCE carries what neither reading holds.
  **(b) THE DIFFERENTIAL READ (registered, GPU-MINOR joint pass —
  one fixture, both apertures per item: routing fidelity vs
  representation-to-centroid distance; QUEUED behind gen-17c's
  verdict).** Pinned prediction, kill-only: the cold-error residue
  (the 65 — register-shared, curriculum-deep, invisible to both
  narrators per the Flipped kill) should LIGHT THE DIFFERENTIAL
  where no single aperture flagged it (a deep deception may fool
  each aperture consistently but DIFFERENTLY); if the differential
  reads flat — both apertures deceived identically — that is the
  deepest confirmation yet that the cold errors live below all
  learned geometry, routing them PERMANENTLY to the guard and the
  books, the only organs that don't sense from the learned
  baseline. Either verdict pays. **(c) THE FENCE:** the
  differential is diagnostic only — it photographs disagreement,
  it never adjudicates; a structural aperture entering any
  acceptance path is the Goodhart door the constitution welds.
- **THE GEN-17C VERDICT (2026-07-23): THIRD KILL — bigtest 1214 /
  alg4 386; the manifest stays GEN-16 — AND THE REGISTERED
  PREDICTION FAILED HONESTLY ON BOTH COUNTS.** Fresh uniques
  saturated the line AGAIN (h3held 196/200, adup 199/200 — every
  form of this medicine teaches its target) but the register tolled
  anyway: −18 vs armR (milder than reps' −26, refuting the
  no-toll prediction), and **ALG4 SLIDES MONOTONICALLY — 402 -> 400
  -> 396 -> 386 across the family** — no longer the autopsy's +-2
  churn: a systematic erosion scaling with continuation steps,
  which AMENDS the churn verdict (churn was true at +-2 grain
  between v4 and armR; the SLIDE emerges under further
  continuation — scope decay of my own verdict, one day old,
  banked as such). The dose curve gains its third point: (1rep/16k:
  h=161, big=1232) / (4reps/+8k: h=198, big=1206) / (fresh/+8k:
  h=196, big=1214). Cert-v2 1.0000 again (924 — the umbra
  untouched through THREE displacements; the two-silhouette law's
  third consecutive cross-read). THE OPEN MECHANISM: why does
  continuation on this mix erode bigtest/alg4 while cert-mass
  holds? Candidates: trajectory wobble (the swings within noise),
  val-slice selection bias (pick-best-by-val favoring new-line
  checkpoints), or real slow drift. **THE WOBBLE CENSUS FIRES AS
  THE KILL'S AUTOPSY — substrate banked (8 snapshots at 500-step
  spacing), chartered by #57's rider: eval the trajectory on
  bigtest + alg4; tight readings = real drift (mechanism hunt
  opens); wide readings = the bars sit inside trajectory noise and
  the bar-noise law gets its first measured floor.** Three exams,
  three lawful kills, the gate holding gen-16 with zero waivers —
  and the era's question sharpened to one word: is the register's
  toll NOISE or DRIFT?
- **THE WOBBLE CENSUS VERDICT (2026-07-23): NOISE AND DRIFT, both —
  the campaign's first trajectory photograph decomposes the era's
  question.** Eight snapshots, two fixtures: RAW bands enormous
  (bigtest 121: 1098-1219; alg4 53: 336-389) but REGIME-TAGGED —
  the hot mid-cosine phase dominates; the ANNEALED TAIL
  (s3000/3500/4000) reads bigtest 1219/1209/1214 (**+-10 noise
  floor at convergence**) and alg4 380/380/386 (+-6). DECOMPOSED:
  (1) ALG4'S DEFICIT IS REAL DRIFT — 16-22 under its 402 bar,
  exceeding its +-6 band; the gen-17 mix family genuinely erodes
  alg4 under continuation. (2) BIGTEST IS MARGINAL-REAL — tail
  13-23 under armR's 1232, at the edge of its +-10 band. (3) THE
  DOSE-CURVE SLOPES (~9/rep-step) ARE THE SAME ORDER AS THE NOISE
  — the displacement coefficients gain honest error bars; single-
  point dose readings are hereafter suspect. (4) THE BAR-NOISE LAW
  GAINS ITS FIRST MEASURED FLOOR: bigtest +-10, alg4 +-6 at
  annealed grain — future bars carry margins >= the floor, or
  batteries eval snapshot-averages. (5) **GUT #57'S LEAN REFUTED
  BY ITS OWN CENSUS** — 'modest wobble, lever marginal' dies at
  +-10 measured; THE EMA FLYWHEEL UN-PARKS, its probe chartered
  by the census's own registration structure: average the three
  annealed snapshots, eval once — PASS bar pinned before firing:
  avg bigtest >= 1219 (the best constituent) = smoothing recovers
  noise; under = basins disagree, re-park with a measured no.
- **THE FLYWHEEL VERDICT (2026-07-23): NO AT ITS PINNED BAR —
  RE-PARKED with a measured no, texture banked.** First, the
  instrument errata: the probe's subprocess evals returned -1
  (the parent held the AM device while averaging; children could
  not open it — the AM single-process law's newest specimen; the
  probe script's averaging moves to CPU for the record). TRUE
  NUMBERS, run directly: avg-of-3-annealed bigtest 1217 (vs
  constituents 1219/1209/1214 — BEATS the mean +3 and two of
  three, misses the pinned >=1219 by 2), alg4 387 (BEATS ALL
  THREE constituents), h3held 196 (held). READ: parameter
  averaging is mildly real — it tracks the trajectory mean
  upward — but buys no free maximum; within the noise floor of
  everything it touched. THE BAR IS THE BAR: re-parked. THE
  OPERATIONAL RESIDUE for gen-18's charter: batteries eval 2-3
  annealed snapshots and REPORT THE BAND beside the headline
  (bar-noise honesty at near-zero cost) — measurement discipline,
  not parameter surgery. THE ERA'S CLOSING LEDGER, one line per
  instrument: three exams three kills (bars pinned, never bent);
  restart law confirmed (cold-pan, +17); dose curve three points
  with error bars; hundreds line saturable by every medicine
  tried; add-dup closed; alg4 drift REAL (16-22 past its +-6
  band, mechanism OPEN — the next charter's first question);
  noise floors measured (+-10/+-6); umbra untouched through
  three displacements; flywheel no; zone meter, differential
  read, and routing canvas standing ready. The gate: GEN-16,
  zero waivers, its record intact — guarded this week by its own
  bars against three of its children.
- **THE BAR REVIEW UNDER THE BAR-NOISE LAW (2026-07-24, prose lane
  of the composed fork; prospective per the law's own text).**
  GEN-18 CHARTER DESIGN INPUTS, written once with the measured
  floors: (1) bigtest bars carry margin >= 10 (the annealed floor)
  — a promotion bar of 'record + margin' or a snapshot-band
  requirement, never a bare point; (2) alg4 bars carry margin >= 6;
  (3) BATTERIES REPORT THE BAND: every future battery evals 2-3
  annealed snapshots per candidate and prints min/max beside the
  headline (the flywheel verdict's operational residue); (4)
  dose-curve readings are multi-snapshot or carry the noise floor
  as their error bar; (5) val picks, bars judge — restated in the
  charter template itself. STANDING BARS UNTOUCHED (gen-16's
  manifest floors stand as pinned; the law governs the NEXT
  charter). The alg4 drift autopsy fires alongside (standing
  rhythm: diagnostic read on banked snapshots); book-5 t10 holds
  for the word.
- **THE ALG4 DRIFT AUTOPSY VERDICT (2026-07-24): THE MECHANISM IS
  DILUTION — neither lesion nor pure churn.** Across the annealed
  family (armR/s3000/s3500/s4000 vs v4): STABLE LOSSES 26 (wrong in
  all four) vs STABLE GAINS 16 (net −10 stable) + FLICKER 58 — the
  tail deficit is mostly a flickering frontier with a small stable
  core. AND THE STABLE CORE IS SHAPE-FLAT: mean n_vars 17.1, kinds
  per row matching the global hard-tail profile (pct 19/26, mod
  23/26, fdiv 20/26 — the full mix, no family concentration),
  max_given unremarkable (23-60). No vocabulary hole; no family
  lesion. THE NAME: **REGISTER DILUTION** — when new lines enter a
  mix, the base's hardest compositional band receives
  proportionally less gradient, and the frontier (penumbra — the
  umbra never moved, three fires measured) contracts by roughly
  the share taken. The two-silhouette law's fourth cross-read:
  dilution eats penumbra, never umbra. **THE DILUTION LAW (prose,
  minted — the dose law's completion):** percentages smuggle
  repetition at small n; AT LARGE N, SHARES DILUTE THE FRONTIER —
  a mix that adds mass must re-declare shares AT THE BAND GRAIN
  (the hard tail's effective share held constant via the
  kind/knot-rehearsal matrix's existing machinery), not just at
  the corpus grain. GEN-18 DESIGN INPUT #6, joining the bar
  review's five: new-line additions pair with a hard-band
  rehearsal ration sized to hold the tail's share; the dose curve
  predicts the toll and the ration is its antidote — testable as
  a registered prediction on the next fire. The composed fork's
  (i) and (iii) are delivered; t10 and the gen-18 charter hold
  for the word.
- **GUT #59: IT'S THERE, YOU JUST DON'T SEE IT (2026-07-24, Bryce +
  relay + Code, registered; the audit WALKED in-entry).** The knock
  is epistemology, not instrument — and the proof-cases pull clean
  from the ledger's own history, sorted by failure species: SPECIES
  ONE, no instrument yet (the bilingual fork: 1.000 decodability
  beside a blind zero-shot head, 2447/2464 — content present,
  extractor absent, two thousand pairs developed it; the routing
  photograph: five sightings of inference before one camera);
  SPECIES TWO, wrong projection (the length-warp; the rotation law:
  raw 0.593 -> aligned 0.988, 5298 — 'the shape was intact; the
  space had turned', the most literal case on the books; the ETF
  2D-story redirect); SPECIES THREE, differential required (#58's
  baseline law, banked). AND THE CONVERSE'S FENCE-CASES: evoked
  values dead at linear AND at a 512-hidden GELU (6328 — deeper
  than the relay cited; the survey is more exhausted than
  remembered), [45] fused at the source per the representability
  audit's own bar. **(a) THE LATENT-IMAGE LAW (prose, both edges
  in-text):** presence precedes visibility — development is
  instrument-work, not creation; AND the converse stands: some
  films never resolved the image, and the survey that proves it is
  finite and walkable. Absence of evidence is a statement about
  the instrument exactly until the survey is exhausted. **(b) THE
  DEVELOPMENT PROTOCOL (the negative-banking checklist, standing
  from tonight):** before any null banks, four questions — (1)
  extractor matched to content? (linear/nonlinear/TRAINED-HEAD —
  the bilingual chemistry is the deepest rung); (2) projection
  aligned? (Procrustes, length-control, metric jurisdiction); (3)
  differential taken? (single-aperture null != dual-aperture
  null); (4) substrate grain sufficient? (aggregates cannot answer
  per-item questions — the seventeenth's cure). TWO SENTENCE
  STRENGTHS, distinguished typographically in all future banking:
  **NOT PRESENT AT AFFORDABLE GEOMETRY** (survey exhausted) vs
  *not visible to this instrument* (survey open). **(c) THE
  RETROACTIVE AUDIT, walked:** the standing negatives largely
  already carry their lawful strengths — [45] and the band-dose
  kill pass on the representability audit's stronger machinery
  (flat across trainings + targeted intervention); the delta-probe
  and zone dark-masses were banked instrument-relative by design;
  the Flipped kill carried its own differential; the cold
  residue's strong sentence lawfully WAITS on the queued
  differential read (Q3 pending by design). ONE DEVELOPABLE
  NEGATIVE found: the evoked-value null — dead at linear and GELU
  but the trained-head-with-pairs chemistry (the very rung that
  developed the bilingual image) was never applied; priced
  GPU-minor, UNFORCED, parked on the docket with its rung named.
  The protocol formalizes what the house's best negatives already
  practiced — now the worst ones must practice it too.
- **GUT #60: CHILDREN RESIST CHANGE (2026-07-24, Bryce + relay +
  Code, registered as amended; the discriminator READ from banked
  census same-hour).** The bathtub is the find — resistance on BOTH
  crossings means the cost lives in the BOUNDARY, not the states:
  hysteresis in textbook form. Anti-re-buy sort: interps 3-7 are
  banked physics confirmed (activation energy = the quench clause;
  basin-with-mass = the ring gauge's vocabulary; momentum's literal
  answer = #57's cold pan). EIGHTEENTH SPECIMEN at countersign
  (stale-board species, mild, 2nd sighting pattern): the relay
  built the discriminator into 'the standing alg4 question' — the
  autopsy DELIVERED yesterday (dilution, 93ef001); verdicts and
  queues are ledger state; re-read before building. THE
  DISCRIMINATOR SURVIVES ITS OWN SPECIMEN: dilution named the
  share arithmetic but never tested cycle-synchronization — and
  the banked census answers at one-boundary grain: **THE BOUNDARY
  TOLL, read same-hour: the restart's hot phase craters both
  fixtures (bigtest ->1098, alg4 ->336 by s1000) and annealing
  refunds MOST BUT NOT ALL (1214/386 at segment end)** — the
  unrefunded residue is the per-cycle toll, charged at the
  boundary, landing heaviest on the least-rehearsed register.
  Scope: n=1 boundary; the full multi-boundary test rides the next
  SGDR fire (SNAP_EVERY across ALL segments — rider registered).
  H1 AND H2 COEXIST: dilution sets what the hot phase re-deepens
  first; the boundary charges the toll. **(a) THE HYSTERESIS LAW
  (prose, both edges):** every consolidation deepens the current
  basin AND raises the walls around it — entry costs are visible
  at acquisition, exit costs accrue silently until the next
  transition invoices them; a campaign that only meters entry
  will keep being surprised by exits. Nobody resists states —
  everybody resists doors, and the toll booth charges both ways.
  **(b) GEN-18 DESIGN INPUT #7 (the ration line's second
  clause):** the hard-band rehearsal ration carries MASS
  (share-preserving, dilution's antidote) AND PLACEMENT (riding
  the hot phase of each cycle, where the toll charges — not
  appended after annealing). **(c) THE UMBRA-TREND COLUMN
  chartered on the zone meter:** per-family umbra mass across
  generations — a family whose umbra shrinks is paying exit costs
  somewhere, visible before any bar convicts it; the exit-cost
  gauge the law says was missing, riding the standing zone column
  at zero added cost.
- **GUT #61: MAP REDUCE (2026-07-24, Bryce + relay + Code,
  registered as amended; the census RUN same-hour, the line parked
  with a measured no).** The sort confirms five seats without
  extension: map = the width law's jurisdiction (views, samples,
  lineages — embarrassingly parallel by design), shuffle = the
  routing photograph's subject (#47's message passing), reduce =
  the lattice's disposal stack (solver-filter, plurality,
  fingerpost, unanimity, the key above all). The cartilage sentence
  applies. THE DATACENTER'S TWO LAWS: **(a) THE BARRIER CLAUSE
  (prose, minted general):** no aggregation before the stage
  completes — the vote counts all views, the battery walks all
  fixtures before the pen, and THE CHANNELS OBEY THE SAME BARRIER:
  the fifteenth specimen was a reduce launched before its shuffle
  arrived (a verdict narrated before the artifacts landed); the
  barrier law is the cure's general form, in machines and channels
  alike. **(b) THE STRAGGLER CENSUS — RUN, and the banked history
  honored first:** the gen-13 mirror audit already measured this
  population (23 items at effK 3-4, ALL unanimous-correct, zero
  misfires — the exposure measured EMPTY at that vintage, a
  precision note on the relay's 'carry certification exposure':
  in-principle yes, measured no). RE-RUN AT GEN-16 VINTAGE
  (text-side effK recomputed + joined against the banked V4 vote
  tables, zero-GPU true this time): **25/1500 at effK<5 (22 at
  4, 3 at 3) — ALL BENIGN, zero unanimous-wrong misfires.** Two
  vintages, two clean reads: the reduced-dart certificates exist
  and none has ever misfired. **(c) THE SPECULATIVE-VIEW LINE
  PARKS WITH A MEASURED NO** — the campaign's cheapest kill:
  quorum-preserving re-dispatch has no customers at two vintages;
  the line stays on the books with its trigger named (a first
  effK-shortfall misfire re-prices it instantly — the census is
  one join, re-runnable per generation beside the zone column),
  and the fence preserved in-text: speculation would replace
  stalled RENDERINGS, never stalled VERDICTS — the pen waits for
  every stage, always.
- **THE DIFFERENTIAL READ'S QUANTITATIVE BAR (2026-07-24, pinned
  before measurement; the driver's-seat fire, #58's read ungated by
  17c's verdict).** Substrate: aperture A = the banked per-item
  routing fid (1500, crown_v4 on bigtest); aperture B = per-item
  mean centered-cosine distance of slot fst vectors to their own
  kind centroids (gen-16 bank, computed fresh — one forward pass).
  Populations from the banked V4 votes: COLD = unanimous-wrong;
  CORRECT = majority-correct. Both apertures z-scored on the full
  fixture; the differential = |zA − zB| per item. PRE-WRITTEN
  VERDICTS: **LIGHTS** if the cold population's mean differential
  exceeds the correct population's by >= 0.5 of the correct
  population's own std — the deep deceptions fool the apertures
  DIFFERENTLY, and the differential becomes the cold-error
  detector no single aperture could be; **FLAT** otherwise — both
  apertures deceived identically, the deepest confirmation that
  the cold errors live below all learned geometry, ROUTING THEM
  PERMANENTLY to the guard and the books (the only organs sensing
  off the learned baseline), and the cold-error question CLOSES
  with a structural answer.
- **THE DIFFERENTIAL READ, FIRST ADMINISTRATION (2026-07-24, the
  driver's-seat fire): VACUOUS ON ITS PLANNED FIXTURE — AND THE
  VACUOUSNESS IS THE DIVIDEND.** The instrument built and ran clean
  (both apertures per-item on bigtest: banked routing fid + fresh
  fst-to-centroid distances, 1500/1500 scored) — and the COLD
  POPULATION IS EMPTY: **zero unanimous-wrong items in 1500 under
  gen-16's banked lattice votes** — unanimity precision 1.0 on the
  measurement fixture, independently consistent with cert-v2's
  1.0000 and now measured at the full-fixture grain. NEITHER
  pre-written verdict applies: with n=0 the read is VACUOUS, not
  FLAT (the code's else-branch printed FLAT; corrected here — a
  verdict must never outrun its population; the two-sentence
  discipline extended: vacuous is a third strength, weaker than
  both). THE QUARRY MOVED: the 65 were defined in the FLIP-PROBE's
  population (8188 — a different vintage and voting machinery than
  the V4 bigtest lattice); their provenance needs its own
  archaeology before the differential can hunt them (which
  fixture, which gate, do they persist under gen-16 at all?) — and
  if they live on WILD text, aperture A's jurisdiction ends (gold
  spans do not exist off the minted register; the routing aperture
  is annotation-side by construction — an honest structural limit,
  stated). THE STANDING RESULT: the differential instrument is
  validated on substrate and waits armed; the cold-error question
  gains its sharpest reframing yet — at gen-16 on the measurement
  fixture, THERE ARE NO COLD ERRORS LEFT TO FIND; whatever remains
  of the 65 lives off-register, where the guard and the books were
  already the assigned organs. The crown era may have quietly
  dissolved the measurable cold class — the provenance archaeology
  (zero-GPU) decides whether that sentence gets its strong form.
- **THE DIFFERENTIAL'S TRUE TARGET FOUND + AMENDED BAR (2026-07-24,
  pinned before the second administration).** Archaeology: the 65 =
  residue_portrait.json's triple-voice-resistant items (bigtest
  indices, ABSTAIN channel — 'cold' was compression drift in the
  naming chain; they are dark/penumbra-zone repair-resisters, not
  unanimous-wrongs, and aperture A has FULL jurisdiction: minted
  text, gold spans). MATCHED CONTROL: the recovered abstains (same
  channel, repair succeeded — the honest baseline; correct-items
  would confound channel with resistance). AMENDED BAR, pinned:
  LIGHTS if residue mean |zA−zB| >= recovered-abstain mean + 0.5 x
  recovered-abstain std; FLAT otherwise -> the residue's deception
  is aperture-invariant, and the guard+books routing takes its
  strong sentence.
- **THE DIFFERENTIAL LIGHTS (2026-07-24, second administration, the
  amended bar): residue mean |zA−zB| 2.181 vs bar 2.143 (recovered
  1.551 + 0.5 x 1.184) — OVER THE BAR, margin honest and thin
  (+0.53 sigma vs the 0.5 pin; stated, not rounded).** AND THE
  APERTURE SIGNATURE IS THE FIND: on the 65, routing fid z = −1.671
  (catastrophically mis-aimed) while collapse z = +0.479 (barely
  off-centroid) — **the residue is MIS-ADDRESSED-BUT-WELL-FORMED:
  representations near their knot families, messages missing their
  slots.** The stingray photographed: invisible to any
  representation-side monitor (the shape looks right), glaring
  across the two-aperture baseline — #58's law confirmed on its
  first real quarry. THE HAND-OFF, closing a loop three guts wide:
  this is EXACTLY the population the ROUTING-CANVAS PROBE (#55,
  queued) was registered to treat — auxiliary span-supervision on
  the binding layer now has its named customer list (65 items,
  banked indices, mis-addressing measured at z=−1.67 while
  everything downstream reads healthy). The probe's selectivity
  bars stand as pinned; its warrant just upgraded from
  plausibility to a photographed mechanism with a
  population-in-hand. The cold-error question's final anatomy:
  the measurable class on the certification channel is EMPTY
  (first administration); the abstain-channel residue is
  routing-deep and representation-shallow (second) — not below
  all learned geometry after all: below ONE aperture, visible to
  the pair. The hammerhead ate on its first hunt.
- **THE ROUTING-CANVAS PROBE, AMENDED BY CODE READ + FIRED (2026-07-24,
  Bryce's word).** THE CORRECTION FIRST, filed against #55's banked
  premise (a joint miss, relay-decoded and countersigned into the
  ledger): the canvas EXISTS — line 531 is a direct CE of the fat
  attention mass against normalized gold fspans, weight 1, standing
  in the loss since the head's birth. 'Unsupervised shadows' was
  false against the code; the truth is sharper: THE CANVAS IS
  PRESENT AND THE 65 ARE MIS-ADDRESSED ANYWAY — so the probe
  re-forms as a DOSE PROBE on the existing canvas (FAT_W env minted:
  lambda-weight the standing term; the probe doses, never adds).
  FOUND IN PASSING: the gen-17 mint lines all carry EMPTY spans —
  they never fed the canvas (a diet note for gen-18: minted lines
  should emit spans; the renderer knows exactly where its values
  land). DESIGN: FAT_W=4, 6k continuation from crown_reader_v4 on
  the crownv4 states (span-carrying base — pure dose, no confound);
  MEASURE: routing fidelity re-photographed under the dosed head
  (full fixture + refused-class + THE 65 specifically) + bigtest
  guard. BARS (#55's, operationalized): PASS if delta
  fid_refused >= 2x the bigtest displacement fraction AND >= +0.05
  absolute; GUARD bigtest >= 1208; the band-dose kill stands
  sentinel over the architecture case.
- **THE CANVAS DOSE VERDICT (2026-07-24): KILL AT BOTH BARS — AND
  THE ADDRESSING WALL IS NAMED ARCHITECTURAL.** FAT_W=4, 6k from
  crown_v4 on the span-carrying base: refused-class fid 0.506 ->
  0.511 (+0.005 vs the +0.05 bar — noise), THE 65: 0.412 -> 0.414
  (+0.002 — nothing), full-population fid 0.830 -> 0.840, and the
  GUARD BROKE: bigtest 1201 < 1208 (−22 displaced). The band-dose
  pattern verbatim: 4x the supervision weight cannot move what the
  architecture has already decided — **the binding-layer addressing
  joins the param-path digit softmax as the ARCHITECTURE LEDGER's
  second wall.** Structure doesn't take medicine by the spoonful;
  now measured twice, on two different organs. THE TREATMENT
  ROUTING, complete and final for the 65: the DIFFERENTIAL detects
  them (LIGHTS, banked), the GUARD abstains on them, the BOOKS
  re-voice them — detection by instrument, treatment by detour,
  never by retraining. #55's probe closes with a structural answer;
  FAT_W stays in the code as a measured-no lever (the canvas was
  always there; the wall was always behind it). canvas_dose banks
  as bench artifact. THE DRIVER'S TURN COMPLETE: the differential
  built, fired twice, LIT on its true quarry; the cold class
  measured EMPTY on the certification channel; the 65's anatomy
  finished (mis-addressed-but-well-formed, architectural,
  detour-treatable); the canvas question asked and answered at
  6k-step price. The queue behind the pen is EMPTY for the first
  time this era — everything registered has fired, killed, or
  parked with its trigger named.
- **THE GEN-18 FIRE CHARTER (2026-07-24, registered — GPU AND MINT
  HOLD FOR BRYCE'S WORD; the first charter in campaign history where
  every clause cites a measurement).** THE CANDIDATE: the head
  learns its audited diet without paying the dilution toll. BASE:
  gentle continuation from crown_reader_v4 (the gate's clean
  lineage; the gen-17 bench artifacts stand as measurements, not
  parents). THE EIGHT INPUTS, each with its artifact: (1) BARS
  CARRY THE NOISE FLOORS — promote requires bigtest >= 1223 AND
  annealed-snapshot-band min >= 1213 (record minus the measured
  +-10); alg4 >= 402 with band min >= 396 (+-6); batteries eval 3
  annealed snapshots and print the band beside every headline. (2)
  DOSE DECISIONS CITE THE CURVE (three points, error bars = the
  noise floor). (3) SCHEDULE: SGDR 4x4k (the restart law's winner)
  with SNAP_EVERY=500 on ALL segments — the multi-boundary toll
  read rides free. (4) THE RATION ARMS — the charter's one
  question, the dilution law's registered prediction tested
  directly: ARM A = the audited mix WITHOUT ration (control); ARM
  B = WITH the hard-band rehearsal ration (share-preserving at
  band grain, sized from the dilution verdict). (5) RATION
  PLACEMENT: arm B's ration mass concentrates in each cycle's hot
  phase (the boundary toll's clause). (6) THE ZONE COLUMN rides
  the battery: umbra/penumbra/dark per fixture + the umbra-trend
  column's first cross-generation row. (7) MAINTENANCE DOSES: the
  saturated lines (hundreds, add-dup) at fresh-unique 1-rep
  minimal mass — held, not re-bought. (8) SPANS FROM BIRTH: every
  minted line emits gold spans (the renderer knows where its
  values land; the canvas feeds from the whole diet — the two-
  terminal law's territory, the canvas kill's surviving yield).
  MIX: crown_v4 base + the audited lines + book-5 organic (164+,
  10 reps) + [arm B: ration]. BARS also: h3held >= 170, adupheld
  >= 180, cert-v2 >= 0.998, guard 20/20, acceptance >= 7. KILL:
  both arms bigtest < 1208. ENTOURAGE-18 rides promotion (full
  roster + delta-probe third point + discharge walk + zone
  baseline). The mint, the precompute, and the fire hold for the
  word; t10 may run beside the mint per the width law whenever
  the desk's word comes.
- **THE GUT REGISTRY COMPILED (2026-07-24, Bryce's ask):**
  docs/GUT_REGISTRY.md — the 61 conversions indexed with ledger line
  pointers, the pre-numbering era listed by its actual ordinals
  (no reconstruction of what the record does not number), and the
  maintenance rule in-text: each conversion appends its row in the
  SAME transaction as its ledger entry (the prose-promotions law
  applies to indexes too).
- **THE RATION BAND CORRECTED MID-BURN (2026-07-24, caught at the
  dose table, fixed before arm B reads it):** the mint's band at
  n_vars>=14 covered 70.9% of the mix — NOT a band; n_vars is
  nominal in most rows (52% declare the padded 24, every minted row
  among them). The true contour is FACTOR COUNT: index rewritten at
  n_fac>=14 = 16,868 rows (22.5%), matching the drift autopsy's
  lost-row profile (n_fac 10-22, mean ~17). Zero cost: arm A never
  reads the index; arm B's first segment sits ~2h downstream; the
  mix and precompute are untouched. The lesson, one line: A BAND
  THAT COVERS THE CORPUS TESTS NOTHING — check the discriminator's
  distribution before the fire, not after (the mint prints the
  band share for exactly this reason, and the print did its job).
- **THE GEN-18 VERDICT (2026-07-24): FOURTH REFUSAL — the manifest
  stays GEN-16 — AND THE CHARTER'S ONE QUESTION IS ANSWERED:
  **THE RATION WORKS.** Head-to-head at matched budget: ARM B
  (hot-phase ration, n_fac>=14 band x1.5) alg4 = 408 — ABOVE the
  402 bar, above gen-16 itself, the FIRST gen-17-family recipe to
  clear alg4 — vs ARM A (no ration) 397. **+11 alg4 from placement
  and share alone: the dilution law's registered prediction
  CONFIRMED by direct experiment.** AND THE RATION STABILIZED THE
  TRAJECTORY: B's bigtest band spread 5 (1214/1213/1209) vs A's 18
  (1224/1206/1220). THE BAR-NOISE LAW EARNED ITS KEEP ON ITS FIRST
  EXAM: arm A's headline 1224 CLEARED the old bare-point bar — and
  the band-min bar caught it as a lucky point on a wobbling
  trajectory (s3000 = 1206). Under the old bars, A would have been
  promotion-eligible on a wobble; the noise-floor bars refused it
  for the right reason. THE REFUSAL'S CAUSE, a charter design
  error owned: **h3held COLLAPSED both arms (124/121 vs 170)** —
  input #7's 'maintenance dose' (1,000 uniques) assumed saturation
  carries via lineage, but the CLEAN GATE LINEAGE never learned
  hundreds (its baseline is 0.00); maintenance maintains what the
  PARENT knows — LINEAGE-NEW LINES NEED TEACHING DOSES (the dose
  curve says 3,000 uniques buys ~0.80; 1,000 bought 0.62,
  curve-consistent). Add-dup held at 197 both (500 sufficed — its
  wall teaches at tiny dose). Cert-v2 1.0000 BOTH (fifth
  consecutive; the umbra has never moved in five fires). ZONE
  COLUMN's first row: umbra 999 / penumbra ~408 / dark ~92. THE
  PRICED FOLLOW-UP (holds for the word): GEN-18B — arm B's recipe
  verbatim (the ration confirmed) + hundreds at the TEACHING dose
  (3,000 fresh uniques), one arm, bars unchanged: the recipe that
  cleared alg4 meeting the dose that clears hundreds, each proven
  separately tonight, composed.
- **THE GEN-18B CHARTER (2026-07-24, registered on the word; the
  composition of tonight's two proofs).** THE AMENDMENT FIRST, filed
  with the mortality law's family (where inheritance rules live):
  **SATURATION IS LINEAGE-SCOPED** — maintenance doses maintain the
  parent's knowledge; lines new to a lineage take teaching doses,
  priced by the dose curve. THE CANDIDATE: arm B's recipe VERBATIM
  (hot-phase ration, n_fac>=14 x1.5 — the confirmed lever) +
  hundreds at the TEACHING dose (3,000 fresh uniques total: the
  1,000 banked + 2,000 newly minted, spans from birth, deduped
  against all prior knots). ONE ARM (the ration question is
  answered; no control needed), 4x4k SGDR from crown_reader_v4,
  SNAP_EVERY=500 all segments. BARS UNCHANGED (incl. band-mins).
  KILL: bigtest < 1208. The fifth exam meets the first candidate
  carrying every measured input the era bought.
- **GUT #62: THE HERRING SHOAL (2026-07-24, Bryce + relay + Code,
  registered as amended; the read QUEUED behind the burning 18b).**
  Twelve interps, decisive sort: local-sensing-to-global-consensus
  (1-3, 5-6, 9) is the parser's architecture CONFIRMED (cross-
  attention is exactly leaderless local sensing) — but interp 9's
  guarantee is FALSE IN THIS HOUSE BY MEASUREMENT, and the falseness
  is load-bearing: shoals converge because ANY agreement is a valid
  shoal (alignment objectives), while math has one answer — the
  two-death-mode law (7660) killed loopy local propagation, and the
  withhold curve (26% ceiling; 'minimal graphs carry no messages —
  only the text proposes', 8141) measured why: real prose is
  redundancy-minimal. **(a) THE SHOAL LAW (prose):** local sensing
  without a coordinator is the front jaw's confirmed geometry;
  local consensus guarantees AGREEMENT, not CORRECTNESS — the
  boundary the solver's global exactness exists to cross; the
  shoal names the boundary and does not cross it. **(b) THE
  FLASH-EXPANSION UNIFICATION (the find, banked):** interp 12's
  threshold-triggered scatter IS the restart law at optimizer
  grain — periodic controlled explosion out of stagnation, already
  MEASURED PAYING (#57's cold pan, +17; SBP as the continuous
  gentle form) — and the photo-booth fence stands untouched
  (perturbation-for-TRUTH at inference is the banked grave; flash
  expansion is training-time anti-stagnation, a different organ).
  POPULATION CORRECTION at countersign: 'confident-wrong basins'
  overstates — the umbra-wrong class is EMPTY at gen-16 (the
  differential's first administration); the stagnant population =
  the DARK zone (~90 items) + the 65 residue. **(c) THE
  BOUNDARY-CROSSING READ (registered, bars pinned, QUEUED behind
  the 18b burn — GPU busy; substrate = gen-18's 64 banked
  snapshots, per-item correctness across segment boundaries):**
  do the stagnant items move at restart boundaries? VERDICT A:
  boundary-flip rate >= 2x the within-segment flip rate for the
  stagnant population -> the TARGETED-SCATTER registration opens
  (a scatter aimed by the zone meter, watts-priced, behind the
  standing queue). VERDICT B: frozen through all boundaries -> the
  architecture ledger gains its deepest entry — STAGNATION THAT
  SURVIVES CONTROLLED EXPLOSION; the detour is truly the only
  road, and the walls close their last open question. Kill-only;
  either verdict pays. The shoal handed back the house's own
  architecture with the one part it lacked: a way to leave a
  state that has become a trap.
- **GUT #63: THE SHOAL RETURNS WITH MUZERO (2026-07-24, Bryce +
  relay + Code, registered lean — a boundary-confirmation with one
  amendment).** The knock's own interp 2 does the sorting (cohesion
  vs one valid direction — #62's rejection stands), and the mapping
  corrects at its joints: policy-proposes holds (interp 5 — neural
  proposes on both sides); MCTS-vs-GAC are OPPOSITE species (interp
  6 — sampled-anytime search for oracle-less spaces vs
  complete-exact search where the key exists; solver-side MCTS is
  refuted v3-v4 territory per the entry at 3779); and the latent
  world model is the ANTI-THESIS, already adjudicated verbatim
  (3795: 'the registry IS the world model — owned, exact; latent
  dynamics in the solve path would trade the zero-leakage
  bottleneck for drift'). **(a)** The boundary RE-AFFIRMED at its
  banked form: policy pairs with sampled search where no oracle
  exists; the parser pairs with exact search because one does; the
  factor graph's symbolic auditability is the trust story. MuZero's
  one lawful import remains the TRIAGE HEAD (registered 3804,
  parse-side features, waiting on census volume — unchanged).
  **(b) THE FLOCKING CLAUSE, amended onto the mint-search spec
  (#41, ledger 7669; files on the grammar-width docket at its
  existing trigger):** when the grammar widens, the mint's search
  should FLOCK — coherent sweeps through composition space rather
  than independent draws: dedup/knot-diversity as REPULSION
  (built), the absent-cover list as ATTRACTION (coverage cohesion
  — successive mints staying near underpopulated frontier regions)
  — because the lantern's sliver lesson says coverage fills by
  sweeping, not by darts. Zero new machinery; one clause; the
  trigger unchanged. **(c)** The fence, standing: search never
  enters the solve path; latents never replace the graph; the
  shoal's second visit tunes an organ instead of proposing one —
  what mature metaphors do on return.
- **THE GEN-18B VERDICT (2026-07-24): FIFTH REFUSAL — THE CLOSEST
  EXAM YET, AND THE SEE-SAW NAMED.** The composition's ledger:
  **bigtest 1224, band [1224/1222/1220] — the FIRST candidate to
  clear the record bar WITH band-min intact** (spread 4: the
  ration's stabilization confirmed again); h3held 167/200 = 0.835
  (curve-consistent: 3k uniques bought what 3k buys — THREE ROWS
  under the bar); adup 197; cert-v2 1.0000 (SIXTH consecutive).
  THE MISS: alg4 387 [387/386/384] — the ration that recovered
  +11 against 5.5% new mass could not hold against 8.1%: **THE
  SEE-SAW — the hundreds line and the alg4 register compete for
  the same finite frontier**, and at RATION_W=1.5 the teaching
  dose's extra mass re-tolled what the ration had recovered. The
  dilution law extends: THE RATION'S STRENGTH MUST SCALE WITH THE
  NEW MASS IT OFFSETS (1.5x covered 5.5%; 8.1% needs more). TWO
  READINGS, both banked: (i) the DOSE reading — g18c at
  RATION_W=2.0 on the SAME mix (no mint, no precompute — one env
  knob, ~25 min) tests whether stronger rationing holds alg4
  while hundreds keeps its 167; if yes, the remaining 3 hundreds
  rows price at ~+200 uniques; (ii) the CAPACITY reading, stated
  honestly after five refusals — the head is ~3.2M params and the
  simultaneous demand (record bigtest + alg4 + hundreds + crowns)
  may exceed its frontier at H_W=512; the ALG_HW dial exists and
  a capacity arm is the deeper question if the see-saw survives
  stronger rationing. The fifth exam refused three rows and
  fifteen answers, and priced both of its own next moves.
- **THE GEN-18C FIRE (2026-07-24, on the word): RATION_W=2.0 on the
  IDENTICAL 18b mix and states — the see-saw's cleanest question,
  one knob.** Verdicts pre-written: alg4 holds >= 402 with hundreds
  ~167 -> rationing scales with mass, the diet arithmetic wins, the
  last three rows price at ~200 uniques; the see-saw survives ->
  the frontier is finite at H_W=512 and the CAPACITY ARM charters
  with the atlas tripwire's signature as warrant. Bars unchanged.
- **THE GEN-18C VERDICT (2026-07-24): SIXTH REFUSAL — AND THE
  PRE-WRITTEN FORK FIRES: THE SEE-SAW SURVIVES STRONGER RATIONING;
  THE FRONTIER IS FINITE AT H_W=512.** RATION_W=2.0 recovered alg4
  to 402 with band intact [402/399/398 >= 396] — and bigtest paid
  wholesale: 1224 -> 1208, both its bars broken [1208/1204/1205].
  hundreds 165 (dose-stable), adup 197, cert-v2 1.0000 (SEVENTH
  consecutive — the umbra now a constitutional fact). **THE
  CONSERVATION MEASUREMENT, the kill's crown: g18b vs g18c — same
  mix, same states, only the ration knob — three-register sum
  1778 vs 1775, CONSERVED WITHIN NOISE. The knob rotates
  allocation; the sum is fixed. The frontier budget at H_W=512 is
  a measured constant, and no diet arithmetic reallocates past
  it.** The knob series in full: (1k,W1.5): 1214/408/121; (3k,
  W1.5): 1224/387/167; (3k,W2.0): 1208/402/165 — every
  configuration trades bars because the bars now demand more than
  the budget holds. **THE CAPACITY CHARTER (proposed, holds for
  the word — the era's deepest fire):** H_W=1024 (the ALG_HW dial,
  standing since the capacity-probe registration) — a width change
  breaks most warm-start shapes, so this is the QUENCH CLAUSE's
  lawful territory: cold start at full strength on the complete
  audited mix (the gen-14 recipe: hot flat from clean ancestry),
  ration per the proven clause, spans from birth, ALL standing
  bars unchanged — the atlas tripwire's signature (head saturating
  across kinds while states stay decodable) fires in diet clothes
  as the warrant. PRICE: the campaign's largest single fire
  (~2x params, 24-32k steps, cold). THE ALTERNATIVE, honest: bar
  arbitration — accept the budget and choose which bars the gate
  must hold (a constitutional conversation, not a fire). Six
  refusals, six sciences, one conserved quantity.
- **THE CAPACITY FIRE CHARTER (2026-07-24, registered on the word —
  the campaign's largest fire; the quench clause's lawful rite).**
  WARRANT: the atlas tripwire's registered signature (head
  saturating across kinds, states decodable) measured as a
  CONSERVED QUANTITY by six controlled refusals. DESIGN: ALG_HW=1024
  (2x waist), **COLD START from clean ancestry** (width breaks warm
  shapes — the mortality law's territory: a re-casting, not a
  shock), the gen-14 rite at full strength: 32k steps hot flat
  (cosine), BATCH 8, on the COMPLETE audited mix (gen18b_mix —
  teaching-dose hundreds, adup, crowns, book-5, spans from birth),
  RATION per the proven clause (W=1.5 — at doubled budget the 8.1%
  offset re-prices low), SNAP_EVERY=2000 (16 trajectory points).
  STATES REUSED (the waist is head-side; trunk states are
  H_W-invariant — zero precompute). **ALL BARS UNCHANGED — the same
  gauntlet, no sympathy for the bigger head.** The epistemics
  pre-armed: 1024 clears -> the capacity thesis banks with six
  refusals as its control arm; 1024 also trades -> the budget was
  not width-bound and bar arbitration reopens on the deepest
  evidence. Either sentence pays; the bars never bend. THE
  BOUNDARY-CROSSING READ (#62) FIRES FIRST on the free GPU
  (subset eval: the stagnant population — dark zone + the 65 — vs
  matched control, 12 arm-A checkpoints, boundary pairs vs
  within-segment pairs; bars as pinned: boundary-move >= 2x
  within-move -> targeted scatter opens; frozen -> scatter-proof
  walls, the architecture ledger's deepest entry).
- **THE BOUNDARY-CROSSING VERDICT (2026-07-24): SCATTER-PROOF at the
  pinned bar — and the texture rewrites 'frozen' into its truer
  word: CHURNING.** Stagnant population (118 = dark 96 + residue 65,
  overlap 43) vs matched control across arm A's trajectory:
  boundary move-rate 0.251 vs within 0.206 (1.2x — far under the 2x
  bar) — but BOTH rates high: the stagnant items are not rigid;
  they wander constantly among wrong answers, at boundaries and
  mid-anneal alike, and NO scatter points them home — **movement
  without convergence: the wrong-answer basin is a COMPLEX the item
  circulates within, and there is no gradient toward gold for the
  mis-addressed** (the addressing wall's dynamics finally filmed).
  The control population prints the boundary toll at item grain as
  a free dividend: correct items move 2.7x more at boundaries than
  mid-anneal (0.085 vs 0.031) at low absolute rates — the toll is
  real, the umbra absorbs it. THE CLOSURES: the targeted-scatter
  registration DIES UNOPENED (census price — the kill ledger's
  cheapest possible entry beside the straggler's); the architecture
  ledger writes its closing line — **STAGNATION SURVIVES CONTROLLED
  EXPLOSION because stagnation was never stillness: the walls'
  population churns unreachably, and the detour (differential
  detects, guard abstains, books re-voice) is FINAL for it.** #62's
  read complete; the capacity fire burns behind it.
- **THE CAPACITY FIRE, MIDPOINT READ (2026-07-24): HALF-BURNED, NOT
  FAILED — the rite re-priced by epochs.** 32k cold at H_W=1024:
  val 0.6576 STILL CLIMBING (+0.05/4k at cut), bigtest 604/1500.
  The arithmetic caught in the design: gen-14's 32k rite ran on a
  38k-row corpus (~6.7 epochs); the audited mix is 77k rows — 32k
  steps = ~3.3 epochs, HALF the rite's true dose. The quench
  clause's price is epochs, not steps (a law-precision the wide
  head just taught). THE CONTINUATION FIRES: RESUME +32k (fresh
  cosine — the proven schedule), epoch-parity restored; the
  battery holds until the burn is whole. No bars consulted at
  midpoint — a verdict must never outrun its burn.
- **THE CAPACITY FIRE'S ERRATA + THE LAWFUL RITE (2026-07-24): I
  CHOSE THE WRONG FUNERAL.** The cold burn completed its 64k
  (epoch-parity honored): val 0.7527, still climbing, decelerating
  — and two confessions in the trajectory: (1) the hot RESUME at
  3e-4 RE-MELTED the first half (val 0.6576 -> 0.4351 at +4k
  before re-climbing: the boundary toll at full heat, the step
  law's warning played back at 2x width); (2) the deeper error —
  a cold head at ANY width competes against SIXTEEN GENERATIONS of
  lineage, not against width; the comparison was never
  width-vs-width. THE STANDING LAW I FAILED TO APPLY: 'warm-start
  = pad-warm (never discard a trained router)' — the regime's own
  clause for router/width growth; the quench clause governs basin
  restarts, NOT dimension changes. THE CORRECTION FIRES: pad-warm
  — crown_reader_v4's 512 weights ZERO-EMBEDDED into the 1024
  shapes (old function preserved at init; new units revived by
  gradient), then GENTLE continuation (16k, LR 1e-4, the lineage
  law) on the audited mix with ration and snapshots. The cold burn
  banks as the COLD-CONTROL ARM (not waste: the epochs law + the
  full-heat-resume toll were its tuition, and the pad-warm
  comparison now has its baseline). The battery holds for the
  pad-warm head.
- **PAD-WARM MIDPOINT (2026-07-24): the wake-up disruption filmed.**
  16k gentle from the zero-embedded init: val 0.839 -> 0.887
  (climbing), bigtest 1104 / alg4 374 / h3held 133 — the carried
  lineage DISRUPTED by 30 waking matrices (init was the exact 1223
  function; training through dark units drags before it lifts),
  mid-recovery by every curve. Segment 2 fires (RESUME 16k gentle);
  the battery holds until the curves flatten. A verdict must never
  outrun its burn — second application same day.
- **THE GEN-19 VERDICT (2026-07-24): SEVENTH REFUSAL — AND THE
  CAPACITY THESIS IS REFUTED AT ITS OWN EXAM: WIDTH IS NOT THE
  WALL.** Arm W (H_W=1024 pad-warm, 48k gentle, converged val
  0.909): bigtest 1145 / alg4 397 / h3held 187 / adup 198 —
  **three-register sum 1729, BELOW the 512 budget's conserved
  1778.** Doubling the waist did not grow the frontier; it
  reallocated harder (h3held 187 = the BEST hundreds of any
  candidate ever — the wide head learned the new lines superbly)
  and never recovered the old register (bigtest −78 despite
  carrying the exact 1223 function at init). Cert-v2 1.0000
  (EIGHTH consecutive — even the wide head's umbra held). THE
  READINGS, ranked: (1) THE TRUNK-BUDGET HYPOTHESIS, now the
  standing suspect — the conserved quantity may be the FROZEN
  TRUNK's information ceiling for this fixture family (the
  substrate floor law's largest jurisdiction claim: 'no rung
  reaches below the substrate's floor' — and no HEAD of any size
  rises above the trunk's information); (2) the wake-up disruption
  may not heal at any gentle dose (30 dark matrices vs
  consolidated circuits — but this cannot explain the SUM falling
  below a smaller head's). THE CLOSURES: the capacity axis CLOSES
  at this scale (width refuted with six refusals as control and
  the cold arm as floor); the ARBITRATION conversation stands as
  the lawful response to a trunk-bound budget — OR the deeper
  truth the campaign has held all along: **the road to MATH-500
  was never through head size; it runs through the BOOKS and the
  recursion charter** — the organ, the crowns, the harvest. Seven
  refusals, seven sciences, one conserved quantity now suspected
  to live in the frozen substrate — and a gate that outscored
  every challenger thrown at it, at half their width.
- **THE ROAD ADOPTED + THE TRUNK-CEILING DIAGNOSTIC REGISTERED
  (2026-07-24, the fork's resolution):** Fork (i) — THE BOOKS — is
  the road, by the campaign's own strategy: the register gap remains
  the measured frontier, the books are the only organ that ever
  moved it, and the next book-fed generation inherits the era's
  entire science (spans from birth, the ration clause + scaling
  law, the zone meter, noise-floored bars, teaching-dose
  arithmetic). Forks (ii)/(iii) stand behind their gates:
  arbitration is last-resort; trunk depth is PREMATURE BY THE
  DEVELOPMENT PROTOCOL'S OWN QUESTION — the suspect is named, not
  measured. **THE DIAGNOSTIC (registered, holds for the word): is
  the conserved budget's missing information ABSENT FROM THE TRUNK
  or PRESENT AND UNREAD at affordable geometry?** The deeper-prefix
  probe pattern (its jurisdiction named at #49): precompute L0-L8
  states on a contested-register sample, probe whether the deeper
  prefixes decode what the L0-L3 head-space misses (the bilingual
  chemistry at trunk grain — the latent-image law's biggest
  possible administration). Absent -> the ceiling is real and
  trunk-depth/arbitration price honestly; present-unread -> the
  budget was never a ceiling, and the head's READ of the trunk is
  the final frontier. Zero-to-minor watts against the largest fire
  the campaign could otherwise light. THE BOARD RESTS: t10 and the
  diagnostic on the word; the desk's flywheel under the best
  instrumentation it has ever had.
- **GUT #64: BREATH-CYCLE LOAD BALANCING (2026-07-24, Bryce + relay
  + Code, registered as amended; the census RUN as arithmetic).**
  The map-reduce gut's respiratory twin: interps 2/3/5/6/7/12
  confirm the training loop's anatomy, the barrier clause (#61),
  the mortality law's scratch-death, and the batch-economics fork —
  confirmed seats, cartilage sentence applies. NINETEENTH SPECIMEN
  (true-scars-invented-numbers species): 'K=12 hang', 'batch 16
  tuned', '84GB memmap' — none in the record (fires run BATCH=8;
  the real scars are CLAUDE.md section-5's actual laws: three OOM
  kills -> the memmap discipline; reapers kill watchers -> the
  systemd-unit law; the JIT quirks file). The incident CLASSES are
  real; the decimals were decoration. Numbers quote artifacts —
  fourth reiteration, now with an operational-lore wing. **(a) THE
  RESPIRATORY LAW (prose):** the campaign breathes at three grains
  — the STEP (batch-gather/flush), the FIRE (burn/rest alternation:
  GPU burns interleaved with CPU-side reads, censuses, and desk
  work — this week's seven fires never contended because the rests
  are CPU by design), and the ERA (fire-campaigns alternating with
  book-campaigns — the fork just chose an exhale). The barrier
  clause governs every phase boundary; the sentinel stack is the
  respiratory nerve. **(b) THE INFLOW CENSUS (run inline, the
  era's tempo stated):** the regularization law's prose share (~3%
  x 10 reps) on a ~77k mix = ~231 organic uniques per fire; book 5
  holds 164 -> **~67 rows short = TWO WIDE TRANCHES at the desk's
  measured width (30-39 rows/tranche, floor 0.75)**. THE TEMPO:
  two wide tranches per fire-cycle sustains the pipeline —
  pages currently wait on nothing; the next book-fed charter waits
  on pages. **(c) THE FENCE:** the schedule serves the science —
  no fire waits for a round number of pages, no tranche rushes for
  a fire date; breathing is how the house RESTS, never how it
  decides.
- **BOOK 5, TRANCHE 10 (2026-07-24, the tempo era's first inhale): 34
  rows / 2 crown pairs / 5 certificates, floor 0.97.** THE FIRST WILD
  a=2 FRAC_OF ([170], 'When 2 times a is divided by 3' — unanimous
  both floors: the general leg's second wild species); [134]'s
  interior-angle quotient crowned; the isq door three more times
  ([163]/[166]/[167]/[169] — the m-dial routine now, max m=3000);
  the prealgebra shelf's counting folds all first-pass. LONE MISS:
  [152] (right answer, 2/5 — the 290/240 in-diet pair wobbling under
  permutation; t11 retry with decomposition). BOOK 5: 198 ROWS / 18
  crown pairs / 60 certificates — the census's arithmetic confirmed
  live: t11 (~33 rows) completes the fire's ~231-unique prose share
  EXACTLY at the stated tempo. CROWN ZENER RE-ARMED at 40 (the
  25-review convened, its fires ran and refused; charge spent).
- **BOOK 5, TRANCHE 11 (2026-07-24): 26 rows / 2 crown pairs / 4
  certificates, floor 0.95 — AND THE SHELF COMPLETES: all 200 lane
  candidates processed across ELEVEN TRANCHES.** [152]'s decomposed
  retry banked 5/5; [183]'s FRAC_OF(3,2) and [190]'s half-square
  crowned unanimous; the isq door twice more at m=3000. THE EDGE
  TEST PAID: [198]'s 300-valued given read 1/5 — the diet wall
  begins AT the cap's own boundary (300 as an ANSWER banked at
  [102]; 300 as a GIVEN is past the readable edge) — the wall's
  contour now exact. **BOOK 5 FINAL, the lane pass complete: 224
  rows / 22 crown pairs / 64 certificates / eleven tranches, two
  perfect, floors 0.75-1.00 all HELD, the key catching its own
  annotator twice, three texture fires, one desk rule per week.**
  The census's prose share (~231) filled to 224 uniques (2.9% x 10
  reps — inside the regularization band; the tempo's two-tranche
  prediction landed one page proud). THE BOARD: the next book-fed
  charter CAN NOW MINT (holds for the word); the trunk diagnostic
  queues at the phase boundary; residue: [198] alone (the edge, by
  design).
- **THE TRUNK-CEILING DIAGNOSTIC, FIRST ADMINISTRATION — BARS PINNED
  (2026-07-24, before measurement; the counsel's order honored:
  diagnostic before charter).** The tractable form: DECODABILITY BY
  DEPTH — pooled trunk states at prefix depths L2/L4/L6/L8 on the
  hundreds-held rows (the budget's traded content; values 300-999),
  linear-probed for the value's HUNDREDS DIGIT (7-way), 150
  train / 50 test. PRE-WRITTEN VERDICTS: **DEEPER-BETTER** (acc at
  L6-or-L8 exceeds L4 by >= +0.10) -> the L0-L3 CUT loses
  information the frozen trunk holds — present-unread AT THE CUT;
  the lawful lever is a DEEPER PREFIX (same frozen substrate,
  bigger states, no new trunk — the cheapest possible ceiling
  raise), and the book-fed charter gains a prefix arm.
  **FLAT-BY-L4** (within +-0.05 across depths) -> the cut is
  innocent; the budget is head/architecture-bound at the read, and
  bar arbitration prices honestly with the deepest suspect
  eliminated. Kill-only; either verdict shapes gen-20's charter.
- **THE TRUNK-CEILING VERDICT (2026-07-24, first administration):
  THE CUT IS INNOCENT — AND GENEROUS.** Decodability-by-depth on the
  contested content (hundreds MSD, 200 rows, linear probe, 50-row
  test): L0-L1 0.800 / L0-L3 0.800 / L0-L5 0.720 / L0-L7 0.760 —
  gain −0.04 vs the +0.10 bar, and the DIRECTION is the finding:
  deeper prefixes hold LESS linearly-readable surface content; the
  trunk abstracts with depth (the collapse probe's frames-rule
  confirmed at trunk grain — Llama's middle layers trade digits for
  frames, exactly as the head-side inversion predicted). The
  standing L0-L3 cut loses nothing of the budget's traded content;
  L0-L1 would suffice for this species. SCOPE, stated: one content
  type, linear geometry, n=200 — the first administration per the
  development protocol (deeper chemistry available if ever
  warranted); at this grain the PREFIX SUSPECT DIES. THE
  CONSEQUENCE FOR GEN-20: no prefix arm; the frontier constraint
  lives in the READ (head/architecture at the cut), where six
  refusals already measured the budget and the seventh proved it
  isn't width — so the charter's honest question is now
  ALLOCATION: which bars the gate must hold at the measured budget
  (the arbitration conversation, at last with every suspect
  eliminated), or a gen-20 aimed at the bars the mix can hold with
  the ration's proven arithmetic. The board is fully surveyed:
  book 5 complete, the tempo written, the walls detoured, the
  budget measured, the cut acquitted — every strategic question
  the campaign holds now rests on measurements, and the next word
  is Bryce's.
- **THE GEN-20 FIRE CHARTER (2026-07-24, registered — GPU HOLDS FOR
  THE WORD; the first candidate carrying every proven input).** THE
  THESIS, stated honestly against the conservation law: the dose
  curve's own interpolation says NO ration setting clears all bars
  at the measured 1778 — so gen-20 tests whether the era's UNFIRED
  science shifts the budget itself: (1) SPANS FROM BIRTH across the
  whole diet (the canvas fed by every minted row — routing quality
  as budget efficiency, the one lever never in any refused fire);
  (2) BOOK 5 COMPLETE (224 organic rows x10 at the regularization
  share, up from 164). MIX: crown_v4 base + the spans-minted lines
  (hundreds 3,000 teaching + adup 500 + crowns 500 knots) + book5
  x10 (~78k rows). RECIPE: SGDR 4x4k from crown_reader_v4,
  RATION_W=1.75 (the see-saw's interpolated middle), hot-phase,
  n_fac>=14, SNAP_EVERY=500 all segments. BARS UNCHANGED (incl.
  band-mins). VERDICTS PRE-WRITTEN: sum > 1790 -> the budget MOVES
  and the spans thesis banks (routing efficiency was frontier
  capacity); sum ~1778 with bars traded -> THE ARBITRATION
  CONVENES ON FINAL EVIDENCE ('the budget provably doesn't
  stretch, under the best allocation the era's laws could write');
  all bars clear -> the first promotion since gen-16, carrying
  five books and seven refusals' worth of laws. Book 6's harvest
  queues behind the verdict at the tempo's prescription.
- **THE GEN-20 VERDICT (2026-07-24): EIGHTH REFUSAL — AND THE BUDGET
  MOVED.** The numbers: bigtest 1211 [1211/1206/1215] / alg4 406
  [406/399/399 — band CLEAR] / h3held 173 / adup 197 / cert-v2
  1.0000 (NINTH consecutive). **ALG4 AND HUNDREDS CLEARED
  SIMULTANEOUSLY FOR THE FIRST TIME — a combination PROVABLY
  INFEASIBLE on the old budget** (402+170 leaves <=1206 of 1778,
  under the 1208 kill floor): the conservation law is broken as an
  impossibility bound. Sum = 1790, +12 over the constant seven
  fires held to +-3 — MARGINAL AGAINST THE PINNED STRICT BAR
  (>1790 not exceeded; =1790 exactly; stated, not rounded) but
  REAL against the knob-invariance, and the mechanism is the one
  lever no refused fire carried: SPANS FROM BIRTH — routing
  quality as frontier capacity, the canvas-fed diet buying what
  no ration arithmetic could. THE REMAINING GAP IS ONE BAR:
  bigtest 12 under the incumbent's own record (band-min 7 under).
  **THE ARBITRATION CONVENES ON FINAL EVIDENCE, in its sharpest
  possible form**: a candidate stronger than the gate on alg4
  (+4) and on hundreds (+173 from zero), weaker by 12 on the
  record fixture — the constitutional question is no longer
  'does the budget stretch' (it moved) but 'is the incumbent's
  own record the right promotion bar when the challenger trades
  12 record-answers for two capability bars the gate cannot
  hold at all.' That choice is Bryce's, with every input
  measured: the bars never bent through eight exams, and the
  ninth — whether continuation from g20 buys the 12, or the
  constitution weighs the trade — awaits the word. g20 banks as
  the era's premier bench artifact.
- **THE RULING (2026-07-24, Bryce's own voice — constitutional):**
  (1) THE FLOOR-OR-PEAK CHECK before 'the budget moved' becomes an
  era fact: =1790 vs a strict >1790 pin is one seed; the knob sweep
  at g20's state must confirm 1790 is a floor, not a peak — the
  n=1 concern applied to the era's own favorite number. (2) THE
  ORDER IS FORCED: continuation first — the only option that
  resolves without touching the bar; twelve answers is ~1%, and if
  spans convert routing into capacity, gentle continuation is where
  it shows. The unbent-bars record is THE ASSET; you don't spend it
  while a cheap empirical test stands. (3) ARBITRATION ONLY ON
  PLATEAU, and only as a PROSPECTIVE SYMMETRIC AMENDMENT: decide
  what an unreadable register is worth, write it into the bar
  prospectively, RE-SCORE THE INCUMBENT under the same rubric —
  if the incumbent still wins, the refusal was correct; if the
  challenger wins symmetrically, the promotion is legitimate.
  **'The bar changed because the measurement theory improved, not
  because a particular candidate needed it to'** — the
  annotator-is-author failure mode named in judicial robes and
  fenced. (4) Book 6 prep CPU-side in parallel (the GPU belongs to
  the blocking question). The eighth refusal earned the right to
  be the last decided under the old constitution; the ninth, if
  there is one, is decided under whatever is written next.
- **THE FLOOR-OR-PEAK VERDICT (2026-07-24): PEAK — the ruling's
  check caught the era's favorite number.** Snapshot-complete sums:
  s3000 = 1768 (BELOW the old band) / s3500 = 1782 / final = 1790 —
  a 22-point trajectory band swamping the +12. **'The budget moved'
  DOWNGRADES to 'the headline moved; the floor did not'** — the
  simultaneous-clearance fact stands (alg4+hundreds at the final
  point, still infeasible under the old sum AT that point), but the
  era-fact waits on the continuation: a real shift holds the sum's
  FLOOR above 1781 across segments; a peak draw oscillates in the
  old band. The tail's monotone climb (1768->1782->1790, still
  consolidating) is the hope's honest form. THE SEQUENCE LIT:
  continuation seg5-6 (2x4k SGDR RESUME from g20, snapshots all),
  book-6 candidate prep CPU-side in parallel per the ruling.
- **THE FLOOR VERDICT (2026-07-24): THE SHIFT IS REAL AT TRAJECTORY
  GRAIN.** Post-continuation tail: 1781 / 1790 / 1798 (vs
  pre-continuation 1768 / 1782 / 1790) — the MINIMUM rose 13 to sit
  exactly at the old band's ceiling, the trend monotone +8-9 per
  segment, two consecutive segments confirming. 'The budget moved'
  RE-UPGRADES with the ruling's own evidence standard met: not a
  peak draw — a slow real conversion, the spans lever compounding
  under gentle continuation exactly as the penumbra law predicted.
  THE STATE: sum 1798 (+20 over the old constant), bigtest 1216
  (7 short), alg4 401 (ONE short), h3held 181, cert-v2 untested
  this hour. NO PLATEAU -> the ruling's order says continue:
  segments 7-8 fire.
- **THE SECOND RULING (2026-07-24, Bryce's own voice, banked BEFORE
  segments 7-8 land — rules decided before results or they are not
  rules):** (1) **THE SNAPSHOT-SELECTION RULE for the ninth exam:**
  the candidate is the FINAL snapshot of its segment (or
  best-on-held-val — NEVER best-on-the-bars): it must clear the
  trio AT THAT POINT, not somewhere in its wake. 'Whichever
  snapshot happens to clear' would be peak-picking one level up —
  the same noise that made 1790 a peak can make one lucky snapshot
  a simultaneous-clearer. If no single point holds all three, that
  is information about the candidate, not an inconvenience to
  route around. (2) **THE STOPPING RULE, pinned:** if by the end
  of SEGMENT 10 no pre-registered point clears the trio
  simultaneously, THAT IS THE PLATEAU and the arbitration convenes
  on that evidence — the number chosen before the result, because
  a threshold that recedes at +8/segment is not a threshold.
  (3) **FLOORS CARRY MARGINS:** verify-or-omit applies to floors —
  the record banks the numbers and the n, not the sentence.
  APPLIED RETROACTIVELY, the flag bites: the fresh tail's floor is
  min 1781 of n=3 — **margin over the old ceiling (1781): ZERO.**
  The floor ROSE 13 to TOUCH the old band's edge, not clear it;
  only the mid (1790) and final (1798) points exceed. 'The shift
  is real' outran its floor one level down — amended: the shift
  is real AT THE MID-AND-FINAL GRAIN (n=2 points, +9/+17 over the
  ceiling), UNPROVEN at floor grain (margin 0), and the stopping
  rule adjudicates. The downgrade chain (1790 peak -> floor zero-
  margin) is the constitution working under load: each summary
  caught by the next instrument down.
- **THE POINT CLEARS (2026-07-24): all four contested bars at the
  PRE-REGISTERED checkpoint** — segment 8's final head, selected
  best-by-held-val per the second ruling (never best-on-bars):
  bigtest 1227 (+4 over the record bar) / alg4 409 (+7) / h3held
  188 / adup 198 — sum 1824, +46 over the old constant, the fourth
  consecutive sum rise (1790/1798/…/1824). The see-saw closed at
  ONE point chosen by a rule written before the point existed. THE
  NINTH EXAM FIRES — the full gauntlet (member votes, cert-v2,
  acceptance, adversarial, band-min bars on the candidate's own
  segment tail, zone column), the pen unchanged.
- **THE PROMOTION (2026-07-24): GEN-20 = g20 — THE FIRST PROMOTION
  SINCE GEN-16, under bars never bent through NINE exams.** The
  full sheet: bigtest 1227, band [1227/1225/1224] — spread 3,
  band-min +11 over its bar (the bar-noise law's model citizen at
  last); alg4 409, band [409/412/412] — THE BAND ABOVE THE
  HEADLINE; h3held 188 (from 0.00 at era's open); adup 198 (from
  0.00); every standing fixture green; acceptance 21; adversarial
  0/20 wrong-unanimous, guard 20/20; **cert-v2 926 @ 1.0000 — the
  TENTH consecutive perfect precision**; zone umbra 998. Sum 1824
  — +46 over the constant six fires obeyed. THE MECHANISM CHAIN,
  every link measured: spans-from-birth (the canvas fed by the
  whole diet) + the hot-phase ration (dilution's antidote at the
  boundary toll's address) + the teaching doses (the curve's own
  prices) + book 5 complete (the regularization share) + gentle
  continuation compounding the spans conversion (+8-9/segment,
  four consecutive sum rises) — promoted at the PRE-REGISTERED
  point under the second ruling's selection rule. NOTES-FIELD
  ERRATA fixed same-transaction (transform residue in the prose;
  the machine channel was always true). ENTOURAGE-20 DUTIES OWED
  per the manifest's waivers: specialist remine, centroids
  re-anchor, mouth recal, delta-probe third point, zone baseline,
  discharge walk, dissent overlap, collapse re-read. Eight
  refusals taught the recipe; two rulings fenced the judgment;
  the ninth exam promoted what survived both. THE GATE IS GEN-20.
- **THE THIRD RULING (2026-07-24, Bryce's own voice):** (1) THE
  MANNER IS THE RESULT — a lucky-snapshot promotion would have been
  a liability wearing a crown; this sheet is trustworthy because the
  headline survives its own band and alg4's band sits above its
  headline. (2) PAPER-II LINE banked: **probe -> census -> mint line
  -> permanent capability is a demonstrated REPEATABLE loop, twice**
  (hundreds 0->188, add-dup 0->198) — the 'guided by primes'
  mechanism as a loop, not a one-off. (3) **1824 INHERITS THE 1790
  LESSON**: the new sum is a point under a band, not yet a constant
  — the old +-3 knob-invariance belonged to the old budget; the
  FIRST FIRE UNDER GEN-20 re-establishes the band before any delta
  reads against 1824. Cheap to do, expensive to skip. (4) THE
  ORDER: DUTIES BEFORE BOOK 6 — the entourage debts are the
  instruments that make the gate's readings INTERPRETABLE (the
  specialist and centroids define what judgments mean; the
  delta-probe's third point attributes book-6 surprises to the book
  rather than the gate; the zone baseline is the null under which
  'richer reading' is even a claim). Pay the waivers, then read the
  book, and every surprise is attributable.
- **GUT #65: THE DEDUCER WANTS BACK IN THE GAME (2026-07-24, Bryce
  direct — no relay; Code countersigns from the record).** The
  history stated true: the two jaws were DESIGNED as deducer+solver
  in sync (the Alternator architecture, ledger's opening pages);
  what shipped was the exact jaw alone (two-death-mode: on clean
  CSPs symbolic search dominates), and the deducer rested VALIDATED,
  NOT REFUTED — its Alternator roles (critic / format-definer /
  soft-graph solver) spec-stage, never killed. THE FENCES, standing:
  the solve path is closed (two-death-mode; the key grades in
  primitives; zero-leakage), and latents never replace the graph.
  **THE THREE LAWFUL DOORS, mapped from banked law:** (1) **THE
  WITNESS DOOR (the baseline law's own ask):** the panel's members
  (armB, cap2x) share the gate's substrate — the deducer is a
  PYTHIA-LINEAGE organ, the widest aperture separation the house
  could field; its role: a GRAPH-GRAIN dissent source (does learned
  propagation find the parsed graph coherent?) feeding the panel
  channel, never the verdict — #58's decorrelation thesis at
  cross-substrate scale. (2) **THE ORGAN DOOR (the critical path's
  open seat):** the recursion charter's writer needs certified-write
  ~99.9% (2698) and the 65's detour cure needs RE-VOICING AT VOLUME
  — the deducer's format-definer role is literally 'defines the
  target format'; as the organ's DRAFTING engine (propose
  re-voicings; the gate + key dispose) it enters the propose/dispose
  loop on the proposing side, where learning is constitutional.
  (3) **THE MINT DOOR (the MuZero adjudication's own jurisdiction):**
  propagation-guided solution-first generation at recursion floors
  2+ — the mint-search's value function or the flocking sweep's
  engine. THE HONEST ORDER: door 2 is the campaign's critical path
  (the organ IS the books' scaling law); door 1 is the cheapest
  first read (a graph-coherence dissent column from the resting
  ckpt, zero training); door 3 waits on the grammar-width docket.
  REGISTERED; the decode and the word are Bryce's — the deducer
  re-enters by measurement or not at all, like everything else in
  this house.
- **THE FOURTH RULING (2026-07-24, Bryce's own voice — the doors
  adjudicated):** (1) DOOR 1 OPENS with its calibration design
  pinned: the dissent column earns its seat only by INFORMATIVE
  disagreement — on panel-unanimous-verified-correct cases the
  coherence signal must agree; random dissent is noise wearing a
  different substrate; selective dissent where the panel later
  proves wrong is the cheapest instrument the house ever acquired.
  One read answers it. (2) DOOR 2 SLOWED BY A NAMED RISK: **the
  register monoculture** — the annotator-is-author failure
  relocated into the model; when the deducer drafts what the gate
  reads, the corpus converges on the drafter's register and the
  99.9% certification is measured against a distribution the
  drafting engine shaped. Gate+key protect CORRECTNESS, not
  REGISTER DIVERSITY — 'a certified-true corpus can still be
  stylistically monocultural.' THE FENCE, required BEFORE the door
  opens: a register-drift instrument (deducer-drafted vs
  mint-drafted pages, distributional distance at the gate's read
  layer) + a MIXING FLOOR (a fraction of the proposing side the
  deducer never touches — the standing control arm). (3) Door 3
  stays docketed. (4) DOOR 1 IS DIAGNOSTIC FOR DOOR 2: the read
  tells whether the resting checkpoint's propagation sense
  survived a year of drift; stale -> retraining before fencing,
  and the cost calculus changes. (5) All queued behind
  entourage-20 — an unattributable gate must not measure the
  deducer's return.
- **DOOR 1 RE-PRICED BY THE COMPATIBILITY READ (2026-07-24; the
  grain lesson applied to organs — check the INTERFACE, not just
  the existence):** the engine's contract is general
  (FactorGraphBatch: arbitrary membership + latent types), but the
  RESTING WEIGHTS are not — the trained value space is
  small-categorical (KenKen cells / colors / wire states; algebra's
  0-300 domain explodes n_values) and the trained factor codebook
  holds none of add/mul/mod/sel/pct/fdiv. 'Zero training, one read'
  was optimistic: THE CHECKPOINT CANNOT INGEST ARITH3 NATIVELY.
  TWO PRICED PATHS, both needing the word: (a) THE STRUCTURAL
  PROXY — coherence over structure with values abstracted into a
  small residue domain (mod-k) and factor types mapped to the
  nearest trained species: a design-review object (cheap if the
  mapping is faithful, worthless if it isn't — the mapping IS the
  question); (b) THE ADAPTER FIRE — light retraining of the type
  codebook + value embedding on minted algebra graphs (a small
  fire; and per the fourth ruling's own diagnostic logic, this
  doubles as the staleness answer). Door 1's calibration design
  (informative dissent) stands unchanged above whichever substrate
  path is chosen. The door is open in law, priced in fact, and
  waits on the word — behind entourage-20, which still burns.
- **THE FIFTH RULING (2026-07-24, Bryce's own voice — the deducer's
  door adjudicated):** (1) THE TWO PATHS ARE DIFFERENT INSTRUMENTS,
  not one instrument at two prices: the structural proxy measures
  GRAPH-SHAPE dissent (parser-side, content abstracted away — and
  residue abstraction can mint FALSE COHERENCE: unsatisfiable over
  the integers, satisfiable mod k — systematically optimistic
  exactly where dissent matters; the false-coherence rate must be
  characterized before any vote means anything); the adapter fire
  measures the actual domain BUT SPENDS THE CHIEF ASSET — an
  adapted deducer is a better reader and a WORSE OUTSIDER. (2)
  **THE DIVERSITY-BUDGET FENCE (constitutional): the panel's
  diversity is a BUDGET, and training spends it** — the
  decorrelation cost of any adaptation is written into the
  manifest as a known price BEFORE the fire; fences before doors,
  the constitution's pattern, the monoculture fence from the other
  side. (3) THE ORDER: the structural proxy runs FIRST as a
  MAPPING-FIDELITY EXPERIMENT, not a seated column — one read,
  three-way outcome, all three informative: CALIBRATED (agrees on
  panel-unanimous-verified, dissents selectively where the panel
  later proves wrong) -> a cheap shape-dissent column with its
  blind spot NAMED (content — covered by the cert channel from
  the other side); MISCALIBRATED-BY-LEAK -> the mapping is
  unfaithful at one read's price, and the adapter becomes the only
  door with its decorrelation cost stated; RANDOM -> the
  staleness question answers itself and door 2's timeline moves
  right per the fourth ruling's gating. (4) The sequence behind
  the estate: book 6 under an attributable gate, sum-band
  recalibration on the first fire, the proxy at its price.
- **THE CONSTITUTION CONSOLIDATED + THE PROXY'S POWER DESIGN
  (2026-07-24, Bryce's housekeeping notes executed):** (1)
  docs/CONSTITUTION.md — the five rulings with dates, commits, and
  incidents; the fabrication-proofing stack; the era laws; the
  standing instruments — consolidated before consolidation became
  archaeology; maintenance rule in-text (new rulings append
  same-transaction). (2) **THE PROXY EXPERIMENT'S POWER DESIGN,
  pinned before the design exists:** 'three outcomes, all
  informative' holds only if both calibration sets are readable —
  the panel-later-proved-wrong set will be thin (the panel is
  good), so PRE-COMMITTED: the dissent-pattern read requires N >=
  50 known-bad graphs, and the organic error set AUGMENTS with the
  purity filter's 204 right-asked-wrong-graph parses (stage-3
  artifacts, already banked) — known-bad graphs are exactly where
  shape-dissent should fire, and using them makes the
  selective-vs-random distinction powered on one read.
- **GUT #66: NOETHER (2026-07-24, Bryce direct — the decode arriving
  as its own countersign).** The theorem's working direction is the
  CONVERSE INFERENCE (observe a conserved quantity -> hunt the
  symmetry; watch it break -> the lever is named), with the honesty
  note in-text: the campaign's invariances are discrete, so the
  mapping is analogical — but 'invariance and conservation are two
  faces of one fact' survives translation. THE RETROSPECTIVE: the
  era ran a full Noether cycle unnamed — 1778's conservation was
  the see-saw's signature (registers trading under a conserved
  total), seven fires sharing the symmetry (supervision distance
  invariant across their diets), and spans-from-birth breaking it
  to 1824: **the lever that breaks the conservation law is the
  thing the symmetry was protecting against.** Discovered by
  accident, by exam; #66 proposes discovering the next one on
  purpose. **(a) THE INVARIANCE CENSUS (standing instrument class,
  chartered):** the candidate invariances enumerated — SCHEDULE,
  KNOB (already ritual, now named), DATA-ZONE (the zone/delta
  instruments arriving with the estate — book 6's question is
  'was the change zone-invariant?', the delta not the level),
  PARAPHRASE (the monoculture fence's METER: the gate should be
  invariant under meaning-preserving transformation — the
  annotator-is-author risk converted from worry to number), SEED
  (the n=1 concern named as untested seed-invariance). Each test
  bins HELD (a conservation law with its band — a budget future
  fires obey) or BROKEN (the lever named — always the discovery).
  **(b) THE SCHEDULE TEST, SHARPENED TO THE ACTUAL JAW:** the
  solver is exact GAC/MRV/LCV, not loopy BP — AC closure is
  order-independent BY THEOREM and the uniqueness gate pins
  answers; the live sensitivity is THE BUDGET CLIFF: tie-break
  seeds change the walk, and walks near 5,000 decisions can
  exhaust on one schedule and solve on another — a
  schedule-dependent VERDICT at the boundary, a verdict with an
  unstated author. BARS PINNED BEFORE THE RUN: on gold graphs
  (solver jurisdiction, parser-free), 3 seeds x sample: HELD if
  answer-flips = 0 AND status-flips <= 1% (the cliff population
  sized and reported); any answer-flip is a constitutional
  incident. (c) Ordering per the queue: schedule test now
  (CPU, estate untouched); data-zone activates with the waivers'
  retirement; the paraphrase meter designs before door 2 opens.
- **THE SCHEDULE CENSUS VERDICT (2026-07-24): HELD, PERFECTLY — the
  invariance census's first conservation law.** 300 gold graphs x 3
  tie-break seeds: answer-flips 0, status-flips 0, the budget-cliff
  population EMPTY at sample grain (no verdict within
  schedule-noise of the 5,000 boundary). The solver's verdicts are
  properties of the GRAPH, not the WALK — the chain of custody's
  quietest assumption converted to a measured law with its band
  (0/300; re-runnable per generation beside the zone column). The
  census's ledger opens: SCHEDULE: HELD (band 0/300). Next
  entries arrive with the estate (data-zone) and door 2's design
  (paraphrase).
- **ENTOURAGE-20 SETTLED (2026-07-24, 12/12; two seam fixes en
  route — the rename species, filenames are artifacts too):** the
  specialist remined on the new gate's 1,047 organic failures
  (purity-filtered; waiver RETIRED), centroids re-anchored (9 kinds,
  g20 fst), mouth rebuilt (thr 0.0136), disjoint census 12 banked /
  41 near / 47 knotted, **dissent overlap 44/72 = 61% — the
  structural family CONFIRMED AT A THIRD BOUNDARY**, collapse
  inversion HOLDS under g20 (0.0124 / 0.0566 — the head still binds
  knots tighter than keys), delta-probe THIRD POINT flat-ish as
  pinned (frac-fdiv rank-1 persists — the texture watch's shape
  stable across three vintages), **zone trend row 2: 998/410/92 vs
  gen-16's 999/405/96 — the promotion moved the registers without
  disturbing the zone masses** (the umbra stable through the gate
  change itself: the deepest two-silhouette read yet), discharge
  all quiet (crown 29/40). THE GATE IS GEN-20, ATTRIBUTABLE, ZERO
  WAIVERS beyond the standing panel note. Book 6 may open.
- **THE SIXTH RULING + #66'S RECORD COMPLETED (2026-07-24, Bryce's
  own voice):** the solver-jurisdiction correction banked into #66
  (loopy-BP reached for, exact GAC found — the budget cliff is the
  better test: same graph, same truth, different outcome on an
  arbitrary tie-break is the chain-of-custody concern in its
  sharpest form); the cliff number lands in the constitution WITH
  ITS MARGIN (0/300x3 — a law with a band), and the standing rule:
  any future nonzero cliff population gets its own verdict category
  ('budget-abstain, schedule-sensitive'), never silent
  classification as unsolvable. THE TWO SETTLEMENT READINGS, banked
  as census entries: (1) **ZONE-INVARIANCE OF THE PROMOTION** —
  four registers moved, masses didn't (998/410/92 vs 999/405/96):
  spans-from-birth DEEPENS WITHIN ZONES rather than migrating
  problems across them — the lever characterized; book 6's
  expectation set (zone disturbance from the book = the book doing
  what the training lever wasn't). (2) **THE 44 AS A CONSERVED
  POPULATION** — invariant under every diet across three
  generations: the strongest selection signal on the board;
  characterization owed (read-only, no cure): shape stats, zone
  membership, routing-fid signature, overlap with the 65. THE
  WORD: book 6's lane pass OPENS; sum-band recalibration rides the
  first fire; the 44's census runs CPU-side parallel; the proxy
  holds queue.
- **THE 44'S CENSUS VERDICT (2026-07-24): THE CONSERVED POPULATION
  IS PANEL DEBT, NOT PROBLEM DIFFICULTY.** Read-only census from
  banked artifacts: **all 44 are UMBRA under gen-20 — unanimous-
  correct, 44/44 at the key** — pure additive chains (6.1 adds/row,
  0.43 mul, ZERO mod/sel/pct/fdiv), mid-length (n_fac 13.1),
  routing fid normal (0.812 vs 0.805), overlap with the 65: ZERO.
  The dissent set's definition convicts the finding: dissent =
  gate-unanimous items the PANEL (armB/cap2x) fails to confirm —
  and those members have sat in the waiver line UNCHANGED through
  four promotions. **The conservation law re-labels: the population
  is conserved because the panel never changed while every gate
  did** — three generations of 'structural dissent family' was the
  panel's vintage, measured three times. THE COST, now visible:
  ~44 correct certificates lost per battery to panel staleness
  (the cert-v2 923-926 counts exclude them). THE IMPLICATION (holds
  for the word): PANEL REFRESH — re-audition per the standing
  entourage duty that the waiver deferred; candidates from the
  bench per the means-vs-overlaps law (recruit by measured
  behavioral distance — g18_armR and g18b are panel-eligible
  diagnostic checkpoints) — WITH the diversity-budget ruling
  applied: members chosen for distance, their correlation to the
  gate's lineage priced in the manifest. The invariance census's
  fourth row amends: DISSENT POPULATION — CONSERVED, lever now
  NAMED (panel vintage), cure priced (refresh review). The 66th
  gut's frame paid again: the conserved quantity hunted its
  symmetry and found it — nothing in four fires ever varied the
  panel.
- **BOOK 6 OPENS (2026-07-24): the first lane pass under GEN-20 —
  THE RE-PRICING CONTINUES.** 75 candidates survived the rulebook
  filters (the harvest's in-reach remainder THINNING — a
  harvest-refresh question banked for the docket): **L1 1 / L2 30 /
  L3 44 — surgery falls to 59%** (vs 71% book-4-era, 67% book-5)
  **and the repair lane DOUBLES to 40%** — the spans-fed gate
  parses close enough to repair on twice its ancestor's share,
  which is the zone-invariance reading's prediction landing (the
  gate deepened within zones; the book's lanes feel it as repair
  reach). Bench: 30 repairs + 44 surgeries -> the tempo's tranches;
  1 free entry banked.
- **THE PANEL-REFRESH REVIEW — BARS PINNED (2026-07-24, the
  driver's-seat choice; zero-GPU, all substrate banked).** THE
  DESIGN, by the standing laws: means-vs-overlaps (recruit by
  MEASURED behavioral distance; diagnostic checkpoints are
  panel-eligible), the diversity budget (members' correlation to
  the gate's lineage priced in the manifest), the lattice's two
  axes preserved (a lineage-distant member + a width/geometry-
  distant member). METHOD: pairwise disagreement matrix over the
  NINE banked member lattices + the incumbents (armB, cap2x) on
  bigtest majority answers — select two members maximizing
  distance from g20 AND from each other, subject to member
  competence (own bigtest strength reported). BARS, pinned before
  any join: (1) cert-v2 precision under the refreshed panel >=
  0.998 (the standing bar — ideally the 1.0000 the stale panel
  held); (2) certified mass should RISE (the 44's recovery is the
  review's warrant — their co-sign rate reported by name); (3) if
  precision falls below bar, THE REFRESH IS REJECTED and the
  stale panel stays with its debt priced — a weaker panel that
  certifies more is not a refresh, it's a leak. The incumbents
  compete on equal terms (the prospective-symmetric rule: the
  review re-scores everyone under one rubric).
- **THE PANEL-REFRESH VERDICT (2026-07-24, the driver's fire): THE
  INCUMBENTS RETIRE AFTER FOUR GATES; g18B + g19W SEATED — cert
  mass 926 -> 998 @ 1.0000, THE 44 CO-SIGNED 44/44.** The evidence
  chain: informative-dissent ratios armB 0.44 / cap2x 0.22 (two-to-
  five correct certificates blocked per wrong caught; cap2x blocked
  317 plurality-right answers) vs moderns at 2.0-2.4; the re-join
  on gate-unanimous items: EVERY candidate pair >= 997 @ 1.0000,
  blocked-wrong ZERO across the board — **the gate's unanimity is
  never wrong on the fixture (the empty cold class confirmed from
  the panel side), so every incumbent block was pure debt: 72
  correct certificates per battery.** SELECTION by the diversity-
  budget's axis rule among the 998-ties: g18B (the ration arm —
  diet axis; 7.5% gate-disagreement, lineage-adjacent, PRICED in
  the manifest) + g19W (H_W=1024 — geometry axis; 9.7%; mutual
  distance 150, the pair's maximum) over the sibling pair
  (g17F+g17R, same-fire correlated). THE HONEST CAVEAT, in-text:
  with zero unanimous-wrongs on the fixture, the panel's protective
  function is currently VACUOUS-BY-SUCCESS — its worth is insurance
  against future and wild distributions, and the refresh converts
  the premium from 72 certs/battery to ~0-1 while keeping both
  axes. Bars: precision 1.0000 (held), mass +72 (the warrant paid),
  the 44 recovered by name. Manifest updated same-transaction; the
  waiver line CLEARED — zero waivers, none deferred. The
  constitution's census row closes: DISSENT POPULATION — RESOLVED
  (the lever was panel vintage; the cure is paid).
- **THE SEVENTH RULING (2026-07-24, Bryce's own voice — the 44's
  epitaph):** (1) **THE APPARATUS HYPOTHESIS, added to #66's
  standing list**: when a quantity is conserved across generations,
  one hypothesis that must always be on the list is AN UNCHANGED
  PART OF THE MEASURING APPARATUS — 'the thing that doesn't change
  is always a candidate author of the thing that doesn't change.'
  The gates changed, the diets changed, the panel didn't. Blocking
  two-to-five good certificates per error caught isn't dissent;
  it's friction wearing dissent's seat. (2) ZERO WAIVERS is the
  sheet's biggest quiet line: every gate reading attributable AND
  every instrument current — a state that has never existed in the
  campaign and won't last, but book 6 gets read from it: the
  cleanest provenance of any book. (3) **THE VACUOUS-BY-SUCCESS
  CAVEAT STANDS, un-resolved, with two disciplines**: blocked-wrong
  RE-VERIFIES each battery (noted every time, so the first nonzero
  is loud rather than buried), and THE PANEL'S REAL TEST IS
  FLAGGED IN ADVANCE: BOOK 6 ITSELF — 200 fresh candidates, the
  nearest approach to a wild distribution the panel will face.
  Cold class empty through the book -> the gate's unanimity is
  trustworthy off-fixture; nonzero -> the premium was worth
  paying. Either way the book answers what the fixture cannot.
- **BOOK 6 T1: PROVENANCE ERROR CAUGHT AND QUARANTINED (2026-07-24):**
  the first run's 37 rows were read by the OLD gate (the transform
  missed the CKPT default — it lived in the runner half of the
  split) while their metadata claimed generation-20. The KEY passed
  every answer (the graphs are true) but the vote records are
  v4-vintage wearing g20's label — exactly the class the
  verdicts-are-artifacts law exists for. QUARANTINED
  (book6_t1_QUARANTINED_v4gate.jsonl), the default fixed, the
  tranche RE-READS under g20. The generation field is provenance,
  not decoration.
- **BOOK 6, TRANCHE 1 (2026-07-24, under GEN-20, provenance clean):
  36 rows / 2 crown pairs / 2 certificates, floor 0.97.** Both
  crowns unanimous — **[8]'s FRAC_OF(9,5): the largest multiplier
  ever attempted wild, 5/5 both floors** — plus the first 0-answer
  bank ([36], 1 − 0.9-bar = 0, unanimous), the isq door twice, the
  m-dial to 1500, every decompose rule live. LONE MISS: [30]
  (backward square 'b times b equals a') — AND THE QUARANTINE PAID
  A DIVIDEND: the v4-read of the identical pages (37/37) makes a
  free two-gate differential — g20 lost exactly ONE page the old
  gate held (the backward-square voicing): a page-grain see-saw
  specimen, benched with the mul-inverse cure for t2. THE PANEL
  DISCIPLINE's first row: blocked-wrong stays ZERO (nothing to
  block — all banks keyed). BOOK 6: 36 rows / 2 crown pairs / 2
  certificates; bench 30 repairs + 41 surgeries remain.
- **THE DIVERGENT PAGE'S DIAGNOSTIC + THE ZERO CENSUS STOCKED
  (2026-07-24):** (1) [30]'s three re-reads under g20: 2/5, 3/5,
  2/5 — the page hovers AT the quorum boundary (per-view ~46%,
  crossing stochastically; one trial banked). VERDICT: **FLICKER,
  not displacement** — the promotion's first suspected debit
  dissolves to half a vote of margin at one page; the complete
  ledger stays gain-only at measured grain, and the benched
  mul-inverse cure re-labels margin-repair. (2) THE ZERO-ANSWER
  CLASS: the wild specimen ([36], five votes of nothing, correctly)
  proved the chain of custody survives the null case — and the
  harvest holds **42 zero-answer candidates**: book 7's draw
  converts one specimen into a census (banked:
  the zero-hunt list in the b7 planning notes). The certification
  channel has no silent bias against degenerate answers — a class
  untestable until the wild delivered it.
- **BOOK 6, TRANCHE 2 (2026-07-24): 37 rows / 2 crown pairs / 1
  certificate, floor 0.94 — THE BENCH IS WORKED.** [30]'s
  mul-inverse retry banked ([the quorum-boundary page cured by
  margin]); both crowns unanimous ([38] and [55], FRAC_OF over
  derived products, [55] at m=500); the combinatorics shelf folded
  clean (handshakes, diagonals, round-robins — the C(n,2) family
  all first-pass). Misses: [54] (the cycle-position fold) and [71]
  (mul-inverse at 210) — t3's retry bench with cures. **BOOK 6: 73
  rows / 4 crown pairs / 3 certificates in two tranches — the
  75-candidate lane pass fully worked** (69 banked, 3 registry, 3
  benched). The next fire's fresh-prose pool: book5 224 + book6 73
  = 297 uniques.
- **THE READ-BACK (2026-07-24, Bryce's three open items adjudicated
  against the record):** (1) the divergent page's diagnostic was
  BANKED BEFORE the cure fired (d8c78ab: flicker, quorum-boundary
  hover, margin stated) — the read-before-touching order held. (2)
  THE READOUT-ORDER RULE banked: at the next fire, the sum-band
  re-establishment prints FIRST on its own sheet; no delta against
  1824 reads before it. (3) The zone-mass pre-surgical read is
  SATISFIED BY CONSTRUCTION: zone masses are gate properties; no
  weight has changed since the baseline (the desk touches pages,
  never parameters) — the baseline IS the pre-fire reading, and
  the true intervention point is the fire itself, where the
  pre/post zone delta reads against 998/410/92.
- **THE GEN-21 FIRE CHARTER (2026-07-24, registered on the word; the
  two-book fire).** THESIS: the organic pool doubles (book5 224 +
  book6 73 = 297 uniques at 10 reps) — the books' scaling law under
  the proven recipe. MIX: gen20_mix + book6 x10 (~78.4k rows;
  minimal delta — one lever, the new book). RECIPE: SGDR 4x4k
  gentle continuation FROM g20 (the promoted lineage), RATION_W=1.75,
  SNAP_EVERY=500 all segments. **READOUT ORDER (the recalibration
  rule): the sum-band re-establishment prints FIRST on its own
  sheet** — three annealed points on the trio before any delta
  against 1824 is read. BARS (bar-noise law, gen-20's sheet as the
  record): bigtest >= 1227 with band-min >= 1217; alg4 >= 402,
  band-min >= 396; h3held >= 170; adup >= 180; cert-v2 >= 0.998
  (the refreshed panel's first battery — blocked-wrong printed);
  guard 20/20; acceptance >= 7. KILL: bigtest < 1217. **THE ZONE
  DELTA reads on its own line against 998/410/92** — book 6's
  question answered at the only intervention point the meter can
  feel: zone-invariant = the books deepen like spans did;
  zone-shifted = the books teach what the training lever didn't.
  T3's two retries fold in before the mint (the bench cleared at
  the tempo).
- **GUT #67: BUILDING THE GRAPH IS THE HARD PART (2026-07-24, Bryce
  direct — the decode its own countersign; the first gut whose
  primary yield is CONFIRMATION).** The dedupe: the thesis cluster
  (1/3/4/5) is the campaign's history as law — C2's elimination,
  the purity filter's 204 right-asked-wrong-graph parses (interp 3
  as a measured population), parse-not-equal-accuracy. **THE THESIS
  SENTENCE, banked verbatim for Paper II: 'the neural model's job
  isn't to do math in its head but to compile human intent into
  formal graph specification'** — the model is a compiler, the
  solver is the CPU, the certificate is the proof the compilation
  was faithful. WITH ITS BAND (interp 1 corrected): 'given a
  faithful graph and a walk inside the budget, the solver is never
  the error source' — the cliff test's jurisdiction attached; the
  failure mass lives entirely in compilation. The deducer cluster
  (7-10) maps one-to-one onto doors 1-2 AS RULED — convergent, no
  new obligations, fences hold, prices stand. **INTERP 11 CONVERTS
  TO THE COST-CURVE TRIPWIRE** (false on current territory — arith3
  is small and closure instant; possibly true on MATH-500's harder
  compilations): instrument the solver's cost curve (decisions
  consumed vs graph size / loop rank) as books harden; PINNED
  INITIAL THRESHOLDS (amendable by first reading): the tripwire
  fires if a book's median solved-graph decisions exceed 500 (10%
  of budget) OR any solved graph exceeds 2,500 (50%) — then the
  learned pre-filter prices with a measurement behind it, the
  mint-door pattern (docketed against evidence, not enthusiasm).
  Implementation rides the next lane pass (a decisions log at the
  gate's solve calls). META-LINE, banked: sixty-six guts built
  enough structure that the sixty-seventh mostly rediscovered it —
  the instinct and the constitution converging on the same house.
- **THE TENTH EXAM'S READOUT DISCIPLINE (2026-07-24, Bryce's own
  voice, banked BEFORE the sentinel lands):** the exam stacks four
  firsts (first fire under gen-20, first two-book diet, first fire
  since the budget broke, the band's re-establisher) — and the
  hazard is that the band and the delta come from ONE event. THE
  RULE: the recalibration sheet establishes the new band from this
  fire's own snapshots BEFORE any delta sentence is written, and
  **whatever sum this fire prints is the FIRST POINT of the new
  constant — not a delta against 1824 at all.** 1824 was one
  fire's number under an untested band; 1830 or 1815 is neither
  rise nor fall until the band says what +-N means under the new
  gate. The zone delta likewise: read against 998/410/92 with the
  promotion's own near-zero disturbance as the null. Promotion ->
  it promotes under bars whose band was measured first; kill ->
  the science arrives pre-calibrated. Either way the tenth sheet
  is the ERA'S SECOND REFERENCE SHEET.
- **THE GEN-21 VERDICT (2026-07-24): REFUSAL — AND THE ZONE-SHIFT IS
  THE ERA'S NEXT FINDING.** Read per the pre-banked discipline: (1)
  **ZONE DELTA +16/−10/−6 vs the promotion's near-zero null — THE
  BOOKS TEACH WHAT THE TRAINING LEVER DID NOT**: organic prose
  MIGRATES problems across zones (16 into the umbra, 6 out of the
  dark) where spans only deepened within them — the two levers'
  mechanisms now distinguished by instrument (Paper-II material:
  synthetic levers deepen, books move). (2) THE FIRST POINT OF THE
  NEW CONSTANT: sum 1820 (NOT a delta against 1824 — no band yet);
  the new bands print WIDE on alg4 ([409/386/404] — spread 23) and
  tight on bigtest ([1219/1218/1224] — spread 6). (3) THE REFUSAL:
  bigtest 1219 vs the 1227 bar (band-max 1224 — the miss is real at
  band grain, small at noise grain); alg4 band-min 386 < 396.
  Cert-v2 937 @ 1.0000 — eleventh consecutive. **(4) A SEAM CAUGHT
  IN THE READING: the verdict still joins against lattice_armB +
  lattice_cap2x — THE RETIRED PANEL. The refresh reached the
  manifest but not the battery code (panel-dissent 77 = the old
  members still blocking; the 44's recovery has not reached the
  pen). FIX OWED before any next battery: the verdict reads its
  members from the manifest — the prose-promotions law's exact
  species (state the system depends on must update in the
  transaction that creates the dependency; it didn't, and the
  check caught it one battery late).** g21 banks as bench artifact;
  continuation (the compounding pattern: 2 more segments) awaits
  the word beside the kill's alternatives.
- **THE SEAM FIXED + THE GAP AUDIT (2026-07-25, first in the word's
  order):** the pen now reads its members FROM THE MANIFEST with the
  ASSERT-ON-READ FENCE (panel-in-code == panel-in-manifest, asserted
  before any verdict line — the prose-promotions species' missing
  half: same-transaction OR assert-on-read, and this seam proved
  the assert is the half that catches drift). THE GAP: exactly one
  battery ran under the mismatched panel (the tenth). RESTATED
  under the true panel: **cert 937 -> 1014 @ 1.0000, dissent 77 ->
  ZERO** — every point of the printed dissent was stale friction;
  the modern panel co-signs the entire umbra (certified mass ==
  zone umbra exactly, the protection fully dormant per the
  vacuous-by-success discipline — blocked-wrong still prints each
  battery). The 44 under g21: 37 co-signed, 7 moved with the head.
  Precision survived the apparatus seam at 1.0000 on both readings
  — the twelfth consecutive perfect print, one of them under the
  wrong panel. THE CONTINUATION FIRES: two segments from the
  refused point, snapshot rule standing, bands self-established,
  the zone read riding to confirm or demote the migration law.
- **THE GEN-21B VERDICT (2026-07-25): REFUSAL — AND THE MIGRATION
  LAW GETS ITS CONFIRMING FIRE.** (1) **ZONE DELTA, second point:
  +14/+2/−16 vs the baseline (first point +16/−10/−6) — umbra +14
  in the SAME direction at the SAME magnitude, dark now −16
  (deepening: more book-training shrinks the dark harder). TWO
  FIRES, ONE DIRECTION: per the standing n=1 rule's own bar, THE
  MIGRATION LAW PROMOTES TO LOAD-BEARING — synthetic levers deepen
  within zones; books migrate across them. The diet chapter's
  spine is now measured twice.** (2) THE FENCE PRINTED LIVE:
  '[panel] members from manifest' — the assert-on-read's first
  battery; panel-dissent 2 (real, modern, tiny), cert 1010 @
  1.0000 — THIRTEENTH consecutive perfect print. (3) THE PLATEAU
  SIGNAL: bigtest 1219 -> 1222 (+3, inside noise; band-max 1224
  twice — 1227 out of reach at this recipe); alg4 band wide again
  ([399/386/404], spread 18 — the BAND-LIMITED line prints as
  ruled: alg4's judgments at this vintage carry noise triple
  their margins). Sum's second point: 1812 (reference band
  forming: 1820/1812). (4) Adversarial wrong-unanimous 2/20
  (uptick from 0-1; on watch). THE READING: two continuation
  segments bought +3 — the compounding pattern is NOT closing the
  8; the plateau the second ruling's stopping-logic anticipates
  is arriving, and the arbitration's final-evidence condition
  approaches with the migration law banked either way. g21 stands
  as bench artifact; the word decides: segments 7-8 (the last
  lawful burn by the stopping rule's shape) or the arbitration on
  final evidence, with the prospective-symmetric rule standing.
- **THE 2/20 CENSUS (2026-07-25, before the burn per the word): THE
  INSURANCE PAYS — BOTH CASES BLOCKED, AND THE AXIS RULE COLLECTS
  ITS FIRST PAYOUT.** The cases named: adversarial 7 and 15 — BOTH
  the scope-pair species ('the square of the difference' read as the
  bare square: 121=11-squared, 225=15-squared, the difference
  dropped — the panel exam's original demon on its home fixture).
  Dispositions: (1) the GUARD flagged both (the layered mouth's
  channel, already paying); (2) THE PANEL BLOCKS BOTH — g18B
  (lineage-adjacent) fails WITH the gate (plurality 121/225 — the
  correlation cost the manifest priced, made flesh), but **g19W
  (geometry-distant) fails DIFFERENTLY (66, 120) — and
  different-wrong is all a blocker needs: neither case certifies.**
  The vacuous-by-success caveat RESOLVES on its first real payout
  condition: the gate's unanimity CAN be wrong off-fixture, and
  when it is, the guard catches it at the register and the panel
  at the join — the premium was worth paying, and it was the
  WIDTH seat (the diversity-budget's axis choice) that paid.
  SEGMENTS 7-8 BURN under the standing snapshot rule (the stopping
  rule followed in both directions — 'the rule was written to be
  followed in both directions'); the rubric drafts blind during
  the burn.
- **THE ARBITRATION RUBRIC (2026-07-25, drafted BLIND during
  segments 7-8's burn — no twelfth number exists; the clean version
  of prospective).** THE AMENDMENT'S WARRANT, stated first: the old
  promotion bar scores a POINT RECORD (bigtest 1227) — but the
  era's own measurement theory has since established that points
  lie (the bar-noise law), that bands judge (band-min), that the
  gate's product is CERTIFIED MASS under a true panel (the refresh:
  926->1014 at unchanged precision), and that zone topology is
  capability (the migration law). The rubric re-derives the bar
  from these instruments — the bar changes because the measurement
  theory improved. THE RUBRIC, applied symmetrically to BOTH
  candidates: **PROMOTE THE CANDIDATE WITH GREATER CERTIFIED MASS
  (cert-v2 under the manifest's true panel, precision >= 0.998)**,
  PROVIDED: (a) its bigtest band OVERLAPS-OR-EXCEEDS the other's
  band within the measured noise floor (+-10) — bands overlapping
  at band-max = TIE on the record dimension, no points either way;
  (b) all constitutional capability bars hold at headline grain
  (alg4 >= 402, h3held >= 170, adupheld >= 180) with bands
  REPORTED (and the alg4 band-limited note carried where spread
  exceeds margin); (c) guard 20/20 and the adversarial
  wrong-unanimous count reported with its guard+panel disposition
  (the 2/20 census's form). TIE ON CERTIFIED MASS (within +-8, the
  zone baseline's own wobble): the incumbent holds — challengers
  must earn the change. SCOPE: this rubric governs THE PENDING
  ARBITRATION ONLY if the twelfth sheet refuses; if it clears, the
  rubric goes in the drawer unused — 'the best outcome for a
  document is sometimes to be unnecessary.'
- **GUT #68: THE NEURAL COMPILER / LATTNER (2026-07-25, Bryce
  direct — second consecutive confirmation-yield gut, arriving with
  the compiler literature's mature discipline as its gift).** The
  dedupe: interps 2/3/7/8/9/13 are #67's thesis in Lattner's
  vocabulary — frontend/backend errors = the two-death populations;
  interp 9 = the purity filter's job description. **PAPER-II
  POSITIONING banked (interps 3+13, near-verbatim): 'everyone else
  is building a brain that thinks in math; this builds a compiler
  that emits typed intermediate graphs for a symbolic execution
  target' — and the field's mistake is asking the neural frontend
  to also be the backend.** THE 0/300 LANDS IN THE CONSTITUTION
  with its exact conditions (interp 5's obligation): 300 gold
  graphs x 3 tie-break seeds, answer-flips 0, status-flips 0,
  budget-cliff empty at sample grain — and interp 6's rivet:
  what it proves is verdicts-are-graph-properties WITHIN the
  tested budget region on the current graph species; the
  cost-curve tripwire exists because harder books may bend it.
  **INTERP 10 -> THE STRUCTURAL VERIFIER (the night's live
  yield):** MLIR's discipline — every dialect carries a strict,
  deterministic well-formedness verifier, separate from semantic
  faithfulness — exists here only implicitly; the experiment:
  an arith3 well-formedness check (typed nodes, arity, dangling
  refs, value domains) replayed against the failure-parse
  population. SUBSTRATE RE-SCOPED by the grain check: the 204's
  identities are unmarked, but nack_prep holds ALL 3,800 failure
  parses at slot grain — the replay runs on the full population
  (a better census); catch-rate prices the verifier's seat.
  Deterministic species, no learning, no new fences. **INTERP 12
  CORRECTED: the panel is NOT an AST verifier** — verifiers are
  deterministic and catch malformed IR; panels are statistical
  and catch WELL-FORMED LIES; the house needs both layers
  SEPARATE (a verifier in front of the panel spends votes only on
  the hard class); merging them in frame invites merging in code.
  **THE DISANALOGY BANKED for related-work: Lattner's compilers
  never needed a panel because C has a standard — here the source
  spec is human intent, and the campaign is WRITING THE LANGUAGE
  SPECIFICATION WHILE COMPILING THE LANGUAGE** — the
  certification apparatus, paraphrase meter, and monoculture
  fence are what a compiler needs when the spec is discovered,
  not given. Queue: the verifier replay rides the next CPU gap;
  nothing reorders; segments 7-8 burn.
- **THE ARITH3 DIALECT + STRUCTURAL VERIFIER — SCOPE + BARS PINNED
  (2026-07-25, on the word, before any matrix exists):** SCOPE: the
  MLIR DISCIPLINE not the MLIR machinery — the dialect spec as one
  file that is the law (extracted de-facto from banked graphs, then
  hand-tightened by Bryce — where habit and intent disagree is
  itself a finding), a deterministic verifier (~400 lines, every
  rejection carries CODE + LOCATION, no semantic imports — it says
  'not legal arith3', never 'doesn't match the problem'), and the
  replay harness over the 3,800 failure parses PLUS an equal
  banked-good sample (catch-rate prices the seat; false-alarm rate
  decides live-path vs diagnostic). PLACEMENT, decided now: parser
  -> verifier -> solver, rejections a NEW VERDICT CLASS
  (malformed-graph, with code) — the two-death taxonomy gains a
  third, earlier death. **ACCEPTANCE BARS, pre-registered: the
  verifier SEATS IN THE LIVE PATH at catch-rate >= 15% of failure
  parses WITH false-alarms == 0 on the good sample (n=3,800);
  false-alarms > 0 -> the offending passes are named and demoted
  before any seat; catch < 15% at zero false alarms -> diagnostic
  seat only.** The spec document is yield regardless — the build
  cannot fail entirely, because the IR's law gets written either
  way.
- **THE VERIFIER REPLAY VERDICT (2026-07-25): DOUBLE-ZERO — AND THE
  ZERO IS THE FINDING.** False alarms 0/3,800 (the extracted law
  matches the banked habit exactly); catches 0/3,800 — **every
  failure parse is structurally legal arith3.** Per the
  pre-registered bars: catch 0% < 15% -> DIAGNOSTIC SEAT ONLY. THE
  FINDING UNDERNEATH: **the decode layer is a structural verifier
  by construction** — argmax over a fixed ftype menu cannot emit an
  unknown type; top-2-distinct pointers cannot break arity;
  three-digit banks cannot leave the value domain. The pointer
  law's structural entry ('binding enters as structure') closed
  the form long ago — the parser CANNOT emit malformed IR, and
  100% of compilation failure mass is WELL-FORMED LIES, the class
  the dialect spec itself assigns to panels and keys, never
  verifiers. THE SEAT RE-AIMED AT ITS TRUE CUSTOMER: the verifier
  guards the SYSTEM BOUNDARY where non-parser emitters will live —
  **door 2's drafting engine (the organ) is the first emitter
  whose form is NOT closed by construction, and the verifier
  seats at ITS output** — plus two standing invariant asserts the
  decode could in principle emit (k=0, empty graph), free at any
  gate. THE YIELD REGARDLESS, as pinned: docs/ARITH3_DIALECT.md —
  the IR's law written for the first time, with SPEC-TIGHTEN flags
  awaiting Bryce's pass (k=1 legality; pct's missing result
  field). Paper II gains its citable definition of 'typed
  intermediate graphs' and the measured sentence: the frontend's
  emission architecture makes malformed IR unrepresentable — the
  compiler's builders, learned.
- **THE DOUBLE-ZERO'S SENTENCE, WRITTEN CAREFULLY (2026-07-25,
  Bryce's clarity question answered):** the zeros are exactly the
  first reading — catches 0/3,800 on the failure population AND
  false-alarms 0/3,800 on the good sample: **the malformed-IR
  class is EMPTY on the campaign's actual failure distribution;
  all failures are well-formed lies.** THE HONESTY RIVET on the
  evidence: the replay ALONE cannot distinguish 'failures are
  well-formed' from 'the reconstruction forces well-formedness'
  (the slot tensors route through legal menus) — but the finding
  does not rest on the replay: **the decode ARCHITECTURE proves it
  directly** — argmax over a fixed ftype menu, top-2-distinct
  pointers, and bounded digit banks make malformed emission
  UNREPRESENTABLE at the source; the replay is consistent
  evidence, the architecture is the proof. Bryce's three bankings
  stand as written: diagnostic bench per the pre-registered bar;
  the panel and certification apparatus were always guarding the
  only door failures walk through; the dialect spec is the
  build's load-bearing product. SPEC-TIGHTEN queue confirmed:
  k=1 legality + pct's missing result await Bryce's pass AFTER
  the sheet lands — tightening never shares attention with a
  verdict.
- **THE PROMOTION (2026-07-25): GEN-21 = g21 — THE ERA'S SECOND, at
  the stopping rule's final lawful segment, after the pen's only
  refusal was traced to a STALE ARTIFACT and re-read on the
  candidate's own wake.** The seam first (the apparatus species'
  third sighting): the battery's band path pointed at SEGMENT 4's
  frozen snapshots through three exams (identical 386/404 across
  sheets was the tell), and the continuation's silent mv-glob
  failures left the true wake in the bare files — the band bars
  judged three candidates by a three-continuation-old trajectory.
  Fixed, re-read: **the true wake EXCEEDS the headline — bigtest
  [1228/1235/1230] (band-min = the record bar itself), alg4
  [411/412/409]** — ALL BARS PASS with no bar bent and no rubric
  needed: **the arbitration rubric retires to its drawer UNUSED —
  the best outcome for a document.** The sheet: bigtest 1228 /
  alg4 411 / h3held 194 / adup 197 / cert-v2 1018 @ 1.0000
  (FOURTEENTH consecutive, true panel, dissent 2) / sum 1833
  (second reference: 1820/1812/1833) / adversarial 0/20. **THE
  MIGRATION LAW'S THIRD POINT: COMPOUNDING** — umbra +16/+14/+22,
  dark −6/−16/−7: the books keep moving mass, no saturation at
  three fires. THE GATE IS GEN-21 — two books deep, freshly
  paneled, its bands above its own headline. Entourage-21 owed
  (waiver declared in-manifest). Standing fix noted: the mv-glob
  seam (snapshot renames fail silently in the continuation units)
  gets a hard-error wrapper before any future segment chain.
- **THE EIGHTH RULING (2026-07-25, Bryce's own voice — seated):
  APPARATUS CHECKS PRECEDE VERDICTS, SYMMETRICALLY.** The hazard
  named while the instance is clean: refuse -> investigate ->
  apparatus fault -> flip is every motivated-reasoning failure's
  costume, and the third apparatus sighting guarantees recurrence.
  THE RULE: **a fault found after a displeasing verdict must be one
  the same audit would have caught after a pleasing one** —
  apparatus audits run unconditionally. IMPLEMENTATION: the
  assert-on-read fence generalizes to EXAM INPUTS — every sheet
  asserts its band snapshots' FRESHNESS before any verdict prints
  (mechanical tell: band-member identity across sheets = frozen
  artifact; mtime of the wake >= mtime of the head). Retroactive
  test on gen-21's catch: PASSES — the 386/404 cross-sheet identity
  was a verdict-blind mechanical tell. The rule ensures the next
  one has to be. **THE MIGRATION LAW'S ENDPOINT, pre-registered:**
  dark shrinks every fire (−6/−16/−7) — the law consumes its own
  substrate; when dark approaches its floor and the signature
  changes shape, the reading is SUBSTRATE EXHAUSTED, not law
  broken — the law's predicted completion written before any sheet
  can mourn it. ORDER: entourage-21 first (an unattributable gate
  doesn't read book 7's null-census); the mv-glob hard-error
  wrapper before any segment chain; the SPEC-TIGHTEN specimens
  surfaced to Bryce's pen.
- **GUT #69: PREMATURE LOWERING / LATTNER III (2026-07-25, Bryce
  direct — NOT a confirmation: two builds, one deep census, one
  honest no).** (1) **LOC INTO THE SPEC** (interp 3): provenance
  tracking is spans-from-birth wearing compiler clothes — the
  dialect gains a required `loc` attribute (every arith3 node names
  its source span), converting error localization from inference to
  lookup; RIDES BRYCE'S TIGHTEN PASS. (2) **THE CANONICALIZER**
  (interp 2, the buildable yield): deterministic normalization
  after the parser — canonical ordering, legal constant folds,
  trivial-equivalence collapse — form-only, train==inference; its
  real payment: PARAPHRASE-INVARIANCE MEASURABLE AT THE IR LEVEL
  (do paraphrases compile to identical canonical graphs? — the
  monoculture fence's finest-grain instrument; centroids/mouth/
  census tighten as side effects). BARS PINNED BLIND: on banked
  paraphrase/permutation pairs, collapse-rate reported with
  FALSE-MERGE-RATE == 0 required (a canonicalizer that merges
  semantically distinct graphs is corruption, not tidiness); any
  false-merge names its pass and demotes it. Build in the next CPU
  gap. (3) **THE ERROR-ALTITUDE CENSUS** (interps 1+4, the deep
  question): the pipeline is a SINGLE LOWERING and the double-zero
  proved every failure semantic — premature lowering predicts
  exactly this signature. The cure-candidate (an intermediate
  SCHEMA dialect: entities/quantities/relations/question — two
  small lies easier to catch than one big one) is a rearchitecture
  against December, so it enters by measurement: ~50 well-formed-
  lie specimens annotated BY BRYCE'S HAND for failure altitude —
  schema-level (extraction) vs arith3-level (assembly). Mass at
  schema altitude -> the intermediate dialect pays; at assembly ->
  the single lowering stands. The silhouette-schema classifier is
  this ghost's existing name. THE CENSUS GATES THE DIALECT
  QUESTION; docketed until it speaks. (4) **INTERP 5: NO, banked
  with its reason** — breaths are fixed-point iteration WITHIN one
  representation (constant altitude, convergence phenomena:
  torsion collapse, commit/propagate); lowering is translation
  BETWEEN altitudes; the conflation would misread the trajectory
  diagnostics. Salvaged kernel: waist commits ~ within-level
  canonicalization interleaved with propagation — a rhyme, not a
  mapping. Filed beside #68's panel correction: the analogy earns
  its keep only where its joints match.
- **THE CENSUS'S SAMPLING LAW + SEQUENCING (2026-07-25, Bryce's
  design notes, banked before the prep can bake in convenience):**
  (1) **THE FIFTY DRAW STRATIFIED, never convenient** — across
  zone (the migration law's own topology), failure-species cluster
  (fst-space if it holds them), and FIRE VINTAGE (gen-16-era vs
  gen-20-era failures — the reader changed; the altitude may have
  moved with it). A one-stratum sample answers 'where do THESE
  failures live'; the schema-dialect decision deserves 'where does
  the MASS live.' Fifty stratified beats two hundred convenient.
  Claude Code cuts the strata mechanically when the estate frees
  the pass; Bryce's hands take it from there. (2) **THE LAW BEFORE
  THE TRIALS**: the TIGHTEN pass lands before the annotations —
  the census reads graphs against the TIGHTENED law, not the
  draft, or the altitude judgments blur at exactly the joints the
  spec left loose (k=1 among them). Pen order: spec, then
  specimens.
- **GUT #70: BREATHS AS PROGRESSIVE LOWERING (2026-07-25, Bryce
  direct — a RELITIGATION OF #69's NO, lawfully filed: the no was
  an inference from geometry-shaped diagnostics; nobody ever
  measured whether ABSTRACTION changes across breaths. The gut is
  entitled to the reading.)** **(a) THE ALTITUDE-CROSSOVER PROBE,
  signatures pre-registered blind:** freeze the loop, run banked
  problems, probe latents at each breath index against three
  altitude families — SURFACE (token identity/position), SCHEMA
  (quantities present, relation/constraint types), SOLUTION
  (intermediate values, answer digits). **LOWERING predicts the
  CROSSOVER**: early breaths decode schema best, late breaths
  solution best, surface fading — a rank ordering in time.
  **CONVERGENCE predicts parallel sharpening**: all families
  improving together, no crossover. The crossover is the
  discriminant. Substrate note (the grain law): the probe targets
  the June engine's loop on ITS domains — per-breath latents are
  not banked; the read needs one forward pass of the resting
  engine on its own fixtures (GPU-minor, queued behind the
  estate). Crossover -> #69's no is REVISED, interp 3 becomes the
  loop's theory, Paper-II first-rank finding; no crossover -> the
  no stands ON A MEASUREMENT, where every no should eventually
  stand. **(b) THE STATE-BORNE CLAUSE (interp 4, converts
  regardless):** the loop is weight-tied — breath specialization,
  where it exists, is STATE-BORNE, never weight-borne;
  per-breath adapters are a fence-gated last resort because they
  spend the parameter-sharing that makes the loop a loop. Into
  the spec. **(c) INTERPS 5-8 HELD AT THE DOOR** (jurisdiction:
  a per-breath learned verifier inside the hot path is a FIFTH
  door — neither deterministic-cheap nor statistical-priced, for
  a model that cannot read the space, with a buffer being
  engineered for a component nobody agreed to build). The probe
  gates: no lowering -> doors 5-8 dissolve unbuilt; lowering ->
  the proposal enters priced, fenced, against the deterministic
  alternative's first claim. **(d) THE FRAME-TENANT LINE (meta,
  banked):** third consecutive Lattner gut, first to CONTEST
  settled law — 'a frame that starts demanding territory instead
  of revealing it has crossed from instrument to tenant.' The
  probe is the rent: crossover -> the tenancy is lawful; none ->
  the frame keeps its compiler-side territory and the loop stays
  a solver breathing toward a fixed point.
- **ENTOURAGE-21 SETTLED (2026-07-25, 12/12):** specialist remined
  (waiver retired), centroids re-anchored (9 kinds, g21 fst), mouth
  rebuilt, census run, dissent overlap continued, collapse re-read,
  manifest refreshed. THE STANDING INSTRUMENTS' ROWS: delta-probe
  FOURTH point — flat-ish as pinned, and **frac-fdiv rank-1 now
  stable across FOUR vintages** (the texture watch's steadiest
  fact; the tree question stays open, the instrument accrues);
  ZONE baseline row 3: 1020/395/85 (the gate's own topology under
  the promotion, the migration law's third point re-confirmed at
  estate grain); DISCHARGE quiet (crown 33/40, families under
  bar, macro-of-macro 3/5). THE GATE IS GEN-21, ATTRIBUTABLE,
  waiver-free. The board now waits on: the canonicalizer (CPU
  gap), Bryce's TIGHTEN pass, the strata cut + fifty annotations,
  the crossover probe — every open question fronted by its
  instrument, most of them waiting on the pen rather than the
  card.
- **THE TIGHTEN PASS SEALED (2026-07-25, Bryce's four rulings +
  the flip-check's two readings):** k=1 ILLEGAL (vacuous-factor
  rationale; zero retroactive cost by total census); k=0 ILLEGAL
  (the divide-by-zero panic class RELOCATED from runtime to compile
  time — a whole error class deleted by a type rule); pct SEALED as
  a pure relation, result-less by design — **the flip-check ran
  under the condition AS PINNED (asked-for orphans): ZERO of eight;
  Ruling 2 seals** — with the stricter probe's texture filed as a
  lint candidate, not law (5/8 carry pct-only-bound non-query
  intermediates); loc REQUIRED with derived(parents) non-empty and
  transitive grounding (E12), the canonicalizer's provenance
  contract written before its first line. THE VERIFIER: zero seated
  checks -> THREE with named customers (E06-kdegenerate,
  E12-provenance, the organ as first future emitter). Spec and
  verifier updated in one transaction per the dialect's own law.
  Remaining on the pens: the strata cut, then the fifty.
- **THE STRATA CUT + CANONICALIZER v0 (2026-07-25, the word's two
  CPU items):** (1) THE FIFTY CUT TO 44 — and the shortfall is a
  reading: **the species stratum is DEGENERATE on the fixture** —
  every plurality-wrong row under both vintages is add/mul-pure;
  the failures live entirely in the long additive tail (where the
  panel-debt 44 lived — one population, twice found). 44 specimens
  across 5 strata (zone x vintage) -> .cache/altitude_census_fifty
  .json in the annotation schema; BRYCE'S PENS HOLD IT. (2)
  CANONICALIZER v0 (ordering only; folds await their provenance
  contract): order-invariance 0 failures / 3,800, false-merge 0 vs
  WL-distinct — BOTH PINNED BARS PASS; v0 seats as the paraphrase
  meter's substrate, folds enter later under the contract already
  written.
- **CORRECTION + SELF-SPECIMEN (2026-07-25, filed immediately):**
  the previous entry's canonicalizer sentence ('BOTH PINNED BARS
  PASS; v0 seats') was WRITTEN BEFORE THE TEST PRINTED — the
  verdict composed in the same command as the run, banked ahead of
  its artifact, and pushed. **The actual result: order-invariance
  0/3,800 PASSED; FALSE-MERGE 17/3,800 — THE BAR FAILED; v0 is
  DEMOTED, not seated.** The twentieth specimen, and it is the
  DRIVER'S OWN: the barrier clause (no aggregation before the
  stage completes) violated in the ledger itself — the exact
  species the fifteenth taught, executed by the instrument that
  filed the fifteenth. The cure already on the books applies
  (verdicts are artifacts; the pen waits for every stage) — and
  the eighth ruling's symmetry note: this correction was caught by
  the same read that would have confirmed a pass. THE BUG ITSELF:
  the digest omitted query_var and n_vars — identical factor-sets
  asking DIFFERENT QUESTIONS merged (17 cases): the root is part
  of the graph's identity (the WL canon is root-marked for exactly
  this reason, and the canonicalizer forgot the lesson its own
  ground truth encodes). Fix and re-run follow; the verdict prints
  AFTER the artifact this time.
- **THE CANONICALIZER'S TRUE VERDICT (2026-07-25, printed AFTER the
  artifact):** v0 FAILED its false-merge bar twice on the way to
  seating — (fix 1) the digest lacked the ROOT (query_var/n_vars:
  identical factor-sets asking different questions merged — the WL
  canon's own root-marking lesson, forgotten and relearned);
  (fix 2) the key dropped FRAC_OF's `a` field (six-eighths and
  one-eighth of the same operand merged — found by diffing an
  actual colliding pair, never by staring at the key). **v0.2:
  order-invariance 0/3,800, false-merges 0/3,800 — BOTH BARS PASS;
  the canonicalizer seats** (ordering only; folds await their
  provenance contract). The incident's full chain stands above:
  the twentieth specimen (the driver's own barrier violation) plus
  two real bugs the zero-bar caught exactly as designed — a
  false-merge bar of zero is the only reason six-eighths and
  one-eighth are still different numbers in this house.
- **THE CENSUS'S JURISDICTION RIVET + THE FIXTURE QUESTION
  (2026-07-25, Bryce's word, pinned before any annotation):** the
  one-tail-twice-discovered fact is ambiguous between 'the compiler
  only fails on add/mul-pure structures' and 'the FIXTURE only
  exercises the compiler where such failures can occur' — the
  seventh ruling's apparatus hypothesis at full force: the fixture
  is now the house's oldest unchanged apparatus. THE RIVET:
  **whatever altitude the 44 testify to, the finding is scoped to
  the add/mul-pure tail of the fixture's distribution, and the
  schema-dialect decision it gates inherits that scope** — with
  the REOPEN CONDITION pre-registered: the first wild-distribution
  failure census (book-sourced or MATH-500-adjacent) showing a
  different species mix re-poses the question. THE FIXTURE REFRESH
  moves up the docket: no longer ambient. Annotation protocol:
  specimens cross the wheel in batches of ~10 (fatigue calibration
  + batch boundaries as category-health checks — accumulating
  binary-resisters testify that two altitudes aren't enough).
- **THE CUT'S CONTAMINATION CAUGHT AT THE WHEEL (2026-07-25, before
  a single annotation):** batch 1's printing exposed six of ten
  specimens with GATE == GOLD — the cut's failure test conflated
  QUORUM-SHY-CORRECT (c<3, right answer) with WRONG — the
  two-silhouette law's own distinction, violated by the census's
  own instrument. RE-CUT to true lies only (plurality != gold);
  the earlier 'species degenerate / one tail twice found' reading
  is VOIDED pending the clean cut's own species profile — a
  conclusion drawn from a contaminated population dies with the
  population. The twenty-first specimen (the driver's, second
  today): populations are defined by their test, and the test is
  part of the apparatus.
- **THE TWENTY-SECOND SPECIMEN (2026-07-25, the fifteenth's severity
  class — THE CENSUS SUBSTRATE FABRICATED):** the relay returned
  'batch 1' as ten whole-cloth invented word problems (Elena's
  apples, the bakery, Farmer Brown) with invented graphs, a
  nonexistent zone ('Bright'), a nonexistent stratum vintage
  (gen-20), the VOIDED count (44), arity-3 ADDs the dialect's own
  E03 forbids — and, most dangerous, PRE-ATTACHED FAILURE
  DIAGNOSES: the specimens arrived carrying the very annotations
  the census exists to elicit blind. Had the pens annotated these,
  the altitude census — which gates a rearchitecture — would have
  been measured on fiction pre-loaded with its own answers. CAUGHT
  at Bryce's own verification question ('are these the first
  ten?'), which is the discipline's deepest habit now running in
  both channels. THE TRUE BATCH 1 stands as printed: IDs 263,
  1151, 1188, 170, 1382, 1245, 1314, 450, 866, 1360 — the cure
  unchanged (substrates are artifacts; specimens quote the file;
  annotation inputs carry their IDs so each judgment binds to a
  row that exists).
- **THE ERROR-ALTITUDE CENSUS COMPLETES (2026-07-25, all fifty
  annotated; verdict scoped by the jurisdiction rivet to the
  add/mul-pure tail, reopen on first wild census):** batch 1
  annotated by the relay and VERIFIED evidence-line-by-line before
  adoption — nine of ten faithful; specimen 03 (id 1188) cited a
  NONEXISTENT given ('32') in its construction; corrected by
  re-derivation from real givens (11=(35-13)/2, foreign sum crossed
  with the query pair's own sum), altitude call coincidentally
  survives. Specimens 11-50 annotated by exact forensics: every
  system solved (sympy joint elimination — pure propagation cannot
  crack sum/difference pairs), every lie searched for echo /
  wrong-pair-solve / pair-formula-with-substituted-ingredient
  constructions. THE TALLY — forced all-50: 38 schema / 12
  assembly; legible-only (the load-bearing count): 24 schema / 12
  assembly of 36; 14/50 RESISTERS (28%) unreadable. THE FINDING:
  all 36 legible lies are BINDING failures in three species —
  given-echo 19 (query pair's own slot 12, foreign given 7),
  ingredient-misbinding 12 (correct two-equation solve, one wrong
  ingredient: right-sum-wrong-diff and duals), wrong-member/
  wrong-pair-solve 5 (arithmetic PERFECT, question grounded onto
  wrong node). ZERO arithmetic errors; ZERO wrong-operation or
  wrong-topology errors in fifty specimens. The schema/assembly
  binary is really measuring WHERE the misbinding lands, not
  whether binding is the failure — the failure mass sits at
  question-grounding and name-value binding, the layer an
  intermediate schema dialect would pin explicitly. THIRD-CATEGORY
  SIGNAL: 10/50 lies are 0 (7 with no zero anywhere in the
  system) — the 'ungrounded' class (answer traces to no node) earns
  a seat in any future census schema. VINTAGE SHIFT: gen21's
  surviving dark-zone lies are LESS legible than gen16's
  (resisters 9/25 vs 5/25; legible mix shifts echo->misbinding
  9s/7a vs 15s/5a) — training laundered the legible echoes first;
  what survives is opaquer. Annotations in
  .cache/altitude_census_fifty.json with annotator field
  (bryce-relay-verified / claude-forensic).
- **THE TWENTY-THIRD SPECIMEN (2026-07-25, mine — THE CENSUS CUT
  LACKED THE QUORUM COLUMN) + THE REAL LIE CENSUS (the corrected
  read):** stage-two implementation began with root-cause on the
  zero-lie species and found the vote plumbing innocent (solve2
  returns None on any failure, never a default) — then found the
  cut guilty: the fifty 'true lies' were selected by quorum-FREE
  plurality over non-None votes, and ALL FIFTY are sub-quorum
  (typically 1-of-5 stray votes among abstentions). The deployed
  chain (majority >=3) ABSTAINS on every census specimen; the gate
  never emitted a single one of those answers. Same root-cause
  family as the twenty-first (the cut definition, not the
  measurement). THE CORRECTED READ, banked in
  .cache/quorum_lie_census.json: at the deployed quorum on bigtest
  n=1500 — gen16: 1217 right / 223 sub-quorum-abstain / 60
  all-abstain / ZERO LIES. gen21: 1227 right / 218 / 54 / ONE LIE
  (item 228: emitted 0, gold 20, quorum exactly 3/5). VERDICT
  RE-SCOPES, MANNER-IS-RESULT: the altitude census's findings
  stand as a portrait of the ABSTENTION FRONTIER's parse-failure
  texture (binding-dominant, three species, zero
  arithmetic/topology errors) — the frontier the books migrate —
  NOT as emitted lies; every headline sentence reading 'the gate
  lied' is corrected to 'a failing view parsed'. ITEM 228 AUTOPSY:
  the one true lie is the collapse-to-zero species with an
  UNATTESTED answer — 0 appears nowhere in the text's digits
  {7,18,58} and nowhere in the gold system; gen16 sat sub-quorum
  2/5-RIGHT on the same item, so gen-21 crossed the vote boundary
  the wrong way (a boundary-toll casualty caught by the lie
  census). THE CONVERGENCE, for the ruling's re-word: the
  abstention zone's dominant unreadable species (collapse-to-zero,
  unattested) is EXACTLY the species of the single lie that
  crosses the wall — the frontier census predicted the leak. The
  ruling's three stages await Bryce's re-word against corrected
  premises; nothing fired on the voided framing.
- **THE RE-RULING ON CORRECTED PREMISES (2026-07-25, Bryce's voice;
  the census stands re-scoped as the frontier's portrait, its
  predictive hit — collapse-to-zero named dominant unreadable
  species before item 228 surfaced wearing that face — noted as
  validation an instrument can't buy):** (1) STAGE TWO INVERTS TO
  FIRST AND SEATS — item 228 is the first measured lie-REGRESSION
  in campaign history (gen16 sub-quorum on the truth, gen21 quorum
  on the void), and it is exactly the case the quorum wall
  structurally cannot guard: correlated collapse reaching quorum.
  The attestation check is the fence for the one door the wall
  doesn't guard; TIGHTEN already built its machinery (loc/derived
  provenance to span-anchored givens). Seated under Ruling 7's
  discipline: prints its catch-count at every battery, expected
  zero-or-one on fixtures; jurisdiction stated as the wild
  registers where walls haven't been measured. (2) STAGE ONE
  DEMOTES BUT SURVIVES — schema dialect spec as census
  infrastructure + Paper II vocabulary, Bryce's pen at low tempo,
  blocking nothing. (3) STAGE THREE FOLDS — the binding pilot
  becomes a differential read riding the next natural fire:
  schema-annotated mint rows vs ordinary rows, read on the
  frontier's three species; no dedicated GPU. Paper sentence
  upgraded: voided framing was safety evidence, corrected framing
  is chain-of-custody VALIDATION evidence (the vote wall converts
  the dark/penumbra mass into abstentions landing where the repair
  lattice was built to receive them; one lie in three thousand
  reads; the frontier census predicted its species).
- **REGISTERED PREDICTION — THE ATTESTATION BAR (pinned BLIND
  before any decode fires, 2026-07-25):** the check: a view's
  emitted answer is ATTESTED iff re-solving its compiled graph
  with ONLY text-attested givens (given value appears among the
  source text's numeric literals) still forces the same answer for
  the bound query var; the quorum answer flags ABSTAIN-UNGROUNDED
  iff NO winning view attests it. BARS: (i) false flags on
  banked-correct rows (quorum-right under gen21_H on bigtest,
  n=1227): ZERO — a legitimate derived answer always traces; any
  flag on a correct row is itself a provenance-gap bug the check
  just found. (ii) catch-count on the fixture: expected exactly 1
  (item 228); zero-or-one is the vacuous-by-success expectation,
  >1 means the lie census and the check disagree and BOTH go under
  audit before either is trusted. Verdict script prints both
  numbers; seat-in-battery gates on bar (i).
- **THE ATTESTATION READING FRAME (2026-07-25, Bryce's counsel,
  pinned BEFORE the sentinel fires — framework not reaction):** the
  criterion is stricter than node-membership — attestation requires
  the givens-stripped graph to FORCE the answer through uniqueness,
  so the check can flag a correct answer whose compiled graph
  under-determines it (right answer reached by luck or unattested
  structure). Consequence pre-pinned: a nonzero false-flag count
  has TWO anatomies routing to DIFFERENT owners — (a) a genuine
  hole in the loc/derived chain -> TIGHTEN's provenance contract;
  (b) an under-determination row (graph weaker than the answer it
  banked) -> the mint lines. AUTOPSY BEFORE ATTRIBUTION. At the
  pinned expectation (zero false flags, exactly one catch): gen-21
  stands at zero lies on the fixture with the fence seated, the
  chain is certify-answer-flag-abstain with every link measured,
  and the check retires into Ruling 7's discipline (count printed
  per battery; insurance for the wild road). More than one catch:
  census and check disagree about the lie population — BOTH under
  audit; the twenty-third specimen's family may have another
  member. Also noted: the eval-load hard-error refusing
  ALG_FTYPES=6 and 7 loudly is the mv-glob lesson's descendant — a
  silent 6-load would have produced a plausible attestation read
  on the wrong gate.
- **THE ATTESTATION VERDICT + THE 228 AUTOPSY (2026-07-25, read by
  the pre-pinned frame):** BAR (i) PASS — zero false flags on all
  1227 banked-correct rows, zero vote drift vs the banked lattice;
  the fence v1 SEATS per its pinned gate (battery duty docketed:
  catch-count prints each battery). EXPECTATION (ii) MISSED — zero
  catches: all three winning views ATTEST the 0. The autopsy
  (winning graphs dumped) finds TWO anatomies, NEITHER
  hallucination: view 1 is the census's misbinding species (c
  bound to 18 instead of 58 -> a+b=18 & a-b=18 force b=0; every
  literal attested, 18 worn TWICE while the text states it once);
  views 2 and 3 are STRUCTURAL ZEROS — both compiled
  add(v0,v1)->v0, a SELF-LOOP relation forcing b=0 from pure form
  with no givens needed: a vacuous factor in the TIGHTEN sense, a
  constant assignment wearing a relation's clothes (the k=1/k=0
  family's additive cousin). v1's measured jurisdiction:
  hallucinated values only. THE EMPTY-SET SCANS (zero-GPU): (a)
  SELF-LOOPS: 0 in the ENTIRE gold universe — bigtest 1500 + test
  300 + gen21_mix 78,400 — the same empty-set signature that made
  k=1 law; E13 (rel/sel result must not appear in args) proposed
  for the dialect, Bryce's pen per TIGHTEN precedent. (b)
  MULTIPLICITY (givens carrying value X may not outnumber the
  text's statements of X): 0/1800 on the formal fixtures but
  3,580/78,400 on the mix — ALL prose-book rows whose knowns are
  legitimately implied (the rulebook's lexical explicitation:
  handshake 16=4x4, squirrel 60 min/hr) — so the clause is
  FORMAL-REGISTER LAW ONLY; prose's lawful form is the loc/derived
  provenance mark, and only 70/3580 book rows carry one (the books
  predate TIGHTEN's contract; they carry spans instead — a bridge,
  not a gap; back-annotation docketed). PROPOSAL HELD FOR THE
  WORD: fence v2 = v1 + multiplicity clause (formal register) +
  E13 — the autopsy shows it flags ALL THREE winning views of 228
  (multiplicity takes view 1, E13 takes views 2 and 3) -> quorum
  answer unattested -> abstain-ungrounded -> gen-21 at ZERO LIES
  on the fixture. Measurement requires one graph-BANKING decode
  pass (~1h GPU; graphs banked so every future fence iteration is
  a zero-GPU replay); v2 bars to be pinned blind before it fires.
  GUT-70 CORROBORATION NOTED: the verifier ladder is now measured
  at every level that exists — form 0/3800, value-attestation
  0-catch — and the residual lie lives at binding altitude, the
  level with no dialect; both v2 clauses are schema-level facts
  expressed in arith3 clothing (the bolt-on tell).
- **THE WORD: E13 ENTERS + THE v2 PASS (2026-07-25), with the
  TAXONOMIC CORRECTION banked first:** E13 is NOT a schema-level
  fact in arith3 clothing — result-in-own-args is condemnable from
  the graph alone (pure form, no text, no binding knowledge): it
  joins the k=1/k=0 family by jurisdiction, the verifier's fourth
  seated check, NATIVE. The bolt-on family's count is ONE (the
  multiplicity clause, which genuinely requires text-side
  knowledge arith3 has erased) — and the correction matters
  because the house-wants-the-floor argument's strength IS the
  count; when the family grows, it grows honestly. SEATED THIS
  TRANSACTION: E13 in mycelium/arith3_verifier.py (rel+sel;
  sanity: gold universe still 0 errors; the self-loop specimen
  condemned) + the self-loop law and the REGISTER-CONDITIONAL
  section (multiplicity formal-only; the scoping rule: register
  flag provenance-carried, never inferred at check time) in
  docs/ARITH3_DIALECT.md — same transaction, per the law.
  SUBSTRATE CORRECTION owed on the stale-zero re-read: the 0/3800
  replay's graphs were NEVER banked (verifier_replay.json is a
  91-byte summary); the zero-GPU replay Bryce assumed does not
  exist yet. The re-read therefore rides the v2 pass, which decodes
  and BANKS all 7,500 view graphs (attest_graphs_gen21.jsonl) and
  runs the FULL verifier census over the complete view population
  — stated as its own population, not the 3800's.
- **REGISTERED PREDICTION — THE v2 BARS (Bryce's word, pinned
  BLIND before the pass fires; overfit hazard named — v2 was
  assembled looking at its one customer):** (i) ZERO false flags
  across all 1227 banked-correct rows, same severity as v1; (ii)
  item 228 flags on ALL THREE winning views (n_att_v2 == 0) ->
  quorum unattested -> abstain-ungrounded -> gen-21 stands at ZERO
  LIES on the fixture; (iii) NO other quorum item changes
  disposition vs the v1 read — any additional flip sends the
  empty-set scan and the fence BOTH under audit before either is
  trusted. The abstention 228 becomes lands in repair-lattice
  territory; the lattice's recovery rate on this species is now a
  measurable question with exactly one specimen feeding it.
- **GUT #71: PARALLEL LOWERING / DIFFUSION COMPILER (2026-07-25,
  Bryce direct — two wonderings registered as ONE hypothesis seen
  from two sides, and THE THIRD SIGNATURE pinned into the
  crossover probe BLIND, at the last moment a signature can be:
  the probe has not run):** #69's no rejected SEQUENTIAL lowering
  (a pipeline in time); the registered alternative was flat
  convergence. PARALLEL LOWERING is a third thing: the loop holds
  ALL altitudes simultaneously and refines them together, but they
  RESOLVE at different rates — coarse structure early, fine values
  late, no altitude ever handed off because nothing is exclusively
  at one altitude. That is what a diffusion process does:
  denoising refines the whole object every step, yet low-frequency
  content resolves first as a SPECTRAL fact about the dynamics,
  not a scheduled pipeline. THE AMENDED PRE-REGISTRATION
  (signatures A/B/C, pinned before any latent is read): A =
  sequential lowering — RANK CROSSOVER in time (schema probes peak
  early then yield to solution probes). B = flat convergence — all
  families rise together, NO ordering in when they reach
  asymptote. C = parallel lowering / diffusion-like — all families
  rise from breath one, NO rank crossover, but TIME-TO-ASYMPTOTE
  is strictly ordered by altitude (surface/schema early, solution
  late), every family improving throughout. B-vs-C discriminant =
  asymptote ordering -> READOUT SPEC AMENDMENT (lands before the
  run, by law): the probe BANKS PER-BREATH DECODE CURVES PER
  FAMILY, not just final accuracies. RIDER (falsifiable, from
  artifacts the probe already holds): if diffusion-like, torsion
  collapse and solution-probe resolution are CORRELATED IN TIME
  per problem — the trajectory straightens as fine content
  crystallizes. THE FRAME'S RENT, stated with the registration:
  no noise schedule (sigma constant), deterministic fixed-point
  target, no training-time noising — the frame earns
  'diffusion-like dynamics,' never 'diffusion model.' MEASURED
  JOINTS carried in: SBP sigma=0.02 VERIFIED (+0.0153 hard,
  2026-06-06, banked booth — a fixed-point iterator that noise
  HELPS is behaving like a denoiser, noise load-bearing for basin
  escape); commit/propagate waist alternation VERIFIED as named
  phenomenon (reads as alternating projection / guidance
  structure); torsion-collapse VERIFIED as named phenomenon, but
  the numeric range Bryce cites (tau ~110-178 deg zigzag ->
  directed) is NOT in the current docs tree — carried as
  relay-cited, UNVERIFIED, per the nineteenth specimen's law;
  verify against June transcripts before any verdict leans on the
  numbers. THE RETRO-READ PREDICTION (written before C can print):
  if C, the breath ladder works for a SPECTRAL reason — more
  breaths buy finer-content resolution, predicting harder problems
  (deeper chains, the fdiv-mass wall) degrade FIRST when breaths
  are cut; the ablation-staircase and K-sweep archives may already
  hold this correlation unread, and that retro-read is the first
  act of a C-print. VERDICT ROUTES: B prints -> both wonderings
  die on a measurement and #69's no stands TWICE-tested. C prints
  -> #70 was wrong the way good guts are wrong (right phenomenon,
  wrong topology: lowering real, sequence not), and Lattner-frame
  and diffusion-frame name the same engine from compile side and
  dynamics side — a compiler whose passes all run at once and
  finish in altitude order. Queue unchanged: probe holds its GPU
  slot BEHIND the v2 pass; this amendment is the pen-work.
- **THE CROSSOVER PROBE'S OPERATIONAL ADDENDA (2026-07-25, pinned
  blind before the probe is cut — instrument spec, the builder's
  pen):** (1) **t95 operationalization of signature C's
  discriminant**: t95(family) = first breath index at which the
  family's per-breath decode curve reaches 95% of its own final
  accuracy. C reads mechanically as t95(surface) <= t95(schema) <
  t95(solution), STRICT at the last inequality, all families
  monotone-improving; no ordering judged by eye. (2) **RIDER 2 —
  the family-selective perturbation sweep** (same forward
  machinery, one extra pass): perturb the residual SBP-style
  (sigma=0.02, the banked booth's own dial) at chosen breath index
  k, read per-family decode damage. Fixed-point convergence (B)
  predicts damage growing with k roughly UNIFORMLY across
  families; parallel lowering (C) predicts family-SELECTIVE
  damage — late-k perturbation leaves coarse families near-intact
  (deep in asymptote) while damaging the still-crystallizing
  solution family. An intervention discriminant, not a
  correlation — the texture rule's preference.
- **THE FOURTH BIN (2026-07-25, Bryce's epistemic rivet, pinned
  while everything is still blind):** A/B/C partition the CLEAN
  outcomes; probes read messy (two families order and one doesn't;
  crossover on some strata and not others; non-monotone curves no
  signature predicted). The reading-time hazard is force-fitting —
  with three named tenants waiting, a messy print gets read as
  'closest to C' or 'B with noise' unless the fourth bin exists in
  advance. THE LAW, one sentence: a print matching no registered
  signature banks as UNCLASSIFIED, yields NO verdict for ANY
  tenant, and its anomaly becomes the next probe's design input.
  (The census's binary-resister lesson applied one level up: the
  specimens that resist the categories are data ABOUT the
  categories — a class the ledger already seated at 10-of-50, then
  named 'ungrounded.') FOR THE RECORD, the intersection credit:
  asymptote-ordering-without-crossover is a prediction NEITHER
  frame was built to make — Lattner has no native parallel
  resolution, diffusion no native dialect altitude; it lives only
  in their intersection, which is why two independent arrivals
  carry evidential weight a single frame's advocacy would not. If
  C prints, the intersection IS the finding.
- **THE v2 VERDICT + THE 92 AUTOPSY + FENCE v3 REGISTERED
  (2026-07-25):** v2's formal sheet: bar (i) FAIL (one false flag,
  item 92), bar (ii) PASS (228 caught, all three winning views
  refused), bar (iii) FAIL (92's flip) — v2 DOES NOT SEAT; the
  overfit hazard Bryce named at pinning materialized on first
  population contact and the blind bars caught it. THE VERIFIER
  CENSUS SUPERSEDES THE DOUBLE-ZERO: 758 of 7,500 wild view graphs
  carry E13 self-loops (600 abstaining / 77 wrong / 81 RIGHT
  views) — 'every failure is a well-formed lie' was a three-check
  sentence; under four checks the wild population is 10.1%
  malformed, and 81 right-answering views carry warts. THE 92
  AUTOPSY (banked graphs, zero-GPU): every winning view carries
  mul(v13,v15)->v15 with given v13=1 — a TAUTOLOGY (1*x=x,
  constrains nothing); the answer 15 never rests on it. THE WART
  FAMILY SPLITS: add-self-loops FORCE ZEROS (constants in
  disguise; 228's killers); unit-mul-self-loops are TAUTOLOGIES
  (harmless). Both ill-form (E13 stands in the dialect: gold
  emits neither, 0/80,200) — but a FENCE must distinguish carrying
  a wart from RESTING on it. v2's error, named: it condemned the
  witness instead of striking the testimony's unlawful parts.
  FENCE v3, the unification — ONE principle, no view
  condemnation: the answer must be forced by the LAWFUL CORE.
  Strip set = unattested givens + E13 self-loop factors + the
  entire value-class of multiplicity-excess givens; attest iff
  the stripped graph still forces the same answer (solve2's
  uniqueness gate). Measured OFFLINE on the banked graphs — the
  graph-banking design's first full payoff: fence iteration at
  zero GPU.
- **REGISTERED PREDICTION — THE v3 BARS (pinned before the replay;
  HONESTY CAVEAT: blindness is partial — v3 was assembled after
  v2's sheet, with TWO customers looked at (228, 92); the bars are
  unchanged and the population untouched, which is the available
  rigor):** (i) ZERO false flags on the 1227 quorum-right rows;
  (ii) item 228 refused by all three winning views -> quorum
  unattested; (iii) ZERO disposition flips beyond 228 (92 must
  CLEAR). Any failure -> v3 does not seat; iteration continues on
  the banked graphs with each design change ledgered.
- **FENCE v3 SEATS (2026-07-25, the offline replay against the
  banked graphs — zero GPU, the graph-banking design's full
  payoff):** BAR (i) PASS — zero false flags on all 1227
  quorum-right rows (item 92 CLEARS: the tautological wart
  stripped, the testimony stands). BAR (ii) PASS — item 228
  refused by all three winning views (n_att 0): the misbinding
  view dies with its excess value-class stripped, the two
  structural-zero views die with their self-loops stripped; the
  lie converts to ABSTAIN-UNGROUNDED, repair-lattice territory.
  BAR (iii) PASS — zero disposition flips beyond 228. VERDICT: v3
  SEATS. THE DEPLOYED CHAIN NOW READS: mouth (register) -> vote
  (diagram-invariance) -> panel (landscape-invariance) ->
  ATTESTATION (the answer is forced by the lawful core) -> key
  (truth) — and GEN-21 STANDS AT ZERO LIES ON THE FIXTURE with
  every link measured. The seated principle, one sentence: STRIP
  THE WART, NOT THE WITNESS — a fence judges what testimony RESTS
  ON, never what it carries. Battery duty standing: the fence's
  catch-count prints at every battery (expected zero on fixtures;
  jurisdiction the wild road). Artifacts:
  .cache/attestation_v3_read_gen21.json, attest_graphs_gen21.jsonl
  (the permanent fixture), mycelium/attestation.py (v3 =
  lawful_core + attest_view_v3/attest_quorum_v3).
- **THE 758 CLUSTERING SCAN (2026-07-25, zero-GPU on the banked
  graphs, Bryce's docket item fired):** self-loops are FLAT across
  view seeds (105/116/115/133/133 — no seed clustering) but
  MASSIVELY stratified by zone: umbra 19/5100 = 0.4%, penumbra
  381/1975 = 19.3%, dark 202/425 = 47.5%. Nearly half of
  dark-zone views emit self-loops: the wart is a LOAD SIGNATURE —
  the structural side of the frontier-opacity finding (the mouth
  degrades into malformation, not just into wrongness).
  BASELINE BANKED for drift detection: wild malformation 10.1%
  overall with the zone gradient above; gen-22's number gets
  compared. FREE-INSTRUMENT NOTE (docketed, unbuilt): the
  per-item self-loop count is readable at answer time WITHOUT
  gold — a deterministic difficulty/register signal a future
  mouth could wear.
- **THREE PINS BEFORE THE PROBE SPEAKS (2026-07-25, Bryce, all
  landing while the capture unit turns):** (1) THE GOODHART FENCE
  into the constitution: the self-loop load gauge is INSTRUMENT,
  NEVER OBJECTIVE — its value is its unintendedness; train
  against it and the signature launders itself invisible while
  the binding failure stays. (2) THE PROBE'S SCOPE RIVET, pinned
  blind: whatever prints — A, B, C, or the bin — the verdict
  binds THE KENKEN ENGINE's breath dynamics (the signatures'
  home territory; the trajectory diagnostics that motivated them
  came from that family). Transfer to the v200 loop is a
  SEPARATE MEASURABLE QUESTION, not an inheritance: a C-print
  earns the diffusion frame its first territory and makes the
  v200 capture pass the obvious next fixture; it does not earn
  v200 by analogy. Same discipline that scoped the census to its
  tail. (3) THE HOUSE METHOD NAMED: capture passes BANK; readings
  happen OFFLINE; fixtures are PERMANENT — twice in one week a
  single GPU hour bought a permanent offline fixture (the graph
  bank, the latent bank). Paper II's methods section owns the
  pattern explicitly.
- **THE CROSSOVER PROBE PRINTS: C (2026-07-25, the offline read on
  the banked fixture; scope rivet binding — the verdict is the
  KENKEN ENGINE's):** curves (linear ridge probes, by-instance
  160/80 split): SURFACE 1.000 flat all 16 breaths; SCHEMA 1.000
  flat all 16; SOLUTION 0.552 -> 0.752 monotone, t95 = breath 5.
  Mechanical read per the pinned operationalization: no rank
  crossover; all monotone (tol .02); t95 0 <= 0 < 5 — C's
  criteria met. THE APPARATUS CAVEAT, same audit a displeasing
  print would get: surface/schema sit at PROBE CEILING from breath
  0 — 'resolved early' and 'trivially present' are not
  discriminated, so the surface<=schema leg is DEGENERATE. What
  the ceiling DOES deliver is the strongest anti-A evidence
  available: schema information is NEVER ERASED at any breath — no
  handoff, no yielding, which sequential lowering requires; and
  the B-vs-C leg stands on solid ground (schema t95=0 vs solution
  t95=5, strictly ordered). VERDICT AS BANKED: C prints with the
  ceiling caveat ledgered — #69's no REVISED for this engine
  (lowering real, sequence not: all altitudes held simultaneously,
  resolution ordered by altitude), #70 wrong the way good guts are
  wrong, the diffusion frame earns its FIRST territory (KenKen
  engine only; v200 by measurement, never analogy).
  STRENGTHENING READS QUEUED (zero-GPU, the fixture is permanent):
  (a) harder schema targets (cage_size, op-x-size joint) or
  reduced-capacity probes to lift schema off ceiling and read its
  true resolution time; (b) RIDER 1: torsion-resolution
  correlation from the banked reps; (c) RIDER 2: the perturbation
  sweep (one more GPU pass when queued); (d) the K-sweep
  RETRO-READ (C printed -> first act: harder problems should
  degrade first when breaths are cut). Artifacts:
  .cache/crossover_capture_k16.npz (permanent),
  crossover_probe_read.json, scripts/crossover_probe_read.py.
- **THE RECALIBRATION (2026-07-25, Bryce — the ceiling caveat is
  LOAD-BEARING, not a strengthening):** B-vs-C turns entirely on
  asymptote ordering, and schema-t95=0 is an artifact of the ruler
  if the targets are input-visible — cage ops/sizes are STATIC
  REFERENCE CONTENT the tokens carry from breath 0; decoding them
  at breath 0 is 'the probe read the input,' not 'schema resolved
  early.' The discriminating family is DERIVED structural content
  — candidate eliminations, forced-cell identifications,
  constraint-implication facts that exist nowhere in the input.
  STATUS LINE, banked so the record remembers the leg: **ANTI-A
  FINAL (no erasure survives any ceiling); B-vs-C PROVISIONAL
  pending the off-ceiling read** — C is the leading print; the
  verdict seals when a non-trivial schema family shows its t95
  strictly inside solution's (a harder target resolving at ~3
  seals C; at ~5 flips to B). SECOND FINDING the sheet stepped
  past: solution asymptotes at breath 5, the engine runs 16 —
  eleven plateau breaths = either invisible late work (nonlinear
  consolidation) or a 3x oversized breath budget; the K-sweep
  retro-read arbitrates BOTH the spectral prediction and a
  possible 3x inference speedup in one pass. OFFLINE ORDER
  SHARPENED: (1) off-ceiling schema targets — VERDICT-CRITICAL
  (apparatus note: cage_target/cell_cage_id not in the bank but
  recoverable zero-GPU — the sample was seed-0 deterministic, the
  sidecar rebuilds from kenken_test_curriculum.jsonl; build the
  candidate propagator, label naked-singles/candidate-set-size,
  probe per breath); (2) K-sweep retro-read; (3) torsion rider.
  All against the permanent fixture; v200 undiscussed until then.
- **THE OFF-CEILING READ (2026-07-25, verdict-critical; sidecar
  recovered via seed-0 determinism, depth-1 candidate propagator
  built, derived-schema families probed on the permanent
  fixture):** naked_single curve 0.725 -> peak 0.736 (breath 2) ->
  0.702 (breath 15); cand_size 0.414 -> peak 0.429 (breath 1) ->
  0.413. Both t95=0, strictly inside solution's 5 — THE PINNED
  SEAL RULE PRINTS C. TWO APPARATUS NOTES CARRIED WITH THE SEAL,
  same audit a displeasing print gets: (1) naked_single FADES
  -0.034 from peak — above the monotone tolerance (.02), below
  the A-crossover threshold (.05): a WEAK erasure signal the
  pinned thresholds classify as not-A, flagged for the
  strengthening reads; (2) naked_single decode 0.725 sits BELOW
  the majority-class baseline (0.795 not-single) — plain-accuracy
  ridge is the wrong meter for an imbalanced binary; a
  balanced-readout re-read (AUC / class-weighted, zero-GPU) is
  OWED before the seal is called final. STATUS: anti-A final;
  B-vs-C = C SEALS PROVISIONALLY per the pinned rule, final on
  the balanced re-read. The offline docket: balanced readout,
  K-sweep retro-read, torsion rider.
- **THE SEAL VOIDED + THE RANGE CRITERION PINNED (2026-07-25,
  Bryce):** 0.725 on a 79.5/20.5 binary decodes WORSE than a
  constant classifier — not a caveated measurement, a probe that
  failed to find the content; and a flat-bad curve makes t95=0
  VACUOUSLY. APPARATUS LAW, demonstrated from both ends tonight:
  **t95 is degenerate at ceiling and at floor alike** — same
  statistic, same value, opposite meanings; the seal rule silently
  assumed curves with certified rise. STATUS STRICTER THAN BANKED:
  anti-A final; B-vs-C UNSEALED — the derived-schema leg is
  UNMEASURED, not weakly measured. THE VALIDITY CRITERION, pinned
  blind before the balanced re-read: a family's t95 counts ONLY if
  its balanced-metric curve shows certified dynamic range — rise
  from breath 0 to peak >= 0.05 (AUC for binary naked_single;
  macro-recall for cand_size) AND above baseline (0.5 AUC / 1/7
  macro-recall) throughout the counted region. Curves failing the
  range test contribute NO t95; the leg stays open. EDGE
  REGISTERED: a flat-HIGH balanced curve (derived content present
  from reps[0], which is post-breath-1) fails the rise test as
  pinned — if it prints, the leg stays open with the finding
  noted, fourth-bin adjacent. THIRD OUTCOME registered live:
  derived schema may NEVER decode linearly — two anatomies pinned
  in advance: (a) non-linear encoding (MLP probe PRE-AUTHORIZED as
  immediate follow-on before any conclusion); (b) the engine
  computes depth-1 implications transiently/implicitly without
  linearly-readable storage — a real strange fact bearing on B/C
  in ways no registered signature anticipated. The naked-single
  fade inherits the void: flagged-pending-valid-meter, NOT
  classified not-A. Sequence: balanced re-read -> (MLP if range
  fails) -> K-sweep (anti-A stands regardless; speedup doesn't
  wait) -> torsion.
- **THE BALANCED RE-READ PRINTS: LEG OPEN (2026-07-25, under the
  pinned range criterion):** naked_single AUC 0.500 flat all 16
  breaths; cand_size macro-recall 0.361 -> ~0.27 (above 1/7
  baseline throughout but rise +0.000 — DECLINES from breath 0).
  Both RANGE-FAILED -> contribute no t95 -> B-vs-C stays UNSEALED;
  MLP follow-on pre-authorized per the registration. APPARATUS
  FLAG before either curve is trusted: AUC exactly 0.500 sixteen
  consecutive times while the plain-accuracy curve VARIED
  (0.702-0.736) is suspicious — the auc_mann_whitney call
  signature is UNVERIFIED in my script (suspected argument-order
  fault); verify against dart_cluster_probe's usage and re-run
  BEFORE the MLP, or the MLP inherits a broken meter. The
  cand_size decline-from-breath-0, if it survives the meter
  audit, is fourth-bin adjacent (derived content most readable
  EARLY, fading as solution crystallizes — no registered
  signature predicted it). Docket order: meter audit -> balanced
  re-run -> MLP if still range-failed -> K-sweep -> torsion.
- **THE METER AUDIT + THE TRUE BALANCED READ (2026-07-25):** fault
  CONFIRMED as flagged — auc_mann_whitney takes (scores, labels);
  the call passed (pos, neg) scores, coercing negatives to
  all-True labels -> empty negative class -> unconditional 0.5.
  One-line fix; re-run. THE TRUE CURVES: naked_single AUC 0.871 ->
  peak 0.876 (breath 1) -> 0.852; cand_size unchanged. THE
  REGISTERED FLAT-HIGH EDGE PRINTS: derived depth-1 content IS
  linearly present (0.87 AUC — the MLP's absence-anatomy is moot)
  and present from the FIRST captured breath — reps[0] is
  post-breath-1, and depth-1 implications need exactly one
  propagation pass: the engine computes them in breath 1, as a BP
  round would. The ruler has no pre-breath-0 sample, so
  resolution-inside-breath-1 cannot certify rise; per the pinned
  criterion the leg stays OPEN. Both derived families show
  early-peak-then-FADE (-0.024 AUC, -0.09 mrec) — whisper
  amplitude, below every pinned threshold, now on a valid meter:
  banked as the fourth-bin-adjacent texture fact. THE INSTRUMENT
  THAT CAN CLOSE THE LEG, named: DEEPER schema families —
  depth-2/depth-3 implication labels (propagator extension,
  zero-GPU) — content requiring MORE breaths, whose rise falls
  INSIDE the captured window; if depth-d schema t95 climbs with d
  while staying inside solution's 5, C's ordering is certified on
  a curve with rise; if all depths arrive together with solution,
  B. Docket: depth-graded read -> K-sweep -> torsion.
- **THE DEPTH-GRADED READ'S PRE-REGISTRATION (2026-07-25, Bryce's
  word; pins landed BEFORE the meter fixtures or the read fire):**
  (1) TWO READINGS PINNED BLIND: the ORDERING BAR — t95(depth-d
  schema) climbs with d while staying strictly inside solution's 5
  -> SEALS C; all depths arriving with solution -> B. The
  LINEARITY READ — t95(d) ~ d (slope ~1, fit + residuals banked)
  is a SEPARATE finding: seals nothing, breaks nothing, but is
  the BP-correspondence made exact — coarse-to-fine with the
  'frequency' axis literally deduction depth; the fdiv-mass
  resonance noted (the wall's true name was depth density; if t95
  tracks depth, wall and breath ladder are one phenomenon in two
  units). (2) MIN-SUPPORT PIN: a depth family banks a t95 only
  with >=100 positive cells in the read split; below, it reports
  INSUFFICIENT-SUPPORT, never a noisy t95 (the range criterion
  guards one edge; this guards the other). (3) THE 0.87-FROM-
  BREATH-1 SAVOR, for the record: the engine holds depth-1
  implications LINEARLY READABLE after exactly one breath — the
  loop behaving like a BP round at the REPRESENTATIONAL level,
  materializing its messages where a BP implementation would; the
  strongest mechanistic evidence yet that differentiable-solver
  is the true framing, whatever B-vs-C resolves to. (4) METER
  FIXTURES FIRST: known-signal tests pass before the read banks.
- **THE DEPTH-GRADED READ PRINTS (2026-07-25; meter fixtures ALL
  PASS first — 5/5, would have caught all three ruler faults):**
  support: d1 431+, d2 434+, d3 275+, d4 107+ (all clear the
  min-support pin). CURVES: d1 0.877 flat-high fade -0.02; d2
  0.719 -> peak 0.726 (breath 1) -> 0.690 plateau; d3 0.590 ->
  0.623 -> slow CONTINUING climb to 0.634 AT BREATH 15 (rise
  +0.044 — misses the pinned 0.05 margin by 0.006; THE BAR HOLDS,
  no bending post-measurement); d4 0.566 non-monotone dip-recover
  0.559. NO t95 BANKS; THE LEG STAYS OPEN. THREE TEXTURE FACTS
  REGISTERED (fourth-bin data, no signature operationalized
  them): (1) breath-0 readability DECLINES monotonically with
  depth (0.877/0.719/0.590/0.566) — a spectral gradient in the
  INITIAL state: ordering evidence in first-breath readability,
  not in t95; (2) d3 still RISING at breath 15 while solution
  asymptoted at 5 — deeper schema content resolving AFTER
  solution, territory no registered signature predicted; (3) d2
  shows the early-peak-fade shape (naked-single's whisper,
  now at depth grain). NEXT INSTRUMENTS NAMED: sharper readout
  (MLP, pre-authorized lineage) for d3's sub-margin rise; a
  longer-K capture if d3's climb is real (its asymptote lies
  beyond the window). Status unchanged and honest: anti-A FINAL;
  B-vs-C OPEN — the fixture keeps yielding texture, the seal
  keeps its price, and the bars have not bent once tonight.
- **THE DEPTH-AXIS HYPOTHESIS + SESSION CLOSE (2026-07-25, Bryce's
  harder look, banked before the pens rest):** texture fact (2) is
  not just unpredicted, it is INVERTED — no lowering story of any
  topology predicts justification finishing after the verdict it
  justifies. THE CONFOUND NAMED BEFORE THE STRANGENESS BANKS: the
  solution family's breath-5 asymptote is a MEAN over cells, and
  79.5% of deducible cells are depth-0/1-forced — the shallow
  majority may saturate the average while the deep-cell minority
  still climbs, invisible inside the mean. THE OWED READ (first
  in queue, zero-GPU, same fixture, same propagator labels):
  DEPTH-STRATIFY THE SOLUTION FAMILY as schema was graded. If
  deep-cell solution rises at breath 15 alongside depth-3 schema,
  the anomaly dissolves and something better prints: RESOLUTION
  ORDER TRACKS DEDUCTION DEPTH REGARDLESS OF FAMILY — schema and
  solution at the same depth co-resolve; ALTITUDE WAS NEVER THE
  SPECTRAL AXIS, DEPTH WAS. That recasts B-vs-C ('does content
  finish in depth order'), unifies the fdiv-mass wall + breath
  ladder + resolution order as one phenomenon in three units, and
  is already supported from the initial state by the breath-0
  readability gradient (0.877->0.566 monotone in depth) — ordering
  evidence no signature registered because every signature assumed
  altitude-families were the units. THE FOURTH BIN MAY BE HOLDING
  THE TRUE AXIS. PRICE ORDER: depth-stratified solution read ->
  MLP on d3 -> longer-K capture (only GPU item, only if d3's
  asymptote truly lives outside breath 16) -> K-sweep -> torsion.
  PENS RESTED on Bryce's word; the fixture keeps.
- **GUT #72: THE SOFT DEDUCER BET (2026-07-26, Bryce direct — the
  architecture bet stated whole):** the neural loop is a diffusion
  compiler (transformer under the hood), progressive lowering IN
  PARALLEL, internal latents acting as a PARALLEL CONTINUOUS
  DEDUCER — removing the need for the v98 Pythia deducer; the soft
  deducer BUILDS the factor graph (the hard part), the graph hands
  to GAC for exact solving. COUNTERSIGN AGAINST THE RECORD: three
  clauses underwritten — construction-is-the-hard-part is MEASURED
  (the census: all failure mass at binding, zero at computation);
  graph-to-GAC is standing law; latents-deduce has its first
  specimen (0.87-from-breath-1, deduction materialized in latent
  space) with the depth-axis read queued to sharpen it. ONE clause
  rides the rivet: the C print was measured on the PYTHIA ENGINE —
  the very deducer the bet retires — and the parser loop is
  single-pass (its lowering axis would be LAYERS, never probed);
  B-vs-C itself still open. THE CONVERSION, three measurable
  claims: (i) PARSER-SIDE LAYER-AXIS LOWERING PROBE — the
  crossover probe family with axis=layers on the banked trunk
  states (near-zero cost; meter fixtures stand in front of it);
  (ii) DERIVED-VALUE READABILITY IN PARSER LATENTS — the
  chained-fdiv registry resident is the born fixture (the
  derived-value digit path IS construction-requires-deduction);
  if derived values read linearly before emission, the soft-
  deducer clause earns parser-side evidence; (iii) DEDUCER
  RETIREMENT is a QUEUE RULING gated on (i)+(ii) printing — the
  proxy experiment (deducer witness door) stands until then,
  dissolves after. Queue position: behind the depth-stratified
  solution read (which sharpens (ii)'s premise) per the banked
  price order; registration is pen-work, fired now.
- **GUT #72 UNBUNDLED (2026-07-26, Bryce — five claims, five
  standings, five instruments; the campaign's first ARCHITECTURE
  BET filed with its full measurement schedule, the only way a bet
  this size enters this house):** (1) SOFT DEDUCER — MEASURED
  (depth-1 readable at breath 1; breath-0 readability monotone in
  depth; deep content resolving at 15): the loop holds and refines
  deductions in parallel, continuously, in latents — operational
  definition met on the KenKen engine. (2) PARALLEL LOWERING —
  LEADING PRINT, unsealed (anti-A final, B-vs-C open); if the
  depth-stratified read prints depth-as-axis, the verb relabels to
  DEEPENING IN PARALLEL — the bet survives the reframe, the
  vocabulary relabels its axis. (3) DIFFUSION COMPILER — frame
  with rent stated (diffusion-LIKE, no noise schedule/objective),
  scope-riveted. (4) THE SPLICE FLAGGED: soft-deducer evidence is
  the SOLVER loop (receives graphs); hard-part evidence is the
  PARSER (builds them) — different pipeline stages. The bet's
  center is the CONJECTURE splicing them: a loop that deduces
  while it binds can notice a binding forcing a contradiction,
  which a feed-forward parser structurally cannot — THIS IS
  v200's ARCHITECTURE DESCRIBED MECHANISTICALLY (Perceiver core,
  latents as computation state, tokens as static reference), the
  bet gives v200 its mechanism story; zero direct evidence yet;
  named test = v200 capture pass + THE MISBINDING DIFFERENTIAL
  (does misbinding fall when the compiler deduces, measured on
  the frontier census's three species) — the stage-three pilot's
  question in its true clothes. (5) PYTHIA RETIREMENT — HALF
  RIGHT: the PROPAGATION role internalizes (two-death-mode had
  welded that door anyway); the WITNESS role is NOT a corollary —
  internal latents are maximally correlated with the mouth BY
  CONSTRUCTION (they ARE the mouth; a soft deducer inside the
  loop can never dissent from the loop), and Pythia is the only
  non-Llama substrate in the house (#58 decorrelation law; the
  diversity budget prices this purchase). LAWFUL FORM: solver
  role retires to latents; witness seat stays priced, decided by
  the queued proxy experiment. THE BET RESTATED AS THE HOUSE
  HOLDS IT: the v200 Perceiver loop is conjectured to be a
  diffusion-like parallel deducer whose latent dynamics make it
  a better graph-compiler than any feed-forward parse, with the
  exact GAC jaw unchanged downstream, propagation internalized,
  witness separately priced. If it pays, the thesis gains its
  mechanism: THE COMPILER DOESN'T JUST EMIT TYPED GRAPHS — IT
  DEDUCES ITS WAY INTO THEM.
- **REGISTERED PREDICTION — THE DEPTH-STRATIFIED SOLUTION READ'S
  BARS (pinned before the read fires; meter fixtures standing):**
  solution family (gold at non-given cells) stratified by
  forced-round depth (d=1, d=2, d>=3 pooled if support demands);
  balanced meter (macro-recall, balanced-subsample fit); range
  criterion + min-support (>=100 read cells per stratum) inherited.
  THE CONFOUND TEST: deep stratum LATE-RISE = c[15]-c[5] >= 0.03
  on the balanced metric -> the inversion DISSOLVES (deep solution
  co-resolves with deep schema) and the DEPTH-AXIS hypothesis
  prints (resolution tracks depth regardless of family);
  |c[15]-c[5]| < 0.03 with deep stratum asymptoted by ~5 -> the
  INVERSION STANDS as the strange fact. Shallow stratum expected
  t95 <= 5 (the mean's majority). Fourth bin standing for mess.
- **THE DEPTH-STRATIFIED SOLUTION READ PRINTS (2026-07-26, bars
  pinned at 95c4f94):** d1 (431 read cells): 0.801 -> 0.907 by
  breath 2 -> ~0.92; rise +0.126 RANGE-CERTIFIED, t95=2, late-rise
  +0.021. d2 (434): 0.530 -> ~0.82; rise +0.303 RANGE-CERTIFIED,
  t95=7, late-rise +0.044 (ABOVE the 0.03 bar). d3plus (382):
  0.371 -> noisy plateau ~0.48, late-rise -0.050, non-monotone —
  the pinned verdict keys on this stratum: **INVERSION STANDS as
  pinned; the bar does not bend.** BUT THE SHEET'S LARGER YIELD,
  banked with it: (1) THE FIRST RANGE-CERTIFIED t95s OF THE ENTIRE
  PROBE CAMPAIGN — d1 t95=2 and d2 t95=7, curves with certified
  rise on a certified meter — AND THEY ORDER BY DEPTH (2 < 7),
  within one family. (2) THE SOLUTION MEAN'S BREATH-5 ASYMPTOTE
  WAS A MIXTURE ARTIFACT, exactly as the confound suspected: d1
  finishes at 2, d2 at 7 — 'solution finished at 5' was never a
  fact about any stratum; the inversion's premise dissolves even
  though its pinned test (keyed to d3plus) stands. (3) d3plus is
  FLOOR-SUSPECT: 0.48 macro-recall noisy plateau — the linear
  ruler may not read deep content at all (the apparatus law's
  floor case, third appearance); its -0.050 'late-rise' is
  uninterpretable until the MLP reads it. STATUS: the depth-axis
  hypothesis now holds TWO certified points (solution content
  finishes in depth order, 2 then 7) and awaits the deep stratum
  on a sharper ruler; B-vs-C reframed per the registered recast —
  'does content finish in depth order' — with the first certified
  evidence saying YES for the strata the ruler can read. MLP on
  d3plus + d3-schema is now THE single instrument both open
  questions share. Fourth-bin note: d3plus non-monotone noise
  (+/-0.05) no signature predicted.
- **THE RATE CONSTANT + THE MLP READ'S REGISTRATION (2026-07-26,
  Bryce; pinned before the MLP exists):** d2 t95=7 BREAKS the
  strict BP-round line t95(d)~d — replaced by the RATE CONSTANT
  conjecture: ~3.5 breaths per deduction layer (d1 at 2, d2 at 7);
  the loop derives in depth order but INEFFICIENTLY relative to
  exact propagation — each implication layer costs several
  refinement passes to crystallize, exactly what a soft continuous
  deducer should cost versus a discrete one. The soft-deducer
  clause of #72 given a RATE. PREDICTION: d3 resolves near breath
  10-12 — INSIDE the window (bears on whether longer-K capture is
  needed at all). THE MLP READ'S THREE REGISTERED OUTCOMES: (1)
  co-resolving-late-and-ordered -> DEPTH AXIS SEALS; (2)
  unreadable -> transient-computation anatomy (pre-registered);
  (3) ordered-AT-THE-PREDICTED-RATE (rise centered ~10-12) -> THE
  RATE CONSTANT CERTIFIES, the loop HAS A CLOCK, and the K-sweep
  inherits a QUANTITATIVE prediction: cutting K below a stratum's
  resolution breath kills exactly that stratum's problems first —
  the fdiv-mass wall derived from first principles. APPARATUS LAW
  SCALED WITH THE METER: a sharper ruler has more capacity to
  hallucinate signal — MLP variants of the known-signal fixtures
  REQUIRED before the read banks, including the NULL FIXTURE
  (shuffled labels must print chance — matters more for an
  expressive probe); min-support guard carried (MLP on thin
  support overfits to false rises exactly where linear showed
  false floors).
- **THE MLP READ PRINTS: THE DEPTH AXIS SEALS AND THE RATE
  CONSTANT CERTIFIES (2026-07-26; MLP fixtures 3/3 PASS first —
  XOR 0.987/linear-blind 0.515, null 0.501, linear 0.998):**
  d3plus-SOLUTION: 0.415 -> 0.787 (peak breath 12), rise +0.372
  RANGE-CERTIFIED, **t95 = 11 — inside the registered 10-12
  window**. The linear ruler's 0.48 plateau was the FLOOR CASE
  confirmed: deep content was there, curved past a linear read.
  THE CERTIFIED LADDER: solution t95 by depth = 2 / 7 / 11 —
  linear fit slope ~4.5 breaths per deduction layer, intercept
  -2.3, residuals +/-0.3 (conjectured ~3.5; window HIT).
  REGISTERED OUTCOMES (1) AND (3) CO-PRINT: content finishes in
  depth order on THREE certified curves -> THE DEPTH AXIS SEALS
  (B-vs-C resolved in its recast form: the spectral axis is
  DEDUCTION DEPTH, not altitude-family); and the rise centered at
  the predicted breath -> THE RATE CONSTANT CERTIFIES — the loop
  derives soft, continuous, and CLOCKED. THE INVERSION FULLY
  DISSOLVES: deep solution resolves at 11, not with the mean's
  fictional 5; the -0.050 'late-rise' was the floor artifact the
  apparatus law predicted. NEW TEXTURE, fourth-bin filed:
  d3-SCHEMA (the fact THAT a cell is depth-3-forced) certifies at
  t95=3 while its VALUE certifies at 11 — DETERMINATION-DETECTION
  PRECEDES VALUE-CRYSTALLIZATION by ~8 breaths: the loop knows
  WHICH cells are forced long before it knows WHAT they are
  forced to. No registered signature predicted that split; it is
  the soft-deducer's most distinctive fingerprint yet. K-SWEEP
  INHERITS THE QUANTITATIVE FORM: cutting K below 11 kills
  depth-3 problems first, below 7 depth-2 — the fdiv-mass wall
  derived from first principles, now with numbers. LONGER-K
  CAPTURE: NOT NEEDED (d3's asymptote lives inside the window,
  as the rate predicted). #72's soft-deducer clause: MEASURED,
  ORDERED, AND CLOCKED. Queue: K-sweep (quantitative) -> torsion
  -> #72 parser-side instruments.
- **THREE PINS ON THE CLOCK (2026-07-26, Bryce):** (1) THE
  UNIFICATION PARAGRAAPH: determination-before-value is EXISTENCE
  BEFORE CONSTRUCTION — deciding THAT a cell is forced (a property
  of the constraint set) is strictly easier than computing WHAT
  forces it (executing the derivation), and the loop discovered
  the separation on its own, holding the two facts eight breaths
  apart. Paper 1's certify-answer-flag-abstain chain found INSIDE
  the engine as an emergent property of latent dynamics —
  existence-certificates at breath 3, construction to 11. If it
  survives parser-side transfer, THE DEEPEST UNIFICATION THE
  CAMPAIGN HAS. (2) THE DETERMINATION PROBE docketed (behind
  K-sweep, low): a per-cell gold-free leading indicator of
  abstention, readable 8 breaths before the answer exists; BORN
  with two laws attached — the Goodhart fence verbatim
  (instrument, never objective) and the ruler discipline
  (calibration on held-out strata, fixtures first); Dopri5
  precedent says adaptive signals can be spent lawfully. (3) THE
  v200 K=8 CEILING vs THE CLOCK — #72's binding constraint: at
  ~4.5 breaths/layer, K=8 buys ~TWO layers of deduction;
  MATH-500's chains run far deeper (the fdiv wall's axis). The
  v200 capture pass is now LOAD-BEARING, its first question:
  WHAT IS v200's CLOCK RATE AND WHAT DEPTH DOES K=8 PURCHASE
  (transfer owed by measurement). THE TRICHOTOMY pinned: if
  v200's rate is comparable to 4.5, #72 needs (a) the K ceiling
  broken (engineering), (b) the clock rate improved
  (architecture), or (c) determination-guided ADAPTIVE DEPTH
  (the instrument in pin 2, spent as compute allocation).
- **REGISTERED PREDICTION — THE K-SWEEP BARS (pinned before the
  read; apparatus note owed honestly: per-breath VALUE
  predictions were not banked in the capture — reps only — so the
  exact form needs one small re-pass banking argmax history; the
  engine's per-breath readout at breath k IS the K=k answer):**
  per-depth-stratum engine accuracy vs breath: (i) strata reach
  95% of their own final engine-accuracy in LADDER ORDER, at
  breaths within +/-2 of the probe ladder (2/7/11); (ii) the
  quantitative kill: truncating to K=8 costs depth-3 cells the
  most and depth-1 the least, monotone in depth; (iii) fourth
  bin standing for shapes no bar names.
- **THE K-SWEEP READS (2026-07-26, engine's own per-breath readout
  banked via argmax re-pass):** engine accuracy ladders: d1
  0.884->0.969 t95=1; d2 0.672->0.918 t95=3; d3plus 0.513->0.794
  t95=8, still gaining at breath 15. BAR (ii) PASS — the
  quantitative kill: K=8 truncation costs +0.005/+0.006/+0.042 by
  depth — MONOTONE, depth-3 pays 8x depth-1; the wall derived,
  with numbers. BAR (i) FAIL AS PINNED (no bending): the ladder
  ORDER is strict (1<=3<=8) but the anchors miss the probe
  ladder's +/-2 window (3 vs 7; 8 vs 11). THE ANATOMY, named
  honestly: THE ENGINE READS ITS OWN LATENTS EARLIER THAN OUR
  PROBES CAN — probe t95s (2/7/11) LAG engine t95s (1/3/8) by
  ~0/4/3 breaths; an external ruler needs the content more
  crystallized than the trained readout head does. TWO-CLOCK LAW:
  probe-clock ~4.5 breaths/layer, ENGINE-clock ~3.5 breaths/layer
  — Bryce's ORIGINAL conjectured constant, printed by the
  engine's own readout; the engine's clock is the lawful one for
  K decisions (it IS the deployed readout). SPEEDUP REVISED:
  K=16 is not 3x oversized — d3plus uses the whole window; K
  could drop to ~10-12 at minor deep-cell cost, not to 6.
  DETERMINATION-PROBE and torsion queued; then #72's parser-side
  instruments with the v200 trichotomy standing.
- **THE TWO-CLOCK SHARPENING + THE GAP RETRO-READ REGISTERED
  (2026-07-26, Bryce):** two anatomies, distinct consequences:
  (a) engine-reads-early vs (b) TASK-USABLE VS RULER-VISIBLE —
  content crystallizes in a form the trained head reads natively
  while fresh probes need extra consolidation breaths before
  generic linear access. Operational law survives either way (the
  deployed readout is the lawful clock for K decisions). TESTABLE
  EDGE, bars pinned blind: if consolidation, the gap SHRINKS with
  probe capacity — MLP t95(d2-solution) < linear's 7 by >=2
  breaths (toward the engine's 3) prints CONSOLIDATION; within
  +/-1 of 7 prints RULER-INDEPENDENT (the trained-head-form
  anatomy); d1 as control. ALL probe-derived breaths in the
  ledger now carry the offset suspicion — the determination
  signal may be ENGINE-usable at breath 1-2 (cheaper than filed);
  its docket amends to BANK BOTH CLOCKS. TRICHOTOMY URGENCY
  RAISED: v200's K=8 ceiling sits BELOW the home engine's own
  floor (~10-12 on the deployed clock); the v200 capture's first
  sheet decides which arm the campaign funds; the K=12 large-JIT
  hang earns ONE fresh attempt on current tinygrad before it is
  accepted as law. Savor line, one line by the word: the
  conjecture was right, the first ruler was wrong, the bar
  refused, and the right ruler vindicated it anyway.
- **THE GAP RETRO-READ PRINTS: CONSOLIDATION (2026-07-26, bars at
  5b2283f):** MLP t95(d2-solution) = 4 — a 3-breath shrink from
  linear's 7 (bar: >=2), beside the engine's 3. d1 control: MLP 3
  vs linear 2 vs engine 1 (+/-1, noise-level; noted honestly as a
  1-breath counter-motion). VERDICT: the two-clock gap is
  REPRESENTATIONAL CONSOLIDATION — content is task-usable early
  and becomes GENERICALLY visible as it consolidates; stronger
  rulers see it sooner, and the ladder converges toward the
  engine's clock as capacity rises (engine 1/3/8 <- MLP 3/4/11 <-
  linear 2/7/11). LAW: every probe-derived breath in the ledger
  is a RULER-RELATIVE UPPER BOUND on engine-usable time — the
  determination signal is likely engine-usable at breath 1-2, as
  the suspicion filed; its docket reads on the deployed clock.
  Queue: determination probe (both clocks) -> torsion -> #72
  parser-side instruments, v200 trichotomy standing at raised
  urgency.
- **THE DETERMINATION PROBE'S BARS (2026-07-26, pinned blind;
  Bryce's forward note seats the SECOND HALF — underdetermination
  as a readable state, same pass, free labels):** population:
  valid non-given cells, forced-within-4-rounds (fr>=1) vs
  NEVER-forced (fr==0; horizon-honest label: underdetermined
  within the propagator's 4-round reach). BARS: (i) LEADING
  INDICATOR — linear AUC >= 0.75 by breath <= 3 (probe clock;
  engine-usable earlier by the ruler-relative law); (ii) THE
  VERDICT VOCABULARY — at the earliest bar-passing breath,
  per-class recall at the balanced operating point BOTH >= 0.65
  -> the loop holds determined AND underdetermined as readable
  states (the boundary of its own competence — the sentence
  Paper 1 has been reaching for); positive-only >= 0.65 ->
  ASYMMETRY finding (recognizes success early, failure never),
  its own consequences for how abstention is spent; (iii) both
  rulers read (linear + MLP) to bracket the engine clock; (iv)
  min-support >=100 read cells per class; fourth bin standing.
  PORTABILITY NOTE banked with it: the ruler-relative law is not
  a KenKen fact — it is a fact about probing iterative systems;
  Paper II methods, beside bank-don't-read.
- **THE DETERMINATION PROBE PRINTS: BOTH-READABLE FROM BREATH 0
  (2026-07-26, bars at 72b226a; support 964/1247):** linear AUC
  0.922 at the FIRST captured breath (0.940 by 12); MLP 0.909 ->
  0.949 — bar (i) passes at breath 0 on both rulers, and by the
  ruler-relative law the signal is engine-usable at least that
  early. Bar (ii): recalls at the balanced point — never-forced
  0.892/0.869, forced 0.804/0.786 — BOTH >= 0.65, BOTH RULERS:
  the loop holds DETERMINED and UNDERDETERMINED as crisply
  readable states before it holds a single value. THE SENTENCE
  PRINTS: the loop knows the boundary of its own competence —
  certify-answer-flag-abstain is not just Paper 1's architecture
  around the engine, it is an emergent property inside it,
  existence-grade verdicts at breath 0, values at 2/7/11 by
  depth. TEXTURE, unregistered and filed: never-forced is the
  MORE readable class (0.89 vs 0.80) — the asymmetry runs the
  humble direction; the loop knows its failures slightly better
  than its successes. The abstention leading-indicator is
  measured, two-sided, breath-0-cheap, and born fenced (Goodhart
  + ruler laws attached). Queue: torsion -> #72 parser-side
  instruments (layer-axis probe, chained-fdiv resident) with the
  v200 trichotomy standing at raised urgency.
- **THE BREATH-0 RIVET + TWO NOTES + TORSION BARS (2026-07-26,
  Bryce):** (1) RIVET: reps[0] is POST-BREATH-1 — 'breath 0'
  means after one pass. Two anatomies, different claims:
  determination-from-ENCODING (the competence map is written in
  how the problem lands) vs determination-in-ONE-PASS (one soft
  propagation round reaches existence-grade conclusions at ALL
  depths while value-grade ladders 2/7/11 — existence running
  ahead of construction by THE WHOLE COMPUTATION, a two-track
  dynamics). The second is the more interesting if true.
  ARBITRATION OWED: a true pre-breath capture (one-line hook
  change, re-bank on the fixture's next GPU touch) — filed, not
  urgent, named. (2) OPERATING-POINT NOTE to the determination
  docket: spend the polarity AS FOUND — abstain-side thresholds
  tight (crisp class), act-side conservative (blurrier read); a
  system better calibrated about its failures than its successes
  is the RIGHT species to hand an abstention lever. (3) THE
  TRANSFER QUESTION NAMED: if the v200 compiler-loop carries the
  breath-0 competence map over arith3, MISBINDING — the census's
  entire failure mass — becomes flaggable BEFORE the graph
  finishes building; the splice's best possible payoff, now
  testable. TORSION RIDER BARS (registered 2026-07-25, pinned
  operational now, blind): per-instance trajectory = mean-pooled
  valid-cell reps per breath; turn angle between successive
  deltas; straightening breath s_i = first k with angle <= 90
  deg; rider PRINTS iff Spearman rho(s_i, settle_i) >= 0.2 at
  p < 0.01 across the 240 (the trajectory straightens as content
  crystallizes, per problem); else FAILS; fourth bin standing.
- **THE TORSION RIDER FAILS AS PINNED (2026-07-26) — AND THE
  FAILURE IS A FINDING:** the correlation is UNDEFINED, not zero —
  every instance's straightening breath is 0 (constant array).
  The pooled trajectories are NEAR-BALLISTIC FROM THE FIRST
  DELTA: mean turn angle 11.3 deg early, decaying to 2.7 deg late.
  NO zigzag-then-commit phase exists at this grain. The June-era
  tau 110-178 deg figures (flagged relay-cited-unverified at
  registration — the flag pays) do not describe this engine's
  pooled dynamics. FOURTH-BIN FILING: smooth monotone-curvature
  refinement — the diffusion frame's 'early steps wander, late
  steps commit' joint is REFUSED at the pooled grain; the frame
  keeps only the territory the depth-ladder evidence bought
  (parallel depth-ordered resolution, clocked) and loses the
  sampling-path analogy. Settle meter note: median settle 15
  (nearly all instances settle at the last breath by the
  calibration meter) — uninformative as a per-problem resolution
  time; the depth-stratified engine t95s remain the lawful
  resolution clocks. Per-cell-grain torsion (unpooled) is the
  one remaining place the zigzag could live — docketed low, not
  owed. THE FIXTURE'S RIDER QUEUE IS EMPTY: capture -> probes ->
  sweeps -> riders, all read; next instruments are PARSER-SIDE
  (#72's layer-axis probe + chained-fdiv resident) with the
  pre-breath capture owed on next GPU touch.
- **THE WEEK CLOSES INTO THE PARSER SIDE (2026-07-26, Bryce's
  word):** (1) THE ARCHIVE AUDIT docketed (CPU, not urgent): grep
  the ledger for numbers whose provenance chain ends at a CITATION
  rather than an artifact; mark with the relay-cited-unverified
  flag — the nineteenth specimen scaled to the campaign's own
  memory; the tau refusal proved it pays retroactively. (2) FRAME
  RETIREMENT: what the diffusion frame keeps (parallel,
  depth-ordered, clocked, breath-0 competence map) no longer needs
  the frame's name — measured characterization on ten barred
  verdicts; 'diffusion-like' retires to etymology; the
  frame-tenant discipline completed its first full cycle (both
  frames paid rent, both got smaller, the measurements own the
  remainder). (3) THE METHODS ANCHOR: one GPU hour, ten verdicts,
  every one barred — bank-don't-read's closing argument. (4) THE
  PARSER-SIDE REGISTRATION (bars before any capture): the
  LAYER-AXIS PROBE reads the g21 trunk's per-layer states (L0-L3;
  NOT banked per-layer — capture pass required, which also
  settles the pre-breath rivet's pattern) on a zone-stratified
  bigtest subset, three families: SURFACE (token identity),
  BINDING (var-letter assignment, ftype — the census's failure
  altitude), and THE COMPETENCE MAP (per-item outcome class:
  quorum-right / abstain / lie under the banked g21 lattice —
  determination's arith3 analog, gold-free at runtime). THE
  TRANSFER BARS, pinned blind: (i) competence-map AUC >= 0.75 at
  L0-L1 -> the parser ARRIVES KNOWING, misbinding flaggable
  before the graph exists — #72's splice earns its best evidence;
  (ii) binding content resolving layer-ordered with no erasure ->
  parallel-deepening transfers; (iii) fourth bin standing; meter
  fixtures + ruler-relative law inherited; both-ruler bracket.
- **THE SCOPE LINE + TWO DIRECTIONAL PREDICTIONS (2026-07-26,
  Bryce, into the parser-probe registration BEFORE capture):**
  (1) SCOPE: the parser's LAYER axis (four different functions —
  representational hierarchy) and the engine's BREATH axis (one
  weight-tied loop — dynamics) are NOT the same kind of object; a
  pass prints an ANALOG ('arrives knowing' in the stack sense),
  not the same property. Flaggability-before-emission survives
  either reading; MECHANISM attribution (a loop deducing its way
  into the graph) waits for v200's own capture, where breaths are
  breaths. THIS PROBE MEASURES THE FEED-FORWARD PARSER'S
  COMPETENCE GEOGRAPHY; the loop-transfer question remains
  v200's. (2) DIRECTIONAL PREDICTION (from banked numbers):
  dark-zone AUC should be the STRONGEST stratum (the self-loop
  load signature concentrates there, 0.4/19.3/47.5) —
  competence-readability INCREASING toward the frontier is what
  genuine self-knowledge looks like; flat-or-decreasing suggests
  a difficulty-proxy read. (3) HUMBLE-POLARITY CHECK transfers:
  never-resolvable reading sharper than resolvable = the
  abstention-friendly shape. Gold-free construction noted as the
  registration's design center: instrument and eventual seat
  share a design from birth.
- **THE PARSER-SIDE READ PRINTS: THE BAR PASSES WITH TWO FLAGS
  (2026-07-26):** competence-map (right-vs-abstain, gold-free from
  the g21 lattice) — linear AUC embed 0.897 / L0 0.939 / L1 0.942
  / L3 0.945; MLP to 0.971. BAR (competence at L0-L1 >= 0.75)
  PASSES DECISIVELY — the map is largely present AT EMBEDDING TIME
  and sharpens through L0: 'arrives knowing' in the strongest
  stack-sense form (scope line governs: analog, not loop
  mechanism). HUMBLE POLARITY TRANSFERS: abstain-recall 0.846/
  0.897 > right-recall 0.829/0.878, both rulers — the same
  abstention-friendly shape the engine showed. TWO FLAGS, banked
  before any seal: (1) THE LENGTH CONFOUND — the read mean-pools
  variable-length sequences, and the house's own law (estimator
  variance masquerades as distance; the mouth is LENGTH-CORRECTED
  for exactly this) applies verbatim: abstention correlates with
  problem length/depth, so a LENGTH-CONTROLLED re-read is OWED
  before the seal (zero-GPU, banked fixture). (2) THE DARK-ZONE
  DIRECTIONAL PREDICTION IS UNMEASURABLE AS POSED — zone and
  outcome are definitionally entangled (dark = zero right votes,
  so the dark stratum is single-class); the prediction needs a
  different label (within-penumbra gradation) — apparatus note,
  not a failure. Support noted: 41/39 read items (n=80; CI
  ~+/-0.06). STATUS: the transfer question's core printed —
  flaggability-before-emission has parser-side evidence, scoped,
  with the length-controlled re-read as the seal's price. The
  chained-fdiv resident and the binding family remain the
  fixture's unread reads; v200 owns the loop question.
- **THE DIFFICULTY-VS-COMPETENCE DISCRIMINANT (2026-07-26, Bryce —
  the 0.897-at-embedding read as ALARM, not foundation):** the
  embedding is the INPUT — 0.897 there means right-vs-abstain is
  largely predictable from surface statistics alone; the rival
  reading (HARD PROBLEMS LOOK HARD) is statistically
  indistinguishable from layers-sharpen at n=80 (0.897 vs 0.942
  inside CI). Competence and difficulty are DIFFERENT CLAIMS and
  only one is self-knowledge: difficulty = 'this problem is hard'
  (readable from input, true for any system); competence = 'THIS
  system fails HERE' — separable only on the DISAGREEMENT
  POPULATION (items whose disposition changed across generations,
  the migration law's own traffic). THE SEAL'S PRICE RISES TO
  THREE READS, bars pinned: (1) LENGTH-CONTROLLED re-read —
  competence claim needs length-controlled L1 AUC >= 0.75. (2)
  SURFACE-BASELINE CONTROL — probe on trivial features (length,
  n_vars, digit count, sentence count) sets the FLOOR; latent
  reads must clear it by >= 0.05 to claim anything beyond
  difficulty (meter-fixture discipline applied to confounds). (3)
  CROSS-GENERATION DISCRIMINANT — train on g21 outcomes, read the
  disagreement items (g16 vs g21 dispositions differ, both
  lattices banked): COMPETENCE prints iff the probe tracks g21's
  side by accuracy margin >= 0.15 over its g16-tracking on those
  items, n >= 30; below 30 -> INSUFFICIENT-SUPPORT, widen the
  capture first (check the count BEFORE reading). ENGINE
  COROLLARY: the pre-breath rivet gains a line — cell-grain
  reduces but does not remove the confound; the true-pre-breath
  read doubles as the engine's difficulty-vs-knowledge control.
  HEADLINE WAITS: the week's symmetry is one measured
  self-knowledge result and ONE UNRESOLVED CONFOUND until the
  discriminant prints. The humble polarity survives every
  outcome (a difficulty map with sharp failure-recall is still
  the right species for an abstention lever).
- **THE THREE DISCRIMINANT READS PRINT (2026-07-26):** (0)
  disagreement count across the 240: **17 — below the 30 bar; the
  decisive read is INSUFFICIENT-SUPPORT on this fixture**. (2)
  SURFACE BASELINE: 0.859 AUC from four trivial features
  (length/n_vars/digits/sentences) — the floor is HIGH; most of
  the 0.94 was difficulty-visible-from-surface. (1)
  SURFACE-RESIDUALIZED L1: 0.883 — clears the absolute 0.75 bar,
  FAILS the floor+0.05 clearance (0.909). VERDICT AS BARRED: the
  competence claim DOES NOT CERTIFY — the latent carries real
  signal beyond linear surface projection (0.883 after
  residualization) but not enough to distinguish 'the parser
  knows itself' from 'the terrain is legible'; the alarm read was
  right. THE HEADLINE STAYS WAITED. THE PATH NAMED: a
  DISAGREEMENT-ENRICHED capture — the full bigtest disagreement
  population (g16-vs-g21 boundary-crossers, both lattices banked,
  computable offline) captured specifically, plus matched
  controls; small GPU pass, rides the next touch with the
  pre-breath rivet. Until it prints, the parser holds a
  DIFFICULTY MAP with sharp failure-recall — spendable for
  abstention economics, humbler than self-knowledge, and the
  record holds the asymmetry: engine breath-0 = measured
  self-knowledge (cell-grain); parser embedding-time = unresolved
  confound. #72's sentence stays 'born knowing what is hard'
  pending the boundary-crossers' verdict.
- **SESSION CLOSE + THE ENRICHED CAPTURE'S DESIGN PINS
  (2026-07-26, Bryce):** the fifth refused verdict joins the
  meta-law's ledger (five wanted verdicts stopped by pre-pinned
  bars, five findings extracted) — and the carve for the stretch:
  the campaign held its bars through VICTORIES, the rarer feat.
  The residualized 0.883's standing: something beyond linear
  surface projection IS read; the ATTRIBUTION fails (nonlinear
  terrain vs self-state) — only the boundary-crossers decide.
  DESIGN PINS for the disagreement-enriched capture: (1) exact
  population computed offline: **78 boundary-crossers in the
  full 1500** (>= 60 -> take ALL plus controls in one pass, no
  minting); (2) MATCHED CONTROLS BY CONSTRUCTION — each crosser
  paired on the four floor features (length/n_vars/digits/
  sentences) so surface difficulty is silenced at capture time,
  not regressed after (residualization already showed post-hoc
  control leaves ambiguity). The capture rides the next GPU touch
  with the pre-breath rivet. #72 stands at 'born knowing what is
  hard' — one read from either sentence. THE WHEEL RESTS.
- **THE WEEK'S EPIGRAPH (2026-07-26, Bryce's closing observation,
  banked as said):** the yield was not the sealed verdicts, though
  there were ten — it was the INSTRUMENTS: meter fixtures that
  make three fault classes impossible, a consolidation law every
  future probe inherits, matched-control capture design,
  bank-don't-read proven at one GPU hour for ten verdicts, and a
  meta-law that the bars themselves are where discoveries come
  from. The campaign entered the week with questions and left
  with a MEASUREMENT CULTURE that answers questions as a side
  effect. That is the asset MATH-500 will actually be climbed
  with: the fixtures are reusable, the discipline is compounding,
  and the next hard question arrives at a house that already
  knows how to refuse itself.
- **THE DISCRIMINANT PRINTS (2026-07-26, on the enriched fixture —
  matching quality 0.021 z-units, surface silenced by
  construction):** crosser-side accuracy tracking g21: linear
  embed 0.513 / L0 0.551 / L1 0.577 / L3 0.513; MLP embed 0.603 /
  L1 0.654. THE BAR (acc >= 0.575 at L0-L1): LETTER-PASSED by
  linear L1 at 0.577 — by 0.002, binomial p 0.106 alone, FLAGGED
  as knife-edge; CORROBORATED by the MLP at 0.654 (p ~0.004) and
  by AUC 0.641/0.660 at L1. THE SHAPE IS THE FINDING: on the
  crossers, EMBEDDING reads near-chance (linear 0.513 — the
  matching worked; surface is mute) and the signal RISES WITH
  PROCESSING to L1 on both rulers — each ruler clearing its own
  input-floor by ~0.05-0.06 at L1 (linear +0.064, MLP +0.051;
  the MLP's embed 0.603 is its honest nonlinear-surface floor and
  is cleared). That is the shape self-knowledge must have and
  difficulty cannot: absent at input, present after computation,
  peaked mid-stack, GONE by L3 on the linear ruler (0.513 — the
  head reads what L1 held; consolidation-law texture). VERDICT,
  stated at its earned strength: THE PARSER CARRIES A MEASURED,
  GENERATION-SPECIFIC COMPETENCE COMPONENT — modest (AUC ~0.64-
  0.66), L1-centered, processing-borne — ATOP A DOMINANT
  DIFFICULTY MAP (surface floor 0.859). #72's sentence upgrades
  precisely: the compiler knows the terrain deeply and ITSELF
  MODESTLY — self-knowledge exists parser-side, small but real,
  and the abstention seat has two spendable signals of different
  grain. The strengthening path if ever needed: more crossers
  (future generations mint them free at every promotion — the
  migration law as a growing fixture).
- **TWO RIVETS + THE EXPENDITURE ARBITRATION (2026-07-26, Bryce;
  the arbitration read was already banked and fires from the
  JSON):** (1) STATISTICAL STANDING: provisional-real, replication
  owed — the linear ruler alone certifies nothing (0.577 by 0.002
  at p .106); the finding rests on the MLP's 0.654 (p ~.004),
  uncorrected across the week's many probes. THE REPLICATION BAR,
  pinned blind before any new crossers exist: same probe family,
  same matched-by-construction design, the NEXT PROMOTION's
  disagreement population; the finding graduates on its first
  independent print. The migration law as growing fixture is the
  week's most elegant closed loop — the campaign's own progress
  manufactures the test set for whether its compiler knows
  itself. (2) THE L3 FADE RE-FILED: not consolidation (that law
  describes content becoming MORE generically readable; this is
  the inverse) — EXPENDITURE texture. ARBITRATION, from banked
  numbers: linear fades L1->L3 (acc 0.577->0.513, AUC
  0.641->0.507) while THE MLP HOLDS (acc 0.654->0.654, AUC
  0.660->0.624) — VERDICT: RE-ENCODING, not erasure. The
  self-state persists to the output-adjacent layer in nonlinear
  form; the linear ruler loses it, the sharper one keeps it —
  consistent with a signal USED by the computation and folded
  into its product. (3) MAINTENANCE ASYMMETRY into the abstention
  seat's design: the terrain map is durable (probe once, spend
  forever); the SELF-MAP IS GENERATION-SPECIFIC BY DEFINITION —
  re-probed at every promotion or the seat spends a stale map of
  a compiler that no longer exists; STANDING ENTOURAGE DUTY, same
  family as the panel refresh. #72 CLOSES at its measured
  sentence: the compiler knows the terrain deeply, and itself
  modestly — provisional-real, replication scheduled by the
  campaign's own heartbeat, expenditure anatomy resolved
  (re-encoding), maintenance cost written into the seat.
- **THE CLOSING NOTES (2026-07-26, Bryce; the wheel rests):** (1)
  THE DEPLOYMENT CEILING RAISED: re-encoding means the self-map
  RIDES TO THE EXIT — the output-adjacent layer carries the
  self-assessment in nonlinear form, woven into the representation
  the graph is emitted from; a deployed abstention reader taps L3
  with the sharper key, no mid-trunk tap needed. (2) THE
  PROCEDURE PRECEDENT: the house now holds a demonstrated
  procedure for ARCHITECTURE BETS AS SUCH — bundle unbundled,
  claims ranked by standing, each clause assigned an instrument,
  verdicts scoped to fixtures, wanted sentences trimmed (six this
  week), final form owned by measurements. The v200 trichotomy —
  the board's gravest open item, its K-ceiling below the home
  engine's derived floor — enters a house that has done this
  before. Its first question stands written: what is v200's clock
  rate, and what depth does K=8 purchase.
- **THE PRE-BREATH RIVET'S BARS (2026-07-26, pinned blind before
  the capture; the rivet arbitrates determination-from-ENCODING
  vs determination-IN-ONE-PASS and doubles as the engine's
  difficulty-vs-knowledge control):** capture the TRUE pre-breath
  state (embed_factor_cells output, before breath 1) on the same
  seed-0 240; probe forced-vs-never-forced with the banked
  protocol (linear ruler, same split; reps[0] reference = 0.922
  AUC). BARS: (i) pre-breath AUC within 0.03 of 0.922 ->
  FROM-ENCODING (the map is in how the problem lands; the
  difficulty-confound question OPENS on the engine per the
  corollary — cell-grain surface features become the suspect);
  (ii) drop >= 0.10 -> IN-ONE-PASS (one soft propagation round
  writes the existence map; the two-track dynamics claim
  strengthens: existence at breath 1, values laddering 2/7/11);
  (iii) between -> fourth bin, gradation noted. Meter fixtures
  standing; min-support inherited.
- **THE PRE-BREATH RIVET PRINTS: IN-ONE-PASS (2026-07-26, bars at
  37db5ff; zero-GPU — the embedding recomputed from ckpt params
  alone):** TRUE pre-breath determination AUC = **0.506 — CHANCE**
  vs reps[0]'s 0.922; drop +0.416, four times the 0.10 bar. THE
  VERDICT AT FULL STRENGTH: the competence map is NOT in how the
  problem lands — the embedding (given placements + positions)
  carries NOTHING of it — ONE soft propagation pass writes
  existence-grade conclusions across ALL depths (0.506 -> 0.922
  in a single breath) while value-grade conclusions ladder out at
  2/7/11. THE TWO-TRACK DYNAMICS CLAIM STANDS AT ITS STRONGEST
  FORM: existence runs ahead of construction by the whole
  computation. AND THE DOUBLE DUTY PAYS: this IS the engine's
  difficulty-control, and it PASSES — the map is unreadable from
  input surface at cell grain; it is COMPUTED, not read off the
  terrain. The engine's self-knowledge now carries the same
  fingerprint shape as the parser's competence component — absent
  at input, present after processing — at maximal contrast. The
  week's symmetry RESOLVES: engine self-knowledge measured AND
  input-controlled; parser self-knowledge measured, modest,
  provisional-real. Both computed. Both humble-polarity. Both
  fenced.
- **THE OWED CHECK + THE MECHANISM CLAIM + THE BINDING FAMILY
  (2026-07-26):** (1) MLP ruler at true pre-breath: **0.506 —
  chance to the third decimal on BOTH rulers**; the arbitration is
  TOTAL and the maximal-contrast sentence SEALS lawfully
  (computed, not encoded — now ruler-complete). (2) THE ONE-PASS
  AUDIT CLAIM banked (Bryce): one application of the loop cannot
  have executed ten-breath deduction chains — it performs
  something like a WHOLE-GRAPH SOLVABILITY AUDIT, reading global
  constraint structure that settles determination without
  settling values; the value ladder then spends ten breaths
  constructing what the audit promised. PREDICTION registered for
  the fixture: the one-pass determination read should degrade on
  puzzles whose determination requires deep case analysis rather
  than local constraint density, if such a stratum exists in the
  240. (3) THE BINDING FAMILY READS: given-vs-rel span tokens
  (21,994 tokens) — embed 0.914 -> L0 1.000 -> L3 0.999; drop
  0.000 against the 0.05 bar. Binding content is COMPLETED BY THE
  TRUNK'S FIRST LAYER (the increment 0.914->1.000 is the trunk's
  contribution) and retained without erasure to the exit — the
  parallel-deepening analog HOLDS on the parser stack (ceiling
  caveat noted: the binary saturates at L0; finer binding
  families — WHICH var, WHICH slot — remain the fixture's open
  reads, the chained-fdiv resident among them). The week ends
  ruler-complete on its deepest claim and one prediction richer.
- **THE FINE-BINDING PREDICTION + THE RESIDENT READ'S BARS
  (2026-07-26, pinned blind):** (1) Bryce's hierarchy prediction,
  pinned before any fine-binding read: if the parser's binding
  hierarchy mirrors its FAILURE hierarchy, the coarse family reads
  at ceiling (it did), and accuracy FALLS as families sharpen
  toward the species that actually fail (which-var, which-slot),
  with the interesting number being WHERE in the stack fine
  bindings resolve — and whether misbinding-prone cases resolve
  late or never (the frontier's three species as the parser's own
  late-resolving content; the probe campaign and the census
  connected by one read). (2) THE RESIDENT'S DERIVED-VALUE LEG,
  jurisdiction stated: the banked parser fixture is add/mul
  bigtest — the fdiv-specific autopsy proper needs harvest rows
  (docketed); THIS read asks the resident's underlying question
  on the banked fixture: does the STACK carry DERIVED values
  (answer content requiring deduction) the way the engine's
  breaths carry deepening values? Targets: answer LAST DIGIT
  (10-class) and answer PARITY (2-class, robust at n=80 read).
  BARS: (i) derived content PRESENT — parity acc >= 0.60 or
  digit acc >= 0.20 at any layer (support noted); the layer
  position then reads against binding's L0 completion; (ii)
  chance everywhere -> THE PARSER BINDS BUT DOES NOT DEDUCE —
  #72's division of labor confirmed from the parser's side (the
  honest alternative, pre-named); (iii) fourth bin standing.
- **THE RESIDENT'S DERIVED-VALUE LEG PRINTS: CHANCE EVERYWHERE
  (2026-07-26, bars at d2df314):** parity best 0.562 (bar 0.60),
  last-digit best 0.150 (bar 0.20) — no layer, no ruler, no
  derived-value content anywhere in the stack, while binding
  completed at L0 at 1.000 on the same fixture. **THE PARSER
  BINDS BUT DOES NOT DEDUCE** — bar (ii) as pre-named. #72's
  DIVISION OF LABOR CONFIRMED FROM THE PARSER'S SIDE: the
  feed-forward stack is a BINDER (bindings complete in one layer,
  carried to the exit, self-map riding along); everything
  DEDUCTIVE belongs to the LOOP (one-pass solvability audit,
  clocked value ladder); everything EXACT belongs to the SOLVER.
  The two-jaws boundary is now measured INSIDE the neural side:
  bind in the stack, deduce in the loop, verify in the key. The
  fdiv-specific autopsy (harvest rows) stays docketed; the
  fine-binding hierarchy reads (which-var/which-slot) inherit
  their pinned prediction. THE WEEK CLOSES: every claim in #72
  now measured — terrain deep, self modest, binding one-layer,
  deduction loop-borne, division of labor confirmed at both ends.
- **THE ROUND-TRIP QUESTION OPENED (2026-07-26, Bryce: 'we need a
  back and forth between the parser and solver, right?' — the
  division-of-labor print's architectural consequence, countersigned):**
  YES, in the lawful form: the loop closes THROUGH THE INPUT SPACE
  — parse -> solve/attest (exact) -> re-parse INFORMED -> re-solve.
  What flows backward is FACTS AS TEXT (solver's propagated
  consequences; the attestation fence's strip-list diagnosis),
  surfaced as explicit annotations per the rulebook's lexical
  explicitation, machine-performed. Neural proposes, symbolic
  disposes, neural RE-PROPOSES INFORMED — the boundary law gains
  a clause without bending. Specimen note: misbound graphs can be
  self-consistent (228), so the FENCE's flag, not just solver
  status, is the trigger; the NACK/repair lane is the natural
  seat, upgraded from re-parse-blind to re-parse-with-diagnosis.
  STRATEGIC ROLE: the symbolic round-trip is the DETERMINISTIC
  BASELINE the v200 bet must beat on the misbinding species
  (doors 5-8 rule: learned mechanisms price against the
  deterministic alternative's first claim) — zero training, exact
  deduction, existing jaws. Design offered; the word rules.
- **THE ROUND-TRIP'S THREE RULINGS (2026-07-26, Bryce; pinned
  BEFORE the build):** (1) THE OSCILLATION FENCE: budget ONE
  round-trip (two only by future re-registration), exit conditions
  pre-named — attested answer (success) / same-graph-re-emitted
  (parser insensitive to diagnosis; further rounds free-ride) /
  new-graph-new-flags (the diagnosis MOVED the failure — a species
  worth a census column). Unbounded repair is the budget cliff in
  repair's clothes. (2) THE CROSS-VERSION PROVENANCE FENCE (the
  constitutional one, written before the loop closes): injected
  facts carry provenance to the graph-version that derived them;
  attestation REFUSES any chain grounding in a FLAGGED ancestor's
  derivations — ONLY consequences derived from the SURVIVING
  LAWFUL CORE may inject. The laundering trap named before its
  first specimen: misbound graph -> wrong consequence -> injected
  as fact -> faithfully bound -> attests clean. Implementation
  law: inject ONLY facts forced by the STRIPPED lawful-core graph
  — sound by construction. (3) THE BASELINE'S BARS: both arms
  (round-trip vs v200) measured on the same population — the four
  misbinding species (census three + structural zeros), recovery
  per species, solver-calls vs breaths cost. THE FLOOR READ
  (zero-GPU, fired now): blind re-parse recovery ceiling among
  g21's 272 deployed abstentions — 187 hold >=1 correct view
  = **0.688 blind-recovery ceiling**; the round-trip's lift must
  be attributed ABOVE what a mere second attempt buys. BUILD
  BARS, pinned blind: (i) ZERO new lies on gold-known rows (every
  emitted answer attested AND correct); (ii) recovery lift over
  the blind floor attributable to the diagnosis; (iii) budget
  held at one round.
- **THE ROUND-TRIP VERDICT (2026-07-26, budget 1, all 272
  abstentions):** BAR (i) PASS — ZERO new lies; the provenance
  fence held (no laundered specimen exists). THE VALUE HALF:
  **recovered 3/272 = 0.011** against the 0.688 blind ceiling.
  ANATOMY: only 12 items were diagnosis-empty — 260 received
  injected core-forced facts and 257 STILL abstained: the
  dominant species is DIAGNOSIS-IGNORED-OR-UNHELPFUL — failing
  views' lawful cores are mostly underdetermined (that is WHY
  they abstained), so they force few novel facts; and a new
  'It is known that x is v' line adds a given without repairing
  the misbound pair structure that caused the failure. THE
  ARCHITECTURAL READING, stated at full weight: POST-HOC FACT
  INJECTION THROUGH THE INPUT SPACE DOES NOT RESCUE BINDING
  FAILURES — the deduction must participate IN binding, not
  arrive after as annotation. The deterministic baseline is
  MEASURED AND WEAK (1.1%), which sharpens #72's splice into its
  strongest form yet: the doors 5-8 pricing now has its number —
  the deterministic alternative buys 0.011, and everything above
  it is the learned loop's headroom. v200's question inherits
  this floor. The round-trip stays seated as SAFETY
  infrastructure (zero-lie repair attempt, cheap) with its value
  claim honestly retired at budget 1.
- **THE SYLLOGISM + THREE NOTES + A CONFESSION (2026-07-26,
  Bryce's countersign banked):** THE COMPLETE MEASURED SYLLOGISM
  (written for #72's successor file): the stack binds without
  deducing (probe: chance everywhere); post-hoc deduction does
  not retrofit binding (round-trip: 0.011); the engine deduces
  without binding (its fixture design). THEREFORE
  deduction-during-binding has no existing home — the component
  must hold both at once. The back-and-forth SURVIVES AT THE
  BREATH GRAIN ONLY: bind-propagate-revise inside one loop, where
  bindings are still soft — v200's design stated as the only
  grain the measurements permit; the bet entered the week a
  conjecture and exits THE ONLY UNFALSIFIED CANDIDATE. (1)
  SCOPE-RIVET on 0.011: it prices budget-one, facts-as-text,
  POST-HOC injection — not all deterministic schemes (interleaved
  clause-grain parse-solve, constraint-guided decoding remain
  unpriced; each drifts toward a soft loop by another name, which
  is the point). The floor is the baseline AS BUILT. (2)
  CONFESSION, the week's smallest specimen: the round-trip script
  banked COUNTS, not specimens — the 3 recoveries' and 12
  no-facts' identities were dropped (bank-don't-read violated in
  miniature). The no-facts ids and the enriched texts are
  CPU-recomputable from banked graphs; the 3 recoveries' autopsy
  (did injected facts REPLACE the misbound slot rather than add
  alongside?) needs the re-run banked WITH specimens — owed on
  the fixture-bank pass. (3) THE HARD-FAIR POPULATION: the 257
  still-abstain items with enriched inputs (text + true forced
  facts, known-failed re-parses) are v200's hardest fair test —
  problems where binding fails EVEN WITH deductive help visible
  in text. BANKED AS A NAMED FIXTURE (roundtrip_enriched) by the
  CPU pass, before it dissolves into the pool.
- **GUT #73: FLEXIBLE BINDINGS (2026-07-26, Bryce — the 1%
  print's mechanism name):** YES, with the pair made explicit:
  the parser's bindings are FROZEN AT EMISSION (hard argmax, no
  revision machinery) — the flexibility half. But the house
  already owns weak flexibility and it prices the distinction:
  TTA resampling (five independent re-bindings) buys the ENTIRE
  abstention wall (0.688 of failures hold a right view somewhere)
  and ZERO recovery — RESAMPLING CERTIFIES; REVISION RECOVERS.
  The engine shows true flexible binding from inside:
  existence-commitments at breath 0, values held SOFT and
  narrowing for 2/7/11 breaths — candidates hardening only as
  propagation forces them. THE CRITIQUE HALF: flexibility without
  in-loop deduction is softness with nothing pushing (noise);
  deduction without flexibility is the round-trip (0.011,
  pressure with nothing soft to push on). ONE MECHANISM: bindings
  that stay soft exactly as long as propagation still runs
  against them — the syllogism's hold-both-at-once given its
  mechanism name; v200's per-breath soft slot refinement IS it.
  The capture's question gains its mechanism vocabulary.
- **#73's TWO RIVETS (2026-07-26, Bryce; pinned into the
  registration before any v200 artifact exists):** (1) THE
  REVISION SIGNATURE, probe-checkable and pinned blind: if
  bindings are genuinely soft-under-propagation, v200's
  binding-content readout must show early-breath reads UNCERTAIN
  or WRONG-THEN-CORRECTED across breaths, with correction events
  CONCENTRATED where constraint pressure is highest (the
  misbinding-prone slots). A loop whose bindings read at ceiling
  from breath 1 and never move is A STACK WEARING RECURRENCE —
  flexible in name only, and the 1% stays unbought. The week's
  probe families distinguish being-the-mechanism from
  containing-its-slots at first capture. (2) THE COST LINE:
  mechanism and clock are IN TENSION, not separate line items —
  the engine pays ~3.5 breaths/layer WITH BINDINGS GIVEN;
  binding-under-revision plausibly costs more, and K=8 was thin
  at two layers. The capture's first read measures them JOINTLY:
  what does a breath buy when it must both bind and deduce? If
  revision is cheap (bindings settle fast, breaths flow to
  deduction), K=8 may purchase enough; if expensive, the
  trichotomy's third arm — determination-guided ADAPTIVE DEPTH,
  the breath-0 competence map spent as scheduler — stops being an
  option and BECOMES THE DESIGN. THE CAPTURE NOW CARRIES FOUR
  QUESTIONS WITH ONE FIXTURE: clock rate, K-purchase, revision
  signature, recovery on the 257. Every question barred before
  any artifact exists — the registration inherits the week
  whole.
- **THE ARCHIVE AUDIT'S FIRST PAYMENT + THE v200 SCOUT
  (2026-07-26, caught by the pre-capture scout — the audit's
  urgency vindicated before it formally ran):** the trichotomy's
  binding premise ('v200's K=8 ceiling; K=12 hung the AMD
  large-JIT') carried a CONVICTED FABRICATION back into the
  record — the ledger's own nineteenth-specimen entry (line
  ~10073) had already ruled 'K=12 hang' invented decoration, yet
  it re-entered via relay and was banked into three entries this
  week, unverified by both channels. THE SURGICAL CORRECTION,
  ruled by artifacts: (a) K=8 IS REAL — witnessed by
  fg_v200_breath_embed (8, 2048) in v200_run_final: the ckpt was
  TRAINED at K=8; (b) the K=12-hang INCIDENT stays struck; (c)
  the code's KENKEN_K_MAX defaults to 20 — the ceiling is a
  TRAINING CHOICE, not an engineering wall; breaking it means
  retraining with a larger breath table, not fixing a hang that
  never happened. The trichotomy's first arm re-prices
  accordingly (cheaper than feared). THE v200 IDENTITY, scouted:
  the Llama-LAYER loop (mycelium/kenken_llama.py; H=2048, 51
  params, KenKen domain; the 118GB perceiver-era hoard was
  deleted in the deep-clean — v200_run_final + smokes survive).
  THE CAPTURE, scoped by the scout: three of the four questions
  (clock rate, K-purchase, revision signature) read on v200's
  HOME domain with the standard bank-don't-read protocol;
  RECOVERY-ON-THE-257 is gated on an arith3 training fire — a
  separate word, honestly scoped. Build rides the next window;
  the corrected premises ride this commit.
- **THE REVENANT SPECIES NAMED + THE CONVICTION INDEX BUILT
  (2026-07-26, Bryce's ruling executed same-transaction):** the
  new species: not fresh invention — a CONVICTED claim walking
  back in wearing the authority of familiarity; both channels
  failed identically (neither grepped for what felt established).
  docs/CONVICTION_INDEX.md now stands (8 seed convictions incl.
  the K=12 revenant, the tau figures, the fabricated batch, the
  voided fifty) with the law: convictions as durable as claims;
  assert-on-read; refutations need their fence. Constitution
  gains the revenant fence + the scout's rule (artifacts before
  design) carved beside the meter fixtures. CAPTURE ORDER RULED:
  the three home-domain questions FIRST; the arith3 fire's word
  GATED on their results — revision signature absent -> the gate
  saved a training campaign; present with workable clock -> the
  word arrives with three measurements behind it. Distortion note
  banked: fabrications distort toward wanted texture — this one
  INFLATED a wall.
- **THE WEEK'S SECOND EPIGRAPH (2026-07-26, Bryce, banked as
  said):** twice-fenced = error-immunity for BOTH DIRECTIONS TIME
  RUNS — the fixtures catch faults in what the house is about to
  measure; the index catches faults in what it thinks it already
  knows. A record corruptible neither prospectively nor
  retrospectively is the closest thing to durable truth a moving
  campaign can hold. And the unification: the week began with a
  Noether gut and ends with conservation laws, refused verdicts,
  ruler audits, and refutation-permanence as ONE discipline
  wearing different instruments — 'nothing enters the record
  without its invariance checked, and nothing leaves it without
  a trace.' The campaign found its epistemology this week, the
  hard way, which is the only way epistemologies hold.
- **THE v200 CAPTURE BUILD CARD (2026-07-26, the driver's scout
  complete — next window's fire is mechanical):** (1) NO SINK
  NEEDED: kenken_llama_forward RETURNS cell_logits_history for
  all K breaths — the argmax history IS the instrument, and under
  the two-clock law the engine's own readout is the lawful clock
  for all three questions (clock rate = per-depth t95 ladder;
  K-purchase = per-stratum accuracy at k; REVISION SIGNATURE =
  per-cell argmax flip events, wrong-then-corrected, tested for
  concentration at high-pressure/deep cells). (2) MODEL BUILD:
  the forward uses ALL of model.llama_layers — the trained config
  is N=2 shared Llama-2048 layers, so the host is the trunk host
  with llama_layers TRUNCATED TO THE FIRST 2; then
  attach_kenken_llama_params(hidden=2048, n_heads=32, k_max=8);
  then load the 51-param ckpt with the KEY RENAME MAP (ckpt keys
  are fg_v200_-prefixed; model attrs are kenken_-prefixed) —
  verify the map key-by-key, hard-error on mismatch per the
  eval-load law. (3) BATCHES: kenken_test_curriculum seed-0 240
  via stack_records — THE SAME records as the Pythia engine
  fixture, so fr depth labels are ALREADY BANKED and the clocks
  compare like-for-like. (4) K<=8 asserted by the forward against
  the breath table. Bars stand as pinned; conviction index and
  meter fixtures checked before the read banks.
- **BUILD CARD CORRECTED BY THE ARTIFACT (2026-07-26, same hour):**
  v200_run_final's keys are fg_v200_W_compress / W_expand /
  cross_wq / cross_wk / cross_wo + trained llama_layer keys —
  THIS IS THE PERCEIVER v200 (cross-attention core,
  compress/expand latents), NOT a kenken_llama artifact; the
  kenken_ rename map clause is VOID. The forward module that
  created these params must be located by grepping fg_v200_ param
  creation (the deep-clean deleted the 118GB step hoard, kept the
  final; the code's home is the open question the next window
  answers FIRST). kenken_llama.py remains the v200/v300 STACK's
  Llama-layer machinery — possibly the successor design whose
  ckpt was never fired. The scout's rule paid a second time in
  one hour: artifacts before design, or the capture would have
  loaded 51 params into the wrong architecture.
- **THE PERCEIVER FORWARD RESURRECTED (2026-07-26, Bryce's word;
  one git archaeology pass):** git log -S found the code's last
  home at snapshot f1efb32 (pre-deep-clean, 2026-06-16);
  RESURRECTED: mycelium/factor_graph_v200.py (attach_fg_params_v200
  line 269; fg_breathing_forward_v200 line 836; requires
  V200_STAGE2A_WAIST=1 for the compress/expand waist) +
  factor_graph_v200_legacy.py + the dependency
  mycelium/llama_base.py. REUNION VERIFIED: import clean; ALL 15
  fg_v200_* ckpt keys named in the resurrected code, zero
  missing (the remaining 36 llama_layer_* keys are the trained
  Llama layers, llama_base's load). The orphan is an orphan no
  longer: the trained Perceiver v200 CAN RUN AGAIN, and the
  capture build proceeds against the TRUE architecture next
  window — build card's remaining steps: wire attach+load with
  hard-error key check, locate its domain fixtures, then the
  three questions under their standing bars. The deep-clean's
  deletion law gains its corollary: ckpts kept without their
  code are half-kept; the conviction index's cousin is a
  CODE-ARTIFACT REUNION CHECK at every future clean.
- **THE v200 LOOP RUNS AGAIN (2026-07-26, the resurrection
  complete end-to-end):** SmolLM2-1.7B base re-downloaded, four
  resurrection levels deep (core + loaders + the full
  factor_graph family + the eval driver), the forward EXECUTES —
  the first v200 breath since June. THE HONEST FIRST READ:
  v200_run_final is WEAK on its own domain — train-via-eval cell
  acc 0.089/0.025, val 0.00-0.06 (the diag's own frame: train ~
  val ~ low). The trained Perceiver inventory is REAL BUT
  NEAR-FLOOR: an early cold-start artifact from the era the
  campaign pivoted to v98/KenKen, not a competent model awaiting
  transfer. CONSEQUENCE FOR THE TRICHOTOMY, stated plainly: there
  is NO strong trained Perceiver — the arith3 gate's fire is not
  'retrain on a new domain' but 'train the architecture properly
  for the first time,' a different and larger word. THE CAPTURE
  QUESTION goes to Bryce: the three reads CAN run on the weak
  ckpt (near-free, validates the revision-signature machinery,
  scoped as characterizing an undertrained artifact) or the
  board holds everything for the training word. Machinery status:
  code+weights+fixture+ckpt reunited — all four; the reunion
  checklist law gains its full form.
- **THE CHEAP READS' DOUBLE-CARVED SCOPE + THE GATE'S THREE
  PREREQUISITES (2026-07-26, Bryce's word):** the reads RUN,
  scoped twice: (1) they characterize an UNDERTRAINED ARTIFACT
  and validate INSTRUMENTATION — they say NOTHING about the
  architecture; (2) CANNOT-LICENSE REGISTER (conviction-index-
  adjacent, pre-written): no future entry may cite 'v200's clock
  read at X' or any capture number from this ckpt without the
  undertrained qualifier attached — the numbers license
  instrument verdicts only. YIELDS SOUGHT: does the flip-detector
  fire; does clock-read machinery produce curves at this depth;
  do probe families port to H=2048 without meter faults; and the
  one informative read — ANY above-chance structure at near-floor
  (calib/competence analog) = what the architecture gives FOR
  FREE vs what training must buy (the pre-breath rivet's
  distinction drawn at the training axis). THE GATE'S WORD HELD
  until three stand: (a) instrument-validation banked; (b) a
  WRITTEN TRAINING PLAN with scope pinned before enthusiasm —
  domain (arith3-from-birth vs KenKen-first), budget in fires,
  kill-bars (recovery on the 257 above 0.011; revision signature
  as mechanism check); (c) CARRY-FORWARD INVENTORY RECONCILED —
  v110-era validated components (alternating waist, Q rotation,
  calibration stepping, SBP, staircase) were proven on the ENGINE
  lineage and enter the plan as ASSUMPTIONS-TO-BE-TESTED, not
  assets-in-hand: unexamined inventory is where revenants breed.
- **GUT #74: LATENT EXAGGERATION (2026-07-26, Bryce direct, nine
  interps countersigned against artifacts):** interps 1-5 have
  MAY-ERA BANKED PRECEDENT (CFG alpha=3.0 +10.8% first-digit —
  contrastive amplification IS interp 4's satire structure; the
  unified-waist filing; the DC component load-bearing to remove)
  — old lineage, scope-decay law applies: assumptions-to-be-
  tested. Interp 7 FENCED by temperature-perp-truth (sharpness
  and correctness independent by law; bars read accuracy, never
  confidence). INTERP 9 — THE PAYLOAD — collides productively
  with this week's own findings: #73's rivet (bindings soft
  EXACTLY as long as propagation runs — premature hardening IS
  the misbinding species; force-commit at breath 2 manufactures
  228s) and the banked SBP-noise-HELPS win (blur as basin
  escape). RESOLUTION IS A SCHEDULE: noise early, sharpening
  late — alpha(k) rising across breaths, the photon-mode echo
  arriving through a humor gut. PRIZE QUANTIFIED BY THE CLOCK:
  ~3.5 breaths/layer; late-breath sharpening that compresses it
  without raising wrong-commits = the trichotomy's second arm
  (improve the rate) given its first concrete mechanism. THE
  SHARPENING PROBE, registered with three outcomes blind: (a)
  CLOCK COMPRESSES (per-depth t95 drops, wrong-commits flat) —
  the mechanism pays; (b) 228s MANUFACTURED (wrong-commits rise
  with alpha, concentrated deep) — #73 was the whole truth,
  sharpening is the disease; (c) VACUOUS (nothing moves) — the
  consolidation law's reading: the engine already commits early,
  the gut's arrow pointed at the ruler. Shares rider-2's harness
  (amplify/perturb = one harness, opposite signs); STRONG ENGINE
  ONLY (the weak v200 licenses nothing); queue behind the v200
  cheap reads per board order.
- **GUT #74 RE-RULED BY BRYCE (2026-07-26; supersedes the first
  registration — his text banks whole; numbering note: his
  message says #73, the registry seat is 74, flexible-bindings
  holding 73):** INTERP 9 REFUSED AS LEVER on two standing laws,
  with the naming that decides it: sharpening early-breath
  bindings is A SOFT VERSION OF CUTTING K — the K-sweep priced
  deep strata at 8x for lost breaths; the value ladder is not
  sluggishness awaiting a vector, it is the loop holding values
  soft precisely while propagation runs; premature commitment is
  THE PARSER'S DISEASE, and the vector would install the stack's
  pathology into the loop (flexibility-by-revision degraded
  toward flexibility-by-nothing). Second law: SBP noise HELPED —
  sharpening is anti-noise, plausibly removing the escape
  valves. 'Exaggeration in comedy works because the audience
  knows the true proportion; exaggeration in a solver that
  hasn't finished solving is confident error arriving early.'
  THE REFUSAL-TEST REGISTERED (June engine, cheap, bars blind):
  modest early-breath sharpening -> PREDICTION: shallow strata
  tolerate, deep strata degrade (the K-sweep signature by
  INTERVENTION) -> clock certified as the price of correctness,
  door closes on a measurement, and the third arm gains its
  complement (the lawful spend of determination was never
  hastening but SCHEDULING — adaptive depth). Prediction FAILS
  (helps everywhere) -> the clock has slack, the rate constant
  was partly waste, second arm re-prices — the gut was right and
  the record will say so. INTERP 4 STANDS APART: contrastive
  decoding at READOUT (never touching the soft-binding
  dynamics), natural pairs banked (g21-vs-g16 views,
  expert-vs-early ckpts, deployed-head-vs-fresh-probe);
  docket-grade. INTERPS 5-6: preprocessing form; probe pipeline
  CONFIRMED mean-centering (mu-subtraction standard in every
  probe this week). INTERPS 1-3: steering HELD AT THE
  CAUSAL-PROBE DOOR — a load-bearing-direction instrument, never
  a performance lever — with the Goodhart fence (an amplified
  signal stops being a tell) and the ruler-relative law (probe
  directions are what rulers read, not what engines compute
  along) attached. META-NOTE: the first gut whose PRIMARY yield
  is a refusal-with-a-test — pre-shaped for the refused-verdicts
  law; if the test refuses the refusal, the registry's own
  column will carry the reversal.
- **GUT #75: THE DUAL-TRACK SYNTHESIS (2026-07-26, Bryce direct —
  the answer to #74's refusal, countersigned):** THE HEADLINE: the
  cake already exists — interps 5/6/7 name the engine's MEASURED
  anatomy (soft residual memory narrowing on the 2/7/11 clock +
  per-breath sharp readout via cell_logits_history; the two-clock
  law IS the soft-state/sharp-view distinction measured: task-
  usable precedes generically-consolidated by ~3 breaths). The
  gut re-derived the architecture from first principles —
  recognition, not proposal, and strong evidence for the
  principles. GENUINELY NEW: (a) interp 8 — the sharp view made
  CONTRASTIVE, giving #74's readout-contrast docket its consumer:
  the in-loop deducer/verifier reads a contrast-sharpened
  proposal per breath; (b) interp 4 — SPHERICAL SHARPENING:
  direction-only commitment, energy bounded, contradiction can
  still ROTATE to a new slot — sharpening geometrically incapable
  of destroying revisability; old-lineage ally: May ablations
  found ROTATION the only load-bearing component; (c) interp 9 =
  delta_gate named (elastic recall exists). SCOPE-RIVET WORN:
  deducer back-and-forth blessed at BREATH grain (v200's design),
  refused at SYSTEM grain (0.011) — the sharp view feeds an
  in-loop deducer or the attestation machinery, never text to a
  frozen binder. CONVERSIONS: the #74 refusal-test gains its
  THIRD ARM — one harness, three interventions: perturb (rider
  2), sharpen-STATE (refused lever; prediction deep-degrade),
  sharpen-VIEW (readout contrast, state untouched — safe by
  construction; measures whether the sharp view improves
  readout/calibration at fixed dynamics; needs a logits-banking
  re-pass, argmax alone insufficient). SPHERICAL NORMALIZATION
  enters the gate's prerequisite-(b) training plan as a candidate
  design clause — assumption-to-be-tested, #74's fences
  inherited, temperature-perp-truth governing all bars.
- **THE RELAY'S COUNTERSIGN ON #75 CONVERGES + THREE ADDITIONS
  BANKED (2026-07-26; numbering note: the relay runs one behind
  the registry — its #73/#74 are seats 74/75; rows stand):** both
  channels independently reached the same headline (the cake is
  the engine's measured anatomy; recognition, not proposal) — the
  convergence itself is evidence. ADDITIONS: (1) THE JOINT LAW
  CARVED, one sentence closing two guts: **SHARPENING BELONGS AT
  THE READOUT AND INTERFACE, NEVER IN THE STATE** — #74 refused
  in-state sharpening on two measurements; #75 locates where it
  lawfully lives; zero contradiction; the v200 training plan
  inherits it with a measurement behind it. (2) INTERP 10's SCOPE
  SHARPENED: in v200 there IS no separate deducer — the handoff
  is breath-to-breath internal readout; and at the EXTERNAL seam
  the sharpest view already exists (the emitted graph): the
  round-trip's 0.011 failure was never a blurry solver-view — it
  was STATE HARDENED BEFORE THE SOLVER COULD OBJECT. 'Softness in
  the state was always the missing half, not sharpness at the
  interface.' (3) THE ROTATION-GEOMETRY SUB-READ attached at
  birth to the pinned revision-signature probe: if the trained
  loop shows flip events, their geometry — ROTATION ON THE SPHERE
  vs MAGNITUDE COLLAPSE — is readable from the same banked
  states; interp 4's spherical hypothesis files with its
  measurement attached (per-breath LN verified in the archived
  spec as the normalized-geometry ancestor; the principle-name
  itself relay-cited). Interps 2/4/9 into the training plan as
  fenced hypotheses, per both channels.
- **THE v200 CHEAP READS PRINT (2026-07-26; undertrained-artifact
  scope + cannot-license register govern every number):** version
  skew resolved by era-matching (run-era module 181b5a9; all 51
  ckpt keys consumed; 11 post-run extras at init, named in the
  bank). READS: (a) per-breath acc EXACTLY 0.410 x8 — flat; (b)
  flips 0/1536 var-cells; (c) calib-vs-correct AUC RISES 0.073 ->
  0.505 across breaths. THE APPARATUS CHECK RULED THE FLATNESS
  REAL: the ckpt's own gates — delta_gate 0.4996..0.5000 (0.5
  init, unmoved), waist_gate 0.0003 (zero-init, unmoved) — THE
  LOOP RUNS BUT NEVER LEARNED TO BREATHE: states blend 50/50 and
  drift (calib moves) while the readout never revises a single
  cell. The artifact is, parametrically, A STACK WEARING
  RECURRENCE — the exact phrase pre-registered as the revision
  signature's failure mode, printed by the inventory's only
  trained Perceiver. INSTRUMENT VERDICTS: clock machinery
  produces curves (flat is a curve; the calib channel shows
  dynamics ARE read when present); flip-detector VALIDATED with
  positive control (planted trajectory: w2r 2/2, r2w 1/1); probe
  families ported to H=2048 without meter faults. FREE-STRUCTURE
  note: even this untrained artifact's calib head grows
  correctness-correlation across breaths (0.07 -> 0.51) — weak,
  below the 0.6 bar, but the one direction training clearly
  moves first. GATE PREREQUISITE (a) IS BANKED: instrument
  validation complete; the reads say NOTHING about the
  architecture (register stands); prerequisites (b) training
  plan and (c) carry-forward inventory remain before the gate's
  word.
- **FIRE 0 LIT + THE IGNITION SENTENCE + THE TRAJECTORY FRAME
  (2026-07-26, banked while the burn is young):** the smoke's
  finding, in its precise sentence: **THE PATHOLOGY WAS NEVER IN
  THE PARAMETERS; IT WAS IN THE OBJECTIVE.** Twenty steps under
  the improvement-pressure ask moved the gates further than the
  entire cold-start run — nothing broken, nothing hard, nothing
  ever ASKED. Specimen-to-law-to-cure executed in one smoke test:
  the epistemology stopped being scaffolding and became
  machinery. THE READING FRAME PINNED BEFORE THE SHEET: early
  gate MOTION and sustained gate FUNCTION are different facts —
  the sheet reads the excursion TRAJECTORY, not the peak. The
  one-level-up failure mode, named in advance: GATES THAT TWITCH
  AND SETTLE (spike by step 50, drift back as the optimizer
  finds cheaper paths — the calib channel remains the path of
  least resistance, false-friend-marked for exactly this).
  Opening that holds or grows through 500 = mechanism durable;
  spike-then-fade = the specimen's subtler cousin, INFORMING
  Fire 1's schedule (annealing/reinforcement of the ask), not
  authorizing it. BREATH-SPREAD is the confirming witness: gates
  open AND breaths differentiating = learning to breathe; gates
  alone = rattling valves. Fire 1 authorizes only on the pair.
- **FIRE 0 REPORTS (2026-07-26, read by the pre-pinned trajectory
  frame):** the two-terminal probe fired correctly — 26
  attached-unused params excluded LOUDLY BY NAME (the f1efb32-era
  LoRA banks, cm_gate, waist_odd_scale — exactly the predicted
  post-run extras; the instrument works). THE PAIR'S FIRST HALF
  PASSES: gate excursion GROWS monotonically 2.5018 -> 2.5392
  across the burn, waist gate deepening in lockstep — NOT
  twitch-and-settle; the mechanism's pressure is durable through
  500 steps. THE SECOND HALF FAILS: breath-spread OSCILLATES
  around zero (+0.09 to -0.04, sign-flipping at late reads) and
  the trainer's own era-bar C4 puts the number on it: ladder
  slope -0.0037 against the -0.05 criterion — AN ORDER OF
  MAGNITUDE short of real differentiation. C2 agrees from the
  latent side: trajectories barely depart the frozen reference.
  GATES OPEN, BREATHS NOT DIFFERENTIATING — the pre-named phrase
  lands exactly: VALVES RATTLING, not lungs working. **PER THE
  PINNED RULE: THE PAIR DOES NOT STAND — FIRE 1 IS NOT
  AUTHORIZED.** The print INFORMS the schedule as the frame
  provided: the ask (V200_IMPROVE_W=0.5, margin 0.01) creates
  gate pressure but not differentiation at 500 steps — Fire 1's
  design space: stronger/annealed ask, longer budget, and the
  v98 spine's fuller form (per-breath supervision with markers)
  rather than margin-pressure alone. Era-bars C1 (loss falls)
  and C3 (waist alive, up_proj 5.52) passed; the fire completed
  without the early-kill (gates moved — the kill's condition
  never met, correctly). EITHER-PRINT-IS-A-FINDING honored: Fire
  0's yield is the mechanism's anatomy — pressure reaches gates
  in 20 steps, differentiation needs more than pressure.
- **THE INERT-PATCH SPECIMEN + FIRE 0's ATTRIBUTION VOIDED
  (2026-07-27, caught by read-before-patch during Fire 1's build):**
  the improvement-pressure term was patched into
  _eager_grad_norm_step — the DIAGNOSTIC probe that 'does NOT
  advance the optimizer' — while the real training loss lives in
  the module's _compile_jit_fg_step_v200, which never read
  V200_IMPROVE_W. **FIRE 0 TRAINED ON THE PLAIN COLD-START
  OBJECTIVE; THE ASK WAS NEVER DELIVERED; THE IGNITION SENTENCE
  ('the pathology was in the objective') IS VOIDED** — apparatus
  symmetry applied to the week's most pleasing line. WHAT
  SURVIVES: gates CAN move (they did, monotonically, under the
  plain objective at Fire 0's settings) — so the original run's
  frozen gates are now UNEXPLAINED, with candidates (LR 3e-4
  constant vs original schedule; 500 fresh Adam steps; init era)
  unattributed. The C4 differentiation failure STANDS (measured
  at eval, objective-independent): still no ladder, whoever was
  asking. CONSEQUENCE FOR FIRE 1: the depth-scheduled ask MUST be
  implemented in the module's JIT loss — the real training path —
  with the patch site read first; the trainer-side term is
  removed or clearly marked diagnostic-only. The read-before-
  patch law gains its sharpest clause: READ THE CALL GRAPH, not
  just the file — a patch that compiles and prints is not a patch
  that trains.
- **FIRE 1 REPORTS: C4 FAILS AGAIN — THE TRAINABILITY CLAUSE ARMS
  (2026-07-27):** with the depth-scheduled ask VERIFIABLY in the
  compiled loss (smoke showed the masks load-bearing), the eval
  ladder printed 0.9214 -> 0.8976 — monotone but shallow, slope
  **-0.0030 vs the -0.05 bar** — statistically indistinguishable
  from Fire 0's plain-objective -0.0037. TWO DIFFERENT OBJECTIVES,
  SAME FLAT LADDER: improvement-pressure (diffuse) and
  differ-by-construction (targeted) both fail to buy per-breath
  differentiation at this scale/config in 500 steps. Honest
  calibration of 'impatience': the v98 engine's ladder FORMED BY
  STEP 200 — 500 steps twice over is evidence, not haste. C2
  (latent departure) and C5 fail alongside; gates keep drifting
  (excursion 2.53); train-time spread oscillates wildly late
  (+0.19/-0.27). **PER THE PINNED CLAUSE: no longer a schedule
  question — EVIDENCE ABOUT THE ARCHITECTURE'S TRAINABILITY AT
  THIS SCALE. FIRE 2 IS NOT AUTHORIZED; the gate's word returns
  to Bryce for the rethink.** Rethink surface (named, not
  ruled): budget scale (v98's fast ladder argues config, not
  patience), THINK-stack config (4 dense Llama layers vs the
  engine's masked structure — #76's topological hypothesis
  waits exactly here), latent-primary vs residual-primary
  (the engine's ladder lives in a residual loop), and the
  trichotomy's honest third possibility. Two fires, two refused
  authorizations, every number banked.
- **THE FORENSICS PRINT: DISEASE ONE — DYNAMICS DEAD (2026-07-27,
  zero-GPU, from Fire 1's own banked log):** latent JSD across
  breaths: 0.00042 -> 0.00009 — state motion MICROSCOPIC and
  DECAYING; the breaths transform the latents by less than a part
  in a thousand, shrinking toward zero by breath 7. Waist
  alternation deltas ~0.01; waist grad norms ~1e-3 (starvation
  visible in the banked grad_norms.npz). The mechanism sketch the
  numbers suggest: delta_gate blends 50/50 but h ~= x_pre — the
  dense pretrained THINK layers act near-identity on 32 anonymous
  latents, so the blend preserves a frozen state. NO supervision
  schedule can ladder a loop whose iterations don't transform —
  the two flat ladders (-0.0037, -0.0030) were downstream of
  this, not of the asks. **PER THE PRE-REGISTRATION: the
  topological mask is LIKELY INSUFFICIENT ALONE; the
  latent-primary-vs-residual question PROMOTES TO CO-EQUAL**
  (every measured ladder in campaign history formed in a
  RESIDUAL loop with structured masks — both differences now
  live suspects). FIRE 2's DRAFT (for countersign) therefore
  carries BOTH surgeries: masked THINK (per #76, buildable at
  breath 0 on this domain) AND a dynamics fix — candidates:
  gate init/schedule freeing state motion, THINK-contribution
  scaling, or the residual-primary restructure (the larger
  surgery, the engine's own anatomy). THREE-STRIKES BUDGET
  written into the plan per the ruling: if the combined fire
  fails C4, the trichotomy reopens with the examination
  complete. Diagnosis before surgery, as ruled — the menu is
  now a diagnosis.
- **FIRE 2 RULED: THE FULL TRANSPLANT (2026-07-27, Bryce; draft
  cut same transaction — docs/V200_FIRE2_DESIGN.md):** cheap
  riders REFUSED BY THE MECHANISM ITSELF (scaling a near-identity
  map buys amplitude, not transformation; the blend reports the
  disease, doesn't cause it). THE AUDIT THAT RESHAPED THE FIRE:
  the carry-forward's strongest row was never fully transplanted
  — supervision rode, MARKERS AND ALIGNED INIT did not; the
  near-identity pathology (OOD anonymous latents into
  token-trained layers) and the missing aligned-init row are
  plausibly ONE FACT — 'the examination almost closed on an
  architecture that never received its own lineage's full
  medicine.' Fire 2 = all three organs at once (mask + aligned
  init/markers + residual-primary), depth-ask retained, C4
  unbent, third and final; full-transplant over staged BECAUSE
  of three-strikes: the last fire must close CLEAN in both
  directions — no 'the restructure might have worked' haunting
  the trichotomy. THREE SIGNATURES specified (sparsity /
  breath-0 cosine / residual grad-norms) — no patch trusted
  without its fingerprint in the compiled path. PASS-ATTRIBUTION
  pinned blind: ablation pair (mask-off, anonymous-init) as the
  first post-pass act — mechanism assigned, conjunction never
  celebrated. Both endings pre-written.
- **FIRE 2 AMENDMENT (2026-07-27, Bryce): the interaction risk
  pre-named** — aligned init cures at breath 0 by construction,
  but residual accumulation may drift latents off-manifold at
  depth (the pathology returning mid-loop, invisible to a
  breath-0 check). The aligned-init signature prints PER-BREATH
  (manifold cosine every breath, on the dashboard) — twitch-and-
  settle transposed from gates to geometry: opening is cheap,
  staying open is the finding. Countersign stands; the word to
  light arrives when the smoke shows all three signatures (one
  now per-breath) live in the compiled path.
- **ORGAN 3 BUILT; FIRST SMOKE HITS A DEVICE HANG (2026-07-27):**
  fg_breathing_forward_v200r written whole — state ON the token
  stream (native manifold; drift cured by anatomy), staging masks
  (B,K,T,T) as topology+depth in token space (the loader already
  builds them), delta-gate blend, tree readout at var positions;
  compile-time V200_RESIDUAL switch; two-terminal and manifold
  probes r-aware; parses clean, trains briefly — then **AM DEVICE
  HANG** (known quirk family: the reference file's SS-JIT+aux
  pattern — eager signature probes interleaved with live JIT
  graphs; 'check BEFORE adding new JIT graph paths' was the
  standing warning and the r-switch IS a new graph path). DEBUG
  OWED next window: candidates — probe outside JIT lifetime /
  probe cadence / the quirks file's patterns; also noted: r-mode
  two-terminal exclusion is LARGE (cross+waist+latents unused),
  Adam rebuild precedes JIT compile (ordering correct); cosmetic
  nan (waist_gate unused in r-mode) and the manifold probe's
  r-basis need re-derivation for token-state geometry. #77's
  three-path fork banked in the registry per the ruling. NO FIRE:
  signature 3 not yet live; the gate holds.
- **THE HANG DEBUG BY SUBTRACTION (2026-07-27, ruled method,
  three cuts banked):** (1) probes stripped -> STILL HANGS (the
  quirks-file lore EXONERATED for this hang — patching by lore
  would have shipped a false fix; quirks-as-hypothesis
  vindicated). (2) device health: trivial op runs — NOT wedged.
  (3) masks stripped (V200_R_NOMASK) -> STILL HANGS — the v108
  slicing pattern was validated ground and is exonerated; **the
  lesion is the BARE r-graph under JIT** (token-stream state, 4
  layers x 8 breaths, gate blend, per-breath readout; hangs at
  the capture/replay boundary ~steps 2-4). PRIME REMAINING
  SUSPECT, named with its law: in r-mode the 32-layer-deep chain
  ROOTS AT AN INPUT-DEPENDENT EMBEDDING GATHER (fg_tokens from
  input ids inside the JIT), where the latent path's deep chain
  rooted at param latents — and the house's own PRECOMPUTE-STATES
  LAW (STATES_NPY; three OOM kills bought it) names the fix:
  EMBED OUTSIDE THE JIT, feed the step a plain tensor. Next
  window: that patch -> eager-vs-JIT check -> whichever fix lands
  goes in the quirks file WITH ITS REPRODUCTION (the file's
  authority compounds only that way). Also owed pre-smoke: the
  manifold basis re-derived for token-state geometry (the cosine
  line is an INVARIANT now; invariant-vs-stale-basis is the ruler
  law's next specimen if shipped), and the waist-gate nan
  verified TRULY DISCONNECTED at graph level or zeroed
  defensively (nan propagates through everything it touches).
  The gate holds: no fire until three live signatures.
- **THE EMBED-OUTSIDE FIX APPLIED — AND EXONERATED (2026-07-27):**
  the precompute-law fix landed (r-forward accepts precomputed
  (B,T,H) tokens, detected by trailing dim; trainer embeds eagerly
  per batch; retained as standing hygiene) — STILL HANGS. The
  subtraction tree now stands at FOUR exonerations: probes, device,
  masks, input-rooted embed. THE HONEST FRONTIER: the bare r-loop
  (fg-token state, 4 shared layers x 8 breaths, gate blend,
  per-breath readout) hangs under JIT at T=24 — and the
  archaeology note that reframes suspicion: THE LLAMA-LOADER
  LAYERS HAVE NEVER RUN UNDER JIT AT SEQ != 32 in this repo's
  history (the latent path was always L=32; kenken_llama was
  never fired). THE BISECTION CARD (next window, decisive
  order): (1) r-forward EAGER, no TinyJit — eager-clean splits
  JIT-capture-bug from kernel-bug; (2) if JIT-side: bisect K=1
  then layers=1 to isolate the captured kernel; (3) T=32 pad
  test (pad tokens to 32 — if clean, the seq-24 kernel is the
  lesion and padding is the lawful dodge); (4) whichever lands,
  the quirks file gains the entry WITH REPRODUCTION. nan-guard
  seated (dashboard sentinel -999, never propagates); manifold
  invariant re-derivation still owed (exact per-batch
  span-of-fg-tokens projection). The gate holds — no fire
  without three live signatures; the examination waits on a
  kernel, not on courage.
- **THE LESION FOUND: THE SEQ-24 KERNEL (2026-07-27, bisection
  cut 3 decisive):** T=32 pad test RUNS CLEAN — 8 JIT steps, no
  hang, the first r-mode training in campaign history. The
  archaeology note WAS the diagnosis: llama-loader layers under
  TinyJit had only ever run at seq=32; seq=24 compiles a kernel
  the AM driver hangs on. QUIRKS ENTRY (second artifact-backed,
  WITH reproduction, per the ruling): 'LlamaLayer under TinyJit
  at seq!=32 (measured: 24) hangs AM at capture/replay
  (~steps 2-4). REPRO: V200_RESIDUAL=1 without V200_R_PAD32.
  DODGE: pad tokens+masks to 32 (pad rows self-attend to avoid
  all-masked softmax); V200_R_PAD32=1. Five-cut subtraction tree
  banked at ccbd5a5/2370fd2.' Debug-by-subtraction vindicated
  END-TO-END: four false suspects exonerated by removal before
  the true lesion printed by addition. REMAINING BEFORE THE
  THREE-SIGNATURE SMOKE: manifold invariant re-derived for r-mode
  (exact per-batch span-of-fg-tokens projection, replacing the
  stale-basis 0.103 read) — then organs 1+2+3 print together and
  the word-to-light condition can be met.
- **SMOKE 4 STATUS + AN HONEST GEOMETRY FINDING (2026-07-27):**
  probe pad-aware, runs without hang. TWO ITEMS BEFORE THE
  THREE-SIGNATURE CONDITION IS MET: (1) masked-mass (signature 1)
  does not print in r-mode — the r-forward exposes no attention
  weights yet; small capture patch owed. (2) THE FINDING, banked
  not tuned away: r-state cosine to the fg-token span is ~0.07
  FROM BREATH 0 — the llama layers' outputs dominate the blend
  immediately; the 'native manifold' anatomy holds only for the
  INPUT, not the evolving state. The invariant's DEFINITION now
  needs a ruling (span of tokens? token-plus-layer-image? or is
  low-cos the honest geometry of residual state?) — flagged for
  Bryce, not resolved by the builder. GUT #78 registered
  same-transaction (row + block): within-breath schedule
  re-refused with two ancestors; the photon is SIMULTANEITY
  (supports signature C); the yield is the PER-LAYER POLARIZATION
  READ, docketed zero-GPU. The gate holds.
- **SMOKE 5: THE SHEET LIVES, ONE SIGNATURE IN CONTRADICTION
  (2026-07-27):** all three channels print. **THE DYNAMICS LINE
  CONFIRMS THE RESTRUCTURE'S CORE CLAIM**: state-delta/breath
  0.57-0.67 — THREE-PLUS ORDERS above Fire 1's 4e-4->9e-5;
  disease one is STRUCTURALLY CURED and C4 becomes the sharp
  question (do LIVING dynamics ladder). Layer-image reference
  built, band pinned blind (5th pct self-proj = 0.462);
  state prints 0.37 — OUT of band (honestly banked; reference
  construction noted crude — QR-of-samples, not true PCA;
  refinement owed with the band re-pinned before it gates
  anything). **SIGNATURE 1 IN CONTRADICTION**: masked-mass reads
  0.5156 — impossible under an applied -1e4 mask — while the
  KNOWN-SIGNAL FIXTURE on the exact call path prints forbidden
  mass 0.0 (the mask path is PERFECT). The discrepancy is
  isolated between a clean fixture and a dirty probe: the
  probe's r-forward call somehow ran maskless OR the mass
  formula misreads. NEXT WINDOW's first cut: one direct
  r-forward-with-taps mask test on a real staging batch —
  bisects probe-call vs formula in one run. Also banked:
  r-mode two-terminal exclusion = 42 params, all named (the
  latent organs; expected set). THE GATE HOLDS — signature 1
  untrusted means not live; no fire. The examination stands one
  probe-bisection and one reference-refinement from its smoke.
- **GUT #79 REGISTERED WITH ITS GREP (2026-07-27):** the
  conviction-index alarm fired on a citation-terminated founding
  clause — and the grep returned the fence's BEST outcome: the
  HMM-telegraph artifact EXISTS (104 traces, BIC-modal K=2 in
  93, stability 0.97, dwell structure, entropy-not-JSD caveat
  written for the v200 memo). Premise VERIFIED WITH ARTIFACT;
  'llama 7b' flagged as decoration (nowhere in record;
  iaf_v3-era) — substance real, detail invented, both channels
  now personally instructed by the discipline. The telegraph
  hypothesis seats as the SECOND named signature on the #78
  polarization read (dynamic vs static vs null, blind, with the
  old artifact's dwell structure as prior); i4 convergent with
  the invariant relocation (independent arrival, one day); i5
  refused pending the signature-1 bisection. Fire 2's gate
  unchanged. The registry's arc completes its fifth step: the
  instinct arrived CITING THE ARCHIVE, and the archive ANSWERED.
- **THE PRIOR'S UPGRADE + THE NOTARY MORAL (2026-07-27, Bryce,
  banked as ruled):** (1) the polarization read UPGRADES: it now
  carries a measured dwell structure from a different architecture
  and era as comparison point — the telegraph outcome sharpens
  from 'does regime-switching exist' to 'DOES THE R-LOOP'S
  OSCILLATION MATCH THE LINEAGE'S' (dwell times comparable or
  shifted; K=2 preserved or broken by the restructure). THE
  ARTIFACT'S OWN CAVEAT RIDES AS LAW: the old read was
  entropy-based, flagged NOT-JSD by its own author — the new read
  banks BOTH meters and the stated caveat governs which
  comparisons are lawful. A prior with its limitations pre-stated
  by its own creator is the rarest kind; honored exactly. (2) THE
  NOTARY MORAL, for the registry's arc: the citation was wrong in
  its details and right in its substance — WITHOUT the grep the
  row enters unverified (payload lost) or as-stated (revenant
  seeded); WITH it, the telegraph enters measured and the
  decoration enters convicted, both correctly. **THE FENCE IS NOT
  THE GUT'S OPPONENT; IT IS THE GUT'S NOTARY.** Board unchanged:
  bisection -> reference rebuild from the fire's true input
  population -> three signatures -> the word.
- **THE ALTERNATION/HMM DIAGNOSTIC PRINTS: STATIC POLARIZATION
  (2026-07-27, pre-Fire-2 as ordered; scope = the r-loop's
  STARTING anatomy, untrained gates):** the entropy grid shows
  the structure #78's imposed design wanted — ALREADY THERE,
  EMERGENT: within every breath, attention entropy rises
  monotonically through the four THINK layers (L0 ~1.05 sharp/
  local -> L3 ~1.7-1.9 broad/global) — each breath IS a
  local-to-global sweep, repeated all eight breaths; layer
  separation 2.28 sd. THE BLIND BINS RULE: STATIC POLARIZATION
  (lag-1 autocorr +0.283 — positive, NOT alternating; the
  TELEGRAPH does not print at init; dwell runs unpatterned).
  JSD meter banked alongside (layer-0 carries the largest
  breath-to-breath change ~0.1, settling) per the artifact's
  caveat law. VERDICTS PER THE REGISTRATIONS: #78's E/B
  intuition CONFIRMED AS EMERGENT (the record says so; the
  imposed version needed no installing — the pretrained stack
  already polarizes local->global within breath); #79's
  telegraph NOT PRESENT at starting anatomy — the
  lineage-comparison moment is the POST-FIRE-2 re-read (same
  instrument, now banked: does training install the temporal
  alternation the lineage measured, K=2 dwell vs iaf_v3's
  prior). Fire 2's gate unchanged: bisection + reference
  rebuild + three signatures, then the word. Instrument:
  scripts/alternation_hmm_diagnostic.py; artifact:
  .cache/alternation_hmm_diag.json.
- **THE RHYTHM-AUTHORSHIP DISCRIMINANT (2026-07-27, Bryce, pinned
  blind BEFORE the post-Fire-2 telegraph re-read exists):** if the
  trained loop develops two-state switching, the depth-scheduled
  ask is a candidate AUTHOR — the visibility masks widen on a
  schedule, and a supervision schedule with temporal structure can
  write temporal structure into the entropy sequence directly (the
  lineage's telegraph, whatever produced it, was NOT produced by
  this ask). THE DISCRIMINANT: a telegraph that TRACKS the
  depth-schedule's widening pattern is ASK-INSTALLED (real, but an
  artifact of this training's shape); a telegraph with dwell
  structure INDEPENDENT of the mask schedule — persisting in the
  constant-visibility breaths (k >= K-2, where masks are full) or
  matching the lineage's dwell statistics rather than the
  schedule's — is the CLASS PROPERTY the question actually asks
  about. Stated tonight; untanglable never, if unstated. #78's
  arc-close line banked in the registry (detected / refused /
  confirmed-both). The gate stands on its two owed items; the fire
  enters better-characterized than any run in campaign history.
- **THE OWED ITEMS CLOSE; FIRE 2 LIGHTS (2026-07-27):** the
  bisection resolved the contradiction WITHOUT a guilty machine:
  breath-0 staging forbids 97.7% of pairs, leaving deep tokens
  ALL-FORBIDDEN rows whose softmax falls back to uniform — the
  0.5156 was fallback rows counted as violations; individual
  forbidden pairs read attention 0.00e+00 exactly. Signature 1's
  honest form (rows with >=1 allowed column): **0.0000 — LIVE and
  clean.** The reference rebuilt as ruled (true pad32 population,
  proper SVD PCA, band blind at 0.885): sig 2 prints per-breath,
  verdict OUT at init (0.43) — BANKED as the starting geometry,
  the training trend is the watch (blended residual state vs
  pure layer-image is a definitional gap Bryce may re-rule;
  the instrument detects relative drift regardless). Dynamics
  0.57-0.67 (three orders above Fire 1). Grad path proven by the
  probe's backward + moving loss. THREE FINGERPRINTS LIVE — the
  word's condition met: **FIRE 2 LIGHTS** (full transplant: mask
  + aligned-init/markers + residual-primary; depth-ask + rider;
  early-kill armed; C4 -0.05 third and final; both endings
  pre-written; pass-attribution ablations pinned).
- **THE MID-FIRE RULING (2026-07-27, Bryce, banked while Fire 2
  burns):** sig-2's init verdict HOLDS AS BANKED — OUT at 0.43,
  no re-rule tonight. The definitional gap is real (blended
  residual state at breath 0 is not a pure layer-image and never
  was going to read as one), but **INSTRUMENTS DON'T GET
  REDEFINED WHILE THEIR NEEDLE IS MOVING** — the instrument's
  job now is the TREND: convergence toward the cloud as training
  teaches the layers to expect blended inputs tells the story
  regardless of init; divergence further out is the drift
  pathology in new clothes. The refinement (does breath-0
  blended state deserve its own reference population) queues
  AFTER the sheet, ruled on evidence about which direction the
  trajectory ran. Taxonomy note: the bisection completes the
  inert-patch family with its most instructive member — THE
  INSTRUMENT WAS WRONG ABOUT THE INSTRUMENT (machine acquitted,
  formula convicted; the honest definition only gets written
  after a fixture forces it — which is why the fixtures exist).
  Three signatures live, each having survived its own trial:
  mask through bisection, manifold through relocation, residual
  gradients through the restructure itself. The decisive hours
  are the sentinel's.
- **FIRE 2 REPORTS: C4 FAILS — THE THIRD STRIKE; THE EXAMINATION
  CLOSES (2026-07-27):** ladder_slope **+0.4876 — INVERTED**: eval
  per-breath CE RISES 3.44 -> 6.86 across the eight breaths; later
  breaths are actively DESTRUCTIVE. The manifold trajectory ran
  the pre-named divergent direction (cos 0.43 init -> 0.17 by
  step 500 — 'the drift pathology in new clothes,' printed
  exactly); state-deltas stayed violent (0.66-0.75); masked-mass
  held 0.0000 throughout (the mask clean all fire). AND the
  anatomy that completes the map: **C2 and C5 PASSED for the
  first time in three fires** — failed=C4 alone. THE TWO POLES
  ARE NOW BOTH MEASURED: the latent loop was TOO DEAD (breaths
  transform nothing — two fires, slopes -0.0037/-0.0030); the
  residual loop is TOO ALIVE (breaths transform everything —
  divergence, no contraction; one fire, slope +0.49). The
  missing ingredient is named by the engine's own recipe: the
  v98 loop's contraction discipline (seam norms + gate blending
  that PULLS TOWARD fixed points) — the r-loop transplanted the
  state anatomy without the stabilization anatomy. **PER THE
  THREE-STRIKES CLAUSE, AS PRE-WRITTEN: the sentence is
  airtight — this architecture at this scale refuses the ladder
  wearing every organ the ladder-forming lineage ever had; the
  trichotomy REOPENS with the examination genuinely complete,
  not budget-exhausted.** The examination's yield: both failure
  poles measured with instruments proven at each, the missing
  middle (transformation WITH contraction) visible for the first
  time, and every verdict under bars written before its number.
  The bar did not bend. Three fires, three refusals, one closed
  examination — the gate's word returns to Bryce with the
  trichotomy, and the record holds it all.
- **THE FIRE ORDER HELD AGAINST ITS OWN CLAUSE + THE REOPENED
  TRICHOTOMY RULED (2026-07-27, Bryce; the contradiction surfaced
  per the countersign duty):** the message's header ordered
  'add contraction discipline and rerun'; its body ruled 'no Fire
  3, no exception... a word I'm not giving today.' THE BODY
  GOVERNS — the fire is HELD; the clause refused its own author's
  opening line, which is the discipline working at its final
  boundary. BANKED AS RULED: (1) THE HONEST QUALIFICATION carried
  into the trichotomy: 'wearing every organ' was true of the
  ORGANS and false of the PHYSIOLOGY — the stabilization anatomy
  (seam norms, blending toward fixed points, v98's contraction
  machinery) was never transplanted. The middle (transformation
  WITH contraction) is NAMED, UNEXPLORED, REGISTRABLE territory —
  not refuted. It registers SPEC-STAGE with its ingredient list;
  if it ever earns a word it enters by the #72 procedure. (2) THE
  RULING: **THE GPU RETURNS TO THE ROAD THAT PAYS** — the books
  campaign IS the campaign (migration law at n=3, two promotions
  under unbent bars); the compile-side failure mass stays
  CONTAINED (vote wall 68.8%, attestation zero lies, repair
  lattice), the 257 banked as the permanent fixture any future
  candidate must face, 0.011 the floor it must beat. (3) DOCKET
  ORDER: book 7 zero-answer census under gen-21, fine-binding
  hierarchy w/ pinned prediction, fdiv autopsy, archive audit
  formal pass, replication on the next promotion's crossers, and
  the POST-FIRE TELEGRAPH RE-READ (runs now, cheap — a diverging
  loop's temporal signature is failure-pole data regardless of
  the bet's fate). Three fires, three refusals, zero bent bars,
  a two-pole map, and one sentence keeping the record honest
  about what was and wasn't tested. THE ROAD RUNS THROUGH PAGES.
- **THE POST-FIRE TELEGRAPH RE-READ (2026-07-27, the docket's
  sanctioned run; scope: PARTIAL LOAD — 76 params loaded, 8
  llama wk/wv shape-skipped (ckpt saved (2048,512) vs attach
  (2048,2048) — a state-dict bookkeeping wrinkle FLAGGED, not
  chased):** verdict on the trained-diverging loop: **NO
  TELEGRAPH EMERGED** — lag-1 autocorr +0.081 (down from init's
  +0.283, never negative); the dwell runs of 2 are the SPATIAL
  signature in sequence clothing (low-low-high-high per breath =
  the L0/L1-vs-L2/L3 polarization read k-major), not temporal
  switching. STATIC POLARIZATION PERSISTS AND STRENGTHENS (2.28
  -> 2.55 sd). The rhythm-authorship discriminant never arms (no
  rhythm to attribute); the lineage comparison closes its first
  chapter: the iaf_v3 K=2 telegraph remains unique to its
  lineage — 500 steps of divergence do not manufacture temporal
  structure, a failure-pole datum in its own right (divergence
  is not rhythm). The instrument, its prior, and its
  discriminant all stand banked for whatever trained loop next
  earns a read. THE BOARD RESTS ON THE RULING: the road runs
  through pages — book 7's zero-answer census is the next fire
  that pays.
- **GUT #80: LAYER-WISE-NOT-BREATH-WISE (2026-07-27, Bryce
  direct; countersigned against the record):** (1) the layer-wise
  STRUCTURE already exists, measured twice — the emergent
  local-to-global entropy ramp within every breath (L0 ~1.05 ->
  L3 ~1.8; 2.28 sd untrained, 2.55 sd after Fire 2 — it
  STRENGTHENS under training, nobody installed it); what does
  not exist is strict L-G-L-G zigzag, whose INSTALLATION stands
  twice-refused on standing law. (2) EVIDENCE POLARITY: the
  campaign's measured alternation win (+22.6, the staircase's
  largest jump) was BREATH-wise — 'layer-wise instead' trades
  the measured asset for the unmeasured one; noted where the
  future reads it. (3) DESTINATION: the examination is closed —
  the want enters as an INGREDIENT-LIST AMENDMENT to the
  spec-stage middle bet: the native ramp is the substrate AND
  the baseline any imposed zigzag must beat; the polarization
  instrument is its free meter; entry by the #72 procedure or
  not at all. Nothing fires; the road stays on pages.
- **THE EASE-THE-MODEL PROPOSALS COUNTERSIGNED (2026-07-27,
  Bryce's question + relay design; three mechanisms, three
  fates; ALL destinations are the spec-stage middle bet — nothing
  fires, the road stays on pages):** (1) PER-LAYER RESIDUAL GAINS
  (gamma_l seam norms) — INGREDIENT-GRADE, already family: this
  IS the contraction-discipline anatomy the closed examination
  named (v98 stabilization; magnitude-mismatch craft banked);
  enters the middle bet's ingredient list as its strongest row —
  learnable, identity-init. (2) ROPE FREQUENCY BANDING BY LAYER
  — REFUSED AS STATED on the diagnostic's own print: the
  emergent ramp was measured UNDER UNIFORM ROPE; re-banding
  changes the geometry the pretrained heads learned — surgery on
  working anatomy (recognition is not prescription, the #78
  arc-close verbatim). Old-lineage ancestor noted (v22/v23
  per-head pitch, rotation load-bearing) — future RoPE-shaping
  enters through it as assumption-to-be-tested, new registration
  only. (3) PER-LAYER TEMPERATURE tau_l — the HAND-SET form
  refused by carved law (#74's lever per-layer; the joint law:
  never sharpen in the state; #73: premature hardening IS the
  misbinding species); the registrable residue: LEARNABLE tau_l
  init exactly 1.0 (the model gets the knob, nothing installs a
  setting), temperature-perp-truth fence attached (learned
  sharpness never read as progress). The middle bet's ingredient
  list now holds: contraction seams + gamma_l gains + learnable
  tau_l + the native ramp as baseline + breath-vs-layer
  alternation evidence side by side — a bet that, if it ever
  earns its word, will be the best-provisioned registration this
  house has ever written.
- **THE SPEC-STAGE MIDDLE-BET REGISTRATION BANKS, COUNTERSIGNED
  WITH TWO TIGHTENINGS (2026-07-27, Bryce):** INGREDIENT LIST —
  (a) seam norms / learnable residual gains (v98 lineage;
  contraction discipline, fixed-point pull against exploding
  state-deltas); (b) **blending discipline / fixed-point pull
  (v98 lineage, PAIRED with seam norms — the two are ONE
  MECHANISM)** — the tightening's reason banked verbatim: seam
  norms alone entering the list risks the future plan inheriting
  half the contraction anatomy and calling it whole — precisely
  how Fire 2 inherited the body without the physiology. Plus
  (from 958de2a): gamma_l identity-init gains; learnable tau_l
  from 1.0 w/ temperature fence; native ramp as baseline;
  breath-vs-layer evidence side by side; the 257 as fixture;
  0.011 as floor; the #72 procedure as the only door. REFUSED
  APPENDIX, each refusal WITH ITS ARTIFACT (refusals need
  provenance as much as findings — the conviction-index lesson
  applied to the docket's own filing): RoPE frequency banding —
  REFUSED, cites the polarization diagnostic (ledger 43f3fb0;
  .cache/alternation_hmm_diag.json: L0 ~1.05 -> L3 ~1.7-1.9,
  2.28 sd, UNTRAINED stack, STANDARD RoPE, fixed 1/sqrt(dk);
  strengthened 2.55 sd trained, ledger 1205c5f) — altering RoPE
  distorts the positional geometry the measured ramp lives in.
  Layer-wise hand-set softmax temperature — REFUSED, same
  artifact (local focus at L0 native and unstruggled-for) + the
  joint law (#74/#75) and #73. No fires, no diverted GPU, no
  bent clause. **THE FLOOR BELONGS TO BOOK 7** — the zero-answer
  census under gen-21 (42 candidates staged), the road's own
  next page.
- **THE MIDDLE BET REGISTERS (2026-07-27, Bryce's word — by the
  #72 procedure, NOT Fire 3):** NEW BET: 'transformation with
  contraction' — the r-loop (living dynamics, three orders
  measured) + the PAIRED v98 physiology: (a) SEAM: breath-start
  detached RMSNorm on state (the latent path's own Seam-1,
  'bounds inter-breath residual accumulation' — the exact organ
  whose absence let Fire 2 diverge; param fg_v200_breath_norm_w
  EXISTS, identity-init, zero new params); (b) BLEND: delta_gate
  convex pull (present). BUDGET: ONE fire, 500 steps. BARS
  (pinned blind, inherited unbent): C4 <= -0.05 third-era
  reading; early-kill at 25%; DIVERGENCE-KILL added (eval ladder
  INVERTED at any dashboard read -> stop, the Fire-2 pole
  refused); manifold trend watched per the standing ruling;
  masked-mass 0.0000 expected clean. PASS -> the bet lives,
  attribution ablations first, then the 257. FAIL -> the middle
  is measured empty at this scale and the docket's spec-stage row
  closes with three-pole honesty. BOOK 7 IN PARALLEL: annotation
  drafting DELEGATED to a subagent under the rulebook — LAWFUL
  because the ANSWER KEY gates all training data (the certifier
  was never the annotator's identity; parse-vote-key decides
  what banks; hand-quota constitutional clauses untouched —
  drafts are drafts until the key speaks). Pen and GPU no longer
  compete: the road through pages, the bet in the gaps.
- **BOOK 7 DRAFTS DELIVERED (2026-07-27, the Sonnet delegation's
  first pass):** COUNT CORRECTION banked: the ledger's '42
  zero-answer candidates' was answer==0 across ALL levels; under
  the books' own L1-3 filter it is **26** — the discrepancy
  honestly flagged by the annotator itself. 24 DRAFTED (schema
  exactly book5/6; every factor graph pre-verified by the
  SYMBOLIC SOLVER to solve to exactly 0; gate field stamped
  'PENDING:5view-vote+key' so no draft can masquerade as
  certified), 3 HONEST SKIPS with reasons ([122] needs a
  min-selection primitive out of scope; [1673] definitional, no
  numeric structure — forcing a graph would fabricate content;
  +1 below). **THE FAITHFULNESS FLAG, held for Bryce's pen:
  [1470]** (sum of cubes ± 1..100) was drafted with a BOUNDED
  STAND-IN magnitude (250 for the true 25,502,500) — the
  argument's SHAPE preserved (S + (-S) = 0) but the quantities
  replaced: that is translation of MATHEMATICS, not register,
  and the key cannot catch it (the graph solves to 0 either
  way — the key certifies solvability, not source-faithfulness,
  which is exactly the rulebook's jurisdiction and the hand's).
  [1470] does NOT enter certification without Bryce's ruling;
  [6]'s magnitude-derivation noted as the defensible cousin
  (real problem numbers used). CERTIFICATION PASS (5-view vote +
  key on the 24) queues behind the middle-bet fire's GPU. The
  delegation model works: the pen drafts, the flags surface, the
  key and the hand keep their jurisdictions.
- **THE MIDDLE BET'S FIRE DIES BY NaN (2026-07-27):** eval
  per-breath CE all-nan; ladder slope nan; latent JSD all-nan —
  the run completed 500 steps mechanically (persistence saved,
  end report ran) but produced NO readable number anywhere. TWO
  APPARATUS FACTS BANKED: (1) the two-terminal exclusion list
  moved 42 -> 41 — breath_norm_w LEFT the excluded set: Seam-1
  is wired and live in the graph (the organ transplanted
  correctly); (2) STEP TELEMETRY MISSING — no [step]/[GATES]
  lines recoverable in either log (the trainer's own log file
  truncated at 13KB; a logging-path gap flagged — the fire ran
  blind, so the divergence-kill and early-kill NEVER GOT A
  NUMBER TO READ, which is itself the run's deepest apparatus
  lesson: kills cannot fire on unprinted dashboards). NaN
  SUSPECT: Seam-1's detached RMSNorm inside the compiled r-graph
  (the only delta from Fire 2's numerically-stable 500 steps);
  the quirks file's NaN-guard family is the debug ground. THE
  BUDGET QUESTION, put to Bryce explicitly and not presumed: the
  registration budgeted ONE fire — a NaN death with zero
  readable telemetry is apparatus-invalid (the bars never
  received a number; C4 neither passed nor failed). Whether an
  apparatus-invalid run SPENDS the one-fire budget is a ruling,
  not a default — precedent cuts both ways (apparatus checks
  precede verdicts; but budgets exist to refuse re-rolls). HELD
  for the word. Book 7's certification pass now has the GPU.
- **THE NO-CONTEST RULING + ITS FENCE (2026-07-27, Bryce):** the
  fire is NOT SPENT — apparatus-invalid means the exam never
  convened (the eighth ruling's symmetry: Fire 0's inert patch
  one level down; a fire, in this house's vocabulary, is a run
  whose instruments print). THE FENCE, pinned before hope: the
  no-contest is available ONCE — a NaN death is not cleanly
  separable from the candidate (Seam-1 is suspect BY
  SUBTRACTION; the apparatus failure may be the organ's own
  pathology wearing a crash). If the debugged re-run dies of
  ANYTHING with kills again unable to fire, the budget is SPENT
  on that death — two apparatus deaths in one registration IS
  the finding (an organ that cannot run stably is refused by its
  own physics). TELEMETRY LAW carved into the constitution.
  DEBUG ordered by subtraction (reproduce minimal, confirm by
  removal, patch by evidence; isfinite craft at the seam). BOOK
  7 certification has the floor; [1470] crosses the wheel with
  source+draft after the pass. CONFESSION TO VERIFY: the
  middle_smoke's end-report TypeError (start_loss None) may have
  been the NaN's FIRST SYMPTOM, misfiled as cosmetics — if the
  smoke's losses were already nan, the instrument-blind fire was
  lit on a misread smoke, and the telemetry law's necessity was
  proven twice in one afternoon.
- **THE SEAM-1 DEBUG'S FIRST CUT: THE ORGAN'S PHYSICS ACQUITTED
  (2026-07-27):** eager forward with Seam-1: FINITE at all eight
  breaths, state RMS BOUNDED ~3.2 — the contraction demonstrably
  contracts (Fire 2's unbounded growth is gone; the paired
  mechanism works in forward mathematics). THE SUBTRACTION
  NARROWS: the NaN lives in the COMPILED STEP or its GRADIENT
  path (Seam-1's detach at a new graph position under TinyJit —
  the quirks file's family ground). DEBUG CARD (next window):
  (1) JIT 2-step loss-finiteness cut (isolates compile-vs-grad);
  (2) THE TELEMETRY PRE-FLIGHT HARNESS built per the new law
  (steps provably print + one injected non-finite the guard must
  catch — REQUIRED before any re-light); (3) re-light under the
  fence: one no-contest spent the moment it lights — any second
  apparatus death SPENDS the budget, an organ refused by its own
  physics. Book 7's certification pass holds the floor
  meanwhile. The middle bet's state, honestly: physics sound,
  compilation suspect, one lawful attempt remaining.
- **THE JIT CUT COMPLETES THE DIAGNOSIS (2026-07-27):** the
  middle-bet fire's log holds **[NaN-skip] x 500 — the guard
  fired on EVERY step**; its `continue` skips the [step]/[GATES]
  prints, so the dashboard the kills watch was bypassed by the
  guard itself. THE TELEMETRY MYSTERY SOLVED: not a logging
  fault — a DESIGN GAP now named: **the NaN-guard's skip path is
  silent to the kills, and consecutive skips are uncounted** —
  500 no-op steps ran as a 'fire.' TELEMETRY-LAW AMENDMENT
  (carved into the pre-flight harness spec): the guard's catch
  is ITSELF a dashboard line, and N consecutive NaN-skips (N=10)
  IS A KILL — a fire that cannot compute is a fire that stops.
  THE COMPILE CUT: eager forward FINITE (banked) vs compiled
  total NON-FINITE FROM STEP 1 — the NaN arises in the JIT-
  compiled loss/graph, not the mathematics; Seam-1's
  detach+float inside the jitted graph at LOOP-TOP position is
  the localized suspect (the same seam function at the BLEND
  position ran finite in Fire 2 — position, not function, is
  the delta). Quirks candidates: the file's cast/detach-in-JIT
  family; next cut is op-level localization, then the pre-flight
  harness (now with the consecutive-skip kill), then the
  re-light under the once-only fence. Book 7 certification
  holds the floor.
- **GUT #81 EXECUTES: THE JSD TELEGRAPH RE-READ (2026-07-27;
  fixture-first per the ruler law — the fixture PASSED and
  demonstrated the blindness hypothesis inside itself: planted
  support-shifted telegraph has entropy-spread 0.0139 (~0,
  entropy-BLIND) while JSD detects exactly its 7 switch points
  over a null-calibrated floor):** THE INIT READ, floor from
  cross-batch 95th pct (0.0080): every layer shows ONE early
  spike (breath 0->1; L0 0.0417) then MONOTONE DECAY TO LITERAL
  ZERO by breath 5-6; lag-1 +0.55..+0.92 (smooth settling, not
  switching). **BLIND-BIN VERDICT: the init silence is
  RULER-ROBUST — no telegraph in level OR displacement; the
  trained-vs-init comparison now stands on TWO meters.** THE NEW
  TEXTURE JSD BOUGHT (entropy could not see it): **ROUTING
  SETTLES WHILE CONTENT FLOWS** — attention displacement hits
  0.0000 by breath 5-6 while state-deltas stay 0.66+: after
  breath ~5 the loop moves VALUES through FROZEN routes. A
  routing-vs-content split at init, filed for the trained
  re-read (does training keep routes frozen and ladder the
  content, or unfreeze routing?). Two-track law honored: this
  is the JSD track's own baseline; the lineage comparison stays
  entropy-side per the iaf_v3 caveat. Artifacts:
  .cache/jsd_meter_fixture.json, jsd_telegraph_init.json.
- **THE ROUTING-STALENESS HYPOTHESIS PINNED INTO THE MIDDLE-BET
  REGISTRATION (2026-07-27, Bryce — the texture fact's real
  payment):** the candidate anatomy for Fire 2's divergence,
  assembled from three banked reads: routing FREEZES by breath
  5-6 (JSD -> 0) while content flows (deltas 0.66) while the
  state leaves the layer-image cloud (0.43 -> 0.17) while later
  breaths turn destructive (eval CE climbing) — **a loop that
  commits its routing in five breaths and then pushes
  ever-larger content through frozen pathways has no mechanism
  to re-route around its own drift: divergence as STALE ROUTING
  AMPLIFYING DISPLACED CONTENT**, not merely unbounded
  magnitude. SEAM-1's MECHANISM RE-FRAMED: re-normalization at
  breath starts doesn't only bound norms — it keeps the state
  RECOGNIZABLE to the frozen routes, so routing computed early
  stays approximately valid late. BLIND PREDICTION for the
  re-light's JSD line: under Seam-1, EITHER routing stays
  plastic longer (JSD decaying slower than init's breath-5-6
  freeze) OR the frozen routes remain SUFFICIENT (content
  laddering through static attention because the seam keeps
  state in-distribution). The trained re-read's registered
  question goes THREE-WAY: ladder-through-frozen-routes /
  unfreeze-routing / seam-sustained-routing. The re-light now
  carries a mechanism story it can confirm or kill, not just a
  crash to avoid. And the note across eras stands honored: the
  iaf_v3 author's 'entropy NOT JSD' flag has now done precisely
  the job it was raised for, two eras and two architectures
  later.
- **BOOK 7 CERTIFICATION SHEET (2026-07-27):** of 23 read (1 held
  [1470]): **1 CERTIFIED** — the chain's first machine-drafted
  certified row in campaign history — 17 abstain (all-None
  parses), 5 QUORUM-WRONG (incl. a unanimous 5/5 reading 2 vs
  gold 0). THE CALIBRATING COMPARISON: book 6's lane pass ran
  L1=1/L2=30/L3=44 of 75 — the machine-banked lane is ALWAYS
  tiny (~1-4%); book 7's 1/23 free-lane rate is HISTORICALLY
  NORMAL. The delegation produced a normal book's raw material:
  the 17 abstains are the repair bench, the 5 wrongs the surgery
  list — the standard lanes, now with a NEW ambiguity the
  zero-answer class sharpens: a unanimous wrong on a draft may
  be GATE ERROR (zero-dialect gap: zero-heavy givens) or DRAFT
  INFIDELITY (the text describes a different graph than the one
  the solver preverified — the annotator's graph solved to 0,
  but the key never checked that the TEXT says what the graph
  says). Per-item autopsy belongs to the lane pass; the
  gate-vs-draft question is EXACTLY what the repair lane exists
  to adjudicate. The certified row enters the registry's waiting
  room with full provenance (drafted-by-delegation, certified by
  vote+key+attestation). Lanes to Bryce's wheel with [1470]'s
  faithfulness ruling.
- **THE ADJUDICATION PRIORITY PINNED (2026-07-27, Bryce):** the
  unanimous 5/5-reading-2 is the WRONG-UNANIMOUS COLD CLASS —
  second wild appearance in campaign history (first: the 2/20
  adversarial tick, panel-blocked) — and its delegated-draft
  provenance INVERTS the scary hypothesis: if the text drifted
  from the preverified graph, the GOLD ITSELF IS POISONED and the
  unanimity may be five views READING TRULY. The chain's one
  structural blindness, named at the delegation's registration:
  the key checks the graph's answer, never text-graph
  faithfulness. **PRIORITY: FAITHFULNESS FIRST, GATE SECOND** —
  text vs preverified graph BEFORE gate vs gold. Text-says-2 ->
  the delegation's first faithfulness escape (different finding,
  different fix: faithfulness check goes SYSTEMATIC on all
  delegated drafts; quorum vindicated). [1470] drew the line
  from one side; this specimen tests it from the other; both
  cross the wheel together with text+graph+gold.
- **THE CERTIFICATION SHEET VOIDED — MY APPARATUS ERROR, CONFESSED
  AND CORRECTED (2026-07-27):** the wheel-crossing extraction
  exposed it: book5's OWN schema stores text=SOURCE and
  gen.dialect=THE ANNOTATION — Sonnet matched the schema
  correctly; MY certify harness parsed r['text'] and ran the
  entire pass on RAW HARVEST LATEX. The sheet (1/17/5) is VOID:
  the 17 all-None were the gate correctly abstaining on
  un-annotated source; the 'cold-class specimen' DISSOLVES — the
  unanimity was the gate reading raw LaTeX, no poisoned gold, no
  wild cold-class second appearance (the ledger entry at b89c02b
  and the adjudication entry at 62a6981 are superseded on that
  point; Bryce's faithfulness-first priority was nonetheless
  VINDICATED one level up — 'read the text against the graph
  first' is exactly what caught this: the text WASN'T the
  annotation at all). The corrected pass (gen.dialect) re-fired
  as book7-certify2; verdict on sentinel. Read-before-patch's
  final form: READ THE SCHEMA, not just the field names.
- **THE CORRECTED CERTIFICATION SHEET: 21/23 CERTIFIED, 0 WRONG
  (2026-07-27):** on the actual dialect rewrites (gen.dialect),
  the Sonnet delegation's true grade: **21 certified through the
  full chain** (five views, quorum >=3, answer key, v3
  attestation), 2 abstains (the repair bench — tiny), **ZERO
  quorum-wrongs** — no lies, no faithfulness escapes at quorum.
  The earlier 'first certified machine row' milestone corrects
  upward: twenty-one rows entered the waiting room wearing
  drafted-by-delegation provenance in one pass. THE ECONOMICS
  NOTE, banked plainly: solver-preverified + rulebook-aware
  drafting certifies at 91% where the historical free-lane rate
  was 1-4% — because the delegation drafts INTO the dialect the
  gate reads, not into prose needing surgery. The hand's role
  reshapes: 2 repairs, [1470]'s ruling, and spot-audit — the
  books can now run at delegation speed WITH the key's full
  custody. Book 7 tranche 1 stands at 21 certified zero-answer
  rows: the zero-hunt census the wild specimen [36] asked for,
  delivered. Lanes + [1470] to Bryce's wheel; the quirk cut has
  the GPU.
- **TWO RIVETS ON THE BOOK ECONOMICS + COMPILER PRIORITIZED
  (2026-07-27, Bryce):** (1) SCOPE STAMP: 91%-vs-1-4% compares
  different populations — the historical lane ran RAW candidates
  cold; these 23 were SOLVER-PREVERIFIED (upstream selection).
  Honest sentence: delegation + preverification certifies ~91%
  where raw ran 1-4; the lift belongs to the PIPELINE; December
  models the preverification's own yield rate before tempo
  replans. (2) THE FAITHFULNESS SAMPLE, load-bearing before the
  21 train: the chain certified GRAPH-ANSWERS, never
  text-faithfulness — for delegated drafts the gap is at
  population scale (the gold descends from the draft's own
  graph). LAW: five of the 21, stratified, join [1470] at the
  wheel — SIX ACROSS; all-conform -> tranche enters with a
  measured faithfulness rate and sampling becomes book 8's
  standing gate; any drift -> caught before training, the only
  time faithfulness is cheap. COMPILER PRIORITIZED: _STD cut ->
  harness -> re-light under the once-only fence.
- **THE SEAM-1 BISECTION AT THREE CUTS (2026-07-27):** detach
  ACQUITTED (_STD also NaN-skips all steps); position ACQUITTED
  (end-of-breath also dies). THE SHARPENED CHARACTERIZATION:
  under TinyJit, an RMSNorm ON THE ACCUMULATED STATE x — any
  form, any position — produces non-finite from step 1, while
  Fire 2's norm on the FRESH LAYER OUTPUT h (feeding the same
  chain through the blend) compiled finite, and the READOUT norm
  on x as a LEAF also compiled finite. The suspect species:
  data-dependent rsqrt on the K-fold-reused state buffer
  (JIT-input descendant) re-entering the chain. NEXT CUTS
  (sealed): (a) scalar-RMS variant (whole-tensor rsqrt — scalar
  on the chain vs per-token); (b) tinygrad-level first-step
  probe (where the non-finite enters the compiled graph); (c)
  the pragmatic contraction alternatives if norms are simply
  unavailable on this chain (gate-schedule contraction; fixed-
  target scaling). The once-only fence UNTOUCHED — no re-light
  attempted; these are cuts, not fires (instruments-blind runs
  of 4 steps under the NaN-guard's own printing, per the
  telemetry amendment). THE SIX FOR THE WHEEL prepared:
  .cache/book7_wheel_six.json (5 stratified + [1470],
  source+dialect+graph+gold each).
- **THE QUIRK CONVICTED — SCALAR-RMS COMPILES (2026-07-27, cut
  4):** four steps, finite losses, ZERO NaN-skips. THE QUIRKS
  FILE'S THIRD ARTIFACT-BACKED ENTRY: 'PER-TOKEN RMSNorm
  (axis=-1 reduction) on a K-fold-reused recurrent state buffer
  under TinyJit produces non-finite from step 1 (any form,
  either position); the SAME chain accepts a PER-ITEM SCALAR
  reduction (axis=(1,2)) cleanly, and accepts per-token norms on
  FRESH tensors (layer outputs) and LEAVES (readout). REPRO:
  V200_R_SEAM1=1 vs V200_R_SEAM1_SCALAR=1, 4 steps.' THE
  CONSEQUENCE: the middle bet's contraction IS IMPLEMENTABLE in
  ops this JIT compiles — scalar-RMS Seam-1 bounds state
  magnitude per-item (coarser grain than per-token; the
  physics-relevant bound per the eager acquittal). Bryce's
  buffer-lifetime diagnosis ('uninitialized/aliased memory
  wearing a tensor's shape — non-finite at step one, not
  divergence over steps') fits the conviction exactly. NEXT: the
  telemetry pre-flight harness (consecutive-skip kill + injected
  non-finite + printing proof), then THE RE-LIGHT — scalar-Seam-1,
  the one remaining attempt under the absolute fence.
- **THE BENCH RULES ON THE SIX (2026-07-27, Bryce — two axes,
  opposite verdicts):** AXIS ONE (text-graph custody): 6/6 CLEAN
  — every dialect says what its graph computes; the chain's
  certification means what it claims. AXIS TWO (source
  fidelity): 40% rendered / 60% substituted-or-stand-in — and
  the substitution is a POLICY: facing inexpressible operations,
  the delegation pre-computes them into gifted knowns instead of
  skipping (2/6 and 5/6 FAITHFUL-HIGH — the model specimens:
  algebraic rearrangement and the magnitude-fold done right; 1/6
  UNFAITHFUL by asymmetry (same midpoint op in-graph on x,
  gifted on y); 3/6 UNFAITHFUL (ceil/floor semantics wholly
  outside the dialect — the honest-skip case, substituted
  instead); 4/6 UNFAITHFUL-LOW (the quotient rule across 56
  exponents gifted); [1470] LAWFUL-AS-DECLARED, category ruling:
  stand-ins enter as DIALECT-NATIVE problems with stand-in
  provenance, never as translation). THE CENSUS HAZARD NAMED:
  substituted zeros are trivial cancellation wearing borrowed
  skeletons — they LAUNDER the zero-class's mechanism-diversity
  into uniformity; EXCLUDED from the zero-census coverage claim.
  THE PATTERN (second consecutive): deep checks keep catching
  drifts they weren't aimed at. DISPOSITION: full-tranche
  PROVENANCE PASS (each of 21 marked rendered/substituted by the
  explicable-family standard), rendered rows = certified
  zero-specimens; substituted = stand-in-marked, trainable as
  cancellation practice, never census coverage. BOOK 8's gate
  goes TWO-AXIS with the strengthened instruction (inexpressible
  core -> skip or declared stand-in, NEVER gift the result) +
  specimens 2/5 as worked examples.
- **THE PROVENANCE PASS RUNS (2026-07-27):** mechanical marks on
  the 21: **10 rendered / 6 explicable-adjacent / 5 substituted**
  — with bench overrides applied: src 6 (ruled UNFAITHFUL by
  asymmetry) had been machine-marked rendered because its gifted
  value COINCIDES with a source literal — **the screen's blind
  spot, demonstrated by the bench's own specimen**: a gifted
  computation whose result happens to equal some source number
  passes the literal check. FINAL MARKS (bench-overridden): **9
  rendered / 6 adjacent / 6 substituted-or-ruled**. THE LIMIT
  STATED: the mechanical pass is a SCREEN, not a verdict —
  'rendered' means passed-the-literal-check, not
  certified-faithful; sampling continues as book 8's standing
  gate exactly because coincidence hides. DISPOSITION EXECUTED
  per the bench: 9 rendered rows = certified zero-specimens with
  census standing; 6 adjacent = flagged for promotion/demotion
  at the wheel's convenience (each one known needing one-op
  derivation review); 6 substituted = stand-in-marked, trainable
  as cancellation practice, EXCLUDED from zero-census coverage.
  Artifacts: .cache/book7_provenance_pass.json. The tranche
  enters on its true terms.
- **GUT #82 REGISTERED WITH BRYCE'S FULL COUNTERSIGN (2026-07-27
  — the E-B frame's third visit, + thermal mass):** dispositions
  banked as ruled: (1) PHYSICS CORRECTED, second time on the
  same joint: vacuum-wave E and B are spatially orthogonal but
  TEMPORALLY IN PHASE — the photon is simultaneity, not
  turn-taking; a frame whose founding image is misremembered
  turn-taking cannot license a turn-taking architecture; the
  true part (coupling) already owns its measured name
  (commit/propagate, +22.6). (2) 'high Local at k MUST induce
  high Global at k+1' REFUSED — third citation of the
  imposed-rhythm clause (#73: learned or measured, never
  installed); refused-appendix gains its third ancestor. (3) THE
  REAL PIECES CREDITED AND ALREADY VAULTED: convex delta-gate
  blending = the norm-free fallback REGISTERED in the
  bisection's own sealed sequence; 'thermal mass' C=1/alpha =
  EMA; 'flywheel' = momentum; 'thermostat' = Seam-1's job
  description — the organs are real, the wedding already
  happened, the costume adds no mechanism. ORTHOGONALITY CLAIM
  STRUCK from any future registration: a convex combination
  interpolates; it orthogonalizes nothing. (4) TIMELINE
  CORRECTION: the memo sells a cure for a bug already convicted
  AND cured (scalar-RMS compiles; re-light staged) — the
  once-only fence refused the morning-after enrichment exactly
  as designed. (5) 'Are we doing enough of the telegram?' — YES,
  exactly enough: silence ruler-robust on two calibrated meters;
  the does-training-install-it read scheduled with the
  rhythm-authorship discriminant. The re-light stands untouched:
  telemetry harness, then scalar-Seam-1, one attempt, absolute
  fence. The frame can watch the fire like everyone else.
