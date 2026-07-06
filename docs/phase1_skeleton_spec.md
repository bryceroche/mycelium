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
