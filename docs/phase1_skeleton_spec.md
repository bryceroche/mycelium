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
