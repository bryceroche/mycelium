# A Self-Certifying Reasoning Pipeline at 90M Trained Parameters
### (working title — alternates below)

**Authors:** Bryce Roche, Claude (Anthropic)
**Status:** SKELETON v2 (2026-07-08) — supersedes `paper/outline.md` (2026-06-20), which
predates the Phase-1 parser, the repair stack, the certification lattice, and the
registry expansion. The old outline's deducer content survives as §4 and its honest-
negatives discipline survives as the paper's whole voice.

**Title alternates:**
- "Certify, Answer, Flag, Abstain: A Gold-Free Deployment Lattice for Small Reasoning Models"
- "The Dancer Reads Twice: Measured Reasoning at 90M Parameters" (informal/blog)
- "One Engine, Any Factor Graph — and a Pipeline That Knows When It's Wrong"

---

## 0. Framing decision (locked per design discussion)

**Paper B wrapped in Paper C, with Paper A as the vehicle.**
- **Headline artifact (B):** the zero-parameter deployment lattice — certify (0.998) /
  answer / flag / abstain — gold-free, portfolio-combined, frozen as an interface and
  re-validated under domain expansion.
- **Credibility engine (C):** the registered-prediction method — kill bars before
  builds, jurisdictions on claims, ten+ refutations each converted to a law, an
  instrument, or a retired build. Evidence: the git log + spec ledger itself.
- **Vehicle (A):** the two-phase architecture (parser / deducer / symbolic search) and
  the repair stack, presented as the demonstration, not the claim.
- **The one-sentence thesis:** *A 90M-trained-parameter pipeline can read, solve,
  repair, and — most importantly — certify its own answers gold-free, and the
  registered-refutation method that built it is faster than folklore, not slower.*

---

## 1. Abstract (claims checklist — every number has a commit)

- End-to-end **71.5% / 0.833 answered-precision** (composed: one-shot → withhold →
  selective retransmission → TTA vote) on adversarially-hardened algebra word problems
  requiring genuine search decisions; **68.2→71.5%** arc documented (1023/1500 corrected forced-answer floor, grade_equivalence.py).
- **Certification tier:** 5/5 TTA unanimity answers at **0.9982 precision @ 38%
  coverage** (original domain), **1.0000 @ 51.4%** (expanded domain; zero-numerator
  discipline: "bounded near a quarter percent"), dial down through 4/5 and 3/5.
- **Gold-free self-diagnosis:** 0.92–0.95 of wrong parses symbolically detectable;
  abstention portfolio = TTA agreement (AUC **0.840**) + waist-centroid monitor (AUC
  **0.728**) + entropy tier-0 (min-comb **0.812** on silents), rank-sum at the tail.
- **Trained footprint:** ~3.2M parse head + ~3.2M repair head over a frozen Llama-3.2-1B
  L0–L3 slice, ~87M deducer, **zero-parameter** monitor/certifier/confidence stack.
- **Generality, twice:** deducer solves coloring / SAT / KenKen / circuits with zero
  core edits (prior result); parser-side control run shows registry expansion
  **improved the original domain** (+68 answers generality, +18 plumbing, separated by
  pre-registered control).
- **Method:** N registered predictions (count from ledger; ≥10 refuted), 6+ minted
  laws, every mechanism's contribution separately measured.

---

## 2. Introduction

- Hook: big models ship a number and a shrug; this pipeline ships a number, a
  certificate, a flag, or an abstention — and knows which, gold-free.
- The accountability-per-parameter axis. Weight class ≠ parameter count.
- The method is a contribution, not a hygiene footnote: preview the refutation ledger.
- Honest scoping UP FRONT (converts the expected review into an exhibit): generated
  corpus with measured teeth; grading-policy deltas measured (equivalence class ~17%,
  luck-inflation 0.6%); external anchor experiment in §8 (TO BE RUN — the one build
  this paper still owes).

## 3. The two-channel architecture (Paper A, compressed)

- Factor graphs; topology vs semantics channels (C2's death as the founding negative).
- Phase 1 parser: frozen trunk + delta head; union-typed slots; mention spans as
  structure; the pointer law (5 sightings) as a DESIGN RULE — every pointer born
  candidate-restricted + span-supervised.
- Phase 2: the validated deducer (from old outline §§1–2: K=16 shared-weight breaths,
  masks from membership, ladder CE, K_min ≈ D/4 breadth-parallelism, honest negatives
  intact) + the symbolic search tier (predicate + bridge; DSATUR=MRV, AC-3=GAC).
- The interface as-built vs as-designed (one honest paragraph): registry thriving;
  membership-in-place-of-ball (flagged); NACK signal stack where the notebook was
  drawn. "The nouns died; the verbs survived."
- **Fig 3.1** pipeline diagram (exists in spec ASCII; redraw).
- **Table 3.1** factorization result: fac-exact flat across decision-bands; partial
  correlation +0.061 size-controlled → *the solver settles because constraints
  interact; the reader deepens because text doesn't* (depth-head ablation, §6).

## 4. Corpus discipline (the evaluation-honesty section)

- Solution-first generation; uniqueness + round-trip as GENERATION GATES (360/360 …
  4,140/4,140); band labels stamped at birth (decisions-per-problem via search tier);
  teeth knobs (paraphrase, obliques, split-refs, distractors, letter-shuffle,
  irrelevant subsystems) with measured bite.
- Gold format: span-sets per factor (never single spans), mention spans, exact
  digit-spaced targets (the symbolic jaw must chew), query pointer.
- Grading policies measured against each other: strict-graph vs forced-answer vs
  answer-match; equivalence class ~17% (two draws, 17.2/16.6); luck inflation 0.6%.
- **Table 4.1** teeth ablation / detectability by domain (KenKen 100% × 7 points;
  algebra 0.92–0.95; the detectability-density law + integrality-jaw expiration).

## 5. The repair stack (measured joint by joint)

- NACK tiers as a PORTFOLIO (dense rankers blend; rare-precise flags veto): tier-0
  entropy (0.812), symbolic verifier + uniqueness probe, belief-JSD (0.687), library
  cross-check (rare flag, 3.6× enrichment), TTA agreement (0.840).
- Withhold-and-solve: Law-7-at-the-factor-level; 26% free recovery, zero silent-wrong
  at any k; domain-general (sparse-flip refutation + decomposition clause).
- Selective retransmission: flag-dependent objective; two-checkpoint architecture
  (parse specialist / repair specialist, one frozen trunk); field-level flags BEAT
  gold text-localization (structural-entry law's cleanest demo; leakage bound zero).
- Multi-round: 19.6→7.7→1.1→0% decay; hard-partition shape; composed 47% (KenKen) /
  32% (algebra at convergence); the curriculum law (fresh failure mining — the
  self-defeating-curriculum fix).
- **Fig 5.1** per-round decay; **Table 5.1** composed stack by stage × domain.

## 6. The survivor arc & the three jurisdictions (the paper's narrative spine)

- Nine refutations in sequence (teeth-uniform → multiplicity → omission → suspicion →
  binding → routing wall named: states 99.6% decodable, pointer mis-aimed → oracle
  ceiling 13.9% → ratchet leak → beacon 3%): a wall reduced from mystery to priced
  population. **Fig 6.1** the refutation cascade (one figure, nine tombstones).
- The verdict: detect-and-abstain, as MEASUREMENT not surrender.
- **The three jurisdictions** (closing image → discussion thesis):
  - *Prevention* owns confident wrongness: registry expansion cut invisible-wrong 4×
    per capita; nine decode-side mechanisms got single digits. (Prevention law,
    2 sightings, third due at tranche 2.)
  - *Depth* owns the parser's residual (+2 both domains) — Brick-P's four shrinking
    formulations ending in the no-story control; parser has no joint structure a
    single pass can't see.
  - *Detection* owns the remainder: the lattice.
- TTA as the third category: nine mechanisms fixed the estimator; TTA randomized the
  input and averaged. MC-π framing; independence-competence tradeoff law (oracle views
  decorrelate best, score worst).

## 7. The deployment lattice (Paper B — the headline section)

> *Epigraph: "Temperature is orthogonal to truth."* — vote entropy across
> TTA views reads basin DEPTH (deep-correct H=0.000, shallow-correct
> H=0.846, deep-wrong H=0.212, measured 2026-07-11); the mouth reads
> landscape familiarity; the answer key reads truth. No instrument
> substitutes for another — that is the chain-of-custody argument in one
> sentence, and the reason the lattice needs all its links.

- Four rungs: certify / answer / flag / abstain — all zero-parameter, all gold-free.
- (Candidate fifth column, pilot-validated: CORRECT-BUT-SHALLOW — right
  answer, vote-shy; the self-identified retraining-target class.)
- The freeze as an interface contract; expansion acceptance = "the lattice held its
  dials" (Table: tranche-1 acceptance, both rows green and upward).
- Certification channel: unanimity dial (5/5 → 3/5 curve), cross-domain survival,
  zero-numerator confidence intervals stated properly.
- Portfolio combination: correlation 0.464, combo loses AUC but wins every operating
  point abstention uses (metric-decision-structure law, 4th sighting — inside our own
  registration).
- **Fig 7.1** precision-coverage frontier with all dials; **Fig 7.2** portfolio tail.

## 8. External anchor (TO BE RUN — the paper's one open build)

- MATH-500 (or AMC) slice through the pipeline; registry tranche 2 per band-sweep
  (quadratic+modular shipped; sweep decides next relations; inequalities architecture
  question priced here).
- Report WITH the lattice: answered-precision + certified coverage at whatever raw
  accuracy lands — the claim is the lattice's honesty on foreign text, not the
  accuracy number.
- Grading-policy delta measured on the anchor (answer-match vs forced-answer).

## 9. The method (Paper C — could be its own section or woven throughout; DECIDE)

- Registered predictions with density-regimes stated; kill bars before builds; cheap
  disconfirmation (scripts killed two builds: ledger re-parse, deeper prefix).
- The laws, as a table with sighting counts: pointer law (structure or supervision,
  never conditioning); density-regime arithmetic; jurisdictions (causes, selections,
  bars/counters); mass bars; no-silent-fallbacks; structural entry; co-adaptation
  (trained structure is relational); independence-competence; audit-that-confirms;
  zero-numerator discipline.
- Case study sidebar: Brick-P's four formulations in 48h (the story growing SMALLER).
- The two-instance workflow (design channel / build channel, registered handoffs) —
  one paragraph, honest about what it is.

## 10. Related work

- Anthropic global-workspace paper: convergent motifs (small broadcast channel;
  internal-state-more-honest-than-output = our tier-3/monitor bet); the J-lens
  borrowed → structured refutation within 24h (co-adaptation defeats post-hoc
  dimension selection — a measured boundary condition of their method); readout-vs-
  storage rhyme with the routing wall.
- Speculative decoding (draft/verify = propose/dispose; confidence-gated drafts =
  tier-0; suffix decay = compounding parse error).
- Neural collapse (simplex ETF; codebook geometry predicts confusions ρ=0.53).
- TTA & progressive resizing (fastai lineage, stated honestly); selective-repeat ARQ
  (Brick-R, registered, unbuilt); NL-to-formal parsing + verifier pipelines;
  calibration/abstention literature (selective prediction) — the lattice's actual
  competitors; deep supervision (the depth head's true family).

## 11. Honest limitations (write this section FIRST)

- Generated corpora; template ancestry despite teeth (external anchor mitigates, not
  removes). Integer domains; integrality-jaw expiration registered. Inequalities =
  open architecture question. The 396-descendant population: detect-and-abstain only
  (beacon tremor 11/14 recorded, not pursued). Six-cycle Alternator, ball, atlas:
  designed, gated, unbuilt — say so plainly. Equilibrium claim: 2 favorable reads,
  test scheduled, not proven.

## 12. Figures & tables already paid for (inventory)

1. Pipeline diagram (spec ASCII → redraw)
2. Refutation cascade (§6, nine tombstones)
3. Per-round recovery decay (multi-round table)
4. Precision-coverage frontier, all four dials
5. Parse-side silhouette render (the BirdNET picture — token × kind-similarity field)
6. Codebook heptagon + confusion overlay (ρ=0.53)
7. Detectability-vs-exactness curve (7 KenKen points + algebra)
8. Withhold curve with solve-to-GOLD column
9. Factorization: fac-exact flat across bands (+ size-controlled partial)
10. Tranche acceptance table (lattice dials, before/after expansion)
11. Laws table with sighting counts
12. TTA arms: independence-competence tradeoff (oracle vs deployable)

## 13. Release checklist

- Code: repo public (already), tag the paper commit; centroid library artifact;
  eval scripts with pinned seeds; the spec ledger (`docs/phase1_skeleton_spec.md`)
  shipped as supplementary material — it IS the registered-prediction record.
- Reproducibility statement: single 7900 XTX, tinygrad, no CUDA — a selling point.

---

## Open decisions (for Bryce + relay before drafting prose)

1. §9 as standalone section vs woven method-thread? (Lean: standalone, short, with
   the ledger as supplementary — reviewers can audit rather than trust.)
2. Venue: ML conference (NeurIPS/ICLR track) vs systems/deployment venue vs long-form
   (arXiv + blog at theshapeofthought.ai)? The lattice framing fits calibration
   workshops; the method framing fits nothing standard — which may be the point.
3. Title. (Current lean: the certify/answer/flag/abstain form — the lattice IS the brand.)
4. External anchor scope: how many MATH-500 problems, which categories, and does
   tranche 2 gate the paper or ship as "expansion continues"?
