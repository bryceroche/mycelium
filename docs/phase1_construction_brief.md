# Phase 1 — Construction (NL → factor graph): the next chapter

**Status:** kickoff brief (spec-stage). **North star:** MATH-500, Dec 2026.
**Context:** `docs/session_2026_06_26_solving_closed_phase1_pivot.md` (why we're here).

## The frame
Two jaws (the project's two-phase architecture):
- **Jaw 2 — SOLVING (factor graph → answer): DONE + VALIDATED.** Symbolic search tier solves
  clean hard-constraint graphs exactly + fast; the breathing deducer is the general/amortized/
  *differentiable* backend for the soft/uncertain ones. We spent a session proving this jaw needs
  no more (neural-guided clean-CSP search is closed — two death modes mapped).
- **Jaw 1 — CONSTRUCTION (NL → factor graph): NOT BUILT — the work.** This is where "symbolic
  isn't enough" is *genuinely* true: parsing language into the right variables + factors is not a
  solved symbolic problem. The unrefuted frontier.

## Who does what
- **The constructor (the bridge):** a **comprehension-capable model** (LLM-grade) — *not* the
  bare breathing transformer, which hit a reading-comprehension wall on GSM8K (the reason v98
  pivoted to Sudoku). Plausibly augmented with the **cathedral structure** rehomed from solving:
  a **notebook** to accumulate the graph while reading, a **waist** to compress language→latent,
  **coarse-to-fine** to scaffold-then-refine. *(Rehoming is a hypothesis — the notebook was
  refuted for solving; re-testing it for incremental construction is a fresh experiment.)*
- **The breathing transformer (deducer):** the construction partner, NOT the parser. Three jobs:
  1. **Differentiable critic** — solve the proposed graph → training signal whose gradient flows
     back into the constructor (can't backprop through symbolic).
  2. **Format-definer** — its membership + inlet vocabulary *is* the constructor's output target.
  3. **Soft-graph backend** — solves the *uncertain* graphs a probabilistic parse produces.

## THE first design question (decides everything)
**Is the NL→answer signal trained END-TO-END or STAGE-WISE?**
- **End-to-end** (NL → graph → answer, one differentiable pipeline): the differentiable deducer
  is *essential* (gradients can't pass through symbolic) → breathing transformer is the centerpiece.
- **Stage-wise** (parse → harden the graph → symbolic solve): needs **graph-level supervision**
  (gold factor graphs), and symbolic suffices for solving → the deducer is a specialized backend
  for the soft/uncertain cases only.

Answering this first is the gate; it determines the whole architecture.

## Open design questions (the next chapter's agenda)
1. **The target schema** — what *is* a factor graph for a MATH-500 problem? (variables =
   quantities/unknowns; factors = equations/relations/constraints; membership = which quantities
   each relation ties.) Needs a concrete, general schema the constructor emits + the engine consumes.
2. **Training graphs / supervision** — where do gold/weak graphs come from? (annotate? derive from
   solutions? weak supervision from the final answer via the differentiable critic?)
3. **Uncertain-graph handling** — soft membership/factors from the probabilistic parse → exactly
   the deducer's soft-backend role; how to represent + propagate parse uncertainty.
4. **Cathedral-rehoming experiments** — does a notebook genuinely help incremental construction?
   (gated, fresh test — refuted for solving ≠ refuted for construction.)

## Cheap-first probes (keep the discipline: kill-gates before big builds)
- **Feasibility probe:** can a comprehension model parse a *simple* MATH-500-style problem into a
  factor graph in the engine's vocabulary *at all*? (smallest end-to-end slice; no training.)
- **Regime probe:** a small experiment to decide end-to-end vs stage-wise (the first design
  question) before committing the architecture.
- **Honest kill condition:** if construction collapses to "the LLM just solves it directly without
  needing the factor-graph backend," the two-phase value proposition is in question — surface that
  early rather than building around it.

## Honest caveats
- **NL comprehension is the hard part** — and it's the deducer's *weak* axis (construction is
  parsing/induction-adjacent, not the deducer's inference strength). The constructor carries the
  comprehension; the deducer only critiques/defines/solves.
- **Supervision cost** — gold factor graphs are expensive; the end-to-end route (answer-only
  supervision via the differentiable critic) is attractive *if* it trains, which is unproven.
- This whole brief is **spec-stage**. Jaw 2 is the validated asset; Jaw 1 is the bet.
