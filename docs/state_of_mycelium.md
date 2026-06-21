# State of Mycelium — A Brainstorm-Ready Stock-Take

**Author:** Bryce + Claude · **Date:** 2026-06-21 · **Deadline:** Dec 25, 2026 · **North-star:** MATH-500
**Platform:** AMD 7900 XTX (24GB), tinygrad + AM driver, no ROCm/CUDA/PyTorch

> **Purpose.** This is the *honest* whole-project stock-take to brainstorm from. Every
> concept carries a STATUS TAG and the tags are the point. The doc is organized so the
> **solid ground**, the **vision/spec-stage**, the **dead ends (do-not-revisit)**, and
> the **poetic intuitions** are cleanly separated, ending with the live open questions.
>
> **Status legend:**
> - **VALIDATED-LIVE** — built + validated + on `main` now.
> - **VALIDATED-OLD-LINE-RETIRED** — was validated in the pre-v98 residual-stream line
>   (controller/notebook/waist/rotation), but is NOT on the current live path.
> - **SPEC-STAGE** — designed, not built.
> - **REFUTED** — tested + failed (do not re-probe).
> - **SPECULATIVE-METAPHOR** — an intuition/framing, never rigorously tested as a mechanism.
> - **ABANDONED** — tried then dropped.

---

## 0. TL;DR

**Mycelium is a general factor-graph reasoning engine.** It has two BUILT + VALIDATED
halves: (1) a general **deducer** — the v98-lineage breathing transformer (Pythia-410M
L0–L3 shared across K=16 breaths, ~35.7M shared + ~51.5M token-embedding params), framed
as learned loopy belief propagation on a factor graph; and (2) a general symbolic
**search tier** ("the deducer proposes, complete search disposes"), whose only
domain-specific code is a per-factor-type predicate plus a thin bridge. The same engine
solves graph coloring, Boolean SAT, KenKen, and hierarchical Boolean circuits with
**zero general-core edits each**. A would-be geometric front-end (the Poincaré/hyperbolic
mask generator) remains **spec-stage**.

**The one-line honest verdict:** *the engine's proven edge is GENERALITY (one engine, any
factor graph) + AMORTIZED-FAST parallel deduction (~4 levels/breath, breadth-parallel,
layer-sequential, linear-in-depth with a 4× constant-factor amortization) — NOT inference
quality, which on every regime measured belongs to a cheaper bespoke method (symbolic
search on clean CSPs, loopy BP on soft marginals, simulated annealing on soft MAP).*

---

## 1. Component Overview — how the pieces fit/complement

**Status: VALIDATED-LIVE** (the architecture); the geometric front-end inside it is SPEC-STAGE.

Mycelium decomposes a problem into a **typed factor graph** (variable nodes + typed factor
nodes + membership), then runs two complementary, both-domain-agnostic mechanisms on it.
The factor graph itself splits into **two orthogonal channels**, and the whole pipeline
sits in **three tiers**.

```
        TIER 1: physical puzzle           TIER 2: compiler / virtual machine            TIER 3: the core executor (GENERAL)
        (domain-specific instance)        (membership + inlet; deterministic today)     (zero domain identifiers)
   ┌──────────────────────────┐      ┌───────────────────────────────────────┐   ┌──────────────────────────────────────────┐
   │ cells, cages, gates,     │ ───► │  TOPOLOGY channel  → per-head masks     │──►│  THE DEDUCER (breathing transformer)       │
   │ edges, clauses           │      │   (who connects to whom; membership)    │   │   K shared-weight passes = loopy BP rounds │
   │  scripts/build_*_data.py │      │   ─ Poincaré ball would live HERE       │   │   factor_graph_engine.py                   │
   └──────────────────────────┘      │     (SPEC-STAGE; today: factor_masks.py)│   │     ▲ proposes ordering (priors only)      │
                                      │                                         │   └──────────────┬───────────────────────────┘
                                      │  SEMANTICS channel → predicate + inlet  │                  │
                                      │   (what relation must hold)             │   ┌──────────────▼───────────────────────────┐
                                      │   ─ predicate registry (search)         │──►│  THE SEARCH TIER (predicate-driven CSP)    │
                                      │   ─ verification inlet (deducer)        │   │   verifier + GAC + MRV + LCV + backtrack   │
                                      │     csp_registry.py / csp_domains.py    │   │   csp_core.py (sound, complete) disposes   │
                                      └───────────────────────────────────────┘   └──────────────────────────────────────────┘
```

- **The deducer** is learned approximate constraint propagation (loopy BP). It consumes
  *given* membership (topology) and a *given* verification inlet (semantics) and refines
  beliefs over K breaths. Three valid vocabularies for one object: loopy BP rounds = ODE
  energy-descent steps = iterative codec passes.
- **The search tier** is the sound, complete, *symbolic* solver. The deducer's neural
  signal enters it ONLY as ordering priors (MRV/LCV bias), never as commits. On clean
  CSPs the symbolic tier dominates and the neural signal is decorative-to-harmful.
- **The two channels** keep each piece honest: TOPOLOGY (membership → masks / GAC
  incidence) is orthogonal to SEMANTICS (the predicate → verification). Coloring and
  "adjacent-must-be-EQUAL" share topology but have opposite predicates — proof they are
  diagonal in a 2×2, not redundant. Folding semantics into topology is the refuted v100
  "C2 death."
- **The three tiers** localize where domain code lives: Tier 1 (data loaders,
  domain-specific), Tier 2 (mask/inlet generation, should be domain-agnostic), Tier 3
  (the executor, *provably* domain-agnostic). Tier 3 is the validated general core.

**Empirical anchor:** `csp_core.py` + `csp_registry.py` carry zero domain logic (matches
of "coloring/kenken/circuit" are docstrings only). `factor_graph_engine.py` (Tier 3
deducer) and `factor_masks.py` (Tier 2 masks) are likewise generic; the only per-domain
diffs are in `csp_domains.py` (predicate + bridge).

---

## 2. SOLID GROUND (VALIDATED-LIVE)

### 2.1 The generality result — zero general-core edits across four domains
**Status: VALIDATED-LIVE.** The same engine (deducer + search tier) solves **graph
coloring** (binary not-equal, flat non-partition), **Boolean SAT** (n-ary clauses),
**KenKen** (param-carrying arithmetic cages + 7-ary all-different, flat partition), and
**hierarchical Boolean circuits** (AND/OR/NOT layered DAG) with ZERO edits to the general
core. Domain code enters ONLY via predicate + bridge (`csp_domains.py`) and membership +
inlet. *Evidence:* Phase 0 (commit `6af771c`) byte-identical coloring reproduce on 22+
fixtures (`solve_symbolic` new==old, `VERIFY_PARITY` max|Δlogit|=0), GAC≡AC-3, MRV≡DSATUR
*by construction*; leak-hunt added SAT via the public API alone (predicate + bridge, zero
core edits). Phase 2 (commit `afc4a2f`) KenKen `git diff = csp_domains.py only`; symbolic
solves 100% across all givens bands.

### 2.2 The deducer recipe (the breathing executor)
**Status: VALIDATED-LIVE.** Pythia-410M L0–L3, all 4 layers SHARED across K=16 breaths;
the 1024d residual is the persistent state. Each breath: per-breath additive marker → 4
shared transformer layers with **runtime-built per-head masks from membership** →
learnable **`delta_gate`** convex residual blend → per-breath layernorm + **value-codebook
readout** → per-breath **calibration head**. Trained with the **per-breath weighted-CE
ladder** (`loss = Σ_k (1 + k/(K−1))·CE(logits_k, target)`) — the reason K matters.
Validated sub-recipes, each portable: aligned embed↔codebook init (step-0 given→gold);
structured frozen per-head masks (heads carry row/col/cage/global structure, model never
"learns what a row is"); per-breath markers + supervision (without them, K is flat).
*Evidence:* v98 Sudoku 78.3% cell at step 200, B0=0.99→B14=0.60 ladder; byte-identical to
this engine when driven with KenKen inputs.

### 2.3 The general symbolic search tier (Path B)
**Status: VALIDATED-LIVE.** A sound, complete, predicate-driven backtracking CSP solver.
The ONLY domain code is a three-valued predicate `predicate(ftype, params, member_values)
→ {SAT | UNVIOLATED | VIOLATED}` + a thin bridge. Everything else (verifier, GAC, MRV,
LCV, backtracking) derives generically. DSATUR-is-MRV and AC-3-is-GAC are proven by
construction, not asserted. Registration self-enforces the hole-monotonicity (L-MONO)
contract. *Evidence:* Phase 0/2 (commits `6af771c`, `afc4a2f`); KenKen B1→B3 collapses the
tree exactly (g10: 305→0.9 decisions, 386→1.1 backtracks); all-different soundness bug
caught + scope-guarded to the bijection regime.

### 2.4 The corrected parallel claim (~1 level/layer, breadth-parallel, linear in depth)
**Status: VALIDATED-LIVE.** The deducer resolves **~1 deduction level per transformer
layer**, so a breath (4 layers = Pythia L0–L3) advances the propagation front ~4 levels,
and depth-D needs ~D/4 breaths — **LINEAR in depth, a 4× constant-factor amortization, NOT
sub-linear.** Parallelism is **BREADTH** (all nodes at a given level resolve together in
one attention pass), **NOT depth** (the depth axis is resolved sequentially, ~1 level per
layer). This is the engine's real superpower, distinct from inference quality. *Evidence:*
K-sweep on depth-16 circuits (commit `b8d73a1`, `docs/circuit_scaling_results.md`,
reproduced Jun 20): K=4 recovers 95% of K=16 (ratio K=4/K=16 = 0.95 across bands D6–D16);
per-D accuracy flat (D6 0.959 → D16 0.923, no depth ceiling).

### 2.5 We lose to symbolic on clean CSPs + the cheap-scorer principle
**Status: VALIDATED-LIVE (honest negative).** On clean verifiable CSPs, symbolic search
wins for free; learned probabilistic propagation is **net-negative**. Path B hard
3-coloring (commit `649855a`): B3 symbolic AC-3 = 0.95, B1 no-prop = 0.85, B2b neural-prop
= 0.825 (≤ B1). The `CONF_THRESH` sweep nails the mechanism: at threshold = 1.0 (never
commit) B2b ≡ B1 byte-identical (decisions 46.9 = 46.9). The design rule: **a propagation
commit must be logically FORCED (100%), not a confident guess; neural signal = ordering
only.** The **cheap-scorer principle**: a learned whole-graph value/confidence has a
customer ONLY where there is NO CHEAP EXACT SCORER of a complete solution — on hard CSPs
the verifier is free; on soft-opt the objective/cost is free; on soft-MRF cheap baselines
(BP/SA) already close the gap. The "learned-beats-bespoke" regime is scarce-to-empty.

### 2.6 The verification inlet + the predicate registry (the SEMANTICS channel, two consumers)
**Status: VALIDATED-LIVE.** Same constraint truth, two consumers. The **predicate
registry** (`csp_registry.py`) feeds the symbolic search; the **verification inlet** feeds
the deducer (per-cage op-type, log-bucketed target, cage-size added as residual features
every breath, never gated, never an op-type mask channel). Constraint content lives in one
place (data gen + deducer inlet + search predicate). *Evidence:* coloring (no inlet,
symmetric not-equal implied by membership), KenKen (op+target inlet + `cage_pred`),
circuits (gate-type inlet + gate truth tables) — all trained and validated.

---

## 3. THE VISION / SPEC-STAGE (designed, not built)

### 3.1 The three tiers (Tier 1 nodes / Tier 2 compiler-VM / Tier 3 executor)
**Status: VALIDATED-LIVE as a frame; the *learned* Tier-2 is SPEC-STAGE.** Tier 1 =
physical puzzle nodes (the instance). Tier 2 = the compiler/virtual-machine that reads the
physical structure and builds a **universally-formatted virtual factor graph** (membership
+ latent_type + verification inlet). Tier 3 = the core executor (deducer + search tier),
general over any virtual graph. The refinement: today Tier 2 is *deterministic*
(`factor_masks.py`); the Poincaré ball, if ever built, would make it *learned* while
staying general. Tier 3 is the validated payload.

### 3.2 The Poincaré ball = the TOPOLOGY channel (hyperbolic mask generator)
**Status: SPEC-STAGE.** A learned embedding of the factor graph into the Poincaré
(hyperbolic) ball; per-head attention masks generated as a thresholded distance matrix
(`bias = −softplus(α·(d_hyp − r))`), ONE coordinate field per relation (row/col/cage — the
triangle inequality forbids one field for all three), anchored at t=0 to reproduce the
hard mask byte-exact, then relaxed. It is the TOPOLOGY channel — it generates the wiring,
carrying ZERO relation semantics (that's the predicate registry). *Status detail:* frozen
foothold reproduces hard masks exactly (CPU tensor-match 0.000, GPU eval HYP=1==HYP=0,
`d_hyp` JIT-compiles on the AM driver). **Relaxation is BLOCKED for non-partition graphs**
(clique-union co-location vs per-node radial DOF are incompatible in the gradient; Stage-1
slot-anchor cell_acc 0.827→0.827 flat, 3 design passes all DEAD). The radial-depth
"deep prize" it was meant to deliver is **REFUTED** (§4). Spec:
`docs/hyperbolic_mask_generator_spec.md`. Strictly additive; the hard mask is the
permanent fallback (frozen-off is byte-identical).

### 3.3 The predicate registry = the SEMANTICS channel
**Status: VALIDATED-LIVE** (the registry itself is built and live in the search tier; it
appears here because it is the conceptual partner of the spec-stage Poincaré topology
channel). The predicate is the SOLE domain code in the search tier; it is NOT the Poincaré
ball (different channel, different consumer). They are diagonal: topology generates masks,
semantics verifies relations. (Mechanics in §2.6.)

### 3.4 The Cathedral (notebook + 512-dim silhouette + π-cycled memory wave)
**Status: SPEC-STAGE (parked, trigger-gated).** A three-part cross-breath memory mechanism
ported from the (retired) perceiver: (1) **NOTEBOOK** = K-slot tensor storing per-breath
variable-node summaries, written at breath end / read at start via cross-attention; (2)
**512-dim SILHOUETTE WAIST** = compressed common-mode of the cell state (the "silhouette of
the dancer"), used as the cleaner notebook write-source; (3) **π-CYCLED APERIODIC MEMORY
WAVE** = per-breath phase stamp `k·π/K_max` on notebook write/read (NEVER on cell-MP
attention — that failed 3× on the perceiver), phase-indexing slots so memory is ordered
across breaths/levels. **Trigger (precise):** add only when a measured cap appears that the
persistent 1024d residual cannot carry — specifically a deduction-DEPTH cap *with a MEMORY
signature* (lower levels resolve, then DEGRADE as breaths continue = "forgetting"). A
*never-resolves-deep* signature is structural, NOT a cathedral trigger. Default OFF,
A/B'd vs residual-only, bootstrap-safe zero-init. Spec: `docs/cathedral_port_spec.md`.
**The trigger has never fired** (the engine is depth-parallel with no depth ceiling, ρ=0.134
null shows no depth-ordered breathing, so no forgetting signature observed).

### 3.5 Minimize per-domain weight fine-tuning → general-purpose weights
**Status: SPEC-STAGE.** Today the *code* is general but the *weights* are per-domain
(`fg_coloring`, `fg_circuit`, …). The holy grail is a weight-side mirror of the code-side
win: ONE domain-agnostic backbone co-trained on a multi-task mix, with constraint
semantics fed as INPUT (the verification inlet = a neural predicate registry) + a universal
masked codebook readout + optionally tiny per-domain adapters. Switch tasks with
zero/near-zero retraining. Feasibility rests on proven code-side generality + the proven
two-channel separation; the multi-task generalization itself is **unproven**.

### 3.6 The long-term goal: a general-purpose PARALLEL factor-graph solver
**Status: SPEC-STAGE as an end-state; partially realized.** The aspirational end-state:
ONE engine that (i) accepts ANY factor graph, (ii) reasons with generality (no
domain-specific core code), (iii) solves via parallel distributed deduction (K_min ≈ D/4,
breadth-parallel). Items (i) and (ii) are VALIDATED-LIVE; the "parallel" property is
VALIDATED-LIVE (§2.4); the *general-purpose-weights* leg (§3.5) and the *learned-Tier-2*
leg (§3.2) are the remaining spec-stage gaps.

### 3.7 The neural ordering arm (Phase 3 of the search tier)
**Status: SPEC-STAGE.** Deducer guides branch order (MRV-tie entropy bias + LCV value
bias). Reserved for the soft/learned-constraint frontier where symbolic propagation is
unavailable — on clean CSPs symbolic already dominates (§2.5), so there is no search-hard
band to test. Gated on a trained KenKen deducer checkpoint (does not yet exist) and a
soft-constraint testbed. *Note:* the viability gate already FAILED on clean CSPs (G1, §4).

---

## 4. DEAD ENDS / DO-NOT-REVISIT (REFUTED)

> These are sunk-cost, refuted via independent routes. Do not re-probe.

- **The Perceiver (5× refuted, v118–v121).** A cross-breath summary/compression channel.
  Five designs (filter / bootstrap / notebook IB-init / observer energy-select) all
  mechanically engaged (gradients flowed, weights grew) yet converged to identical ceiling
  val[hard]=0.349, *below* the no-perceiver baseline 0.362. **Why dead:** new attention/
  pointer pathways (~30+ positions) don't bootstrap from task gradient on diverse data
  (the bootstrap law). The engine rejects perceiver-shaped channels.
- **v300 perceiver-CORE.** The perceiver-as-executor (Perceiver-IO style, Llama base, 2048d
  waist) collapsed flat at chance. **Why dead:** same gradient void; no structural anchor
  for the attention pathways. Tier 3 is the v98 executor, NOT a perceiver.
- **GSM8K-as-deduction (17 variants, v66–v95, 0–1.7%).** **Why dead:** the bottleneck was
  reading comprehension (NL→graph parsing), NOT the architecture — confirmed by v98
  Sudoku's success on explicit task structure. Deduction ≠ induction; parsing is a
  separate phase. *(Tagged VALIDATED-OLD-LINE-RETIRED as a finding; the GSM8K-as-deduction
  attempt is REFUTED.)*
- **AlphaZero / MCTS on verifiable CSPs (killed).** PUCT (policy+value) over a CSP
  decision tree. **Why dead:** no game / no self-play / no learned value use-case on
  deterministic verifiable CSPs. Viability gate (commit `b973197`): G1 FAIL (policy
  confidently wrong, mean gold-prob 0.261 < uniform 0.333), G4 PASS but weak calibration
  (~0.28). v106 PUCT sweep (commit, Jun 6): all 8 configs regressed (best Δ=−0.0061). MCTS
  *sampling* discards the free exact verifier + GAC pruning that make the regime cheap.
- **Radial-depth deep-prize / "breath cycles traverse the tree" (refuted).** The hypothesis
  that deduction-depth ↔ radial position ↔ breath-count, breath cycle as a depth-ordered
  geodesic engine. **Why dead:** direct test on the Boolean-circuit DAG (commit `98d1bea`),
  ρ(per-node settle-breath, topological depth) = 0.134 (lower-CI 0.115, bar 0.30) → NULL.
  The engine solves hierarchy (~0.97) via *distributed* constraint-satisfaction, not
  depth-ordered traversal — consistent with §2.4 (depth-PARALLEL → settle does not track
  depth, precisely because parallel ≠ sequential-in-order).
- **Soft-MRF frontier — marginals + MAP (buried).** Ising/Potts, the one regime where
  symbolic GAC has no move (soft factors are never VIOLATED) and learned BP "should" shine.
  **Why dead:** two cheap kill-gates (eval-only, no training). Gate-1 marginals (commit
  `4c8f6eb`): BP-vs-exact gap and its localizability are ANTI-correlated (small β =
  localized but tiny gap; large β = big gap but BP non-convergent 17–83% and unlocalizable;
  1/10 gap-cells both localized + converged). Gate-2 MAP/DAG (commit `688f1b5`): cheap SA
  closes 87% of flat Ising MAP (fair stronger SA 100%); DAG noisy-circuit MPE is BP-trivial
  (max-product exact, Hamming 0.000). Deducer-beats-BP/symbolic on quality is REFUTED across
  clean CSPs, soft marginals, soft MAP, flat and DAG.
- **The Bethe / pairwise-readout bolt-on (cheap version killed).** Attempt to extract joint
  marginals from the frozen residual via a pairwise/Bethe readout. **Why dead:** frozen
  probe (commit `edb8644`, `scripts/probe_pairwise_confidence.py`) found NO joint-consistency
  signal in the residual to extract (residual adds nothing: A−C = −0.0009; pairwise flips
  negative under capacity matching).
  The per-cell product-form readout is a structural mean-field ceiling; a real win needs a
  full retrain with a joint objective, not a readout swap.
- **Neural probabilistic propagation on clean CSPs (net-negative).** See §2.5 — costs ~48
  forwards/solve, never beats no-prop. The neural role is ordering only.
- **The Photon line (E-field + B-field oscillation, refuted Jun 9).** Per-breath Q rotation
  phase + sin² waist gate. **Why dead:** v98 warm-start easy 0.79→0.574 (−0.22); v110-step3
  "+0.028" was continuation drift (proper control: α=0.5 hard 0.397 = baseline); folded-
  phase rescue failed its pre-registered criterion. v109's binary commit/propagate
  alternation was already the optimal form.

---

## 5. THE POETIC FRAMINGS / INTUITIONS (Bryce vocabulary, honestly tagged)

> These are how we *think* about the architecture. Some are vocabulary for real behavior;
> most are metaphors never tested as mechanisms; one (resonance) has a real empirical
> finding under a speculative interpretation. The tags say which is which.

- **The energy function / energy landscape** — **SPECULATIVE-METAPHOR** (vocabulary for
  real behavior). The deducer as a learned ODE integrator descending an implicit Hopfield
  energy `E(x)` (Ramsauer 2020: attention = energy descent); per-breath weighted CE = the
  landscape, `delta_gate` = step-size, calibration head = Dopri5-style error estimator.
  *Honest status:* one of three valid vocabularies (loopy BP / ODE / iterative codec),
  consistent with observed geometric energy decay (Sudoku 21.0→0.71 over K=1…20), but a
  post-hoc interpretation — the model is trained end-to-end via CE, not explicitly as
  energy descent.
- **Breathing speed (adaptive-K via the convergence instrument)** — **SPEC-STAGE / partly
  VALIDATED-OLD-LINE.** The min-based gold-free convergence instrument (argmin consecutive-
  breath belief JSD) declares "settled" without peeking at gold; the calibration head is a
  per-breath confidence. The instrument is built and sound and in use. The **Property-2
  claim** (harder puzzles settle later) is **UNVERIFIED**: first read on hard-only K=8 was
  UNTESTABLE-by-restriction (depth-narrow settled set, `rho_no_ceiling` sign-flipped =
  ceiling artifact); K=16 curriculum retrain early peek ρ=0.72 / `rho_no_ceiling`=0.67
  (encouraging but underpowered). No powered verdict banked. Bar: lower-CI Spearman ρ>0.30
  + perm p<0.01.
- **Resonance / "ringing the problem" to find its resonant frequency** — **SPECULATIVE-
  METAPHOR** (with a real empirical finding underneath). The intuition: pluck the problem
  like a tuning fork and let its resonant frequencies emerge; the convergence instrument
  literally "listens" to belief-change dynamics. The *real finding* (v22, old line): a
  constant-slope rotation sweep revealed periodic accuracy dips at multiples of π/8 = head-
  collision resonance (rotating by π/8 puts head h where head h±2 was). That finding is
  real; the "tune to the resonant frequency as a strategy" interpretation is unproven.
- **Surface-area-to-volume ratio of the Poincaré ball in high-dim** — **SPECULATIVE-
  METAPHOR.** In high-dim hyperbolic space surface area grows ~exponentially with radius
  (thin-shell effect), so most volume sits near the boundary → variables can spread while
  staying metrically "close," motivating radial-position-as-abstraction. Geometrically
  sound, mechanistically unproven — and relaxation is gradient-dead anyway, so no
  structural advantage demonstrated.
- **Soap bubbles / surface tension of the gradient boundary** — **SPECULATIVE-METAPHOR.**
  Gradients organize like surface tension on a bubble — spread and settle at boundaries,
  minimizing "surface area," explaining why low-dim bottlenecks (waist) work. Evocative;
  no experimental evidence. The codebook-orthogonality penalty (OFF by default) is loosely
  related but is optimization-engineering, not bubble physics.
- **The π-cycled aperiodic memory wave washing over the problem** — **SPEC-STAGE**
  (cathedral component). `k·π/K_max` phase stamp on notebook write/read only (never cell-MP
  attention — that failed 3× on the perceiver). Built + tested on the perceiver (retired,
  bundled, no isolated evidence). NOT on the current deducer; included in the cathedral
  spec build IF the trigger ever fires.
- **The silhouette of the dancer (512-dim waist = common-mode)** — **SPEC-STAGE**
  (cathedral component). The lossy 512d projection capturing the essential "silhouette" of
  the distributed deduction, used as the cleaner notebook write-source. Speculative claim:
  a single universal shape survives the bottleneck despite per-cell heterogeneity. Ported
  from the perceiver; never validated on the current engine. *(Related: the waist WAS
  load-bearing on the old residual-stream line, Δzero=+3.77/Δswap=+4.59 on v79 — that is
  VALIDATED-OLD-LINE-RETIRED, not the cathedral silhouette.)*
- **The notebook + "50 first dates" analogy → cross-breath memory** — **SPEC-STAGE**
  (cathedral component). Like keeping notes because long-term memory resets each "date"
  (breath): the notebook PINS resolved lower-level deductions across breath boundaries so
  the engine doesn't clobber them. Conceptual framing for why the cathedral might be
  needed on deep DAGs. Untested on the current engine — on every tested regime the single
  1024d residual sufficed (no forgetting signature observed).
- **The diffusion and signal-to-noise (SNR) analogy** — **SPECULATIVE-METAPHOR.** K breaths
  as a reverse-diffusion / denoising process: signal (true constraint) accumulates, noise
  cancels, SNR improves with depth; per-breath calibration is the SNR estimator. Consistent
  with observable monotone entropy drop and the SBP result (training-time noise injection
  helps), but no formal SNR decomposition; the actual mechanism is energy descent /
  constraint-satisfaction, not denoising.

---

## 6. THE EMPIRICAL LEDGER (tried / worked / didn't)

| When | Thing | Outcome | Tag |
|---|---|---|---|
| May 11–14 | Controller / Notebook / LookupTable closed loop (v1–v55) | Closed-loop learned `f(breath_idx)` not `f(rep, breath_idx)`; per-problem std <0.08 → open-loop. Dropped controller (v7: +11/+5/+4). | VALIDATED-OLD-LINE-RETIRED / ABANDONED |
| May 14–25 | Per-breath markers, cross-breath handoff, B-field waist, dual notebook (v9–v45) | L4_MIXED climbed to 96/94/93 (v45). Waist load-bearing (Δzero +3.77). | VALIDATED-OLD-LINE-RETIRED |
| May 28–29 | GSM8K (17 variants v66–v95) | Plateau 0–1.7%; root cause = reading comprehension, not architecture. | REFUTED (GSM8K-as-deduction) |
| May 29 | **v98 Sudoku** | 78.3% cell easy @ step 200; clean per-breath ladder; **paradigm validated**. | **VALIDATED-LIVE** |
| Jun 6 | SBP (stochastic breathing propagation) | Hard 0.3761→0.3914 (+0.0153) at zero inference cost. | VALIDATED-LIVE |
| Jun 6 | v106 PUCT / MCTS | All 8 configs regressed (best −0.0061). | REFUTED |
| Jun 7 | v112b per-node gating vs pairwise bias | Per-position gate engaged (0→0.503); pairwise bias stayed ~0. New project high hard 0.3945. | VALIDATED-LIVE (per-node) |
| Jun 8 | Perceiver 5× (v118–v121) | All converge to hard=0.349 < baseline 0.362. | REFUTED |
| Jun 9 | Photon line (E+B oscillation) | Warm-start −0.22; fold rescue failed pre-registered criterion. | REFUTED |
| Jun 19 | **Phase 0 general search core** (`6af771c`) | Byte-identical coloring; GAC≡AC-3, MRV≡DSATUR; SAT added via API. | **VALIDATED-LIVE** |
| Jun 19 | **Phase 2 KenKen generality** (`afc4a2f`) | One predicate + bridge, `git diff = csp_domains.py only`, 100% solve. | **VALIDATED-LIVE** |
| Jun 19 | Path B coloring — symbolic dominates (`649855a`) | B3 0.95 vs neural-prop 0.825 ≤ no-prop 0.85; CONF_THRESH=1.0 ≡ no-prop. | VALIDATED-LIVE (honest negative) |
| Jun 19 | Radial-depth refuted (`98d1bea`) | ρ=0.134 << bar 0.30. | REFUTED |
| Jun 20 | Soft-MRF gates (`4c8f6eb`, `688f1b5`) | Marginals anti-correlated; MAP cheap-SA 100%; DAG BP-trivial. | REFUTED |
| Jun 20 | Bethe-readout frozen probe (`edb8644`) | No joint-consistency signal in residual. | REFUTED (cheap version) |
| Jun 20 | Circuit scaling (`b8d73a1`) | K=4 ≈ 95% of K=16; per-D flat → ~4 levels/breath, breadth-parallel. | VALIDATED-LIVE |
| Jun 20–21 | **Consolidation** | `docs/general_engine_results.md` + this doc; own the validated result. | VALIDATED-LIVE |

---

## 7. THE KEY REALIZATIONS

- **STOP making the breathing loop FIND structure — it should EXECUTE on a given graph
  (deduction, not induction).** **VALIDATED-LIVE.** Structure-finding (parsing/induction)
  is a separate Phase 1; the breathing loop (Tier 3) executes on *given* membership +
  inlet. The perceiver conflated discover-and-execute; the current design separates them.
  This separation is *why* generality works.
- **Induction vs deduction (the frontier trap).** **VALIDATED-LIVE.** DEDUCTION = propagate
  on a given graph (our strength). INDUCTION = find the rule / build the graph (ARC,
  program synthesis, NL parsing — not the engine's job; there Mycelium is at most a
  bit-part verifier). The recurring trap: relayed proposals center the buried search tier
  on problems whose hard part is induction or is BP-trivial.
- **Hard vs soft constraints.** **VALIDATED-LIVE.** Hard CSP → three-valued predicate
  (SAT/UNVIOLATED/VIOLATED) → exact verifier + symbolic search. Soft MRF → nothing is ever
  VIOLATED → GAC has no admissible move → approximate inference (BP/SA/learned BP). The
  engine's proven regime is hard + symbolic-dominated; the soft regime is where the deducer
  "should" earn its keep — but cheap baselines win every measured soft regime (§4).
- **The cheap-scorer test.** **VALIDATED-LIVE.** A learned value/confidence has a customer
  ONLY where there is no cheap exact scorer. Verifier free on hard CSPs; objective free on
  soft-opt; BP/SA close soft-MRF. The only no-cheap-scorer zone left is NL-uncertain
  constraints (Phase 1, set aside).
- **The grouper's two sets of jaws (Jaw 1 GAC / Jaw 2 search).** **SPECULATIVE-METAPHOR**
  (vivid, modest predictive power). Jaw 1 = constraint propagation (the deducer as
  generalized arc-consistency, and GAC); Jaw 2 = systematic search (MRV/LCV/backtrack). On
  clean CSPs Jaw 1 in the *winning* config is **SYMBOLIC GAC/AC-3 (B3 = 0.95)** — NOT the
  neural deducer-as-GAC, which was net-negative (§2.5) — and it inhales the problem,
  leaving Jaw 2 a tiny residual; on soft constraints Jaw 1 has no VIOLATED move so Jaw 2
  must become approximate inference. The
  two-jaws structure appears empirically (ordering matters: DSATUR 0.85 vs neural-entropy
  0.35) but is a framing, not a validated mechanism.
- **Generality is the prize, not quality.** **VALIDATED-LIVE.** The durable takeaway: the
  contribution is engineering + generality + honesty (one engine, any factor graph,
  amortized-fast parallel deduction), NOT a solver SOTA. On quality the engine beats
  nothing measured.

---

## 8. OPEN QUESTIONS FOR THE BRAINSTORM

The genuinely-live directions (everything in §4 is closed; do not re-open it):

1. **Own the validated result, or position the engine as a verifier component?** The honest
   framings are (a) OWN the general engine (generality + amortized-parallel, the paper
   we can write now), or (b) Mycelium as a fast deterministic VERIFIER inside a bigger
   synthesizer/LLM (the headline is theirs), or (c) hunt a frontier with a precise spec.
   Which posture do we commit to for the Dec-25 MATH-500 deadline?

2. **General-purpose weights via semantics-as-input (§3.5).** Does a single Pythia backbone
   co-trained on a coloring+KenKen+circuits+SAT mix, with constraint semantics fed via the
   verification inlet + a universal codebook readout, generalize as well as per-domain
   weights? This is the weight-side mirror of the proven code-side generality — the
   cleanest *positive* next experiment that does not require beating a bespoke solver.

3. **The no-cheap-scorer frontier hunt.** The cheap-scorer test buried soft-MRF, MCTS, and
   the Bethe bolt-on. The ONE remaining no-cheap-scorer zone is **NL-uncertain / learned
   constraints** (Phase 1: NL→factor-graph parsing, the MATH-500 north-star) where the
   constraint itself is uncertain so a learned confidence might finally have a customer.
   Is Phase 1 the frontier, and if so is the deducer the backend or a bit-part? What is the
   cheap kill-gate we run *before* building?

4. **Does any old-line / spec idea have a live role?**
   - *Cathedral (notebook + silhouette + π-wave):* the trigger (depth-cap + forgetting
     signature) has never fired. Is there a hierarchical-DAG testbed deep enough to fire
     it, or is the residual permanently sufficient?
   - *Resonance / breathing-speed / adaptive-K:* the convergence instrument is real; is a
     *powered* Property-2 verdict (K=16 curriculum retrain) worth banking, and would
     adaptive-K buy real throughput?
   - *Poincaré topology channel:* relaxation is blocked for non-partition graphs and the
     deep-prize is refuted — is there ANY remaining payoff (e.g. transfer across N=5/6/7,
     or partition-only domains) worth the build, or is it permanently shelved?

5. **Amortized-speed as the product (deferred).** If the edge is generality + parallel
   deduction at a fixed forward pass, is the right deliverable a throughput/differentiable
   inference component over heterogeneous factor-graph distributions — and what benchmark
   makes that edge legible?

---

### Cross-references
- `docs/general_engine_results.md` — the consolidated quantitative results (every number traced to a commit).
- `docs/general_factor_graph_search.md` — the search-tier spec (Phases 0+2 built).
- `docs/circuit_scaling_results.md` — the reproduced ~4-levels/breath parallel claim.
- `docs/hyperbolic_mask_generator_spec.md` — the Poincaré/Tier-2 spec (SPEC-STAGE).
- `docs/cathedral_port_spec.md` — the cathedral (notebook + silhouette + π-wave) spec (SPEC-STAGE, trigger-gated).
- `CLAUDE.md` — the durable agent brief + editing rules.
- `memory/` — the per-finding notes (project_*, feedback_*).
