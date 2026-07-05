# Mycelium: A General Factor-Graph Reasoning Engine

ONE engine, ANY factor graph. A small (~87M-param) iterative transformer that
solves constraint problems by **breathing** — K passes through 4 shared
Pythia-410M layers, each pass one round of factor-graph inference in a 1024d
residual stream — driven *purely* from a factor graph (variables + typed factor
nodes + membership), with no problem-specific architecture. **Generality is the
thesis:** the same weights and the same forward run graph coloring, hierarchical
Boolean circuits, and KenKen.

**Author:** Bryce + Claude
**Date:** 2026-06-20
**Deadline:** December 25, 2026
**Platform:** Shadow Glass (AMD 7900 XTX, 24GB) · tinygrad + AM driver · no ROCm
**Target:** MATH-500 (current empirical benches: graph coloring · Boolean
circuits · KenKen CSP)

> **The authoritative current brief is [`CLAUDE.md`](CLAUDE.md)** (direction,
> specs, editing rules) — including the current spec-stage forward design, the
> two-phase **Alternator** (CLAUDE.md §8; summarized below). The two active design notes are the general symbolic
> search tier
> [`docs/general_factor_graph_search.md`](docs/general_factor_graph_search.md)
> (Phases 0+2 built and validated) and — spec-stage only — the hyperbolic mask
> generator
> [`docs/hyperbolic_mask_generator_spec.md`](docs/hyperbolic_mask_generator_spec.md).
> The pre-v98 conceptual arc (sine-wave breath, π-cycled RoPE, Controller /
> Notebook / LookupTable loop) is archived at
> [`docs/archive/`](docs/archive/). The full v100–v300 lineage is preserved in
> git history (removed from the working tree in the Jun-16 clean).

---

## The thesis: one engine, any factor graph

The holy grail is **generality**. A problem's topology should not be hardwired.
A factor graph — variables + typed factor nodes + membership — is the universal
description, and Mycelium is **one engine that consumes that description and
deduces**, with no problem-specific code in its core. The proven shape is two
layers:

```
┌──────────────────────────────────────────────────────────────────────┐
│ THE GENERAL DEDUCER  (the validated v98-lineage breathing executor)    │
│   Iterative factor-graph inference from membership alone. Shared        │
│   Pythia-410M L0-L3, K=16 breaths, per-breath delta_gate + calibration  │
│   head, value-codebook readout, per-breath weighted-CE ladder, a        │
│   gold-free convergence instrument. Per-head attention masks are built  │
│   from membership; the verification inlet carries each factor's         │
│   relation. Byte-identical to the v98 KenKen executor on KenKen inputs. │
│   STATUS: ★ VALIDATED ★ on THREE structurally different factor graphs:  │
│           graph coloring · hierarchical Boolean circuits · KenKen.      │
│           mycelium/factor_graph_engine.py  ·  mycelium/kenken.py        │
└───────────────────────────────┬────────────────────────────────────────┘
                                │  the deducer proposes (advisory ordering)
┌───────────────────────────────┴────────────────────────────────────────┐
│ THE GENERAL SYMBOLIC SEARCH TIER  ("Path B": deducer proposes,          │
│   complete systematic search disposes)                                  │
│   Sound + complete backtracking: verifier / GAC propagation / MRV       │
│   var-ordering / LCV val-ordering, all DERIVED GENERICALLY from one      │
│   per-factor-type predicate. DSATUR = MRV-on-coloring and AC-3 = GAC-   │
│   on-not-equal, proven by construction. Validated on coloring, Boolean  │
│   SAT, and KenKen with ZERO general-core changes.                       │
│   STATUS: ★ Phases 0+2 BUILT + VALIDATED ★ (branch mycelium-factor-     │
│           graph). Neural-ordering arm is spec-stage.                     │
│   docs/general_factor_graph_search.md  ·  mycelium/csp_core.py          │
└──────────────────────────────────────────────────────────────────────┘
```

**What is built versus spec'd — read this carefully, it is the central honesty
discipline of the project.** The **general deducer** and the **general symbolic
search tier (Phases 0+2)** are built and validated. The **Poincaré (hyperbolic)
embedding + hyperbolic mask generator** — the would-be continuous-topology layer
— remain **SPEC-STAGE, not built or tested**
([`docs/hyperbolic_mask_generator_spec.md`](docs/hyperbolic_mask_generator_spec.md)),
and the radial-depth "deep prize" they were meant to pay off has been **refuted
on the DAG testbed** (see below). Nothing in this document states or implies
that the hyperbolic layer is built, working, or validated.

### The two channels — the clean conceptual frame

A factor graph splits into two orthogonal channels, and every component lands in
exactly one:

- **TOPOLOGY** — *who connects to whom.* This is membership, and it drives the
  per-head **attention masks** in the deducer (its structural, geometric
  channel).
- **SEMANTICS** — *what relation must hold.* This is the per-factor-type
  **predicate**. In the symbolic search tier the predicate registry IS the
  semantics channel; in the neural deducer its twin is the **verification
  inlet** (each factor's op / target / size, added as a feature, never as an
  attention channel).

Topology is structural and shared across problems of a class; semantics is one
small function per relation type. Holding them apart is what keeps the engine
general: a new domain contributes only a predicate (+ a thin bridge), never new
core machinery.

### The deducer's real superpower: parallel deduction that scales

The headline empirical finding from the Boolean-circuit DAG testbed
(`mycelium/circuit_data.py`): **distributed deduction is PARALLEL, and it
scales sub-linearly in depth.** A 4-layer transformer performs ~4 attention
hops per breath, so the engine resolves **~4 deduction levels per breath**. It
therefore solves deep circuits in roughly `K_min ≈ D/4` breaths — e.g. depth-16
circuits in ~4 breaths — `acc(K≈4, D=16) ≈ acc(K=16, D=16)`
(`scripts/eval_circuit_scaling.py`). This is the engine's actual advantage:
**parallel breadth, not depth-sequential descent.**

### The general symbolic search tier (Path B)

When constraints are clean and verifiable, the deducer's value is the *ordering
proposal*; a sound, complete symbolic search disposes — "the deducer proposes,
complete systematic search disposes"
([`docs/general_factor_graph_search.md`](docs/general_factor_graph_search.md)).
The design's guarantee: the **only** domain-specific code is a per-factor-type
predicate plus a thin bridge. Everything else — the verifier, GAC propagation,
MRV variable-ordering, LCV value-ordering, backtracking — is derived
generically. Two identities are proven *by construction*, not asserted:
**DSATUR = MRV-on-coloring** and **AC-3 = GAC-on-not-equal**.

It is validated on three structurally different domains with **zero general-core
changes** each:

- **graph coloring** — binary not-equal factors;
- **Boolean SAT** — n-ary clause factors (GAC on a clause = unit propagation);
- **KenKen** — param-carrying arithmetic cages + 7-ary all-different; this
  needed only a new predicate + bridge + one specialized all-different
  propagator slotted through the *existing* dispatch seam.

### Where neural search would earn its keep (the honest limit)

The key empirical finding, stated honestly: **on clean verifiable CSPs,
symbolic search dominates.** It solves for free with smaller trees; the learned
*probabilistic* propagation arm was **net-negative** — a propagation commit must
be logically FORCED, not a confident guess. So the neural deducer's demonstrated
value is **generality + parallel deduction, NOT search value or ordering.**
Neural search would only pay off where symbolic propagation is *unavailable* —
soft, learned, or natural-language-specified constraints. That frontier is the
forward research direction (below), not a current result.

---

## The general deducer (the breathing transformer)

Everything below is built and validated. The deducer is the v98-lineage
breathing executor — **not a perceiver** (the perceiver is retired; see below).
`mycelium/factor_graph_engine.py` parameterizes it for an arbitrary typed factor
graph and is **byte-identical to the v98 KenKen executor (`mycelium/kenken.py`)
when driven with KenKen inputs** at matching hyperparameters.

### The one-paragraph mechanism

A small iterative transformer (4 Pythia-410M L0-L3 layers, **shared** across
all K breaths, h=1024, 16 heads, ~87M params) performs factor-graph inference
by K passes through the same weights. Each breath: add a per-breath additive
marker → 4-layer transformer with a **structured per-head attention mask**
encoding the factor topology → an optional projection waist → a learnable
per-breath `delta_gate` residual blend → per-breath layernorm + value-codebook
readout → per-breath calibration head. The K breaths are JIT-unrolled into one
graph. Training uses a per-breath weighted CE — the **ladder**,
`loss = Σ_k (1 + k/(K-1)) · CE(logits_k, target)` — which is what makes the K
axis do work.

### Per-breath forward pass

Each of K breaths runs the same code on the same shared weights:

1. **Embed.** Breath 0 embeds the input grid into `x ∈ ℝ^(B×T×1024)`;
   otherwise `x = x_prev_breath`.
2. **Add per-breath marker.** An orthogonal additive `breath_embed[k]`
   separates per-breath gradients without forcing specialization.
3. **4 Pythia L0-L3 layers (shared across breaths).** Attention uses the
   structured per-head mask. Head-dim 64, FFN 4096, standard RMSNorm + RoPE
   for token positions. **No π-cycled RoPE across breaths** — the legacy
   diversity mechanism is replaced by the per-head structural mask.
4. **(Optional) projection waist.** An explicit lossy compression step
   (the "exhale").
5. **Delta-gate residual update.**
   `x_{k+1} = x_pre + sigmoid(gate_k) · (x_post − x_pre)`. The gates are a
   single learnable `(K_max,)` tensor — static across problems, NOT
   controller-emitted.
6. **Per-breath layernorm + codebook readout.** A value codebook reads the
   residual stream as a soft distribution over cell values.
7. **Per-breath calibration head.** Scalar confidence per breath, trained
   against the detached argmax-correctness target. Conceptually the Dopri5-
   style error estimator for adaptive K.

### The KenKen instantiation

The current executor is `mycelium/kenken.py` — a **direct mirror of the v98
Sudoku design** (box → arithmetic cage):

- Variable-N (N∈{5,6,7}) Latin-square + arithmetic-cage CSP on a fixed 7×7 =
  49-cell grid.
- **K=16 breaths** (the curriculum default).
- **Hard row/col/cage attention masks**: a 5/5/5/1 head split (5 heads each
  for row / col / cage AllDifferent cliques + 1 global head). The cage clique
  is a symmetric per-puzzle membership mask built per batch.
- **A per-cage verification inlet** — arithmetic enters as a *feature* added
  to each cage cell's residual stream (target value + op + cage-size,
  log-magnitude bucketed), never as an op-type attention mask.
- **A 7-value codebook** (values 1..7) readout.
- **A gold-free convergence instrument**: the Property-2 adaptive-depth
  telegraph (below) declares a puzzle *settled* when consecutive per-cell
  beliefs stop moving, with no access to the answer.

### The architectural lever per topology

The "instrument" (Pythia + iterated layers + masked attention) is universal.
What specializes it to a problem is the **topology of the attention mask**,
the **readout codebook**, and **K**. Three "musical keys" recur:

| Key | Topology | Rhythm |
|---|---|---|
| Cyclic | Loopy graph, symmetric AllDifferent cliques | Symmetric per-head masks (Sudoku / KenKen) |
| Directional | Tree / DAG, asymmetric functional constraints | Parallel level-resolution (~4 levels / breath; Boolean circuits) |
| Chain | Sequential dependencies within a variable | Adjacent-only attention |

Same instrument, different scale. Coloring / Sudoku / KenKen are the cyclic key,
where symmetric AllDifferent masks work; the Boolean-circuit DAG is the
directional key, where the engine resolves several depth-levels per breath (the
parallel-deduction finding above). The same shared weights and the same forward
run all three — only the membership-derived masks, the readout codebook, and K
change.

### Three vocabularies for one object

The same execution is described three ways, all valid:

- **JPEG codec.** Each breath is a learned compression codec: Transform
  (attention) → Quantize (waist) → Encode (delta_gate) → psychoacoustic model
  (the per-breath CE).
- **ODE integrator.** `dx/dt = −∇E(x)` on the factor-graph energy E. 4 layers
  per breath = RK4-like stages; K breaths = K integration steps; delta_gate =
  step size; calibration head = Dopri5-style adaptive-timestep error
  estimator.
- **Approximate belief propagation.** Per-head masks = factor topology; K
  breaths = K loopy-BP rounds; per-breath weighted CE = monotonic belief
  refinement. The signature is geometric energy decay (Sudoku: 21.0 → 0.71
  over K=1…20) and the multi-orders-of-magnitude correlation between cell
  accuracy and puzzle accuracy (joint MAP, not independent marginals).

These are three vocabularies for one mathematical object — **a learned
approximate iterative solver for joint MAP inference on a factor graph,
structured as an ODE integrator with energy-based dynamics.** Foundation:
attention IS one Hopfield energy-descent step (Ramsauer et al., 2020). See
[`paper/outline.md`](paper/outline.md),
`memory/project_factor_graph_framing.md`, and
`memory/project_ode_integrator_framing.md`.

---

## Empirical status (honest)

### Generality — one engine, three structurally different factor graphs

The load-bearing result of this arc: the **same** breathing deducer — same
shared weights, same forward — solves three factor graphs that share *nothing*
structurally:

- **graph coloring** (binary not-equal cliques),
- **hierarchical Boolean circuits** (a layered AND/OR/NOT DAG;
  `mycelium/circuit_data.py`),
- **KenKen** (Latin-square + param-carrying arithmetic cages).

The only per-domain inputs are the membership-derived masks, the readout
codebook, and the verification inlet — no domain-specific architecture. The
general wrapper (`mycelium/factor_graph_engine.py`) is **byte-identical to the
v98 KenKen executor on KenKen inputs**, which pins generality to a validated
baseline rather than a fresh claim.

### Parallel deduction scales sub-linearly in depth

On the Boolean-circuit DAG, the engine resolves **~4 deduction levels per breath**
(a 4-layer transformer ≈ 4 attention hops). It solves deep circuits in roughly
`K_min ≈ D/4` breaths — `acc(K≈4, D=16) ≈ acc(K=16, D=16)`
(`scripts/eval_circuit_scaling.py`). The engine is **depth-parallel, not
depth-sequential** — this is its real superpower.

### Symbolic search dominates on clean CSPs (the honest negative)

On clean, verifiable CSPs the **general symbolic search tier** (coloring / SAT /
KenKen, zero general-core changes) solves with smaller trees, for free. The
learned **probabilistic** propagation arm was **net-negative**: committing the
deducer's confident guesses *causes* backtracks. The lesson, banked: *a
propagation commit must be logically FORCED, not a confident guess.* The neural
deducer's demonstrated value is therefore generality + parallel deduction, **not**
search value/ordering — neural search would only earn its keep where symbolic
propagation is unavailable (soft / learned / NL constraints).
See [`docs/general_factor_graph_search.md`](docs/general_factor_graph_search.md).

### Radial-depth "deep prize" — REFUTED on the DAG testbed (honest null)

The hyperbolic layer's would-be payoff was *deduction-depth ↔ radial position ↔
breath-count.* On the Boolean-circuit DAG — exactly the hierarchical testbed
reserved for this verdict — it is a **clean null** (rho ≈ 0.13 against the
matched depth-shuffle control; `scripts/analyze_circuit_rho.py`). The mechanism
explains the null: the engine is **depth-PARALLEL** (~4 levels per breath), so it
does not perform depth-ordered radial traversal. The radial-depth claim is not
the expected payoff; treat the hyperbolic layer's transfer/interpolation value as
unproven (see "What is spec-stage" below).

### The KenKen reframe — what the "0 puzzle-acc ceiling" actually was

The apparent **"0 puzzle-acc ceiling"** on hard puzzles was an **eval-regime
artifact, not an architecture wall.** v98 Sudoku scores ~97% cell / 79% puzzle
when graded on its **easy band (43%-givens)**; the *same model* collapses to
~5% puzzle on its **33%-givens band**. KenKen made this explicit and reframed
the project's central empirical question accordingly. The headline number was
never about whether the breather can solve — it was about which band it was
graded on.

### Property-2 first read — UNTESTABLE-by-restriction

Property-2 is the adaptive-depth claim: *harder puzzles should take more
breaths to settle.* The **first read was UNTESTABLE by restriction**, and the
honesty matters:

- On the **hard-only K=8 model**, the settled set was **depth-narrow** — there
  was no breath-count spread *with* depth, so the correlation is undefined
  (restriction of range), not weak.
- The companion `rho ≈ 0.5` that initially looked promising was a **ceiling
  artifact**: the `rho_no_ceiling` control (drop puzzles that hit the K
  ceiling) **sign-flipped**, which is the tell that the raw correlation was
  carried by ceiling-pinned puzzles.

A **K=16 curriculum retrain** is the powered retry. An early peek (underpowered,
not a verdict) showed the settled set **deepening** and **N=5 settled rho = 0.72
with rho_no_ceiling = 0.67** — encouraging and *consistent*, but underpowered
against the bar; no powered verdict has been banked. The analyzer was patched so
a restriction-of-range no longer reports as a win — it reports **UNTESTABLE** —
and `rho_no_ceiling` is now a **required companion control** on every read.
(Adaptive-depth is a separate question from the radial-depth null above:
Property-2 asks whether *harder* puzzles settle later in breath-count, not
whether breath-count tracks radial position.)

### Property-2 read discipline (the bar)

The convergence instrument and its read are deliberately conservative. The
discipline (in `scripts/analyze_kenken_property2.py`):

- **Min-based instrument.** breath-count = gold-free argmin of the consecutive
  belief change; no peeking at the answer.
- **settled = correct.** Only puzzles the instrument calls *settled* count
  toward the primary statistic.
- **Depth-spread first.** If the settled set has no breath-count spread across
  depth, the read is **UNTESTABLE**, not "weak" — restriction of range is not
  a null result.
- **`rho_no_ceiling` companion control.** Report rho both with and without
  ceiling-hit puzzles; a sign flip invalidates the raw rho.
- **The bar:** HILL-STANDS requires **lower-CI Spearman rho > 0.30 AND
  permutation p < 0.01** in ≥2/3 of qualifying bins (a bin qualifies only with
  settled-n ≥ 50 and frac-settled-strict ≥ 0.80).

### The v98 Sudoku K-sweep — the central paper claim

The most important measurement in the project: a K-sweep on easy puzzles where
**constraint energy decays geometrically at ~0.5× per ~3 breaths**, 21.0 → 0.71
over K=1…20. That geometric decay is the mathematical signature of loopy-BP
convergence on a factor graph with cycles. The **multi-orders-of-magnitude
correlation** between cell accuracy and puzzle accuracy is the diagnostic that
the model is solving *structures* (joint MAP), not classifying independent
cells. See `memory/project_v98_sudoku_validates_paradigm.md` and
[`paper/outline.md`](paper/outline.md).

---

## Editing rules (load-bearing)

These rules are still load-bearing in current code:

- **No mid-breath token generation.** Reasoning stays in the 1024d residual
  stream; tokens (if any) are generated once at the end. Empirically: "had had
  had" collapse within 2 autoregressive loops if violated.
- **Diversity must be structural, not learned.** Row/col/cage masks and
  topological staging are geometric. Every v1–v3 learned diversity mechanism
  (scales, soft tokens, codebooks, fingerprints) collapsed to constant within
  one epoch.
- **Factor per-NODE, not per-EDGE.** Prefer per-position gating (each position
  gets its own activation pattern in the shared backbone) over pairwise
  structures (learned attention biases, edge-strength tensors). The v112b
  finding: a per-position residual gate became load-bearing while a pairwise
  attention-bias channel refused to engage. Edges are already captured by the
  binary masks; per-node activation patterns are what is missing. See
  `memory/project_v112b_phase1_validates_factorization.md`.
- **Attention-bootstrap principle.** A new attention pathway needs direct
  supervision (or a known-good init) for ~500 steps — task gradient through
  softmax is too weak to grow one from random. Codebook selection (≤32-way)
  bootstraps from task gradient alone; pointer attention (~30+ positions) does
  not. This is why the engine **derives its masks from membership** (a known-good
  structural input) rather than learning a topology from scratch — and why the
  perceiver, which had no such anchor, hit a gradient void. (It is also why the
  spec-stage hyperbolic generator was designed to *anchor* to the hard mask and
  relax; in practice that relaxation is gradient-dead off the partition case —
  see "What is spec-stage" below.)
- **Property-2 read discipline** (above): min-based instrument, settled =
  correct, depth-spread first, `rho_no_ceiling` control, bar = lower-CI
  rho > 0.30 + p < 0.01.
- **KV cache invariants.** Size to actual seq length; pad eval batches to a
  fixed batch size so compiled graphs match; compile once during first eval,
  replay for the rest.
- **Bryce wants root-cause perf fixes**, not workarounds, when perf is the
  bottleneck.
- **Gradient separation (legacy, currently moot).** If a Controller-like
  decision module is ever reintroduced, its gradient must never reach
  transformer weights — separate optimizer, parameter-change smoke. No
  Controller is in the current code path.

---

## Specifications

**Initialization.** Pythia-410M layers 0-3 (attention + FFN weights), token
embeddings (50304 × 1024), untied output head. In v98+ all 4 layers share the
SAME weights across all K breaths — no phase-specific copies.

**Model dimensions.** Hidden 1024, 16 heads × head-dim 64, FFN 4096, vocab
50304, max seq 512. 4 transformer layers in the iterated stack, shared across
breaths.

**K.** KenKen / coloring / circuits use K=16; v98 Sudoku used K=20. No global
ceiling — AMD 7900 XTX JIT capacity (~20 breaths) is the practical limit. On the
circuit DAG, `K_min ≈ D/4` suffices (parallel deduction).

**Parameters.** ~87M total: ~35.7M shared transformer processing + ~51.5M
token embeddings. The 7-value codebook and verification inlet add a small tail.

**Memory.** ~5GB mixed precision on a 7900 XTX, ~19GB headroom. Batch 32–64 at
FIXED_LEN=160 for the 49-cell grid. KV cache sized to actual sequence length.

**Substrate laws (AM-driver landmines).** No `dtypes.float32` literal inside
the JIT step; `scores.clip(-1e4, 1e4)` for numerical stability; `where()`-gated
NaN guard (multiply-gate fails because NaN×0=NaN); scalar `isfinite`. For the
hyperbolic metric specifically: clamp `|z|² ≤ 1 − 1e-5` and the arccosh
argument `≥ 1 + 1e-7`, and mirror the −1e4 block magnitude (not −inf) so the
softmax stays finite. See `memory/reference_tinygrad_am_quirks.md` and
[`docs/hyperbolic_mask_generator_spec.md`](docs/hyperbolic_mask_generator_spec.md) §4.

**Platform.** AMD Radeon RX 7900 XTX (24GB GDDR6). Tinygrad + AM custom
userspace driver (working since 2026-05-11 — Secure Boot off +
`vm.compact_unevictable_allowed=0`). No ROCm, no CUDA, no PyTorch. Ubuntu 24.04.

---

## The two-phase Alternator (spec-stage design — 2026-07-04)

The forward frontier named in the roadmap — problems where symbolic propagation isn't
enough because the constraints are **NL-specified** — now has a concrete design. The
Alternator interleaves parsing and solving so the factor graph is built *iteratively
under deductive feedback*. **Everything in this section is SPEC-STAGE — designed, not
built. The validated deducer is untouched and remains the regression anchor.** The full
brief (interface contracts, null hypotheses, brick ladder, kill criteria) is
[`CLAUDE.md`](CLAUDE.md) §8.

```
     tokens ──┐
              ▼
┌───────────────────────────┐  SYN: graph delta (registry + ball)
│ PHASE 1 — PARSER          │ ────────────────────────────────┐
│ Llama-base 2048d L0–L3    │                                 ▼
│ weight-invariant          │            ┌─────────────────────────────────┐
└─────────────▲─────────────┘            │ PHASE 2 — DEDUCER               │
              │                          │ Pythia 1024d L0–L3 — the        │
   NACK: re-parse request                │ VALIDATED v98-lineage engine,   │
   + notebook state                      │ untouched; never sees NL        │
              │                          └───────────────┬─────────────────┘
┌─────────────┴─────────────┐   waist common mode        │  ACK: settled state
│ PERCEIVER (monitor —      │◄───────────────────────────┘
│ session state · spectral  │──► NOTEBOOK (accumulate ledger + replace scratch)
│ segmenter · global lats)  │
└───────────────────────────┘        × 6 cycles · the 7th breath decodes the KV cache
```

- **Two trunks, both weight-invariant across all six cycles** (not one shared weight
  set). Per-cycle variation is *input-conditioned* — notebook state + NACK — which is the
  anti-gradient-tug-of-war design and the **zero-LoRA null hypothesis**. Progressive
  resizing runs coarse→fine across cycles: early cycles parse global scaffold, late
  cycles refine exact predicates.
- **The interface is exactly three objects** — the two channels plus memory. ONE
  canonical **predicate registry** (semantics), ONE **Poincaré ball** (topology —
  parser-emitted differentiable masks; the §"spec-stage" relaxation caveat is the hard
  risk), ONE **notebook** (temporal memory from the deducer's silhouette read — see
  CLAUDE.md §8.6 for the waist-vs-tap split; the spec's 512d readout waist is unbuilt,
  and all existing silhouette evidence comes from the final-breath readout-LN *tap*:
  append-only ledger for committed deductions, replace-scratch for hypotheses).
  Topology ≠ memory — the ball is not the notebook.
- **A TCP-style handshake justifies the alternation.** SYN = parse delta; ACK = settled
  state via notebook; **NACK** = deducer contradiction routed backward so the next cycle
  re-parses the offending region (the factor-graph error-localization role reborn).
  Alternation earns its cost only if the NACK path works.
- **The perceiver returns, narrow.** Perceiver-as-core stays retired; Brick-1
  ([`docs/perceiver_poincare_design.md`](docs/perceiver_poincare_design.md) §9) showed a
  small latent bank breathes against Poincaré anchors. Its new job is monitor, not
  engine: track the handshake, host the global-broadcast latents (spatial channel,
  distinct from the notebook's temporal channel), and act as a **learned spectral
  segmenter** — latents as matched filters untangling the ~4–5 superposed step
  signatures in the waist silhouette, classifying each against registry centroids,
  emitting the unmatched residual as the NACK.
- **Compression lives only at the waist** (layers run full-width), with a
  Matryoshka-style 512→128 nested-dim schedule — over training time (handicap) and/or
  over cycles (coarse=narrow, fine=wide) — and a companion instrument: the 0.85
  valid/invalid common-mode separation (the *learned-nonlinear deconfounded* read;
  PCA-linear floor 0.658 — CLAUDE.md §8.6) measured as a function of prefix width. A
  dormant in-loop 256d bottleneck already exists in the deducer code (gate-closed,
  never trained — parked for objective reasons, not lack of signal).
- **Three load-bearing assumptions are unvalidated** (conditioning suffices; matched
  filters segment; the *fully differentiable* NACK — though a v0 NACK assembles from
  validated parts: calibration head as session health + symbolic per-factor VIOLATED
  flags as parse-cycle input), gated by a brick ladder starting with **Brick-0**: frozen
  latents reading an *existing* silhouette-tap common mode must beat both the 0.658
  PCA-linear floor and fixed analytic matched filters before anything is wired in.
  Brick-B (segmentation) is ungated from the alternation via composed-problem
  supervision — run problems with known constituent factors through the trained engine
  and recover them (linearity of superposition checked first, not assumed).

---

## What is spec-stage (NOT built — do not imply otherwise)

- **The Poincaré (hyperbolic) embedding + the hyperbolic mask generator** (the
  would-be continuous-topology layer,
  [`docs/hyperbolic_mask_generator_spec.md`](docs/hyperbolic_mask_generator_spec.md)).
  **Spec only.** The geometry can reproduce hard masks **frozen** (byte-exact),
  but its transfer/interpolation payoff is **unproven**, and mask relaxation is
  **gradient-dead for non-partition factor graphs** (it has no gradient to learn
  from off the clean-partition case).
- **The radial-depth "deep prize"** (deduction-depth ↔ radial position ↔
  breath-count) is **REFUTED** on the Boolean-circuit DAG testbed (honest null,
  rho ≈ 0.13). The engine is depth-PARALLEL, not depth-sequential, so it does
  not do depth-ordered radial traversal. This is not the expected payoff and
  should not be framed as one.
- **The two-phase Alternator** (the section above + CLAUDE.md §8) — the six-cycle
  parse/solve loop, the TCP handshake, the perceiver-as-monitor, the Matryoshka waist
  schedule. **Spec only**; three load-bearing assumptions unvalidated; Brick-0 not yet
  run.
- **The neural-ordering search arm** (the deducer guiding which branch to try
  first) is spec-stage; on clean CSPs symbolic search already dominates, so this
  arm is reserved for the soft/learned-constraint frontier where symbolic
  propagation is unavailable.

---

## What we carry forward — and what we left behind

**Forward (the validated general-deducer design):**

- Pythia-410M L0-L3 init, FULL weight sharing across breaths.
- Iterative prefill (K passes, residual stream as persistent state).
- Per-breath `delta_gate` (replaces the v1-v95 controller-emitted gate).
- Per-breath calibration head (Dopri5-style error estimator).
- Per-breath weighted CE supervision (the ladder).
- Structured per-head attention masks (replaces π-cycled RoPE).
- Value-codebook readout.
- Per-node residual gating (v112b): same shared backbone, different
  per-position activations — the mycelium principle made architectural.
- The copy-machine principle (no mid-breath token gen) and the
  structural-not-learned diversity rule.
- **Generality, made architectural** (`mycelium/factor_graph_engine.py`): masks
  + readout + verification inlet derived from membership and a per-factor-type
  predicate, so a new domain adds a predicate + bridge, never new core code.
- **The general symbolic search tier** (`mycelium/csp_core.py` — Path B): a
  sound, complete fallback that solves clean CSPs for free; the deducer's role
  there is advisory ordering only.

**Left behind:**

- **The perceiver as ENGINE — RETIRED.** Refuted 5× as an add-on (v118–v121), and the
  v300 perceiver-CORE failed flat at chance. The earlier Mycelium blueprint
  used a perceiver as the executor; we **replaced it with the validated v98
  executor**. The engine is the v98-lineage breathing deducer, **not** a
  perceiver. See `memory/project_v121_perceiver_5x_refuted.md` and
  `memory/project_v118_ablation_perceiver_diagnosis.md`. (Brick-1 later
  validated a small latent bank breathing against Poincaré anchors; the only
  sanctioned revival is the narrow spec-stage **monitor/segmenter** role in the
  Alternator section — never the core.)
- Llama 1B, LoRA atoms / continuous scales, straight-through estimators, soft
  token diversity (all v1–v3).
- Controller, Notebook, LookupTable (v1–v95 — module code preserved for import
  compatibility, called by no current trainer).
- π-cycled per-breath RoPE and sine-modulated within-breath temperature
  (v1–v95 — replaced by structured masks and a single fixed temperature).
- The WaistController GSM8K-via-AR-decode paradigm (v54–v95).
- **The full v100–v300 lineage** (v100–v121 residual-stream factor graphs, the
  v105 digit family, the v200/v300 perceiver-core) — **preserved in git
  history**, removed from the working tree in the Jun-16 clean.

---

## Repository map

```
mycelium/
  kenken.py            — the v98 KenKen breather (the regression oracle; never touched)
  factor_graph_engine.py — the GENERAL deducer (parameterizes kenken.py for any
                          typed factor graph; byte-identical on KenKen inputs)
  circuit_data.py      — the hierarchical Boolean-circuit DAG testbed
  graph_coloring_data.py — the graph-coloring testbed
  csp_core.py          — the GENERAL symbolic search core (verifier / GAC / MRV /
                          LCV / backtracking; ZERO domain identifiers)
  csp_registry.py      — the per-factor-type predicate registry (the SEMANTICS channel)
  csp_domains.py       — domain content: coloring + KenKen predicates and bridges
  pythia.py            — Pythia weight loader
  breathing.py         — Pythia L0-L3 backbone + LEGACY Controller / Notebook /
                          LookupTable (import-compat only; no trainer calls it)

scripts/
  factor_graph_train.py       — the general-deducer trainer (coloring / circuits)
  kenken_train.py             — KenKen trainer (K=16 curriculum)
  build_kenken_data.py        — KenKen puzzle generator
  search_coloring.py / search_kenken.py — the symbolic search drivers
  eval_circuit_scaling.py     — the parallel-deduction (K_min ≈ D/4) read
  analyze_circuit_rho.py      — the radial-depth null read (rho ≈ 0.13)
  analyze_kenken_property2.py — the gold-free Property-2 convergence read

docs/
  general_factor_graph_search.md    — ★ the search tier (Phases 0+2 built) ★
  hyperbolic_mask_generator_spec.md — the hyperbolic layer design (SPEC-STAGE)
  archive/                          — pre-v98 historical content
    vision_v1_to_v95.md             — original architecture vision (deprecated)
    empirical_v45_to_v95.md         — GSM8K WaistController era results
    closed_loop_seven_components.md — the pre-pivot seven-component framing

paper/
  outline.md           — current paper draft

CLAUDE.md              — ★ the authoritative current brief ★
```

The v100–v300 modules (`factor_graph_v1*`, `v200`, `v300`, the perceiver-core,
the v105 digit family) and their docs were removed from the tree in the Jun-16
clean; they live in git history.

---

## Design-doc + memory-note pointers

- `docs/general_factor_graph_search.md` — the general symbolic search tier (the
  predicate-driven core; the Path-B honest negative; Phases 0+2 built).
- `mycelium/factor_graph_engine.py` (module docstring) — the general deducer and
  its byte-identical-on-KenKen contract.
- `mycelium/circuit_data.py` (module docstring) — the Boolean-circuit DAG testbed
  (where parallel deduction was measured and the radial-depth claim was refuted).
- `memory/project_kenken_property2_first_read_untestable.md` — the
  UNTESTABLE-by-restriction first read; rho_no_ceiling sign-flip; K=16 plan.
- `memory/project_v98_sudoku_validates_paradigm.md` — the six-component recipe.
- `memory/project_factor_graph_framing.md` — breathing = learned approximate BP.
- `memory/project_ode_integrator_framing.md` — the ODE-integrator identity.
- `memory/project_musical_keys_topology.md` — topology determines breathing
  rhythm (cyclic vs directional vs chain).
- `memory/project_v112b_phase1_validates_factorization.md` — per-node gating
  validated, pairwise attention bias refuted.
- `memory/project_v121_perceiver_5x_refuted.md` and
  `memory/project_v118_ablation_perceiver_diagnosis.md` — why the perceiver is
  retired.
- `memory/project_big_paper_strategy.md` — paper holding strategy.
- `memory/reference_tinygrad_am_quirks.md` — the AM-driver / JIT substrate laws.

---

## Roadmap

**Proven this arc.** One general deducer on three structurally different factor
graphs (coloring / circuits / KenKen); parallel deduction that scales
sub-linearly in depth (`K_min ≈ D/4`); a general symbolic search tier (Phases
0+2) validated on coloring / SAT / KenKen with zero general-core changes.

**The forward direction — the real frontier.** On clean verifiable CSPs,
symbolic search already wins, so the neural deducer's *generality* is the prize
to push, not its search value. The target is a problem class where **symbolic
propagation isn't enough** — soft / probabilistic / learned / NL-specified
constraints — while keeping the engine **strictly general** (no domain-specific
core code) and **minimizing the weight retraining** needed to switch tasks.
This is forward research, not a banked result. **Its chosen instantiation is the
two-phase Alternator (spec-stage — the section above; CLAUDE.md §8), entered via
the brick ladder starting at Brick-0.**

**Still open (not closed).** Bank a *powered* Property-2 (adaptive-depth)
verdict from a K=16 curriculum retrain, against the conservative bar above
(min-based instrument, settled = correct, `rho_no_ceiling` control). This is the
*harder-puzzles-settle-later* question, distinct from the refuted radial-depth
claim.

**Reserved / spec-stage (gated, additive, never on the critical path).** The
hyperbolic mask generator
([`docs/hyperbolic_mask_generator_spec.md`](docs/hyperbolic_mask_generator_spec.md))
remains spec-only: the radial-depth prize is refuted on the DAG and relaxation is
gradient-dead off the partition case, so it is pursued (if at all) only as a
strictly-additive experiment behind a flag — with it off, the forward is
byte-identical to the validated engine, so the working engine is never at risk.

**Deadline.** December 25, 2026 — MATH-500.
