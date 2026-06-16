# Mycelium: The Breathing Transformer

A small (~87M-param) iterative transformer that solves constraint problems
by **breathing** — K passes through 4 shared Pythia-410M layers, each pass
one round of factor-graph inference in a 1024d residual stream.

**Author:** Bryce + Claude
**Date:** 2026-06-16
**Deadline:** December 25, 2026
**Platform:** Shadow Glass (AMD 7900 XTX, 24GB) · tinygrad + AM driver · no ROCm
**Target:** MATH-500 (current empirical bench: KenKen CSP / Property-2 adaptive-depth)

> **The authoritative current brief is [`CLAUDE.md`](CLAUDE.md)** (direction,
> specs, editing rules). The active next-direction design note is
> [`docs/hyperbolic_mask_generator_spec.md`](docs/hyperbolic_mask_generator_spec.md).
> The pre-v98 conceptual arc (sine-wave breath, π-cycled RoPE, Controller /
> Notebook / LookupTable loop) is archived at
> [`docs/archive/`](docs/archive/). The full v100–v300 lineage is preserved in
> git history (removed from the working tree in the Jun-16 clean).

---

## The vision: a three-tier architecture around a learned Poincaré ball

The project is building toward **one architecture in three tiers**. The bottom
tier is built and validated; the upper two are the active research program.

```
┌──────────────────────────────────────────────────────────────────────┐
│ TIER 1 — STRUCTURAL MAPPING  (continuous topology embedding)           │
│   A problem's geometry + dependency-logic → continuous coordinates     │
│   in a learned Poincaré (hyperbolic) ball.                             │
│   STATUS: SPEC-STAGE — not built or tested.                            │
└───────────────────────────────┬────────────────────────────────────────┘
                                │  coordinates
┌───────────────────────────────┴────────────────────────────────────────┐
│ TIER 2 — THE COMPILER  (the "virtual factor graph")                    │
│   THE HYPERBOLIC MASK GENERATOR. Compiles the Tier-1 coordinates into   │
│   the attention masks the executor consumes — instead of hardwiring     │
│   them. A differentiable virtual machine.                              │
│   STATUS: SPEC-STAGE — foothold NOT yet built or tested.               │
│   docs/hyperbolic_mask_generator_spec.md                               │
└───────────────────────────────┬────────────────────────────────────────┘
                                │  attention masks
┌───────────────────────────────┴────────────────────────────────────────┐
│ TIER 3 — THE CORE EXECUTOR  (the validated v98 KenKen breather)        │
│   Pure iterative deduction on whatever masks Tier 2 provides. Shared    │
│   Pythia-410M L0-L3, K=16 breaths, per-breath delta_gate + calibration  │
│   head, value-codebook readout, per-breath weighted-CE ladder,          │
│   gold-free convergence instrument.                                    │
│   STATUS: ★ VALIDATED AND LIVE ★  (the Property-2 K=16 curriculum run   │
│           is training now). mycelium/kenken.py                         │
└──────────────────────────────────────────────────────────────────────┘
```

**What is built versus spec'd — read this carefully, it is the central
honesty discipline of the project.** ONLY **Tier 3** is validated, built, and
live. **Tiers 1 and 2** — the Poincaré embedding and the hyperbolic mask
generator — are the **active research program**: spec'd in
[`docs/hyperbolic_mask_generator_spec.md`](docs/hyperbolic_mask_generator_spec.md),
with the foothold **not yet built or tested**. The three-tier picture is "the
architecture we are building toward," with Tier 3 done and Tiers 1–2 as the
next, spec-stage work. Nothing in this document states or implies that Tiers
1–2 are built, working, or validated.

### Why hyperbolic (Tier 1)

Problem topologies are **hierarchical**: a cell sits inside a cage inside a
board; a DAG sub-computation nests inside a larger computation. Hyperbolic
space embeds hierarchy with low distortion — trees fit in a Poincaré ball
almost isometrically where they crowd badly in Euclidean space. The radial
coordinate becomes an **abstraction level**: near the origin = abstract /
high-level, near the boundary = concrete / leaf-level. Mapping a problem's
structure to a point cloud in this ball replaces rigid one-hot problem IDs
with a **continuous structural signature**, which is what makes
interpolation and transfer across problem classes even thinkable.

### Why the generator is buildable where the perceiver wasn't (Tier 2)

Tier 2 generates each attention mask from the Tier-1 coordinates. The
mechanism (per
[`docs/hyperbolic_mask_generator_spec.md`](docs/hyperbolic_mask_generator_spec.md)):

- **One coordinate field PER RELATION** (row / col / cage). This is not a
  convenience — it is a structural necessity. A single metric space cannot
  hold row ∪ col ∪ cage at once: cell A=(0,0) must be close to B=(0,1) (same
  row) and to C=(1,0) (same col), yet B and C must be far — and the triangle
  inequality forbids it (`d(B,C) ≤ d(A,B)+d(A,C)`). v98 already resolves this
  with separate head-groups; the geometry mirrors that exactly.
- **Closed-form, anchored at t=0 to reproduce the v98 hard mask exactly.**
  Each relation's groups become max-separated anchors on a shell of the ball;
  the bias is `bias = -softplus(α·(d_hyp − r))`, with `r` and `α` *dialed*
  (not fitted) so within-group ≈ 0 and between-group ≈ −1e4. Match to ~1e-3.
- **Then RELAXED.** The coordinates unfreeze and co-train with the executor.

This anchor discipline is **why Tier 2 is buildable where the perceiver was
not**. The hyperbolic generator initializes to the validated hard mask and
then relaxes — **learning relaxes a known geometry, it never discovers one
from random.** That neutralizes the attention-bootstrap wall (task gradient
through softmax is too weak to grow a new attention pathway from scratch —
see the editing rules) that killed every perceiver attempt. The earlier
"Mycelium blueprint" placed a *perceiver* as the executor; **that perceiver
is retired** (see below), and the executor is now the validated v98 breather.
Do not confuse this three-tier architecture with the retired perceiver-core.

### The deep prize: the geodesic engine

The payoff that makes the hyperbolic detour worth taking is a single chain of
identities:

> **deduction-depth ↔ radial traversal ↔ breath-count.**

If the mask radius `r` can move inward as breaths progress, the breath cycle
becomes a **geodesic engine**. The "exhale" (the waist projection) drives the
representation *inward* toward the origin — toward abstraction — which
**auto-widens the attention horizon** (a smaller `r` lets a cell attend
further). The "inhale" descends back *outward* to project onto concrete local
nodes. Each breath is then a literal radial traversal of the hierarchy, and
the number of breaths a problem needs is read off the depth it must climb.

This is pursued in **phases, deliberately NOT bundled** (per the spec's
roadmap):

1. **Foothold — static global `r` (per relation).** Prove the geometry can
   *hold* the mask AND that ONE shared coordinate field generalizes across
   N=5/6/7. One variable only.
2. **The climb — monotonic `r_k` per breath.** A non-decreasing radius
   schedule: the continuous form of the v100 topological-staging mask. Read
   off whether the learned schedule accelerates on deeper puzzles.
3. **The ultimate — `r = f(|z|)`.** The horizon becomes a function of radial
   position: the waist's inward climb auto-widens the horizon and the climb
   *is* the expansion. Earn it; do not start here (that is the perceiver
   bootstrap mistake repeated).

**Testbed caveat (keep this).** KenKen is **flat** — lateral row/col/cage
cliques, no nested DAG. The radial-depth bloom that Tiers 1–2 are designed to
exploit is richest on **hierarchical** problems (e.g. GSM8K DAGs). So the
N=5/6/7 foothold cleanly proves the static geometry, but a *muted* KenKen
radial signal is the geometry faithfully reflecting a flat problem — NOT a
manifold failure. Reserve the radial-depth verdict for a DAG testbed.

---

## Tier 3 — the validated executor (the breathing transformer)

Everything below is built, validated, and live. Tier 3 is the foundation the
upper tiers extend; it is the v98 executor — **not a perceiver**.

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
   (the "exhale" Tier 2's geodesic roadmap will eventually drive radially).
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
- **K=16 breaths** (the live curriculum run).
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
| Directional | Tree / DAG, asymmetric functional constraints | Topological staging (the climb) |
| Chain | Sequential dependencies within a variable | Adjacent-only attention |

Same instrument, different scale. Sudoku/KenKen are the cyclic key, where
symmetric rotational breathing works. Tier 2's monotonic-`r_k` climb is the
**continuous form of the directional-key staging mask** — which is exactly why
a hierarchical DAG testbed is where the geometry will bloom.

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

The **live K=16 curriculum run is training now.** Early peek (underpowered,
not a verdict): the settled set is **deepening**, and **N=5 settled rho = 0.72
with rho_no_ceiling = 0.67** — encouraging and *consistent*, but underpowered
against the bar. The analyzer was patched so a restriction-of-range no longer
reports as a win — it reports **UNTESTABLE** — and `rho_no_ceiling` is now a
**required companion control** on every read.

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
  not. This is precisely why Tier 2 **anchors to the validated hard mask and
  relaxes** rather than learning a mask from scratch — and why the perceiver,
  which had no such anchor, hit a gradient void.
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

**K.** KenKen K=16 (live curriculum); v98 Sudoku used K=20. No global ceiling —
AMD 7900 XTX JIT capacity (~20 breaths) is the practical limit.

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

## What we carry forward — and what we left behind

**Forward (the validated Tier-3 design):**

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

**Left behind:**

- **The perceiver — RETIRED.** Refuted 5× as an add-on (v118–v121), and the
  v300 perceiver-CORE failed flat at chance. The earlier Mycelium blueprint
  used a perceiver as the executor; we **replaced it with the validated v98
  executor**. Tier 3 is the v98 breather, not a perceiver — and the hyperbolic
  generator's anchor-and-relax discipline (Tier 2) is precisely the thing the
  perceiver lacked. See `memory/project_v121_perceiver_5x_refuted.md` and
  `memory/project_v118_ablation_perceiver_diagnosis.md`.
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
  kenken.py            — the live Tier-3 executor (v98 KenKen breather)
  pythia.py            — Pythia weight loader
  breathing.py         — Pythia L0-L3 backbone + LEGACY Controller / Notebook /
                          LookupTable (import-compat only; no trainer calls it)

scripts/
  kenken_train.py             — KenKen trainer (the live K=16 curriculum run)
  build_kenken_data.py        — KenKen puzzle generator
  analyze_kenken_property2.py — the gold-free Property-2 convergence read

docs/
  hyperbolic_mask_generator_spec.md — ★ the active Tier-1/Tier-2 design note ★
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

## Memory-note pointers

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

**Live now.** The Property-2 K=16 curriculum run (Tier 3). The read is gated on
the bar above; the first read was UNTESTABLE-by-restriction and the K=16 run is
the powered retry.

**Active research program (Tiers 1–2).** The hyperbolic mask generator, per
[`docs/hyperbolic_mask_generator_spec.md`](docs/hyperbolic_mask_generator_spec.md),
in deliberately-unbundled phases:

1. **Foothold** — frozen, calibrated, per-relation fields; replication sanity
   (byte-match the boolean mask) then N=5/6/7 generalization (does ONE field
   serve all three N). Not yet built or tested.
2. **The climb** — monotonic `r_k` per breath (continuous topological staging).
3. **The ultimate** — `r = f(|z|)`, the geodesic engine, on a hierarchical DAG
   testbed where the radial-depth bloom can actually appear.

Each phase is **strictly additive and gated behind a banked result**: with
`KENKEN_HYP_MASK` off, the forward is byte-identical to the validated executor,
so the working engine is never at risk — the geometry only ships if it beats
the frozen-hard-mask baseline.

**Deadline.** December 25, 2026 — MATH-500.
