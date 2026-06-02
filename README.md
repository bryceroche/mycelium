# Mycelium v4: The Breathing Transformer

A small (87M-param) iterative transformer that performs factor-graph
inference via K passes through 4 shared Pythia-410M layers.

**Author:** Bryce + Claude
**Date:** 2026-06-01 (refactored after the v98 Sudoku pivot)
**Deadline:** December 25, 2026
**Platform:** Shadow Glass (AMD 7900 XTX, 24GB) · tinygrad + AM driver · no ROCm
**Target:** MATH-500 (current empirical benchmarks: Sudoku, GSM8K via factor graphs)

> The conceptual arc that motivated the project (sine-wave breath, π-cycled
> RoPE, BirdNET-parallel heads, Controller / Notebook / LookupTable closed
> feedback loop) was largely deprecated by the v98 Sudoku pivot of 2026-05-29.
> The original vision is archived at
> [`docs/archive/vision_v1_to_v95.md`](docs/archive/vision_v1_to_v95.md) and
> the pre-pivot empirical state at
> [`docs/archive/empirical_v45_to_v95.md`](docs/archive/empirical_v45_to_v95.md).
> This README describes what the system IS, not what it was envisioned to be.

---

## 1. What this is

The breathing transformer is a small iterative model that performs
factor-graph inference. The 4 Pythia-410M L0-L3 layers are shared
across K breaths; each breath does one pass through them and accumulates
into a 1024d residual stream. Structured per-head attention masks encode
the factor topology (which cells affect which other cells). A learnable
per-breath `delta_gate` scalar controls how much each breath updates the
running state; a per-breath calibration head produces a scalar
"confidence" signal. A variant-specific codebook reads out the residual
stream as a soft distribution over digit values; per-breath weighted
cross-entropy supervision (the "ladder") makes K breaths matter.

The architecture is three-instances-of-one-design:

- **v98 Sudoku** — 81 cells × 9 digits, K=20 breaths, row/col/box
  attention masks. **97.65% cell / 79.0% puzzle accuracy on easy.**
- **v100-v107 (factor graphs, number-level)** — variable number of cells
  with values in [0, 99] or [0, 199], K=10 breaths, per-op-type
  attention masks + topological staging. v107 (hybrid 200) reaches
  **24.2% on GSM8K** after Haiku-distilled factor-graph extraction.
- **v105 family (factor graphs, digit-level)** — same as v100 but with
  per-position digit codebooks (5 positions × 10 digits) and RoPE on
  the digit axis. Currently at ~5% plateau; mean-field collapse
  diagnosed via linear probe; Y-soft adjacent-only attention fix is
  under test.

All three share the same backbone, training loop structure, and
"iterative prefill in a 1024d residual stream" execution model. The
legacy machinery from the original vision (`Controller`, `Notebook`,
`LookupTable`, sine-modulated temperature, π-cycled per-breath RoPE)
still lives in `mycelium/breathing.py` so the legacy import surface stays
clean, but no current trainer calls those code paths.

For the original conceptual vision, see
[`docs/archive/vision_v1_to_v95.md`](docs/archive/vision_v1_to_v95.md).
For the paper draft, see [`paper/outline.md`](paper/outline.md).
For the natural-language → factor-graph parser (Phase 1) spec, see
[`docs/phase1_nl_parser_spec.md`](docs/phase1_nl_parser_spec.md).

---

## 2. The architecture, precisely

### 2.1 Per-breath forward pass

Each of K breaths runs the same code on the same shared weights:

1. **Embed.** If breath 0, the input (Sudoku grid / factor graph nodes)
   is embedded into `x ∈ ℝ^(B×T×1024)`. Otherwise `x = x_prev_breath`.
2. **Add per-breath marker.** A small additive embedding
   `breath_embed[k]` (orthogonal across breaths) is added to the
   residual, separating per-breath gradients without forcing
   specialization.
3. **4 Pythia L0-L3 layers (shared across breaths).** Attention uses
   the architecture's structured per-head mask
   (§2.2). Within each layer, head-dim 64, FFN intermediate 4096,
   standard RMSNorm + RoPE for token positions. **No π-cycled RoPE
   across breaths** — the legacy diversity mechanism is replaced by the
   per-head structural mask.
4. **(v105.1.2+ only) IB semantic codebook attention.** 32-entry
   semantic codebook from IB clustering on Pythia embeddings; LoRA-init
   gate so step-0 forward equals no-codebook baseline.
5. **(v105.1.2+ only) Projection waist (1024 → 512 → 1024).** Explicit
   lossy "Quantize" step in the codec framing. LoRA-init gate so the
   gate lifts gradually during training rather than disrupting warm-start.
6. **Delta gate residual update.**
   `x_{k+1} = x_pre + sigmoid(gate_k) * (x_post - x_pre)`. The gates
   are a single learnable `(K_max,)` tensor — static across problems,
   not controller-emitted.
7. **Per-breath layernorm + codebook readout.** The codebook is variant-
   specific (§2.2). Output is a soft distribution over digit values.
8. **Per-breath calibration head.** Scalar per-breath, trained against
   the detached argmax-correctness target. Conceptually the
   Dopri5-style error estimator for adaptive K; not yet used for early
   stopping in the trainer (it's training-only signal so far).

The K breaths are unrolled into a single tinygrad JIT graph at compile
time. The per-breath weighted CE
(`loss = Σ_k (1 + k/(K-1)) * CE(logits_k, target)`) is the "ladder"
that makes the K axis do work.

### 2.2 The architectural lever per problem topology

The "instrument" (Pythia + iterated layers + masked attention) is
universal. What differs across problem classes is the topology of the
structured attention mask, the readout codebook, and K. Together with
the per-breath ladder loss, these three knobs specialize the architecture
to a problem topology (the "musical keys" framing — see §2.3).

| Problem topology | Attention mask | Codebook | K |
|---|---|---|---|
| Cyclic (Sudoku) | 5 heads each for row / col / box (per-cell AllDifferent cliques) + 1 global head | 9-digit single-cell | 20 |
| Tree DAG (arithmetic) | Per-op-type masks (4 ops × 4 heads) + topological staging (breath k sees up to depth k) | 100-bin / 200-bin number-level | 10 |
| Chain (digit decomposition) | Y-soft adjacent-only within-variable attention (testing as of 2026-06-01) | 10-digit × 5 positions, RoPE on the digit axis | 8 |

**Sudoku (cyclic key).** The 27 AllDifferent constraints are baked
into the attention masks: 5 attention heads each restrict attention to
row-mates / column-mates / box-mates of the current cell, plus 1
global head. K=20 because loopy BP on a 27-constraint factor graph
needs that many iterations to converge for hard puzzles. See
`mycelium/sudoku.py`.

**Number-level factor graph (directional key).** Topological staging
mask GROWS across breaths: at breath 0 the model only "sees" observed
leaves, by breath k=depth the full DAG is visible. The per-op-type
masks specialize 4 heads each to add / sub / mul / div factor types.
See `mycelium/factor_graph_v100.py` through `factor_graph_v107.py`.

**Digit-level factor graph (chain key).** Same as number-level but
the readout is per-position (5 positions × 10 digits each). The
position axis carries its own RoPE (right-aligned: ones-digit always
RoPE position 0). The Y-soft fix (testing) restricts within-variable
attention to adjacent positions only, breaking the mean-field collapse
where all positions of one variable averaged together. See
`mycelium/factor_graph_v105_1_2.py` and `factor_graph_v105_4.py`.

### 2.3 The three conceptual frameworks (all valid views)

The same execution is described by three different vocabularies:

- **JPEG codec.** Each breath is a learned compression codec:
  Transform (basis rotation via attention) → Quantize (waist
  projection, v105+) → Encode (delta_gate carries survivors to next
  breath) → Psychoacoustic model (per-breath CE is the learned model
  of what to preserve).
- **ODE integrator.** `dx/dt = -∇E(x)` with the factor graph energy
  E. 4 transformer layers per breath = 4 RK4-like stages (each layer
  is one gradient estimate; residual stream is the running sum); K
  breaths = K integration steps; delta_gate = adaptive step size;
  calibration head = Dopri5-style error estimator for adaptive K.
- **Approximate belief propagation.** Factor graph inference via
  message passing; the model learns the messages. Per-head attention
  masks = factor topology; K breaths = K loopy-BP rounds; per-breath
  weighted CE = monotonic belief refinement. The signature is
  geometric energy decay (Sudoku: 21.0 → 0.71 over K=1 to K=20 at
  rate ~0.5×/3 breaths) and the 7-orders-of-magnitude correlation
  between cell accuracy and puzzle accuracy (joint MAP, not
  independent marginals).

These are not three theories; they are three vocabularies for the same
mathematical object — **a learned approximate iterative solver for
joint MAP inference on a factor graph, structured as an ODE integrator
with energy-based dynamics**. Foundation: attention IS one Hopfield
energy descent step (Ramsauer et al., 2020). See
`memory/project_ode_integrator_framing.md` and `paper/outline.md`.

### 2.4 The musical keys: topology determines breathing rhythm

The architecture's "instrument" is universal but each problem class is
in a different topological **key** that requires its own breathing
**rhythm**:

| Key | Topology | Rhythm |
|---|---|---|
| Cyclic | Loopy graph, symmetric AllDiff cliques | Symmetric per-head masks (Sudoku) |
| Directional | Tree, asymmetric functional constraints | Topological staging (v100+) |
| Chain | Sequential dependencies within a variable | Adjacent-only attention (v105 Y-soft) |
| Cadence | Forward + backward cycles | Alternating direction (untested) |

v98's Sudoku success and v99's negative result on arithmetic DAGs are
the same finding viewed from two angles. Rotation breathing works for
cyclic key; it fails on directional key, where the right rhythm is
staging. Same instrument, different scale. See
`memory/project_musical_keys_topology.md`.

### 2.5 The two-phase system

Comprehension (NL → factor graph) and inference (factor graph → answer)
are different computational regimes that should use different model sizes:

```
┌──────────────────────────────────────────────────────────┐
│ Phase 1: COMPREHENSION (large model, one-shot)            │
│ "Janet has 16 eggs..."                                    │
│         ↓                                                 │
│ NL Parser (Haiku / fine-tuned T5-small / DistilBERT)      │
│         ↓                                                 │
│ Factor Graph: variables, factors, observed, query         │
└────────────────────┬─────────────────────────────────────┘
                     ↓
┌────────────────────┴─────────────────────────────────────┐
│ Phase 2: INFERENCE (small model, iterative, on device)    │
│ Factor Graph                                              │
│         ↓                                                 │
│ Breathing Transformer (87M, 377MB)                        │
│ dx/dt = -∇E(x, constraints)                               │
│ K breaths of ODE integration                              │
│         ↓                                                 │
│ Converged variable assignments → answer                   │
└──────────────────────────────────────────────────────────┘
```

The 410M GSM8K failure was Phase 1 being undersized for comprehension.
v98 Sudoku worked because Phase 1 is trivial (grid IS factor graph)
and the architecture is sized correctly for Phase 2. The breathing
transformer IS Phase 2. Building Phase 1 (NL parser) is the path to
GSM8K, not redesigning the inference engine. See
`docs/phase1_nl_parser_spec.md`.

---

## 3. Empirical status (current)

```
v98 Sudoku:         97.65% cell / 79.0% puzzle on easy (validated)
                    83.33% cell / 6.5% puzzle on medium
                    76.16% cell / 0.0% puzzle on hard
v100-v107 (number): 40-48% cell_acc on synthetic [0,99] factor graphs
v107 (hybrid 200):  24.2% cell_acc on GSM8K factor graphs
v105 family (digit): 12 attempts hit ~5% plateau. Linear probe
                    diagnostic (Jun 1) identified mean-field collapse;
                    Y-soft attention mask fix currently testing.

Phase 1 (NL parser): designed (`mycelium/phase1_classifier.py`); not
                    yet trained.
```

### 3.1 The v98 Sudoku breakthrough (May 29)

**The most important measurement in the project.** K-sweep at K ∈ {1,
3, 5, 8, 12, 15, 18, 20}, n=200 per difficulty:

```
K    easy puzzle  easy cell  medium puzzle  medium cell  avg energy (easy)
 1     0.0%        82.1%       0.0%           69.1%       21.0
 3    10.0%        91.9%       0.0%           76.6%        7.2
 5    33.5%        94.8%       0.0%           79.8%        3.5
 8    56.0%        96.4%       1.0%           81.3%        1.8
12    72.5%        97.3%       2.5%           82.4%        1.1
15    75.0%        97.5%       5.5%           82.8%        0.86
18    77.0%        97.6%       6.0%           83.2%        0.75
20    79.0%        97.65%      6.5%           83.33%       0.71
```

The constraint energy decays **geometrically at rate ~0.5× per ~3 K** —
the mathematical signature of loopy BP convergence on a factor graph
with cycles.

The **7-orders-of-magnitude correlation** at medium difficulty:
independent-cell prediction baseline is 0.833^81 ≈ 3×10⁻⁷; observed
puzzle accuracy is 6.5%; ratio is 2×10⁵ above independent baseline. The
model is solving structures (joint MAP), not classifying independent
cells.

### 3.2 The journey and key findings

- **v98 Sudoku validated the factor-graph framing.** After 17 GSM8K
  architectural variants (v82-v97) plateaued at 0-1.7%, the strategic
  pivot to Sudoku revealed the breathing transformer is approximate
  joint MAP inference on a factor graph. Six-component recipe
  documented in `memory/project_v98_sudoku_validates_paradigm.md`.
- **v99 was the directional-key failure mode.** Same architecture on
  arithmetic DAGs converged to a uniform-distribution fixed point —
  9% accuracy flat across K=1 to K=10. The moment-matching constraint
  energy has a trivial low-energy attractor. Architecture works
  mechanically; the breathing rhythm was wrong for that topology.
- **v100 fixed it with topological staging.** Mask GROWS across
  breaths; later breaths see deeper DAG nodes; aligned init
  (`state_embed[i] = digit_codebook[i]`) was the v98 unlock v99
  missed. Result: 40-48% on synthetic factor graphs (number-level).
- **v107 reached GSM8K.** Hybrid 200-bin codebook (handles up to
  4-digit numbers); 24.2% cell accuracy on GSM8K factor graphs
  (Phase 2 only, oracle factor graphs from Haiku distillation).
- **v105 family hit a wall on digits.** 12 attempts at per-position
  digit prediction stalled at ~5%. Linear probe diagnostic (Jun 1)
  identified mean-field collapse — across positions of one variable
  the hidden states become identical, so different position
  predictions are impossible. Y-soft adjacent-only attention is the
  fix being tested.

See `paper/outline.md` for the full empirical writeup with figures.

---

## 4. Editing rules (load-bearing)

These rules are still load-bearing in current code:

- **No mid-breath token generation.** Reasoning stays in the 1024d
  residual stream. Generation (when present, e.g. v80-era AR decode
  variants) happens once at the end. Empirically: "had had had"
  collapse within 2 autoregressive loops if violated.
- **Diversity must be structural, not learned.** v98 row/col/box
  masks, v100 topological staging masks, v105 per-digit RoPE are all
  geometric. Every v1-v3 learned diversity mechanism (scales, soft
  tokens, codebooks, fingerprints) collapsed to constant within one
  epoch.
- **Digit-spaced for arithmetic** (partial). v98 uses single-cell
  digits; v100+ uses number bins or per-digit codebooks. Whole-number
  BPE tokens force memorization. The v105 family explores per-position
  digit prediction; this is still load-bearing but the position layout
  matters (MSD-first + right-aligned RoPE + valid_mask, see
  `memory/feedback_digit_vs_number_prediction.md`).
- **KV cache invariants.** Size to actual seq length; pad eval
  batches to fixed batch_size so compiled graphs match; compile once
  during first eval, replay for the rest of the run.
- **Bryce wants root-cause perf fixes**, not workarounds, when perf is
  the bottleneck.
- **Gradient separation rule (legacy, currently moot).** If a
  Controller-like decision module is ever reintroduced, its gradient
  must never reach transformer weights. Use a separate optimizer and
  verify with the parameter-change smoke. No Controller is in the
  current code path, so this is dormant.

---

## 5. Specifications

### Initialization

Pythia-410M layers 0-3 (attention + FFN weights), token embeddings
(50304 × 1024), and untied output head. In v98+ all 4 layers share the
SAME weights across all K breaths (no phase-specific copies — that
was a v1-v95 design and was dropped in the pivot).

### Model dimensions

Hidden 1024, 16 heads × head_dim 64, FFN 4096, vocab 50304, max seq 512.
4 transformer layers in the iterated stack (shared across breaths).

K varies per architecture: Sudoku=20, v100=10, v105=8, v107=10. No
global ceiling — JIT capacity on the AMD 7900 XTX is the practical
limit (~20 breaths in the current trainer graphs).

### Parameters

- Sudoku: ~87M total (35.7M shared transformer processing + 51.5M
  token embeddings; no separate output head, codebook is 9 digits).
- v100/v107 (number-level): ~87M same as Sudoku, with a 100- or 200-bin
  codebook in place of digit values.
- v105.1.2+ (digit-level + waist + IB codebook): adds ~1M from the
  1024 → 512 → 1024 LoRA-init waist and ~1.6M from the 32 × 1024
  semantic codebook. Both gated by zero-init scalars so step 0 forward
  equals the no-extras baseline.

### Memory

~5GB total mixed precision on a 7900 XTX, ~19GB headroom. Batch size
32-64 at FIXED_LEN=160 for Sudoku; 8-16 at FIXED_LEN=256 for factor
graphs. KV cache sized to actual sequence length.

### Inference engine

JIT-fused KV cache, per-batch position tracking, fixed-batch eval that
compiles once and reuses across checkpoints. 42.8× over the original
uncached path on the L3 eval set; bit-for-bit identical outputs. See
`memory/project_inference_engine.md`.

### Platform

AMD Radeon RX 7900 XTX (24GB GDDR6, ~120 TFLOPS FP16). Tinygrad
framework with AM custom userspace driver (working since 2026-05-11 —
Secure Boot off + `vm.compact_unevictable_allowed=0`). No ROCm, no
CUDA, no PyTorch. Ubuntu 24.04.

---

## 6. Repository structure

```
mycelium/
  breathing.py                — Pythia L0-L3 backbone, BreathingTransformer
                                 class (contains LEGACY Controller / Notebook /
                                 LookupTable that v98+ doesn't call — preserved
                                 for backward import compatibility only)
  controller.py               — LEGACY: Notebook + Controller (v1-v95 era)
  lookup_table.py             — LEGACY: 16×1024 prime-op matcher (v1-v95 era)
  sudoku.py + sudoku_data.py  — v98 Sudoku architecture and data
  factor_graph_v100..v107.py  — number-level factor graph variants
  factor_graph_v105_1_2.py    — digit-level + AR conditioning
                                 (Bryce's "die on this hill")
  factor_graph_v105_3.py      — LSD-first refactor of v105.1.2
  factor_graph_v105_4.py      — hierarchical codebooks (mag head,
                                 per-pos digits, hier IB)
  phase1_classifier.py        — DistilBERT-based NL parser (Phase 1)
  pythia.py                   — Pythia weight loader

scripts/
  v98_sudoku_{prod,smoke}.sh  — Sudoku train + smoke
  sudoku_train.py             — Sudoku trainer
  eval_v98_sudoku.py          — Sudoku eval
  v98_k_sweep.sh              — K-sweep for figure 2 of the paper
  v100..v107_factor_graph_*   — number-level factor graph train/eval
  v105_*_factor_graph_*       — digit-level factor graph variants
  v105_1_2_v2_*               — current digit-level prod runs
                                 (lateral / number-MSE / Fourier / log-uniform / AR-MSD)
  phase1_*                    — Phase 1 build, train, eval
  diag_*                      — diagnostics:
                                  diag_v98_per_breath_convergence.py
                                  diag_v105_*_per_position_acc.py
                                  diag_v105_4_linear_probe.py
                                  diag_ib_clustering.py + diag_ib_tree_export.py

docs/
  archive/                    — pre-v98 historical content
    vision_v1_to_v95.md       — original architecture vision (deprecated)
    empirical_v45_to_v95.md   — GSM8K WaistController era results
    closed_loop_seven_components.md — pre-pivot CLAUDE.md §2
  phase1_nl_parser_spec.md    — Phase 1 design doc

paper/
  outline.md                  — current paper draft
                                 ("The Shape of Thought: Iterative
                                  Reasoning Through Learned Energy
                                  Descent on Factor Graphs")
  figures/                    — paper figures
  references.bib              — citations

.cache/                       — checkpoints, data, IB centroids
                                 (gitignored). Key files:
  sudoku_ckpts/v98_prod_final.safetensors      — v98 final
  fg_v107_ckpts/v107_prod_step1000.safetensors — v107 GSM8K champion
  v98_ksweep/K*.log                            — K-sweep results
```

---

## 7. Roadmap

**End-of-month goal (June 2026).** Workshop-tier paper draft submission:
v98 Sudoku result, v100/v107 factor-graph generalization, energy decay
characterization, three-vocabulary framing (codec / ODE / BP). Held
for 3-4 weeks to add Phase 1 (NL parser) + IB-anchored Phase 2
codebook + end-to-end GSM8K result before submission. See
`memory/project_big_paper_strategy.md`.

**Open architectural questions:**

- Does Y-soft adjacent-only within-variable attention fix the
  mean-field collapse in v105.1.2 v2? (testing as of 2026-06-01)
- Does the explicit Quantize step (v105.1.2 v2 waist) lift accuracy
  vs the no-waist baseline (v100)? Codec hypothesis under test.
- Can a small Phase 1 classifier (~60-90M params) parse GSM8K NL to
  factor graphs at >85% accuracy? T5-small vs DistilBERT spec at
  `docs/phase1_nl_parser_spec.md`.
- v106 PUCT search on v107: when BP is uncertain at high-entropy
  positions, branch the digit codebook tree under PUCT scoring with
  calibration as the value signal. Code at
  `mycelium/factor_graph_v106.py`; RUN DEFERRED until v105 BP produces
  useful per-position digit distributions. Design memo:
  `memory/project_v106_mcts_design.md`.

**Deadline.** December 25, 2026 — MATH-500 (current bench is
GSM8K-via-factor-graphs at the v107 24% level).

---

## Appendix: Conceptual archive

For the project's original conceptual vision (sine-wave breath,
π-cycled per-head RoPE, BirdNET-parallel heads, Controller / Notebook
/ LookupTable closed feedback loop), see:

- [`docs/archive/vision_v1_to_v95.md`](docs/archive/vision_v1_to_v95.md) — original architecture vision (deprecated)
- [`docs/archive/empirical_v45_to_v95.md`](docs/archive/empirical_v45_to_v95.md) — GSM8K WaistController era results
- [`docs/archive/closed_loop_seven_components.md`](docs/archive/closed_loop_seven_components.md) — the seven-component framing

These describe what the project WAS, not what it IS. They are preserved
as design history.
