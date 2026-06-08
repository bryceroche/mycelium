# Mycelium v4: The Breathing Transformer — Agent Brief

**Author:** Bryce + Claude · **Deadline:** Dec 25, 2026 · **Target:** MATH-500
**Platform:** Shadow Glass (AMD 7900 XTX, 24GB) · tinygrad + AM driver · no ROCm

For the conceptual writeup see `README.md`. For the paper see `paper/outline.md`.
For the pre-v98 historical vision see `docs/archive/`.

---

## 1. The architecture in one paragraph (current)

A small iterative transformer (4 Pythia-410M L0-L3 layers SHARED across
all K breaths, h=1024, 16 heads, ~87M params total) that performs
factor-graph inference by K passes through the same weights. Each
breath: add a per-breath additive marker → 4-layer transformer with a
structured per-head attention mask encoding the factor topology → an
optional projection waist (v105+ codec families) and IB semantic
codebook → a learnable per-breath `delta_gate` residual blend → per-
breath layernorm + variant codebook readout → per-breath calibration
head. K breaths are JIT-unrolled into a single graph. Training: per-
breath weighted CE (`loss = Σ_k (1 + k/(K-1)) * CE(logits_k, target)`),
the "ladder" that makes K matter. Three variants of one design: v98
Sudoku (K=20, 9-digit cell codebook, row/col/box masks); v100-v107
(number-level, K=10, per-op-type masks + topological staging); v105
family (per-position digit codebook with right-aligned RoPE).

---

## 2. Components, as-built (v98+)

Three architecture families share this core pattern. Entry points:
`sudoku_breathing_forward` (sudoku.py), `fg_breathing_forward_v100`
(factor_graph_v100.py), `fg_breathing_forward_v105_1_2`
(factor_graph_v105_1_2.py).

| Component | What it does | Implementation |
|---|---|---|
| **Iterative shared-weight prefill** | K passes through Pythia L0-L3, SAME weights every breath, residual stream stays at 1024d throughout | `sudoku.py`, `factor_graph_v100.py`, `factor_graph_v105_1_2.py` |
| **Per-breath additive embedding** | Orthogonal markers added to residual per breath; replaces the "breath_idx schedule" of v9-v11 era | `breath_embed` tensor in each forward |
| **Structured per-head attention masks** | Sudoku: 5 row + 5 col + 5 box + 1 global. v100/v107: per-op-type masks (4 ops × 4 heads). The actual "alternation" mechanism — replaces π-cycled RoPE | `_build_*_masks` helpers per file |
| **Topological staging mask (v100/v107)** | Per-breath visibility expands depth-by-depth across the DAG; later breaths see deeper nodes | `staging_mask` in fg loaders |
| **Per-breath delta_gate** | Learnable convex residual blend: `x = x_pre + gate_k * (h - x_pre)`. Static `(K_max,)` tensor, NOT controller-emitted | `model.*_delta_gate` |
| **Per-breath calibration head** | Scalar confidence per breath, trained with detached argmax-correctness target. Conceptual hook for adaptive K (Dopri5-style error estimator); not yet used for stopping | `*_calib_head_*` weights |
| **Variant codebook (readout)** | sudoku: 9-digit value codebook. v100: 100-bin domain codebook. v107: hybrid 200-bin. v105.1.2: 10-digit codebook × 5 positions (with right-aligned RoPE on digit axis) | `*_codebook` per file |
| **IB semantic codebook + projection waist (v105.1.2+)** | 32-entry semantic codebook from IB clustering on Pythia embeddings + 1024→512→1024 LoRA-init waist. Both gated by zero-init scalars so step 0 forward is identical to no-extras baseline | `factor_graph_v105_1_2.py` |
| **Per-breath weighted CE supervision (the "ladder")** | Loss weighted by `1 + k/(K-1)` so later breaths matter more; this is what makes K matter at all | training loop in each trainer |

K varies per architecture: Sudoku=20, v100=10, v105.1.2=8, v107=10.
No global ceiling (JIT capacity on AMD 7900 XTX is the practical limit).

For LEGACY machinery (Controller, Notebook, LookupTable, sine-modulated
within-breath temperature, π-cycled per-breath RoPE) preserved in
`mycelium/breathing.py` but not called by current trainers, see
`docs/archive/closed_loop_seven_components.md`. Those modules still
import cleanly because `BreathingTransformer.__init__` instantiates
them, but no current trainer calls the methods that USE them.

---

## 3. Empirical status (current)

```
v98 Sudoku       97.65% cell / 79.0% puzzle  (.cache/sudoku_ckpts/v98_prod_final.safetensors)
v100 (number)    40.7% cell_acc  on [0,99]
v101 (number)    47.6% cell_acc  on [0,99]
v103 (number)    46.6% cell_acc  on [0,99]
v104 (number)    47.7% cell_acc  on [0,99]
v107 (hybrid)    24.2% cell_acc  on GSM8K  (.cache/fg_v107_ckpts/v107_prod_step1000.safetensors)
v105 family     ~3-5% plateau on per-position digit prediction
                 Jun 1: per-layer cos_sim probe pins collapse to Pythia L0
                        (input 0.74 → post-L0 0.90 → post-L3 0.991)
                 Jun 2: 4 anti-collapse experiments all hit the ceiling
                        v105.5 PPFFN-after/before-waist:   cos 0.999
                        v105.5 + hard block within-var:    cos 0.991
                        v105.6 per-position L0 W_in:       cos 0.999 (per-pos
                                                           weights stay cos
                                                           0.999 to each other
                                                           — no specialization)
                        v105.6 + aux distinctness loss:    distinct climbs
                                                           0.99→0.998 (CE-driven
                                                           collapse > aux pressure)
                 → readout-side AND replacement-architecture both fail
v105.9 (pooled AR digit decoder, Jun 2):
                 Drop per-position digit pressure; pool 5 tokens → cell_hidden
                 → AR digit chain. Per-breath ladder Δ=1.37 (13× target) —
                 STRONGEST breath refinement in v105 history. But val plateaus
                 at ~3-5%. Linear probe on cell_hidden at step 2000:
                 magnitude probe 61% (chance 25%), 200-bin probe 6.3% (chance
                 0.5%), R² on log(value) = -614. cell_hidden encodes scale
                 but NOT precise value. Supervision gap, not capacity gap.
v105.8 (200-bin number readout, RUNNING Jun 2):
                 Direct per-NUMBER CE on pooled cell_hidden. Tests whether
                 breathing learns precise value when given direct supervision.
                 If bin acc > 30% → v105.10 = dual readout (number CE trains
                 breathing + AR digit decoder reads it for OOD generalization).
v105.12 (Jun 3) OUTCOME B closure: prefill-isolation + Fourier digit init +
                 codebook annealing + 15K horizon. Step 5K OOD: per-digit
                 acc = 10.8% (chance), cell_acc = 0% — RESPONSIVE conditioning
                 but NOT ACCURATE. Pool readout flat 5-6% through entire
                 anneal. Per-digit OOD compositionality closed for v105 family.
v105.13 (Jun 4) wave-guide retrofit: 1500 steps of v105.12 + skip-mask on LoRA
                 correction. Pool readout flat (easy 0.000→0.008, hard 0.042→
                 0.051). L0 collapse confound: preserve channel preserves
                 nothing per-position-distinct because L0 averages upstream.
v108 (Jun 4)   v107 single-token base + 5-level tree codebook output. K-sweep
                 anomaly: pos4 hard DECREASES with K (0.191→0.132). BP fixed
                 point trades marginal pos4 acc for joint cell consistency.
                 Drop Monte Carlo framing from paper.
v109 (Jun 4)   v108 + 512d LoRA waist + alternation (waist on EVEN breaths
                 only). Step 500 cell_acc lift over v108: easy +0.037, medium
                 +0.044, HARD +0.083 (nearly tripled, 0.050 → 0.133). K-sweep
                 FLIP: v108 pos4 hard -0.059 with K → v109 +0.007 (non-
                 decreasing). Medium pos4 monotonic 0.126→0.173→0.189.
v109a (Jun 4)  Ablation: waist EVERY breath, no alternation. Also flips
                 K-sweep (-0.059 → +0.007) AND lifts cell_acc +0.011-0.037.
                 Attribution: WAIST = dynamics-fixer (K-sweep flip);
                 ALTERNATION = marginal lift on hard (+0.066 cell_acc atop
                 v109a). Three-row ablation table cleanly attributable.
v109 prod (Jun 4) 10K steps: easy 0.399, med 0.333, hard 0.293. 5.9× over
                 v108 on hard. Step 8500: all 3 simultaneously > 0.30.
v109pi (Jun 4)  v109 + per-breath Q rotation by phase k·π/K_max (Option A,
                 uniform across positions — NOT RoPE, no frequency × pos).
                 Chained smoke→cont9 (9500 effective steps from
                 v109_prod_step8500). cont9_step500 (step 9000) hit triple
                 new-high: easy 0.502, med 0.434, hard 0.337. Step 9400 hard
                 0.344 ALL-TIME HIGH on hard. Easy +0.11, med +0.09, hard
                 +0.04 over v109 baseline.
v109pi diag (Jun 5) K-sweep on cont9_step500: 9/9 SIGNALS MONOTONICALLY
                 RISE across {easy,med,hard}×{cell_acc,pos4_acc} from K=2
                 to K=8. Strongest K-sweep in project. OOD eval looked
                 spectacular (cell 0.89-0.93) but is a MEASUREMENT ARTIFACT:
                 200-bin input/output caps at 9999, OOD values clamp to bin
                 199, model trivially "predicts" 09999 = gold. Honest OOD
                 needs digit-decomposed input (v108b-style). Fair add/sub
                 IND comparison: v109pi NOT specifically better at add/sub.
v110-acc/photon/full/step1/step2/step3 (Jun 5):
                 +accumulate notebook (5.2M params), +sin² photon waist
                 gate, +Goldilocks step-size penalty (calibration-driven).
                 v110-step3 prod chain to step 9500: PROJECT ALL-TIMES.
                 Easy 0.610 (crossed 60%), med 0.509 (crossed 50%),
                 hard 0.399 (near 0.40). Calibration head transition
                 decorative → load-bearing produces strongest in-dist
                 result.
v106 PUCT search on v110-step3 (Jun 6) — ROBUST NEGATIVE.
                 8 configs swept (UCB1+eager, true PUCT+JIT, weights
                 ∈{0.5/0.3/0.2,...,0.0/1.0/0.0}, c∈{0.5,1.5,2.5},
                 rollouts ∈{30,100}). ALL regress vs BP-only by
                 0.006-0.012. JIT delivered 14.8× (23.5s → 1.59s/puzzle)
                 but BP makes confidently-wrong locally-consistent fixed
                 points on hard; search amplifies miscalibration.
MC-BP inference (Jun 6): noise on x at start of breath, N=16 forwards,
                 average tree_logits. noise=0.01 on cont8_step1000:
                 hard 0.3761 → 0.3823 (Δ=+0.0061, 2 correct flips / 0
                 wrong on 50-puzzle subset). Mirror image of PUCT's
                 -0.0061. Independence WAS the missing piece.
SBP training (Jun 6) — HEADLINE TRAINING-TIME WIN.
                 50/50 alternating noise=randn*0.02 on x at start of
                 breath during training, 500 steps from cont8_step1000.
                 Hard 0.3761 → 0.3914 (Δ=+0.0153, 2.5× the MC-BP signal,
                 ZERO inference overhead). MC-BP averaging adds nothing
                 on SBP-trained ckpt (model is too sharp for alt fixed
                 point search). Mechanism = denoising score matching
                 (Vincent 2011) at the residual-stream injection point.
                 12-min training, 0.14s/puzzle inference. SBP is the
                 paper's training-time row.
fb arc (Jun 6) tested 3 tree-logits feedback variants (softmax-soft,
                 fb+SBP, Gumbel-ST discrete). ALL underperform SBP
                 alone on hard cell_acc. Path C (fb+SBP) refutes the
                 synergy hypothesis: fb provides a noise-immune channel
                 that partially cancels SBP's attractor sharpening.
                 Path D (Gumbel) produces the BIGGEST MC-BP signal of
                 the arc (+0.0153 at noise=0.02) but from a worse det
                 baseline.
eg (Jun 6) — entropy-controlled waist gate. Modulates the photon gate
                 by tanh(entropy_scale * mean tree_logits entropy of
                 previous breath). Result: hurts hard cell_acc by
                 -0.015 BUT lifts EASY QUERY by +0.12 (0.26 → 0.38,
                 0.40 with MC-BP). Two different winners on two
                 different metrics: SBP for hard cell_acc, eg for easy
                 query_acc.
v111 (Jun 6) coherent-photon: sin² rotation envelope + sin² waist
                 envelope + per-dim modulation. Warm-start from
                 cont8_step1000 FAILED — rotation envelope change too
                 disruptive; loss oscillated 0.5-0.7 across 1500 steps
                 then destabilized to 0.98 at step 1550. Cold-start
                 required (deferred).
v112 (Jun 6) dual-notebook (COMMIT writes on even breaths, PROPAGATION
                 writes on odd; both read every breath). Cold-start to
                 step 3000: hard 0.073, easy 0.048 — too early to
                 evaluate; mechanism works (both W_o growing parallel).
v112b Phase 1 (Jun 7) ★ NEW PROJECT HIGH ★ shared learnable factor
                 graph topology. ONE tensor (T=24, latent_dim=64) →
                 67K new params total. Adds attention bias =
                 topology @ topology.T (REFUTED — bias_scale stayed
                 ~0) AND per-position residual gate = tanh(topology
                 @ W_res) (VALIDATED — Wres_norm grew 0→0.503).
                 Trained 5000 effective steps warm-start from
                 cont8_step1000.
                 50-puzzle apples-to-apples results:
                   HARD:  0.3761 → 0.3945 (+0.0184) ★ beats SBP 0.3914
                   MED:   0.4910 → 0.5180 (+0.0270) ★
                   EASY:  0.5726 → 0.6371 (+0.0645) ★★ biggest gain
                   QUERY easy: 0.2600 → 0.4600 (+0.20) ★★★ massive
                 Mechanism: per-NODE gating of shared backbone is the
                 right factorization. Per-EDGE structure already
                 captured by existing binary masks. Topology vectors
                 learned anti-correlated identities (sim01: +0.024 →
                 -0.154). Step 4500 + final ckpts produce identical
                 50-puzzle results — robust signal. Phase 2 should
                 pursue per-position gating mechanisms (waist routing,
                 codebook attention, delta_gate scaling), NOT pairwise
                 structures.
Phase 1 parser   Designed (`mycelium/phase1_classifier.py`); not yet trained.
```

The **central paper claim** is the v98 Sudoku constraint energy curve:
geometric decay at rate ~0.5×/3 breaths, energy 21.0 → 0.71 over K=1
to K=20 on easy. This IS the signature of loopy BP convergence.

**Key memory notes:**
- `memory/project_v112b_phase1_validates_factorization.md` — Jun 7 NEW
  PROJECT HIGH; per-node gating validated, pairwise attention bias
  refuted; phase 2 direction
- `memory/project_v112b_shared_topology_design.md` — design memo behind
  the v112b architecture
- `memory/project_v109pi_diagnostics_jun5.md` — K-sweep 9/9 monotonic rise;
  OOD eval is a binning artifact; fair add/sub IND comparison
- `memory/project_v109pi_chain_to_9500.md` — π-cycled Q rotation chain results
- `memory/project_v109_prod_step10000_verdict.md` — v109 10K final numbers
- `memory/project_v109_ablation_clean_attribution.md` — three-row table:
  waist flips K-sweep, alternation adds marginal lift on hard
- `memory/project_v109_alternation_breakthrough.md` — v109 smoke first signal
- `memory/project_v108_k_sweep_verdict.md` — refutes MC framing; BP fixed
  point is the right characterization
- `memory/project_v98_sudoku_validates_paradigm.md` — six-component recipe
- `memory/project_factor_graph_framing.md` — breathing = learned approx BP
- `memory/project_ode_integrator_framing.md` — the deepest math identity
- `memory/project_musical_keys_topology.md` — topology determines rhythm
- `memory/feedback_digit_vs_number_prediction.md` — v105 digit-level lessons
- `memory/project_big_paper_strategy.md` — paper holding strategy
- `memory/project_v105_collapse_unbreakable.md` — Jun 2 finding: collapse
  at Pythia L0, readout-side fixes mechanically can't recover, v105.9
  pooled-AR breathes beautifully (Δ=1.37) but cell_hidden lacks number
  precision (R²=-614); v105.8 tests whether direct number CE fixes that

For the pre-v98 empirical state (v45/v55/v59 L4_MIXED, v77-v81 GSM8K
DAG paradigm), see `docs/archive/empirical_v45_to_v95.md`.

---

## 4. Specifications (tight)

- **Init:** Pythia-410M L0-3 (attn + FFN weights + token embeddings
  50304×1024). All 4 layers are SHARED across all K breaths in v98+
  (no phase-specific copies — that was a v1-v95 design and was
  dropped in the pivot).
- **Dimensions:** h=1024, 16 heads × head_dim 64, FFN 4096, vocab
  50304, max seq 512. 4 transformer layers in the iterated stack.
- **K_max per architecture:** Sudoku=20, v100=10, v105.1.2=8, v107=10.
  No global ceiling; AMD JIT capacity is the limit (~20 breaths).
- **Parameters:** ~35.7M shared transformer processing + 51.5M token
  embeddings = ~87M for v98/v100/v107. v105.1.2+ adds ~1M (waist) +
  ~1.6M (IB codebook). No separate Controller in active code path.
- **Memory:** ~5GB mixed precision, ~19GB headroom. BATCH=32-64 for
  Sudoku at FIXED_LEN=160; BATCH=8-16 for factor graphs at
  FIXED_LEN=256. KV cache sized to actual seq length.
- **Platform:** AMD 7900 XTX, tinygrad, AM driver (working since
  2026-05-11 — Secure Boot off + `vm.compact_unevictable_allowed=0`).
  Ubuntu 24.04. No ROCm, no CUDA, no PyTorch.

---

## 5. Editing rules

- **No mid-breath token generation.** Reasoning stays in 1024d
  residual stream; tokens (if any) generated once at the end.
  (Empirical: "had had had" within 2 autoregressive loops if violated.)
- **Diversity must be structural, not learned.** v98 row/col/box
  masks, v100 topological staging masks, v105 per-digit RoPE are all
  geometric/structural. Every v1-v3 learned diversity mechanism
  (scales, soft tokens, codebooks, fingerprints) collapsed to constant
  within one epoch.
- **Digit-spaced for arithmetic.** v98 uses single-cell digits; v100+
  uses bins or per-digit codebooks. Whole-number BPE tokens force
  memorization.
- **KV cache invariants:** size to actual seq length; pad eval batches
  to fixed batch_size so compiled graphs match; compile once during
  first eval, replay for the rest of the run.
- **Bryce wants root-cause perf fixes**, not workarounds, when perf is
  the bottleneck.
- **Factor per-NODE, not per-EDGE (Jun 7, v112b finding).** When adding
  new structure, prefer per-position gating mechanisms (each position
  gets its own activation pattern in the shared backbone) over pairwise
  structures (learned attention biases, edge-strength tensors). v112b's
  attention-bias channel REFUSED to engage (bias_scale stayed ~0) while
  its per-position residual gate became the load-bearing component
  (Wres_norm grew 0 → 0.503 monotonically). Edges are already captured
  by binary masks; per-node activation patterns are what's missing.
  See `memory/project_v112b_phase1_validates_factorization.md`.
- **If a Controller-like module is ever reintroduced:** separate
  optimizer, gradient never flows through transformer, verify with
  parameter-change smoke. Currently moot — no Controller in active
  code path.

---

## 6. Current work in progress (as of 2026-06-07)

- **v112b Phase 1 — NEW PROJECT HIGH (Jun 7).** Shared learnable factor
  graph topology tensor (T=24, latent_dim=64). 67K new params lifted
  ALL THREE difficulties from cont8_step1000 baseline: HARD 0.3761 →
  0.3945 (+0.0184, beats SBP 0.3914), MED 0.4910 → 0.5180 (+0.0270),
  EASY 0.5726 → 0.6371 (+0.0645), easy QUERY 0.2600 → 0.4600 (+0.20).
  Mechanism finding: per-position residual gate VALIDATED (Wres_norm
  grew 0 → 0.503 monotonically); pairwise attention bias REFUTED
  (bias_scale stayed ~0). The factorization that helps is per-NODE
  gating, not per-EDGE structure. Step 4500 + final ckpts produce
  identical 50-puzzle results. See
  `memory/project_v112b_phase1_validates_factorization.md`.
- **v110-step3 prod chain — DONE (Jun 5).** Goldilocks step-size
  regulation (calibration-driven `step_k/(1-calib_k)` CoV penalty)
  reached project all-times: easy 0.610, med 0.509, hard 0.399.
  cont8_step1000 is the warm-start anchor for all Jun 6-7 experiments.
- **The search-vs-sampling arc — DONE (Jun 6).** PUCT (correlated tree-
  path samples) regresses -0.006-0.012. MC-BP inference (independent
  noise) lifts +0.0061. SBP training (independent noise) lifts +0.0153.
  Independence quality is the discriminating factor. SBP is the paper's
  previous training-time row.
- **v109pi K-sweep — DONE (Jun 5).** cont9_step500 ckpt at K∈{2,4,8}.
  9/9 signals monotonically rise across {easy,med,hard}×{cell,pos4}.
  Strongest K-sweep in project. See
  `memory/project_v109pi_diagnostics_jun5.md`.
- **Paper draft.** Held at workshop tier (`paper/outline.md`). The
  v112b factorization breakthrough is a clean new row. Hold strategy
  in `memory/project_big_paper_strategy.md`: don't ship until Phase 1
  classifier + IB-anchored Phase 2 codebook + end-to-end GSM8K are
  added. Target: top-conference-tier upgrade.
- **Phase 1 classifier — built, not trained.** DistilBERT-based NL
  parser at `mycelium/phase1_classifier.py`. Spec at
  `docs/phase1_nl_parser_spec.md`. Build script at
  `scripts/build_phase1_classifier_data.py`. Smoke at
  `scripts/phase1_classifier_smoke.sh`.
- **v106 PUCT search — ROBUST NEGATIVE (Jun 6).** 8 configs tested on
  v110-step3; all regress by 0.006-0.012. Locally-consistent-but-
  globally-wrong attractor problem means search amplifies BP
  miscalibration. Preserved at `mycelium/factor_graph_v106_step3.py`
  for future use when BP quality improves.

---

## 7. What we carry forward (and what we left behind)

**Forward (v98+ design):**
- Pythia-410M L0-L3 init
- Partial weight sharing — actually FULL sharing across breaths in v98+
- Iterative prefill (K passes, residual stream as persistent state)
- Per-breath `delta_gate` (replaces v1-v95 controller-emitted gate)
- Per-breath calibration head (Dopri5-style error estimator)
- Per-breath weighted CE supervision (the ladder)
- Structured per-head attention masks (replaces π-cycled RoPE)
- Variant codebook readout (replaces token vocab when no AR decode)
- JIT-fused KV cache (when AR decode is used)
- The copy machine principle (no mid-breath token gen)
- Structural-not-learned diversity rule
- **Per-node residual gating via shared topology tensor (v112b, Jun 7)** —
  one learnable `node_topology` tensor parameterizes each position's
  identity; per-position gate `tanh(topology @ W_res_gate)` modulates
  the shared backbone's activation pattern. 67K new params lifted ALL
  THREE difficulties (easy +0.0645, med +0.0270, hard +0.0184 — NEW
  PROJECT HIGH). The mycelium principle made architectural: same shared
  backbone, different per-position activations.
- **SBP training (v110-step3, Jun 6)** — 50/50 alternating residual-stream
  noise during training. Denoising score matching at the injection
  point. Zero inference cost. Sharpens attractors away from confidently-
  wrong fixed points. Composes with v112b per Phase 2 hypothesis.

**Left behind:**
- Llama 1B (replaced by Pythia-410M L0-3 in v4-era)
- LoRA atoms and continuous scales (v1-v3)
- Straight-through gradient estimator (v1-v3)
- Soft token diversity mechanisms (v1-v3)
- PyTorch / ROCm / Windows (project-wide)
- **Controller, Notebook, LookupTable** (v1-v95 — module code preserved
  in `mycelium/breathing.py` and `controller.py` for import compat, not
  called by any v98+ trainer)
- **π-cycled per-breath RoPE** (v1-v95 — replaced by structured
  per-head masks)
- **Sine-modulated within-breath temperature** (v1-v95 — replaced by
  single fixed temp `1/sqrt(head_dim)`)
- **WaistController paradigm** (v54-v95 — the compressed 512d waist
  → AR decode interface plateaued at <5% on GSM8K. v105.1.2+ restores
  the waist as a quantize step in the codec framing, but read-out is
  via codebook not AR decode.)
- **GSM8K AR-decode-via-waist** (v80-v81 — replaced by factor-graph
  paradigm; v107 Phase 2 reads factor graphs via Haiku distillation,
  not AR-from-prompt)

---

## 8. Active research threads (sequenced)

**Updated sequence (Jun 7):**

The current load-bearing result is **v112b Phase 1 = v110-step3 +
shared learnable factor graph topology + per-position residual gate**
(Jun 7): all three difficulties lifted (easy +0.0645, med +0.0270, hard
+0.0184 — NEW PROJECT HIGH at 0.3945). Mechanism finding: per-node
gating works, pairwise attention bias doesn't. See
`memory/project_v112b_phase1_validates_factorization.md`.

Phase 2 should extend the per-node gating principle to other components
(waist routing, codebook attention, delta_gate scaling). Phase 3 (likely
paper-2) is the v112b + SBP composition test.

1. **v112b Phase 2** — extend per-node gating to other components.
   Three concrete tests: (a) per-position waist routing
   `softmax(topology @ W_route)` driving which waist channels each node
   uses; (b) per-position codebook attention `softmax(topology @ W_cb)`
   per-position attending to codebook entries; (c) per-position
   delta_gate scaling `sigmoid(topology @ w_delta)` per-position step
   size. Each is one new projection on top of the validated topology
   tensor. Build expectation: 2-3 hours per variant, smoke test before
   the next.
2. **v112b + SBP composition** — orthogonal mechanism test. SBP
   sharpens attractors via training noise; v112b factors per-node
   computation via architecture. If they compose, hard cell_acc should
   exceed 0.40. Cheap to test: warm-start v112b cont1_final + SBP-style
   alternating noise for 500-2000 steps. This is the paper-2 row.
3. **v109pi K=16 retrain** — the strongest K-sweep in the project
   motivates retraining at K=16 (current K=8). Question: does the
   monotonic rise extend to K=16, or saturate?
4. **Phase 1 small NL→factor-graph model** —
   `memory/project_phase1_segment_classify_design.md`. Segment-and-classify
   on DistilBERT; flat BIO span tagging + ~8-op codebook classification
   (Phase 1A) + deterministic compiler → DAG (Phase 1B). Required for the
   "fully on-device" deployment story. Now load-bearing: Phase 2
   (inference engine) is empirically strong enough to ship.

**Other open threads (lower priority, parallel-tractable):**

- **v98 ablation — which constraint masks are load-bearing?** —
  pre-paper figure-2 work (v98 Sudoku is the paper's central claim, so
  the constraint-mask ablation supports it).
- **Notebook-as-MCTS-state** —
  `memory/project_notebook_mcts_design.md`. Deferred until v112b Phase
  2 is validated; if per-node gating extends to delta_gate, MCTS on
  per-node step decisions is the natural extension.

For the v1-v95 era research threads (E-and-B oscillation, BirdNET head
specialization, hierarchical IB → MCTS, photon zero-crossing, JSD
attention analysis, IB-clustered codebook, multi-head WaistController,
etc.), see `docs/archive/`.
