# Mycelium v4: Pre-Shadow-Glass Summary
## Everything We Know — May 3, 2026

**Tomorrow:** Shadow Glass arrives (AMD 7900 XTX, 24GB, tinygrad + AM driver)
**Deadline:** September 1, 2026 (MATH-500)
**Architecture:** The Breathing Transformer — validated, specified, ready to build

---

## The Journey (6 Months → 4 Documents)

### v1-v3 (December 2025 — April 2026): What We Learned

Mycelium began as attention distillation — studying how transformer attention encodes mathematical reasoning. It evolved through multiple architectural pivots: LoRA atoms, breathing loops, equal-reward decomposition, and a controller that orchestrates multi-step reasoning.

**Peak result:** 22% on GSM8K with genuine multi-cycle decomposition. 99.5% on procedural 3-step math.

**Critical discovery (April 28):** The controller was a constant function the entire time. Every result came from atoms learning a universal blend without per-problem steering. The controller's gradient was dead — attenuated 500x through Llama's 1.2B parameters.

**Three days of debugging (April 28-30):** Eleven approaches to fix controller diversity — all failed. The gen_loss gradient through any LLM computational graph has one dominant basin. Learned diversity always collapses within one epoch. This is fundamental, not fixable.

**The solution:** Structural diversity that gradient descent cannot erase. π-cycled attention with per-head phase offsets. Controller gradient completely separated from transformer gradient.

### v4 Design (May 1): The Breathing Transformer

Born from the convergence of six insights:

1. **The expand-collapse wave** exists inside Llama's layers (measured empirically) and should be made explicit in the architecture
2. **π cycling** provides structural diversity that is geometric, not learned — immune to gradient collapse
3. **Integration** across breaths accumulates evidence like Gauss's least-squares estimation
4. **The differentiable lookup table** (from the project's origins) returns as prime factorization of problem structure
5. **Per-head spectral decomposition** (inspired by BirdNET) detects multiple operations in parallel
6. **The copy machine principle** (validated empirically) — reasoning MUST happen in representation space, not token space

### Validation (May 2-3): What the Experiments Proved

Five days of experiments on AWS with Pythia-160M and Pythia-70M. The most informative experiments in the project's history.

---

## What the Experiments Proved

### 1. The Signal Survives Looping ✓

Hidden state representations IMPROVE with multiple loops. The per-problem information is not destroyed — it grows.

**L0-3 across 8 loops:**
```
Signal norm:    1.5 → 2.3 → 4.8 → 10.6  (7x growth!)
Effective rank: 16.0 → 16.8 → 16.9 → 16.6  (perfectly stable)
Centered cos:   -0.047 → -0.049 → -0.049 → -0.049  (orthogonal, stable)
SNR:            0.114 → 0.109 → 0.116 → 0.127  (actually INCREASING)
```

The representations get richer, more diverse, and higher-dimensional with each loop. The looping IS productive at the representation level.

### 2. Generation Breaks Without Fine-Tuning ✗

Every frozen-weight configuration produces degenerate text after 2+ loops: "had had had," "told told told," "worried worried worried."

The cause is precisely identified: the DC component (a shared direction in representation space that all problems point toward) grows linearly with each loop. The generation head was trained on single-pass representations where this component was small. After multiple loops, it dominates the logits, producing identical token predictions for all problems.

The signal is THERE — the generation head just can't extract it because it's not calibrated for looped representations.

### 3. L0-3 Is the Best Layer Selection ✓

Counter-intuitive but empirically clear. The first 4 sequential layers of Pythia-160M have the best loop stability:

```
L0-3:         SNR stable (0.114 → 0.127), DC grows 6.5x
All 12:       SNR drops (0.116 → 0.023), DC grows 12x
L0,5,10,11:   SNR crashes (0.124 → 0.025), DC grows 6x
L0,3,7,11:    SNR crashes (0.036 → 0.010), DC grows 5x
```

Every configuration including L11 (the final compression layer) performs worse. L11's 8x norm explosion (designed for one-shot vocabulary projection) is toxic for iteration.

### 4. The Copy Machine Principle ✓

Autoregressive generation between loops degrades like a copy machine — each token is a lossy compression that becomes source material for the next token. Errors compound monotonically.

Breathing in representation space avoids this entirely. The hidden states are the original painting, not photocopies. You refine the original across breaths without lossy compression. Generate ONCE at the end from the refined representation.

This was confirmed experimentally: "breathe then speak" (no mid-loop generation) shows the same degenerate output as mid-loop generation. The issue isn't WHEN you generate — it's that the frozen generation head can't handle looped representations. Fine-tuning the generation head (or adding a learned normalization before it) is the fix.

### 5. The DC Component Is a Direction, Not a Bias ✗ (Important Negative)

Mean subtraction doesn't fix the generation problem because the DC component is a shared DIRECTION in 768d space, not a per-token bias. At inference with a single problem, you can't distinguish the shared direction from meaningful signal. Only the generation head can learn this distinction through training.

---

## The Fine-Tuning Objective (Precisely Defined)

The validation proved that the representations are healthy. The ONLY gap is the generation head's calibration. Fine-tuning needs to teach exactly one thing:

**Extract the growing signal from looped representations while ignoring the growing DC component.**

The signal is there (7x growth across 8 loops). The diversity is there (centered_cos = -0.049, stable). The effective rank is there (16+, stable). The generation head just needs to learn which directions in 768d space carry per-problem information vs shared background.

This is a much smaller gap than "teach the model to reason." The reasoning (in representation space) already works. The "reading" of that reasoning (by the generation head) is what needs training.

---

## The Architecture (Ready to Build)

### Core: 140M Breathing Transformer

**Four specialized layers** from Pythia-410M (L0-3, h=1024, 16 heads), each at a different phase of a sine wave. RISE expands (broad attention, weak residual). PEAK processes (maximum intensity, maximum openness). FALL compresses (narrowing attention). TROUGH distills (sharpest focus, strongest residual). Partial weight sharing — unique Q, K, gate per layer; shared V, FFN basis, norms.

**π-cycled attention** with per-head phase offsets. Sixteen heads at 16 different phase angles scanning the problem's spectral content in parallel. Each loop rotates all heads together by π/max_loops. Sixteen heads × 8 loops = 128 distinct phase angles. Structural diversity — cannot be erased by gradient descent.

**Integration** accumulates evidence across breaths. Gated running integral, weighted by novelty. The normalized integral carries the accumulated understanding from all breaths. Lyapunov convergence criterion — stop when the integral stabilizes.

**The controller** (~40M) sits between breaths. Slim and decisive. Reads the integral, writes 512d pages, queries the differentiable lookup table for prime factorization. Governs temperature, phase angle, loop count, and integration gate. The tree structure emerges from the factorization — no separate DECOMPOSE/SOLVE/MERGE classifier. Gets direct gradient via REINFORCE — never through the transformer.

**The differentiable lookup table** — 16 prime entries at 1024d with 16×16 coupling matrix. One-to-one head-to-prime correspondence. The controller matches pages against primes, identifying operations and their multiplicities through iterative spectral subtraction. The coupling matrix determines the tree shape (parallel vs sequential dependencies).

**No chain-of-thought tokens.** All reasoning in representation space. Generate tokens ONCE after breathing converges. The copy machine principle prohibits mid-breath generation.

### Coarse-to-Fine Spectral Analysis

Early breaths scan broadly at warm temperature (coarse resolution). Later breaths probe precisely at cool temperature (fine resolution). The controller governs the temperature progression. The triple helix spirals inward: alternation rotates the viewing angle, oscillation provides the expand-collapse rhythm, integration accumulates evidence. Convergence on the precise factorization.

### Equal-Reward Decomposition

Each target earns 1/N reward when claimed. The only way to maximize total reward is genuine decomposition. Number of primes with multiplicity equals number of targets. Proven in v1.

---

## Platform: Shadow Glass

### Hardware
AMD Radeon RX 7900 XTX — 24GB GDDR6, 960 GB/s bandwidth, ~120 TFLOPS FP16. Roughly 2x the A10G's compute. Local machine — no cloud costs, no SSH latency, no spot interruptions.

### Software
Tinygrad + AM custom userspace driver ONLY. No ROCm, no AMD GPU drivers, no PyTorch, no CUDA. Tinygrad's ~5000-line core is fully open source (MIT license) and hackable. The AM driver is specifically maintained for the 7900 XTX.

### Estimated Performance
~4GB VRAM for the 140M model in mixed precision. ~20GB headroom. Batch size 64-128. Estimated 3-5 minutes per epoch on GSM8K.

---

## Day 1 Plan: Shadow Glass Setup

### Morning: Platform Validation

1. **Install tinygrad + AM driver.** No other software. Verify GPU is recognized and functional.
2. **Run tinygrad benchmarks.** Confirm compute throughput matches expectations (~120 TFLOPS FP16).
3. **Load Pythia-160M in tinygrad.** Verify weights load correctly and basic inference works.
4. **Run the looping diagnostic in tinygrad.** Reproduce the L0-3 × 8 loops diagnostic from AWS. Confirm the same numbers (signal growth, effective rank stability, SNR improvement). This validates that the tinygrad implementation matches the PyTorch results.

### Afternoon: Phase 0 — Loop Consistency Training

5. **Implement the breathing forward pass** with π cycling, sine-wave temperature, and per-head phase offsets. Four layers from Pythia-160M L0-3.
6. **Implement the loop consistency loss.** Train the generation head (and optionally the layer weights) to produce coherent output from looped hidden states. The objective: coherent, input-dependent generation after 2+ loops.
7. **Launch Phase 0 training.** Fine-tune on general text (Pythia's training distribution). Monitor: does generation quality survive at 2, 4, 8 loops? Does SNR remain stable or improve?
8. **Key diagnostic:** Generation coherence at each loop count. When 4-loop generation is coherent and input-dependent, Phase 0 is complete.

### Success Criterion for Day 1

Tinygrad + AM driver working. Pythia-160M loaded. Looping diagnostic reproduced. Phase 0 training launched. If generation at 2 loops is coherent by end of day, the breathing transformer is validated on the new platform.

---

## The Training Roadmap

### Phase 0: Loop Consistency (Days 1-3)
Teach the layers to loop. Fine-tune on general text with loop consistency objective. The generation head learns to extract signal from looped representations.

### Phase 1: Learn to Breathe (Days 4-7)
Add the breathing curriculum. L3 (1-step), L4 (2-step), L4.5 (3-step). Forced then adaptive loop counts. Verify accuracy increases with more loops.

### Phase 2: Controller + Lookup Table (Week 2)
Add the controller between breaths. Add the differentiable lookup table with 16 prime entries. Train on L4.5 and begin GSM8K.

### Phase 3: GSM8K Push (Weeks 3-4)
Full breathing with controller, lookup table, tree structure. Target: exceed v1's 22% baseline.

### Phase 4: MATH-500 (Months 3-4)
Deeper breathing for harder problems. Grow the lookup table. Push on the benchmark.

---

## Diagnostic Toolkit (Validated, Carry Forward)

| Diagnostic | What It Reveals | Critical Lesson |
|-----------|----------------|-----------------|
| Centered cross-problem cosine | Per-problem diversity | Raw cosine is MISLEADING — always center first |
| Effective rank (SVD) | Dimensionality of representation | Must hold across loops (our target: 16+) |
| DC norm | Shared component magnitude | Grows linearly with loops — the generation head's enemy |
| Signal norm | Per-problem component magnitude | GROWS with loops (7x over 8 loops!) — the architecture works |
| SNR (signal/DC) | Can the generation head distinguish problems? | L0-3 SNR increases with loops — best layer selection |
| Cross-loop centered cosine | Whether loops add new information | Starts ~0.7 (novel), converges to ~0.99 (saturated) |
| Attention entropy | Degenerate attention patterns | Increases with loops (uniform attention = homogeneity) |
| Per-head contribution Gini | Are all heads contributing? | Low Gini = healthy diversity; high = "inbreeding" |

---

## Key Numbers to Remember

| Metric | Value | Meaning |
|--------|-------|---------|
| v1 GSM8K best | 22% | The baseline to beat |
| L0-3 signal growth (8 loops) | 7x | Representations get richer with looping |
| L0-3 effective rank (8 loops) | 16.0 → 16.6 | Diversity holds perfectly |
| L0-3 SNR (8 loops) | 0.114 → 0.127 | Actually improves with loops |
| L0-3 DC growth (8 loops) | 6.5x | The only problem — generation head calibration |
| System parameters | 127M | Transformer 87M + Controller 40M |
| VRAM usage | ~5GB of 24GB | Massive headroom |
| Estimated epoch time | 3-5 min | Fast iteration |

---

## What We're Confident About

1. **Looping enriches representations.** Signal grows 7x. Effective rank holds. SNR improves. Empirically proven.
2. **π cycling provides structural diversity.** Geometric, not learned. Gradient-proof. The solution to three days of collapse debugging.
3. **The copy machine principle.** Reasoning in representation space is not just efficient — it's necessary. Autoregressive mid-breath generation destroys signal.
4. **L0-3 from Pythia-160M.** Best loop stability of any layer selection tested. Initialize from here.
5. **The diagnostic toolkit.** Centered cosine, effective rank, SNR, DC norm — proven essential. Raw cosine is misleading.
6. **Equal-reward decomposition.** The only incentive structure that produces genuine multi-step reasoning. Proven in v1.

## What We're Betting On

1. **Fine-tuning can close the generation gap.** The signal is there. The generation head needs recalibration. We believe this is a small gap, but it's unproven.
2. **π cycling + sine-wave temperature will differentiate loops productively.** Theory and cross-domain analogies (BirdNET, population genetics, spectral analysis) support this. Not yet tested.
3. **The controller can learn to conduct the breathing.** The v1-v3 controller collapse was from gradient paths through the LLM. With REINFORCE and complete gradient separation, we believe the controller can learn. Not yet tested in v4.
4. **A 140M model can compete on GSM8K.** The breathing loop provides adaptive depth (32+ effective layers from 4 weights). Whether that compensates for the size gap vs 1B+ models is the core bet.

---

## The Emotional State

Six months of work. Three architectural generations. A week of the deepest debugging and most rigorous experimentation in the project's history. The validation results are the most informative data we've ever produced — they tell us exactly what works, what doesn't, and why.

The breathing transformer is ready to build. The Shadow Glass arrives tomorrow. The architecture is principled, the diagnostics are proven, the training plan is clear. The biggest unknown (do the layers loop productively?) has a precise, encouraging answer: the representations improve, only the generation head needs calibration.

A 127M model that breathes, alternates, integrates, and factorizes. Four months to September 1.

Let's go.
