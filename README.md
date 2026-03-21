# Mycelium v16 — Breathing Models

> **A small model that cannot solve hard problems in a single pass can solve them
> by learning to decompose into steps that fit within its capacity.**
>
> Compression happens in latent space. Gradients flow through. The model is born breathing.

---

## Current State (April 1, 2026)

```
Architecture:        SmolLM2-135M + CompressionHead (32 floats) + StateInjector (4 pseudo-tokens)
Status:              TWO-STEP ARITHMETIC PROVEN — 0% → 80.4% through breathing

LANDMARK RESULT:
  Single-shot accuracy:     0%
  Breathing accuracy:       80.4% (both steps correct)
  Ablation (real vs zeros): 96% vs 6% (+90 point delta)
  State vector:             32 floats (1,024 bits)
  New parameters:           ~600K (0.4% of model)

Next:                Three-step arithmetic with 64-float state
```

---

## What Mycelium Does

Mycelium trains a vanilla transformer (SmolLM2-135M) to reason in **expand-collapse cycles** with **differentiable latent-space compression**. Instead of text-based memory, the model compresses its reasoning into a state vector that persists between breaths.

The model never sees its full solution history. It sees only the problem and pseudo-token embeddings derived from its last state vector. The state vector IS the memory.

---

## Why Latent-Space Compression

Previous approaches (v10-v15) used text-based compression with [EXPAND]/[COLLAPSE] markers. The problem: **text is discrete, you can't backprop through it.**

The new approach:
- **State vector**: Continuous floats (not text tokens) — scales with difficulty (32 → 64 → 128 → 256 → 512)
- **Fully differentiable**: Gradients flow from answer loss back through every breath
- **Autoencoder warmup**: Solves cold start by pretraining on oracle collapses

```
Old: hidden_states -> [COLLAPSE] text -> next breath (no gradients)
New: hidden_states -> CompressionHead -> state vector -> StateInjector -> next breath (gradients!)
```

---

## The Breathing Cycle

```
Input:   [state_tokens (4)] + [BREATH n/max] + [problem_tokens]
         |
         SmolLM2-135M generates reasoning
         |
Output:  hidden_states -> CompressionHead -> state vector
         |
         state vector -> StateInjector -> pseudo-tokens for next breath
```

The model loops until `\boxed{}` or max breaths.

---

## The Entropy Wave

Token-level entropy across breaths forms a sine wave:

```
H(t)
 |.--.
 |/   \  .--.
 |      \/   \  .--.
 |            \/   \  ..
 |                  \/  \-> answer
 |------------------------------- breaths
```

Each breath: entropy rises during generation (exploring), falls during compression (committing). The LOCAL oscillation is the breathing rhythm. The GLOBAL envelope descends toward the answer.

**Thermodynamic view:** Mathematical reasoning is heat engine cycles. Problem = high energy. Answer = low energy. Each breath removes entropy. Larger model = more entropy per cycle = fewer breaths needed.

---

## Architecture

```
SmolLM2-135M:       135M params (base model)
CompressionHead:    Perceiver-style encoder (hidden -> state vector)
StateInjector:      State -> pseudo-tokens (prepended to input)
StateDecoder:       Phase 0 only (discarded after autoencoder warmup)
```

### CompressionHead
Perceiver-style cross-attention: learned queries attend to hidden states, project to state vector.

### StateInjector
Splits state vector into chunks, projects each to d_model (576), adds positional embeddings.

---

## Three-Phase Training

### Phase 0: Autoencoder Warmup
Train encoder+decoder on oracle collapses. Solves cold start — gives the compressor a meaningful latent space before any breathing. Decoder is discarded after.

### Phase 1: Frozen Compressor
Train transformer + injector to USE state vectors. Compressor frozen from Phase 0.

**State scale warmup:** 0.1 → 1.0 over first 5 epochs (prevents pseudo-token interference)

### Phase 2: End-to-End
Unfreeze everything. Gradients flow from answer loss through state vector. Compression co-evolves with generation.

---

## Data

```
Phase 0:  Oracle collapses from GSM8K/MATH solutions
Phase 1:  Breath-structured sequences from strong model
Phase 2:  GSM8K → MATH train
Never:    MATH-500 (evaluation only)
```

---

## File Structure

```
src/
  compression_head.py      # Perceiver encoder: hidden -> state
  state_injector.py        # State -> pseudo-tokens
  state_decoder.py         # Decoder for autoencoder (Phase 0 only)
  breathing_model.py       # Full recurrent loop

scripts/
  train_autoencoder.py     # Phase 0: autoencoder warmup
  train_phase1.py          # Phase 1: frozen compressor
  train_two_step.py        # Two-step arithmetic training
  train_two_step_v2.py     # Two-step v2 (state scale warmup)
  ablation_state_vs_zeros.py  # State ablation testing
  eval_breathing.py        # Evaluation
  eval_phase1_quick.py     # Quick eval
  generate_step_solutions*.py  # Data generation

configs/
  autoencoder.yaml         # Phase 0 config
  phase1_gsm8k.yaml        # Phase 1 config
```

---

## Milestones

```
M0:    COMPLETE — SmolLM2-135M baseline (0% GSM8K, 0% MATH-500)
M0.5:  COMPLETE — Architecture built (CompressionHead, StateInjector, etc.)
M0.75: COMPLETE — Oracle collapse data (206K breaths from 36K GSM8K solutions)
M1:    COMPLETE — Phase 0 autoencoder warmup (loss 1.59 → 0.44, t-SNE structured)

M2:    COMPLETE — Two-step arithmetic (LANDMARK RESULT)
       32-float state, 4 pseudo-tokens, breath counter, supervised
       Training: Epoch 1 B1=58%/B2=7% → Epoch 10 B1=88%/B2=91%/Both=80.4%
       Ablation: real state 96% vs zeros 6% (+90 point delta)
       0% single-shot → 80.4% with breathing. Core thesis proven.

M3:    Three-step arithmetic (NEXT)
       ((a+b)*c)-d with 64-float state, 3 breaths

M4:    Easy GSM8K (128-float state, 3-5 breaths)
M5:    Full GSM8K (256-float state, 10 breaths, autonomous decomposition)
M6:    MATH L1-L3 (512-float state)
M7:    THE HEADLINE — MATH L4-L5 (capacity extension on competition math)
M8:    Scaling analysis (50M → 1.7B)
```

---

## Critical Rules

```
 1. SmolLM2-135M BASE (not instruct) — no single-shot habits
 2. State vector ramps with difficulty: 32 → 64 → 128 → 256 → 512
 3. State scale warmup: 0.1 → 1.0 over first 5 epochs
 4. Breath counter [BREATH n/max] in prompt
 5. Ablation (real state vs zeros) at every major checkpoint
 6. Temperature = 0 for all evaluations
 7. N >= 100 problems for any accuracy measurement
 8. MATH-500 NEVER in training data
 9. One difficulty level at a time: arithmetic → GSM8K → MATH
10. Increase state size only when accuracy plateaus
```

---

## Hardware

```
Training:    SmolLM2-135M + ~600K params fits easily on A10G (24GB)
Phase 0:     ~2-4 hours on A10G
Phase 1:     ~8-16 hours on A10G
Phase 2:     ~24-48 hours on A10G
Cost:        ~$50-100 on AWS spot for full pipeline
```

---

## Deadline

**MATH-500 benchmark: April 22, 2026**

Target: breathing accuracy > single-shot on Level 4-5 problems.

---

> *Good expansion is expansion that's easy to collapse.*
> *Good collapse is collapse that sets up the next expansion.*
> *The model descends the energy landscape one breath at a time.*
> *Small steps. Continuous gradients. Always learning.*
