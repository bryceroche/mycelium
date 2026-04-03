# Mycelium v18 — Integrated Thinking

> **A transformer that THINKS before it SPEAKS.**
>
> Multiple forward passes to understand. A single generation pass to answer.
> 108M parameters deciding what goes on a 64-float sticky note.

---

## Current State (April 3, 2026)

```
Architecture:        Llama 3.2 1B-Instruct + 7-Layer AllLayerPerceiver (108M)
Status:              ARCHITECTURE PIVOT — Integrated Thinking

LANDMARK RESULTS (prior work):
  SmolLM2 two-step:   0% → 80.4% (32 floats, +90 point ablation)
  SmolLM2 three-step: 0% → 52%   (64 floats)
  Llama two-step:     0% → 83%   (tight bottleneck scales)

Target:              GSM8K thinking accuracy > 35% (single-shot baseline)
```

---

## What Mycelium Does

Mycelium trains a transformer to reason through **multiple internal passes** before generating any text. Each pass:
1. Processes the problem through all 16 Llama layers
2. A 7-layer Perceiver reads ALL layers and compresses to 64 floats
3. State accumulates via residual connection
4. When confident, generate the final answer

The model never generates text during thinking. It gets multiple forward passes to understand, then speaks from accumulated understanding.

---

## The Architecture

```
                    ┌──────────────────────────────────────────────────────┐
                    │                                                      │
                    ▼                                                      │
[problem tokens] + [4 state pseudo-tokens]                                │
        │                                                                 │
        ▼                                                                 │
   Layer 1 → Layer 2 → ... → Layer 16                                     │
        │         │                │                                      │
        │         │    (all layers saved)                                 │
        ▼         ▼                ▼                                      │
   ┌──────────────────────────────────┐                                   │
   │  7-LAYER PERCEIVER (108M params) │                                   │
   │  Pass-conditioned layer gate     │                                   │
   │  4 queries → 64 floats           │ ← TIGHT BOTTLENECK               │
   └──────────────────────────────────┘                                   │
        │                                                                 │
        ▼                                                                 │
   state = state + alpha * compressed   ← RESIDUAL                       │
        │                                                                 │
        ├──→ ConfidenceHead → ready?                                      │
        │         │                                                       │
        │     NO  │  YES → GENERATE ANSWER                                │
        │         │                                                       │
        └─────────────────────────────────────────────────────────────────┘
```

---

## Why This Works

**Tight bottleneck (64 floats)**: Forces incremental thinking. The model can't solve GSM8K in one pass — it needs 3-5 passes, each extracting value + context + intent.

**Massive compressor (108M params)**: The asymmetry is the point. 108M parameters deciding what goes on a 64-float sticky note. Like a brilliant editor writing a one-sentence summary.

**All-layer reading**: The Perceiver reads ALL 16 transformer layers with pass-conditioned attention. Early passes focus on parsing layers (1-8), later passes on reasoning layers (12-16).

**Residual accumulation**: `state = state + alpha * delta`. Each pass ADDS to understanding, doesn't replace. Gradients flow directly through addition.

**No generation during thinking**: Forward passes without text generation are cheap. 10 thinking passes ≈ generating 200 tokens.

---

## Components

```
Llama 3.2 1B-Instruct:  1.23B   (35% GSM8K single-shot — strong gradient signal)
AllLayerPerceiver:      ~108M   (7 layers, pass-conditioned, all-layer reading)
StateInjector:          ~130K   (64 floats → 4 pseudo-tokens)
ConfidenceHead:         ~2K     (when to stop thinking?)
────────────────────────────────
Total:                  ~1.34B
Bottleneck:             64 floats (2,048 bits)
```

---

## Training

**Phase 1**: Freeze transformer, train perceiver + injector + confidence head (5-10 epochs)

**Phase 2**: Unfreeze everything, end-to-end (10-20 epochs)

**Deep supervision**: Every thinking pass gets direct gradient (predicts answer from current state)

**State scale warmup**: 0.1 → 1.0 over first 5 epochs

---

## Why This Beats Previous Approaches

| Approach | Problem | Integrated Thinking |
|----------|---------|-------------------|
| External loop (v16) | Slow (generation each pass), lossy | Fast (forward passes only), rich |
| Hourglass (v4) | Distribution mismatch mid-layers | Compression after all layers |
| Text breathing (v10-v15) | No gradients through text | Continuous compression, full gradients |

---

## Milestones

```
COMPLETE:  Tight bottleneck proof-of-concept (80.4% two-step, 52% three-step)
COMPLETE:  Llama two-step (83%)

CURRENT:   Build integrated thinking architecture
           GSM8K thinking accuracy > 35% (single-shot baseline)
           GSM8K thinking accuracy > 45% (+10 points)

FUTURE:    MATH L1-L5 (capacity extension on competition math)
```

---

## Key Insight

Current LLMs generate tokens immediately — one forward pass per word. This architecture decouples thinking from speaking. The model processes the problem multiple times internally, accumulating compressed insights, before producing a single token of output.

When it finally speaks, it speaks from a place of accumulated understanding.

---

## Deadline

**MATH-500 benchmark: April 22, 2026**

Target: thinking accuracy > single-shot on Level 4-5 problems.

---

> *Think hard, compress tight, speak from understanding.*
