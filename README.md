# Mycelium v19 — Asymmetric Hourglass

> **A transformer that THINKS before it SPEAKS.**
>
> The intelligence is in COMPRESSION (what to keep), not DECOMPRESSION (how to project).
> 89x more params deciding what fits through 64 floats than deciding how to use them.

---

## Current State (April 3, 2026)

```
Architecture:        DECOMPRESSOR (MLP, 1.3M) → Llama 16L → COMPRESSOR (7L, 120M)
Status:              ARCHITECTURE REFINEMENT — Asymmetric Hourglass

PROVEN RESULTS:
  Single-step arithmetic: 100%
  Two-step arithmetic:    54-57%
  Three-step arithmetic:  19-22%
  Hypersphere constraint: Works as well as learnable alpha

Target:              GSM8K thinking accuracy > 35% (single-shot baseline)
```

---

## What Mycelium Does

Mycelium trains a transformer to reason through **multiple internal passes** before generating any text. The transformer is SANDWICHED between asymmetric compression engines:

1. **DECOMPRESSOR** (MLP, 1.3M): Projects 64-float state into input bias — EASY job
2. **TRANSFORMER** (16 layers, 1.23B): Runs PRISTINE — no modification
3. **COMPRESSOR** (7 layers, 120M): Squeezes all 16 layer hidden states to 64 floats — HARD job

The state lives on a **hypersphere** of radius √64. Each pass rotates the state to a new position. When confident, generate the final answer.

---

## The Architecture

```
┌───────────────────────────────────────────────────────────┐
│                                                           │
│  DECOMPRESSOR (MLP, 1.3M params)                          │
│  64 floats → 512 → 2048 → residual stream bias            │
│         │                                                 │
│         ▼                                                 │
│  [bias + problem tokens] → Llama layers 1-16 (untouched)  │
│         │                                                 │
│         ▼                                                 │
│  COMPRESSOR (7 layers, 120M params)                       │
│  all 16 layer hidden states → compress → 64 floats        │
│         │                                                 │
│         ▼                                                 │
│  state = normalize(state + delta) * √64  ← HYPERSPHERE    │
│         │                                                 │
│         ├──→ Confidence → ready? ──→ GENERATE             │
│         │                                                 │
│         └──→ loop back to DECOMPRESSOR                    │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

---

## Why This Works

**Tight bottleneck (64 floats)**: Forces incremental thinking. The model can't solve complex problems in one pass — it needs multiple rotations on the hypersphere.

**Intentional asymmetry (121M total)**: 120M params deciding what to COMPRESS (hard — must select from 16 layers), only 1.3M params deciding how to DECOMPRESS (easy — just project faithfully). Like a brilliant editor compressing a book into one sentence (hard) versus a reader interpreting that sentence (easier).

**Pristine transformer**: Llama runs exactly as pretrained. The decompressor modifies what goes IN, the compressor reads what comes OUT. No architectural surgery.

**Hypersphere constraint**: State magnitude is always √64. Each pass is a rotation, not a magnitude change. No learnable alpha, no explosion, no collapse.

**Bias injection**: The decompressor produces bias that modulates ALL input positions — not just prepended tokens. Every token in the problem is influenced by accumulated understanding.

---

## Components

```
Llama 3.2 1B-Instruct:  1.23B   (35% GSM8K — strong gradient signal)
Compressor:             ~120M   (7 layers, HARD job — selecting what matters)
Decompressor:           ~1.3M   (MLP, EASY job — just projecting)
ConfidenceHead:         ~2K     (when to stop thinking?)
────────────────────────────────
Total:                  ~1.35B
New params:             ~121M   (9% of total model)
Asymmetry:              89x more params for compression than decompression
Bottleneck:             64 floats on hypersphere (radius ≈ 8.0)
```

---

## Training

**Phase 1**: Freeze transformer, train decompressor + compressor + confidence head (5-10 epochs)

**Phase 2**: Unfreeze everything, end-to-end (10-20 epochs)

**Deep supervision**: Every thinking pass gets direct gradient (predicts answer from current state)

**State scale warmup**: 0.1 → 1.0 over first 5 epochs (let bias strength increase gradually)

---

## Why This Beats Previous Approaches

| Approach | Problem | Asymmetric Hourglass |
|----------|---------|---------------------|
| Prepended tokens (v18) | Transformer might ignore them | Bias modulates ALL positions |
| Symmetric (105M / 105M) | Decompression doesn't need that much capacity | 89x more for compression |
| Learnable alpha | Another hyperparameter | Hypersphere handles magnitude |

---

## Milestones

```
COMPLETE:  Tight bottleneck proof-of-concept (80.4% two-step, 52% three-step)
COMPLETE:  Llama curriculum (100% / 57% / 22% on L0/L1/L2)
COMPLETE:  Hypersphere constraint validation

CURRENT:   Build asymmetric hourglass architecture
           GSM8K thinking accuracy > 35% (single-shot baseline)
           GSM8K thinking accuracy > 45% (+10 points)

FUTURE:    MATH L1-L5 (capacity extension on competition math)
```

---

## Key Insight

The transformer is SANDWICHED, not modified. It processes exactly as Llama was pretrained to process.

The **compressor** has the HARD job: "read everything the transformer thought across all 16 layers and squeeze the most important findings into 64 floats." This is why it gets 120M params.

The **decompressor** has the EASY job: "take 64 floats and project them into something the transformer can use." This is why it only needs 1.3M params.

The asymmetry is intentional: **compression is selection, decompression is projection.**

---

## Deadline

**MATH-500 benchmark: April 22, 2026**

Target: thinking accuracy > single-shot on Level 4-5 problems.

---

> *Think hard, compress tight, speak from understanding.*
