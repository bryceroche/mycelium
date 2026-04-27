# Mycelium — Breathing Models

Differentiable recurrent reasoning for small language models. A frozen 1B-parameter LLM that scores 0% on chained math learns to chain through a breathing loop — EXPAND in natural language, COLLAPSE to 64 floats, repeat. Each cycle: one breath, one pattern matched, one result computed, one page written.

**The thesis:** decomposition is everything. The model doesn't become smarter — it learns to break hard problems into pieces it can already solve.

**Lead:** Bryce Roche · **Target:** MATH-500 · **Deadline:** July 1, 2026

---

## Results

| Task | Base Model | With Breathing | Improvement |
|------|-----------|----------------|-------------|
| Single-step arithmetic | 70% | **100%** | 1.4x |
| Two-step arithmetic | 0% | **94.8%** | inf |
| Three-step arithmetic | 0% | **83.4%** | inf |
| Word operations | 0.6% | **53.4%** | 89x |
| Named quantities | 18.8% | **96.0%** | 5.1x |
| Two-step word problems | 6.0% | **91.0%** | 15.2x |
| Three-step word problems | — | **94.5%** | Growing notebook |
| **GSM8K grade school math** | **2.2%** | **training** | **BreathingController** |

**Key:** Llama 3.2 1B (frozen) + ~272M trainable parameters. The base model scores 0% on chained arithmetic. Our architecture provides the chaining through a differentiable compression loop.

---

## How It Works

### The Breathing Loop

```
         ┌─────────────────────────────────────────────────┐
         │                                                 │
         ▼                                                 │
    ┌─────────┐     ┌────────────┐     ┌──────────┐      │
    │  EXPAND  │────▶│  OBSERVE   │────▶│  RECORD  │──────┘
    │  (Llama) │     │(Controller)│     │ + PLAN   │
    └─────────┘     └────────────┘     └──────────┘
    
    EXPAND:   Llama reads problem with atom-modified attention
              Full natural language thinking (the inhale)
    
    OBSERVE:  Controller reads ALL 16 Llama layers
              One act of self-reflection (the exhale)
    
    RECORD:   Page head → 64 floats → append to notebook
    PLAN:     Scale head → 64 atom scales → next cycle's attention
```

Each cycle: think → observe → record + plan → think again. The system observes itself.

### The Pattern Library (64 LoRA Atoms)

64 rank-6 Low-Rank Adaptation atoms modify how Llama pays attention. They BLEND continuously — the controller outputs 64 scales and the attention modification is the weighted sum.

```
Low-frequency atoms:  broad patterns ("half as many in May" as one unit)
High-frequency atoms: fine patterns ("48" → a number)
The model takes the BIGGEST bite that fits per cycle.
```

### The Growing Notebook

Each cycle appends a FRESH 64-float page. No blending. No overwriting. The controller cross-attends over all previous pages — like flipping through a real notebook.

### Per-Cycle Intermediate Targets

Each cycle predicts ONE intermediate result, not the final answer.

```
Problem: "Jamie had 160 cookies, gave 63 away, got 20 more"

Cycle 1 → "Jamie had 160 cookies."                        → 160
Cycle 2 → "He gave away 63. 160 - 63 = 97 remaining."    → 97
Cycle 3 → "He got 20 more. 97 + 20 = 117 total."         → 117

Each cycle BREATHES: full natural sentence → 64 floats
```

---

## Architecture

```
Component                    Params      Role
──────────────────────────────────────────────────────────────
Llama 3.2 1B (frozen)        1,230M      Thinks (reads with modified attention)
BreathingController           190M       Observes (reads hidden states → page + scales)
64 LoRA Atoms (rank 6)         82M       Pattern library (attention modifications)
Confidence Head               2.5M       When to stop breathing
Mobius Transform                65K       Page diversity on hypersphere
──────────────────────────────────────────────────────────────
Total:                       ~1.50B
Trainable:                    ~272M
Frozen:                       1.23B
```

Two trainable pillars: **Controller** (observe) + **Atoms** (breathe).

---

## Three Principles

**Decomposition is everything.** Break hard problems into easy pieces. Each cycle handles one piece.

**Patterns are the vocabulary.** 64 atoms blend continuously — an infinite space of attention patterns. The controller navigates to the right blend per cycle.

**The system observes itself.** One controller reads what Llama computed, records understanding, and plans the next step. Record and plan from shared self-reflection.

---

## What's Next

```
1. GSM8K with BreathingController              ← IN PROGRESS
2. Confidence head (variable cycle count)
3. Gentle training wheel removal
4. MATH-500 benchmark (July 1 deadline)
```

---

## What's Novel

- **Differentiable decomposition** — the model learns to decompose through gradient descent
- **Unified self-observer** — one network produces both record (page) and plan (scales) from shared understanding
- **Growing notebook** — append-only compressed memory, no degradation with depth
- **64-atom continuous pattern library** — Fourier-initialized, no mode collapse, no softmax
- **Breathing rhythm** — expand in natural language, collapse through 64-float bottleneck
- **Per-cycle intermediate targets** — each cycle does one step, not the whole problem
- **Generation-only training** — no separate answer head, extraction from generated text
- **Number augmentation** — anti-memorization through per-epoch number randomization
- **Scale diversity loss** — direct pressure for cycle differentiation

---

See `CLAUDE.md` for full technical context, known bugs, training setup, and design decisions.
