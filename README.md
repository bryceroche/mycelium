# Mycelium — Breathing Models

Differentiable recurrent reasoning for small language models. A frozen 1B-parameter LLM that scores 0% on chained math learns to chain through a breathing loop — EXPAND in natural language, COLLAPSE to 64 floats, repeat. Each cycle matches one pattern, computes one step, writes one page in a growing notebook.

**The thesis:** decomposition is everything. The model doesn't become smarter — it learns to break hard problems into pieces it can already solve.

**Lead:** Bryce Roche · **Target:** MATH-500 · **Deadline:** July 1, 2026

---

## Results

| Task | Base Model | With Breathing | Improvement |
|------|-----------|----------------|-------------|
| Single-step arithmetic | 70% | **100%** | 1.4x |
| Two-step arithmetic | 0% | **94.8%** | ∞ |
| Three-step arithmetic | 0% | **83.4%** | ∞ |
| Word operations | 0.6% | **53.4%** | 89x |
| Named quantities | 18.8% | **96.0%** | 5.1x |
| **Two-step word problems** | **6.0%** | **91.0%** | **15.2x** |
| **Three-step word problems** | — | **94.5%** | **Growing notebook** |
| Four-step word problems | — | *training* | In progress |
| GSM8K grade school math | 2.2% | **17.8%** | 8.1x |

**Key:** Llama 3.2 1B (frozen) + 203M trainable parameters. The base model scores 0% on chained arithmetic. Our architecture provides the chaining through a differentiable compression loop.

---

## How It Works

### The Breathing Loop

```
         ┌─────────────────────────────────────────────────┐
         │                                                 │
         ▼                                                 │
    ┌─────────┐     ┌───────────┐     ┌──────────┐       │
    │  EXPAND  │────▶│  COLLAPSE │────▶│  RECORD  │───────┘
    │  (Llama) │     │(Perceiver)│     │(Notebook) │
    └─────────┘     └───────────┘     └──────────┘
    
    EXPAND:   Llama reads problem with atom-modified attention
              Full natural language thinking (the inhale)
    
    COLLAPSE: Perceiver compresses 16 layers → 64 floats
              Ruthless compression through bottleneck (the exhale)
    
    RECORD:   Fresh page appended to growing notebook
              Nothing overwritten. Perfect recall. Notebook grows.
```

Each cycle: one breath, one pattern matched, one result computed, one page written.

### The Pattern Library (64 LoRA Atoms)

64 rank-6 Low-Rank Adaptation atoms modify how Llama pays attention. They BLEND continuously — the hypernetwork outputs 64 scales and the attention modification is the weighted sum. The 64-dimensional atom space is a continuous manifold of attention patterns. Each point is a unique pattern recognizer.

```
Atom blends are like color mixing:
  Atom 0 + Atom 5 = a composite pattern neither produces alone
  (red + blue = purple — a new color from blending primaries)
  
Low-frequency atoms:  broad patterns ("half as many in May" as one unit)
High-frequency atoms: fine patterns ("48" → a number)
The model takes the BIGGEST bite that fits per cycle.
```

### The Growing Notebook

Each cycle appends a FRESH 64-float page. No blending. No overwriting. The hypernetwork reads all pages via cross-attention — like flipping through a real notebook. Information from cycle 1 is preserved perfectly at cycle 8.

```
Cycle 1: perceiver → page_1 → append     notebook = [page_1]
Cycle 2: perceiver → page_2 → append     notebook = [page_1, page_2]
Cycle 3: perceiver → page_3 → append     notebook = [page_1, page_2, page_3]
                                          
Hypernetwork cross-attends over ALL pages → selects next atom blend
Answer head reads each cycle's OWN page → extracts the intermediate result
```

### Per-Cycle Intermediate Targets

Each cycle predicts ONE intermediate result, not the final answer. Cycle 1 extracts the first number. Cycle 2 computes the first operation. Cycle 3 computes the second. The model takes one bite per cycle.

```
Problem: "Jamie had 160 cookies, gave 63 away, got 20 more"

Cycle 1 → "Jamie had 160 cookies."                        → 160
Cycle 2 → "He gave away 63. 160 - 63 = 97 remaining."    → 97
Cycle 3 → "He got 20 more. 97 + 20 = 117 total."         → 117

Each cycle BREATHES: full natural sentence (expand) → 64 floats (collapse)
```

---

## Architecture

```
Component                    Params      Role
──────────────────────────────────────────────────────────────
Llama 3.2 1B (frozen)        1,230M      Comprehends (reads with modified attention)
7-Layer Perceiver             105M       Collapses (16 layers → 64 floats)
64 LoRA Atoms (rank 6)         82M       Pattern library (Fourier-initialized)
Atom Hypernetwork              10M       Pattern matcher (navigates atom space)
Answer Head                   100K       Verifier (digit prediction per cycle)
Cycle Message Generator       533K       Direct bypass (16 floats, no compression)
Page-to-Tokens                0.5M       Generation bridge
Confidence Head                79K       Stop decision (when done thinking)
──────────────────────────────────────────────────────────────
Total:                       ~1.43B
Trainable:                    ~203M     (14.2%)
Frozen:                       1.23B     (85.8%)
Notebook:                     64N floats (grows with cycles)
```

---

## Six Breakthroughs

1. **Per-cycle intermediate targets** — each cycle does ONE job. CoT targets made cycles redundant.
2. **Hybrid generation loss** — 1000x gradient to atoms. Answer head alone: 0.0002 gradient (starving).
3. **Page delta for answer head** — broke 5% → 89% ceiling. Cycle 2 was copying cycle 1 (60% of errors).
4. **Text injection** — Llama needs TEXT, not continuous vectors. "Step 1 result: 160\n" as actual tokens.
5. **Natural sentence targets** — the model must BREATHE. Terse targets suppressed expansion.
6. **Growing notebook** — remove residual gate, append fresh pages. Cycle 2 hit 85.5% on epoch 1.

---

## Three Principles

**Decomposition is everything.** Break hard problems into easy pieces. Each cycle handles one piece.

**Patterns are the vocabulary.** 64 atoms blend continuously — an infinite space of attention patterns. The hypernetwork navigates to the right blend per cycle.

**Match the largest pattern that fits.** The Panama hat principle. "Half as many clips in May" is one pattern, not five words. Bigger bites = fewer cycles = less error compounding.

---

## What's Next

```
1. L4.7 (4-step) → L4.9 (5-step) curriculum     ← in progress
2. GSM8K per-cycle decomposition (7,473 problems)
3. Confidence head (variable cycle count per problem)
4. Gentle training wheel removal (self-decomposition)
5. MATH-500 benchmark (July 1 deadline)
```

---

## What's Novel

- **Differentiable decomposition** — the model learns to decompose through gradient descent
- **Growing notebook** — append-only compressed memory, no degradation with depth
- **64-atom continuous pattern library** — Fourier-initialized, no mode collapse, no softmax
- **Breathing rhythm** — expand in natural language, collapse through 64-float bottleneck
- **Per-cycle intermediate targets** — each cycle does one step, not the whole problem
- **Page delta** (for blended architectures) — one line fix, 5% → 89%
- **Text injection** — bridge between compressed state and LLM's native text format

---

See `CLAUDE.md` for full technical context, known bugs, training setup, and design decisions.
