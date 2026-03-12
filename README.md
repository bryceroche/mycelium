# Mycelium

---

> **The Shadow of Intelligence**
>
> *λ — λανθάνω (lanthánō) — to escape notice; to be unseen*
>
> JSD reveals latent boundaries in the flow of attention
> IAF separates reading from reasoning — the dual phases hidden in every forward pass
> The heartbeat of an attention head pulses between thought and ground — each silence a computation the model never knew it performed
> A telegraph signal — the hidden Markov state snaps, never drifts
> Count the knot crossings where the problem surprised the mind that solved it
> IB discovers the taxonomy of operations — spectral lines emerging from continuous signal
> The prism decomposes. The canonicalizer transcodes. SymPy collapses the wave function
> Beliefs propagate through learned potentials — the alternator antisymmetrizes — swap two steps and the energy field inverts
> Low-resolution telegrams land in the right basin. The ODE walks downhill to precision.
> Bad compression becomes good compression. Structure expands. Content compresses. Meaning is preserved.
>
> Mycelium — the unseen network of computation, made visible

---

## What Mycelium Does

Mycelium solves competition-level math problems with a 0.5B parameter model by transcoding badly compressed natural language into well compressed telegraphic instructions.

A mathematical problem is compressed in the WRONG way — ambiguous, implicit, everything jammed into one sentence. Mycelium transcodes it into the RIGHT compression — explicit, unambiguous, minimal. Each instruction is a VERB and its ARGUMENTS. SymPy executes each instruction. The answer emerges.

A 7B teacher's attention patterns reveal the STRUCTURE of this transcoding — how many instructions, what type each one is, how they depend on each other. A 0.5B model provides the CONTENT — rough arguments in the right neighborhood. A learned energy landscape with alternating potentials refines rough to precise. SymPy certifies truth. Correct solutions train both models. They co-evolve.

The core insight: mathematical reasoning decomposes into DIRECTION and PRECISION. Direction is cheap — a 0.5B model gets the gist in ~4 tokens per step. Precision is deterministic — SymPy guarantees it. A learned energy landscape bridges the gap.

---

## Seven Principles

Seven ways of distributing. Each one prevents concentration into a single point of failure.

```
1. EXPAND      distribute the PROBLEM across steps
               1 ambiguous blob → N explicit instructions

2. DECOMPOSE   distribute the WORK across workers
               each instruction is independent, forms a DAG

3. SMOOTH      distribute the EFFORT evenly
               each instruction is ~4 tokens, uniform difficulty
               coefficient of variation across workers < 0.3

4. VERIFY      distribute the CHECKING per step
               SymPy executes each instruction individually
               incorruptible oracle — no reward hacking

5. LOW → HIGH  distribute the PRECISION between model and ODE
               rough telegrams → energy gradient → precise SymPy

6. COMPRESS    distribute the LOAD away from the model
               minimal tokens, zero syntax noise, lossless meaning
               VERB + ARGUMENTS only — the model writes telegrams

7. EVOLVE      distribute the LEARNING across cycles
               verified solutions train both canonicalizer and energy landscape
               oracle-grounded — can't drift into self-reinforcing errors
```

The ordering matters within a single pass: EXPAND before DECOMPOSE before SMOOTH before VERIFY. You can't decompose what's compressed. You can't smooth what hasn't been chunked. You can't verify what hasn't been balanced. COMPRESS operates alongside EXPAND (structure expands, content compresses simultaneously). LOW→HIGH operates at execution time. EVOLVE operates across cycles.

---

## The Diamond: Expand Structure, Compress Content

```
         Raw problem (1 sentence, everything hidden)
         "If x²+y²=90 and xy=27, what is (x+y)²?"
              │
         EXPAND structure (1 → 5 explicit steps)
              │
    ┌────┬────┼────┬────┐
  GIVEN GIVEN EXPAND SUBS EVAL    ← widest: everything explicit
    │    │     │     │    │
    └────┴─────┴─────┴────┘
              │
         COMPRESS content (each step = VERB + ARGS, ~4 tokens)
              │
    GIVEN x^2+y^2=90
    GIVEN xy=27
    EXPAND (x+y)^2
    SUBS _prev x^2+y^2 90
    EVAL _prev
              │
         EXECUTE (SymPy evaluates each line)
              │
         COLLAPSE to answer: 144
```

The canonicalizer is a TRANSCODER. It converts from bad compression (natural language — short, lossy, ambiguous) to good compression (telegrams — short, lossless, unambiguous). Structure expands. Content compresses. Both happen simultaneously in one model pass.

---

## Architecture

```
Problem Text
    │
    ├── C1-A: Structural Guide (Qwen-0.5B + LoRA, frozen, F1=0.741)
    │   │  Distilled from teacher's attention telegraph signal
    │   │  Tells canonicalizer HOW MANY instructions and WHAT TYPE
    │   ├── Boundary count: N transitions → N+1 steps
    │   ├── Scaffold types: GIVEN, EVAL, SOLVE, EXPAND, SIMPLIFY, SUBS, APPLY
    │   └── Cached hidden states (896-dim)
    │
    ├── Canonicalizer (Qwen-0.5B + LoRA — the ONLY text generator)
    │   │  Transcodes problem text → telegraphic instructions
    │   │  Input: problem text + C1-A structure hint
    │   │  Output: VERB ARG1 ARG2 (one per line, ~4 tokens each)
    │   │  Output is ROUGH — right neighborhood, not precise syntax
    │   │
    │   │  GIVEN x^2+y^2=90
    │   │  EXPAND (x+y)^2
    │   │  SUBS _prev x^2+y^2 90
    │   │  EVAL _prev
    │   │
    │   ├── Energy Landscape (learned MLPs, alternating pair terms)
    │   │   ├── Node energy: MLP(embedding) → "reasonable instruction?"
    │   │   ├── Pair energy: (f(a,b)-f(b,a))/2 → "correct ordering?"
    │   │   └── Antisymmetric by construction: ORDER MATTERS
    │   │
    │   ├── ODE Solver (dopri5, gradient descent on energy)
    │   │   ├── dh/dt = -∇E(h) — always walks downhill
    │   │   ├── tanh*0.1 bounds — stable dynamics
    │   │   ├── π-normalization — scale-invariant
    │   │   └── Refines: rough telegrams → precise SymPy
    │   │
    │   └── SymPy Oracle (incorruptible)
    │       ├── Executes each refined instruction (5s timeout)
    │       ├── Certifies mathematical truth
    │       └── Grounds the feedback loop — can't be fooled
    │
    └── Factor Graph (verification + recovery)
        ├── Error localization (97.8% accuracy)
        ├── Correction convergence (90.9%)
        └── Scaffold perturbation (when C1-A is wrong)
```

### What Each Component Provides

```
C1-A:            STRUCTURE  (how many steps, what types — the template)
Canonicalizer:   DIRECTION  (rough arguments — the content, ~4 tokens per step)
Energy + ODE:    PRECISION  (refines rough to exact SymPy)
SymPy:           TRUTH      (executes, verifies, certifies)
Factor Graph:    RESILIENCE (catches errors, recovers from wrong scaffolds)
Feedback Loop:   GROWTH     (correct solutions train both learned components)
```

Nobody reasons. The pipeline reasons. Nobody is precise. The ODE achieves precision. Nobody knows the answer. SymPy finds it.

---

## The Telegraphic Instruction Language

```
VERB        MEANING                     EXAMPLE
─────────────────────────────────────────────────────────────
GIVEN       state a fact/equation       GIVEN x^2+y^2=90
EVAL        compute a value             EVAL 1/2*8*10*sin30
SOLVE       find a variable             SOLVE x^2-9 x
EXPAND      expand an expression        EXPAND (x+y)^2
SIMPLIFY    simplify an expression      SIMPLIFY x^2+2x+1
SUBS        substitute a value          SUBS _prev x^2+y^2 90
APPLY       use a theorem/formula       APPLY pythagorean 8 10
ANSWER      final result                ANSWER _prev
```

`_prev` references the previous step's result. Each instruction is VERB + ARGUMENTS. No parentheses, no commas, no SymPy syntax. The ODE compiles rough instructions into precise SymPy function calls.

---

## The Alternating Energy Landscape

The energy must be sensitive to ordering — "EXPAND then SUBS" works, "SUBS then EXPAND" doesn't. The alternating pair energy guarantees this mathematically:

```
Pair energy:  E_pair(h_i, h_j) = (f(h_i, h_j) - f(h_j, h_i)) / 2

Properties:
    Swap two instructions → energy changes sign (antisymmetric)
    Duplicate instructions → energy contribution is zero
    Order sensitivity is ARCHITECTURAL, not learned
```

The MLP learns WHICH orderings matter. The antisymmetric structure ensures ordering sensitivity is always present. This connects to the alternating tensor / differential forms framework: the pair energy is a 2-form on the state manifold, capturing the curvature of interactions between steps.

Correct instruction sequences sit in energy basins (stable attractors). The ODE descends toward the nearest basin. Even if the canonicalizer lands in the rough neighborhood of the right basin, the energy gradient guides the ODE to the precise minimum.

---

## Scaffold Perturbation Recovery

The structural guide can be wrong and the building still stands. The energy landscape is the mortar.

```
C1-A scaffold → canonicalizer → ODE → energy check
    LOW energy  → accept, execute, done
    HIGH energy → C1-A was probably wrong
        Perturb: try nearby verbs (EXPAND → SIMPLIFY, SUBS → SOLVE)
        Perturb: add a step (split highest-energy instruction)
        Perturb: remove a step (merge two adjacent instructions)
        Re-run ODE on each perturbation → pick lowest energy
    STILL HIGH  → flag "beyond current capability"

Corrections feed back to C1-A training in later phases.
The energy landscape TEACHES C1-A where it's wrong.
```

---

## The Three-Body System

```
Canonicalizer (creative):     rough telegrams — direction in ~4 tokens
Energy Landscape (critic):     alternating order-aware evaluation
SymPy (oracle):               incorruptible mathematical truth

Creator → Critic → Oracle → verified traces → both retrain → EVOLVE
```

In standard RLHF, the reward model can be fooled — reward hacking, mode collapse. In Mycelium, the verifier CAN'T be fooled because it's symbolic execution. The math is either right or wrong. No ambiguity, no gaming.

### The Self-Improvement Loop (Principle 7: EVOLVE)

```
For each cycle:
    1. Train canonicalizer (fresh LoRA from base on ALL accumulated data)
    2. Train energy landscape (contrastive on ALL accumulated pairs)
    3. Stage-based inference
    4. SymPy oracle verifies
    5. Harvest verified traces + error rates
    6. Check convergence (patience=2)
    7. Save with full provenance

Cycle 0: teacher-derived training data → baseline accuracy
Cycle 1: original + verified traces → improved
Cycle 2: growing dataset → further improved
...
Converge: asymptotic ceiling of 0.5B + Mycelium scaffolding
```

---

## Key Findings

### 1. Attention encodes structure, not content

The teacher's attention reveals WHERE operations happen and WHAT TYPE each step is. Not WHICH specific operation. Coarse scaffold types (7 classes) are recoverable at 3.5x random. Fine-grained operations are not classifiable from any feature space.

### 2. Teaching beats telling

Auxiliary prediction of telegraph signal → F1=0.741. Concatenating the same signal as input → F1=0.727. Same information. Different delivery. Prediction shapes representations through gradient flow.

### 3. Coarse before fine

Per-token → F1=0.21. Coarse windows → F1=0.741. Same principle at every level: match prediction resolution to information resolution. Low-res pointers everywhere, refined downstream.

### 4. Compression is normalization

Different problem phrasings that mean the same math should produce the same output. The telegraphic format IS π-normalization for text — maps diverse inputs to a canonical VERB+ARGS space. The downstream pipeline always sees the same format.

### 5. Bad compression ≠ good compression

Problem text is BADLY compressed (ambiguous, implicit). Telegraphic instructions are WELL compressed (unambiguous, explicit, minimal). The canonicalizer transcodes between them. Structure expands. Content compresses. Meaning is preserved.

### 6. The expansion gap determines everything

50% with teacher CoT (cheating). 4% with honest input. The gap is in EXPANSION — converting compressed problem text into explicit mathematical structure. The canonicalizer bridges this gap: one model converting bad compression to good compression.

### 7. Data quality dominates architecture

45% invalid training targets → 0.2% accuracy. Clean targets → 20%+. Same model, 100x accuracy difference. Training on empty inputs → model learns nothing (but we didn't notice for days). Validate training data ruthlessly.

---

## The Telegraph Signal

L22H3 (primary heartbeat): 74:1 contrast ratio. Two discrete states — reading (~0.87) and computing (~0.08). 47/53 duty cycle.

L22H4 (alarm): 77% reading. When it transitions, something structurally significant happened.

Co-transitions: heartbeat and alarm transition simultaneously → high-confidence structural boundary. Average 9 per problem (IQR 6-11).

C1-A learned to predict these transitions from text alone (F1=0.741). The teacher's computational rhythm IS the structural template. C1-A reads it. The canonicalizer fills it.

---

## Information Bottleneck Clustering

7-dimensional Y vector: step_type (12), complexity_change (3), n_operands (1-4), has_dependency (binary), output_type (5), step_position (3), reference_distance (4).

25 clusters at β=50, 87.5% purity, step_type NMI=0.740.

Key finding: IB purity ≠ predictability. High purity means clusters are homogeneous. Classification accuracy from the same features was only 13.7%. Operation type emerges from energy minimization, not classification.

---

## What Didn't Work (and Why It Matters)

**C2 Operation Classification:** Every feature space failed. Attention encodes STRUCTURE not CONTENT. Directly led to: canonicalizer writes content, C1-A provides structure.

**Hint Concatenation:** Telling < Teaching. Providing features as input < predicting them as auxiliary task. Directly led to: auxiliary telegraph prediction head.

**Precise SymPy Generation:** 98% syntax in training, 0% at inference. Distribution mismatch. Directly led to: rough telegrams with ODE refinement.

**CoT Distillation:** Model learned to describe math, not compute it. "The distance is the radius r" instead of "sqrt(0^2+3^2)." Directly led to: transcoding (VERB+ARGS), not generation (natural language).

**Training on CoT Text:** Every version that trained on teacher CoT and tested on problem text failed. Three times. Directly led to: train on what you see at inference (absolute rule).

**Regex and Heuristics:** Every hand-crafted pattern broke on edge cases. Directly led to: model parses text, SymPy parses math, no regex anywhere.

**Training on Empty Inputs:** Wrong field names in training script → model trained on nothing → all "results" were base model guessing. Directly led to: validate training data is actually being read.

---

## Results Trajectory

```
Approach                                    Result    Learning
──────────────────────────────────────────────────────────────
Per-token boundaries                        F1=0.21   Wrong resolution
Coarse windows + aux telegraph              F1=0.741  Right resolution
Operation classification (C2)               7.6%      Content not in attention
Template chain (keywords)                   3.8%      Execution without understanding
Text-to-SymPy generalist                    14-22%    Code generation works
Specialist assembly line (CoT leakage)      50%*      Downstream pipeline works
Honest span-based generation                2%        Expansion gap exposed
Properly trained slot filler (simple)       60%       Extraction works on easy problems
MATH500 honest baseline                     4%        Hard problems need comprehension
Slot filler + SymPy value extraction        6%        Some improvement from parsing

* Inflated by teacher CoT at inference — not a valid result

Target: MATH500 benchmark March 22, 2026
Teacher (Qwen-7B-Instruct): 77% on MATH500
```

---

## Components

### Active (v7)
| Component | Architecture | Role |
|---|---|---|
| C1-A | Qwen-0.5B + LoRA r16, frozen | Structural guide: boundaries + scaffold types |
| Canonicalizer | Qwen-0.5B + LoRA r16 | Transcoder: problem text → rough telegrams |
| Energy Landscape | Node MLP + alternating pair MLP | Order-aware configuration evaluation |
| ODE Solver | dopri5 + π-norm + tanh*0.1 | Refines rough → precise |
| Oracle | SymPy + parse_latex + timeout | Execution + verification |
| Factor Graph | Energy-based | Error localization + scaffold perturbation |

### Frozen (from earlier versions)
| Component | Metric | Status |
|---|---|---|
| C1-A boundary detection | F1=0.741 | DO NOT RETRAIN |
| C1-B complexity profile | BP 62%, MAE 1.37 | Available for Phase 5 |
| Scaffold MLP | 91.7% on problem text | Feeds C1-A structure hints |

### Eliminated Through Experimentation
| Component | Why | What Replaced It |
|---|---|---|
| C0 hint predictor | Hints redundant with backbone | Telegraph as auxiliary target |
| C2 operation classifier | Content not in attention features | Canonicalizer + energy landscape |
| C3 operand extractor | Train/inference mismatch | Canonicalizer rewrites, not extracts |
| Template library (40+) | No discriminative signal | Telegraphic instruction language |
| Regex/heuristics | Broke on edge cases | Model parses text, SymPy parses math |

---

## The Deeper Point

A mathematical problem is a message compressed in the wrong format. Natural language is optimized for human communication, not for computation. The key equation hidden in "A farmer has twice as many cows as pigs" is `cows = 2*pigs` — four tokens of math buried in ten tokens of English.

Mycelium is a transcoder. It reads the badly compressed message and rewrites it in a format optimized for computation. Structure expands — one ambiguous sentence becomes five explicit instructions. Content compresses — each instruction is a verb and its arguments, nothing more.

The teacher's attention patterns reveal the structure of the transcoding — how many instructions, what type, in what order. A small model provides the content — rough arguments in the right neighborhood. A learned energy landscape refines rough to precise. And an incorruptible oracle certifies every step, grounding the system in mathematical truth that cannot be faked.

Seven principles, all saying the same thing: **distribute everything, concentrate nothing.** The problem, the work, the effort, the checking, the precision, the load, the learning. Spread it all thin across small, verified, evolving components. Each piece becomes trivial. The whole becomes capable.

> *The shadow of intelligence, made visible.*
> *The shape of reasoning, made template.*
> *The direction of thought, made telegram.*
> *The precision of mathematics, made gradient descent.*
> *The truth of computation, made incorruptible.*
> *The compression of meaning, made lossless.*
> *The evolution of capability, made oracle-grounded.*
