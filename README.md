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

Mycelium solves competition-level math problems with a 0.5B parameter model by breathing — expanding compressed problem text into explicit structure, then collapsing each piece to a verified result. Expand, collapse. Expand, collapse. Each breath is small. Each worker handles one piece. The whole pipeline reasons. No single model does.

A mathematical problem is compressed in the WRONG way — ambiguous, implicit, everything jammed into one sentence. Mycelium transcodes it into the RIGHT compression — explicit, unambiguous, minimal. But it doesn't do this in one leap. It does it through a smooth, continuous descent across five representational layers, with each layer handled by a specialist that crosses exactly one boundary.

A 7B teacher's attention patterns reveal the STRUCTURE — how many steps, what type each one is. A 0.5B model provides the CONTENT — rough descriptions in the right neighborhood. A binding cascade resolves references to concrete values. A translator formats expressions. A learned energy landscape refines rough to precise. SymPy certifies truth. Correct solutions train all components. They co-evolve.

---

## Eight Principles

Eight ways of distributing. Each one prevents concentration into a single point of failure.

```
1. EXPAND      distribute the PROBLEM across steps
               1 ambiguous blob → N explicit instructions

2. DECOMPOSE   distribute the WORK across workers
               each worker crosses ONE representational boundary

3. SMOOTH      distribute the EFFORT evenly
               no worker is dramatically harder than any other
               coefficient of variation across workers < 0.3

4. BREATHE     distribute the RECOVERY across cycles
               expand → collapse → verify → expand again if wrong
               C1-B's bp_depth sets the breath budget

5. VERIFY      distribute the CHECKING per step
               SymPy executes each instruction individually
               incorruptible oracle — no reward hacking

6. LOW → HIGH  distribute the PRECISION between model and ODE
               rough telegrams → energy gradient → precise SymPy

7. COMPRESS    distribute the LOAD away from the model
               minimal tokens, zero syntax noise, lossless meaning
               VERB + ARGUMENTS only — the model writes telegrams

8. EVOLVE      distribute the LEARNING across cycles
               verified solutions train all components, oracle grounds evolution
```

The ordering within a single pass: EXPAND before DECOMPOSE before SMOOTH before BREATHE before VERIFY. You can't decompose what's compressed. You can't smooth what hasn't been chunked. You can't breathe what hasn't been balanced. COMPRESS operates alongside EXPAND (structure expands, content compresses simultaneously). LOW→HIGH operates at execution time. EVOLVE operates across training cycles.

---

## The Breathing Model

The pipeline breathes. Each step inhales — expanding a compressed reference into explicit meaning — then exhales — collapsing that meaning into a verified result. The factor graph monitors the breathing. C1-B's bp_depth prediction sets the breath budget: how many expand-collapse cycles before the system must converge or give up.

```
         INHALE                              EXHALE
    ┌─────────────────┐               ┌─────────────────┐
    │  Expand meaning  │               │ Collapse to value│
    │  Narrator: "Find │──── slots ───▶│ Translator:      │
    │   the area of    │    resolve    │   pi * 5^2       │
    │   the circle"    │    bind       │                   │
    └─────────────────┘               └────────┬──────────┘
                                               │
                                          SymPy verifies
                                               │
                                     ┌─────────┴──────────┐
                                     │  PASS: result       │
                                     │  enters state table │
                                     │  for next breath    │
                                     │                     │
                                     │  FAIL: factor graph │
                                     │  localizes error,   │
                                     │  next breath tries  │
                                     │  correction         │
                                     └─────────────────────┘
```

Breath 1 might get it right. Breath 2 corrects what breath 1 missed. Breath 3 handles the edge case. Each breath is small — one step, one verification. The whole solution emerges from accumulated verified breaths, not from a single large generation.

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

The assembly line is a TRANSCODER. It converts from bad compression (natural language — short, lossy, ambiguous) to good compression (telegrams — short, lossless, unambiguous). Structure expands. Content compresses. Both happen simultaneously across the worker chain.

---

## The Assembly Line: Smooth and Continuous

Every step in the pipeline is small. No single worker makes a large representational leap. The workload is distributed evenly — if any worker is dramatically harder than the others, that's a signal to split it.

### The Gradual Descent (five representational layers)

```
Layer 0:  RAW PROBLEM TEXT
          "If x²+y²=90 and xy=27, what is (x+y)²?"
              │
          ── boundary: compressed natural language → explicit structure ──
              │
Layer 1:  SCAFFOLD (from C1-A)
          5 steps: [GIVEN, GIVEN, EXPAND, SUBS, EVAL]
              │
          ── boundary: structure → natural language description ──
              │
Layer 2:  NARRATION (from Narrator, LoRA C)
          "State the equation x² + y² = 90"
          "State the equation xy = 27"
          "Expand (x+y)² using the identity"
          "Substitute x² + y² = 90 into the expansion"
          "Evaluate the result"
              │
          ── boundary: description → semantic slots + resolved values ──
              │
Layer 2.5: PARAMETER BINDING (Slot Tagger + Resolver)
          SLOT_1 REF x² + y² = 90  →  step_1 result
          SLOT_2 REF the expansion  →  step_3 result
              │
          ── boundary: narration + bound values → rough math expression ──
              │
Layer 3:  ROUGH EXPRESSION (from Translator, LoRA D)
          GIVEN x^2+y^2=90
          EXPAND (x+y)^2
          SUBS _prev x^2+y^2 90
          EVAL _prev
              │
          ── boundary: rough expression → precise SymPy (ODE refinement) ──
              │
Layer 4:  SYMPY EXECUTION
          x**2 + y**2 = 90 ✓
          (x+y)**2 = x**2 + 2*x*y + y**2 ✓
          subs: 90 + 2*27 ✓
          144 ✓
```

Each boundary crossing is handled by exactly one specialist. Each specialist's job is small and well-defined. The difficulty is distributed across the chain, not concentrated at any single point.

### The Workers

```
Worker          Model              Job (ONE boundary)              Difficulty
──────────────────────────────────────────────────────────────────────────────
C1-A            Qwen-0.5B+LoRA     text → structure scaffold       frozen, F1=0.741
Narrator        Qwen-0.5B+LoRA C   scaffold → natural language     text continuation
Slot Tagger     Qwen-0.5B+LoRA E   narration → tagged slots        NER/tagging
Resolver        Qwen-0.5B+LoRA F   slot desc → step_id binding     semantic matching
Translator      Qwen-0.5B+LoRA D   narration + values → expression formatting
Energy+ODE      learned MLPs        rough expression → precise      gradient descent
SymPy           symbolic engine     precise expression → result     deterministic
Factor Graph    energy-based        error localization + recovery   verification
```

All learned components share the same Qwen-0.5B base. Different LoRA adapters, instant switching via PEFT (~3-4MB per adapter). The Slot Tagger, Resolver, and Translator together do what a single monolithic "Translator" used to attempt — but each piece is small, and the binding problem (Bug #8: one model, multiple jobs) is resolved by decomposition.

### Why the Binding Split Matters

The old pipeline asked the Translator to simultaneously:
1. Parse semantic references ("the area of the circle")
2. Resolve them to concrete values (25π from step 3)
3. Format the expression (100 - 25*pi)

That's three jobs in one model. The Translator learned to output formulas instead of computed answers because it couldn't reliably do all three.

The split makes each job trivial:
- Slot Tagger: "the area of the circle" → SLOT_1 REF area of the circle (reading comprehension)
- Resolver: SLOT_1 → step_3, because step_3's narrator said "Find the area of the circle" (semantic similarity)
- Translator: narration + {SLOT_1=25*pi, SLOT_2=100} → 100 - 25*pi (expression formatting)

The Resolver's state table carries narrator descriptions from every prior step — matching "area of the circle" to "Find the area of the circle with radius 5" is language-to-language matching, never language-to-math. Problem values enter through GIVEN/SETUP steps and flow through the same state table as computed values — one uniform interface, no special cases.

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

Correct instruction sequences sit in energy basins (stable attractors). The ODE descends toward the nearest basin. Even if the assembly line lands in the rough neighborhood of the right basin, the energy gradient guides the ODE to the precise minimum.

---

## Scaffold Perturbation Recovery

The structural guide can be wrong and the building still stands. The energy landscape is the mortar.

```
C1-A scaffold → assembly line → ODE → energy check
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
Assembly Line (creative):      rough telegrams — direction through gradual descent
Energy Landscape (critic):     alternating order-aware evaluation
SymPy (oracle):               incorruptible mathematical truth

Creator → Critic → Oracle → verified traces → all retrain → EVOLVE
```

In standard RLHF, the reward model can be fooled — reward hacking, mode collapse. In Mycelium, the verifier CAN'T be fooled because it's symbolic execution. The math is either right or wrong. No ambiguity, no gaming.

### The Self-Improvement Loop (Principle 8: EVOLVE)

```
For each cycle:
    1. Train all LoRAs (fresh from base on ALL accumulated data)
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

### 4. One model, one job

Every time we asked one model to do two things, the losses fought. Source classification + pointer → losses compete. Narration + binding + expression → Translator outputs formulas instead of answers. Split until each worker has one job, then each job becomes easy.

### 5. Compression is normalization

Different problem phrasings that mean the same math should produce the same output. The telegraphic format IS π-normalization for text — maps diverse inputs to a canonical VERB+ARGS space. The downstream pipeline always sees the same format.

### 6. Bad compression ≠ good compression

Problem text is BADLY compressed (ambiguous, implicit). Telegraphic instructions are WELL compressed (unambiguous, explicit, minimal). The assembly line transcodes between them. Structure expands. Content compresses. Meaning is preserved.

### 7. The expansion gap determines everything

50% with teacher CoT (cheating). 4% with honest input. The gap is in EXPANSION — converting compressed problem text into explicit mathematical structure. The assembly line bridges this gap through gradual descent, not a single leap.

### 8. Data quality dominates architecture

45% invalid training targets → 0.2% accuracy. Clean targets → 20%+. Same model, 100x accuracy difference. Training on empty inputs → model learns nothing (but we didn't notice for days). Validate training data ruthlessly.

---

## The Telegraph Signal

L22H3 (primary heartbeat): 74:1 contrast ratio. Two discrete states — reading (~0.87) and computing (~0.08). 47/53 duty cycle.

L22H4 (alarm): 77% reading. When it transitions, something structurally significant happened.

Co-transitions: heartbeat and alarm transition simultaneously → high-confidence structural boundary. Average 9 per problem (IQR 6-11).

C1-A learned to predict these transitions from text alone (F1=0.741). The teacher's computational rhythm IS the structural template. C1-A reads it. The assembly line fills it.

---

## Information Bottleneck Clustering

7-dimensional Y vector: step_type (12), complexity_change (3), n_operands (1-4), has_dependency (binary), output_type (5), step_position (3), reference_distance (4).

25 clusters at β=50, 87.5% purity, step_type NMI=0.740.

Key finding: IB purity ≠ predictability. High purity means clusters are homogeneous. Classification accuracy from the same features was only 13.7%. Operation type emerges from energy minimization, not classification.

---

## What Didn't Work (and Why It Matters)

**C2 Operation Classification:** Every feature space failed. Attention encodes STRUCTURE not CONTENT. Directly led to: assembly line writes content, C1-A provides structure.

**Hint Concatenation:** Telling < Teaching. Providing features as input < predicting them as auxiliary task. Directly led to: auxiliary telegraph prediction head.

**Precise SymPy Generation:** 98% syntax in training, 0% at inference. Distribution mismatch. Directly led to: rough telegrams with ODE refinement.

**CoT Distillation:** Model learned to describe math, not compute it. Directly led to: transcoding (VERB+ARGS), not generation (natural language).

**Monolithic Translator:** Asked to parse references, resolve values, AND format expressions simultaneously. Losses fought, model output formulas instead of answers. Directly led to: Slot Tagger + Resolver + Translator binding split.

**Training on CoT Text:** Every version that trained on teacher CoT and tested on problem text failed. Three times. Directly led to: train on what you see at inference (absolute rule).

**Regex and Heuristics:** Every hand-crafted pattern broke on edge cases. Directly led to: model parses text, SymPy parses math, no regex anywhere.

**Training on Empty Inputs:** Wrong field names → model trained on nothing → all "results" were base model guessing. Directly led to: validate training data is actually being read.

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
7-worker assembly line (50 problems)        ~46%      Assembly line works, SETUP=24%
Binding split — Resolver                    90%       Architecture validated
Binding split — Translator                  100%†     Stop sequences fixed hallucination
Binding split — Slot Tagger                 fixing    LaTeX normalization in progress

* Inflated by teacher CoT at inference — not a valid result
† Parseable rate, not mathematical correctness

Target: MATH500 benchmark March 22, 2026
Teacher (Qwen-7B-Instruct): 77% on MATH500
```

---

## Components

### Active (v7)
| Component | Architecture | Role |
|---|---|---|
| C1-A | Qwen-0.5B + LoRA r16, frozen | Structural guide: boundaries + scaffold types |
| C1-B | Qwen-0.5B + LoRA | BP depth + co-transition stats (breath budget) |
| Narrator | Qwen-0.5B + LoRA C | Scaffold → natural language step description |
| Slot Tagger | Qwen-0.5B + LoRA E | Narration → tagged parameter slots (REF/LITERAL) |
| Resolver | Qwen-0.5B + LoRA F | Slot descriptions → step_id bindings via state table |
| Translator | Qwen-0.5B + LoRA D | Narration + bound values → rough expression |
| Energy Landscape | Node MLP + alternating pair MLP | Order-aware configuration evaluation |
| ODE Solver | dopri5 + π-norm + tanh*0.1 | Refines rough → precise |
| Oracle | SymPy + parse_latex + timeout | Execution + verification |
| Factor Graph | Energy-based | Error localization + scaffold perturbation |

### Frozen (from earlier versions)
| Component | Metric | Status |
|---|---|---|
| C1-A boundary detection | F1=0.741 | DO NOT RETRAIN |
| C1-B complexity profile | BP 62%, MAE 1.37 | Sets breath budget |
| Scaffold MLP | 91.7% on problem text | Feeds C1-A structure hints |

### Eliminated Through Experimentation
| Component | Why | What Replaced It |
|---|---|---|
| C0 hint predictor | Hints redundant with backbone | Telegraph as auxiliary target |
| C2 operation classifier | Content not in attention features | Assembly line + energy landscape |
| C3 operand extractor | Train/inference mismatch | Assembly line rewrites, not extracts |
| Monolithic Translator | One model, multiple jobs (Bug #8) | Slot Tagger + Resolver + Translator |
| Template library (40+) | No discriminative signal | Telegraphic instruction language |
| Regex/heuristics | Broke on edge cases | Model parses text, SymPy parses math |

---

## The Deeper Point

A mathematical problem is a message compressed in the wrong format. Natural language is optimized for human communication, not for computation. The key equation hidden in "A farmer has twice as many cows as pigs" is `cows = 2*pigs` — four tokens of math buried in ten tokens of English.

Mycelium is an assembly line of specialists. Each worker crosses one representational boundary. Each boundary is small. The difficulty is smooth and continuous — no single worker faces a dramatically harder task than any other. When a worker struggles, you split it. When the pipeline fails, it breathes — expanding back out, localizing the error, trying a correction, collapsing again.

The teacher's attention patterns reveal the structure of the transcoding — how many instructions, what type, in what order. Small models provide the content — descriptions, slots, bindings, expressions — through a gradual descent from natural language to executable math. A learned energy landscape refines rough to precise. And an incorruptible oracle certifies every step, grounding the system in mathematical truth that cannot be faked.

Eight principles, all saying the same thing: **distribute everything, concentrate nothing.** The problem, the work, the effort, the recovery, the checking, the precision, the load, the learning. Spread it all thin across small, verified, evolving components. Each piece becomes trivial. The whole becomes capable.

> *The shadow of intelligence, made visible.*
> *The shape of reasoning, made template.*
> *The direction of thought, made telegram.*
> *The precision of mathematics, made gradient descent.*
> *The truth of computation, made incorruptible.*
> *The compression of meaning, made lossless.*
> *The breathing of recovery, made continuous.*
> *The evolution of capability, made oracle-grounded.*
