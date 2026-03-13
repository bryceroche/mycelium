# The Gradual Descent
# From Natural Language to Symbolic Execution in Small Steps

---

## The Problem: One Big Jump Fails

Every failed approach in Mycelium's history asked a small model to make
a large representational jump in a single step:

```
v2 seq2seq:         "She sold half her clips" → Mul(48, Rational(1,2))
v6 C3 extraction:   "half the price of cheese" → extract span → SymPy
v7 canonicalizer:   "How many divisors?" → factorint(196)
v7 operand split:   "distance between (2,-6) and (-4,3)" → [2, -6, -4, 3]
```

Each jump crosses multiple representational boundaries at once:
natural language → mathematical semantics → symbolic notation → executable code.

A 0.5B model can't bridge that gap. It learns the *form* of the target
(valid-looking SymPy) without the *substance* (correct mathematics).

---

## The Solution: Many Small Steps

Instead of one jump, descend gradually. Each step crosses ONE
representational boundary. Each step is a natural text continuation
task — what LMs actually do well.

```
LAYER 0: Raw Problem (fully natural language, everything implicit)
         "If x² + y² = 90 and xy = 27, what is the value of (x+y)²?"

    ↓ C1-A: structural scaffolding (learned from attention heartbeat)
    
LAYER 1: Scaffold (structure explicit, content still implicit)
         5 steps: [SETUP, SETUP, EXPAND, SUBSTITUTE, EVALUATE]
         Boundary count reveals complexity. Classification reveals intent.
         
    ↓ Narrator (LoRA C): describe each step in natural language
    
LAYER 2: Narration (intent explicit, values still in natural language)
         Step 1: "State that x² + y² equals 90"
         Step 2: "State that xy equals 27"
         Step 3: "Expand (x+y)² using the identity"
         Step 4: "Substitute the known values x² + y² = 90 and xy = 27"
         Step 5: "Evaluate the final expression"
         
    ↓ Translator (LoRA D): convert description to rough math
    
LAYER 3: Rough Expression (math explicit, notation still rough)
         Step 1: GIVEN x^2+y^2=90
         Step 2: GIVEN xy=27
         Step 3: EXPAND (x+y)^2
         Step 4: SUBS _prev x^2+y^2 90 2*xy 54
         Step 5: EVAL 90+54
         
    ↓ Oracle: SymPy parses and executes (relaxed parsing)
    
LAYER 4: Precise Execution (symbolic, verified, incorruptible)
         Step 1: Eq(x**2 + y**2, 90)         ✓
         Step 2: Eq(x*y, 27)                  ✓
         Step 3: x**2 + 2*x*y + y**2          ✓
         Step 4: 90 + 2*27                     ✓
         Step 5: 144                           ✓ → answer
```

---

## The Representational Boundaries

Five layers, four boundaries. Each boundary is ONE transformation.

```
    LAYER 0                    LAYER 1
    ─────────                  ─────────
    Raw problem         →      Scaffold
    "everything hidden"        "structure visible"
    
    BOUNDARY: C1-A extracts STRUCTURE from attention patterns
    What crosses: step count, step types, dependencies
    What doesn't: mathematical content, specific values
    
    
    LAYER 1                    LAYER 2
    ─────────                  ─────────
    Scaffold             →      Narration
    "5 steps, typed"           "intent in natural language"
    
    BOUNDARY: Narrator translates step type → description
    What crosses: mathematical INTENT (what to do)
    What doesn't: precise notation, symbolic form
    THIS IS A NATURAL LANGUAGE TASK — LMs excel here
    
    
    LAYER 2                    LAYER 3
    ─────────                  ─────────
    Narration            →      Rough expression
    "expand (x+y)²"           "EXPAND (x+y)^2"
    
    BOUNDARY: Translator converts NL description → math notation
    What crosses: symbolic FORM (how to write it)
    What doesn't: precise SymPy syntax, execution semantics
    Small vocabulary mapping — each description maps to ~1-3 patterns
    
    
    LAYER 3                    LAYER 4
    ─────────                  ─────────
    Rough expression     →      SymPy execution
    "EXPAND (x+y)^2"          "x**2 + 2*x*y + y**2"
    
    BOUNDARY: Oracle parses rough notation → precise SymPy → execute
    What crosses: PRECISION (exact syntax, evaluation)
    What doesn't: nothing — SymPy is the final authority
    Deterministic. Incorruptible. No model involved.
```

---

## Why Each Boundary Must Be Small

The lesson from every Mycelium version:

```
v2:  Problem → SymPy         (3 boundaries at once → 22% accuracy)
v6:  Problem → operation+span (2 boundaries → 55% ceiling)
v7a: Problem → telegram       (2 boundaries → 4% accuracy)
v7b: Problem → operands       (1.5 boundaries → 2% accuracy, wrong selection)
v7c: Problem → narration → expression (1 boundary each → ???)
```

Each time we reduce the boundary size, the task becomes more learnable.
The limit is one boundary per model. Each worker crosses exactly one
representational gap. The gap must be small enough that pattern matching
(what 0.5B models do) is sufficient — no reasoning required.

---

## The Diamond Shape (Revisited)

```
             Problem text
             (1 sentence, everything compressed)
                  │
             C1-A EXPANDS structure
                  │
        ┌────┬────┼────┬────┐
      step  step step step step     ← widest: N explicit steps
        │    │    │    │    │
        ↓    ↓    ↓    ↓    ↓
      Narrator EXPANDS intent        ← each step gets a description
        │    │    │    │    │
        ↓    ↓    ↓    ↓    ↓
      Translator COMPRESSES to math  ← description → expression
        │    │    │    │    │
        └────┴────┼────┴────┘
                  │
             Oracle COLLAPSES to answer
             (1 value, everything verified)
```

The top half EXPANDS: one blob → N steps → N descriptions.
The bottom half COMPRESSES: N descriptions → N expressions → 1 answer.

Each expansion makes the implicit explicit.
Each compression removes natural language and adds precision.

The widest point — N natural language descriptions — is where the
most information is visible and the most error attribution is possible.

---

## Connection to the Seven Principles

```
EXPAND:     Layer 0→1→2 expands structure and intent
DECOMPOSE:  Each step is independent, processed in parallel
SMOOTH:     Each boundary is ~equal difficulty across step types
VERIFY:     Layer 4 oracle checks every step
LOW→HIGH:   Layers descend from rough NL to precise SymPy
COMPRESS:   Layer 2→3→4 compresses to minimal executable form
EVOLVE:     Verified traces improve every worker each cycle
```

---

## Connection to Breathing

The gradual descent is the SPATIAL dimension (across layers).
Breathing is the TEMPORAL dimension (across iterations).

```
                    Breath 1          Breath 2          Breath 3
                    ────────          ────────          ────────
Layer 0 (problem):  read              read              read
Layer 1 (scaffold): generate          (locked)          (locked)
Layer 2 (narrate):  attempt 1         retry failures    retry remaining
Layer 3 (translate): attempt 1        retry failures    retry remaining
Layer 4 (execute):  verify            verify            verify
```

Each breath descends through all layers.
Each subsequent breath has more context from previous collapses.

The spatial gradient (layers) ensures small jumps.
The temporal gradient (breaths) ensures convergence.

Together they form a 2D optimization surface:
    - Horizontal: gradual representational descent
    - Vertical: iterative refinement with growing context

---

## Assembly Line Workers

```
┌──────────────────────────────────────────────────────────────┐
│                    ASSEMBLY LINE                             │
│                                                              │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐     │
│  │  C1-A   │→ │Narrator │→ │Translator│→ │  Oracle  │     │
│  │scaffold │  │ LoRA C  │  │ LoRA D   │  │  SymPy   │     │
│  │         │  │         │  │          │  │          │     │
│  │structure│  │  intent │  │   math   │  │precision │     │
│  │from attn│  │  in NL  │  │ notation │  │  verify  │     │
│  └─────────┘  └─────────┘  └──────────┘  └──────────┘     │
│       │            │             │             │             │
│   boundary      boundary     boundary      boundary         │
│    count       description   expression    execution         │
│                                                              │
│  Same Qwen-0.5B base model throughout.                       │
│  Different LoRA adapter per worker. 3-4MB swap.              │
│  Each worker: ONE boundary, ONE output format.               │
│                                                              │
│  C1-B watches from above:                                    │
│    - How many breaths to budget?                             │
│    - Where are the phase boundaries?                         │
│    - Is the sequence converging or diverging?                │
└──────────────────────────────────────────────────────────────┘
```

---

## What Each Worker Does NOT Need to Know

This is as important as what they do:

```
C1-A does NOT need to know:
    - What the mathematical expressions look like
    - What the answer is
    - How many breaths the problem will need
    It only knows: where the boundaries are, what type each step is

Narrator does NOT need to know:
    - SymPy syntax
    - How to write mathematical expressions
    - Precise numerical values
    It only knows: given this step type and context, describe the intent

Translator does NOT need to know:
    - The original problem text
    - Why this step is being performed
    - The full solution strategy
    It only knows: given this description and these values, write the expression

Oracle does NOT need to know:
    - Natural language
    - Problem context
    - Why the expression was written
    It only knows: parse this string, execute it, return the result
```

Each worker is deliberately ignorant of everything outside its boundary.
This is not a limitation — it's the design.
Ignorance prevents shortcutting. Specialization enables mastery.

---

## The Mantra (Updated)

```
Expand, smooth, decompose, verify, low→high, compress, evolve.

Expand the STRUCTURE while compressing the CONTENT.
Descend GRADUALLY through representational layers.
Breathe: expand possibilities, collapse to verified truth.
Each worker crosses ONE boundary.
Each breath has MORE context than the last.
The oracle is INCORRUPTIBLE.
The system EVOLVES through verified traces.
```
