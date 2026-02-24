# Mycelium v7: Strategy-Level Decomposition

## The Hierarchy of Mathematical Thought

v6 decomposes reasoning into **atomic operations** — the spectral lines. DIVIDE, SOLVE, INTEGRATE, SUBSTITUTE. Each is a single computational step that sympy executes in microseconds.

But mathematicians don't think in atomic operations. They think in **strategies**: "complete the square," "integration by parts," "proof by contradiction," "construct a system of equations." A strategy is a recurring PATTERN of operations with characteristic dependency structure. It's the difference between knowing individual chess moves and knowing openings.

v7 decomposes reasoning into strategies — the **spectral bands** that contain the spectral lines.

---

## What a Strategy Is (and Isn't)

A strategy is NOT a new model. It's a recurring **DAG motif** — a subgraph pattern that appears across many different problems.

**Integration by parts:**
```
ASSIGN(u) → ASSIGN(dv) → DIFFERENTIATE(u→du) → INTEGRATE(dv→v) → SUBSTITUTE(uv - ∫v·du)
```

**Complete the square:**
```
DIVIDE(normalize leading coefficient) → ADD(half of b, squared) → FACTOR(perfect square) → SOLVE(for variable)
```

**System of equations by substitution:**
```
SOLVE(eq1 for x) → SUBSTITUTE(x into eq2) → SOLVE(eq2 for y) → SUBSTITUTE(y back into x)
```

**Proof by strong induction:**
```
EVALUATE(base case) → ASSUME(k ≤ n) → SUBSTITUTE(n+1) → SIMPLIFY → VERIFY(matches assumption)
```

Each strategy is a template of templates — a meta-pattern in the DAG that v6's C5 wires implicitly but never names.

---

## The Two Levels of Spectral Decomposition

### Level 1: Operations (v6 — current)
- **Prism:** IAF separates white light into spectral lines
- **Discovery:** IB clusters CoT steps by computational function
- **Result:** ~40-80 atomic operation templates (ADD, SIN, SOLVE, ...)
- **Granularity:** Single computation step
- **Verification:** Sympy executes each step

### Level 2: Strategies (v7 — proposed)
- **Prism:** DAG topology analysis separates strategy patterns from individual operations
- **Discovery:** Graph motif mining across thousands of solved problem DAGs
- **Result:** ~20-50 strategy templates (COMPLETE_THE_SQUARE, INTEGRATION_BY_PARTS, ...)
- **Granularity:** Multi-step reasoning pattern
- **Verification:** The full DAG executes correctly as a unit

The relationship is hierarchical: strategies contain operations, operations contain operands. Three levels of the spectral taxonomy:

```
Strategy (spectral band)
  └── Operation (spectral line)
        └── Operand (wavelength)
```

---

## How Strategy Discovery Works

### Step 1: Build the DAG Corpus

v6 already produces DAGs for every training problem: C2 assigns templates, C5 wires dependencies. Collect these into a corpus of ~7K labeled DAGs from MATH.

Each DAG is a directed graph where:
- Nodes = (operation template, operand types)
- Edges = data dependencies

### Step 2: Graph Motif Mining

Find recurring subgraph patterns across the corpus. A "motif" is a DAG fragment that appears in many different problems with the same topology.

Standard algorithms: gSpan, SUBDUE, or simpler frequent subgraph mining. The DAGs are small (3-10 nodes) so this is computationally cheap.

**Expected motifs:**
- Linear chains: A → B → C (sequential computation)
- Fan-in: A → C, B → C (combining independent results)
- Fan-out: A → B, A → C (using one result in two ways)
- Diamond: A → B, A → C, B → D, C → D (parallel paths reconverging)

### Step 3: Strategy Clustering via IB

Now we have a REAL two-variable IB setup:
- **X** = DAG motif (the subgraph structure)
- **Y** = problem category or difficulty level

IB discovers which motifs cluster into coherent strategies. The diamond pattern with {SOLVE, SOLVE, SUBSTITUTE, SIMPLIFY} is "system of equations by substitution." The linear chain with {ASSIGN, DIFFERENTIATE, INTEGRATE, SUBSTITUTE} is "integration by parts."

### Step 4: Strategy Templates

Output: a library of named strategy templates, each defined by:
- A DAG topology (which operations, in what dependency order)
- Slot types (what kinds of operands fill each operation)
- Frequency and difficulty distribution
- Example problems

---

## New Architecture Component: C0 — Strategy Selector

v7 adds ONE model to the pipeline — C0, which runs BEFORE C2:

```
Problem text (full superposition)
    → C0: "what strategy?" (strategy collapse — narrows operation space)
    → C2: "what operations?" (operation collapse — within strategy)
    → C3: "what expressions?" (operand extraction — per operation)
    → C5: "what dependencies?" (DAG wiring — constrained by strategy template)
    → C6: "what answer format?"
    → Sympy (full collapse)
```

**C0's job:** Given problem text, predict which strategy template applies. This is single-label classification (one strategy per problem, typically) over ~20-50 strategy classes.

**Why C0 helps:** It provides a top-down constraint on the entire pipeline.

Without C0, C2 predicts {DIVIDE, ADD, SOLVE, SUBSTITUTE} independently and C5 has to figure out how they wire together. With C0, C2 knows "this is a COMPLETE_THE_SQUARE problem" and only needs to fill in the slots: which coefficient to normalize, what value to add, where the perfect square is. C5's job becomes trivial — the DAG topology is given by the strategy template.

**In WFC terms:** C0 performs the COARSEST measurement first. Instead of collapsing individual operation types (C2's job), C0 collapses the **strategy space** — a much lower-dimensional measurement that eliminates huge swaths of the hypothesis space in one step.

**In spectral terms:** C0 selects the spectral band. C2 then identifies individual lines within that band. Much easier than scanning the full spectrum from scratch.

---

## C0 Training Data

Strategy labels come from the DAG motif mining in Step 2. For each training problem:
1. Build the ground-truth DAG from v6 labels
2. Match it against the strategy template library
3. Assign the best-matching strategy

Training pair: (problem text) → strategy template ID

The model learns to recognize "this problem is asking me to complete the square" from natural language cues: "find the vertex of the parabola," "express in the form (x-h)² + k," "minimize the quadratic."

---

## The Strategy Heartbeat

The heartbeat signal from IAF traces encodes strategy as well as operation count. A COMPLETE_THE_SQUARE problem has a characteristic pulse pattern:

```
[read] ████ [compute: normalize] ██ [read] ██ [compute: add term] ████ [read] █ [compute: factor] ██████ [compute: solve] ███
```

Different strategies have different rhythms — different pulse counts, different pulse widths, different read/compute ratios. A PROOF_BY_INDUCTION problem has a distinctive long read phase (parsing the inductive hypothesis) followed by rapid-fire compute pulses. A SYSTEM_OF_EQUATIONS problem has two parallel read-compute cycles that converge.

**The heartbeat is the strategy's fingerprint.** FFT or wavelet analysis on the pulse pattern could classify strategies directly from the attention signal, without any text analysis at all. The attention head's rhythm IS the strategy.

---

## Training Progression

```
v6 (current):  Learn operations from IAF attention patterns
                Result: C2 predicts {DIVIDE, ADD, SOLVE}

v7 (proposed): Learn strategies from DAG motif patterns
                Result: C0 predicts COMPLETE_THE_SQUARE
                        C2 fills slots within the strategy template
                        C5 is mostly determined by strategy topology
```

The critical insight: **v6 must work first.** Strategy discovery requires a corpus of correct DAGs, which requires v6's pipeline to produce them. v7 is a meta-learning layer on top of v6, not a replacement for it.

---

## What v7 Unlocks

### 1. Planning Before Execution

v6 is reactive — it processes each clause and discovers the strategy implicitly as it goes. v7 is proactive — it identifies the strategy upfront and uses it to guide every subsequent decision. This mirrors how expert mathematicians work: recognize the problem type, recall the strategy, apply it.

### 2. Transfer Across Domains

The SUBSTITUTION strategy has the same DAG topology whether applied in algebra (substitute x into equation), calculus (u-substitution in integration), or physics (substitute values into formula). v7's strategy templates transfer across domains because they're structural, not content-dependent.

### 3. Harder Problems

MATH Level 5 problems often require multi-strategy reasoning: "complete the square, THEN integrate, THEN evaluate at bounds." v7 can compose strategies by chaining C0 predictions: [COMPLETE_THE_SQUARE → INTEGRATE → EVALUATE]. Each strategy expands into its operation DAG, and the full DAG is their concatenation.

### 4. Explainability

"I used integration by parts" is a more useful explanation than "I did ASSIGN, DIFFERENTIATE, INTEGRATE, SUBSTITUTE in sequence." Strategy-level decomposition provides human-readable reasoning traces for free.

### 5. Failure Diagnosis

If a problem fails at sympy, v7 can distinguish:
- **Wrong strategy:** C0 predicted COMPLETE_THE_SQUARE but the problem needed FACTORING → retrain C0
- **Right strategy, wrong operand:** C0 was correct but C3 extracted the wrong coefficient → retrain C3
- **Right strategy, wrong wiring:** C0 was correct but C5 miswired within the template → retrain C5

Error attribution becomes hierarchical: strategy-level first, operation-level second, operand-level third.

---

## Estimated Timeline

| Phase | Task | Prerequisite |
|-------|------|-------------|
| 1 | v6 pipeline working (C2→C3→C5→sympy) | Current work |
| 2 | Generate DAG corpus from v6 on full MATH | v6 accuracy >30% |
| 3 | Graph motif mining on DAG corpus | DAG corpus |
| 4 | Strategy template library | Motif mining |
| 5 | Train C0 strategy selector | Strategy labels |
| 6 | Integrate C0 into pipeline | C0 trained |
| 7 | Evaluate v7 vs v6 | Full pipeline |

v6 first. Strategies require correct operation DAGs. Can't learn the opening if you can't see the individual moves.

---

## The Spectral Hierarchy — Complete Picture

```
λανθάνω — the unseen structure, now revealed at every scale

Level 0: The heartbeat (IAF pulse rhythm)
         → The raw signal. Read-compute-read-compute.

Level 1: Operations (IB spectral lines)
         → What computation each pulse performs.
         → ADD, SIN, SOLVE, INTEGRATE

Level 2: Strategies (DAG motif spectral bands)
         → How operations compose into reasoning patterns.
         → COMPLETE_THE_SQUARE, INTEGRATION_BY_PARTS

Level 3: Problem archetypes (strategy composition)
         → How strategies chain to solve complex problems.
         → "Optimization = DIFFERENTIATE + SOLVE + EVALUATE"

The teacher's 7B attention contains all levels simultaneously —
white light holding every frequency from the fundamental heartbeat
to the highest harmonic of compositional strategy.

Mycelium v6 extracts levels 0-1.
Mycelium v7 extracts level 2.
The unseen network grows deeper.
```
