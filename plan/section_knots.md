# Mycelium: Theme Park Brainstorming — Implementation Handoff

## 1. CRITICAL ARCHITECTURE CHANGE: Split C3 into C3 + C4

**This is the top priority.** The current C3 does three jobs (find operands, resolve dependencies, build expressions) and hallucinates because of it. Split into two:

### C3 — Operand Locator (learned, pointer model)
- **Task:** Given full problem text + template tag + list of prior results, POINT at which operands to use
- **Input:** `[TEMPLATE: DIV] full problem text [PRIOR_1: 24] [PRIOR_2: 15]`
- **Output:** `[TEXT_48, IMPLICIT_half]` — provenance labels, NOT expression text
- **Architecture:** Qwen-0.5B with a pointer head. For each operand slot, predict source type + location
- **Key constraint:** C3 can only SELECT from what exists. It cannot generate numbers.

**Provenance taxonomy (the closed set C3 selects from):**
```
TEXT_<position>  — operand from problem text at token position N
PRIOR_<N>        — result from computation step N
IMPLICIT_<word>  — implied value ("half"→2, "double"→2, "triple"→3, "twice"→2)
CONSTANT_<value> — domain constant (60 min/hr, 100 for percent, 365 days/yr)
```

### C4 — Expression Builder (deterministic lookup table, NOT a neural network)
- **Task:** Given template + ordered operands, assemble the sympy expression
- **Input:** template=DIV, operands=[48, 2]
- **Output:** `48 / 2`
- **Implementation:** Lookup table / simple function

```python
def assemble(template, operands):
    if template == "ADD":   return f"{operands[0]} + {operands[1]}"
    if template == "SUB":   return f"{operands[0]} - {operands[1]}"
    if template == "MUL":   return f"{operands[0]} * {operands[1]}"
    if template == "DIV":   return f"{operands[0]} / {operands[1]}"
    if template == "SQRT":  return f"sqrt({operands[0]})"
    if template == "SQUARE": return f"({operands[0]})**2"
    if template == "SIN":   return f"sin({operands[0]})"
    if template == "COS":   return f"cos({operands[0]})"
    if template == "LOG":   return f"log({operands[0]})"
    # ... etc
```

### Why This Solves the Hallucination Problem
- C3 can't hallucinate numbers — it only points at things that exist
- C4 can't hallucinate structure — the template determines it
- Combined: if the operands are correct, the expression is correct

### The DAG Builds Itself
Every time C3 outputs `PRIOR_N`, that IS a directed edge from step N to the current step. No separate C5 dependency resolver needed. The graph emerges from C3's provenance labels.

```
Step 1: C2=DIV, C3=[TEXT_48, IMPLICIT_half], C4→"48/2", sympy→24
Step 2: C2=ADD, C3=[PRIOR_1, TEXT_48],       C4→"24+48", sympy→72
                     ↑
                     This IS the DAG edge
```

### Training Data for C3 (pointer model)
Reformat existing C3 training pairs:
1. Take each (problem_text, template, expression) triple
2. Parse the gold expression to identify each operand
3. For each operand, determine provenance:
   - Does it appear in the problem text? → TEXT_<position>
   - Is it a result from a prior step? → PRIOR_<N>
   - Is it implied by a word like "half"? → IMPLICIT_<word>
   - Is it a domain constant? → CONSTANT_<value>
4. Output training pair: (input text + priors) → list of provenance labels

Use Track A (all operands in text, 6,453 examples) for initial training.
Use Track B (derived values, 20,458 examples) to train PRIOR_N resolution.

---

## 2. SIMPLIFIED PIPELINE

### Old (6 models):
```
C1 (segmenter) → C2 (classifier) → C3 (extractor) → C4 (bridging) → C5 (dependencies) → C6 (answer type) → sympy
```

### New (2 learned models + 1 lookup table + sympy):
```
C2: What operations?     → {DIV, ADD}          (MiniLM-22M, multi-label sigmoid, threshold 0.3)
Loop (ordered by MCTS):
    C3: Which operands?  → [TEXT_48, PRIOR_1]   (Qwen-0.5B, pointer model)
    C4: Build expression → "48 / 2"             (deterministic lookup table)
    Sympy: Evaluate      → 24
    Add result to PRIOR list
Final sympy result = answer (format determined by sympy output type, no C6 needed)
```

### What's Dead and Why
- **C1 (segmenter):** Position measurement is low-information when operands overlap. WFC says measure type first.
- **C5 (dependencies):** Replaced by C3's PRIOR_N provenance labels. The DAG builds itself.
- **C6 (answer type):** Sympy's output type IS the answer format. Integer→integer, Rational→fraction, etc.

### Current Results to Beat
- 18% accuracy on 50 MATH problems (with old C3 causal LM, beam k=10, weighted voting)
- 44% ceiling (correct answer in beam but not ranked first)
- Baseline: 17.2%

---

## 3. LOG BASE 10 — HARTLEYS FOR INFORMATION MEASUREMENT

### MCTS Search Space Budgeting
A problem with N operations and k beams has k^N paths. Log₁₀(search space) = the number of decimal digits needed to enumerate all paths.
- 3 operations, 10 beams: log₁₀(10³) = 3
- 5 operations, 10 beams: log₁₀(10⁵) = 5

Use log₁₀ to budget compute per problem. Each MATH level gets a compute budget in hartleys:
- Level 1: budget = 2 hartleys (100 paths max)
- Level 5: budget = 5 hartleys (100,000 paths max)

### Confidence Calibration
C2 sigmoid outputs in log₁₀ space:
- 0.9 → -0.05 (very confident)
- 0.09 → -1.05 (uncertain)
- 0.009 → -2.05 (very uncertain)

The decoherence threshold θ maps cleanly: each integer in log₁₀ space = order of magnitude of certainty.

### IB Information in Hartleys
Instead of nats (natural log) or bits (log₂), measure mutual information in hartleys (log₁₀). "This template carries 2.3 hartleys of information about the computation" = each template is worth ~2 decimal digits of information.

---

## 4. KNOT THEORY — DAG TOPOLOGY AS PROBLEM COMPLEXITY

### The Core Insight
The teacher's chain-of-thought is a 1D projection of multi-dimensional reasoning. When step 3 references step 1's result while step 4 references step 2 — that's a CROSSING in the projection. The linear CoT forces parallel threads to interleave, creating knots.

**Mycelium unknots the reasoning** — untangles the linear CoT back into its true DAG structure.

### Crossing Number = Problem Complexity
- **Crossing number 0** (unknot): linear chain, A→B→C, no crossings. Easy.
- **Crossing number 1** (trefoil): one dependency crosses another. Medium.
- **Crossing number 3+**: heavily tangled dependencies. Hard.

Two problems can have the SAME heartbeat count (same number of operations) but completely different knot topology. Heartbeats measure quantity; crossings measure structural complexity.

### Reidemeister Moves = Equivalent DAG Rewrites
Knot theory's three elementary moves correspond to:
1. **R1 (twist/untwist):** Remove redundant intermediate results
2. **R2 (poke/unpoke):** Reorder independent operations
3. **R3 (slide):** Restructure a three-step dependency chain

Two strategies that look different but have the same knot invariants are computationally equivalent — same topology, different projection.

### For v7 Strategy Templates
Knot invariants (Jones polynomial, Alexander polynomial) could classify strategies. "Complete the square" and "integration by parts" may have different operations but the same crossing structure — the same knot type. Strategy templates grouped by knot class.

### Implementation (Future)
1. Build DAG from C3's provenance labels
2. Compute crossing number from DAG edge intersections when drawn in CoT order
3. Correlate crossing number with MATH difficulty level
4. Use knot invariants to classify v7 strategy templates

---

## 5. OPERATION ORDERING WITHOUT C5

Without C5, who decides operation order? Two approaches:

### Approach A: MCTS Explores All Orderings
For N operations, try all N! permutations. For typical problems (2-5 ops):
- 2 ops: 2 permutations
- 3 ops: 6 permutations
- 4 ops: 24 permutations
- 5 ops: 120 permutations

Each permutation runs the C3→C4→sympy loop. Valid final answers survive. Majority vote on survivors. This is brute-forceable for small N and elegant — the correct ordering is the one that produces valid sympy at every step.

### Approach B: Greedy Execution
Try each remaining operation. If C3 can find all operands (text + available priors), execute it. If not, skip and try another. Operations naturally sort themselves by dependency — an operation that needs PRIOR_1 can only run after step 1 completes.

```python
remaining = C2_predicted_operations.copy()
priors = []
while remaining:
    for op in remaining:
        result = try_execute(op, problem_text, priors)
        if result is not None:
            priors.append(result)
            remaining.remove(op)
            break
    else:
        break  # stuck — no operation can execute
```

### Recommendation
Start with Approach B (greedy). It's simpler and handles 80% of cases. Add MCTS permutation search for problems where greedy gets stuck.

---

## 6. CURRENT STATUS (as of theme park day)

### What's Working
- **C2 classifier:** 99.64% any-correct, 100% all-correct at threshold 0.3 (MiniLM-22M on MATH, 15 labels + heartbeat auxiliary)
- **C3 causal LM:** 98% valid sympy, 18% E2E accuracy, 44% correct-in-beam with k=10
- **Heartbeat detection:** Calibrated at min_run=5, monotonic Level 1→3.8 to Level 5→14.8
- **IB templates:** 30 clusters via true IB with Y labels and elbow detection
- **Infrastructure:** S3 daemon, Lambda MapReduce, DDP training on g5.48xlarge

### What's Next (Priority Order)
1. **Split C3 into C3 (pointer) + C4 (assembler)** — this doc
2. **Build C3 pointer training data** — reformat existing pairs with provenance labels
3. **Train C3 pointer model** — start with Track A, then add Track B
4. **E2E test with new pipeline** — C2→C3→C4→sympy on 50 MATH problems
5. **Beat 18% and push toward 44% ceiling**

### Models and Data
- C2 model: `s3://mycelium-data/models/c2_heartbeat/`
- C3 training data: `s3://mycelium-data/c3_training/` and `s3://mycelium-data/c3_span_training/`
- IB templates: `s3://mycelium-data/ib_true/templates.json`
- Heartbeats: `s3://mycelium-data/pulse_analysis/math_heartbeats.json`
- IAF data: `s3://mycelium-data/backups/medusa_iaf_2026-02-26/`

### Key Principles
- No heuristics in the inference path — everything learned or deterministic
- C4 assembler is deterministic, not heuristic (template→expression is a mathematical fact)
- Measure type before position (WFC)
- The DAG builds itself from C3's provenance labels
- MCTS handles uncertainty — let all candidates play out through sympy
