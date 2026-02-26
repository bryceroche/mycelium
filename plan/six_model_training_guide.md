# Mycelium v6: Training Guide for C1–C6 Specialists

All six models start from **Qwen-0.5B-Instruct** and get a task-specific head. Total inference footprint: ~3B parameters. The 7B teacher model is only used at training time to generate IAF attention data — it is never needed at inference.

---

## Shared Setup

**Base model:** Qwen-0.5B-Instruct (all six)

**Training data source:** All training labels come from IAF extraction on the 7B teacher. When the teacher generates a correct chain-of-thought (verified by matching the gold answer), its attention patterns become ground-truth labels for the students. No human annotation required — execution is a free verifier.

**Key principle:** Nothing in the inference path is heuristic or hand-coded. Every decision is learned from attention distillation.

---

## C1 — Segmenter (IO Tagger)

**Task:** Given a math problem, tag each token as Inside (part of an operational clause) or Outside (not).

**Head type:** Token classification, 2 classes (I, O)

**Training data source:** JSD (Jensen-Shannon Divergence) boundaries from IAF traces. During 7B generation, attention shifts sharply when the model moves from one reasoning step to the next. These JSD spikes mark clause boundaries. Contiguous low-JSD regions between spikes are the operational clauses.

**How to build training pairs:**
1. Take a problem's IAF trace from the 7B teacher
2. Compute JSD between consecutive generation tokens' attention distributions over the input
3. JSD spikes → boundary positions (token indices)
4. Convert token indices to character positions using tokenizer offsets (critical — the token↔character alignment bug destroyed an entire day of work)
5. Map character positions back to input tokens
6. Tokens inside clause spans get label I, everything else gets label O

**Why IO, not BIO:** BIO tagging has a severe class imbalance — there's only one B tag per span vs many I tags. The model learns to never predict B, resulting in 0% usable regions. IO tagging avoids this entirely. Contiguous runs of I tokens = clauses.

**Current status:** 86.5% F1, trained on GSM8K data. May need retraining on MATH data if MATH clause structures differ significantly from GSM8K.

**What good output looks like:**
```
Problem: "Natalia sold clips to 48 of her friends in April, and then she sold half as many in May."
C1 output: OOOOO [sold clips to 48 of her friends in April] OOO [sold half as many in May] O
```

---

## C2 — Classifier (Span → Operation Template)

**Task:** Given a clause (from C1), classify it into one of ~100 operation templates discovered by Information Bottleneck clustering.

**Head type:** Sequence classification, ~100 classes

**Training data source:** IB (Information Bottleneck) template discovery on IAF data. IB takes CoT span embeddings as input X and execution result type as target Y, then anneals a temperature parameter β from low to high. At low β, everything collapses into one cluster. As β increases, clusters split — first into coarse categories (arithmetic vs algebra vs geometry), then into fine-grained templates (linear_solve, quadratic_formula, modular_arithmetic, etc.).

**How to build training pairs:**
1. Run IB on MATH IAF data (needs 500+ problems with valid CoT)
2. IB produces template assignments: each CoT span gets a template ID
3. Training pair = (clause text, template ID)
4. On GSM8K, IB found 86 templates with 100% purity. MATH should produce 50–100+ templates covering operations like POW, SQRT, LOG, TRIG, MOD, SOLVE, SUBSTITUTE, etc.

**Interim approach:** We have a structural role classifier (76.6% accuracy, 7 roles: EQUATION, CONSTRAINT, DEFINITION, OBJECTIVE, COMPUTATION, CONDITION, SETUP) trained on manually derived labels. This gets replaced by IB templates once they're available, expanding from ~7 roles to ~100 fine-grained templates.

**What good output looks like:**
```
Clause: "sold half as many clips in May"
C2 output: DIVIDE (template #14)

Clause: "find all values of x such that x² + 3x - 10 = 0"
C2 output: QUADRATIC_SOLVE (template #67)
```

---

## C3 — Extractor (Operation → Arguments)

**Task:** Given a clause + its operation template (from C2), extract the specific operands as a sympy-parseable expression.

**Head type:** Generative (keeps the LM head from Qwen-0.5B)

**Training data source:** Merged spans combining IAF semantic signal with operand top_positions. This is the critical innovation for MATH — on GSM8K, operands are always in the clause text ("5 apples at $2 each" → grab 5 and 2). On MATH, 56% of expressions have implicit operands that live elsewhere in the problem.

**How to build training pairs:**
1. From IAF data, get the top_positions for each generation token — these show which input tokens the 7B attended to when computing each step
2. IAF semantic signal identifies WHICH clause is being processed (high IAF = reading, low IAF = computing)
3. Operand top_positions identify WHERE the numbers come from
4. Merge both signals: the clause provides context ("sold half as many"), the top_positions provide operands (48 from elsewhere in the problem)
5. Training pair = (problem text + highlighted clause + role label) → sympy expression
6. Only include pairs where the expression executes correctly (execution = free verifier)

**The merged-span insight:** IAF and operand matching are orthogonal (validated: 0.5% token overlap). IAF captures semantic tokens ("sold", "half", "May"). Operand matching captures numbers ("48", "2"). Combined, C3 gets everything it needs to extract without reasoning.

**What good output looks like:**
```
Input:  "Natalia sold clips to 48 of her friends... sold half as many in May"
        Clause: [DIVIDE] "sold half as many clips in May"
C3 output: 48/2

Input:  "If f(x) = x² + 3x - 10, find f(2)"
        Clause: [SUBSTITUTE] "find f(2)"
C3 output: (2)**2 + 3*(2) - 10
```

**Current status:** Training on 8,986 GSM8K pairs. MATH pairs pending IAF extraction completion.

---

## C4 — Bridging (Implicit Operations)

**Task:** Detect implicit operations not stated in the problem text — things like unit conversions, intermediate algebraic steps, or unstated assumptions that the 7B teacher performs but that have no corresponding clause.

**Head type:** Sequence classification (bridging template types)

**Training data source:** IB bridging templates from IAF data. During 7B CoT generation, some computation steps attend to the model's own prior outputs rather than the input text (very low IAF). These "internal computation" steps are bridging operations — the model is doing math that wasn't explicitly asked for.

**How to build training pairs:**
1. From IAF traces, identify generation spans where IAF stays very low (< 0.01) for sustained periods — the model is computing internally, not reading
2. Map these spans to the CoT text to see what the teacher wrote during that internal computation
3. IB clustering on these spans reveals bridging template categories
4. Training pair = (preceding context + available results) → bridging operation type

**Why this might be redundant:** If merged spans give C3 everything it needs, bridging operations may not be necessary. Validate C3 with merged spans first. Only train C4 if error attribution shows failures where the model needed an implicit step that no clause provided.

**What good output looks like:**
```
Context: Step 1 produced x = 5, Step 2 needs x in meters but x is in feet
C4 output: UNIT_CONVERT (feet → meters, multiply by 0.3048)

Context: Problem gives m + r = 10 and m × r = 21, next step needs m alone
C4 output: SUBSTITUTE (express m from equation 1, plug into equation 2)
```

---

## C5 — Dependency Resolver (DAG Wiring)

**Task:** Given all clauses from C1, determine which steps depend on which. Output a directed acyclic graph (DAG) of dependencies.

**Head type:** Pairwise classification — for each pair (step_i, step_j), predict whether step_i's result feeds into step_j.

**Training data source:** IAF attention-to-input-over-time patterns. When the 7B generates step 3 and attends back to tokens it generated during step 1, that's a dependency: step 3 depends on step 1's result.

**How to build training pairs:**
1. From IAF traces, segment generation into steps (using JSD boundaries)
2. For each step, check its top_positions: does it attend to tokens from a previous step's generation span?
3. If step_j attends to step_i's generated tokens, there's a dependency edge: i → j
4. Training pair = (step_i text, step_j text) → DEPENDS / INDEPENDENT

**Why not just sequential chaining:** GSM8K is mostly sequential — step 2 uses step 1, step 3 uses step 2. MATH has complex dependencies: two equations might be independently derived then combined in a third step. A DAG captures this; a linear chain cannot.

**What good output looks like:**
```
Problem: "If m = r + 2 and m × r = 98, find m"
Clauses: [S1: m = r + 2] [S2: m × r = 98] [S3: find m]
C5 output DAG:
  S1 → S3
  S2 → S3
  (S1 and S2 are independent; S3 depends on both)
```

---

## C6 — Goal Resolver (Answer Type)

**Task:** Determine what the question is asking for — what form the final answer should take.

**Head type:** Sequence classification (answer type categories)

**Training data source:** IB clustering on the objective clauses from IAF data. The 7B teacher's final generation steps reveal the answer type through attention patterns — whether it converges to a single number, a set of values, an expression, a proof conclusion, etc.

**How to build training pairs:**
1. From IAF data, identify the final computation step (last low-IAF span before \boxed{} output)
2. Classify the gold answer format: integer, fraction, expression, set, coordinate, etc.
3. IB clustering on objective clause embeddings discovers natural answer type categories
4. Training pair = (problem text + objective clause) → answer type

**Why this matters:** Sympy can solve an equation but needs to know what to return. "Find all values of x" needs a set. "What is the remainder" needs an integer. "Express in simplest form" needs a simplified expression. Without C6, the executor doesn't know what to hand back.

**What good output looks like:**
```
Problem: "Find all integer values of x such that x² < 20"
C6 output: INTEGER_SET

Problem: "What is 17 mod 5?"
C6 output: INTEGER

Problem: "Express (3+2i)(1-i) in the form a+bi"
C6 output: COMPLEX_STANDARD_FORM
```

---

## Training Order

C1 → C2 → C3 → C5 → C4 → C6

Each model's output becomes part of the next model's input at inference time. But training can be partially parallelized since they use different slices of the IAF data:

- **C1** uses JSD boundaries (available now)
- **C2** uses IB templates (needs IAF data — available soon)
- **C3** uses merged spans with top_positions (needs IAF data — available soon)
- **C5** uses inter-step attention patterns (needs IAF data — available soon)
- **C4** uses low-IAF bridging spans (needs IAF data, validate need after C3)
- **C6** uses answer format from gold labels + IB (needs IAF data)

Once IAF extraction finishes, C2, C3, C5, and C6 can all start building training data in parallel.

---

## End-to-End Inference Pipeline

```
Problem text
    │
    ▼
   C1 (Segmenter) ──→ clause spans
    │
    ▼
   C2 (Classifier) ──→ each clause gets an operation template
    │
    ▼
   C3 (Extractor) ──→ each clause gets a sympy expression
    │
    ▼
   C5 (Dep Resolver) ──→ clauses wired into a DAG
    │
    ▼
   C4 (Bridging) ──→ implicit operations inserted where needed
    │
    ▼
   C6 (Goal Resolver) ──→ answer format determined
    │
    ▼
   Sympy Executor ──→ walks the DAG, executes expressions, returns answer in correct format
```

Six 0.5B forward passes + deterministic sympy execution. No autoregressive generation at inference time (except C3 which is generative). Total: ~3B learned parameters distilled from a 7B+ teacher.
