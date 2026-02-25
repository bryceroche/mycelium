# Mycelium v6: Training Pipeline (v3 — IAF + Clause Expansion)

---

## Architecture Summary

**Teacher:** Qwen2.5-Math-7B-Instruct (single model, triple purpose)
**Student:** Two 0.5B models (segmenter + classifier = 1B total at inference)
**Distillation gap:** 7x parameters (7B → 1B)
**No 72B required. No CoT at inference.**

The 7B provides three signals from one forward pass:
1. **CoT text** — the reasoning trace
2. **JSD boundaries** — where computation steps are in the CoT
3. **IAF mapping** — which problem text tokens each CoT step attended to

IAF is the bridge between CoT and problem text. **Clause expansion** converts
scattered IAF token highlights into linguistically coherent operation regions
that the 0.5B segmenter can learn to detect.

---

## The Core Idea

```
TRAINING TIME:                              INFERENCE TIME:

  Problem Text                                Problem Text
      │                                           │
      ▼                                           ▼
  7B generates CoT ◄── attention ──┐        ┌──────────────────┐
      │                            │        │ 0.5B Segmenter   │
      ▼                            │        │ (BIO on problem  │
  JSD segments CoT                 │        │  text → clauses) │
  into computation steps           │        └────────┬─────────┘
      │                            │                 │ spans
      ▼                            │                 ▼
  IAF maps each step ─────────────┘        ┌──────────────────┐
  to problem text TOKENS                   │ 0.5B Classifier  │
      │                                    │ (span → op label)│
      ▼                                    └────────┬─────────┘
  Clause expansion:                                 │
  tokens → enclosing clauses                  (region, op, args)
      │                                             │
      ▼                                             ▼
  (clause, op, args)                          Execute DAG
  training pairs                                    │
      │                                             ▼
      ▼                                          Answer
  Train 0.5B segmenter on clauses
  Train 0.5B classifier on spans
```

---

## Training Pipeline

### Phase 1: Generate Teacher Traces with Attention

```
Input:  12,500 MATH problems
Model:  Qwen2.5-Math-7B-Instruct
Output: For each problem:
        - CoT text + \boxed{} answer
        - Per-token attention matrices (generation phase)
```

**Config:**
- `attn_implementation="eager"` (required for attention capture)
- `output_attentions=True`
- Two-phase option if OOM: generate CoT first (SDPA), then forward pass with eager

**Current status:** Complete. 12,500 problems, 68.8% solve rate, ~8,600 correct traces.

### Phase 2: Extract JSD Span Boundaries (on CoT)

For each correct trace, segment the CoT into computation steps:

```
1. Compute per-token JSD between consecutive attention distributions
   JSD(t) = JS_divergence(attn_dist[t-1], attn_dist[t])

2. Smooth with Savitzky-Golay filter (window=5, polyorder=2)

3. Detect peaks → computation step boundaries
   ~1 boundary per 10-15 tokens

4. Define CoT spans as token ranges between consecutive boundaries
```

**Current status:** Complete. 7,842 JSD files extracted.

### Phase 3: Compute IAF Mapping (CoT → Problem Text Tokens)

For each CoT span from Phase 2:

```
1. Extract attention from generation-phase tokens in this span
   to input (problem text) tokens

2. Compute Input Attention Fraction (IAF):
   For each problem text token j:
     IAF(j, span) = sum of attention from span's gen tokens to token j
                    / total attention from span's gen tokens to all input tokens

3. Threshold IAF to identify high-attention problem text tokens
```

**What IAF captures (validated):**
- Semantic context: "Natalia", "altogether", "clips", "May"
- Orthogonal to operand matching (0.5% token overlap)
- Operand matching finds: "48", "half"
- Combined = complete picture for each operation

**Current status:** Complete. 1,516 stratified sample.

**Output:** For each CoT span → scattered high-attention tokens in problem text

### Phase 4: Clause Expansion (Tokens → Regions) ← THE KEY STEP

Raw IAF gives scattered tokens: "48", "half", "sold" light up individually.
The BIO tagger can't learn from scattered tokens (36% F1 proved this).
Clause expansion converts these into contiguous, linguistically coherent regions.

```
1. Parse problem text with spacy (dependency parse + sentence segmentation)

2. For each set of IAF-highlighted tokens from a CoT span:
   a. Find the enclosing clause for each highlighted token
   b. Merge overlapping clauses
   c. Result: one contiguous region per operation

   Example:
   IAF highlights: "48", "half", "sold"
   Enclosing clause: "she sold half as many clips in May"
   → Tag entire clause as one operation region

3. Assign operation label from the corresponding CoT span parse:
   CoT span: "48 / 2 = 24" → operation = DIV
   Problem clause: "she sold half as many clips in May" → label = DIV

4. Extract arguments:
   a. Numbers from the clause: 48
   b. Implicit operands from keywords: "half" → 0.5 or /2
   c. Operand matching + IAF semantic context = robust extraction
```

**Why this works:**
- IAF tells us WHICH tokens matter (semantic + numeric)
- Spacy tells us the BOUNDARY of the phrase containing those tokens
- Together: clean, contiguous, linguistically natural operation regions
- The 0.5B learns "she sold half as many clips in May" = DIV
  not scattered tokens "48" + "half" + "sold" = DIV

**Output:** (problem_text_clause, operation_label, arguments) tuples

### Phase 5: Execution Validation

```
1. Build computation DAG from parsed operations
2. Execute symbolically
3. Compare pipeline answer to gold answer

Correct → KEEP as validated training example
Wrong   → DISCARD
```

Execution correctness is a free verifier. Only traces that produce
the right answer become training data.

### Phase 6: Train 0.5B Student Models

**Model 1: Segmenter (BIO tagger on problem text)**
```
Task:      Token-level BIO tagging on problem text
Input:     Problem text tokens
Labels:    BIO tags from clause-expanded IAF regions
           B-OP at clause start, I-OP for continuation, O elsewhere
Model:     Qwen2-0.5B fine-tuned for sequence labeling
Training:  Validated examples from Phase 5
Eval:      Span F1 against held-out clause boundaries
```

The segmenter learns to find operation-bearing CLAUSES in problem text.
Not scattered tokens, not arbitrary windows — whole linguistic units
that correspond to computation steps.

**Model 2: Classifier (operation label per span)**
```
Task:      Predict operation label for each detected span
Input:     Problem text clause (from segmenter output)
Labels:    Operation labels from Phase 4
Model:     Qwen2-0.5B with classification head
Training:  Validated examples from Phase 5
Eval:      Classification accuracy on held-out spans
```

### Phase 7: Self-Improving Loop

```
Round N:
  1. Run 0.5B segmenter on problem text → find clause-level spans
  2. Run 0.5B classifier on each span → operation labels
  3. Extract arguments, build DAG, execute → answer
  4. Compare to gold → correct traces join training set
  5. Optional: 7B mop-up on failures (CoT + attention → IAF → clause expansion)
  6. Retrain both 0.5B models on expanded data
  7. Repeat until delta < 1%
```

---

## Inference Pipeline (1B Total, No CoT)

```
Problem Text
    │
    ▼
┌──────────────────────────┐
│ 0.5B Segmenter           │
│ (BIO tagging)            │
│                          │
│ Finds operation-bearing  │
│ clauses in problem text  │
│                          │
│ "she sold half as many   │
│  clips in May"  → B-OP  │
│                  I-OP... │
└────────────┬─────────────┘
             │ clause-level spans
             ▼
┌──────────────────────────┐
│ 0.5B Classifier          │
│ (operation labeling)     │
│                          │
│ Each clause → op type    │
│ "sold half as many" → DIV│
└────────────┬─────────────┘
             │ (clause, operation, args)
             ▼
┌──────────────────────────┐
│ Argument Extractor       │
│ (regex + heuristics)     │
│                          │
│ Extract numbers from     │
│ clause text + keywords   │
│ "half" → /2, "48" → 48  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ DAG Builder + Executor   │
│ (heuristics + symbolic)  │
│                          │
│ Topological sort         │
│ Execute operations       │
│ Return final answer      │
└────────────┬─────────────┘
             │
             ▼
          Answer
```

**1B parameters. No 7B. No 72B. No CoT generation.**

---

## Data Flow Summary

```
MATH Problems (12,500)
        │
        ▼
  7B Instruct (forward pass with attention hooks)
        │
        ├── CoT text (69% correct)
        │
        ├── JSD ──► Segment CoT into computation steps
        │
        └── IAF ──► Map each step to problem text TOKENS
                                │
                                ▼
                    Clause Expansion (spacy)
                    tokens → enclosing clauses
                                │
                                ▼
                    Parse operations from CoT spans
                                │
                                ▼
                    Merge: (problem_clause, op, args)
                                │
                                ▼
                    Execute & validate
                                │
                                ▼
                    Validated training pairs
                    (problem_clause → operation)
                                │
                        ┌───────┴───────┐
                        ▼               ▼
                  Train 0.5B      Train 0.5B
                  Segmenter       Classifier
                  (BIO on         (clause →
                   problem text)   op label)
                        │               │
                        └───────┬───────┘
                                │
                                ▼
                    Self-Improving Loop
                                │
                                ▼
                    1B inference (no CoT)
```

---

## Why Clause Expansion Is Essential

| Approach | F1 | Problem |
|---|---|---|
| Raw IAF tokens → BIO | 36.1% | Scattered tokens, no contiguous regions |
| Clause-expanded IAF → BIO | TBD | Linguistically coherent, learnable spans |

Raw IAF highlights individual tokens that the 7B attended to. But BIO
tagging needs contiguous regions to learn from. Clause expansion is the
post-processing step that bridges this gap:

```
Raw IAF:          "... Natalia ... 48 ... half ... sold ... May ..."
                       ↑         ↑      ↑       ↑       ↑
                   scattered high-attention tokens

Clause expanded:  "she sold half as many clips in May"
                   |←────── one contiguous B/I-OP region ──────→|
```

The information is identical — same tokens drive the region. But the
representation is learnable. The segmenter sees a phrase boundary
pattern, not scattered token activations.

---

## Key Numbers

| Item | Value |
|---|---|
| Teacher model | Qwen2.5-Math-7B-Instruct |
| Student models | 2x Qwen2-0.5B (segmenter + classifier) |
| Total inference parameters | ~1B |
| Parameter reduction from teacher | 7x |
| MATH dataset | 12,500 problems |
| 7B solve rate | 68.8% |
| JSD extractions | 7,842 |
| IAF extractions | 1,516 (stratified) |
| Current training pairs | 2,820 |
| Classifier v2 accuracy | 60.4% |
| BIO F1 (raw IAF tokens) | 36.1% (failed) |
| BIO F1 (clause-expanded) | TBD — next experiment |

---

## What 72B Is Still Good For (Optional)

Not in the critical path. Useful for:

1. **CoT mop-up on hard problems** — Geometry (52%), Precalculus (52%) where 7B fails
2. **Ceiling estimation** — compare 7B JSD vs 72B JSD boundary quality
3. **Training data diversity** — hard-category traces the 7B can't produce

All training-time investments. None required at inference.

---

## Open Questions

1. **Clause-expanded BIO F1:** Does expanding IAF tokens to clause boundaries produce learnable BIO labels? This is the immediate next experiment.

2. **7B JSD pipeline accuracy:** We have 7,842 JSD files but haven't run the full E2E with 7B JSD boundaries + 0.5B classifier yet. Need this number to confirm 7B JSD ≈ 72B JSD.

3. **Spacy clause boundary quality:** Do spacy's dependency parses produce sensible clause boundaries for math problem text? Math phrasing can be unusual ("how many more than twice the number of...").

4. **Multi-operation clauses:** Some clauses may contain multiple operations ("she sold 48 clips and bought 12 more"). Need strategy for these — possibly sub-clause splitting.

5. **Non-numeric answers:** Current pipeline handles ~69% of MATH (numeric only). Extending to symbolic answers requires DSL expansion.

6. **Self-improving loop dynamics:** How many rounds to convergence? When to add 7B mop-up? How to balance easy vs. hard category training data?
