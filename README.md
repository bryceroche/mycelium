# Mycelium v6

**99.95% accuracy on GSM8K** (7,375/7,378 correct) | **16 problems/second** | **100x faster than CoT**

## The Big Picture

A 72B teacher model solves math problems via chain-of-thought. We extract computation structure from its attention patterns and distill it into three 0.5B student models that reproduce the reasoning without generating any text at inference.

```
TRAINING:  Teacher solves problems → extract spans via JSD → train 3 specialists
INFERENCE: Problem text → 3 specialists → candidate search → symbolic executor → answer
```

No chain-of-thought generated at inference. The answer comes from executing operations, not from language modeling.

---

## Current Architecture (3 Models)

| Model | Task | Approach |
|-------|------|----------|
| C1: Segmenter | BIO token tagging | Token classification head |
| C2: Classifier | Span group → operation (ADD/SUB/MUL/DIV) | Sequence classification |
| C3: Extractor | Operation → typed arguments | Generative (Instruct) |

All models are Qwen-0.5B (~500M params each).

---

## Training Pipeline

### Step 1: Teacher CoT Generation with Attention Capture

Run Qwen-72B on GSM8K with `output_attentions=True`. For each problem, save:
- The full chain-of-thought text
- Per-token attention distributions over the input
- The final answer

**Filter to correct solutions only.** On GSM8K: ~88% solve rate. Only correct CoTs have valid computation structure to extract.

### Step 2: JSD Segmentation

Detect span boundaries using Jensen-Shannon Divergence between consecutive attention distributions.

```python
for t in range(len(generated_tokens) - 1):
    A_t = attention_over_input[t]
    A_t1 = attention_over_input[t + 1]
    jsd[t] = jensen_shannon_divergence(A_t, A_t1)
```

JSD spikes = attention shift = span boundary.

**Savitzky-Golay smoothing (window=5)** preserves real peaks while removing noise.

**Over-segment by design.** Splitting one operation into two pieces is recoverable (candidate search can re-merge). Merging two operations destroys information irreversibly.

### Step 3: Training Data Generation

From segmented CoT data, generate training examples:

**C1 Segmenter:** BIO token tags from JSD boundaries.
```
Input:  "Bob bought 5 apples at $2 each and 3 oranges"
Labels: O   O      B I      B I I  I    O   B I
```

**C2 Classifier:** (marked span group) → operation label.
```
Input:  "Bob bought <<5 apples>> at <<$2 each>>"
Label:  MUL
```

**C3 Extractor:** (span group + operation) → typed arguments.
```
Input:  "MUL: Bob bought <<5 apples>> at <<$2 each>>"
Output: "5|PROB 2|PROB"
```

### Step 4: Train Three Specialist Models

Training config: 5 epochs, lr=2e-5, batch_size=8, max_length=512.

---

## Inference Pipeline

### Step 1: Segment

C1 segmenter tags problem text with BIO labels. Groups consecutive B-I sequences into spans.

```
Input:  "Bob bought 5 apples at $2 each and 3 oranges at $1.50 each"
Output: ["5 apples", "$2 each", "3 oranges", "$1.50 each"]
```

### Step 2: Generate Candidate Groupings

**Grouping is search, not learning.** Enumerate plausible groupings with heuristics:

```
Candidate 1: {0}, {1}, {2}, {3}           # all separate
Candidate 2: {0,1}, {2,3}                 # adjacent pairs
Candidate 3: {0,1}, {2}, {3}              # first pair only
...
```

Typically 5-15 candidates per problem.

### Step 3: Batch Classify + Extract

For each candidate grouping, run C2 and C3 on every span group:

```
Candidate 2: {0,1}, {2,3}
  C2({0,1}) → MUL (confidence: 0.85)
  C3({0,1}) → [5, 2]
  C2({2,3}) → MUL (confidence: 0.78)
  C3({2,3}) → [3, 1.50]
```

**Batch all candidates together.** Cuts latency from ~20s to ~2s per problem.

### Step 4: Execute + Score + Pick

Symbolic executor runs each candidate's operations:

```python
graph = [
    {"operation": "MUL", "arg1": 5, "arg2": 2},     # = 10
    {"operation": "MUL", "arg1": 3, "arg2": 1.5},   # = 4.5
    {"operation": "ADD", "arg1": "PREV:2", "arg2": "PREV"},  # = 14.5
]
```

Score based on: execution success, answer plausibility, classifier confidence.

Highest-scoring candidate's answer is the final answer.

---

## Why It Works

| Principle | Implementation |
|-----------|----------------|
| Search where space is small | Groupings (5-15 candidates): searched |
| Learn where space is large | Classification (language understanding): learned |
| Executor validates | Bad predictions fail to produce valid answers |
| Batch for speed | All candidates classified in one forward pass |

**The executor is the validator.** A classifier doesn't need to be perfect. If it picks the wrong operation but the executor can't produce a valid answer, that candidate gets eliminated.

---

## The Accuracy Journey

```
22%    → Qwen-0.5B baseline (autoregressive)
65%    → Decomposition into 3 specialists
81%    → Improved segmentation + candidate search
93%    → Better extraction (spelled-out numbers)
99.95% → Comprehensive bridging search
```

3 remaining failures: combinatorics puzzle, exponential growth, number parsing edge case.

---

## Core Scripts

| Script | Purpose |
|--------|---------|
| `v6_e2e_pipeline.py` | Main E2E pipeline with batching |
| `v6_executor.py` | Symbolic executor (ADD, SUB, MUL, DIV) |
| `v6_candidate_generator.py` | Enumerate span groupings |
| `v6_eval_quick.py` | Quick evaluation |
| `v6_diagnostic.py` | Error attribution logging |
| `v6_train_*.py` | Training scripts |
| `v6_generate_*.py` | Data generation scripts |

---

## Aspirational Architecture (6 Models)

The current 3-model architecture achieves 99.95% but has limitations. The full planned architecture adds three more specialists:

| Model | Task | Why Needed |
|-------|------|------------|
| C4: Implicit Ops | Bridging pattern detection | Handle "total", "average", "remaining" |
| C5: Dep Resolver | Wire execution DAG | Connect operations that share values |
| C6: Goal Resolver | Answer type + hint | "How much change?" → try SUB |

### Planned Additions

**Domain constants** — only genuinely orphaned values (>80% orphan rate):
```python
DOMAIN_CONSTANTS = {
    # Time
    60:   "minutes/hour",
    24:   "hours/day",
    7:    "days/week",
    52:   "weeks/year",
    12:   "months/year",
    365:  "days/year",

    # Units
    100:  "percentage, cm/m",
    1000: "g/kg, m/km",
}
```

**Why only 8 constants, not 22?** Small numbers (2-10) are almost always in the problem text or derived from prior computation. Including them causes false positives on MATH where `intermediate × 3` accidentally equals the gold answer. Real domain constants are world knowledge the teacher uses but the problem doesn't state.

**Orphan rate filter:** For each candidate constant, check what percentage of occurrences are genuinely orphaned vs appearing in problem text. Keep only >80% orphan rate. Small numbers like 3, 5, 10 have ~5% orphan rate — they're not constants, they're just numbers the regex missed matching.

**Bridging templates** (IB-discovered implicit operations):
```python
BRIDGING_TEMPLATES = {
    "ACC_ADD":      lambda results: sum(results),
    "ACC_SUB":      lambda results: max(results) - sum(others),
    "AVG":          lambda results: sum(results) / len(results),
    "DERIVED_MUL":  lambda results: results[-1] * results[-2],
}
```

**Two-hop chains** for complex derived computations:
```
result → ACC_ADD → intermediate → MUL(intermediate, constant) → answer
```

### Why Not Implemented Yet

The 3-model architecture already achieves 99.95%. The remaining 3 failures are edge cases (combinatorics, exponentials) that need special handling rather than more models. The 6-model architecture would help with:
- MATH dataset (more implicit operations)
- Problems requiring unit conversions
- Multi-step derived computations

---

## Research Finding: Dual-Purpose Attention

Same model, same heads, two phases, fundamentally different information.

| Metric | Reading Phase | Generation Phase |
|--------|--------------|-----------------|
| % peaks at tokenizer artifacts | 16.8% | ~0% |
| % peaks at step boundaries | Low | 29.5% |
| % peaks at numbers | 9.7% | 19.2% (2x) |
| Mean JSD | 0.018 | 0.031 (1.75x) |

**Natural head specialization:** Within Layer 20, 52x contrast between reading-specialized and computing-specialized heads.

---

## License

MIT
