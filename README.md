# Mycelium

---

> λ — λανθάνω (lanthánō) — to escape notice; to be unseen
> 
> JSD reveals latent structure in attention flow  
> IAF reconstructs operator influence across layers  
> Structure parser assembles the compositional program  
> 
> Mycelium — the unseen network of computation

---

**99.95% on GSM8K** (7,375/7,378) | **17.2% on MATH** (and climbing) | **No CoT at inference** | **1B parameters**

## Documentation

| Document | Description |
|----------|-------------|
| [Field Guide](plan/field_guide.md) | Lessons learned, bugs encountered, tricks that work |
| [Training Pipeline](plan/training_pipeline.md) | Full IAF + clause expansion training pipeline |
| [Training Guide (C1-C6)](plan/mycelium_c1_c6_training_guide.md) | Step-by-step guide for training all six specialist models |
| [Paper Draft](plan/paper.md) | Technical paper with methodology and results |

---

## The Big Picture

A 7B teacher model solves math problems via chain-of-thought. We extract **three signals** from its attention patterns — span boundaries (JSD), problem text mapping (IAF), and CoT operation structure — and distill them into small student models that reproduce the reasoning without generating any text.

```
TRAINING:  7B solves problems → JSD segments CoT → IAF maps to problem text → train specialists
INFERENCE: Problem text → specialists parse structure → assemble sympy program → execute → answer
```

No chain-of-thought at inference. The answer comes from understanding problem structure and executing symbolically, not from language modeling.

---

## Results

### GSM8K (Word Problems)

```
22%    → Qwen-0.5B baseline (autoregressive)
65%    → Decomposition into 3 specialists
81%    → Improved segmentation + candidate search
93%    → Better extraction (spelled-out numbers)
99.95% → Comprehensive bridging search
```

3 remaining failures: combinatorics puzzle, exponential growth, number parsing edge case.

### MATH (Competition Mathematics)

IO tagger finds clauses → those ARE the spans
Role classifier labels each clause → that IS the structure
Expression extractor pulls sympy expressions
Program assembler builds the solve() call
Sympy executes


## Architecture

### GSM8K: Three Specialists

| Model | Task | Approach |
|-------|------|----------|
| C1: Segmenter | BIO token tagging | Token classification head |
| C2: Classifier | Span group → operation (ADD/SUB/MUL/DIV) | Sequence classification |
| C3: Extractor | Operation → typed arguments | Generative (Instruct) |

All models are Qwen-0.5B (~500M params each). Inference uses candidate search over span groupings with symbolic execution as the validator.

### MATH: Structure Parser (In Development)

The MATH architecture reframes the problem. Instead of classifying arithmetic operations, the pipeline **parses problem structure** and assembles a sympy program.

**Why roles instead of operations?** MATH problems are 72% algebraic (contain variables, not just numbers). Classifying ADD/SUB/MUL is meaningless when the problem is "If $m = r + 2$ and $mr = 98$, find $m$." The pipeline needs to understand that one clause is an EQUATION, another is a CONSTRAINT, and the question is the OBJECTIVE — then assemble the solve() call.

## Training Pipeline

### Three Signals from One Forward Pass

The 7B teacher provides everything from a single forward pass with attention hooks:

| Signal | What It Captures | How It's Extracted |
|--------|-----------------|-------------------|
| **CoT text** | The reasoning trace | Standard generation |
| **JSD boundaries** | Where computation steps are | Jensen-Shannon Divergence between consecutive attention distributions, Savitzky-Golay smoothed (w=5) |
| **IAF mapping** | Which problem text each step attended to | Input Attention Fraction — generation→input attention flow |

### Label Generation

```
7B generates CoT
    │
    ├── JSD segments CoT into computation steps
    │
    └── IAF maps each step to problem text tokens
                │
                ▼
        Clause expansion (spacy)
        scattered tokens → coherent phrases
                │
                ▼
        LaTeX parser (sympy) + classifier fallback
        extract operation/role labels
                │
                ▼
        Execution validation
        keep only traces that produce correct answers
                │
                ▼
        Training pairs: (problem_text_clause → role/operation)
```

### Self-Improving Loop

```
Round N:
  1. Run student models on all problems
  2. Correct traces → add to training set (execution-validated)
  3. 7B mop-up on failures → more training data
  4. Retrain students on expanded data
  5. Repeat until convergence
```

Each round: more correct traces → better training data → better models → more correct traces. Execution correctness is a **free verifier** — no human annotation needed at any stage.

## Reading vs generation attention (Prefill vs Decode)

Same model, same heads, two phases, fundamentally different information:

| Metric | Reading Phase | Generation Phase |
|--------|--------------|-----------------|
| % peaks at tokenizer artifacts | 16.8% | ~0% |
| % peaks at step boundaries | Low | 29.5% |
| % peaks at numbers | 9.7% | 19.2% (2x) |
| Mean JSD | 0.018 | 0.031 (1.75x) |

**Reading-phase attention** organizes by linguistic structure — syntax, entities, sentence boundaries. **Generation-phase attention** organizes by computational structure — operation steps, operand routing, result production.

Mycelium v3–v5 failed on reading-phase features. v6 succeeded on generation-phase features. The same pipeline, different attention phase, completely different outcome.

**IAF and operand matching are orthogonal** (0.5% token overlap):
- IAF captures semantic context: "Natalia", "altogether", "clips", "May"
- Operand matching finds numbers: "48", "half"
- Combined = complete operation specification

---

## Error Attribution and Benefits of inference model specialization

**Important** 
Splitting span segmentation, classification, and extraction into separate components allows for precise error attribution.  It also allows the qwen .5b inference model to specialize for each task.
---

## Inference

```
Problem Text → IO Tagger → Role Classifier → Expression Extractor → Program Assembler → Sympy → Answer
```

---

## Core Scripts

**Inference** (`inference/`):

| Script | Purpose |
|--------|---------|
| `e2e_pipeline.py` | GSM8K E2E pipeline with batching |
| `executor.py` | Symbolic executor (ADD, SUB, MUL, DIV + lambdas) |
| `candidate_generator.py` | Enumerate span groupings |
| `diagnostic.py` | Error attribution logging |
| `track1_sympy.py` | Direct LaTeX→sympy evaluation (MATH baseline) |
| `track2_structure.py` | Structure parser pipeline (MATH, in development) |

**Training** (`train/`):

| Script | Purpose |
|--------|---------|
| `train_segmenter.py` | Train BIO/IO segmenter |
| `train_classifier.py` | Train operation/role classifier |
| `train_extractor.py` | Train argument extractor |
| `generate_classifier_data.py` | Generate classifier training data |
| `generate_extractor_data.py` | Generate extractor training data |

**Data Generation** (`scripts/`):

| Script | Purpose |
|--------|---------|
| `generate_7b_cot.py` | Generate 7B CoT on MATH |
| `extract_jsd.py` | Extract JSD boundaries from attention |
| `extract_iaf.py` | Extract IAF mappings |
| `clause_expansion.py` | Expand IAF tokens to clause boundaries |
| `latex_parser.py` | Hybrid LaTeX + classifier label parser |
| `error_attribution.py` | Diagnose which component fails |

---

## Lessons Learned

### Critical Bugs

| Bug | Symptom | Impact | Lesson |
|-----|---------|--------|--------|
| Token→char index misalignment | Mid-word spans: "rmine the va" | Poisoned ALL downstream models for hours | Always visualize your spans |
| GSM8K segmenter on MATH | Scattered BIO tags | 5% E2E vs 78% with JSD | Domain mismatch kills transfer |
| BIO label imbalance | Model predicts I-OP, never B-OP | 0% usable regions | Use IO tagging instead |
| `\boxed{}` masking real eval | 91% accuracy (all from boxed) | Pipeline never actually tested | Disable shortcuts for honest eval |
| Classifier on CoT, eval on problem | Nonsense predictions | 0.5% E2E | Training/inference distributions must match |
| `max_new_tokens=512` | 43% solve rate | Truncated correct answers | Check generation config |

### Tricks That Work

1. **JSD + Savitzky-Golay (w=5)** — peaks in generation-phase attention = computation step boundaries
2. **IAF** — maps CoT computation back to problem text regions (orthogonal to operand matching)
3. **Clause expansion** — converts scattered IAF tokens to linguistically coherent regions via spacy
4. **IO tagging over BIO** — avoids label imbalance, 100% problem coverage
5. **Hybrid parser** — LaTeX patterns primary, classifier fallback, execution validation filters all
6. **Execution validation** — correct answer = valid labels, free supervision at scale
7. **Error attribution** — always diagnose before guessing, run the diagnostic script
8. **Role classification over operation classification** — EQUATION/CONSTRAINT/OBJECTIVE captures problem structure, not just arithmetic

### Mantras

1. **Always visualize your spans.** If they look like "rmine the va" you have a bug.
2. **Execution is a free verifier.** If the answer is right, the labels are right.
3. **Generation-phase attention, not reading-phase.** Computation structure is only visible during generation.
4. **Disable `\boxed{}` for honest eval.** Otherwise you're measuring the teacher, not the pipeline.
5. **Error attribution before guessing.** Run the diagnostic, don't assume where the bug is.
6. **The self-improving loop is the endgame.** Execution-validated traces → better models → more correct traces.

---
**The Scaling Story**
The scaling story is wild when you think about it.
Inference cost: A frontier model doing chain-of-thought on a hard MATH problem might generate 2000+ tokens autoregressively. That's 2000 forward passes through hundreds of billions of parameters. Mycelium does one forward pass through each 0.5B specialist — six passes total, each through a tiny model. Orders of magnitude cheaper.
Knowledge leverage: Those 3B parameters could contain distilled reasoning structure from a model 1,000x their size. They're not trying to know everything — they just know how to decompose, classify, extract, and wire up math problems. Sympy handles the actual math for free.  This could apply to any domain where you have a small set of primitives that compose in predictable ways.
The real unlock: The teacher only runs at training time. At inference, it's gone. You pay the big model cost once to generate attention patterns, then amortize that across unlimited cheap inference. It's like a master mathematician teaching six apprentices each one specific skill, then retiring.

