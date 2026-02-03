# The Big 4
1. Dual-Signal Architecture (Attention + Embeddings)
2. Attention Distillation Breakthrough
3. New Favorite Pattern
4. How to use Beads

## Terminology
- **Welford's** — Online algorithm for calculating running mean/variance
- **DAG** — Directed Acyclic Graph (computation graph)
- **Span** — Contiguous tokens forming a semantic unit (e.g., "half the eggs")
- **SET** — Initial value assignment operation
- **Attention sink** — Token that receives attention from many others (usually the subject/entity)
- **Z-score** — Standard deviations from learned mean, used for classification

## Core Principle: Failures Are Valuable Data Points
**Let the system fail.** This is how it learns.
- Record every failure — it feeds the learning loop
- Do not fallback to LLM reasoning
- Accumulated failure patterns (not individual failures) refine thresholds
- Success/failure stats drive classification decisions

The goal is NOT 100% accuracy on every run. The goal is collecting data that makes the system smarter over time. A misclassified span provides valuable signal for threshold adjustment.

## Attention-Based Decomposition
**Decomposition is the crux.** Everything downstream of step-level intermediate representation (IR) is solved.

Transformer attention reveals semantic spans. The "Panama Hats" problem: "panama" = country, but "panama hats" = completely different meaning. We need the **longest span** that matches a semantic unit.


Standard embedding models learn lexical similarity, not operational similarity. They think "x + y" and "x * y" are similar because the tokens overlap. This is useless for math — we need to group by what computations do, not what they look like.

Our solution extracts implicit structure from LLMs and distills it into a lightweight classifier. The key insight: transformer attention patterns reveal semantic spans. When processing "she sold half her eggs," the attention weights show "half," "eggs," and "sold" attending to each other. That cluster is the model recognizing a single operation: multiply eggs by 0.5.

We extract attention matrices from a 7B model on 10K math problems and discovered two orthogonal signals. First, attention magnitude from numbers to verbs: state verbs like "has" produce ~0.05, action verbs like "sold" produce ~0.07. This distinguishes SET from transformations. Second, verb embeddings cluster semantically — "sold/spent/lost" cluster together (subtraction), "bought/received/earned" cluster together (addition).

We use Welford's algorithm to learn these thresholds online rather than hardcoding. Classification becomes a z-score: how many standard deviations from each operation's learned mean?

After classifying spans, we build a computation graph. Entity binding comes from attention sinks — all spans attend to the problem's subject, telling us they chain together. "Lay 16 eggs" → SET(16), "eats 3" → SUB(result, 3), execute in order.

## Dual-Signal Architecture (Path C Hybrid)

**Two orthogonal signals for span detection and classification:**

1. **Attention patterns** — Structural relationships between tokens (which tokens attend to each other)
2. **Embeddings** — Semantic similarity for template matching

**The hybrid approach:**
- **Training time**: Extract attention patterns from Qwen 7B (large model) to learn span templates
- **Inference time**: Use fine-tuned MiniLM (22M params, 318x smaller) to detect spans and match against learned templates

This gives us the quality of a 7B model at the cost of a 22M model.

## Attention Distillation Breakthrough

**Key discovery**: Model architecture matters more than size for attention correlation.

We tested correlation between various models and Qwen 7B attention patterns:

| Model | Params | Architecture | Correlation with Qwen 7B |
|-------|--------|--------------|--------------------------|
| **MiniLM-L6 (fine-tuned)** | **22M** | Encoder | **0.945** |
| MiniLM-L6 (baseline) | 22M | Encoder | 0.58 |
| MiniLM-L12 | 33M | Encoder | 0.54 |
| RoBERTa-large ST | 355M | Encoder | 0.52 |
| ELECTRA-base | 110M | Encoder | 0.49 |
| Qwen2-0.5B | 500M | Decoder | 0.31 |
| BERT-base | 110M | Encoder | 0.30 |

**Surprising findings:**
1. Smaller sentence-transformer models beat larger generic models
2. Same-family Qwen2-0.5B only achieves 0.31 correlation with Qwen 7B
3. Bidirectional encoders (MiniLM) correlate better than causal decoders (Qwen, GPT)
4. MiniLM's prior attention distillation training made it ideal for further distillation

**Fine-tuning approach:**
- Train MiniLM to match Qwen's connectivity patterns (span groupings)
- Learn optimal head weights (heads 5 & 8 most important)
- Learn optimal layer weights (layers 4 & 5 most important)
- Result: 0.58 → 0.945 correlation (+63% improvement)

**Model location**: `models/minilm_attention_finetuned.pt` (90MB)

## New Favorite Pattern
We want to consolidate methods - for example all database connections should go through a data layer instead of having multiple database connections.  Same with Signature creation, or leaf_node rejection of dag_steps.  We want to consolidate method calls for features to simplify our codebase and reduce the chance of bugs


# How to use Beads

```bash
bd prime        # Load context from beads
bd ready        # See available work
```

**When you encounter a bug or feature idea, create a beads issue to track it.**

```bash
bd create --title="Bug: description" --type=bug
bd create --title="Feature: description" --type=feature
```

Don't fix and forget - always track issues in beads.

## Workflow

1. Check `bd ready` for available issues
2. `bd update <id> --status=in_progress` to claim work
3. Make changes
4. `bd close <id> --reason="..."` when done
5. `bd sync` to sync changes

See `AGENTS.md` for detailed guidance.
