# The Big 4
1. Attention Signals (Entropy, Received, Connectivity)
2. Dual-Signal Architecture (Attention + Embeddings)
3. Attention Distillation (Qwen 7B → MiniLM 22M)
4. How to use Beads

## Terminology
- **Attention Entropy** — Low entropy = important token (focused attention). High entropy = diffuse attention.
- **Attention Received** — Which tokens get looked back to. High received attention = structurally important (entities, operators).
- **Span Connectivity** — How strongly tokens within a span attend to each other. High connectivity = cohesive semantic unit.
- **Centroid Embedding** — Average embedding of a span's tokens. Used for template matching.
- **Welford's** — Online algorithm for calculating running mean/variance without storing all data.
- **DAG** — Directed Acyclic Graph (computation graph).
- **Span** — Contiguous tokens forming a semantic unit (e.g., "half the eggs").
- **SET** — Initial value assignment operation.
- **Attention Sink** — Token that receives attention from many others (usually the subject/entity).
- **Z-score** — Standard deviations from learned mean, used for classification.

## Core Principle: Failures Are Valuable Data Points
**Let the system fail.** This is how it learns.
- Record every failure — it feeds the learning loop
- Do not fallback to LLM reasoning
- Accumulated failure patterns (not individual failures) refine thresholds
- Success/failure stats drive classification decisions

The goal is NOT 100% accuracy on every run. The goal is collecting data that makes the system smarter over time. A misclassified span provides valuable signal for threshold adjustment.

## Attention Signals

Three signals extracted from attention matrices:

**1. Attention Entropy (per token)**
- Low entropy → token attends to specific targets → important structural role
- High entropy → token attends broadly → less discriminative
- Use case: Identify operators and key nouns

**2. Attention Received (per token)**
- Sum of attention each token receives from all other tokens
- High received → many tokens look back to this one → entity or anchor
- Use case: Find subjects ("Janet"), referenced quantities

**3. Span Connectivity (per span)**
- Average mutual attention between tokens in a candidate span
- High connectivity → tokens form cohesive unit → valid span
- Low connectivity → tokens don't belong together → split or reject
- Use case: Validate span boundaries, detect multi-token operations

## Ground Truth from Qwen 7B

Qwen 7B attention patterns capture:
- **Span structure** — Which tokens group together (connectivity)
- **Entity binding** — Which tokens reference the same entity (received attention)
- **Centroid embeddings** — Semantic fingerprint of each operation type

We extract these patterns on 10K math problems to build a library of span templates with their attention signatures and centroid embeddings.

## Attention Distillation: Qwen 7B → MiniLM 22M

**The problem**: Qwen 7B is too expensive for inference.

**The solution**: Distill attention patterns into MiniLM (318x smaller).

**Why MiniLM works:**
- Bidirectional encoder (sees full context, unlike causal Qwen)
- Prior distillation training made it a good student
- Sentence-transformer architecture already optimized for semantic similarity

**Fine-tuning process:**
1. Extract Qwen 7B attention matrices on training set
2. Train MiniLM to match Qwen's span connectivity patterns
3. Learn optimal head weights (heads 5 & 8 most important: 0.108)
4. Learn optimal layer weights (layers 4 & 5 most important: 0.18)
5. Result: 0.58 → 0.945 correlation (+63% improvement)

**Distillation results:**

| Model | Params | Correlation with Qwen 7B |
|-------|--------|--------------------------|
| **MiniLM-L6 (fine-tuned)** | **22M** | **0.945** |
| MiniLM-L6 (baseline) | 22M | 0.58 |
| Qwen2-0.5B | 500M | 0.31 |
| BERT-base | 110M | 0.30 |

Key insight: Architecture > size. Same-family Qwen2-0.5B only achieves 0.31 correlation.

## Dual-Signal Architecture

**Two orthogonal signals for robust matching:**

1. **Attention signal** — Structural relationships (which tokens attend to each other)
2. **Embedding signal** — Semantic similarity (centroid distance)

**Why dual signals?**
- Attention alone can have false positives (similar structure, different meaning)
- Embeddings alone miss structural relationships
- Combined: more robust span detection and classification

**The hybrid pipeline:**
- **Training**: Qwen 7B → extract attention patterns + centroid embeddings → span templates
- **Inference**: MiniLM (22M) → match attention patterns + embeddings → classify spans

Quality of 7B model at cost of 22M model.

## New Favorite Pattern
Consolidate methods. All database connections go through a data layer. All span detection through one interface. All embedding lookups through cache. Reduces bugs, simplifies codebase.

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
