# The Big 3
1. Attention-Based Decomposition
2. New Favorite Pattern
3. How to use Beads

## Attention-Based Decomposition
**Decomposition is the crux.** Everything downstream of step-level intermediate representation (IR) is solved.

Transformer attention reveals semantic spans. The "Panama Hats" problem: "panama" = country, but "panama hats" = completely different meaning. We need the **longest span** that matches a semantic unit.

In math: "half the price of the cheese" is ONE operation, not three.

```
Attention patterns reveal these spans:
- "half" attends strongly to "price" AND "cheese"
- This cluster = single semantic unit
- Map span → operation: cheese_price * 0.5
```

**The Vision:**
```
Math Problem (NL)
       ↓
Transformer Attention Weights
       ↓
Span Clustering (find semantic units)
       ↓
Span → Operation Mapping (learned)
       ↓
Computation Graph / AST
       ↓
Execute (solved)
```

**Span → Operation Patterns:**
- "half the X" → `X * 0.5`
- "X more than Y" → `Y + X`
- "X percent of Y" → `Y * (X/100)`
- "twice as many as X" → `X * 2`

This mapping can be learned from (problem, solution) pairs using attention analysis.

## Structural Distillation

This is a different kind of distillation:

**Traditional distillation:** Train small model to mimic big model's outputs (soft labels, behavior cloning)

**Our approach:** Extract the *structure* that big models learn, then train tiny encoder + classifier.

The insight is that large models learn span→operation mappings implicitly in their attention patterns. We make that explicit.

### Prototype: Frozen Embeddings + Nearest Neighbor
Skip training entirely for the prototype. Use frozen embeddings and nearest neighbor:

```
span text → frozen encoder (sentence-transformers) → nearest neighbor → operation_id
```

1. Embed ~10 prototype spans per operation
2. At inference: embed query span, find nearest prototype
3. If it works, validates the approach before any training

### Later: Tiny Encoder + Classifier
Once prototype validates, optionally train for better accuracy:

```
span text → tiny encoder (distilBERT, ~66M params) → operation_id
```

**Why this works:** We're not doing open-ended generation - we're doing **classification into a finite set of operations**. That's fundamentally easier than general LLM reasoning.

### The Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  1. COLLECTION                                              │
│     GSM8K (10K problems)                                    │
│           ↓                                                 │
│     Attention Extraction (DeepSeek 7B)                      │
│           ↓                                                 │
│     SpanCollector → collected_spans.jsonl                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  2. CLUSTERING                                              │
│     Embed all spans → vectors                               │
│           ↓                                                 │
│     K-means cluster → ~100 operation clusters               │
│           ↓                                                 │
│     Spans that cluster = same operation                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  3. TAGGING                                                 │
│     Tag ONE span per cluster → propagates to all            │
│     10,000 spans → ~100 manual tags                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  4. GRAPH EXECUTION                                         │
│     Spans → Computation Graph → Execute → Answer            │
└─────────────────────────────────────────────────────────────┘
```

### Key Insight: Attention Shows Entity Binding

Everything in "Janet's ducks lay 16 eggs. She eats 3." attends to **Janet**.
This isn't noise — it shows all operations bind to the same entity.

```
           ┌──────────────────────────────────────┐
           │           "Janet"                    │ ← SUBJECT (attention sink)
           │         (the variable)               │
           └──────────────────────────────────────┘
                 ↑         ↑         ↑
           ┌─────┴──┐ ┌────┴───┐ ┌───┴────┐
           │ lay 16 │ │ eats 3 │ │bakes 4 │ ← OPERATIONS
           │  SET   │ │SUBTRACT│ │SUBTRACT│
           └────────┘ └────────┘ └────────┘
```

### Computation Graph

Named variables with multi-input references:
```python
Graph("Janet"):
  eggs = SET(16)           # "lay 16 eggs"
  after_eat = SUB(eggs, 3) # "eats three"
  final = SUB(after_eat, 4) # "bakes with four"
→ Result: 9
```

**The bet:** Math reasoning isn't about "intelligence" — it's about recognizing which operation template applies. Big models learned this implicitly. We extract it via clustering.

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
