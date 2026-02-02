# Mycelium

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

**The bet:** Math reasoning isn't about "intelligence" — it's about recognizing which operation template applies. Big models learned this implicitly. We extract it via embeddings.

## License
MIT — Bryce Roche ([github.com/bryceroche/mycelium](https://github.com/bryceroche/mycelium))
Built with [Claude Code](https://claude.ai/claude-code)
