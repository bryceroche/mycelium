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

**Our approach:** Extract the *structure* that big models learn, then execute it deterministically

The insight is that large models learn span→operation mappings implicitly in their attention patterns. We make that explicit:

1. **One-time extraction:** Use big model attention to learn which spans cluster together
2. **Build a mapping table:** "half the X" → `X * 0.5`, etc.
3. **Inference is tiny:** Just pattern match spans and execute the graph — no LLM needed

You pay the "big model tax" once during training to learn the decomposition patterns. At inference, it's just:
- Tokenize
- Match spans (lookup table)
- Execute arithmetic

**The bet:** Math reasoning isn't about "intelligence" — it's about recognizing which operation template applies. Big models learned this implicitly. We extract it into something a tiny model (or no model) can run.

## License
MIT — Bryce Roche ([github.com/bryceroche/mycelium](https://github.com/bryceroche/mycelium))
Built with [Claude Code](https://claude.ai/claude-code)
