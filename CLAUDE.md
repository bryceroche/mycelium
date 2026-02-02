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

**Our approach:** Extract the *structure* that big models learn, then execute it deterministically

The insight is that large models learn span→operation mappings implicitly in their attention patterns. We make that explicit.

### The Problem with Lookup Tables
A raw mapping table would explode - "half the X" vs "half of X" vs "50% of X" vs "X divided by two" are all the same operation with infinite surface forms.

### The Solution: Tiny Encoder + Classifier
Instead of string matching, we embed spans and classify into operations:

```
span text → tiny encoder (distilBERT, ~66M params) → operation_id
```

The encoder learns that "half the X" ≈ "50% of X" in embedding space. We're distilling the mapping table into weights.

**Why this works:** We're not doing open-ended generation - we're doing **classification into a finite set of operations**. That's fundamentally easier than general LLM reasoning.

### Training Pipeline
1. **One-time extraction:** Use big model attention to learn which spans cluster together
2. **Generate (span, operation) pairs:** From attention analysis on solved problems
3. **Train tiny classifier:** ~50-100M param model to map spans → operation_ids
4. **Inference is tiny:** Embed span, classify, execute the operation

You pay the "big model tax" once during training to extract decomposition patterns. At inference:
- Tokenize
- Embed spans (tiny encoder)
- Classify → operation_id
- Execute arithmetic

**The bet:** Math reasoning isn't about "intelligence" — it's about recognizing which operation template applies. Big models learned this implicitly. We extract it into a tiny classifier.

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
