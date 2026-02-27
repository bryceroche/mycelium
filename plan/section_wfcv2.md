# Mycelium WFC Architecture: Build Prompt

## Context

We've been iterating on a 6-model pipeline (C1→C2→C3→C4→C5→C6) for math problem solving. C1 (segmenter/relevance scorer) has failed repeatedly:
- IO tagging: 34.5% F1
- Regression (global): labels were 93% hot — flat, useless
- Multi-channel (per-step): channels collapsed to identical patterns because teacher attention genuinely overlaps across steps

**Key insight from Wave Function Collapse theory:** we were forcing the wrong measurement first. Position measurement (which tokens belong to which step) is low-information because operands are shared across steps. TYPE measurement (what operations does this problem need) is high-information and collapses the hypothesis space sharply.

**Decision: Kill C1. C2 becomes the first measurement.**

## New Architecture (4 learned models + sympy)

```
Problem text (full superposition — no segmentation)
    → C2: "what operations?" (type collapse)
    → C3: "what expression for each operation?" (position+value collapse)
    → C5: "what depends on what?" (dependency collapse)
    → C6: "what answer format?" (output constraint)
    → Sympy: execute DAG (full collapse)
```

## C2 — Operation Set Detector (FIRST measurement)

**Old task:** Given a segmented clause, classify it into one template.
**New task:** Given FULL problem text, predict the SET of operations needed.

This is multi-label classification, not single-label. A problem might need {DIVIDE, ADD} or {QUADRATIC_SOLVE, SUBSTITUTE, EVALUATE}.

**Head:** Multi-label classification over 58 IB templates (or 38 operator classes). Sigmoid per class, not softmax. Threshold at 0.5 for each.

**Training data:** From IAF data, each problem has N computation steps, each labeled with an IB template. The multi-label target is the set of all templates used in that problem.

**Input:** Full problem text (no segmentation, no relevance field).
**Output:** Set of operation types with confidences. E.g., {DIVIDE: 0.92, ADD: 0.87, MUL: 0.34}

**MCTS branching:** When confidence is ambiguous (MUL at 0.34 — include or not?), branch: one path includes MUL, one doesn't.

**Existing C2 model:** Trained on 17,101 IB-labeled pairs at 73.2% accuracy. This was single-label on clauses. Needs retraining as multi-label on full problem text.

**Training data location:** s3://mycelium-data/ — IB template labels in ib_results/, IAF data in medusa backups.

## C3 — Expression Extractor (SECOND measurement — per operation)

**Old task:** Given a clause + template, extract expression.
**New task:** Given FULL problem text + ONE operation type from C2, extract the sympy expression for that operation.

C3 runs once PER operation that C2 identified. If C2 says {DIVIDE, ADD}, C3 runs twice:
- C3(problem_text, DIVIDE) → "48/2"
- C3(problem_text, ADD) → "24 + 48"

C3's own attention mechanism learns WHERE the operands are — it doesn't need C1 to tell it. The operation type constrains the search: DIVIDE + "half as many" + numbers in the text is enough signal.

**Head:** Generative (LM head, beam search width 2-3).
**Input format:** `[TEMPLATE: DIVIDE] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.`
**Output:** `48 / 2`

**Training data:** Same merged-span pairs we already built (22,419 at data/extraction_multispan.json), but reformat input to be full problem text + template tag instead of extracted clause + template.

**MCTS branching:** Top-3 beams per operation. Second major branching point.

**Existing C3 model:** s3://mycelium-data/models/c3_extractor_partial_freeze_v1/ — needs retraining with new input format.

## C5 — Dependency Resolver (unchanged)

**Task:** Given all operations and their extracted expressions, predict pairwise dependencies.
**Head:** Pairwise classification (DEPENDS/INDEPENDENT).
**Data:** Structural detection — if step_j uses step_i's computed result as operand → DEPENDS.
**Currently training.** Subsample to ~50K examples (keep all DEPENDS, 5x INDEPENDENT).

## C6 — Answer Type (unchanged)

**Task:** Given problem text, predict answer format.
**Head:** Sequence classification.
**Currently retraining** on MATH-only data with clipped weights (max 5x).

## C4 — Bridging (DEFERRED, probably unnecessary)

With C3 seeing the full problem text per operation, it has all the context it needs. Bridging was a patch for when C3 only saw a narrow clause. May not be needed. Validate after C2→C3→C5→C6→sympy is working.

## E2E Inference Flow

```python
# 1. C2: What operations does this problem need?
operations = C2(problem_text)  # e.g., {DIVIDE: 0.92, ADD: 0.87}

# 2. C6: What answer format?
answer_type = C6(problem_text)  # e.g., INTEGER

# 3. C3: For each operation, extract expression (with beam search)
expressions = {}
for op in operations:
    expressions[op] = C3.beam_search(problem_text, op, beam_width=3)

# 4. C5: Wire dependencies between operations
dag = C5(expressions)

# 5. MCTS: Explore combinations of beams
for candidate_path in mcts_explore(expressions, dag):
    result = sympy.execute(candidate_path)
    if result.valid and matches_type(result, answer_type):
        candidates.append(result)

# 6. Select: majority vote or highest confidence
return select_answer(candidates)
```

## Immediate Actions

1. **Rebuild C2 training data** as multi-label (problem_text → set of operation types). Source: existing IB labels grouped by problem.
2. **Rebuild C3 training data** as (full_problem_text + template_tag → expression). Source: existing extraction pairs, just change input format.
3. **Train C2** multi-label classifier. Frozen backbone first, unfreeze last 4 if <80%.
4. **Train C3** with new input format. Partial freeze (layers 0-15 frozen, 16-23 + LM head trainable).
5. **Run E2E test** on 50-100 problems: C2→C3→sympy (skip C5/C6 initially, assume sequential).
6. **Error attribution** on the 50-100 results to find next bottleneck.

## Key Files

- IB templates: s3://mycelium-data/ (ib_results/)
- IAF data: s3://mycelium-data/backups/medusa_iaf_2026-02-26/ (16.3GB)
- C3 training pairs: data/extraction_multispan.json (22,419 pairs)
- Existing models: s3://mycelium-data/models/
- Training guide: See mycelium_c1_c6_training_guide_v3.md (note: C1 is now KILLED)

## Success Criteria

C2→C3→sympy on 50 MATH problems. Target: beat 17.2% (Track 1 baseline). If this works with 4 models, we've validated that C1 was unnecessary overhead and the WFC-informed architecture is correct.
