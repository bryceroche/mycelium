# Mycelium v6: C1–C6 Training Guide (v3)

All models: Qwen-0.5B-Instruct with task-specific heads. 7B teacher used only at training time via IAF extraction. Everything in inference is learned — no heuristics. Sympy is the executor (calculator), not a reasoner.

**Core principle: Progressive wave function collapse.** No component forces a hard decision. Each outputs a distribution or ranked hypotheses. Only sympy forces full collapse.

---

## C1 — Relevance Scorer (NOT a segmenter)

**Task:** Given problem text, output a relevance score (0–1) per token. No spans. No boundaries. A continuous field.

**Head:** Regression (single float per token). MSE loss.

**Labels:** Directly from IAF top_positions. For each computation step the teacher performs, certain problem tokens receive high attention. Normalize per-token cumulative attention across all steps to [0, 1]. That's the label. No thresholding, no IO conversion, no contiguity forcing.

**Why not IO tagging:** The teacher's attention is scattered (non-contiguous clusters of 1–3 tokens), not ribbons. Forcing contiguous spans = premature collapse = 34.5% F1. The relevance field preserves the teacher's actual attention structure.

**Inference output:** A soft heatmap over problem tokens. Downstream models decide what to attend to — C1 doesn't commit.

---

## C2 — Classifier (Relevance Field → Template)

**Task:** Given problem text + C1 relevance field, classify each high-relevance cluster into one of 100 IB-discovered templates.

**Head:** Sequence classification, 100 classes.

**Labels:** 17,101 IB-labeled spans from β-annealing (100 templates, 100% operator purity). Includes renamed T54_RESULT (final answers) and T42_DEFINE (variable setup).

**Training input format:** Problem text with token-level relevance weights from teacher attention (ground truth C1 output). At inference, uses C1's predicted relevance field.

**Inference output:** Top-3 templates with probabilities per cluster. Primary MCTS branching point. High confidence (>90%) collapses to one; ambiguous clauses maintain 2–3 hypotheses.

**Forward constraint:** Template restricts what C3 extracts (QUADRATIC_SOLVE → polynomial coefficients).

---

## C3 — Extractor (Template + Relevance → Expression)

**Task:** Given problem text + relevance field + template, extract sympy-parseable expression.

**Head:** Generative (LM head, beam search width 2–3).

**Labels:** Merged-span training pairs. IAF semantic signal (which clause) + top_positions (which operands) → sympy expression. Only include pairs where execution matches gold answer.

**Key insight:** Merged spans solve the 84.7% missing operand problem. On MATH, 56% of expressions have implicit operands elsewhere in the text. The relevance field from C1 + operand positions from IAF give C3 both context and numbers without requiring contiguous spans.

**Inference output:** Top-3 candidate expressions via beam search. Second MCTS branching point.

---

## C4 — Bridging (Implicit Operations) — DEFERRED

Detects implicit operations (unit conversions, algebraic rearrangements) from low-IAF generation spans where the teacher computes internally. **Train only if error attribution shows C3 missing implicit steps.**

---

## C5 — Dependency Resolver (DAG Wiring)

**Task:** For each pair of operation clusters, predict whether one depends on the other's output.

**Head:** Pairwise classification (DEPENDS / INDEPENDENT).

**Labels:** From IAF traces — when generation step_j attends to tokens generated during step_i (low IAF + attention to prior output), that's a dependency edge.

**Inference output:** Edge probabilities. Usually unambiguous (1–2 DAG hypotheses). Entanglement: dependencies constrain C3 operand choices.

---

## C6 — Goal Resolver (Answer Type)

**Task:** Determine required answer format (integer, fraction, set, expression, etc.). **Head:** Sequence classification. **Labels:** T54_RESULT spans from IB + gold answer format. Runs early in pipeline — backward constraint prunes MCTS paths with incompatible output format.

---

## Inference: Wave Function Collapse

```
Problem → C1 (relevance field, no collapse) → C6 (answer type, backward constraint)
  → C2 (top-3 templates per cluster) → C3 (top-3 expressions via beam)
  → C5 (DAG wiring) → C4 (bridging if needed) → Sympy (FULL COLLAPSE)
  → Select: majority vote or highest confidence among valid results
```

Easy problems: 1–3 paths, ~15ms. Hard problems: 20–50 paths, ~200ms. All ~1000× cheaper than 70B CoT.

---

## Training Priority

| Order | Model | Data Status | Action |
|-------|-------|-------------|--------|
| 1 | C1 | IAF top_positions ready | Retrain as regression head |
| 1 | C2 | 17,101 IB pairs ready | Train now |
| 2 | C3 | Building merged-span pairs | Train on GSM8K + MATH combined |
| 2 | C5 | Building pairwise pairs | Train from IAF cross-step attention |
| 2 | C6 | Building from T54 + gold | Train from IB results |
| 3 | C4 | Deferred | Only if error attribution demands it |

C1 and C2 train in parallel. C3, C5, C6 train in parallel once data is built. ~3B total parameters at inference.
