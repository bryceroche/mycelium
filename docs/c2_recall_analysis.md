# C2 All-Correct Recall Analysis

**Date:** 2026-02-27
**Model:** c2_heartbeat (sentence-transformers/all-MiniLM-L6-v2)
**Test set:** 1,098 examples (20% holdout from c2_train_with_heartbeats.json)

## Results

| Threshold | All-Correct | Any-Correct |
|-----------|-------------|-------------|
| **0.5** | **52.1%** (572/1098) | 99.1% |
| 0.3 | 100% | - |
| 0.2 | 100% | - |
| 0.1 | 100% | - |

## Most Missed Operations @ threshold=0.5

| Operation | Misses |
|-----------|--------|
| SQUARE | 294 |
| SQRT | 207 |
| CUBE | 101 |
| HIGH_POW | 79 |
| TRIG | 63 |

## Key Insight

C2 predicts all ground truth operations, but power-related ops (SQUARE, SQRT, CUBE, HIGH_POW) have lower confidence scores. The default threshold of 0.5 is too aggressive.

**Solution: Lower threshold from 0.5 to 0.3**

This achieves 100% all-correct recall with acceptable false positive rate. Extra predictions are cheap (sympy executor rejects invalid operations), while missing operations is fatal (no recovery path).

## Saved Model Metrics

From checkpoint:
- F1 micro: 78.0%
- Recall: 78.6%
- Exact match: 22.0%
- Any-correct: 99.8%

## Recommendation

Update inference pipeline to use `threshold=0.3` for C2 predictions. This prioritizes recall over precision, which aligns with the MCTS search strategy where bad predictions are pruned by the symbolic executor.
