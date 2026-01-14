# Signature Diversity and Learning

## Optimizing for Diversity

**Before (redundant):**
```
"calculate 20% of X"  ●●
"find 20% of total"   ●●
"compute 20% of Y"    ●
```

**After (diverse):**
```
"calculate 20%"       ●
"find the tip"        ●
"discount amount"     ●
"tax portion"         ●
```

Same function, but examples **span the semantic space**.

This also helps **deduplication** - avoid storing near-identical signatures.

---

## Runtime vs Post-Mortem Decision Matrix

| Situation | Action |
|-----------|--------|
| Runtime + high similarity | **Execute** |
| Runtime + low similarity | **Decompose further** |
| Post-mortem + success + close to existing | **Merge into signature** |
| Post-mortem + success + far from all | **Create new signature** |

Key insight: **Don't create signatures at runtime** - only learn from post-mortem successes.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────┐
│                    LLM DECOMPOSER                       │
│                                                         │
│  Input:  Problem + Signature Menu (diverse, quality-    │
│          weighted)                                      │
│  Output: Steps with descriptions                        │
│                                                         │
│  Menu = "Phrasebook" with evenly-spaced examples per    │
│         function                                        │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               CLASSIFICATION (per step)                 │
│                                                         │
│  embed(step) → k-NN → (func_name, similarity)           │
│                                                         │
│  similarity >= threshold?  → EXECUTE                    │
│  similarity < threshold?   → DECOMPOSE FURTHER          │
│  similarity not improving? → STOP, TRY EXECUTING        │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               POST-MORTEM (on success)                  │
│                                                         │
│  For each (step, func) that worked:                     │
│    - Close to existing sig? → MERGE (update centroid    │
│                               + descriptions)           │
│    - Far from all sigs?     → CREATE new signature      │
│                                                         │
│  Thresholds guided by Welford stats                     │
└─────────────────────────────────────────────────────────┘
```

---

## Key Decisions

| Component | Decision |
|-----------|----------|
| **Signature structure** | Prototype with multiple descriptions (centroid averaging) |
| **Merge threshold** | Welford-guided per-function |
| **Menu building** | Quality-weighted farthest-point sampling |
| **Runtime low-sim** | Decompose further (don't create sigs) |
| **Learning** | Post-mortem creates/merges from successes |
| **Recursion depth** | Similarity trend (continue while improving) |

---

## Ready to Implement

We have clear specs for:

1. **`LLMDecomposer`** - Takes problem + menu, outputs steps
2. **`solve_with_decomposition()`** - The main loop with similarity-trend recursion
3. **`learn_from_success()`** - Post-mortem signature creation/merging
4. **Enhanced `build_diverse_menu()`** - Quality-weighted sampling
