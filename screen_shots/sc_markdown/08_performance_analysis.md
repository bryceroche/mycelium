# Performance Analysis

## 5k Signatures - Brute Force

Even with 5k signatures we can still do fast brute-force nearest neighbor:

```python
# Vectorized with numpy (normalized embeddings)
all_centroids = np.array([sig.centroid for sig in signatures])  # Shape: (5000, 768)
query = step_embedding  # Shape: (768,)

similarities = all_centroids @ query  # Single matrix-vector multiply
best_idx = np.argmax(similarities)
```

**Time: ~0.1 - 0.5ms**

NumPy does 5000 × 768 = 3.8M multiply-adds almost instantly.

---

## For Reference

| Signatures | Brute Force | Notes |
|------------|-------------|-------|
| 1,000 | < 0.1ms | Trivial |
| 5,000 | < 0.5ms | Still trivial |
| 50,000 | ~5ms | Still fine |
| 500,000 | ~50ms | Maybe consider FAISS |

---

## The Real Bottleneck

| Operation | Time | Notes |
|-----------|------|-------|
| Embedding API call | ~100-500ms | **THIS dominates** |
| Brute force search | ~0.5ms | Rounding error |
| Function execution | ~0.01ms | Instant |

The embedding API call is 100-1000x slower than the search. At this scale, we don't need FAISS or ANN - brute force is fine.
