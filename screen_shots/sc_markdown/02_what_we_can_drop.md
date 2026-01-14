# What We Can Drop

## Removed Components

- **DSL generation** - no more LLM writing code
- **DSL parsing/eval** - just call functions directly
- **Complex execution engine** - Python does it for us

---

## The Core That Remains

```
┌─────────────────────────────────────┐
│          SIGNATURE TREE             │
│                                     │
│  • Embeddings for semantic similarity│
│  • Welford for variance tracking    │
│  • MCTS for exploration/routing     │
│  • Leaf nodes = function pointers   │
└─────────────────────────────────────┘
```

---

## Simplified Leaf Node

**Before:**
```python
leaf_node = {
    "id": "add_quantities",
    "dsl_code": "def solve(a, b): return a + b",  # LLM-generated
    "embedding": [...],
}
```

**After:**
```python
leaf_node = {
    "id": "add",
    "func": operator.add,  # Python function pointer
    "embedding": [...],    # For routing
}
```

---

## What's Actually New

1. **Function Registry** - curated list of Python functions
2. **LLM Decomposer** - prompts for recursive decomposition (we started this in mathdecomp)
3. **Simpler execution** - just call functions

---

**Bottom line:** We keep the tree, embeddings, Welford, MCTS. We swap DSL for function pointers.
