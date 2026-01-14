# Mycelium Architecture Summary

## The Three Components

| Component | Role | Learns |
|-----------|------|--------|
| **LLM** | Recursive decomposition, argument extraction | How to decompose (via prompting) |
| **Tree** | Semantic → function mapping, k-NN classification | Which descriptions map to which functions |
| **Python** | Deterministic execution | Nothing - just executes |

---

## The Refactor: DSL → Function Pointers

| Before | After |
|--------|-------|
| Leaf node = DSL code string | Leaf node = function pointer |
| Execute = eval DSL | Execute = `call_function(name, *args)` |
| LLM writes code | LLM just decomposes |
| Complex DSL executor (4 files) | Simple function registry (1 file) |

---

## File Changes

- **Keep (9):** data_layer/*, step_signatures/{models,db,utils}.py, embedder, config
- **Modify (5):** solver.py, models.py, db.py, schema.py, executor.py
- **Drop (8):** dsl_executor, dsl_types, math_layer, sympy_layer, gts_*, expression_tree, chain_nodes
- **New (2):** function_registry.py, grader.py

---

## Tree Size at Maturity

```
Functions:     ~150-200 (current: 57)
Signatures:    ~300-1000 (2-5 semantic variants per function)
```

One function can have multiple signatures:
```
add:
  ├── "combine two prices"       → signature_42
  ├── "sum the quantities"       → signature_87
  └── "total distance traveled"  → signature_156
```

---

## Network Shape

```
INPUT (narrow)     →  1 step embedding
       ↓
MIDDLE (wide)      →  300-1000 signature prototypes
       ↓
OUTPUT (narrow)    →  1 of 200 functions
```

The wide middle captures **many-to-one** mapping from natural language → functions.

---

## Signatures as Examples (Feedback Loop)

1. Tree learns which descriptions succeed for which functions
2. High-success signatures become few-shot examples for LLM
3. LLM produces better decompositions
4. Tree gets even better signal

**Signatures become a learned vocabulary** - curated from experience.

---

## Performance

| Signatures | Brute-Force k-NN | Notes |
|------------|------------------|-------|
| 1,000 | < 0.1ms | Trivial |
| 5,000 | < 0.5ms | Still trivial |
| 50,000 | ~5ms | Still fine |

**Real bottleneck:** Embedding API call (~100-500ms), not search.

---

## The Vision

```
Problem Text
     │
     ▼
┌─────────────────────────────────┐
│  LLM Decomposer                 │
│  (with signature examples)      │
└─────────────────────────────────┘
     │
     ▼ [atomic steps with func names]
┌─────────────────────────────────┐
│  Signature Store (k-NN)         │
│  300-1000 prototypes            │
└─────────────────────────────────┘
     │
     ▼ [best matching signature]
┌─────────────────────────────────┐
│  Function Registry              │
│  call_function(name, *args)     │
└─────────────────────────────────┘
     │
     ▼
   Result
```

---

## Key Decisions

1. **Drop DSL** - Just call Python functions directly
2. **Function Registry** - Curated ~57-200 functions across 7 tiers
3. **Signatures as Prototypes** - Multiple semantic examples per function
4. **Flat Store** - Brute-force k-NN (no tree traversal needed at this scale)
5. **LLM Decomposition** - With signature examples as few-shot guidance
