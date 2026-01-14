# Architecture Discussion Notes

These notes capture the brainstorming session about refactoring mycelium from DSL-based execution to a function pointer architecture.

## Documents

| # | File | Topic |
|---|------|-------|
| 1 | [01_separation_of_concerns.md](01_separation_of_concerns.md) | LLM vs Tree vs Python responsibilities |
| 2 | [02_what_we_can_drop.md](02_what_we_can_drop.md) | DSL removal, simplified leaf nodes |
| 3 | [03_codebase_audit.md](03_codebase_audit.md) | Keep/Modify/Drop file analysis |
| 4 | [04_new_architecture.md](04_new_architecture.md) | File structure with status |
| 5 | [05_tree_size_estimation.md](05_tree_size_estimation.md) | ~300-1000 signatures at maturity |
| 6 | [06_signatures_as_examples.md](06_signatures_as_examples.md) | Using signatures to teach LLM |
| 7 | [07_network_shape.md](07_network_shape.md) | Narrow-wide-narrow architecture |
| 8 | [08_performance_analysis.md](08_performance_analysis.md) | Brute force is fast enough |
| 9 | [09_flat_prototype_pivot.md](09_flat_prototype_pivot.md) | Hierarchy → flat prototype store |

---

## Key Decisions

1. **Drop DSL** - Just call Python functions directly
2. **Function Registry** - Curated list of ~57-200 functions
3. **Signatures as Prototypes** - Multiple semantic examples per function
4. **Flat Store** - Brute-force k-NN, no tree traversal needed
5. **LLM Decomposition** - With signature examples as few-shot guidance

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
