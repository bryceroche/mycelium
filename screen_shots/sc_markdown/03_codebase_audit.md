# Codebase Audit

## KEEP (Core Infrastructure)

| File | Purpose | Why Keep |
|------|---------|----------|
| `data_layer/connection.py` | SQLite connection management | Database access layer |
| `data_layer/schema.py` | Database schema | Store signatures, stats |
| `data_layer/mcts.py` | DAG tracking, post-mortem | Track (step, signature) pairs |
| `step_signatures/models.py` | StepSignature dataclass | Leaf node representation |
| `step_signatures/db.py` | Signature CRUD, routing | Tree operations |
| `step_signatures/utils.py` | Cosine similarity, centroid cache | Embedding math |
| `embedder.py` | Generate embeddings | Semantic routing |
| `embedding_cache.py` | Cache embeddings | Performance |
| `config.py` | Configuration | Settings |

---

## MODIFY (Adapt to New Architecture)

| File | Current | Change To |
|------|---------|-----------|
| `step_signatures/models.py` | `dsl_script: str` | `func_pointer: str` (registry key) |
| `step_signatures/db.py` | Routes to DSL | Routes to function |
| `solver.py` | Orchestrates DSL execution | Orchestrates function calls |
| `mathdecomp/schema.py` | Has `op: Operator` enum | Remove, use `func: str` |
| `mathdecomp/executor.py` | Custom execution | Call function registry |

---

## DROP (No Longer Needed)

| File | Why Drop |
|------|----------|
| `step_signatures/dsl_executor.py` | No more DSL - just call functions |
| `step_signatures/dsl_types.py` | DSL types obsolete |
| `step_signatures/math_layer.py` | AST-based eval → direct function call |
| `step_signatures/sympy_layer.py` | Just use `sympy.solve` directly |
| `gts_model.py` | GTS prefix notation → LLM decomposition |
| `gts_decomposer.py` | GTS wrapper → LLM decomposer |
| `expression_tree.py` | GTS expression trees → function DAG |
| `local_decomposer.py` | Stub - replace with mathdecomp |
| `chain_nodes.py` | Chaining is now just DAG of function calls |

---

## NEW (Need to Create)

| File | Purpose |
|------|---------|
| `function_registry.py` | Curated Python function pointers |
| `mathdecomp/llm_decomposer.py` | LLM recursive decomposition prompts |

---

## Summary

- **KEEP:** 9 files (data layer, signatures, embeddings)
- **MODIFY:** 5 files (swap DSL → function pointers)
- **DROP:** 8 files (DSL execution, GTS, expression trees)
- **NEW:** 2 files (function registry, LLM decomposer)

**Net result:** Simpler codebase. The DSL layer (4 files) becomes 1 file (function registry).
