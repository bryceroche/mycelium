# Separation of Concerns

## The Three Components

| Component | Role |
|-----------|------|
| **LLM** | Natural language understanding, recursive decomposition, argument extraction |
| **Tree** | Learns semantic → function mapping, clusters similar operations, MCTS routing |
| **Python** | Deterministic execution, guaranteed correct arithmetic |

## What Each Component Learns

| Component | What it learns |
|-----------|----------------|
| LLM | How to decompose (via prompting/fine-tuning) |
| Tree | Which semantic descriptions map to which functions (via MCTS post-mortem) |
| Python | Nothing - it's deterministic |

---

## What Maps Directly

| New Component | Existing Infrastructure | Status |
|---------------|------------------------|--------|
| Tree Routing | `step_signatures.py`, embeddings, MCTS, Welford | Keep |
| Decomposition Schema | `mathdecomp/schema.py` | Keep (simplify) |
| Variance Tracking | Welford algorithm | Keep |
| Clustering | Embedding similarity | Keep |

---

## What Changes

| Old | New |
|-----|-----|
| Leaf node = DSL code string | Leaf node = function pointer |
| Execute = eval DSL | Execute = call Python function |
| LLM writes DSL | LLM just decomposes (no code) |
