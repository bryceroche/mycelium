# Pivot: Flat Prototype Store

## The Realization

The current epic is about zero LLM calls for decomposition. But our brainstorming has pivoted to:

- **LLM does decomposition** (with signature examples as guidance)
- **Tree is a flat classifier** (prototype store, not hierarchy)

This is a significant pivot.

---

## Code Changes Needed

| Area | Current | New |
|------|---------|-----|
| Tree structure | Hierarchical (routers + leaves) | Flat prototype store |
| Routing | Tree traversal | Brute-force k-NN |
| Decomposition | Local text splitter (planned) | LLM with signature examples |
| schema.py | signature_relationships table | Maybe not needed |
| db.py | Complex tree operations | Simple CRUD + k-NN |
| New file | - | function_registry.py |

---

## Beads Issues - Proposed Updates

### Close/Obsolete:
- `mycelium-1b8w.3` (local text splitter) - LLM does this now

### Update:
- `mycelium-1b8w` (epic) - Reframe around flat classifier + LLM decomposer
- `mycelium-1b8w.4` (recursive decomposition) - Now LLM-powered

### Keep:
- `mycelium-1b8w.2` (coverage/residual) - Still useful
- `mycelium-1b8w.5` (failure tracking) - Still relevant

### New issues needed:
- Flatten tree to prototype store
- Create function_registry.py
- Build LLM decomposer with signature examples
