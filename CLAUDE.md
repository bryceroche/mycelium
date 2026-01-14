# Claude Code Instructions

## Quick Start

```bash
bd prime        # Load context from beads
bd ready        # See available work
```

## Design Philosophy: Build for Scale

**The system must grow on its own.** Avoid patterns that require manual maintenance or LLM fallback:

- **No hardcoded mappings** - If you're tempted to write a dict like `{"compute_gcd": "find greatest common divisor"}`, stop. Infer it dynamically from the data itself.
- **No manual enumerations** - New step_types, operators, or patterns should work automatically without code changes.
- **No LLM crutches** - The system should execute deterministically. LLM fallback means the DSL/signature failed—fix the root cause.
- **Self-extending** - Signatures, DSLs, and embeddings should compound. Each solved problem makes the next one easier.

Think: *"Will this still work when we have 10,000 signatures?"* If not, find another way.

## DAG of DAGs Structure

**The signature hierarchy must be a strict DAG.** No cycles allowed.

- Parent umbrellas route to child signatures
- Children can have multiple parents (DAG, not tree)
- Cycle detection runs on every `add_child()` call
- Self-references are rejected
- If adding A→B would create a cycle (B is already an ancestor of A), reject it

This structure enables:
- Deterministic routing (no infinite loops)
- Clean credit propagation upward
- Depth tracking for branching decisions
- Multiple inheritance (a signature can specialize multiple umbrellas)

The DB enforces this at write time. Don't try to work around it.

## Core Principle: Failures Are Data

**Let signatures fail.** This is how the system learns.

- Don't mask DSL failures with LLM execution
- Record every failure—it feeds the refinement loop
- Failed signatures get demoted, decomposed, or fixed
- Success/failure stats drive routing decisions

The goal is NOT 100% accuracy on every run. The goal is collecting data that makes the system smarter over time. A failed DSL today becomes training data for a better DSL tomorrow.

## Learning Mechanisms

### Centroid Averaging

Signature centroids are running averages of all matched embeddings. Each time a step matches a signature, the centroid updates:

```
new_sum = old_sum + new_embedding
new_count = old_count + 1
new_centroid = new_sum / new_count
```

This makes centroids more stable and representative over time. High-traffic signatures become semantic attractors.

### Credit Propagation

When a problem is solved correctly, success credit propagates up the signature DAG to parent umbrellas with decay:

- Direct signatures get +1 success
- Parent umbrellas get `decay^depth` credit (default 0.5 per level)
- Max propagation depth is configurable (default 3 levels)

This lets umbrella signatures accumulate credit from their children's successes, improving routing decisions.

### Depth-Based Branching

Shallow nodes (low depth) are more likely to **create new children** when decomposed. Deep nodes are more likely to **repoint to existing signatures**.

Why this works:
- Shallow = high uncertainty, worth exploring new branches
- Deep = should be converging on known patterns
- Natural depth limit: very deep nodes that still fail get auto-demoted
- Prevents unbounded depth while allowing organic growth at the frontier

The system self-limits: if you're 5 levels deep and still failing, you're on the wrong path. Let it fail, record the data, try a different route next time.

## DSL Bootstrap: Generate from Structure

**Don't wait for examples—generate DSLs from problem structure.**

When the planner extracts values like `{base: 2, exponent: 10}`, immediately create a working DSL:
```json
{"type": "math", "script": "base ** exponent", "params": ["base", "exponent"]}
```

This solves the cold-start problem:
- No need for successful examples to bootstrap
- DSLs are inferred from semantic parameter names
- The planner already knows the math structure—use it
- Each new problem type creates its own DSL pattern

The planner extracts values with semantic names (`total_distance`, `speed`, `time`). These names ARE the DSL—just connect them with the right operator.

## Key Rule

**When you encounter a bug or feature idea, create a beads issue to track it.**

```bash
bd create --title="Bug: description" --type=bug
bd create --title="Feature: description" --type=feature
```

Don't fix and forget - always track issues in beads.

## Project Structure

- `src/mycelium/` - Main source code
- `solver.py` - Main solver
- `step_signatures.py` - Step-level signature database
- `prompt_templates.py` - Centralized prompt registry

## Workflow

1. Check `bd ready` for available issues
2. `bd update <id> --status=in_progress` to claim work
3. Make changes
4. `bd close <id> --reason="..."` when done
5. `bd sync` to sync changes

See `AGENTS.md` for detailed guidance.
