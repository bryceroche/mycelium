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
