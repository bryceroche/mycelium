# Claude Code Instructions

## Quick Start

```bash
bd prime        # Load context from beads
bd ready        # See available work
```

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
