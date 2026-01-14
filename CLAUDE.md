# Claude Code Instructions

Bryce is the user.  He likes to make edits to this file.  Please do not overwrite his edits.  
This file is critically important to the project.  Please do not overwrite it.
We want to treate this file as our source of truth.  Every bug fix, optimization or feature should be implemented with this file in mind.
Please always keep this file in context window.

With fresh DB:
 - Start with easy problems (GSM8K or MATH L1-L2)
 - Need some successes to learn from - failures alone don't teach what works

Key Questions
how do clusters form? Emerge naturally from umbrella promotions
How do we route? MCTS, but learned from embeddings
Cluster Centroid - Average of all descendant leaf embeddings 
how does learning work at each level? Learn "these operation types cluster together"

when to create a new root -- only one root.  Every problem goes through the first root signature.
how deep should the hierarchy go?  unbounded UMBRELLA_MAX_DEPTH 
should routing ever skip leves.  No it should not
how do hints work in hierarchical world?  surface clusters as operational types available.

What's the root's initial state? First signature IS the root
Threshold per level? Same threshold at all levels
When does umbrella promotion happen? on failure
How to migrate existing signatures?  start fresh

The Meta-Insight
  The same pattern applies at every level:
  - Problem decomposition: Break into sub-problems
  - Signature organization: Break into sub-clusters
  - Routing: Decompose the search space
  It's turtles all the way down. 🐢

slow decay.  Calculate signature use count /  ((cache) count of total num problems)   SLOW decay not a fast

## Core Principle: Failures Are Valuable Data Points
**Let signatures fail.** This is how the system learns.
- Record every failure—it feeds the refinement loop
- Do not fallback to LLM reasoning
- Failed signatures get decomposed
- Success/failure stats drive routing decisions

The goal is NOT 100% accuracy on every run. The goal is collecting data that makes the system smarter over time. A failed DSL provides valuable signal.

## Learning Mechanisms
Centroid Averaging
Cluster Centroid - Average of all descendant leaf embeddings 
More matches → more stable centroid → high-traffic signatures become semantic attractors.

### Credit Propagation
When a problem is solved correctly, success credit propagates up the signature DAG to parent umbrellas with decay:
- Direct signatures get +1 success
- Parent umbrellas get `decay^depth` credit (default 0.5 per level)
- Max propagation depth is configurable (default 3 levels)

This lets umbrella signatures accumulate credit from their children's successes, improving routing decisions.


## Semantic Embedding First
**Always prefer embedding similarity over keyword matching.**


## How to use Beads

```bash
bd prime        # Load context from beads
bd ready        # See available work
```

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
