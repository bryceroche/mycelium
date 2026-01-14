# Claude Code Instructions

## Quick Start

```bash
bd prime        # Load context from beads
bd ready        # See available work
```

Bryce is the user.  He likes to make edits to this file.  Please do not overwrite his edits.  


## Design Philosophy: Build for Scale
**The system must grow on its own.** Avoid patterns that require manual maintenance or LLM fallback:
We want the system to be self-sufficient and scalable.  We do not want the system to rely on patches from Claude or rely on LLM Fallback.

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

## Core Principle: Failures Are Valuable Data Points

**Let signatures fail.** This is how the system learns.

- Don't mask DSL failures with LLM execution
- Record every failure—it feeds the refinement loop
- Failed signatures get demoted, decomposed, or fixed
- Success/failure stats drive routing decisions

The goal is NOT 100% accuracy on every run. The goal is collecting data that makes the system smarter over time. A failed DSL today becomes training data for a better DSL tomorrow.

## Learning Mechanisms
Centroid Averaging
Centroids are running averages: new_centroid = (old_sum + embedding) / new_count
More matches → more stable centroid → high-traffic signatures become semantic attractors.

### Credit Propagation
When a problem is solved correctly, success credit propagates up the signature DAG to parent umbrellas with decay:
- Direct signatures get +1 success
- Parent umbrellas get `decay^depth` credit (default 0.5 per level)
- Max propagation depth is configurable (default 3 levels)

This lets umbrella signatures accumulate credit from their children's successes, improving routing decisions.


## Embedding Model Limitations & Signature-Guided Learning

**Known limitation:** MathBERT embeddings cluster by vocabulary, not operational semantics. Generic anchors like "permutations arrangements" don't match well to domain phrases like "choices for the gold medal" (similarity ~0.35).

**Solution: Let successful signatures become the anchors.**

### The Cold-Start Challenge
- Generic operation anchors have weak discrimination
- Novel problem phrasings don't match any anchor above threshold
- Result: `decompose` fallback → fail → record failure data

### Signature-Guided Learning (How the system compounds)
1. **High threshold rejects bad matches** (`DSL_OPERATION_INFERENCE_THRESHOLD = 0.6`)
   - Uncertain matches → `decompose` type → fail cleanly
   - No wrong DSLs polluting the system

2. **Failures are learning data**
   - Failed signatures get recorded with their embeddings
   - Stats accumulate (uses, successes, failure patterns)
   - System learns what DOESN'T work

3. **Successful signatures become domain anchors**
   - When a permutation problem finally succeeds, its signature has embedding for "choices for medal"
   - Future "medal selection" problems match THIS signature, not generic anchors
   - The solved problem's vocabulary becomes the new anchor

4. **Bi-directional NL refinement**
   - **Signatures → Planner**: `clarifying_questions`, `param_descriptions`
   - **Planner → Signatures**: `extracted_values`, step descriptions
   - Each successful solve teaches both sides what to look for

### Why This Works
- First solve of a problem type is hard (generic anchors don't match)
- Second solve is easier (matches the first solve's signature)
- 100th solve is trivial (strong centroid from many examples)
- **Each solved problem makes similar problems easier**

Key insight: Don't enrich generic anchors with domain vocabulary. Let the system LEARN domain vocabulary from successful solves. This scales automatically.


### Depth-Based Branching
We want the system to prefer routing to existing signatures first and then only create new ones when necessary.
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

## Semantic Embedding First

**Always prefer embedding similarity over keyword matching.**

Bad (keyword-based):
```python
if "sum" in step_type or "add" in step_type:
    return "+"
```

Good (embedding-based):
```python
# Embed the step description, compare to operation anchors
sim_add = cosine_similarity(step_emb, addition_anchor)
sim_sub = cosine_similarity(step_emb, subtraction_anchor)
return max(operations, key=lambda op: similarities[op])
```

Why this matters:
- Keywords don't generalize ("calculate total" ≠ "compute sum" but same meaning)
- Embeddings capture semantic equivalence
- New patterns work without code changes
- Scales with the embedding model's knowledge

Use rich anchor texts that describe the operation semantically:
```python
ADDITION_ANCHOR = "combining quantities, finding total, summing values together"
SUBTRACTION_ANCHOR = "finding difference, taking away, how much more or less"
```

Key Questions
how do clusters form? Emerge naturally from umbrella promotions
How do we route? Like a decision tree, but learned from embeddings
Cluster Centroid - Average of all descendant leaf embeddings 
how does learning work at each level? Learn "these operation types cluster together"

slow decay with if unused cache total num problems and compare to each sig to see it's % use.  this should be a SLOW decay not a fast one.


when to create a new root -- only one root.  Every problem goes through the first root signature.
how deep should the hierarchy go?  unbounded UMBRELLA_MAX_DEPTH 
should routing ever skip leves.  No it should not
how do hints work in hierarchical world?  surface clusters as operational types available.


The Meta-Insight
  The same pattern applies at every level:
  - Problem decomposition: Break into sub-problems
  - Signature organization: Break into sub-clusters
  - Routing: Decompose the search space
  It's turtles all the way down. 🐢


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
