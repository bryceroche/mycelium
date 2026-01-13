# Parent-to-Child Routing Flow Audit

## Overview

The routing system enables semantic umbrellas (generic step types) to route to more specific child signatures. This supports multi-level routing where children can also be umbrellas.

## Database State

- **Total semantic umbrellas**: 80
- **Multi-level chains** (child is also umbrella): 5
  - `count_combinations` → `count_permutations_with_replacement`
  - `count_combinations` → `count_combinations_with_constraints`
  - `synthesize_results` → `Simplify_Expression`
  - `count_items` → `count_total_items`
  - `factor_expression` → `factor_polynomial`

## Flow Diagram

```
Step Execution
     │
     ▼
┌─────────────────────────────────────────┐
│ _execute_step_with_signature()          │
│ (solver.py:997-1468)                    │
│                                         │
│ 1. Embed step                           │
│ 2. Find/create signature                │
│ 3. Check if semantic umbrella           │
└────────────────┬────────────────────────┘
                 │
     ┌───────────┴───────────┐
     │                       │
     ▼                       ▼
┌────────────┐       ┌──────────────────┐
│ NOT        │       │ IS umbrella      │
│ umbrella   │       │ with children    │
└─────┬──────┘       └────────┬─────────┘
      │                       │
      │                       ▼
      │              ┌────────────────────────────────┐
      │              │ _route_to_child_signature()    │
      │              │ (solver.py:1603-1888)          │
      │              │                                │
      │              │ 1. Check max depth (3)         │
      │              │ 2. Parse child specs           │
      │              │ 3. LLM selects child           │
      │              │ 4. Embed confidence calc       │
      │              │ 5. Execute child DSL           │
      │              └────────────┬───────────────────┘
      │                           │
      │              ┌────────────┴────────────┐
      │              │                         │
      │              ▼                         ▼
      │      ┌──────────────┐         ┌───────────────┐
      │      │ DSL SUCCESS  │         │ DSL FAILED    │
      │      └──────┬───────┘         └───────┬───────┘
      │             │                         │
      │             │                         ▼
      │             │                ┌────────────────────┐
      │             │                │ Child is umbrella? │
      │             │                └────────┬───────────┘
      │             │                         │
      │             │              ┌──────────┴──────────┐
      │             │              │                     │
      │             │              ▼                     ▼
      │             │      ┌──────────────┐     ┌──────────────┐
      │             │      │ YES: Recurse │     │ NO: Return   │
      │             │      │ deeper       │     │ None         │
      │             │      └──────┬───────┘     └──────┬───────┘
      │             │             │                    │
      │             │             │ (back to top)      │
      │             │             │                    │
      │             ▼             │                    │
      │      ┌─────────────────┐  │                    │
      │      │ Return          │  │                    │
      │      │ StepResult      │◀─┘                    │
      │      │ (routing)       │                       │
      │      └─────────────────┘                       │
      │                                                │
      │                                                │
      ▼                                                ▼
┌───────────────────────────────────────────────────────────┐
│ Continue to DSL/Formula/LLM execution                     │
│ (solver.py:1180-1424)                                     │
└───────────────────────────────────────────────────────────┘
```

## Code Path Details

### 1. Entry Point: `_execute_step_with_signature` (line 1053-1072)

```python
# Check if this is a semantic umbrella that should route to child
routing_attempted = False
routing_path: list[str] = []
if signature.is_semantic_umbrella and signature.child_signatures:
    routing_attempted = True
    routing_path = [signature.step_type]  # At least the parent was tried
    routed_child = await self._route_to_child_signature(
        parent=signature,
        step=step,
        context=context,
        step_descriptions=step_descriptions,
        problem=problem,
        embedding=embedding,
        depth=decomposition_depth,  # ⚠️ Uses decomposition_depth, not routing_depth
    )
    if routed_child is not None:
        return routed_child
    # Routing failed - log and continue to other methods
```

### 2. Child Selection: `_route_to_child_signature` (line 1603-1888)

**Initialization (line 1640-1659)**
```python
# Initialize routing path with parent
if routing_path is None:
    routing_path = [parent.step_type]
else:
    routing_path = routing_path + [parent.step_type]

# Check max depth to prevent infinite recursion
if depth >= RECURSIVE_MAX_DEPTH:
    return None

# Parse child specs
child_specs = json.loads(parent.child_signatures)
```

**LLM Selection + Embedding Confidence (line 1662-1739)**
- Builds prompt with child options
- Computes embedding similarity for each child
- LLM selects pattern number
- Falls back to embedding ranking if LLM unparseable

**Child DSL Execution (line 1762-1861)**
```python
if child_sig.dsl_script:
    # Build context, check lift/semantic avoidance
    dsl_result, dsl_success, dsl_confidence = await execute_dsl_with_llm_matching(...)

    if dsl_success:
        # Return StepResult with full routing path
        return StepResult(
            execution_method="routing",
            routing_path=routing_path + [child_sig.step_type],
            ...
        )
```

### 3. Multi-Level Recursion (line 1864-1880)

```python
else:
    # DSL failed - check if child has its own children to route deeper
    current_path = routing_path + [child_sig.step_type]
    if child_sig.is_semantic_umbrella and child_sig.child_signatures:
        logger.info("[routing] DSL failed path=%s conf=%.2f, routing deeper",
                   " -> ".join(current_path), dsl_confidence)
        # Recurse to grandchildren
        deeper_result = await self._route_to_child_signature(
            parent=child_sig,
            step=step,
            context=context,
            step_descriptions=step_descriptions,
            problem=problem,
            embedding=embedding,
            depth=depth + 1,
            routing_path=current_path,
        )
        if deeper_result is not None:
            return deeper_result
```

## Issues Identified

### Issue 1: Depth Parameter Confusion (MEDIUM)

**Location**: `_execute_step_with_signature` line 1064

**Problem**: `decomposition_depth` is passed as `depth` to routing, but decomposition and routing are separate concerns.

```python
routed_child = await self._route_to_child_signature(
    ...
    depth=decomposition_depth,  # Should this be routing_depth=0?
)
```

**Impact**: If a step is decomposed to depth 2, it starts routing at depth 2, limiting routing to only 1 more level (if RECURSIVE_MAX_DEPTH=3).

**Recommendation**: Use separate counters for decomposition and routing depths.

### Issue 2: Failed Routing Path Not Captured (LOW)

**Location**: `_execute_step_with_signature` line 1055-1058

**Problem**: When routing fails at all levels and falls back to LLM, the final StepResult only has `routing_path = [parent.step_type]`, not the full failed path.

```python
routing_attempted = True
routing_path = [signature.step_type]  # Only parent, not full failed path
```

**Impact**: Debugging difficult - can't see which child paths were attempted before fallback.

**Recommendation**: Have `_route_to_child_signature` return both result and attempted_path, or store in thread-local for debugging.

### Issue 3: Child Not Found Handling (OK)

**Location**: Line 1743-1754

**Status**: ✅ Handled correctly with detailed logging

```python
if child_sig is None:
    logger.error(
        "[routing] ORPHAN: parent_id=%d parent_type='%s' -> orphan_child_id=%d ...",
        ...
    )
    return None
```

### Issue 4: Embedding Cache Growth (LOW)

**Location**: Line 1697-1701

**Problem**: `_child_embedding_cache` grows unboundedly over solver lifetime.

```python
if child_text in self._child_embedding_cache:
    child_emb = self._child_embedding_cache[child_text]
else:
    child_emb = self.embedder.embed(child_text)
    self._child_embedding_cache[child_text] = child_emb  # Never cleared
```

**Impact**: Memory growth in long-running processes.

**Recommendation**: Add LRU eviction or clear on new problem.

## Multi-Level Flow Example

For `count_combinations` → `count_permutations_with_replacement`:

```
Step: "Count ways to arrange 5 items with replacement"
         │
         ▼
┌─────────────────────────────────┐
│ Match: count_combinations (id=3)│
│ is_semantic_umbrella = True     │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ _route_to_child_signature(parent=3, depth=0)│
│ LLM selects: count_permutations_with_replacement│
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ Execute DSL for child id=71                 │
│ DSL: n ** r                                 │
└────────────────┬────────────────────────────┘
                 │
     ┌───────────┴───────────┐
     │                       │
     ▼                       ▼
SUCCESS                   FAILED
  │                          │
  │                          ▼
  │                   ┌──────────────────────────┐
  │                   │ Child id=71 is umbrella? │
  │                   │ → NO (leaf node)         │
  │                   │ → Return None            │
  │                   └──────────────────────────┘
  │                          │
  │                          ▼
  │                   Fall back to LLM
  │
  ▼
Return StepResult(
  routing_path=["count_combinations", "count_permutations_with_replacement"],
  execution_method="routing",
  dsl_executed=True
)
```

## Fixed Issues

### Complete LLM Removal (FIXED)

**Problem**: LLM was used as fallback when DSL/routing/decomposition failed.

**Fix**: Removed ALL LLM fallback. System now uses only DSL execution and decomposition.

**Changes Made**:

1. **Routing tries ALL children** - not just LLM's first choice
2. **Umbrella failure returns error** - not LLM fallback
3. **Step execution section rewritten** - removed ~120 lines of LLM code
4. **DSL failure triggers decomposition** - not LLM

**New Architecture (NO LLM for step solving)**:
```
Step Execution
     │
     ▼
┌─────────────────────────────────────┐
│ Is Semantic Umbrella?               │
└──────────────┬──────────────────────┘
               │
     ┌─────────┴─────────┐
     │                   │
    YES                  NO
     │                   │
     ▼                   ▼
┌────────────────┐  ┌────────────────┐
│ Try ALL        │  │ Try DSL        │
│ Children       │  │ Try Formula    │
└───────┬────────┘  └───────┬────────┘
        │                   │
        │ All failed        │ Failed
        ▼                   ▼
┌────────────────┐  ┌────────────────┐
│ Decompose      │  │ Decompose      │
└───────┬────────┘  └───────┬────────┘
        │                   │
        │ Failed            │ Failed
        ▼                   ▼
┌────────────────────────────────────┐
│ Return ERROR (never LLM)           │
│ "ERROR: DSL failed and             │
│  decomposition failed"             │
└────────────────────────────────────┘
```

**Benefits**:
- 100% deterministic execution
- Failures are explicit (not hidden LLM guesses)
- DSL coverage gaps become visible
- Easier debugging and improvement

## Remaining Recommendations

1. **Separate routing depth from decomposition depth** to allow full routing hierarchy independent of decomposition level.

2. **Track failed routing paths** for better debugging - return `(result, attempted_paths)` or use structured logging.

3. **Add routing metrics** to SolverResult:
   - `max_routing_depth_reached`
   - `routing_fallback_count`
   - `full_routing_paths_attempted`

4. **Add cache eviction** for `_child_embedding_cache` to prevent memory leaks.

5. **Consider parallel child evaluation** - try top-2 children by embedding confidence simultaneously, return first success.
