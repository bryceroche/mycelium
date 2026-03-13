# Breathing: Expand-Collapse Iteration
# (addition to Mycelium v7 doc)

---

## The Breathing Model

The pipeline breathes. Each breath is an expand-collapse cycle.

```
EXPAND:   generate possibilities (narrate, translate, propose)
COLLAPSE: SymPy verifies, keep what works, discard what doesn't
EXPAND:   regenerate failures with new context from successes
COLLAPSE: SymPy verifies again
...until convergence or budget exhausted
```

This mirrors the teacher's attention heartbeat:
    reading (expand) → computing (collapse) → reading → computing

The pipeline inhales (generate), exhales (verify), and each breath
has better context than the last.

---

## How Breathing Works

```python
def solve_with_breathing(problem, scaffold, max_breaths):
    results = [None] * len(scaffold)
    narrations = [None] * len(scaffold)
    
    for breath in range(max_breaths):
        for i, step_type in enumerate(scaffold):
            if results[i] is not None:
                continue  # already collapsed — skip
            
            # EXPAND: narrate with context from collapsed steps
            context = [(narrations[j], results[j]) 
                       for j in range(i) if results[j] is not None]
            narrations[i] = narrator(problem, step_type, context)
            
            # EXPAND: translate narration → expression
            telegram = translator(narrations[i], available_values(results))
            
            # COLLAPSE: SymPy executes — incorruptible
            exec_result = oracle.execute(telegram, results)
            if exec_result.success:
                results[i] = exec_result.result  # collapsed, locked in
        
        # All collapsed? Done.
        if all(r is not None for r in results):
            return results[-1]
    
    return None  # budget exhausted
```

### Key Properties

```
1. Each breath has MORE CONTEXT than the last
   Breath 1: step 3 narrator doesn't know step 1-2 results
   Breath 2: step 3 narrator sees step 1-2 concrete values
   → Better context → better narration → better translation

2. Each breath is CHEAPER than the last
   Breath 1: processes all N steps
   Breath 2: processes only failures (often 1-2 steps)
   Breath 3: processes remaining stubborn failures
   → Work concentrates on hard steps automatically

3. Collapsed steps are LOCKED
   Once SymPy verifies a step, it never re-runs
   Results are concrete values, not model outputs
   → No error accumulation from successful steps

4. SMOOTH distributes naturally
   Easy steps collapse in breath 1 (low difficulty)
   Hard steps get multiple attempts (more breathing room)
   Total compute adapts to per-step difficulty
```

---

## C1-B Sets the Breath Budget

C1-B predicts belief propagation depth — "how many iteration cycles
does this problem need to converge." This IS the breath budget.

```
C1-B bp_depth → breathing budget:

    bp_depth ≈ 1.0 → 1 breath   (easy, linear chain, most collapse first try)
    bp_depth ≈ 2.0 → 2 breaths  (medium, some steps need context from others)
    bp_depth ≈ 3.0 → 3 breaths  (hard, iterative refinement)
    bp_depth > 3.0 → 3 breaths + MCTS scaffold search (very hard)
```

C1-B also predicts co-transition statistics — P(step_type_i | step_type_{i-1}).
Low-probability transitions mark subproblem boundaries.
Breathing naturally concentrates at boundaries because cross-phase
steps are hardest to get right without full context.

```
Co-transitions reveal subproblems:

    GIVEN → GIVEN → COMPUTE → EVAL     ← all high probability, one phase
                              ↓
    EVAL → GIVEN                        ← LOW probability = phase boundary
                              ↓
    GIVEN → SOLVE → SUBS → EVAL        ← second phase, new subproblem
```

Steps right after a phase boundary are the most likely to fail in
breath 1 (no context from previous phase yet) and succeed in
breath 2 (previous phase results now available).

---

## Escalation: Breathing → Perturbation → MCTS

Three levels of iteration, escalating cost:

```
Level 1: BREATHING (same scaffold, retry failed steps)
    Cost: ~1 narrator + translator call per failed step per breath
    When: default for every problem
    Budget: C1-B bp_depth breaths

Level 2: PERTURBATION (modify scaffold, re-breathe)
    Cost: full breathing cycle on modified scaffold
    When: all breaths exhausted, some steps still fail
    Actions:
        - Swap verb: EXPAND → SIMPLIFY
        - Split step: one step → two
        - Merge steps: two → one
    Budget: 3-5 perturbations

Level 3: MCTS (search scaffold space)
    Cost: multiple perturbation + breathing cycles
    When: perturbation didn't help
    Actions: structured scaffold mutations via UCB tree search
    Budget: problem-level compute budget from C1-B
    
    Each MCTS node = one scaffold variant
    Each rollout = breathing cycle on that scaffold
    Reward = negative energy of best partial solution
```

```
Easy problem (bp_depth=1):
    Breath 1 → all steps collapse → answer ✓
    Total: 1 narrator pass + 1 translator pass + 1 oracle pass

Medium problem (bp_depth=2):
    Breath 1 → steps 1,2,4 collapse, step 3 fails
    Breath 2 → step 3 gets context from 1,2 → collapses → answer ✓
    Total: 1.25 narrator passes + 1.25 translator passes + 1.25 oracle passes

Hard problem (bp_depth=3+):
    Breath 1-3 → step 5 keeps failing
    Perturbation → swap step 5 verb EXPAND→SIMPLIFY
    Breath 1 with new scaffold → step 5 collapses → answer ✓
    Total: ~4 passes + scaffold mutation

Very hard problem:
    MCTS explores 3 scaffold variants × 2 breaths each
    Two variants produce same answer → high confidence
    Total: ~8-12 passes (still cheaper than 70B inference)
```

---

## Energy Landscape + Breathing

The energy landscape guides breathing decisions:

```
After each breath:
    Compute per-step energy for uncollapsed steps
    
    High energy = wrong basin → re-narrate with new context
    Medium energy = right basin, imprecise → ODE refine then retry
    Low energy but execution fails → notation issue → oracle retry
    
    Focus next breath on HIGHEST energy uncollapsed step first
    → Greedy: fix the worst step, its result may unblock others
```

The energy landscape also detects when breathing won't help:

```
If energy INCREASES across breaths → diverging, not converging
    → Stop breathing, escalate to perturbation
    
If energy is flat across breaths → stuck in local minimum
    → Perturbation needed to escape the basin

If energy drops each breath → converging
    → Keep breathing, don't escalate prematurely
```

---

## The Full Assembly Line with Breathing

```
Input: MATH problem

Stage 1: C1-A scaffold + C1-B breath budget
    → 5 steps: [SETUP, SETUP, COMPUTE, SUBS, EVAL]
    → budget: 2 breaths

Stage 2: Breathing loop
    Breath 1:
        Narrator (LoRA C): describe each step in natural language
        Translator (LoRA D): convert descriptions to expressions
        Oracle: execute each expression
        → Steps 1,2,3,5 collapse. Step 4 fails (wrong substitution target)
    
    Breath 2:
        Narrator: re-describe step 4 with steps 1-3 concrete results
        Translator: re-translate with actual values available
        Oracle: re-execute
        → Step 4 collapses. Full chain complete.

Stage 3: Answer verification
    Final result vs expected format
    Energy landscape: is total sequence energy low?
    
Output: answer
```

---

## Inference Cost Model

```
Per problem:
    C1-A inference:     ~50ms  (one pass, all problems batched)
    C1-B inference:     ~50ms  (one pass, all problems batched)
    Narrator per step:  ~30ms  (Qwen-0.5B, ~10 tokens out)
    Translator per step: ~30ms (Qwen-0.5B, ~10 tokens out)
    Oracle per step:    ~10ms  (SymPy execution)
    
Easy problem (5 steps, 1 breath):
    100ms + 5×(30+30+10) = 450ms total

Medium problem (5 steps, 2 breaths, 1 retry):
    100ms + 5×70 + 1×70 = 520ms total

Hard problem (7 steps, 3 breaths, 5 perturbations):
    100ms + 7×70×1.5 + 5×7×70 = ~3s total

All under 5 seconds. All on a single A10G.
```
