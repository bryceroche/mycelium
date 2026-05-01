# Handoff: Equal-Reward Decomposition Incentive

## One-Sentence Summary

Each intermediate target and the final answer are worth EQUAL reward (1/N where N = total targets), each consumed once, in any order — making decomposition the ONLY way to maximize reward and one-shot solving structurally capped at 1/N.

---

## The Problem: No Incentive to Decompose

Diagnostic 5 revealed the truth about L4.5's "100% accuracy":

```
Cycle 1: 100/100 = 100%  (solves step 1 perfectly)
Cycle 2:   6/100 =   6%  (garbage)
Cycle 3:   0/100 =   0%  (garbage)
Final:     0/100 =   0%  (never reaches final answer)

The model does ALL its work in cycle 1.
Cycles 2-3 are dead. The breathing loop never chains.
The "100% on L4.5" was 100% on STEP 1, not 100% final.
```

And the atom blend is a CONSTANT — every atom at 0.463 for every problem. The controller learned "turn everything on at max clamp." No selection. No adaptation. One universal blend.

Why? Because the current incentive structure REWARDS one-shot:

```
Current incentives:
  Cycle 1 matches target[0]: FULL reward (gen_loss × 1.0)
  Cycle 2 matches target[1]: FULL reward
  Cycle 3 matches final:     FULL reward
  
  But cycle 1 gets full reward just for step 1!
  No reason to continue. Cycles 2-3 are free to produce garbage.
  The rational strategy is: maximize cycle 1, ignore everything else.
```

---

## The Fix: Equal Reward, Consumed Once, Any Order

Each target is worth 1/N reward. N = total number of targets (intermediates + final). Each target can only be claimed ONCE. Order doesn't matter.

```
Problem: "Natalia sold 48 clips in April, half as many in May. Total?"
Targets: [48, 24, 72]  →  N=3  →  each worth 1/3

Strategy A — ONE-SHOT (current broken behavior):
  Cycle 1: "#### 48" → matches [48], consumed    → reward = 1/3
  Cycle 2: garbage   → matches nothing            → reward = 0
  Cycle 3: garbage   → matches nothing            → reward = 0
  Total: 1/3  ← CAPPED. Can never get more than 1/3 with one-shot.

Strategy B — FULL DECOMPOSITION:
  Cycle 1: "#### 48"  → matches [48], consumed   → reward = 1/3
  Cycle 2: "#### 24"  → matches [24], consumed   → reward = 1/3
  Cycle 3: "#### 72"  → matches [72], consumed   → reward = 1/3
  Total: 3/3 = FULL REWARD ← only achievable by decomposing!

Strategy C — PANAMA HAT (big bite, different order):
  Cycle 1: "#### 72"  → matches [72], consumed   → reward = 1/3
  Cycle 2: "#### 48"  → matches [48], consumed   → reward = 1/3
  Cycle 3: "#### 24"  → matches [24], consumed   → reward = 1/3
  Total: 3/3 = FULL REWARD ← any order works!

Strategy D — PARTIAL DECOMPOSITION:
  Cycle 1: "#### 48"  → matches [48], consumed   → reward = 1/3
  Cycle 2: "#### 48"  → ALREADY CONSUMED           → reward = 0 (copying!)
  Cycle 3: "#### 72"  → matches [72], consumed   → reward = 1/3
  Total: 2/3 ← better than one-shot, worse than full

Strategy E — SKIP TO FINAL:
  Cycle 1: "#### 72"  → matches [72], consumed   → reward = 1/3
  Cycle 2: garbage    → matches nothing            → reward = 0
  Cycle 3: garbage    → matches nothing            → reward = 0
  Total: 1/3 ← even getting the final answer first only earns 1/3!
```

---

## Why This Works

The incentive structure has elegant properties:

### 1. Decomposition is ALWAYS better than one-shot

```
One-shot:        1/N reward  (capped, no matter which target)
Full decompose:  N/N reward  (maximum, only way to achieve it)

The gap GROWS with problem complexity:
  2-step: one-shot = 0.5, decomposed = 1.0  (2x better)
  3-step: one-shot = 0.33, decomposed = 1.0 (3x better)
  5-step: one-shot = 0.2, decomposed = 1.0  (5x better)

Harder problems have STRONGER incentive to decompose.
```

### 2. Order doesn't matter

```
[48, 24, 72] in any order = same reward.
[72, 48, 24] = same reward.
[24, 72, 48] = same reward.

The model discovers its OWN decomposition order.
Panama hat (final answer first) is valid.
Standard order (step-by-step) is valid.
Any valid path is rewarded equally.
```

### 3. Copying is naturally penalized

```
Cycle 1: "#### 48" → consumed
Cycle 2: "#### 48" → ALREADY CONSUMED → zero reward
Cycle 3: "#### 48" → ALREADY CONSUMED → zero reward

Repeating the same answer earns NOTHING after the first claim.
The model must produce DIFFERENT answers to earn more reward.
```

### 4. Scales with problem complexity

```
1-target problem:  1 cycle needed → 1/1 = full reward
3-target problem:  3 cycles needed → 1/3 each → 3/3 full
5-target problem:  5 cycles needed → 1/5 each → 5/5 full

The number of productive cycles matches the problem's complexity.
The confidence head learns: "I've claimed 3/5, keep going."
```

### 5. Self-consistent wrong answers get partial credit

```
Cycle 2 gets wrong input from cycle 1 but computes correctly:
  Step 1: #### 120 (wrong, should be 240)
  Step 2: 120 / 3 = 40 #### 40 (correct MATH on wrong input)
  
  #### 40 doesn't match any target → no 1/N reward
  BUT self-consistent check passes → 0.1 gen_loss weight
  The model learns: "your division is correct, your input was wrong"
  
  The computation SKILL is rewarded even when the answer is wrong.
```

---

## Implementation

```python
def compute_cycle_reward(predicted, available_targets, consumed_targets,
                          num_total_targets, is_self_consistent):
    """
    Compute reward for one cycle's prediction.
    
    Args:
        predicted: extracted number from generation
        available_targets: targets not yet consumed
        consumed_targets: targets already claimed
        num_total_targets: N (for 1/N reward calculation)
        is_self_consistent: did the model's own equation check out?
    
    Returns:
        gen_weight: multiplier for this cycle's gen_loss
        new_available: updated available targets
        new_consumed: updated consumed targets
    """
    reward_per_target = 1.0 / num_total_targets
    
    if predicted is None:
        # No extraction — minimal learning signal
        return 0.05, available_targets, consumed_targets
    
    if predicted in consumed_targets:
        # Copying a consumed target — ZERO reward
        return 0.0, available_targets, consumed_targets
    
    # Check against available targets
    for i, target in enumerate(available_targets):
        if predicted == target:
            # MATCH! Claim this target.
            new_available = available_targets[:i] + available_targets[i+1:]
            new_consumed = consumed_targets + [target]
            return reward_per_target, new_available, new_consumed
    
    # No match — but is the computation self-consistent?
    if is_self_consistent:
        # Correct math on wrong input — reward the skill
        return 0.1, available_targets, consumed_targets
    else:
        # Wrong math — minimal signal
        return 0.05, available_targets, consumed_targets
```

### Training Loop

```python
def train_step(model, controller, atom_sets, problem, 
               num_cycles, tokenizer, kv_cache):
    """
    Training with equal-reward decomposition incentive.
    
    Each target worth 1/N. Consumed once. Any order.
    The ONLY way to maximize reward is to decompose.
    """
    all_targets = problem['all_targets']  # [48, 24, 72]
    num_targets = len(all_targets)
    
    available_targets = list(all_targets)
    consumed_targets = []
    
    notebook = Notebook()
    total_loss = 0.0
    total_reward = 0.0
    prev_results = []
    
    # Cycle 0: comprehend + cache
    # ... (same as before) ...
    
    for cycle in range(num_cycles):
        # Run cycle (multi-pass with interleaved observation)
        gen_logits, predicted, confidence = think_one_cycle(
            model, controller, atom_sets, kv_cache,
            notebook, cycle, prev_results, tokenizer
        )
        
        # Compute generation loss
        gen_loss = weighted_generation_loss(
            gen_logits, problem['cycle_gen_targets'][cycle],
            tokenizer, eos_weight=5.0
        )
        
        # Compute reward (1/N per matched target)
        with torch.no_grad():
            text = tokenizer.decode(gen_logits.argmax(-1)[0])
            predicted = extract_answer(text)
            self_consistent = check_computation_correct(text)
        
        gen_weight, available_targets, consumed_targets = compute_cycle_reward(
            predicted, available_targets, consumed_targets,
            num_targets, self_consistent
        )
        
        # Weighted loss: reward × gen_loss
        total_loss += gen_weight * gen_loss
        total_reward += gen_weight
        
        # Text injection for next cycle
        prev_results.append(predicted if predicted else 0)
        
        # Confidence: have we claimed all targets?
        if len(consumed_targets) == num_targets:
            break  # all targets claimed — stop early
    
    # Smooth fading still applies (for autonomous decomposition later)
    # But now intermediate targets have EQUAL weight to final
    
    # Regularizers
    total_loss += 0.01 * isotropic_reg(raw_pages)
    total_loss += 0.1 * confidence_entropy_loss(notebook)
    
    return total_loss, total_reward
```

---

## How This Fixes the Constant-Blend Problem

```
BEFORE (cycle 1 gets full reward):
  Controller: "turn all atoms on at max → cycle 1 matches target[0] → full reward"
  No incentive to vary blend. No incentive to use cycles 2-3.
  Rational strategy: maximize cycle 1.

AFTER (each target worth 1/N):
  Controller: "turn all atoms on at max → cycle 1 matches target[0] → 1/3 reward"
  Controller: "I need 2 MORE correct answers to get full reward."
  Controller: "cycle 2 must produce a DIFFERENT correct answer."
  Controller: "I need to change my atom blend for cycle 2!"
  
  The controller HAS to vary its blends to claim different targets.
  The universal blend might claim target[0] at cycle 1.
  But to claim target[1] at cycle 2, it needs a DIFFERENT blend.
  
  Decomposition becomes the RATIONAL STRATEGY.
```

The diversity loss on scales becomes UNNECESSARY — the equal-reward structure naturally forces diverse blends because diverse targets require diverse computations.

```
Before: diversity loss says "be different" (artificial pressure)
After:  reward structure says "be different to earn more" (natural incentive)

The model WANTS to produce different answers.
Different answers require different atom blends.
The diversity emerges from the incentive, not from a loss term.
```

---

## Interaction with Existing Components

### Flexible Loss with Consumption

Already built! The consumption mechanism is the core of this system. Each target consumed once, any order. We just change the REWARD from fixed 1.0 to proportional 1/N.

### Confidence Head

The confidence head learns a NEW signal: "how many targets have I claimed?"

```
0/N claimed: confidence low → keep cycling
1/N claimed: confidence low → keep cycling  
2/N claimed: moderate → maybe keep going
N/N claimed: high → STOP (all targets claimed!)

The confidence head naturally learns to count consumed targets.
It stops when decomposition is complete, not after a fixed number of cycles.
```

### Smooth Fading

Still applies! As accuracy climbs:

```
Low accuracy:   prescribed targets active (learning decomposition)
High accuracy:  targets fade → model discovers own decomposition
                But the 1/N reward structure REMAINS
                The model still needs to claim N unique answers
                It just discovers its OWN intermediates instead of ours
```

### Multi-Pass Atoms

Each cycle's multi-pass processing aims to claim ONE target. The inner loop (atom passes) processes thoroughly. The outer loop (cycles) claims targets. Depth × length, with each cycle incentivized to claim something NEW.

### Self-Consistency Check

Unchanged. A cycle that computes correctly on wrong upstream input gets partial credit (0.1). This rewards computation SKILL even when the chain has errors. But only TARGET MATCHES earn 1/N reward.

---

## Monitoring

```
1. Reward per epoch:
   total_reward / num_problems
   Should climb from 1/N (one-shot) toward N/N (full decomposition)
   
   Epoch 1:  avg reward = 0.33 (one-shot, claims 1/3 targets)
   Epoch 5:  avg reward = 0.50 (partial decomposition)
   Epoch 10: avg reward = 0.80 (most targets claimed)
   
   This is THE metric for decomposition quality.

2. Targets claimed per problem:
   avg targets_consumed / total_targets
   Should climb toward 1.0 (all targets claimed)

3. Per-cycle accuracy:
   cycle_1: should stay high (already works)
   cycle_2: should CLIMB (incentivized to claim target[1])
   cycle_3: should CLIMB (incentivized to claim target[2])
   
   The even distribution across cycles = genuine decomposition.

4. Blend diversity:
   Are atom blends different per cycle?
   Should emerge NATURALLY (different targets need different blends)
   No diversity loss needed — the reward structure forces it.

5. Decomposition order:
   Does the model use prescribed order [48, 24, 72]?
   Or its own order [72, 48, 24]?
   Both are valid. Observing the chosen order tells us what the model finds natural.

6. Confidence head calibration:
   Does it learn to stop when all targets are claimed?
   Plot: confidence vs fraction of targets consumed
   Should be monotonically increasing.
```

---

## Expected Trajectory

```
Phase 1 (epochs 1-5): learning to decompose
  Cycle 1: claims first target (already can)  → 1/N reward
  Cycle 2: starts claiming second target      → 2/N reward sometimes
  Total reward: 0.33 → 0.50
  The model discovers: "different cycles need different answers"

Phase 2 (epochs 5-15): decomposition improves
  Most cycles claim their targets
  Total reward: 0.50 → 0.80
  Atom blends diversify per cycle (emergent, not forced)
  
Phase 3 (epochs 15+): near-complete decomposition
  Almost all targets claimed per problem
  Total reward: 0.80 → 0.95
  The model has learned to chain reasoning

Phase 4 (smooth fading): autonomous decomposition
  Prescribed targets fade
  The model discovers its OWN intermediates
  Still earns reward by matching ANY valid path to the final answer
```

---

## What NOT to Do

```
- Do NOT give full reward for any single target match.
  Each target is worth 1/N. This is the CORE incentive.
  Full reward for one target = no incentive to decompose.

- Do NOT prescribe target ordering.
  Any order earns the same reward.
  The model discovers its natural decomposition order.

- Do NOT remove the consumption mechanism.
  Consumption prevents claiming the same target twice.
  Without it, the model would repeat the easiest answer every cycle.

- Do NOT set wrong-answer weight to 0.0.
  The model needs SOME gradient from wrong cycles to learn.
  0.05 for wrong math. 0.1 for self-consistent. 0.0 for copying only.

- Do NOT add diversity loss.
  The reward structure naturally forces diverse blends.
  Adding diversity loss on top is redundant and might conflict.

- Do NOT use this with single-step training.
  Single-step has N=1 targets → 1/1 = full reward always.
  This incentive only matters for multi-step problems.
  Graduate to multi-step to use this incentive.
```

---

## The Elegant Picture

```
The ONLY way to maximize reward is to decompose.

One-shot:  1/N reward.  Capped. Insufficient.
Decompose: N/N reward.  Maximum. The only path to full reward.

The model WANTS to produce different answers per cycle.
Different answers require different atom blends.
Different blends emerge from the incentive, not from loss pressure.

No prescribed order. No prescribed roles.
Just: "claim as many unique targets as you can."
The model discovers that decomposition is the optimal strategy.

Incentive-driven decomposition.
Not loss-driven. Not architecture-driven. Not prescribed.
The model CHOOSES to decompose because it's the rational strategy.
```
