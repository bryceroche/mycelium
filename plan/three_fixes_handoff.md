# Handoff: Three Fixes for the 17.8% GSM8K Ceiling

## Current Result

```
GSM8K: 2.2% baseline → 17.8% (8.1x improvement)
Problem: plateaued at epoch 6, overfitting by epoch 7
Three root causes identified:
  1. L4→L5 is a cliff (100% → 17.8%), not a ramp
  2. Overfitting from seeing same problems repeatedly
  3. Earlier cycles get weaker gradient than later cycles
```

---

## Fix 1: Gradient Scaling Per Cycle

### The Problem

Cycle 1 is farthest from the answer loss. Its gradient flows backward through cycles 2 and 3's compression steps, attenuating at each step. Cycle 1's LoRA and perceiver output get weak gradient — they can't learn as effectively as cycle 3's.

### The Fix

Scale each page's gradient inversely to its distance from the loss. Forward pass unchanged. Backward pass amplified for earlier cycles.

```python
def scale_gradient(tensor, scale):
    """Scale gradient during backward without affecting forward."""
    # Forward: returns tensor unchanged
    # Backward: gradient multiplied by scale
    return tensor + (scale - 1.0) * tensor.detach()

# In the thinking loop:
for pass_num in range(num_passes):
    page, strategy = perceiver(hidden_states, pass_num)
    
    # Earlier cycles get amplified gradient
    # cycle 0 = 3x, cycle 1 = 2x, cycle 2 = 1x
    grad_scale = float(num_passes - pass_num)
    page = scale_gradient(page, grad_scale)
    
    state_pages.append(page)
```

### Why Not Deep Supervision

Deep supervision asks each cycle to predict the final answer. But each cycle should focus on its CHUNK — parse, compute, or verify — not try to solve the whole problem. Deep supervision fights the tight bottleneck by incentivizing each cycle to cram the full answer into 64 floats.

Gradient scaling is better: it says "your chunk was useful for the final answer" (amplified gradient from the end), not "predict the final answer from your chunk" (deep supervision).

### Implementation

One line per cycle in the thinking loop. No architectural change. No extra compute. No extra memory.

---

## Fix 2: Fresh Data Every Epoch (Anti-Overfitting)

### The Problem

Each thinking cycle does a small chunk of work. "56 - 2 = 54" is a tiny fact. 20K tiny facts are easily memorized. By epoch 3, the model has seen each problem 3 times and starts memorizing. Answer loss drops to 0.0000 while eval accuracy collapses.

This is structural: the CHUNKS are the right size for thinking but too small for learning. More epochs can't help because the model memorizes faster than it generalizes.

### The Fix

Generate FRESH problems every epoch. Never see the same problem twice. The model can't memorize what it hasn't seen before.

For procedural levels (L0-L4): generate new problems at the start of each epoch.

```python
def generate_epoch_data(level, num_problems, epoch_seed):
    """Generate fresh problems for this epoch. Different seed = different problems."""
    rng = random.Random(epoch_seed)
    
    if level == 'L3':
        return generate_named_quantity_problems(num_problems, rng)
    elif level == 'L4':
        return generate_two_step_word_problems(num_problems, rng)
    elif level == 'L4.5':
        return generate_three_step_word_problems(num_problems, rng)
    # etc.

# In training loop:
for epoch in range(max_epochs):
    train_data = generate_epoch_data(level, num_problems=20000, epoch_seed=epoch * 1000)
    train_one_epoch(model, train_data)
```

For GSM8K (L5): can't generate procedurally — real problems from the dataset. Instead, augment:

```python
def augment_gsm8k_problem(problem, answer, rng):
    """Same structure, different numbers/names/objects."""
    # Swap numbers: multiply all numbers by a random factor
    factor = rng.choice([0.5, 1.5, 2, 3, 0.25])
    # Swap names: "Jamie" → "Sarah", "cookies" → "apples"
    # Swap context: "store" → "bakery", "Monday" → "Wednesday"
    # Recompute answer with new numbers
    return augmented_problem, new_answer
```

With augmentation, 7,473 GSM8K problems become effectively infinite. Each epoch sees the same STRUCTURES but different NUMBERS. The model must learn the pattern, not memorize specific facts.

### Key Principle

The model should never see the same (problem, answer) pair twice. The STRUCTURE repeats (that's learning). The SPECIFICS don't (that prevents memorizing).

---

## Fix 3: Fill the L4→L5 Gap

### The Problem

```
L4:  answers [1, 200], 2-step, simple operations    → 100%
L5:  answers [1, 100000+], 3-5 step, complex ops    → 17.8%

That's a cliff, not a ramp.
```

### The Fix

Add intermediate levels between L4 and L5:

```
L4:    2-step word problems, answers [1, 200]              → 100%
L4.5:  2-step word problems, answers [1, 2000]             → ???
       Same structure as L4 but larger numbers
       Tests: can the model handle bigger arithmetic?

L4.7:  3-step word problems, answers [1, 5000]             → ???
       More steps, larger numbers
       Tests: can the model chain 3 word-problem steps?

L4.9:  GSM8K easy subset (2-3 step, straightforward)       → ???
       Real GSM8K problems filtered for simplicity
       Tests: can the model handle real GSM8K formatting?

L5:    Full GSM8K                                           → 17.8%
```

Each level adds ONE new challenge:
- L4.5: bigger numbers (same structure)
- L4.7: more steps (slightly more complex)
- L4.9: real GSM8K format (bridge to actual data)

### Generating L4.5 and L4.7

Same procedural generators as L4, with wider number ranges and optional third step:

```python
def generate_L45_problem(rng):
    """2-step, answers [1, 2000]"""
    # Same templates as L4 but numbers drawn from [10, 500]
    a = rng.randint(10, 500)
    b = rng.randint(1, a)
    # "A store has {a} cookies. They sell {b}..."
    
def generate_L47_problem(rng):
    """3-step, answers [1, 5000]"""
    # Three operations chained
    a = rng.randint(10, 500)
    b = rng.randint(1, 100)
    c = rng.randint(1, 100)
    # "A store has {a} cookies. They sell {b} on Monday,
    #  buy {c} on Tuesday, and give away half..."
```

### L4.9: GSM8K Easy Subset

Filter GSM8K training set for the simplest problems:

```python
def is_easy_gsm8k(problem, solution):
    steps = count_steps(solution)
    max_number = max_number_in_solution(solution)
    return steps <= 3 and max_number < 1000
```

This gives the model real GSM8K formatting and language before facing the full difficulty.

### Curriculum

```
L4   → L4.5 → L4.7 → L4.9 → L5
Each: warm start from previous, 20K fresh problems per epoch
      CoT targets, dual LoRA, early stopping patience=3
```

---

## Implementation Order

```
1. Add gradient scaling to thinking loop (one line per cycle)
2. Add fresh data generation per epoch (modify training loop)
3. Generate L4.5 problems and train (quick test of both fixes)
4. If L4.5 improves over L4's warm-start baseline → continue to L4.7, L4.9
5. Retrain GSM8K with all three fixes
```

### Changes to Training Script

```python
def train(model, level, max_epochs=12, patience=3):
    best_acc = 0
    patience_counter = 0
    
    for epoch in range(max_epochs):
        # FIX 2: Fresh data every epoch
        train_data = generate_epoch_data(level, num_problems=20000, epoch_seed=epoch)
        
        for batch in train_data:
            optimizer.zero_grad()
            state_pages = []
            
            for pass_num in range(num_passes):
                page, strategy = model.think_one_pass(batch, state_pages, pass_num)
                
                # FIX 1: Gradient scaling
                grad_scale = float(num_passes - pass_num)
                page = scale_gradient(page, grad_scale)
                
                state_pages.append(page)
            
            # Generation loss + contrastive loss
            loss = compute_loss(model, state_pages, batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Eval
        acc = evaluate(model, eval_data)
        if acc > best_acc:
            best_acc = acc
            save_checkpoint(model)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_acc
```

---

## What to Monitor

```
1. Per-cycle gradient norms: are earlier cycles getting meaningful gradient now?
   Before scaling: cycle 1 grad << cycle 3 grad
   After scaling:  cycle 1 grad ≈ cycle 3 grad (compensated)

2. Train/eval gap: does fresh data per epoch reduce overfitting?
   Before: ans_loss → 0.0000 by epoch 4, accuracy collapses
   After:  ans_loss should stay higher (new problems each epoch), accuracy should hold

3. Level-by-level accuracy: does filling the gap help?
   L4.5 > 80%?  L4.7 > 60%?  L4.9 > 30%?  L5 > 20%?

4. Blend trajectory: does verification increase on harder levels?
   L4.5 blend vs L5 blend — harder problems should verify more
```

---

## What NOT to Do

```
- Do NOT use deep supervision. Each cycle does a chunk, not the whole problem.
- Do NOT reuse training data across epochs. Fresh data prevents memorization.
- Do NOT skip intermediate levels. The L4→L5 cliff needs filling.
- Do NOT set gradient scaling too high. 3x for 3 cycles is the starting point.
  If cycles=6, scale would be 6x for cycle 1 — might destabilize. Cap at 3-4x.
- Do NOT change the architecture. These are training fixes, not architecture fixes.
```
