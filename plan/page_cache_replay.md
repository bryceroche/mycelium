# Handoff: Page Cache + Replay Buffer + Per-Cycle Curriculum

## One-Sentence Summary

Cache early cycle pages that have graduated (high per-step accuracy). Focus training compute on later cycles where accuracy is lowest. Use a replay buffer to train later cycles against varying quality of earlier thinking. Training gets faster as more cycles graduate.

---

## The Problem

Every training step runs ALL cycles from scratch:

```
Step 1: Llama fwd → perceive → page 1 → Llama fwd → perceive → page 2 → ... → page 5 → loss → backward
        5 Llama forwards + 1 full backward = ~8 seconds per step

But cycles 1-2 might already be at 95% per-step accuracy.
We're spending 40% of compute on cycles that don't need training.
```

## The Solution: Cache and Focus

### Phase 1: Identify Graduated Cycles

Use per-step accuracy diagnostics to measure each cycle's contribution:

```python
def measure_per_step_accuracy(model, eval_problems, max_passes=5):
    """At each pass, generate answer from current pages. Track when it becomes correct."""
    per_step_acc = [0.0] * max_passes
    
    for problem, gold in eval_problems:
        state_pages = []
        for pass_num in range(max_passes):
            page = model.think_one_pass(problem, state_pages, pass_num)
            state_pages.append(page)
            
            # Would the current pages produce the right answer?
            with torch.no_grad():
                answer = model.generate_from_pages(state_pages, problem)
                if extract_number(answer) == gold:
                    per_step_acc[pass_num] += 1
    
    per_step_acc = [a / len(eval_problems) for a in per_step_acc]
    return per_step_acc
    
# Example output:
# [0.45, 0.62, 0.78, 0.85, 0.88]
# Cycle 1: 45% → still learning
# Cycle 5: 88% → close to final accuracy
```

A cycle "graduates" when its per-step accuracy stops improving across epochs:

```
Graduation threshold: per-step accuracy stable for 2+ epochs
                      OR per-step accuracy > 90%
```

### Phase 2: Cache Graduated Pages

```python
class PageCache:
    def __init__(self):
        self.cache = {}  # problem_hash → {epoch: [pages]}
        self.graduated_up_to = 0  # cycles 0..graduated_up_to are cached
    
    def store(self, problem_hash, pages, epoch):
        """Cache pages from a full training run."""
        if problem_hash not in self.cache:
            self.cache[problem_hash] = {}
        self.cache[problem_hash][epoch] = [p.detach().cpu() for p in pages]
    
    def load(self, problem_hash, up_to_pass, epoch=None):
        """
        Load cached pages up to a specific pass.
        If epoch=None, pick a random cached epoch (replay buffer).
        """
        if problem_hash not in self.cache:
            return None
        
        available_epochs = list(self.cache[problem_hash].keys())
        if not available_epochs:
            return None
        
        if epoch is None:
            epoch = random.choice(available_epochs)  # replay buffer
        elif epoch not in available_epochs:
            epoch = max(available_epochs)  # use most recent
        
        pages = self.cache[problem_hash][epoch][:up_to_pass]
        return [p.to(device).requires_grad_(False) for p in pages]
    
    def update_graduation(self, per_step_acc, threshold=0.90):
        """Advance the graduation frontier based on per-step accuracy."""
        for i, acc in enumerate(per_step_acc):
            if acc >= threshold:
                self.graduated_up_to = i + 1
            else:
                break
        print(f"Graduated cycles: 0-{self.graduated_up_to - 1}")
```

### Phase 3: Training With Cache

```python
def train_step_with_cache(model, problem, gold_answer, cache, max_passes=5):
    problem_hash = hash(problem)
    graduated = cache.graduated_up_to
    
    # Decide where to start this step
    r = random.random()
    
    if graduated > 0 and r < 0.7:
        # 70%: start from the graduation frontier (fast — skip graduated cycles)
        cached_pages = cache.load(problem_hash, graduated)
        if cached_pages is not None:
            state_pages = cached_pages
            start_pass = graduated
        else:
            state_pages = []
            start_pass = 0  # cache miss — full run
    elif graduated > 0 and r < 0.9:
        # 20%: start from one cycle before frontier (keep frontier cycle fresh)
        start_from = max(0, graduated - 1)
        cached_pages = cache.load(problem_hash, start_from)
        if cached_pages is not None:
            state_pages = cached_pages
            start_pass = start_from
        else:
            state_pages = []
            start_pass = 0
    else:
        # 10%: full run from scratch (keep early cycles learning)
        state_pages = []
        start_pass = 0
    
    # Run remaining cycles
    for pass_num in range(start_pass, max_passes):
        page = model.think_one_pass(problem, state_pages, pass_num)
        state_pages.append(page)
    
    # Loss and backward (gradient only flows to cycles that ran)
    loss = compute_loss(model, state_pages, gold_answer)
    loss.backward()
    
    # Update cache on full runs
    if start_pass == 0:
        cache.store(problem_hash, state_pages, epoch=current_epoch)
    
    return loss.item(), start_pass
```

---

## The Replay Buffer

The cache stores pages from MULTIPLE epochs. When loading cached pages, we randomly sample from different epochs. This means later cycles train against varying quality of earlier thinking:

```
Cache for problem "Natalia clips":
  Epoch 2 pages: [noisy_p1, noisy_p2, noisy_p3]    (early training, rough)
  Epoch 5 pages: [better_p1, better_p2, better_p3]   (mid training, decent)
  Epoch 8 pages: [good_p1, good_p2, good_p3]         (late training, clean)
```

When training cycle 4:

```
Step 100: load epoch 2 pages → cycle 4 sees noisy inputs → learns robustness
Step 101: load epoch 8 pages → cycle 4 sees clean inputs → learns precision
Step 102: load epoch 5 pages → cycle 4 sees decent inputs → learns adaptation
```

Later cycles become ROBUST to imperfect earlier thinking. Like practicing the boss fight with different starting conditions — sometimes full health, sometimes half health.

### Replay Buffer Management

```python
class ReplayBuffer(PageCache):
    def __init__(self, max_epochs_stored=5):
        super().__init__()
        self.max_epochs = max_epochs_stored
    
    def store(self, problem_hash, pages, epoch):
        """Store with eviction of old epochs."""
        super().store(problem_hash, pages, epoch)
        
        # Evict oldest epochs if too many stored
        if problem_hash in self.cache:
            epochs = sorted(self.cache[problem_hash].keys())
            while len(epochs) > self.max_epochs:
                del self.cache[problem_hash][epochs.pop(0)]
    
    def load_diverse(self, problem_hash, up_to_pass, current_epoch):
        """
        Load from a random epoch, biased toward recent but with diversity.
        """
        if problem_hash not in self.cache:
            return None
        
        available = list(self.cache[problem_hash].keys())
        if not available:
            return None
        
        # 60% most recent, 40% random older epoch
        if random.random() < 0.6:
            epoch = max(available)
        else:
            epoch = random.choice(available)
        
        pages = self.cache[problem_hash][epoch][:up_to_pass]
        return [p.to(device).requires_grad_(False) for p in pages]
```

---

## The Training Frontier Moves Forward

As training progresses, more cycles graduate and get cached. Training gets faster:

```
Epoch 1:   graduated=0  → all 5 cycles every step  → 5 forwards/step
Epoch 3:   graduated=1  → cache cycle 1             → avg 4.3 forwards/step
Epoch 5:   graduated=2  → cache cycles 1-2          → avg 3.3 forwards/step
Epoch 8:   graduated=3  → cache cycles 1-3          → avg 2.6 forwards/step
Epoch 12:  graduated=4  → cache cycles 1-4          → avg 1.7 forwards/step
```

Each epoch is faster than the last. Compute flows to where learning is happening.

### Graduation Check Per Epoch

```python
def train_epoch_with_cache(model, train_data, eval_data, cache, epoch, max_passes=5):
    # Train
    for batch in train_data:
        train_step_with_cache(model, batch, cache, max_passes)
    
    # Evaluate per-step accuracy
    per_step_acc = measure_per_step_accuracy(model, eval_data, max_passes)
    print(f"Per-step accuracy: {per_step_acc}")
    
    # Update graduation frontier
    old_graduated = cache.graduated_up_to
    cache.update_graduation(per_step_acc, threshold=0.90)
    new_graduated = cache.graduated_up_to
    
    if new_graduated > old_graduated:
        print(f"Cycles 0-{new_graduated-1} GRADUATED! Caching enabled.")
        
        # Populate cache: run all training problems through the graduated cycles
        print("Populating cache for graduated cycles...")
        with torch.no_grad():
            for problem in train_data:
                problem_hash = hash(problem)
                state_pages = []
                for pass_num in range(new_graduated):
                    page = model.think_one_pass(problem, state_pages, pass_num)
                    state_pages.append(page)
                cache.store(problem_hash, state_pages, epoch)
    
    # Full evaluation
    accuracy = evaluate(model, eval_data, max_passes)
    return accuracy, per_step_acc
```

---

## Connection to GSM8K Per-Step Data

We have diagnostic data showing where GSM8K accuracy breaks down per cycle. This directly informs the cache strategy:

```
Expected GSM8K per-step profile:
  Cycle 1 (parse):      high accuracy (base model understands language)
  Cycle 2 (1st compute): moderate (single operations work)
  Cycle 3 (2nd compute): lower (chaining is hard)
  Cycle 4 (verify):      lowest (checking 5-step solutions is hardest)
  Cycle 5 (answer):      depends on extraction quality

Training allocation:
  Cycles 1-2: graduate early, cache by epoch 3-5
  Cycle 3:    graduate mid-training, cache by epoch 8-10
  Cycles 4-5: never fully graduate on hard problems, always train
```

The hardest GSM8K problems might NEVER graduate cycles 3-5. That's fine — those cycles keep training. The easy problems graduate all cycles quickly — training steps on easy problems are fast (load all cached pages, only run generation).

---

## Memory Management

Caching pages for all training problems:

```
Per problem: 5 passes × 64 floats × 4 bytes = 1.28 KB
5 epochs of replay buffer: × 5 = 6.4 KB per problem

GSM8K training: 7,473 problems × 6.4 KB = ~48 MB
Procedural L2-L4: 20,000 problems × 6.4 KB = ~128 MB

Total cache: < 200 MB — fits easily in RAM (no GPU memory needed)
```

The cache lives on CPU. Pages are moved to GPU only when loaded for training. Negligible memory overhead.

---

## Speed Impact

```
Without cache (current):
  5 Llama forwards per step × 7,473 problems = 37,365 forwards per epoch
  At ~0.3s per forward = ~3.1 hours per epoch

With cache (graduated=2, epoch 5+):
  Avg 3.3 forwards per step × 7,473 = 24,661 forwards per epoch
  At ~0.3s per forward = ~2.1 hours per epoch
  Speedup: 1.5x

With cache (graduated=3, epoch 8+):
  Avg 2.6 forwards per step × 7,473 = 19,430 forwards per epoch
  At ~0.3s per forward = ~1.6 hours per epoch
  Speedup: 1.9x

With cache (graduated=4, epoch 12+):
  Avg 1.7 forwards per step × 7,473 = 12,704 forwards per epoch
  At ~0.3s per forward = ~1.1 hours per epoch
  Speedup: 2.8x
```

Training accelerates as the model improves. The better the model gets, the faster it trains. Virtuous cycle.

---

## What to Monitor

```
1. Per-step accuracy per epoch:
   Track which cycles are improving and which have plateaued.
   Graduation decisions are based on this.

2. Cache hit rate:
   What fraction of steps load from cache vs run from scratch?
   Should increase over training as more cycles graduate.

3. Training speed:
   Seconds per epoch should decrease as graduation advances.
   Plot forwards_per_step across epochs.

4. Replay diversity:
   Are later cycles seeing cached pages from multiple epochs?
   Check that the replay buffer has 3-5 epoch versions per problem.

5. Robustness check:
   Run full end-to-end eval (no cache) periodically.
   Accuracy should match cached training performance.
   If cached training accuracy >> full eval accuracy, the cache is stale.
```

---

## What NOT to Do

```
- Do NOT cache pages with requires_grad=True. Cached pages are DETACHED.
  Gradient only flows through cycles that actually run.

- Do NOT graduate too aggressively. Use 90% threshold and stable-for-2-epochs.
  Premature graduation means stale cached pages that hurt later cycles.

- Do NOT skip full runs entirely. Always run 10% of steps from scratch.
  This keeps early cycles fresh and populates the cache with updated pages.

- Do NOT store cached pages on GPU. Use CPU storage, move to GPU on load.
  The cache can grow large but RAM is cheap.

- Do NOT use the cache for eval. Always run full end-to-end for evaluation.
  The cache is a training optimization, not an eval shortcut.

- Do NOT cache during epoch 1. All cycles need full gradient in early training.
  Enable caching only after the first graduation check (epoch 2-3).
```

---

## Implementation Order

```
1. Add per-step accuracy measurement to eval (measure_per_step_accuracy)
2. Implement PageCache with replay buffer
3. Add graduation logic to training loop
4. Modify train_step to load from cache probabilistically
5. Test on L3 first (fast iteration, proven results)
6. Deploy on GSM8K (where 5-pass training is expensive and cache matters most)
```
