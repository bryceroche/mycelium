# Handoff: Confidence Head + Gentle Training Wheel Removal

## One-Sentence Summary

Train the confidence head to decide when to stop thinking (enabling variable-length decomposition), then GENTLY remove teacher decompositions by blending teacher and self-discovered intermediates over a curriculum — not a hard cutoff that collapses the model.

---

## Part 1: Confidence Head (Variable Cycle Count)

### Why This Is Essential

The model currently uses fixed num_passes. Every problem gets the same number of cycles regardless of difficulty. This prevents self-decomposition:

```
Current:  every problem → 3 cycles (hardcoded)
          Easy: "5 + 3" → 3 cycles (wasteful, 2 cycles of nothing)
          Hard: "multi-step GSM8K" → 3 cycles (insufficient, needs 6)

Needed:   model decides per problem
          Easy: "5 + 3" → 1 cycle → confident → stop
          Medium: "160 - 63 + 20" → 3 cycles → confident → stop
          Hard: GSM8K → 6 cycles → confident → stop
```

Without variable cycles, the model can't adapt its decomposition to the problem. It can't decide "this needs 2 more steps" vs "I'm done." The confidence head IS the decomposition termination signal.

### Training the Confidence Head

The confidence head reads the page delta and predicts: "would the answer head get the right answer if I stopped now?"

```python
class ConfidenceHead(nn.Module):
    def __init__(self, page_size=64, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(page_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, page_delta):
        return self.net(page_delta)  # (batch, 1) — probability of being done
```

Training signal — compare the answer head's prediction to the gold answer AT EACH CYCLE:

```python
def train_confidence(model, problem_ids, cycle_targets, final_answer):
    state_pages = []
    conf_loss = 0.0
    
    for cycle in range(num_cycles):
        page = model.think_one_pass(problem_ids, state_pages, cycle)
        state_pages.append(page)
        
        # What would the answer head predict if we stopped now?
        if cycle == 0:
            delta = page
        else:
            delta = page - state_pages[-2]
        
        pred = model.answer_head.decode(delta, cycle)
        
        # Is it correct?
        is_correct = (pred == final_answer)
        
        # Confidence should predict whether we'd be right
        conf = model.confidence_head(delta)
        target = torch.tensor([[float(is_correct)]], device=page.device)
        conf_loss += F.binary_cross_entropy(conf, target)
    
    return conf_loss / num_cycles
```

The confidence head learns:
```
After cycle 1: answer_head predicts 160, gold is 117 → WRONG → conf target = 0.0
After cycle 2: answer_head predicts 97, gold is 117  → WRONG → conf target = 0.0
After cycle 3: answer_head predicts 117, gold is 117 → RIGHT → conf target = 1.0

The head learns: "don't stop until the answer head would get it right"
```

### Inference With Variable Cycles

```python
def solve(model, problem_ids, max_cycles=12, conf_threshold=0.85):
    state_pages = []
    prev_results = []
    
    for cycle in range(max_cycles):
        # Inject previous results as text
        ctx = format_previous_results(prev_results)
        full_input = tokenize(ctx + problem_text)
        
        page = model.think_one_pass(full_input, state_pages, cycle)
        state_pages.append(page)
        
        # Check confidence
        delta = page - state_pages[-2] if cycle > 0 else page
        conf = model.confidence_head(delta)
        
        # Extract intermediate for text injection
        pred = model.answer_head.decode(delta, cycle)
        prev_results.append(pred)
        
        if conf > conf_threshold and cycle >= 1:
            break  # model says "I'm done"
    
    return pred  # last prediction is the final answer
```

Easy problems stop early. Hard problems think longer. The decomposition length adapts to the problem.

### Training Schedule for Confidence

```
Phase 1 (current):  fixed cycles, confidence head trains as auxiliary loss
                    weight: 0.1 (doesn't interfere with main training)
                    
Phase 2:            confidence head influences cycle count during training
                    if conf > 0.95 on a graduated cycle, skip remaining computation
                    saves compute on easy problems
                    
Phase 3:            full variable cycles
                    the model decides when to stop
                    no hardcoded num_passes
```

---

## Part 2: Smooth Fading of Teacher Decompositions

### The Problem

We currently provide per-cycle intermediate targets: [160, 97, 117]. The model learns to produce THESE specific intermediates in THIS specific order. But:

```
With teacher targets:     model EXECUTES our decomposition (puppet)
Without teacher targets:  model must DISCOVER decomposition (intelligent)
```

Any DISCRETE change to the training signal destabilizes the system. We learned this the hard way — every graduation threshold, loss skip, and teacher forcing cutoff broke something. The solution must be SMOOTH and CONTINUOUS.

### The Core Principle: No Discrete Jumps

Every hyperparameter transition must be smooth. If you find yourself writing `if metric > threshold: change_something`, replace it with a sigmoid ramp. The gradient-based training lives in a continuous world. Discrete jumps are foreign objects.

### Smooth Fading: Per-Cycle Target Weight

The intermediate cycle target weights FADE smoothly as final accuracy climbs. No thresholds. No stages. One continuous function:

```python
def per_cycle_target_weight(final_accuracy, cycle, total_cycles):
    """
    Smoothly fade intermediate targets as the model masters the task.
    
    Final cycle: ALWAYS has full weight (the answer must be right).
    Intermediate cycles: fade from 1.0 to 0.0 as accuracy climbs.
    
    The model EARNS freedom through demonstrated competence.
    The transition is smooth — no discrete jumps, no equilibrium disruption.
    """
    if cycle == total_cycles - 1:
        return 1.0  # final cycle always fully supervised
    
    # Smooth sigmoid fade centered at 80% accuracy
    # At 50% accuracy: weight ≈ 1.0 (full teacher, model still learning)
    # At 75% accuracy: weight ≈ 0.8 (mostly teacher, starting to ease off)
    # At 80% accuracy: weight ≈ 0.5 (half teacher, half autonomous)
    # At 85% accuracy: weight ≈ 0.2 (mostly free, teacher as safety net)
    # At 90% accuracy: weight ≈ 0.05 (nearly autonomous)
    # At 95% accuracy: weight ≈ 0.01 (fully autonomous in practice)
    fade = torch.sigmoid(torch.tensor((final_accuracy - 0.80) * 15.0))
    return float(1.0 - fade)
```

The sigmoid curve is smooth, continuous, and S-shaped. The transition happens gradually over the 70-90% accuracy range. No cliff. No jump. The training dynamics evolve continuously.

### Implementation

```python
def train_step_smooth(model, problem_ids, cycle_targets, final_answer, 
                       final_accuracy, num_cycles):
    """
    Training with smooth fading of intermediate targets.
    
    As final_accuracy climbs, intermediate cycle targets fade smoothly.
    The model transitions from executing prescribed decompositions
    to discovering its own — continuously, without disruption.
    """
    notebook = []
    total_loss = 0.0
    
    for cycle in range(num_cycles):
        page, gen_text = model.think_one_pass(problem_ids, notebook, cycle)
        notebook.append(page)
        
        # Compute smooth target weight for this cycle
        weight = per_cycle_target_weight(final_accuracy, cycle, num_cycles)
        
        if weight > 0.01:  # teacher signal active (smooth, not discrete)
            # Teacher target: prescribed intermediate
            gen_loss = generation_loss(page, problem_ids, cycle_targets[cycle])
            head_loss = answer_head_loss(page, cycle_targets[cycle], cycle)
            
            if cycle == 0:
                total_loss += weight * (1.0 * gen_loss + 0.5 * head_loss)
            else:
                total_loss += weight * (0.1 * gen_loss + 5.0 * head_loss)
        
        if cycle == num_cycles - 1:
            # Final cycle: ALWAYS supervised with correct answer
            final_head_loss = answer_head_loss(page, final_answer, cycle)
            final_gen_loss = generation_loss(
                page, problem_ids, f"The answer is {final_answer}."
            )
            total_loss += 5.0 * final_head_loss + 1.0 * final_gen_loss
    
    # Notebook diversity: pages must be different from each other
    # (prevents all cycles from doing the same thing when targets fade)
    for i in range(len(notebook)):
        for j in range(i+1, len(notebook)):
            cos = F.cosine_similarity(notebook[i], notebook[j], dim=-1)
            total_loss += 0.1 * F.relu(cos - 0.3).mean()
    
    # Confidence loss (always active)
    conf_loss = train_confidence(model, notebook, final_answer)
    total_loss += 0.1 * conf_loss
    
    return total_loss
```

### The Smooth Curve in Action

```
Accuracy 50%:  intermediate weight ≈ 1.00
               "Follow the recipe exactly."
               Full teacher targets. Every cycle prescribed.
               The model learns mechanics and pattern library.

Accuracy 70%:  intermediate weight ≈ 0.85
               "Follow the recipe, but you can season to taste."
               Targets still strong. Small freedom at margins.
               Model starts noticing which targets matter most.

Accuracy 80%:  intermediate weight ≈ 0.50
               "I'll show you half the steps."
               Equal blend. Teacher and self-discovered.
               The model starts finding its own decomposition.

Accuracy 85%:  intermediate weight ≈ 0.18
               "You mostly know what you're doing."
               Targets are faint suggestions, not commands.
               The model's own decomposition dominates.

Accuracy 90%:  intermediate weight ≈ 0.05
               "Cook on your own. I'll just taste the final dish."
               Targets nearly zero. Fully emergent decomposition.
               Only the final answer is meaningfully supervised.

Accuracy 95%:  intermediate weight ≈ 0.01
               "Master chef. I trust you completely."
               The model decomposes novel problems entirely on its own.
```

### Why Smooth Fading Works

**No equilibrium disruption.** The training dynamics evolve continuously. At every point, the loss landscape changes infinitesimally. The optimizer's momentum and learning rate remain calibrated. No sudden gradient shocks.

**Competence earns freedom.** Higher accuracy → lower target weight → more freedom → potentially better decomposition → higher accuracy. A virtuous cycle. The model discovers that its OWN decompositions sometimes work better than the prescribed ones.

**Self-correcting.** If the model's accuracy drops (because its self-discovered decomposition was bad), target weight increases (sigmoid moves backward), and the teacher signal strengthens. The system self-stabilizes.

**No stages to manage.** No "Phase 1, Phase 2, Phase 3." No manual decisions about when to increase autonomy. The sigmoid handles everything automatically based on one metric: final accuracy.

### Gradient Flow During Fading

As intermediate targets fade, gradient to early cycles must come from the FINAL cycle's loss, flowing backward through the notebook:

```
Final cycle loss → hypernetwork reads all pages → attention over notebook →
  gradient flows to each page proportional to attention weight →
  early pages get gradient saying "encode useful intermediates"
```

With the growing notebook (no blending), this gradient path is clean. The hypernetwork cross-attends over all pages. Pages that contribute to the final answer get strong attention and strong gradient. Pages that don't contribute get weak gradient and evolve slowly.

The notebook diversity loss (pages must differ) ensures early cycles don't collapse to noise when their targets fade — they must still contribute SOMETHING unique, even if we don't prescribe WHAT.

### The Key Insight

The teacher targets are TRAINING WHEELS. They're essential early — without them, cycles collapse to one pass. But they're also CONSTRAINING — they prescribe OUR decomposition, not the model's.

Smooth fading is NOT removing training wheels abruptly (falling). It's NOT keeping them forever (never learning to ride). It's GRADUALLY raising them as balance improves. The model never notices the transition. One day it's riding on its own.

### What Self-Discovered Decomposition Looks Like

After training with increasing autonomy, the model might discover decompositions different from ours:

```
OUR decomposition (prescribed):
  Cycle 1: extract 48 from "Natalia sold 48 clips"
  Cycle 2: compute 48/2 = 24 from "half as many"
  Cycle 3: compute 48+24 = 72 from "how many total"

MODEL'S decomposition (discovered):
  Cycle 1: extract 48 AND recognize "half" → encode both
  Cycle 2: compute 48 + 24 = 72 directly (combined two steps!)
  Confidence: high → stop after 2 cycles
  
  The model found a SHORTER decomposition. Bigger bites. Fewer cycles.
  This IS the Panama hat principle — it matched a larger pattern.
```

Or for a harder problem, the model might take SMALLER steps than we prescribed:

```
OUR decomposition:
  Cycle 1: extract 48
  Cycle 2: compute 48/2 = 24
  Cycle 3: compute 48+24 = 72

MODEL'S decomposition (for a weaker model):
  Cycle 1: extract 48
  Cycle 2: recognize "half" means /2
  Cycle 3: compute 48/2 = 24
  Cycle 4: recognize "total" means addition
  Cycle 5: compute 48+24 = 72
  
  Smaller model takes smaller bites. More cycles. Same answer.
```

The decomposition granularity adapts to the model's capability. Larger models take bigger bites (fewer cycles). Smaller models take smaller bites (more cycles). Same architecture, same training, different emergent decomposition.

---

## Verification: Is Self-Decomposition Working?

### Diagnostic 1: Do autonomous problems get correct answers?

```python
# Compare accuracy on teacher vs autonomous problems
teacher_acc = accuracy(problems_with_teacher_targets)
autonomous_acc = accuracy(problems_without_teacher_targets)

# If autonomous_acc ≈ teacher_acc: self-decomposition works
# If autonomous_acc << teacher_acc: model relies on teacher targets
```

### Diagnostic 2: What intermediates does the model discover?

```python
# For autonomous problems, record what each cycle predicts
for problem in autonomous_problems:
    intermediates = []
    for cycle in range(max_cycles):
        page = model.think_one_pass(...)
        pred = model.answer_head.decode(page_delta)
        intermediates.append(pred)
        if model.confidence_head(page_delta) > 0.85:
            break
    
    print(f"Problem: {problem}")
    print(f"Our decomposition: {teacher_targets}")
    print(f"Model's decomposition: {intermediates}")
    print(f"Same? Different? Fewer steps? More steps?")
```

### Diagnostic 3: Does decomposition vary by problem difficulty?

```python
# Easy problems should use fewer cycles
# Hard problems should use more cycles
easy_cycles = mean([num_cycles for p in easy_problems])
hard_cycles = mean([num_cycles for p in hard_problems])

# If easy_cycles < hard_cycles: model adapts decomposition to difficulty
# If easy_cycles ≈ hard_cycles: model uses fixed strategy regardless
```

### Diagnostic 4: Pattern space clustering

```python
# Project all page deltas into pattern space
# Do they cluster by operation type WITHOUT being told to?
deltas = collect_all_page_deltas(eval_problems)
projected = pattern_projector(deltas)  # (N, 16)

# Cluster and label
clusters = kmeans(projected, k=8)
for cluster_id in range(8):
    problems_in_cluster = get_problems(clusters == cluster_id)
    print(f"Cluster {cluster_id}: {analyze_operation_types(problems_in_cluster)}")
    
# If clusters align with operation types: patterns emerged naturally
# If clusters are random: no meaningful pattern structure learned
```

---

## Implementation Order

```
NOW:     Fix graduated cycle weight (0.1 anchor, not detach or skip)
         Push L4.5 to graduation
         
NEXT:    Train confidence head as auxiliary loss (0.1 weight)
         Train alongside main per-cycle loss
         Don't use for stopping yet — just learn the signal
         
THEN:    Enable variable cycles at inference time
         Use confidence head to stop early
         Verify: easy problems → fewer cycles, hard → more

AFTER:   Begin gentle training wheel removal
         Start at autonomy_rate = 0.0 (current)
         Increase based on accuracy thresholds
         Monitor autonomous_acc vs teacher_acc
         
GOAL:    autonomy_rate = 1.0
         Model decomposes novel problems on its own
         Confidence head decides cycle count
         Pattern library (atoms) handles all operation types
         No teacher annotations needed
```

---

## What NOT to Do

```
- Do NOT remove teacher targets all at once.
  The model collapses. It's never had to plan its own decomposition.
  Gradual blending (0% → 20% → 50% → 80% → 100%) is essential.

- Do NOT hardcode a list of pattern types.
  The patterns should be LEARNED, not prescribed.
  The 64 atoms and their continuous blends ARE the pattern space.
  Let clusters emerge from training, not from our definitions.

- Do NOT force the model to use a specific number of cycles.
  The confidence head decides. Easy → few cycles. Hard → many.
  Forcing cycle count prevents adaptive decomposition.

- Do NOT evaluate self-decomposition on teacher-annotated accuracy.
  The model might find BETTER decompositions than ours.
  Judge by FINAL ANSWER accuracy, not intermediate accuracy.
  The intermediates are means, not ends.

- Do NOT increase autonomy faster than the model can handle.
  If autonomous_acc drops more than 10% below teacher_acc, 
  REDUCE autonomy_rate. The model isn't ready.
  Let it earn autonomy through demonstrated competence.
```

---

## The Long-Term Vision

```
Phase 1 (now):    Teacher decompositions — learn the mechanics
Phase 2 (next):   Confidence head — learn WHEN to stop
Phase 3 (then):   Gentle autonomy — learn to decompose on own
Phase 4 (goal):   Fully autonomous — decompose novel problems

The model progresses from puppet → apprentice → journeyman → master.

Puppet:      executes prescribed decompositions
Apprentice:  follows teacher most of the time, experiments sometimes  
Journeyman:  mostly self-directed, teacher for hard cases
Master:      decomposes any problem, adapts strategy to difficulty

The 64 atoms are the master's toolkit.
The hypernetwork is the master's judgment.
The confidence head is the master's sense of "done."
The pages are the master's working memory.
The breathing loop is the master's method: one bite at a time.
```
