# Handoff: Per-Cycle Intermediate Targets — One Bite Per Cycle

## One-Sentence Summary

Train each thinking cycle to compute ONE intermediate result, not the final answer. The answer head predicts the intermediate result for EACH cycle — cycle 1 extracts the first number, cycle 2 computes the first operation, cycle 3 computes the second. No CoT generation. No text extraction. The answer falls out of the last page's answer head prediction.

---

## The Problem With CoT Targets

CoT targets teach the model to solve the ENTIRE problem in one generation pass:

```
CoT target: "Natalia sold 48 clips. She sold half as many in May.
             48 / 2 = 24. 48 + 24 = 72. The answer is 72."

What the model learns: "generate all reasoning steps in one shot"
What happens to cycles: they're redundant — one pass generates everything
Result: the thinking loop is unused, 10/64 page dims, 17% ceiling
```

The CoT target makes the cycles unnecessary. The model bypasses the breathing loop and dumps everything into text generation. The architecture's core design — incremental thinking across cycles — is undermined by the training target.

---

## The Fix: Per-Cycle Intermediate Targets

Each cycle gets its OWN target — the intermediate result for that step of the problem:

```
Problem: "Natalia sold clips to 48 friends in April.
          She sold half as many in May. How many total?"

Decomposition:
  Step 1: Extract first quantity       → 48
  Step 2: Compute half                 → 24  (48 / 2)
  Step 3: Compute total                → 72  (48 + 24)

Per-cycle targets:
  Cycle 1 target: 48  (extract)
  Cycle 2 target: 24  (compute)
  Cycle 3 target: 72  (compute)
  Final answer:   72  (same as last cycle)
```

Each cycle's answer head predicts ONE number. That number is the result of THIS cycle's thinking — not the final answer but the INTERMEDIATE result for this step.

---

## Why This Is the Missing Piece

### 1. Every Cycle Gets Direct Gradient

```
BEFORE (CoT):
  Cycle 1: no target (just runs, hopes to help generation later)
  Cycle 2: no target (same)
  Cycle 3: no target (same)
  Generation: target = full CoT text (gradient must flow backward through all cycles)
  
  Result: cycles 1-2 get attenuated gradient. They learn slowly or not at all.

AFTER (per-cycle):
  Cycle 1: target = 48 → answer_head_loss → gradient directly to page 1
  Cycle 2: target = 24 → answer_head_loss → gradient directly to page 2
  Cycle 3: target = 72 → answer_head_loss → gradient directly to page 3
  
  Result: every cycle gets STRONG, DIRECT gradient. No attenuation.
```

### 2. Pages MUST Use More Dimensions

```
BEFORE:
  All cycles optimize for one final answer (72)
  The perceiver encodes "72" in ~10 dims
  54 dims unused (no incentive to use them)

AFTER:
  Cycle 1 must encode 48
  Cycle 2 must encode 24
  Cycle 3 must encode 72
  Each cycle needs DIFFERENT dims for DIFFERENT numbers
  The residual gate preserves earlier numbers while adding new ones
  The perceiver is FORCED to use more dims to track multiple intermediates
```

### 3. Cycles Are No Longer Redundant

```
BEFORE:
  Every cycle tries to produce the same final answer
  The model learns: "one good pass is enough, repeat it"
  Cycles 2-5 are copies of cycle 1 (the copying problem returns)

AFTER:
  Cycle 1 MUST produce 48 (not 72)
  Cycle 2 MUST produce 24 (not 48, not 72)
  Cycle 3 MUST produce 72 (the final answer)
  Each cycle has a UNIQUE job. Copying another cycle gives the WRONG target.
```

### 4. No Generation Fragility

```
BEFORE:
  Think → generate CoT text → regex extract number → hope it's right
  Failure modes: MCQ generation, number spam, format bugs, extraction errors

AFTER:
  Think → answer head reads last page → digit prediction → number
  No text generation. No regex. No format issues.
  The answer is a direct neural network output, not parsed text.
```

---

## Architecture Changes

### Training: Per-Cycle Answer Head Loss

```python
def train_step_per_cycle(model, problem_ids, cycle_targets, max_passes=None):
    """
    Train with per-cycle intermediate targets.
    
    cycle_targets: list of ints, one per thinking step
                   e.g. [48, 24, 72] for a 3-step problem
    """
    if max_passes is None:
        max_passes = len(cycle_targets)
    
    state_pages = []
    total_loss = 0.0
    
    for pass_num in range(max_passes):
        # Think one cycle
        page = model.think_one_pass(problem_ids, state_pages, pass_num)
        state_pages.append(page)
        
        # Per-cycle target: what should this cycle's page encode?
        if pass_num < len(cycle_targets):
            target = cycle_targets[pass_num]
            cycle_loss = model.answer_head_loss(page, target)
            total_loss += cycle_loss
    
    # Contrastive loss (still needed — prevents page collapse)
    contrastive_loss = compute_contrastive(state_pages)
    total_loss += 0.05 * contrastive_loss
    
    # Scale regularization (still needed — prevents tanh saturation)
    total_loss += 0.1 * model.get_scale_reg()
    
    return total_loss
```

### Inference: Read Last Page

```python
def solve(model, problem_ids, max_passes=8):
    """
    Solve by thinking N cycles, reading the answer from the last page.
    No text generation needed.
    """
    state_pages = []
    
    for pass_num in range(max_passes):
        page = model.think_one_pass(problem_ids, state_pages, pass_num)
        state_pages.append(page)
        
        # Check confidence
        if pass_num >= 1:
            conf, smooth = model.confidence_head(state_pages)
            if conf > 0.9 and smooth > 0.7:
                break
    
    # Answer from last page — no generation
    answer = model.answer_head.decode(state_pages[-1])
    return answer
```

### Loss Function (Simplified)

```python
total_loss = (sum(per_cycle_answer_losses)    # each cycle predicts its intermediate
              + 0.05 * contrastive_loss        # prevent page collapse
              + 0.1 * confidence_loss          # when to stop
              + 0.1 * scale_reg)               # prevent tanh saturation

# NO generation loss. NO CoT loss. The answer head IS the primary loss.
# The model never generates text during training.
```

---

## Training Data: Per-Step Decompositions

Each problem needs a decomposition into intermediate results. This is simpler than SymPy annotations — just the intermediate NUMBERS, not symbolic expressions.

### Format

```json
{
    "problem": "Natalia sold clips to 48 friends in April. She sold half as many in May. How many total?",
    "cycle_targets": [48, 24, 72],
    "final_answer": 72,
    "num_steps": 3
}
```

### Generating Decompositions

For procedural data (L3-L4), decompositions are known at generation time:

```python
def generate_L4_problem():
    a = random.randint(10, 100)
    b = random.randint(1, a)
    c = random.randint(1, 50)
    answer = a - b + c
    
    problem = f"Jamie had {a} cookies. He gave {b} away. Then he got {c} more."
    
    return {
        "problem": problem,
        "cycle_targets": [a, a - b, answer],  # extract, subtract, add
        "final_answer": answer,
    }
```

For GSM8K, use a larger model (Claude/GPT-4) to decompose once:

```
Prompt: "Break this problem into numbered steps. 
         For each step, give ONLY the numeric result.
         
         Problem: Natalia sold clips to 48 friends in April...
         
         Step 1 result: 48
         Step 2 result: 24
         Step 3 result: 72"
```

Store as JSON. One-time annotation cost for 7,473 problems.

---

## Curriculum With Per-Cycle Targets

Each level adds one more step. The model gradually learns to use more cycles:

```
L3 (1 step):
  "Jamie had 56 cookies and gave 2 away."
  cycle_targets = [54]
  1 cycle. Model learns: "read the page, predict the answer."
  Page dims needed: ~5 (just the answer)

L4 (2 steps):
  "Had 56, gave 2 away, then got 10 more."
  cycle_targets = [54, 64]
  2 cycles. Model learns: "cycle 1 does first op, cycle 2 does second."
  Page dims needed: ~10 (two intermediate numbers)

L4.5 (3 steps):
  "Had 100, lost 15, then doubled what remained."
  cycle_targets = [100, 85, 170]
  3 cycles. Model uses 3 passes with different atom configs.
  Page dims needed: ~15

L4.7 (4 steps):
  "Bought 50 at $3, sold 30 at $5, kept the rest."
  cycle_targets = [150, 150, 100, -50]
  4 cycles. Model tracks multiple quantities.
  Page dims needed: ~25

L4.9 (5 steps, easy GSM8K):
  Simple GSM8K problems with known decompositions.
  cycle_targets = [v1, v2, v3, v4, answer]
  5 cycles. Full use of the breathing loop.
  Page dims needed: ~35

L5 (GSM8K):
  Full GSM8K.
  cycle_targets = [v1, v2, ..., v_n, answer]
  Variable cycles (3-8). Confidence head decides when to stop.
  Page dims needed: ~40-50
```

### Why This Solves the 10/64 Dimension Problem

At L3, the model needs 5 dims to encode one answer. Fine — 5/64 used.

At L4, the model needs to encode TWO intermediates. It can't reuse the same 5 dims for both (the residual gate preserves cycle 1's encoding while cycle 2 adds its own). It MUST use additional dims. Maybe 10/64.

At L4.7, four intermediates. 20/64 dims.

At L5, 5-8 intermediates. 40-50/64 dims.

The curriculum FORCES dimensional expansion. Each level demands more page bandwidth. The perceiver learns to use more dimensions because the per-cycle targets REQUIRE it.

```
Per-dim variance diagnostic MUST show:
  After L3:    5-10 dims active
  After L4:    10-15 dims active
  After L4.5:  15-20 dims active
  After L4.7:  20-30 dims active
  After L5:    30-50 dims active

If variance doesn't spread → add per-dim variance regularizer
If variance spreads → curriculum is working as designed
```

---

## The Deeper Insight: Decomposition IS Pattern Matching

Per-cycle targets aren't just about arithmetic steps. They're about PATTERN RECOGNITION ORDER. Each cycle matches one pattern from the model's library and reports the result.

"Half as many clips in May" is not five separate words to parse — it's ONE pattern: "half as many [X] in [time] → X_new = X_previous / 2". The model should recognize the WHOLE PHRASE as a unit, like "Panama hat" is one concept, not "Panama" + "hat".

```
Over-decomposed (too many cycles, too small patterns):
  Cycle 1: "she" → subject
  Cycle 2: "sold" → verb
  Cycle 3: "half" → divide by 2
  Cycle 4: "as many" → reference previous
  Six cycles for one operation. Wasteful.

Right-sized (one pattern per cycle):
  Cycle 1: "sold [X] clips in April" → v1 = 48 (one pattern, one cycle)
  Cycle 2: "half as many in May" → v2 = v1 / 2 (one pattern, one cycle)
  Cycle 3: "how many total" → answer = v1 + v2 (one pattern, one cycle)
  Three cycles. Each matches the LARGEST pattern that fits.
```

### The 64 Atoms ARE the Pattern Library

The 64 atoms aren't matched one by one. They're BLENDED. The hypernetwork outputs 64 independent scales and the attention modification is the weighted sum of all active atoms:

```
Cycle 1 scales: [0.8, 0.0, 0.3, -0.2, 0.0, 0.9, 0.0, ...]

LoRA_output = 0.8 * atom_0 + 0.3 * atom_2 + (-0.2) * atom_3 + 0.9 * atom_5 + ...
```

This is like color blending. Red + blue = purple. Atom 0 + atom 5 = a COMPOSITE attention pattern that neither atom represents alone:

```
Atom 0 alone:    attends to sentence subjects ("Natalia", "she")
Atom 5 alone:    attends to numbers ("48", "24")  
Atom 0 + 5:      attends to subject-number relationships ("Natalia sold 48")
                  A composite pattern from blending two atoms
```

The pattern library isn't 64 discrete patterns. It's a CONTINUOUS 64-dimensional space. Each point is a unique blend of atoms producing a unique attention modification. The hypernetwork navigates this space, finding the point that recognizes the right pattern for each cycle.

```
64 atoms with continuous scales [-1.0 to +1.0]:
  The space of possible attention patterns is infinite
  Every blend is a unique pattern recognizer
  The hypernetwork learns to navigate to the right blend per cycle
  
Cycle 1: navigate to point [0.8, 0.0, 0.3, ...] → recognizes "quantity assignment"
Cycle 2: navigate to point [0.1, 0.7, 0.0, ...] → recognizes "fractional relationship"
Cycle 3: navigate to point [0.0, 0.2, 0.0, ...] → recognizes "sum query"
```

### Fourier Atoms Enable Multi-Scale Patterns

The Fourier initialization gives atoms different SCALES:

```
Low-frequency atoms (0-15):   recognize LARGE patterns
  "half as many in May" as ONE unit
  "X percent more than Y" as ONE unit
  "bought N at $M each" as ONE unit

High-frequency atoms (48-63): recognize SMALL patterns
  "48" → a number
  "half" → divide by 2
  "total" → sum
```

Easy problems activate mostly low-frequency atoms — one large pattern per cycle, few cycles needed. Hard problems with unfamiliar structure activate more high-frequency atoms — smaller patterns, more cycles needed. The decomposition granularity adapts to the problem.

### The Curriculum Builds the Pattern Library

Each stepping stone adds patterns to the atom library:

```
L3:   learns single-operation patterns
      "X gave away Y" → subtraction
      "X got Y more" → addition

L4:   learns two-step composition patterns
      "X gave away Y then got Z" → subtract then add

L4.5: learns multiplicative relationship patterns
      "half of X" → X/2
      "three times as many" → X*3

L4.7: learns percentage and rate patterns
      "X% of Y" → Y * X/100
      "at $Y each" → N * Y

L5:   COMBINES patterns from all levels
      Recognizes large composite patterns from the library
      Falls back to smaller patterns when composite doesn't match
```

The per-cycle targets teach which pattern to match at each cycle. The Fourier atoms provide the pattern templates. The curriculum grows the library. The hypernetwork learns to navigate to the right blend.

### Phase 2: Self-Guided Decomposition (Future)

In Phase 2, when we remove teacher targets, the model uses its learned pattern library to decompose on its own:

```
Phase 1: We say "cycle 1 extracts 48, cycle 2 computes 24"
Phase 2: The model says "I recognize two patterns here. Two cycles."
         It matches the largest patterns it knows and allocates one cycle each.
         
Small model (1B):   small patterns, many cycles (fine decomposition)
Large model (8B):   large patterns, few cycles (coarse decomposition)
Same architecture, same training, different decomposition — adapted to capacity.
```

The number of cycles = number of patterns needed. The confidence head says "I've matched all the patterns I can find — stop." The decomposition granularity emerges from the model's pattern library size, which emerges from its capacity, which emerges from its parameter count.

---

## How the Thinking Loop Finally Works

```
Problem: "Natalia sold 48 clips in April, half in May, total?"

Cycle 1:
  Atoms: low-frequency activated (broad parsing)
  Llama: reads problem, focuses on "48 friends"
  Perceiver: compresses → page 1
  Answer head: reads page 1 → predicts 48 ✓
  Page 1 encodes: [48 in dims 5-10, problem type in dims 0-4]

Cycle 2:
  Hypernetwork: reads page 1 (knows 48 was extracted)
  Atoms: mid-frequency activated (computation focus)
  Llama: re-reads, focuses on "half as many"
  Perceiver: compresses → page 2 (with residual from page 1)
  Answer head: reads page 2 → predicts 24 ✓
  Page 2 encodes: [48 preserved in dims 5-10, 24 in dims 15-20]

Cycle 3:
  Hypernetwork: reads pages 1-2 (knows 48 and 24)
  Atoms: different activation (total computation)
  Llama: re-reads, focuses on "how many total"
  Perceiver: compresses → page 3 (with residual from pages 1-2)
  Answer head: reads page 3 → predicts 72 ✓
  Page 3 encodes: [48 in dims 5-10, 24 in dims 15-20, 72 in dims 25-30]

Confidence: high → stop
Final answer: 72 (from answer head on page 3, no generation needed)
```

Each cycle:
1. Reads the problem with DIFFERENT attention (atoms change each cycle)
2. Compresses understanding into the page (perceiver)
3. Preserves previous results (residual gate)
4. Predicts THIS cycle's intermediate (answer head with per-cycle target)
5. The hypernetwork reads all pages to decide next cycle's atom config

The cycles COORDINATE because the hypernetwork reads accumulated pages. "I see 48 in page 1 and the problem says 'half' — cycle 2 should focus on division." The pages are the communication channel between cycles.

---

## What We Keep vs What We Drop

```
KEEP (proven, essential):
  ✓ 64 rank-6 LoRA atoms with Fourier init (45/64 active)
  ✓ 10M hypernetwork with skip_pass_embed (reads pages, no shortcut)
  ✓ 105M perceiver with page communication (sees previous pages)
  ✓ Residual gate (information persists across cycles)
  ✓ Skip connections (gradient highway)
  ✓ Hard pre-tanh clamp (prevents saturation)
  ✓ Contrastive loss (prevents page collapse)
  ✓ Answer head (sign + length + per-digit prediction)
  ✓ Confidence head (when to stop)
  ✓ Pi-harmonic page encoding
  ✓ Haar wavelet preprocessing
  ✓ Fresh data per epoch
  ✓ Per-component learning rates

DROP (not needed with per-cycle targets):
  ✗ CoT generation loss (replaced by per-cycle answer head loss)
  ✗ Text generation during training (no generation at all)
  ✗ SymPy decoder (premature — revisit after pages use 40+ dims)
  ✗ Pattern memory (premature — revisit after GSM8K >25%)
  ✗ Generation prefix "The answer is" (no generation)

PARK (for later):
  ○ SymPy decoder (add once pages are rich, curriculum complete)
  ○ Pattern memory (add once GSM8K >25%)
  ○ Binary tree search (inference-time, after training works)
  ○ MCMC exploration (inference-time)
  ○ Page cache + replay buffer (training speedup for later epochs)
  ○ Adaptive data weighting (after basic training works)
  ○ Unfreezing Llama (if parsing becomes the bottleneck)
```

---

## Implementation Order

```
Tomorrow:
  1. Generate per-cycle decompositions for L3 (trivial — 1 step)
  2. Modify training loop: per-cycle answer head loss, no generation loss
  3. Train L3 with per-cycle targets (should hit 90%+ in 1-2 epochs)
  4. Verify: per-dim variance spreading? More than 10 dims active?

This week:
  5. Generate per-cycle decompositions for L4 (2 steps)
  6. Train L4 with warm start from L3
  7. Verify: per-dim variance spreading to ~15 dims?
  8. Generate L4.5, L4.7, L4.9 decompositions
  9. Train through stepping stones, checking variance at each level

Next week:
  10. Generate GSM8K per-cycle decompositions (use Claude/GPT-4)
  11. Train L5 with full per-cycle targets
  12. Target: >25% on GSM8K with per-cycle training
  13. Run per-dim variance diagnostic: using 30+ dims?

Later:
  14. Add SymPy decoder (pages now rich enough to read)
  15. Add pattern memory
  16. MATH-500 evaluation
```

---

## What to Monitor

```
PER-CYCLE ACCURACY (new, most important):
  After cycle 1: does answer head predict the first intermediate correctly?
  After cycle 2: does it predict the second intermediate?
  After cycle 3: does it predict the third?
  
  Per-cycle accuracy tells us EXACTLY which step the model struggles with.
  If cycle 1 is 90% but cycle 3 is 20%, the model extracts numbers
  but can't chain computations.

PER-DIM VARIANCE (must spread with curriculum):
  After L3: 5-10 dims active
  After L4: 10-15 dims active
  After L4.5: 15-20 dims active
  After L5: 30-50 dims active
  
  If variance doesn't spread: add regularizer.
  If it spreads: curriculum is working.

EXISTING METRICS (still track):
  page_cos: should drop as more dims are used
  active_atoms: should stay high (40+) with Fourier init
  Jacobian spectral radius: should stay near 1.0
  scale_reg: should stay low (<1.0) with hard clamp
```

---

## The Key Insight

The model's problem was never arithmetic — Llama can compute 48/2=24 just fine. The problem was CHAINING: carrying intermediate results across cycles with only 10/64 page dimensions active.

Per-cycle intermediate targets force each cycle to encode a SPECIFIC result. The perceiver must use DIFFERENT dimensions for DIFFERENT intermediates. The curriculum gradually increases the number of steps (and thus the number of dimensions needed).

The breathing loop was designed for incremental thinking. CoT targets undermined this by asking for everything in one shot. Per-cycle targets align the training objective with the architectural design: one bite per cycle, one intermediate per page, one step at a time.

The model breathes. Each breath takes in one piece. The notebook fills with intermediates. The answer falls out of the last page.
