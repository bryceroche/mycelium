# Mycelium v16 — Breathing Models

A vanilla transformer (SmolLM2-135M) trained from birth to reason in expand-collapse cycles, guided by a learned energy landscape. The model solves problems beyond its single-shot capacity by taking small, verified steps downhill through the landscape.

> Good expansion is expansion that's easy to collapse.
> Good collapse is collapse that sets up the next expansion.

---

Only train on the AWS VM — PEM key in ~/.ssh/mycelium-key.pem
Limit timeouts to less than 2 minutes
Always use vLLM + stage-based batching when possible, otherwise HF Transformers
Temperature = 0 (greedy) for all evals
MATH-500 NEVER in training data

---

## Current State (March 28, 2026)

```
Architecture:        SmolLM2-135M base (vanilla transformer, no modifications)
Context window:      8,192 tokens
Approach:            Continued pretraining on breath-structured data
Status:              Building training data pipeline

Core principle:      The model is born breathing. Not bolted on, not fine-tuned onto it.
                     The expand-collapse rhythm is the model's native mode of thought.

Previous approach:   Qwen-0.5B + external bottleneck module (v10)
Why we moved on:     Every attempt to bolt breathing onto a single-shot model
                     either matched or hurt vanilla. The model must be born breathing.
```

---

## The Breathing Cycle

Each breath is one forward pass with bounded context:

```
[PROBLEM] <problem statement>
[STATE] <last collapsed state, or "none" for first breath>
[EXPAND] <model generates reasoning — HIGH entropy, creative, exploratory>
[COLLAPSE] <model generates compressed state — LOW entropy, verified, precise>
```

Critical constraints:
- Model sees ONLY [PROBLEM] + last [COLLAPSE] — full history is never available
- The collapse IS the memory — lossy, forcing the model to decide what matters
- Expand budget: ~7,000 tokens (most of context window)
- Collapse budget: ~100–200 tokens (~35–70X compression)
- Max breaths per problem: 20 (upper bound, most problems need 6–10)
- Termination: model generates \boxed{} when confident, or stops at max breaths

The expand phase is high entropy — the model explores, considers, reasons freely.
The collapse phase is low entropy — the model commits to verified mathematical statements.

---

## The Energy Landscape

The energy landscape is a topological surface over mathematical reasoning states. It tells the model where it is and which direction is downhill (toward the answer).

### Learned Potentials (MLPs)

**E_node — "Is this step correct?"**
- Scores the quality of each mathematical joint (each [COLLAPSE] output)
- Trained on hidden states / text features from verified solutions
- High quality joint = low energy. Malformed/wrong = high energy.
- Current: 82% accuracy on Qwen-0.5B hidden states
- TODO: Retrain on text-based features (SymPy expressions) for model portability

**E_edge — "How difficult is this gap?"**
- Scores the difficulty of the transition between two joints
- Easy transition = low energy. Hard leap = high energy.
- Current: 96.5% AUC on Qwen-0.5B features
- TODO: Retrain on text-based features for model portability

**Wave Coherence Reward — "What's the shape of the whole trajectory?"**
- Amplitude uniformity (no single step dramatically harder than others)
- Frequency regularity (evenly spaced joints)
- Joint quality, progress, ordering
- Captures global topology of the solution path

Together: E_node = elevation at each point. E_edge = slope between points. Wave coherence = shape of the whole trail.

### What the Energy Landscape Enforces

The ONLY training signal is: **move downhill in the energy landscape, within tight step-size bands.**

```
reward = signed_progress  (energy_prev - energy_curr, positive = downhill)

constraints:
  - Step must be DOWNHILL (toward lower energy / correct answer)
  - Step must be within TIGHT BANDS (not too big, not too small)
  - Steps too large → penalty (model bit off more than it can chew)
  - Steps too small → penalty (model is stalling / not making progress)
  - Steps uphill → penalty (model went in wrong direction)
```

We do NOT separately reward preservation. If the model drops critical state from its collapse, the next expansion will flounder, producing an uphill step — the energy landscape catches it.

### Elegance Metric

```
elegance = min(total_energy_distance) across multiple solutions to same problem
```

If we solve a problem several times, the most elegant solution is the one that traveled the shortest total distance through the energy landscape. Fewer steps, more direct, less wandering.

### Alternating Energy Levels

The energy landscape captures order of operations through alternating levels:
- Expand phases INCREASE local entropy (exploring possibilities)
- Collapse phases DECREASE local entropy (committing to verified math)
- The alternation is the breathing rhythm itself
- A correct solution descends the global energy while locally oscillating expand/collapse

---

## IB Operation Types

Information Bottleneck analysis discovered 8 natural operation types from teacher attention patterns (β=50, NMI=0.740, 85%+ purity):

### 3 Expand Types (MCTS candidates)
```
These are the creative/exploratory operations where branching helps:
- SETUP: Problem interpretation and variable identification
- SOLVE: Core mathematical manipulation
- SUBSTITUTE: Value insertion and simplification
```

MCTS on expand types: at each expand step, generate multiple candidate expansions, evaluate each with E_edge (which path leads to easier collapse?), select best. The tree search explores the expand space where creativity matters.

### 5 Collapse Types (Deterministic)
```
These are mechanical/verification operations — no branching needed:
- COMPUTE: Arithmetic evaluation
- SIMPLIFY: Expression reduction
- THEOREM: Known result application
- ANSWER: Final answer formatting
- VERIFY: SymPy equivalence check
```

No MCTS on collapse types. Collapse is deterministic — compress the expand output into a verified joint. SymPy handles verification. Branching here wastes compute.

### IB Types in the Energy Landscape

The operation type determines the expected step characteristics:
- SETUP steps cross fewer contour lines (establishing, not solving)
- SOLVE steps cross the most contour lines (core mathematical progress)
- COMPUTE/SIMPLIFY steps are short, precise descents
- The energy landscape bands adjust per operation type

---

## Training Data

### Collection
```
1. Collect correct solutions from Qwen-7B (or stronger) on:
   - GSM8K (~7.5K problems, grade school math)
   - MATH train (~7.5K problems, competition math)
   Generate 4–8 solutions per problem, keep only SymPy-verified correct

2. Segment at natural joints using 99.5% joint detection pipeline
   Each joint = LaTeX expression SymPy can parse

3. Format as breath sequences
   Each segment between joints → one expand-collapse cycle

4. Score and filter by step-size bands
   Reject solutions where any step crosses too many contour lines
   Keep most elegant solutions (min total energy distance)
```

### Format
Each breath is a SEPARATE training example. The model never sees the full solution:

```
# Breath 1:
[PROBLEM] Solve for x: x² - 5x + 6 = 0
[STATE] none
[EXPAND] I need to factor this quadratic. Looking for two numbers
  that multiply to 6 and add to -5. Those would be -2 and -3.
[COLLAPSE] x² - 5x + 6 = (x-2)(x-3) = 0

# Breath 2 (sees ONLY the collapse from breath 1):
[PROBLEM] Solve for x: x² - 5x + 6 = 0
[STATE] x² - 5x + 6 = (x-2)(x-3) = 0
[EXPAND] Setting each factor to zero: x-2=0 gives x=2, x-3=0 gives x=3.
[COLLAPSE] x = 2 or x = 3. \boxed{x \in \{2, 3\}}
```

### Data Purity
Strong model generates solutions single-shot, but the training model never sees the full solution — only individual breath cycles. The mathematical steps are valid regardless of how they were generated. What matters is the structural format: one expand, one collapse, bounded context.

---

## Three-Phase Training

Everything ramps smoothly. No sudden changes. This is the deepest lesson from all prior work.

### Phase 1: Imitation — "This is what breathing looks like"
```
Objective:  Continued pretraining (next-token prediction) on breath-structured data
States:     Ground truth (from strong model solutions)
Data:       GSM8K easy → GSM8K full → MATH easy
Collapse:   Generous budget (~200 tokens), gradually tighten
Duration:   Until model produces valid [EXPAND]/[COLLAPSE] format, >90% parse rate

Limitation: Model never faces consequences of its own bad collapses.
            Learning from perfect states, not its own.
```

### Phase 2: Self-Play — "This is what YOUR breathing feels like"
```
Objective:  Close distribution gap between training and inference
Method:     DAgger-style — model reads its own collapses
Mix:        80% teacher states → 50/50 → 100% self-generated (gradual)
Data:       GSM8K + MATH L1-L3
Duration:   Until model functions with self-generated states, <10% degradation

Key moment: Model faces consequences of bad collapses for the first time.
            Telephone-resistant compression emerges here.
```

### Phase 3: Optimization — "This is how to breathe WELL"
```
Objective:  GRPO with energy landscape reward
Training:   Unrolled breath chains — model generates full solutions
Reward:     Signed energy progress per breath, within step-size bands
Data:       Full MATH spectrum
MCTS:       On expand types (SETUP, SOLVE, SUBSTITUTE) — branch and select
Duration:   Until breathing accuracy > single-shot on previously unsolvable problems

This is where min(max(step_distance)) emerges from training.
```

### Difficulty Curriculum (Smooth Ramp)
```
Week 1:     Trivial chains (counting, variable tracking) — learn format
Week 1-2:   Arithmetic, linear equations (GSM8K easy) — learn compression
Week 2-3:   Full GSM8K, easy MATH — learn state management
Week 3-4:   MATH L1-L3 — learn real mathematical breathing
Week 4+:    Full MATH spectrum — capacity-aware decomposition emerges
```

---

## Architecture Details

### Model
```
SmolLM2-135M (HuggingFace, BASE, not instruct)
  Parameters:     135M
  Context window:  8,192 tokens
  Hidden dim:      576
  Layers:          30
  Attention heads: 9

Why base (not instruct): No ingrained single-shot habits.
  The breathing pattern is the first problem-solving strategy it learns.
  
Why 135M: Small enough that single-shot capacity ceiling is genuinely low.
  The "breathing extends capacity" result is undeniable at this scale.
```

### Token Budget Per Breath
```
[PROBLEM]   ~100-500 tokens (varies by problem)
[STATE]     ~100-200 tokens (last collapse, the lossy pointer)
[EXPAND]    remainder of context (~7,000 tokens max)
[COLLAPSE]  ~100-200 tokens (compressed state for next breath)

Compression ratio: ~35-70X (expand → collapse)
The ratio is adaptive per problem — model learns its own compression level.
```

### No Additional Parameters
```
Zero architectural modifications. Zero bottleneck modules.
The compression is what the model PRODUCES at [COLLAPSE].
The model itself is the compressor.
The token budget is the bottleneck.
```

---

## Theoretical Framework

### Keyframes and Interframes
- [EXPAND] phases are KEYFRAMES — full resolution, creative, exploratory
- [COLLAPSE] phases are INTERFRAMES — compressed, derivative, lossy pointers
- Like video compression: keyframes carry full information, interframes encode deltas
- The collapse is a low-resolution pointer into the energy landscape
  "I'm approximately HERE" — enough to orient, not enough to reconstruct

### Rate-Distortion Codec
- The breathing model is a rate-distortion codec for mathematical reasoning
- Rate = collapse token budget (the channel capacity)
- Distortion = information loss at each compression (how lossy the pointer is)
- The model learns the optimal rate-distortion tradeoff for its capacity
- Smaller models need higher rates (more tokens per collapse) = less compression
- Larger models achieve lower rates = more aggressive compression

### Capacity-Granularity Relationship
- The teacher (7B) processes ~16 micro-operations per breath
- The student (135M) can't hold all 16 simultaneously
- It externalizes working memory through multiple breaths
- Step size scales with model capacity — publishable scaling result
- Train at 50M, 135M, 360M, 1.7B → measure emergent step size at each scale

### π-Normalization
- Model-agnostic attention thresholds for joint detection
- Normalizes across different model scales and architectures
- Ensures joint detection pipeline works identically on SmolLM2 as on Qwen

### Laplace Spectral Fingerprint
- Eigenvalues of the graph Laplacian on the dependency graph
- Captures topological structure of the solution path
- Can be used to condition the energy landscape — similar solution structures
  should have similar energy profiles regardless of specific mathematical content
- Open thread: needs rework for text-based features

---

## Speculative Directions (Flagged Honestly)

These ideas came up in brainstorming. Some may be deep, some may be noise.
Each needs rigorous testing before committing.

### Diffusion Model for Low-Resolution Pointers
- The collapse is a low-res pointer. Could a diffusion model "super-resolve" it?
- Denoise the lossy collapse into a richer state for the next expansion
- Risk: adds complexity to an aggressively simple architecture
- Test: does denoised state lead to better next-expansion quality?

### Energy Resonance
- Do correct solutions exhibit resonant patterns in the energy landscape?
- Frequency of expand/collapse oscillation might correlate with problem difficulty
- Natural frequency = the problem's "breathing rate"
- If the model matches this frequency, it's in resonance = optimal stepping

### Reflection Step
- After N breaths, pause and generate a [REFLECT] phase
- Model reviews its collapsed state and asks: "Am I on track?"
- Could catch systematic drift that per-step energy doesn't detect
- Where in the loop? Every 5 breaths? When energy progress stalls?
- Needs experimentation. Gut says this matters but unproven.

### Bidirectional Breathing
- Forward: expand from problem toward answer
- Backward: expand from answer toward problem (when answer structure is known)
- Meet in the middle — reduces total breaths needed
- Only works when answer form is constrained (e.g., "find the integer")
- May connect to bidirectional search in MCTS

---

## Infrastructure

### Models on S3
```
REUSE (retrain on text features for SmolLM2):
s3://mycelium-data-v7/models/e_edge_v1/                    # Gap difficulty (96.5% AUC)
s3://mycelium-data-v7/models/e_node_base_v1/                # Joint quality (82%)
s3://mycelium-data-v7/models/grpo_lora_v1/                  # Wave coherence LoRA (reference)

FROZEN (read-only reference):
s3://mycelium-data/models/c1a_coarse_v6_aux_telegraph/      # Boundary detection F1=0.741
s3://mycelium-data/models/c1b_sequence_v5/                  # BP depth prediction
s3://mycelium-data/ib_cluster_to_type.json                  # 8 IB type mapping (3 expand + 5 collapse)

DON'T USE:
s3://mycelium-data-v7/models/cycle*_specialists/            # LoRA specialists (suppress reasoning)
```

### Data on S3
```
Training data:
s3://mycelium-data-v7/training_data/sequential_v1/         # 4,280 train / 475 val
s3://mycelium-data-v7/grpo_lite/                            # 99,968 scored candidates
s3://mycelium-data-v7/self_improvement/                     # Verified traces (42K steps)

Analysis:
s3://mycelium-data/iaf_density_analysis/                    # IAF micro-transition data
                                                            # (117 chunks × 200MB, 3GB memory)
                                                            # Contains 6-10 steps per problem finding
```

### Key Scripts (from v10, adapt for v16)
```
src/wave_reward.py                  # Wave coherence reward function
src/pipeline_v8.py                  # Joint detection + dependency graphs
src/oracle.py                       # SymPy verification
scripts/robust_latex_parser.py      # LaTeX→SymPy with edge case handling
scripts/train_grpo_lora.py          # GRPO training loop (basis for Phase 3)
scripts/generate_grpo_candidates.py # Candidate generation with vLLM
scripts/eval_math500_grpo.py        # MATH-500 evaluation
```

### New Scripts (v16)
```
src/
  energy_landscape.py      # E_node + E_edge on text features (retrained MLPs)
  breath_format.py         # Breath sequence formatting and parsing
  step_bands.py            # Step-size band enforcement and scoring
  mcts_expand.py           # MCTS on expand types (SETUP, SOLVE, SUBSTITUTE)

scripts/
  collect_solutions.py     # Generate solutions from strong model
  segment_breaths.py       # Joint detection → breath sequence formatting
  train_phase1.py          # Phase 1: imitation (continued pretraining)
  train_phase2.py          # Phase 2: self-play (DAgger)
  train_phase3.py          # Phase 3: optimization (GRPO + energy landscape)
  eval_breathing.py        # Breathing inference + evaluation
  eval_single_shot.py      # Single-shot baseline for comparison
  measure_step_size.py     # Emergent step-size analysis across model scales
```

### VMs
```
AWS EC2 g5.xlarge (~$1/hr, A10G 24GB) — primary training
AWS EC2 g5.48xlarge (~$16/hr) — large batch generation if needed
SSH: ssh -i ~/.ssh/mycelium-key.pem ubuntu@<IP>
```

---

## Bug List (Carried Forward + New)

```
CARRIED FROM v10:
27. max_tokens=512 truncates before \boxed{}. Use 2048+.
28. parse_latex() on non-LaTeX corrupts. Only call if contains backslash.
29. Implicit mult breaks function names. Protect reserved words.
30. Verification bottleneck: 29b ≠ 29*b to naive parser.
31. LoRA specialists SUPPRESS reasoning. Don't use.
34. Sudden bottleneck → collapse (no gradient signal). Fix: gradual training.

NEW FOR v16:
35. Temperature must be 0 (greedy) for eval. 0.7 caused 8→24% baseline shift.
36. vLLM process isolation kills global state. Run bottleneck BETWEEN calls, not inside.
37. Small sample sizes lie. N=20 → 35%, N=100 → 26% on same data. Always N≥100.
38. Always verify baseline on SAME problem set before comparing.
39. E_node/E_edge trained on Qwen hidden states — must retrain on text features for SmolLM2.
40. DataCollatorForLanguageModeling overwrites label masking (bugs 26/27 from v7).
```

---

## Milestones

```
M0: Training data pipeline
    Collect solutions, segment at joints, format as breath sequences
    Validate: ~20K+ breath-structured training examples
    Validate: step-size distribution matches expected 6-10 steps/problem

M1: SmolLM2-135M single-shot baseline
    Evaluate on GSM8K and MATH-500 (all difficulty levels)
    Validate: establish capacity ceiling per difficulty level
    Expect: very low on MATH L4-L5 (this is the bar breathing must clear)

M2: Phase 1 — Imitation
    Continued pretraining on breath-structured data
    Validate: >90% format compliance ([EXPAND]/[COLLAPSE] tags)
    Validate: >80% SymPy parse rate on [COLLAPSE] outputs
    Validate: model can chain 3+ breaths without format degradation

M3: Phase 2 — Self-Play
    DAgger with self-generated states
    Validate: <10% accuracy degradation vs teacher states
    Validate: model survives 8+ breaths of its own collapses (telephone test)

M4: Phase 3 — Optimization
    GRPO with energy landscape reward + step-size bands
    Validate: breathing accuracy > single-shot on at least one difficulty tier
    Validate: step-size distribution shows tight bands (min(max(step_distance)))

M5: THE RESULT — Capacity Extension
    Evaluate on MATH L4-L5 problems where single-shot gets 0%
    Validate: non-zero accuracy through breathing
    THIS IS THE HEADLINE: small model solves previously unsolvable problems

M6: Scaling Analysis
    Train at 50M, 135M, 360M, 1.7B
    Measure: emergent step size, compression ratio, capacity extension per scale
    Validate: clean scaling relationship between params and step granularity
```

---

## Critical Rules

```
 1. SmolLM2-135M BASE (not instruct) — breathing is the first strategy it learns
 2. Model sees ONLY [PROBLEM] + last [COLLAPSE] — never full history
 3. Energy landscape is the ONLY training signal — no separate preservation reward
 4. Signed progress: downhill = good, uphill = bad, stalling = bad
 5. Step-size bands: tight, not too big, not too small
 6. MCTS on expand types (SETUP, SOLVE, SUBSTITUTE) ONLY — collapse is deterministic
 7. Smooth ramp in training difficulty — no sudden changes, ever
 8. Phase transitions are gradual (mix ratios, not switches)
 9. Temperature = 0 for all evaluations
10. N ≥ 100 problems for any accuracy measurement
11. Always verify baseline on same test set before comparing
12. vLLM + stage batching for generation, HF Transformers for training
13. MATH-500 NEVER in training data
14. Save checkpoints at each phase transition for analysis
15. Handoff doc before closing any session
```

---

## Guardrails

```
1. Log energy landscape distance per breath during training
2. If loss spikes >2x, pause difficulty ramp for 2 epochs
3. If collapse parse rate drops below 50%, increase collapse budget
4. If telephone degradation >20% per breath, stay in Phase 2 longer
5. Keep single-shot baseline for comparison at every checkpoint
6. Validate on 10 examples before scaling to 500
7. One clean measurement is worth more than 10 fast experiments
```

---

## MATH-500 Deadline: April 22, 2026

Canonical evaluation: HuggingFaceH4/MATH-500 only.
Target: breathing accuracy > single-shot on L4-L5 problems.
The undeniable result: solving previously unsolvable problems.
