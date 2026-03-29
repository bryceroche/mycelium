# Mycelium v16 — Breathing Models

> **A small model that cannot solve hard problems in a single pass can solve them
> by learning to decompose into steps that fit within its capacity.**
>
> The breathing rhythm is not bolted on — it is the model's native mode of thought.

---

## What Mycelium Does

Mycelium trains a vanilla transformer (SmolLM2-135M) from birth to reason in **expand-collapse cycles**, guided by a **learned energy landscape**. The model takes small, verified steps downhill through the landscape, each step within tight step-size bands. It externalizes working memory through compression — each collapse is a lossy pointer that says "I'm approximately here," enough to orient the next expansion.

The model never sees its full solution history. It sees only the problem and its last collapsed state. The collapse IS the memory.

---

## Why Breathing

Current language models reason in a single continuous pass. For hard problems, the entire reasoning chain must fit in the model's working memory simultaneously. Small models hit a capacity ceiling — not because they lack mathematical knowledge, but because they can't hold enough intermediate state.

Breathing solves this through cognitive offloading. Like a human doing long multiplication on paper instead of in their head, the model writes down intermediate results (collapses), then reads them back to continue (next expansion). The paper doesn't make you smarter. It lets you solve problems that exceed your working memory.

---

## The Expand-Collapse Cycle

```
[PROBLEM]  Solve for x: x² - 5x + 6 = 0
[STATE]    none
[EXPAND]   I need to factor this quadratic. Looking for two numbers
           that multiply to 6 and add to -5. Those are -2 and -3.
[COLLAPSE] x² - 5x + 6 = (x-2)(x-3) = 0

           ↓ collapse becomes next state (everything else is discarded)

[PROBLEM]  Solve for x: x² - 5x + 6 = 0
[STATE]    x² - 5x + 6 = (x-2)(x-3) = 0
[EXPAND]   Setting each factor to zero: x-2=0 gives x=2, x-3=0 gives x=3.
[COLLAPSE] x = 2 or x = 3. \boxed{x \in \{2, 3\}}
```

**Expand** = high entropy. Creative, exploratory, full thinking space (~7,000 tokens).
**Collapse** = low entropy. Compressed, verified, precise (~100–200 tokens, 35–70X compression).

The expand phase is a keyframe — full resolution reasoning.
The collapse phase is an interframe — a low-resolution pointer into the energy landscape.

The model loops until `\boxed{}` or max 20 breaths.

---

## The Energy Landscape

The energy landscape is a topological surface over mathematical reasoning states. High energy = far from solved. Low energy = close to the answer. The model descends this surface one breath at a time.

### Learned Potentials

| Component | What It Measures | Current Performance |
|-----------|-----------------|-------------------|
| **E_node** | Is this step correct? (joint quality) | 82% accuracy |
| **E_edge** | How difficult is this gap? (transition quality) | 96.5% AUC |
| **Wave coherence** | What's the shape of the whole trajectory? | 1.8 vs -0.1 correct/incorrect separation |

These are learned MLPs trained on verified solution trajectories. Together they define the mountain the model descends: E_node = elevation at each point. E_edge = slope between points. Wave coherence = shape of the whole trail.

### The Only Training Signal

The model is rewarded for one thing: **moving downhill within tight step-size bands.**

```
Step is downhill + right size  →  reward
Step is downhill + too big     →  penalty (bit off too much)
Step is downhill + too small   →  penalty (stalling)
Step is uphill                 →  penalty (wrong direction)
Step is flat                   →  penalty (no progress)
```

No separate preservation signal. If the model drops critical state from its collapse, the next expansion fails, the next step goes uphill, and the energy signal catches it. One signal. One landscape. Descend.

### Elegance

When the model solves a problem multiple times, the most elegant solution is the one with the shortest total distance through the energy landscape. Fewest steps, most direct path, least wandering.

```
elegance = min(total_energy_distance_to_answer)
```

---

## IB Operation Types

Information Bottleneck analysis discovered 8 natural operation types from teacher attention patterns:

**3 Expand Types (creative — MCTS candidates):**
- SETUP — problem interpretation, variable identification
- SOLVE — core mathematical manipulation
- SUBSTITUTE — value insertion, simplification

**5 Collapse Types (mechanical — deterministic, no branching):**
- COMPUTE, SIMPLIFY, THEOREM, ANSWER, VERIFY

MCTS explores the expand space where branching matters. Collapse is deterministic — compress and verify via SymPy. Don't waste compute branching on compression.

---

## Architecture

Aggressively simple. Zero novel components. Zero additional parameters.

```
Model:              SmolLM2-135M base (HuggingFace, NOT instruct)
Context window:     8,192 tokens
Hidden dim:         576
Parameters:         135M (all from base model, nothing added)
Modifications:      None. Vanilla transformer.

The breathing rhythm emerges from training data structure and curriculum.
The token budget is the bottleneck. The model itself is the compressor.
```

Why SmolLM2-135M base: small enough that single-shot ceiling is genuinely low. Large enough to learn mathematical reasoning in small steps. Base (not instruct) means no single-shot habits — breathing is the first and only problem-solving strategy it learns.

---

## Training

### Data
Collect correct solutions from Qwen-0.5B (joint detection already validated at 99.5% on this model), segment at natural joints, format as breath sequences. Each breath is a separate training example — the model never sees the full solution. Target: ~20K+ breath-structured examples from GSM8K + MATH train.

### Three Phases (Smooth Transitions)

**Phase 1: Imitation** — "This is what breathing looks like"
Standard continued pretraining on breath-structured data. Ground-truth states. The model learns the format, the tags, what valid collapses look like. Starts with trivial problems, ramps smoothly toward real math.

**Phase 2: Self-Play** — "This is what YOUR breathing feels like"
DAgger-style. The model reads its own collapses. Like the telephone game — the model must learn to write collapses that survive the round trip. Gradual mix: 80% teacher states → 100% self-generated.

**Phase 3: Optimization** — "This is how to breathe WELL"
GRPO with energy landscape reward. Unrolled breath chains. Per-breath credit assignment. MCTS on expand types. min(max(step_distance)) emerges from training, calibrated to the model's own capacity.

### Difficulty Ramp
```
Trivial (counting, tracking) → GSM8K easy → GSM8K full → MATH L1-L3 → Full MATH
```
Each transition is gradual. Mix ratios shift smoothly. No cliffs.

---

## The Undeniable Result

Take problems that SmolLM2-135M gets **0% on single-shot** (MATH Level 4–5). Show **non-zero accuracy with breathing**. The model solves problems it literally cannot solve in one pass, by decomposing them into steps that fit within its capacity.

This is capacity extension, not accuracy improvement.

### Scaling Story
Train the same architecture at 50M, 135M, 360M, 1.7B. Measure emergent step size at each scale. Expect: step size proportional to model capacity. Smaller models take smaller steps. All traverse the same energy landscape. A small model with many breaths can reach the same valley as a large model with few breaths.

---

## What We Tried (v1–v15)

| Approach | Result | Lesson |
|----------|--------|--------|
| LoRA specialists (8 IB types) | 7.7% vs 24% vanilla | Specialists suppress reasoning |
| All real-time interventions | ≤ vanilla | Model resists intervention during generation |
| Contrastive bottleneck | 18% vs 20% baseline | Classified, didn't compress for continuation |
| Distilled bottleneck | 19% = baseline | Near-identity, not compressing yet |
| Hourglass architecture | 0% | Distribution mismatch from layer restructuring |
| Full fine-tune | Destroyed capability | Catastrophic forgetting |
| Gradual external bottleneck (v10) | +2pp untrained | Right direction, but bolting onto single-shot model |
| GRPO-LoRA with wave coherence | +4pp | Proves reward works, but still single-shot model |

**The deepest lesson:** every attempt to retrofit breathing onto a single-shot model either matches or hurts vanilla. The model must be born breathing.

---

## Theoretical Framework

| Concept | Role |
|---------|------|
| **Rate-distortion codec** | Breathing = lossy compression at each collapse. Rate = collapse budget. Distortion = information loss. Model learns optimal tradeoff for its capacity. |
| **Keyframes/interframes** | Expands are keyframes (full resolution). Collapses are interframes (compressed pointers). Like video compression. |
| **Capacity-granularity** | Larger models process ~16 micro-ops per breath. Student (135M) can't hold all 16. Externalizes through multiple breaths. Step size scales with capacity. |
| **π-normalization** | Model-agnostic attention thresholds for joint detection. Works across scales. |
| **Bidirectional breathing** | Forward: problem→answer. Backward: answer→problem. Both descend same energy landscape. Reflection = geometric transformation, not metacognition. |
| **Alternating energy** | Expand increases local entropy, collapse decreases it. Correct solutions descend globally while oscillating locally. |

---

## Quick Start

```bash
# Establish single-shot baseline
python scripts/eval_single_shot.py --model SmolLM2-135M --dataset math500

# Generate breath-structured training data
python scripts/collect_solutions.py --model Qwen-0.5B --dataset gsm8k
python scripts/segment_breaths.py --input solutions/ --output breaths/

# Phase 1: Imitation training
python scripts/train_phase1.py --data breaths/ --epochs 20

# Evaluate breathing
python scripts/eval_breathing.py --checkpoint phase1_final.pt --dataset math500

# Compare
python scripts/eval_breathing.py --compare-single-shot
```

---

## Hardware

```
Training:    SmolLM2-135M fits easily on A10G (24GB) or even T4 (16GB)
Generation:  vLLM + stage batching for fast rollouts
Time:        Phase 1 ~4-8 hours, Phase 2 ~8-16 hours, Phase 3 ~24-48 hours
Cost:        ~$50-100 on AWS spot for full training pipeline
```

---

## Deadline

**MATH-500 benchmark: April 22, 2026**

Target: breathing accuracy > single-shot on Level 4–5 problems.

---

> *Good expansion is expansion that's easy to collapse.*
> *Good collapse is collapse that sets up the next expansion.*
> *The model descends the energy landscape one breath at a time.*
> *Small steps. Tight bands. Always downhill.*
