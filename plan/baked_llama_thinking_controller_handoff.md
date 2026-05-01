# Handoff: Baked Llama + Thinking Controller

**Author:** Bryce + Claude (brainstorm session)
**Date:** April 30, 2026
**Status:** Ready for implementation
**Priority:** CRITICAL — radical simplification based on two days of evidence
**Replaces:** All previous atom/scale/controller architectures

---

## The Insight

Two days of evidence proved: **there is one dominant basin for atom scales.** Every architecture where the controller produces continuous scales collapses to the same universal blend. The controller can't differentiate in scale space because scale space has no per-problem structure.

**The radical simplification:** Bake the universal blend into Llama permanently. Remove atoms, scales, profiles, and the ST gradient entirely. The controller's job isn't to adjust Llama — it's to think.

---

## Architecture

### Two Components. That's It.

```
1. FROZEN MATH-MODE LLAMA (1.2B, never changes at runtime)
   - Base Llama 3.2 1B weights
   - + L1 LoRA baked in (arithmetic mode, proven at 99.5%)
   - + L2 LoRA baked in (universal math blend from best v1/v2 checkpoint)
   - Reads problems, generates equations
   - KV cache valid for entire problem lifetime

2. TRAINABLE THINKING CONTROLLER (~350M, the only thing that learns)
   - Reads Llama's hidden states
   - Iterates on its own pages (inner loop = thinking)
   - Makes discrete decisions (decompose/solve/merge, stop/continue)
   - Drives tree structure
   - Manages notebook memory across cycles
```

No atoms. No scales. No profiles. No ST gradient. No monkey-patching. No LoRA managers.

---

## Baking L2 Into Llama (One-Time Setup)

```python
# Run once. Llama becomes a permanent math engine.

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Bake L1 (arithmetic mode — already done)
l1_checkpoint = torch.load("checkpoints/l45_best_atoms.pt")
for name, param in model.named_parameters():
    if is_lora_target(name):
        A = l1_checkpoint[f"{name}.lora_A"]
        B = l1_checkpoint[f"{name}.lora_B"]
        param.data += 0.46 * (A @ B)

# Bake L2 (universal math blend — best v1/v2 scales)
l2_checkpoint = torch.load("checkpoints/gsm8k_best_atoms2.pt")
best_scales = l2_checkpoint["best_universal_scales"]  # (64,)
for name, param in model.named_parameters():
    if is_lora_target(name):
        for atom_idx in range(64):
            A = l2_checkpoint[f"atom_{atom_idx}.lora_A.{name}"]
            B = l2_checkpoint[f"atom_{atom_idx}.lora_B.{name}"]
            param.data += best_scales[atom_idx] * (A @ B)

model.save_pretrained("checkpoints/baked_math_llama/")
# This is the final frozen model. No runtime modification ever.
```

---

## KV Cache Architecture

With fully baked weights, the KV cache is valid across ALL cycles and ALL inner passes. Nothing about Llama changes — same weights, same input, same KV values.

### Cache Flow

```
PROBLEM ARRIVES:
  ┌─────────────────────────────────────────────────┐
  │ Llama encodes full problem text                  │
  │ → KV cache (stored)                              │  Cost: 1.0
  │ → Hidden states (passed to controller)           │  Computed ONCE
  └─────────────────────────────────────────────────┘
         │
         │  KV cache reused for ALL subsequent generation
         ▼
  ┌─────────────────────────────────────────────────┐
  │ CYCLE 1:                                         │
  │   Controller thinks (3 passes)      Cost: 0.45   │
  │   Llama generates from cached KV    Cost: 0.3    │
  │                                     Total: 0.75  │
  ├─────────────────────────────────────────────────┤
  │ CYCLE 2:                                         │
  │   Controller thinks (3 passes)      Cost: 0.45   │
  │   Llama generates from cached KV    Cost: 0.3    │
  │                                     Total: 0.75  │
  ├─────────────────────────────────────────────────┤
  │ CYCLE 3:                                         │
  │   Controller thinks (3 passes)      Cost: 0.45   │
  │   Llama generates from cached KV    Cost: 0.3    │
  │                                     Total: 0.75  │
  └─────────────────────────────────────────────────┘

  TOTAL: 1.0 + 3 × 0.75 = 3.25 Llama-equivalents
  vs PREVIOUS: ~8.35 (multiple full Llama forwards per cycle)
  SPEEDUP: ~2.5x
```

### Implementation

```python
class BakedLlamaWithCache:
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
    
    def encode_problem(self, input_ids):
        """Run once per problem. Cache KV and extract hidden states."""
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                use_cache=True,
                output_hidden_states=True
            )
        self.cached_kv = outputs.past_key_values
        hidden_states = outputs.hidden_states  # tuple of (batch, seq, 2048)
        return hidden_states
    
    def generate_from_cache(self, prompt_continuation_ids, max_new_tokens=64):
        """Generate using cached KV. Only computes new tokens."""
        with torch.no_grad():
            outputs = self.model.generate(
                prompt_continuation_ids,
                past_key_values=self.cached_kv,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # temperature=0 for eval
                use_cache=True
            )
        return outputs
```

### Cache Invalidation Rules

- **Never** within a problem. KV cache is valid for the entire problem lifetime.
- **New problem** → recompute. Different input tokens = different KV values.
- **Between training steps** → no invalidation needed. Llama's weights never change.

### What About Cycle-Specific Context?

Each cycle generates text ("equation #### answer"). This text needs to be part of the context for subsequent cycles. Two approaches:

**Option A: Extend KV cache.** After each cycle's generation, append the new tokens' KV values to the cached KV. The cache grows per cycle but each extension is cheap (only new tokens).

```python
def extend_cache_with_generation(self, generated_ids):
    """Add generated tokens to KV cache for next cycle."""
    with torch.no_grad():
        outputs = self.model(
            generated_ids,
            past_key_values=self.cached_kv,
            use_cache=True
        )
    self.cached_kv = outputs.past_key_values  # extended cache
```

**Option B: Text injection.** Prepend previous cycle's results as text to the generation prompt. Reuses the base KV cache but adds a small re-encoding cost for the injected text. Simpler but slightly more compute.

**Recommendation:** Option A (extend KV cache). It's the natural approach — the cache grows as the conversation develops, exactly how autoregressive models work. Each cycle adds ~20 tokens to the cache, which is negligible.

---

## Controller Inner Loop: Thinking in Pages

### The Core Loop

The controller doesn't adjust Llama. It thinks. Each inner pass reads Llama's hidden states plus its own accumulated pages and writes a new page. The decisions come after thinking.

```python
def controller_think(controller, hidden_states, notebook, cycle_num,
                      max_passes=3, energy_threshold=0.15):
    """
    Controller iterates on its own understanding.
    Each pass: read → write page → refine.
    Decisions come after thinking.
    """
    cycle_pages = []
    
    for pass_num in range(max_passes):
        # Controller reads everything available
        trunk_output = controller.encode(
            hidden_states=hidden_states,
            notebook=notebook,          # pages from previous cycles
            current_pages=cycle_pages,  # pages from this cycle's thinking
            cycle_num=cycle_num,
            pass_num=pass_num
        )
        
        # Write a page (record current understanding)
        page = controller.page_head(trunk_output)           # (batch, 256) on hypersphere
        energy = controller.energy_head(trunk_output)       # (batch, 1)
        cycle_pages.append(page)
        
        # Adaptive stopping: done thinking?
        if energy.mean() < energy_threshold:
            break
    
    # Make decisions after thinking
    decisions = controller.decide(trunk_output)
    # decisions.action: DECOMPOSE / SOLVE / MERGE (3-way Gumbel-softmax)
    # decisions.branch_embed: tree position (64d, L2-clipped)
    # decisions.confidence: how sure am I (sigmoid)
    
    # Append all thinking pages to notebook
    for page in cycle_pages:
        notebook.append(page, cycle_num)
    
    return decisions, cycle_pages
```

### What Pages Contain

Pages are 256-dim vectors on the hypersphere. They encode the controller's understanding — not explicit symbols, but learned representations. Through training, pages learn to encode things like:

- "This problem involves two entities" (structural understanding)
- "The first step result was correct" (progress tracking)
- "I need to subtract, not add" (operation awareness)
- "This is similar to fraction problems I've seen" (problem categorization)

We don't dictate what pages encode. The controller learns to write useful pages because the notebook attention reads them in subsequent passes and cycles. Pages that encode useful information lead to better decisions → better accuracy → more gradient signal.

### Why Pages Can't Collapse

Pages are driven by Llama's hidden states, which differ per problem. Unlike scales (where one direction minimizes loss for all problems), pages encode *what the controller observed*, which is inherently per-problem. The controller can't write the same page for "48 + 12" and "Sally had $20 less" because the hidden states it reads are completely different.

The inner loop amplifies this: pass 2 reads pass 1's page plus the hidden states, so even small differences in pass 1 pages compound. By pass 3, the accumulated pages are rich with per-problem information.

---

## Full Breathing Cycle

```python
def breathing_loop(llama, controller, problem_text, max_cycles=3):
    # 1. Llama reads problem ONCE
    input_ids = tokenize(problem_text)
    hidden_states = llama.encode_problem(input_ids)
    
    notebook = TreeNotebook()
    claimed_targets = []
    
    for cycle in range(max_cycles):
        # 2. Controller thinks (inner loop — cheap, controller only)
        decisions, pages = controller_think(
            controller, hidden_states, notebook, cycle
        )
        
        # 3. Act on decisions
        if decisions.action == DECOMPOSE:
            # Create child branches (recurse)
            for child in create_children(decisions):
                child_results = breathing_loop(
                    llama, controller, problem_text,
                    notebook=notebook.branch(child)
                )
                claimed_targets.extend(child_results)
        
        elif decisions.action == SOLVE:
            # 4. Llama generates from cached KV (cheap — only new tokens)
            generation = llama.generate_from_cache(
                build_generation_prompt(cycle, notebook)
            )
            
            # 5. Extract answer, claim target
            answer = extract_answer(generation)
            if answer in available_targets:
                claimed_targets.append(answer)
            
            # 6. Extend KV cache with generated text
            llama.extend_cache_with_generation(generation)
        
        elif decisions.action == MERGE:
            # Combine child results
            generation = llama.generate_from_cache(
                build_merge_prompt(notebook)
            )
            answer = extract_answer(generation)
            claimed_targets.append(answer)
        
        # 7. Check stopping criterion
        if decisions.confidence > 0.9 or all_targets_claimed:
            break
    
    return claimed_targets
```

---

## Controller Architecture (~350M)

```
CONTROLLER COMPONENTS:

State Encoder (~160M):
  - Perceiver cross-attention reading Llama hidden states (2048d × seq_len)
  - 8 attention heads, latent dim 1024
  - 4 layers
  - Produces compressed state vector (1024d)

Page Attention (~100M):
  - Attends to accumulated notebook pages
  - Hierarchical: ancestor pages, sibling pages, current-cycle pages
  - 6 layers, 8 heads
  - Input: state vector + all pages
  - Output: context-enriched representation (1024d)

Trunk (~60M):
  - 4-layer MLP integrating state + page context
  - LayerNorm + GELU
  - Output: trunk representation (2048d)

Decision Heads (~30M):
  - Page head: MLP → 256d → L2 normalize (hypersphere)
  - Action head: Linear → 3d → Gumbel-softmax (DECOMPOSE/SOLVE/MERGE)
  - Energy head: Linear → 1d → sigmoid (stopping criterion)
  - Confidence head: Linear → 1d → sigmoid
  - Branch embed head: Linear → 64d → L2 clip
```

---

## Training

### What Trains

Only the controller. Llama is frozen with baked weights. 

### Losses (All Direct — No ST Gradient, No Gradient Through Llama)

```python
# 1. Generation reward (REINFORCE on action decisions)
reward = 1.0 if answer_correct else 0.0
baseline = batch_reward.mean()
advantage = reward - baseline

# Action loss: reinforce good decisions
action_log_prob = F.log_softmax(action_logits, dim=-1)
action_loss = -(advantage * action_log_prob[selected_action]).mean()

# 2. Energy calibration (supervised)
if answer_correct:
    energy_loss = energy                     # push toward 0
else:
    energy_loss = F.relu(0.7 - energy)       # push toward 1

# 3. Page diversity (contrastive)
# Pages from different problems should differ
# Pages from same problem's inner loop should show progression
page_loss = contrastive_page_loss(all_pages, problem_ids, pass_nums)

# 4. Combined
total_loss = action_loss + 0.5 * energy_loss + 0.1 * page_loss
total_loss.backward()
controller_optimizer.step()
```

### No ST Gradient

The ST gradient is removed entirely. It was the wrong signal — uniform across problems, pointed toward one basin. The controller learns through:
- REINFORCE on discrete decisions (inherently per-problem via advantage)
- Supervised calibration on energy (inherently per-problem via correctness)
- Contrastive learning on pages (inherently per-problem via different hidden states)

### Teacher Forcing for Generation

Llama's generation is teacher-forced during training (standard cross-entropy on target tokens). This loss trains nothing (Llama is frozen) — it's only used to compute the reward signal for the controller.

```python
# Compute gen_loss for reward calculation only
with torch.no_grad():
    gen_outputs = llama.generate_from_cache(prompt)
    answer = extract_answer(gen_outputs)
    reward = 1.0 if answer in targets else 0.0
```

### Curriculum

Same as v2 plan:
1. **L3** (1-step) → verify controller can make SOLVE decisions and pages work
2. **L4** (2-step) → verify multi-cycle breathing with notebook
3. **L4.5** (3-step) → verify 3-cycle decomposition
4. **GSM8K** → does the thinking controller beat v1's 22%?
5. **MATH-500** → final benchmark

---

## Diagnostics

### Controller Thinking (Every Epoch)

| Metric | Healthy Range | What It Means |
|--------|---------------|---------------|
| `page_xproblem_cos` | < 0.8 | Different problems produce different pages |
| `page_progression_cos` | < 0.9 | Successive passes write different pages (thinking evolves) |
| `page_active_dims` | > 50/256 | Pages encode rich information |
| `avg_thinking_passes` | 1.5-3.0 | Adaptive stopping is working |
| `energy_correct_mean` | < 0.3 | Energy calibrated for correct answers |
| `energy_wrong_mean` | > 0.7 | Energy calibrated for wrong answers |

### Decision Quality (Every Epoch)

| Metric | Healthy Range | What It Means |
|--------|---------------|---------------|
| `action_entropy` | > 0.3 (when tree enabled) | Not collapsed to always-SOLVE |
| `decompose_accuracy` | Higher than solve-only | Decomposing helps when chosen |
| `confidence_calibration` | Correlation < -0.3 | Confidence tracks actual correctness |
| `per_cycle_accuracy` | Decreasing gracefully | Later cycles are harder but still productive |

---

## Compute Budget

```
Component               Per Problem    Notes
────────────────────────────────────────────────────
Llama encode            1.0            Once, cached
Controller think (×3)   0.45           3 passes × 0.15 each
Llama generate          0.3            From cached KV, new tokens only
────────────────────────────────────────────────────
Per cycle total         0.75
3-cycle problem         1.0 + 3×0.75 = 3.25
────────────────────────────────────────────────────
Previous architecture   ~8.35          Multiple full Llama forwards
Speedup                 ~2.5x
```

### Memory Budget

```
Component                    Memory (fp16)
──────────────────────────────────────────
Baked Llama (frozen)         ~2.4 GB
KV cache (per problem)       ~200 MB
Controller (350M)            ~700 MB
Controller optimizer         ~1.4 GB
Notebook pages               ~10 MB
Activations/gradients        ~1-2 GB
──────────────────────────────────────────
Total                        ~6-7 GB (fits A10G 24GB easily)
```

---

## What We Removed (And Why)

| Removed | Why |
|---------|-----|
| L2 atoms (82M params) | Baked into Llama. One dominant basin — let it be permanent. |
| Scale vectors (64 dims) | Controller can't learn per-problem scales. Two days of evidence. |
| AtomLoRAManager | No runtime LoRA swapping needed. |
| Monkey-patching | No weight modification at runtime. |
| ST gradient | Uniform across problems. Wrong signal for controller. |
| Profile library | Continuous scale selection still collapses. |
| Codebook | Same collapse problem as continuous scales. |
| Scale head | Controller doesn't produce scales anymore. |
| Separate atoms2 optimizer | No atoms2 to optimize. |

### What We Kept

| Kept | Why |
|------|-----|
| Baked L1 + L2 in Llama | Proven math ability. Universal blend is genuinely good. |
| Controller (~350M) | Thinking, decisions, memory. The only trainable component. |
| Tree structure | Decompose/solve/merge decisions. Controller's primary value. |
| Notebook pages | Controller's memory across cycles. Where per-problem variation lives. |
| Energy/confidence | Adaptive stopping. Lyapunov convergence. |
| Equal-reward (1/N per target) | Still needed to incentivize decomposition. |
| Number augmentation (0.8-1.2x) | Prevents memorization. |
| KV cache | Now permanent — never invalidated within a problem. |

---

## Risks / Mitigation

| Risk | Mitigation |
|------|-----------|
| Baked universal blend isn't optimal for all GSM8K problems | It got v1 to 22%. The controller adds value through structural decisions, not scale tweaking. |
| REINFORCE variance too high | K=3 actions is tiny. Batch-average baseline reduces variance. Reward is binary (correct/wrong) which is low variance. |
| Controller pages collapse to constant | Pages are driven by different hidden states per problem — can't collapse the way scales did. Hidden states are the input, not a learned output. |
| Without scale adjustment, ceiling is v1's 22% | The controller's value is in decomposition strategy, stopping decisions, and tree structure — not in LoRA tweaking. If structural decisions add 0% accuracy, the thesis needs revision. |
| Llama can't generate different equation types without scale variation | Llama with baked math mode generates diverse equations in v1. The prompt (including previous cycle results) steers the generation, not the scales. |

---

## Phase 0 Smoke Test

Before connecting to Llama, verify the controller can:

```python
# 1. Write different pages for different inputs
# 2. Make different action decisions for different inputs
# 3. Evolve pages across inner loop passes (page_2 ≠ page_1)
# 4. Energy estimates vary by input
# 5. Gradient flows to all components through REINFORCE

inputs = [torch.randn(1, 2048) for _ in range(10)]  # 10 different "hidden states"
rewards = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]  # alternating

# Train for 200 steps
# Verify: page_xproblem_cos < 0.5 and action decisions vary
```

---

## Implementation Checklist

1. Bake L2 universal blend into Llama weights (one-time script)
2. Implement `BakedLlamaWithCache` (encode_problem, generate_from_cache, extend_cache)
3. Strip all atom/scale/profile code from controller
4. Implement `controller_think` inner loop (page writing, adaptive stopping)
5. Implement REINFORCE loss on action decisions
6. Implement energy calibration loss
7. Implement contrastive page loss
8. Wire tree structure (DECOMPOSE creates branches, MERGE combines)
9. Add page-based diagnostics (xproblem_cos, progression_cos, active_dims)
10. Run Phase 0 smoke test
11. Run L3 → L4 → L4.5 curriculum
12. Run GSM8K → compare to v1's 22%
13. If GSM8K > 22%: proceed to MATH-500
14. If GSM8K ≤ 22%: controller structural decisions aren't adding value, rethink thesis
