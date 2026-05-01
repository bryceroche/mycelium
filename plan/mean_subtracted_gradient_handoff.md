# Handoff: Mean-Subtracted Gradient — Spectral Decomposition of the Gradient Flow

**Author:** Bryce + Claude (brainstorm session)
**Date:** May 1, 2026
**Status:** Ready for implementation
**Priority:** CRITICAL — addresses the root cause of every controller collapse
**Depends on:** v3 architecture (baked Llama + thinking controller + soft tokens as only cycle channel)

---

## The Root Cause (Finally Understood)

Three days of evidence proved: the gen_loss gradient flowing back through Llama's computational graph has one dominant mode — "activate math processing." This mode is the same for every problem. Every controller output trained by this gradient collapses to constant because the dominant mode overwhelms any per-problem signal.

```
Full gradient = dominant_mode (same for all problems) + residual (per-problem)
                     ~95% of magnitude                    ~5% of magnitude

Controller trained on full gradient → learns dominant_mode → constant function
Controller trained on residual only → learns per-problem variation → diversity
```

**The fix:** Subtract the dominant mode from the gradient before it reaches the controller. The controller only sees the per-problem residual — the part of the gradient that says "how should THIS problem's soft tokens differ from the AVERAGE."

This is spectral decomposition of the gradient. The dominant eigenvector (math mode) is projected out. The remaining components carry per-problem information. The controller is trained on the AC signal, not the DC offset.

---

## Why This Is Better Than Every Previous Approach

| Approach | Signal Quality | Problem |
|----------|---------------|---------|
| Full gradient through Llama | Rich (high-dim vector) | One-basin collapse |
| ST estimator | Medium (high-dim, attenuated) | Same one-basin, 500x weaker |
| REINFORCE | Poor (scalar reward) | Too sparse for 322M params |
| Diversity losses | None (zero gradient at fixed point) | Can't break symmetry |
| **Mean-subtracted gradient** | **Rich (high-dim, per-problem)** | **Dominant mode removed** |

The mean-subtracted gradient gives the controller a rich, high-dimensional, per-problem training signal — the same quality as direct gradient, but with the component that causes collapse removed. It's the best of both worlds: gradient-quality signal without one-basin collapse.

---

## Architecture

### Two Gradient Paths, Two Roles

```
SOFT TOKENS = base_tokens + delta_tokens

base_tokens:  Learned universal prefix (like baked scales)
              Trained by: dominant mode of gradient (batch mean)
              Expected behavior: converge to one optimal value (one basin — let it)
              
delta_tokens: Per-problem adjustment from controller
              Trained by: residual gradient (dominant mode subtracted)
              Expected behavior: different for every problem (per-problem signal only)
```

Llama sees `base_tokens + delta_tokens`. The base provides the universal math-mode activation. The delta provides per-problem steering. They're trained by different components of the same gradient and don't fight.

### Soft Tokens as Only Cycle Channel

From the earlier design decision: remove text injection of previous cycle results. Soft tokens are the ONLY way cross-cycle information reaches Llama. This makes the delta tokens essential — without per-cycle variation, Llama has no memory of previous steps.

```
Cycle 1: [base + delta_1] + problem text → generate step 1
         Controller reads output, writes pages

Cycle 2: [base + delta_2] + problem text → generate step 2
         delta_2 ≠ delta_1 because controller's pages changed
         Llama's ONLY knowledge of step 1's result is through delta_2

Cycle 3: [base + delta_3] + problem text → generate step 3
         delta_3 encodes full history from pages
```

---

## Implementation

### Computing Per-Sample Gradients

The key operation: compute the gradient of gen_loss with respect to soft_tokens separately for each problem in the batch.

```python
def compute_mean_subtracted_gradient(gen_loss_per_sample, soft_tokens):
    """
    gen_loss_per_sample: (batch_size,) — per-problem generation loss
    soft_tokens: (batch_size, num_soft, embed_dim) — requires_grad=True
    
    Returns: residual gradient (batch_size, num_soft, embed_dim)
             with dominant mode subtracted
    """
    batch_size = gen_loss_per_sample.shape[0]
    per_problem_grads = []
    
    for i in range(batch_size):
        grad_i = torch.autograd.grad(
            gen_loss_per_sample[i],
            soft_tokens,
            retain_graph=True,
            create_graph=False
        )[0][i]  # gradient for problem i
        per_problem_grads.append(grad_i)
    
    grads = torch.stack(per_problem_grads)  # (batch, num_soft, embed_dim)
    
    # Dominant mode = batch mean (the "math mode" component)
    dominant_mode = grads.mean(dim=0, keepdim=True)  # (1, num_soft, embed_dim)
    
    # Residual = per-problem signal with dominant mode removed
    residual = grads - dominant_mode  # (batch, num_soft, embed_dim)
    
    # Normalize per problem for consistent magnitude
    residual_flat = residual.view(batch_size, -1)  # (batch, num_soft * embed_dim)
    residual_normalized = residual_flat / (residual_flat.norm(dim=-1, keepdim=True) + 1e-8)
    residual_normalized = residual_normalized.view_as(residual)
    
    return dominant_mode.squeeze(0), residual_normalized
```

### Training Loop

```python
# Forward pass
hidden_states = baked_llama.encode(problem_text)  # cached
pages, soft_tokens = controller.think(hidden_states, notebook, cycle_num)

# soft_tokens = base_tokens + delta_tokens
base_tokens = controller.base_token_param  # learned parameter, not from controller network
delta_tokens = controller.soft_token_head(pages)  # from controller's thinking
final_soft_tokens = base_tokens + delta_tokens

# Generate with soft tokens as only cycle context
gen_loss_per_sample = baked_llama.generate_teacher_forced(
    final_soft_tokens, problem_text, target_text
)  # (batch_size,) — per-sample loss

# ============================================================
# GRADIENT DECOMPOSITION
# ============================================================

# Step 1: Compute mean-subtracted gradient
dominant_mode, residual_grad = compute_mean_subtracted_gradient(
    gen_loss_per_sample, final_soft_tokens
)

# Step 2: Train base tokens with dominant mode (universal prefix)
base_loss = (base_tokens * dominant_mode.detach()).sum()
base_loss.backward(retain_graph=True)
base_optimizer.step()
base_optimizer.zero_grad()

# Step 3: Train controller with residual (per-problem steering)
delta_loss = (delta_tokens * residual_grad.detach()).sum()
delta_loss.backward()
controller_optimizer.step()
controller_optimizer.zero_grad()

# ============================================================
# REINFORCE for discrete decisions (tree actions, stopping)
# ============================================================
reward = compute_reward(generated_answer, targets)
baseline = reward.mean()
advantage = reward - baseline
action_loss = -(advantage * action_log_probs).mean()
action_loss.backward()
controller_optimizer.step()
controller_optimizer.zero_grad()
```

### Efficient Per-Sample Gradient (Optimization)

Computing per-sample gradients with a loop is expensive. Two alternatives:

**Option A: Use functorch/vmap (PyTorch 2.0+)**
```python
from torch.func import grad, vmap

# Vectorized per-sample gradient
per_sample_grad_fn = vmap(grad(lambda st, gl: gl, argnums=0), in_dims=(0, 0))
grads = per_sample_grad_fn(soft_tokens, gen_loss_per_sample)
```

**Option B: Approximate with micro-batches**
```python
# Split batch into micro-batches of 1, compute gradient, stack
# Simpler than vmap, works on all PyTorch versions
# For batch_size=8, this is 8 backward passes — acceptable at 25s/epoch
```

**Option C: First-order approximation**
```python
# Instead of exact per-sample gradient, use the Jacobian-vector product
# Less accurate but only requires one backward pass
# Good enough if the per-problem variation is mostly in the loss magnitude
# rather than the gradient direction

# Per-sample loss weights
weights = gen_loss_per_sample - gen_loss_per_sample.mean()  # centered
weighted_loss = (weights.detach() * gen_loss_per_sample).sum()
weighted_loss.backward()
# The gradient of weighted_loss is approximately the "residual" direction
```

**Recommendation:** Start with Option B (loop) for correctness. Optimize to Option A (vmap) if speed is needed. Option C is a fallback if per-sample gradients are too expensive.

---

## Why This Won't Collapse

### The Math

The dominant mode is subtracted from the gradient. The controller ONLY receives the residual. The residual is zero-mean across the batch by construction:

```
residual_i = grad_i - mean(grad)
sum(residual_i) = sum(grad_i) - N * mean(grad) = 0
```

The controller cannot learn a single output that satisfies all residual gradients simultaneously, because they sum to zero. Any constant output has zero correlation with the residual signal. The ONLY way to reduce the residual loss is to produce DIFFERENT outputs for different problems.

This is the mathematical guarantee that was missing from every previous approach. Diversity losses had degenerate gradients at the fixed point. The mean-subtracted gradient has NO fixed point at constant output — constant output gives zero gradient, which means the controller is at a saddle point, not a minimum. Any perturbation that correlates with the residual reduces loss, so the controller will always move away from constant.

### The Intuition

The dominant mode says: "every problem wants you to move in this direction."
The residual says: "problem A wants you to move slightly left of average, problem B wants slightly right."

Training on the dominant mode → move in the shared direction → constant function.
Training on the residual → move differently for each problem → diversity.

The base tokens handle the shared direction. The delta tokens handle the per-problem deviations. Clean separation.

---

## Connection to Spectral / Resonance Framework

The mean subtraction IS the DC removal in Fourier analysis:

- **Full gradient** = the raw signal (DC + all frequencies)
- **Dominant mode (batch mean)** = the DC component (constant offset, same everywhere)
- **Residual** = the AC signal (oscillations, varies per problem)
- **Per-problem residual vector** = that problem's frequency signature

The controller is trained on the AC signal only. It learns to resonate with each problem's unique frequency because the DC component (which would drown out the frequencies) has been removed.

### Future: Full SVD Decomposition

Mean subtraction removes eigenmode 0. For richer analysis:

```python
grads_centered = grads - grads.mean(dim=0)
U, S, V = torch.svd(grads_centered)

# S = singular values = amplitude of each gradient mode
# V = singular vectors = the "natural frequencies" of problem variation  
# U = per-problem coefficients = each problem's spectral signature

# If S = [10, 8, 6, 0.1, 0.01, ...]
#   → 3 meaningful modes of per-problem variation
#   → Controller should learn a 3-mode spectral decomposition

# If S = [10, 0.1, 0.01, ...]
#   → Only 1 meaningful mode after DC removal
#   → Per-problem variation is essentially 1-dimensional
```

The SVD reveals how many dimensions of per-problem variation exist in the gradient. This tells you the controller's effective capacity requirement — if there are only 3 significant modes, the controller only needs to learn 3 different behaviors, not 64.

**Don't implement SVD initially.** Mean subtraction captures the main benefit. SVD is a diagnostic tool to understand the gradient landscape, and a future optimization to align the controller's architecture with the actual dimensionality of per-problem variation.

---

## Diagnostics

### Gradient Spectrum (Run Periodically)

```python
# How much per-problem signal exists in the gradient?
grads = compute_per_sample_grads(gen_loss, soft_tokens)
dominant = grads.mean(dim=0)
residuals = grads - dominant

dominant_magnitude = dominant.norm()
residual_magnitudes = residuals.norm(dim=-1).mean()

ratio = residual_magnitudes / dominant_magnitude
# ratio = 0.01 → per-problem signal is 1% of dominant (weak but present)
# ratio = 0.1  → per-problem signal is 10% (strong)
# ratio > 0.3  → rich per-problem gradient (ideal)
```

### Diversity After Training

| Metric | Target | What It Means |
|--------|--------|---------------|
| `delta_token_cos` | < 0.9 | Controller producing different deltas per problem |
| `delta_token_cos` across cycles | < 0.8 | Different cycles get different deltas |
| `base_token_magnitude` | Stable | Base converging to universal prefix |
| `delta_token_magnitude` | > 0 | Controller contributing, not zeroed out |
| `residual_grad_ratio` | > 0.01 | Per-problem signal exists in gradient |

### Critical Test: Does Diversity Survive Training?

```
Epoch 0 (init):  delta_cos = 0.46 (diverse, from sharp attention)
Epoch 1:         delta_cos = ???

Previous approaches: → 1.0 (collapsed)
This approach:       → should stay < 0.9 (residual gradient maintains diversity)
```

If delta_cos stays below 0.9 after epoch 1, the mean subtraction is working. The one-basin collapse is broken.

---

## Training Configuration

```python
# Learning rates
base_token_lr = 1e-3     # base converges fast to universal prefix
controller_lr = 1e-4     # controller learns per-problem deltas more carefully

# Soft token configuration
num_soft_tokens = 4       # 4 virtual tokens prepended to generation
soft_token_dim = 2048     # matches Llama's embedding dim

# Inner loop
num_thinking_passes = 3   # for trajectory-based diversity
attention_temperature = 0.1  # sharp attention for amplification

# Generation
remove_text_injection = True  # soft tokens are ONLY cycle channel
```

---

## Risks / Mitigation

| Risk | Mitigation |
|------|-----------|
| Residual gradient is too small (ratio < 0.01) | Normalize residual per-sample to unit length — preserves direction even at small magnitude |
| Per-sample gradient computation is expensive | Start with loop (Option B), optimize to vmap (Option A) if needed. At batch=8 and 25s/epoch, 8 backward passes are ~3x slowdown — acceptable |
| Removing text injection hurts accuracy (Llama loses cycle context) | The whole point — soft tokens MUST carry this information. If accuracy drops, the soft tokens will learn to carry it because they're the only channel |
| Base tokens converge but delta stays near zero | Monitor delta_token_magnitude. If zero, the residual gradient has no useful signal — check residual_grad_ratio diagnostic |
| Controller learns to put everything in base, nothing in delta | Base is a single learned parameter (not per-problem). It CAN'T carry per-problem info. The delta is the only per-problem path. |

---

## What NOT to Do

- **Don't train delta tokens with the full gradient.** That's what collapsed every previous approach. Only the residual.
- **Don't train base tokens with the residual.** Base should converge to the universal prefix. Give it the dominant mode.
- **Don't add diversity losses.** The mean-subtracted gradient IS the diversity signal. No auxiliary losses needed.
- **Don't add fingerprints.** The residual gradient provides per-problem differentiation directly. No structural tricks needed.
- **Don't skip the inner loop.** The sharp attention trajectory provides diversity at init which the residual gradient can then preserve and shape. Without the inner loop, the controller starts constant and the residual gradient alone may not break the symmetry.

---

## Implementation Checklist

1. Modify gen_loss computation to return per-sample losses (not batch-reduced)
2. Implement `compute_mean_subtracted_gradient` function
3. Split soft tokens into `base_tokens` (nn.Parameter) + `delta_tokens` (from controller)
4. Separate backward passes: base with dominant mode, controller with residual
5. Remove text injection of previous cycle results
6. Keep REINFORCE for discrete decisions (tree actions, stopping)
7. Add gradient spectrum diagnostic (dominant vs residual magnitude ratio)
8. Add delta_token_cos diagnostic (per-problem diversity)
9. Run Phase 0 smoke test: does delta_cos survive training?
10. Run L3: does accuracy climb with diverse delta tokens?
11. Run GSM8K: does accuracy exceed 22% with per-problem steering?

---

## Summary

The one-basin collapse was caused by the dominant mode of the gradient — "activate math mode" — drowning out per-problem signal. The fix is spectral decomposition: subtract the dominant mode, train the controller on the residual only. The controller gets rich, high-dimensional, per-problem gradient signal without the component that causes collapse.

Base tokens learn the universal prefix (DC component). Delta tokens learn per-problem adjustments (AC component). The gradient is decomposed into the right signal for each role. No fighting, no collapse, no diversity tricks needed.

The mathematical guarantee: mean-subtracted gradients sum to zero across the batch. No constant output can satisfy them. The controller MUST produce different outputs to reduce loss. Diversity isn't encouraged — it's mathematically required.
