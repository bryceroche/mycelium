# Handoff: Straight-Through Controller Gradient Fix

**Author:** Bryce + Claude (brainstorm session)
**Date:** April 28, 2026
**Status:** Ready for implementation
**Depends on:** Current baked-L1 architecture with frozen L1, L2 atoms, K=3 inner passes
**Priority:** HIGH — controller gradient is 500x weaker than atoms2, effectively frozen

---

## Problem

The BreathingController produces L2 atom scales that steer Llama's generation. But the gradient from gen_loss back to the controller flows through the full 1.2B-parameter frozen Llama forward pass, diluting it by ~500x:

```
gen_loss → Llama output → L2 LoRA application → scales → controller
                ↑
        1.2B frozen params dilute gradient here
```

**Measured gradient norms:**
- atoms2: 0.32 (healthy, direct gradient through LoRA)
- controller: 0.0006 (500x weaker, effectively frozen)

**Isolated test confirmed:** when controller is directly in the computation graph without Llama in between, controller gradient norm is 560. The gradient direction is correct — it's just attenuated to uselessness by passing through Llama.

**Impact:** The controller cannot learn to steer L2 atoms. The model achieves 43% cycle 1 accuracy purely from atoms2 learning on its own without intelligent steering. Fixing the controller gradient could unlock a significant accuracy jump.

---

## Solution: Straight-Through Estimator

After each cycle's forward pass and generation loss computation, compute the gradient of gen_loss with respect to the controller's scale output directly, then inject it back as an auxiliary loss that bypasses Llama.

### Core Mechanism

```python
# During each cycle's forward pass:
# 1. Controller produces scales
scales = controller(hidden_states, cycle_num, pass_num, notebook, prev_scales)

# 2. Scales are applied to L2 atoms, Llama generates, gen_loss computed
# ... (existing code) ...

# 3. NEW: Compute direct gradient of gen_loss w.r.t. scales
scale_grad = torch.autograd.grad(
    gen_loss, 
    scales, 
    retain_graph=True,
    allow_unused=True
)[0]

# 4. NEW: Inject as auxiliary controller loss
if scale_grad is not None:
    controller_loss = (scales * scale_grad.detach()).sum()
    total_loss = total_loss + controller_loss
```

### Why This Works

The `scale_grad` vector tells us exactly how each of the 64 atom scales should change to reduce gen_loss. This is the same information the existing gradient path carries — but at full strength instead of 500x attenuated.

- `scale_grad.detach()` prevents double-counting: the original weak gradient path through Llama still exists, and this auxiliary loss adds the direct signal on top.
- The controller learns: "when I see these hidden states at this cycle/pass, I should produce scales that move in this direction."
- No REINFORCE variance, no baseline, no reward shaping. The gradient direction is exact.

### Intuition

Think of it as the controller getting a "cheat sheet" from the atoms. The atoms know which direction to move (their gradient is strong at 0.32). The straight-through estimator tells the controller: "the scales you produced should have been shifted by this much in this direction to get a better answer." The controller learns to anticipate what scale adjustments will help, rather than discovering it through the attenuated Llama path.

---

## Implementation Details

### Where to Add

In `train_per_cycle.py`, after gen_loss is computed for each cycle and before the backward pass. The straight-through loss should be accumulated alongside gen_loss into total_loss.

### Handling Multiple Cycles

Each cycle produces its own scales and gen_loss. Compute the straight-through gradient per cycle:

```python
cycle_losses = []

for cycle in range(num_cycles):
    scales = controller(...)
    # ... Llama forward, generation ...
    gen_loss = compute_gen_loss(...)
    
    # Straight-through controller gradient
    scale_grad = torch.autograd.grad(
        gen_loss, scales, retain_graph=True, allow_unused=True
    )[0]
    
    if scale_grad is not None:
        st_loss = (scales * scale_grad.detach()).sum()
        cycle_losses.append(gen_loss + st_loss)
    else:
        cycle_losses.append(gen_loss)

total_loss = sum(cycle_losses)
total_loss.backward()
```

### Handling K Inner Passes

The controller fires K=3 times per cycle in the inner loop, but only the final pass's scales are used for generation. The straight-through gradient applies to the **final scales** that produced the generation output. The inner loop iterations refine toward those final scales — the controller already gets signal for its refinement behavior through the final scales' gradient.

### Loss Weighting

Start with weight 1.0 on the straight-through loss (no scaling). The gradient magnitude from `scale_grad` is naturally calibrated to the gen_loss scale. If the controller starts learning too aggressively (scales oscillating wildly), add a damping factor:

```python
st_loss = st_weight * (scales * scale_grad.detach()).sum()
```

Try st_weight in [0.1, 0.5, 1.0]. Start at 1.0.

### Gradient Clipping

The controller may receive much stronger gradients than before (500x stronger). Ensure gradient clipping is applied to the controller's parameters:

```python
torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=1.0)
```

This prevents the sudden strong gradient from destabilizing the controller in early training.

---

## Diagnostics

### Verify the Fix

After implementing, check gradient norms again:

```
grad norms: controller=X.XXX atoms2=0.XXXX
```

The controller gradient should now be in the same order of magnitude as atoms2 (0.1-1.0 range), not 500x smaller.

### Monitor Controller Learning

Track these per epoch:
1. **Scale diversity across problems:** Are different problems getting different L2 blends? (Should be yes — this means the controller is steering per-problem.)
2. **Scale diversity across inner passes:** Is the controller adjusting scales between K=1, K=2, K=3? (Should show decreasing cosine similarity, meaning refinement is happening.)
3. **Controller weight drift:** How fast are controller parameters moving? (Should be steady, not explosive.)

### Compare Against Baseline

The current run (controller effectively frozen) is the baseline. After the fix:
- If accuracy jumps: controller steering was the bottleneck. The breathing loop is now fully functional.
- If accuracy stays the same: the controller wasn't the bottleneck. The atoms can find good configurations without steering, and the issue is elsewhere (language parsing, augmentation, etc.).
- If accuracy drops: the direct gradient is too strong or noisy. Reduce st_weight or increase gradient clipping.

---

## Risk / Mitigation

| Risk | Mitigation |
|------|-----------|
| Controller gradient too strong, destabilizes training | Gradient clipping at max_norm=1.0; reduce st_weight if needed |
| scale_grad is None (scales not in computation graph) | Check allow_unused=True; verify scales requires_grad=True |
| Double-counting gradient (weak path + direct path) | scale_grad.detach() prevents backprop through the direct path a second time |
| retain_graph=True increases memory | Memory cost is one extra graph retention per cycle; monitor GPU memory |
| Controller learns to produce extreme scales | Existing scale clamping [-0.5, 0.5] prevents this |

---

## What NOT to Do

- **Don't use REINFORCE.** REINFORCE with 64 atom scales is high-variance and requires careful baselining. The straight-through estimator gives exact gradients with zero variance.
- **Don't just bump controller LR by 500x.** This amplifies noise along with signal. The straight-through approach gives clean full-strength signal at normal LR.
- **Don't remove the existing gradient path through Llama.** It still provides correct (if weak) gradient and acts as a regularizer. The straight-through loss is additive.
- **Don't apply straight-through to inner loop intermediate passes.** Only the final scales that produce generation output get the direct gradient. Inner passes are implicitly trained through their effect on final scales.

---

## Success Criteria

- Controller gradient norm in same order of magnitude as atoms2 (~0.1-1.0)
- L2 scale diversity across problems increases (controller is steering per-problem)
- Accuracy climbs past epoch 5 without declining (controller learning prevents plateau)
- GSM8K final accuracy exceeds 22% baseline

---

## Implementation Priority

1. Add `torch.autograd.grad(gen_loss, scales, retain_graph=True)` after gen_loss computation
2. Add auxiliary straight-through loss to total_loss
3. Add gradient clipping on controller parameters
4. Log controller gradient norms to verify fix
5. Run and compare against current baseline
