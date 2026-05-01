# Handoff: Controller Owns Scales — Full Authority Architecture

**Author:** Bryce + Claude (brainstorm session)
**Date:** April 30, 2026
**Status:** Ready for implementation
**Priority:** HIGH — resolves the fundamental atoms2-vs-controller optimization conflict
**Depends on:** v2 controller (Phase 0 passed), baked L1, additive fingerprint, extraction fix

---

## Problem

Every architecture variant so far has the same failure mode: atoms2 and the controller fight over the scales, and atoms2 always wins.

```
atoms2 gradient on scales:     ~0.3  (direct through LoRA, strong)
controller gradient on scales: ~0.01 (through ST estimator, weak)
```

atoms2 pulls scales toward a universal saturated blend (±0.46) that's 80% good for all problems. The controller tries to introduce per-problem variation but gets overpowered 30x. Every compromise (delta scales, diversity loss, fingerprint) is fighting against atoms2's gradient dominance.

The pattern repeats across every run:

| Approach | Result |
|----------|--------|
| v1: controller + atoms2 share scales | Controller dead (constant function) |
| v2: ST gradient + fingerprint | Brief diversity, atoms2 re-saturates |
| v2: diversity losses | Zero gradient at fixed point |
| v2: delta scales (base + controller) | atoms2 dominates base, delta ignored |

**Root cause:** Shared ownership of the scale vector. As long as atoms2 has direct gradient on scales, it will always overpower the controller.

---

## Solution: Controller Owns Scales 100%

**The controller is the only thing that produces the scale vector.** atoms2 keeps its A/B matrices (what each atom does) but has no direct path to the scales (when each atom is used). Clean separation of concerns:

- **atoms2:** Provides good tools (learns A/B matrices that encode useful attention modifications)
- **Controller:** Decides which tools to use (produces the full scale vector per problem)

```
OLD (fighting):
  atoms2 gradient → scales ← ST gradient from controller
  Result: atoms2 wins, controller collapses

NEW (separated):
  atoms2 gradient → A/B matrices only (what atoms DO)
  controller gradient → scales only (WHEN to use atoms)
  Result: no conflict, each optimizes its own domain
```

---

## Architecture

### Forward Pass

```python
def breathing_cycle(llama, atoms2, controller, hidden_states, notebook,
                    cycle_num, fingerprint_proj):
    
    # 1. atoms2 suggests a scale vector (non-binding advisory input)
    suggested_scales = atoms2.suggest(hidden_states)  # (batch, 64)
    
    # 2. Problem fingerprint (frozen random projection, per-problem diversity)
    fingerprint = compute_fingerprint(hidden_states, fingerprint_proj)  # (batch, 64)
    
    # 3. Controller decides (FULL AUTHORITY over scales)
    raw_scales = controller(
        hidden_states=hidden_states,
        suggested_scales=suggested_scales.detach(),  # advisory only, no gradient to atoms2
        fingerprint=fingerprint,
        notebook=notebook,
        cycle_num=cycle_num
    )
    
    # 4. Controller owns the output
    final_scales = 0.46 * torch.tanh(raw_scales)  # bounded Euclidean, full range
    
    # 5. Apply to Llama
    output = llama_forward_with_atoms(llama, atoms2, final_scales)
    
    return output, final_scales
```

### Key Design Decisions

**`.detach()` on suggested_scales:** The suggestion is input to the controller but gradient does NOT flow back through it to atoms2. This prevents atoms2 from indirectly controlling scales through the suggestion pathway. atoms2 only gets gradient through its A/B matrices via gen_loss → Llama → LoRA.

**Controller has full range (±0.46):** No delta, no base+offset. The controller produces the complete scale vector. If it discovers that ±0.4 is optimal for most problems, it learns that — but it gets there through its own gradient, not because atoms2 dragged it there.

**Fingerprint remains additive:** The fingerprint still adds per-problem diversity to the raw scales before tanh. This guarantees diversity even if the controller's learned component is temporarily constant.

### atoms2 Suggestion Mechanism

atoms2 provides a "what I think the scales should be" signal as input to the controller. This is lightweight — a single linear projection from atoms2's internal state:

```python
class AtomSuggestion(nn.Module):
    """Non-binding scale suggestion from atoms2's perspective."""
    def __init__(self, hidden_dim=2048, num_atoms=64):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, num_atoms)
    
    def forward(self, hidden_states):
        # Pool over sequence length
        pooled = hidden_states.mean(dim=1)  # (batch, hidden_dim)
        return self.proj(pooled)  # (batch, 64)
```

This is trained as part of atoms2 (gets gradient through gen_loss) but its output is detached before reaching the controller. atoms2 learns what scales it *wishes* were applied. The controller learns whether to follow that suggestion.

**Why include the suggestion at all?** atoms2 has direct, strong gradient from gen_loss. It learns quickly what scale configurations reduce loss. The suggestion gives the controller a "warm start" signal — "atoms2 thinks you should do this." The controller can follow the suggestion for easy problems (where the universal blend works) and deviate for hard problems (where per-problem steering matters). This is the "backseat driving" — informed advice without authority.

---

## Gradient Flow

### atoms2: A/B Matrices Only

```
gen_loss → Llama output → LoRA application → atoms2.A, atoms2.B
```

atoms2 learns what each atom does to attention. Strong gradient (~0.3), direct path. This is unchanged from v1 — atoms2 is good at this.

### atoms2: Suggestion Head

```
gen_loss → Llama output → LoRA application → atoms2.A, atoms2.B
                                           → suggestion_head.proj
```

The suggestion head gets gradient from gen_loss through the same LoRA path. It learns "what scales would reduce gen_loss?" But this gradient does NOT reach the actual scales — `.detach()` blocks it.

### Controller: Scale Vector

```
gen_loss → scales → ST estimator → controller
```

The controller gets gradient via the straight-through estimator. Per-sample normalized, bypasses Llama. This is the ONLY gradient path to the scales.

```python
# Separate backward passes (cardinal rule)

# Pass 1: atoms2 A/B matrices + suggestion head
gen_loss.backward(retain_graph=True)
atoms2_optimizer.step()  # updates A, B, suggestion_head

# Pass 2: controller via ST estimator
scale_grad = torch.autograd.grad(gen_loss, final_scales, retain_graph=False)[0]
scale_grad_normalized = scale_grad / (scale_grad.norm(dim=-1, keepdim=True) + 1e-8)
st_loss = (final_scales * scale_grad_normalized.detach()).sum()
st_loss.backward()
controller_optimizer.step()  # updates controller only
```

### Gradient Summary

```
Component           Optimizes         Gradient Source     Magnitude
────────────────────────────────────────────────────────────────────
atoms2 A/B          What atoms do     gen_loss (direct)   ~0.3
atoms2 suggestion   Scale advice      gen_loss (direct)   ~0.1
Controller          Scale vector      ST estimator        ~0.01-0.1
Fingerprint         (frozen)          None                N/A
Llama + baked L1    (frozen)          None                N/A
```

No component has gradient on another's parameters. No fighting.

---

## Controller Input Specification

The controller receives multiple information streams:

```python
controller_input = {
    # Primary observation (what's happening)
    'hidden_states': hidden_states,          # (batch, seq_len, 2048) — Llama's internal state
    
    # Advisory signal (what atoms2 suggests)
    'suggested_scales': suggested.detach(),  # (batch, 64) — atoms2's non-binding recommendation
    
    # Diversity guarantee (per-problem identity)
    'fingerprint': fingerprint,              # (batch, 64) — frozen random projection
    
    # Memory (what happened in previous cycles)
    'notebook': notebook,                    # list of TreeNodes with pages
    
    # Positional context
    'cycle_num': cycle_num,                  # which breathing cycle
    'pass_num': pass_num,                    # which inner loop pass
    'prev_scales': prev_scales,              # what scales were used last cycle
}
```

The controller processes these through its state encoder, trunk, and scale head to produce the final scale vector. Every input is either a direct observation or an advisory signal — nothing constrains the controller's output.

---

## Initialization

### Controller Scale Head
Initialize with small weights (std=0.01) so initial scales are near zero. The fingerprint provides immediate per-problem diversity. As training progresses, the controller learns to produce larger, more informed scale vectors.

**Do NOT initialize bias to ±0.65** (the previous approach for atoms2's working regime). That was needed when atoms2 was fighting the controller toward saturation. Now the controller owns the scales and can find the right operating regime through its own gradient.

### atoms2
Load from L4/L4.5 checkpoint. The A/B matrices already know math. Only the suggestion head is new (initialize with small weights).

### Learning Rates
```python
controller_lr = 1e-3    # controller is the primary learner now
atoms2_lr = 1e-4        # atoms2 adapts slowly, mostly preserving L4.5 knowledge
suggestion_lr = 1e-4    # suggestion head adapts with atoms2
```

The controller gets a higher LR than atoms2. This is the opposite of v1 where atoms2 had all the gradient and the controller had none. Now the controller is the driver.

---

## Expected Behavior

### On L3/L4 (Easy Problems)
- atoms2 suggestion is accurate (universal blend works)
- Controller learns to roughly follow the suggestion
- scale_xproblem_cos may be high (limited need for per-problem variation)
- Accuracy 95%+ quickly
- This is fine — on easy problems, the universal blend IS the right answer

### On GSM8K (Hard Problems)
- atoms2 suggestion is the universal blend (~14-22% accurate at best)
- Controller learns to DEVIATE from suggestion for specific problem types
- scale_xproblem_cos drops as controller discovers per-problem patterns
- Accuracy exceeds the universal blend ceiling (>22%)
- The delta between following and deviating from the suggestion IS the controller's contribution

### The Test
If GSM8K accuracy with controller-owns-scales exceeds v1's 22% (universal blend), the controller is adding value. If accuracy is the same, the controller hasn't learned useful per-problem steering and the suggestion was sufficient.

---

## Diagnostics

### Controller Health (Every Epoch)

| Metric | What It Means |
|--------|--------------|
| `scale_xproblem_cos` | < 0.99 = controller differentiating |
| `scale_mid_frac` | > 0.3 = not saturated (controller chose this, not forced) |
| `scale_vs_suggestion_cos` | How closely controller follows atoms2's advice |
| `suggestion_quality` | Accuracy when using suggestion directly (baseline) |
| `ctrl_grad` | 0.01-1.0 range = healthy learning |
| `dead_dims` | < 10/64 = most dimensions active |

### Key New Metric: scale_vs_suggestion_cos

```python
# How much does the controller deviate from atoms2's suggestion?
suggestion_cos = F.cosine_similarity(
    final_scales, suggested_scales.detach(), dim=-1
).mean()
```

- **1.0:** Controller blindly follows suggestion (hasn't learned anything useful)
- **0.7-0.9:** Controller mostly follows but adjusts for specific problems
- **< 0.5:** Controller has learned its own strategy, largely ignoring suggestion

On easy problems (L3/L4), expect ~0.9 (follow suggestion, it's good).
On hard problems (GSM8K), expect this to decrease as the controller learns when to deviate.

### Per-Problem Analysis

For GSM8K, track which problems the controller deviates most on:
- Do high-deviation problems have higher accuracy than following the suggestion?
- Do certain problem types (multi-entity, conditional, fraction) trigger more deviation?
- Is the controller learning meaningful categories or just adding noise?

---

## Risks / Mitigation

| Risk | Mitigation |
|------|-----------|
| Controller starts near zero, atoms2 needs ±0.4 scales | Fingerprint + early training will push scales outward; monitor accuracy in first 5 epochs |
| Controller produces garbage scales, worse than universal blend | atoms2 suggestion provides a safety net — controller can learn to follow it initially |
| ST gradient still too weak for the controller to learn | Controller now has higher LR (1e-3) and no competing gradient on scales |
| atoms2 A/B matrices degrade without direct scale feedback | atoms2 still gets gen_loss gradient on A/B; it just can't optimize the scales directly |
| Controller learns to always follow suggestion (no added value) | This would show as scale_vs_suggestion_cos ≈ 1.0 on GSM8K; if so, controller architecture needs more capacity or richer observation |

---

## What NOT to Do

- **Don't let atoms2 gradient reach the scales.** The `.detach()` on suggested_scales is critical. Any gradient leak from atoms2 to scales recreates the fighting problem.
- **Don't initialize scale head bias to match atoms2's preferred regime.** Let the controller find its own operating point.
- **Don't freeze atoms2 entirely.** atoms2 needs to keep learning better A/B matrices for GSM8K patterns. Just block its path to the scales.
- **Don't remove the fingerprint.** Even with controller-owns-scales, the fingerprint guarantees per-problem diversity during early training before the controller has learned.
- **Don't add diversity losses.** The controller should differentiate because it's *useful* (improves accuracy per-problem), not because a loss forces it. If it stays constant, the problem is the gradient signal, not the lack of a diversity penalty.

---

## Implementation Checklist

1. Add `AtomSuggestion` module to atoms2 (one linear layer)
2. Modify `breathing_cycle` — controller produces final_scales directly, no base_scales parameter
3. Add `.detach()` on suggested_scales before passing to controller
4. Wire suggested_scales as input to controller alongside hidden_states, fingerprint, notebook
5. Separate optimizer steps: atoms2_optimizer (A/B + suggestion), controller_optimizer (everything else)
6. Set learning rates: controller 1e-3, atoms2 1e-4
7. Add `scale_vs_suggestion_cos` diagnostic
8. Initialize scale head with small weights (std=0.01), no bias trick
9. Remove any base_scales parameter or shared scale optimization
10. Run on L3 first to verify accuracy recovers, then GSM8K
