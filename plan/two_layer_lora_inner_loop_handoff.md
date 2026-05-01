# Handoff: Two-Layer LoRA with Inner/Outer Loop & Three-Tier KV Caching

**Author:** Bryce + Claude (brainstorm session)
**Date:** April 28, 2026
**Status:** Ready for implementation
**Depends on:** atom_lora.py (existing atoms/atoms2 infrastructure), BreathingController (existing cycle_embed/pass_embed/prev_scale_project)

---

## Motivation

The current single-layer atom architecture achieves 99.5% on L4.5 (procedural 3-step) but plateaus around 22% on GSM8K. Diagnostics reveal the bottleneck:

- L4.5 atoms are **template-invariant**: same atoms at max clamp (0.463) for every problem, but different atoms per cycle. This is a universal "math mode" transform.
- GSM8K requires **inferential language parsing** ("if Sally had $20 less, she would have $80" → 80 + 20 = 100) which the current atoms can't do because they're simultaneously trying to be a math engine AND a language parser.

**Solution:** Separate the two jobs into two LoRA layers with an inner loop for refinement.

---

## Architecture

### Two Layers

**Layer 1 — Math Engine (slow-moving foundation)**
- Initialized from L4.5-trained `model.atoms` checkpoint
- Transforms Llama from token predictor → arithmetic computer
- Trained at **low learning rate** (10x lower than layer 2, e.g., layer_1_lr = 1e-4, layer_2_lr = 1e-3)
- NOT frozen — flexible enough to adapt if GSM8K needs a different math mode than L4.5
- If layer 1 barely moves during training → freezing was fine, can hard-freeze later
- If layer 1 shifts significantly → good that we kept it flexible

**Layer 2 — Per-Step Steering (fast-moving adaptation)**
- Fresh initialization on `model.atoms2`
- Handles problem-specific language parsing and step planning
- Controller selects blend per inner pass based on Llama's hidden states
- This is where "if Sally had $20 less" → "100 - 20 = 80" gets learned

**Composition:** LoRA is additive. Llama sees `W_base + ΔW_layer1 + ΔW_layer2`. Layer 1 modifies weights first (math mode), layer 2 modifies on top (step-specific steering).

### Inner/Outer Loop

```
OUTER LOOP (breathing cycles, advances decomposition):
  for cycle in 1..N:

    INNER LOOP (refinement passes, adjusts layer 2):
      for pass in 1..K:
        Controller reads Llama hidden states
        Controller outputs: layer_2_scales, page_update, confidence
        Apply layer_1_scales (stable) + layer_2_scales (adjusted) to Llama
        Llama generates equation + answer

    Extract answer from final inner pass → claim target
    Append page to notebook
    Advance to next cycle
```

- **Outer loop** = decomposition across cycles (cycle 1 claims target 1, cycle 2 claims target 2, etc.)
- **Inner loop** = refinement within a cycle (controller reacts to Llama's output, adjusts layer 2, tries again)
- Controller fires at **every inner pass** — it reads the result and decides how to adjust
- Start with **K=2** inner passes (infrastructure already exists via `thinking_pass_multipass`)
- Experiment with K=3 once K=2 is validated

### Controller Inputs Per Inner Pass

The controller already has the embeddings for this. Ensure all are wired:

- `cycle_embed`: which breathing cycle (outer loop position)
- `pass_embed`: which inner pass (inner loop position)
- `prev_scale_project`: what the layer 2 atom blend was on the previous pass
- Llama hidden states: what Llama just generated
- Notebook pages: accumulated history from previous cycles

---

## Three-Tier KV Caching

What you cache defines what's stable. What's stable defines each layer's job.

### Tier 1 — Base KV Cache
- **What:** Llama's vanilla forward pass on the problem text (no atoms)
- **When computed:** Once per problem
- **Reused across:** All cycles, all inner passes
- **Already implemented**

### Tier 2 — Base + Layer 1 KV Cache
- **What:** Llama's forward pass with layer 1 atoms applied
- **When computed:** Once per cycle (layer 1 scales may vary slightly per cycle)
- **Reused across:** All K inner passes within that cycle
- **This is the new win:** If K=3 inner passes, saves 2 full forward passes per cycle
- **Staleness:** Layer 1 weights change during training (low LR), but within a single forward pass they're fixed. Recompute per training step, not across batches.

### Tier 3 — Layer 2 (Never Cached)
- **What:** The layer 2 atom modification
- **Recomputed:** Every inner pass — this is the adaptation that changes
- **Cost:** Only the delta from layer 2, applied on top of the cached tier 2 KV

### Cache Invalidation Rules
- Layer 1 weights update → tier 2 cache stale (recompute next forward pass)
- New problem → tier 1 and tier 2 recomputed
- New cycle → tier 2 recomputed (controller may adjust layer 1 scales per cycle)
- New inner pass → only layer 2 recomputed (tier 2 cached)

### Implementation Note
The tier 2 cache requires storing KV states after layer 1 modification but before layer 2. This may require a hook in the forward pass between the two LoRA applications, or computing them as separate sequential operations rather than a single fused ΔW.

---

## Training Configuration

```python
# Learning rates
layer_1_lr = 1e-4      # slow-moving math engine
layer_2_lr = 1e-3      # fast-moving step steering
controller_lr = 5e-4   # controller adapts to both layers

# Layer 1 initialization
model.atoms.load_state_dict(l45_checkpoint['atoms'])  # from L4.5 best

# Layer 2 initialization
model.atoms2.reset_parameters()  # fresh init

# Loss weights (from reweighting experiment)
correct_weight = 1.0
self_consistent_weight = 0.5
wrong_weight = 0.3

# Inner loop
num_inner_passes = 2  # K=2 to start

# Curriculum
# Start directly on GSM8K — layer 1 already has L4.5 math competence
```

---

## Diagnostics to Run

**Before training (on L4.5 checkpoint):**
1. Confirm layer 1 atoms are template-invariant (already done — yes)
2. Confirm different atoms per cycle (already done — yes)
3. Record layer 1 atom scales as reference baseline

**During training:**
4. Track layer 1 weight drift — how much do the atoms move from L4.5 init?
5. Track layer 2 atom diversity — are different blends used per problem? (Should be yes, unlike layer 1)
6. Compare inner pass 1 vs pass 2 accuracy — does the refinement loop help?
7. Monitor scale cosine similarity between inner passes (should be <0.9, indicating adjustment)

**At convergence:**
8. Run `diag_debug_cycles.py` on GSM8K failures — which language patterns still fail?
9. Compare layer 1 final vs layer 1 init — if barely moved, hard-freeze for efficiency

---

## Success Criteria

- GSM8K accuracy exceeds 22% baseline (current best with single-layer atoms)
- Layer 2 shows problem-specific variation (different blends per problem, unlike layer 1)
- Inner pass 2 accuracy > inner pass 1 accuracy (refinement loop is working)
- Layer 1 remains relatively stable (small drift from L4.5 init)

---

## Risk / Mitigation

| Risk | Mitigation |
|------|-----------|
| Layer 2 fresh init disrupts layer 1's math mode | Low LR on layer 1, monitor weight drift |
| Inner loop adds compute cost | Tier 2 caching — inner passes only pay layer 2 cost |
| Controller can't learn to drive two layers | Already has pass_embed + prev_scale_project; if struggling, try K=2 before K=3 |
| Layer 1 drifts too far from math mode | Add regularization loss: L2 penalty on (atoms - atoms_init) |
| Tier 2 KV cache implementation is complex | Start without tier 2 cache, add as optimization after architecture is validated |

---

## Implementation Priority

1. **Wire up two-layer forward pass** — layer 1 + layer 2 composed additively, controller fires per inner pass
2. **Initialize layer 1 from L4.5, fresh layer 2** — separate optimizers or param groups with different LRs
3. **Train on GSM8K with K=2 inner passes** — validate the architecture works
4. **Add tier 2 KV caching** — optimization after correctness is confirmed
5. **Experiment with K=3** — only if K=2 shows inner-loop improvement
