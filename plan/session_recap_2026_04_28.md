# Session Recap — April 28, 2026

**Project:** Mycelium Breathing Models
**Session:** Architecture brainstorming + training monitoring
**Duration:** Full day session

---

## Where We Started

GSM8K training with single-layer atom architecture, equal-reward decomposition. Best result: **22% final accuracy** at epoch 6 (100-sample eval), with genuine multi-cycle decomposition confirmed.

## Key Results

### Single-Layer Runs
- Equal-reward single-layer peaked at 22% (epoch 6), declined to 15% by epoch 9
- Rebalanced loss weights (correct=1.0, self-consistent=0.5, wrong=0.3) disrupted initially but recovered — confirmed gradient starvation at old 0.05 wrong-answer weight was real

### Dual-Layer Architecture
- Reached 49% cycle 1 accuracy (up from 35-44% single-layer)
- But accuracy peaked early and declined — same pattern as single-layer
- Diagnosed as L1 atom drift: GSM8K training erodes the L4.5 math foundation

### Baked-L1 Configuration (Current)
- Froze L1 by baking into Llama's base weights — mathematically identical, no runtime cost
- K=3 inner passes with KV caching (K=3 at K=2 speed)
- 441-sample eval (4x less noise than 100-sample)
- 3x number token weighting + gentle augmentation (0.8-1.2)
- **Result: epoch 3 accuracy held at 13.6% — first run that didn't decline at epoch 3**
- 14.4 min/epoch (down from 25 min) — 2x speedup from baked L1

### Controller Gradient Bug (CRITICAL FINDING)
- Controller gradient norm: 0.0006 vs atoms2: 0.32 — **500x imbalance**
- Gradient flows through 1.2B frozen Llama params and gets diluted to uselessness
- Controller is effectively frozen — L2 atoms learn without steering
- 43% cycle 1 achieved *despite* the controller barely learning

---

## Architecture Decisions Made

### Two-Layer LoRA (Implemented)
- **Layer 1 (math engine):** L4.5-trained atoms, now baked into Llama base weights permanently. No runtime overhead.
- **Layer 2 (language steering):** Fresh atoms trained on GSM8K. Controller selects blend per inner pass.
- **Rationale:** Diagnostics showed L4.5 atoms are template-invariant (same atoms at max clamp for every problem). Universal math mode — perfect for freezing.

### Inner/Outer Loop (Implemented)
- **Outer loop:** Breathing cycles advance decomposition (claim targets 1, 2, 3)
- **Inner loop:** K=3 controller iterations refine L2 scales without additional Llama forwards
- Controller approach: iterate on L1 hidden states (cheap ~15% of Llama cost), then one L1+L2 forward for generation

### Three-Tier KV Caching (Partially Implemented)
- Tier 1 (base KV): computed once per problem — done
- Tier 2 (base+L1 KV): with baked L1, this IS tier 1 now — free
- Tier 3 (L2): recomputed every inner pass — by design
- Full Option B decomposition (cache input activations, add A2·B2·X delta) deferred — current approach is fast enough

### Straight-Through Controller Gradient (Handoff Written, Not Yet Implemented)
- Direct gradient injection: `scale_grad = torch.autograd.grad(gen_loss, scales, retain_graph=True)`
- Bypasses Llama dilution entirely — controller gets exact gradient at full strength
- Expected to be the biggest remaining unlock

---

## Research Validation

### DeepMind Paper: "The Topological Trouble With Transformers" (arXiv:2604.17121)
- Published April 18, 2026 — 10 days ago
- Core thesis: feedforward transformers have fundamental depth exhaustion limits for state tracking; recurrence is necessary
- Mycelium independently converged on the same architecture class (coarse recurrence with explicit state)
- Breathing loop = their "coarse recurrence"; notebook pages = their "explicit state tracking"; inner loop = their "depth recurrence"
- Actionable insight: adjoint sensitivity method for memory-efficient backprop through deep recurrence — future optimization, not current blocker
- **Key takeaway: theoretical validation that we're on the right track**

---

## Handoff Documents Produced

1. **two_layer_lora_inner_loop_handoff.md** — Two-layer architecture, inner/outer loop design, three-tier caching, training configuration
2. **straight_through_controller_gradient_handoff.md** — Controller gradient fix, straight-through estimator design, diagnostics, risk mitigation

---

## Open Questions / Next Steps

1. **Immediate:** Watch current baked-L1 run — does accuracy keep climbing past epoch 5? (First run that didn't decline at epoch 3)
2. **Next implementation:** Straight-through controller gradient — the controller is effectively frozen and fixing it could unlock a major accuracy jump
3. **If accuracy plateaus:** Investigate whether augmentation (even at 0.8-1.2) is still too aggressive, or whether the gen_loss/accuracy disconnect (#5 from Claude Code's analysis) needs token-level weighting adjustments
4. **Future optimization:** Adjoint sensitivity method for memory-efficient backprop through many breathing cycles (needed when scaling to N>3 cycles for harder problems)
5. **Future experiment:** Bake successful L2 atoms into weights after convergence → add L3 layer (recursive deepening of the LoRA stack)

---

## Updated CLAUDE.md Entries Needed

- Architecture is now v28+ with baked L1, dual-layer LoRA, K=3 inner passes
- Add to failed experiments: "Full LR on L1 during GSM8K training → math drift, accuracy peaks epoch 2-3 then declines"
- Add to key discoveries: "Controller gradient 500x attenuated through frozen Llama — straight-through estimator required"
- Update file structure: new handoff docs in plan/
- Update results: GSM8K 13.6% on 441-sample eval (more reliable than previous 22% on 100-sample)
