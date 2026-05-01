# Design Doc: Energy-Based Inner Loop with Adaptive Convergence

**Author:** Bryce + Claude (brainstorm session)
**Date:** April 28, 2026
**Status:** Future work — implement after controller is alive and tree structure is validated
**Depends on:** Straight-through controller gradient, tree-structured breathing, scaled controller (~436M)
**Inspired by:** Yann LeCun's Energy-Based Models / Factor Graphs, Lyapunov stability theory

---

## Motivation

The inner loop currently runs a fixed K=3 iterations. Simple problems that the controller solves on pass 1 waste passes 2 and 3. Hard problems where the controller is still uncertain at pass 3 would benefit from more iterations. Fixed K is a compromise.

The insight from two convergent ideas:

**Energy-Based Models:** Inference is optimization — start from an initial state, iteratively minimize an energy function until you reach a coherent low-energy configuration. The controller's confidence signal is already an energy estimate. Train it properly and it becomes the stopping criterion.

**Lyapunov Stability:** A dynamical system converges if there exists a function V where V̇ < 0 at each step. The energy function IS the Lyapunov function. If energy decreases at each inner pass, the loop is converging. If it stops decreasing, the loop has found its fixed point and should stop.

**Combined:** The inner loop becomes energy minimization with a convergence guarantee. Iterate until the energy (confidence) is low enough, with a Lyapunov-style proof that it will terminate.

---

## Architecture

### Confidence Head as Energy Function

The controller already outputs a confidence scalar. Reframe it as an energy function:

```
E(scales, hidden_states, notebook) → scalar

Low energy  = controller believes these scales will produce a correct answer
High energy = controller believes these scales are wrong
```

**Current usage:** Confidence is trained weakly as a byproduct. 
**Proposed usage:** Confidence is trained explicitly as an energy function with contrastive loss.

### Contrastive Energy Training

After each cycle, we know whether the generated answer was correct. Use this as supervision:

```python
# Controller produces energy estimate alongside scales
energy = controller.energy_head(controller_state)

# After generation, we know if the answer was correct
if answer_correct:
    # Push energy DOWN for good scale configurations
    energy_loss = energy  # minimize
else:
    # Push energy UP for bad scale configurations
    energy_loss = max(0, margin - energy)  # maximize (hinge)

total_loss += energy_weight * energy_loss
```

This is exactly the EBM contrastive loss: low energy for compatible (correct) configurations, high energy for incompatible (wrong) configurations. The margin prevents the model from assigning infinite energy to wrong answers.

### Adaptive Inner Loop

Replace fixed K with energy-based stopping:

```python
def adaptive_inner_loop(controller, hidden_states, notebook, 
                         max_K=6, energy_threshold=0.1, 
                         convergence_threshold=0.01):
    
    scales = controller.initial_scales(hidden_states)
    prev_energy = float('inf')
    
    for k in range(max_K):
        # Controller refines scales and estimates energy
        scales, page, energy = controller(
            hidden_states, scales, notebook, cycle_num, pass_num=k
        )
        
        # Lyapunov check: is energy decreasing?
        energy_delta = prev_energy - energy
        
        # Stop if energy is low enough (confident)
        if energy < energy_threshold:
            break
            
        # Stop if energy isn't decreasing (converged or stuck)
        if energy_delta < convergence_threshold:
            break
            
        prev_energy = energy
    
    return scales, page, energy, k  # k = actual passes used
```

### Lyapunov Convergence Guarantee

For the inner loop to provably converge, we need V̇ < 0 at each step. The energy function E is V. We can add a soft regularization that encourages energy to decrease at each inner pass:

```python
# During training, track energy across inner passes
energies = []  # energy at each inner pass

for k in range(K):
    scales, page, energy = controller(...)
    energies.append(energy)

# Lyapunov regularization: energy should decrease monotonically
lyapunov_loss = 0
for k in range(1, len(energies)):
    # Penalize energy increases (V̇ > 0)
    energy_increase = max(0, energies[k] - energies[k-1])
    lyapunov_loss += energy_increase

total_loss += lyapunov_weight * lyapunov_loss
```

This doesn't force strict monotonic decrease (which could be too constraining) but penalizes violations. The controller learns that inner passes should refine toward lower energy.

---

## Connection to Existing Architecture

### The Straight-Through Estimator IS the Energy Gradient

The straight-through gradient:
```python
scale_grad = torch.autograd.grad(gen_loss, scales, retain_graph=True)[0]
```

This is ∂E/∂scales — the gradient of the energy (gen_loss) with respect to the scales. The inner loop controller iterations are already performing gradient descent on the energy surface, just implicitly. Making the energy explicit (via the confidence head) and the descent explicit (via Lyapunov regularization) formalizes what the system is already doing.

### The Tree Structure Maps to Factor Graphs

LeCun's factor graph framing:
- **Variable nodes** = tree nodes (subproblem answers) — the "hollow circles" that need to be inferred
- **Factor nodes** = merge/decompose constraints — local compatibility functions
- **Energy** = sum of local factor energies — how coherent is the whole tree?

The tree-structured notebook already encodes this. Each node has a page (variable state) and the merge operation checks compatibility (factor). The total energy is the sum of per-node energies.

```python
# Total tree energy = sum of node energies
tree_energy = sum(node.energy for node in tree.nodes)

# Factor energies at merge nodes
for merge_node in tree.merge_nodes:
    children = tree.get_children(merge_node)
    # Merge is compatible if children's answers combine correctly
    factor_energy = compatibility(merge_node.answer, 
                                   [c.answer for c in children])
    tree_energy += factor_energy
```

### The v7 Factor Graph Comes Full Circle

The v7 architecture had explicit energy terms:
- structural_consistency → tree structure constraints
- dependency_flow → parent-child relationships in tree
- execution_validity → per-node generation quality (gen_loss)
- verification → merge node compatibility checking

The tree-structured breathing with energy-based inner loop IS the v7 factor graph, implemented as a differentiable recurrent system instead of a static graph.

---

## Training Strategy

### Phase 1: Train Energy Head on Existing Architecture
- Add energy head to controller (single scalar output)
- Train with contrastive loss alongside existing gen_loss
- Keep fixed K=3 — don't change the loop yet
- Monitor: does energy correlate with answer correctness?
- Success criterion: energy < 0.3 for correct answers, energy > 0.7 for wrong answers

### Phase 2: Adaptive K with Energy Stopping
- Replace fixed K with energy-threshold stopping
- Set max_K=6 as safety cap
- Add Lyapunov regularization (penalize energy increases across inner passes)
- Monitor: what's the average K per problem? (Should vary — easy problems K=1, hard problems K=4+)
- Monitor: does energy decrease monotonically within inner loops?
- Success criterion: accuracy improves or matches fixed K=3 with lower average compute

### Phase 3: Energy-Based Tree Evaluation
- Extend energy to tree-level: total energy = sum of node energies + factor energies
- Merge nodes evaluate compatibility of children
- Tree-level energy guides structural decisions (decompose when energy is high, merge when children have low energy)
- The tree grows until total energy is minimized

---

## Diagnostics

| Diagnostic | What It Reveals |
|-----------|----------------|
| Energy vs correctness correlation | Is the energy head learning? (Should be negative correlation) |
| Energy trajectory per inner loop | Is energy decreasing? (Lyapunov V̇ < 0?) |
| Average K per problem | Is adaptive stopping working? (Should vary by difficulty) |
| Energy at stopping point | What threshold works? (Calibrate energy_threshold) |
| Per-node energy in tree | Which subproblems are hard? (High energy = uncertain) |
| Factor energy at merge nodes | Are merges coherent? (Low = children compatible) |

---

## Risks / Mitigation

| Risk | Mitigation |
|------|-----------|
| Energy head doesn't correlate with correctness | Start with supervised contrastive loss; don't go adaptive until correlation is strong |
| Adaptive K makes training unstable (variable-length computation graphs) | Cap max_K=6; pad shorter loops to fixed length for batching |
| Lyapunov regularization is too strict, prevents exploration | Use soft penalty (hinge loss) not hard constraint; weight low (0.01) |
| Energy function is easy to game (assign low energy to everything) | Contrastive loss with margin prevents flat energy surfaces |
| Variable K breaks batching efficiency | Within a batch, use max K across batch, mask out stopped examples |

---

## What NOT to Do

- **Don't start from random noise.** Llama's comprehension pass gives a much better starting point than LeCun's Langevin dynamics initialization. The inner loop refines from a warm start, not from noise.
- **Don't implement full EBM inference with MCMC.** The controller's gradient-based refinement is simpler and sufficient. Langevin dynamics adds noise injection which is unnecessary here.
- **Don't normalize the energy into probabilities.** The partition function is intractable and unnecessary. Raw energy values with contrastive training are enough.
- **Don't add this before the controller is alive.** If scale_xproblem_cos is still 1.0, the energy head will also be constant. Fix the controller first.

---

## Implementation Priority

1. Add energy head to controller (tiny — one linear layer)
2. Add contrastive energy loss to training
3. Verify energy correlates with correctness (diagnostic)
4. Add Lyapunov regularization on inner loop energy trajectory
5. Implement adaptive K stopping criterion
6. Extend to tree-level energy (after tree structure is validated)

---

## Summary

The inner loop is already performing energy minimization implicitly. This design makes it explicit: train the confidence head as an energy function, use it to adaptively stop the inner loop, and regularize it with Lyapunov-style monotonic decrease. The tree structure naturally maps to LeCun's factor graph — variable nodes are subproblem answers, factor nodes are merge constraints, and total energy measures tree coherence. This unifies the breathing loop, tree structure, energy landscape (v7), and adaptive compute into a single framework.
