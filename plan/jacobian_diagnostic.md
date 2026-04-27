# Handoff: Jacobian Diagnostic for the Thinking Loop

## One-Sentence Summary

Compute the Jacobian ∂page_{k+1}/∂page_k to inspect whether the bottleneck carries structured reasoning or whether the loop has collapsed. The Jacobian's sparsity reveals information routing, its block structure reveals frequency bands, and its eigenvalues reveal loop stability. Standard dynamical systems analysis applied to our recurrent thinking architecture.

---

## What the Jacobian Tells Us

The Jacobian is a 64×64 matrix: entry (i,j) = "how much does dimension i of the next page change when I nudge dimension j of the current page?"

```
J[i][j] = ∂page_{k+1}[i] / ∂page_k[j]
```

Three failure modes are directly visible:

```
FIXED-POINT COLLAPSE:
  J ≈ 0 everywhere
  Pages don't influence each other. The loop is a no-op.
  Nudging any dimension of page_k has no effect on page_{k+1}.
  Diagnosis: "the thinking loop is dead"

NO STRUCTURE:
  J ≈ 0.1 everywhere (uniform)
  Everything influences everything equally. No specialization.
  The bottleneck is mush — no information routing.
  Diagnosis: "the thinking loop is alive but unstructured"

HEALTHY REASONING:
  J is sparse with strong entries at specific (i,j) pairs
  Specific page dimensions drive specific downstream dimensions.
  "Dimension 17 (encodes a number) drives dimension 5 (encodes a computation)"
  Diagnosis: "the thinking loop has structured information flow"
```

---

## Implementation

### Core: Compute the Jacobian

```python
import torch

def compute_page_jacobian(model, problem_ids, state_pages, pass_num):
    """
    Compute ∂page_{k+1} / ∂page_k.
    
    Returns: (64, 64) Jacobian matrix
    
    For a tiny change in each dimension of the current page,
    how does the next page change?
    """
    page_size = 64
    
    # Detach the current page and enable gradients
    current_page = state_pages[-1].detach().clone().requires_grad_(True)
    pages_with_grad = [p.detach() for p in state_pages[:-1]] + [current_page]
    
    # Forward: compute the next page from the current state
    # This runs: pages → hypernetwork → atom scales → LoRA → Llama → perceiver → next page
    next_page = model.think_one_pass(problem_ids, pages_with_grad, pass_num)
    # next_page: (batch=1, 64)
    
    # Compute Jacobian row by row
    jacobian = torch.zeros(page_size, page_size, device=next_page.device)
    
    for i in range(page_size):
        # Gradient of next_page[i] with respect to current_page
        grad = torch.autograd.grad(
            outputs=next_page[0, i],
            inputs=current_page,
            retain_graph=True,
            create_graph=False,
        )[0]  # (1, 64)
        
        jacobian[i] = grad[0]
    
    return jacobian.detach()
```

### Efficient Batched Version

```python
def compute_page_jacobian_batched(model, problem_ids, state_pages, pass_num):
    """
    Faster Jacobian using torch.autograd.functional.jacobian.
    Requires wrapping the forward pass as a function of the page.
    """
    current_page = state_pages[-1].detach().clone()
    frozen_pages = [p.detach() for p in state_pages[:-1]]
    
    def forward_fn(page):
        """Map current page → next page."""
        pages = frozen_pages + [page.unsqueeze(0)]  # add batch dim
        next_page = model.think_one_pass(problem_ids, pages, pass_num)
        return next_page.squeeze(0)  # remove batch dim
    
    # PyTorch computes the full Jacobian
    J = torch.autograd.functional.jacobian(forward_fn, current_page.squeeze(0))
    # J: (64, 64)
    
    return J.detach()
```

---

## Diagnostic Metrics

### 1. Sparsity (Information Routing)

```python
def jacobian_sparsity(J, threshold=0.01):
    """
    What fraction of entries are near-zero?
    High sparsity = structured routing (specific dims drive specific dims).
    Low sparsity = uniform (everything drives everything).
    """
    return (J.abs() < threshold).float().mean().item()

# Healthy: >80% sparse (most connections inactive, few strong paths)
# Collapsed: ~100% sparse (all connections dead)
# Unstructured: <50% sparse (dense, no specialization)
```

### 2. Spectral Radius (Loop Stability)

```python
def jacobian_stability(J):
    """
    Eigenvalue analysis of the Jacobian.
    Spectral radius (max |eigenvalue|) determines loop behavior.
    """
    eigenvalues = torch.linalg.eigvals(J)
    magnitudes = eigenvalues.abs()
    
    spectral_radius = magnitudes.max().item()
    mean_magnitude = magnitudes.mean().item()
    
    # How many eigenvalues are in the stable range?
    stable_count = ((magnitudes > 0.5) & (magnitudes < 1.5)).sum().item()
    contracting_count = (magnitudes < 0.1).sum().item()
    expanding_count = (magnitudes > 2.0).sum().item()
    
    return {
        'spectral_radius': spectral_radius,
        'mean_eigenvalue_magnitude': mean_magnitude,
        'stable_eigenvalues': stable_count,      # |λ| ∈ [0.5, 1.5]
        'contracting_eigenvalues': contracting_count,  # |λ| < 0.1
        'expanding_eigenvalues': expanding_count,  # |λ| > 2.0
    }

# Healthy: spectral_radius ≈ 0.8-1.2, most eigenvalues stable
# Collapsing: spectral_radius << 1, most eigenvalues contracting
# Diverging: spectral_radius >> 1, some eigenvalues expanding
```

### 3. Block Diagonal Ratio (Frequency Band Structure)

```python
def jacobian_frequency_structure(J, num_bands=4):
    """
    Does the Jacobian have block-diagonal structure?
    If the pi-harmonic encoding works, frequency bands should be somewhat independent:
    low-freq dims drive low-freq dims, high-freq dims drive high-freq dims.
    """
    page_size = J.size(0)
    band_size = page_size // num_bands
    
    within_band_energy = 0.0
    total_energy = J.abs().sum().item()
    
    for b in range(num_bands):
        start = b * band_size
        end = (b + 1) * band_size
        block = J[start:end, start:end]
        within_band_energy += block.abs().sum().item()
    
    # Ratio: what fraction of Jacobian energy is within frequency bands?
    band_ratio = within_band_energy / (total_energy + 1e-8)
    
    return band_ratio

# Random matrix: band_ratio ≈ 1/num_bands = 0.25 (no structure)
# Perfect block diagonal: band_ratio = 1.0 (complete band independence)
# Pi-harmonic working: band_ratio > 0.4 (some band structure)
```

### 4. Information Flow Paths (Top Connections)

```python
def jacobian_top_connections(J, top_k=10):
    """
    Which page dimensions most strongly influence which downstream dimensions?
    Reveals the information routing structure.
    """
    # Flatten and find top-k absolute values
    flat = J.abs().flatten()
    values, indices = flat.topk(top_k)
    
    connections = []
    for val, idx in zip(values, indices):
        i = idx.item() // J.size(1)  # output dim
        j = idx.item() % J.size(1)   # input dim
        connections.append({
            'from_dim': j,
            'to_dim': i,
            'strength': val.item(),
            'from_band': j // (J.size(0) // 4),  # which frequency band
            'to_band': i // (J.size(0) // 4),
        })
    
    return connections

# Example output:
# from_dim=17 → to_dim=5, strength=0.82, band 1→band 0
# "Mid-frequency input dim 17 strongly drives low-frequency output dim 5"
# Interpretation: "a number extracted at pass k drives a computation at pass k+1"
```

---

## Full Diagnostic Script

```python
def diagnose_thinking_loop(model, eval_problems, max_passes=5, num_problems=10):
    """
    Full Jacobian diagnostic across problems and passes.
    Run after training to understand the thinking loop's structure.
    """
    print("=" * 60)
    print("JACOBIAN DIAGNOSTIC: Thinking Loop Structure")
    print("=" * 60)
    
    all_sparsity = []
    all_spectral = []
    all_band_ratio = []
    
    for prob_idx, (problem_ids, gold) in enumerate(eval_problems[:num_problems]):
        state_pages = []
        print(f"\nProblem {prob_idx}: gold={gold}")
        
        for pass_num in range(max_passes):
            page = model.think_one_pass(problem_ids, state_pages, pass_num)
            state_pages.append(page)
            
            if pass_num > 0:
                J = compute_page_jacobian(model, problem_ids, state_pages, pass_num)
                
                sparsity = jacobian_sparsity(J)
                stability = jacobian_stability(J)
                band_ratio = jacobian_frequency_structure(J)
                top_conns = jacobian_top_connections(J, top_k=5)
                
                all_sparsity.append(sparsity)
                all_spectral.append(stability['spectral_radius'])
                all_band_ratio.append(band_ratio)
                
                print(f"  Pass {pass_num-1}→{pass_num}:")
                print(f"    Sparsity:        {sparsity:.1%}")
                print(f"    Spectral radius: {stability['spectral_radius']:.3f}")
                print(f"    Stable eigenvals: {stability['stable_eigenvalues']}/64")
                print(f"    Contract eigenvals: {stability['contracting_eigenvalues']}/64")
                print(f"    Band ratio:      {band_ratio:.3f}")
                print(f"    Top connection:  dim {top_conns[0]['from_dim']}→{top_conns[0]['to_dim']} "
                      f"(strength {top_conns[0]['strength']:.3f}, "
                      f"band {top_conns[0]['from_band']}→{top_conns[0]['to_band']})")
    
    print("\n" + "=" * 60)
    print("SUMMARY ACROSS ALL PROBLEMS AND PASSES")
    print("=" * 60)
    print(f"  Mean sparsity:        {sum(all_sparsity)/len(all_sparsity):.1%}")
    print(f"  Mean spectral radius: {sum(all_spectral)/len(all_spectral):.3f}")
    print(f"  Mean band ratio:      {sum(all_band_ratio)/len(all_band_ratio):.3f}")
    
    # Diagnosis
    mean_sparsity = sum(all_sparsity) / len(all_sparsity)
    mean_spectral = sum(all_spectral) / len(all_spectral)
    mean_band = sum(all_band_ratio) / len(all_band_ratio)
    
    print(f"\nDIAGNOSIS:")
    if mean_sparsity > 0.95:
        print(f"  ⚠ FIXED-POINT COLLAPSE: Jacobian nearly zero. Loop is dead.")
    elif mean_sparsity > 0.8:
        print(f"  ✓ STRUCTURED: Sparse Jacobian with specific routing paths.")
    elif mean_sparsity > 0.5:
        print(f"  ~ MODERATE: Some structure but also diffuse connections.")
    else:
        print(f"  ⚠ UNSTRUCTURED: Dense Jacobian. No specialization.")
    
    if mean_spectral < 0.3:
        print(f"  ⚠ CONTRACTING: Loop losing information each cycle.")
    elif mean_spectral < 1.5:
        print(f"  ✓ STABLE: Information preserved across cycles.")
    else:
        print(f"  ⚠ EXPANDING: Loop amplifying — potentially unstable.")
    
    if mean_band > 0.4:
        print(f"  ✓ FREQUENCY STRUCTURE: Pi-harmonic bands are working.")
    else:
        print(f"  ~ NO BAND STRUCTURE: Frequency encoding not creating independent bands.")


# Usage:
# diagnose_thinking_loop(model, eval_data, max_passes=5, num_problems=10)
# Takes ~30 seconds for 10 problems × 4 pass transitions × 64 backward passes
```

---

## When to Run

```
1. After each level's training completes (L3, L4, L5)
   Compare structure across difficulty levels.
   Expect: L3 sparse + contracting, L5 less sparse + stable.

2. When page_cos = 1.0 (fixed-point collapse suspected)
   Jacobian should confirm: all entries near zero.

3. When accuracy plateaus unexpectedly
   Jacobian reveals whether the bottleneck is the problem.
   Structured but low accuracy → bottleneck is fine, problem is elsewhere.
   Unstructured → bottleneck needs work.

4. After adding new components (wavelet, pi-harmonic, etc.)
   Does the new component change the Jacobian structure?
   Band ratio should increase with pi-harmonic encoding.
```

---

## What to Look For by Level

```
L3 (easy):
  Sparsity: high (>85%) — simple problems need few routing paths
  Spectral radius: <1.0 — loop converges quickly (1-2 passes enough)
  Band ratio: modest (0.3-0.5) — some frequency structure
  Active connections: 5-10 strong paths

L4 (medium):
  Sparsity: moderate (70-85%) — more routing paths needed
  Spectral radius: ≈1.0 — loop is cycling, not converging immediately
  Band ratio: higher (0.4-0.6) — frequency bands more defined
  Active connections: 10-20 strong paths

L5 (GSM8K, hard):
  Sparsity: lower (50-70%) — complex problems use more paths
  Spectral radius: ≈1.0 — loop needs to cycle multiple times
  Band ratio: highest (0.5+) — frequency bands carrying different information
  Active connections: 20-40 strong paths
  
Trend across difficulty: less sparse, more stable, more frequency structure.
```

---

## Connection to Other Diagnostics

```
Jacobian sparsity     ↔  atom activation count (sparse J ↔ few atoms active)
Jacobian eigenvalues  ↔  entropy flow smoothness (stable eigenvalues ↔ smooth flow)
Jacobian band ratio   ↔  pi-harmonic encoding effectiveness
Jacobian top paths    ↔  which atom-page dimension pairs matter most
```

The Jacobian is the DEEPEST diagnostic. It reveals the MECHANICS of the loop — not just "are pages different?" (contrastive) or "are atoms active?" (sparsity) but "how does information actually flow from one cycle to the next?"

---

## Parameter Cost

Zero. The Jacobian is computed from the existing model using autograd. No new parameters, no architecture changes. Pure diagnostic.
