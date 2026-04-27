"""
Jacobian Diagnostic for the Thinking Loop.

Computes the Jacobian dpage_{k+1}/dpage_k to inspect whether the bottleneck
carries structured reasoning or whether the loop has collapsed.

The Jacobian is a 64x64 matrix where entry (i,j) = "how much does dimension i
of the next page change when I nudge dimension j of the current page?"

Three failure modes are directly visible:
- FIXED-POINT COLLAPSE: J ~ 0 everywhere (pages don't influence each other)
- NO STRUCTURE: J ~ uniform (everything influences everything, no specialization)
- HEALTHY REASONING: J is sparse with strong entries at specific (i,j) pairs

Usage:
  python scripts/diag_jacobian.py --checkpoint checkpoints/atom_lora_L3_best.pt --level L3
  python scripts/diag_jacobian.py --checkpoint checkpoints/atom_lora_gsm8k_best.pt --level L5 --num_problems 10

API usage:
    from scripts.diag_jacobian import (
        compute_page_jacobian,
        compute_page_jacobian_batched,
        jacobian_sparsity,
        jacobian_stability,
        jacobian_frequency_structure,
        jacobian_top_connections,
        diagnose_thinking_loop,
    )

    # After computing a Jacobian J (64x64 matrix):
    sparsity = jacobian_sparsity(J)  # fraction of near-zero entries
    stability = jacobian_stability(J)  # eigenvalue analysis dict
    band_ratio = jacobian_frequency_structure(J)  # block-diagonal structure
    connections = jacobian_top_connections(J)  # top information paths
"""
import argparse
import json
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/home/ubuntu/mycelium')

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from typing import Dict, List, Any, Tuple, Optional

from scripts.atom_lora import AtomLoRAModel, AtomAdditiveLoRAManager, AnswerHead


def jacobian_sparsity(J: torch.Tensor, threshold: float = 0.01) -> float:
    """
    Compute the sparsity of the Jacobian matrix.

    Sparsity measures what fraction of Jacobian entries are near-zero (below
    the threshold). This reveals information routing structure:

    - High sparsity (>80%): Structured routing. Specific page dimensions drive
      specific downstream dimensions. The bottleneck is doing its job - forcing
      the model to route information through narrow paths.

    - Low sparsity (<50%): Unstructured. Everything influences everything.
      No specialization, no information routing. The bottleneck is mush.

    - Very high sparsity (>95%): Possible fixed-point collapse. The thinking
      loop may be dead - pages don't influence each other.

    Args:
        J: Jacobian matrix of shape (page_size, page_size), typically (64, 64).
           Entry J[i,j] = d(page_{k+1}[i]) / d(page_k[j]).
        threshold: Absolute value below which an entry is considered "near-zero".
                  Default 0.01.

    Returns:
        Sparsity as a float in [0, 1].
        1.0 = all entries near-zero (collapsed).
        0.0 = no entries near-zero (dense).

    Interpretation guide:
        >95%: COLLAPSED - loop is dead, pages don't influence each other
        80-95%: STRUCTURED - healthy, specific routing paths
        50-80%: MODERATE - some structure, some diffuse connections
        <50%: UNSTRUCTURED - dense, no specialization
    """
    return (J.abs() < threshold).float().mean().item()


def jacobian_stability(J: torch.Tensor) -> Dict[str, Any]:
    """
    Eigenvalue analysis of the Jacobian for loop stability.

    The spectral radius (maximum eigenvalue magnitude) determines whether the
    thinking loop is stable, contracting, or expanding:

    - Spectral radius < 1: Contracting loop. Information decays each cycle.
      Early cycles' influence fades away. May be fine for easy problems
      (converge quickly) but problematic for hard problems (lose early thinking).

    - Spectral radius ~ 1: Stable loop. Information is preserved across cycles.
      Ideal for complex reasoning that needs multiple passes.

    - Spectral radius > 1: Expanding loop. Small perturbations grow.
      Potentially unstable, but may indicate the model is amplifying
      important signals. Watch for exploding values.

    The eigenvalue DISTRIBUTION also matters:
    - Many stable eigenvalues (|lambda| in [0.5, 1.5]): Healthy information flow
    - Many contracting eigenvalues (|lambda| < 0.1): Dimensions being crushed
    - Any expanding eigenvalues (|lambda| > 2.0): Potential instability

    Args:
        J: Jacobian matrix of shape (page_size, page_size), typically (64, 64).

    Returns:
        Dictionary with keys:
            - 'spectral_radius': float, max |eigenvalue|
            - 'mean_eigenvalue_magnitude': float, average |eigenvalue|
            - 'stable_eigenvalues': int, count with |lambda| in [0.5, 1.5]
            - 'contracting_eigenvalues': int, count with |lambda| < 0.1
            - 'expanding_eigenvalues': int, count with |lambda| > 2.0

    Interpretation guide:
        spectral_radius < 0.3: CONTRACTING - loop losing info each cycle
        spectral_radius in [0.8, 1.2]: STABLE - healthy information preservation
        spectral_radius > 1.5: EXPANDING - watch for instability
    """
    # Compute eigenvalues (may be complex)
    eigenvalues = torch.linalg.eigvals(J)
    magnitudes = eigenvalues.abs()

    spectral_radius = magnitudes.max().item()
    mean_magnitude = magnitudes.mean().item()

    # Count eigenvalues in different stability regimes
    stable_count = ((magnitudes > 0.5) & (magnitudes < 1.5)).sum().item()
    contracting_count = (magnitudes < 0.1).sum().item()
    expanding_count = (magnitudes > 2.0).sum().item()

    return {
        'spectral_radius': spectral_radius,
        'mean_eigenvalue_magnitude': mean_magnitude,
        'stable_eigenvalues': int(stable_count),
        'contracting_eigenvalues': int(contracting_count),
        'expanding_eigenvalues': int(expanding_count),
    }


def jacobian_frequency_structure(J: torch.Tensor, num_bands: int = 4) -> float:
    """
    Measure block-diagonal structure in the Jacobian (frequency band analysis).

    If the pi-harmonic encoding works correctly, the page dimensions should
    organize into frequency bands that are somewhat independent:
    - Low-frequency dims (0-15): coarse info (problem type, magnitude)
    - Mid-frequency dims (16-40): key operations (numbers, operations)
    - High-frequency dims (40-63): precise details (exact values, corrections)

    Block-diagonal structure means low-freq dims primarily drive low-freq dims,
    high-freq dims primarily drive high-freq dims. Cross-band connections are
    weaker than within-band connections.

    The band ratio measures what fraction of Jacobian energy is within
    frequency bands vs between them:

    Args:
        J: Jacobian matrix of shape (page_size, page_size), typically (64, 64).
        num_bands: Number of frequency bands to divide the page into.
                  Default 4 (16 dims per band for 64-dim pages).

    Returns:
        Band ratio as a float in [0, 1].
        - 0.25 (for num_bands=4): No structure, random matrix
        - 1.0: Perfect block-diagonal (complete band independence)
        - >0.4: Some frequency band structure (pi-harmonic working)

    Interpretation guide:
        ~0.25: NO STRUCTURE - random, pi-harmonic not helping
        0.3-0.4: WEAK STRUCTURE - some band organization emerging
        0.4-0.6: MODERATE STRUCTURE - frequency bands somewhat independent
        >0.6: STRONG STRUCTURE - clear frequency band separation
    """
    page_size = J.size(0)
    band_size = page_size // num_bands

    within_band_energy = 0.0
    total_energy = J.abs().sum().item()

    if total_energy < 1e-8:
        # Jacobian is essentially zero - collapsed
        return 0.0

    for b in range(num_bands):
        start = b * band_size
        end = (b + 1) * band_size
        block = J[start:end, start:end]
        within_band_energy += block.abs().sum().item()

    band_ratio = within_band_energy / total_energy
    return band_ratio


def jacobian_top_connections(J: torch.Tensor, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Find the strongest information flow paths in the Jacobian.

    Returns the top-k strongest connections: which input dimensions most
    strongly influence which output dimensions. This reveals the information
    routing structure - what the model considers important enough to route
    through the bottleneck.

    Each connection includes:
    - from_dim: which page dimension in pass k (input)
    - to_dim: which page dimension in pass k+1 (output)
    - strength: absolute value of the Jacobian entry
    - from_band: which frequency band the input is in (0=lowest freq)
    - to_band: which frequency band the output is in

    Example interpretation:
        from_dim=17 -> to_dim=5, strength=0.82, band 1->band 0
        "Mid-frequency dim 17 strongly drives low-frequency dim 5"
        Possible meaning: "a number extracted at pass k drives a high-level
        computation decision at pass k+1"

    Args:
        J: Jacobian matrix of shape (page_size, page_size), typically (64, 64).
        top_k: Number of top connections to return. Default 10.

    Returns:
        List of dicts, each with keys:
            - 'from_dim': int, input dimension index
            - 'to_dim': int, output dimension index
            - 'strength': float, |J[to_dim, from_dim]|
            - 'from_band': int, frequency band of input (0 = lowest)
            - 'to_band': int, frequency band of output
    """
    page_size = J.size(0)
    num_bands = 4
    band_size = page_size // num_bands

    # Flatten and find top-k absolute values
    flat = J.abs().flatten()
    values, indices = flat.topk(min(top_k, flat.numel()))

    connections = []
    for val, idx in zip(values, indices):
        i = idx.item() // J.size(1)  # output dim (row)
        j = idx.item() % J.size(1)   # input dim (column)
        connections.append({
            'from_dim': j,
            'to_dim': i,
            'strength': val.item(),
            'from_band': j // band_size,
            'to_band': i // band_size,
        })

    return connections


def compute_page_jacobian(
    model: AtomLoRAModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    state_pages: List[torch.Tensor],
    pass_num: int,
    prev_mid_states: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Compute dpage_{k+1} / dpage_k for a single thinking pass.

    For a tiny change in each dimension of the current page,
    how does the next page change?

    The computation chain is:
        current_page -> hypernetwork -> atom_scales -> LoRA -> Llama -> perceiver -> next_page

    The Jacobian captures the full end-to-end gradient through this chain.

    Args:
        model: AtomLoRAModel instance.
        input_ids: Token IDs for the problem, shape (1, seq_len).
        attention_mask: Attention mask, shape (1, seq_len).
        state_pages: List of page tensors from previous passes.
                    The last page is the one we differentiate w.r.t.
        pass_num: Current pass number (0-indexed) for the NEXT pass.
        prev_mid_states: Optional list of previous mid-layer states for skip connection.

    Returns:
        Jacobian matrix of shape (page_size, page_size), typically (64, 64).
        Entry J[i,j] = d(next_page[i]) / d(current_page[j]).
    """
    if len(state_pages) == 0:
        raise ValueError("Need at least one page to compute Jacobian")

    page_size = model.page_size
    device = input_ids.device

    # Detach the current (last) page and enable gradients
    current_page = state_pages[-1].detach().clone().requires_grad_(True)
    pages_with_grad = [p.detach() for p in state_pages[:-1]] + [current_page]

    # Forward pass: compute the next page from the current state
    # Generate atom scales from pages + pass number
    atom_scales = model.hypernet(pages_with_grad, pass_num)

    # Apply atom LoRA via monkey-patching
    manager = AtomAdditiveLoRAManager(model.transformer)
    manager.apply(model.atoms, atom_scales)

    try:
        outputs = model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = list(outputs.hidden_states[1:])
    finally:
        manager.remove()

    # Compress all 16 layers -> page delta (strategy discarded)
    page_delta, _strategy, _mid = model.compressor(
        hidden_states, pass_num, prev_mid_states=prev_mid_states,
    )

    # Normalize on hypersphere
    next_page = F.normalize(page_delta, dim=-1) * model.page_radius

    # Add Fourier structural identity (after normalization)
    next_page = model.fourier_page.apply(next_page, pass_num)

    # v24.8: Apply residual gate blending with current_page
    # This is the KEY step that should shift eigenvalues by ~0.5
    if hasattr(model, 'residual_gate') and model.residual_gate is not None:
        next_page = model.residual_gate(next_page, current_page)

    # Compute Jacobian row by row
    jacobian = torch.zeros(page_size, page_size, device=device)

    for i in range(page_size):
        # Gradient of next_page[i] with respect to current_page
        grad = torch.autograd.grad(
            outputs=next_page[0, i],
            inputs=current_page,
            retain_graph=True,
            create_graph=False,
        )[0]  # (1, page_size)

        jacobian[i] = grad[0]

    return jacobian.detach()


def compute_page_jacobian_batched(
    model: AtomLoRAModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    state_pages: List[torch.Tensor],
    pass_num: int,
    prev_mid_states: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Faster Jacobian computation using torch.autograd.functional.jacobian.

    This is more efficient than row-by-row computation when the forward
    pass is expensive. Uses PyTorch's built-in vectorized Jacobian computation.

    Args:
        model: AtomLoRAModel instance.
        input_ids: Token IDs for the problem, shape (1, seq_len).
        attention_mask: Attention mask, shape (1, seq_len).
        state_pages: List of page tensors from previous passes.
        pass_num: Current pass number (0-indexed) for the NEXT pass.
        prev_mid_states: Optional list of previous mid-layer states for skip connection.

    Returns:
        Jacobian matrix of shape (page_size, page_size), typically (64, 64).
    """
    if len(state_pages) == 0:
        raise ValueError("Need at least one page to compute Jacobian")

    current_page = state_pages[-1].detach().clone()
    frozen_pages = [p.detach() for p in state_pages[:-1]]

    def forward_fn(page):
        """Map current page -> next page."""
        # Add batch dim
        page_batched = page.unsqueeze(0)  # (1, page_size)
        pages = frozen_pages + [page_batched]

        # Generate atom scales
        atom_scales = model.hypernet(pages, pass_num)

        # Apply LoRA and run transformer
        manager = AtomAdditiveLoRAManager(model.transformer)
        manager.apply(model.atoms, atom_scales)

        try:
            outputs = model.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = list(outputs.hidden_states[1:])
        finally:
            manager.remove()

        # Compress to page delta
        page_delta, _strategy, _mid = model.compressor(
            hidden_states, pass_num, prev_mid_states=prev_mid_states,
        )

        # Normalize and apply Fourier encoding
        next_page = F.normalize(page_delta, dim=-1) * model.page_radius
        next_page = model.fourier_page.apply(next_page, pass_num)

        # v24.8: Apply residual gate blending with current_page
        if hasattr(model, 'residual_gate') and model.residual_gate is not None:
            next_page = model.residual_gate(next_page, page_batched)

        return next_page.squeeze(0)  # remove batch dim

    # PyTorch computes the full Jacobian
    J = torch.autograd.functional.jacobian(forward_fn, current_page.squeeze(0))
    # J: (page_size, page_size)

    return J.detach()


def diagnose_thinking_loop(
    model: AtomLoRAModel,
    dataset,
    device,
    max_passes: int = 5,
    num_problems: int = 10,
    use_batched: bool = False,
) -> Dict[str, Any]:
    """
    Full Jacobian diagnostic across problems and passes.

    Run this after training to understand the thinking loop's structure.
    Prints detailed diagnostics and returns summary statistics.

    Args:
        model: AtomLoRAModel instance.
        dataset: Evaluation dataset with 'problem' and 'answer' keys.
        device: Torch device.
        max_passes: Number of thinking passes to analyze. Default 5.
        num_problems: Number of problems to analyze. Default 10.
        use_batched: If True, use batched Jacobian (faster but more memory).

    Returns:
        Dictionary with:
            - 'sparsity': list of sparsity values
            - 'spectral_radius': list of spectral radii
            - 'band_ratio': list of band ratios
            - 'jacobians': list of Jacobian matrices (numpy)
            - 'mean_sparsity': average sparsity
            - 'mean_spectral': average spectral radius
            - 'mean_band_ratio': average band ratio

    Interpretation guide by level:

        L3 (easy):
            Sparsity: high (>85%) - simple problems need few routing paths
            Spectral radius: <1.0 - loop converges quickly
            Band ratio: modest (0.3-0.5) - some frequency structure

        L4 (medium):
            Sparsity: moderate (70-85%) - more routing paths needed
            Spectral radius: ~1.0 - loop is cycling, not converging immediately
            Band ratio: higher (0.4-0.6) - frequency bands more defined

        L5 (GSM8K, hard):
            Sparsity: lower (50-70%) - complex problems use more paths
            Spectral radius: ~1.0 - loop needs to cycle multiple times
            Band ratio: highest (0.5+) - frequency bands carrying different info

    Trend across difficulty: less sparse, more stable, more frequency structure.
    """
    print("=" * 60)
    print("JACOBIAN DIAGNOSTIC: Thinking Loop Structure")
    print("=" * 60)

    compute_fn = compute_page_jacobian_batched if use_batched else compute_page_jacobian

    all_sparsity = []
    all_spectral = []
    all_band_ratio = []
    all_jacobians = []
    max_length = 128

    model.eval()

    for prob_idx in range(min(num_problems, len(dataset))):
        sample = dataset[prob_idx]
        problem = sample['problem']
        gold = sample.get('answer', 'N/A')

        # Tokenize
        inputs = model.tokenizer(
            [problem], return_tensors='pt', padding=True,
            truncation=True, max_length=max_length,
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        state_pages = []
        mid_states_history = []

        print(f"\nProblem {prob_idx}: gold={gold}")

        for pass_num in range(max_passes):
            # Run thinking pass
            with torch.no_grad():
                page, atom_scales, current_mid_states = model.thinking_pass(
                    input_ids, attention_mask, state_pages, pass_num,
                    prev_mid_states=mid_states_history if mid_states_history else None,
                )
            state_pages.append(page.detach())
            mid_states_history.append(current_mid_states)

            if pass_num > 0:
                # Compute Jacobian for transition from pass (pass_num-1) to pass_num
                # We use the pages up to pass_num-1 and compute how pass_num depends on them
                pages_for_jacobian = state_pages[:-1]

                J = compute_fn(
                    model, input_ids, attention_mask,
                    pages_for_jacobian, pass_num,
                    prev_mid_states=mid_states_history[:-1] if len(mid_states_history) > 1 else None,
                )

                sparsity = jacobian_sparsity(J)
                stability = jacobian_stability(J)
                band_ratio = jacobian_frequency_structure(J)
                top_conns = jacobian_top_connections(J, top_k=5)

                all_sparsity.append(sparsity)
                all_spectral.append(stability['spectral_radius'])
                all_band_ratio.append(band_ratio)
                all_jacobians.append(J.cpu().numpy())

                print(f"  Pass {pass_num-1}->{pass_num}:")
                print(f"    Sparsity:        {sparsity:.1%}")
                print(f"    Spectral radius: {stability['spectral_radius']:.3f}")
                print(f"    Stable eigenvals: {stability['stable_eigenvalues']}/64")
                print(f"    Contract eigenvals: {stability['contracting_eigenvalues']}/64")
                print(f"    Band ratio:      {band_ratio:.3f}")
                print(f"    Top connection:  dim {top_conns[0]['from_dim']}->{top_conns[0]['to_dim']} "
                      f"(strength {top_conns[0]['strength']:.3f}, "
                      f"band {top_conns[0]['from_band']}->{top_conns[0]['to_band']})")

    print("\n" + "=" * 60)
    print("SUMMARY ACROSS ALL PROBLEMS AND PASSES")
    print("=" * 60)

    mean_sparsity = sum(all_sparsity) / len(all_sparsity) if all_sparsity else 0
    mean_spectral = sum(all_spectral) / len(all_spectral) if all_spectral else 0
    mean_band = sum(all_band_ratio) / len(all_band_ratio) if all_band_ratio else 0

    print(f"  Mean sparsity:        {mean_sparsity:.1%}")
    print(f"  Mean spectral radius: {mean_spectral:.3f}")
    print(f"  Mean band ratio:      {mean_band:.3f}")

    # Diagnosis
    print(f"\nDIAGNOSIS:")
    if mean_sparsity > 0.95:
        print(f"  [!] FIXED-POINT COLLAPSE: Jacobian nearly zero. Loop is dead.")
    elif mean_sparsity > 0.8:
        print(f"  [+] STRUCTURED: Sparse Jacobian with specific routing paths.")
    elif mean_sparsity > 0.5:
        print(f"  [~] MODERATE: Some structure but also diffuse connections.")
    else:
        print(f"  [!] UNSTRUCTURED: Dense Jacobian. No specialization.")

    if mean_spectral < 0.3:
        print(f"  [!] CONTRACTING: Loop losing information each cycle.")
    elif mean_spectral < 1.5:
        print(f"  [+] STABLE: Information preserved across cycles.")
    else:
        print(f"  [!] EXPANDING: Loop amplifying - potentially unstable.")

    if mean_band > 0.4:
        print(f"  [+] FREQUENCY STRUCTURE: Pi-harmonic bands are working.")
    else:
        print(f"  [~] NO BAND STRUCTURE: Frequency encoding not creating independent bands.")

    return {
        'sparsity': all_sparsity,
        'spectral_radius': all_spectral,
        'band_ratio': all_band_ratio,
        'jacobians': all_jacobians,
        'mean_sparsity': mean_sparsity,
        'mean_spectral': mean_spectral,
        'mean_band_ratio': mean_band,
    }


def plot_jacobian_summary(results: Dict[str, Any], output_path: str):
    """Plot Jacobian diagnostic summary.

    Args:
        results: dict from diagnose_thinking_loop
        output_path: where to save PNG
    """
    jacobians = results['jacobians']
    if not jacobians:
        print("No Jacobians to plot")
        return

    n_jacobians = len(jacobians)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    title = f"Jacobian Diagnostic ({n_jacobians} transitions)\n"
    title += f"Mean sparsity: {results['mean_sparsity']:.1%} | "
    title += f"Mean spectral radius: {results['mean_spectral']:.3f} | "
    title += f"Mean band ratio: {results['mean_band_ratio']:.3f}"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Top-left: Mean Jacobian heatmap
    ax = axes[0, 0]
    mean_J = np.mean(jacobians, axis=0)
    im = ax.imshow(mean_J, cmap='RdBu_r', aspect='equal',
                   vmin=-np.abs(mean_J).max(), vmax=np.abs(mean_J).max())
    ax.set_xlabel('Input dimension (page_k)')
    ax.set_ylabel('Output dimension (page_{k+1})')
    ax.set_title('Mean Jacobian')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Top-middle: Absolute Jacobian (shows magnitude of influence)
    ax = axes[0, 1]
    abs_J = np.mean(np.abs(jacobians), axis=0)
    im2 = ax.imshow(abs_J, cmap='hot', aspect='equal')
    ax.set_xlabel('Input dimension')
    ax.set_ylabel('Output dimension')
    ax.set_title('Mean |Jacobian| (influence strength)')
    plt.colorbar(im2, ax=ax, shrink=0.8)

    # Top-right: Sparsity by frequency band (4x4 block structure)
    ax = axes[0, 2]
    band_structure = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            block = abs_J[i*16:(i+1)*16, j*16:(j+1)*16]
            band_structure[i, j] = block.mean()
    im3 = ax.imshow(band_structure, cmap='hot', aspect='equal')
    ax.set_xlabel('Input band (frequency)')
    ax.set_ylabel('Output band (frequency)')
    ax.set_title('Band-level structure (16-dim bands)')
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Low', 'Mid-low', 'Mid-high', 'High'])
    ax.set_yticks(range(4))
    ax.set_yticklabels(['Low', 'Mid-low', 'Mid-high', 'High'])
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{band_structure[i, j]:.3f}', ha='center', va='center',
                    fontsize=10, color='white' if band_structure[i, j] > band_structure.max()/2 else 'black')
    plt.colorbar(im3, ax=ax, shrink=0.8)

    # Bottom-left: Sparsity histogram across transitions
    ax = axes[1, 0]
    ax.hist(results['sparsity'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(results['mean_sparsity'], color='red', linestyle='--',
               label=f'mean={results["mean_sparsity"]:.2f}')
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Count')
    ax.set_title('Sparsity distribution')
    ax.legend()

    # Bottom-middle: Spectral radius histogram
    ax = axes[1, 1]
    ax.hist(results['spectral_radius'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(results['mean_spectral'], color='red', linestyle='--',
               label=f'mean={results["mean_spectral"]:.2f}')
    ax.axvline(1.0, color='green', linestyle=':', label='stable=1.0')
    ax.set_xlabel('Spectral radius')
    ax.set_ylabel('Count')
    ax.set_title('Spectral radius distribution')
    ax.legend()

    # Bottom-right: Band ratio histogram
    ax = axes[1, 2]
    ax.hist(results['band_ratio'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(results['mean_band_ratio'], color='red', linestyle='--',
               label=f'mean={results["mean_band_ratio"]:.2f}')
    ax.axvline(0.25, color='orange', linestyle=':', label='random=0.25')
    ax.set_xlabel('Band ratio')
    ax.set_ylabel('Count')
    ax.set_title('Band ratio distribution (higher = more structure)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved Jacobian diagnostic plot to {output_path}")


# ============================================================================
# Standalone test (no checkpoint needed)
# ============================================================================

def test_jacobian_standalone():
    """Simple test that can run without a checkpoint.

    Tests all the metric functions on synthetic Jacobian matrices.
    """
    print("=" * 60)
    print("JACOBIAN DIAGNOSTIC: Standalone Test")
    print("=" * 60)

    torch.manual_seed(42)
    page_size = 64

    # Test 1: Random Jacobian (should be ~unstructured)
    print("\nTest 1: Random Jacobian")
    print("-" * 40)
    J_random = torch.randn(page_size, page_size) * 0.1

    sparsity = jacobian_sparsity(J_random)
    print(f"  Sparsity (threshold=0.01): {sparsity:.1%}")
    assert 0 <= sparsity <= 1, "Sparsity should be in [0, 1]"

    stability = jacobian_stability(J_random)
    print(f"  Spectral radius: {stability['spectral_radius']:.3f}")
    print(f"  Stable eigenvalues: {stability['stable_eigenvalues']}/64")
    print(f"  Contracting eigenvalues: {stability['contracting_eigenvalues']}/64")
    print(f"  Expanding eigenvalues: {stability['expanding_eigenvalues']}/64")

    band_ratio = jacobian_frequency_structure(J_random)
    print(f"  Band ratio: {band_ratio:.3f} (expected ~0.25 for random)")
    assert 0 <= band_ratio <= 1, "Band ratio should be in [0, 1]"

    connections = jacobian_top_connections(J_random, top_k=3)
    print(f"  Top 3 connections:")
    for c in connections:
        print(f"    dim {c['from_dim']}->{c['to_dim']}: {c['strength']:.3f} (band {c['from_band']}->{c['to_band']})")

    # Test 2: Sparse Jacobian (should be >99% sparse)
    print("\nTest 2: Sparse Jacobian")
    print("-" * 40)
    J_sparse = torch.zeros(page_size, page_size)
    J_sparse[0, 0] = 1.0
    J_sparse[10, 5] = 0.8
    J_sparse[32, 17] = -0.5

    sparsity = jacobian_sparsity(J_sparse)
    print(f"  Sparsity: {sparsity:.1%} (expected >99%)")
    assert sparsity > 0.99, f"Sparse Jacobian should be >99% sparse, got {sparsity:.1%}"

    connections = jacobian_top_connections(J_sparse, top_k=3)
    print(f"  Top connections: {[(c['from_dim'], c['to_dim']) for c in connections]}")
    assert connections[0]['from_dim'] == 0 and connections[0]['to_dim'] == 0, "Should find (0,0) first"

    # Test 3: Block-diagonal Jacobian (should have high band ratio)
    print("\nTest 3: Block-diagonal Jacobian")
    print("-" * 40)
    J_block = torch.zeros(page_size, page_size)
    for b in range(4):
        start = b * 16
        end = (b + 1) * 16
        J_block[start:end, start:end] = torch.randn(16, 16) * 0.5

    band_ratio = jacobian_frequency_structure(J_block, num_bands=4)
    print(f"  Band ratio: {band_ratio:.3f} (expected ~1.0 for perfect block-diagonal)")
    assert band_ratio > 0.9, f"Block-diagonal should have band ratio >0.9, got {band_ratio:.3f}"

    # Test 4: Collapsed Jacobian (should be 100% sparse)
    print("\nTest 4: Collapsed (zero) Jacobian")
    print("-" * 40)
    J_zero = torch.zeros(page_size, page_size)

    sparsity = jacobian_sparsity(J_zero)
    print(f"  Sparsity: {sparsity:.1%} (expected 100%)")
    assert sparsity == 1.0, "Zero Jacobian should be 100% sparse"

    stability = jacobian_stability(J_zero)
    print(f"  Spectral radius: {stability['spectral_radius']:.6f} (expected ~0)")

    band_ratio = jacobian_frequency_structure(J_zero)
    print(f"  Band ratio: {band_ratio:.3f} (expected 0 for zero matrix)")

    # Test 5: Eigenvalue stability test
    print("\nTest 5: Eigenvalue stability regimes")
    print("-" * 40)

    # Identity-like (stable)
    J_stable = torch.eye(page_size) * 0.9
    stability = jacobian_stability(J_stable)
    print(f"  0.9 * I: spectral_radius={stability['spectral_radius']:.3f}, stable_count={stability['stable_eigenvalues']}")
    assert stability['spectral_radius'] < 1.0, "Should be contracting"

    # Expanding
    J_expand = torch.eye(page_size) * 2.5
    stability = jacobian_stability(J_expand)
    print(f"  2.5 * I: spectral_radius={stability['spectral_radius']:.3f}, expanding_count={stability['expanding_eigenvalues']}")
    assert stability['expanding_eigenvalues'] == 64, "All eigenvalues should be expanding"

    # Contracting
    J_contract = torch.eye(page_size) * 0.05
    stability = jacobian_stability(J_contract)
    print(f"  0.05 * I: spectral_radius={stability['spectral_radius']:.3f}, contracting_count={stability['contracting_eigenvalues']}")
    assert stability['contracting_eigenvalues'] == 64, "All eigenvalues should be contracting"

    print("\n" + "=" * 60)
    print("All standalone tests PASSED!")
    print("=" * 60)


# ============================================================================
# CLI main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Jacobian diagnostic for thinking loop structure analysis',
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to atom LoRA checkpoint (if None, runs standalone test)')
    parser.add_argument('--level', type=str, default='L3',
                        choices=['L3', 'L4', 'L4.5', 'L4.7', 'L4.9', 'L5'],
                        help='Dataset level for eval problems')
    parser.add_argument('--num_problems', type=int, default=10,
                        help='Number of problems to analyze')
    parser.add_argument('--num_passes', type=int, default=5,
                        help='Number of thinking passes')
    parser.add_argument('--output', type=str, default='logs/',
                        help='Output path (file or directory)')
    parser.add_argument('--num_atoms', type=int, default=64,
                        help='Number of LoRA atoms (must match checkpoint)')
    parser.add_argument('--atom_rank', type=int, default=6,
                        help='Rank of each LoRA atom (must match checkpoint)')
    parser.add_argument('--batched', action='store_true',
                        help='Use batched Jacobian computation (faster but more memory)')
    parser.add_argument('--test', action='store_true',
                        help='Run standalone test (no checkpoint needed)')
    parser.add_argument('--skip_pass_embed', action='store_true',
                        help='Use skip_pass_embed architecture (v24.6)')
    args = parser.parse_args()

    # If no checkpoint or --test flag, run standalone test
    if args.test or args.checkpoint is None:
        test_jacobian_standalone()
        return

    # Otherwise, run full diagnostic with checkpoint
    try:
        from scripts.train_atom_lora import make_eval_dataset, try_warm_start
    except ImportError:
        print("ERROR: Could not import train_atom_lora. Make sure you're in the mycelium directory.")
        print("Running standalone test instead...")
        test_jacobian_standalone()
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Determine output path
    output_path = args.output
    if os.path.isdir(output_path) or output_path.endswith('/'):
        os.makedirs(output_path, exist_ok=True)
        ckpt_name = os.path.basename(args.checkpoint).replace('.pt', '')
        output_path = os.path.join(output_path, f'jacobian_{ckpt_name}.png')

    # Load model
    print(f"Loading AtomLoRAModel (atoms={args.num_atoms}, rank={args.atom_rank}, skip_pass_embed={args.skip_pass_embed})...")
    model = AtomLoRAModel(num_atoms=args.num_atoms, atom_rank=args.atom_rank, skip_pass_embed=args.skip_pass_embed)
    model.compressor = model.compressor.to(device=device, dtype=torch.bfloat16)
    model.hypernet = model.hypernet.to(device=device, dtype=torch.bfloat16)
    model.atoms = model.atoms.to(device=device, dtype=torch.bfloat16)
    model.confidence_head = model.confidence_head.to(device)
    model.residual_gate = model.residual_gate.to(device=device, dtype=torch.bfloat16)  # v24.8
    model.fourier_page = model.fourier_page.to(device)
    model.probe_head = model.probe_head.to(device)

    answer_head = AnswerHead(page_size=model.page_size).to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    try_warm_start(model, answer_head, args.checkpoint)

    # Load eval dataset
    print(f"Loading eval dataset: level={args.level}, n={args.num_problems}")
    eval_dataset = make_eval_dataset(args.level, num_samples=args.num_problems)
    actual_n = min(args.num_problems, len(eval_dataset))
    print(f"  Got {actual_n} problems")

    # Run diagnostic
    print(f"\nRunning Jacobian diagnostic...")
    print(f"  Passes: {args.num_passes}")
    print(f"  Batched: {args.batched}")

    results = diagnose_thinking_loop(
        model, eval_dataset, device,
        max_passes=args.num_passes,
        num_problems=actual_n,
        use_batched=args.batched,
    )

    # Plot summary
    plot_jacobian_summary(results, output_path)

    # Save metrics to JSON
    metrics_path = output_path.replace('.png', '_metrics.json')
    metrics_to_save = {
        'checkpoint': args.checkpoint,
        'level': args.level,
        'num_problems': actual_n,
        'num_passes': args.num_passes,
        'mean_sparsity': results['mean_sparsity'],
        'mean_spectral_radius': results['mean_spectral'],
        'mean_band_ratio': results['mean_band_ratio'],
        'sparsity_list': results['sparsity'],
        'spectral_radius_list': results['spectral_radius'],
        'band_ratio_list': results['band_ratio'],
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == '__main__':
    main()
