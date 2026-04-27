"""
Entropy flow visualization diagnostics for Mycelium breathing models.

This script provides visualization tools for analyzing the entropy flow
across thinking cycles. The main function `visualize_entropy_flow()` creates
a 4-subplot figure showing:
1. Page deltas - information change per cycle
2. Atom entropy - thinking complexity per cycle
3. Surprise - unexpected changes (red if > 2.0, green otherwise)
4. Cumulative smoothness - steadiness of thinking flow

Usage:
    from scripts.diag_entropy_flow import visualize_entropy_flow

    # After running thinking passes, collect:
    # - state_pages: list of (batch, 64) or (64,) tensors
    # - atom_scales_history: list of (batch, 64) or (64,) tensors

    visualize_entropy_flow(state_pages, atom_scales_history)
    # Saves to 'entropy_flow.png'

    # Or with custom path:
    visualize_entropy_flow(state_pages, atom_scales_history, save_path='my_plot.png')
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Optional

from entropy_flow import EntropyTracker, SurpriseDetector


def visualize_entropy_flow(
    state_pages: List[torch.Tensor],
    atom_scales_history: List[torch.Tensor],
    save_path: str = 'entropy_flow.png'
) -> None:
    """
    Plot the entropy flow across thinking cycles for a single problem.

    Creates a 4-subplot figure showing:
    1. Page deltas (bar chart) - ||page_k - page_{k-1}||
    2. Atom entropy per cycle (line plot) - H(atom_scales)
    3. Surprise per cycle (bar chart, red if > 2.0, green otherwise)
    4. Cumulative smoothness (line plot) - 1 - CV(deltas[:i])

    Args:
        state_pages: List of page tensors, one per thinking cycle.
                    Each tensor has shape (batch, page_size) or (page_size,).
                    If batched, uses the first sample (index 0).
                    Must have at least 2 pages for delta computation.
        atom_scales_history: List of atom scale tensors, one per cycle.
                            Each tensor has shape (batch, num_atoms) or (num_atoms,).
                            If batched, uses the first sample (index 0).
        save_path: Path to save the figure. Default 'entropy_flow.png'.

    Returns:
        None. Saves figure to save_path and prints confirmation.

    Raises:
        ValueError: If state_pages has fewer than 2 pages.
    """
    if len(state_pages) < 2:
        raise ValueError("Need at least 2 pages to compute deltas")

    # Handle batched input - take first sample
    pages = []
    for p in state_pages:
        if p.dim() == 2:
            pages.append(p[0])  # Take first sample from batch
        else:
            pages.append(p)

    atom_scales = []
    for s in atom_scales_history:
        if s.dim() == 2:
            atom_scales.append(s[0])  # Take first sample from batch
        else:
            atom_scales.append(s)

    # Compute page deltas using EntropyTracker
    deltas = []
    for i in range(1, len(pages)):
        delta = EntropyTracker.page_delta_norm(pages[i], pages[i-1])
        deltas.append(delta.item())

    # Compute atom entropy per cycle using EntropyTracker
    atom_ents = []
    for scales in atom_scales:
        ent = EntropyTracker.atom_entropy(scales)
        atom_ents.append(ent.item())

    # Compute surprise using SurpriseDetector.compute_surprise_batch
    # Need to convert deltas to tensors
    delta_tensors = [torch.tensor([d]) for d in deltas]
    surprises = SurpriseDetector.compute_surprise_batch(delta_tensors)
    surprise_vals = [s.item() for s in surprises]

    # Compute cumulative smoothness: 1 - CV(deltas[:i])
    smooth_vals = []
    for i in range(1, len(deltas) + 1):
        partial_deltas = torch.tensor(deltas[:i])
        if i == 1:
            # Single delta - no variance, treat as perfectly smooth
            smooth_vals.append(1.0)
        else:
            mean_d = partial_deltas.mean().clamp(min=1e-8)
            std_d = partial_deltas.std()
            cv = std_d / mean_d
            smooth_vals.append(max(0.0, 1.0 - cv.item()))

    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))

    # Plot 1: Page deltas (bar chart)
    axes[0].bar(range(len(deltas)), deltas)
    axes[0].set_title('Page Delta Per Cycle (information change)')
    axes[0].set_ylabel('||page_k - page_{k-1}||')
    axes[0].set_xlabel('Cycle')

    # Plot 2: Atom entropy per cycle (line plot)
    axes[1].plot(atom_ents, 'o-')
    axes[1].set_title('Atom Entropy Per Cycle (thinking complexity)')
    axes[1].set_ylabel('H(atom_scales)')
    axes[1].set_xlabel('Cycle')

    # Plot 3: Surprise per cycle (bar chart, red if > 2.0, green otherwise)
    colors = ['red' if s > 2.0 else 'green' for s in surprise_vals]
    axes[2].bar(range(len(surprise_vals)), surprise_vals, color=colors)
    axes[2].axhline(y=2.0, color='red', linestyle='--', label='threshold')
    axes[2].set_title('Surprise Per Cycle (red = unexpected)')
    axes[2].set_ylabel('|actual - expected| / std')
    axes[2].set_xlabel('Cycle')
    axes[2].legend()

    # Plot 4: Cumulative smoothness (line plot)
    axes[3].plot(smooth_vals, 'o-', color='blue')
    axes[3].set_title('Cumulative Smoothness (higher = steadier flow)')
    axes[3].set_ylabel('1 - CV(deltas)')
    axes[3].set_xlabel('Cycle')
    axes[3].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    """
    Test the visualization with random mock data.
    Creates 5 pages with 64 dimensions and verifies the output file is created.
    """
    import os

    print("Testing visualize_entropy_flow with mock data...")
    print("=" * 50)

    # Create random mock data (5 pages, 64 dims)
    torch.manual_seed(42)
    num_pages = 5
    page_size = 64
    num_atoms = 64

    # Mock state pages - simulating a thinking trajectory
    state_pages = [torch.randn(page_size) for _ in range(num_pages)]

    # Mock atom scales - simulating hypernetwork outputs
    atom_scales_history = [torch.randn(num_atoms) * 0.5 for _ in range(num_pages)]

    # Test with default save path
    test_path = 'entropy_flow.png'

    # Remove existing file if present
    if os.path.exists(test_path):
        os.remove(test_path)

    # Run visualization
    visualize_entropy_flow(state_pages, atom_scales_history)

    # Verify file was created
    if os.path.exists(test_path):
        file_size = os.path.getsize(test_path)
        print(f"SUCCESS: {test_path} created ({file_size} bytes)")
    else:
        print(f"FAILED: {test_path} was not created")
        sys.exit(1)

    # Test with batched input
    print("\nTesting with batched input...")
    batch_size = 4
    batched_pages = [torch.randn(batch_size, page_size) for _ in range(num_pages)]
    batched_atoms = [torch.randn(batch_size, num_atoms) * 0.5 for _ in range(num_pages)]

    test_path_batched = 'entropy_flow_batched.png'
    if os.path.exists(test_path_batched):
        os.remove(test_path_batched)

    visualize_entropy_flow(batched_pages, batched_atoms, save_path=test_path_batched)

    if os.path.exists(test_path_batched):
        file_size = os.path.getsize(test_path_batched)
        print(f"SUCCESS: {test_path_batched} created ({file_size} bytes)")
        # Clean up batched test file
        os.remove(test_path_batched)
    else:
        print(f"FAILED: {test_path_batched} was not created")
        sys.exit(1)

    # Test error case: fewer than 2 pages
    print("\nTesting error case (fewer than 2 pages)...")
    try:
        visualize_entropy_flow([torch.randn(page_size)], [torch.randn(num_atoms)])
        print("FAILED: Should have raised ValueError")
        sys.exit(1)
    except ValueError as e:
        print(f"SUCCESS: Caught expected error: {e}")

    print("\n" + "=" * 50)
    print("All tests PASSED!")
    print("=" * 50)
