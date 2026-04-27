"""
Entropy flow tracking for Mycelium breathing models.

This module provides utilities for measuring the information content and
dynamics of thinking across cycles. The core idea: good thinking should
show smooth, steady entropy reduction - like electricity following the
path of least resistance.

Three key measurements:
1. page_entropy: How much information/uncertainty is in a page
2. page_delta_norm: How much did the page change between cycles
3. atom_entropy: How distributed is the atom activation pattern

Components:
- EntropyTracker: Static methods for entropy measurements
- SurpriseDetector: Detects unexpected entropy changes per cycle
- EntropyFlowConfidence: GRU-based confidence head that tracks page dynamics
- compute_smoothness_target: Helper to compute smoothness training targets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class EntropyTracker:
    """
    Tracks entropy flow across thinking cycles.

    All methods are static - no state is maintained. These are pure
    measurements that can be composed with other tracking mechanisms
    (like SurpriseDetector) for full entropy flow analysis.
    """

    @staticmethod
    def page_entropy(page: torch.Tensor) -> torch.Tensor:
        """
        Estimate entropy of a page using its value distribution.

        Higher entropy = more information = more uncertainty.
        Lower entropy = values concentrated in fewer dimensions.

        The page values are converted to a probability-like distribution
        by taking absolute values and applying softmax. This treats the
        page as encoding "how much attention" each dimension deserves.

        Args:
            page: Tensor of shape (batch, page_size) or (page_size,)
                  Typically page_size=64 in the Mycelium architecture.

        Returns:
            Tensor of shape (batch,) or scalar - entropy per sample.
            Range: [0, log(page_size)] where max occurs at uniform distribution.
        """
        # Handle both batched and unbatched input
        if page.dim() == 1:
            page = page.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Normalize page to a probability-like distribution
        # Using abs() because page values can be negative (hypersphere)
        probs = F.softmax(page.abs(), dim=-1)  # (batch, page_size)

        # Shannon entropy: H = -sum(p * log(p))
        # Add epsilon to prevent log(0)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # (batch,)

        if squeeze_output:
            entropy = entropy.squeeze(0)

        return entropy

    @staticmethod
    def page_delta_norm(page_current: torch.Tensor, page_previous: torch.Tensor) -> torch.Tensor:
        """
        Compute the L2 norm of change between consecutive pages.

        Measures how much "information changed" between thinking cycles.
        Large delta = lots of new information processed.
        Small delta = thinking is converging / little change.

        In smooth thinking, deltas should be roughly consistent across
        cycles - each cycle doing its fair share of work. Highly variable
        deltas indicate choppy, unreliable reasoning.

        Args:
            page_current: Current page tensor (batch, page_size) or (page_size,)
            page_previous: Previous page tensor, same shape as page_current

        Returns:
            Tensor of shape (batch,) or scalar - L2 norm of difference.
        """
        # Handle both batched and unbatched input
        if page_current.dim() == 1:
            page_current = page_current.unsqueeze(0)
            page_previous = page_previous.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # L2 norm of the difference
        delta_norm = (page_current - page_previous).norm(dim=-1)  # (batch,)

        if squeeze_output:
            delta_norm = delta_norm.squeeze(0)

        return delta_norm

    @staticmethod
    def atom_entropy(atom_scales: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of atom activation pattern.

        High entropy = many atoms active with similar magnitudes (complex thinking).
        Low entropy = few dominant atoms (simple/focused thinking).

        This measures the "complexity" of the LoRA configuration being used.
        For the 64-atom architecture, max entropy = log(64) ~ 4.16 when all
        atoms contribute equally.

        The atom scales can be positive or negative (tanh output), so we
        use absolute values to get activation "intensity" regardless of sign.

        Args:
            atom_scales: Tensor of shape (batch, num_atoms) or (num_atoms,)
                        Typically num_atoms=64 in the v24 architecture.
                        Values typically in [-1, 1] from tanh activation.

        Returns:
            Tensor of shape (batch,) or scalar - entropy per sample.
            Range: [0, log(num_atoms)] where max occurs at uniform activation.
        """
        # Handle both batched and unbatched input
        if atom_scales.dim() == 1:
            atom_scales = atom_scales.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Convert to probability distribution using absolute values
        # (scales can be negative from tanh, but we care about magnitude)
        probs = F.softmax(atom_scales.abs(), dim=-1)  # (batch, num_atoms)

        # Shannon entropy
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # (batch,)

        if squeeze_output:
            entropy = entropy.squeeze(0)

        return entropy


class SurpriseDetector:
    """
    Detects unexpected entropy changes per cycle.

    Surprise measures how much a cycle's entropy delta deviates from the
    running average. High surprise means the cycle encountered something
    unexpected - either a breakthrough (large positive) or regression
    (unexpected negative).

    Uses exponential moving average to track running statistics, allowing
    the detector to adapt to the problem's natural rhythm while still
    catching anomalies.

    Attributes:
        momentum: Weight for exponential moving average (default 0.9).
                  Higher = slower adaptation, more stable expectations.
        running_delta: EMA of page delta means across cycles.
        running_var: EMA of page delta variances across cycles.
    """

    def __init__(self, momentum=0.9):
        """
        Initialize the surprise detector.

        Args:
            momentum: Weight for exponential moving average (0.0-1.0).
                      0.9 means 90% of the running stat comes from history,
                      10% from the current observation.
        """
        self.momentum = momentum
        self.running_delta = None
        self.running_var = None

    def compute_surprise(self, page_deltas):
        """
        Compute surprise scores for each cycle based on page deltas.

        Surprise = |actual - expected| / std

        The first cycle always has zero surprise (no expectation yet).
        Subsequent cycles compare against running statistics updated
        via exponential moving average.

        Args:
            page_deltas: List of (batch,) tensors - one per cycle.
                         Each tensor contains the L2 norm of page changes
                         for that cycle across the batch.

        Returns:
            List of (batch,) tensors - surprise score per cycle.
            Higher values indicate more unexpected changes.
        """
        import math

        surprises = []

        for i, delta in enumerate(page_deltas):
            if i == 0:
                # First cycle - no expectation yet
                surprises.append(torch.zeros_like(delta))
                self.running_delta = delta.mean().item()
                self.running_var = delta.var().item() + 1e-8
            else:
                # Surprise = |actual - expected| / std
                expected = self.running_delta
                std = math.sqrt(self.running_var)
                surprise = ((delta - expected).abs() / (std + 1e-8))
                surprises.append(surprise)

                # Update running stats with EMA
                current_mean = delta.mean().item()
                current_var = delta.var().item()
                self.running_delta = (self.momentum * self.running_delta
                                      + (1 - self.momentum) * current_mean)
                self.running_var = (self.momentum * self.running_var
                                    + (1 - self.momentum) * current_var)

        return surprises

    @staticmethod
    def compute_surprise_batch(page_deltas):
        """
        Simpler batch version - surprise relative to mean of previous deltas.

        No running statistics needed. Each cycle's surprise is computed
        based on all previous deltas in the current sequence. Useful for
        one-off computation where you don't need to maintain state across
        multiple problems.

        Args:
            page_deltas: List of (batch,) tensors - one per cycle.

        Returns:
            List of (batch,) tensors - surprise score per cycle.
            First cycle always returns zeros (no prior expectation).
        """
        surprises = []
        for i, delta in enumerate(page_deltas):
            if i == 0:
                surprises.append(torch.zeros_like(delta))
            elif i == 1:
                # Only one prior - can't compute meaningful std
                # Use a reasonable default: surprise = |actual - expected|
                # (no normalization since std is undefined)
                expected = page_deltas[0]
                surprise = (delta - expected).abs()
                surprises.append(surprise)
            else:
                # Expected = mean of all previous deltas
                prev_deltas = torch.stack(page_deltas[:i], dim=0)  # (i, batch)
                expected = prev_deltas.mean(dim=0)  # (batch,)
                std = prev_deltas.std(dim=0).clamp(min=1e-8)
                surprise = (delta - expected).abs() / std
                surprises.append(surprise)

        return surprises


class EntropyFlowConfidence(nn.Module):
    """
    GRU-based confidence head that tracks page dynamics across thinking cycles.

    This module processes a sequence of pages through a GRU to understand the
    FLOW of thinking, not just the final state. It outputs two signals:

    1. confidence: "Am I done thinking?" (probability the answer is ready)
    2. smoothness: "Was my thinking steady?" (probability the flow was smooth)

    The key insight: choppy thinking with high confidence is suspicious.
    Smooth thinking with low confidence just needs more cycles.

    Architecture:
        1. page_project: Linear(page_size, hidden) - projects pages to hidden dim
        2. flow_gru: GRU(hidden, hidden, num_layers=2) - learns flow dynamics
        3. confidence_head: Linear(hidden, 64) -> ReLU -> Linear(64, 1) -> Sigmoid
        4. smoothness_head: Linear(hidden, 64) -> ReLU -> Linear(64, 1) -> Sigmoid

    The GRU processes pages sequentially, learning to detect patterns in how
    pages evolve. The final hidden state captures the overall quality and
    trajectory of thinking, which the two heads decode into interpretable
    signals.

    Parameter cost: ~216K parameters (negligible compared to the 105M perceiver)
        - page_project: 64 * 128 = 8K
        - flow_gru (2-layer): ~200K
        - confidence_head: 128*64 + 64 + 64*1 + 1 = 8.3K
        - smoothness_head: same = 8.3K

    Example:
        >>> confidence_head = EntropyFlowConfidence(page_size=64, hidden=128)
        >>> state_pages = [torch.randn(batch, 64) for _ in range(5)]  # 5 thinking cycles
        >>> confidence, smoothness = confidence_head(state_pages)
        >>> # confidence: (batch, 1) in [0, 1] - probability answer is ready
        >>> # smoothness: (batch, 1) in [0, 1] - probability thinking was smooth
    """

    def __init__(self, page_size: int = 64, hidden: int = 128):
        """
        Initialize the EntropyFlowConfidence module.

        Args:
            page_size: Dimension of each page vector. Default 64 matches
                      the Mycelium architecture's 64-float bottleneck.
            hidden: Hidden dimension for the GRU and intermediate layers.
                   Default 128 provides good capacity without overfitting.
        """
        super().__init__()

        # Project pages to hidden dim
        self.page_project = nn.Linear(page_size, hidden)

        # GRU tracks the DYNAMICS of page changes
        # 2 layers with dropout for regularization
        # Learns to detect smooth vs choppy entropy reduction
        self.flow_gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # Confidence head: "Am I done thinking?"
        # ReLU activation for non-linearity, Sigmoid for [0,1] output
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Smoothness head: "Was my thinking steady?"
        # Same architecture, different learned weights
        self.smoothness_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, state_pages: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a sequence of pages and output confidence and smoothness.

        Args:
            state_pages: List of page tensors, one per thinking cycle.
                        Each tensor has shape (batch, page_size) or (page_size,).
                        Must have at least 1 page.

        Returns:
            confidence: Tensor of shape (batch, 1) in [0, 1].
                       High values indicate the model believes the answer is ready.
            smoothness: Tensor of shape (batch, 1) in [0, 1].
                       High values indicate the thinking flow was steady/smooth.

        Raises:
            ValueError: If state_pages is empty.
        """
        if len(state_pages) == 0:
            raise ValueError("state_pages must contain at least one page")

        # Handle unbatched input (page_size,) -> (1, page_size)
        if state_pages[0].dim() == 1:
            state_pages = [p.unsqueeze(0) for p in state_pages]
            squeeze_output = True
        else:
            squeeze_output = False

        # Stack pages into (batch, num_pages, page_size)
        pages = torch.stack(state_pages, dim=1)

        # Project to hidden dimension: (batch, num_pages, hidden)
        pages_proj = self.page_project(pages)

        # GRU processes pages sequentially - learns flow dynamics
        # Output: (batch, num_pages, hidden), hidden_state: (num_layers, batch, hidden)
        flow_output, _ = self.flow_gru(pages_proj)

        # Take the last output - summarizes the entire flow
        last_flow = flow_output[:, -1, :]  # (batch, hidden)

        # Decode into confidence and smoothness
        confidence = self.confidence_head(last_flow)   # (batch, 1)
        smoothness = self.smoothness_head(last_flow)   # (batch, 1)

        # Handle unbatched case
        if squeeze_output:
            confidence = confidence.squeeze(0)
            smoothness = smoothness.squeeze(0)

        return confidence, smoothness


def compute_smoothness_target(page_deltas: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute the smoothness target from a sequence of page deltas.

    Smoothness measures how evenly the page deltas are distributed across
    thinking cycles. It is defined as:

        smoothness = 1 - coefficient_of_variation(page_deltas)
        smoothness = 1 - (std / mean)

    Even deltas (smooth flow) -> high smoothness (close to 1.0)
    Uneven deltas (choppy flow) -> low smoothness (close to 0.0)

    This is used as a training target for the smoothness head. The model
    learns to predict how smooth its thinking was, which correlates with
    answer reliability.

    Args:
        page_deltas: List of (batch,) tensors containing the L2 norm of
                    page changes for each thinking cycle. Must have at
                    least 1 delta (ideally 2+ for meaningful variance).

    Returns:
        smoothness: Tensor of shape (batch, 1) in [0, 1].
                   Clamped to ensure valid probability range.

    Example:
        >>> # Smooth thinking: even deltas
        >>> smooth_deltas = [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0]),
        ...                  torch.tensor([1.0, 1.0])]
        >>> smoothness = compute_smoothness_target(smooth_deltas)
        >>> # smoothness close to 1.0

        >>> # Choppy thinking: uneven deltas
        >>> choppy_deltas = [torch.tensor([0.5, 0.5]), torch.tensor([5.0, 5.0]),
        ...                  torch.tensor([0.5, 0.5])]
        >>> smoothness = compute_smoothness_target(choppy_deltas)
        >>> # smoothness close to 0.0
    """
    if len(page_deltas) == 0:
        raise ValueError("page_deltas must contain at least one delta")

    # Stack deltas: (batch, num_passes)
    deltas = torch.stack(page_deltas, dim=1)

    # Handle single delta case: std is undefined, return perfect smoothness
    if deltas.shape[1] == 1:
        # Single delta = no variance to measure, treat as perfectly smooth
        return torch.ones(deltas.shape[0], 1, device=deltas.device, dtype=deltas.dtype)

    # Compute coefficient of variation
    # CV = std / mean - measures relative variability
    mean_delta = deltas.mean(dim=1, keepdim=True).clamp(min=1e-8)
    std_delta = deltas.std(dim=1, keepdim=True)

    cv = std_delta / mean_delta

    # Smoothness = 1 - CV, clamped to [0, 1]
    smoothness = torch.clamp(1.0 - cv, 0.0, 1.0)  # (batch, 1)

    return smoothness


if __name__ == "__main__":
    """Simple tests to verify EntropyTracker and SurpriseDetector methods work correctly."""

    print("Testing EntropyTracker...")
    print("=" * 50)

    # Test 1: page_entropy with uniform distribution should give max entropy
    print("\n1. page_entropy test:")
    uniform_page = torch.ones(64)
    uniform_entropy = EntropyTracker.page_entropy(uniform_page)
    max_entropy = torch.log(torch.tensor(64.0))
    print(f"   Uniform page entropy: {uniform_entropy.item():.4f}")
    print(f"   Max possible entropy: {max_entropy.item():.4f}")
    assert abs(uniform_entropy.item() - max_entropy.item()) < 0.01, "Uniform should give max entropy"
    print("   PASSED: Uniform page gives max entropy")

    # Test 2: page_entropy with concentrated values should give low entropy
    concentrated_page = torch.zeros(64)
    concentrated_page[0] = 10.0  # One dominant value
    concentrated_entropy = EntropyTracker.page_entropy(concentrated_page)
    print(f"   Concentrated page entropy: {concentrated_entropy.item():.4f}")
    assert concentrated_entropy.item() < max_entropy.item() / 2, "Concentrated should give low entropy"
    print("   PASSED: Concentrated page gives low entropy")

    # Test 3: page_entropy batched
    batch_pages = torch.randn(8, 64)
    batch_entropy = EntropyTracker.page_entropy(batch_pages)
    assert batch_entropy.shape == (8,), f"Expected shape (8,), got {batch_entropy.shape}"
    print(f"   Batch entropy shape: {batch_entropy.shape}")
    print("   PASSED: Batched input works correctly")

    # Test 4: page_delta_norm
    print("\n2. page_delta_norm test:")
    page1 = torch.randn(64)
    page2 = page1.clone()  # Identical pages
    delta_zero = EntropyTracker.page_delta_norm(page1, page2)
    print(f"   Delta between identical pages: {delta_zero.item():.6f}")
    assert delta_zero.item() < 1e-6, "Identical pages should have zero delta"
    print("   PASSED: Identical pages give zero delta")

    # Test 5: page_delta_norm with different pages
    page3 = torch.randn(64)
    delta_nonzero = EntropyTracker.page_delta_norm(page1, page3)
    print(f"   Delta between different pages: {delta_nonzero.item():.4f}")
    assert delta_nonzero.item() > 0, "Different pages should have nonzero delta"
    print("   PASSED: Different pages give nonzero delta")

    # Test 6: page_delta_norm batched
    batch_current = torch.randn(8, 64)
    batch_previous = torch.randn(8, 64)
    batch_delta = EntropyTracker.page_delta_norm(batch_current, batch_previous)
    assert batch_delta.shape == (8,), f"Expected shape (8,), got {batch_delta.shape}"
    print(f"   Batch delta shape: {batch_delta.shape}")
    print("   PASSED: Batched delta works correctly")

    # Test 7: atom_entropy
    print("\n3. atom_entropy test:")
    uniform_atoms = torch.ones(64)
    uniform_atom_ent = EntropyTracker.atom_entropy(uniform_atoms)
    print(f"   Uniform atom entropy: {uniform_atom_ent.item():.4f}")
    print(f"   Max possible: {max_entropy.item():.4f}")
    assert abs(uniform_atom_ent.item() - max_entropy.item()) < 0.01, "Uniform atoms should give max entropy"
    print("   PASSED: Uniform atoms give max entropy")

    # Test 8: atom_entropy with sparse activation (need large values to dominate softmax)
    sparse_atoms = torch.zeros(64)
    sparse_atoms[0] = 10.0  # Large value to dominate softmax
    sparse_atom_ent = EntropyTracker.atom_entropy(sparse_atoms)
    print(f"   Sparse atom entropy: {sparse_atom_ent.item():.4f}")
    assert sparse_atom_ent.item() < max_entropy.item() / 2, "Concentrated activation should give low entropy"
    print("   PASSED: Concentrated atoms give low entropy")

    # Test 9: atom_entropy handles negative values (from tanh)
    mixed_atoms = torch.randn(64) * 0.5  # Mix of positive and negative
    mixed_atom_ent = EntropyTracker.atom_entropy(mixed_atoms)
    print(f"   Mixed sign atom entropy: {mixed_atom_ent.item():.4f}")
    assert mixed_atom_ent.item() > 0, "Should produce valid entropy"
    print("   PASSED: Handles negative values correctly")

    # Test 10: atom_entropy batched
    batch_atoms = torch.randn(8, 64)
    batch_atom_ent = EntropyTracker.atom_entropy(batch_atoms)
    assert batch_atom_ent.shape == (8,), f"Expected shape (8,), got {batch_atom_ent.shape}"
    print(f"   Batch atom entropy shape: {batch_atom_ent.shape}")
    print("   PASSED: Batched atom entropy works correctly")

    # ========================================
    # SurpriseDetector tests
    # ========================================
    print("\n" + "=" * 50)
    print("Testing SurpriseDetector...")
    print("=" * 50)

    # Test 11: Basic functionality with running stats
    print("\n4. SurpriseDetector.compute_surprise test:")
    detector = SurpriseDetector(momentum=0.9)

    # Simulate 5 cycles with batch size 4
    batch_size = 4
    page_deltas = [
        torch.tensor([1.0, 1.1, 0.9, 1.0]),   # cycle 0: baseline
        torch.tensor([1.0, 1.0, 1.0, 1.0]),   # cycle 1: similar to baseline
        torch.tensor([1.0, 1.0, 1.0, 1.0]),   # cycle 2: similar
        torch.tensor([5.0, 5.0, 5.0, 5.0]),   # cycle 3: SURPRISE! big jump
        torch.tensor([1.2, 1.1, 1.0, 1.1]),   # cycle 4: back to normal
    ]

    surprises = detector.compute_surprise(page_deltas)

    # Verify first cycle has zero surprise
    assert torch.allclose(surprises[0], torch.zeros(batch_size)), \
        "First cycle should have zero surprise"
    print(f"   Cycle 0 surprise: {surprises[0].mean():.4f} (expected: 0.0)")
    print("   PASSED: First cycle has zero surprise")

    # Verify that cycle 3 (big jump) has high surprise
    assert surprises[3].mean() > surprises[1].mean(), \
        "Big jump cycle should have higher surprise than steady cycle"
    print(f"   Cycle 1 surprise (steady): {surprises[1].mean():.4f}")
    print(f"   Cycle 3 surprise (big jump): {surprises[3].mean():.4f}")
    print("   PASSED: Big jump cycle has higher surprise")

    # Verify running stats were updated
    assert detector.running_delta is not None, "Running delta should be set"
    assert detector.running_var is not None, "Running var should be set"
    print(f"   Final running_delta: {detector.running_delta:.4f}")
    print(f"   Final running_var: {detector.running_var:.4f}")
    print("   PASSED: Running stats updated correctly")

    # Test 12: Static batch method
    print("\n5. SurpriseDetector.compute_surprise_batch test:")

    # Create deltas with enough variance that the std is meaningful
    # The batch method uses mean/std of ALL previous deltas, so:
    # - cycle 1: only one prior (cycle 0), std=0 -> infinite surprise for any difference
    # - cycle 2+: can compute meaningful std from multiple priors
    batch_deltas_for_batch_test = [
        torch.tensor([1.0, 1.0, 1.0, 1.0]),   # cycle 0: baseline
        torch.tensor([1.5, 1.5, 1.5, 1.5]),   # cycle 1: slightly different (will have high surprise due to single prior)
        torch.tensor([1.2, 1.2, 1.2, 1.2]),   # cycle 2: now we have 2 priors with some variance
        torch.tensor([5.0, 5.0, 5.0, 5.0]),   # cycle 3: SURPRISE! big jump (should have highest surprise vs cycles 2,4)
        torch.tensor([1.3, 1.3, 1.3, 1.3]),   # cycle 4: back to normal
    ]

    batch_surprises = SurpriseDetector.compute_surprise_batch(batch_deltas_for_batch_test)

    # First cycle should still be zero
    assert torch.allclose(batch_surprises[0], torch.zeros(batch_size)), \
        "Batch: First cycle should have zero surprise"
    print(f"   Batch cycle 0 surprise: {batch_surprises[0].mean():.4f}")
    print("   PASSED: Batch first cycle has zero surprise")

    # Big jump (cycle 3) should show higher surprise than cycle 4 (which is after the model adapts)
    # Note: cycle 1 has infinite surprise due to single prior with std=0, so we compare cycles 2,3,4
    assert batch_surprises[3].mean() > batch_surprises[2].mean(), \
        "Batch: Big jump cycle should have higher surprise than steady cycle"
    assert batch_surprises[3].mean() > batch_surprises[4].mean(), \
        "Batch: Big jump cycle should have higher surprise than return-to-normal cycle"
    print(f"   Batch cycle 2 surprise (steady): {batch_surprises[2].mean():.4f}")
    print(f"   Batch cycle 3 surprise (big jump): {batch_surprises[3].mean():.4f}")
    print(f"   Batch cycle 4 surprise (back to normal): {batch_surprises[4].mean():.4f}")
    print("   PASSED: Batch big jump has higher surprise than adjacent cycles")

    # Test 13: Edge case - single cycle
    print("\n6. Single cycle edge case test:")
    detector_single = SurpriseDetector()
    single_delta = [torch.tensor([2.0, 2.0])]
    single_surprise = detector_single.compute_surprise(single_delta)
    assert len(single_surprise) == 1, "Should return one surprise tensor"
    assert torch.allclose(single_surprise[0], torch.zeros(2)), \
        "Single cycle should have zero surprise"
    print(f"   Single cycle surprise: {single_surprise[0].mean():.4f}")
    print("   PASSED: Single cycle returns zero surprise")

    # Test 14: Momentum affects adaptation
    print("\n7. Momentum affects adaptation rate test:")
    detector_fast = SurpriseDetector(momentum=0.5)  # fast adaptation
    detector_slow = SurpriseDetector(momentum=0.99)  # slow adaptation

    # Same deltas
    deltas = [
        torch.tensor([1.0, 1.0]),
        torch.tensor([2.0, 2.0]),  # jump
    ]

    _ = detector_fast.compute_surprise(deltas)
    _ = detector_slow.compute_surprise(deltas)

    # Fast adapter should have moved running_delta more toward 2.0
    print(f"   Fast adapter running_delta (m=0.5): {detector_fast.running_delta:.4f}")
    print(f"   Slow adapter running_delta (m=0.99): {detector_slow.running_delta:.4f}")
    assert detector_fast.running_delta > detector_slow.running_delta, \
        "Fast adapter should have higher running_delta after jump"
    print("   PASSED: Momentum affects adaptation rate correctly")

    # Test 15: Verify batch version with only two cycles
    print("\n8. Two cycles batch version test:")
    two_deltas = [
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([2.0, 2.0, 2.0]),
    ]
    two_surprises = SurpriseDetector.compute_surprise_batch(two_deltas)
    assert len(two_surprises) == 2, "Should have 2 surprise tensors"
    # With only one prior delta, std would be 0, but we clamp at 1e-8
    # Expected = 1.0, actual = 2.0, so surprise = |2-1|/1e-8 which is huge
    # This is expected behavior - with single prior point, any deviation is infinite surprise
    print(f"   Two-cycle test - cycle 0 surprise: {two_surprises[0].mean():.4f}")
    print(f"   Two-cycle test - cycle 1 surprise: {two_surprises[1].mean():.4f}")
    print("   PASSED: Two cycle batch version works")

    # ========================================
    # EntropyFlowConfidence tests
    # ========================================
    print("\n" + "=" * 50)
    print("Testing EntropyFlowConfidence...")
    print("=" * 50)

    # Test 16: Basic initialization
    print("\n9. EntropyFlowConfidence initialization test:")
    confidence_head = EntropyFlowConfidence(page_size=64, hidden=128)
    print(f"   page_project weight shape: {confidence_head.page_project.weight.shape}")
    print(f"   flow_gru input_size: {confidence_head.flow_gru.input_size}")
    print(f"   flow_gru hidden_size: {confidence_head.flow_gru.hidden_size}")
    print(f"   flow_gru num_layers: {confidence_head.flow_gru.num_layers}")
    assert confidence_head.page_project.weight.shape == (128, 64), \
        "page_project should map 64 -> 128"
    assert confidence_head.flow_gru.num_layers == 2, "GRU should have 2 layers"
    print("   PASSED: EntropyFlowConfidence initializes correctly")

    # Test 17: Forward pass with batched input
    print("\n10. EntropyFlowConfidence forward (batched) test:")
    batch_size = 4
    num_pages = 5
    page_size = 64
    state_pages = [torch.randn(batch_size, page_size) for _ in range(num_pages)]

    confidence, smoothness = confidence_head(state_pages)

    assert confidence.shape == (batch_size, 1), \
        f"Expected confidence shape ({batch_size}, 1), got {confidence.shape}"
    assert smoothness.shape == (batch_size, 1), \
        f"Expected smoothness shape ({batch_size}, 1), got {smoothness.shape}"
    print(f"   confidence shape: {confidence.shape}")
    print(f"   smoothness shape: {smoothness.shape}")

    # Check outputs are in [0, 1] (from Sigmoid)
    assert (confidence >= 0).all() and (confidence <= 1).all(), \
        "Confidence should be in [0, 1]"
    assert (smoothness >= 0).all() and (smoothness <= 1).all(), \
        "Smoothness should be in [0, 1]"
    print(f"   confidence values (sample): {confidence[:2, 0].tolist()}")
    print(f"   smoothness values (sample): {smoothness[:2, 0].tolist()}")
    print("   PASSED: Forward pass works with batched input")

    # Test 18: Forward pass with unbatched input
    print("\n11. EntropyFlowConfidence forward (unbatched) test:")
    unbatched_pages = [torch.randn(page_size) for _ in range(3)]
    conf_unbatched, smooth_unbatched = confidence_head(unbatched_pages)

    assert conf_unbatched.shape == (1,), \
        f"Expected unbatched confidence shape (1,), got {conf_unbatched.shape}"
    assert smooth_unbatched.shape == (1,), \
        f"Expected unbatched smoothness shape (1,), got {smooth_unbatched.shape}"
    print(f"   unbatched confidence shape: {conf_unbatched.shape}")
    print(f"   unbatched smoothness shape: {smooth_unbatched.shape}")
    print("   PASSED: Forward pass works with unbatched input")

    # Test 19: Single page input
    print("\n12. EntropyFlowConfidence single page test:")
    single_page = [torch.randn(batch_size, page_size)]
    conf_single, smooth_single = confidence_head(single_page)

    assert conf_single.shape == (batch_size, 1), \
        f"Expected shape ({batch_size}, 1), got {conf_single.shape}"
    print(f"   Single page confidence shape: {conf_single.shape}")
    print("   PASSED: Single page input works")

    # Test 20: Empty input raises error
    print("\n13. EntropyFlowConfidence empty input test:")
    try:
        confidence_head([])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"   Caught expected error: {e}")
        print("   PASSED: Empty input raises ValueError")

    # Test 21: Different page counts produce different outputs
    print("\n14. EntropyFlowConfidence page count sensitivity test:")
    torch.manual_seed(42)
    base_pages = [torch.randn(2, 64) for _ in range(3)]
    conf_3, smooth_3 = confidence_head(base_pages)

    # Add one more page
    extra_page = torch.randn(2, 64)
    conf_4, smooth_4 = confidence_head(base_pages + [extra_page])

    # Outputs should be different (GRU processes the sequence)
    assert not torch.allclose(conf_3, conf_4), \
        "Different page counts should produce different confidence"
    print(f"   3-page confidence: {conf_3[0, 0].item():.4f}")
    print(f"   4-page confidence: {conf_4[0, 0].item():.4f}")
    print("   PASSED: Different page counts produce different outputs")

    # Test 22: Gradient flow
    print("\n15. EntropyFlowConfidence gradient flow test:")
    grad_pages = [torch.randn(2, 64, requires_grad=True) for _ in range(3)]
    conf_grad, smooth_grad = confidence_head(grad_pages)

    # Backprop through both outputs
    loss = conf_grad.sum() + smooth_grad.sum()
    loss.backward()

    # Check gradients flow to input pages
    for i, page in enumerate(grad_pages):
        assert page.grad is not None, f"Page {i} should have gradients"
        assert page.grad.abs().sum() > 0, f"Page {i} gradients should be non-zero"
    print("   All input pages received gradients")
    print("   PASSED: Gradients flow correctly")

    # ========================================
    # compute_smoothness_target tests
    # ========================================
    print("\n" + "=" * 50)
    print("Testing compute_smoothness_target...")
    print("=" * 50)

    # Test 23: Perfectly smooth deltas (all equal)
    print("\n16. compute_smoothness_target smooth deltas test:")
    smooth_deltas = [
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([1.0, 1.0, 1.0]),
    ]
    smoothness_target = compute_smoothness_target(smooth_deltas)

    assert smoothness_target.shape == (3, 1), \
        f"Expected shape (3, 1), got {smoothness_target.shape}"
    print(f"   smoothness_target shape: {smoothness_target.shape}")
    print(f"   smoothness_target for equal deltas: {smoothness_target[0, 0].item():.4f}")

    # Equal deltas should have high smoothness (close to 1.0)
    # std = 0, mean > 0, so CV = 0, smoothness = 1.0
    assert smoothness_target[0, 0].item() > 0.99, \
        "Equal deltas should give smoothness close to 1.0"
    print("   PASSED: Equal deltas give high smoothness")

    # Test 24: Choppy deltas (high variance)
    print("\n17. compute_smoothness_target choppy deltas test:")
    choppy_deltas = [
        torch.tensor([0.1, 0.1]),
        torch.tensor([10.0, 10.0]),
        torch.tensor([0.1, 0.1]),
    ]
    choppy_smoothness = compute_smoothness_target(choppy_deltas)

    print(f"   choppy_smoothness: {choppy_smoothness[0, 0].item():.4f}")

    # High variance relative to mean -> low smoothness
    assert choppy_smoothness[0, 0].item() < 0.5, \
        "Choppy deltas should give low smoothness"
    print("   PASSED: Choppy deltas give low smoothness")

    # Test 25: Smoothness is clamped to [0, 1]
    print("\n18. compute_smoothness_target clamping test:")
    # CV > 1 would give smoothness < 0, but clamping should keep it at 0
    extreme_choppy = [
        torch.tensor([0.01, 0.01]),
        torch.tensor([100.0, 100.0]),
    ]
    extreme_smoothness = compute_smoothness_target(extreme_choppy)

    print(f"   extreme_smoothness: {extreme_smoothness[0, 0].item():.4f}")
    assert extreme_smoothness[0, 0].item() >= 0.0, \
        "Smoothness should be clamped to >= 0"
    assert extreme_smoothness[0, 0].item() <= 1.0, \
        "Smoothness should be clamped to <= 1"
    print("   PASSED: Smoothness is properly clamped to [0, 1]")

    # Test 26: Single delta
    print("\n19. compute_smoothness_target single delta test:")
    single_delta = [torch.tensor([2.0, 2.0, 2.0])]
    single_smoothness = compute_smoothness_target(single_delta)

    assert single_smoothness.shape == (3, 1), \
        f"Expected shape (3, 1), got {single_smoothness.shape}"
    # With single delta, std = 0, so smoothness should be 1.0
    print(f"   single_smoothness: {single_smoothness[0, 0].item():.4f}")
    print("   PASSED: Single delta works")

    # Test 27: Empty input raises error
    print("\n20. compute_smoothness_target empty input test:")
    try:
        compute_smoothness_target([])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"   Caught expected error: {e}")
        print("   PASSED: Empty input raises ValueError")

    # Test 28: Gradient flow
    print("\n21. compute_smoothness_target gradient flow test:")
    grad_deltas = [torch.tensor([1.0, 2.0], requires_grad=True) for _ in range(3)]
    smooth_target = compute_smoothness_target(grad_deltas)
    smooth_target.sum().backward()

    for i, delta in enumerate(grad_deltas):
        assert delta.grad is not None, f"Delta {i} should have gradients"
    print("   All deltas received gradients")
    print("   PASSED: Gradients flow correctly")

    # Test 29: Integration - typical thinking scenario
    print("\n22. Integration test - typical thinking scenario:")
    # Simulate 5 passes with batch size 8
    batch_sz = 8
    pages = []
    deltas = []
    prev_page = None

    for _ in range(5):
        page = torch.randn(batch_sz, 64)
        pages.append(page)
        if prev_page is not None:
            delta = EntropyTracker.page_delta_norm(page, prev_page)
            deltas.append(delta)
        prev_page = page

    # Get confidence from pages
    conf_head = EntropyFlowConfidence()
    conf_out, smooth_out = conf_head(pages)

    # Get smoothness target from deltas
    if len(deltas) > 0:
        smooth_target = compute_smoothness_target(deltas)

        # We could train: loss = MSE(smooth_out, smooth_target)
        print(f"   pages: {len(pages)}, deltas: {len(deltas)}")
        print(f"   confidence: {conf_out.mean().item():.4f}")
        print(f"   smoothness (predicted): {smooth_out.mean().item():.4f}")
        print(f"   smoothness (target): {smooth_target.mean().item():.4f}")
    print("   PASSED: Integration works as expected")

    print("\n" + "=" * 50)
    print("All EntropyTracker tests PASSED!")
    print("All SurpriseDetector tests PASSED!")
    print("All EntropyFlowConfidence tests PASSED!")
    print("All compute_smoothness_target tests PASSED!")
    print("=" * 50)
