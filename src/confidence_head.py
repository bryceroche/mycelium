"""
ConfidenceHead: The Judge (Mycelium v20)

A simple 2-layer MLP that predicts whether the current state is ready to generate
an answer. Takes the 64-float state vector from the hypersphere and outputs a
scalar confidence in [0, 1].

Role in the Thinking Loop:
    1. After each thinking pass, the Perceiver compresses to a 64-float state delta
    2. State accumulates on hypersphere: state = normalize(state + delta) * sqrt(64)
    3. ConfidenceHead takes this state and predicts "ready to answer?"
    4. If confidence > threshold (e.g., 0.8): stop thinking, generate answer
    5. If confidence < threshold: loop back, apply LoRA, think again

The confidence head learns to recognize when accumulated thinking is sufficient:
    - Easy problems: few passes, high confidence early
    - Hard problems: many passes, confidence grows gradually

Architecture:
    Input:  64-float state vector (on hypersphere, magnitude sqrt(64))
    Hidden: 32 neurons with ReLU activation
    Output: 1 scalar through Sigmoid -> [0, 1]

Parameters: ~2.1K
    - Linear(64, 32):  64 * 32 + 32 = 2,080
    - Linear(32, 1):   32 * 1 + 1   = 33
    - Total:                        = 2,113

Training:
    During training, confidence is supervised with a soft target that increases
    with pass number: target = min((pass + 1) / max_passes + 0.1, 0.95)
    Loss: MSE between predicted confidence and target.

Usage:
    confidence_head = ConfidenceHead(state_size=64)
    conf = confidence_head(state)  # (batch, 1) in [0, 1]

    if conf.item() > 0.8:
        # Ready to generate answer
        pass
"""

import torch
import torch.nn as nn


class ConfidenceHead(nn.Module):
    """
    Reads the state vector and outputs a confidence score between 0 and 1.

    The confidence indicates whether the model has accumulated enough understanding
    to produce a correct answer. Used to decide when to stop thinking and generate.

    Args:
        state_size: Dimension of the input state vector (default: 64)
        hidden_size: Dimension of the hidden layer (default: 32)
    """

    def __init__(self, state_size: int = 64, hidden_size: int = 32) -> None:
        super().__init__()

        self.state_size = state_size
        self.hidden_size = hidden_size

        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

        # Initialize weights for stable early training
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for reasonable confidence values at start."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                # Xavier uniform for stable gradients
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, state_vector: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence score from state vector.

        Args:
            state_vector: The accumulated state from thinking passes.
                         Shape: (batch, 64) or (batch, state_size)

        Returns:
            Confidence score between 0 and 1.
            Shape: (batch, 1)

        Example:
            >>> head = ConfidenceHead(state_size=64)
            >>> state = torch.randn(4, 64)  # batch of 4
            >>> conf = head(state)
            >>> conf.shape
            torch.Size([4, 1])
            >>> assert (conf >= 0).all() and (conf <= 1).all()
        """
        return self.net(state_vector)

    def count_parameters(self) -> int:
        """
        Count total trainable parameters.

        Returns:
            Integer count of trainable parameters (~2,113)
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters (alias for count_parameters)."""
        return self.count_parameters()


def test_confidence_head() -> None:
    """
    Test ConfidenceHead functionality.

    Verifies:
        1. Output shape is correct for batched inputs
        2. Output is bounded in [0, 1]
        3. Parameter count matches expected ~2.1K
        4. Gradients flow through the network
        5. Works with hypersphere-normalized inputs
    """
    import math

    print("Testing ConfidenceHead...")
    print("-" * 50)

    # Initialize
    head = ConfidenceHead(state_size=64, hidden_size=32)
    print(f"State size: 64")
    print(f"Parameter count: {head.count_parameters():,}")
    assert head.count_parameters() == 2113, f"Expected 2113 params, got {head.count_parameters()}"
    print("Parameter count verified: 2,113")
    print()

    # Test 1: Basic forward pass (batched)
    print("Test 1: Batched input (batch_size=4)")
    state = torch.randn(4, 64)
    conf = head(state)
    print(f"  Input shape:  {state.shape}")
    print(f"  Output shape: {conf.shape}")
    print(f"  Output range: [{conf.min().item():.4f}, {conf.max().item():.4f}]")
    assert conf.shape == (4, 1), f"Expected shape (4, 1), got {conf.shape}"
    assert (conf >= 0).all() and (conf <= 1).all(), "Confidence must be in [0, 1]"
    print("  PASSED")
    print()

    # Test 2: Gradient flow
    print("Test 2: Gradient flow")
    state_grad = torch.randn(2, 64, requires_grad=True)
    conf_grad = head(state_grad)
    loss = conf_grad.sum()
    loss.backward()
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient flows to input: {state_grad.grad is not None}")
    assert state_grad.grad is not None, "Gradients should flow to input"
    print(f"  Input grad norm: {state_grad.grad.norm().item():.6f}")
    print("  PASSED")
    print()

    # Test 3: Single sample
    print("Test 3: Single sample")
    single_state = torch.randn(1, 64)
    single_conf = head(single_state)
    print(f"  Input shape:  {single_state.shape}")
    print(f"  Output shape: {single_conf.shape}")
    assert single_conf.shape == (1, 1), f"Single sample shape: {single_conf.shape}"
    print("  PASSED")
    print()

    # Test 4: Zeros state (initial state)
    print("Test 4: Zeros state (initial condition)")
    zeros_state = torch.zeros(1, 64)
    zeros_conf = head(zeros_state)
    print(f"  Input: zeros tensor")
    print(f"  Output: {zeros_conf.item():.4f}")
    assert zeros_conf.shape == (1, 1), "Should handle zeros state"
    print("  PASSED")
    print()

    # Test 5: Hypersphere-normalized input (realistic usage)
    print("Test 5: Hypersphere-normalized input")
    batch_size = 8
    state_sphere = torch.randn(batch_size, 64)
    state_sphere = torch.nn.functional.normalize(state_sphere, dim=-1) * math.sqrt(64)
    conf_sphere = head(state_sphere)
    print(f"  Input magnitude: {state_sphere.norm(dim=-1).mean().item():.2f} (expected: {math.sqrt(64):.2f})")
    print(f"  Output shape: {conf_sphere.shape}")
    print(f"  Output range: [{conf_sphere.min().item():.4f}, {conf_sphere.max().item():.4f}]")
    assert conf_sphere.shape == (batch_size, 1)
    assert (conf_sphere >= 0).all() and (conf_sphere <= 1).all()
    print("  PASSED")
    print()

    # Test 6: count_parameters method
    print("Test 6: count_parameters() method")
    num_params = head.count_parameters()
    print(f"  count_parameters(): {num_params:,}")
    assert num_params == 2113, f"Expected 2113 params, got {num_params}"
    # Verify alias
    assert head.get_num_parameters() == num_params, "Alias should return same value"
    print("  PASSED")
    print()

    print("-" * 50)
    print("All tests passed!")
    print()

    # Summary
    print("ConfidenceHead Summary:")
    print("  Architecture: Linear(64->32) -> ReLU -> Linear(32->1) -> Sigmoid")
    print("  Parameters:   2,113 (~2.1K)")
    print("  Input:        64-float state vector on hypersphere")
    print("  Output:       Scalar confidence in [0, 1]")
    print("  Role:         Decides when to stop thinking and generate answer")


if __name__ == "__main__":
    test_confidence_head()
