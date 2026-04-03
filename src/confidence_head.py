"""
ConfidenceHead: The Readiness Judge

A tiny MLP that reads the 64-float state vector and judges whether the model
has accumulated enough information to confidently generate an answer.

Architecture:
    64 -> 32 (ReLU) -> 1 (Sigmoid)

The confidence head enables adaptive compute: easy problems stop after 2-3 passes
(high confidence), while hard problems continue for 8-10 passes (lower confidence).

Parameters: ~2.1K
    - Linear(64, 32):  64 * 32 + 32 = 2,080
    - Linear(32, 1):   32 * 1 + 1 = 33
    - Total:           2,113

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

    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_confidence_head() -> None:
    """Verify ConfidenceHead implementation."""
    print("Testing ConfidenceHead...")

    # Test basic forward pass
    head = ConfidenceHead(state_size=64, hidden_size=32)
    state = torch.randn(4, 64)
    conf = head(state)

    assert conf.shape == (4, 1), f"Expected shape (4, 1), got {conf.shape}"
    assert (conf >= 0).all() and (conf <= 1).all(), "Confidence must be in [0, 1]"

    # Test parameter count (~2.1K expected)
    num_params = head.get_num_parameters()
    print(f"  Parameters: {num_params:,}")
    assert 2000 <= num_params <= 2200, f"Expected ~2.1K params, got {num_params}"

    # Test gradient flow
    state = torch.randn(2, 64, requires_grad=True)
    conf = head(state)
    loss = conf.sum()
    loss.backward()
    assert state.grad is not None, "Gradients should flow to input"

    # Test single sample
    single_state = torch.randn(1, 64)
    single_conf = head(single_state)
    assert single_conf.shape == (1, 1), f"Single sample shape: {single_conf.shape}"

    # Test with zeros state (initial state)
    zeros_state = torch.zeros(1, 64)
    zeros_conf = head(zeros_state)
    assert zeros_conf.shape == (1, 1), "Should handle zeros state"

    print("  All tests passed!")
    print(f"  Output range: [{conf.min().item():.3f}, {conf.max().item():.3f}]")


if __name__ == "__main__":
    test_confidence_head()
