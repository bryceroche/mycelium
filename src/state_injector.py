"""
StateInjector: Converts 64-float state vector to 4 pseudo-tokens for Llama input.

The StateInjector is the "notebook reader" in the Mycelium v18 architecture.
It takes the compressed 64-float state vector and projects it into 4 pseudo-tokens
that are prepended to the transformer's input embeddings.

Architecture:
    64 floats -> split into 4 chunks of 16 floats each
    Each 16-float chunk -> MLP (16 -> 56 -> 2048) to reach d_model
    Add learned position embeddings
    Apply LayerNorm

The expansion (16 -> 56 -> 2048) ensures no information is lost at injection.
The bottleneck is in the 64 floats, not the projection.

Parameter count: ~130K
    - project (MLP 16 -> 56 -> 2048):
        - fc1: 16 * 56 + 56 = 952
        - fc2: 56 * 2048 + 2048 = 116,736
    - position_embed: 4 * 2048 = 8,192
    - LayerNorm: 2048 * 2 = 4,096
    - Total: ~130K
"""

import torch
import torch.nn as nn
from typing import Optional


class StateInjector(nn.Module):
    """
    Converts a 64-float state vector into 4 pseudo-tokens for Llama 3.2 1B input.

    The state vector is split into 4 chunks of 16 floats each. Each chunk is
    projected through a two-layer MLP to the transformer's hidden dimension (2048)
    and given a learned position embedding. The output is normalized and ready
    to be prepended to the transformer's input embeddings.

    Args:
        state_size: Size of the state vector (default: 64)
        d_model: Transformer hidden dimension (default: 2048 for Llama 3.2 1B)
        num_tokens: Number of pseudo-tokens to generate (default: 4)
        hidden_dim: Hidden dimension for the projection MLP (default: 56)

    Example:
        >>> injector = StateInjector(state_size=64, d_model=2048, num_tokens=4)
        >>> state = torch.randn(8, 64)  # batch of 8
        >>> tokens = injector(state)  # (8, 4, 2048)
        >>> # Prepend to input: torch.cat([tokens, input_embeds], dim=1)
    """

    def __init__(
        self,
        state_size: int = 64,
        d_model: int = 2048,
        num_tokens: int = 4,
        hidden_dim: int = 56,
    ) -> None:
        super().__init__()

        if state_size % num_tokens != 0:
            raise ValueError(
                f"state_size ({state_size}) must be divisible by "
                f"num_tokens ({num_tokens})"
            )

        self.state_size = state_size
        self.d_model = d_model
        self.num_tokens = num_tokens
        self.chunk_size = state_size // num_tokens  # 64 // 4 = 16
        self.hidden_dim = hidden_dim

        # Two-layer MLP projection: 16 -> 56 -> 2048
        # This gives ~130K parameters total as specified
        self.project = nn.Sequential(
            nn.Linear(self.chunk_size, hidden_dim),  # 16 -> 56
            nn.GELU(),  # Smooth nonlinearity
            nn.Linear(hidden_dim, d_model),  # 56 -> 2048
        )

        # Learned position embeddings for the 4 pseudo-tokens
        self.position_embed = nn.Parameter(torch.randn(num_tokens, d_model))

        # Normalize output for stable injection into transformer
        self.norm = nn.LayerNorm(d_model)

        # Initialize weights for stable training start
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable training."""
        # Xavier initialization for projection layers
        for module in self.project:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # Small initialization for position embeddings
        nn.init.normal_(self.position_embed, mean=0.0, std=0.02)

    def forward(self, state_vector: torch.Tensor) -> torch.Tensor:
        """
        Convert state vector to pseudo-tokens.

        Args:
            state_vector: Compressed state from the AllLayerPerceiver.
                Shape: (batch, 64) or (batch, state_size)

        Returns:
            Pseudo-tokens to prepend to transformer input.
            Shape: (batch, 4, 2048) or (batch, num_tokens, d_model)

        Raises:
            ValueError: If state_vector has wrong shape.
        """
        if state_vector.dim() != 2:
            raise ValueError(
                f"Expected 2D state_vector (batch, {self.state_size}), "
                f"got shape {state_vector.shape}"
            )

        if state_vector.size(-1) != self.state_size:
            raise ValueError(
                f"Expected state_vector with size {self.state_size} in last dim, "
                f"got {state_vector.size(-1)}"
            )

        batch_size = state_vector.size(0)

        # Split into chunks: (batch, 64) -> (batch, 4, 16)
        chunks = state_vector.reshape(batch_size, self.num_tokens, self.chunk_size)

        # Project each chunk through MLP: (batch, 4, 16) -> (batch, 4, 2048)
        tokens = self.project(chunks)

        # Add position embeddings (broadcast across batch)
        tokens = tokens + self.position_embed

        # Normalize for stable injection
        tokens = self.norm(tokens)

        return tokens

    def get_empty_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Create a zero-initialized state vector.

        Used for the first thinking pass when no prior state exists.

        Args:
            batch_size: Number of states to create.
            device: Device for the tensor (defaults to module's device).
            dtype: Data type for the tensor (defaults to float32).

        Returns:
            Zero state vector of shape (batch_size, state_size).
        """
        if device is None:
            device = self.position_embed.device
        if dtype is None:
            dtype = self.position_embed.dtype

        return torch.zeros(batch_size, self.state_size, device=device, dtype=dtype)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _test_state_injector():
    """Sanity check for StateInjector."""
    print("Testing StateInjector...")

    # Create module
    injector = StateInjector(state_size=64, d_model=2048, num_tokens=4)
    print(f"  Parameters: {injector.num_parameters:,}")

    # Test forward pass
    batch_size = 8
    state = torch.randn(batch_size, 64)
    tokens = injector(state)

    assert tokens.shape == (batch_size, 4, 2048), f"Wrong shape: {tokens.shape}"
    print(f"  Input:  {state.shape}")
    print(f"  Output: {tokens.shape}")

    # Test empty state
    empty = injector.get_empty_state(batch_size=4)
    assert empty.shape == (4, 64), f"Wrong empty shape: {empty.shape}"
    assert (empty == 0).all(), "Empty state should be zeros"
    print(f"  Empty:  {empty.shape}")

    # Test gradient flow
    state.requires_grad_(True)
    tokens = injector(state)
    loss = tokens.sum()
    loss.backward()
    assert state.grad is not None, "Gradients should flow through"
    print("  Gradient flow: OK")

    # Verify parameter count is ~130K
    param_count = injector.num_parameters
    assert 120_000 < param_count < 140_000, f"Expected ~130K params, got {param_count:,}"
    print(f"  Parameter count verified: {param_count:,} (~130K)")

    print("StateInjector tests passed!")


if __name__ == "__main__":
    _test_state_injector()
