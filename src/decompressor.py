"""
Decompressor: Lightweight MLP that EXPANDS 64-float state into input bias.

The decompressor has the EASY job: take 64 floats and expand to 2048-dim bias.
That's going from small to big — there are many valid ways to do this.
It doesn't need to make hard decisions about what to keep.
It just needs to faithfully translate the 64 floats into something the
transformer's residual stream can use.

Contrast with the Compressor (HARD job):
- Compressor: 16 layers × seq × 2048 → 64 floats (must SELECT what matters)
- Decompressor: 64 floats → 2048 bias (just PROJECT faithfully)

The asymmetry makes sense: a brilliant editor compressing a book into one
sentence (hard, needs skill) versus a reader interpreting that sentence
(easier, the sentence just needs to be clear).

Architecture (lightweight):
    Input: 64 floats (state on hypersphere)

    1. Concatenate state with pass embedding: 64 + 64 = 128
    2. MLP: 128 → 512 → 2048

    Output: bias vector (batch, 1, 2048) to ADD to all input embeddings

Parameters: ~1.2M (vs 90M in the heavy version)
"""

import torch
import torch.nn as nn


class Decompressor(nn.Module):
    """
    Lightweight MLP that expands 64-float state into input bias.

    The decompressor is intentionally simple because expansion is easy.
    The intelligence is in the COMPRESSION (what to keep), not the
    DECOMPRESSION (how to project it back).

    Args:
        state_size: Size of the input state vector (default: 64)
        d_model: Transformer hidden dimension (default: 2048)
        d_hidden: Hidden dimension of MLP (default: 512)
        max_passes: Maximum thinking passes for embedding (default: 20)
    """

    def __init__(
        self,
        state_size: int = 64,
        d_model: int = 2048,
        d_hidden: int = 512,
        max_passes: int = 20,
    ) -> None:
        super().__init__()

        self.state_size = state_size
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.max_passes = max_passes

        # Pass embedding: small, just tells the MLP which pass we're on
        self.pass_embed = nn.Embedding(max_passes, state_size)  # 20 × 64 = 1.3K

        # Simple 2-layer MLP: (state + pass_embed) → hidden → output
        # Input: state_size * 2 (state concatenated with pass embedding)
        # Or we can add them: state + pass_embed, keeping input size = state_size
        self.mlp = nn.Sequential(
            nn.Linear(state_size, d_hidden),      # 64 → 512
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),        # 512 → 512 (optional depth)
            nn.GELU(),
            nn.Linear(d_hidden, d_model),         # 512 → 2048
        )

        # Output normalization for stable training
        self.output_norm = nn.LayerNorm(d_model)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable training."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # Small initialization for final layer (bias starts subtle)
        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.1)

    def forward(
        self,
        state: torch.Tensor,
        pass_num: int,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Expand state into bias vector for input modulation.

        Args:
            state: The accumulated state vector (batch, 64)
            pass_num: Which thinking pass (0-indexed)
            scale: Scale factor for state warmup (default: 1.0)

        Returns:
            bias: Vector to ADD to input embeddings (batch, 1, 2048)
                  Broadcasts across all sequence positions
        """
        device = state.device

        # Add pass conditioning
        pass_idx = torch.tensor(pass_num, device=device)
        pass_context = self.pass_embed(pass_idx)  # (state_size,)
        x = state + pass_context  # broadcast add: (batch, state_size)

        # Simple MLP expansion
        bias = self.mlp(x)  # (batch, d_model)
        bias = self.output_norm(bias)

        # Add sequence dimension for broadcasting
        bias = bias.unsqueeze(1)  # (batch, 1, d_model)

        # Apply scale (for state warmup: 0.1 → 1.0)
        bias = bias * scale

        return bias

    def count_parameters(self) -> dict:
        """Count parameters in each component."""
        counts = {
            'pass_embed': sum(p.numel() for p in self.pass_embed.parameters()),
            'mlp': sum(p.numel() for p in self.mlp.parameters()),
            'output_norm': sum(p.numel() for p in self.output_norm.parameters()),
        }
        counts['total'] = sum(counts.values())
        return counts


def _test_decompressor():
    """Verify lightweight Decompressor implementation."""
    print("Testing Lightweight Decompressor...")

    # Create module
    decompressor = Decompressor(
        state_size=64,
        d_model=2048,
        d_hidden=512,
        max_passes=20,
    )

    # Count parameters
    param_counts = decompressor.count_parameters()
    total_params = sum(p.numel() for p in decompressor.parameters())

    print(f"\nParameter counts:")
    for name, count in param_counts.items():
        if count > 1e6:
            print(f"  {name}: {count / 1e6:.2f}M")
        else:
            print(f"  {name}: {count:,}")

    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
    print(f"Savings vs heavy version: ~89M params")

    # Test forward pass
    batch_size = 2
    state = torch.randn(batch_size, 64)

    bias = decompressor(state, pass_num=0)
    print(f"\nInput state: ({batch_size}, 64)")
    print(f"Output bias: {bias.shape}")
    assert bias.shape == (batch_size, 1, 2048), f"Expected (batch, 1, 2048), got {bias.shape}"

    # Test scale parameter
    bias_scaled = decompressor(state, pass_num=0, scale=0.1)
    assert torch.allclose(bias_scaled, bias * 0.1, atol=1e-5), "Scale should multiply output"
    print("Scale parameter: OK")

    # Test different passes produce different outputs
    bias_p0 = decompressor(state, pass_num=0)
    bias_p5 = decompressor(state, pass_num=5)
    assert not torch.allclose(bias_p0, bias_p5), "Different passes should produce different bias"
    print("Pass conditioning: OK")

    # Test gradient flow
    state = torch.randn(batch_size, 64, requires_grad=True)
    bias = decompressor(state, pass_num=0)
    loss = bias.sum()
    loss.backward()
    assert state.grad is not None, "Gradients should flow to input state"
    print("Gradient flow: OK")

    # Test broadcasting with input embeddings
    seq_len = 128
    input_embeds = torch.randn(batch_size, seq_len, 2048)
    modulated = input_embeds + bias  # bias broadcasts across seq_len
    assert modulated.shape == (batch_size, seq_len, 2048), "Bias should broadcast"
    print("Broadcasting: OK")

    print("\nAll tests passed!")


if __name__ == "__main__":
    _test_decompressor()
