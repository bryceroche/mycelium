"""
Compressor: 7-layer Perceiver that COMPRESSES all 16 transformer layers into 64 floats.

This is one half of the Symmetric Hourglass (v19). The Compressor reads ALL 16
hidden states from the transformer and squeezes them into a 64-float state delta.
The delta is then used to rotate the state on the hypersphere.

Key features:
- Reads ALL 16 transformer layers (not just the final one)
- Pass-conditioned layer gate: learns which layers matter for each thinking pass
- 4 learned queries compressed through 7 perceiver layers → 64 floats
- ~105M parameters (symmetric with Decompressor)

Architecture:
    Input: 16 tensors of shape (batch, seq_len, 2048) from Llama's layers

    1. Pass embedding conditions the queries and layer gate
    2. Layer gate (softmax) weights the 16 layers differently per pass
    3. Weighted combination projected from 2048 → 1024
    4. 7 perceiver layers: cross-attn + self-attn + FFN with residuals
    5. Output projection: 1024 → 16 per query → 64 total floats

    Output: state delta (batch, 64) to rotate on hypersphere
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Compressor(nn.Module):
    """
    7-layer Perceiver that reads all transformer layers with pass-conditioned attention.

    The compression is SYMMETRIC with the Decompressor: 105M params deciding how to
    compress, 105M params deciding how to expand. Like two brilliant editors
    collaborating on a one-sentence summary.

    Args:
        num_transformer_layers: Number of Llama layers to read (default: 16)
        d_transformer: Hidden dimension of Llama (default: 2048)
        d_internal: Internal perceiver dimension (default: 1024)
        num_queries: Number of learned queries (default: 4)
        num_layers: Depth of perceiver stack (default: 7)
        state_size: Output state vector size (default: 64)
        max_passes: Maximum thinking passes for embedding (default: 20)
    """

    def __init__(
        self,
        num_transformer_layers: int = 16,
        d_transformer: int = 2048,
        d_internal: int = 1024,
        num_queries: int = 4,
        num_layers: int = 7,
        state_size: int = 64,
        max_passes: int = 20,
    ) -> None:
        super().__init__()

        self.num_transformer_layers = num_transformer_layers
        self.d_transformer = d_transformer
        self.d_internal = d_internal
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.state_size = state_size
        self.max_passes = max_passes

        # Learned queries (4 queries, each in perceiver space)
        self.queries = nn.Parameter(torch.randn(num_queries, d_internal) * 0.02)

        # Pass embedding: conditions queries and layer gate
        self.pass_embed = nn.Embedding(max_passes, d_internal)

        # Pass-conditioned layer gate
        # Learns which of Llama's 16 layers to focus on per pass
        # Pass 1 (parsing): might focus on layers 1-8 (basic features)
        # Pass 5 (solving): might focus on layers 12-16 (answer-oriented)
        self.layer_gate = nn.Sequential(
            nn.Linear(d_internal, 64),
            nn.ReLU(),
            nn.Linear(64, num_transformer_layers),
        )

        # Project from Llama's space to perceiver's internal space
        self.input_project = nn.Linear(d_transformer, d_internal)  # 2048 → 1024

        # 7-layer perceiver stack
        # Each layer: cross-attn + self-attn + FFN with residuals and LayerNorm
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(
                    embed_dim=d_internal,
                    num_heads=8,
                    kdim=d_internal,
                    vdim=d_internal,
                    batch_first=True,
                ),
                'cross_norm': nn.LayerNorm(d_internal),
                'self_attn': nn.MultiheadAttention(
                    embed_dim=d_internal,
                    num_heads=8,
                    batch_first=True,
                ),
                'self_norm': nn.LayerNorm(d_internal),
                'ffn': nn.Sequential(
                    nn.Linear(d_internal, d_internal * 4),
                    nn.GELU(),
                    nn.Linear(d_internal * 4, d_internal),
                ),
                'ffn_norm': nn.LayerNorm(d_internal),
            })
            for _ in range(num_layers)
        ])

        # Final tight projection: 1024 → 16 per query → 64 total
        self.project_out = nn.Linear(d_internal, state_size // num_queries)  # 1024 → 16

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small values for stable training."""
        # Initialize queries with small random values
        nn.init.normal_(self.queries, mean=0.0, std=0.02)

        # Initialize projection layers
        nn.init.xavier_uniform_(self.input_project.weight)
        nn.init.zeros_(self.input_project.bias)
        nn.init.xavier_uniform_(self.project_out.weight)
        nn.init.zeros_(self.project_out.bias)

        # Initialize layer gate to start with uniform attention
        # (all layers equally weighted initially)
        nn.init.zeros_(self.layer_gate[0].weight)
        nn.init.zeros_(self.layer_gate[0].bias)
        nn.init.zeros_(self.layer_gate[2].weight)
        nn.init.zeros_(self.layer_gate[2].bias)

    def forward(
        self,
        all_layer_hidden_states: List[torch.Tensor],
        pass_num: int,
    ) -> torch.Tensor:
        """
        Compress all transformer layer hidden states into a 64-float state delta.

        Args:
            all_layer_hidden_states: List of 16 tensors, each (batch, seq_len, 2048)
                                     Hidden states from each Llama layer.
            pass_num: Which thinking pass (0-indexed, used for pass-conditioned attention)

        Returns:
            state_delta: (batch, 64) compressed state delta for hypersphere rotation
        """
        assert len(all_layer_hidden_states) == self.num_transformer_layers, \
            f"Expected {self.num_transformer_layers} layer hidden states, got {len(all_layer_hidden_states)}"

        batch_size = all_layer_hidden_states[0].size(0)
        device = all_layer_hidden_states[0].device
        dtype = all_layer_hidden_states[0].dtype

        # Get pass embedding and condition queries
        pass_idx = torch.tensor(pass_num, device=device)
        pass_context = self.pass_embed(pass_idx)  # (d_internal,)

        # Expand queries for batch and add pass context
        queries = (self.queries + pass_context).unsqueeze(0).expand(batch_size, -1, -1)
        # queries: (batch, num_queries, d_internal) = (batch, 4, 1024)

        # Pass-conditioned layer importance (softmax for normalized weights)
        layer_logits = self.layer_gate(pass_context)  # (16,)
        layer_weights = F.softmax(layer_logits, dim=-1)

        # Stack all layer hidden states
        stacked = torch.stack(all_layer_hidden_states, dim=0)  # (16, batch, seq, 2048)

        # Weighted combination of ALL transformer layers
        weights = layer_weights.view(self.num_transformer_layers, 1, 1, 1)
        combined = (stacked * weights).sum(dim=0)  # (batch, seq, 2048)

        # Project to perceiver dimension
        kv = self.input_project(combined.to(dtype=self.input_project.weight.dtype))
        # kv: (batch, seq, d_internal) = (batch, seq, 1024)

        # Ensure queries match dtype
        queries = queries.to(dtype=kv.dtype)

        # 7 layers of deep compression processing
        for layer in self.layers:
            # Cross-attend: queries read from transformer representations
            attended, _ = layer['cross_attn'](
                query=queries,
                key=kv,
                value=kv,
            )
            queries = layer['cross_norm'](queries + attended)

            # Self-attend: queries coordinate with each other
            refined, _ = layer['self_attn'](
                query=queries,
                key=queries,
                value=queries,
            )
            queries = layer['self_norm'](queries + refined)

            # FFN: nonlinear processing
            ffn_out = layer['ffn'](queries)
            queries = layer['ffn_norm'](queries + ffn_out)

        # queries: (batch, 4, 1024)

        # Project to tight bottleneck
        state_delta = self.project_out(queries)  # (batch, 4, 16)

        return state_delta.flatten(start_dim=1)  # (batch, 64)

    def get_layer_weights(self, pass_num: int, device: torch.device = None) -> torch.Tensor:
        """
        Get the learned layer importance weights for a given pass.

        Useful for visualization: see which Llama layers the compressor
        focuses on at different stages of thinking.

        Args:
            pass_num: Which thinking pass
            device: Device to use (defaults to module's device)

        Returns:
            layer_weights: (16,) softmax weights over Llama layers
        """
        if device is None:
            device = self.queries.device

        pass_idx = torch.tensor(pass_num, device=device)
        pass_context = self.pass_embed(pass_idx)
        layer_logits = self.layer_gate(pass_context)
        layer_weights = F.softmax(layer_logits, dim=-1)

        return layer_weights

    def count_parameters(self) -> dict:
        """Count parameters in each component for verification."""
        counts = {
            'queries': self.queries.numel(),
            'pass_embed': sum(p.numel() for p in self.pass_embed.parameters()),
            'layer_gate': sum(p.numel() for p in self.layer_gate.parameters()),
            'input_project': sum(p.numel() for p in self.input_project.parameters()),
            'perceiver_layers': sum(
                sum(p.numel() for p in layer.parameters())
                for layer in self.layers
            ),
            'project_out': sum(p.numel() for p in self.project_out.parameters()),
        }
        counts['total'] = sum(counts.values())
        return counts


def _test_compressor():
    """Quick sanity check that the module works."""
    print("Testing Compressor...")

    # Create module
    compressor = Compressor(
        num_transformer_layers=16,
        d_transformer=2048,
        d_internal=1024,
        num_queries=4,
        num_layers=7,
        state_size=64,
        max_passes=20,
    )

    # Count parameters
    param_counts = compressor.count_parameters()
    print(f"\nParameter counts:")
    for name, count in param_counts.items():
        if count > 1e6:
            print(f"  {name}: {count / 1e6:.1f}M")
        else:
            print(f"  {name}: {count:,}")

    total_params = sum(p.numel() for p in compressor.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.1f}M")
    print(f"Target: ~105M")

    # Test forward pass
    batch_size = 2
    seq_len = 128

    # Simulate 16 layer hidden states
    all_layer_hidden_states = [
        torch.randn(batch_size, seq_len, 2048)
        for _ in range(16)
    ]

    # Forward pass
    state_delta = compressor(all_layer_hidden_states, pass_num=0)
    print(f"\nInput: 16 × (batch={batch_size}, seq={seq_len}, 2048)")
    print(f"Output: {state_delta.shape}")
    assert state_delta.shape == (batch_size, 64), f"Expected (batch, 64), got {state_delta.shape}"

    # Test layer weights
    print("\nLayer weights per pass:")
    for pass_num in [0, 5, 10, 19]:
        weights = compressor.get_layer_weights(pass_num)
        top3 = weights.detach().topk(3)
        print(f"  Pass {pass_num}: top layers {top3.indices.tolist()} with weights {top3.values.tolist()}")

    # Test gradient flow
    state_delta.sum().backward()
    print("\nGradient flow: OK")

    print("\nAll tests passed!")


if __name__ == "__main__":
    _test_compressor()
