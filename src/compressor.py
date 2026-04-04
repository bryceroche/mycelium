"""
Compressor (AllLayerPerceiver): 7-layer Perceiver that COMPRESSES all 16 transformer
layers into 64 floats.

Mycelium v20 Architecture - The compressor is "THE NOTE-TAKER" in the thinking loop.
It reads ALL 16 hidden states from Llama with pass-conditioned attention and squeezes
them into a 64-float state delta. The delta rotates the state on the hypersphere,
which then conditions the LoRA weights for the next thinking pass.

Key features:
- Reads ALL 16 transformer layers (not just the final one)
- Pass-conditioned layer gate: softmax weights over layers (e.g., early passes focus
  on parsing layers 1-8, later passes focus on answer-oriented layers 12-16)
- Cross-attention uses FULL sequence (not pooled to 1 token)
- 4 learned queries compressed through 7 perceiver layers → 64 floats
- ~105M parameters

Architecture:
    Input: 16 tensors of shape (batch, seq_len, 2048) from Llama's layers

    1. Pass embedding conditions the queries and layer gate
    2. Layer gate (softmax) weights the 16 layers differently per pass
    3. Weighted combination projected from 2048 → 1024 (FULL sequence preserved)
    4. 7 perceiver layers: cross-attn + self-attn + FFN with residuals
    5. Output projection: 1024 → 16 per query → 64 total floats

    Output: state delta (batch, 64) to rotate on hypersphere

Parameters (~105M):
    - Learned queries:           4 × 1024 = 4K
    - Pass embedding:            20 × 1024 = 20K
    - Layer gate:                1024 × 64 + 64 × 16 = 66K
    - Input projection:          2048 × 1024 = 2.1M
    - 7 perceiver layers:        ~102M (cross-attn + self-attn + FFN each)
    - Output projection:         1024 × 16 = 16K
    - Total:                     ~105M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Compressor(nn.Module):
    """
    AllLayerPerceiver: 7-layer Perceiver that reads all transformer layers with
    pass-conditioned attention.

    Mycelium v20 Architecture Component - "THE NOTE-TAKER"

    This compressor reads ALL 16 Llama layer hidden states with the FULL sequence
    (not pooled). Pass-conditioned layer gating learns which layers matter at each
    stage of thinking: early passes focus on parsing (layers 1-8), later passes
    focus on answer-oriented layers (12-16).

    The 4 learned queries cross-attend to the full sequence through 7 perceiver
    layers, then compress to 64 floats (16 floats per query). This 64-float state
    delta rotates the state on the hypersphere, which conditions the LoRA weights
    for the next thinking pass.

    Args:
        num_transformer_layers: Number of Llama layers to read (default: 16)
        d_transformer: Hidden dimension of Llama (default: 2048)
        d_perceiver: Internal perceiver dimension (default: 1024)
        num_queries: Number of learned queries (default: 4)
        num_perceiver_layers: Depth of perceiver stack (default: 7)
        state_size: Output state vector size (default: 64)
        max_passes: Maximum thinking passes for embedding (default: 20)
    """

    def __init__(
        self,
        num_transformer_layers: int = 16,
        d_transformer: int = 2048,
        d_perceiver: int = 1024,
        num_queries: int = 4,
        num_perceiver_layers: int = 7,
        state_size: int = 64,
        max_passes: int = 20,
    ) -> None:
        super().__init__()

        self.num_transformer_layers = num_transformer_layers
        self.d_transformer = d_transformer
        self.d_perceiver = d_perceiver
        self.num_queries = num_queries
        self.num_perceiver_layers = num_perceiver_layers
        self.state_size = state_size
        self.max_passes = max_passes

        # Learned queries (4 queries, each in perceiver space)
        self.queries = nn.Parameter(torch.randn(num_queries, d_perceiver) * 0.02)

        # Pass embedding: conditions queries and layer gate
        self.pass_embed = nn.Embedding(max_passes, d_perceiver)

        # Pass-conditioned layer gate (softmax weights over 16 transformer layers)
        # Learns which of Llama's 16 layers to focus on per pass:
        #   - Pass 1 (parsing): might focus on layers 1-8 (basic features, numbers)
        #   - Pass 5 (solving): might focus on layers 12-16 (answer-oriented)
        # The softmax ensures normalized weights that sum to 1.0
        self.layer_gate = nn.Sequential(
            nn.Linear(d_perceiver, 64),
            nn.ReLU(),
            nn.Linear(64, num_transformer_layers),
        )

        # Project from Llama's space to perceiver's internal space
        # FULL sequence is preserved (not pooled to 1 token)
        self.input_project = nn.Linear(d_transformer, d_perceiver)  # 2048 → 1024

        # 7-layer perceiver stack
        # Each layer: cross-attn (queries attend to FULL sequence) + self-attn + FFN
        # with residuals and LayerNorm throughout
        self.perceiver_layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(
                    embed_dim=d_perceiver,
                    num_heads=8,
                    kdim=d_perceiver,
                    vdim=d_perceiver,
                    batch_first=True,
                ),
                'cross_norm': nn.LayerNorm(d_perceiver),
                'self_attn': nn.MultiheadAttention(
                    embed_dim=d_perceiver,
                    num_heads=8,
                    batch_first=True,
                ),
                'self_norm': nn.LayerNorm(d_perceiver),
                'ffn': nn.Sequential(
                    nn.Linear(d_perceiver, d_perceiver * 4),
                    nn.GELU(),
                    nn.Linear(d_perceiver * 4, d_perceiver),
                ),
                'ffn_norm': nn.LayerNorm(d_perceiver),
            })
            for _ in range(num_perceiver_layers)
        ])

        # Final tight projection: 1024 → 16 per query → 64 total
        self.project_out = nn.Linear(d_perceiver, state_size // num_queries)  # 1024 → 16

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
        pass_context = self.pass_embed(pass_idx)  # (d_perceiver,)

        # Expand queries for batch and add pass context
        queries = (self.queries + pass_context).unsqueeze(0).expand(batch_size, -1, -1)
        # queries: (batch, num_queries, d_perceiver) = (batch, 4, 1024)

        # Pass-conditioned layer importance (softmax for normalized weights)
        layer_logits = self.layer_gate(pass_context)  # (16,)
        layer_weights = F.softmax(layer_logits, dim=-1)

        # Stack all layer hidden states
        stacked = torch.stack(all_layer_hidden_states, dim=0)  # (16, batch, seq, 2048)

        # Weighted combination of ALL transformer layers
        weights = layer_weights.view(self.num_transformer_layers, 1, 1, 1)
        combined = (stacked * weights).sum(dim=0)  # (batch, seq, 2048)

        # Project to perceiver dimension (FULL sequence preserved)
        kv = self.input_project(combined.to(dtype=self.input_project.weight.dtype))
        # kv: (batch, seq, d_perceiver) = (batch, seq, 1024)

        # Ensure queries match dtype
        queries = queries.to(dtype=kv.dtype)

        # 7 layers of deep compression processing
        # Cross-attention uses FULL sequence (kv has shape batch, seq, 1024)
        for layer in self.perceiver_layers:
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
        """
        Count parameters in each component for verification.

        Expected totals for v20 defaults:
            - queries:          4K
            - pass_embed:       20K
            - layer_gate:       ~66K
            - input_project:    ~2.1M
            - perceiver_layers: ~102M
            - project_out:      ~16K
            - Total:            ~105M
        """
        counts = {
            'queries': self.queries.numel(),
            'pass_embed': sum(p.numel() for p in self.pass_embed.parameters()),
            'layer_gate': sum(p.numel() for p in self.layer_gate.parameters()),
            'input_project': sum(p.numel() for p in self.input_project.parameters()),
            'perceiver_layers': sum(
                sum(p.numel() for p in layer.parameters())
                for layer in self.perceiver_layers
            ),
            'project_out': sum(p.numel() for p in self.project_out.parameters()),
        }
        counts['total'] = sum(counts.values())
        return counts


def _test_compressor():
    """Quick sanity check that the module works with v20 architecture."""
    print("Testing Compressor (AllLayerPerceiver) for Mycelium v20...")

    # Create module with v20 defaults
    compressor = Compressor(
        num_transformer_layers=16,
        d_transformer=2048,
        d_perceiver=1024,
        num_queries=4,
        num_perceiver_layers=7,
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
