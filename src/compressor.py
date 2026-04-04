"""
Compressor v2: Now with STATE-CONDITIONED QUERIES.

The key change: queries now know what is already in the state, so they can
extract NEW information instead of re-extracting the same thing.

Change from v1:
    self.state_project = nn.Linear(state_size, d_perceiver)  # 64 → 1024
    queries = self.queries + pass_context + state_context   # NEW: what is already encoded
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class Compressor(nn.Module):
    """
    AllLayerPerceiver v2: Now with state-conditioned queries.
    
    The queries now know:
    1. What pass we are on (pass_context)
    2. What is already encoded in state (state_context)
    
    This closes the communication loop: "I am on pass 2, state already has 48,
    extract something NEW."
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
        
        # NEW: State projection - tells queries what is already encoded
        # This closes the communication loop
        self.state_project = nn.Linear(state_size, d_perceiver)  # 64 → 1024

        # Pass-conditioned layer gate (softmax weights over 16 transformer layers)
        self.layer_gate = nn.Sequential(
            nn.Linear(d_perceiver, 64),
            nn.ReLU(),
            nn.Linear(64, num_transformer_layers),
        )

        # Project from Llama space to perceiver space
        self.input_project = nn.Linear(d_transformer, d_perceiver)  # 2048 → 1024

        # 7-layer perceiver stack
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
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.input_project.weight)
        nn.init.zeros_(self.input_project.bias)
        nn.init.xavier_uniform_(self.project_out.weight)
        nn.init.zeros_(self.project_out.bias)
        
        # Initialize state_project to start with small influence
        nn.init.xavier_uniform_(self.state_project.weight)
        nn.init.zeros_(self.state_project.bias)

        # Initialize layer gate to start with uniform attention
        nn.init.zeros_(self.layer_gate[0].weight)
        nn.init.zeros_(self.layer_gate[0].bias)
        nn.init.zeros_(self.layer_gate[2].weight)
        nn.init.zeros_(self.layer_gate[2].bias)

    def forward(
        self,
        all_layer_hidden_states: List[torch.Tensor],
        pass_num: int,
        current_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compress all transformer layer hidden states into a 64-float state delta.

        Args:
            all_layer_hidden_states: List of 16 tensors, each (batch, seq_len, 2048)
            pass_num: Which thinking pass (0-indexed)
            current_state: Current accumulated state (batch, 64) - NEW parameter
                          If None, no state conditioning is applied (backward compatible)

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
        
        # NEW: Get state context - tells queries what is already encoded
        if current_state is not None:
            # current_state: (batch, 64)
            state_context = self.state_project(current_state.to(dtype=self.state_project.weight.dtype))
            # state_context: (batch, d_perceiver)
            
            # Expand queries for batch and add BOTH pass and state context
            # queries = base_queries + pass_context (broadcast) + state_context (per-sample)
            queries = self.queries.unsqueeze(0) + pass_context.unsqueeze(0) + state_context.unsqueeze(1)
            # queries: (batch, num_queries, d_perceiver)
        else:
            # Backward compatible: just use pass context
            queries = (self.queries + pass_context).unsqueeze(0).expand(batch_size, -1, -1)

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
        # kv: (batch, seq, d_perceiver)

        # Ensure queries match dtype
        queries = queries.to(dtype=kv.dtype)

        # 7 layers of deep compression processing
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

        # Project to tight bottleneck
        state_delta = self.project_out(queries)  # (batch, 4, 16)

        return state_delta.flatten(start_dim=1)  # (batch, 64)

    def get_layer_weights(self, pass_num: int, device: torch.device = None) -> torch.Tensor:
        """Get the learned layer importance weights for a given pass."""
        if device is None:
            device = self.queries.device

        pass_idx = torch.tensor(pass_num, device=device)
        pass_context = self.pass_embed(pass_idx)
        layer_logits = self.layer_gate(pass_context)
        layer_weights = F.softmax(layer_logits, dim=-1)

        return layer_weights

    def count_parameters(self) -> dict:
        """Count parameters in each component."""
        counts = {
            'queries': self.queries.numel(),
            'pass_embed': sum(p.numel() for p in self.pass_embed.parameters()),
            'state_project': sum(p.numel() for p in self.state_project.parameters()),  # NEW
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


if __name__ == "__main__":
    print("Testing Compressor v2 with state conditioning...")
    
    compressor = Compressor()
    
    # Count parameters
    param_counts = compressor.count_parameters()
    print(f"\nParameter counts:")
    for name, count in param_counts.items():
        if count > 1e6:
            print(f"  {name}: {count / 1e6:.1f}M")
        else:
            print(f"  {name}: {count:,}")
    
    # Test forward pass WITH state
    batch_size = 2
    seq_len = 128
    
    all_layer_hidden_states = [torch.randn(batch_size, seq_len, 2048) for _ in range(16)]
    current_state = torch.randn(batch_size, 64)
    
    state_delta = compressor(all_layer_hidden_states, pass_num=1, current_state=current_state)
    print(f"\nWith state conditioning: {state_delta.shape}")
    
    # Test backward compatibility (no state)
    state_delta_no_state = compressor(all_layer_hidden_states, pass_num=1, current_state=None)
    print(f"Without state (backward compat): {state_delta_no_state.shape}")
    
    # Test gradient flow
    state_delta.sum().backward()
    print("\nGradient flow: OK")
    print("\nAll tests passed!")
