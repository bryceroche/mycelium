"""
Compressor v3: Two-headed output — State (64) + Strategy (512)

State:    64 floats, accumulates on hypersphere, persistent memory
Strategy: 512 floats, overwrites each pass, rich signal to hypernetwork

The hypernetwork goes from starving (64 → 256) to well-fed (576 → 256).
Each LoRA scale now has ~2.25 floats of information instead of ~0.25.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class Compressor(nn.Module):
    """
    Two-headed Perceiver: outputs both state delta and strategy vector.
    
    State delta:  64 floats, accumulates on hypersphere (persistent memory)
    Strategy:     512 floats, overwritten each pass (attention instructions)
    
    The strategy channel feeds the hypernetwork with rich signal without
    expanding the bottleneck. Memory is still 64 floats.
    """

    def __init__(
        self,
        num_transformer_layers: int = 16,
        d_transformer: int = 2048,
        d_perceiver: int = 1024,
        num_queries: int = 4,
        num_perceiver_layers: int = 7,
        state_size: int = 64,
        strategy_size: int = 512,  # NEW: strategy vector size
        max_passes: int = 20,
    ) -> None:
        super().__init__()

        self.num_transformer_layers = num_transformer_layers
        self.d_transformer = d_transformer
        self.d_perceiver = d_perceiver
        self.num_queries = num_queries
        self.num_perceiver_layers = num_perceiver_layers
        self.state_size = state_size
        self.strategy_size = strategy_size
        self.max_passes = max_passes

        # Learned queries
        self.queries = nn.Parameter(torch.randn(num_queries, d_perceiver) * 0.02)

        # Pass embedding
        self.pass_embed = nn.Embedding(max_passes, d_perceiver)

        # Layer gate
        self.layer_gate = nn.Sequential(
            nn.Linear(d_perceiver, 64),
            nn.ReLU(),
            nn.Linear(64, num_transformer_layers),
        )

        # Input projection
        self.input_project = nn.Linear(d_transformer, d_perceiver)

        # 7-layer perceiver stack
        self.perceiver_layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(
                    embed_dim=d_perceiver, num_heads=8,
                    kdim=d_perceiver, vdim=d_perceiver, batch_first=True,
                ),
                'cross_norm': nn.LayerNorm(d_perceiver),
                'self_attn': nn.MultiheadAttention(
                    embed_dim=d_perceiver, num_heads=8, batch_first=True,
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

        # TWO OUTPUT HEADS:
        # 1. State head: 64 floats (accumulates, tight bottleneck)
        self.state_head = nn.Linear(d_perceiver * num_queries, state_size)
        
        # 2. Strategy head: 512 floats (overwrites, rich signal to hypernetwork)
        self.strategy_head = nn.Linear(d_perceiver * num_queries, strategy_size)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.input_project.weight)
        nn.init.zeros_(self.input_project.bias)
        nn.init.xavier_uniform_(self.state_head.weight)
        nn.init.zeros_(self.state_head.bias)
        nn.init.xavier_uniform_(self.strategy_head.weight)
        nn.init.zeros_(self.strategy_head.bias)
        nn.init.zeros_(self.layer_gate[0].weight)
        nn.init.zeros_(self.layer_gate[0].bias)
        nn.init.zeros_(self.layer_gate[2].weight)
        nn.init.zeros_(self.layer_gate[2].bias)

    def forward(
        self,
        all_layer_hidden_states: List[torch.Tensor],
        pass_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress hidden states into state delta AND strategy vector.

        Args:
            all_layer_hidden_states: List of 16 tensors, each (batch, seq_len, 2048)
            pass_num: Which thinking pass (0-indexed)

        Returns:
            state_delta: (batch, 64) — accumulates on hypersphere
            strategy:    (batch, 512) — overwrites, feeds hypernetwork
        """
        assert len(all_layer_hidden_states) == self.num_transformer_layers

        batch_size = all_layer_hidden_states[0].size(0)
        device = all_layer_hidden_states[0].device
        dtype = all_layer_hidden_states[0].dtype

        # Pass conditioning
        pass_idx = torch.tensor(pass_num, device=device)
        pass_context = self.pass_embed(pass_idx)
        queries = (self.queries + pass_context).unsqueeze(0).expand(batch_size, -1, -1)

        # Layer gating
        layer_logits = self.layer_gate(pass_context)
        layer_weights = F.softmax(layer_logits, dim=-1)

        # Combine layers
        stacked = torch.stack(all_layer_hidden_states, dim=0)
        weights = layer_weights.view(self.num_transformer_layers, 1, 1, 1)
        combined = (stacked * weights).sum(dim=0)

        # Project to perceiver space
        kv = self.input_project(combined.to(dtype=self.input_project.weight.dtype))
        queries = queries.to(dtype=kv.dtype)

        # 7 perceiver layers
        for layer in self.perceiver_layers:
            attended, _ = layer['cross_attn'](query=queries, key=kv, value=kv)
            queries = layer['cross_norm'](queries + attended)
            
            refined, _ = layer['self_attn'](query=queries, key=queries, value=queries)
            queries = layer['self_norm'](queries + refined)
            
            ffn_out = layer['ffn'](queries)
            queries = layer['ffn_norm'](queries + ffn_out)

        # Flatten queries: (batch, num_queries * d_perceiver)
        flat = queries.flatten(start_dim=1)

        # TWO OUTPUTS:
        state_delta = self.state_head(flat)    # (batch, 64) — memory
        strategy = self.strategy_head(flat)     # (batch, 512) — instructions

        return state_delta, strategy

    def count_parameters(self) -> dict:
        counts = {
            'queries': self.queries.numel(),
            'pass_embed': sum(p.numel() for p in self.pass_embed.parameters()),
            'layer_gate': sum(p.numel() for p in self.layer_gate.parameters()),
            'input_project': sum(p.numel() for p in self.input_project.parameters()),
            'perceiver_layers': sum(
                sum(p.numel() for p in layer.parameters())
                for layer in self.perceiver_layers
            ),
            'state_head': sum(p.numel() for p in self.state_head.parameters()),
            'strategy_head': sum(p.numel() for p in self.strategy_head.parameters()),
        }
        counts['total'] = sum(counts.values())
        return counts


if __name__ == "__main__":
    print("Testing Compressor v3 with strategy channel...")
    
    compressor = Compressor()
    counts = compressor.count_parameters()
    
    print(f"\nParameter counts:")
    for name, count in counts.items():
        if count > 1e6:
            print(f"  {name}: {count / 1e6:.1f}M")
        else:
            print(f"  {name}: {count:,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    all_layer_hidden_states = [torch.randn(batch_size, seq_len, 2048) for _ in range(16)]
    
    state_delta, strategy = compressor(all_layer_hidden_states, pass_num=1)
    
    print(f"\nOutput shapes:")
    print(f"  state_delta: {state_delta.shape} (memory, accumulates)")
    print(f"  strategy:    {strategy.shape} (instructions, overwrites)")
    
    # Gradient flow
    (state_delta.sum() + strategy.sum()).backward()
    print("\nGradient flow: OK")
    print("\nAll tests passed!")
