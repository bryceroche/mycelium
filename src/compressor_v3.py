"""
Compressor v3: Two-headed output — State (64) + Strategy (512)

State:    64 floats, accumulates on hypersphere, persistent memory
Strategy: 512 floats, overwrites each pass, rich signal to hypernetwork

The hypernetwork goes from starving (64 → 256) to well-fed (576 → 256).
Each LoRA scale now has ~2.25 floats of information instead of ~0.25.

v24.3: Optional Haar wavelet preprocessing for 2x input compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .haar_wavelet import LayerWiseWavelet


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
        strategy_size: int = 64,  # shrunk from 512 → 64 to close bypass
        max_passes: int = 20,
        use_wavelet: bool = True,  # v24.3: Haar wavelet preprocessing (default ON)
        wavelet_levels: int = 4,
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
        self.use_wavelet = use_wavelet

        # v24.3: Optional wavelet preprocessing (2x input compression)
        if use_wavelet:
            self.wavelet = LayerWiseWavelet(
                num_layers=num_transformer_layers,
                max_level=wavelet_levels,
                truncate_finest=True,
                share_weights=True,
            )
        else:
            self.wavelet = None

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

        # Skip connection: cross-attend to previous passes' mid-layer states
        self.skip_attn = nn.MultiheadAttention(d_perceiver, num_heads=8, batch_first=True)
        self.skip_norm = nn.LayerNorm(d_perceiver)

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
        prev_mid_states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compress hidden states into state delta AND strategy vector.

        Args:
            all_layer_hidden_states: List of 16 tensors, each (batch, seq_len, 2048)
            pass_num: Which thinking pass (0-indexed)
            prev_mid_states: Optional list of (batch, num_queries, d_perceiver) tensors
                from previous passes' mid-layer (after layer 4). If None or empty,
                skip connection is not applied (backward compatible).

        Returns:
            state_delta:       (batch, 64) — accumulates on hypersphere
            strategy:          (batch, 512) — overwrites, feeds hypernetwork
            current_mid_states: (batch, num_queries, d_perceiver) — detached mid-layer
                states for future passes to cross-attend to
        """
        assert len(all_layer_hidden_states) == self.num_transformer_layers

        batch_size = all_layer_hidden_states[0].size(0)
        device = all_layer_hidden_states[0].device
        dtype = all_layer_hidden_states[0].dtype

        # v24.3: Optional wavelet preprocessing (2x compression before perceiver)
        if self.wavelet is not None:
            all_layer_hidden_states = self.wavelet(all_layer_hidden_states)

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

        # Layers 0-3 (first half)
        for layer in self.perceiver_layers[:4]:
            attended, _ = layer['cross_attn'](query=queries, key=kv, value=kv)
            queries = layer['cross_norm'](queries + attended)

            refined, _ = layer['self_attn'](query=queries, key=queries, value=queries)
            queries = layer['self_norm'](queries + refined)

            ffn_out = layer['ffn'](queries)
            queries = layer['ffn_norm'](queries + ffn_out)

        # Save mid-layer states (detached — no gradient through stored states)
        current_mid_states = queries.detach().clone()

        # Skip connection: cross-attend to previous passes' mid-layer states
        if prev_mid_states is not None and len(prev_mid_states) > 0:
            prev = torch.cat(prev_mid_states, dim=1)  # (batch, P*num_queries, d_perceiver)
            skip_out, _ = self.skip_attn(query=queries, key=prev, value=prev)
            queries = self.skip_norm(queries + skip_out)

        # Layers 4-6 (second half)
        for layer in self.perceiver_layers[4:]:
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

        return state_delta, strategy, current_mid_states

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
            'skip_attn': sum(p.numel() for p in self.skip_attn.parameters()),
            'skip_norm': sum(p.numel() for p in self.skip_norm.parameters()),
            'state_head': sum(p.numel() for p in self.state_head.parameters()),
            'strategy_head': sum(p.numel() for p in self.strategy_head.parameters()),
        }
        if self.wavelet is not None:
            counts['wavelet'] = self.wavelet.count_parameters()
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

    state_delta, strategy, mid_states = compressor(all_layer_hidden_states, pass_num=0)

    print(f"\nOutput shapes:")
    print(f"  state_delta:  {state_delta.shape} (memory, accumulates)")
    print(f"  strategy:     {strategy.shape} (instructions, overwrites)")
    print(f"  mid_states:   {mid_states.shape} (perceiver mid-layer, private memory)")

    # Gradient flow (no prev_mid_states)
    (state_delta.sum() + strategy.sum()).backward()
    print("\nGradient flow (no skip): OK")

    # Test with skip connection
    compressor.zero_grad()
    mid_history = [mid_states]
    state_delta2, strategy2, mid_states2 = compressor(
        all_layer_hidden_states, pass_num=1, prev_mid_states=mid_history,
    )
    (state_delta2.sum() + strategy2.sum()).backward()
    print("Gradient flow (with skip): OK")
    print(f"  mid_states2: {mid_states2.shape}")

    # Test with wavelet preprocessing (v24.3)
    print("\n--- Testing with Wavelet Preprocessing (v24.3) ---")
    compressor_wavelet = Compressor(use_wavelet=True, wavelet_levels=4)
    counts_wavelet = compressor_wavelet.count_parameters()

    print(f"\nParameter counts (with wavelet):")
    for name, count in counts_wavelet.items():
        if count > 1e6:
            print(f"  {name}: {count / 1e6:.1f}M")
        else:
            print(f"  {name}: {count:,}")

    state_delta_w, strategy_w, mid_states_w = compressor_wavelet(
        all_layer_hidden_states, pass_num=0
    )

    print(f"\nWith wavelet:")
    print(f"  Input:  {len(all_layer_hidden_states)} layers x {all_layer_hidden_states[0].shape}")
    print(f"  state_delta:  {state_delta_w.shape}")
    print(f"  strategy:     {strategy_w.shape}")
    print(f"  mid_states:   {mid_states_w.shape}")

    # Gradient flow
    compressor_wavelet.zero_grad()
    (state_delta_w.sum() + strategy_w.sum()).backward()
    wavelet_grad = compressor_wavelet.wavelet.wavelet.level_weights.grad
    print(f"  Wavelet level_weights grad: {wavelet_grad}")
    print("Gradient flow (wavelet): OK")

    print("\nAll tests passed!")
