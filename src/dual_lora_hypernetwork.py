"""
DualPageHypernetwork: two sets of LoRA templates (forward + verify) with
a learned blend weight that transitions from computation to verification.

Forward templates: narrow, sequential attention for computation.
Verify templates: broad, relational attention for consistency checking.

The hypernetwork outputs forward_scales, verify_scales, and blend per pass.
The AdditiveLoRAManager applies blended LoRA:
    q_forward = (x @ A_forward) * forward_scales @ B_forward
    q_verify  = (x @ A_verify)  * verify_scales  @ B_verify
    q_lora    = (1 - blend) * q_forward + blend * q_verify

Expected blend trajectory (learned, not hardcoded):
    Pass 1: blend ≈ 0.1  (mostly computing)
    Pass 2: blend ≈ 0.3  (computing + starting to check)
    Pass 3: blend ≈ 0.7  (mostly verifying)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class DualPageHypernetwork(nn.Module):
    def __init__(
        self,
        d_model: int = 2048,
        d_kv: int = 512,
        page_size: int = 64,
        strategy_size: int = 512,
        rank: int = 4,
        num_layers: int = 16,
        num_projections: int = 4,
        attn_dim: int = 256,
        num_query_heads: int = 4,
        num_attn_heads: int = 4,
        max_passes: int = 10,
        pass_embed_dim: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_kv = d_kv
        self.page_size = page_size
        self.strategy_size = strategy_size
        self.rank = rank
        self.num_layers = num_layers
        self.num_projections = num_projections
        self.num_query_heads = num_query_heads
        self.attn_dim = attn_dim

        self.proj_dims = [d_model, d_kv, d_kv, d_model]
        self.proj_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

        num_scales = num_layers * num_projections * rank

        # Forward templates (computation-focused attention)
        self.A_forward = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, d_model, rank) * 0.01)
            for _ in range(num_projections)
        ])
        self.B_forward = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, rank, self.proj_dims[i]) * 0.01)
            for i in range(num_projections)
        ])

        # Verify templates (consistency-checking attention)
        self.A_verify = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, d_model, rank) * 0.01)
            for _ in range(num_projections)
        ])
        self.B_verify = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, rank, self.proj_dims[i]) * 0.01)
            for i in range(num_projections)
        ])

        # Page attention (shared between forward and verify)
        self.page_project = nn.Linear(page_size, attn_dim)
        self.page_query = nn.Parameter(torch.randn(num_query_heads, attn_dim) * 0.02)
        self.page_attn = nn.MultiheadAttention(
            attn_dim, num_heads=num_attn_heads, batch_first=True,
        )
        self.page_norm = nn.LayerNorm(attn_dim)

        # Pass embedding
        self.pass_embed = nn.Embedding(max_passes, pass_embed_dim)
        self.pass_embed_dim = pass_embed_dim

        # Combine → forward_scales (256) + verify_scales (256) + blend (1)
        combined_dim = num_query_heads * attn_dim + strategy_size + pass_embed_dim
        self.combine = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.GELU(),
            nn.Linear(512, num_scales * 2 + 1),
        )
        self._num_scales = num_scales

    def compute_scales_and_blend(
        self,
        state_pages: List[torch.Tensor],
        strategy: torch.Tensor,
        pass_num: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            forward_scales: (B, num_scales) tanh-bounded
            verify_scales:  (B, num_scales) tanh-bounded
            blend:          (B, 1) sigmoid-bounded [0=forward, 1=verify]
        """
        batch_size = strategy.size(0)
        device = strategy.device

        pass_t = torch.tensor([pass_num], device=device)
        pass_emb = self.pass_embed(pass_t).expand(batch_size, -1)

        if len(state_pages) == 0:
            return (
                torch.zeros(batch_size, self._num_scales, device=device, dtype=strategy.dtype),
                torch.zeros(batch_size, self._num_scales, device=device, dtype=strategy.dtype),
                torch.zeros(batch_size, 1, device=device, dtype=strategy.dtype),
            )

        pages = torch.stack(state_pages, dim=1)
        pages_proj = self.page_project(pages)
        queries = self.page_query.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.page_attn(query=queries, key=pages_proj, value=pages_proj)
        attended = self.page_norm(attended)
        page_summary = attended.flatten(start_dim=1)

        combined = torch.cat([page_summary, strategy, pass_emb], dim=-1)
        out = self.combine(combined)

        forward_scales = torch.tanh(out[:, :self._num_scales])
        verify_scales = torch.tanh(out[:, self._num_scales:self._num_scales * 2])
        blend = torch.sigmoid(out[:, self._num_scales * 2:])  # (B, 1)

        return forward_scales, verify_scales, blend

    def forward(
        self,
        state_pages: List[torch.Tensor],
        strategy: torch.Tensor,
        pass_num: int = 0,
    ) -> Tuple[Dict, Dict, torch.Tensor]:
        """
        Returns:
            forward_mods: lora_mods dict for forward templates
            verify_mods:  lora_mods dict for verify templates
            blend:        (B, 1) blend weight
        """
        forward_scales, verify_scales, blend = self.compute_scales_and_blend(
            state_pages, strategy, pass_num,
        )
        batch_size = forward_scales.size(0)

        forward_scales = forward_scales.reshape(
            batch_size, self.num_layers, self.num_projections, self.rank,
        )
        verify_scales = verify_scales.reshape(
            batch_size, self.num_layers, self.num_projections, self.rank,
        )

        forward_mods = {}
        verify_mods = {}
        for layer_idx in range(self.num_layers):
            forward_mods[layer_idx] = {}
            verify_mods[layer_idx] = {}
            for proj_idx, proj_name in enumerate(self.proj_names):
                forward_mods[layer_idx][proj_name] = {
                    'A': self.A_forward[proj_idx][layer_idx],
                    'B': self.B_forward[proj_idx][layer_idx],
                    'scales': forward_scales[:, layer_idx, proj_idx, :],
                }
                verify_mods[layer_idx][proj_name] = {
                    'A': self.A_verify[proj_idx][layer_idx],
                    'B': self.B_verify[proj_idx][layer_idx],
                    'scales': verify_scales[:, layer_idx, proj_idx, :],
                }

        return forward_mods, verify_mods, blend

    def warm_start_from_single(self, single_hypernet_state: dict):
        """
        Warm-start forward templates from a single-LoRA checkpoint.
        Verify templates stay randomly initialized (they learn from scratch).
        Shared components (page_attn, combine, pass_embed) are loaded where compatible.
        """
        loaded = 0
        skipped = 0
        own_state = self.state_dict()

        for key, value in single_hypernet_state.items():
            # Map single-LoRA template names to forward templates
            mapped_key = key
            if key.startswith('A_templates.'):
                mapped_key = key.replace('A_templates.', 'A_forward.')
            elif key.startswith('B_templates.'):
                mapped_key = key.replace('B_templates.', 'B_forward.')

            if mapped_key in own_state and own_state[mapped_key].shape == value.shape:
                own_state[mapped_key] = value
                loaded += 1
            else:
                # Try loading shared components directly
                if key in own_state and own_state[key].shape == value.shape:
                    own_state[key] = value
                    loaded += 1
                else:
                    skipped += 1

        self.load_state_dict(own_state, strict=False)
        print(f"  dual hypernet warm start: loaded {loaded}, skipped {skipped}")
        return loaded, skipped


if __name__ == '__main__':
    h = DualPageHypernetwork()
    pages = [torch.randn(2, 64) for _ in range(3)]
    strategy = torch.randn(2, 512)

    forward_mods, verify_mods, blend = h(pages, strategy, pass_num=1)
    print(f"layers={len(forward_mods)}")
    print(f"forward q_scales={forward_mods[0]['q_proj']['scales'].shape}")
    print(f"verify q_scales={verify_mods[0]['q_proj']['scales'].shape}")
    print(f"blend={blend.shape} values={blend.squeeze().tolist()}")

    # Empty pages
    f0, v0, b0 = h([], strategy)
    print(f"empty: forward sum={f0[0]['q_proj']['scales'].abs().sum():.4f}")

    # Param count
    total = sum(p.numel() for p in h.parameters())
    forward_params = sum(p.numel() for p in list(h.A_forward.parameters()) + list(h.B_forward.parameters()))
    verify_params = sum(p.numel() for p in list(h.A_verify.parameters()) + list(h.B_verify.parameters()))
    print(f"Total params: {total:,}")
    print(f"Forward templates: {forward_params:,}")
    print(f"Verify templates: {verify_params:,}")
    print(f"Shared (attn+combine+pass): {total - forward_params - verify_params:,}")
    print("OK")
