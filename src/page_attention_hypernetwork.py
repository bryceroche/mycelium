"""
PageAttentionHypernetwork: cross-attention over a list of 64-float "pages" + a
512-float strategy → 256 LoRA scales, packaged in the same lora_mods dict
shape that AdditiveLoRAManager expects.

Replaces StateConditionedLoRA's flat Linear(576→256) hypernet. Templates
(A/B) live here so this module is a drop-in for v21.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class PageAttentionHypernetwork(nn.Module):
    def __init__(
        self,
        d_model: int = 2048,
        d_kv: int = 512,
        page_size: int = 64,
        strategy_size: int = 64,
        rank: int = 4,
        num_layers: int = 16,
        num_projections: int = 4,
        attn_dim: int = 256,
        num_query_heads: int = 4,
        num_attn_heads: int = 4,
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

        # Templates (same shape as StateConditionedLoRA — warm-startable)
        self.A_templates = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, d_model, rank) * 0.01)
            for _ in range(num_projections)
        ])
        self.B_templates = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, rank, self.proj_dims[i]) * 0.01)
            for i in range(num_projections)
        ])

        # Page attention
        self.page_project = nn.Linear(page_size, attn_dim)
        self.page_query = nn.Parameter(torch.randn(num_query_heads, attn_dim) * 0.02)
        self.page_attn = nn.MultiheadAttention(
            attn_dim, num_heads=num_attn_heads, batch_first=True,
        )
        self.page_norm = nn.LayerNorm(attn_dim)

        # Combine page summary + strategy → scales
        num_scales = num_layers * num_projections * rank
        combined_dim = num_query_heads * attn_dim + strategy_size
        self.combine = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.GELU(),
            nn.Linear(512, num_scales),
            nn.Tanh(),
        )

    def compute_scales(
        self,
        state_pages: List[torch.Tensor],
        strategy: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Returns (all_scales: (B, num_scales), attn_weights or None)."""
        batch_size = strategy.size(0)
        if len(state_pages) == 0:
            num_scales = self.num_layers * self.num_projections * self.rank
            return (
                torch.zeros(batch_size, num_scales, device=strategy.device, dtype=strategy.dtype),
                None,
            )
        pages = torch.stack(state_pages, dim=1)            # (B, P, page_size)
        pages_proj = self.page_project(pages)              # (B, P, attn_dim)
        queries = self.page_query.unsqueeze(0).expand(batch_size, -1, -1)
        attended, attn_weights = self.page_attn(
            query=queries, key=pages_proj, value=pages_proj,
        )                                                  # (B, Q, attn_dim)
        attended = self.page_norm(attended)
        page_summary = attended.flatten(start_dim=1)       # (B, Q*attn_dim)
        combined = torch.cat([page_summary, strategy], dim=-1)
        scales = self.combine(combined)                    # (B, num_scales)
        return scales, attn_weights

    def forward(
        self,
        state_pages: List[torch.Tensor],
        strategy: torch.Tensor,
    ) -> Dict[int, Dict[str, Dict[str, torch.Tensor]]]:
        all_scales, _ = self.compute_scales(state_pages, strategy)
        batch_size = all_scales.size(0)
        all_scales = all_scales.reshape(
            batch_size, self.num_layers, self.num_projections, self.rank,
        )
        lora_mods: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}
        for layer_idx in range(self.num_layers):
            lora_mods[layer_idx] = {}
            for proj_idx, proj_name in enumerate(self.proj_names):
                lora_mods[layer_idx][proj_name] = {
                    'A': self.A_templates[proj_idx][layer_idx],
                    'B': self.B_templates[proj_idx][layer_idx],
                    'scales': all_scales[:, layer_idx, proj_idx, :],
                }
        return lora_mods


if __name__ == '__main__':
    h = PageAttentionHypernetwork()
    pages = [torch.randn(2, 64) for _ in range(3)]
    strategy = torch.randn(2, 512)
    mods = h(pages, strategy)
    print(f"layers={len(mods)} q_scales={mods[0]['q_proj']['scales'].shape}")
    # Empty pages
    mods0 = h([], strategy)
    print(f"empty pages q_scales sum={mods0[0]['q_proj']['scales'].abs().sum().item():.4f}")
    print("OK")
