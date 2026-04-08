"""
PageToTokens: cross-attention over a list of 64-float pages → N pseudo-tokens
at d_model. Used during the generation pass (LoRA OFF) to give Llama a gentle
state injection that doesn't override the language head.
"""

import torch
import torch.nn as nn
from typing import List


class PageToTokens(nn.Module):
    def __init__(
        self,
        page_size: int = 64,
        d_model: int = 2048,
        num_tokens: int = 8,
        attn_dim: int = 256,
        num_heads: int = 4,
    ):
        super().__init__()
        self.page_size = page_size
        self.d_model = d_model
        self.num_tokens = num_tokens

        self.page_project = nn.Linear(page_size, attn_dim)
        self.queries = nn.Parameter(torch.randn(num_tokens, attn_dim) * 0.02)
        self.attn = nn.MultiheadAttention(attn_dim, num_heads=num_heads, batch_first=True)
        self.output_project = nn.Linear(attn_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

        # Small init on output so initial pseudo-tokens are tiny perturbations
        nn.init.normal_(self.output_project.weight, std=0.01)
        nn.init.zeros_(self.output_project.bias)

    def forward(self, state_pages: List[torch.Tensor]) -> torch.Tensor:
        """
        state_pages: list of (B, page_size) tensors
        returns: (B, num_tokens, d_model)
        """
        pages = torch.stack(state_pages, dim=1)            # (B, P, page_size)
        pages_proj = self.page_project(pages)              # (B, P, attn_dim)
        batch_size = pages.size(0)
        q = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.attn(query=q, key=pages_proj, value=pages_proj)
        tokens = self.output_project(attended)             # (B, num_tokens, d_model)
        return self.norm(tokens)


if __name__ == '__main__':
    p2t = PageToTokens()
    pages = [torch.randn(2, 64) for _ in range(3)]
    tokens = p2t(pages)
    print(f"OK shape={tokens.shape}")  # (2, 8, 2048)
