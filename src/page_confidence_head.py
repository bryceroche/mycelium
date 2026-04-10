"""
PageConfidenceHead: reads accumulated pages via cross-attention and predicts
whether the model has thought enough to produce a correct answer.

Trained with CORRECTNESS signal: at each pass, check if pages would produce
the right answer. Target = 1.0 if correct, 0.0 if not. No efficiency penalty.

Usage:
    head = PageConfidenceHead()
    conf = head(state_pages)  # (B, 1) in [0, 1]
"""
import torch
import torch.nn as nn
from typing import List


class PageConfidenceHead(nn.Module):
    def __init__(self, page_size: int = 64, hidden: int = 128, num_heads: int = 4):
        super().__init__()
        self.page_project = nn.Linear(page_size, hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads=num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, hidden) * 0.02)
        self.output = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, state_pages: List[torch.Tensor]) -> torch.Tensor:
        """
        state_pages: list of (B, page_size) tensors
        returns: (B, 1) confidence in [0, 1]
        """
        pages = torch.stack(state_pages, dim=1)       # (B, P, page_size)
        pages_proj = self.page_project(pages)          # (B, P, hidden)
        batch_size = pages.size(0)
        q = self.query.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 1, hidden)
        attended, _ = self.attn(query=q, key=pages_proj, value=pages_proj)
        return self.output(attended.squeeze(1))        # (B, 1)


if __name__ == '__main__':
    head = PageConfidenceHead()
    pages = [torch.randn(4, 64) for _ in range(3)]
    conf = head(pages)
    print(f"OK shape={conf.shape} range=[{conf.min():.3f}, {conf.max():.3f}]")
    print(f"Params: {sum(p.numel() for p in head.parameters()):,}")
