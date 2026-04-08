"""
PseudoTokenHead: maps the (state, strategy) bottleneck into N "soft prompt"
embedding vectors that get prepended to the prompt during generation.

The idea: LoRA is great for thinking (powerful, rewires attention) but
catastrophic for generation (overspecializes to "emit number, stop"). Pseudo-
tokens are a much gentler form of state injection — they sit at the start of
the context as if they were ordinary tokens, biasing attention without
overriding the language head's natural distribution.

Hybrid pipeline:
    1. Thinking passes:    LoRA ON,  pseudo-tokens NOT used
    2. Generation:         LoRA OFF, pseudo-tokens prepended to embeddings

The state after the final thinking pass is what feeds the pseudo-token head.
"""

import torch
import torch.nn as nn


class PseudoTokenHead(nn.Module):
    """
    Maps (state, strategy) -> num_tokens soft embeddings of dimension d_model.

    These embeddings are concatenated to the front of the regular token
    embeddings before generation. They occupy attention positions but never
    correspond to vocabulary tokens.

    Args:
        state_size:    dimension of the state vector (default 64)
        strategy_size: dimension of the strategy side channel (default 512)
        d_model:       transformer hidden size (Llama 3.2 1B: 2048)
        num_tokens:    number of pseudo tokens to produce (default 4)
        hidden:        MLP hidden width (default 512)
    """

    def __init__(
        self,
        state_size: int = 64,
        strategy_size: int = 512,
        d_model: int = 2048,
        num_tokens: int = 4,
        hidden: int = 512,
    ):
        super().__init__()
        self.state_size = state_size
        self.strategy_size = strategy_size
        self.d_model = d_model
        self.num_tokens = num_tokens

        in_dim = state_size + strategy_size
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_tokens * d_model),
        )

        # Small init so the prepended tokens start as a tiny perturbation
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, state: torch.Tensor, strategy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state:    (batch, state_size)
            strategy: (batch, strategy_size)
        Returns:
            (batch, num_tokens, d_model) — soft prompt embeddings
        """
        x = torch.cat([state, strategy], dim=-1)
        out = self.mlp(x)
        return out.view(-1, self.num_tokens, self.d_model)
