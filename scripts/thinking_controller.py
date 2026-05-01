"""
Thinking Controller (~350M) — the only trainable component.

Reads Llama's hidden states, thinks in pages (inner loop), makes discrete
decisions (decompose/solve/merge), and injects soft tokens into Llama's
generation for differentiable steering.

No atoms. No scales. No ST gradient. No monkey-patching.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Tree data structures
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    page: torch.Tensor              # (256d) on hypersphere
    branch_embed: torch.Tensor      # (64d) L2-clipped
    action: str                     # 'decompose' / 'solve' / 'merge'
    cycle_num: int = 0
    pass_num: int = 0
    parent_idx: int = -1
    children_idx: list = field(default_factory=list)
    claimed_target: Optional[int] = None
    generation: str = ""
    energy: float = 1.0


class TreeNotebook:
    def __init__(self):
        self.nodes: List[TreeNode] = []

    def append(self, node: TreeNode) -> int:
        idx = len(self.nodes)
        self.nodes.append(node)
        return idx

    def __len__(self):
        return len(self.nodes)


# ---------------------------------------------------------------------------
# State Encoder (~150M) — Perceiver reading Llama hidden states
# ---------------------------------------------------------------------------

class StateEncoder(nn.Module):
    def __init__(self, hidden_dim=2048, latent_dim=1536, num_heads=8,
                 num_layers=4, num_llama_layers=16):
        super().__init__()
        self.latent_dim = latent_dim

        self.layer_weights = nn.Parameter(torch.ones(num_llama_layers) / num_llama_layers)

        self.input_project = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(
                    embed_dim=latent_dim, num_heads=num_heads, batch_first=True,
                ),
                'cross_norm': nn.LayerNorm(latent_dim),
                'self_attn': nn.MultiheadAttention(
                    embed_dim=latent_dim, num_heads=num_heads, batch_first=True,
                ),
                'self_norm': nn.LayerNorm(latent_dim),
                'ff': nn.Sequential(
                    nn.Linear(latent_dim, latent_dim * 4),
                    nn.GELU(),
                    nn.Linear(latent_dim * 4, latent_dim),
                ),
                'ff_norm': nn.LayerNorm(latent_dim),
            }))

        self.latent_query = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)

    def forward(self, hidden_states_all_layers):
        weights = F.softmax(self.layer_weights, dim=0)
        combined = sum(w * h for w, h in zip(weights, hidden_states_all_layers))
        kv = self.input_project(combined)

        batch_size = kv.shape[0]
        query = self.latent_query.expand(batch_size, -1, -1)

        for layer in self.layers:
            residual = query
            query = layer['cross_norm'](query)
            query, _ = layer['cross_attn'](query, kv, kv)
            query = query + residual

            residual = query
            query = layer['self_norm'](query)
            query, _ = layer['self_attn'](query, query, query)
            query = query + residual

            residual = query
            query = layer['ff_norm'](query)
            query = layer['ff'](query) + residual

        return query.squeeze(1)


# ---------------------------------------------------------------------------
# Page Attention (~90M) — attends to accumulated notebook pages
# ---------------------------------------------------------------------------

class PageAttention(nn.Module):
    def __init__(self, state_dim=1536, page_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.state_dim = state_dim
        self.page_project = nn.Linear(page_dim, state_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(
                    embed_dim=state_dim, num_heads=num_heads, batch_first=True,
                ),
                'norm1': nn.LayerNorm(state_dim),
                'ff': nn.Sequential(
                    nn.Linear(state_dim, state_dim * 4),
                    nn.GELU(),
                    nn.Linear(state_dim * 4, state_dim),
                ),
                'norm2': nn.LayerNorm(state_dim),
            }))

        self.no_history = nn.Parameter(torch.zeros(1, state_dim))

    def forward(self, state, pages):
        """
        Args:
            state: (batch, state_dim)
            pages: list of (batch, page_dim) tensors, or empty list
        """
        if len(pages) == 0:
            return self.no_history.expand(state.shape[0], -1)

        # Stack and project pages
        page_stack = torch.stack(pages, dim=1)  # (batch, N, page_dim)
        kv = self.page_project(page_stack)       # (batch, N, state_dim)

        query = state.unsqueeze(1)

        for layer in self.layers:
            residual = query
            query = layer['norm1'](query)
            query, _ = layer['attn'](query, kv, kv)
            query = query + residual

            residual = query
            query = layer['norm2'](query)
            query = layer['ff'](query) + residual

        return query.squeeze(1)


# ---------------------------------------------------------------------------
# Thinking Controller (~350M)
# ---------------------------------------------------------------------------

class ThinkingController(nn.Module):
    """
    The only trainable component. Thinks in pages, makes decisions,
    and produces soft tokens to steer Llama's generation.
    """

    def __init__(self, hidden_dim=2048, latent_dim=1536,
                 page_dim=256, max_cycles=16, num_soft_tokens=4,
                 num_encoder_layers=4, num_page_layers=4,
                 num_trunk_layers=4, num_heads=8,
                 num_llama_layers=16, branch_embed_dim=64,
                 max_branch_norm=0.95):
        super().__init__()
        self.latent_dim = latent_dim
        self.page_dim = page_dim
        self.hidden_dim = hidden_dim
        self.num_soft_tokens = num_soft_tokens

        # State encoder (~150M)
        self.state_encoder = StateEncoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            num_llama_layers=num_llama_layers,
        )

        # Page attention (~90M)
        self.page_attn = PageAttention(
            state_dim=latent_dim,
            page_dim=page_dim,
            num_heads=num_heads,
            num_layers=num_page_layers,
        )

        # Positional embeddings
        self.cycle_embed = nn.Embedding(max_cycles, 32)
        self.pass_embed = nn.Embedding(8, 32)

        # Trunk (~80M)
        trunk_input_dim = latent_dim * 2 + 32 + 32  # state + context + cycle + pass
        trunk_dim = latent_dim * 2  # 2048
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, trunk_dim),
            nn.LayerNorm(trunk_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        trunk_layers = []
        for _ in range(num_trunk_layers - 1):
            trunk_layers.extend([
                nn.Linear(trunk_dim, trunk_dim),
                nn.LayerNorm(trunk_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
        self.trunk_body = nn.Sequential(*trunk_layers)
        self.trunk_dim = trunk_dim

        # --- Output heads ---

        # Page head: "what did I understand?" (256d hypersphere)
        self.page_head = nn.Sequential(
            nn.Linear(trunk_dim, 512),
            nn.GELU(),
            nn.Linear(512, page_dim),
        )

        # Soft token head: pages → Llama embedding space (differentiable steering)
        # Produces N_soft tokens of dimension hidden_dim (2048)
        self.soft_token_head = nn.Sequential(
            nn.Linear(trunk_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, num_soft_tokens * hidden_dim),
        )

        # Action head: DECOMPOSE / SOLVE / MERGE (Gumbel-softmax)
        self.action_head = nn.Sequential(
            nn.Linear(trunk_dim, 256),
            nn.GELU(),
            nn.Linear(256, 3),
        )

        # Energy head: thinking convergence (sigmoid)
        self.energy_head = nn.Sequential(
            nn.Linear(trunk_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

        # Confidence head: stopping criterion (sigmoid)
        self.confidence_head = nn.Sequential(
            nn.Linear(trunk_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

        # Branch embedding head: tree position (64d, L2-clipped)
        self.branch_embed_head = nn.Sequential(
            nn.Linear(trunk_dim, 256),
            nn.GELU(),
            nn.Linear(256, branch_embed_dim),
        )
        self.max_branch_norm = max_branch_norm

    def think(self, hidden_states_all_layers, all_pages, cycle_num,
              max_passes=3, energy_threshold=0.15):
        """
        Inner thinking loop. Reads hidden states + accumulated pages,
        writes new pages, decides when to stop thinking.

        Args:
            hidden_states_all_layers: list of (batch, seq_len, 2048) from Llama
            all_pages: list of (batch, 256) pages from previous cycles
            cycle_num: current cycle index
            max_passes: max thinking iterations
            energy_threshold: stop when energy drops below this

        Returns:
            trunk_output: (batch, trunk_dim) final trunk state after thinking
            cycle_pages: list of (batch, 256) pages written during thinking
            energies: list of (batch, 1) energy values per pass
        """
        # Encode Llama's hidden states (once per cycle)
        state = self.state_encoder(hidden_states_all_layers)  # (batch, latent_dim)

        device = state.device
        cycle_pages = []
        energies = []

        for pass_num in range(max_passes):
            # Read all available pages (previous cycles + current thinking)
            available_pages = all_pages + cycle_pages

            # Page attention
            context = self.page_attn(state, available_pages)

            # Positional context
            cycle_ctx = self.cycle_embed(
                torch.tensor(min(cycle_num, 15), device=device)
            ).expand(state.size(0), -1)
            pass_ctx = self.pass_embed(
                torch.tensor(min(pass_num, 7), device=device)
            ).expand(state.size(0), -1)

            # Trunk
            trunk_input = torch.cat([state, context, cycle_ctx, pass_ctx], dim=-1)
            trunk_out = self.trunk(trunk_input)
            trunk_out = self.trunk_body(trunk_out)

            # Write page (record thinking)
            raw_page = self.page_head(trunk_out)
            page = F.normalize(raw_page, dim=-1)  # unit hypersphere
            cycle_pages.append(page)

            # Energy (thinking convergence)
            energy = torch.sigmoid(self.energy_head(trunk_out))
            energies.append(energy)

            # Adaptive stopping
            if energy.mean().item() < energy_threshold:
                break

        return trunk_out, cycle_pages, energies

    def decide(self, trunk_output):
        """Make decisions after thinking.

        Returns:
            action_logits: (batch, 3) — DECOMPOSE/SOLVE/MERGE
            confidence: (batch, 1) — stopping criterion
            branch_embed: (batch, 64) — tree position
        """
        action_logits = self.action_head(trunk_output)
        confidence = torch.sigmoid(self.confidence_head(trunk_output))

        raw_embed = self.branch_embed_head(trunk_output)
        embed_norm = raw_embed.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        clip_scale = (self.max_branch_norm / embed_norm).clamp(max=1.0)
        branch_embed = raw_embed * clip_scale

        return action_logits, confidence, branch_embed

    def make_soft_tokens(self, trunk_output):
        """Project thinking into Llama's embedding space.

        Args:
            trunk_output: (batch, trunk_dim) from thinking

        Returns:
            soft_tokens: (batch, N_soft, hidden_dim) for injection into Llama
        """
        raw = self.soft_token_head(trunk_output)  # (batch, N_soft * hidden_dim)
        soft_tokens = raw.view(
            trunk_output.size(0), self.num_soft_tokens, self.hidden_dim
        )
        return soft_tokens


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    print("=" * 60)
    print("ThinkingController — Architecture Summary")
    print("=" * 60)

    ctrl = ThinkingController()
    total, trainable = count_parameters(ctrl)
    print(f"\nTotal parameters: {total:,}")
    print(f"Trainable: {trainable:,}")
    print(f"Size (bf16): {total * 2 / 1e6:.1f} MB")

    components = {
        'state_encoder': ctrl.state_encoder,
        'page_attn': ctrl.page_attn,
        'trunk + trunk_body': nn.Sequential(ctrl.trunk, ctrl.trunk_body),
        'page_head': ctrl.page_head,
        'soft_token_head': ctrl.soft_token_head,
        'action_head': ctrl.action_head,
        'energy_head': ctrl.energy_head,
        'confidence_head': ctrl.confidence_head,
        'branch_embed_head': ctrl.branch_embed_head,
    }

    print("\nComponent breakdown:")
    for name, module in components.items():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name:25s}: {n:>12,} ({n/total*100:5.1f}%)")

    # Test forward
    print("\nSmoke test...")
    batch = 2
    # Fake Llama hidden states (16 layers)
    hidden = [torch.randn(batch, 20, 2048) for _ in range(16)]

    trunk_out, pages, energies = ctrl.think(hidden, all_pages=[], cycle_num=0)
    print(f"  Thinking: {len(pages)} pages, {len(energies)} energy values")
    print(f"  Page shape: {pages[0].shape}, norm: {pages[0].norm(dim=-1).mean():.3f}")

    action_logits, conf, embed = ctrl.decide(trunk_out)
    print(f"  Action logits: {action_logits.shape}")
    print(f"  Confidence: {conf.mean():.3f}")

    soft = ctrl.make_soft_tokens(trunk_out)
    print(f"  Soft tokens: {soft.shape}")  # (batch, N_soft, 2048)

    # Check page diversity across different inputs
    hidden2 = [torch.randn(batch, 20, 2048) for _ in range(16)]
    trunk2, pages2, _ = ctrl.think(hidden2, all_pages=[], cycle_num=0)
    soft2 = ctrl.make_soft_tokens(trunk2)

    cos = F.cosine_similarity(
        soft.view(batch, -1)[:1], soft2.view(batch, -1)[:1]
    ).item()
    print(f"\n  Soft token diversity (different inputs): cos={cos:.4f}")
    print(f"  {'PASS' if cos < 0.95 else 'FAIL'}: different inputs → different soft tokens")
