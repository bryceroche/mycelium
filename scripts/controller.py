"""
Mycelium v2 BreathingController (~380M).

Clean reimplementation with direct gradient path as the cardinal rule.
The controller OWNS the scales 100%. atoms2 provides A/B matrices (tools),
the controller decides which tools to use (scales).

Components:
  - StateEncoder: Perceiver reading Llama's hidden states → 1536d (~150M)
  - TreeNotebookAttention: Hierarchical attention over tree nodes (~90M)
  - Trunk: Integration MLP (state + context → 3072d) (~100M)
  - Decision heads: scales, branch_embed, branch_action, page, energy, confidence (~25M)
  - Fingerprint: frozen random projection for per-problem diversity
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
    """A node in the breathing tree."""
    page: torch.Tensor              # (256d) on hypersphere
    branch_embed: torch.Tensor      # (64d) Euclidean with L2 clipping
    hidden_pool: torch.Tensor       # (1536d) compressed state from encoder
    action: str                     # 'decompose' / 'solve' / 'merge'
    parent_idx: int = -1            # -1 for root
    children_idx: list = field(default_factory=list)
    claimed_target: Optional[int] = None
    generation: str = ""
    energy: float = 1.0


class TreeNotebook:
    """Hierarchical memory for the breathing tree."""

    def __init__(self):
        self.nodes: List[TreeNode] = []

    def append(self, node: TreeNode) -> int:
        idx = len(self.nodes)
        self.nodes.append(node)
        return idx

    def __len__(self):
        return len(self.nodes)

    def get_ancestors(self, idx: int) -> List[int]:
        ancestors = []
        current = idx
        while current >= 0 and current < len(self.nodes):
            parent = self.nodes[current].parent_idx
            if parent >= 0:
                ancestors.append(parent)
            current = parent
        return ancestors

    def get_siblings(self, idx: int) -> List[int]:
        if idx < 0 or idx >= len(self.nodes):
            return []
        parent = self.nodes[idx].parent_idx
        if parent < 0:
            return []
        return [c for c in self.nodes[parent].children_idx if c != idx]

    def get_children(self, idx: int) -> List[int]:
        if idx < 0 or idx >= len(self.nodes):
            return []
        return self.nodes[idx].children_idx


# ---------------------------------------------------------------------------
# State Encoder (~150M) — Perceiver reading Llama hidden states
# ---------------------------------------------------------------------------

class StateEncoder(nn.Module):
    """Cross-attention Perceiver that compresses Llama's hidden states."""

    def __init__(self, hidden_dim=2048, latent_dim=1536, num_heads=8,
                 num_layers=4, num_llama_layers=16):
        super().__init__()
        self.latent_dim = latent_dim

        # Learned weighted combination of Llama layers
        self.layer_weights = nn.Parameter(torch.ones(num_llama_layers) / num_llama_layers)

        # Project Llama hidden dim to latent dim
        self.input_project = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

        # Perceiver layers
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

        # Learned latent query
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

        return query.squeeze(1)  # (batch, latent_dim)


# ---------------------------------------------------------------------------
# Tree Notebook Attention (~90M)
# ---------------------------------------------------------------------------

class TreeNotebookAttention(nn.Module):
    """Hierarchical attention over tree notebook."""

    def __init__(self, state_dim=1536, page_dim=256, embed_dim=64,
                 num_heads=8, num_layers=4):
        super().__init__()
        self.state_dim = state_dim
        self.page_project = nn.Linear(page_dim + embed_dim, state_dim)

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

    def forward(self, state, notebook: TreeNotebook, current_embed=None):
        if len(notebook) == 0:
            return self.no_history.expand(state.shape[0], -1)

        pages = torch.stack([n.page for n in notebook.nodes])
        embeds = torch.stack([n.branch_embed for n in notebook.nodes])
        page_cat = torch.cat([pages, embeds], dim=-1)
        kv = self.page_project(page_cat).unsqueeze(0).expand(state.shape[0], -1, -1)

        attn_mask = None
        if current_embed is not None:
            dists = torch.cdist(
                current_embed.unsqueeze(1),
                embeds.unsqueeze(0).expand(state.shape[0], -1, -1),
            ).squeeze(1)
            attn_mask = -dists.unsqueeze(1)

        query = state.unsqueeze(1)

        for layer in self.layers:
            residual = query
            query = layer['norm1'](query)
            query, _ = layer['attn'](query, kv, kv, attn_mask=attn_mask)
            query = query + residual

            residual = query
            query = layer['norm2'](query)
            query = layer['ff'](query) + residual

        return query.squeeze(1)


# ---------------------------------------------------------------------------
# BreathingController (~380M) — Tree-structured reasoning engine
# ---------------------------------------------------------------------------

class BreathingController(nn.Module):
    """
    v2 BreathingController — OWNS scales 100%.

    atoms2 provides A/B matrices (tools).
    Controller decides scales (which tools to use per problem).
    Cardinal rule: controller gradient never flows through Llama.
    """

    def __init__(self, hidden_dim=2048, latent_dim=1536,
                 page_dim=256, num_atoms=64, max_cycles=16,
                 num_encoder_layers=4, num_notebook_layers=4,
                 num_trunk_layers=4, num_heads=8,
                 num_llama_layers=16, branch_embed_dim=64,
                 max_branch_norm=0.95):
        super().__init__()
        self.latent_dim = latent_dim
        self.page_dim = page_dim
        self.num_atoms = num_atoms
        self.branch_embed_dim = branch_embed_dim
        self.max_branch_norm = max_branch_norm

        # State encoder (~150M)
        self.state_encoder = StateEncoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            num_llama_layers=num_llama_layers,
        )

        # Notebook attention (~90M)
        self.notebook_attn = TreeNotebookAttention(
            state_dim=latent_dim,
            page_dim=page_dim,
            embed_dim=branch_embed_dim,
            num_heads=num_heads,
            num_layers=num_notebook_layers,
        )

        # Positional embeddings
        self.cycle_embed = nn.Embedding(max_cycles, 32)
        self.pass_embed = nn.Embedding(8, 32)
        self.depth_embed = nn.Embedding(8, 32)
        self.prev_scale_project = nn.Linear(num_atoms, 64)

        # Trunk (~100M): integrate state + context + positional
        trunk_input_dim = latent_dim * 2 + 32 + 32 + 32 + 64
        trunk_dim = 4096  # wider than latent_dim*2 for more processing capacity
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

        # Problem fingerprint: frozen random projection → additive on raw scales
        self.register_buffer(
            'fingerprint_proj',
            torch.randn(hidden_dim, num_atoms) / math.sqrt(hidden_dim),
        )

        # Decomposed scale output: cycle component + problem component
        #
        # Cycle scales: coarse structure, shared across problems within a cycle.
        #   "cycle 1 = comprehension mode, cycle 2 = computation mode"
        #   Absorbs the uniform ST gradient (that's its job).
        #   Input: cycle_embed + notebook context (no per-problem hidden states)
        self.num_strategies = 16
        self.codebook = nn.Parameter(torch.randn(self.num_strategies, num_atoms) * 0.1)
        self.cycle_strategy_head = nn.Sequential(
            nn.Linear(32 + latent_dim, 512),  # cycle_embed + notebook context
            nn.GELU(),
            nn.Linear(512, self.num_strategies),
        )

        # Problem delta: fine structure, per-problem adjustment on top of cycle scales.
        #   "this problem involves fractions, that one involves conditionals"
        #   Gets the residual per-problem signal after cycle component is accounted for.
        #   Input: trunk output (includes hidden states + fingerprint diversity)
        self.problem_delta_head = nn.Sequential(
            nn.Linear(trunk_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_atoms),
        )

        self.page_head = nn.Sequential(
            nn.Linear(trunk_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, page_dim),
        )

        self.branch_embed_head = nn.Sequential(
            nn.Linear(trunk_dim, 512),
            nn.GELU(),
            nn.Linear(512, branch_embed_dim),
        )

        self.branch_action_head = nn.Sequential(
            nn.Linear(trunk_dim, 256),
            nn.GELU(),
            nn.Linear(256, 3),
        )

        self.energy_head = nn.Sequential(
            nn.Linear(trunk_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(trunk_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, hidden_states_all_layers, notebook: TreeNotebook,
                cycle_num: int = 0, pass_num: int = 0, depth: int = 0,
                prev_scales=None, current_embed=None):
        """
        Full controller forward pass. OWNS the scale output.

        Returns:
            scales, page, branch_embed, branch_action_logits,
            energy, confidence, trunk_output
        """
        # 1. Encode Llama's hidden states
        state = self.state_encoder(hidden_states_all_layers)

        # 2. Read notebook history
        context = self.notebook_attn(state, notebook, current_embed)

        # 3. Positional embeddings
        device = state.device
        cycle_ctx = self.cycle_embed(
            torch.tensor(min(cycle_num, 15), device=device)
        ).expand(state.size(0), -1)
        pass_ctx = self.pass_embed(
            torch.tensor(min(pass_num, 7), device=device)
        ).expand(state.size(0), -1)
        depth_ctx = self.depth_embed(
            torch.tensor(min(depth, 7), device=device)
        ).expand(state.size(0), -1)

        if prev_scales is not None:
            prev_ctx = self.prev_scale_project(prev_scales.to(state.dtype))
        else:
            prev_ctx = torch.zeros(state.size(0), 64, device=device, dtype=state.dtype)

        # 4. Trunk
        trunk_input = torch.cat([state, context, cycle_ctx, pass_ctx, depth_ctx, prev_ctx], dim=-1)
        trunk_out = self.trunk(trunk_input)
        trunk_out = self.trunk_body(trunk_out)

        # 5. Problem fingerprint — additive bias for per-problem diversity
        fp_weights = F.softmax(self.state_encoder.layer_weights.detach(), dim=0)
        fp_hidden = sum(w * h.mean(dim=1) for w, h in zip(fp_weights, hidden_states_all_layers))
        fingerprint = fp_hidden.to(trunk_out.dtype) @ self.fingerprint_proj.to(trunk_out.dtype)

        # 6. Decomposed scales: cycle component + problem component
        #
        # Cycle scales: select strategy from codebook using cycle + notebook context
        # (no per-problem hidden states — absorbs the uniform gradient)
        cycle_input = torch.cat([cycle_ctx, context], dim=-1)  # (batch, 32 + latent_dim)
        cycle_logits = self.cycle_strategy_head(cycle_input)     # (batch, K)
        cycle_attn = F.softmax(cycle_logits, dim=-1)             # (batch, K)
        cycle_scales = cycle_attn @ self.codebook                # (batch, 64)

        # Problem delta: per-problem adjustment from trunk (includes hidden states)
        problem_delta = self.problem_delta_head(trunk_out)       # (batch, 64)

        # Compose: cycle base + problem adjustment + fingerprint diversity
        raw_scales = cycle_scales + 0.3 * problem_delta + fingerprint
        scales = 0.46 * torch.tanh(raw_scales)

        page = F.normalize(self.page_head(trunk_out), dim=-1)

        raw_embed = self.branch_embed_head(trunk_out)
        embed_norm = raw_embed.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        clip_scale = (self.max_branch_norm / embed_norm).clamp(max=1.0)
        branch_embed = raw_embed * clip_scale

        branch_action_logits = self.branch_action_head(trunk_out)
        energy = torch.sigmoid(self.energy_head(trunk_out))
        confidence = torch.sigmoid(self.confidence_head(trunk_out))

        return scales, page, branch_embed, branch_action_logits, energy, confidence, trunk_out

    def forward_simple(self, hidden_state, prev_scales=None):
        """Simplified forward for smoke test — no Llama, no notebook."""
        device = hidden_state.device
        batch_size = hidden_state.size(0)

        state = self.state_encoder.input_project(hidden_state)
        context = self.notebook_attn.no_history.expand(batch_size, -1)

        cycle_ctx = self.cycle_embed(torch.tensor(0, device=device)).expand(batch_size, -1)
        pass_ctx = self.pass_embed(torch.tensor(0, device=device)).expand(batch_size, -1)
        depth_ctx = self.depth_embed(torch.tensor(0, device=device)).expand(batch_size, -1)

        if prev_scales is not None:
            prev_ctx = self.prev_scale_project(prev_scales.to(state.dtype))
        else:
            prev_ctx = torch.zeros(batch_size, 64, device=device, dtype=state.dtype)

        trunk_input = torch.cat([state, context, cycle_ctx, pass_ctx, depth_ctx, prev_ctx], dim=-1)
        trunk_out = self.trunk(trunk_input)
        trunk_out = self.trunk_body(trunk_out)

        fingerprint = hidden_state.to(trunk_out.dtype) @ self.fingerprint_proj.to(trunk_out.dtype)

        # Decomposed: cycle + problem
        cycle_input = torch.cat([cycle_ctx, context], dim=-1)
        cycle_logits = self.cycle_strategy_head(cycle_input)
        cycle_attn = F.softmax(cycle_logits, dim=-1)
        cycle_scales = cycle_attn @ self.codebook

        problem_delta = self.problem_delta_head(trunk_out)
        scales = 0.46 * torch.tanh(cycle_scales + 0.3 * problem_delta + fingerprint)
        page = F.normalize(self.page_head(trunk_out), dim=-1)

        raw_embed = self.branch_embed_head(trunk_out)
        embed_norm = raw_embed.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        clip_scale = (self.max_branch_norm / embed_norm).clamp(max=1.0)
        branch_embed = raw_embed * clip_scale

        branch_action_logits = self.branch_action_head(trunk_out)
        energy = torch.sigmoid(self.energy_head(trunk_out))
        confidence = torch.sigmoid(self.confidence_head(trunk_out))

        return scales, page, branch_embed, branch_action_logits, energy, confidence


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    print("=" * 60)
    print("BreathingController v2 — Architecture Summary")
    print("=" * 60)

    ctrl = BreathingController()
    total, trainable = count_parameters(ctrl)
    print(f"\nTotal parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Size (bf16): {total * 2 / 1e6:.1f} MB")

    components = {
        'state_encoder': ctrl.state_encoder,
        'notebook_attn': ctrl.notebook_attn,
        'trunk + trunk_body': nn.Sequential(ctrl.trunk, ctrl.trunk_body),
        'cycle_strategy_head': ctrl.cycle_strategy_head,
        'problem_delta_head': ctrl.problem_delta_head,
        'page_head': ctrl.page_head,
        'branch_embed_head': ctrl.branch_embed_head,
        'branch_action_head': ctrl.branch_action_head,
        'energy_head': ctrl.energy_head,
        'confidence_head': ctrl.confidence_head,
    }

    print("\nComponent breakdown:")
    for name, module in components.items():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name:25s}: {n:>12,} ({n/total*100:5.1f}%)")

    # Forward pass test
    print("\nSmoke test forward pass...")
    x = torch.randn(4, 2048)
    scales, page, embed, action, energy, conf = ctrl.forward_simple(x)
    print(f"  scales: {scales.shape}, range [{scales.min():.3f}, {scales.max():.3f}]")

    # Diversity check
    from torch.nn.functional import cosine_similarity
    cos_vals = []
    for i in range(4):
        for j in range(i+1, 4):
            cos_vals.append(cosine_similarity(scales[i:i+1], scales[j:j+1]).item())
    print(f"  scale diversity (pairwise cos): {sum(cos_vals)/len(cos_vals):.4f}")
    print(f"  page norm: {page.norm(dim=-1).mean():.3f}")
    print(f"  branch_embed norm: {embed.norm(dim=-1).mean():.3f}")
    print(f"\n{'PASS' if sum(cos_vals)/len(cos_vals) < 0.95 else 'FAIL'}: controller produces diverse scales")
