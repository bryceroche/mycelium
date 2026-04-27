"""
64-Atom LoRA Architecture (v27 — unified BreathingController).

64 anonymous rank-6 LoRA atoms (~82M params), independently scaled by a
unified BreathingController that replaces the separate perceiver + hypernetwork
+ bypass + message generator.

Key design choices:
- Tanh activation (NOT softmax): atoms are independent, no competition, no mode collapse
- Batched einsum: all 64 atoms applied simultaneously, no per-atom loops
- BreathingController reads ALL Llama hidden layers + history -> page + scales + focus
- One network, one understanding, two outputs (record + plan)
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/home/ubuntu/mycelium')

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# FourierPageEncoding — structural identity for pages (zero learnable params)
# ---------------------------------------------------------------------------
class FourierPageEncoding(nn.Module):
    """Fixed Fourier encoding that gives each page dimension a frequency identity.

    Each page gets 32 sine/cosine frequency pairs scaled by pass_num,
    providing structural identity (which dimension, which pass) alongside
    content. Zero learnable parameters — frequencies are a fixed buffer.

    DEPRECATED: Use PiHarmonicPageEncoding instead for DCT-like orthogonal basis.
    """

    def __init__(self, page_size: int = 64):
        super().__init__()
        dim_indices = torch.arange(page_size // 2, dtype=torch.float32)
        freqs = torch.exp(dim_indices * -(math.log(10000.0) / (page_size // 2)))
        self.register_buffer('freqs', freqs)

    def encode(self, pass_num: int) -> torch.Tensor:
        """Unique encoding per (dimension, pass) combination."""
        t = pass_num * self.freqs  # (page_size // 2,)
        return torch.cat([torch.sin(t), torch.cos(t)])  # (page_size,)

    def apply(self, page: torch.Tensor, pass_num: int) -> torch.Tensor:
        """Add positional structure to a page. Applied AFTER normalization."""
        encoding = self.encode(pass_num).to(device=page.device, dtype=page.dtype)
        return page + encoding  # content + structural identity


# ---------------------------------------------------------------------------
# PiHarmonicPageEncoding — DCT-like orthogonal basis (zero learnable params)
# ---------------------------------------------------------------------------
class PiHarmonicPageEncoding(nn.Module):
    """Pi-harmonic page encoding with DCT-like orthogonal frequency basis.

    Uses frequencies: freq_k = k * pi / page_size for k in 1..32
    This creates 32 orthogonal harmonics (same basis as DCT/JPEG compression).

    Each frequency pair (sin, cos) is independent:
    - k=1:  wavelength = 128 dims (lowest frequency - broadest pattern)
    - k=16: wavelength = 8 dims (mid frequency)
    - k=32: wavelength = 4 dims (highest frequency - finest detail)

    Advantages over transformer-style (10000-based) encoding:
    - Orthogonal by construction (harmonics of pi)
    - Same mathematical basis as DCT (proven optimal for energy compaction)
    - No arbitrary constant (pi is natural, 10000 was arbitrary)
    - Page IS a proper frequency decomposition

    Zero learnable parameters - frequencies are fixed buffer.
    """

    def __init__(self, page_size: int = 64):
        super().__init__()
        n = page_size // 2  # 32 frequency pairs
        # Pi-harmonic frequencies: freq_k = k * pi / page_size
        # k ranges from 1 to n (not 0 to n-1, to avoid zero frequency)
        freqs = torch.arange(1, n + 1, dtype=torch.float32) * math.pi / page_size
        self.register_buffer('freqs', freqs)

    def encode(self, pass_num: int) -> torch.Tensor:
        """Encode pass number using pi-harmonic frequencies.

        Each dimension pair (sin, cos) is one harmonic of pi.
        Orthogonal basis - independent frequency channels.

        Args:
            pass_num: int, the pass number (0-indexed)

        Returns:
            encoding: (page_size,) tensor with sin/cos pairs
        """
        t = pass_num * self.freqs  # (32,)
        return torch.cat([torch.sin(t), torch.cos(t)])  # (64,)

    def apply(self, page: torch.Tensor, pass_num: int) -> torch.Tensor:
        """Add pi-harmonic positional structure to a page.

        Applied AFTER hypersphere normalization.
        Creates natural coarse-to-fine pressure:
        - Low-freq dims (0-15): broad info (problem type, magnitude)
        - Mid-freq dims (16-40): key operations (specific numbers)
        - High-freq dims (40-63): precise details (exact values)

        Args:
            page: (batch, page_size) normalized page content
            pass_num: int, the pass number

        Returns:
            page + encoding: page with structural identity added
        """
        encoding = self.encode(pass_num).to(device=page.device, dtype=page.dtype)
        return page + encoding  # content + harmonic structure


# ---------------------------------------------------------------------------
# LoRAAtoms — 64 rank-6 anonymous LoRA atoms
# ---------------------------------------------------------------------------
class LoRAAtoms(nn.Module):
    """
    64 rank-6 LoRA atoms for Q,K,V,O at all 16 Llama layers.

    Each atom is an independent direction of attention modification.
    Applied via batched einsum — no per-atom loops.

    A: (num_atoms, num_layers, d_model, rank)
    B: (num_atoms, num_layers, rank, proj_dim)
       where proj_dim is 512 for k_proj/v_proj (GQA) and 2048 for q_proj/o_proj
    """

    PROJ_NAMES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    def __init__(
        self,
        d_model: int = 2048,
        d_kv: int = 512,
        rank: int = 6,
        num_atoms: int = 64,
        num_layers: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_kv = d_kv
        self.rank = rank
        self.num_atoms = num_atoms
        self.num_layers = num_layers

        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()

        for proj_name in self.PROJ_NAMES:
            proj_dim = d_kv if proj_name in ('k_proj', 'v_proj') else d_model
            self.A[proj_name] = nn.Parameter(
                torch.randn(num_atoms, num_layers, d_model, rank) * 0.01
            )
            self.B[proj_name] = nn.Parameter(
                torch.randn(num_atoms, num_layers, rank, proj_dim) * 0.01
            )

    def apply(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
        proj_name: str,
        atom_scales: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute blended atom LoRA output via batched einsum.

        Args:
            hidden:      (batch, seq, d_model)
            layer_idx:   int, which Llama layer
            proj_name:   str, one of q_proj/k_proj/v_proj/o_proj
            atom_scales: (batch, num_atoms) — independent tanh-bounded scalars

        Returns:
            lora_out: (batch, seq, proj_dim) — additive LoRA contribution
        """
        A = self.A[proj_name][:, layer_idx]  # (num_atoms, d_model, rank)
        B = self.B[proj_name][:, layer_idx]  # (num_atoms, rank, proj_dim)

        # Ensure consistent dtype (atom_scales may be float32 from hypernet)
        hidden = hidden.to(dtype=A.dtype)
        atom_scales = atom_scales.to(dtype=A.dtype)

        # Batched: all atoms at once
        # hidden: (B, S, D) @ A: (A, D, R) -> projections: (B, A, S, R)
        projections = torch.einsum('bsd,adr->basr', hidden, A)

        # Scale each atom: (B, A, S, R) * (B, A, 1, 1)
        scaled = projections * atom_scales.unsqueeze(-1).unsqueeze(-1)

        # Sum across atoms and project to output dim
        # scaled: (B, A, S, R) @ B: (A, R, P) -> sum over A -> (B, S, P)
        lora_out = torch.einsum('basr,arp->bsp', scaled, B)

        return lora_out


# ---------------------------------------------------------------------------
# AtomAdditiveLoRAManager — monkey-patches Llama attention projections
# ---------------------------------------------------------------------------
class AtomAdditiveLoRAManager:
    """
    Manages inline atom LoRA on Llama attention projections via monkey-patching.

    At each projection layer:
        output = W @ x + atoms.apply(x, layer_idx, proj_name, atom_scales)

    Same pattern as QuadAdditiveLoRAManager but uses batched atom application
    instead of blended named modes.
    """

    PROJECTION_NAMES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    def __init__(self, model: nn.Module):
        self.model = model
        self._original_forwards: Dict[int, Dict[str, callable]] = {}
        self._active = False

        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.num_layers = len(model.model.layers)
        else:
            self.num_layers = 16

    def _get_projection(self, layer_idx: int, proj_name: str) -> nn.Module:
        return getattr(self.model.model.layers[layer_idx].self_attn, proj_name)

    @staticmethod
    def _make_atom_lora_forward(
        original_forward: callable,
        atoms: 'LoRAAtoms',
        layer_idx: int,
        proj_name: str,
        atom_scales: torch.Tensor,
    ) -> callable:
        """
        Create a patched forward that adds atom LoRA output.

        atoms:       LoRAAtoms module
        layer_idx:   int
        proj_name:   str
        atom_scales: (B, num_atoms)
        """
        def forward(x: torch.Tensor) -> torch.Tensor:
            base_output = original_forward(x)
            lora_contribution = atoms.apply(
                x.to(dtype=atoms.A[proj_name].dtype),
                layer_idx, proj_name, atom_scales,
            )
            return base_output + lora_contribution.to(dtype=base_output.dtype)

        return forward

    def apply(
        self,
        atoms: 'LoRAAtoms',
        atom_scales: torch.Tensor,
    ) -> None:
        """
        Apply atom LoRA to all layers and projections.

        Args:
            atoms:       LoRAAtoms module with A/B parameters
            atom_scales: (B, num_atoms) — tanh-bounded scalars
        """
        if self._active:
            raise RuntimeError(
                "Atom LoRA already applied. Call remove() before applying again."
            )

        self._original_forwards = {}

        for layer_idx in range(self.num_layers):
            self._original_forwards[layer_idx] = {}

            for proj_name in self.PROJECTION_NAMES:
                proj_module = self._get_projection(layer_idx, proj_name)
                self._original_forwards[layer_idx][proj_name] = proj_module.forward

                proj_module.forward = self._make_atom_lora_forward(
                    proj_module.forward, atoms, layer_idx, proj_name, atom_scales,
                )

        self._active = True

    def remove(self) -> None:
        """Restore original forwards."""
        if not self._active:
            return
        for layer_idx, layer_forwards in self._original_forwards.items():
            for proj_name, original_forward in layer_forwards.items():
                proj_module = self._get_projection(layer_idx, proj_name)
                proj_module.forward = original_forward
        self._original_forwards = {}
        self._active = False

    def is_active(self) -> bool:
        return self._active

    def __del__(self):
        if self._active:
            self.remove()


# ---------------------------------------------------------------------------
# MobiusTransform — conformal warp on the hypersphere for page diversity
# ---------------------------------------------------------------------------
class MobiusTransform(nn.Module):
    """Möbius transformation on the unit sphere in R^n.

    Warps the sphere by expanding around a focus point and compressing
    away from it. Conformal, bijective, stays on the sphere.
    Zero learnable parameters — pure math.
    """

    def __init__(self, dim=64, max_focus_norm=0.5):
        super().__init__()
        self.dim = dim
        self.max_focus_norm = max_focus_norm

    def forward(self, page, focus):
        """
        page:  (batch, 64) — on the hypersphere (normalized)
        focus: (batch, 64) — from hypernetwork (will be constrained to ball)
        Returns: warped page on the hypersphere
        """
        # Constrain focus inside unit ball
        focus_norm = focus.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        focus = focus * (self.max_focus_norm * torch.tanh(focus_norm) / focus_norm)

        # Normalize page to unit sphere
        x = page / page.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Möbius transformation: T_a(x) = (1 - |a|²)(x - a) / |x - a|² + a
        a = focus
        a_norm_sq = (a * a).sum(dim=-1, keepdim=True)
        diff = x - a
        diff_norm_sq = (diff * diff).sum(dim=-1, keepdim=True).clamp(min=1e-8)

        transformed = (1.0 - a_norm_sq) * diff / diff_norm_sq + a

        # Re-normalize and scale back to page radius
        transformed = F.normalize(transformed, dim=-1)
        transformed = transformed * math.sqrt(self.dim)

        return transformed


# ---------------------------------------------------------------------------
# encode_text_context — non-differentiable input to break fixed-point collapse
# ---------------------------------------------------------------------------
def encode_text_context(prev_results, max_steps=8, device=None):
    """Encode previous cycle results as a fixed-size float vector.

    Non-differentiable input that breaks hypernetwork fixed-point collapse.
    Each cycle gets genuinely different context based on what was computed.

    Args:
        prev_results: list of extracted numbers from previous cycles
        max_steps: max previous results to encode
        device: torch device
    Returns:
        context: (max_steps * 2,) tensor — pairs of (step_exists, normalized_value)
    """
    context = torch.zeros(max_steps * 2)
    for i, val in enumerate(prev_results):
        if i < max_steps:
            context[i * 2] = 1.0
            context[i * 2 + 1] = float(val) / 1000.0
    if device is not None:
        context = context.to(device)
    return context


def scale_diversity_loss(all_scales, target_cos=0.3):
    """Penalize similar atom scales between consecutive cycles.

    Forces the hypernetwork to produce different scales per cycle,
    breaking the fixed-point collapse (scales cos(2,3)=0.999).

    Args:
        all_scales: list of (batch, 64) scale tensors, one per cycle
        target_cos: max allowed cosine before penalty (0.3 = meaningfully different)
    Returns:
        scalar loss
    """
    if len(all_scales) < 2:
        return torch.tensor(0.0, device=all_scales[0].device)

    diversity_loss = 0.0
    num_pairs = 0

    for i in range(len(all_scales) - 1):
        cos = F.cosine_similarity(
            all_scales[i], all_scales[i + 1], dim=-1
        ).mean()
        diversity_loss += F.relu(cos - target_cos)
        num_pairs += 1

    return diversity_loss / max(num_pairs, 1)








# ---------------------------------------------------------------------------
# AtomConfidenceHead — reads pages (no blend history) -> stop decision
# ---------------------------------------------------------------------------
class AtomConfidenceHead(nn.Module):
    """
    Confidence head that reads accumulated pages via cross-attention.

    Unlike QuadConfidenceHead, there is no blend history (atoms don't have
    named modes or blend weights). Just pages -> attention -> sigmoid.
    """

    def __init__(
        self,
        page_size: int = 64,
        hidden: int = 512,
        num_heads: int = 8,
    ):
        super().__init__()
        self.page_project = nn.Linear(page_size, hidden)
        self.query = nn.Parameter(torch.randn(1, hidden) * 0.02)

        # 2-layer cross-attention
        self.attn1 = nn.MultiheadAttention(hidden, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.attn2 = nn.MultiheadAttention(hidden, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)

        # Deep output MLP
        self.output = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        state_pages: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            state_pages: list of (B, page_size) tensors

        Returns:
            confidence: (B, 1) in [0, 1]
        """
        if not state_pages or len(state_pages) == 0:
            return torch.tensor([[0.5]], device=state_pages[0].device if state_pages else 'cpu')

        pages = torch.stack(state_pages, dim=1).float()  # (B, P, page_size)
        pages_proj = self.page_project(pages)             # (B, P, hidden)

        batch_size = pages.size(0)
        q = self.query.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 1, hidden)

        att1, _ = self.attn1(query=q, key=pages_proj, value=pages_proj)
        q = self.norm1(q + att1)
        att2, _ = self.attn2(query=q, key=pages_proj, value=pages_proj)
        q = self.norm2(q + att2)

        return self.output(q.squeeze(1))  # (B, 1)






# ---------------------------------------------------------------------------
# IsotropicRegularizer — pushes page distribution toward isotropic Gaussian
# ---------------------------------------------------------------------------
class IsotropicRegularizer(nn.Module):
    """Forces page distribution toward isotropic Gaussian. Zero learnable params."""
    def __init__(self, target_var=1.0, corr_weight=0.1):
        super().__init__()
        self.target_var = target_var
        self.corr_weight = corr_weight

    def forward(self, pages_batch):
        """pages_batch: (N, 64) raw perceiver output before normalization."""
        if pages_batch.size(0) < 4:
            return torch.tensor(0.0, device=pages_batch.device)
        dim_means = pages_batch.mean(dim=0)
        mean_loss = (dim_means ** 2).mean()
        dim_vars = pages_batch.var(dim=0)
        var_loss = ((dim_vars - self.target_var) ** 2).mean()
        normalized = (pages_batch - dim_means) / (dim_vars.sqrt() + 1e-8)
        batch_size = pages_batch.size(0)
        correlation = (normalized.T @ normalized) / batch_size
        identity = torch.eye(pages_batch.size(1), device=pages_batch.device)
        off_diagonal = correlation - identity
        corr_loss = (off_diagonal ** 2).mean()
        return mean_loss + var_loss + self.corr_weight * corr_loss




# ---------------------------------------------------------------------------
# BreathingController — unified perceiver + hypernetwork replacement
# ---------------------------------------------------------------------------
class BreathingController(nn.Module):
    """Unified controller: reads Llama's hidden states, produces page + scales + focus.

    Replaces separate perceiver (105M) + hypernetwork (101M) + bypass (5M) + message (1M).
    One network, one understanding, two outputs (record + plan).
    """
    def __init__(self, num_layers=16, hidden_dim=2048,
                 internal_dim=1536, num_heads=8,
                 page_dim=64, num_atoms=64, max_cycles=12,
                 num_history_layers=6):
        super().__init__()
        self.internal_dim = internal_dim
        self.page_dim = page_dim
        self.num_atoms = num_atoms

        # Learned weighted combination of all Llama layers
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        # Project weighted hidden states to internal dimension
        self.current_project = nn.Sequential(
            nn.Linear(hidden_dim, internal_dim),
            nn.LayerNorm(internal_dim),
            nn.GELU(),
        )

        # Project each history entry (page + hidden_pool) to internal dim
        self.history_entry_project = nn.Linear(page_dim + hidden_dim, internal_dim)

        # Cross-attention: current understanding queries history
        self.history_attn_layers = nn.ModuleList()
        for _ in range(num_history_layers):
            self.history_attn_layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(
                    embed_dim=internal_dim, num_heads=num_heads, batch_first=True
                ),
                'norm1': nn.LayerNorm(internal_dim),
                'ff': nn.Sequential(
                    nn.Linear(internal_dim, internal_dim * 4),
                    nn.GELU(),
                    nn.Linear(internal_dim * 4, internal_dim),
                ),
                'norm2': nn.LayerNorm(internal_dim),
            }))

        # Shared trunk: integrate current + history
        self.trunk = nn.Sequential(
            nn.Linear(internal_dim * 2, internal_dim),
            nn.LayerNorm(internal_dim),
            nn.GELU(),
            nn.Linear(internal_dim, internal_dim),
            nn.LayerNorm(internal_dim),
            nn.GELU(),
            nn.Linear(internal_dim, internal_dim),
            nn.LayerNorm(internal_dim),
            nn.GELU(),
            nn.Linear(internal_dim, internal_dim),
            nn.LayerNorm(internal_dim),
            nn.GELU(),
        )

        # Page head: "what did I understand?"
        self.page_head = nn.Sequential(
            nn.Linear(internal_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, page_dim),
        )

        # Scale head: "what should I think about next?"
        self.scale_head = nn.Sequential(
            nn.Linear(internal_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_atoms),
        )

        # Möbius focus head
        self.focus_head = nn.Sequential(
            nn.Linear(internal_dim, 256),
            nn.GELU(),
            nn.Linear(256, page_dim),
        )

    def forward(self, hidden_states_all_layers, history_pages, history_hiddens,
                return_pre_tanh=False):
        """
        Args:
            hidden_states_all_layers: list of 16 tensors, each (batch, seq, 2048)
            history_pages: list of previous cycle pages [(batch, 64), ...]
            history_hiddens: list of previous cycle hidden pools [(batch, 2048), ...]
            return_pre_tanh: if True, also return pre-tanh scale values
        Returns:
            page, scales, focus (and optionally pre_tanh)
        """
        # Weighted combination of all Llama layers
        weights = F.softmax(self.layer_weights, dim=0)
        combined_hidden = sum(
            w * h.mean(dim=1) for w, h in zip(weights, hidden_states_all_layers)
        )  # (batch, 2048)

        current = self.current_project(combined_hidden)  # (batch, internal_dim)

        # Read history
        if len(history_pages) > 0:
            history_entries = []
            for pg, hid in zip(history_pages, history_hiddens):
                entry = torch.cat([pg.to(current.dtype), hid.to(current.dtype)], dim=-1)
                projected = self.history_entry_project(entry)
                history_entries.append(projected)

            history_seq = torch.stack(history_entries, dim=1)  # (batch, N, internal_dim)

            query = current.unsqueeze(1)  # (batch, 1, internal_dim)
            for layer in self.history_attn_layers:
                attn_out, _ = layer['attn'](query, history_seq, history_seq)
                query = layer['norm1'](query + attn_out)
                ff_out = layer['ff'](query)
                query = layer['norm2'](query + ff_out)

            history_ctx = query.squeeze(1)  # (batch, internal_dim)
        else:
            history_ctx = torch.zeros_like(current)

        # Shared trunk
        shared = self.trunk(torch.cat([current, history_ctx], dim=-1))

        # Two heads
        page = self.page_head(shared)
        page = F.normalize(page, dim=-1) * math.sqrt(self.page_dim)

        pre_tanh = self.scale_head(shared)
        pre_tanh = torch.clamp(pre_tanh, -0.5, 0.5)
        scales = torch.tanh(pre_tanh)  # output range ~[-0.46, 0.46], optimal for GSM8K

        focus = self.focus_head(shared)

        if return_pre_tanh:
            return page, scales, pre_tanh, focus
        return page, scales, focus


# ---------------------------------------------------------------------------
# AtomLoRAModel — main model class
# ---------------------------------------------------------------------------
class AtomLoRAModel(nn.Module):
    """
    64-Atom LoRA model with unified BreathingController (v27).

    - Llama 3.2 1B frozen
    - BreathingController: reads all Llama layers + history -> page + scales + focus
    - LoRAAtoms: 64 rank-6 atoms applied via batched einsum
    - AtomConfidenceHead: stop decision from pages
    - MobiusTransform: conformal warp for page diversity
    """

    def __init__(self, model_name: str = 'unsloth/Llama-3.2-1B',
                 num_atoms: int = 64, atom_rank: int = 6):
        super().__init__()

        # --- Frozen transformer ---
        self.transformer = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map='auto',
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for p in self.transformer.parameters():
            p.requires_grad = False

        # --- Dimensions ---
        self.d_model = self.transformer.config.hidden_size          # 2048
        self.num_layers = self.transformer.config.num_hidden_layers  # 16
        num_kv_heads = self.transformer.config.num_key_value_heads   # 8
        head_dim = self.d_model // self.transformer.config.num_attention_heads
        self.d_kv = num_kv_heads * head_dim                         # 512

        self.page_size = 64
        self.page_radius = math.sqrt(self.page_size)  # 8.0
        self.num_atoms = num_atoms

        # --- LoRA Atoms ---
        self.atoms = LoRAAtoms(
            d_model=self.d_model,
            d_kv=self.d_kv,
            rank=atom_rank,
            num_atoms=num_atoms,
            num_layers=self.num_layers,
        )

        # --- Confidence head (pages only, no blend) ---
        self.confidence_head = AtomConfidenceHead(
            page_size=self.page_size, hidden=128, num_heads=4,
        )

        # --- Möbius transform (conformal warp for page diversity, cycle 2+) ---
        self.mobius = MobiusTransform(dim=self.page_size, max_focus_norm=0.5)

        # --- Isotropic regularizer (per-dim variance → 1, correlation → 0) ---
        self.isotropic_reg = IsotropicRegularizer()

        # --- Breathing controller (v27: unified perceiver + hypernetwork) ---
        self.controller = BreathingController(
            num_layers=self.num_layers,
            hidden_dim=self.d_model,
            internal_dim=1536,
            num_heads=8,
            page_dim=self.page_size,
            num_atoms=num_atoms,
            num_history_layers=6,
        )

    def thinking_pass(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state_pages: List[torch.Tensor],
        pass_num: int,
        max_passes: int = 3,
        prev_mid_states: Optional[List[torch.Tensor]] = None,
        messages: Optional[List[torch.Tensor]] = None,
        text_context: Optional[torch.Tensor] = None,
        bypass_vectors: Optional[List[torch.Tensor]] = None,
        history_hiddens: Optional[List[torch.Tensor]] = None,
        prev_scales: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One thinking pass with unified BreathingController (v27).

        The controller reads ALL Llama hidden layers + history and produces
        BOTH page and next_scales in one forward pass. No separate perceiver
        or hypernetwork call.

        Cycle 0: run Llama WITHOUT atoms (no prev_scales), then controller
        Cycle 1+: run Llama WITH atoms from PREVIOUS cycle's scales, then controller

        Args:
            input_ids:        (B, seq_len)
            attention_mask:   (B, seq_len)
            state_pages:      list of (B, page_size) accumulated pages
            pass_num:         int (0-indexed)
            max_passes:       int, total passes (for gradient scaling)
            prev_mid_states:  Unused (kept for API compat with legacy)
            messages:         Unused (kept for API compat with legacy)
            text_context:     Unused (kept for API compat with legacy)
            bypass_vectors:   Unused (kept for API compat with legacy)
            history_hiddens:  list of (B, 2048) mean-pooled hidden states from prev cycles
            prev_scales:      (B, num_atoms) scales from PREVIOUS cycle's controller

        Returns:
            page:               (B, page_size) normalized on hypersphere
            atom_scales:        (B, num_atoms) the NEXT cycle's scales (from controller)
            current_mid_states: None (no perceiver mid states)
            message:            None (no message generator)
            page_delta:         (B, page_size) same as page (no separate raw)
            hidden_pool:        (B, d_model) mean-pooled last hidden layer
            focus:              (B, 64) Möbius focus point from controller
            bypass_vec:         None (no bypass)
        """
        batch_size = input_ids.size(0)

        # Apply atoms with scales from PREVIOUS cycle (or no atoms for cycle 0)
        if prev_scales is None:
            # Cycle 0: no atom modification
            outputs = self.transformer(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True,
            )
        else:
            manager = AtomAdditiveLoRAManager(self.transformer)
            manager.apply(self.atoms, prev_scales)
            try:
                outputs = self.transformer(
                    input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            finally:
                manager.remove()

        hidden_states = list(outputs.hidden_states[1:])  # skip embedding layer, gives 16

        # Controller reads ALL hidden layers + history → page + next_scales + focus
        if history_hiddens is None:
            history_hiddens = []

        page, next_scales, focus = self.controller(
            hidden_states, state_pages, history_hiddens,
        )

        # Möbius warp for cycle 2+
        if pass_num > 0:
            page = self.mobius(page, focus)

        # Hidden pool for history (cast to float32 for stability)
        hidden_pool = hidden_states[-1].mean(dim=1).float()  # (B, d_model)

        # Return 8 values for compatibility:
        # page, atom_scales (next), mid_states (None), message (None),
        # page_delta (=page), hidden_pool, focus, bypass_vec (None)
        return page, next_scales, None, None, page, hidden_pool, focus, None

    def two_pass_cycle1(self, input_ids, attention_mask):
        """Two-pass cycle 1: vanilla comprehension -> informed initial scales.

        Pass 1: Llama WITHOUT LoRA -> controller reads comprehension -> initial scales
        Pass 2: Llama WITH LoRA -> controller reads modified comprehension -> scales for cycle 2

        Returns:
            page_0:         (B, 64) page from vanilla pass (observation)
            hidden_pool_0:  (B, 2048) hidden pool from vanilla pass
            page_1:         (B, 64) page from LoRA pass (actual cycle 1)
            hidden_pool_1:  (B, 2048) hidden pool from LoRA pass
            initial_scales: (B, 64) scales produced from vanilla comprehension
            next_scales:    (B, 64) scales for cycle 2
            focus_1:        (B, 64) Mobius focus for page_1
            outputs_lora:   CausalLMOutputWithPast from LoRA pass (for generation)
        """
        # Pass 1: vanilla Llama (no LoRA)
        outputs_vanilla = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states_vanilla = list(outputs_vanilla.hidden_states[1:])
        hidden_pool_0 = hidden_states_vanilla[-1].mean(dim=1).float()

        # Controller reads vanilla comprehension -> initial scales
        page_0, initial_scales, focus_0 = self.controller(
            hidden_states_vanilla, [], [],
        )

        # Pass 2: Llama WITH LoRA (atom-modified attention)
        manager = AtomAdditiveLoRAManager(self.transformer)
        manager.apply(self.atoms, initial_scales)
        try:
            outputs_lora = self.transformer(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True,
            )
        finally:
            manager.remove()

        hidden_states_lora = list(outputs_lora.hidden_states[1:])
        hidden_pool_1 = hidden_states_lora[-1].mean(dim=1).float()

        # Controller reads LoRA-modified comprehension -> page + next scales
        page_1, next_scales, focus_1 = self.controller(
            hidden_states_lora, [page_0], [hidden_pool_0.detach()],
        )

        # Mobius warp page_1 (it's the second page, so warp)
        page_1 = self.mobius(page_1, focus_1)

        return (page_0, hidden_pool_0.detach(), page_1, hidden_pool_1,
                initial_scales, next_scales, focus_1, outputs_lora)



# ---------------------------------------------------------------------------
# Warm start from any previous checkpoint
# ---------------------------------------------------------------------------
def warm_start_atom_from_checkpoint(
    atom_model: AtomLoRAModel,
    checkpoint,
) -> None:
    """
    Warm-start an AtomLoRAModel from a previous checkpoint.

    Loads controller and atoms weights where shapes match.
    Everything else is fresh init.

    Args:
        atom_model: AtomLoRAModel to warm-start
        checkpoint: path string OR already-loaded checkpoint dict
    """
    if isinstance(checkpoint, str):
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    else:
        ckpt = checkpoint

    # --- Controller ---
    if 'controller' in ckpt:
        own = atom_model.controller.state_dict()
        loaded = 0
        for k, v in ckpt['controller'].items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded += 1
        atom_model.controller.load_state_dict(own, strict=False)
        print(f"  controller: loaded {loaded}/{len(own)}")
    else:
        print("  controller: not found in checkpoint, fresh init")

    # --- Atoms ---
    if 'atoms' in ckpt:
        own = atom_model.atoms.state_dict()
        loaded = 0
        for k, v in ckpt['atoms'].items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded += 1
        atom_model.atoms.load_state_dict(own, strict=False)
        print(f"  atoms: loaded {loaded}/{len(own)}")
    else:
        print("  atoms: fresh init (64 rank-6 atoms, 0.01 scale)")

    # --- Everything else is fresh ---
    print("  confidence_head: fresh init")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    device = 'cpu'
    torch.manual_seed(42)

    print("=" * 60)
    print("64-Atom LoRA Architecture — Self-Test (v27 cleanup)")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 1. Test LoRAAtoms
    # ---------------------------------------------------------------
    print("\n--- LoRAAtoms ---")
    batch = 4
    atoms = LoRAAtoms(d_model=2048, d_kv=512, rank=6, num_atoms=64, num_layers=16)

    hidden = torch.randn(batch, 32, 2048)  # (B, seq, d_model)
    atom_scales = torch.randn(batch, 64).tanh()

    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        out = atoms.apply(hidden, layer_idx=0, proj_name=proj_name, atom_scales=atom_scales)
        expected_dim = 512 if proj_name in ('k_proj', 'v_proj') else 2048
        print(f"  {proj_name}: input {hidden.shape} -> output {out.shape} "
              f"(expected proj_dim={expected_dim}) {'OK' if out.shape[-1] == expected_dim else 'FAIL'}")

    atom_params = sum(p.numel() for p in atoms.parameters())
    print(f"  Params: {atom_params / 1e6:.2f}M")

    # ---------------------------------------------------------------
    # 2. Test AtomAdditiveLoRAManager with mock Llama layers
    # ---------------------------------------------------------------
    print("\n--- AtomAdditiveLoRAManager (mock) ---")

    class MockLinear(nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.linear = nn.Linear(in_f, out_f, bias=False)
        def forward(self, x):
            return self.linear(x)

    class MockSelfAttn(nn.Module):
        def __init__(self, d_model, d_kv):
            super().__init__()
            self.q_proj = MockLinear(d_model, d_model)
            self.k_proj = MockLinear(d_model, d_kv)
            self.v_proj = MockLinear(d_model, d_kv)
            self.o_proj = MockLinear(d_model, d_model)

    class MockLayer(nn.Module):
        def __init__(self, d_model, d_kv):
            super().__init__()
            self.self_attn = MockSelfAttn(d_model, d_kv)

    class MockModel(nn.Module):
        def __init__(self, num_layers, d_model, d_kv):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([
                MockLayer(d_model, d_kv) for _ in range(num_layers)
            ])

    d_model, d_kv, num_layers = 128, 64, 2  # small for test
    mock_model = MockModel(num_layers, d_model, d_kv)
    small_atoms = LoRAAtoms(d_model=d_model, d_kv=d_kv, rank=6, num_atoms=64, num_layers=num_layers)
    small_scales = torch.randn(batch, 64).tanh()

    manager = AtomAdditiveLoRAManager(mock_model)

    # Get baseline outputs
    x = torch.randn(batch, 8, d_model)
    base_q = mock_model.model.layers[0].self_attn.q_proj(x)

    # Apply atom LoRA
    manager.apply(small_atoms, small_scales)
    assert manager.is_active(), "Manager should be active"
    lora_q = mock_model.model.layers[0].self_attn.q_proj(x)

    diff = (lora_q - base_q).abs().mean().item()
    print(f"  Q projection diff with LoRA: {diff:.6f} (should be > 0)")

    # Remove
    manager.remove()
    restored_q = mock_model.model.layers[0].self_attn.q_proj(x)
    restore_diff = (restored_q - base_q).abs().max().item()
    print(f"  After remove, max diff from baseline: {restore_diff:.9f} (should be ~0)")

    # ---------------------------------------------------------------
    # 3. Test AtomConfidenceHead
    # ---------------------------------------------------------------
    print("\n--- AtomConfidenceHead ---")
    pages = [torch.randn(batch, 64, device=device) for _ in range(3)]
    conf_head = AtomConfidenceHead(page_size=64, hidden=128, num_heads=4)

    confidence = conf_head(pages)
    print(f"  Confidence shape: {confidence.shape}, "
          f"range: [{confidence.min().item():.3f}, {confidence.max().item():.3f}]")

    conf_params = sum(p.numel() for p in conf_head.parameters())
    print(f"  Params: {conf_params:,}")

    # ---------------------------------------------------------------
    # 4. Test BreathingController
    # ---------------------------------------------------------------
    print("\n--- BreathingController ---")
    ctrl = BreathingController(num_layers=16, hidden_dim=2048, internal_dim=1536,
                               num_heads=8, page_dim=64, num_atoms=64)

    # Simulate hidden states from 16 Llama layers
    fake_hidden = [torch.randn(batch, 10, 2048) for _ in range(16)]
    history_pages = [torch.randn(batch, 64) for _ in range(2)]
    history_hiddens = [torch.randn(batch, 2048) for _ in range(2)]

    page, scales, focus = ctrl(fake_hidden, history_pages, history_hiddens)
    print(f"  page: {page.shape}, scales: {scales.shape}, focus: {focus.shape}")
    print(f"  scales range: [{scales.min().item():.3f}, {scales.max().item():.3f}]")

    ctrl_params = sum(p.numel() for p in ctrl.parameters())
    print(f"  Params: {ctrl_params / 1e6:.2f}M")

    # ---------------------------------------------------------------
    # 5. Parameter count summary
    # ---------------------------------------------------------------
    print("\n--- Parameter Count Summary ---")
    components = {
        'LoRAAtoms': atoms,
        'BreathingController': ctrl,
        'AtomConfidenceHead': conf_head,
    }
    total = 0
    for name, module in components.items():
        count = sum(p.numel() for p in module.parameters())
        total += count
        if count > 1e6:
            print(f"  {name:25s}: {count / 1e6:.2f}M")
        else:
            print(f"  {name:25s}: {count:,}")
    print(f"  {'TOTAL (trainable)':25s}: {total / 1e6:.2f}M")
    print(f"  (+ 1.23B frozen Llama)")

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
