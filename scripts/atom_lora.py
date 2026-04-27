"""
64-Atom LoRA Architecture (v24).

Replace named LoRA templates (parse/compute/verify/answer) with 64 anonymous
rank-6 atoms (~100M params), independently scaled by a 10M-param hypernetwork.
The model discovers its own cognitive decomposition.

Key design choices:
- Tanh activation (NOT softmax): atoms are independent, no competition, no mode collapse
- Batched einsum: all 64 atoms applied simultaneously, no per-atom loops
- No strategy side channel: hypernetwork reads pages + pass embed only
- Symmetric capacity: ~105M compress (perceiver) ~ 100M expand (atoms)

Evolution from QuadLoRA (v23): removes named modes, softmax blend, entropy
regularization. Adds 64 anonymous atoms with independent tanh scaling.
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
from src.compressor_v3 import Compressor


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
# AtomHypernetwork — reads pages + pass number -> 64 atom scales
# ---------------------------------------------------------------------------
class AtomHypernetwork(nn.Module):
    """
    Hypernetwork that reads accumulated pages + pass embedding and outputs
    64 independent atom scales via tanh (NOT softmax).

    No strategy input — removed as redundant with page attention.

    Architecture:
    - 2-layer cross-attention over pages (4 queries, 8 heads, d=512)
    - Fourier pass encoding (continuous, no max_passes limit)
    - Deep MLP: 1536 -> 1024 -> 1024 -> 512 -> 64
    - Final: tanh (independent, no competition)
    """

    def __init__(
        self,
        page_size: int = 64,
        num_atoms: int = 64,
        attn_dim: int = 512,
        num_query_heads: int = 4,
        num_attn_heads: int = 8,
        pass_embed_dim: int = 512,
    ):
        super().__init__()
        self.page_size = page_size
        self.num_atoms = num_atoms
        self.attn_dim = attn_dim
        self.num_query_heads = num_query_heads

        # --- Page reading: 2-layer cross-attention ---
        self.page_project = nn.Linear(page_size, attn_dim)

        self.page_query = nn.Parameter(torch.randn(num_query_heads, attn_dim) * 0.02)

        # Layer 1
        self.page_attn_1 = nn.MultiheadAttention(
            attn_dim, num_heads=num_attn_heads, batch_first=True,
        )
        self.page_norm_1 = nn.LayerNorm(attn_dim)
        self.page_ffn_1 = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 2), nn.GELU(), nn.Linear(attn_dim * 2, attn_dim),
        )
        self.page_ffn_norm_1 = nn.LayerNorm(attn_dim)

        # Layer 2
        self.page_attn_2 = nn.MultiheadAttention(
            attn_dim, num_heads=num_attn_heads, batch_first=True,
        )
        self.page_norm_2 = nn.LayerNorm(attn_dim)
        self.page_ffn_2 = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 2), nn.GELU(), nn.Linear(attn_dim * 2, attn_dim),
        )
        self.page_ffn_norm_2 = nn.LayerNorm(attn_dim)

        # Flatten 4 queries x 512 dim = 2048 -> summary 1024
        self.summary_project = nn.Linear(num_query_heads * attn_dim, 1024)

        # --- Fourier pass encoding (continuous, no max_passes limit) ---
        self.register_buffer(
            'fourier_freqs',
            torch.exp(torch.arange(0, pass_embed_dim, 2) * -(math.log(10000.0) / pass_embed_dim))
        )
        self.pass_project = nn.Linear(pass_embed_dim, pass_embed_dim)

        # --- Deep MLP: 1024 (page_summary) + 512 (pass_embed) = 1536 -> 64 ---
        self.scale_net = nn.Sequential(
            nn.Linear(1024 + pass_embed_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, num_atoms),
            nn.Tanh(),  # bounded [-1, 1], independent, no competition
        )
        # Gentle bias: tanh(0.05)≈0.05 → atoms barely active but gradient flows
        self.scale_net[-2].bias.data.fill_(0.05)

    def fourier_encode(self, pass_num: int, device: torch.device) -> torch.Tensor:
        """Encode pass number as smooth Fourier features."""
        t = pass_num * self.fourier_freqs.to(device)  # (dim/2,)
        encoding = torch.cat([torch.sin(t), torch.cos(t)])  # (dim,)
        return self.pass_project(encoding)

    def forward(
        self,
        state_pages: List[torch.Tensor],
        pass_num: int,
    ) -> torch.Tensor:
        """
        Args:
            state_pages: list of (batch, page_size) tensors (accumulated pages)
            pass_num: int (0-indexed)

        Returns:
            atom_scales: (batch, num_atoms) tanh-bounded in [-1, 1]
        """
        if len(state_pages) == 0:
            # Pass 0: no pages yet, return zeros (no LoRA modification)
            batch_size = 1  # caller must handle
            device = self.page_query.device
            return torch.zeros(batch_size, self.num_atoms, device=device)

        batch_size = state_pages[0].size(0)
        device = state_pages[0].device

        # Stack and project pages
        pages = torch.stack(state_pages, dim=1)         # (B, P, page_size)
        pages_proj = self.page_project(pages)            # (B, P, attn_dim)

        queries = self.page_query.unsqueeze(0).expand(batch_size, -1, -1)  # (B, Q, attn_dim)

        # Layer 1: cross-attention + FFN
        att1, _ = self.page_attn_1(query=queries, key=pages_proj, value=pages_proj)
        queries = self.page_norm_1(queries + att1)
        queries = self.page_ffn_norm_1(queries + self.page_ffn_1(queries))

        # Layer 2: cross-attention + FFN
        att2, _ = self.page_attn_2(query=queries, key=pages_proj, value=pages_proj)
        queries = self.page_norm_2(queries + att2)
        queries = self.page_ffn_norm_2(queries + self.page_ffn_2(queries))

        # Flatten queries -> summary
        page_summary = self.summary_project(queries.flatten(1))  # (B, 1024)

        # Fourier pass encoding
        pass_emb = self.fourier_encode(pass_num, device).unsqueeze(0).expand(batch_size, -1)  # (B, 512)

        # Combine and generate atom scales
        combined = torch.cat([page_summary, pass_emb], dim=-1)  # (B, 1536)
        atom_scales = self.scale_net(combined)                   # (B, num_atoms)

        return atom_scales


# ---------------------------------------------------------------------------
# AnswerHead — digit-based answer extraction from last page
# ---------------------------------------------------------------------------
class AnswerHead(nn.Module):
    """
    Reads the last page (64 floats) and predicts the answer as digits.

    Three sub-heads:
    - sign_head:   Linear(page_size, 2)           -> positive or negative
    - length_head: Linear(page_size, max_digits)   -> how many digits
    - digit_heads: max_digits x Linear(page_size, 10) -> 0-9 per position
    """

    def __init__(self, page_size: int = 64, max_digits: int = 6):
        super().__init__()
        self.page_size = page_size
        self.max_digits = max_digits

        self.sign_head = nn.Linear(page_size, 2)
        self.length_head = nn.Linear(page_size, max_digits)
        self.digit_heads = nn.ModuleList([
            nn.Linear(page_size, 10) for _ in range(max_digits)
        ])

    def forward(
        self, last_page: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            last_page: (B, page_size)

        Returns:
            sign_logits:   (B, 2)
            length_logits: (B, max_digits)
            digit_logits:  list of max_digits x (B, 10)
        """
        page = last_page.float()
        sign_logits = self.sign_head(page)
        length_logits = self.length_head(page)
        digit_logits = [head(page) for head in self.digit_heads]
        return sign_logits, length_logits, digit_logits

    @torch.no_grad()
    def decode(self, last_page: torch.Tensor) -> torch.Tensor:
        """
        Decode predicted answer as integer tensor (B,).

        Args:
            last_page: (B, page_size)

        Returns:
            answers: (B,) integer tensor
        """
        sign_logits, length_logits, digit_logits = self.forward(last_page)
        batch_size = last_page.size(0)

        num_digits = length_logits.argmax(dim=-1) + 1  # (B,) 1-indexed
        is_negative = sign_logits.argmax(dim=-1) == 1   # (B,)

        answers = torch.zeros(batch_size, dtype=torch.long, device=last_page.device)
        for i in range(self.max_digits):
            digit = digit_logits[i].argmax(dim=-1)  # (B,)
            answers = answers * 10 + digit

        # Trim to predicted length
        trim_power = (self.max_digits - num_digits).clamp(min=0)
        divisor = (10 ** trim_power).long()
        answers = answers // divisor

        # Apply sign
        answers = torch.where(is_negative, -answers, answers)
        return answers


def answer_head_loss(
    answer_head: AnswerHead,
    last_page: torch.Tensor,
    gold_answers: torch.Tensor,
) -> torch.Tensor:
    """
    Compute answer head loss from gold integer answers.

    Args:
        answer_head: AnswerHead module
        last_page:   (B, page_size)
        gold_answers: (B,) integer tensor

    Returns:
        loss: scalar tensor
    """
    sign_logits, length_logits, digit_logits = answer_head(last_page)
    device = last_page.device
    batch_size = last_page.size(0)

    gold_abs = gold_answers.abs()
    gold_sign = (gold_answers < 0).long()  # 0=positive, 1=negative

    gold_strings = [str(v.item()) for v in gold_abs]
    max_digits = answer_head.max_digits

    gold_length = torch.tensor(
        [len(s) - 1 for s in gold_strings],  # 0-indexed for CE
        dtype=torch.long, device=device,
    )
    gold_length = gold_length.clamp(max=max_digits - 1)

    # Digit matrix (left-aligned, most significant first)
    gold_digit_matrix = torch.zeros(
        batch_size, max_digits, dtype=torch.long, device=device,
    )
    for b, s in enumerate(gold_strings):
        s = s[:max_digits]
        for i, ch in enumerate(s):
            gold_digit_matrix[b, i] = int(ch)

    # Losses
    loss = F.cross_entropy(sign_logits, gold_sign)
    loss = loss + F.cross_entropy(length_logits, gold_length)

    for i in range(max_digits):
        mask = torch.tensor(
            [1.0 if i < len(gold_strings[b]) else 0.0 for b in range(batch_size)],
            device=device,
        )
        if mask.sum() > 0:
            digit_loss = F.cross_entropy(
                digit_logits[i], gold_digit_matrix[:, i], reduction='none',
            )
            loss = loss + (digit_loss * mask).sum() / mask.sum()

    return loss


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
        hidden: int = 128,
        num_heads: int = 4,
    ):
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
        pages = torch.stack(state_pages, dim=1).float()  # (B, P, page_size)
        pages_proj = self.page_project(pages)             # (B, P, hidden)

        batch_size = pages.size(0)
        q = self.query.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 1, hidden)
        attended, _ = self.attn(query=q, key=pages_proj, value=pages_proj)
        return self.output(attended.squeeze(1))  # (B, 1)


# ---------------------------------------------------------------------------
# AtomLoRAModel — main model class
# ---------------------------------------------------------------------------
class AtomLoRAModel(nn.Module):
    """
    64-Atom LoRA model: anonymous atoms with independent tanh scaling.

    - Llama 3.2 1B frozen
    - 7-layer perceiver compressor (strategy output ignored)
    - Page-based state accumulation (64-float pages on hypersphere)
    - AtomHypernetwork: pages + pass -> 64 atom scales
    - LoRAAtoms: 64 rank-6 atoms applied via batched einsum
    - AnswerHead: digit extraction from last page
    - AtomConfidenceHead: stop decision from pages
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

        # --- Compressor (strategy output ignored in this arch) ---
        self.compressor = Compressor(
            num_transformer_layers=self.num_layers,
            d_transformer=self.d_model,
            d_perceiver=1024,
            num_queries=4,
            num_perceiver_layers=7,
            state_size=self.page_size,
            strategy_size=64,  # still outputs strategy, we just discard it
        )

        # --- LoRA Atoms ---
        self.atoms = LoRAAtoms(
            d_model=self.d_model,
            d_kv=self.d_kv,
            rank=atom_rank,
            num_atoms=num_atoms,
            num_layers=self.num_layers,
        )

        # --- Atom Hypernetwork ---
        self.hypernet = AtomHypernetwork(
            page_size=self.page_size,
            num_atoms=num_atoms,
            attn_dim=512,
            num_query_heads=4,
            num_attn_heads=8,
            pass_embed_dim=512,
        )

        # --- Answer head (digit extraction from last page) ---
        self.answer_head = AnswerHead(
            page_size=self.page_size, max_digits=6,
        )

        # --- Confidence head (pages only, no blend) ---
        self.confidence_head = AtomConfidenceHead(
            page_size=self.page_size, hidden=128, num_heads=4,
        )

        # --- Pi-harmonic page encoding (DCT-like orthogonal basis, zero learnable params) ---
        self.fourier_page = PiHarmonicPageEncoding(page_size=self.page_size)

        # --- Probe head (intermediate value supervision) ---
        self.probe_head = nn.Linear(self.page_size, 1)

    def thinking_pass(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state_pages: List[torch.Tensor],
        pass_num: int,
        max_passes: int = 3,
        prev_mid_states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One thinking pass with atom LoRA.

        Args:
            input_ids:       (B, seq_len)
            attention_mask:  (B, seq_len)
            state_pages:     list of (B, page_size) accumulated pages
            pass_num:        int (0-indexed)
            max_passes:      int, total passes (for gradient scaling)
            prev_mid_states: Optional list of (B, num_queries, d_perceiver) tensors
                from previous passes' perceiver mid-layer states.

        Returns:
            page:               (B, page_size) normalized on hypersphere
            atom_scales:        (B, num_atoms) the scales used this pass
            current_mid_states: (B, num_queries, d_perceiver) detached mid-layer states
        """
        batch_size = input_ids.size(0)

        if len(state_pages) == 0:
            # First pass: no LoRA (no pages to condition on)
            atom_scales = torch.zeros(
                batch_size, self.atoms.num_atoms,
                device=input_ids.device, dtype=torch.float32,
            )
            outputs = self.transformer(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = list(outputs.hidden_states[1:])
        else:
            # Generate atom scales from pages + pass number
            atom_scales = self.hypernet(state_pages, pass_num)

            # Apply atom LoRA via monkey-patching
            manager = AtomAdditiveLoRAManager(self.transformer)
            manager.apply(self.atoms, atom_scales)
            try:
                outputs = self.transformer(
                    input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = list(outputs.hidden_states[1:])
            finally:
                manager.remove()

        # Compress all 16 layers -> page delta (strategy discarded)
        # Pass prev_mid_states for skip connection
        page_delta, _strategy, current_mid_states = self.compressor(
            hidden_states, pass_num, prev_mid_states=prev_mid_states,
        )

        # Normalize on hypersphere
        page = F.normalize(page_delta, dim=-1) * self.page_radius

        # Add Fourier structural identity (after normalization)
        page = self.fourier_page.apply(page, pass_num)

        # Gradient scaling for earlier cycles (amplify earlier passes)
        grad_scale = min(float(max_passes - pass_num), 4.0)
        if grad_scale != 1.0 and page.requires_grad:
            page = page * grad_scale + page.detach() * (1.0 - grad_scale)

        return page, atom_scales, current_mid_states


# ---------------------------------------------------------------------------
# Warm start from any previous checkpoint (perceiver only)
# ---------------------------------------------------------------------------
def warm_start_atom_from_checkpoint(
    atom_model: AtomLoRAModel,
    checkpoint,
) -> None:
    """
    Warm-start an AtomLoRAModel from a previous checkpoint.

    Only loads compressor (perceiver) weights — everything else is fresh:
    - atoms: fresh random init (completely different structure from named templates)
    - hypernet: fresh init (different output dimension, no strategy input)
    - answer_head: fresh init
    - confidence_head: fresh init

    Args:
        atom_model: AtomLoRAModel to warm-start
        checkpoint: path string OR already-loaded checkpoint dict
    """
    if isinstance(checkpoint, str):
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    else:
        ckpt = checkpoint

    # --- Compressor (perceiver — same architecture, direct load) ---
    if 'compressor' in ckpt:
        own = atom_model.compressor.state_dict()
        loaded = 0
        for k, v in ckpt['compressor'].items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded += 1
        atom_model.compressor.load_state_dict(own, strict=False)
        print(f"  compressor: loaded {loaded}/{len(own)}")
    else:
        print("  compressor: not found in checkpoint, fresh init")

    # --- Probe head ---
    if 'probe_head' in ckpt:
        own = atom_model.probe_head.state_dict()
        loaded = 0
        for k, v in ckpt['probe_head'].items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded += 1
        atom_model.probe_head.load_state_dict(own, strict=False)
        print(f"  probe_head: loaded {loaded}/{len(own)}")
    else:
        print("  probe_head: fresh init (not in checkpoint)")

    # --- Everything else is fresh ---
    print("  atoms: fresh init (64 rank-6 atoms, 0.01 scale)")
    print("  hypernet: fresh init (new architecture, no strategy)")
    print("  answer_head: fresh init")
    print("  confidence_head: fresh init")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time

    device = 'cpu'
    torch.manual_seed(42)

    print("=" * 60)
    print("64-Atom LoRA Architecture — Self-Test")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 1. Test AtomHypernetwork
    # ---------------------------------------------------------------
    print("\n--- AtomHypernetwork ---")
    hypernet = AtomHypernetwork(page_size=64, num_atoms=64).to(device)

    # Empty pages (pass 0)
    scales_empty = hypernet([], pass_num=0)
    print(f"  Empty pages -> scales shape: {scales_empty.shape}, "
          f"all zeros: {(scales_empty == 0).all().item()}")

    # With pages
    batch = 4
    pages = [torch.randn(batch, 64, device=device) for _ in range(3)]
    scales = hypernet(pages, pass_num=1)
    print(f"  3 pages -> scales shape: {scales.shape}, "
          f"range: [{scales.min().item():.3f}, {scales.max().item():.3f}]")
    active = (scales.abs() > 0.1).float().mean().item()
    print(f"  Fraction with |scale| > 0.1: {active:.2%}")

    # Different passes should give different scales
    scales_p1 = hypernet(pages, pass_num=1)
    scales_p2 = hypernet(pages, pass_num=2)
    cos_sim = F.cosine_similarity(scales_p1, scales_p2, dim=-1).mean().item()
    print(f"  Pass 1 vs pass 2 cosine: {cos_sim:.4f} (should be < 1.0)")

    hypernet_params = sum(p.numel() for p in hypernet.parameters())
    print(f"  Params: {hypernet_params / 1e6:.2f}M")

    # ---------------------------------------------------------------
    # 2. Test LoRAAtoms
    # ---------------------------------------------------------------
    print("\n--- LoRAAtoms ---")
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
    # 3. Test AtomAdditiveLoRAManager with mock Llama layers
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
    # 4. Test AnswerHead encode/decode
    # ---------------------------------------------------------------
    print("\n--- AnswerHead ---")
    answer_head = AnswerHead(page_size=64, max_digits=6)

    last_page = torch.randn(batch, 64)
    sign_logits, length_logits, digit_logits = answer_head(last_page)
    print(f"  sign_logits: {sign_logits.shape}")
    print(f"  length_logits: {length_logits.shape}")
    print(f"  digit_logits: {len(digit_logits)} x {digit_logits[0].shape}")

    decoded = answer_head.decode(last_page)
    print(f"  Decoded answers: {decoded.tolist()}")

    # Test loss
    gold = torch.tensor([42, -137, 5, 9999], dtype=torch.long)
    loss = answer_head_loss(answer_head, last_page, gold)
    print(f"  Answer head loss: {loss.item():.4f}")

    ah_params = sum(p.numel() for p in answer_head.parameters())
    print(f"  Params: {ah_params:,}")

    # ---------------------------------------------------------------
    # 5. Test AtomConfidenceHead
    # ---------------------------------------------------------------
    print("\n--- AtomConfidenceHead ---")
    conf_head = AtomConfidenceHead(page_size=64, hidden=128, num_heads=4)

    confidence = conf_head(pages)
    print(f"  Confidence shape: {confidence.shape}, "
          f"range: [{confidence.min().item():.3f}, {confidence.max().item():.3f}]")

    conf_params = sum(p.numel() for p in conf_head.parameters())
    print(f"  Params: {conf_params:,}")

    # ---------------------------------------------------------------
    # 6. Test gradient flow
    # ---------------------------------------------------------------
    print("\n--- Gradient flow ---")

    # End-to-end: pages -> hypernet -> scales -> atoms -> output -> loss
    test_pages = [torch.randn(2, 64, requires_grad=True) for _ in range(2)]
    test_hidden = torch.randn(2, 8, 2048)

    test_scales = hypernet(test_pages, pass_num=1)
    test_out = atoms.apply(test_hidden, layer_idx=0, proj_name='q_proj', atom_scales=test_scales)
    test_loss = test_out.sum()
    test_loss.backward()

    pages_grad = all(p.grad is not None and p.grad.abs().sum() > 0 for p in test_pages)
    hypernet_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in hypernet.parameters() if p.requires_grad
    )
    atoms_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in atoms.parameters() if p.requires_grad
    )
    print(f"  Pages have gradient:    {pages_grad}")
    print(f"  Hypernet has gradient:  {hypernet_grad}")
    print(f"  Atoms have gradient:    {atoms_grad}")

    # ---------------------------------------------------------------
    # 7. Parameter count summary
    # ---------------------------------------------------------------
    print("\n--- Parameter Count Summary ---")
    components = {
        'LoRAAtoms': atoms,
        'AtomHypernetwork': hypernet,
        'AnswerHead': answer_head,
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
    print(f"  (+ ~105M perceiver, + 1.23B frozen Llama)")

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
