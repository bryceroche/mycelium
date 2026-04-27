"""
AtomHypernetwork ported to tinygrad.

100M hypernetwork: reads notebook pages + messages -> 64 atom scales.
Architecture: 6-layer cross-attention (contextual path) + direct path (gradient highway),
blended via learned sigmoid parameter, clamped [-3,3] + tanh.

Original: scripts/atom_lora.py (PyTorch)
"""

from tinygrad import Tensor, dtypes
from tinygrad.nn import Linear, LayerNorm
import math
from typing import List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Manual multi-head attention (tinygrad has no nn.MultiheadAttention)
# ---------------------------------------------------------------------------

def multi_head_attention(q: Tensor, k: Tensor, v: Tensor, num_heads: int) -> Tensor:
    """
    Multi-head attention with scaled dot-product.

    Args:
        q: (B, S_q, D) query tensor
        k: (B, S_kv, D) key tensor
        v: (B, S_kv, D) value tensor
        num_heads: number of attention heads

    Returns:
        (B, S_q, D) attended output
    """
    B, S, D = q.shape
    head_dim = D // num_heads
    q = q.reshape(B, S, num_heads, head_dim).transpose(1, 2)       # (B, H, S_q, hd)
    k = k.reshape(B, -1, num_heads, head_dim).transpose(1, 2)      # (B, H, S_kv, hd)
    v = v.reshape(B, -1, num_heads, head_dim).transpose(1, 2)      # (B, H, S_kv, hd)
    attn = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)         # (B, H, S_q, S_kv)
    attn = attn.softmax(axis=-1)
    out = (attn @ v).transpose(1, 2).reshape(B, S, D)              # (B, S_q, D)
    return out


# ---------------------------------------------------------------------------
# Cross-attention layer (attn + norm + ffn + norm, with residual)
# ---------------------------------------------------------------------------

class CrossAttentionLayer:
    """One cross-attention block: Q/K/V projections + LN + FFN + LN."""

    def __init__(self, attn_dim: int, num_heads: int):
        self.num_heads = num_heads
        # Q/K/V projections for cross-attention
        self.q_proj = Linear(attn_dim, attn_dim)
        self.k_proj = Linear(attn_dim, attn_dim)
        self.v_proj = Linear(attn_dim, attn_dim)
        self.o_proj = Linear(attn_dim, attn_dim)
        # Post-attention norm
        self.attn_norm = LayerNorm(attn_dim)
        # FFN: expand 4x then contract
        self.ffn_up = Linear(attn_dim, attn_dim * 4)
        self.ffn_down = Linear(attn_dim * 4, attn_dim)
        # Post-FFN norm
        self.ffn_norm = LayerNorm(attn_dim)

    def __call__(self, queries: Tensor, kv: Tensor) -> Tensor:
        """
        Args:
            queries: (B, Q, attn_dim) — learnable query heads
            kv: (B, P, attn_dim) — projected pages (keys and values)
        Returns:
            (B, Q, attn_dim) — updated queries
        """
        # Cross-attention with residual
        q = self.q_proj(queries)
        k = self.k_proj(kv)
        v = self.v_proj(kv)
        att_out = self.o_proj(multi_head_attention(q, k, v, self.num_heads))
        queries = self.attn_norm(queries + att_out)
        # FFN with residual
        ffn_out = self.ffn_down(self.ffn_up(queries).gelu())
        queries = self.ffn_norm(queries + ffn_out)
        return queries


# ---------------------------------------------------------------------------
# AtomHypernetwork
# ---------------------------------------------------------------------------

class AtomHypernetwork:
    """
    100M hypernetwork: reads notebook pages + messages -> 64 atom scales.

    Two paths blended via learned sigmoid:
    - DIRECT PATH: last_page -> Linear -> GELU -> Dropout -> Linear -> logits
      (gradient highway bypassing cross-attention softmax)
    - CONTEXTUAL PATH: 6-layer cross-attention over all pages -> deep MLP -> logits
      (rich multi-page context reading)

    Final: blend * direct + (1-blend) * contextual, clamp [-3,3], tanh.
    """

    def __init__(
        self,
        page_size: int = 64,
        num_atoms: int = 64,
        attn_dim: int = 1024,
        num_query_heads: int = 8,
        num_attn_heads: int = 16,
        num_attn_layers: int = 6,
        message_dim: int = 32,
        max_messages: int = 12,
    ):
        self.page_size = page_size
        self.num_atoms = num_atoms
        self.attn_dim = attn_dim
        self.num_query_heads = num_query_heads
        self.message_dim = message_dim
        self.max_messages = max_messages

        # ===== DIRECT PATH (gradient highway -- reads last page only) =====
        self.direct_lin1 = Linear(page_size, 512)
        self.direct_lin2 = Linear(512, num_atoms)

        # ===== BLEND (learnable mixing of direct + contextual) =====
        self.blend_logit = Tensor.zeros(1)  # scalar, sigmoid -> (0, 1)

        # ===== CONTEXTUAL PATH: 6-layer cross-attention (~100M) =====
        self.page_project = Linear(page_size, attn_dim)
        # Learnable query heads: (num_query_heads, attn_dim), small init
        self.page_query = Tensor.randn(num_query_heads, attn_dim) * 0.02

        # N cross-attention layers
        self.cross_attn_layers = [
            CrossAttentionLayer(attn_dim, num_attn_heads)
            for _ in range(num_attn_layers)
        ]

        # Flatten queries -> summary: (Q * attn_dim) -> 2048
        self.summary_project = Linear(num_query_heads * attn_dim, 2048)

        # ===== MESSAGE PROJECTION =====
        # Flatten max_messages * message_dim -> 512 -> 512
        self.msg_lin1 = Linear(message_dim * max_messages, 512)
        self.msg_lin2 = Linear(512, 512)

        # ===== DEEP MLP: 2560 -> 2048 -> 1024 -> 512 -> 64 =====
        # (summary 2048 + msg_summary 512 = 2560)
        self.mlp_lin1 = Linear(2560, 2048)
        self.mlp_norm1 = LayerNorm(2048)
        self.mlp_lin2 = Linear(2048, 1024)
        self.mlp_norm2 = LayerNorm(1024)
        self.mlp_lin3 = Linear(1024, 512)
        self.mlp_lin4 = Linear(512, num_atoms)

    def _direct_path(self, last_page: Tensor) -> Tensor:
        """Direct path: last page -> 512 -> num_atoms (gradient highway)."""
        x = self.direct_lin1(last_page).gelu()
        # Note: dropout is training-only; tinygrad Tensor.dropout handles this
        x = x.dropout(0.1)
        return self.direct_lin2(x)

    def _message_summary(self, messages: Optional[List[Tensor]], batch_size: int) -> Tensor:
        """Read and project inter-cycle messages -> (B, 512)."""
        if messages is not None and len(messages) > 0:
            msg_list = list(messages)
            # Pad to max_messages with zeros
            if len(msg_list) < self.max_messages:
                pad_shape = msg_list[0].shape
                pad = [Tensor.zeros(*pad_shape) for _ in range(self.max_messages - len(msg_list))]
                msg_list = msg_list + pad
            # Concatenate along last dim: (B, message_dim * max_messages)
            msg_cat = msg_list[0]
            for m in msg_list[1:self.max_messages]:
                msg_cat = msg_cat.cat(m, dim=-1)
            # Project: 2-layer MLP
            msg_summary = self.msg_lin2(self.msg_lin1(msg_cat).gelu())
        else:
            msg_summary = Tensor.zeros(batch_size, 512)
        return msg_summary

    def _contextual_path(self, pages: Tensor, batch_size: int) -> Tensor:
        """
        Contextual path: cross-attention over all pages -> deep MLP -> logits.

        Args:
            pages: (B, P, page_size) stacked pages
            batch_size: int
        Returns:
            (B, num_atoms) contextual logits (pre-tanh)
        """
        # Project pages to attn_dim
        pages_proj = self.page_project(pages)  # (B, P, attn_dim)

        # Expand learnable queries for the batch
        queries = self.page_query.unsqueeze(0).expand(batch_size, -1, -1)  # (B, Q, attn_dim)

        # N-layer cross-attention + FFN with residual
        for layer in self.cross_attn_layers:
            queries = layer(queries, pages_proj)

        # Flatten queries -> summary
        page_summary = self.summary_project(queries.reshape(batch_size, -1))  # (B, 2048)
        return page_summary

    def _scale_mlp(self, combined: Tensor) -> Tensor:
        """Deep MLP: 2560 -> 2048 -> 1024 -> 512 -> num_atoms."""
        x = self.mlp_lin1(combined).gelu().dropout(0.1)
        x = self.mlp_norm1(x)
        x = self.mlp_lin2(x).gelu().dropout(0.1)
        x = self.mlp_norm2(x)
        x = self.mlp_lin3(x).gelu()
        x = self.mlp_lin4(x)
        return x

    def __call__(
        self,
        state_pages: List[Tensor],
        pass_num: int = 0,
        messages: Optional[List[Tensor]] = None,
        return_pre_tanh: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            state_pages: list of (B, page_size) tensors (accumulated pages)
            pass_num: int (0-indexed), kept for API compat (not used in v24.6+)
            messages: optional list of (B, message_dim) tensors
            return_pre_tanh: if True, also return pre-tanh values

        Returns:
            atom_scales: (B, num_atoms) tanh-bounded in [-1, 1]
            pre_tanh: (B, num_atoms) raw values (only if return_pre_tanh=True)
        """
        # Pass 0 with no pages: return zeros
        if len(state_pages) == 0:
            zeros = Tensor.zeros(1, self.num_atoms)
            if return_pre_tanh:
                return zeros, zeros
            return zeros

        batch_size = state_pages[0].shape[0]

        # Stack pages: (B, P, page_size)
        pages = state_pages[0].unsqueeze(1)
        for p in state_pages[1:]:
            pages = pages.cat(p.unsqueeze(1), dim=1)

        last_page = pages[:, -1, :]  # (B, page_size)

        # ===== DIRECT PATH (gradient highway) =====
        direct_logits = self._direct_path(last_page)  # (B, num_atoms)

        # ===== CONTEXTUAL PATH =====
        page_summary = self._contextual_path(pages, batch_size)  # (B, 2048)

        # Read messages
        msg_summary = self._message_summary(messages, batch_size)  # (B, 512)

        # Combine and run through deep MLP
        combined = page_summary.cat(msg_summary, dim=-1)  # (B, 2560)
        context_logits = self._scale_mlp(combined)  # (B, num_atoms)

        # ===== BLEND direct + contextual =====
        blend = self.blend_logit.sigmoid()  # scalar in (0, 1)
        pre_tanh = blend * direct_logits + (Tensor.ones(1) - blend) * context_logits

        # Hard clamp [-3, 3]: tanh(3) = 0.995, gradient = 0.01 -- small but non-zero
        pre_tanh = pre_tanh.clip(-3.0, 3.0)

        # Final activation
        atom_scales = pre_tanh.tanh()  # (B, num_atoms)

        if return_pre_tanh:
            return atom_scales, pre_tanh
        return atom_scales
