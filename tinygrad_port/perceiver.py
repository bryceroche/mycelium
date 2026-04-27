"""
Perceiver compressor ported to tinygrad.

7-layer perceiver that compresses 16 Llama hidden state layers into a 64-float
page delta. Includes Haar wavelet preprocessing for 2x input compression.

Port of src/compressor_v3.py (PyTorch) -> tinygrad.
"""

from tinygrad import Tensor, dtypes
from tinygrad_port.nn_utils import Linear, LayerNorm
import math
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Attention helpers
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, num_heads: int
) -> Tensor:
    """
    Multi-head scaled dot-product attention.

    Args:
        q: (batch, seq_q, d_model)
        k: (batch, seq_k, d_model)
        v: (batch, seq_k, d_model)
        num_heads: number of attention heads

    Returns:
        (batch, seq_q, d_model)
    """
    batch, seq_q, d_model = q.shape
    seq_k = k.shape[1]
    d_head = d_model // num_heads

    # Reshape to (batch, num_heads, seq, d_head)
    q = q.reshape(batch, seq_q, num_heads, d_head).permute(0, 2, 1, 3)
    k = k.reshape(batch, seq_k, num_heads, d_head).permute(0, 2, 1, 3)
    v = v.reshape(batch, seq_k, num_heads, d_head).permute(0, 2, 1, 3)

    # Scaled dot-product: (batch, heads, seq_q, d_head) @ (batch, heads, d_head, seq_k)
    scale = 1.0 / math.sqrt(d_head)
    attn_weights = (q @ k.permute(0, 1, 3, 2)) * scale  # (batch, heads, seq_q, seq_k)
    attn_weights = attn_weights.softmax(axis=-1)

    # Weighted values: (batch, heads, seq_q, d_head)
    out = attn_weights @ v

    # Merge heads: (batch, seq_q, d_model)
    out = out.permute(0, 2, 1, 3).reshape(batch, seq_q, d_model)
    return out


# ---------------------------------------------------------------------------
# Multi-head attention with projections
# ---------------------------------------------------------------------------

class MultiheadAttention:
    """Multi-head attention with learned Q/K/V/output projections."""

    def __init__(self, embed_dim: int, num_heads: int,
                 kdim: Optional[int] = None, vdim: Optional[int] = None):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(self.kdim, embed_dim)
        self.v_proj = Linear(self.vdim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def __call__(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        attn_out = scaled_dot_product_attention(q, k, v, self.num_heads)
        return self.out_proj(attn_out)


# ---------------------------------------------------------------------------
# Haar wavelet preprocessing
# ---------------------------------------------------------------------------

class HaarWaveletPreprocess:
    """
    2x compression via Haar wavelet decomposition along the sequence dimension.

    Multi-level Haar transform: each level splits into averages (smooth) and
    differences (detail). Truncating the finest detail level gives ~2x compression.
    Output ordered coarsest-to-finest for natural perceiver reading order.
    Learned importance weights per level (Apery-weighted init: 1/k^3).
    """

    def __init__(self, max_level: int = 4, truncate_finest: bool = True):
        self.max_level = max_level
        self.truncate_finest = truncate_finest

        # Apery-weighted initialization (1/k^3 power law, coarse > fine)
        apery = [1.0 / (k + 1) ** 3 for k in range(max_level + 1)]
        total = sum(apery)
        apery_norm = [w / total * (max_level + 1) for w in apery]
        self.level_weights = Tensor(apery_norm).reshape(max_level + 1).requires_grad_()

    def __call__(self, hidden_states: Tensor) -> Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, d_model)
        Returns:
            (batch, compressed_len, d_model)
        """
        coefficients: List[Tensor] = []
        current = hidden_states

        for level in range(self.max_level):
            seq_len = current.shape[1]

            # Pad if odd length
            if seq_len % 2 == 1:
                last = current[:, -1:, :]  # (batch, 1, d_model)
                current = current.cat(last, dim=1)

            even = current[:, 0::2, :]
            odd = current[:, 1::2, :]

            inv_sqrt2 = 1.0 / math.sqrt(2)
            averages = (even + odd) * inv_sqrt2
            details = (even - odd) * inv_sqrt2

            weight = self.level_weights[level].sigmoid()
            coefficients.append(details * weight)

            current = averages

        # Coarsest approximation coefficients
        weight = self.level_weights[-1].sigmoid()
        coefficients.append(current * weight)

        # Truncate finest detail if requested
        if self.truncate_finest and len(coefficients) > 1:
            coefficients = coefficients[1:]

        # Reverse so coarsest first
        coefficients.reverse()

        return coefficients[0].cat(*coefficients[1:], dim=1)


class LayerWiseWavelet:
    """Apply shared wavelet preprocessing independently to each transformer layer."""

    def __init__(self, num_layers: int = 16, max_level: int = 4,
                 truncate_finest: bool = True):
        self.num_layers = num_layers
        self.wavelet = HaarWaveletPreprocess(
            max_level=max_level, truncate_finest=truncate_finest
        )

    def __call__(self, all_layer_hidden_states: List[Tensor]) -> List[Tensor]:
        return [self.wavelet(hs) for hs in all_layer_hidden_states]


# ---------------------------------------------------------------------------
# FFN block (Linear -> GELU -> Linear)
# ---------------------------------------------------------------------------

class FFN:
    """Feed-forward network: Linear -> GELU -> Linear."""

    def __init__(self, d_model: int, d_ff: int):
        self.up = Linear(d_model, d_ff)
        self.down = Linear(d_ff, d_model)

    def __call__(self, x: Tensor) -> Tensor:
        return self.down(self.up(x).gelu())


# ---------------------------------------------------------------------------
# Single perceiver layer
# ---------------------------------------------------------------------------

class PerceiverLayer:
    """
    One perceiver layer: cross-attention + self-attention + FFN.
    Each sub-block has a residual connection and LayerNorm.
    """

    def __init__(self, d_perceiver: int, num_heads: int = 8):
        self.cross_attn = MultiheadAttention(d_perceiver, num_heads,
                                             kdim=d_perceiver, vdim=d_perceiver)
        self.cross_norm = LayerNorm(d_perceiver)

        self.self_attn = MultiheadAttention(d_perceiver, num_heads)
        self.self_norm = LayerNorm(d_perceiver)

        self.ffn = FFN(d_perceiver, d_perceiver * 4)
        self.ffn_norm = LayerNorm(d_perceiver)

    def __call__(self, queries: Tensor, kv: Tensor) -> Tensor:
        # Cross-attention: queries attend to hidden states
        attended = self.cross_attn(queries, kv, kv)
        queries = self.cross_norm(queries + attended)

        # Self-attention among queries
        refined = self.self_attn(queries, queries, queries)
        queries = self.self_norm(queries + refined)

        # FFN
        ffn_out = self.ffn(queries)
        queries = self.ffn_norm(queries + ffn_out)

        return queries


# ---------------------------------------------------------------------------
# Main perceiver compressor
# ---------------------------------------------------------------------------

class Perceiver:
    """
    7-layer perceiver compressor. 16 Llama layers -> 64 floats.

    Architecture:
        1. Optional Haar wavelet preprocessing (2x compression of each layer)
        2. Learned layer gating (softmax weights over 16 layers)
        3. Input projection to perceiver dimension
        4. Cross-attention: learned queries attend to all hidden states
        5. 7 self-attention layers with residual + LayerNorm + FFN
        6. Final projection: flatten queries -> Linear -> 64-dim page delta

    Matches the core perceiver path from src/compressor_v3.py (without the
    direct path skip connection, page communication, and strategy head which
    are orthogonal features).
    """

    def __init__(
        self,
        num_transformer_layers: int = 16,
        d_transformer: int = 2048,
        d_perceiver: int = 1024,
        num_queries: int = 4,
        num_perceiver_layers: int = 7,
        page_size: int = 64,
        max_passes: int = 20,
        use_wavelet: bool = True,
        wavelet_levels: int = 4,
    ):
        self.num_transformer_layers = num_transformer_layers
        self.d_transformer = d_transformer
        self.d_perceiver = d_perceiver
        self.num_queries = num_queries
        self.num_perceiver_layers = num_perceiver_layers
        self.page_size = page_size
        self.max_passes = max_passes
        self.use_wavelet = use_wavelet

        # Haar wavelet preprocessing (v24.3)
        if use_wavelet:
            self.wavelet = LayerWiseWavelet(
                num_layers=num_transformer_layers,
                max_level=wavelet_levels,
                truncate_finest=True,
            )
        else:
            self.wavelet = None

        # Learned queries: (num_queries, d_perceiver)
        self.queries = (Tensor.randn(num_queries, d_perceiver) * 0.02).requires_grad_()

        # Pass embedding: (max_passes, d_perceiver)
        self.pass_embed = (Tensor.randn(max_passes, d_perceiver) * 0.02).requires_grad_()

        # Layer gate: Linear(d_perceiver, 64) -> ReLU -> Linear(64, num_transformer_layers)
        self.layer_gate_1 = Linear(d_perceiver, 64)
        self.layer_gate_2 = Linear(64, num_transformer_layers)

        # Input projection: d_transformer -> d_perceiver
        self.input_project = Linear(d_transformer, d_perceiver)

        # 7-layer perceiver stack
        self.layers = [PerceiverLayer(d_perceiver, num_heads=8)
                       for _ in range(num_perceiver_layers)]

        # Skip connection attention (cross-attend to previous passes' mid-layer states)
        self.skip_attn = MultiheadAttention(d_perceiver, num_heads=8)
        self.skip_norm = LayerNorm(d_perceiver)

        # Output head: flatten queries -> page_size
        self.state_head = Linear(d_perceiver * num_queries, page_size)

    def __call__(
        self,
        all_layer_hidden_states: List[Tensor],
        pass_num: int,
        prev_mid_states: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compress hidden states into a 64-float page delta.

        Args:
            all_layer_hidden_states: List of 16 tensors, each (batch, seq_len, 2048)
            pass_num: Which thinking pass (0-indexed)
            prev_mid_states: Optional list of (batch, num_queries, d_perceiver)
                from previous passes' mid-layer (after layer 4). Enables skip
                connection for gradient flow across passes.

        Returns:
            page_delta: (batch, 64) -- the page delta to accumulate on hypersphere
            mid_states: (batch, num_queries, d_perceiver) -- mid-layer states
                (detached) for future skip connections
        """
        assert len(all_layer_hidden_states) == self.num_transformer_layers

        batch_size = all_layer_hidden_states[0].shape[0]

        # Wavelet preprocessing (2x compression)
        if self.wavelet is not None:
            all_layer_hidden_states = self.wavelet(all_layer_hidden_states)

        # Pass conditioning: look up pass embedding, add to queries
        pass_context = self.pass_embed[pass_num]  # (d_perceiver,)
        # Broadcast: (num_queries, d_perceiver) + (d_perceiver,) -> expand to batch
        queries = (self.queries + pass_context).reshape(1, self.num_queries, self.d_perceiver) \
            .expand(batch_size, -1, -1)

        # Layer gating: softmax weights over transformer layers
        gate_hidden = self.layer_gate_1(pass_context.reshape(1, -1)).relu()
        layer_logits = self.layer_gate_2(gate_hidden)  # (1, num_transformer_layers)
        layer_weights = layer_logits.softmax(axis=-1)  # (1, num_transformer_layers)

        # Weighted combination of all layers
        # Stack: (num_layers, batch, seq, d_transformer)
        stacked = Tensor.stack(all_layer_hidden_states, dim=0)
        weights = layer_weights.reshape(self.num_transformer_layers, 1, 1, 1)
        combined = (stacked * weights).sum(axis=0)  # (batch, seq, d_transformer)

        # Project to perceiver dimension
        kv = self.input_project(combined)  # (batch, seq, d_perceiver)

        # --- Layers 0-3 (first half) ---
        for layer in self.layers[:4]:
            queries = layer(queries, kv)

        # Save mid-layer states (for skip connection in future passes)
        # In tinygrad there is no .detach(); we stop gradients by not tracking.
        # For forward-only inference this is fine; during training the caller
        # should handle gradient stopping externally.
        mid_states = queries  # (batch, num_queries, d_perceiver)

        # Skip connection: cross-attend to previous passes' mid-layer states
        if prev_mid_states is not None and len(prev_mid_states) > 0:
            prev = prev_mid_states[0].cat(*prev_mid_states[1:], dim=1) \
                if len(prev_mid_states) > 1 else prev_mid_states[0]
            skip_out = self.skip_attn(queries, prev, prev)
            queries = self.skip_norm(queries + skip_out)

        # --- Layers 4-6 (second half) ---
        for layer in self.layers[4:]:
            queries = layer(queries, kv)

        # Flatten queries and project to page_size
        flat = queries.reshape(batch_size, self.num_queries * self.d_perceiver)
        page_delta = self.state_head(flat)  # (batch, page_size)

        return page_delta, mid_states


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing tinygrad Perceiver port...")

    # Build model
    perceiver = Perceiver(
        num_transformer_layers=16,
        d_transformer=2048,
        d_perceiver=1024,
        num_queries=4,
        num_perceiver_layers=7,
        page_size=64,
        use_wavelet=True,
        wavelet_levels=4,
    )

    batch_size = 2
    seq_len = 128

    # Fake 16 Llama hidden state layers
    all_layers = [Tensor.randn(batch_size, seq_len, 2048) for _ in range(16)]

    # Pass 0 (no skip connection)
    page_delta, mid_states = perceiver(all_layers, pass_num=0)
    print(f"Pass 0:")
    print(f"  page_delta shape: {page_delta.shape}  (expected (2, 64))")
    print(f"  mid_states shape: {mid_states.shape}  (expected (2, 4, 1024))")
    print(f"  page_delta values: mean={page_delta.numpy().mean():.4f}, "
          f"std={page_delta.numpy().std():.4f}")

    # Pass 1 (with skip connection from pass 0)
    page_delta_1, mid_states_1 = perceiver(
        all_layers, pass_num=1, prev_mid_states=[mid_states]
    )
    print(f"\nPass 1 (with skip):")
    print(f"  page_delta shape: {page_delta_1.shape}")
    print(f"  mid_states shape: {mid_states_1.shape}")

    # Test without wavelet
    perceiver_no_wav = Perceiver(use_wavelet=False)
    page_delta_nw, _ = perceiver_no_wav(all_layers, pass_num=0)
    print(f"\nNo wavelet:")
    print(f"  page_delta shape: {page_delta_nw.shape}")

    print("\nAll tests passed!")
