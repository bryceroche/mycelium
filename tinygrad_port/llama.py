"""
Llama 3.2 1B loader for tinygrad.

Loads safetensor weights directly (no HuggingFace transformers dependency).
All Llama parameters are frozen — the breathing architecture trains around them.

Llama 3.2 1B config:
  dim=2048, n_layers=16, n_heads=32, n_kv_heads=8,
  vocab_size=128256, hidden_dim=8192, rope_theta=500000.0

Weight mapping:  HuggingFace safetensor name  ->  our attribute path
  model.embed_tokens.weight                   ->  tok_embeddings.weight
  model.layers.N.self_attn.{q,k,v,o}_proj.weight  ->  layers[N].attention.w{q,k,v,o}.weight
  model.layers.N.mlp.gate_proj.weight         ->  layers[N].feed_forward.w1.weight
  model.layers.N.mlp.down_proj.weight         ->  layers[N].feed_forward.w2.weight
  model.layers.N.mlp.up_proj.weight           ->  layers[N].feed_forward.w3.weight
  model.layers.N.input_layernorm.weight       ->  layers[N].attention_norm.weight
  model.layers.N.post_attention_layernorm.weight -> layers[N].ffn_norm.weight
  model.norm.weight                           ->  norm.weight
  lm_head.weight                              ->  output.weight
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tinygrad import Tensor, dtypes
from tinygrad.nn import Linear, Embedding
from tinygrad.nn.state import safe_load, get_parameters


# ---------------------------------------------------------------------------
# RMSNorm (Llama uses RMSNorm, not LayerNorm)
# ---------------------------------------------------------------------------
class RMSNorm:
    def __init__(self, dim: int, eps: float = 1e-6):
        self.weight = Tensor.ones(dim)
        self.eps = eps

    def __call__(self, x: Tensor) -> Tensor:
        return x * (x.float().square().mean(axis=-1, keepdim=True) + self.eps).rsqrt() * self.weight


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
#   Llama 3.2 uses theta=500000.0 (NOT the older 10000.0).
# ---------------------------------------------------------------------------
class RotaryEmbedding:
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 500000.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        # freqs: (dim//2,)  —  one frequency per pair of dimensions
        freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2).float() / dim))
        # positions: (max_seq_len,)
        t = Tensor.arange(max_seq_len).float()
        # outer product -> (max_seq_len, dim//2)
        freqs = t.unsqueeze(1) * freqs.unsqueeze(0)
        # Precompute cos/sin tables: (max_seq_len, dim//2)
        self.cos_cached = freqs.cos()
        self.sin_cached = freqs.sin()

    def __call__(self, x: Tensor, start_pos: int = 0) -> Tensor:
        """Apply rotary embeddings to x of shape (B, n_heads, S, head_dim)."""
        _, _, S, D = x.shape
        cos = self.cos_cached[start_pos:start_pos + S]  # (S, D//2)
        sin = self.sin_cached[start_pos:start_pos + S]  # (S, D//2)
        # Reshape for broadcast: (1, 1, S, D//2)
        cos = cos.reshape(1, 1, S, D // 2)
        sin = sin.reshape(1, 1, S, D // 2)
        # Split x into even/odd pairs
        x1 = x[..., :D // 2]   # first half
        x2 = x[..., D // 2:]   # second half
        # Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return out1.cat(out2, dim=-1)


# ---------------------------------------------------------------------------
# Grouped-Query Attention (GQA)
#   32 query heads, 8 KV heads -> 4:1 ratio
#   dim=2048 -> head_dim=64, KV projection dim=8*64=512
# ---------------------------------------------------------------------------
class Attention:
    def __init__(self, dim: int = 2048, n_heads: int = 32, n_kv_heads: int = 8):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads  # 64
        self.n_rep = n_heads // n_kv_heads  # 4

        self.wq = Linear(dim, n_heads * self.head_dim, bias=False)      # 2048 -> 2048
        self.wk = Linear(dim, n_kv_heads * self.head_dim, bias=False)   # 2048 -> 512
        self.wv = Linear(dim, n_kv_heads * self.head_dim, bias=False)   # 2048 -> 512
        self.wo = Linear(dim, dim, bias=False)                          # 2048 -> 2048

    def __call__(self, x: Tensor, rope: RotaryEmbedding, mask: Optional[Tensor] = None,
                 start_pos: int = 0) -> Tensor:
        B, S, D = x.shape

        q = self.wq(x).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)      # (B, 32, S, 64)
        k = self.wk(x).reshape(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, 8, S, 64)
        v = self.wv(x).reshape(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, 8, S, 64)

        # Apply RoPE to Q and K
        q = rope(q, start_pos)
        k = rope(k, start_pos)

        # GQA: repeat K, V heads to match Q head count (8 -> 32, repeat 4x)
        if self.n_rep > 1:
            k = k.repeat((1, self.n_rep, 1, 1))  # (B, 32, S, 64)
            v = v.repeat((1, self.n_rep, 1, 1))  # (B, 32, S, 64)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, 32, S, S)

        if mask is not None:
            attn = attn + mask

        attn = attn.softmax(axis=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)  # (B, S, 2048)
        return self.wo(out)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward  (gate_proj, up_proj, down_proj)
#   hidden_dim=8192 for Llama 3.2 1B
# ---------------------------------------------------------------------------
class FeedForward:
    def __init__(self, dim: int = 2048, hidden_dim: int = 8192):
        self.w1 = Linear(dim, hidden_dim, bias=False)       # gate_proj
        self.w2 = Linear(hidden_dim, dim, bias=False)       # down_proj
        self.w3 = Linear(dim, hidden_dim, bias=False)       # up_proj

    def __call__(self, x: Tensor) -> Tensor:
        return self.w2(self.w1(x).silu() * self.w3(x))


# ---------------------------------------------------------------------------
# Transformer Block (pre-norm residual)
# ---------------------------------------------------------------------------
class TransformerBlock:
    def __init__(self, dim: int = 2048, n_heads: int = 32,
                 n_kv_heads: int = 8, hidden_dim: int = 8192):
        self.attention = Attention(dim, n_heads, n_kv_heads)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def __call__(self, x: Tensor, rope: RotaryEmbedding,
                 mask: Optional[Tensor] = None, start_pos: int = 0) -> Tensor:
        h = x + self.attention(self.attention_norm(x), rope, mask, start_pos)
        return h + self.feed_forward(self.ffn_norm(h))


# ---------------------------------------------------------------------------
# Llama 3.2 1B  (16 layers, 2048 dim, 32 Q heads, 8 KV heads)
# ---------------------------------------------------------------------------
class Llama:
    def __init__(self, dim: int = 2048, n_layers: int = 16, n_heads: int = 32,
                 n_kv_heads: int = 8, vocab_size: int = 128256,
                 hidden_dim: int = 8192, max_seq_len: int = 4096):
        self.dim = dim
        self.n_layers = n_layers
        self.tok_embeddings = Embedding(vocab_size, dim)
        self.layers = [TransformerBlock(dim, n_heads, n_kv_heads, hidden_dim)
                       for _ in range(n_layers)]
        self.norm = RMSNorm(dim)
        self.output = Linear(dim, vocab_size, bias=False)
        self.rope = RotaryEmbedding(dim // n_heads, max_seq_len)

    def __call__(self, tokens: Tensor, output_hidden_states: bool = False,
                 start_pos: int = 0) -> Tuple[Tensor, Optional[List[Tensor]]]:
        """
        Forward pass.

        Args:
            tokens: (B, S) integer token ids
            output_hidden_states: if True, return list of hidden states from
                each layer (needed by the perceiver in the breathing loop)
            start_pos: position offset for RoPE (for KV-cache inference)

        Returns:
            logits: (B, S, vocab_size)
            hidden_states: list of (B, S, dim) tensors if requested, else None
        """
        x = self.tok_embeddings(tokens)
        hidden_states = [x] if output_hidden_states else None

        # Causal mask
        S = tokens.shape[1]
        mask = Tensor.full((S, S), float("-inf")).triu(1).reshape(1, 1, S, S)

        for layer in self.layers:
            x = layer(x, self.rope, mask, start_pos)
            if output_hidden_states:
                hidden_states.append(x)

        x = self.norm(x)
        logits = self.output(x)

        if output_hidden_states:
            hidden_states.append(x)  # post-norm hidden state

        return logits, hidden_states


# ---------------------------------------------------------------------------
# Weight loading: HuggingFace safetensor -> our model
# ---------------------------------------------------------------------------

# Map from HF name prefix to (our attribute path, param name within that object)
_HF_TO_OURS = {
    "model.embed_tokens.weight":                         ("tok_embeddings", "weight"),
    "model.norm.weight":                                 ("norm", "weight"),
    "lm_head.weight":                                    ("output", "weight"),
}

# Per-layer mappings: HF suffix -> our attribute chain within layers[N]
_LAYER_HF_TO_OURS = {
    "self_attn.q_proj.weight":              ("attention.wq", "weight"),
    "self_attn.k_proj.weight":              ("attention.wk", "weight"),
    "self_attn.v_proj.weight":              ("attention.wv", "weight"),
    "self_attn.o_proj.weight":              ("attention.wo", "weight"),
    "mlp.gate_proj.weight":                 ("feed_forward.w1", "weight"),
    "mlp.down_proj.weight":                 ("feed_forward.w2", "weight"),
    "mlp.up_proj.weight":                   ("feed_forward.w3", "weight"),
    "input_layernorm.weight":               ("attention_norm", "weight"),
    "post_attention_layernorm.weight":       ("ffn_norm", "weight"),
}


def _resolve_attr(obj, dotted_path: str):
    """Resolve 'attention.wq' style paths, handling list indices in layers."""
    for part in dotted_path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def load_llama_weights(model: Llama, weights_path: str) -> int:
    """
    Load Llama 3.2 1B weights from one or more safetensor files.

    Args:
        model: our Llama instance
        weights_path: path to a single .safetensors file, or a directory
            containing model-*.safetensors shards

    Returns:
        Number of parameters loaded.
    """
    path = Path(weights_path)
    if path.is_dir():
        # Load all safetensor shards in the directory
        shard_files = sorted(path.glob("*.safetensors"))
        if not shard_files:
            raise FileNotFoundError(f"No .safetensors files in {path}")
        weights: Dict[str, Tensor] = {}
        for sf in shard_files:
            weights.update(safe_load(str(sf)))
    else:
        weights = safe_load(str(path))

    loaded = 0

    for hf_name, tensor in weights.items():
        # --- Global (non-layer) weights ---
        if hf_name in _HF_TO_OURS:
            attr_path, param_name = _HF_TO_OURS[hf_name]
            target = _resolve_attr(model, attr_path)
            setattr(target, param_name, tensor)
            loaded += 1
            continue

        # --- Per-layer weights: model.layers.N.<suffix> ---
        if hf_name.startswith("model.layers."):
            parts = hf_name.split(".", 3)  # ['model', 'layers', 'N', suffix]
            if len(parts) < 4:
                continue
            layer_idx = int(parts[2])
            suffix = parts[3]
            if suffix in _LAYER_HF_TO_OURS:
                attr_chain, param_name = _LAYER_HF_TO_OURS[suffix]
                target = _resolve_attr(model.layers[layer_idx], attr_chain)
                setattr(target, param_name, tensor)
                loaded += 1
                continue

        # Weights we don't need (e.g. rotary_emb.inv_freq — we compute our own)
        # Silently skip.

    return loaded


def freeze_llama(model: Llama) -> None:
    """Freeze all Llama parameters (exclude from optimizer)."""
    for p in get_parameters(model):
        p.requires_grad = False


# ---------------------------------------------------------------------------
# Convenience: build + load + freeze in one call
# ---------------------------------------------------------------------------
def build_llama(weights_path: str, freeze: bool = True) -> Llama:
    """
    Build Llama 3.2 1B, load weights, optionally freeze.

    Args:
        weights_path: path to safetensors file or directory of shards
        freeze: if True (default), freeze all parameters

    Returns:
        Loaded (and optionally frozen) Llama model.
    """
    model = Llama()
    n = load_llama_weights(model, weights_path)
    print(f"[llama] loaded {n} weight tensors from {weights_path}")
    if freeze:
        freeze_llama(model)
        print("[llama] all parameters frozen")
    return model
