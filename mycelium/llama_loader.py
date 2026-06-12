"""Llama-compatible model loader for v200 perceiver-CORE architecture.

Loads LlamaForCausalLM safetensors weights (HuggingFace format) into
a tinygrad-compatible LlamaLayer stack for use as the THINK backbone.

Supported models (all LlamaForCausalLM, hidden_size=2048):
  - HuggingFaceTB/SmolLM2-1.7B  (non-gated, recommended for development)
  - meta-llama/Llama-3.2-1B      (gated — requires HF_TOKEN + access grant)

HF safetensors key naming (standard LlamaForCausalLM):
  model.embed_tokens.weight                        (vocab_size, H)
  model.layers.{l}.input_layernorm.weight          (H,)
  model.layers.{l}.self_attn.q_proj.weight         (H, H)  [n_heads*head_dim, H]
  model.layers.{l}.self_attn.k_proj.weight         (kv_H, H)  [n_kv_heads*head_dim, H]
  model.layers.{l}.self_attn.v_proj.weight         (kv_H, H)
  model.layers.{l}.self_attn.o_proj.weight         (H, H)
  model.layers.{l}.post_attention_layernorm.weight (H,)
  model.layers.{l}.mlp.gate_proj.weight            (ffn, H)
  model.layers.{l}.mlp.up_proj.weight              (ffn, H)
  model.layers.{l}.mlp.down_proj.weight            (H, ffn)
  model.norm.weight                                (H,)

All Linear weights in HF are (out, in); we transpose to (in, out) for x @ w.
RMSNorm has no bias.

Mirror of mycelium/loader.py pattern (Pythia loader).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from tinygrad import Tensor, dtypes, Device
from tinygrad.nn.state import safe_load


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_CACHE_DIR = os.path.join(_PROJECT_ROOT, ".cache", "llama-3.2-1b")

# Default: SmolLM2-1.7B (non-gated, identical architecture to Llama-3.2-1B at 2048d)
# Override with LLAMA_WEIGHTS env var pointing to any LlamaForCausalLM safetensors file.
_DEFAULT_SMOLLM_PATH = os.path.join(
    LLAMA_CACHE_DIR,
    "models--HuggingFaceTB--SmolLM2-1.7B",
    "snapshots",
    "effd688a12921b4cc83e3312b6feb579f70f9c71",
    "model.safetensors",
)
_LLAMA_HF_REPO = "HuggingFaceTB/SmolLM2-1.7B"


@dataclass
class LlamaConfig:
    """Llama architecture hyperparameters (LlamaForCausalLM / SmolLM2-1.7B)."""
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int = 32   # = n_heads → no GQA for SmolLM2; Llama-3.2-1B uses 8
    vocab_size: int = 49152
    rms_norm_eps: float = 1e-5
    rope_theta: float = 130000.0
    max_position_embeddings: int = 8192

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads  # same when no GQA

    @property
    def n_rep(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


# Config matched to SmolLM2-1.7B (default base) and Llama-3.2-1B
SMOLLM2_1_7B_CFG = LlamaConfig(
    hidden_size=2048, intermediate_size=8192, num_hidden_layers=24,
    num_attention_heads=32, num_key_value_heads=32, vocab_size=49152,
    rms_norm_eps=1e-5, rope_theta=130000.0,
)

# Llama-3.2-1B uses GQA: 32 query heads, 8 KV heads
LLAMA_3_2_1B_CFG = LlamaConfig(
    hidden_size=2048, intermediate_size=8192, num_hidden_layers=32,
    num_attention_heads=32, num_key_value_heads=8, vocab_size=128256,
    rms_norm_eps=1e-5, rope_theta=500000.0,
)


# ---------------------------------------------------------------------------
# RMS Norm
# ---------------------------------------------------------------------------

def _rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """RMSNorm: x / rms(x) * weight (no bias)."""
    rms = (x.float().pow(2).mean(axis=-1, keepdim=True) + eps).sqrt()
    return (x.float() / rms * weight.float()).cast(x.dtype)


# ---------------------------------------------------------------------------
# RoPE utilities
# ---------------------------------------------------------------------------

def _build_rope_freqs(head_dim: int, max_seq_len: int, theta: float) -> tuple[Tensor, Tensor]:
    """Precompute cos/sin for RoPE. Returns (cos, sin) each shape (max_seq_len, head_dim)."""
    half_dim = head_dim // 2
    freqs = 1.0 / (theta ** (np.arange(0, half_dim, dtype=np.float32) * 2 / head_dim))
    t = np.arange(max_seq_len, dtype=np.float32)
    angles = np.outer(t, freqs)  # (max_seq_len, half_dim)
    # Interleave cos/sin to match HF LlamaRotaryEmbedding: cos/sin on each pair
    cos_full = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)  # (max_seq_len, head_dim)
    sin_full = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)
    return (
        Tensor(cos_full, dtype=dtypes.float).contiguous().realize(),
        Tensor(sin_full, dtype=dtypes.float).contiguous().realize(),
    )


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate half as in HF LlamaRotaryEmbedding."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return Tensor.cat(-x2, x1, dim=-1)


def _apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, seq_len: int) -> tuple[Tensor, Tensor]:
    """Apply RoPE to q and k. q/k shape: (B, n_heads, S, head_dim)."""
    cos_s = cos[:seq_len].reshape(1, 1, seq_len, -1).cast(q.dtype)
    sin_s = sin[:seq_len].reshape(1, 1, seq_len, -1).cast(q.dtype)
    q_rot = q * cos_s + _rotate_half(q) * sin_s
    k_rot = k * cos_s + _rotate_half(k) * sin_s
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# LlamaLayer: single Llama transformer block (pre-norm, SwiGLU, RoPE, optional GQA)
# ---------------------------------------------------------------------------

class LlamaLayer:
    """Single Llama transformer block (pre-norm RMSNorm, SwiGLU FFN, full/GQA attention).

    Designed for v200: called with (x, rope_cos, rope_sin, attn_mask=None).
    Input/output: (B, S, H) — S is the latent sequence dimension (32 for v200).
    No KV cache — v200 latent self-attention sequence is short (32 tokens).
    """

    def __init__(self, cfg: LlamaConfig):
        self.cfg = cfg
        H = cfg.hidden_size
        kv_H = cfg.num_key_value_heads * cfg.head_dim
        ffn = cfg.intermediate_size

        # Attention projections — (in, out) layout for x @ w
        self.wq = Tensor.zeros(H, H, dtype=dtypes.float).contiguous()
        self.wk = Tensor.zeros(H, kv_H, dtype=dtypes.float).contiguous()
        self.wv = Tensor.zeros(H, kv_H, dtype=dtypes.float).contiguous()
        self.wo = Tensor.zeros(H, H, dtype=dtypes.float).contiguous()

        # FFN projections (SwiGLU: gate and up are input projections, down is output)
        self.w_gate = Tensor.zeros(H, ffn, dtype=dtypes.float).contiguous()
        self.w_up   = Tensor.zeros(H, ffn, dtype=dtypes.float).contiguous()
        self.w_down = Tensor.zeros(ffn, H, dtype=dtypes.float).contiguous()

        # RMSNorm weights (no bias)
        self.attn_norm = Tensor.ones(H, dtype=dtypes.float).contiguous()
        self.ffn_norm  = Tensor.ones(H, dtype=dtypes.float).contiguous()

        self._scale = 1.0 / math.sqrt(cfg.head_dim)

    def __call__(self, x: Tensor, rope_cos: Tensor, rope_sin: Tensor,
                 attn_mask: Tensor | None = None) -> Tensor:
        cfg = self.cfg
        B, S, H = x.shape
        nh = cfg.num_attention_heads
        nkv = cfg.num_key_value_heads
        hd = cfg.head_dim

        # --- Pre-attention RMSNorm ---
        h = _rms_norm(x, self.attn_norm, cfg.rms_norm_eps).cast(x.dtype)

        # --- Attention ---
        q = (h @ self.wq.cast(x.dtype)).reshape(B, S, nh,  hd).transpose(1, 2)  # (B, nh, S, hd)
        k = (h @ self.wk.cast(x.dtype)).reshape(B, S, nkv, hd).transpose(1, 2)  # (B, nkv, S, hd)
        v = (h @ self.wv.cast(x.dtype)).reshape(B, S, nkv, hd).transpose(1, 2)  # (B, nkv, S, hd)

        q, k = _apply_rope(q, k, rope_cos, rope_sin, S)

        # GQA repeat KV heads if needed
        if cfg.n_rep > 1:
            k = k.repeat((1, 1, cfg.n_rep, 1)).reshape(B, nh, S, hd)
            v = v.repeat((1, 1, cfg.n_rep, 1)).reshape(B, nh, S, hd)

        # Scaled dot product attention
        scores = (q @ k.transpose(-2, -1)) * self._scale   # (B, nh, S, S)
        if attn_mask is not None:
            scores = scores + attn_mask.cast(scores.dtype)
        attn = scores.clip(-1e4, 1e4).softmax(-1).cast(v.dtype)
        ctx = (attn @ v).transpose(1, 2).reshape(B, S, H)  # (B, S, H)
        attn_out = ctx @ self.wo.cast(x.dtype)

        x = x + attn_out

        # --- Pre-FFN RMSNorm ---
        h2 = _rms_norm(x, self.ffn_norm, cfg.rms_norm_eps).cast(x.dtype)

        # --- SwiGLU FFN ---
        gate = (h2 @ self.w_gate.cast(x.dtype)).silu()
        up   = (h2 @ self.w_up.cast(x.dtype))
        ffn_out = (gate * up) @ self.w_down.cast(x.dtype)

        return x + ffn_out

    def forward_return_weights(self, x: Tensor, rope_cos: Tensor, rope_sin: Tensor,
                               attn_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Same as __call__ but also returns post-softmax attention weights.

        Returns (x_out, attn_w) where attn_w is (B, nh, S, S).
        Used by v200 #237+ instrumentation (per-latent THINK attention entropy, §7).
        NOTE: keep consistent with __call__ — any change there must mirror here.
        Eager-only; not called inside JIT paths.
        """
        cfg = self.cfg
        B, S, H = x.shape
        nh = cfg.num_attention_heads
        nkv = cfg.num_key_value_heads
        hd = cfg.head_dim

        # --- Pre-attention RMSNorm ---
        h = _rms_norm(x, self.attn_norm, cfg.rms_norm_eps).cast(x.dtype)

        # --- Attention ---
        q = (h @ self.wq.cast(x.dtype)).reshape(B, S, nh,  hd).transpose(1, 2)
        k = (h @ self.wk.cast(x.dtype)).reshape(B, S, nkv, hd).transpose(1, 2)
        v = (h @ self.wv.cast(x.dtype)).reshape(B, S, nkv, hd).transpose(1, 2)

        q, k = _apply_rope(q, k, rope_cos, rope_sin, S)

        if cfg.n_rep > 1:
            k = k.repeat((1, 1, cfg.n_rep, 1)).reshape(B, nh, S, hd)
            v = v.repeat((1, 1, cfg.n_rep, 1)).reshape(B, nh, S, hd)

        scores = (q @ k.transpose(-2, -1)) * self._scale   # (B, nh, S, S)
        if attn_mask is not None:
            scores = scores + attn_mask.cast(scores.dtype)
        attn = scores.clip(-1e4, 1e4).softmax(-1).cast(v.dtype)
        ctx = (attn @ v).transpose(1, 2).reshape(B, S, H)
        attn_out = ctx @ self.wo.cast(x.dtype)

        x = x + attn_out

        h2 = _rms_norm(x, self.ffn_norm, cfg.rms_norm_eps).cast(x.dtype)
        gate = (h2 @ self.w_gate.cast(x.dtype)).silu()
        up   = (h2 @ self.w_up.cast(x.dtype))
        ffn_out = (gate * up) @ self.w_down.cast(x.dtype)

        return x + ffn_out, attn

    def parameters(self) -> list[Tensor]:
        return [self.wq, self.wk, self.wv, self.wo,
                self.w_gate, self.w_up, self.w_down,
                self.attn_norm, self.ffn_norm]


# ---------------------------------------------------------------------------
# Weight loading utilities (mirrors mycelium/loader.py pattern)
# ---------------------------------------------------------------------------

def _gpu(t: Tensor) -> Tensor:
    """Materialize a safetensors-backed tensor on the default device."""
    return t.to(Device.DEFAULT).realize()


def _assign(dst: Tensor, src: Tensor) -> None:
    """Copy src into dst, preserving tensor identity (optimizer wiring intact)."""
    if src.shape != dst.shape:
        src = src.reshape(dst.shape)
    if src.dtype != dst.dtype:
        src = src.cast(dst.dtype)
    if src.device != dst.device:
        src = src.to(dst.device)
    dst.assign(src).realize()


def _get_weights_path() -> str:
    """Resolve Llama weights path: LLAMA_WEIGHTS env > default SmolLM2 path."""
    env_path = os.environ.get("LLAMA_WEIGHTS", "")
    if env_path and os.path.exists(env_path):
        return env_path
    if os.path.exists(_DEFAULT_SMOLLM_PATH):
        return _DEFAULT_SMOLLM_PATH
    raise FileNotFoundError(
        f"Llama weights not found.\n"
        f"  Expected: {_DEFAULT_SMOLLM_PATH}\n"
        f"  Or set LLAMA_WEIGHTS=/path/to/model.safetensors\n"
        f"  Download: cd {_PROJECT_ROOT} && .venv/bin/python -c \"\n"
        f"    from huggingface_hub import hf_hub_download\n"
        f"    hf_hub_download('{_LLAMA_HF_REPO}', 'model.safetensors',\n"
        f"                    cache_dir='.cache/llama-3.2-1b')\"\n"
    )


def load_llama_weights(path: str | None = None) -> Dict[str, Tensor]:
    """Load Llama/SmolLM2 safetensors. Returns the raw HF state dict."""
    if path is None:
        path = _get_weights_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Llama weights not found at {path}")
    print(f"[llama_loader] loading weights from {path}", flush=True)
    sd = safe_load(path)
    print(f"[llama_loader] loaded {len(sd)} keys", flush=True)
    return sd


def _load_llama_layer_weights(
    layer: LlamaLayer,
    sd: Dict[str, Tensor],
    layer_idx: int,
    cfg: LlamaConfig,
) -> None:
    """Load HF Llama layer weights into a LlamaLayer via .assign()."""
    p = f"model.layers.{layer_idx}"

    # Attention (HF stores (out, in) → transpose to (in, out))
    _assign(layer.wq,   _gpu(sd[f"{p}.self_attn.q_proj.weight"]).T)
    _assign(layer.wk,   _gpu(sd[f"{p}.self_attn.k_proj.weight"]).T)
    _assign(layer.wv,   _gpu(sd[f"{p}.self_attn.v_proj.weight"]).T)
    _assign(layer.wo,   _gpu(sd[f"{p}.self_attn.o_proj.weight"]).T)

    # FFN (gate=w1, up=w3, down=w2 in HF naming; all transposed)
    _assign(layer.w_gate, _gpu(sd[f"{p}.mlp.gate_proj.weight"]).T)
    _assign(layer.w_up,   _gpu(sd[f"{p}.mlp.up_proj.weight"]).T)
    _assign(layer.w_down, _gpu(sd[f"{p}.mlp.down_proj.weight"]).T)

    # RMSNorm (no transpose needed — shape is (H,))
    _assign(layer.attn_norm, _gpu(sd[f"{p}.input_layernorm.weight"]))
    _assign(layer.ffn_norm,  _gpu(sd[f"{p}.post_attention_layernorm.weight"]))


def attach_llama_layers(
    model: Any,
    n_layers: int = 4,
    sd: Dict[str, Tensor] | None = None,
    cfg: LlamaConfig | None = None,
    layer_offset: int = 0,
) -> None:
    """Build and attach n_layers LlamaLayer instances to model.

    Attaches:
      model.llama_layers    : list[LlamaLayer]   (n_layers instances)
      model.llama_rope_cos  : Tensor (max_pos, H)
      model.llama_rope_sin  : Tensor (max_pos, H)
      model.llama_cfg       : LlamaConfig
      model.llama_embed     : Tensor (vocab_size, H)  — embed_tokens weight

    Uses layers layer_offset..layer_offset+n_layers-1 from the Llama checkpoint.
    Default is L0-L3 (layer_offset=0).

    If sd is None, calls load_llama_weights() to fetch from disk/cache.
    """
    if cfg is None:
        cfg = SMOLLM2_1_7B_CFG
    if sd is None:
        sd = load_llama_weights()

    H = cfg.hidden_size
    layers = []
    for i in range(n_layers):
        ckpt_idx = layer_offset + i
        layer = LlamaLayer(cfg)
        _load_llama_layer_weights(layer, sd, ckpt_idx, cfg)
        layers.append(layer)
        print(f"[llama_loader] loaded layer L{ckpt_idx} → model.llama_layers[{i}]", flush=True)

    model.llama_layers = layers
    model.llama_cfg = cfg

    # RoPE frequencies (precomputed, not trained)
    rope_cos, rope_sin = _build_rope_freqs(
        cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta
    )
    model.llama_rope_cos = rope_cos.contiguous().realize()
    model.llama_rope_sin = rope_sin.contiguous().realize()

    # Token embedding (used to embed factor graph token IDs)
    embed_w = _gpu(sd["model.embed_tokens.weight"])
    model.llama_embed = Tensor(
        embed_w.realize().numpy().astype(np.float32),
        dtype=dtypes.float,
    ).contiguous().realize()

    n_params = sum(
        int(np.prod(t.shape)) for layer in layers for t in layer.parameters()
    )
    print(
        f"[llama_loader] attached {n_layers} layers (L{layer_offset}..L{layer_offset+n_layers-1}) "
        f"+ embed ({cfg.vocab_size}×{H}) "
        f"+ rope  ({cfg.hidden_size}d, theta={cfg.rope_theta})\n"
        f"  layer params: {n_params/1e6:.1f}M  "
        f"  embed params: {cfg.vocab_size*H/1e6:.1f}M",
        flush=True,
    )


def llama_layer_parameter_count(cfg: LlamaConfig, n_layers: int = 4) -> int:
    """Quick estimate of layer parameter count (excludes embed/rope)."""
    H, ffn = cfg.hidden_size, cfg.intermediate_size
    kv_H = cfg.num_key_value_heads * cfg.head_dim
    per_layer = (
        H * H +       # wq
        H * kv_H +    # wk
        H * kv_H +    # wv
        H * H +       # wo
        H * ffn +     # w_gate
        H * ffn +     # w_up
        ffn * H +     # w_down
        H + H         # attn_norm, ffn_norm
    )
    return per_layer * n_layers
