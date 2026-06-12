"""Llama-family tinygrad loader for Mycelium v200 (Stage 1B unified).

Stage 1B CONSOLIDATION of the Stage 1A SmolLM2-only loader and the
inline LlamaBase32 class from scripts/v200_llama32_smoke.py into one
GQA-aware class.

Handles:
  - SmolLM2-1.7B  (hidden=2048, 32 Q heads, 32 KV heads — no GQA)
  - Llama-3.2-1B  (hidden=2048, 32 Q heads,  8 KV heads — GQA 4:1)

GQA detection: reads n_kv_heads from HF config.json. If n_kv_heads ==
n_heads: non-GQA path (wk/wv each H×H). If n_kv_heads < n_heads: GQA path
(wk/wv each H×kv_dim; expanded at attention time by repeating each KV head
n_q_per_kv times).

Architecture notes vs Pythia-410M (loader.py):
  - RMSNorm (no bias), not LayerNorm. Uses tinygrad nn.RMSNorm.
  - SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x)). Three matrices.
  - Separate Q/K/V projections (not fused QKV like Pythia).
  - RoPE on Q and K (full head_dim rotation, not Pythia's partial 25%).
  - No attention bias (attention_bias=False in both SmolLM2 and Llama-3.2-1B).
  - SmolLM2: tie_word_embeddings=True. Llama-3.2-1B: tie_word_embeddings=True.

Interface contract (Stage 1A → 1B, per docs/v200_brief.md §15):
  model.embed            — Embedding (vocab_size, hidden_size)
  model.layers[k]        — LlamaLayer, k in 0..3
    .attn                — LlamaAttention (GQA-aware)
      .wq                — Tensor (hidden_size, hidden_size)
      .wk                — Tensor (hidden_size, kv_dim)  [kv_dim = n_kv_heads * head_dim]
      .wv                — Tensor (hidden_size, kv_dim)
      .wo                — Tensor (hidden_size, hidden_size)
    .mlp                 — LlamaMLP
      .gate_proj         — Tensor (hidden_size, intermediate_size)
      .up_proj           — Tensor (hidden_size, intermediate_size)
      .down_proj         — Tensor (intermediate_size, hidden_size)
    .attn_ln             — tinygrad RMSNorm (input_layernorm)
    .mlp_ln              — tinygrad RMSNorm (post_attention_layernorm)
    .forward(x, cos, sin) -> Tensor (B, T, H)
    .forward_with_taps(x, cos, sin) -> (Tensor, dict)
  model.ln_f             — RMSNorm (hidden_size) — final model norm
  model.hidden_size      — int
  model.vocab_size       — int
  model.n_heads          — int (Q heads)
  model.n_kv_heads       — int (KV heads; equals n_heads for non-GQA)
  model.head_dim         — int
  model.kv_dim           — int (n_kv_heads * head_dim)
  model.is_gqa           — bool

  model.forward(token_ids)       -> (B, T, H)
  model.forward_with_taps(token_ids) -> (B, T, H), [taps_per_layer]
    where taps_per_layer is a list of 4 dicts per layer.

Tap keys per layer:
  "pre_ln_resid"    — residual BEFORE attention LN (= input x)
  "post_attn_resid" — residual AFTER attention add
  "post_mlp_resid"  — residual AFTER MLP add (= output)

Weight layout convention:
  HF safetensors: linear weights as (out_dim, in_dim) [PyTorch convention].
  Stored here as (in_dim, out_dim) so forward is x @ W (not x @ W.T).
  GQA wk/wv: (kv_dim, hidden_size) in HF → .T → (hidden_size, kv_dim).

wv-sharing status (Jun 11):
  REFUTED on SmolLM2-1.7B (cos=0.5126). UNCLEAR on Llama-3.2-1B (cos=0.9715).
  v200 default: each layer's OWN wv (conservative, per §4 of v200_brief.md).
  Stage 1C training-time test queued at v1.1 to settle the Llama-3.2-1B pin.

JIT notes (Stage 1B is EAGER-ONLY):
  - .cast(dtypes.float32) inside JIT → segfault on AM driver.
  - Per-param isnan() inside JIT → kernel count overload.
  - See memory/reference_tinygrad_am_quirks.md for the full list.
"""

import json
import math
import os
from typing import Dict, List, Optional, Tuple

from tinygrad import Tensor, dtypes, Device
from tinygrad.nn import Embedding, RMSNorm
from tinygrad.nn.state import safe_load


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOLLM2_CACHE = os.path.join(_PROJECT_ROOT, ".cache", "smollm2-1.7b", "model.safetensors")
SMOLLM2_CFG   = os.path.join(_PROJECT_ROOT, ".cache", "smollm2-1.7b", "config.json")
LLAMA_CACHE   = os.path.join(_PROJECT_ROOT, ".cache", "llama-3.2-1b-weights", "model.safetensors")
LLAMA_CFG     = os.path.join(_PROJECT_ROOT, ".cache", "llama-3.2-1b-weights", "config.json")


# ---------------------------------------------------------------------------
# Per-model config dataclass (parsed from HF config.json)
# ---------------------------------------------------------------------------

class _ModelConfig:
    """Parsed HF config.json for one LlamaForCausalLM checkpoint."""

    def __init__(self, cfg: dict):
        self.hidden_size      = cfg["hidden_size"]
        self.num_heads        = cfg["num_attention_heads"]
        self.num_kv_heads     = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
        self.intermediate     = cfg["intermediate_size"]
        self.vocab_size       = cfg["vocab_size"]
        self.rope_theta       = cfg.get("rope_theta", 10000.0)
        self.rms_eps          = cfg.get("rms_norm_eps", 1e-5)
        # head_dim may be explicit (Llama-3.2-1B has it) or derived
        self.head_dim         = cfg.get("head_dim", self.hidden_size // self.num_heads)
        self.is_gqa           = (self.num_kv_heads != self.num_heads)
        self.kv_dim           = self.num_kv_heads * self.head_dim
        self.n_q_per_kv       = self.num_heads // self.num_kv_heads if self.is_gqa else 1

    @classmethod
    def from_path(cls, cfg_path: str) -> "_ModelConfig":
        with open(cfg_path) as f:
            return cls(json.load(f))

    @classmethod
    def from_weights_path(cls, weights_path: str) -> "_ModelConfig":
        """Infer config.json location from weights_path."""
        cfg_path = os.path.join(os.path.dirname(weights_path), "config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"config.json not found alongside weights at {cfg_path}. "
                "Provide a weights_path that lives next to config.json."
            )
        return cls.from_path(cfg_path)


# ---------------------------------------------------------------------------
# SmolLM2 fallback constants (used when no config.json is available)
# ---------------------------------------------------------------------------

_SMOLLM2_DEFAULT = _ModelConfig({
    "hidden_size": 2048,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "intermediate_size": 8192,
    "vocab_size": 49152,
    "rope_theta": 130000.0,
    "rms_norm_eps": 1e-5,
    "head_dim": 64,
})

_MAX_SEQ_LEN = 8192   # practical upper bound for RoPE tables


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def _build_rope_tables(seq_len: int, head_dim: int, rope_theta: float
                       ) -> Tuple[Tensor, Tensor]:
    """Precompute RoPE cos/sin tables (full head_dim rotation).

    Returns (cos, sin) each of shape (seq_len, head_dim), realized so
    they are concrete GPU buffers rather than lazy op graphs.
    """
    half = head_dim // 2
    inv_freq = Tensor(
        [1.0 / (rope_theta ** (2 * i / head_dim)) for i in range(half)],
        dtype=dtypes.float,
    )
    pos = Tensor.arange(seq_len, dtype=dtypes.float)
    angles = pos.reshape(-1, 1) * inv_freq.reshape(1, -1)   # (seq, half)
    angles_full = Tensor.cat(angles, angles, dim=-1)         # (seq, head_dim)
    cos_t = angles_full.cos().contiguous().realize()
    sin_t = angles_full.sin().contiguous().realize()
    return cos_t, sin_t


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE rotation.

    x:   (B, n_heads, T, head_dim)
    cos: (T, head_dim)  — broadcast over B and n_heads
    sin: (T, head_dim)
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = Tensor.cat(-x2, x1, dim=-1)
    c = cos.reshape(1, 1, cos.shape[0], cos.shape[1])
    s = sin.reshape(1, 1, sin.shape[0], sin.shape[1])
    return x * c + rotated * s


# ---------------------------------------------------------------------------
# Attention (GQA-aware)
# ---------------------------------------------------------------------------

class LlamaAttention:
    """GQA-aware multi-head attention.

    If n_kv_heads == n_heads (no GQA, e.g. SmolLM2):
      wk/wv are (hidden_size, hidden_size).
      Forward is standard full-rank attention.

    If n_kv_heads < n_heads (GQA, e.g. Llama-3.2-1B with 8 KV / 32 Q):
      wk/wv are (hidden_size, kv_dim) where kv_dim = n_kv_heads * head_dim.
      Forward expands K/V by repeating each head n_q_per_kv times.

    Weight layout after load: (in_dim, out_dim) so forward uses x @ W.
    """

    def __init__(self, cfg: _ModelConfig):
        self._cfg = cfg
        h  = cfg.hidden_size
        kd = cfg.kv_dim
        # wq always full: (H, H)
        self.wq = Tensor.zeros(h, h)
        # wk/wv sized to KV heads: (H, kv_dim)
        self.wk = Tensor.zeros(h, kd)
        self.wv = Tensor.zeros(h, kd)
        self.wo = Tensor.zeros(h, h)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor,
                attn_mask: Optional[Tensor] = None) -> Tensor:
        """Self-attention with RoPE on Q and K; GQA KV expansion if needed.

        x:   (B, T, H)
        cos: (T, head_dim)
        sin: (T, head_dim)
        Returns (B, T, H).
        """
        cfg = self._cfg
        B, T, H = x.shape
        nh  = cfg.num_heads
        nkv = cfg.num_kv_heads
        hd  = cfg.head_dim

        # Linear projections
        q = x @ self.wq   # (B, T, H)
        k = x @ self.wk   # (B, T, kv_dim)
        v = x @ self.wv   # (B, T, kv_dim)

        # Reshape to multi-head
        q = q.reshape(B, T, nh,  hd).transpose(1, 2)   # (B, nh, T, hd)
        k = k.reshape(B, T, nkv, hd).transpose(1, 2)   # (B, nkv, T, hd)
        v = v.reshape(B, T, nkv, hd).transpose(1, 2)   # (B, nkv, T, hd)

        # Apply RoPE to Q and K
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # GQA: expand K/V to match Q head count
        if cfg.is_gqa:
            k = k.repeat((1, cfg.n_q_per_kv, 1, 1))   # (B, nh, T, hd)
            v = v.repeat((1, cfg.n_q_per_kv, 1, 1))   # (B, nh, T, hd)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(hd)
        scores = (q @ k.transpose(-2, -1)) * scale
        # Score clamping — AM-driver-safe numerical stability (not a cast)
        scores = scores.clip(-1e4, 1e4)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn_w = scores.softmax(-1)

        # Aggregate values
        out = attn_w @ v                              # (B, nh, T, hd)
        out = out.transpose(1, 2).reshape(B, T, H)   # (B, T, H)
        return out @ self.wo

    def forward_return_weights(self, x: Tensor, cos: Tensor, sin: Tensor,
                                attn_mask: Optional[Tensor] = None
                                ) -> Tuple[Tensor, Tensor]:
        """Same as forward but also returns attention weights (B, nh, T, T).

        Used by v200 instrumentation (cross-attn head-group entropy).
        NOTE: keep consistent with forward() — any change there must mirror here.
        """
        cfg = self._cfg
        B, T, H = x.shape
        nh  = cfg.num_heads
        nkv = cfg.num_kv_heads
        hd  = cfg.head_dim

        q = x @ self.wq
        k = x @ self.wk
        v = x @ self.wv

        q = q.reshape(B, T, nh,  hd).transpose(1, 2)
        k = k.reshape(B, T, nkv, hd).transpose(1, 2)
        v = v.reshape(B, T, nkv, hd).transpose(1, 2)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if cfg.is_gqa:
            k = k.repeat((1, cfg.n_q_per_kv, 1, 1))
            v = v.repeat((1, cfg.n_q_per_kv, 1, 1))

        scale = 1.0 / math.sqrt(hd)
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = scores.clip(-1e4, 1e4)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn_w = scores.softmax(-1)   # (B, nh, T, T) — returned

        out = (attn_w @ v).transpose(1, 2).reshape(B, T, H) @ self.wo
        return out, attn_w


# ---------------------------------------------------------------------------
# MLP (SwiGLU)
# ---------------------------------------------------------------------------

class LlamaMLP:
    """SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x)).

    Weight layout after load: (in_dim, out_dim).
    gate_proj: (hidden_size, intermediate_size)
    up_proj:   (hidden_size, intermediate_size)
    down_proj: (intermediate_size, hidden_size)
    """

    def __init__(self, cfg: _ModelConfig):
        h = cfg.hidden_size
        i = cfg.intermediate
        self.gate_proj = Tensor.zeros(h, i)
        self.up_proj   = Tensor.zeros(h, i)
        self.down_proj = Tensor.zeros(i, h)

    def forward(self, x: Tensor) -> Tensor:
        gate = (x @ self.gate_proj).silu()
        up   = x @ self.up_proj
        return (gate * up) @ self.down_proj


# ---------------------------------------------------------------------------
# Transformer layer with tap hooks
# ---------------------------------------------------------------------------

class LlamaLayer:
    """One Llama transformer layer.

    Standard forward:
      x = x + attn(attn_ln(x))
      x = x + mlp(mlp_ln(x))

    forward_with_taps() additionally returns per-step residuals for the §10
    v1.1 energy channel ‖Δz_j‖. Baking it in here means v1.1 is a feature
    add (activate the taps path), not a refactor.
    """

    def __init__(self, cfg: _ModelConfig):
        self.attn    = LlamaAttention(cfg)
        self.mlp     = LlamaMLP(cfg)
        self.attn_ln = RMSNorm(cfg.hidden_size, eps=cfg.rms_eps)
        self.mlp_ln  = RMSNorm(cfg.hidden_size, eps=cfg.rms_eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor,
                attn_mask: Optional[Tensor] = None) -> Tensor:
        """Standard forward without tap capture."""
        x = x + self.attn.forward(self.attn_ln(x), cos, sin, attn_mask)
        x = x + self.mlp.forward(self.mlp_ln(x))
        return x

    def forward_with_taps(self, x: Tensor, cos: Tensor, sin: Tensor,
                          attn_mask: Optional[Tensor] = None
                          ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Forward with intermediate residual taps.

        Returns:
          (x_post, taps) where taps contains:
            "pre_ln_resid"    — residual BEFORE attention LN (= input x)
            "post_attn_resid" — residual AFTER attention add
            "post_mlp_resid"  — residual AFTER MLP add (= x_post)
        """
        pre_ln      = x
        x_post_attn = x + self.attn.forward(self.attn_ln(x), cos, sin, attn_mask)
        x_post_mlp  = x_post_attn + self.mlp.forward(self.mlp_ln(x_post_attn))
        taps = {
            "pre_ln_resid":    pre_ln,
            "post_attn_resid": x_post_attn,
            "post_mlp_resid":  x_post_mlp,
        }
        return x_post_mlp, taps

    def forward_with_attn_weights(self, x: Tensor, cos: Tensor, sin: Tensor,
                                   attn_mask: Optional[Tensor] = None
                                   ) -> Tuple[Tensor, Tensor]:
        """Forward returning (x_post, attn_weights).

        attn_weights: (B, n_heads, T, T) — after softmax, for entropy computation.
        Used by v200 instrumentation to compute self-attention entropy per head.
        """
        x_pre_attn = self.attn_ln(x)
        attn_out, attn_w = self.attn.forward_return_weights(
            x_pre_attn, cos, sin, attn_mask
        )
        x_post_attn = x + attn_out
        x_post_mlp  = x_post_attn + self.mlp.forward(self.mlp_ln(x_post_attn))
        return x_post_mlp, attn_w


# ---------------------------------------------------------------------------
# Top-level unified loader
# ---------------------------------------------------------------------------

class LlamaBase:
    """GQA-aware Llama-family base model: embed + L0-L3 + ln_f.

    Auto-detects GQA from the config.json that sits alongside the weights
    file. Handles SmolLM2-1.7B (no GQA) and Llama-3.2-1B (GQA 8 KV / 32 Q)
    through the same code path; the branch is inside LlamaAttention.

    Constructor:
      LlamaBase(weights_path=None)
        weights_path: path to model.safetensors. If None, tries env vars
        LLAMA_WEIGHTS / SMOLLM2_WEIGHTS, then local cache dirs, then download.

    Class method:
      LlamaBase.load(model_id_or_path)
        Convenience wrapper; also accepts HF model IDs as path hints.

    Interface:
      model.embed           — Embedding (vocab_size, hidden_size)
      model.layers[0..3]    — LlamaLayer list (GQA-aware attn)
      model.ln_f            — RMSNorm (hidden_size)
      model.rope_cos        — Tensor (MAX_SEQ_LEN, head_dim)
      model.rope_sin        — Tensor (MAX_SEQ_LEN, head_dim)
      model.hidden_size, .vocab_size, .n_heads, .n_kv_heads, .head_dim
      model.is_gqa          — bool
      model.kv_dim          — int (n_kv_heads * head_dim)

      model.forward(token_ids)           -> (B, T, H)
      model.forward_with_taps(token_ids) -> (B, T, H), [taps_per_layer]
      model.forward_wv_shared(token_ids) -> (B, T, H)  [wv-sharing test]
    """

    def __init__(self, weights_path: Optional[str] = None):
        path = self._resolve_path(weights_path)
        cfg  = _ModelConfig.from_weights_path(path)

        self.hidden_size = cfg.hidden_size
        self.vocab_size  = cfg.vocab_size
        self.n_heads     = cfg.num_heads
        self.n_kv_heads  = cfg.num_kv_heads
        self.head_dim    = cfg.head_dim
        self.is_gqa      = cfg.is_gqa
        self.kv_dim      = cfg.kv_dim
        self._cfg        = cfg

        # Build model structure
        self.embed  = Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers: List[LlamaLayer] = [LlamaLayer(cfg) for _ in range(4)]
        self.ln_f   = RMSNorm(cfg.hidden_size, eps=cfg.rms_eps)

        # Precompute RoPE tables (realized buffers)
        self.rope_cos, self.rope_sin = _build_rope_tables(
            _MAX_SEQ_LEN, cfg.head_dim, cfg.rope_theta
        )

        self._load_weights(path)

    @classmethod
    def load(cls, model_id_or_path: Optional[str] = None) -> "LlamaBase":
        """Convenience: accepts HF model IDs or absolute paths.

        Examples:
          LlamaBase.load()                               # auto-detect
          LlamaBase.load("meta-llama/Llama-3.2-1B")     # treated as hint
          LlamaBase.load("/path/to/model.safetensors")   # explicit path
        """
        # If it looks like an HF model ID (contains a slash but no OS path sep)
        if (model_id_or_path is not None
                and "/" in model_id_or_path
                and not model_id_or_path.startswith("/")
                and not os.path.exists(model_id_or_path)):
            # Use the model ID as a hint; resolution falls through to auto-detect
            return cls(weights_path=None)
        return cls(weights_path=model_id_or_path)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_path(weights_path: Optional[str]) -> str:
        """Find weights file: explicit arg > env vars > local cache > download."""
        if weights_path is not None:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"Specified weights path does not exist: {weights_path}"
                )
            return weights_path

        # Env overrides
        for env_key, default_path in [
            ("LLAMA_WEIGHTS",   LLAMA_CACHE),
            ("SMOLLM2_WEIGHTS", SMOLLM2_CACHE),
        ]:
            env_val = os.environ.get(env_key)
            if env_val and os.path.exists(env_val):
                print(f"[llama_base] Using {env_key}={env_val}")
                return env_val
            if os.path.exists(default_path):
                print(f"[llama_base] Found weights at {default_path}")
                return default_path

        # Attempt download of SmolLM2-1.7B as fallback
        print("[llama_base] No local weights found. Attempting download of SmolLM2-1.7B...")
        try:
            from huggingface_hub import snapshot_download
            local_dir = os.path.join(_PROJECT_ROOT, ".cache", "smollm2-1.7b")
            snapshot_download(
                "HuggingFaceTB/SmolLM2-1.7B",
                local_dir=local_dir,
                ignore_patterns=["*.bin", "flax_*", "tf_*", "*.msgpack"],
            )
            if os.path.exists(SMOLLM2_CACHE):
                print(f"[llama_base] Download complete: {SMOLLM2_CACHE}")
                return SMOLLM2_CACHE
        except Exception:
            pass

        raise FileNotFoundError(
            "BLOCKER: No Llama/SmolLM2 weights found and download failed.\n"
            "Options:\n"
            "  1. Set LLAMA_WEIGHTS=/path/to/Llama-3.2-1B/model.safetensors\n"
            "  2. Set SMOLLM2_WEIGHTS=/path/to/SmolLM2-1.7B/model.safetensors\n"
            "  3. Run: huggingface-cli download HuggingFaceTB/SmolLM2-1.7B "
            "--local-dir .cache/smollm2-1.7b\n"
            "  4. For Llama-3.2-1B: requires HF token + access grant"
        )

    def _gpu(self, t: Tensor) -> Tensor:
        return t.to(Device.DEFAULT).realize()

    def _assign(self, dst: Tensor, src: Tensor) -> None:
        if src.shape != dst.shape:
            src = src.reshape(dst.shape)
        if src.dtype != dst.dtype:
            # JIT-NOTE: dst.dtype (not hardcoded float32) avoids AM driver segfault in JIT
            src = src.cast(dst.dtype)
        if src.device != dst.device:
            src = src.to(dst.device)
        dst.assign(src).realize()

    def _load_weights(self, path: str) -> None:
        """Load HF safetensors into L0-L3 + embed + ln_f.

        HF convention: linear weights are (out_dim, in_dim).
        We transpose to (in_dim, out_dim) so forward uses x @ W.

        GQA weight dimensions for Llama-3.2-1B:
          q_proj.weight: (hidden_size, hidden_size) → .T → (hidden_size, hidden_size)
          k_proj.weight: (kv_dim, hidden_size)      → .T → (hidden_size, kv_dim)
          v_proj.weight: (kv_dim, hidden_size)      → .T → (hidden_size, kv_dim)
          o_proj.weight: (hidden_size, hidden_size) → .T → (hidden_size, hidden_size)
        """
        print(f"[llama_base] Loading weights from {path}")
        sd = safe_load(path)
        print(f"[llama_base] Loaded {len(sd)} weight tensors")

        # Token embedding
        self._assign(self.embed.weight, self._gpu(sd["model.embed_tokens.weight"]))

        # Layers 0-3
        for i in range(4):
            layer = self.layers[i]
            p = f"model.layers.{i}"

            self._assign(layer.attn.wq, self._gpu(sd[f"{p}.self_attn.q_proj.weight"]).T)
            self._assign(layer.attn.wk, self._gpu(sd[f"{p}.self_attn.k_proj.weight"]).T)
            self._assign(layer.attn.wv, self._gpu(sd[f"{p}.self_attn.v_proj.weight"]).T)
            self._assign(layer.attn.wo, self._gpu(sd[f"{p}.self_attn.o_proj.weight"]).T)

            self._assign(layer.mlp.gate_proj, self._gpu(sd[f"{p}.mlp.gate_proj.weight"]).T)
            self._assign(layer.mlp.up_proj,   self._gpu(sd[f"{p}.mlp.up_proj.weight"]).T)
            self._assign(layer.mlp.down_proj, self._gpu(sd[f"{p}.mlp.down_proj.weight"]).T)

            self._assign(layer.attn_ln.weight, self._gpu(sd[f"{p}.input_layernorm.weight"]))
            self._assign(layer.mlp_ln.weight,  self._gpu(sd[f"{p}.post_attention_layernorm.weight"]))

        # Final norm
        self._assign(self.ln_f.weight, self._gpu(sd["model.norm.weight"]))
        print(f"[llama_base] Weights loaded. L0-L3 + embed + ln_f ready. "
              f"GQA={self.is_gqa} n_heads={self.n_heads} n_kv_heads={self.n_kv_heads} "
              f"kv_dim={self.kv_dim}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _get_rope(self, T: int) -> Tuple[Tensor, Tensor]:
        """Slice RoPE tables to the actual sequence length."""
        return self.rope_cos[:T], self.rope_sin[:T]

    def forward(self, token_ids: Tensor,
                attn_mask: Optional[Tensor] = None) -> Tensor:
        """Forward through embed + L0-L3 + ln_f.

        token_ids: (B, T) integer tensor
        Returns:   (B, T, H)

        No causal mask by default. Pass attn_mask=(B, 1, T, T) or (T, T)
        for causal or structured masking.
        """
        B, T = token_ids.shape
        x = self.embed(token_ids)
        cos, sin = self._get_rope(T)
        for layer in self.layers:
            x = layer.forward(x, cos, sin, attn_mask)
        return self.ln_f(x)

    def forward_with_taps(self, token_ids: Tensor,
                          attn_mask: Optional[Tensor] = None
                          ) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
        """Forward with per-layer residual taps for instrumentation.

        Returns (x_final, layer_taps) where:
          x_final:    (B, T, H) — after L0-L3 + ln_f
          layer_taps: list of 4 dicts (one per layer), each with:
            "pre_ln_resid":    Tensor (B, T, H)
            "post_attn_resid": Tensor (B, T, H)
            "post_mlp_resid":  Tensor (B, T, H)
        """
        B, T = token_ids.shape
        x = self.embed(token_ids)
        cos, sin = self._get_rope(T)

        all_taps: List[Dict[str, Tensor]] = []
        for layer in self.layers:
            x, taps = layer.forward_with_taps(x, cos, sin, attn_mask)
            all_taps.append(taps)

        x = self.ln_f(x)
        return x, all_taps

    # ------------------------------------------------------------------
    # wv-sharing variant (for Pythia-era pin portability test)
    # ------------------------------------------------------------------

    def forward_wv_shared(self, token_ids: Tensor) -> Tensor:
        """Forward with L0's wv broadcast to L1-L3 (Pythia-era pin test).

        For GQA backbones (e.g. Llama-3.2-1B): L0's wv is (hidden, kv_dim).
        For non-GQA backbones (e.g. SmolLM2): L0's wv is (hidden, hidden).
        In both cases L0's V projection is used for all layers.

        Stage 1A/1B measured:
          SmolLM2-1.7B: cos=0.5126 → REFUTED
          Llama-3.2-1B: cos=0.9715 → UNCLEAR
        v200 uses each layer's own wv (conservative default per §4).
        """
        cfg = self._cfg
        B, T = token_ids.shape
        x = self.embed(token_ids)
        cos, sin = self._get_rope(T)

        wv_L0 = self.layers[0].attn.wv   # (hidden, kv_dim)
        nh    = cfg.num_heads
        nkv   = cfg.num_kv_heads
        hd    = cfg.head_dim

        for layer in self.layers:
            x_ln = layer.attn_ln(x)
            q = x_ln @ layer.attn.wq
            k = x_ln @ layer.attn.wk
            v = x_ln @ wv_L0             # <- L0's wv for all layers

            q = q.reshape(B, T, nh,  hd).transpose(1, 2)
            k = k.reshape(B, T, nkv, hd).transpose(1, 2)
            v = v.reshape(B, T, nkv, hd).transpose(1, 2)

            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

            if cfg.is_gqa:
                k = k.repeat((1, cfg.n_q_per_kv, 1, 1))
                v = v.repeat((1, cfg.n_q_per_kv, 1, 1))

            scale  = 1.0 / math.sqrt(hd)
            scores = (q @ k.transpose(-2, -1)) * scale
            scores = scores.clip(-1e4, 1e4)
            attn_w = scores.softmax(-1)
            out    = (attn_w @ v).transpose(1, 2).reshape(B, T, cfg.hidden_size)
            out    = out @ layer.attn.wo

            x_post_attn = x + out
            x = x_post_attn + layer.mlp.forward(layer.mlp_ln(x_post_attn))

        return self.ln_f(x)

    # ------------------------------------------------------------------
    # Latent forward (used by factor_graph_v200 THINK phase)
    # ------------------------------------------------------------------

    def forward_latents(self, z: Tensor) -> Tensor:
        """Forward latents (B, L, H) through L0-L3 self-attention + ln_f.

        Latents are (B, L=32, H=2048) — they pass through the backbone as if
        they were the residual stream of a length-L sequence. No token_ids
        embedding is performed; this is purely for the THINK phase.

        Returns (B, L, H).
        """
        L = z.shape[1]
        cos, sin = self._get_rope(L)
        for layer in self.layers:
            z = layer.forward(z, cos, sin)
        return self.ln_f(z)

    def forward_latents_with_taps(self, z: Tensor
                                   ) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
        """Same as forward_latents but with per-layer residual taps."""
        L = z.shape[1]
        cos, sin = self._get_rope(L)
        all_taps = []
        for layer in self.layers:
            z, taps = layer.forward_with_taps(z, cos, sin)
            all_taps.append(taps)
        return self.ln_f(z), all_taps

    def forward_latents_with_attn_weights(self, z: Tensor
                                           ) -> Tuple[Tensor, List[Tensor]]:
        """Same as forward_latents but collects per-layer self-attn weights.

        Returns (z_out, [attn_w_layer0, attn_w_layer1, attn_w_layer2, attn_w_layer3])
        Each attn_w_layerN: (B, n_heads, L, L) — softmax attention weights.
        Used by v200 to compute self-attention entropy per head for Gate B.
        """
        L = z.shape[1]
        cos, sin = self._get_rope(L)
        all_attn_weights = []
        for layer in self.layers:
            z, attn_w = layer.forward_with_attn_weights(z, cos, sin)
            all_attn_weights.append(attn_w)
        return self.ln_f(z), all_attn_weights

    def forward_latents_with_taps_and_attn_weights(
        self, z: Tensor
    ) -> Tuple[Tensor, List[Dict[str, Tensor]], List[Tensor]]:
        """Combined: per-layer residual taps AND per-layer self-attn weights.

        Returns (z_out, taps_per_layer, attn_weights_per_layer).
        Avoids the double-forward that separate calls to forward_latents_with_taps
        + forward_latents_with_attn_weights would require.

        taps_per_layer:  list of 4 dicts (pre_ln_resid, post_attn_resid, post_mlp_resid)
        attn_weights_per_layer: list of 4 Tensors (B, n_heads, L, L)
        """
        L = z.shape[1]
        cos, sin = self._get_rope(L)
        all_taps = []
        all_attn_weights = []
        for layer in self.layers:
            # Compute attn+taps in one pass
            pre_ln      = z
            x_ln        = layer.attn_ln(z)
            attn_out, attn_w = layer.attn.forward_return_weights(x_ln, cos, sin)
            x_post_attn = z + attn_out
            x_post_mlp  = x_post_attn + layer.mlp.forward(layer.mlp_ln(x_post_attn))
            z = x_post_mlp
            all_taps.append({
                "pre_ln_resid":    pre_ln,
                "post_attn_resid": x_post_attn,
                "post_mlp_resid":  x_post_mlp,
            })
            all_attn_weights.append(attn_w)
        return self.ln_f(z), all_taps, all_attn_weights
