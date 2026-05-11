"""Mycelium v4 breathing transformer.

Architecture matches Pythia-410M (GPT-NeoX) so weights load cleanly:
  - Standard 2-weight FFN (w_in -> GELU -> w_out), biases everywhere
  - Two LayerNorms per block (input_LN, post_attn_LN), parallel residual
  - Partial RoPE: only first rotary_dim=16 of head_dim=64 dims are rotated

Mycelium v4 modifications:
  - 4 phase-specific layers (RISE/PEAK/FALL/TROUGH) sharing V/O/FFN-out/LNs
  - Phase-specific Q, K, FFN-input projection (and biases)
  - π-cycled RoPE: per-head + per-loop phase offset applied to Q only
  - Sine-wave temperature modulation per phase
  - Gated running integral across loops (controller stub: gate=1)
"""
import math
import os
from typing import List

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.nn import Embedding
from tinygrad.engine.jit import TinyJit

from mycelium.config import Config
from mycelium.lookup_table import LookupTable
from mycelium.controller import Controller


# Ablation flags — read once at module import. Each disables one closed-loop
# component for Phase 2/3 directional screening. Default off: behavior matches
# the un-ablated architecture exactly when none are set.
ABLATE_TEMP        = int(os.environ.get("ABLATE_TEMP", "0")) > 0          # pin temperature multiplier to 1.0
ABLATE_STEP_MULT   = int(os.environ.get("ABLATE_STEP_MULT", "0")) > 0     # pin RoPE step_mult to 1.0
ABLATE_GATE        = int(os.environ.get("ABLATE_GATE", "0")) > 0          # pin integration gate to 1.0 (uniform breath weighting)
ABLATE_INTEGRATION = int(os.environ.get("ABLATE_INTEGRATION", "0")) > 0   # no cross-breath integral; last-breath-only
ABLATE_NOTEBOOK    = int(os.environ.get("ABLATE_NOTEBOOK", "0")) > 0      # clear notebook before every controller call
ABLATE_ROTATION    = int(os.environ.get("ABLATE_ROTATION", "0")) > 0      # uniform RoPE phase (no per-head / per-loop offset)

_active_ablations = [n for n, v in [
    ("TEMP", ABLATE_TEMP), ("STEP_MULT", ABLATE_STEP_MULT), ("GATE", ABLATE_GATE),
    ("INTEGRATION", ABLATE_INTEGRATION), ("NOTEBOOK", ABLATE_NOTEBOOK),
    ("ROTATION", ABLATE_ROTATION)] if v]
if _active_ablations:
    print(f"[ABLATE] active: {_active_ablations}", flush=True)


# ---------- partial RoPE with π cycling ----------

def _rope_base(seq_len: int, rotary_dim: int, base: int):
    """Half-rotation RoPE table for the rotated portion only.

    cos/sin: shape (seq_len, rotary_dim). The full head_dim is split into
    [rotated | unrotated]; only the first rotary_dim slots get the cos/sin.
    """
    half = rotary_dim // 2
    inv_freq = Tensor([1.0 / (base ** (2 * i / rotary_dim)) for i in range(half)], dtype=dtypes.float)
    pos = Tensor.arange(seq_len, dtype=dtypes.float)
    angles = pos.reshape(-1, 1) * inv_freq.reshape(1, -1)            # (seq, half)
    angles_full = Tensor.cat(angles, angles, dim=-1)                 # (seq, rotary_dim)
    return angles_full.cos().contiguous(), angles_full.sin().contiguous()


def _rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Half-rotation RoPE applied to the rotated slice only.

    x:        (..., rotary_dim)
    cos, sin: broadcast to x's shape
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = Tensor.cat(-x2, x1, dim=-1)
    return x * cos + rotated * sin


class RoPE:
    """π-cycled rotary position embedding, partial (Pythia-style 25%).

    Standard RoPE on both Q and K for relative position. Then Q gets an extra
    rotation by alpha(h, l) = h*pi/n_heads + l*pi/max_loops. Q-only application
    means q·k shifts by alpha (uniform offset on both would cancel).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        rd = cfg.rotary_dim
        cos, sin = _rope_base(cfg.max_seq_len, rd, cfg.rope_base)
        # Realize so the tables are real buffers (not lazy ops). Lazy ops here
        # tangle with autograd and silently swallow gradients on consumer tensors.
        self.cos = cos.reshape(1, 1, cfg.max_seq_len, rd).contiguous().realize()
        self.sin = sin.reshape(1, 1, cfg.max_seq_len, rd).contiguous().realize()

        head_phase = [h * math.pi / cfg.n_heads for h in range(cfg.n_heads)]
        if ABLATE_ROTATION:
            head_phase = [0.0] * cfg.n_heads  # no per-head phase diversity
        self.alpha_cos: List[Tensor] = []
        self.alpha_sin: List[Tensor] = []
        for l in range(cfg.max_loops):
            loop_phase = 0.0 if ABLATE_ROTATION else l * math.pi / cfg.max_loops
            alphas = [hp + loop_phase for hp in head_phase]
            ac = Tensor(alphas, dtype=dtypes.float).cos().reshape(1, cfg.n_heads, 1, 1).contiguous().realize()
            asn = Tensor(alphas, dtype=dtypes.float).sin().reshape(1, cfg.n_heads, 1, 1).contiguous().realize()
            self.alpha_cos.append(ac)
            self.alpha_sin.append(asn)

    def apply_at_tensor_pos(self, q: Tensor, k: Tensor, loop_idx: int, t_pos_t: Tensor):
        """Cached generation: S=1, t_pos as Tensor (shape () or (1,) for shared
        position across batch, or (B,) for per-batch positions). Builds the position-
        specific cos/sin via mask + sum over the full table.
        """
        cfg = self.cfg
        rd = cfg.rotary_dim
        max_len = int(self.cos.shape[2])
        # Determine if t_pos is per-batch or scalar
        per_batch = (t_pos_t.ndim == 1 and int(t_pos_t.shape[0]) > 1)
        B = int(t_pos_t.shape[0]) if per_batch else 1

        pos = Tensor.arange(max_len)
        if per_batch:
            # Per-batch position: mask shape (B, 1, max_len, 1)
            mask = (pos.reshape(1, max_len) == t_pos_t.reshape(B, 1)).reshape(B, 1, max_len, 1)
        else:
            mask = (pos == t_pos_t).reshape(1, 1, max_len, 1)
        cos_at = (self.cos * mask.cast(self.cos.dtype)).sum(axis=2, keepdim=True).cast(q.dtype)
        sin_at = (self.sin * mask.cast(self.sin.dtype)).sum(axis=2, keepdim=True).cast(q.dtype)

        q_rot, q_pass = q[..., :rd], q[..., rd:]
        k_rot, k_pass = k[..., :rd], k[..., rd:]
        q_rot = _rotate(q_rot, cos_at, sin_at)
        k_rot = _rotate(k_rot, cos_at, sin_at)

        ac = self.alpha_cos[loop_idx].cast(q.dtype)
        asn = self.alpha_sin[loop_idx].cast(q.dtype)
        q_rot = _rotate(q_rot, ac, asn)

        return Tensor.cat(q_rot, q_pass, dim=-1), Tensor.cat(k_rot, k_pass, dim=-1)

    def apply(self, q: Tensor, k: Tensor, loop_idx: int, start_pos: int = 0):
        """q, k: (B, n_heads, seq, head_dim). Rotate first rotary_dim slots, leave rest.

        start_pos: offset into the position table (for KV-cached generation, when
        the new token sits at position T_past, not 0).
        """
        S = q.shape[2]
        rd = self.cfg.rotary_dim
        cos = self.cos[:, :, start_pos:start_pos + S, :].cast(q.dtype)
        sin = self.sin[:, :, start_pos:start_pos + S, :].cast(q.dtype)

        q_rot, q_pass = q[..., :rd], q[..., rd:]
        k_rot, k_pass = k[..., :rd], k[..., rd:]

        q_rot = _rotate(q_rot, cos, sin)
        k_rot = _rotate(k_rot, cos, sin)

        # π-cycled phase offset on Q only (rotated portion)
        ac = self.alpha_cos[loop_idx].cast(q.dtype)
        asn = self.alpha_sin[loop_idx].cast(q.dtype)
        q_rot = _rotate(q_rot, ac, asn)

        q = Tensor.cat(q_rot, q_pass, dim=-1)
        k = Tensor.cat(k_rot, k_pass, dim=-1)
        return q, k


# ---------- weight initializers ----------

def _linear_w(in_dim: int, out_dim: int, dtype=dtypes.half) -> Tensor:
    """Pythia init: normal(0, 0.02). Stored (in, out) so x @ w needs no transpose."""
    return (Tensor.randn(in_dim, out_dim, dtype=dtype) * 0.02).contiguous()


def _bias(dim: int, dtype=dtypes.half) -> Tensor:
    return Tensor.zeros(dim, dtype=dtype).contiguous()


def _layernorm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    """LayerNorm with FP32 internal compute, casts output back to input dtype."""
    in_dt = x.dtype
    x32 = x.cast(dtypes.float)
    mean = x32.mean(axis=-1, keepdim=True)
    var = ((x32 - mean) ** 2).mean(axis=-1, keepdim=True)
    out = (x32 - mean) * (var + eps).rsqrt()
    return (out * gamma + beta).cast(in_dt)


# ---------- shared & phase-specific weights ----------

class SharedWeights:
    """Weights tied across all 4 phase layers — initialized from Pythia layer 0."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.wv = _linear_w(cfg.hidden, cfg.hidden)
        self.bv = _bias(cfg.hidden)
        self.wo = _linear_w(cfg.hidden, cfg.hidden)
        self.bo = _bias(cfg.hidden)
        self.w_out = _linear_w(cfg.ffn, cfg.hidden)             # FFN dense_4h_to_h
        self.b_out = _bias(cfg.hidden)
        # Pythia has separate input_layernorm and post_attention_layernorm (use_parallel_residual=True).
        self.in_ln_g = Tensor.ones(cfg.hidden, dtype=dtypes.float).contiguous()
        self.in_ln_b = Tensor.zeros(cfg.hidden, dtype=dtypes.float).contiguous()
        self.post_ln_g = Tensor.ones(cfg.hidden, dtype=dtypes.float).contiguous()
        self.post_ln_b = Tensor.zeros(cfg.hidden, dtype=dtypes.float).contiguous()

    def parameters(self):
        return [self.wv, self.bv, self.wo, self.bo, self.w_out, self.b_out,
                self.in_ln_g, self.in_ln_b, self.post_ln_g, self.post_ln_b]


class BreathingLayer:
    """One phase of the breath — RISE, PEAK, FALL, or TROUGH.

    Phase-specific: Q, K, FFN-in (and their biases). Shared: V, O, FFN-out, LNs.
    Temperature: T = exp(temp_amp * sin(phase)); divides QK^T inside softmax.
    """

    def __init__(self, cfg: Config, phase: float, shared: SharedWeights, rope: RoPE):
        self.cfg = cfg
        self.phase = phase
        self.shared = shared
        self.rope = rope

        # Phase-specific
        self.wq = _linear_w(cfg.hidden, cfg.hidden)
        self.bq = _bias(cfg.hidden)
        self.wk = _linear_w(cfg.hidden, cfg.hidden)
        self.bk = _bias(cfg.hidden)
        self.w_in = _linear_w(cfg.hidden, cfg.ffn)              # FFN dense_h_to_4h
        self.b_in = _bias(cfg.ffn)

        # Sine-wave temperature: T = exp(amp * sin(phase))
        self.temperature = math.exp(cfg.temp_amp * math.sin(phase))
        self.attn_scale = 1.0 / (math.sqrt(cfg.head_dim) * self.temperature)

    def parameters(self):
        return [self.wq, self.bq, self.wk, self.bk, self.w_in, self.b_in]

    def __call__(self, x: Tensor, loop_idx: int, attn_mask: Tensor | None = None,
                 temp_mult: Tensor | float = 1.0) -> Tensor:
        return self._forward(x, loop_idx, kv_cache=None, return_kv=False,
                             attn_mask=attn_mask, temp_mult=temp_mult)[0]

    def forward_with_kv(self, x: Tensor, loop_idx: int, attn_mask: Tensor | None = None):
        """Full-sequence forward that also returns the post-RoPE K, V tensors.

        attn_mask: optional (B, S) bool/{0,1} tensor — 1 for valid, 0 for padding.
        When provided, padding positions don't influence attention (added as -inf to
        scores) and don't get gradient signal.
        """
        return self._forward(x, loop_idx, kv_cache=None, return_kv=True, attn_mask=attn_mask)

    def forward_cached_step(self, x_new: Tensor, loop_idx: int, kv_cache):
        """Single-token (S=1) forward with cached past K/V. Returns (out, (k_full, v_full))."""
        return self._forward(x_new, loop_idx, kv_cache=kv_cache, return_kv=True)

    def forward_cached_step_batched(self, x_new: Tensor, loop_idx: int,
                                    k_buf: Tensor, v_buf: Tensor, t_pos_t: Tensor,
                                    prompt_mask: Tensor | None = None):
        """Batched single-token cached forward.

        x_new:        (B, 1, H)
        k_buf, v_buf: (B, n_heads, max_seq_len, head_dim) — buffers shared across batch
        t_pos_t:      0-dim or shape (1,) Tensor — uniform write position
        prompt_mask:  (B, max_seq_len) — 1 where the cache position holds a valid (non-pad)
                      prompt token. The new-token slot itself is added to the valid mask
                      via the causal `pos <= t_pos_t` comparison (no need to update the
                      prompt_mask between calls because future slots are zero by default
                      and t_pos_t monotonically advances).
        """
        cfg = self.cfg
        max_len = int(k_buf.shape[2])
        B = int(x_new.shape[0])

        attn_in = _layernorm(x_new, self.shared.in_ln_g, self.shared.in_ln_b, cfg.layer_norm_eps)
        mlp_in = _layernorm(x_new, self.shared.post_ln_g, self.shared.post_ln_b, cfg.layer_norm_eps)
        attn_in_dt = attn_in.cast(x_new.dtype) if attn_in.dtype != x_new.dtype else attn_in
        mlp_in_dt = mlp_in.cast(x_new.dtype) if mlp_in.dtype != x_new.dtype else mlp_in

        q_new = (attn_in_dt @ self.wq + self.bq).reshape(B, 1, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k_new = (attn_in_dt @ self.wk + self.bk).reshape(B, 1, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        v_new = (attn_in_dt @ self.shared.wv + self.shared.bv).reshape(B, 1, cfg.n_heads, cfg.head_dim).transpose(1, 2)

        q_new, k_new = self.rope.apply_at_tensor_pos(q_new, k_new, loop_idx, t_pos_t)

        pos = Tensor.arange(max_len)
        per_batch = (t_pos_t.ndim == 1 and int(t_pos_t.shape[0]) > 1)
        if per_batch:
            write_at = (pos.reshape(1, max_len) == t_pos_t.reshape(B, 1)).reshape(B, 1, max_len, 1)
            causal = (pos.reshape(1, max_len) <= t_pos_t.reshape(B, 1)).reshape(B, 1, 1, max_len)
        else:
            write_at = (pos == t_pos_t).reshape(1, 1, max_len, 1)
            causal = (pos <= t_pos_t).reshape(1, 1, 1, max_len)
        k_new_b = k_new.expand(B, cfg.n_heads, max_len, cfg.head_dim)
        v_new_b = v_new.expand(B, cfg.n_heads, max_len, cfg.head_dim)
        k_buf_new = write_at.where(k_new_b, k_buf)
        v_buf_new = write_at.where(v_new_b, v_buf)

        # Per-batch t_pos already excludes Phase-A padding via causal mask (we only
        # attend to positions < t_pos_t[b], and shorter prompts have their first
        # generated token overwrite the Phase-A padding before any later token tries
        # to attend to it). prompt_mask kept as optional knob for caller-provided masking.
        if prompt_mask is not None:
            pmask = prompt_mask.reshape(B, 1, 1, max_len).cast(dtypes.bool)
            valid = causal & pmask
        else:
            valid = causal

        scores = q_new @ k_buf_new.transpose(-2, -1) * self.attn_scale
        scores = valid.where(scores, Tensor(-float("inf"), dtype=scores.dtype))
        attn = scores.softmax(-1).cast(v_buf_new.dtype)
        ctx = (attn @ v_buf_new).transpose(1, 2).reshape(B, 1, cfg.hidden)
        attn_out = ctx @ self.shared.wo + self.shared.bo

        ff = (mlp_in_dt @ self.w_in + self.b_in).gelu()
        ffn_out = ff @ self.shared.w_out + self.shared.b_out

        out = x_new + attn_out + ffn_out
        return out, k_buf_new, v_buf_new

    def forward_cached_step_jit(self, x_new: Tensor, loop_idx: int,
                                k_buf: Tensor, v_buf: Tensor, t_pos_t: Tensor):
        """Single-token cached forward, t_pos as Tensor scalar (JIT-replay friendly).
        Same compute as forward_cached_step_fixed but no Python-int slicing — every op
        has stable shape so a single TinyJit graph handles all positions."""
        cfg = self.cfg
        max_len = int(k_buf.shape[2])
        attn_dtype = x_new.dtype

        attn_in = _layernorm(x_new, self.shared.in_ln_g, self.shared.in_ln_b, cfg.layer_norm_eps)
        mlp_in = _layernorm(x_new, self.shared.post_ln_g, self.shared.post_ln_b, cfg.layer_norm_eps)
        attn_in_dt = attn_in.cast(attn_dtype) if attn_in.dtype != attn_dtype else attn_in
        mlp_in_dt = mlp_in.cast(attn_dtype) if mlp_in.dtype != attn_dtype else mlp_in

        q_new = (attn_in_dt @ self.wq + self.bq).reshape(1, 1, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k_new = (attn_in_dt @ self.wk + self.bk).reshape(1, 1, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        v_new = (attn_in_dt @ self.shared.wv + self.shared.bv).reshape(1, 1, cfg.n_heads, cfg.head_dim).transpose(1, 2)

        q_new, k_new = self.rope.apply_at_tensor_pos(q_new, k_new, loop_idx, t_pos_t)

        pos = Tensor.arange(max_len)
        write_at = (pos == t_pos_t).reshape(1, 1, max_len, 1)
        k_new_b = k_new.expand(1, cfg.n_heads, max_len, cfg.head_dim)
        v_new_b = v_new.expand(1, cfg.n_heads, max_len, cfg.head_dim)
        k_buf_new = write_at.where(k_new_b, k_buf)
        v_buf_new = write_at.where(v_new_b, v_buf)

        valid = (pos <= t_pos_t).reshape(1, 1, 1, max_len)
        scores = q_new @ k_buf_new.transpose(-2, -1) * self.attn_scale
        scores = valid.where(scores, Tensor(-float("inf"), dtype=scores.dtype))
        attn = scores.softmax(-1).cast(v_buf_new.dtype)
        ctx = (attn @ v_buf_new).transpose(1, 2).reshape(1, 1, cfg.hidden)
        attn_out = ctx @ self.shared.wo + self.shared.bo

        ff = (mlp_in_dt @ self.w_in + self.b_in).gelu()
        ffn_out = ff @ self.shared.w_out + self.shared.b_out

        out = x_new + attn_out + ffn_out
        return out, k_buf_new, v_buf_new

    def forward_cached_step_fixed(self, x_new: Tensor, loop_idx: int,
                                  k_buf: Tensor, v_buf: Tensor, t_pos):
        """Single-token forward with FIXED-SHAPE K/V buffers + position-mask attention.

        t_pos may be a Python int or a 0-dim Tensor (the latter is required for
        TinyJit replay to handle multiple positions with one compiled graph).
        """
        cfg = self.cfg
        max_len = int(k_buf.shape[2])
        attn_dtype = x_new.dtype

        attn_in = _layernorm(x_new, self.shared.in_ln_g, self.shared.in_ln_b, cfg.layer_norm_eps)
        mlp_in = _layernorm(x_new, self.shared.post_ln_g, self.shared.post_ln_b, cfg.layer_norm_eps)
        attn_in_dt = attn_in.cast(attn_dtype) if attn_in.dtype != attn_dtype else attn_in
        mlp_in_dt = mlp_in.cast(attn_dtype) if mlp_in.dtype != attn_dtype else mlp_in

        q_new = (attn_in_dt @ self.wq + self.bq).reshape(1, 1, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k_new = (attn_in_dt @ self.wk + self.bk).reshape(1, 1, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        v_new = (attn_in_dt @ self.shared.wv + self.shared.bv).reshape(1, 1, cfg.n_heads, cfg.head_dim).transpose(1, 2)

        # NOTE: For now t_pos is treated as a Python int (RoPE uses .start_pos slicing
        # which must resolve at trace time). When t_pos is a Tensor, this path will
        # need to use full-length cos/sin tables and apply with masking. For initial
        # JIT smoke we keep it as int and accept N JIT compiles for first N positions.
        q_new, k_new = self.rope.apply(q_new, k_new, loop_idx, start_pos=t_pos)

        pos = Tensor.arange(max_len, dtype=dtypes.int)
        write_at = (pos == t_pos).reshape(1, 1, max_len, 1)
        k_new_b = k_new.expand(1, cfg.n_heads, max_len, cfg.head_dim)
        v_new_b = v_new.expand(1, cfg.n_heads, max_len, cfg.head_dim)
        k_buf_new = write_at.where(k_new_b, k_buf)
        v_buf_new = write_at.where(v_new_b, v_buf)

        valid = (pos <= t_pos).reshape(1, 1, 1, max_len)
        scores = q_new @ k_buf_new.transpose(-2, -1) * self.attn_scale
        scores = valid.where(scores, Tensor(-float("inf"), dtype=scores.dtype))
        attn = scores.softmax(-1).cast(v_buf_new.dtype)
        ctx = (attn @ v_buf_new).transpose(1, 2).reshape(1, 1, cfg.hidden)
        attn_out = ctx @ self.shared.wo + self.shared.bo

        ff = (mlp_in_dt @ self.w_in + self.b_in).gelu()
        ffn_out = ff @ self.shared.w_out + self.shared.b_out

        out = x_new + attn_out + ffn_out
        return out, k_buf_new, v_buf_new

    def _forward(self, x: Tensor, loop_idx: int, kv_cache, return_kv: bool,
                 attn_mask: Tensor | None = None, temp_mult: Tensor | float = 1.0):
        cfg = self.cfg
        B, S, H = x.shape

        attn_in = _layernorm(x, self.shared.in_ln_g, self.shared.in_ln_b, cfg.layer_norm_eps)
        mlp_in = _layernorm(x, self.shared.post_ln_g, self.shared.post_ln_b, cfg.layer_norm_eps)
        attn_in_dt = attn_in.cast(x.dtype) if attn_in.dtype != x.dtype else attn_in
        mlp_in_dt = mlp_in.cast(x.dtype) if mlp_in.dtype != x.dtype else mlp_in

        q = (attn_in_dt @ self.wq + self.bq).reshape(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = (attn_in_dt @ self.wk + self.bk).reshape(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        v = (attn_in_dt @ self.shared.wv + self.shared.bv).reshape(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)

        # Adaptive temperature: built-in sine schedule × 1/temp_mult (higher mult → softer attention).
        # When temp_mult=1.0 (default), behavior is identical to the original fixed sine schedule.
        if isinstance(temp_mult, Tensor):
            scale = self.attn_scale * (1.0 / temp_mult).cast(q.dtype).reshape(B, 1, 1, 1)
        else:
            scale = self.attn_scale / float(temp_mult)

        if kv_cache is None:
            q, k = self.rope.apply(q, k, loop_idx, start_pos=0)
            scores = q @ k.transpose(-2, -1) * scale
            mask = Tensor.ones(S, S, dtype=scores.dtype).tril().reshape(1, 1, S, S)
            scores = scores.masked_fill(mask == 0, float("-inf"))
            if attn_mask is not None:
                # attn_mask shape: (B, S) — 1 valid, 0 padding. Broadcast to (B, 1, 1, S).
                key_mask = attn_mask.reshape(B, 1, 1, S).cast(dtypes.bool)
                scores = key_mask.where(scores, Tensor(-float("inf"), dtype=scores.dtype))
            attn = scores.softmax(-1).cast(v.dtype)
            ctx = (attn @ v).transpose(1, 2).reshape(B, S, H)
            attn_out = ctx @ self.shared.wo + self.shared.bo
            ff = (mlp_in_dt @ self.w_in + self.b_in).gelu()
            ffn_out = ff @ self.shared.w_out + self.shared.b_out
            out = x + attn_out + ffn_out
            return (out, (k, v)) if return_kv else (out, None)

        # Cached path — single new token (S==1) attending over (cached past + itself).
        k_past, v_past = kv_cache
        t_past = int(k_past.shape[2])
        q, k = self.rope.apply(q, k, loop_idx, start_pos=t_past)
        k_full = Tensor.cat(k_past, k, dim=2)        # (B, n_heads, T_past+1, head_dim)
        v_full = Tensor.cat(v_past, v, dim=2)
        # No causal mask: new token can attend to all past + itself.
        scores = q @ k_full.transpose(-2, -1) * scale
        attn = scores.softmax(-1).cast(v_full.dtype)
        ctx = (attn @ v_full).transpose(1, 2).reshape(B, S, H)
        attn_out = ctx @ self.shared.wo + self.shared.bo
        ff = (mlp_in_dt @ self.w_in + self.b_in).gelu()
        ffn_out = ff @ self.shared.w_out + self.shared.b_out
        out = x + attn_out + ffn_out
        return out, (k_full, v_full)


# ---------- block: 4 phases × N loops + integration ----------

class BreathingBlock:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.shared = SharedWeights(cfg)
        self.rope = RoPE(cfg)
        # Phases trace one full sine wave: 0, π/2, π, 3π/2.
        phases = [i * 2 * math.pi / cfg.n_phases for i in range(cfg.n_phases)]
        self.layers = [BreathingLayer(cfg, ph, self.shared, self.rope) for ph in phases]

    def parameters(self):
        ps = list(self.shared.parameters())
        for layer in self.layers:
            ps.extend(layer.parameters())
        return ps

    def breathe_once(self, x: Tensor, loop_idx: int) -> Tensor:
        for layer in self.layers:
            x = layer(x, loop_idx)
        return x

    def breathe(self, x: Tensor, n_loops: int) -> Tensor:
        """Loop the 4-layer breath n_loops times with gated running integral.

        Stub controller: gate=1 every breath, always continue. Returns the normalized
        integral (running mean) of all breath outputs.
        """
        integral = Tensor.zeros_like(x)
        gate_sum = 0.0
        for l in range(n_loops):
            x = self.breathe_once(x, l)
            integral = integral + x
            gate_sum += 1.0
        return integral / gate_sum


# ---------- top-level model ----------

class BreathingTransformer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.embed = Embedding(cfg.vocab_size, cfg.hidden)
        self.block = BreathingBlock(cfg)
        # Final LN on the integrated representation.
        self.ln_f_g = Tensor.ones(cfg.hidden, dtype=dtypes.float).contiguous()
        self.ln_f_b = Tensor.zeros(cfg.hidden, dtype=dtypes.float).contiguous()
        # Output head (untied — Pythia has separate embed_out weight, no bias).
        self.embed_out = _linear_w(cfg.hidden, cfg.vocab_size)
        # Closed-loop component #4: prime-operation lookup table. 16 entries × 1024d.
        # Random orthogonal init; joint-trained via aux op-classification CE so the
        # entries align with the model's actual operation directions.
        self.lookup_table = LookupTable(n_entries=cfg.n_lookup_entries,
                                        hidden=cfg.hidden,
                                        seed=cfg.seed_lookup)
        # Closed-loop component #5: the controller. State reader + decision heads.
        # Step B scaffold; notebook (Step C) and adaptive wiring (Step D) follow.
        self.controller = Controller(cfg)

    def parameters(self):
        """Parameters trained on the main loss (transformer + lookup table).
        The controller has gradient separation per the spec — its parameters
        are returned by controller_parameters() and trained by a separate
        optimizer (Step F) via REINFORCE on outcomes + auxiliary signals."""
        return ([self.embed.weight, self.ln_f_g, self.ln_f_b, self.embed_out]
                + self.block.parameters()
                + self.lookup_table.parameters())

    def controller_parameters(self):
        """Controller-only parameters. Trained via a separate optimizer with a
        non-overlapping signal (gradient separation enforced by construction —
        the main loss never reaches these params)."""
        return self.controller.parameters()

    def state_dict(self) -> dict:
        """Single source of truth for ckpt save/load. New components register here."""
        sd = {
            "embed.weight": self.embed.weight,
            "embed_out": self.embed_out,
            "ln_f.g": self.ln_f_g,
            "ln_f.b": self.ln_f_b,
            "lookup_table.weight": self.lookup_table.weight,
        }
        sw = self.block.shared
        for a in ("wv", "bv", "wo", "bo", "w_out", "b_out",
                 "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b"):
            sd[f"shared.{a}"] = getattr(sw, a)
        for i, layer in enumerate(self.block.layers):
            for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
                sd[f"phase{i}.{a}"] = getattr(layer, a)
        sd.update(self.controller.state_dict())
        return sd

    def load_state_dict(self, sd_ck: dict, strict: bool = False) -> dict:
        """Load tensors from sd_ck into the model. With strict=False, missing keys
        in sd_ck are skipped (current weights kept) — important for resuming from
        ckpts saved before new components (e.g., lookup_table) existed.

        Returns a dict {missing: [...], unexpected: [...]} for visibility.
        """
        from tinygrad import Device
        targets = self.state_dict()
        missing, unexpected = [], []
        for name, dst in targets.items():
            if name not in sd_ck:
                missing.append(name)
                if strict:
                    raise KeyError(f"missing key in checkpoint: {name}")
                continue
            src = sd_ck[name].to(dst.device).realize()
            if src.shape != dst.shape: src = src.reshape(dst.shape)
            if src.dtype != dst.dtype: src = src.cast(dst.dtype)
            dst.assign(src).realize()
        for name in sd_ck:
            if name not in targets:
                unexpected.append(name)
        Device[Device.DEFAULT].synchronize()
        return {"missing": missing, "unexpected": unexpected}

    def hidden_states(self, tokens: Tensor, n_loops: int, return_per_loop: bool = False):
        """Forward pass returning hidden states. If return_per_loop, returns a list of
        states after each breath (length n_loops+1, including the embedded input).
        Otherwise returns the final integrated representation (post final LN).
        """
        x = self.embed(tokens).cast(dtypes.half)
        if not return_per_loop:
            x = self.block.breathe(x, n_loops)
            return _layernorm(x, self.ln_f_g, self.ln_f_b, self.cfg.layer_norm_eps)
        # explicit loop for diagnostics
        states = [x]
        integral = Tensor.zeros_like(x)
        gate_sum = 0.0
        for l in range(n_loops):
            x = self.block.breathe_once(x, l)
            states.append(x)
            integral = integral + x
            gate_sum += 1.0
        final = _layernorm(integral / gate_sum, self.ln_f_g, self.ln_f_b, self.cfg.layer_norm_eps)
        return states, final

    def __call__(self, tokens: Tensor, n_loops: int) -> Tensor:
        return self.hidden_states(tokens, n_loops, return_per_loop=False)

    def breathe_with_lookup_jit(self, tokens: Tensor, n_loops: int):
        """JIT-cached version of breathe_with_lookup, returning only (final_hidden,
        last_breath_match_weights). The per-breath lists aren't returned because
        TinyJit doesn't return Python lists; we keep only what the training step
        actually consumes (the LAST breath's match weights, used by the aux CE).

        Cached per n_loops in self._jit_breathe_forwards. First call at each
        n_loops compiles (~30-60s); subsequent calls replay as a single graph.

        Bypasses the per-step py_overhead growth we observed (870ms → 2274ms in
        100 steps without JIT). The JIT replays a fixed-shape compiled graph,
        so no lazy Python state accumulates between calls.
        """
        n_loops = int(n_loops)
        if not hasattr(self, "_jit_breathe_forwards"):
            self._jit_breathe_forwards = {}
        if n_loops not in self._jit_breathe_forwards:
            n_loops_captured = n_loops

            @TinyJit
            def _fwd(toks: Tensor):
                final, match_weights, _ = self.breathe_with_lookup(toks, n_loops_captured)
                return final, match_weights[-1]

            self._jit_breathe_forwards[n_loops] = _fwd
        return self._jit_breathe_forwards[n_loops](tokens)

    def call_jit(self, tokens: Tensor, n_loops: int) -> Tensor:
        """JIT-cached version of __call__ (the plain forward without lookup table
        per-breath queries). Used for cycles that don't need the aux loss.
        Cached per n_loops in self._jit_forwards."""
        n_loops = int(n_loops)
        if not hasattr(self, "_jit_forwards"):
            self._jit_forwards = {}
        if n_loops not in self._jit_forwards:
            n_loops_captured = n_loops

            @TinyJit
            def _fwd(toks: Tensor):
                return self(toks, n_loops_captured)

            self._jit_forwards[n_loops] = _fwd
        return self._jit_forwards[n_loops](tokens)

    def breathe_controlled(self, tokens: Tensor, max_loops: int, notebook,
                           rep_position: int = -1, detach_rep_for_ctrl: bool = True,
                           detach_decisions_into_transformer: bool = False,
                           adaptive: bool = False, min_loops: int = 1):
        """Closed-loop adaptive breathing — the full 7/7 system in action.

        Per breath:
          1. Run the 4 layer-passes at the current temperature (multiplier from controller).
          2. Add this breath's contribution to the running integral, weighted by gate.
          3. Read the integrated rep at rep_position (the 'controller's eyes').
          4. Run the controller(rep, notebook) → page is appended to notebook,
             attention reads tree of prior pages, decision heads emit
             {temperature, gate, stop_logit} for the NEXT breath.
          5. If adaptive=True and l+1 >= min_loops and mean(stop_logit) > 0, break.

        Adaptive stopping (inference-only by default — training keeps adaptive=False
        so loss computation is straightforward over a fixed unrolled loop):
          - adaptive=True: after each breath's controller call, halt if the batch-mean
            stop_logit crosses zero. Adds one .numpy() sync per breath; cheap at
            inference. Each breath has access to the stop_logit it emitted, so the
            controller can learn (via compute-penalty) to halt early on easy problems.
          - min_loops: guarantees at least this many breaths run before early-stop is
            considered. Default 1 — the controller can't bail at breath 0.

        Gradient separation:
          - detach_rep_for_ctrl=True: the rep fed into the controller is detached,
            so the controller's loss can't update transformer params.
          - detach_decisions_into_transformer=False: the controller's outputs
            (temperature, gate) flow into the transformer's computation WITH
            gradient. This is correct for controller training (controller learns
            from how its decisions affected the transformer's behavior). For
            main-loss training of the transformer, set True so transformer
            gradient doesn't update controller params.

        Returns:
          final_hidden: (B, T, hidden) post final LN — the same surface as __call__
          decisions:    list of dicts (one per breath taken)
          n_breaths:    int — actual number of breaths run (≤ max_loops)
          match_weights: list of (B, T, n_entries) per-breath lookup table queries
        """
        cfg = self.cfg
        x = self.embed(tokens).cast(dtypes.half)
        B = int(x.shape[0])
        notebook.clear()

        integral = Tensor.zeros_like(x)
        gate_total = Tensor.zeros((B,), dtype=dtypes.float).realize()
        decisions_per_breath = []
        match_weights = []

        # Initial decisions (from raw input) — controller's "first look" before any breathing
        rep = x[:, rep_position, :].cast(dtypes.float)
        if detach_rep_for_ctrl:
            rep = rep.detach()
        decisions = self.controller(rep, notebook=notebook)
        decisions_per_breath.append(decisions)

        # Adaptive phase index — accumulated as a float across breaths. RoPE table
        # is integer-indexed, so we round and clamp into [0, max_loops-1] when
        # querying. Uniform default (step_mult=1.0) reproduces the existing
        # 0,1,2,...,max_loops-1 sequence exactly.
        phase_idx_float = 0.0
        actual_n_breaths = max_loops

        for l in range(max_loops):
            temp_mult = decisions["temperature"]                      # (B,)
            gate = decisions["gate"]                                  # (B,)
            step_mult = decisions["step_mult"]                        # (B,)
            if ABLATE_TEMP:
                # Pin to 1.0 but keep gradient graph connected (zero grad flows
                # back through the controller's temperature head). Replacing with
                # Tensor.ones_like severs the path and breaks ctrl_opt.step.
                temp_mult = temp_mult * 0.0 + 1.0
            if ABLATE_STEP_MULT:
                step_mult = step_mult * 0.0 + 1.0
            if ABLATE_GATE:
                gate = gate * 0.0 + 1.0
            if detach_decisions_into_transformer:
                temp_mult = temp_mult.detach()
                gate = gate.detach()
                step_mult = step_mult.detach()

            # Per-batch step_mult is averaged across the batch for the (shared)
            # RoPE phase index. Per-batch fractional indexing would require
            # interpolating cos/sin tables; this is the simpler v1.
            step_avg = float(step_mult.mean().realize().numpy())
            current_loop_idx = max(0, min(int(round(phase_idx_float)), cfg.max_loops - 1))

            # Run the 4-layer breath at this breath's temperature + adaptive phase
            for layer in self.block.layers:
                x = layer(x, current_loop_idx, temp_mult=temp_mult)
            phase_idx_float += step_avg

            # Add to integral with gate weighting.
            # Ablation: when ABLATE_INTEGRATION, overwrite instead of accumulate
            # (last-breath-only — no cross-breath memory in the integral path).
            if ABLATE_INTEGRATION:
                integral = x.cast(dtypes.float) * gate.cast(dtypes.float).reshape(B, 1, 1)
                gate_total = gate.cast(dtypes.float)
            else:
                integral = integral + x.cast(dtypes.float) * gate.cast(dtypes.float).reshape(B, 1, 1)
                gate_total = gate_total + gate.cast(dtypes.float)

            # Per-breath lookup-table query against the running integral, normalized
            running = integral / (gate_total + 1e-6).reshape(B, 1, 1)
            running_normed = _layernorm(running, self.ln_f_g, self.ln_f_b, cfg.layer_norm_eps)
            match_weights.append(self.lookup_table(running_normed))

            # Controller reads the running integral and emits decisions for next breath.
            # Ablation: when ABLATE_NOTEBOOK, clear notebook before each call so the
            # controller never sees prior-breath pages (no cross-breath memory).
            rep = running_normed[:, rep_position, :].cast(dtypes.float)
            if detach_rep_for_ctrl:
                rep = rep.detach()
            if ABLATE_NOTEBOOK:
                notebook.clear()
            decisions = self.controller(rep, notebook=notebook)
            decisions_per_breath.append(decisions)

            # Adaptive early-stop: after we've run at least min_loops breaths, halt
            # when the controller's batch-mean stop_logit crosses zero. One sync per
            # breath; only enabled at inference.
            if adaptive and (l + 1) >= min_loops:
                stop_mean = float(decisions["stop_logit"].mean().realize().numpy())
                if stop_mean > 0.0:
                    actual_n_breaths = l + 1
                    break

        # Final integrated rep: gate-weighted mean
        final = integral / (gate_total + 1e-6).reshape(B, 1, 1)
        final = _layernorm(final, self.ln_f_g, self.ln_f_b, cfg.layer_norm_eps)
        return final, decisions_per_breath, actual_n_breaths, match_weights

    def breathe_with_lookup(self, tokens: Tensor, n_loops: int):
        """Forward pass returning (final_hidden, per_breath_match_weights, integrated_per_breath).

        Queries the model's lookup table once per breath against the running integral
        normalized to date. Returned shapes:
          final_hidden:                (B, T, hidden)         — same as __call__
          per_breath_match_weights:    list of n_loops × (B, T, n_entries)
          integrated_per_breath:       list of n_loops × (B, T, hidden) post-LN

        Used by the controller to read per-breath operation matches and by training
        to apply auxiliary lookup-CE loss against ground-truth op labels.
        """
        x = self.embed(tokens).cast(dtypes.half)
        integral = Tensor.zeros_like(x)
        match_weights = []
        integrated_per_breath = []
        for l in range(n_loops):
            x = self.block.breathe_once(x, l)
            integral = integral + x
            running = integral / float(l + 1)
            running_normed = _layernorm(running, self.ln_f_g, self.ln_f_b, self.cfg.layer_norm_eps)
            integrated_per_breath.append(running_normed)
            match_weights.append(self.lookup_table(running_normed))
        # Final hidden — bit-for-bit equal to __call__'s output
        final = _layernorm(integral / float(n_loops),
                           self.ln_f_g, self.ln_f_b, self.cfg.layer_norm_eps)
        return final, match_weights, integrated_per_breath

    def cached_generate_batch(self, batch_prompt_ids: list, n_loops: int, max_new: int,
                               stop_token_ids=None, stop_seq=None,
                               vocab_active: int = 50277,
                               cache_max_len: int | None = None) -> list:
        """Batched cached generation. Processes B prompts in parallel with shared
        K/V cache buffers.

        cache_max_len: dimension of the K/V buffers (defaults to cfg.max_seq_len).
        For short generations (e.g. L3-spaced arithmetic with ~30-token sequences),
        passing a smaller value (e.g. 32) reduces cache size 16× and unlocks much
        larger batch sizes within the GPU's memory budget. Must be >= max_prompt
        + max_new.
        """
        from tinygrad import Tensor as _T
        cfg = self.cfg
        n_layers = cfg.n_phases
        B = len(batch_prompt_ids)
        stop_set = set(stop_token_ids or [])
        seq = list(stop_seq or [])
        seq_len = len(seq)

        real_lens = [len(p) for p in batch_prompt_ids]
        max_prompt = max(real_lens)
        # cache_max_len defaults to model max but can be much smaller for short gens
        if cache_max_len is None:
            cache_max_len = cfg.max_seq_len
        assert max_prompt + max_new <= cache_max_len, (
            f"cache_max_len={cache_max_len} too small for prompt={max_prompt} + new={max_new}"
        )
        assert cache_max_len <= cfg.max_seq_len, "cache_max_len cannot exceed RoPE table size"
        max_len = cache_max_len

        # Right-pad prompts to max_prompt with PAD=0
        padded = [p + [0] * (max_prompt - len(p)) for p in batch_prompt_ids]
        prompts_t = _T(padded, dtype=dtypes.int).realize()

        # Phase A attention mask: 1 for real prompt positions [0..real_len), 0 for padding.
        prompt_attn_mask_phase_a_np = np.zeros((B, max_prompt), dtype=np.int32)
        for b, rl in enumerate(real_lens):
            prompt_attn_mask_phase_a_np[b, :rl] = 1
        attn_mask_phase_a = _T(prompt_attn_mask_phase_a_np, dtype=dtypes.int).realize()

        # gen_mask is unused now (per-batch t_pos handles correctness) but kept for compat
        gen_mask_np = np.ones((B, max_len), dtype=np.int32)
        gen_attn_mask = _T(gen_mask_np, dtype=dtypes.int).contiguous().realize()

        # ---- Stage 1: Phase A breathing on padded prompts ----
        # Pass attn_mask only when there's actual padding (mixed prompt lengths). For
        # uniform-length batches the mask is all-1s and we skip it to keep the code path
        # identical to the uncached forward — easier correctness verification.
        same_len = all(rl == real_lens[0] for rl in real_lens)
        attn_mask_arg = None if same_len else attn_mask_phase_a

        x = self.embed(prompts_t).cast(dtypes.half)
        integral = Tensor.zeros_like(x)
        cache_k = [[None] * n_loops for _ in range(n_layers)]
        cache_v = [[None] * n_loops for _ in range(n_layers)]
        for loop in range(n_loops):
            for li, layer in enumerate(self.block.layers):
                x, (k_part, v_part) = layer.forward_with_kv(x, loop_idx=loop, attn_mask=attn_mask_arg)
                pad_n = max_len - max_prompt
                k_full = k_part.pad(((0, 0), (0, 0), (0, pad_n), (0, 0))).contiguous().realize()
                v_full = v_part.pad(((0, 0), (0, 0), (0, pad_n), (0, 0))).contiguous().realize()
                cache_k[li][loop] = k_full
                cache_v[li][loop] = v_full
            integral = integral + x
        integrated_rep = (integral / float(n_loops)).realize()

        # First token: per-batch gather at real_lens[i] - 1.
        # Build per-batch index: pos == (real_len - 1)
        pos_arange = Tensor.arange(max_prompt).reshape(1, max_prompt, 1)
        last_idx = (Tensor(real_lens, dtype=dtypes.int).reshape(B, 1, 1) - 1)
        last_mask = (pos_arange == last_idx).cast(dtypes.half)  # (B, max_prompt, 1)
        h_at_last = (integrated_rep * last_mask).sum(axis=1, keepdim=True)  # (B, 1, H)
        h_normed = _layernorm(h_at_last, self.ln_f_g, self.ln_f_b, cfg.layer_norm_eps)
        logits = (h_normed @ self.embed_out).cast(dtypes.float)
        logits = logits[:, :, :vocab_active]
        first_ids = logits.argmax(axis=-1).realize().numpy().reshape(B)
        outs = [[int(first_ids[b])] for b in range(B)]
        is_done = [False] * B
        for b in range(B):
            if outs[b][0] in stop_set:
                is_done[b] = True
            elif seq_len > 0 and outs[b][-seq_len:] == seq:
                is_done[b] = True
        if all(is_done):
            return outs

        # ---- Stage 2: batched per-token generation, JIT-fused ----
        # JIT body: embed(prev_id) → breathing → argmax → next_id. Both embed and
        # argmax inside JIT cuts ~2 kernel launches per step and lets the compiler
        # fuse the embedding lookup with the first matmul + the final logit projection
        # with the argmax reduction.
        #
        # JITs are cached in a dict keyed on (B, n_loops, vocab_active) so a typical
        # eval that sweeps multiple loop counts (EVAL_LOOPS=[1,2,4,8]) compiles each
        # graph once at the first eval and replays them at zero compile cost on every
        # subsequent eval cycle.
        if not hasattr(self, "_cached_batch_jits"):
            self._cached_batch_jits = {}
        jit_key = (B, n_loops, vocab_active)
        if jit_key not in self._cached_batch_jits:
            import time as _t_jit
            _jit_compile_start = _t_jit.perf_counter()
            print(f"[JIT] compile cached_generate_batch: B={B} n_loops={n_loops} vocab={vocab_active}...", flush=True)
            ln_g = self.ln_f_g
            ln_b_t = self.ln_f_b
            embed_out = self.embed_out
            embed_w = self.embed.weight
            layers = self.block.layers
            layer_norm_eps = cfg.layer_norm_eps
            n_loops_local = n_loops
            n_layers_local = n_layers
            B_local = B
            vocab_active_local = vocab_active

            @TinyJit
            def _step(prev_id_t, t_pos_t, *kv_flat):
                total = n_layers_local * n_loops_local
                ck = list(kv_flat[:total])
                cv = list(kv_flat[total:])
                # In-graph embedding lookup
                x = embed_w[prev_id_t].cast(dtypes.half)
                integral = Tensor.zeros(B_local, 1, cfg.hidden, dtype=dtypes.half).contiguous()
                new_ck = [None] * total
                new_cv = [None] * total
                for loop in range(n_loops_local):
                    for li in range(n_layers_local):
                        idx = li * n_loops_local + loop
                        x, k_new, v_new = layers[li].forward_cached_step_batched(
                            x, loop, ck[idx], cv[idx], t_pos_t, None  # per-batch t_pos handles masking
                        )
                        new_ck[idx] = k_new
                        new_cv[idx] = v_new
                    integral = integral + x
                integrated = integral / float(n_loops_local)
                x_normed = _layernorm(integrated, ln_g, ln_b_t, layer_norm_eps)
                logits = x_normed @ embed_out  # (B, 1, vocab) — half is fine for argmax
                logits_active = logits[:, :, :vocab_active_local]
                next_id_t = logits_active.argmax(axis=-1).cast(dtypes.int).realize()  # (B, 1)
                return (next_id_t, *new_ck, *new_cv)

            self._cached_batch_jits[jit_key] = _step
            print(f"[JIT] cached_generate_batch graph registered for replay "
                  f"(cache size={len(self._cached_batch_jits)}) — first call will compile lazily.", flush=True)

        jit_step = self._cached_batch_jits[jit_key]

        # Per-batch t_pos: each batch item starts at its real prompt's last position + 1.
        # All advance by 1 each step, so t_pos[b] = real_lens[b] + step.
        step = 0
        t_pos_per = [real_lens[b] for b in range(B)]
        t_pos_t = _T(t_pos_per, dtype=dtypes.int).contiguous().realize()
        # Persistent prev_id_t buffer (B, 1) seeded with first generated tokens
        prev_id_t = _T([[outs[b][0]] for b in range(B)], dtype=dtypes.int).contiguous().realize()
        packed_kv = (
            [cache_k[li][lp] for li in range(n_layers) for lp in range(n_loops)]
            + [cache_v[li][lp] for li in range(n_layers) for lp in range(n_loops)]
        )

        total = n_layers * n_loops
        for _ in range(max_new - 1):
            # Stop if any batch item has run out of cache slots
            if max(t_pos_per) >= max_len:
                break

            outputs = jit_step(prev_id_t, t_pos_t, *packed_kv)
            next_id_t = outputs[0]
            new_kv = outputs[1:]
            for li in range(n_layers):
                for lp in range(n_loops):
                    idx = li * n_loops + lp
                    cache_k[li][lp] = new_kv[idx]
                    cache_v[li][lp] = new_kv[total + idx]
            packed_kv = (
                [cache_k[li][lp] for li in range(n_layers) for lp in range(n_loops)]
                + [cache_v[li][lp] for li in range(n_layers) for lp in range(n_loops)]
            )

            step += 1
            t_pos_per = [real_lens[b] + step for b in range(B)]
            t_pos_t.assign(_T(t_pos_per, dtype=dtypes.int)).realize()

            # Single sync per step: pull next_ids to CPU for stop check
            next_ids = next_id_t.numpy().reshape(B)
            # Feed back into next step via persistent buffer
            prev_id_t.assign(next_id_t).realize()

            for b in range(B):
                if is_done[b]:
                    continue
                outs[b].append(int(next_ids[b]))
                if int(next_ids[b]) in stop_set:
                    is_done[b] = True
                elif seq_len > 0 and outs[b][-seq_len:] == seq:
                    is_done[b] = True
            if all(is_done):
                break

        return outs

    def cached_generate(self, prompt_ids: list, n_loops: int, max_new: int,
                        stop_token_ids=None, stop_seq=None, vocab_active: int = 50277):
        """Fast cached inference that bit-for-bit matches the uncached path.

        Per-loop, per-layer K/V cache. Each new token does N cached breaths through 4
        layers each (vs uncached: N × 4 × full_seq forward operations per token). The
        integral is recomputed for the new token only by accumulating its per-loop
        outputs.

        Cache: (n_phases × n_loops) × {K, V} fixed-shape buffers of (1, n_heads,
        max_seq_len, head_dim). Total ~32 MB at N=8.
        """
        from tinygrad import Tensor as _T
        cfg = self.cfg
        max_len = cfg.max_seq_len
        n_layers = cfg.n_phases
        stop_set = set(stop_token_ids or [])
        seq = list(stop_seq or [])
        seq_len = len(seq)
        prompt_len = len(prompt_ids)
        assert prompt_len <= max_len, f"prompt {prompt_len} exceeds max_seq_len {max_len}"

        # ---- Stage 1: full breathing on prompt, save K/V at every (layer, loop) ----
        # cache[layer_idx][loop_idx] = (K_buf, V_buf) padded to max_len
        x = self.embed(_T([prompt_ids], dtype=dtypes.int).realize()).cast(dtypes.half)
        integral = Tensor.zeros_like(x)
        cache_k = [[None] * n_loops for _ in range(n_layers)]
        cache_v = [[None] * n_loops for _ in range(n_layers)]
        for loop in range(n_loops):
            for li, layer in enumerate(self.block.layers):
                x, (k_part, v_part) = layer.forward_with_kv(x, loop_idx=loop)
                pad_n = max_len - int(k_part.shape[2])
                k_full = k_part.pad(((0, 0), (0, 0), (0, pad_n), (0, 0))).contiguous().realize()
                v_full = v_part.pad(((0, 0), (0, 0), (0, pad_n), (0, 0))).contiguous().realize()
                cache_k[li][loop] = k_full
                cache_v[li][loop] = v_full
            integral = integral + x
        integrated_rep = (integral / float(n_loops)).realize()

        # First token: project integrated_rep[:, -1, :] (matches training exactly).
        h_normed = _layernorm(integrated_rep, self.ln_f_g, self.ln_f_b, cfg.layer_norm_eps)
        logits = (h_normed[:, -1:, :] @ self.embed_out).cast(dtypes.float)
        logits = logits[:, :, :vocab_active]
        next_id = int(logits.argmax(axis=-1).realize().numpy()[0, 0])
        out = [next_id]
        if next_id in stop_set:
            return out
        if seq_len > 0 and out[-seq_len:] == seq:
            return out

        # ---- Stage 2: per-token incremental generation, JIT with in-place cache updates ----
        # The 32 cached forwards fuse into ~4 batched kernels via TinyJit. The remaining
        # bottleneck is data movement: returning new cache tensors from the JIT triggers
        # 32 × 2MB AMD<-AMD copies per token. We eliminate these by mutating the cache
        # buffers IN-PLACE via .assign() inside the JIT — only logits is returned.
        if not hasattr(self, "_cached_token_jit") or getattr(self, "_cached_jit_n_loops", None) != n_loops:
            ln_g = self.ln_f_g
            ln_b = self.ln_f_b
            embed_out = self.embed_out
            layers = self.block.layers
            layer_norm_eps = cfg.layer_norm_eps
            n_loops_local = n_loops
            n_layers_local = n_layers

            @TinyJit
            def _token_step(x_new, t_pos_t, *kv_flat):
                total = n_layers_local * n_loops_local
                ck = list(kv_flat[:total])
                cv = list(kv_flat[total:])
                integral = Tensor.zeros(1, 1, cfg.hidden, dtype=dtypes.half).contiguous()
                new_ck = [None] * total
                new_cv = [None] * total
                x = x_new
                for loop in range(n_loops_local):
                    for li in range(n_layers_local):
                        idx = li * n_loops_local + loop
                        x, k_new, v_new = layers[li].forward_cached_step_jit(
                            x, loop, ck[idx], cv[idx], t_pos_t
                        )
                        new_ck[idx] = k_new
                        new_cv[idx] = v_new
                    integral = integral + x
                integrated = integral / float(n_loops_local)
                x_normed = _layernorm(integrated, ln_g, ln_b, layer_norm_eps)
                logits = (x_normed @ embed_out).cast(dtypes.float).realize()
                return (logits, *new_ck, *new_cv)

            self._cached_token_jit = _token_step
            self._cached_jit_n_loops = n_loops

        t_pos = prompt_len
        t_pos_t = Tensor([t_pos], dtype=dtypes.int).contiguous().realize()
        packed_kv = (
            [cache_k[li][lp] for li in range(n_layers) for lp in range(n_loops)]
            + [cache_v[li][lp] for li in range(n_layers) for lp in range(n_loops)]
        )
        total_n = n_layers * n_loops

        for _ in range(max_new - 1):
            if t_pos >= max_len:
                break
            tok_t = _T([[next_id]], dtype=dtypes.int).realize()
            new_emb = self.embed(tok_t).cast(dtypes.half).contiguous().realize()
            outputs = self._cached_token_jit(new_emb, t_pos_t, *packed_kv)
            logits = outputs[0]
            new_kv = outputs[1:]
            for li in range(n_layers):
                for lp in range(n_loops):
                    idx = li * n_loops + lp
                    cache_k[li][lp] = new_kv[idx]
                    cache_v[li][lp] = new_kv[total_n + idx]
            packed_kv = (
                [cache_k[li][lp] for li in range(n_layers) for lp in range(n_loops)]
                + [cache_v[li][lp] for li in range(n_layers) for lp in range(n_loops)]
            )

            t_pos += 1
            t_pos_t.assign(Tensor([t_pos], dtype=dtypes.int)).realize()

            logits_active = logits[:, :, :vocab_active]
            next_id = int(logits_active.argmax(axis=-1).realize().numpy()[0, 0])
            out.append(next_id)
            if next_id in stop_set:
                break
            if seq_len > 0 and len(out) >= seq_len and out[-seq_len:] == seq:
                break
        return out
