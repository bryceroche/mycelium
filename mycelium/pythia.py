"""Unmodified GPT-NeoX (Pythia) layer for baseline comparison.

This is a clean, single Pythia layer matching HF transformers' modeling_gpt_neox.py
exactly. Used to reproduce AWS looping diagnostic results before testing the v4
breathing modifications.

Differences from BreathingLayer:
  - No weight sharing across layers
  - No π-cycled phase offset (standard RoPE only)
  - No sine-wave temperature (fixed 1/sqrt(head_dim) scaling)
  - No integration / looping at the architecture level
"""
import math
from tinygrad import Tensor, dtypes
from tinygrad.nn import Embedding

from mycelium.config import Config
from mycelium.breathing import _rope_base, _rotate, _layernorm, _linear_w, _bias


class PythiaRoPE:
    """Standard partial RoPE — no per-head, no per-loop offset."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        rd = cfg.rotary_dim
        cos, sin = _rope_base(cfg.max_seq_len, rd, cfg.rope_base)
        # Realize at init — see RoPE in breathing.py for the autograd reason.
        self.cos = cos.reshape(1, 1, cfg.max_seq_len, rd).contiguous().realize()
        self.sin = sin.reshape(1, 1, cfg.max_seq_len, rd).contiguous().realize()

    def apply(self, q: Tensor, k: Tensor):
        S = q.shape[2]
        rd = self.cfg.rotary_dim
        cos = self.cos[:, :, :S, :].cast(q.dtype)
        sin = self.sin[:, :, :S, :].cast(q.dtype)

        q_rot, q_pass = q[..., :rd], q[..., rd:]
        k_rot, k_pass = k[..., :rd], k[..., rd:]
        q_rot = _rotate(q_rot, cos, sin)
        k_rot = _rotate(k_rot, cos, sin)
        return Tensor.cat(q_rot, q_pass, dim=-1), Tensor.cat(k_rot, k_pass, dim=-1)


class PythiaLayer:
    """Single GPT-NeoX block with parallel residual: y = x + Attn(in_LN(x)) + MLP(post_LN(x))."""

    def __init__(self, cfg: Config, rope: PythiaRoPE):
        self.cfg = cfg
        self.rope = rope

        # All weights phase-independent and not shared across layers.
        self.wq = _linear_w(cfg.hidden, cfg.hidden)
        self.bq = _bias(cfg.hidden)
        self.wk = _linear_w(cfg.hidden, cfg.hidden)
        self.bk = _bias(cfg.hidden)
        self.wv = _linear_w(cfg.hidden, cfg.hidden)
        self.bv = _bias(cfg.hidden)
        self.wo = _linear_w(cfg.hidden, cfg.hidden)
        self.bo = _bias(cfg.hidden)

        self.w_in = _linear_w(cfg.hidden, cfg.ffn)
        self.b_in = _bias(cfg.ffn)
        self.w_out = _linear_w(cfg.ffn, cfg.hidden)
        self.b_out = _bias(cfg.hidden)

        self.in_ln_g = Tensor.ones(cfg.hidden, dtype=dtypes.float).contiguous()
        self.in_ln_b = Tensor.zeros(cfg.hidden, dtype=dtypes.float).contiguous()
        self.post_ln_g = Tensor.ones(cfg.hidden, dtype=dtypes.float).contiguous()
        self.post_ln_b = Tensor.zeros(cfg.hidden, dtype=dtypes.float).contiguous()

        self.attn_scale = 1.0 / math.sqrt(cfg.head_dim)

    def parameters(self):
        return [self.wq, self.bq, self.wk, self.bk, self.wv, self.bv, self.wo, self.bo,
                self.w_in, self.b_in, self.w_out, self.b_out,
                self.in_ln_g, self.in_ln_b, self.post_ln_g, self.post_ln_b]

    def __call__(self, x: Tensor) -> Tensor:
        cfg = self.cfg
        B, S, H = x.shape

        attn_in = _layernorm(x, self.in_ln_g, self.in_ln_b, cfg.layer_norm_eps)
        mlp_in = _layernorm(x, self.post_ln_g, self.post_ln_b, cfg.layer_norm_eps)
        attn_in_dt = attn_in.cast(x.dtype) if attn_in.dtype != x.dtype else attn_in
        mlp_in_dt = mlp_in.cast(x.dtype) if mlp_in.dtype != x.dtype else mlp_in

        q = (attn_in_dt @ self.wq + self.bq).reshape(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = (attn_in_dt @ self.wk + self.bk).reshape(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        v = (attn_in_dt @ self.wv + self.bv).reshape(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)

        q, k = self.rope.apply(q, k)

        scores = q @ k.transpose(-2, -1) * self.attn_scale
        mask = Tensor.ones(S, S, dtype=scores.dtype).tril().reshape(1, 1, S, S)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = scores.softmax(-1).cast(v.dtype)
        ctx = (attn @ v).transpose(1, 2).reshape(B, S, H)
        attn_out = ctx @ self.wo + self.bo

        ff = (mlp_in_dt @ self.w_in + self.b_in).gelu()
        ffn_out = ff @ self.w_out + self.b_out

        return x + attn_out + ffn_out


class PythiaStack:
    """N consecutive Pythia layers, embedding, and final LN. Used for baseline diagnostic."""

    def __init__(self, cfg: Config, n_layers: int):
        self.cfg = cfg
        self.n_layers = n_layers
        self.embed = Embedding(cfg.vocab_size, cfg.hidden)
        self.rope = PythiaRoPE(cfg)
        self.layers = [PythiaLayer(cfg, self.rope) for _ in range(n_layers)]
        self.ln_f_g = Tensor.ones(cfg.hidden, dtype=dtypes.float).contiguous()
        self.ln_f_b = Tensor.zeros(cfg.hidden, dtype=dtypes.float).contiguous()

    def parameters(self):
        ps = [self.embed.weight, self.ln_f_g, self.ln_f_b]
        for layer in self.layers:
            ps.extend(layer.parameters())
        return ps

    def hidden_states(self, tokens: Tensor):
        """Returns list of states: [embedded, after_layer_0, after_layer_1, ...]."""
        x = self.embed(tokens).cast(dtypes.half)
        states = [x]
        for layer in self.layers:
            x = layer(x)
            states.append(x)
        return states

    def __call__(self, tokens: Tensor) -> Tensor:
        x = self.hidden_states(tokens)[-1]
        return _layernorm(x, self.ln_f_g, self.ln_f_b, self.cfg.layer_norm_eps)
