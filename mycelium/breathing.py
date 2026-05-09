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
from typing import List
from tinygrad import Tensor, dtypes
from tinygrad.nn import Embedding

from mycelium.config import Config


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
        self.alpha_cos: List[Tensor] = []
        self.alpha_sin: List[Tensor] = []
        for l in range(cfg.max_loops):
            loop_phase = l * math.pi / cfg.max_loops
            alphas = [hp + loop_phase for hp in head_phase]
            ac = Tensor(alphas, dtype=dtypes.float).cos().reshape(1, cfg.n_heads, 1, 1).contiguous().realize()
            asn = Tensor(alphas, dtype=dtypes.float).sin().reshape(1, cfg.n_heads, 1, 1).contiguous().realize()
            self.alpha_cos.append(ac)
            self.alpha_sin.append(asn)

    def apply(self, q: Tensor, k: Tensor, loop_idx: int):
        """q, k: (B, n_heads, seq, head_dim). Rotate first rotary_dim slots, leave rest."""
        S = q.shape[2]
        rd = self.cfg.rotary_dim
        cos = self.cos[:, :, :S, :].cast(q.dtype)
        sin = self.sin[:, :, :S, :].cast(q.dtype)

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

    def __call__(self, x: Tensor, loop_idx: int) -> Tensor:
        cfg = self.cfg
        B, S, H = x.shape

        # Pythia parallel residual: y = x + Attn(in_LN(x)) + MLP(post_LN(x))
        attn_in = _layernorm(x, self.shared.in_ln_g, self.shared.in_ln_b, cfg.layer_norm_eps)
        mlp_in = _layernorm(x, self.shared.post_ln_g, self.shared.post_ln_b, cfg.layer_norm_eps)
        attn_in_dt = attn_in.cast(x.dtype) if attn_in.dtype != x.dtype else attn_in
        mlp_in_dt = mlp_in.cast(x.dtype) if mlp_in.dtype != x.dtype else mlp_in

        # ---- attention ----
        q = (attn_in_dt @ self.wq + self.bq).reshape(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = (attn_in_dt @ self.wk + self.bk).reshape(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        v = (attn_in_dt @ self.shared.wv + self.shared.bv).reshape(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)

        q, k = self.rope.apply(q, k, loop_idx)

        scores = q @ k.transpose(-2, -1) * self.attn_scale
        # Causal mask: position i can only attend to positions <= i.
        # Without this, training leaks: bidirectional attention lets the model see
        # future tokens, breaking the autoregressive LM objective.
        mask = Tensor.ones(S, S, dtype=scores.dtype).tril().reshape(1, 1, S, S)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = scores.softmax(-1).cast(v.dtype)
        ctx = (attn @ v).transpose(1, 2).reshape(B, S, H)
        attn_out = ctx @ self.shared.wo + self.shared.bo

        # ---- FFN: GELU(x @ w_in + b_in) @ w_out + b_out ----
        ff = (mlp_in_dt @ self.w_in + self.b_in).gelu()
        ffn_out = ff @ self.shared.w_out + self.shared.b_out

        return x + attn_out + ffn_out


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

    def parameters(self):
        return [self.embed.weight, self.ln_f_g, self.ln_f_b, self.embed_out] + self.block.parameters()

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
