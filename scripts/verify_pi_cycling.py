"""Verify pi-cycled RoPE produces structurally different attention per loop.

The mathematical claim: a phase shift alpha(h, l) = h*pi/n_heads + l*pi/max_loops
applied to Q (only) rotates the q.k bilinear form. So the softmax(QK^T) matrix should
differ across loop indices, even with identical input.

This script:
  1. Runs the same input through the model at loop indices 0..max_loops-1
  2. Captures the softmax attention matrix from layer 0, head 0
  3. Reports cross-loop similarity (cosine of flattened attention matrices)
  4. Reports per-head diversity within one breath (head 0 vs head 8 at same loop)

If pi cycling is structural, cross-loop cosine should be << 1 (different patterns)
and the diversity should be the same regardless of weight values.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from tinygrad import Tensor, Device, dtypes
from mycelium import Config


def attention_at_loop(cfg, x, wq, wk, wv, rope, loop_idx):
    """Compute softmax attention pattern (B, heads, seq, seq) for a given loop_idx."""
    B, S, H = x.shape
    q = (x @ wq).reshape(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)
    k = (x @ wk).reshape(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)
    q, k = rope.apply(q, k, loop_idx)
    scale = 1.0 / math.sqrt(cfg.head_dim)
    scores = q @ k.transpose(-2, -1) * scale
    return scores.softmax(-1)


def main():
    from mycelium.breathing import RoPE, _linear_weight

    cfg = Config()
    B, S = 4, 64
    rope = RoPE(cfg)

    # Fixed weights and input — only loop_idx changes.
    Tensor.manual_seed(42)
    x = Tensor.randn(B, S, cfg.hidden, dtype=dtypes.half).realize()
    wq = _linear_weight(cfg.hidden, cfg.hidden)
    wk = _linear_weight(cfg.hidden, cfg.hidden)
    wv = _linear_weight(cfg.hidden, cfg.hidden)

    print(f"Input: B={B} seq={S} hidden={cfg.hidden}")
    print(f"Loops: 0..{cfg.max_loops - 1}, heads: 0..{cfg.n_heads - 1}\n")

    # 1. Attention patterns at each loop_idx, head 0
    patterns = []
    for l in range(cfg.max_loops):
        attn = attention_at_loop(cfg, x, wq, wk, wv, rope, l)  # (B, H, S, S)
        patterns.append(attn.cast(dtypes.float).numpy())

    print("Cross-loop cosine similarity, head 0 (1.0 = identical):")
    p0_h0 = patterns[0][:, 0].reshape(-1)  # flatten across batch and seq positions
    for l in range(cfg.max_loops):
        p_h0 = patterns[l][:, 0].reshape(-1)
        dot = (p0_h0 * p_h0).sum()
        denom = ((p0_h0 ** 2).sum() ** 0.5) * ((p_h0 ** 2).sum() ** 0.5)
        cos = float(dot / denom)
        diff = float(((p0_h0 - p_h0) ** 2).mean() ** 0.5)
        print(f"  loop {l}: cos vs loop 0 = {cos:.4f}  rms diff = {diff:.4f}")

    print("\nPer-head diversity at loop 0 (cos of head 0 vs head h):")
    p0 = patterns[0]
    for h in range(cfg.n_heads):
        p_h = p0[:, h].reshape(-1)
        p_h0 = p0[:, 0].reshape(-1)
        dot = (p_h0 * p_h).sum()
        denom = ((p_h0 ** 2).sum() ** 0.5) * ((p_h ** 2).sum() ** 0.5)
        cos = float(dot / denom)
        print(f"  head {h:2d}: cos = {cos:.4f}")


if __name__ == "__main__":
    main()
