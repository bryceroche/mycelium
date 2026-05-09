"""Mycelium v4 model dimensions.

Matches Pythia-410M's first 4 transformer blocks so we can later init from those weights.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Pythia-410M dimensions (first 4 layers of 24)
    vocab_size: int = 50304
    max_seq_len: int = 512
    hidden: int = 1024
    n_heads: int = 16
    head_dim: int = 64  # hidden // n_heads
    ffn: int = 4096

    # Pythia-410M specifics
    rotary_pct: float = 0.25        # only first 16 of 64 head_dim get rotated
    layer_norm_eps: float = 1e-5

    # Breathing
    n_phases: int = 4          # RISE, PEAK, FALL, TROUGH
    max_loops: int = 8
    rope_base: int = 10000     # standard RoPE base; head/loop offsets layered on top

    # Temperature: T = exp(temp_amp * sin(phase)) — broadens at PEAK (sin=1), sharpens at TROUGH (sin=-1)
    temp_amp: float = 0.5

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.rotary_pct)
