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

    # Lookup table — closed-loop component #4
    n_lookup_entries: int = 16        # capacity = 16 prime operations (matches 16 heads)
    seed_lookup: int = 11             # random orthogonal init seed

    # Controller — closed-loop components #3, #5, #6, #7
    page_dim: int = 512               # controller's thinking dimension (notebook page)
    controller_n_layers: int = 3      # notebook attention depth
    controller_n_heads: int = 8       # multi-head attention inside controller

    # WaistController hidden width (DECOUPLED from cfg.hidden so the base model
    # can scale up (e.g. Pythia-1B at H=2048) without ballooning the controller
    # params. At H=2048 with controller_hidden=1024: cross-attn K/V become
    # rectangular (2048 → 1024); a final up-projection 1024 → cfg.hidden lets
    # the tied embed_out work. Default 1024 matches Pythia-410M's H (no change
    # for existing models).
    controller_hidden: int = 1024

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.rotary_pct)

    @property
    def controller_head_dim(self) -> int:
        return self.controller_hidden // self.n_heads
