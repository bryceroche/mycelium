"""Controller — the breathing transformer's conductor.

Step B scope: state reader + decision heads. The controller reads the integrated
representation at a single position and emits adaptive decisions:
  - Temperature multiplier for the next breath
  - Integration gate (how much the next breath contributes to the running integral)
  - Continue/stop logit (whether to keep breathing)

It also writes a 512-dim page recording what this breath understood. The page is
the basic unit the notebook (Step C) will store and the controller's attention
will read in subsequent breaths.

Decisions are emitted but NOT yet wired back into breathing — that's Step D, where
gradient separation also kicks in (controller gradient never flows through the
transformer). Step B is just the forward path.
"""
from __future__ import annotations
import math
import numpy as np
from tinygrad import Tensor, dtypes

from mycelium.config import Config


def _linear_w(in_dim: int, out_dim: int, dtype=dtypes.float) -> Tensor:
    """Kaiming-style init for ReLU/GELU activations."""
    return (Tensor.randn(in_dim, out_dim, dtype=dtype) * math.sqrt(2.0 / in_dim)).contiguous()


def _zeros(*shape, dtype=dtypes.float) -> Tensor:
    return Tensor.zeros(*shape, dtype=dtype).contiguous()


def _layernorm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float) -> Tensor:
    in_dt = x.dtype
    x = x.cast(dtypes.float)
    mean = x.mean(axis=-1, keepdim=True)
    var = ((x - mean).square()).mean(axis=-1, keepdim=True)
    out = (x - mean) / (var + eps).sqrt()
    return (out * gamma + beta).cast(in_dt)


class Controller:
    """Reads the integrated representation, writes a 512d page, emits adaptive
    decisions for the next breath.

    Param count (Step B): ~525K — light scaffold. Step C (notebook + tree
    attention) and Step E (richer state reader) will grow this toward the
    spec's ~40M target.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        h, p = cfg.hidden, cfg.page_dim
        # State reader: 1024d hidden → 512d page. Two-layer MLP with GELU + LN.
        self.reader_w1 = _linear_w(h, p)
        self.reader_b1 = _zeros(p)
        self.reader_w2 = _linear_w(p, p)
        self.reader_b2 = _zeros(p)
        self.page_ln_g = Tensor.ones(p, dtype=dtypes.float).contiguous()
        self.page_ln_b = _zeros(p)
        # Decision heads (page → scalar)
        self.temp_w = _linear_w(p, 1)
        self.temp_b = _zeros(1)
        self.gate_w = _linear_w(p, 1)
        self.gate_b = _zeros(1)
        self.stop_w = _linear_w(p, 1)
        self.stop_b = _zeros(1)

    def parameters(self):
        return [self.reader_w1, self.reader_b1,
                self.reader_w2, self.reader_b2,
                self.page_ln_g, self.page_ln_b,
                self.temp_w, self.temp_b,
                self.gate_w, self.gate_b,
                self.stop_w, self.stop_b]

    def state_dict(self) -> dict:
        return {
            "controller.reader_w1": self.reader_w1, "controller.reader_b1": self.reader_b1,
            "controller.reader_w2": self.reader_w2, "controller.reader_b2": self.reader_b2,
            "controller.page_ln_g": self.page_ln_g, "controller.page_ln_b": self.page_ln_b,
            "controller.temp_w": self.temp_w, "controller.temp_b": self.temp_b,
            "controller.gate_w": self.gate_w, "controller.gate_b": self.gate_b,
            "controller.stop_w": self.stop_w, "controller.stop_b": self.stop_b,
        }

    def __call__(self, rep: Tensor) -> dict:
        """rep: (B, hidden) — integrated representation at a single position.
        Returns dict with per-batch decisions:
          page:        (B, page_dim)        — the controller's 512d note
          temperature: (B,) in (0.5, 2.0)   — multiplier on baseline temperature
          gate:        (B,) in (0, 1)       — integration gate for next breath
          stop_logit:  (B,)                 — pre-sigmoid; high = stop
        """
        rep = rep.cast(dtypes.float)
        h = (rep @ self.reader_w1 + self.reader_b1).gelu()
        h = (h @ self.reader_w2 + self.reader_b2)
        page = _layernorm(h, self.page_ln_g, self.page_ln_b, self.cfg.layer_norm_eps)

        temp_logit = (page @ self.temp_w + self.temp_b).reshape(-1)        # (B,)
        gate_logit = (page @ self.gate_w + self.gate_b).reshape(-1)        # (B,)
        stop_logit = (page @ self.stop_w + self.stop_b).reshape(-1)        # (B,)
        # Bound temperature to (0.5, 2.0) — keeps the schedule in a sensible range
        temperature = temp_logit.sigmoid() * 1.5 + 0.5
        gate = gate_logit.sigmoid()
        return {
            "page": page,
            "temperature": temperature,
            "gate": gate,
            "stop_logit": stop_logit,
        }
