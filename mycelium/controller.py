"""Controller — the breathing transformer's conductor.

Step B scope: state reader + decision heads. The controller reads the integrated
representation at a single position and emits adaptive decisions:
  - Temperature multiplier for the next breath
  - Integration gate (how much the next breath contributes to the running integral)
  - Continue/stop logit (whether to keep breathing)

It also writes a 512-dim page recording what this breath understood. The page is
stored in the Notebook and read on subsequent breaths via tree-structured attention.

Step C: Notebook (per-problem list of pages) + multi-layer notebook attention.
  - The controller's first pass produces a "raw page" from the integrated rep.
  - Notebook attention then refines the page by reading prior pages of the same
    problem (across breaths AND outer cycles).
  - Decision heads operate on the refined page.

Decisions are emitted but NOT yet wired back into breathing — that's Step D, where
gradient separation also kicks in (controller gradient never flows through the
transformer).
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


class Notebook:
    """Per-problem memory: a list of 512d pages, one per breath, persisting across
    inner breathing loops AND outer execution cycles. Each page is a (B, page_dim)
    tensor; the notebook stacks them along axis 1 for attention.

    Caller is responsible for clear() at the start of each new problem (or batch),
    and for append()ing the controller's page after each breath.
    """

    def __init__(self):
        self.pages: list = []   # each entry: (B, 1, page_dim) for easy concat

    def append(self, page: Tensor):
        """page: (B, page_dim) — the controller's note for the current breath."""
        self.pages.append(page.reshape(page.shape[0], 1, page.shape[1]))

    def stacked(self) -> Tensor | None:
        """(B, N_pages, page_dim) or None if empty."""
        if not self.pages:
            return None
        if len(self.pages) == 1:
            return self.pages[0]
        return self.pages[0].cat(*self.pages[1:], dim=1)

    def clear(self):
        self.pages = []

    def __len__(self):
        return len(self.pages)


class _MultiHeadSelfAttn:
    """Plain multi-head self-attention block over a sequence of pages.
    No causal mask — every page can attend to every other (full closed-loop reads
    of ancestors, siblings, children). LN + residual on output."""

    def __init__(self, cfg: Config):
        p, n_h = cfg.page_dim, cfg.controller_n_heads
        assert p % n_h == 0, f"page_dim {p} must be divisible by n_heads {n_h}"
        self.cfg = cfg
        self.n_h = n_h
        self.head_dim = p // n_h
        self.q_w = _linear_w(p, p); self.q_b = _zeros(p)
        self.k_w = _linear_w(p, p); self.k_b = _zeros(p)
        self.v_w = _linear_w(p, p); self.v_b = _zeros(p)
        self.o_w = _linear_w(p, p); self.o_b = _zeros(p)
        # FFN
        self.ff1_w = _linear_w(p, p * 4); self.ff1_b = _zeros(p * 4)
        self.ff2_w = _linear_w(p * 4, p); self.ff2_b = _zeros(p)
        # LNs
        self.ln1_g = Tensor.ones(p, dtype=dtypes.float).contiguous(); self.ln1_b = _zeros(p)
        self.ln2_g = Tensor.ones(p, dtype=dtypes.float).contiguous(); self.ln2_b = _zeros(p)

    def parameters(self):
        return [self.q_w, self.q_b, self.k_w, self.k_b, self.v_w, self.v_b,
                self.o_w, self.o_b, self.ff1_w, self.ff1_b, self.ff2_w, self.ff2_b,
                self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b]

    def state_dict_with_prefix(self, prefix: str) -> dict:
        names = ["q_w","q_b","k_w","k_b","v_w","v_b","o_w","o_b",
                 "ff1_w","ff1_b","ff2_w","ff2_b","ln1_g","ln1_b","ln2_g","ln2_b"]
        return {f"{prefix}.{n}": getattr(self, n) for n in names}

    def __call__(self, x: Tensor) -> Tensor:
        """x: (B, N, page_dim) → (B, N, page_dim)"""
        B, N, P = x.shape
        h, hd = self.n_h, self.head_dim
        # Pre-LN attention
        x_n = _layernorm(x, self.ln1_g, self.ln1_b, self.cfg.layer_norm_eps)
        q = (x_n @ self.q_w + self.q_b).reshape(B, N, h, hd).transpose(1, 2)   # (B, h, N, hd)
        k = (x_n @ self.k_w + self.k_b).reshape(B, N, h, hd).transpose(1, 2)
        v = (x_n @ self.v_w + self.v_b).reshape(B, N, h, hd).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))             # (B, h, N, N)
        attn = scores.softmax(-1)
        ctx = (attn @ v).transpose(1, 2).reshape(B, N, P)
        x = x + (ctx @ self.o_w + self.o_b)
        # Pre-LN FFN
        ff_in = _layernorm(x, self.ln2_g, self.ln2_b, self.cfg.layer_norm_eps)
        ff = (ff_in @ self.ff1_w + self.ff1_b).gelu()
        ff = ff @ self.ff2_w + self.ff2_b
        return x + ff


class Controller:
    """Reads the integrated representation, writes a 512d page, refines it via
    notebook attention over prior pages, emits adaptive decisions.

    Param count (Step C): ~6.6M with 3 attention layers — page-dim 512, ffn 2048.
    Step E (richer state reader / Perceiver) can grow toward the spec's ~40M.
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
        # Notebook attention layers — full attention over [prior pages, current page]
        self.notebook_layers = [_MultiHeadSelfAttn(cfg) for _ in range(cfg.controller_n_layers)]
        # Decision heads operate on the post-notebook-attention page
        self.temp_w = _linear_w(p, 1)
        self.temp_b = _zeros(1)
        self.gate_w = _linear_w(p, 1)
        self.gate_b = _zeros(1)
        self.stop_w = _linear_w(p, 1)
        self.stop_b = _zeros(1)

    def parameters(self):
        ps = [self.reader_w1, self.reader_b1,
              self.reader_w2, self.reader_b2,
              self.page_ln_g, self.page_ln_b,
              self.temp_w, self.temp_b,
              self.gate_w, self.gate_b,
              self.stop_w, self.stop_b]
        for layer in self.notebook_layers:
            ps.extend(layer.parameters())
        return ps

    def state_dict(self) -> dict:
        sd = {
            "controller.reader_w1": self.reader_w1, "controller.reader_b1": self.reader_b1,
            "controller.reader_w2": self.reader_w2, "controller.reader_b2": self.reader_b2,
            "controller.page_ln_g": self.page_ln_g, "controller.page_ln_b": self.page_ln_b,
            "controller.temp_w": self.temp_w, "controller.temp_b": self.temp_b,
            "controller.gate_w": self.gate_w, "controller.gate_b": self.gate_b,
            "controller.stop_w": self.stop_w, "controller.stop_b": self.stop_b,
        }
        for i, layer in enumerate(self.notebook_layers):
            sd.update(layer.state_dict_with_prefix(f"controller.notebook.{i}"))
        return sd

    def _read_page(self, rep: Tensor) -> Tensor:
        """rep (B, hidden) → raw page (B, page_dim) via 2-layer MLP + LN."""
        rep = rep.cast(dtypes.float)
        h = (rep @ self.reader_w1 + self.reader_b1).gelu()
        h = (h @ self.reader_w2 + self.reader_b2)
        return _layernorm(h, self.page_ln_g, self.page_ln_b, self.cfg.layer_norm_eps)

    def _decision_heads(self, page: Tensor) -> dict:
        """page (B, page_dim) → decisions dict."""
        temp_logit = (page @ self.temp_w + self.temp_b).reshape(-1)
        gate_logit = (page @ self.gate_w + self.gate_b).reshape(-1)
        stop_logit = (page @ self.stop_w + self.stop_b).reshape(-1)
        temperature = temp_logit.sigmoid() * 1.5 + 0.5
        gate = gate_logit.sigmoid()
        return {"page": page, "temperature": temperature, "gate": gate, "stop_logit": stop_logit}

    def __call__(self, rep: Tensor, notebook: Notebook | None = None) -> dict:
        """rep: (B, hidden) — integrated representation at a single position.
        notebook: optional Notebook of prior pages from this problem.

        Pipeline: rep → state-reader page → append to notebook → notebook attention
        over all pages → take the LAST position (= current page, post-attention)
        → decision heads on that refined page.
        """
        page = self._read_page(rep)                              # (B, page_dim)
        if notebook is None:
            return self._decision_heads(page)

        # Append current page; run attention over [prior, current]
        notebook.append(page)
        x = notebook.stacked()                                   # (B, N, page_dim)
        for layer in self.notebook_layers:
            x = layer(x)
        # Refined current page is the last position
        refined = x[:, -1, :]                                    # (B, page_dim)
        # Replace the last entry in the notebook with the refined page so subsequent
        # breaths see a consistent, attended history rather than the raw reader output.
        notebook.pages[-1] = refined.reshape(refined.shape[0], 1, refined.shape[1])
        return self._decision_heads(refined)
