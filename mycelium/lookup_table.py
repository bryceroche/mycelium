"""LookupTable — the prime-operation library that lives inside the model.

A `n_entries × hidden`-dim cosine-similarity matcher. Initialized random orthogonal
(matches the spec's prime basis intuition); learns the model's actual operation
directions via auxiliary cross-entropy on operation classification when joint-trained.

Each query returns a `(..., n_entries)` tensor of cosine similarities — the per-prime
match weights the controller will read. No softmax inside the table; the controller
applies its own normalization downstream.
"""
from __future__ import annotations
import numpy as np
from tinygrad import Tensor, dtypes


class LookupTable:
    def __init__(self, n_entries: int = 16, hidden: int = 1024, seed: int = 11):
        self.n_entries = n_entries
        self.hidden = hidden
        # Random orthogonal init — the prime basis at t=0 (the closed-loop spec's
        # "spectral matching" framing). Joint training pulls each entry toward the
        # model's actual operation direction.
        rng = np.random.default_rng(seed)
        M = rng.standard_normal((hidden, n_entries))
        Q, _ = np.linalg.qr(M)               # (hidden, n_entries), columns orthonormal
        self.weight = Tensor(Q.T.astype(np.float32), requires_grad=True).contiguous().realize()

    def parameters(self):
        return [self.weight]

    def __call__(self, x: Tensor) -> Tensor:
        """Cosine-similarity match.
        x: (..., hidden) any leading dims
        returns: (..., n_entries) cosine similarities in [-1, 1]
        """
        x_n = x / (x.square().sum(axis=-1, keepdim=True).sqrt() + 1e-6)
        w_n = self.weight / (self.weight.square().sum(axis=-1, keepdim=True).sqrt() + 1e-6)
        # x_n: (..., hidden), w_n: (n_entries, hidden) → matmul on last dim
        return x_n @ w_n.transpose()
