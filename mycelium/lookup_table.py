"""LookupTable — the prime-operation library that lives inside the model.

A `n_entries × hidden`-dim cosine-similarity matcher. Initialized random orthogonal
(matches the spec's prime basis intuition); learns the model's actual operation
directions via auxiliary cross-entropy on operation classification when joint-trained.

Each query returns a `(..., n_entries)` tensor of cosine similarities — the per-prime
match weights the controller will read. No softmax inside the table; the controller
applies its own normalization downstream.

Op label convention: 0=+, 1=-, 2=*, 3=/. The first 4 entries of the table are the
"live" classes for ARITH; the remaining 12 stay free for additional operations
encountered at L4+ (fraction-of, comparison, sequential dependency, etc.).
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from tinygrad import Tensor, dtypes


_OP_TO_IDX = {"+": 0, "-": 1, "*": 2, "/": 3}


def op_label_from_text(text: str) -> int:
    """Returns 0/1/2/3 for first occurrence of +/-/*/, or -1 if no op character is
    present anywhere in `text`. Used to derive ground-truth labels for the aux loss
    from MathExample.problem + gen_targets."""
    for ch in text:
        if ch in _OP_TO_IDX:
            return _OP_TO_IDX[ch]
    return -1


def find_eq_position(tokens_1d, eq_token_ids) -> int:
    """First index in tokens_1d whose token id is in eq_token_ids, or -1 if absent.
    eq_token_ids may be an int or any iterable of ints (BPE encodes "=" and " =" as
    different tokens; pass both via eq_token_ids_for(tok) below)."""
    if isinstance(eq_token_ids, int):
        eq_token_ids = {eq_token_ids}
    else:
        eq_token_ids = set(eq_token_ids)
    for i, t in enumerate(tokens_1d):
        if t in eq_token_ids:
            return i
    return -1


def eq_token_ids_for(tok) -> list[int]:
    """All tokens this tokenizer uses to represent '=' (with and without leading space).
    BPE encodes them differently; both can appear in encoded inputs."""
    return list(set(tok.encode("=").ids + tok.encode(" =").ids))


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
