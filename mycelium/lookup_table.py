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
import os
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
        # v28 prototype retrieval: each lookup entry gets a VALUE — a 1024d "ideal
        # rep" prototype for problems matching this key. Forward: at some point in
        # the rep stream, query lookup → get match weights → weighted sum of values
        # → inject into rep. Continuous Hopfield-network / soft attention over a
        # learned prototype library.
        #
        # Information bottleneck (LOOKUP_IB_DIM > 0): values stored at LOOKUP_IB_DIM
        # instead of hidden, with a learned (IB_DIM, hidden) project_up matrix.
        # Forces compressed representation that can't encode example specifics —
        # only the "procedure shape" survives. Default (LOOKUP_IB_DIM=0) keeps
        # values at hidden dim (no bottleneck).
        ib_dim = int(os.environ.get("LOOKUP_IB_DIM", "0"))
        self.ib_dim = ib_dim
        value_dim = ib_dim if ib_dim > 0 else hidden
        # Init: random fallback for both values and proj_up.
        self.values = (Tensor.randn(n_entries, value_dim, dtype=dtypes.float) * 0.02).contiguous()
        if ib_dim > 0:
            self.value_proj_up = (Tensor.randn(ib_dim, hidden, dtype=dtypes.float) * 0.02).contiguous()
        else:
            self.value_proj_up = Tensor.eye(hidden).cast(dtypes.float).contiguous()

        # Optional data-driven init from saved file:
        #   .npy:  values only (n_entries, value_dim), normalized to small-init magnitude
        #   .npz:  values + proj_up (the LDA-compressed pair)
        init_path = os.environ.get("LOOKUP_VALUES_INIT_PATH", "")
        if init_path and os.path.exists(init_path):
            import numpy as _np
            if init_path.endswith(".npz"):
                data = _np.load(init_path)
                if "values" in data and data["values"].shape == (n_entries, value_dim):
                    self.values = Tensor(data["values"].astype(_np.float32), dtype=dtypes.float).contiguous()
                if "proj_up" in data and ib_dim > 0 and data["proj_up"].shape == (ib_dim, hidden):
                    self.value_proj_up = Tensor(data["proj_up"].astype(_np.float32), dtype=dtypes.float).contiguous()
            elif init_path.endswith(".npy"):
                raw = _np.load(init_path).astype(_np.float32)
                if raw.shape == (n_entries, value_dim):
                    norms = _np.linalg.norm(raw, axis=-1, keepdims=True) + 1e-9
                    target_norm = _np.sqrt(value_dim) * 0.02
                    values_np = (raw / norms) * target_norm
                    self.values = Tensor(values_np, dtype=dtypes.float).contiguous()

    def parameters(self):
        return [self.weight, self.values, self.value_proj_up]

    def __call__(self, x: Tensor) -> Tensor:
        """Cosine-similarity match.
        x: (..., hidden) any leading dims
        returns: (..., n_entries) cosine similarities in [-1, 1]
        """
        x_n = x / (x.square().sum(axis=-1, keepdim=True).sqrt() + 1e-6)
        w_n = self.weight / (self.weight.square().sum(axis=-1, keepdim=True).sqrt() + 1e-6)
        # x_n: (..., hidden), w_n: (n_entries, hidden) → matmul on last dim
        return x_n @ w_n.transpose()

    def retrieve(self, match_weights: Tensor) -> Tensor:
        """Given match weights (..., n_entries), return weighted-sum of values (..., hidden).
        Softmax over match weights first to make a proper probability over prototypes.
        When LOOKUP_IB_DIM > 0, values are stored at IB_DIM and project_up to hidden.

        Temperature LOOKUP_TEMP (env var) controls sharpness:
          T=1.0 (default): cosine similarities range [-1, 1] → softmax near-uniform
                           because differences between top match and others are tiny.
          T=10-20: sharp routing — top-1 entry dominates weights.
        """
        T = float(os.environ.get("LOOKUP_TEMP", "1.0"))
        weights = (match_weights * T).softmax(axis=-1)
        retrieved_low = weights @ self.values
        if self.ib_dim > 0:
            return retrieved_low @ self.value_proj_up
        return retrieved_low
