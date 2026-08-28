"""Per-step optimal-stopping calibration training.

Each L4-style problem has N intermediate-answer "=" positions in its token
sequence. The model claims each step's answer when its confidence crosses a
threshold; each correct claim earns 1/N reward, each wrong claim a penalty.
Different steps can be claimed at different breaths — the model learns to
allocate breath budget per step difficulty.

This module provides:
  ConfidenceHead     — small MLP from (hidden,) reps to scalar confidence in (0,1)
  find_all_eq_positions — multi-step variant of find_eq_position
  build_step_targets — extract per-step (eq_pos, digit_positions, digit_ids) from a token sequence
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from tinygrad import Tensor, dtypes


def _linear_w(in_dim: int, out_dim: int) -> Tensor:
    """fp32 init — matches the post-cast_model_fp32 dtype of trainable params."""
    return (Tensor.randn(in_dim, out_dim, dtype=dtypes.float) * 0.02).contiguous()


def _zeros(dim: int) -> Tensor:
    return Tensor.zeros(dim, dtype=dtypes.float).contiguous()


class ConfidenceHead:
    """Reads a rep at a step's "=" position and emits a scalar confidence in (0,1).

    Two-layer MLP with GELU. Output is sigmoid'd so it's bounded. Trained
    jointly with the transformer via REINFORCE on the optimal-stopping objective
    (claim_when_confident-and-correct → reward; claim_when_confident-and-wrong →
    penalty).
    """

    def __init__(self, hidden: int, mid: int = 256):
        self.hidden = hidden
        self.mid = mid
        self.w1 = _linear_w(hidden, mid)
        self.b1 = _zeros(mid)
        self.w2 = _linear_w(mid, 1)
        self.b2 = _zeros(1)

    def parameters(self):
        return [self.w1, self.b1, self.w2, self.b2]

    def state_dict(self) -> dict:
        return {
            "confidence.w1": self.w1, "confidence.b1": self.b1,
            "confidence.w2": self.w2, "confidence.b2": self.b2,
        }

    def __call__(self, x: Tensor) -> Tensor:
        """x: (..., hidden). Returns: (...,) — confidence in (0,1)."""
        x = x.cast(dtypes.float)
        h = (x @ self.w1 + self.b1).gelu()
        out = (h @ self.w2 + self.b2).reshape(*x.shape[:-1])
        return out.sigmoid()


def find_all_eq_positions(tokens_1d, eq_token_ids) -> List[int]:
    """All indices in tokens_1d whose token id is in eq_token_ids."""
    if isinstance(eq_token_ids, int):
        eq_token_ids = {eq_token_ids}
    else:
        eq_token_ids = set(eq_token_ids)
    return [i for i, t in enumerate(tokens_1d) if t in eq_token_ids]


def extract_digit_runs_after_eq(tokens_1d, eq_positions: List[int], digit_token_ids: set,
                                 max_digits: int = 4) -> List[Tuple[int, List[int], List[int]]]:
    """For each eq_pos, find the contiguous run of digit tokens that follows.

    Returns: list of (eq_pos, digit_positions, digit_ids) per step.
    digit_positions: token indices of the digit run (length up to max_digits)
    digit_ids: the actual token ids at those positions (the ground-truth answer)

    The run ends at the first non-digit token (e.g., space, "+", "-", " items", etc).
    For digit-spaced encoding, digits are individual tokens like " 3", " 1", etc.
    """
    out = []
    n = len(tokens_1d)
    for eq in eq_positions:
        digit_positions: List[int] = []
        digit_ids: List[int] = []
        # Start scanning from the token AFTER "=". The "=" rep predicts the first
        # digit at position eq+1.
        i = eq + 1
        while i < n and len(digit_positions) < max_digits and tokens_1d[i] in digit_token_ids:
            digit_positions.append(i)
            digit_ids.append(int(tokens_1d[i]))
            i += 1
        if digit_positions:  # only include steps with at least 1 digit
            out.append((eq, digit_positions, digit_ids))
    return out


def digit_token_ids_for(tok) -> set:
    """Set of token IDs that represent individual digits (0-9) in digit-spaced encoding.

    Each digit appears as " 0", " 1", ..., " 9" with a leading space — that's
    what space_digits() produces. We also include the bare "0".."9" forms in
    case BPE merges differently in any context.
    """
    ids = set()
    for d in "0123456789":
        ids.update(tok.encode(d).ids)
        ids.update(tok.encode(" " + d).ids)
    return ids
