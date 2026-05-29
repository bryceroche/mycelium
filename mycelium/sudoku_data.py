"""Sudoku data loader for v98.

Streams JSONL puzzles, optionally filters by difficulty, supports a per-step
curriculum (start with 100% easy at step 0, anneal to uniform across
difficulties by curriculum_anneal_steps).
"""
import json
import os
import random
from typing import Iterator

import numpy as np
from tinygrad import Tensor, dtypes


DIFFICULTIES = ["easy", "medium", "hard", "expert"]


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def index_by_difficulty(records: list[dict]) -> dict[str, list[dict]]:
    idx: dict[str, list[dict]] = {d: [] for d in DIFFICULTIES}
    for r in records:
        d = r.get("difficulty", "easy")
        if d not in idx:
            idx[d] = []
        idx[d].append(r)
    return idx


class SudokuLoader:
    """Iterates puzzles in batches. Supports:
      - difficulty_filter: only sample from one difficulty band (smoke mode)
      - curriculum: per-step difficulty distribution that anneals from
        100% easy (step 0) to uniform across all-available (curriculum_anneal_steps)
    """

    def __init__(self, path: str, batch_size: int = 8,
                 difficulty_filter: str | None = None,
                 curriculum: bool = False,
                 curriculum_anneal_steps: int = 1000,
                 seed: int = 0):
        self.path = path
        self.batch_size = batch_size
        self.difficulty_filter = difficulty_filter
        self.curriculum = curriculum
        self.curriculum_anneal_steps = curriculum_anneal_steps
        self.rng = random.Random(seed)
        self.numpy_rng = np.random.RandomState(seed)

        records = load_jsonl(path)
        if difficulty_filter is not None:
            records = [r for r in records if r.get("difficulty") == difficulty_filter]
            assert records, f"No records in {path} match difficulty={difficulty_filter}"

        self.records = records
        self.by_diff = index_by_difficulty(records)
        # Only keep non-empty bands
        self.active_diffs = [d for d in DIFFICULTIES if len(self.by_diff[d]) > 0]
        print(f"[sudoku_data] loaded {len(records)} from {os.path.basename(path)}; "
              f"by difficulty: {[(d, len(self.by_diff[d])) for d in self.active_diffs]}",
              flush=True)

    def _step_difficulty_weights(self, step: int) -> dict[str, float]:
        """Curriculum: at step 0, 100% easy. By curriculum_anneal_steps, uniform
        across active difficulties."""
        if not self.curriculum or "easy" not in self.active_diffs:
            return {d: 1.0 / len(self.active_diffs) for d in self.active_diffs}
        progress = min(1.0, step / max(1.0, float(self.curriculum_anneal_steps)))
        # easy weight: 1.0 → 1/N; others: 0.0 → 1/N
        N = len(self.active_diffs)
        weights = {}
        for d in self.active_diffs:
            if d == "easy":
                weights[d] = (1.0 - progress) + progress * (1.0 / N)
            else:
                weights[d] = progress * (1.0 / N)
        # Normalize (paranoia)
        s = sum(weights.values())
        return {d: w / s for d, w in weights.items()}

    def sample_batch(self, step: int = 0) -> tuple[Tensor, Tensor, list[dict]]:
        """Return (inputs (B,81) int Tensor, golds (B,81) int Tensor, metadata list)."""
        if self.difficulty_filter is not None:
            picks = self.rng.choices(self.records, k=self.batch_size)
        else:
            weights = self._step_difficulty_weights(step)
            inputs = []
            for _ in range(self.batch_size):
                diff = self.rng.choices(self.active_diffs,
                                         weights=[weights[d] for d in self.active_diffs], k=1)[0]
                inputs.append(self.rng.choice(self.by_diff[diff]))
            picks = inputs

        # Build int arrays
        in_arr = np.array([r["input"] for r in picks], dtype=np.int32)        # (B, 81)
        sol_arr = np.array([r["solution"] for r in picks], dtype=np.int32)    # (B, 81)
        return (
            Tensor(in_arr, dtype=dtypes.int).contiguous().realize(),
            Tensor(sol_arr, dtype=dtypes.int).contiguous().realize(),
            picks,
        )

    def iter_eval(self, batch_size: int | None = None) -> Iterator[tuple[Tensor, Tensor, list[dict]]]:
        """Iterate through ALL records in order (no shuffle), in batches.
        For eval. Pads the last batch with repeats so all batches are full.
        """
        bs = batch_size or self.batch_size
        n = len(self.records)
        for start in range(0, n, bs):
            batch = self.records[start:start + bs]
            # Pad to fixed size if needed
            while len(batch) < bs:
                batch.append(self.records[0])
            in_arr = np.array([r["input"] for r in batch], dtype=np.int32)
            sol_arr = np.array([r["solution"] for r in batch], dtype=np.int32)
            yield (
                Tensor(in_arr, dtype=dtypes.int).contiguous().realize(),
                Tensor(sol_arr, dtype=dtypes.int).contiguous().realize(),
                batch,
            )

    def __len__(self):
        return len(self.records)
