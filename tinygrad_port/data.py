"""
Data loading for per-cycle answer head training, ported to tinygrad.

No DataLoader -- simple Python iteration with numpy shuffling.
"""
import json
import numpy as np
from tinygrad import Tensor, dtypes


class PerCycleDataset:
    """Load per-cycle training data from JSONL files.

    Each line: {"problem": "...", "cycle_targets": [48, 24, 72],
                "final_answer": 72, "num_steps": 3}
    """

    def __init__(self, jsonl_path, max_passes=5):
        self.samples = []
        self.max_passes = max_passes
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def get_batch(self, indices):
        """Get a batch by indices. Returns dict with padded tensors.

        Args:
            indices: list of int -- sample indices

        Returns:
            dict with:
                'problems':          list of str
                'cycle_targets':     Tensor (B, max_steps) int32
                'cycle_mask':        Tensor (B, max_steps) float32
                'cycle_gen_targets': list of max_steps lists, each of B strings
                'final_answers':     list of int/str
                'num_steps':         list of int
                'max_steps':         int
        """
        batch = [self.samples[i] for i in indices]

        problems = [s['problem'] for s in batch]
        finals = [s['final_answer'] for s in batch]
        num_steps_list = [
            min(s.get('num_steps', len(s['cycle_targets'])), self.max_passes)
            for s in batch
        ]
        max_steps = max(num_steps_list) if num_steps_list else 1

        padded_targets = []
        mask = []
        gen_targets_by_cycle = [[] for _ in range(max_steps)]

        for s in batch:
            ct = s['cycle_targets'][:self.max_passes]
            gt = s.get('cycle_gen_targets', [str(c) for c in ct])
            gt = gt[:self.max_passes]
            pad_len = max_steps - len(ct)
            padded_targets.append(ct + [0] * pad_len)
            mask.append([1.0] * len(ct) + [0.0] * pad_len)
            for i in range(max_steps):
                if i < len(gt):
                    gen_targets_by_cycle[i].append(gt[i])
                else:
                    gen_targets_by_cycle[i].append("0")  # placeholder

        return {
            'problems': problems,
            'cycle_targets': Tensor(padded_targets, dtype=dtypes.int32),
            'cycle_mask': Tensor(mask, dtype=dtypes.float32),
            'cycle_gen_targets': gen_targets_by_cycle,
            'final_answers': finals,
            'num_steps': num_steps_list,
            'max_steps': max_steps,
        }


class SubsetDataset:
    """Wraps a PerCycleDataset to expose only a contiguous range of indices.

    Used as a fallback eval set when no separate eval file exists.
    """

    def __init__(self, dataset, start, end):
        self.dataset = dataset
        self.start = start
        self.end = end

    def __len__(self):
        return self.end - self.start

    def get_sample(self, idx):
        """Get a single sample dict (same keys as the raw JSONL)."""
        return self.dataset.samples[self.start + idx]


def batch_iterator(dataset, batch_size, shuffle=True):
    """Simple batch iterator -- no DataLoader needed.

    Yields dicts from dataset.get_batch().

    Args:
        dataset:    PerCycleDataset instance
        batch_size: int
        shuffle:    bool, whether to shuffle each epoch

    Yields:
        dict -- batch from dataset.get_batch()
    """
    indices = list(range(len(dataset)))
    if shuffle:
        np.random.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i + batch_size]
        yield dataset.get_batch(batch_idx)
