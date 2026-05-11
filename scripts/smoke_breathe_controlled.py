"""Smoke the full 7/7 closed loop on whatever DEV tinygrad picks.

Inference only. Builds BreathingTransformer (which owns transformer + lookup +
controller), allocates a Notebook, runs breathe_controlled on a tiny synthetic
batch, and prints per-breath decisions. Crash-free completion plus sane shapes
means the 7/7 loop runs end-to-end on this stack.

Usage:
  DEV=PCI+AMD .venv/bin/python scripts/smoke_breathe_controlled.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv

from mycelium import Config, BreathingTransformer
from mycelium.controller import Notebook


def main():
    cfg = Config()
    B = getenv("B", 2)
    SEQ = getenv("SEQ", 64)
    MAX_LOOPS = getenv("MAX_LOOPS", 4)

    print(f"device={Device.DEFAULT}")
    print(f"cfg: hidden={cfg.hidden} heads={cfg.n_heads} ffn={cfg.ffn} "
          f"phases={cfg.n_phases} max_loops={cfg.max_loops}")
    print(f"smoke: B={B} seq={SEQ} max_loops={MAX_LOOPS}")
    print()

    Tensor.training = False

    t0 = time.perf_counter()
    model = BreathingTransformer(cfg)
    Device[Device.DEFAULT].synchronize()
    print(f"built model in {time.perf_counter() - t0:.1f}s")

    tokens = Tensor.randint(B, SEQ, low=0, high=cfg.vocab_size).realize()
    notebook = Notebook()

    t0 = time.perf_counter()
    final_hidden, decisions, n_breaths, match_weights = model.breathe_controlled(
        tokens, max_loops=MAX_LOOPS, notebook=notebook,
    )
    final_hidden.realize()
    Device[Device.DEFAULT].synchronize()
    elapsed = time.perf_counter() - t0
    print(f"breathe_controlled: {elapsed:.2f}s, n_breaths={n_breaths}")
    print(f"final_hidden shape: {tuple(int(s) for s in final_hidden.shape)}")
    print(f"match_weights: {len(match_weights)} breaths, "
          f"each shape {tuple(int(s) for s in match_weights[0].shape)}")
    print()
    print("per-breath decisions:")
    for i, d in enumerate(decisions):
        parts = []
        for k, v in d.items():
            try:
                arr = v.numpy() if hasattr(v, "numpy") else v
                if hasattr(arr, "mean"):
                    parts.append(f"{k}={float(arr.mean()):+.3f}")
                else:
                    parts.append(f"{k}={arr}")
            except Exception:
                parts.append(f"{k}=<{type(v).__name__}>")
        print(f"  breath {i}: " + " ".join(parts))


if __name__ == "__main__":
    main()
