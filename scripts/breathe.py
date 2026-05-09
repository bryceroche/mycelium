"""First breath — forward pass through the v4 breathing transformer.

Verifies shapes, parameter count, and per-step time at B=64, seq=512, loops in {1, 4, 8}.
No training, no controller, no generation. Just the breath.
"""
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.engine.jit import TinyJit

from mycelium import Config, BreathingTransformer


def count_params(params):
    total = 0
    for p in params:
        n = 1
        for s in p.shape:
            n *= int(s)
        total += n
    return total


def main():
    cfg = Config()
    B = getenv("B", 64)
    SEQ = getenv("SEQ", 512)
    WARMUP = getenv("WARMUP", 2)
    ITERS = getenv("ITERS", 5)
    LOOPS = [int(s) for s in getenv("LOOPS", "1,4,8").split(",")]

    print(f"device={Device.DEFAULT}")
    print(f"cfg: hidden={cfg.hidden} heads={cfg.n_heads} ffn={cfg.ffn} phases={cfg.n_phases} max_loops={cfg.max_loops}")
    print(f"input: B={B} seq={SEQ}")
    print()

    print("Building model...", flush=True)
    t0 = time.perf_counter()
    model = BreathingTransformer(cfg)
    Device[Device.DEFAULT].synchronize()
    print(f"  built in {time.perf_counter() - t0:.1f}s")

    n = count_params(model.parameters())
    print(f"  params: {n/1e6:.1f}M ({n:,})")
    print()

    # Synthetic input tokens
    tokens = Tensor.randint(B, SEQ, low=0, high=cfg.vocab_size).realize()

    for n_loops in LOOPS:
        print(f"--- n_loops = {n_loops} ---", flush=True)

        # Build a JIT'd forward at this loop count. TinyJit captures the trace on the
        # first 2 calls and replays the kernel sequence on subsequent calls.
        @TinyJit
        def fwd(tok):
            return model(tok, n_loops).realize()

        # Per-iteration timing to see JIT compile vs replay separation
        for i in range(WARMUP):
            t = time.perf_counter()
            out = fwd(tokens)
            Device[Device.DEFAULT].synchronize()
            print(f"  warmup {i}: {(time.perf_counter()-t)*1000:.1f} ms", flush=True)

        per_iter = []
        for i in range(ITERS):
            t = time.perf_counter()
            out = fwd(tokens)
            Device[Device.DEFAULT].synchronize()
            per_iter.append((time.perf_counter()-t)*1000)
        dt = sum(per_iter)/1000.0

        ms = dt / ITERS * 1000
        print(f"  output shape: {out.shape}, dtype: {out.dtype}")
        print(f"  per-iter ms: {[f'{m:.1f}' for m in per_iter]}")
        print(f"  ms/iter: {ms:.2f}  ({ms/n_loops:.2f} ms/loop)")
        print(f"  first row sample: {out[0, 0, :4].cast(dtypes.float).tolist()}")
        print()


if __name__ == "__main__":
    main()
