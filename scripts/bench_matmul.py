"""Matmul TFLOPS benchmark on the 7900 XTX.

Runs square FP16 matmuls at several sizes, JIT-compiled with optional BEAM search,
prints achieved TFLOPS vs the 7900 XTX FP16 theoretical (~120 TFLOPS).
"""
import time
from tinygrad import Tensor, dtypes, Device, GlobalCounters
from tinygrad.helpers import getenv
from tinygrad.engine.jit import TinyJit

def main():
    SIZES = [int(s) for s in getenv("SIZES", "1024,2048,4096,8192").split(",")]
    WARMUP = getenv("WARMUP", 3)
    ITERS = getenv("ITERS", 10)
    DTYPE = dtypes.half if getenv("HALF", 1) else dtypes.float

    print(f"device={Device.DEFAULT} dtype={DTYPE} BEAM={getenv('BEAM',0)} JIT={getenv('JIT',1)}")
    print(f"theoretical FP16 ~120 TFLOPS (7900 XTX)")
    print(f"{'N':>6}  {'iters':>5}  {'ms/iter':>10}  {'TFLOPS':>10}")

    for N in SIZES:
        a = Tensor.randn(N, N, dtype=DTYPE).realize()
        b = Tensor.randn(N, N, dtype=DTYPE).realize()

        @TinyJit
        def step(x, y):
            return (x @ y).realize()

        for _ in range(WARMUP):
            step(a, b)
        Device[Device.DEFAULT].synchronize()

        GlobalCounters.reset()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            c = step(a, b)
        Device[Device.DEFAULT].synchronize()
        dt = time.perf_counter() - t0

        flops = 2 * N**3 * ITERS
        tflops = flops / dt / 1e12
        print(f"{N:>6}  {ITERS:>5}  {dt/ITERS*1e3:>10.3f}  {tflops:>10.2f}", flush=True)


if __name__ == "__main__":
    main()
