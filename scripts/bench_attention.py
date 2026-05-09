"""Pythia-shaped attention block benchmark.

Mycelium v4 uses Pythia-410M dimensions: h=1024, 16 heads, head_dim=64.
This benchmarks one breath's worth of self-attention at seq_len=512.
"""
import time
from tinygrad import Tensor, dtypes, Device, GlobalCounters
from tinygrad.helpers import getenv
from tinygrad.engine.jit import TinyJit


def main():
    B = getenv("B", 8)
    SEQ = getenv("SEQ", 512)
    H = getenv("H", 1024)
    HEADS = getenv("HEADS", 16)
    HEAD_DIM = H // HEADS
    WARMUP = getenv("WARMUP", 3)
    ITERS = getenv("ITERS", 20)
    DTYPE = dtypes.half if getenv("HALF", 1) else dtypes.float

    print(f"device={Device.DEFAULT} dtype={DTYPE} BEAM={getenv('BEAM',0)} JIT={getenv('JIT',1)}")
    print(f"B={B} seq={SEQ} h={H} heads={HEADS} head_dim={HEAD_DIM}")

    x = Tensor.randn(B, SEQ, H, dtype=DTYPE).realize()
    wq = Tensor.randn(H, H, dtype=DTYPE).realize()
    wk = Tensor.randn(H, H, dtype=DTYPE).realize()
    wv = Tensor.randn(H, H, dtype=DTYPE).realize()
    wo = Tensor.randn(H, H, dtype=DTYPE).realize()

    USE_SDPA = getenv("SDPA", 1)

    @TinyJit
    def attn(x, wq, wk, wv, wo):
        q = (x @ wq).reshape(B, SEQ, HEADS, HEAD_DIM).transpose(1, 2)
        k = (x @ wk).reshape(B, SEQ, HEADS, HEAD_DIM).transpose(1, 2)
        v = (x @ wv).reshape(B, SEQ, HEADS, HEAD_DIM).transpose(1, 2)
        if USE_SDPA:
            o = q.scaled_dot_product_attention(k, v)
        else:
            s = (q @ k.transpose(-2, -1)) / (HEAD_DIM ** 0.5)
            p = s.softmax(-1)
            o = p @ v
        o = o.transpose(1, 2).reshape(B, SEQ, H)
        return (o @ wo).realize()

    for _ in range(WARMUP):
        attn(x, wq, wk, wv, wo)
    Device[Device.DEFAULT].synchronize()

    GlobalCounters.reset()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        out = attn(x, wq, wk, wv, wo)
    Device[Device.DEFAULT].synchronize()
    dt = time.perf_counter() - t0

    flops_proj = 4 * 2 * B * SEQ * H * H
    flops_qk = 2 * B * HEADS * SEQ * SEQ * HEAD_DIM
    flops_av = 2 * B * HEADS * SEQ * SEQ * HEAD_DIM
    flops = (flops_proj + flops_qk + flops_av) * ITERS
    tflops = flops / dt / 1e12

    print(f"ms/iter: {dt/ITERS*1e3:.3f}")
    print(f"TFLOPS:  {tflops:.2f}")
    print(f"output: shape={out.shape} dtype={out.dtype}")


if __name__ == "__main__":
    main()
