"""Smoke test: tinygrad on the 7900 XTX.

Confirms backend selection, runs a tiny matmul + grad, prints the result.
"""
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import DEV

print(f"DEV target: {DEV}")
print(f"Default device: {Device.DEFAULT}")

a = Tensor.randn(64, 128, dtype=dtypes.float32).realize()
b = Tensor.randn(128, 32, dtype=dtypes.float32).realize()
c = (a @ b).realize()
print(f"a@b shape={c.shape} dtype={c.dtype} device={c.device}")
print(f"first row: {c[0, :4].tolist()}")

x = Tensor.randn(8, 16, requires_grad=True)
y = (x * 2 + 1).sum()
y.backward()
print(f"grad mean: {x.grad.mean().item():.4f} (expect 2.0)")
