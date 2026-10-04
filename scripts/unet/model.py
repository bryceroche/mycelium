"""scripts/unet/model.py — THE U-NET PICTURE BAKE-OFF, the reader. A NEW
module; no existing script touched. A small 2D U-Net (tinygrad) over a
(C,T,T) picture (pictures.py), output (4,T,T) per-pixel class logits
(targets.py's CLASS_NONE/GIVEN/ARG/RESOP). 2 down/up levels + a bottleneck
(3 resolutions total), 16/32/64 channels, skip connections, GroupNorm.

THE PER-BREATH MASK (THE MANDATORY-ROAD LAW's shape, applied to an
unrelated organ: "one graph for every breath, never slicing"): the SAME
U-Net graph runs once per breath over the SAME input picture, with the 3
WHERE channels (same-sentence/clause/mention, when the picture carries
them) multiplied by a FIXED per-breath {0,1} schedule (content channels
always weight 1) and the breath index appended as one extra constant
conditioning channel. This is done by REPLICATING the batch along a
breath axis and multiplying by a precomputed CONSTANT mask tensor --
never by python-side slicing of the channel dimension, so one call
(`forward_breaths`) is one static graph for every breath, JIT-stable
(fixed shapes, no data-dependent control flow). Breaths are folded into
the batch dimension (B,K,...)->(B*K,...) so the same Conv2d/GroupNorm
weights are shared and reused across breaths within one pass, matching
the family's own "weights shared across breaths" convention
(mycelium/factor_graph_engine.py's K=16 breath sharing) even though this
U-Net has no recurrence of its own (every breath sees the one fixed
picture, never the previous breath's output -- intentionally: this is a
reading organ, not an iterative solver).
"""
import os
import sys

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.nn import Conv2d, GroupNorm

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")


def where_schedule(K_B):
    """(K_B,3) f32 {0,1}: ramps (1,0,0) -> (1,1,0) -> (1,1,1) across
    breaths in three roughly-equal stages (K_B=7 -> stage boundaries at
    breaths 2 and 4, matching the brief's "1,0,0 -> 1,1,0 -> 1,1,1")."""
    b1 = max(1, K_B // 3)
    b2 = max(b1 + 1, (2 * K_B) // 3)
    sched = np.zeros((K_B, 3), np.float32)
    for b in range(K_B):
        sched[b, 0] = 1.0
        sched[b, 1] = 1.0 if b >= b1 else 0.0
        sched[b, 2] = 1.0 if b >= b2 else 0.0
    return sched


def estimate_flops(c_in, base, T, K_B, batch, n_classes=4):
    """Rough forward-pass multiply-add count for the architecture below
    (2*Cin*Cout*k^2*H*W per conv; backward is conventionally ~2x forward,
    so total train-step FLOPs ~3x this). Printed at construction time."""
    def conv(cin, cout, k, h, w):
        return 2 * cin * cout * k * k * h * w
    T0, T1, T2 = T, T // 2, T // 4
    total = 0
    total += conv(c_in, base, 3, T0, T0) + conv(base, base, 3, T0, T0)
    total += conv(base, base * 2, 3, T1, T1) + conv(base * 2, base * 2, 3, T1, T1)
    total += conv(base * 2, base * 4, 3, T2, T2) + conv(base * 4, base * 4, 3, T2, T2)
    total += conv(base * 4 + base * 2, base * 2, 3, T1, T1) + conv(base * 2, base * 2, 3, T1, T1)
    total += conv(base * 2 + base, base, 3, T0, T0) + conv(base, base, 3, T0, T0)
    total += conv(base, n_classes, 1, T0, T0)
    return total * batch * K_B


class ConvBlock:
    def __init__(self, cin, cout):
        self.conv = Conv2d(cin, cout, 3, padding=1)
        groups = min(8, cout)
        while cout % groups != 0:
            groups -= 1
        self.gn = GroupNorm(groups, cout)

    def __call__(self, x):
        return self.gn(self.conv(x)).relu()

    def params(self):
        return [self.conv.weight, self.conv.bias, self.gn.weight, self.gn.bias]


class UNet:
    """c_content: channels that are ALWAYS weight 1 (cosine + PCA bands,
    9 for this bake-off). n_where: 0 (Aprime) or 3 (Adouble/Adouble_shuf)
    -- the WHERE channels the per-breath schedule modulates. K_B: breath
    count (the conditioning scalar's range; ALG_BREATH's sibling here,
    an independent knob -- see UN_BREATHS in train.py)."""

    def __init__(self, c_content, n_where, K_B, base=16, n_classes=4, seed=0):
        Tensor.manual_seed(seed)
        self.c_content = c_content
        self.n_where = n_where
        self.c_in = c_content + n_where
        self.K_B = K_B
        self.base = base
        self.n_classes = n_classes
        c0 = self.c_in + 1   # +1 breath-scalar conditioning channel
        self.enc0a, self.enc0b = ConvBlock(c0, base), ConvBlock(base, base)
        self.enc1a, self.enc1b = ConvBlock(base, base * 2), ConvBlock(base * 2, base * 2)
        self.bota, self.botb = ConvBlock(base * 2, base * 4), ConvBlock(base * 4, base * 4)
        self.dec1a, self.dec1b = ConvBlock(base * 4 + base * 2, base * 2), ConvBlock(base * 2, base * 2)
        self.dec0a, self.dec0b = ConvBlock(base * 2 + base, base), ConvBlock(base, base)
        self.head = Conv2d(base, n_classes, 1)
        # FIXED (non-trained) constants: the breath where-schedule and the
        # breath conditioning scalar. Registered as plain numpy here;
        # turned into Tensors lazily in forward_breaths so device/dtype
        # always matches the input picture's.
        self._where_sched_np = where_schedule(K_B) if n_where else None   # (K,3)
        self._breath_scalar_np = (np.arange(K_B, dtype=np.float32) /
                                   max(1, K_B - 1))                        # (K,) in [0,1]
        flops = estimate_flops(self.c_in, base, 256, K_B, 1)
        print(f"[unet/model] UNet c_in={self.c_in} base={base} K_B={K_B}: "
              f"~{flops/1e9:.2f} GFLOP/image-breath-set at T=256 fwd-only "
              f"(x{K_B} breaths folded into batch; backward ~2x more)",
              flush=True)

    def parameters(self):
        ps = []
        for blk in (self.enc0a, self.enc0b, self.enc1a, self.enc1b,
                    self.bota, self.botb, self.dec1a, self.dec1b,
                    self.dec0a, self.dec0b):
            ps += blk.params()
        ps += [self.head.weight, self.head.bias]
        return ps

    def _unet_body(self, x):
        e0 = self.enc0b(self.enc0a(x))
        p0 = e0.avg_pool2d(2)
        e1 = self.enc1b(self.enc1a(p0))
        p1 = e1.avg_pool2d(2)
        b = self.botb(self.bota(p1))
        u1 = b.interpolate(size=e1.shape[-2:], mode="nearest")
        d1 = self.dec1b(self.dec1a(Tensor.cat(u1, e1, dim=1)))
        u0 = d1.interpolate(size=e0.shape[-2:], mode="nearest")
        d0 = self.dec0b(self.dec0a(Tensor.cat(u0, e0, dim=1)))
        logits = self.head(d0)
        return logits.clip(-1e4, 1e4)   # quirks: clip before any log/softmax downstream

    def forward_breaths(self, x_base, where_sched=None):
        """x_base: (B, c_in, T, T) float. Returns (B, K_B, n_classes, T, T)
        logits -- ONE call through the body per (batch*breath) image, the
        where-channels and the breath-scalar channel baked in via constant
        multiply/concat (never a python slice of the channel axis).
        `where_sched` overrides the model's own (K_B,3) schedule (numpy
        array) -- used ONLY by train.py's smoke-test ablation check (never
        inside the jitted training step, so the step's graph shape/consts
        never change step to step)."""
        B, Cin, T, _ = x_base.shape
        assert Cin == self.c_in, (Cin, self.c_in)
        K = self.K_B
        x_rep = x_base.reshape(B, 1, Cin, T, T).expand(B, K, Cin, T, T).reshape(B * K, Cin, T, T)
        if self.n_where:
            sched_np = where_sched if where_sched is not None else self._where_sched_np
            ones = np.ones((K, self.c_content), np.float32)
            mask_np = np.concatenate([ones, sched_np.astype(np.float32)], axis=1)   # (K, c_in)
            mask = Tensor(mask_np, dtype=dtypes.float, device=x_base.device)
            mask_b = mask.reshape(1, K, Cin, 1, 1).expand(B, K, Cin, T, T).reshape(B * K, Cin, T, T)
            x_rep = x_rep * mask_b
        bscal = Tensor(self._breath_scalar_np, dtype=dtypes.float, device=x_base.device)
        bscal_b = bscal.reshape(1, K, 1, 1, 1).expand(B, K, 1, T, T).reshape(B * K, 1, T, T)
        x_in = Tensor.cat(x_rep, bscal_b, dim=1)   # (B*K, c_in+1, T, T)
        logits = self._unet_body(x_in)
        return logits.reshape(B, K, self.n_classes, T, T)


if __name__ == "__main__":
    # tiny CPU shape/finite check -- no training, no gradients
    for n_where in (0, 3):
        m = UNet(c_content=9, n_where=n_where, K_B=4, base=8, seed=0)
        x = Tensor(np.random.default_rng(0).standard_normal((2, m.c_in, 32, 32)).astype(np.float32))
        out = m.forward_breaths(x)
        onp = out.numpy()
        print(f"n_where={n_where}: out shape={onp.shape} finite={np.isfinite(onp).all()}")
        if n_where:
            off = np.zeros((4, 3), np.float32)
            out0 = m.forward_breaths(x, where_sched=off).numpy()
            print(f"  ablated-schedule output differs from default: "
                  f"{not np.allclose(onp, out0)}")
