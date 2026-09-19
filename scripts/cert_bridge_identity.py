"""cert_bridge_identity.py — THE IDENTITY CHECK (THE CERTIFICATE PASS,
2026-09-18): mycelium.loop_bridge.cert_spotlight (the in-graph road) must
be bit-exact with Bridge(fat, sent).spotlight(F, beta, mode) (the numpy
road _wheel_turn / _wheel_bias use) on the same inputs, for both modes,
including rows with no flags at all.

Usage: .venv/bin/python3 scripts/cert_bridge_identity.py
"""
import os
import sys

os.environ.setdefault("DEV", "CPU")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
from tinygrad import Tensor

from mycelium.loop_bridge import Bridge, cert_spotlight


def main():
    rng = np.random.default_rng(2026)
    B, LT, T = 4, 32, 48
    n_sent = 6
    fails = 0
    for trial in range(20):
        fat = rng.random((B, LT, T)).astype(np.float32)
        # sentences 0..n_sent-1 packed contiguously per row, with a
        # trailing pad run (sentinel sentence id, mimicking the head's
        # padding convention: pad tokens still carry SOME sent id, never
        # matched by a real flagged slot's source since fat's argmax
        # over a random attention row essentially never lands on padding
        # when padding is a small trailing suffix).
        sent = np.zeros((B, T), np.int64)
        for b in range(B):
            n_pad = rng.integers(0, T // 4)
            n_real = T - n_pad
            bounds = np.sort(rng.choice(np.arange(1, n_real), size=n_sent - 1, replace=False))
            ids = np.zeros(n_real, np.int64)
            cur = 0
            prev = 0
            for k, bnd in enumerate(list(bounds) + [n_real]):
                ids[prev:bnd] = k
                prev = bnd
            sent[b, :n_real] = ids
            sent[b, n_real:] = n_sent  # a sentence id no flagged slot ever sources (pad sentinel)
        F_rows = (rng.random((B, LT)) > 0.6).astype(np.float32)
        F_rows[0] = 0.0   # row 0: no flags at all, every trial
        for mode in ("union", "own"):
            for beta in (3.0, 1.0, 0.0):
                ref = Bridge(fat, sent).spotlight(F_rows, beta, mode)
                got = cert_spotlight(Tensor(F_rows), Tensor(fat), Tensor(sent.astype(np.int32)),
                                      beta, mode).numpy()
                if not np.array_equal(ref, got):
                    fails += 1
                    d = np.abs(ref - got)
                    print(f"[cert-identity] MISMATCH trial={trial} mode={mode} beta={beta} "
                          f"max|diff|={d.max()} at {np.unravel_index(d.argmax(), d.shape)}")
    if fails:
        print(f"[cert-identity] FAIL: {fails} mismatching (trial, mode, beta) combinations")
        sys.exit(1)
    print("[cert-identity] PASS: cert_spotlight == Bridge.spotlight bit-exact "
          "(union/own, beta in {3.0,1.0,0.0}, rows with no flags, 20 random fixtures)")


if __name__ == "__main__":
    main()
