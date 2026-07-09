"""Forward-parity gate: factor_graph_layer_forward ≡ kenken_layer_forward at S=49.

The general engine now routes its breath loop through the S-agnostic
`factor_graph_layer_forward` (so it can run at s_max != 49, e.g. dual-view s_max=98)
instead of the oracle `kenken_layer_forward` (which hard-asserts S==49). This gate
certifies the twin is BYTE-IDENTICAL to the oracle at S=49 with rotation off — the
only path the KenKen anchor ever uses — so the swap cannot have changed the
validated KenKen behavior. (Replaces the correctness role of a primal-only control
run; the existing 0.796 number came from the old oracle path and can't certify new code.)

  DEV=AMD .venv/bin/python3 scripts/test_general_layer_parity.py
"""
import sys
import numpy as np
from tinygrad import Tensor, dtypes

sys.path.insert(0, ".")
from mycelium import Config                                    # noqa: E402
from mycelium.loader import load_breathing, _load_state       # noqa: E402
from mycelium.kenken import kenken_layer_forward              # oracle (asserts S==49)
from mycelium.factor_graph_engine import factor_graph_layer_forward  # general twin

S = 49
B = 3


def main():
    cfg = Config()
    print(f"loading breathing model (h={cfg.hidden} n_heads={cfg.n_heads})...", flush=True)
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    layer = list(model.block.layers)[0]

    rng = np.random.RandomState(20260625)
    x_np = (rng.randn(B, S, cfg.hidden) * 0.5).astype(np.float32)
    # attn_bias: {0 allow, -1e4 block}, per-batch per-head, like the real masks.
    allow = (rng.rand(B, cfg.n_heads, S, S) > 0.5).astype(np.float32)
    bias_np = (1.0 - allow) * (-1e4)

    x = Tensor(x_np, dtype=dtypes.float)
    bias = Tensor(bias_np, dtype=dtypes.float)

    out_oracle = kenken_layer_forward(layer, x, bias).cast(dtypes.float).numpy()
    out_general = factor_graph_layer_forward(layer, x, bias).cast(dtypes.float).numpy()

    max_abs = float(np.max(np.abs(out_oracle - out_general)))
    print(f"\n  out shape: {out_oracle.shape}")
    print(f"  max|Δ| (general - oracle) @ S=49 = {max_abs:.3e}")
    ok = max_abs == 0.0
    print(f"  PARITY: {'PASS (byte-identical)' if ok else 'FAIL'}")
    # Sanity: also confirm the general layer RUNS at S=98 (the dual-view size) — no assert.
    x98 = Tensor((rng.randn(B, 98, cfg.hidden) * 0.5).astype(np.float32), dtype=dtypes.float)
    b98 = Tensor(((rng.rand(B, cfg.n_heads, 98, 98) > 0.5).astype(np.float32) - 1.0) * 1e4 * 0 +
                 (1.0 - (rng.rand(B, cfg.n_heads, 98, 98) > 0.5).astype(np.float32)) * (-1e4),
                 dtype=dtypes.float)
    out98 = factor_graph_layer_forward(layer, x98, b98).cast(dtypes.float).numpy()
    print(f"  general layer at S=98 runs: out shape {out98.shape}  (oracle would assert here)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
