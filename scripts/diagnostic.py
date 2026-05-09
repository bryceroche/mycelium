"""AWS looping diagnostic, reproduced on Shadow Glass with v4 architecture.

For each model variant, captures hidden states at every checkpoint (layer for the
sequential baseline, breath for the looping breathing transformer) and computes:

  signal_norm       — avg ||h_i|| across batch and tokens
  cross_problem_cos — centered cosine between per-problem mean hiddens
                      (near 0 = orthogonal across problems = preserved diversity)
  eff_rank          — effective rank of the per-problem mean state matrix
                      (computed as exp(entropy of normalized singular values))
  snr_db            — signal-to-noise ratio in dB, where signal = variance across
                      problems (per-problem mean - global mean), noise = variance
                      within problem (per-token deviation from problem mean)

AWS reference for Pythia layer outputs: signal 3.9->6.4, eff rank 15-16, cos -0.05.
"""
import sys
import os
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes

from mycelium import Config
from mycelium.loader import _load_state, load_pythia_baseline, load_breathing


def hidden_metrics(h_np: np.ndarray) -> dict:
    """h_np: (B, S, H) float32. Returns scalar diagnostic metrics."""
    B, S, H = h_np.shape
    flat = h_np.reshape(B * S, H)

    # Signal norm: average ||x_i||
    signal_norm = float(np.linalg.norm(flat, axis=1).mean())

    # Per-problem mean over tokens: (B, H)
    per_problem = h_np.mean(axis=1)  # (B, H)

    # Global mean across problems: (H,)
    global_mean = per_problem.mean(axis=0, keepdims=True)
    centered = per_problem - global_mean  # (B, H)

    # Centered cross-problem cosine: average pairwise cos of centered per-problem vectors.
    # Excludes self-pairs.
    norms = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-12
    unit = centered / norms
    cos_mat = unit @ unit.T  # (B, B)
    n = B
    if n > 1:
        # average of off-diagonal entries
        cross_cos = float((cos_mat.sum() - np.trace(cos_mat)) / (n * (n - 1)))
    else:
        cross_cos = 1.0

    # Effective rank of per-problem mean state matrix
    s = np.linalg.svd(centered, compute_uv=False)
    s_norm = s / (s.sum() + 1e-12)
    s_norm = s_norm[s_norm > 1e-12]
    entropy = -(s_norm * np.log(s_norm)).sum()
    eff_rank = float(np.exp(entropy))

    # SNR: signal = var across problems (centered per-problem means), noise = var within problem
    sig_var = float((centered ** 2).mean())
    within = h_np - per_problem[:, None, :]  # (B, S, H), per-token deviations from problem mean
    noise_var = float((within ** 2).mean())
    snr_db = 10.0 * math.log10((sig_var + 1e-12) / (noise_var + 1e-12))

    return {
        "signal_norm": signal_norm,
        "cross_cos": cross_cos,
        "eff_rank": eff_rank,
        "snr_db": snr_db,
    }


def run_pythia_baseline(cfg: Config, sd: dict, tokens: Tensor, n_layers: int):
    print(f"\n=== Pythia baseline ({n_layers} sequential layers, no sharing, no π) ===\n")
    model = load_pythia_baseline(cfg, n_layers=n_layers, sd=sd)
    states = model.hidden_states(tokens)
    Device[Device.DEFAULT].synchronize()

    print(f"{'checkpoint':>14}  {'signal':>8}  {'cross_cos':>10}  {'eff_rank':>10}  {'snr_db':>8}")
    for i, s in enumerate(states):
        h_np = s.cast(dtypes.float).realize().numpy()
        m = hidden_metrics(h_np)
        label = "embed" if i == 0 else f"layer {i-1}"
        print(f"{label:>14}  {m['signal_norm']:>8.3f}  {m['cross_cos']:>10.4f}  {m['eff_rank']:>10.2f}  {m['snr_db']:>8.2f}")


def run_breathing(cfg: Config, sd: dict, tokens: Tensor, n_loops: int, label: str):
    print(f"\n=== Breathing transformer — {label} ({n_loops} loops, sharing + π) ===\n")
    model = load_breathing(cfg, sd=sd)
    states, _ = model.hidden_states(tokens, n_loops=n_loops, return_per_loop=True)
    Device[Device.DEFAULT].synchronize()

    print(f"{'checkpoint':>14}  {'signal':>8}  {'cross_cos':>10}  {'eff_rank':>10}  {'snr_db':>8}")
    for i, s in enumerate(states):
        h_np = s.cast(dtypes.float).realize().numpy()
        m = hidden_metrics(h_np)
        label_i = "embed" if i == 0 else f"breath {i}"
        print(f"{label_i:>14}  {m['signal_norm']:>8.3f}  {m['cross_cos']:>10.4f}  {m['eff_rank']:>10.2f}  {m['snr_db']:>8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "breathing", "both"], default="both")
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_loops", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = Config()
    print(f"device={Device.DEFAULT}  B={args.batch}  seq={args.seq}  seed={args.seed}")

    sd = _load_state()

    Tensor.manual_seed(args.seed)
    tokens = Tensor.randint(args.batch, args.seq, low=0, high=cfg.vocab_size).realize()

    if args.mode in ("baseline", "both"):
        run_pythia_baseline(cfg, sd, tokens, args.n_layers)

    if args.mode in ("breathing", "both"):
        run_breathing(cfg, sd, tokens, args.n_loops, label=f"{args.n_loops} loops")


if __name__ == "__main__":
    main()
