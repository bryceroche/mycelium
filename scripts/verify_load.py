"""Sanity check the Pythia weight loader.

Lists state-dict keys, confirms baseline forward runs, computes a perplexity-ish
sanity number on a known input.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinygrad import Tensor, Device, dtypes
from mycelium import Config
from mycelium.loader import _load_state, load_pythia_baseline


def main():
    cfg = Config()
    sd = _load_state()
    print(f"Loaded state dict with {len(sd)} tensors")

    keys = sorted(sd.keys())
    print("\nFirst 5 layer keys:")
    for k in keys[:5]:
        print(f"  {k}: shape={sd[k].shape} dtype={sd[k].dtype}")

    print("\nKey families:")
    families = {}
    for k in keys:
        # Strip layer index
        parts = k.split(".")
        if len(parts) >= 3 and parts[0] == "gpt_neox" and parts[1] == "layers":
            family = "gpt_neox.layers.N." + ".".join(parts[3:])
        else:
            family = k
        families.setdefault(family, []).append(k)

    for fam, ks in sorted(families.items()):
        sample = ks[0]
        print(f"  {fam}: {len(ks)} tensors, sample shape={sd[sample].shape} dtype={sd[sample].dtype}")

    print("\n--- Building baseline (4 layers from Pythia) ---")
    model = load_pythia_baseline(cfg, n_layers=4, sd=sd)
    Device[Device.DEFAULT].synchronize()

    # Smoke forward: shape and basic dynamics
    B, S = 2, 16
    tokens = Tensor.randint(B, S, low=0, high=cfg.vocab_size).realize()
    states = model.hidden_states(tokens)
    Device[Device.DEFAULT].synchronize()
    print(f"\nForward through {len(states) - 1} layers, B={B} S={S}:")
    for i, s in enumerate(states):
        ns = (s.cast(dtypes.float) ** 2).sum().realize()
        norm_sq = float(ns.numpy())
        norm = (norm_sq ** 0.5) / (B * S) ** 0.5
        print(f"  state {i}: shape={s.shape} dtype={s.dtype} avg_norm={norm:.3f}")


if __name__ == "__main__":
    main()
