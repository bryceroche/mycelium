"""Instrument the converged ARITH_HARD controller — what decisions is it making?

Distinguishes two hypotheses:
  (A) Controller learned "constant defaults" — stuck in the trivial basin
      because trivial outputs (everything = 1.0) achieved the loss minimum on
      ARITH_HARD. Per-breath std across the batch will be ~0; means will be
      near the sigmoid midpoints.
  (B) Controller learned structured variation — emits different decisions
      per example and per breath, but the downstream computation ignores them
      (which is what the inference ablations showed). Per-breath std > 0 and
      decisions vary across breaths.

The fixes implied by (A) vs (B) are completely different, so this 5-minute
instrumentation determines the next move.

Usage:
    DEV=PCI+AMD CKPT=.cache/arith_hard_ckpts/arith_hard_v1_step1500.safetensors \\
        .venv/bin/python scripts/instrument_controller.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math
from mycelium.controller import Notebook


def cast_fp32(model):
    def _c(o, a):
        t = getattr(o, a)
        if t.dtype == dtypes.half:
            setattr(o, a, t.cast(dtypes.float).contiguous().realize())
    _c(model.embed, "weight"); _c(model, "embed_out")
    sw = model.block.shared
    for a in ("wv","bv","wo","bo","w_out","b_out"): _c(sw, a)
    for layer in model.block.layers:
        for a in ("wq","bq","wk","bk","w_in","b_in"): _c(layer, a)


def collate_problem_ids(tok, examples, fixed_len):
    """Right-pad each example's tokenized problem to fixed_len with 0s."""
    ids_list = [tok.encode(ex.problem).ids for ex in examples]
    B = len(ids_list)
    out = np.zeros((B, fixed_len), dtype=np.int32)
    for b, ids in enumerate(ids_list):
        L = min(len(ids), fixed_len)
        out[b, :L] = ids[:L]
    return out


def main():
    ckpt = getenv("CKPT", ".cache/arith_hard_ckpts/arith_hard_v1_step1500.safetensors")
    assert os.path.exists(ckpt), f"missing ckpt: {ckpt}"
    B = getenv("B", 32)
    MAX_LOOPS = getenv("MAX_LOOPS", 8)
    SEED = getenv("SEED", 42)
    FIXED_LEN = getenv("FIXED_LEN", 32)

    cfg = Config()
    print(f"=== instrument controller on {os.path.basename(ckpt)} ===")
    print(f"B={B}  max_loops={MAX_LOOPS}  fixed_len={FIXED_LEN}\n")

    print("generating ARITH_HARD problems...")
    examples = generate_math("ARITH_HARD", B + 50, seed=SEED, digit_spacing=True)[:B]
    for i in range(3):
        print(f"  [{i}] {examples[i].problem!r} -> {examples[i].answer}")
    print()

    print("loading model + ckpt...")
    sd = _load_state(); model = load_breathing(cfg, sd=sd); cast_fp32(model); del sd
    sd_ck = safe_load(ckpt)
    info = model.load_state_dict(sd_ck, strict=False)
    print(f"  loaded. missing={len(info['missing'])} unexpected={len(info['unexpected'])}\n")

    tok = load_tokenizer()
    Tensor.training = False

    tokens_np = collate_problem_ids(tok, examples, FIXED_LEN)
    tokens = Tensor(tokens_np, dtype=dtypes.int).realize()

    notebook = Notebook()
    t0 = time.perf_counter()
    final_hidden, decisions_per_breath, n_breaths, match_weights = model.breathe_controlled(
        tokens, max_loops=MAX_LOOPS, notebook=notebook,
    )
    final_hidden.realize()
    Device[Device.DEFAULT].synchronize()
    print(f"breathe_controlled: {time.perf_counter()-t0:.1f}s, n_breaths={n_breaths}\n")

    # decisions_per_breath has max_loops+1 entries (initial + per-breath)
    # Each entry: dict with temperature, gate, step_mult, stop_logit (all (B,) tensors)
    keys = ["temperature", "gate", "step_mult", "stop_logit"]
    # Pull all to numpy: shape (n_decisions, B) per key
    stats = {k: np.stack([d[k].numpy() for d in decisions_per_breath], axis=0) for k in keys}
    n_dec = stats["temperature"].shape[0]

    print(f"--- per-breath statistics across batch of {B} examples ---\n")
    for k in keys:
        arr = stats[k]                            # (n_dec, B)
        mean_per_dec = arr.mean(axis=1)
        std_per_dec = arr.std(axis=1)
        # Range of means across breaths — does the controller modulate over time?
        breath_range = mean_per_dec.max() - mean_per_dec.min()
        # Per-example variation across breaths (avg std over breaths, per example)
        per_ex_breath_std = arr.std(axis=0).mean()   # (B,) → scalar
        print(f"  {k}:")
        header = "    breath: " + "  ".join(f"{i:6d}" for i in range(n_dec))
        print(header)
        print(f"    mean:   " + "  ".join(f"{v:+.3f}" for v in mean_per_dec))
        print(f"    std:    " + "  ".join(f"{v:.3f}" for v in std_per_dec))
        print(f"    range of means across breaths: {breath_range:.3f}")
        print(f"    avg per-example std across breaths: {per_ex_breath_std:.3f}")
        print()

    # Verdict
    print("--- verdict ---")
    print()
    for k in keys:
        arr = stats[k]
        std_within_batch = arr.std(axis=1).max()
        std_across_breaths = arr.std(axis=0).mean()
        # Heuristic thresholds
        if std_within_batch < 0.01 and std_across_breaths < 0.01:
            verdict = "STUCK (constant) — hypothesis (A): never left trivial basin"
        elif std_within_batch < 0.05 and std_across_breaths < 0.05:
            verdict = "near-constant — small variation, probably noise"
        else:
            verdict = "STRUCTURED variation — hypothesis (B): controller modulates but ablation showed it's ignored"
        print(f"  {k:15s}: within-batch std max={std_within_batch:.3f}  across-breath std avg={std_across_breaths:.3f}  → {verdict}")

    print()
    print(f"Notes:")
    print(f"  - temperature ∈ (0.5, 2.0) via sigmoid * 1.5 + 0.5, default midpoint 1.25")
    print(f"  - gate ∈ (0.0, 1.0) via sigmoid, default midpoint 0.5")
    print(f"  - step_mult ∈ (0.5, 2.0) via sigmoid * 1.5 + 0.5, default midpoint 1.25")
    print(f"  - stop_logit raw — sign matters (positive = stop)")


if __name__ == "__main__":
    main()
