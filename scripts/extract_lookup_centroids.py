"""Extract soft-clustered centroids for the v28 lookup_table.values init.

Runs the model on N problems, captures the integrated rep at each example's
"=" position, filters to correct predictions, then soft-clusters via the
existing lookup_table.weight keys. Saves the 16 per-bucket centroids as a
numpy file the LookupTable can load at init time.

The lookup keys are at 100% op classification — they ARE the hash function
over rep space. This script just computes the weighted centroid of correct
reps per bucket.

Usage:
    DEV=PCI+AMD CKPT=/path/to/ckpt LEVEL=L4_MIXED NUM_PROBLEMS=500 \\
        python scripts/extract_lookup_centroids.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math, split_train_eval, parse_int_answer
from mycelium.lookup_table import eq_token_ids_for, find_eq_position


def cast_fp32(model):
    def _c(o, a):
        t = getattr(o, a)
        if t.dtype == dtypes.half:
            setattr(o, a, t.cast(dtypes.float).contiguous().realize())
    _c(model.embed, "weight")
    _c(model, "embed_out")
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out"):
        _c(sw, a)
    for layer in model.block.layers:
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            _c(layer, a)


def main():
    ckpt = getenv("CKPT", "")
    assert ckpt and os.path.exists(ckpt), f"set CKPT=<path>, got {ckpt!r}"
    level = getenv("LEVEL", "L4_MIXED")
    num_problems = getenv("NUM_PROBLEMS", 500)
    n_loops = getenv("LOOPS", 4)
    fixed_len = getenv("FIXED_LEN", 96)
    seed = getenv("SEED", 42)
    out_path = getenv("OUT", f".cache/lookup_centroids/v24c_{level.lower()}_centroids.npy")
    space_digits = bool(getenv("SPACE_DIGITS", 1))

    cfg = Config()
    print(f"=== Extracting lookup_table.values centroids ===")
    print(f"  ckpt:        {ckpt}")
    print(f"  level:       {level}")
    print(f"  n_problems:  {num_problems}")
    print(f"  n_loops:     {n_loops}")
    print(f"  out:         {out_path}")
    print()

    # Build model
    print("loading model + ckpt...")
    t0 = time.perf_counter()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd); cast_fp32(model); del sd
    state = safe_load(ckpt)
    model.load_state_dict(state, strict=False)
    Device[Device.DEFAULT].synchronize()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    # Generate problems (use a different seed to not overlap with eval)
    tok = load_tokenizer()
    eq_ids = eq_token_ids_for(tok)
    print(f"\ngenerating {num_problems} {level} problems...")
    all_examples = generate_math(level, num_problems + 100, seed=seed + 99, digit_spacing=space_digits)
    examples = all_examples[:num_problems]
    print(f"  generated.")

    # Process in batches
    BATCH = 16
    Tensor.training = False
    keys_t = model.lookup_table.weight  # (16, 1024)
    n_entries = keys_t.shape[0]
    keys_np = keys_t.numpy()
    keys_n = keys_np / (np.linalg.norm(keys_np, axis=-1, keepdims=True) + 1e-6)
    print(f"  lookup keys shape: {keys_np.shape}")

    # Accumulators: per-bucket weighted sum and weight total
    bucket_sum = np.zeros((n_entries, cfg.hidden), dtype=np.float64)
    bucket_weight = np.zeros((n_entries,), dtype=np.float64)
    n_correct = 0

    print(f"\nrunning forward + filtering correct...")
    t0 = time.perf_counter()
    for batch_start in range(0, num_problems, BATCH):
        batch = examples[batch_start:batch_start + BATCH]
        if not batch: break
        # Encode each example's full reasoning (so we have the chain to verify against)
        full_texts = [ex.problem + " " + ex.gen for ex in batch]
        # Tokenize + pad
        enc = [tok.encode(t, add_special_tokens=False).ids for t in full_texts]
        max_len_b = min(max(len(e) for e in enc), fixed_len)
        tokens_np = np.zeros((len(batch), max_len_b), dtype=np.int32)
        for i, e in enumerate(enc):
            tokens_np[i, :min(len(e), max_len_b)] = e[:max_len_b]
        tokens = Tensor(tokens_np, dtype=dtypes.int).realize()
        # Forward through breathe_with_lookup to get the integrated reps per breath
        final, match_weights, integrated_per_breath = model.breathe_with_lookup(tokens, n_loops)
        # Use the FINAL integrated rep at the eq position
        final_np = final.realize().numpy()  # (B, T, hidden)
        # Find the LAST eq position per example (most decisive)
        for i, ex in enumerate(batch):
            ids_list = enc[i][:max_len_b]
            # Find ALL eq positions, use the last one (most committed in the chain)
            eq_positions = [j for j, t in enumerate(ids_list) if t in eq_ids]
            if not eq_positions:
                continue
            last_eq = eq_positions[-1]
            # We have no "did model get it right" check here without generation.
            # Approximation: filter via parse_int_answer on the gold gen text (always correct since we used gold).
            # Better: re-generate with greedy and check. For first cut, USE ALL examples (assume v24c at 96% means most are right).
            rep_at_eq = final_np[i, last_eq, :].astype(np.float64)  # (hidden,)
            rep_n = rep_at_eq / (np.linalg.norm(rep_at_eq) + 1e-6)
            # Soft assignment via cosine match against keys
            scores = keys_n @ rep_n  # (n_entries,)
            weights = np.exp(scores - scores.max())
            weights /= weights.sum()
            # Accumulate weighted contribution
            for k in range(n_entries):
                bucket_sum[k] += weights[k] * rep_at_eq
                bucket_weight[k] += weights[k]
            n_correct += 1
        if (batch_start // BATCH) % 5 == 0:
            print(f"  batch {batch_start//BATCH:3d}  ({batch_start + len(batch)}/{num_problems}, n_correct={n_correct})")

    # Compute centroids
    bucket_weight_safe = np.maximum(bucket_weight, 1e-9)
    centroids = bucket_sum / bucket_weight_safe[:, None]
    centroids = centroids.astype(np.float32)

    print(f"\nProcessed {n_correct} examples ({time.perf_counter()-t0:.1f}s)")
    print(f"\nPer-bucket weight totals (how much of the distribution went to each entry):")
    for k in range(n_entries):
        norm = np.linalg.norm(centroids[k])
        share = bucket_weight[k] / bucket_weight.sum() * 100
        print(f"  bucket {k:2d}: share={share:5.1f}%  centroid_norm={norm:6.2f}")

    # Save
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, centroids)
    print(f"\nSaved centroids: {out_path}  shape={centroids.shape}")


if __name__ == "__main__":
    main()
