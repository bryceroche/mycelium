"""Per-op centroid extraction for v28c.

DECOMPOSITION-based extraction: for each correct example, extract reps at the
"=" position of EACH cycle (not just the final one). Hard-assign each cycle's
rep to its gold op label (0=+, 1=-, 2=*, 3=/). Compute centroid per op.

Key difference vs v28b's soft-clustered centroids:
- Soft-clustered (v28b): all cycles' final-rep grouped via softmax(rep · keys),
  produces near-identical centroids because reps live in narrow math-problem cone.
- Hard-assigned per-op (v28c): EACH CYCLE'S rep grouped by ground-truth op label
  from the cycle's reasoning text. Centroids reflect "rep when doing ADD" vs
  "rep when doing SUB" — operations with genuinely different internal procedures.

Usage:
    DEV=PCI+AMD CKPT=/path/to/ckpt LEVEL=L4_MIXED NUM_PROBLEMS=500 \\
        python scripts/extract_per_op_centroids.py
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
from mycelium.l3_data import generate_math, encode_cycles, parse_int_answer
from mycelium.lookup_table import (
    eq_token_ids_for, find_eq_position, op_label_from_text,
)


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
    out_path = getenv("OUT", f".cache/lookup_centroids/v24c_{level.lower()}_per_op_centroids.npy")
    space_digits = bool(getenv("SPACE_DIGITS", 1))

    cfg = Config()
    print(f"=== Per-op centroid extraction (v28c) ===")
    print(f"  ckpt:        {ckpt}")
    print(f"  level:       {level}")
    print(f"  n_problems:  {num_problems}")
    print(f"  n_loops:     {n_loops}")
    print(f"  out:         {out_path}")
    print()

    print("loading model + ckpt...")
    t0 = time.perf_counter()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd); cast_fp32(model); del sd
    state = safe_load(ckpt)
    model.load_state_dict(state, strict=False)
    Device[Device.DEFAULT].synchronize()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    tok = load_tokenizer()
    eq_ids = eq_token_ids_for(tok)
    print(f"\ngenerating {num_problems} {level} problems...")
    all_examples = generate_math(level, num_problems + 200, seed=seed + 99,
                                   digit_spacing=space_digits)
    examples = all_examples[:num_problems]
    print(f"  generated.")

    # Per-op accumulators. 4 ops: 0=+, 1=-, 2=*, 3=/
    n_ops = 4
    op_sum = np.zeros((n_ops, cfg.hidden), dtype=np.float64)
    op_count = np.zeros((n_ops,), dtype=np.int64)
    op_names = ["+", "-", "*", "/"]

    BATCH = 16
    Tensor.training = False
    n_cycles_processed = 0

    print(f"\nrunning per-cycle forward + per-op grouping...")
    t0 = time.perf_counter()
    # We process ONE problem at a time for simplicity (each problem has 2+ cycles).
    # Could batch by stacking cycles, but for ~500 problems × 2 cycles this is fine.
    for ex_idx, ex in enumerate(examples):
        cycle_encodings = encode_cycles(tok, ex)
        for cyc_idx, (ids_list, prefix_len, total_len) in enumerate(cycle_encodings):
            # Determine op from cycle's gen_target text
            cycle_text = ex.gen_targets[cyc_idx] if cyc_idx < len(ex.gen_targets) else ""
            op_label = op_label_from_text(cycle_text)
            if op_label < 0 or op_label >= n_ops:
                continue  # skip cycles with no detectable op

            # Pad to fixed_len and run forward
            ids = ids_list[:fixed_len]
            tokens_np = np.zeros((1, fixed_len), dtype=np.int32)
            tokens_np[0, :len(ids)] = ids
            tokens = Tensor(tokens_np, dtype=dtypes.int).realize()

            final, _, _ = model.breathe_with_lookup(tokens, n_loops)
            final_np = final.realize().numpy()[0]   # (T, hidden)

            # Find the "=" position WITHIN the target span (after prefix_len)
            target_span = ids[prefix_len:]
            eq_offset = -1
            for i, t in enumerate(target_span):
                if t in eq_ids:
                    eq_offset = i
                    break
            if eq_offset < 0:
                continue  # no = found in target
            eq_pos = prefix_len + eq_offset
            if eq_pos >= fixed_len:
                continue

            rep = final_np[eq_pos, :].astype(np.float64)
            op_sum[op_label] += rep
            op_count[op_label] += 1
            n_cycles_processed += 1
        if (ex_idx + 1) % 50 == 0:
            print(f"  examples {ex_idx+1}/{num_problems}  cycles={n_cycles_processed}  per_op_counts={op_count.tolist()}")

    centroids = np.zeros((cfg.n_lookup_entries, cfg.hidden), dtype=np.float32)
    for op_i in range(n_ops):
        if op_count[op_i] > 0:
            centroids[op_i] = (op_sum[op_i] / op_count[op_i]).astype(np.float32)
    # Entries 4-15: keep as small random init (they correspond to ops we don't have data for)
    rng = np.random.default_rng(11)
    for i in range(n_ops, cfg.n_lookup_entries):
        v = rng.standard_normal(cfg.hidden).astype(np.float32) * 0.02
        # Scale to match the random init norm we'd otherwise use
        centroids[i] = v

    print(f"\nProcessed {n_cycles_processed} cycles ({time.perf_counter()-t0:.1f}s)")
    print(f"\nPer-op centroid summary:")
    for op_i in range(n_ops):
        norm = np.linalg.norm(centroids[op_i])
        print(f"  op {op_i} ({op_names[op_i]:3s}): count={op_count[op_i]:4d}  centroid_norm={norm:6.2f}")
    print(f"  ops 4-15:  filled with random N(0, 0.02²) init")

    # Pairwise cosine similarity (the diagnostic we care about)
    print(f"\nPairwise cosine similarity of op centroids (entries 0-3):")
    norms = np.linalg.norm(centroids[:n_ops], axis=-1, keepdims=True) + 1e-9
    c_n = centroids[:n_ops] / norms
    sim = c_n @ c_n.T
    print(f"     {'  '.join(op_names)}")
    for i in range(n_ops):
        row = '  '.join([f'{sim[i,j]:+.2f}' for j in range(n_ops)])
        print(f"  {op_names[i]}: {row}")
    off_diag = sim[~np.eye(n_ops, dtype=bool)]
    print(f"\n  off-diag mean: {off_diag.mean():+.3f}  (vs 0.986 for v28b non-decomposed)")
    print(f"  off-diag min:  {off_diag.min():+.3f}")
    print(f"  off-diag max:  {off_diag.max():+.3f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, centroids)
    print(f"\nSaved centroids: {out_path}  shape={centroids.shape}")


if __name__ == "__main__":
    main()
