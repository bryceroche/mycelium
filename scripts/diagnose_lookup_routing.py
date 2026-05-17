"""Diagnose the lookup table routing: is the softmax too uniform?

Loads a trained ckpt, runs forward on some L4_MIXED problems, and reports the
softmax entropy and max-weight statistics of the lookup retrieve step.

Interpretation:
- max_weight ~ 1/n_entries (0.0625 for n=16): UNIFORM — softmax is killing routing
- max_weight > 0.3: peaked — routing IS sharp
- entropy near log(n_entries) (2.77 for n=16): uniform
- entropy near 0: peaked

Usage:
    DEV=PCI+AMD CKPT=/path/to/ckpt python scripts/diagnose_lookup_routing.py
"""
import os
import sys
import time

os.environ.setdefault("DEV", "PCI+AMD")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math, encode_cycles, parse_int_answer
from mycelium.lookup_table import eq_token_ids_for, op_label_from_text


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
    ckpt = getenv("CKPT", "/home/bryce/mycelium/.cache/arith_ckpts/arith_v34_lda_lookup_step1500.safetensors")
    level = getenv("LEVEL", "L4_MIXED")
    num_problems = getenv("NUM_PROBLEMS", 50)
    n_loops = getenv("LOOPS", 4)
    fixed_len = getenv("FIXED_LEN", 96)
    seed = getenv("SEED", 42)

    cfg = Config()
    print(f"=== Lookup routing diagnostic ===")
    print(f"  ckpt: {os.path.basename(ckpt)}")
    print(f"  level: {level}  N: {num_problems}")
    print()

    print("loading model + ckpt...")
    t0 = time.perf_counter()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd); cast_fp32(model); del sd
    state = safe_load(ckpt)
    info = model.load_state_dict(state, strict=False)
    Device[Device.DEFAULT].synchronize()
    print(f"  loaded ({time.perf_counter()-t0:.1f}s)")
    if info.get("missing"):
        print(f"  missing keys: {info['missing'][:5]}")

    tok = load_tokenizer()
    eq_ids = eq_token_ids_for(tok)

    print(f"\ngenerating {num_problems} {level} problems...")
    examples = generate_math(level, num_problems + 50, seed=seed + 99, digit_spacing=True)
    examples = examples[:num_problems]

    Tensor.training = False
    print(f"\nrunning forwards + capturing match weights...")
    all_max_weights = []
    all_entropies = []
    all_top1_ops = []
    gold_ops = []
    n_cycles = 0

    for ex_idx, ex in enumerate(examples):
        cycle_encodings = encode_cycles(tok, ex)
        for cyc_idx, (ids_list, prefix_len, total_len) in enumerate(cycle_encodings):
            cycle_text = ex.gen_targets[cyc_idx] if cyc_idx < len(ex.gen_targets) else ""
            op_label = op_label_from_text(cycle_text)
            if op_label < 0 or op_label >= 4:
                continue
            ids = ids_list[:fixed_len]
            tokens_np = np.zeros((1, fixed_len), dtype=np.int32)
            tokens_np[0, :len(ids)] = ids
            tokens = Tensor(tokens_np, dtype=dtypes.int).realize()

            # Get the final rep AND compute lookup match weights
            final, _, _ = model.breathe_with_lookup(tokens, n_loops)
            scores = model.lookup_table(final).realize().numpy()[0]  # (T, n_entries)

            # Find = position
            target_span = ids[prefix_len:]
            eq_offset = -1
            for i, t in enumerate(target_span):
                if t in eq_ids:
                    eq_offset = i
                    break
            if eq_offset < 0:
                continue
            eq_pos = prefix_len + eq_offset
            if eq_pos >= fixed_len:
                continue

            # Apply softmax at this position WITH temperature
            T = float(os.environ.get("LOOKUP_TEMP", "1.0"))
            score_at_eq = scores[eq_pos] * T  # (n_entries,)
            w = np.exp(score_at_eq - score_at_eq.max())
            w = w / w.sum()
            max_weight = w.max()
            entropy = -np.sum(w * np.log(np.maximum(w, 1e-12)))
            top1 = int(np.argmax(w))
            all_max_weights.append(max_weight)
            all_entropies.append(entropy)
            all_top1_ops.append(top1)
            gold_ops.append(op_label)
            n_cycles += 1

    all_max_weights = np.array(all_max_weights)
    all_entropies = np.array(all_entropies)
    all_top1_ops = np.array(all_top1_ops)
    gold_ops = np.array(gold_ops)

    n_entries = cfg.n_lookup_entries
    uniform_weight = 1.0 / n_entries
    uniform_entropy = np.log(n_entries)

    print(f"\nProcessed {n_cycles} cycles")
    print(f"\n=== Softmax routing statistics (at '=' position) ===")
    print(f"  uniform max-weight (baseline):    {uniform_weight:.4f}")
    print(f"  observed max-weight (mean ± std): {all_max_weights.mean():.4f} ± {all_max_weights.std():.4f}")
    print(f"  observed max-weight (min, max):   ({all_max_weights.min():.4f}, {all_max_weights.max():.4f})")
    print()
    print(f"  uniform entropy (baseline):       {uniform_entropy:.4f}")
    print(f"  observed entropy (mean ± std):    {all_entropies.mean():.4f} ± {all_entropies.std():.4f}")
    print()
    print(f"INTERPRETATION:")
    if all_max_weights.mean() < 2 * uniform_weight:
        print(f"  ⚠️  Softmax is NEAR-UNIFORM. Routing is effectively averaging all entries.")
        print(f"     Fix: scale scores before softmax (temperature).")
    elif all_max_weights.mean() > 0.5:
        print(f"  ✓ Softmax is SHARP. Routing IS picking specific entries.")
    else:
        print(f"  ~ Softmax is MODERATE. Some routing, some blending.")

    # Top-1 routing accuracy (does the lookup pick the correct op?)
    # Note: only meaningful if first 4 entries are the 4 ops in order
    print(f"\n=== Top-1 entry vs gold op label (op_idx 0=+ 1=- 2=* 3=/) ===")
    op_names = ['+', '-', '*', '/', 'other_4_15']
    for op_i in range(4):
        mask = gold_ops == op_i
        if mask.sum() == 0:
            continue
        top1_for_op = all_top1_ops[mask]
        bins = np.bincount(top1_for_op, minlength=n_entries)
        top3_idx = np.argsort(-bins)[:3]
        line = f"  gold {op_names[op_i]} ({mask.sum()} cycles): "
        line += f"top-1 entry distribution top-3 = {top3_idx.tolist()} ({bins[top3_idx].tolist()})"
        print(line)


if __name__ == "__main__":
    main()
