"""Per-layer effective rank diagnostic for v66 step 3000.

Validates (or refutes) the "128d straw" hypothesis: are shared weights
constraining all layers to a common ~128d subspace?

Captures layer outputs at each of the 4 phase layers + at start (post-embed)
across multiple breaths, computes PCA energy concentration at each.

Output:
  If all ≈ 128d → shared weights ARE the bottleneck (independent weights unlock dims)
  If increasing rank with layer depth → layers stack dims, shared weights not limiting
  If layer 0 high, layer 3 low → progressive concentration (current arch already expand-collapse)
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load


def env_v66():
    os.environ.setdefault('DEV', 'PCI+AMD')
    os.environ['NOTEBOOK_DAG'] = '0'
    os.environ['TWO_PHASE'] = '0'
    os.environ['PROMPT_REFRESH_ALPHA'] = '0.1'
    os.environ['BOUNDARY_AUX_WEIGHT'] = '0.1'
    os.environ['CONTROLLER_DECODE'] = '1'
    os.environ['CONTROLLER_N_LAYERS'] = '2'
    os.environ['PER_BREATH_DECODE'] = '1'
    os.environ['BFIELD_WAIST'] = '512'
    os.environ['BFIELD_END_OF_BREATH'] = '1'
    os.environ['WAIST_CODEBOOK_N'] = '64'
    os.environ['WAIST_CODEBOOK_INJECT_WEIGHT'] = '1.0'
    os.environ['NOTEBOOK_V24'] = '1'
    os.environ['NOTEBOOK_ACCUMULATE_ENABLED'] = '0'
    os.environ['NOTEBOOK_DUAL'] = '1'
    os.environ['NOTEBOOK_POOL_MODE'] = 'attn'
    os.environ['PER_HEAD_PITCH'] = '1'
    os.environ['SINE_TEMP'] = '1'
    os.environ['SINE_TEMP_MAX'] = '2.0'
    os.environ['SINE_TEMP_MIN'] = '0.7'
    os.environ['CONSTANT_RADIUS'] = '1'
    os.environ['BREATH_TIME_EMBED'] = '1'
    os.environ['CROSS_BREATH_HANDOFF'] = '1'
    os.environ['ABLATE_BREATH_ROTATION'] = '1'


def effective_rank_stats(reps):
    """Given (N, H) reps, return cumulative energy at K=64, 128, 256, 512, 768."""
    reps = reps - reps.mean(axis=0)
    # SVD on the centered matrix
    if reps.shape[0] > 5000:
        idx = np.random.RandomState(42).choice(reps.shape[0], 5000, replace=False)
        reps = reps[idx]
    U, S, Vt = np.linalg.svd(reps, full_matrices=False)
    total_energy = (S ** 2).sum()
    cum = (S ** 2).cumsum() / total_energy
    # rank at 99% energy
    rank_99 = int(np.argmax(cum >= 0.99)) + 1
    rank_95 = int(np.argmax(cum >= 0.95)) + 1
    return {
        'K=64': float(cum[63]) if len(cum) > 63 else 1.0,
        'K=128': float(cum[127]) if len(cum) > 127 else 1.0,
        'K=256': float(cum[255]) if len(cum) > 255 else 1.0,
        'K=512': float(cum[511]) if len(cum) > 511 else 1.0,
        'K=768': float(cum[767]) if len(cum) > 767 else 1.0,
        'rank_95': rank_95,
        'rank_99': rank_99,
    }


def main():
    env_v66()
    from mycelium.config import Config
    from mycelium.loader import _load_state, load_breathing
    from mycelium.data import load_tokenizer
    from mycelium.l3_data import load_gsm8k_steps
    from scripts.eval_ckpt_controller_segmented import cast_model_fp32

    cfg = Config()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_model_fp32(model)
    Device[Device.DEFAULT].synchronize()
    del sd
    ckpt = safe_load('.cache/gsm8k_steps_ckpts/v66_sched_sampling_step3000.safetensors')
    info = model.load_state_dict(ckpt, strict=False)
    print(f'Loaded v66: missing={len(info["missing"])}')
    del ckpt
    tok = load_tokenizer()

    # Class-level monkey-patch (dunders can't be patched per-instance).
    # Map each model.block.layers[i] to its index by id().
    from mycelium.breathing import BreathingLayer
    captures = {0: [], 1: [], 2: [], 3: []}
    capture_input = []
    layer_to_idx = {id(layer): i for i, layer in enumerate(model.block.layers)}
    orig_call = BreathingLayer.__call__

    def wrapped_call(self, x, loop_idx, attn_mask=None, temp_mult=1.0, alpha=None):
        li = layer_to_idx.get(id(self), -1)
        if li == 0:
            capture_input.append(x.cast(dtypes.float).numpy().reshape(-1, x.shape[-1]))
        out = orig_call(self, x, loop_idx, attn_mask=attn_mask, temp_mult=temp_mult, alpha=alpha)
        if li >= 0:
            captures[li].append(out.cast(dtypes.float).numpy().reshape(-1, out.shape[-1]))
        return out

    BreathingLayer.__call__ = wrapped_call

    # Run on 100 train examples
    all_examples = load_gsm8k_steps('.cache/gsm8k_steps_v1_train.jsonl', min_k=2, max_k=6, bucket_by_k=False)
    examples = all_examples[:100]
    Tensor.training = False

    print(f"Running v66 forward on {len(examples)} examples...")
    t0 = time.perf_counter()
    BATCH = 2
    FIXED_LEN = 320
    K_LOOPS = 3
    for b_start in range(0, len(examples), BATCH):
        batch = examples[b_start:b_start + BATCH]
        prompts = [tok.encode(ex.problem).ids for ex in batch]
        tokens_np = np.zeros((len(batch), FIXED_LEN), dtype=np.int32)
        for i, p in enumerate(prompts):
            tokens_np[i, :len(p)] = p
        tokens = Tensor(tokens_np, dtype=dtypes.int).realize()
        _ = model.breathe_with_lookup(tokens, n_loops=K_LOOPS, return_per_breath_x=True)

    # Restore class-level method
    BreathingLayer.__call__ = orig_call
    print(f"  done in {time.perf_counter() - t0:.0f}s")
    print(f"  captures: input {len(capture_input)} batches, layer0 {len(captures[0])}, layer3 {len(captures[3])}")

    # Concat and compute rank
    print()
    print(f'{"position":>20} | {"K=64":>6} {"K=128":>6} {"K=256":>6} {"K=512":>6} {"K=768":>6} | {"rank_95":>7} {"rank_99":>7}')
    print('-' * 90)

    pre_layer0 = np.concatenate(capture_input, axis=0)
    print(f'{"pre layer0 (input)":>20} |', end='')
    s = effective_rank_stats(pre_layer0)
    print(f' {s["K=64"]:>6.3f} {s["K=128"]:>6.3f} {s["K=256"]:>6.3f} {s["K=512"]:>6.3f} {s["K=768"]:>6.3f} | {s["rank_95"]:>7d} {s["rank_99"]:>7d}')

    for li in range(4):
        layer_out = np.concatenate(captures[li], axis=0)
        s = effective_rank_stats(layer_out)
        print(f'{"post layer" + str(li):>20} |', end='')
        print(f' {s["K=64"]:>6.3f} {s["K=128"]:>6.3f} {s["K=256"]:>6.3f} {s["K=512"]:>6.3f} {s["K=768"]:>6.3f} | {s["rank_95"]:>7d} {s["rank_99"]:>7d}')

    # Interpretation hint
    print()
    print("Interpretation:")
    print("  - If all ranks ≈ 128 → shared weights bottleneck (v69 independent weights help)")
    print("  - If increasing rank with depth → layers stack dims, shared not limiting")
    print("  - If layer 0 high, layer 3 low → progressive concentration (architecture already expand-collapse)")


if __name__ == "__main__":
    main()
