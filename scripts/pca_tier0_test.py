"""Tier 0 of v69 collapse design: zero-learned-params PCA top-K test.

1. Extract v66's pre-waist x reps from ~1000 GSM8K train examples
2. Compute PCA basis (1024, 1024)
3. Eval v66 on GSM8K test with apply_bfield_waist replaced by PCA→top-K→inverse-PCA
4. Compare K=128, K=256, K=512 to v66's 2.7% baseline

If K=128 ≈ 2.7%: 87.5% of waist is discardable → build full JPEG pipeline
If K=128 drops, K=256 holds: moderate compression viable
If both drop: PCA isn't the right basis (nonlinear structure), need codebook
"""
import os
import sys
import time
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load, safe_save


def env_v66_arch():
    """Set env vars matching v66's architecture."""
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


def extract_reps(model, tok, n_examples=1000, fixed_len=320, K_loops=3):
    """Run v66 on n_examples and collect the x_in to apply_bfield_waist.
    Returns (N, hidden) numpy array where N ≈ n_examples * K_loops * fixed_len.

    Strategy: monkey-patch apply_bfield_waist to record its input, then run forward.
    """
    from mycelium.l3_data import load_gsm8k_steps, split_train_eval
    all_examples = load_gsm8k_steps('.cache/gsm8k_steps_v1_train.jsonl', min_k=2, max_k=6, bucket_by_k=False)
    # Use train data — these are what v66 was trained on
    examples = all_examples[:n_examples]

    captured_reps = []

    orig_apply = model.block.apply_bfield_waist
    def capturing_apply(x, return_compressed=False):
        # Capture x as numpy array
        x_np = x.cast(dtypes.float).numpy()
        # Flatten (B, T, H) → (B*T, H) and append
        H = x_np.shape[-1]
        captured_reps.append(x_np.reshape(-1, H))
        return orig_apply(x, return_compressed=return_compressed)

    model.block.apply_bfield_waist = capturing_apply

    print(f"Extracting reps from {n_examples} examples...")
    Tensor.training = False
    t0 = time.perf_counter()
    BATCH = 2
    for b_start in range(0, n_examples, BATCH):
        batch = examples[b_start:b_start + BATCH]
        prompts = [tok.encode(ex.problem).ids for ex in batch]
        # Pad to fixed_len, only forward (no decode)
        tokens_np = np.zeros((len(batch), fixed_len), dtype=np.int32)
        for i, p in enumerate(prompts):
            tokens_np[i, :len(p)] = p
        tokens = Tensor(tokens_np, dtype=dtypes.int).realize()
        _ = model.breathe_with_lookup(tokens, n_loops=K_loops, return_per_breath_x=True)
        if (b_start // BATCH) % 50 == 0:
            print(f"  {b_start}/{n_examples} ({time.perf_counter() - t0:.0f}s elapsed)", flush=True)

    model.block.apply_bfield_waist = orig_apply
    reps = np.concatenate(captured_reps, axis=0)
    print(f"Extracted {reps.shape[0]} representations of dim {reps.shape[1]} in {time.perf_counter() - t0:.0f}s")
    return reps


def compute_pca(reps, max_samples=10000):
    """Compute PCA basis from extracted reps.
    Returns (basis: (H, H), singular_values: (H,)).
    """
    # Subsample if too many
    if reps.shape[0] > max_samples:
        idx = np.random.RandomState(42).choice(reps.shape[0], max_samples, replace=False)
        reps = reps[idx]
    # Center
    mean = reps.mean(axis=0)
    centered = reps - mean
    # SVD: centered = U @ diag(S) @ V.T; V is the PCA basis (columns are principal components)
    print(f"Computing SVD on {centered.shape} matrix...")
    t0 = time.perf_counter()
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    V = Vt.T  # (H, H) — columns are principal components
    print(f"  SVD done in {time.perf_counter() - t0:.0f}s, top-5 singular values: {S[:5]}")
    print(f"  cumulative energy: K=128: {(S[:128]**2).sum() / (S**2).sum():.3f}, K=256: {(S[:256]**2).sum() / (S**2).sum():.3f}, K=512: {(S[:512]**2).sum() / (S**2).sum():.3f}")
    return V.astype(np.float32), S.astype(np.float32), mean.astype(np.float32)


def eval_with_pca_topk(model, tok, pca_basis_np, pca_mean_np, K, num_eval=300, fixed_len=400, batch=2, max_new=120):
    """Replace apply_bfield_waist with PCA top-K and eval."""
    from mycelium.l3_data import load_gsm8k_steps, split_train_eval, parse_int_answer
    from scripts.eval_ckpt_controller_segmented import segmented_generate_kv_batch, segmented_generate_batch

    H = pca_basis_np.shape[0]
    pca_basis = Tensor(pca_basis_np, dtype=dtypes.float).realize()
    pca_basis_T = Tensor(pca_basis_np.T.copy(), dtype=dtypes.float).realize()
    pca_mean = Tensor(pca_mean_np, dtype=dtypes.float).realize()
    # Mask: (H,) with 1.0 in first K, 0.0 elsewhere
    mask_np = np.zeros(H, dtype=np.float32)
    mask_np[:K] = 1.0
    keep_mask = Tensor(mask_np, dtype=dtypes.float).realize()

    orig_apply = model.block.apply_bfield_waist

    def pca_topk_apply(x, return_compressed=False):
        x_f = x.cast(dtypes.float)
        # Center
        x_centered = x_f - pca_mean.reshape(1, 1, -1)
        # Project to PCA basis
        x_pca = x_centered @ pca_basis  # (B, T, H)
        # Top-K: multiply by mask (zero bottom dims)
        x_quantized = x_pca * keep_mask.reshape(1, 1, -1)
        # Inverse PCA
        x_recon = x_quantized @ pca_basis_T
        # Add mean back
        out = (x_recon + pca_mean.reshape(1, 1, -1)).cast(x.dtype)
        if return_compressed:
            # Return first 512 PCA coords (post-mask) — matches v66's bf_w=512 for downstream consumers.
            # At K=128, first 128 are non-zero, next 384 are zero (from mask).
            compressed = x_quantized[:, :, :512].cast(x.dtype)
            return out, compressed
        return out

    model.block.apply_bfield_waist = pca_topk_apply

    # Run eval
    all_examples = load_gsm8k_steps('.cache/gsm8k_steps_v1_test.jsonl', min_k=2, max_k=6, bucket_by_k=False)
    _, eval_examples = split_train_eval(all_examples, n_eval=num_eval, seed=42)

    examples_by_k = {}
    for ex in eval_examples:
        examples_by_k.setdefault(len(ex.gen_targets), []).append(ex)

    Tensor.training = False
    correct = 0
    total = 0
    per_k = {}
    t0 = time.perf_counter()
    for group_K in sorted(examples_by_k):
        group = examples_by_k[group_K]
        gc, gt = 0, 0
        for b_start in range(0, len(group), batch):
            batch_exs = group[b_start:b_start + batch]
            prompt_ids = [tok.encode(ex.problem).ids for ex in batch_exs]
            gen_per_ex = segmented_generate_batch(model, prompt_ids, tok, K=group_K,
                                                    fixed_len=fixed_len, max_new=max_new)
            for i, ex in enumerate(batch_exs):
                gen_text = tok.decode(gen_per_ex[i])
                parsed = parse_int_answer(gen_text)
                ok = (parsed == ex.answer)
                if ok:
                    correct += 1; gc += 1
                total += 1; gt += 1
        per_k[group_K] = (gc, gt)
        print(f"  K={group_K}: {gc}/{gt}", flush=True)

    model.block.apply_bfield_waist = orig_apply
    dt = time.perf_counter() - t0
    acc = correct / max(total, 1) * 100
    print(f"\n=== PCA top-K (K={K}) acc: {acc:.1f}% ({correct}/{total})  ({dt:.0f}s) ===")
    print("per-K breakdown:")
    for k, (c, t) in sorted(per_k.items()):
        pct = c / max(t, 1) * 100
        print(f"  K={k}: {pct:.1f}% ({c}/{t})")
    return acc, per_k


def main():
    env_v66_arch()

    from mycelium.config import Config
    from mycelium.loader import _load_state, load_breathing
    from mycelium.data import load_tokenizer
    from scripts.eval_ckpt_controller_segmented import cast_model_fp32

    cfg = Config()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_model_fp32(model)
    Device[Device.DEFAULT].synchronize()
    del sd
    ckpt = safe_load('.cache/gsm8k_steps_ckpts/v66_sched_sampling_step3000.safetensors')
    info = model.load_state_dict(ckpt, strict=False)
    print(f'Loaded v66 step 3000: missing={len(info["missing"])}, unexpected={len(info["unexpected"])}')
    del ckpt
    tok = load_tokenizer()

    pca_path = '/tmp/v66_pca_basis.npz'
    if os.path.exists(pca_path):
        print(f"Loading cached PCA basis from {pca_path}")
        data = np.load(pca_path)
        basis = data['basis']
        mean = data['mean']
        sing = data['singular_values']
        print(f"  basis: {basis.shape}, cumulative energy K=128: {(sing[:128]**2).sum() / (sing**2).sum():.3f}, K=256: {(sing[:256]**2).sum() / (sing**2).sum():.3f}")
    else:
        # Extract reps
        N_EXAMPLES = int(os.environ.get("PCA_N_EXAMPLES", "200"))
        reps = extract_reps(model, tok, n_examples=N_EXAMPLES, fixed_len=320, K_loops=3)
        # Save raw reps (optional)
        np.savez('/tmp/v66_reps.npz', reps=reps)
        # Compute PCA
        basis, sing, mean = compute_pca(reps, max_samples=10000)
        np.savez(pca_path, basis=basis, singular_values=sing, mean=mean)
        print(f"Saved PCA basis to {pca_path}")

    # Run evals
    Ks = [int(k) for k in os.environ.get("PCA_KS", "128,256,512").split(",")]
    results = {}
    for K in Ks:
        print(f"\n========== Eval with K={K} ==========")
        acc, per_k = eval_with_pca_topk(model, tok, basis, mean, K=K,
                                          num_eval=int(os.environ.get("NUM_EVAL", "300")),
                                          fixed_len=400, batch=2, max_new=120)
        results[K] = {'acc': acc, 'per_k': per_k}

    print(f"\n=== SUMMARY ===")
    print(f"v66 baseline (no PCA): 2.7% (8/300)")
    for K, r in sorted(results.items()):
        print(f"K={K}: {r['acc']:.1f}%")


if __name__ == "__main__":
    main()
