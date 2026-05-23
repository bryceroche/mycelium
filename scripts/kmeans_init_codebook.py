"""K-means codebook initializer for v70 warm-start.

Runs a forward pass on ~200 training examples with a source checkpoint,
captures the WAIST INPUT (pre-collapse) representations, then fits k-means
with k=256 clusters. Outputs a safetensors file with cluster centers that
can be used to warm-start v70's collapse_codebook_keys (and optionally
collapse_codebook_values).

Usage example (shown in docstring, also at bottom of script):

    # 1. Build the codebook init file from v69 step 3000
    CKPT=.cache/gsm8k_steps_ckpts/v69_collapse_step3000.safetensors \\
      SRC_LABEL=v69_collapse_step3000 \\
      python scripts/kmeans_init_codebook.py

    # 2. Merge into a new warm-start ckpt (patch then save)
    python - <<'EOF'
    import numpy as np
    from tinygrad.nn.state import safe_load, safe_save
    from tinygrad import Tensor, dtypes

    src  = safe_load(".cache/gsm8k_steps_ckpts/v69_collapse_step3000.safetensors")
    km   = safe_load(".cache/gsm8k_steps_ckpts/v69_collapse_step3000_kmeans_codebook.safetensors")

    # Overwrite codebook keys with k-means centers
    src["block.collapse_codebook_keys"]   = km["collapse_codebook_keys"]
    # Use zero-init values (LoRA-style: gradient will build them from scratch)
    src["block.collapse_codebook_values"] = km["collapse_codebook_values"]

    safe_save(src, ".cache/gsm8k_steps_ckpts/v70_warm_start.safetensors")
    print("Saved v70_warm_start.safetensors")
    EOF

Env vars:
    CKPT            Source ckpt path (required).
    SRC_LABEL       Label for output filename (default: derived from CKPT basename).
    N_EXTRACT       Number of training examples to extract reps from (default 200).
    N_SUBSAMPLE     Max reps to feed k-means (default 10000). Subsampled if more.
    K_CLUSTERS      Number of cluster centers = codebook size (default 256).
    K_MEANS_ITERS   Fixed iterations for k-means (default 100).
    K_LOOPS         Breath depth for extraction (default 4).
    FIXED_LEN       Sequence length (default 320).
    BATCH           Batch size for extraction (default 2).
    VALUES_ZERO     If 1 (default), set values = zeros. If 0, set values = centers.
    GSM8K_TRAIN     GSM8K training JSONL path.
    OUT_DIR         Output directory (default .cache/gsm8k_steps_ckpts).
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must set env vars BEFORE mycelium imports (module-level reads)
os.environ.setdefault('COLLAPSE_V69',             '1')
os.environ.setdefault('COLLAPSE_WAIST_DIM',        '128')
os.environ.setdefault('COLLAPSE_CODEBOOK_N',       '256')
os.environ.setdefault('COLLAPSE_TAU',              '1.0')
os.environ.setdefault('COLLAPSE_GATE_BIAS',        '2.0')
os.environ.setdefault('COLLAPSE_ENTROPY_REG',      '0.01')
os.environ.setdefault('TWO_PHASE',                 '0')
os.environ.setdefault('NOTEBOOK_DAG',              '0')
os.environ.setdefault('PROMPT_REFRESH_ALPHA',      '0.1')
os.environ.setdefault('BOUNDARY_AUX_WEIGHT',       '0.1')
os.environ.setdefault('CONTROLLER_DECODE',         '1')
os.environ.setdefault('CONTROLLER_N_LAYERS',       '2')
os.environ.setdefault('PER_BREATH_DECODE',         '1')
os.environ.setdefault('BFIELD_WAIST',              '512')
os.environ.setdefault('BFIELD_END_OF_BREATH',      '1')
os.environ.setdefault('BFIELD_ENFORCED',           '0')
os.environ.setdefault('BFIELD_ALPHA',              '1.0')
os.environ.setdefault('WAIST_CODEBOOK_N',          '64')
os.environ.setdefault('WAIST_CODEBOOK_INJECT_WEIGHT', '1.0')
os.environ.setdefault('NOTEBOOK_V24',              '1')
os.environ.setdefault('NOTEBOOK_ACCUMULATE_ENABLED', '0')
os.environ.setdefault('NOTEBOOK_DUAL',             '1')
os.environ.setdefault('NOTEBOOK_POOL_MODE',        'attn')
os.environ.setdefault('NOTEBOOK_INIT_SCALE',       '0.02')
os.environ.setdefault('STOCH_DEPTH_P',             '0.10')
os.environ.setdefault('LABEL_SMOOTHING',           '0.1')
os.environ.setdefault('WEIGHT_DECAY',              '0.05')
os.environ.setdefault('PER_HEAD_PITCH',            '1')
os.environ.setdefault('SINE_TEMP',                 '1')
os.environ.setdefault('SINE_TEMP_MAX',             '2.0')
os.environ.setdefault('SINE_TEMP_MIN',             '0.7')
os.environ.setdefault('CONSTANT_RADIUS',           '1')
os.environ.setdefault('BREATH_TIME_EMBED',         '1')
os.environ.setdefault('BREATH_TIME_INIT_SCALE',    '0.0')
os.environ.setdefault('CROSS_BREATH_HANDOFF',      '1')
os.environ.setdefault('ABLATE_BREATH_ROTATION',    '1')
os.environ.setdefault('QUADRATURE_HEADS',          '0')
os.environ.setdefault('SCHED_SAMPLE_RATE',         '0.3')
os.environ.setdefault('BOUNDARY_POS_WEIGHT',       '5.0')
os.environ.setdefault('DEV',                       'PCI+AMD')

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load, safe_save

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import load_gsm8k_steps
from scripts.eval_ckpt_controller_segmented import cast_model_fp32


# ---------------------------------------------------------------------------
# Rep extraction — capture x BEFORE apply_collapse_v69
# ---------------------------------------------------------------------------

def extract_pre_collapse_reps(model, tok, examples, K_loops, fixed_len, batch_size):
    """Capture the input to apply_collapse_v69 for each (breath, token) position.

    Monkey-patches model.block.apply_collapse_v69 to record x before the
    collapse pipeline. Each call receives x of shape (B, T, hidden).

    Returns: numpy array (M, hidden) where M = examples × K_loops × T_nonempty.
             Only non-padding positions (token != 0) are included when possible,
             but we don't filter here — the full (B*T) is captured and the
             k-means will naturally weight populated positions more after subsampling.
    """
    captured = []

    orig_apply = model.block.apply_collapse_v69

    def capturing_apply(x, return_compressed=False):
        # Capture pre-collapse input as numpy
        x_np = x.cast(dtypes.float).numpy()  # (B, T, hidden)
        B, T, H = x_np.shape
        captured.append(x_np.reshape(-1, H))
        return orig_apply(x, return_compressed=return_compressed)

    model.block.apply_collapse_v69 = capturing_apply

    Tensor.training = False
    t0 = time.perf_counter()
    print(f"Extracting pre-collapse reps from {len(examples)} examples (K={K_loops})...", flush=True)

    for b_start in range(0, len(examples), batch_size):
        batch = examples[b_start:b_start + batch_size]
        prompts = [tok.encode(ex.problem).ids for ex in batch]
        tokens_np = np.zeros((len(batch), fixed_len), dtype=np.int32)
        for i, p in enumerate(prompts):
            p_trunc = p[:fixed_len]
            tokens_np[i, :len(p_trunc)] = p_trunc
        tokens = Tensor(tokens_np, dtype=dtypes.int).realize()
        _ = model.breathe_with_lookup(tokens, n_loops=K_loops)

        if (b_start // batch_size) % 20 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {b_start}/{len(examples)}  ({elapsed:.0f}s elapsed)", flush=True)

    model.block.apply_collapse_v69 = orig_apply

    reps = np.concatenate(captured, axis=0)  # (M, hidden)
    elapsed = time.perf_counter() - t0
    print(f"Extracted {reps.shape[0]:,} reps of dim {reps.shape[1]} in {elapsed:.0f}s", flush=True)
    return reps


# ---------------------------------------------------------------------------
# Simple k-means (numpy, fixed iterations, no fancy convergence check)
# ---------------------------------------------------------------------------

def kmeans_fixed_iters(reps, k, n_iters=100, seed=42):
    """K-means with fixed number of iterations. No sklearn dependency.

    Uses standard Lloyd's algorithm:
        1. Init: k++ style (first center = random; each subsequent = weighted
           by squared distance to nearest existing center).
        2. Assign each point to nearest center.
        3. Recompute centers as mean of assigned points.
        4. Repeat for n_iters.

    Args:
        reps:    (M, D) float32 — input representations.
        k:       int — number of cluster centers.
        n_iters: int — fixed iterations (no convergence check).
        seed:    int — random seed.

    Returns:
        centers: (k, D) float32 — cluster centers.
        labels:  (M,) int32 — cluster assignments.
    """
    rng = np.random.RandomState(seed)
    M, D = reps.shape

    # K-means++ init
    print(f"  K-means++ init (k={k})...", flush=True)
    first_idx = rng.randint(0, M)
    centers = [reps[first_idx].copy()]
    for c_i in range(1, k):
        # Squared distances to nearest center — vectorized over current centers
        dists = np.full(M, np.inf, dtype=np.float64)
        for c in centers:
            d = ((reps - c) ** 2).sum(axis=1)
            dists = np.minimum(dists, d)
        probs = dists / dists.sum()
        next_idx = rng.choice(M, p=probs)
        centers.append(reps[next_idx].copy())
        if c_i % 32 == 0:
            print(f"    init center {c_i}/{k}", flush=True)
    centers = np.stack(centers, axis=0).astype(np.float32)  # (k, D)

    print(f"  Lloyd iterations: {n_iters}", flush=True)
    t0 = time.perf_counter()
    for iteration in range(n_iters):
        # Assign: (M, k) distances — computed in chunks to avoid OOM
        chunk = 1000
        labels = np.empty(M, dtype=np.int32)
        for start in range(0, M, chunk):
            end = min(start + chunk, M)
            diff = reps[start:end, None, :] - centers[None, :, :]  # (chunk, k, D)
            dists = (diff ** 2).sum(axis=2)  # (chunk, k)
            labels[start:end] = dists.argmin(axis=1)

        # Recompute centers
        new_centers = np.zeros_like(centers)
        counts = np.zeros(k, dtype=np.int32)
        for j in range(k):
            mask = (labels == j)
            if mask.sum() > 0:
                new_centers[j] = reps[mask].mean(axis=0)
                counts[j] = mask.sum()
            else:
                # Empty cluster: reinit to a random point
                new_centers[j] = reps[rng.randint(0, M)].copy()
                counts[j] = 1

        # Measure shift
        shift = np.linalg.norm(new_centers - centers, axis=1).mean()
        centers = new_centers

        if (iteration + 1) % 10 == 0 or iteration == 0:
            elapsed = time.perf_counter() - t0
            empty = int((counts == 0).sum())
            print(f"    iter {iteration+1:3d}/{n_iters}  shift={shift:.4f}  empty={empty}  ({elapsed:.0f}s)", flush=True)

        if shift < 1e-6:
            print(f"    converged early at iter {iteration+1}")
            break

    return centers.astype(np.float32), labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    CKPT        = os.environ.get("CKPT", "")
    SRC_LABEL   = os.environ.get("SRC_LABEL", "")
    N_EXTRACT   = int(os.environ.get("N_EXTRACT", "200"))
    N_SUBSAMPLE = int(os.environ.get("N_SUBSAMPLE", "10000"))
    K_CLUSTERS  = int(os.environ.get("K_CLUSTERS", "256"))
    K_MEANS_ITERS = int(os.environ.get("K_MEANS_ITERS", "100"))
    K_LOOPS     = int(os.environ.get("K_LOOPS", "4"))
    FIXED_LEN   = int(os.environ.get("FIXED_LEN", "320"))
    BATCH       = int(os.environ.get("BATCH", "2"))
    VALUES_ZERO = int(os.environ.get("VALUES_ZERO", "1")) > 0
    GSM8K_TRAIN = os.environ.get("GSM8K_TRAIN",
                                  os.environ.get("GSM8K_STEPS_PATH",
                                                 ".cache/gsm8k_steps_v1_train.jsonl"))
    OUT_DIR     = os.environ.get("OUT_DIR", ".cache/gsm8k_steps_ckpts")

    if not CKPT:
        print("ERROR: set CKPT= to the source ckpt path", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(CKPT):
        print(f"ERROR: ckpt not found: {CKPT}", file=sys.stderr)
        sys.exit(1)
    if not SRC_LABEL:
        SRC_LABEL = os.path.splitext(os.path.basename(CKPT))[0]

    out_path = os.path.join(OUT_DIR, f"{SRC_LABEL}_kmeans_codebook.safetensors")

    print("=== k-means codebook initializer ===")
    print(f"  src ckpt:   {CKPT}")
    print(f"  k clusters: {K_CLUSTERS}")
    print(f"  n_extract:  {N_EXTRACT} examples")
    print(f"  n_subsample:{N_SUBSAMPLE}")
    print(f"  k_iters:    {K_MEANS_ITERS}")
    print(f"  K_loops:    {K_LOOPS}")
    print(f"  values:     {'zero-init' if VALUES_ZERO else 'same as keys (centers)'}")
    print(f"  output:     {out_path}")
    print("")

    # Load model
    cfg = Config()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_model_fp32(model)
    Device[Device.DEFAULT].synchronize()
    del sd

    ckpt_sd = safe_load(CKPT)
    info = model.load_state_dict(ckpt_sd, strict=False)
    print(f"  loaded ckpt; missing={len(info['missing'])}, unexpected={len(info['unexpected'])}")
    del ckpt_sd
    Device[Device.DEFAULT].synchronize()

    # Check collapse params
    hidden = model.block.collapse_codebook_keys.shape[1]
    print(f"  hidden dim: {hidden}")

    # Load training data
    tok = load_tokenizer()
    all_examples = load_gsm8k_steps(GSM8K_TRAIN, min_k=2, max_k=6, bucket_by_k=False)
    # Use first N_EXTRACT (training examples, not shuffled — deterministic)
    examples = all_examples[:N_EXTRACT]
    print(f"  {len(examples)} training examples loaded from {GSM8K_TRAIN}")
    print("")

    # Extract pre-collapse reps
    Tensor.training = False
    reps = extract_pre_collapse_reps(model, tok, examples, K_LOOPS, FIXED_LEN, BATCH)
    # reps: (M, hidden)

    # Subsample if too many
    M = reps.shape[0]
    if M > N_SUBSAMPLE:
        idx = np.random.RandomState(42).choice(M, N_SUBSAMPLE, replace=False)
        reps_sub = reps[idx]
        print(f"Subsampled {M:,} → {N_SUBSAMPLE:,} reps for k-means", flush=True)
    else:
        reps_sub = reps
        print(f"Using all {M:,} reps (no subsampling needed)", flush=True)
    print("")

    # Run k-means
    print(f"Running k-means (k={K_CLUSTERS}, iters={K_MEANS_ITERS})...", flush=True)
    t0 = time.perf_counter()
    centers, labels = kmeans_fixed_iters(reps_sub, k=K_CLUSTERS, n_iters=K_MEANS_ITERS, seed=42)
    elapsed = time.perf_counter() - t0
    print(f"K-means done in {elapsed:.0f}s", flush=True)

    # Quick quality check
    n_clusters_populated = len(set(labels.tolist()))
    center_norms = np.linalg.norm(centers, axis=1)
    print(f"  Populated clusters: {n_clusters_populated}/{K_CLUSTERS}")
    print(f"  Center norm: mean={center_norms.mean():.4f}  std={center_norms.std():.4f}  "
          f"min={center_norms.min():.4f}  max={center_norms.max():.4f}")

    # Build output tensors
    keys_np = centers.astype(np.float32)          # (K, hidden)
    if VALUES_ZERO:
        vals_np = np.zeros_like(keys_np)           # (K, hidden) zeros — LoRA-style
    else:
        vals_np = keys_np.copy()                   # (K, hidden) same as keys

    keys_t = Tensor(keys_np, dtype=dtypes.float).contiguous().realize()
    vals_t = Tensor(vals_np, dtype=dtypes.float).contiguous().realize()

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    out_sd = {
        "collapse_codebook_keys":   keys_t,
        "collapse_codebook_values": vals_t,
    }
    safe_save(out_sd, out_path)
    print(f"\nSaved codebook init to {out_path}")
    print(f"  collapse_codebook_keys:   {keys_np.shape}  (k-means cluster centers)")
    print(f"  collapse_codebook_values: {vals_np.shape}  ({'zeros' if VALUES_ZERO else 'same as keys'})")
    print("")

    # Print how to merge into a v70 warm-start ckpt
    print("=" * 60)
    print("To build a v70 warm-start ckpt from these centers:")
    print("")
    print(f"  src_ckpt = '{CKPT}'")
    print(f"  km_ckpt  = '{out_path}'")
    print("")
    print("  from tinygrad.nn.state import safe_load, safe_save")
    print("  src = safe_load(src_ckpt)")
    print("  km  = safe_load(km_ckpt)")
    print("  src['block.collapse_codebook_keys']   = km['collapse_codebook_keys']")
    print("  src['block.collapse_codebook_values'] = km['collapse_codebook_values']")
    print(f"  safe_save(src, '{OUT_DIR}/v70_warm_start.safetensors')")
    print("=" * 60)


if __name__ == "__main__":
    main()
