"""Codebook usage histogram for v69 collapse pipeline.

Loads a v69 checkpoint, runs forward on ~100 GSM8K test problems, and captures
the `match_weights` tensor inside `apply_collapse_v69` via monkey-patch. Then
computes:

  1. Per-entry usage frequency — which codebook entries get attended to.
  2. Entropy distribution — histogram of per-token entropy values.
  3. Effective number of "live" entries — entries that receive > 1% of mass
     anywhere across all (B, T) positions.
  4. Top-5 most-used entries' value norms.
  5. Match entropy over training (scalar from _collapse_last_match_entropy, if
     the ckpt training log was captured separately).

Expected outcome (sad case, matching observations so far):
  - Near-uniform usage across all 256 entries.
  - Entropy ≈ 5.544 / log(256)=5.545 — essentially max entropy.
  - Effective live entries ≈ 256 (all interchangeable).

Hopeful case (would indicate successful prototype learning):
  - Peaked distribution: some entries dominate, others near-zero.
  - Entropy 3.0–4.5: moderate specialization.
  - 20–80 live entries.

Env vars:
  CKPT            Path to v69 ckpt (required)
  NUM_EVAL        Number of test problems (default 100)
  FIXED_LEN       Sequence length (default 400)
  BATCH           Batch size (default 2)
  TRAIN_LOOPS     K inner breaths (default 4)
  GSM8K_PATH      GSM8K test JSONL (default .cache/gsm8k_steps_v1_test.jsonl)
  HIST_OUT        Output path for text histogram (default /tmp/v69_codebook_hist.txt)

Usage:
  CKPT=.cache/gsm8k_steps_ckpts/v69_collapse_step3000.safetensors \\
    python scripts/codebook_usage_histogram.py
"""
import os
import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must set arch env vars BEFORE importing mycelium (module-level os.environ reads)
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
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import load_gsm8k_steps, split_train_eval
from scripts.eval_ckpt_controller_segmented import cast_model_fp32


def extract_match_weights(model, tok, examples, K_loops, fixed_len, batch_size):
    """Monkey-patch apply_collapse_v69 to capture match_weights.

    Returns:
        all_mw: list of numpy arrays, each (B, T, N). Caller stacks/flattens.
    """
    all_mw = []

    orig_apply = model.block.apply_collapse_v69

    def capturing_apply(x, return_compressed=False):
        x_f = x.cast(dtypes.float)
        # Replicate codebook match (no side effects on model state)
        keys = model.block.collapse_codebook_keys  # (N, hidden)
        N = keys.shape[0]
        hidden = keys.shape[1]
        scores = (x_f @ keys.T) / (float(hidden) ** 0.5)

        COLLAPSE_TAU = float(os.environ.get('COLLAPSE_TAU', '1.0'))
        mw = (scores / COLLAPSE_TAU).softmax(axis=-1)  # (B, T, N)
        # Store as numpy — call .numpy() eagerly (outside JIT, this is fine)
        all_mw.append(mw.numpy())  # (B, T, N)
        # Now call the real method
        return orig_apply(x, return_compressed=return_compressed)

    model.block.apply_collapse_v69 = capturing_apply

    print(f"Running forward on {len(examples)} examples (K={K_loops})...", flush=True)
    Tensor.training = False
    t0 = time.perf_counter()

    for b_start in range(0, len(examples), batch_size):
        batch = examples[b_start:b_start + batch_size]
        prompts = [tok.encode(ex.problem).ids for ex in batch]
        tokens_np = np.zeros((len(batch), fixed_len), dtype=np.int32)
        for i, p in enumerate(prompts):
            p_trunc = p[:fixed_len]
            tokens_np[i, :len(p_trunc)] = p_trunc
        tokens = Tensor(tokens_np, dtype=dtypes.int).realize()
        _ = model.breathe_with_lookup(tokens, n_loops=K_loops)
        if (b_start // batch_size) % 10 == 0:
            print(f"  {b_start}/{len(examples)} ({time.perf_counter() - t0:.0f}s)", flush=True)

    model.block.apply_collapse_v69 = orig_apply
    print(f"Done in {time.perf_counter() - t0:.0f}s. Captured {len(all_mw)} batches.", flush=True)
    return all_mw


def compute_stats(all_mw_list):
    """Compute usage statistics from list of (B, T, N) match_weight arrays.

    Returns dict with:
        entry_freq:       (N,) float — fraction of (B*T) positions where each
                          entry is the argmax (hard assignment)
        entry_softmass:   (N,) float — mean soft probability mass per entry
                          (softer measure of usage; sums to 1.0)
        entropy_mean:     float — mean per-position entropy across all examples
        entropy_std:      float — std of per-position entropy
        entropy_hist:     (10,) int — histogram of per-position entropy in bins
                          from 0 to log(N)
        live_entries_1pct: int — entries that received > 1% of mass in at least
                            one (B, T) position
        live_entries_hard: int — entries that are argmax at least once
        top5_entry_ids:   list of int — top 5 by soft mass
        N:                int — codebook size
        n_positions:      int — total (B, T) positions analysed
    """
    # Stack all batches: flatten to (M, N) where M = total positions
    chunks = []
    for mw in all_mw_list:
        # mw: (B, T, N)
        B, T, N = mw.shape
        chunks.append(mw.reshape(-1, N))  # (B*T, N)
    all_positions = np.concatenate(chunks, axis=0)  # (M, N)
    M, N = all_positions.shape

    # Per-entry soft mass (mean over all positions)
    entry_softmass = all_positions.mean(axis=0)  # (N,)

    # Hard argmax assignment — which entry "wins" each position
    argmax_ids = all_positions.argmax(axis=1)  # (M,)
    entry_counts = np.bincount(argmax_ids, minlength=N)
    entry_freq = entry_counts / M

    # Per-position entropy: -sum(p log p)
    log_mw = np.log(all_positions + 1e-12)
    per_pos_entropy = -(all_positions * log_mw).sum(axis=1)  # (M,)
    entropy_mean = float(per_pos_entropy.mean())
    entropy_std = float(per_pos_entropy.std())
    max_entropy = float(np.log(N))

    n_bins = 10
    hist_edges = np.linspace(0.0, max_entropy, n_bins + 1)
    entropy_hist, _ = np.histogram(per_pos_entropy, bins=hist_edges)

    # Live entries
    live_entries_hard = int((entry_counts > 0).sum())
    # max mass any position assigns to each entry
    max_per_entry = all_positions.max(axis=0)  # (N,)
    live_entries_1pct = int((max_per_entry > 0.01).sum())

    # Top 5 by soft mass
    top5 = list(np.argsort(entry_softmass)[::-1][:5])

    return {
        "N": N,
        "n_positions": M,
        "entry_freq": entry_freq,
        "entry_softmass": entry_softmass,
        "entropy_mean": entropy_mean,
        "entropy_std": entropy_std,
        "entropy_hist": entropy_hist,
        "hist_edges": hist_edges,
        "live_entries_hard": live_entries_hard,
        "live_entries_1pct": live_entries_1pct,
        "top5_entry_ids": top5,
    }


def print_report(stats, codebook_values_np, out_path):
    """Print a human-readable report and optionally write to file."""
    N = stats["N"]
    M = stats["n_positions"]
    entropy_mean = stats["entropy_mean"]
    max_entropy = float(np.log(N))
    norm_entropy = entropy_mean / max_entropy
    live_hard = stats["live_entries_hard"]
    live_1pct = stats["live_entries_1pct"]
    top5 = stats["top5_entry_ids"]
    entry_softmass = stats["entry_softmass"]

    lines = [
        "=" * 60,
        "v69 codebook usage histogram",
        "=" * 60,
        f"  Codebook size N={N}",
        f"  Positions analysed: {M:,}",
        "",
        "--- ENTROPY ---",
        f"  Mean per-position entropy: {entropy_mean:.4f} / {max_entropy:.4f} (max = log({N}))",
        f"  Normalized entropy:        {norm_entropy:.4f}  (1.0 = fully uniform = broken)",
        f"  Std:                       {stats['entropy_std']:.4f}",
        "",
        "  Histogram (0 → log(N) in 10 bins):",
    ]
    hist_edges = stats["hist_edges"]
    for i, count in enumerate(stats["entropy_hist"]):
        lo = hist_edges[i]
        hi = hist_edges[i + 1]
        bar = "#" * (count * 40 // max(stats["entropy_hist"].max(), 1))
        lines.append(f"    [{lo:.2f}, {hi:.2f}): {count:5d}  {bar}")

    lines += [
        "",
        "--- USAGE DISTRIBUTION ---",
        f"  Live entries (hard argmax ≥1 time): {live_hard} / {N}",
        f"  Live entries (max soft mass > 1%):  {live_1pct} / {N}",
        "",
        "  Top-5 entries by mean soft mass:",
    ]
    for rank, entry_id in enumerate(top5):
        mass = entry_softmass[entry_id]
        freq = stats["entry_freq"][entry_id]
        # value norm from codebook_values
        val_norm = float(np.linalg.norm(codebook_values_np[entry_id]))
        lines.append(
            f"    #{rank+1}: entry {entry_id:3d}  soft_mass={mass:.4f}  "
            f"argmax_freq={freq:.4f}  value_norm={val_norm:.4f}"
        )

    lines += [
        "",
        "--- SOFT MASS DISTRIBUTION (all entries) ---",
        "  (expected uniform = {:.4f} each)".format(1.0 / N),
    ]
    softmass_sorted = np.sort(entry_softmass)[::-1]
    # Show top-20 and bottom-20
    lines.append("  Top 20 entries by soft mass:")
    for i in range(min(20, N)):
        entry_id = int(np.argsort(entry_softmass)[::-1][i])
        lines.append(f"    [{i:3d}] entry {entry_id:3d}: {entry_softmass[entry_id]:.5f}")
    if N > 40:
        lines.append("  ...")
        lines.append("  Bottom 20 entries by soft mass:")
        for i in range(min(20, N)):
            entry_id = int(np.argsort(entry_softmass)[i])
            lines.append(f"    [{N-20+i:3d}] entry {entry_id:3d}: {entry_softmass[entry_id]:.5f}")

    lines += [
        "",
        "=" * 60,
        "INTERPRETATION:",
        f"  Uniform baseline entropy: {max_entropy:.4f}",
        f"  Current:                  {entropy_mean:.4f}  ({norm_entropy*100:.1f}% of max)",
        "",
        "  near-max entropy → codebook is interchangeable (gate+proj doing the work)",
        "  50-80% of max    → moderate specialization",
        "  <50% of max      → codebook prototypes are genuinely diverse",
        "=" * 60,
    ]

    report = "\n".join(lines)
    print(report)
    if out_path:
        with open(out_path, "w") as f:
            f.write(report + "\n")
        print(f"\nReport written to {out_path}")


def main():
    CKPT       = os.environ.get("CKPT", "")
    NUM_EVAL   = int(os.environ.get("NUM_EVAL", "100"))
    FIXED_LEN  = int(os.environ.get("FIXED_LEN", "400"))
    BATCH      = int(os.environ.get("BATCH", "2"))
    K_LOOPS    = int(os.environ.get("TRAIN_LOOPS", "4"))
    GSM8K_PATH = os.environ.get("GSM8K_PATH",
                                 os.environ.get("GSM8K_STEPS_PATH",
                                                ".cache/gsm8k_steps_v1_test.jsonl"))
    HIST_OUT   = os.environ.get("HIST_OUT", "/tmp/v69_codebook_hist.txt")

    if not CKPT:
        print("ERROR: set CKPT= to the v69 ckpt path", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(CKPT):
        print(f"ERROR: ckpt not found: {CKPT}", file=sys.stderr)
        sys.exit(1)

    print(f"=== v69 codebook usage histogram ===")
    print(f"  ckpt:      {CKPT}")
    print(f"  n_eval:    {NUM_EVAL}")
    print(f"  fixed_len: {FIXED_LEN}")
    print(f"  K_loops:   {K_LOOPS}")
    print(f"  gsm8k:     {GSM8K_PATH}")
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
    print(f"  loaded; missing={len(info['missing'])}, unexpected={len(info['unexpected'])}")
    del ckpt_sd
    Device[Device.DEFAULT].synchronize()

    # Verify collapse params present
    cb_keys = model.block.collapse_codebook_keys.numpy()    # (N, hidden)
    cb_vals = model.block.collapse_codebook_values.numpy()  # (N, hidden)
    N = cb_keys.shape[0]
    hidden = cb_keys.shape[1]
    print(f"  collapse_codebook_keys:   {cb_keys.shape}  norm_mean={np.linalg.norm(cb_keys, axis=1).mean():.4f}")
    print(f"  collapse_codebook_values: {cb_vals.shape}  norm_mean={np.linalg.norm(cb_vals, axis=1).mean():.4f}")
    print("")

    # Load test data
    tok = load_tokenizer()
    all_examples = load_gsm8k_steps(GSM8K_PATH, min_k=2, max_k=6, bucket_by_k=False)
    _, eval_examples = split_train_eval(all_examples, n_eval=NUM_EVAL, seed=42)
    print(f"  {len(eval_examples)} eval examples loaded")

    # Extract match_weights via monkey-patch
    Tensor.training = False
    all_mw = extract_match_weights(model, tok, eval_examples, K_LOOPS, FIXED_LEN, BATCH)

    if not all_mw:
        print("ERROR: no match_weights captured — is COLLAPSE_V69=1?", file=sys.stderr)
        sys.exit(1)

    # Compute stats
    print("Computing statistics...", flush=True)
    stats = compute_stats(all_mw)

    # Print report
    print_report(stats, cb_vals, HIST_OUT)

    # Also dump JSON-friendly summary
    summary = {
        "ckpt": CKPT,
        "N": stats["N"],
        "n_positions": stats["n_positions"],
        "entropy_mean": stats["entropy_mean"],
        "entropy_std": stats["entropy_std"],
        "entropy_normalized": stats["entropy_mean"] / float(np.log(stats["N"])),
        "live_entries_hard": stats["live_entries_hard"],
        "live_entries_1pct": stats["live_entries_1pct"],
        "top5_entry_ids": stats["top5_entry_ids"],
        "top5_soft_mass": [float(stats["entry_softmass"][i]) for i in stats["top5_entry_ids"]],
    }
    summary_path = HIST_OUT.replace(".txt", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary written to {summary_path}")


if __name__ == "__main__":
    main()
