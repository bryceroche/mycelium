"""Stage 1A verification smoke for LlamaBase (SmolLM2-1.7B).

Artifacts produced:
  .cache/v200_smoke/llama_load.log              — this script's stdout (tee'd)
  .cache/v200_smoke/llama_weights.sha256        — SHA256 of weight tensors
  .cache/v200_smoke/llama_load.provenance.json  — four-axis provenance sidecar

Completion criterion (per §11 of docs/v200_brief.md):
  Final line of llama_load.log must be "SMOKE PASSED <metrics>" or
  "SMOKE FAILED <reason>".

Checks performed:
  1. Model loads without error
  2. SHA256 of loaded weight tensors saved for reproducibility
  3. B=2, T=64 random forward (baseline): shape, no-NaN, per-layer norms
  4. B=2, T=64 wv-shared forward: same checks, norm ratio vs baseline
  5. Print comparison table (wv-sharing portability assessment)
  6. Peak GPU memory estimate
  7. Timing per-forward
"""

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

# Ensure project root is on path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from tinygrad import Tensor, dtypes, Device
from tinygrad.nn.state import safe_load

import mycelium.llama_base as llama_mod
from mycelium.llama_base import LlamaBase, SMOLLM2_CACHE, LLAMA_CACHE


# ---------------------------------------------------------------------------
# Logging: tee to file and stdout
# ---------------------------------------------------------------------------

LOG_DIR  = os.path.join(_PROJECT_ROOT, ".cache", "v200_smoke")
LOG_PATH = os.path.join(LOG_DIR, "llama_load.log")
SHA_PATH = os.path.join(LOG_DIR, "llama_weights.sha256")
PROV_PATH = os.path.join(LOG_DIR, "llama_load.provenance.json")

os.makedirs(LOG_DIR, exist_ok=True)

_log_fh = open(LOG_PATH, "w", buffering=1)


def log(msg: str = "") -> None:
    print(msg)
    _log_fh.write(msg + "\n")
    _log_fh.flush()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tensor_l2_norm(t: Tensor) -> float:
    """Compute L2 norm of a tensor (mean over batch, then L2 over feature dims)."""
    # Realize to get numpy
    arr = t.float().numpy()
    import numpy as np
    # Mean per-sample norm: average over B, T, then L2 over H
    # arr shape: (B, T, H) or (B, T, heads, head_dim)
    arr_flat = arr.reshape(arr.shape[0], -1, arr.shape[-1]) if arr.ndim == 4 else arr
    norms = np.linalg.norm(arr_flat, axis=-1)  # (B, T)
    return float(norms.mean())


def tensor_max_abs(t: Tensor) -> float:
    """Max absolute value."""
    import numpy as np
    return float(np.abs(t.float().numpy()).max())


def check_nan(t: Tensor, name: str) -> bool:
    """Returns True if tensor contains NaN or Inf."""
    import numpy as np
    arr = t.float().numpy()
    has_nan = bool(np.isnan(arr).any())
    has_inf = bool(np.isinf(arr).any())
    if has_nan or has_inf:
        log(f"  WARN: {name} contains NaN={has_nan} Inf={has_inf}")
    return has_nan or has_inf


def compute_weights_sha256(weights_path: str) -> str:
    """Compute SHA256 of the raw bytes of the safetensors weight file."""
    sha = hashlib.sha256()
    chunk = 1 << 20  # 1MB chunks
    with open(weights_path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            sha.update(buf)
    return sha.hexdigest()


def get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=_PROJECT_ROOT, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_tinygrad_sha() -> str:
    try:
        import tinygrad
        tg_dir = os.path.dirname(tinygrad.__file__)
        tg_git = os.path.join(os.path.dirname(tg_dir), ".git")
        if os.path.exists(tg_git):
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True,
                cwd=os.path.dirname(tg_dir), timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        return "editable-install-no-git"
    except Exception:
        return "unknown"


def format_shape(t: Tensor) -> str:
    return str(list(t.shape))


# ---------------------------------------------------------------------------
# Per-layer stats from taps
# ---------------------------------------------------------------------------

def per_layer_stats(model: LlamaBase, token_ids: Tensor) -> dict:
    """Run forward_with_taps and collect per-layer L2 norms."""
    import numpy as np
    x_final, all_taps = model.forward_with_taps(token_ids)

    stats = {}
    for i, taps in enumerate(all_taps):
        pre  = taps["pre_ln_resid"].float().numpy()
        attn = taps["post_attn_resid"].float().numpy()
        mlp  = taps["post_mlp_resid"].float().numpy()

        pre_norm  = float(np.linalg.norm(pre.reshape(pre.shape[0], -1, pre.shape[-1]),  axis=-1).mean())
        attn_norm = float(np.linalg.norm(attn.reshape(attn.shape[0], -1, attn.shape[-1]), axis=-1).mean())
        mlp_norm  = float(np.linalg.norm(mlp.reshape(mlp.shape[0], -1, mlp.shape[-1]),  axis=-1).mean())

        # Step size: ‖post_mlp - pre_ln‖ (the §10 v1.1 energy channel)
        delta = mlp - pre
        delta_norm = float(np.linalg.norm(delta.reshape(delta.shape[0], -1, delta.shape[-1]), axis=-1).mean())

        stats[f"layer{i}"] = {
            "pre_ln_norm":  pre_norm,
            "post_attn_norm": attn_norm,
            "post_mlp_norm":  mlp_norm,
            "delta_norm":     delta_norm,
        }

    final_norm = float(np.linalg.norm(
        x_final.float().numpy().reshape(x_final.shape[0], -1, x_final.shape[-1]), axis=-1
    ).mean())
    stats["ln_f_output_norm"] = final_norm
    return stats, x_final


# ---------------------------------------------------------------------------
# Main smoke
# ---------------------------------------------------------------------------

def run_smoke() -> int:
    """Run the full smoke. Returns 0 on pass, 1 on failure."""
    import numpy as np

    log("=" * 70)
    log("Mycelium v200 Stage 1A — LlamaBase Smoke Test")
    log(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    log(f"Device:    {Device.DEFAULT}")
    log("=" * 70)

    # ------------------------------------------------------------------
    # 1. Resolve weights path
    # ------------------------------------------------------------------
    log("\n[1] Resolving weights path...")
    try:
        weights_path = LlamaBase._resolve_path(None)
        log(f"    weights_path = {weights_path}")
        weights_size_gb = os.path.getsize(weights_path) / (1024**3)
        log(f"    file size    = {weights_size_gb:.2f} GB")
    except FileNotFoundError as e:
        log(f"SMOKE FAILED weights not found: {e}")
        return 1

    # ------------------------------------------------------------------
    # 2. SHA256 of weights file
    # ------------------------------------------------------------------
    log("\n[2] Computing SHA256 of weights file...")
    t0 = time.time()
    sha256 = compute_weights_sha256(weights_path)
    t_sha = time.time() - t0
    log(f"    sha256       = {sha256}")
    log(f"    hashing time = {t_sha:.1f}s")
    with open(SHA_PATH, "w") as f:
        f.write(f"{sha256}  {weights_path}\n")
    log(f"    saved to     {SHA_PATH}")

    # ------------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------------
    log("\n[3] Loading model (LlamaBase)...")
    t0 = time.time()
    try:
        model = LlamaBase(weights_path)
        t_load = time.time() - t0
        log(f"    Load time:   {t_load:.1f}s")
        log(f"    hidden_size: {model.hidden_size}")
        log(f"    vocab_size:  {model.vocab_size}")
        log(f"    n_heads:     {model.n_heads}")
        log(f"    head_dim:    {model.head_dim}")
        log(f"    n_layers:    {len(model.layers)}")
    except Exception as e:
        log(f"SMOKE FAILED model load error: {e}")
        import traceback
        log(traceback.format_exc())
        return 1

    # ------------------------------------------------------------------
    # 4. Baseline forward (B=2, T=64, random tokens)
    # ------------------------------------------------------------------
    log("\n[4] Baseline forward (B=2, T=64, random tokens, no wv-sharing)...")
    Tensor.manual_seed(42)
    token_ids = Tensor(
        np.random.randint(0, model.vocab_size, (2, 64), dtype=np.int32)
    )

    # Shape check
    log(f"    token_ids shape: {list(token_ids.shape)}")

    t0 = time.time()
    try:
        baseline_stats, x_baseline = per_layer_stats(model, token_ids)
        t_fwd_base = time.time() - t0
    except Exception as e:
        log(f"SMOKE FAILED baseline forward error: {e}")
        import traceback
        log(traceback.format_exc())
        return 1

    log(f"    Forward time: {t_fwd_base:.3f}s")
    log(f"    Output shape: {list(x_baseline.shape)}")

    # Shape assertion
    assert list(x_baseline.shape) == [2, 64, 2048], \
        f"Expected [2, 64, 2048], got {list(x_baseline.shape)}"

    # NaN check
    baseline_has_nan = check_nan(x_baseline, "baseline output")
    if baseline_has_nan:
        log("SMOKE FAILED NaN/Inf in baseline forward output")
        return 1

    log(f"\n    Per-layer norms (baseline):")
    log(f"    {'Layer':>8}  {'pre_ln':>10}  {'post_attn':>10}  {'post_mlp':>10}  {'delta':>10}")
    for i in range(4):
        s = baseline_stats[f"layer{i}"]
        log(f"    {'L'+str(i):>8}  {s['pre_ln_norm']:>10.4f}  {s['post_attn_norm']:>10.4f}  "
            f"{s['post_mlp_norm']:>10.4f}  {s['delta_norm']:>10.4f}")
    log(f"    {'ln_f_out':>8}  {baseline_stats['ln_f_output_norm']:>10.4f}")

    # ------------------------------------------------------------------
    # 5. wv-shared forward (L0's wv broadcast to L1-L3)
    # ------------------------------------------------------------------
    log("\n[5] wv-shared forward (L0 wv broadcast to L1-L3, Pythia-era pin test)...")
    log("    NOTE: The Pythia-era wv-sharing pin is UNVERIFIED on Llama/SmolLM2")
    log("    (different attention structure). Emitting both configs' stats so")
    log("    portability is measured, not assumed. See docs/v200_brief.md §4.")

    t0 = time.time()
    try:
        x_wvshared = model.forward_wv_shared(token_ids)
        t_fwd_wvs = time.time() - t0
    except Exception as e:
        log(f"SMOKE FAILED wv-shared forward error: {e}")
        import traceback
        log(traceback.format_exc())
        return 1

    log(f"    Forward time: {t_fwd_wvs:.3f}s")
    log(f"    Output shape: {list(x_wvshared.shape)}")

    wvs_has_nan = check_nan(x_wvshared, "wv-shared output")
    if wvs_has_nan:
        log("  WARN: wv-shared output has NaN/Inf (pin may break representations)")

    # Per-layer stats for wv-shared via direct taps approach
    # Use model's forward_with_taps but patch wv temporarily
    # We'll compute norms from the final output and a simplified analysis
    x_wvs_np = x_wvshared.float().numpy()
    wvs_out_norm = float(np.linalg.norm(
        x_wvs_np.reshape(x_wvs_np.shape[0], -1, x_wvs_np.shape[-1]), axis=-1
    ).mean())

    x_base_np = x_baseline.float().numpy()
    base_out_norm = baseline_stats["ln_f_output_norm"]

    # Cosine similarity between baseline and wv-shared outputs (mean over B,T)
    x_base_flat = x_base_np.reshape(-1, 2048)
    x_wvs_flat  = x_wvs_np.reshape(-1, 2048)
    cos_sims = (x_base_flat * x_wvs_flat).sum(-1) / (
        np.linalg.norm(x_base_flat, axis=-1) * np.linalg.norm(x_wvs_flat, axis=-1) + 1e-8
    )
    mean_cos_sim = float(cos_sims.mean())
    min_cos_sim  = float(cos_sims.min())

    # L2 distance between outputs
    l2_diff = float(np.linalg.norm(x_base_flat - x_wvs_flat, axis=-1).mean())
    norm_ratio = wvs_out_norm / (base_out_norm + 1e-8)

    log(f"\n    wv-sharing comparison:")
    log(f"    Metric                    Baseline       wv-shared      Ratio/Delta")
    log(f"    ln_f_output_norm          {base_out_norm:>10.4f}     {wvs_out_norm:>10.4f}     "
        f"ratio={norm_ratio:.4f}")
    log(f"    mean_cos_sim(base,wvs)    {'N/A':>10}     {'N/A':>10}     {mean_cos_sim:.6f}")
    log(f"    min_cos_sim(base,wvs)     {'N/A':>10}     {'N/A':>10}     {min_cos_sim:.6f}")
    log(f"    mean_L2_dist(base,wvs)    {'N/A':>10}     {'N/A':>10}     {l2_diff:.4f}")

    # Interpretation
    log(f"\n    wv-sharing portability assessment:")
    if mean_cos_sim > 0.99:
        log(f"    HIGH cos-sim ({mean_cos_sim:.4f}) — wv-sharing barely changes output.")
        log(f"    Pin may be portable BUT representations are very similar,")
        log(f"    raising question of whether wv-sharing adds constraint vs redundancy.")
    elif mean_cos_sim > 0.90:
        log(f"    MODERATE cos-sim ({mean_cos_sim:.4f}) — wv-sharing changes output noticeably.")
        log(f"    Pin portability is unclear; full training comparison required.")
    else:
        log(f"    LOW cos-sim ({mean_cos_sim:.4f}) — wv-sharing substantially changes output.")
        log(f"    Pin likely NOT portable to SmolLM2 attention structure.")

    if wvs_has_nan:
        log(f"    VERDICT: NaN in wv-shared — pin BREAKS SmolLM2 representations.")
    else:
        log(f"    VERDICT: No NaN in either config. Portability requires training comparison.")

    # ------------------------------------------------------------------
    # 6. Timing stats
    # ------------------------------------------------------------------
    log("\n[6] Timing stats...")
    # Run a second forward to get warmed-up timing
    t0 = time.time()
    _ = model.forward(token_ids)
    t_fwd2 = time.time() - t0
    log(f"    baseline forward (warm):  {t_fwd2:.3f}s")
    t0 = time.time()
    _ = model.forward_wv_shared(token_ids)
    t_fwd_wvs2 = time.time() - t0
    log(f"    wv-shared forward (warm): {t_fwd_wvs2:.3f}s")

    # ------------------------------------------------------------------
    # 7. Peak memory estimate
    # ------------------------------------------------------------------
    log("\n[7] Peak memory estimate...")
    # Rough estimate: weight count × bytes per param
    # SmolLM2 first 4 layers: 4 × (4 attn weight matrices + 3 MLP + 2 LN)
    # Each attn weight: 2048×2048 = 4M, × 4 = 16M per layer
    # Each MLP: gate+up(2048×8192) + down(8192×2048) = 2×2048×8192 + 8192×2048
    # = 3 × 2048 × 8192 = 50.3M per layer
    # Plus embed: 49152×2048 = 100.7M params
    attn_params_per_layer = 4 * 2048 * 2048
    mlp_params_per_layer  = 2 * 2048 * 8192 + 8192 * 2048
    ln_params_per_layer   = 2 * 2048
    total_params = (
        49152 * 2048 +                                    # embed
        4 * (attn_params_per_layer + mlp_params_per_layer + ln_params_per_layer) +
        2048                                              # ln_f
    )
    # bfloat16 = 2 bytes
    weight_mem_gb = total_params * 2 / (1024**3)
    log(f"    L0-L3 + embed + ln_f:    {total_params/1e6:.1f}M params")
    log(f"    Weight memory (bf16):    {weight_mem_gb:.2f} GB")
    log(f"    (Full SmolLM2-1.7B has ~1.7B params; we load only L0-L3 "
        f"+ embed + ln_f)")

    # Try to get actual GPU memory info if available
    try:
        # tinygrad GPU memory
        from tinygrad.runtime.ops_amd import AMDevice
        log(f"    GPU device: AMD (AM driver)")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 8. Provenance sidecar
    # ------------------------------------------------------------------
    log("\n[8] Writing provenance sidecar...")
    git_sha = get_git_sha()
    tg_sha  = get_tinygrad_sha()
    now_iso = datetime.now(timezone.utc).isoformat()

    # Determine which model was loaded
    ckpt_name = "unknown"
    if "smollm2" in weights_path.lower() or "SmolLM2" in weights_path:
        ckpt_name = "SmolLM2-1.7B-hf-snapshot"
    elif "llama" in weights_path.lower():
        ckpt_name = "llama-3.2-1B-base-hf-snapshot"

    provenance = {
        "what": {
            "metric":     "llama_smoke_forward_stats",
            "units":      "raw activations + L2 norms",
            "shape":      "[B=2, T=64, H=2048] after L0-L3 + ln_f",
            "head_group": None,
        },
        "where": {
            "file": LOG_PATH,
            "key":  None,
        },
        "when": {
            "timestamp_iso": now_iso,
            "git_sha":       git_sha,
            "config_diff":   None,
            "step":          0,
        },
        "with_what": {
            "ckpt":  ckpt_name,
            "split": None,
            "seed":  42,
            "env": {
                "tinygrad_sha": tg_sha,
                "device":       f"AM driver/AMD 7900 XTX ({Device.DEFAULT})",
                "env_vars": {
                    "LLAMA_WEIGHTS":   os.environ.get("LLAMA_WEIGHTS",   "unset"),
                    "SMOLLM2_WEIGHTS": os.environ.get("SMOLLM2_WEIGHTS", "unset"),
                },
            },
        },
    }

    with open(PROV_PATH, "w") as f:
        json.dump(provenance, f, indent=2)
    log(f"    Saved: {PROV_PATH}")

    # ------------------------------------------------------------------
    # 9. Summary + SMOKE PASSED / FAILED line
    # ------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"  Model:           {ckpt_name}")
    log(f"  Weights path:    {weights_path}")
    log(f"  SHA256:          {sha256[:16]}...")
    log(f"  Load time:       {t_load:.1f}s")
    log(f"  Baseline fwd:    {t_fwd2:.3f}s (warm, B=2, T=64)")
    log(f"  wv-shared fwd:   {t_fwd_wvs2:.3f}s (warm, B=2, T=64)")
    log(f"  Output shape:    {list(x_baseline.shape)}")
    log(f"  Baseline NaN:    {baseline_has_nan}")
    log(f"  wv-shared NaN:   {wvs_has_nan}")
    log(f"  wv-sharing cos:  {mean_cos_sim:.6f}")
    log(f"  ln_f norm (base):{base_out_norm:.4f}")
    log(f"  ln_f norm (wvs): {wvs_out_norm:.4f}")

    # Check pass conditions
    pass_conditions = [
        (not baseline_has_nan, "baseline forward is NaN-free"),
        (list(x_baseline.shape) == [2, 64, 2048], "output shape is [2, 64, 2048]"),
        (os.path.exists(SHA_PATH), "SHA256 file exists"),
        (os.path.exists(PROV_PATH), "provenance file exists"),
    ]

    all_pass = all(cond for cond, _ in pass_conditions)
    for cond, desc in pass_conditions:
        status = "PASS" if cond else "FAIL"
        log(f"  [{status}] {desc}")

    log("")
    if all_pass:
        metrics = (
            f"model={ckpt_name} "
            f"load={t_load:.1f}s "
            f"fwd={t_fwd2:.3f}s "
            f"shape=[2,64,2048] "
            f"nan=False "
            f"wv_cos={mean_cos_sim:.4f} "
            f"sha={sha256[:8]}"
        )
        log(f"SMOKE PASSED {metrics}")
        return 0
    else:
        failed = [desc for cond, desc in pass_conditions if not cond]
        log(f"SMOKE FAILED {'; '.join(failed)}")
        return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np  # needed for random token generation
    exit_code = run_smoke()
    _log_fh.close()
    sys.exit(exit_code)
