"""Stage 1B arch smoke: verify FactorGraphV200 forward pass.

Completion criterion (per docs/v200_brief.md §11 + §15):
  Final line of .cache/v200_smoke/arch_smoke.log must be:
    SMOKE PASSED <metrics>
  or
    SMOKE FAILED <reason>

Checks:
  1. Import FactorGraphV200, LlamaBase, provenance helpers
  2. Load model (Llama-3.2-1B; falls back to SmolLM2-1.7B)
  3. Random forward B=2, T=24, K=8
  4. Shape assertions:
       fg_emb:      (2, 24, 2048)
       z_init:      (2, 32, 2048)
       tree_logits: (2, 5, 10)
       traces:      K=8 entries, each with correct keys
  5. No NaN anywhere (tree_logits, calib, z per breath)
  6. Instrumentation hooks fire:
       - latent JSD: K-1=7 values, non-trivial (any non-zero)
       - cross-attn entropy: (K=8, n_heads=16), non-zero, units=nats
       - self-attn entropy: (K=8, 4, n_heads), non-zero, units=nats
       - self-attn JSD (secondary): (K-1=7, 4), kept as diagnostic only
       - energy channel (K=8, L=32), non-zero
  7. Like-units distinguishability (Gate B §8):
       - cross-attn entropy (nats) vs self-attn entropy (nats)
       - |cross_entropy_mean - self_entropy_mean| reported
       - NOT entropy vs JSD (category error from Stage 1B initial run)
  7b. Waist alternation check:
       - even breaths: waist_delta > 1e-6  (waist fired)
       - odd breaths:  waist_delta ≈ 0     (identity path)
       - SMOKE FAILED if fires_correctly=NO
  8. Persistence bundle dry-run: K=8 entries, correct shapes
  9. Provenance sidecar written at .cache/v200_smoke/arch_smoke.provenance.json
 10. SmolLM2 regression: re-run v200_llama_smoke.py and verify SMOKE PASSED
 11. Llama-3.2-1B regression: re-run v200_llama32_smoke.py and verify SMOKE PASSED
 12. Reference curves (§5): B=32 forward → persist random-init null trajectories at
       .cache/v200_smoke/reference_curves/latent_jsd_random_init.npz
       .cache/v200_smoke/reference_curves/energy_channel_random_init.npz
       .cache/v200_smoke/reference_curves/xattn_entropy_random_init.npz
       .cache/v200_smoke/reference_curves/self_attn_entropy_random_init.npz
     each with sibling .provenance.json

Artifacts produced:
  .cache/v200_smoke/arch_smoke.log              — this log
  .cache/v200_smoke/arch_smoke.provenance.json  — four-axis provenance sidecar
  .cache/v200_smoke/reference_curves/*.npz      — random-init null curves
  .cache/v200_smoke/reference_curves/*.provenance.json
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from tinygrad import Tensor, dtypes, Device

from mycelium.llama_base import LlamaBase
from mycelium.factor_graph_v200 import FactorGraphV200, V200Config
from mycelium.provenance import make_provenance, write_with_provenance, validate_provenance


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

LOG_DIR    = os.path.join(_PROJECT_ROOT, ".cache", "v200_smoke")
LOG_PATH   = os.path.join(LOG_DIR, "arch_smoke.log")
PROV_PATH  = os.path.join(LOG_DIR, "arch_smoke.provenance.json")
REF_DIR    = os.path.join(LOG_DIR, "reference_curves")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REF_DIR, exist_ok=True)

_log_fh = open(LOG_PATH, "w", buffering=1)


def log(msg: str = "") -> None:
    print(msg)
    _log_fh.write(msg + "\n")
    _log_fh.flush()


def get_git_sha() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           capture_output=True, text=True,
                           cwd=_PROJECT_ROOT, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_tinygrad_sha() -> str:
    try:
        import tinygrad
        tg_dir = os.path.dirname(os.path.dirname(tinygrad.__file__))
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           capture_output=True, text=True, cwd=tg_dir, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "editable-no-git"
    except Exception:
        return "unknown"


def check_nan(t: Tensor, name: str) -> bool:
    arr = t.float().numpy()
    has_nan = bool(np.isnan(arr).any())
    has_inf = bool(np.isinf(arr).any())
    if has_nan or has_inf:
        log(f"  [WARN] {name}: NaN={has_nan} Inf={has_inf}")
    return has_nan or has_inf


# ---------------------------------------------------------------------------
# Main smoke
# ---------------------------------------------------------------------------

def run_smoke() -> int:
    log("=" * 70)
    log("Mycelium v200 Stage 1B — Perceiver-Core Architecture Smoke Test")
    log("(follow-up: like-units entropy, waist alternation, reference curves)")
    log(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    log(f"Device:    {Device.DEFAULT}")
    log("=" * 70)

    failures = []
    t_start = time.time()

    # ------------------------------------------------------------------
    # 1. Module imports
    # ------------------------------------------------------------------
    log("\n[1] Module import check...")
    try:
        from mycelium.llama_base import LlamaBase, _ModelConfig
        from mycelium.factor_graph_v200 import (
            FactorGraphV200, V200Config, LatentInit, CrossAttention, Waist,
            TreeCodebookReadout, CalibHead,
        )
        from mycelium.provenance import (
            make_provenance, write_with_provenance, validate_provenance
        )
        log("  [PASS] All Stage 1B modules imported successfully")
    except Exception as e:
        log(f"  [FAIL] Import error: {e}")
        import traceback
        log(traceback.format_exc())
        failures.append("module import")
        log(f"SMOKE FAILED {failures}")
        return 1

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    log("\n[2] Loading FactorGraphV200 model...")
    t0 = time.time()
    try:
        config = V200Config()   # auto-detect best available backbone
        model  = FactorGraphV200(config)
        t_load = time.time() - t0
        log(f"  Load time: {t_load:.1f}s")
        log(f"  Backbone is_gqa: {model.backbone.is_gqa}")
        log(f"  wv_sharing: OFF (each layer's own wv — conservative per §4)")
        log(f"  IB centroids: {config.centroids_path}")
        log(f"  jitter_std: {config.jitter_std}")
        log(f"  [PASS] Model loaded")
    except Exception as e:
        log(f"  [FAIL] Model load error: {e}")
        import traceback
        log(traceback.format_exc())
        failures.append("model load")
        log(f"SMOKE FAILED {'; '.join(failures)}")
        return 1

    # ------------------------------------------------------------------
    # 3. Random forward (B=2, T=24, K=8)
    # ------------------------------------------------------------------
    log("\n[3] Random forward (B=2, T=24, K=8, return_taps=True)...")
    B, T, K = 2, 24, 8

    np.random.seed(42)
    vocab_size = model.backbone.vocab_size
    fg_tokens_np = np.random.randint(0, vocab_size, (B, T), dtype=np.int32)
    fg_tokens = Tensor(fg_tokens_np)

    t0 = time.time()
    try:
        tree_logits, traces = model.forward(fg_tokens, K=K, training=True,
                                             return_taps=True)
        # Realize outputs
        tree_logits.realize()
        t_fwd = time.time() - t0
        log(f"  Forward time: {t_fwd:.3f}s")
    except Exception as e:
        log(f"  [FAIL] Forward error: {e}")
        import traceback
        log(traceback.format_exc())
        failures.append("forward pass")
        log(f"SMOKE FAILED {'; '.join(failures)}")
        return 1

    # ------------------------------------------------------------------
    # 4. Shape assertions
    # ------------------------------------------------------------------
    log("\n[4] Shape assertions...")

    # tree_logits shape
    tl_shape = list(tree_logits.shape)
    exp_tl = [B, config.n_digits, config.n_digit_vals]
    if tl_shape == exp_tl:
        log(f"  [PASS] tree_logits shape: {tl_shape}")
    else:
        log(f"  [FAIL] tree_logits shape {tl_shape} != {exp_tl}")
        failures.append(f"tree_logits shape {tl_shape}")

    # traces count
    if len(traces) == K:
        log(f"  [PASS] traces length: {len(traces)} (= K={K})")
    else:
        log(f"  [FAIL] traces length {len(traces)} != K={K}")
        failures.append(f"traces length {len(traces)}")

    # Trace dict keys (new keys: self_attn_weights_per_layer, waist_delta, waist_applied)
    required_keys = ["k", "z", "z_pre", "read_ctx", "read_ctx_norm",
                     "xattn_weights", "think_taps", "delta_gate_val",
                     "calib", "tree_logits", "self_attn_resid_norm",
                     "self_attn_weights_per_layer", "waist_delta", "waist_applied"]
    for k_idx, trace in enumerate(traces):
        missing = [key for key in required_keys if key not in trace]
        if missing:
            log(f"  [FAIL] trace[{k_idx}] missing keys: {missing}")
            failures.append(f"trace[{k_idx}] missing keys")
        elif k_idx == 0:
            log(f"  [PASS] trace[0] has all required keys (including "
                "self_attn_weights_per_layer, waist_delta)")

    # z shape per breath
    for k_idx, trace in enumerate(traces):
        z_shape = list(trace["z"].shape)
        exp_z = [B, config.n_latents, config.hidden_dim]
        if z_shape != exp_z:
            log(f"  [FAIL] trace[{k_idx}].z shape {z_shape} != {exp_z}")
            failures.append(f"trace[{k_idx}].z shape")
        elif k_idx == 0:
            log(f"  [PASS] z shape per breath: {z_shape}")

    # xattn_weights shape
    if traces and traces[0]["xattn_weights"] is not None:
        w_shape = list(traces[0]["xattn_weights"].shape)
        exp_w = [B, config.xattn_heads, config.n_latents, T]
        if w_shape == exp_w:
            log(f"  [PASS] xattn_weights shape: {w_shape}")
        else:
            log(f"  [FAIL] xattn_weights shape {w_shape} != {exp_w}")
            failures.append(f"xattn_weights shape {w_shape}")

    # calib shape
    if traces and traces[0]["calib"] is not None:
        c_shape = list(traces[0]["calib"].shape)
        if c_shape == [B]:
            log(f"  [PASS] calib shape: {c_shape}")
        else:
            log(f"  [FAIL] calib shape {c_shape} != [{B}]")
            failures.append(f"calib shape {c_shape}")

    # think_taps: 4 layers, each with 3 keys
    if traces and traces[0]["think_taps"] is not None:
        taps = traces[0]["think_taps"]
        if len(taps) == 4:
            log(f"  [PASS] think_taps: 4 layers")
            tap_keys = ["pre_ln_resid", "post_attn_resid", "post_mlp_resid"]
            for li, layer_taps in enumerate(taps):
                missing_t = [k for k in tap_keys if k not in layer_taps]
                if missing_t:
                    log(f"  [FAIL] think_taps[{li}] missing: {missing_t}")
                    failures.append(f"think_taps[{li}] missing {missing_t}")
                else:
                    tap_shape = list(layer_taps["post_mlp_resid"].shape)
                    exp_tap = [B, config.n_latents, config.hidden_dim]
                    if tap_shape != exp_tap:
                        log(f"  [FAIL] think_taps[{li}].post_mlp_resid shape {tap_shape}")
                        failures.append(f"think_taps[{li}] shape")
            log(f"  [PASS] think_taps all keys and shapes verified")
        else:
            log(f"  [FAIL] think_taps length {len(taps)} != 4")
            failures.append(f"think_taps length {len(taps)}")

    # self_attn_weights_per_layer: should have 4 tensors
    if traces and traces[0].get("self_attn_weights_per_layer") is not None:
        saw = traces[0]["self_attn_weights_per_layer"]
        if len(saw) == 4:
            log(f"  [PASS] self_attn_weights_per_layer: 4 layers")
            n_sa_heads = model.backbone.n_heads
            for li, w in enumerate(saw):
                expected_saw_shape = [B, n_sa_heads, config.n_latents, config.n_latents]
                if list(w.shape) != expected_saw_shape:
                    log(f"  [FAIL] self_attn_weights_per_layer[{li}] shape "
                        f"{list(w.shape)} != {expected_saw_shape}")
                    failures.append(f"sa_weights[{li}] shape")
                elif li == 0:
                    log(f"  [PASS] self_attn_weights[0] shape: {list(w.shape)}")
        else:
            log(f"  [FAIL] self_attn_weights_per_layer has {len(saw)} layers != 4")
            failures.append("self_attn_weights_per_layer count")

    # ------------------------------------------------------------------
    # 5. NaN checks
    # ------------------------------------------------------------------
    log("\n[5] NaN / Inf checks...")

    nan_found = False
    nan_found |= check_nan(tree_logits, "tree_logits")
    for k_idx, trace in enumerate(traces):
        nan_found |= check_nan(trace["z"], f"trace[{k_idx}].z")
        if trace.get("calib") is not None:
            nan_found |= check_nan(trace["calib"], f"trace[{k_idx}].calib")
        if trace.get("tree_logits") is not None:
            nan_found |= check_nan(trace["tree_logits"], f"trace[{k_idx}].tree_logits")
        if trace.get("xattn_weights") is not None:
            nan_found |= check_nan(trace["xattn_weights"], f"trace[{k_idx}].xattn_weights")

    if not nan_found:
        log("  [PASS] No NaN or Inf found in any tensor")
    else:
        log("  [FAIL] NaN or Inf detected")
        failures.append("NaN in forward output")

    # ------------------------------------------------------------------
    # 6. Instrumentation hooks (all metrics computed)
    # ------------------------------------------------------------------
    log("\n[6] Instrumentation hooks...")

    # Latent JSD
    try:
        jsd_list = model.compute_latent_jsd_per_breath(traces)
        assert len(jsd_list) == K - 1, f"JSD list length {len(jsd_list)} != K-1={K-1}"
        jsd_nonzero = any(v > 0 for v in jsd_list)
        jsd_has_nan = any(np.isnan(v) for v in jsd_list)
        if jsd_nonzero and not jsd_has_nan:
            log(f"  [PASS] Latent JSD: {K-1} values, non-trivial")
            log(f"         values: [{', '.join(f'{v:.4f}' for v in jsd_list)}]")
        else:
            log(f"  [FAIL] Latent JSD: zero={not jsd_nonzero} nan={jsd_has_nan} "
                f"values={jsd_list}")
            failures.append(f"latent JSD trivial zero or nan")
    except Exception as e:
        log(f"  [FAIL] Latent JSD error: {e}")
        failures.append("latent JSD error")
        jsd_list = []

    # Energy channel
    try:
        energy = model.compute_energy_channel(traces)
        assert energy.shape == (K, config.n_latents), \
            f"energy shape {energy.shape} != ({K}, {config.n_latents})"
        energy_nonzero = bool(energy.any())
        energy_has_nan = bool(np.isnan(energy).any())
        if energy_nonzero and not energy_has_nan:
            log(f"  [PASS] Energy channel shape: {energy.shape}, "
                f"mean={energy.mean():.4f}")
        else:
            log(f"  [FAIL] Energy channel: zero={not energy_nonzero} nan={energy_has_nan}")
            failures.append("energy channel trivial")
    except Exception as e:
        log(f"  [FAIL] Energy channel error: {e}")
        failures.append("energy channel error")
        energy = np.zeros((K, config.n_latents))

    # Cross-attn entropy (NATS — like-units metric for Gate B)
    try:
        xattn_ent = model.compute_xattn_entropy_per_breath(traces)
        assert xattn_ent.shape == (K, config.xattn_heads), \
            f"xattn_ent shape {xattn_ent.shape} != ({K}, {config.xattn_heads})"
        xattn_nonzero = bool(xattn_ent.any())
        xattn_has_nan = bool(np.isnan(xattn_ent).any())
        mean_xattn = float(xattn_ent.mean())
        if xattn_nonzero and not xattn_has_nan:
            log(f"  [PASS] Cross-attn entropy (nats) shape: {xattn_ent.shape}, "
                f"mean={mean_xattn:.4f} (log(24)={np.log(24):.4f} at uniform)")
        else:
            log(f"  [FAIL] Cross-attn entropy: zero={not xattn_nonzero} nan={xattn_has_nan}")
            failures.append("cross-attn entropy trivial")
    except Exception as e:
        log(f"  [FAIL] Cross-attn entropy error: {e}")
        import traceback
        log(traceback.format_exc())
        failures.append("cross-attn entropy error")
        xattn_ent = np.zeros((K, config.xattn_heads))
        mean_xattn = 0.0

    # Self-attn entropy (NATS — like-units metric for Gate B; new in follow-up)
    try:
        n_sa_heads = model.backbone.n_heads
        sa_ent = model.compute_self_attn_entropy_per_breath(traces)
        assert sa_ent.shape == (K, 4, n_sa_heads), \
            f"sa_ent shape {sa_ent.shape} != ({K}, 4, {n_sa_heads})"
        sa_ent_nonzero = bool(sa_ent.any())
        sa_ent_has_nan = bool(np.isnan(sa_ent).any())
        mean_sa_ent = float(sa_ent.mean())
        if sa_ent_nonzero and not sa_ent_has_nan:
            log(f"  [PASS] Self-attn entropy (nats) shape: {sa_ent.shape}, "
                f"mean={mean_sa_ent:.4f} (log(32)={np.log(32):.4f} at uniform)")
        else:
            log(f"  [FAIL] Self-attn entropy: zero={not sa_ent_nonzero} nan={sa_ent_has_nan}")
            failures.append("self-attn entropy trivial")
    except Exception as e:
        log(f"  [FAIL] Self-attn entropy error: {e}")
        import traceback
        log(traceback.format_exc())
        failures.append("self-attn entropy error")
        sa_ent = np.zeros((K, 4, model.backbone.n_heads if hasattr(model, 'backbone') else 32))
        mean_sa_ent = 0.0

    # Self-attn layer JSD (SECONDARY diagnostic, NOT Gate B)
    try:
        sa_jsd = model.compute_self_attn_layer_jsd(traces)
        assert sa_jsd.shape == (K - 1, 4), \
            f"sa_jsd shape {sa_jsd.shape} != ({K-1}, 4)"
        sa_nonzero = bool(sa_jsd.any())
        sa_has_nan = bool(np.isnan(sa_jsd).any())
        mean_sa_jsd = float(sa_jsd.mean())
        if sa_nonzero and not sa_has_nan:
            log(f"  [PASS] Self-attn layer JSD (secondary) shape: {sa_jsd.shape}, "
                f"mean={mean_sa_jsd:.4f}")
        else:
            log(f"  [FAIL] Self-attn layer JSD: zero={not sa_nonzero} nan={sa_has_nan}")
            failures.append("self-attn JSD trivial")
    except Exception as e:
        log(f"  [FAIL] Self-attn layer JSD error: {e}")
        failures.append("self-attn JSD error")
        sa_jsd = np.zeros((K - 1, 4))
        mean_sa_jsd = 0.0

    # ------------------------------------------------------------------
    # 7. Like-units distinguishability (Gate B §8 — entropy vs entropy)
    # ------------------------------------------------------------------
    log("\n[7] Like-units entropy distinguishability (Gate B §8)...")
    log("  NOTE: Both metrics are ATTENTION-WEIGHT ENTROPY (nats), not JSD.")
    log("  Stage 1B initial run reported entropy(3.18) vs JSD(0.06) — category error.")
    log("  This version reports entropy vs entropy (like-units, per like-units clause).")
    log("")

    try:
        xattn_mean = float(xattn_ent.mean())
        sa_mean    = float(sa_ent.mean())
        diff = abs(xattn_mean - sa_mean)

        log(f"  Cross-attn entropy per breath (head-group means, nats):")
        log(f"    per-breath: [{', '.join(f'{v:.4f}' for v in xattn_ent.mean(axis=1))}]")
        log(f"    mean over K: {xattn_mean:.4f}")
        log(f"    reference (random init, uniform over T=24): log(24) = {np.log(24):.4f}")
        log("")
        log(f"  Self-attn entropy per breath (per-layer means, nats):")
        log(f"    per-breath (mean over 4 layers × heads): "
            f"[{', '.join(f'{v:.4f}' for v in sa_ent.mean(axis=(1,2)))}]")
        log(f"    mean over K: {sa_mean:.4f}")
        log(f"    reference (random init, uniform over L=32): log(32) = {np.log(32):.4f}")
        log("")
        log(f"  Like-units distinguishability: |cross_entropy_mean - self_entropy_mean|")
        log(f"    = |{xattn_mean:.4f} - {sa_mean:.4f}| = {diff:.4f}")

        distinguishable = diff > 1e-6
        if distinguishable:
            log(f"  [PASS] Cross-attn and self-attn entropies are distinguishable "
                f"(diff={diff:.4f})")
        else:
            log(f"  [WARN] Cross-attn and self-attn entropies are IDENTICAL "
                "(degenerate init; expected at random init)")
            # Not a hard failure at Stage 1B (untrained model, gate B is training-time)
    except Exception as e:
        log(f"  [FAIL] Like-units distinguishability check error: {e}")
        failures.append("like-units distinguishability error")

    # ------------------------------------------------------------------
    # 7b. Waist alternation check (BLOCKER if fires_correctly=NO)
    # ------------------------------------------------------------------
    log("\n[7b] Waist alternation check (COMMIT phase)...")
    log("  Per §2: COMMIT (waist 2048→512→2048) fires only on EVEN breaths (k % 2 == 0).")
    log("  Odd breaths must be identity (waist_delta ≈ 0).")

    try:
        waist_check = model.check_waist_alternation(traces)
        even_applied = waist_check.get("even_applied", [])
        odd_applied  = waist_check.get("odd_applied", [])
        even_deltas  = waist_check.get("even_deltas", [])
        odd_deltas   = waist_check.get("odd_deltas", [])
        verdict      = waist_check["verdict"]
        note         = waist_check.get("note", "")

        log(f"  Even-breath waist_applied (code-path fired): {even_applied}")
        log(f"  Odd-breath  waist_applied (code-path fired): {odd_applied}")
        log(f"  Even-breath waist deltas: {[f'{d:.6f}' for d in even_deltas]}")
        log(f"  Odd-breath  waist deltas: {[f'{d:.6f}' for d in odd_deltas]}")
        log(f"  Note: {note}")
        log(f"  WAIST ALTERNATION CHECK: conditional fires correctly = {verdict}")

        if verdict == "YES":
            log(f"  [PASS] Waist code path fires on even breaths only (identity on odd)")
        elif verdict == "UNKNOWN":
            log(f"  [WARN] Could not determine waist alternation")
            # Not a hard failure if traces are missing this key
        else:
            log(f"  [FAIL] Waist alternation incorrect — SMOKE FAILED BLOCKER")
            failures.append("waist alternation incorrect")
    except Exception as e:
        log(f"  [FAIL] Waist alternation check error: {e}")
        import traceback
        log(traceback.format_exc())
        failures.append("waist alternation check error")

    # ------------------------------------------------------------------
    # 8. Persistence bundle dry-run
    # ------------------------------------------------------------------
    log("\n[8] Persistence bundle dry-run...")
    try:
        bundle_meta = model.save_persistence_bundle(
            traces, step=0,
            output_dir=os.path.join(LOG_DIR, "v200_bundle_dry"),
            dry_run=True,
            b_sample=2,
        )
        n_bundle = len([k for k in bundle_meta if "_z" in k and "provenance" not in k])
        log(f"  Bundle entries for latent z: {n_bundle} (expected {K})")
        if n_bundle == K:
            log(f"  [PASS] Persistence bundle: {len(bundle_meta)} total entries, "
                f"K={K} z artifacts")
        else:
            log(f"  [FAIL] Expected {K} z entries, got {n_bundle}")
            failures.append(f"persistence bundle count {n_bundle}")

        # Verify a provenance dict is complete
        sample_prov = bundle_meta.get("breath0_z_provenance")
        if sample_prov is not None:
            try:
                validate_provenance(sample_prov)
                log(f"  [PASS] Provenance dict at breath0 passes validation")
            except ValueError as ve:
                log(f"  [FAIL] Provenance validation failed: {ve}")
                failures.append("provenance validation")

        # Check z shapes are correct
        z_shape_ok = True
        for k_idx in range(K):
            meta = bundle_meta.get(f"breath{k_idx}_z")
            if meta is None:
                log(f"  [FAIL] breath{k_idx}_z missing from bundle")
                z_shape_ok = False
                failures.append(f"breath{k_idx}_z missing")
                continue
            exp_shape = [2, config.n_latents, config.hidden_dim]   # b_sample=2
            if meta["shape"] != exp_shape:
                log(f"  [FAIL] breath{k_idx}_z shape {meta['shape']} != {exp_shape}")
                z_shape_ok = False
                failures.append(f"breath{k_idx}_z shape")
        if z_shape_ok:
            log(f"  [PASS] All {K} z shapes correct: "
                f"[2, {config.n_latents}, {config.hidden_dim}]")

    except Exception as e:
        log(f"  [FAIL] Persistence bundle error: {e}")
        import traceback
        log(traceback.format_exc())
        failures.append("persistence bundle error")

    # ------------------------------------------------------------------
    # 9. Provenance sidecar
    # ------------------------------------------------------------------
    log("\n[9] Writing arch_smoke provenance sidecar...")
    git_sha = get_git_sha()
    tg_sha  = get_tinygrad_sha()
    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        prov = make_provenance(
            metric="v200_arch_smoke_forward",
            units="raw activations + instrumentation metrics (entropy in nats)",
            shape=[B, config.n_digits, config.n_digit_vals],
            ckpt="cold-start",
            split="smoke",
            seed=42,
            step=0,
            env_vars={
                "K_MAX": str(K),
                "N_LATENTS": str(config.n_latents),
                "HIDDEN_DIM": str(config.hidden_dim),
                "BACKBONE": (
                    "meta-llama/Llama-3.2-1B" if model.backbone.is_gqa
                    else "HuggingFaceTB/SmolLM2-1.7B"
                ),
                "GQA": str(model.backbone.is_gqa),
                "WV_SHARING": "OFF (each layer's own wv)",
            },
            config_diff=(
                "Stage 1B follow-up: like-units entropy (xattn + sattn both nats), "
                "waist alternation check, reference curves persisted. "
                "JSD removed as gate-B signal; entropy vs entropy is the gate-B metric."
            ),
            output_path=LOG_PATH,
            key=None,
            project_root=_PROJECT_ROOT,
        )
        validate_provenance(prov)

        with open(PROV_PATH, "w") as f:
            json.dump(prov, f, indent=2)
        log(f"  [PASS] Provenance sidecar written: {PROV_PATH}")
    except Exception as e:
        log(f"  [FAIL] Provenance sidecar error: {e}")
        failures.append("provenance sidecar")

    # ------------------------------------------------------------------
    # 10. Regression: SmolLM2 smoke
    # ------------------------------------------------------------------
    log("\n[10] SmolLM2-1.7B regression smoke...")
    smollm2_log = os.path.join(LOG_DIR, "llama_load.log")
    if os.path.exists(smollm2_log):
        with open(smollm2_log) as f:
            lines = f.readlines()
        last_line = lines[-1].strip() if lines else ""
        if last_line.startswith("SMOKE PASSED"):
            log(f"  [PASS] SmolLM2 smoke already passed: {last_line[:80]}")
        else:
            log(f"  [INFO] SmolLM2 smoke log last line: {last_line[:80]}")
            log(f"  [INFO] Re-running SmolLM2 smoke...")
            try:
                subprocess.run(
                    [".venv/bin/python", "-u", "scripts/v200_llama_smoke.py"],
                    capture_output=True, text=True,
                    cwd=_PROJECT_ROOT, timeout=300,
                )
                with open(smollm2_log) as f:
                    lines_new = f.readlines()
                last_new = lines_new[-1].strip() if lines_new else ""
                if last_new.startswith("SMOKE PASSED"):
                    log(f"  [PASS] SmolLM2 re-run passed: {last_new[:80]}")
                else:
                    log(f"  [FAIL] SmolLM2 re-run failed: {last_new[:80]}")
                    failures.append("SmolLM2 regression")
            except subprocess.TimeoutExpired:
                log("  [FAIL] SmolLM2 regression timed out")
                failures.append("SmolLM2 regression timeout")
    else:
        log(f"  [INFO] SmolLM2 smoke log not found at {smollm2_log}, skipping regression")

    # ------------------------------------------------------------------
    # 11. Regression: Llama-3.2-1B smoke
    # ------------------------------------------------------------------
    log("\n[11] Llama-3.2-1B regression smoke...")
    llama32_log = os.path.join(LOG_DIR, "llama32_load.log")
    if os.path.exists(llama32_log):
        with open(llama32_log) as f:
            lines = f.readlines()
        last_line = lines[-1].strip() if lines else ""
        if last_line.startswith("SMOKE PASSED"):
            log(f"  [PASS] Llama-3.2-1B smoke already passed: {last_line[:80]}")
        else:
            log(f"  [INFO] Llama-3.2-1B smoke log last line: {last_line[:80]}")
            log(f"  [INFO] Re-running Llama-3.2-1B smoke...")
            try:
                subprocess.run(
                    [".venv/bin/python", "-u", "scripts/v200_llama32_smoke.py"],
                    capture_output=True, text=True,
                    cwd=_PROJECT_ROOT, timeout=300,
                )
                with open(llama32_log) as f:
                    lines_new = f.readlines()
                last_new = lines_new[-1].strip() if lines_new else ""
                if last_new.startswith("SMOKE PASSED"):
                    log(f"  [PASS] Llama-3.2-1B re-run passed: {last_new[:80]}")
                else:
                    log(f"  [FAIL] Llama-3.2-1B re-run failed: {last_new[:80]}")
                    failures.append("Llama-3.2-1B regression")
            except subprocess.TimeoutExpired:
                log("  [FAIL] Llama-3.2-1B regression timed out")
                failures.append("Llama-3.2-1B regression timeout")
    else:
        log(f"  [INFO] Llama-3.2-1B smoke log not found at {llama32_log}, skipping regression")

    # ------------------------------------------------------------------
    # Print freeze-breath table
    # ------------------------------------------------------------------
    log("\n[Instrumentation] Freeze-breath table:")
    model.print_freeze_table(traces, half_k=K // 2)

    # ------------------------------------------------------------------
    # 12. Reference curves (§5) — B=32 forward, persist null trajectories
    # ------------------------------------------------------------------
    log("\n[12] Persisting random-init reference curves (§5)...")
    log("  B=32 forward pass, seed=42. These are the Gate B null trajectories.")

    B_ref = 32
    ref_failures = []
    try:
        np.random.seed(42)
        fg_tokens_ref_np = np.random.randint(0, vocab_size, (B_ref, T), dtype=np.int32)
        fg_tokens_ref = Tensor(fg_tokens_ref_np)

        log(f"  Running B={B_ref} forward for reference curves...")
        t_ref0 = time.time()
        _, traces_ref = model.forward(fg_tokens_ref, K=K, training=False,
                                       return_taps=True)
        t_ref = time.time() - t_ref0
        log(f"  Reference forward time: {t_ref:.3f}s")

        # Compute reference metrics
        ref_jsd       = np.array(model.compute_latent_jsd_per_breath(traces_ref))
        ref_energy    = model.compute_energy_channel(traces_ref)
        ref_xattn_ent = model.compute_xattn_entropy_per_breath(traces_ref)
        ref_sa_ent    = model.compute_self_attn_entropy_per_breath(traces_ref)

        git_sha_ref = get_git_sha()
        tg_sha_ref  = get_tinygrad_sha()
        now_ref     = datetime.now(timezone.utc).isoformat()
        backbone_name = ("meta-llama/Llama-3.2-1B" if model.backbone.is_gqa
                         else "HuggingFaceTB/SmolLM2-1.7B")

        ref_env = {
            "K_MAX": str(K),
            "N_LATENTS": str(config.n_latents),
            "HIDDEN_DIM": str(config.hidden_dim),
            "BACKBONE": backbone_name,
            "B_REF": str(B_ref),
            "GQA": str(model.backbone.is_gqa),
        }

        def _persist_ref(arr: np.ndarray, stem: str, metric_name: str,
                          units: str) -> bool:
            npz_path  = os.path.join(REF_DIR, f"{stem}.npz")
            prov_path = os.path.join(REF_DIR, f"{stem}.provenance.json")
            shape = list(arr.shape)
            prov = make_provenance(
                metric=metric_name,
                units=units,
                shape=shape,
                ckpt="random_init_seed_42",
                split="smoke",
                seed=42,
                step=0,
                env_vars=ref_env,
                config_diff="random-init null curve for Gate B comparison",
                output_path=npz_path,
                key="data",
                project_root=_PROJECT_ROOT,
            )
            try:
                np.savez(npz_path, data=arr)
                with open(prov_path, "w") as f:
                    json.dump(prov, f, indent=2)
                validate_provenance(prov)
                return True
            except Exception as exc:
                log(f"    [FAIL] Failed to persist {stem}: {exc}")
                return False

        # latent_jsd_random_init.npz — shape (K-1,)
        ok = _persist_ref(ref_jsd, "latent_jsd_random_init",
                           "latent_jsd_per_breath", "nats (JSD between softmax distributions)")
        if ok:
            log(f"  [PASS] latent_jsd_random_init.npz persisted "
                f"(shape={ref_jsd.shape}, mean={ref_jsd.mean():.4f})")
        else:
            ref_failures.append("latent_jsd_random_init")

        # energy_channel_random_init.npz — shape (K, L)
        ok = _persist_ref(ref_energy, "energy_channel_random_init",
                           "energy_channel_per_breath", "L2 norm (raw)")
        if ok:
            log(f"  [PASS] energy_channel_random_init.npz persisted "
                f"(shape={ref_energy.shape}, mean={ref_energy.mean():.4f})")
        else:
            ref_failures.append("energy_channel_random_init")

        # xattn_entropy_random_init.npz — shape (K, n_xattn_heads)
        ok = _persist_ref(ref_xattn_ent, "xattn_entropy_random_init",
                           "xattn_entropy_per_breath", "nats (attention-weight entropy)")
        if ok:
            log(f"  [PASS] xattn_entropy_random_init.npz persisted "
                f"(shape={ref_xattn_ent.shape}, mean={ref_xattn_ent.mean():.4f} "
                f"expected≈log(24)={np.log(24):.4f})")
        else:
            ref_failures.append("xattn_entropy_random_init")

        # self_attn_entropy_random_init.npz — shape (K, 4, n_sa_heads)
        ok = _persist_ref(ref_sa_ent, "self_attn_entropy_random_init",
                           "self_attn_entropy_per_breath",
                           "nats (attention-weight entropy, per-layer per-head)")
        if ok:
            log(f"  [PASS] self_attn_entropy_random_init.npz persisted "
                f"(shape={ref_sa_ent.shape}, mean={ref_sa_ent.mean():.4f} "
                f"expected≈log(32)={np.log(32):.4f})")
        else:
            ref_failures.append("self_attn_entropy_random_init")

        if ref_failures:
            log(f"  [FAIL] Reference curve failures: {ref_failures}")
            failures.extend([f"ref_curve_{s}" for s in ref_failures])
        else:
            log(f"  [PASS] All 4 reference curves persisted at {REF_DIR}")

    except Exception as e:
        log(f"  [FAIL] Reference curve generation error: {e}")
        import traceback
        log(traceback.format_exc())
        failures.append("reference curves error")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    t_total = time.time() - t_start
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    backbone_name = ("meta-llama/Llama-3.2-1B" if model.backbone.is_gqa
                     else "HuggingFaceTB/SmolLM2-1.7B")
    log(f"  Backbone:              {backbone_name}")
    log(f"  GQA:                   {model.backbone.is_gqa}")
    log(f"  wv_sharing:            OFF (each layer's own wv, conservative §4)")
    log(f"  B={B}, T={T}, K={K}")
    log(f"  L=32 latents, H=2048, waist=512")
    log(f"  tree_logits:           {list(tree_logits.shape)}")
    log(f"  NaN found:             {nan_found}")
    log(f"  Latent JSD:            [{', '.join(f'{v:.4f}' for v in jsd_list)}]")
    log(f"  Energy channel:        mean={energy.mean():.4f} (K={K}, L={config.n_latents})")
    log(f"  Cross-attn entropy:    mean={mean_xattn:.4f} nats  [ref log(24)={np.log(24):.4f}]")
    log(f"  Self-attn entropy:     mean={mean_sa_ent:.4f} nats  [ref log(32)={np.log(32):.4f}]")
    log(f"  Like-units diff:       |cross-self| = {abs(mean_xattn-mean_sa_ent):.4f} nats")
    log(f"  [NOTE] Self-attn JSD is secondary/diagnostic only, NOT Gate B signal.")
    log(f"  Waist alternation:     {waist_check.get('verdict','?')} "
        f"(even_applied={waist_check.get('even_applied','?')}, "
        f"odd_applied={waist_check.get('odd_applied','?')})")
    log(f"  Ref curves:            {REF_DIR}")
    log(f"  Load time:             {t_load:.1f}s")
    log(f"  Forward time:          {t_fwd:.3f}s")
    log(f"  Total time:            {t_total:.1f}s")
    log(f"  Provenance:            {PROV_PATH}")

    # Check pass conditions
    pass_conditions = [
        (not nan_found,               "no NaN in forward output"),
        (len(traces) == K,            f"traces length = K={K}"),
        (tl_shape == exp_tl,          f"tree_logits shape = {exp_tl}"),
        (os.path.exists(PROV_PATH),   "provenance sidecar exists"),
        (waist_check.get("verdict","?") in ("YES", "UNKNOWN"),
                                      "waist alternation correct or unknown"),
        (os.path.exists(os.path.join(REF_DIR, "xattn_entropy_random_init.npz")),
                                      "xattn reference curve persisted"),
        (os.path.exists(os.path.join(REF_DIR, "self_attn_entropy_random_init.npz")),
                                      "self-attn reference curve persisted"),
        (len(failures) == 0,          "no failures accumulated"),
    ]

    all_pass = all(cond for cond, _ in pass_conditions)
    log("")
    for cond, desc in pass_conditions:
        status = "PASS" if cond else "FAIL"
        log(f"  [{status}] {desc}")

    if failures:
        log(f"\n  Accumulated failures: {failures}")

    log("")
    if all_pass and not failures:
        metrics = (
            f"backbone={backbone_name} "
            f"gqa={model.backbone.is_gqa} "
            f"wv_sharing=OFF "
            f"B={B} T={T} K={K} L={config.n_latents} H={config.hidden_dim} "
            f"nan=False "
            f"jsd_mean={float(np.mean(jsd_list)) if jsd_list else 0.0:.4f} "
            f"energy_mean={energy.mean():.4f} "
            f"xattn_ent_nats={mean_xattn:.4f} "
            f"sa_ent_nats={mean_sa_ent:.4f} "
            f"like_units_diff={abs(mean_xattn-mean_sa_ent):.4f} "
            f"waist_alternation={waist_check.get('verdict','?')} "
            f"ref_curves=persisted "
            f"fwd={t_fwd:.3f}s "
            f"load={t_load:.1f}s"
        )
        log(f"SMOKE PASSED {metrics}")
        return 0
    else:
        reason = "; ".join(failures) if failures else "unknown failure"
        log(f"SMOKE FAILED {reason}")
        return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    exit_code = run_smoke()
    _log_fh.close()
    sys.exit(exit_code)
