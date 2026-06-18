"""probe_cell_smoothing.py — Over-smoothing + value-basin diagnostic for the cell-MP perceiver.

Loads a perceiver checkpoint and runs perceiver_breathing_forward EAGERLY
(no TinyJit) with PERCEIVER_PROBE_SMOOTH=1 to capture per-breath diagnostics.

TWO DIAGNOSTIC SECTIONS in one run:

SECTION A — OVER-SMOOTHING (peer-cosine)
  Pairwise cosine similarity between constraint-peer cell hidden states.
  OVER-SMOOTHING HYPOTHESIS: repeated cell<->cell message-passing drives peer
  cells to near-identical representations (cosine->1) by the last breath,
  collapsing the final readout to chance.

  CONFIRMED if peer_cos RISES across breaths toward ~0.9-1.0 (especially b6->b7)
              AND/OR the gap (peer_cos - nonpeer_cos) COLLAPSES late.
  REFUTED   if peer_cos stays moderate (<~0.7) and roughly flat across breaths.

SECTION B — VALUE-BASIN / MODE-COLLAPSE (value-distribution drift)
  Does the K-breath iteration drift toward over-predicting a few high-frequency
  values ("Christmas-drawing" collapse), regardless of the puzzle?

  Per breath, computed over VALID cells only (cell_valid mask applied):
    pred_entropy[k]  : H(p_pred) in nats — entropy of the marginal
                       predicted-value distribution (argmax histogram over
                       N_MAX values, normalised to a probability vector).
                       Falling entropy = predictions concentrating.
    top_value_frac[k]: fraction of valid cells whose argmax is the SINGLE
                       most-common predicted value at breath k.
                       Rising = a dominant "Christmas" basin forming.
    kl_pred_true[k]  : KL(p_pred || p_true) in nats — divergence of the
                       predicted value mix from the true value mix.
                       Rising = drifting away from the true distribution.
  true_entropy       : reference (constant) — H of the TRUE value distribution
                       over all valid cells in the probe batches.

  TRUE SOLUTION VALUES are read from batch.gold (shape (B, 49) int, values
  1..7 for valid cells; 0 for padding), sourced from kenken_data.py field
  "gold_solution" / KenKenBatch.gold — the per-cell gold digit used as the CE
  target in perceiver_train.py (gold_idx = (gold - 1).clip(0, N_MAX - 1)).

  BASIN / MODE-COLLAPSE CONFIRMED:
    pred_entropy FALLS across breaths (especially below true_entropy),
    top_value_frac RISES toward a dominant single value, AND kl_pred_true RISES.
    The iteration is drifting toward high-prior basins.

  BASIN REFUTED:
    pred_entropy stays >= true_entropy and roughly flat/rising, AND
    top_value_frac stays near the true most-common-value frequency.
    No basin drift — predictions track the true value mix.

  NOTE — distinguish from over-smoothing-to-chance:
    A UNIFORM collapse looks like: pred_entropy HIGH (~ln(7) ≈ 1.95 nats),
    top_value_frac LOW (~1/7 ≈ 0.14) — that is OVER-SMOOTHING-TO-CHANCE, NOT
    a value-basin.  A value-basin has LOW pred_entropy + HIGH top_value_frac.

Usage:
    # antismooth config (default):
    PERCEIVER_TASK=1 PERCEIVER_CELL_MP=1 PERCEIVER_PERSIST_CELLS=1 \\
    PERCEIVER_CELL_RENORM=1 PERCEIVER_THINK_RENORM=1 \\
    PERCEIVER_CELL_MP_DECAY=1 PERCEIVER_CELL_IDENTITY=1 \\
    PERCEIVER_BALL_PATH=per_constraint \\
    CKPT_PATH=.cache/perceiver_ckpts/perceiver_antismooth/<ckpt>.safetensors \\
    python scripts/probe_cell_smoothing.py

    # Override checkpoint / data / batches:
    CKPT_PATH=.cache/perceiver_ckpts/perceiver_brick2_deduction/perceiver_brick2_deduction_final.safetensors
    KENKEN_TEST=.cache/kenken_test.jsonl  PROBE_BATCHES=3  BATCH=8
"""
from __future__ import annotations

import os
import sys

# Env must be set BEFORE importing perceiver_poincare (module-level flags read at import).
# The script reads them from the environment; set defaults matching the trained run.
os.environ.setdefault("PERCEIVER_TASK", "1")
os.environ.setdefault("PERCEIVER_CELL_MP", "1")
os.environ.setdefault("PERCEIVER_PERSIST_CELLS", "1")
os.environ.setdefault("PERCEIVER_CELL_RENORM", "1")
os.environ.setdefault("PERCEIVER_THINK_RENORM", "1")
os.environ.setdefault("PERCEIVER_CELL_MP_DECAY", "1")
os.environ.setdefault("PERCEIVER_CELL_IDENTITY", "1")
os.environ.setdefault("PERCEIVER_BALL_PATH", "per_constraint")
os.environ.setdefault("PERCEIVER_K_MAX", "8")
os.environ.setdefault("PERCEIVER_N_GLOBAL", "4")
os.environ.setdefault("PERCEIVER_PROBE_SMOOTH", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.kenken_data import load_jsonl
from mycelium.perceiver_poincare import (
    PERCEIVER_K_MAX, PERCEIVER_N_GLOBAL,
    attach_perceiver_params, perceiver_state_dict,
    perceiver_breathing_forward,
)
from mycelium.perceiver_poincare_data import PerceiverLoader, latent_capacity


# ---- fp32 cast (mirror perceiver_train.cast_layers_fp32) ---------------------

def cast_layers_fp32(model):
    def _cast(obj, attr):
        t = getattr(obj, attr)
        if t.dtype == dtypes.half:
            setattr(obj, attr, t.cast(dtypes.float).contiguous().realize())
    _cast(model.embed, "weight")
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out"):
        _cast(sw, a)
    for layer in model.block.layers:
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            _cast(layer, a)


# ---- checkpoint load (mirror kenken_train.load_ckpt pattern) -----------------

def load_ckpt(model, path: str) -> None:
    """Load a perceiver safetensors checkpoint into model via .assign().

    Merges the backbone state dict (ln_f, shared, phase*) and the perceiver
    state dict (perc.*) keys.  Missing keys are skipped with a warning; shape
    mismatches attempt a reshape, else skip.
    """
    sd = safe_load(path)
    # Build target dict: backbone keys + perceiver keys (same as model_state_dict
    # in perceiver_train.py but constructed from the live model objects).
    targets: dict[str, Tensor] = {}
    # backbone
    targets["ln_f.g"] = model.ln_f_g
    targets["ln_f.b"] = model.ln_f_b
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out",
              "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b"):
        targets[f"shared.{a}"] = getattr(sw, a)
    for i, layer in enumerate(model.block.layers):
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            targets[f"phase{i}.{a}"] = getattr(layer, a)
    # perceiver
    targets.update(perceiver_state_dict(model))

    missing = []
    for name, dst in targets.items():
        if name not in sd:
            missing.append(name)
            continue
        src = sd[name].to(dst.device).realize()
        if src.shape != dst.shape:
            try:
                src = src.reshape(dst.shape)
            except Exception:
                missing.append(f"{name}(shape mismatch {src.shape} vs {dst.shape})")
                continue
        if src.dtype != dst.dtype:
            src = src.cast(dst.dtype)
        dst.assign(src).realize()

    if missing:
        print(f"  [load_ckpt] skipped {len(missing)} keys: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    else:
        print(f"  [load_ckpt] all {len(targets)} keys loaded OK")


# ---- main --------------------------------------------------------------------

def main():
    CKPT_PATH = os.environ.get(
        "CKPT_PATH",
        ".cache/perceiver_ckpts/perceiver_brick2_deduction/"
        "perceiver_brick2_deduction_final.safetensors",
    )
    KENKEN_TEST = os.environ.get("KENKEN_TEST", ".cache/kenken_test.jsonl")
    KENKEN_TRAIN = os.environ.get("KENKEN_TRAIN", ".cache/kenken_train.jsonl")
    PROBE_BATCHES = int(os.environ.get("PROBE_BATCHES", "3"))
    BATCH = int(os.environ.get("BATCH", "8"))
    K = int(os.environ.get("PERCEIVER_K_MAX", str(PERCEIVER_K_MAX)))
    BALL_PATH = os.environ.get("PERCEIVER_BALL_PATH", "per_constraint").strip().lower()
    N_GLOBAL = int(os.environ.get("PERCEIVER_N_GLOBAL", str(PERCEIVER_N_GLOBAL)))

    print("=" * 66)
    print("probe_cell_smoothing.py — over-smoothing + value-basin diagnostic")
    print(f"  ckpt    : {CKPT_PATH}")
    print(f"  data    : {KENKEN_TEST}")
    print(f"  K={K}  B={BATCH}  batches={PROBE_BATCHES}  ball_path={BALL_PATH}")
    print(f"  device  : {Device.DEFAULT}")
    print("=" * 66)

    # ---- build corpus shape (needed for L_max / stable JIT shapes) -----------
    train_recs = load_jsonl(KENKEN_TRAIN)
    test_recs  = load_jsonl(KENKEN_TEST)
    corpus_n_cages_max = max(
        max(len(r["cages"]) for r in train_recs),
        max(len(r["cages"]) for r in test_recs),
    )
    N_CAGES_MAX = int(os.environ.get("KENKEN_N_CAGES_MAX", str(corpus_n_cages_max)))
    L_MAX = latent_capacity(N_CAGES_MAX, N_GLOBAL)
    print(f"  n_cages_max={N_CAGES_MAX}  L_max={L_MAX}")

    # ---- build model ---------------------------------------------------------
    print("\n[1] Building model (Pythia-410M + perceiver params)...")
    cfg = Config()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    cast_layers_fp32(model)
    attach_perceiver_params(model, hidden=cfg.hidden, n_heads=cfg.n_heads,
                            L_max=L_MAX, k_max=K)

    # ---- load checkpoint -----------------------------------------------------
    print(f"\n[2] Loading checkpoint: {CKPT_PATH}")
    load_ckpt(model, CKPT_PATH)

    # ---- build data loader ---------------------------------------------------
    test_loader = PerceiverLoader(KENKEN_TEST, batch_size=BATCH, seed=99,
                                  n_cages_max=N_CAGES_MAX, n_global=N_GLOBAL)

    # ---- run eager forward, accumulate both diagnostics ----------------------
    print(f"\n[3] Running {PROBE_BATCHES} eval batch(es) EAGERLY (no JIT)...")
    Tensor.training = False

    # --- Section A: over-smoothing accumulators ---
    # Per-breath sums of peer / nonpeer cosine scalars returned by smooth_history.
    peer_sums    = [0.0] * K
    nonpeer_sums = [0.0] * K

    # --- Section B: value-basin accumulators ---
    # Per-breath, we accumulate numpy arrays to average over batches.
    # Basin metrics are computed from cell_logits_history (list of K Tensors,
    # each (B, 49, N_MAX)) masked by cell_valid (B, 49) float.
    # True solution values are in batch.gold (B, 49) int, values 1..7, 0=padding.
    # The logit index i corresponds to predicted value i+1 (gold_idx = gold - 1).
    N_MAX = int(os.environ.get("PERCEIVER_N_MAX", "7"))  # values 1..N_MAX
    # Accumulators: sums over batches for averaging.
    pred_entropy_sum  = np.zeros(K, dtype=np.float64)  # H(p_pred) in nats
    top_value_frac_sum = np.zeros(K, dtype=np.float64) # frac cells = mode value
    kl_pred_true_sum  = np.zeros(K, dtype=np.float64)  # KL(p_pred||p_true) nats
    true_entropy_sum  = 0.0                             # reference, constant per batch
    # Initialise last-batch true_counts for the ref footer (overwritten each batch).
    true_counts = np.ones(N_MAX, dtype=np.float64)
    total_valid = float(N_MAX)

    batch_count = 0

    for batch in test_loader.iter_eval(batch_size=BATCH):
        result = perceiver_breathing_forward(
            model, batch, K=K, ball_path=BALL_PATH, collect_engagement=False)
        # When PERCEIVER_PROBE_SMOOTH=1 the return is a 6-tuple
        # (cell_logits_history, eng_history, sharp_reg, codebook_ortho_reg,
        #  value_balance_reg, smooth_history).
        assert len(result) == 6, (
            f"Expected 6-tuple from perceiver_breathing_forward with PROBE_SMOOTH=1, "
            f"got {len(result)}-tuple.  Is PERCEIVER_PROBE_SMOOTH=1 set?")
        cell_logits_history, _eng_history, _sharp_reg, _cb_ortho, _val_bal, smooth_history = result

        # --- Section A: over-smoothing ---
        assert len(smooth_history) == K, (
            f"smooth_history has {len(smooth_history)} entries; expected {K}")
        for k, (pc, npc) in enumerate(smooth_history):
            peer_sums[k]    += pc
            nonpeer_sums[k] += npc

        # --- Section B: value-basin ---
        # cell_valid: (B, 49) float Tensor — 1.0=valid, 0.0=pad.
        # batch.gold:  (B, 49) int Tensor  — gold digit 1..N_MAX, 0=pad.
        cv_np   = batch.cell_valid.realize().numpy().astype(np.float32)  # (B, 49)
        gold_np = batch.gold.realize().numpy().astype(np.int32)          # (B, 49)

        # Build the true value distribution over ALL valid cells in this batch.
        # gold values are 1..N_MAX; map to 0-indexed bins 0..N_MAX-1.
        valid_mask_flat = cv_np.reshape(-1) > 0.5               # (B*49,) bool
        gold_flat       = gold_np.reshape(-1)                   # (B*49,) int
        true_vals       = gold_flat[valid_mask_flat]            # valid gold digits (1..N_MAX)
        true_vals_idx   = (true_vals - 1).clip(0, N_MAX - 1)   # 0-indexed
        true_counts     = np.bincount(true_vals_idx, minlength=N_MAX).astype(np.float64)
        total_valid     = true_counts.sum()
        if total_valid < 1.0:
            batch_count += 1
            if batch_count >= PROBE_BATCHES:
                break
            continue
        p_true = true_counts / total_valid                      # (N_MAX,) normalised

        # H(p_true) in nats — constant reference for this batch.
        true_ent = -np.sum(p_true * np.log(p_true + 1e-12))
        true_entropy_sum += true_ent

        # Per-breath basin metrics.
        for k in range(K):
            # Realise the k-th breath logits: (B, 49, N_MAX).
            logits_np = cell_logits_history[k].realize().numpy().astype(np.float32)

            # Argmax over values dim -> predicted value index per cell: (B, 49).
            pred_idx = np.argmax(logits_np, axis=-1)           # (B, 49) int, 0..N_MAX-1

            # Restrict to valid cells.
            pred_valid = pred_idx.reshape(-1)[valid_mask_flat] # (n_valid,) int

            # Build predicted value histogram — normalise to distribution.
            pred_counts = np.bincount(pred_valid, minlength=N_MAX).astype(np.float64)
            n_valid_pred = pred_counts.sum()
            p_pred = pred_counts / (n_valid_pred + 1e-12)       # (N_MAX,)

            # (1) pred_entropy: H(p_pred) in nats.
            pred_ent = -np.sum(p_pred * np.log(p_pred + 1e-12))
            pred_entropy_sum[k] += pred_ent

            # (2) top_value_frac: fraction of valid cells with the modal predicted value.
            top_frac = pred_counts.max() / (n_valid_pred + 1e-12)
            top_value_frac_sum[k] += top_frac

            # (3) kl_pred_true: KL(p_pred || p_true) — divergence of predicted mix
            #     from the true value distribution.
            #     KL = sum_v p_pred[v] * log(p_pred[v] / p_true[v])
            #     Guard: skip bins where p_pred=0 (those contribute 0 to KL).
            safe_mask = p_pred > 1e-12
            kl = np.sum(
                p_pred[safe_mask] * np.log(p_pred[safe_mask] / (p_true[safe_mask] + 1e-12))
            )
            kl_pred_true_sum[k] += kl

        batch_count += 1
        if batch_count >= PROBE_BATCHES:
            break

    print(f"  ran {batch_count} batch(es)  (B={BATCH} each  ->  "
          f"~{batch_count * BATCH} puzzles total)\n")

    # ==========================================================================
    # SECTION A — OVER-SMOOTHING: peer-cosine table + verdict
    # ==========================================================================
    print("=" * 66)
    print("SECTION A — OVER-SMOOTHING (peer-cosine)")
    print("=" * 66)
    print(f"  {'breath':>6}  {'peer_cos':>9}  {'nonpeer_cos':>11}  "
          f"{'gap(peer-non)':>13}")
    print("-" * 66)
    for k in range(K):
        pc  = peer_sums[k]   / batch_count
        npc = nonpeer_sums[k] / batch_count
        gap = pc - npc
        print(f"  {k:>6}  {pc:>9.4f}  {npc:>11.4f}  {gap:>13.4f}")
    print("-" * 66)

    last_peer  = peer_sums[K - 1] / batch_count
    first_peer = peer_sums[0]     / batch_count
    delta_peer = last_peer - first_peer
    last_gap   = (peer_sums[K - 1] - nonpeer_sums[K - 1]) / batch_count
    first_gap  = (peer_sums[0]     - nonpeer_sums[0])     / batch_count
    delta_gap  = last_gap - first_gap

    print()
    print("VERDICT (over-smoothing)")
    print(f"  peer_cos b0={first_peer:.4f}  ->  b{K-1}={last_peer:.4f}  "
          f"(delta={delta_peer:+.4f})")
    print(f"  peer-nonpeer gap: b0={first_gap:.4f}  ->  b{K-1}={last_gap:.4f}  "
          f"(delta={delta_gap:+.4f})")
    print()
    if last_peer > 0.85 and delta_peer > 0.10:
        sm_verdict = "OVER-SMOOTHING CONFIRMED"
        sm_detail  = (f"peer_cos rises to {last_peer:.3f} (>0.85) and climbs "
                      f"{delta_peer:+.3f} across breaths.  The cell-MP drives peer "
                      f"cells to near-identical representations -> readout collapses "
                      f"to chance at the last breath.")
    elif last_peer > 0.70 and delta_peer > 0.05:
        sm_verdict = "OVER-SMOOTHING LIKELY (moderate)"
        sm_detail  = (f"peer_cos reaches {last_peer:.3f} (>0.70) and rises "
                      f"{delta_peer:+.3f}.  Some smoothing; may explain partial "
                      f"last-breath degradation.")
    else:
        sm_verdict = "OVER-SMOOTHING REFUTED"
        sm_detail  = (f"peer_cos stays at {last_peer:.3f} (<0.70) and changes only "
                      f"{delta_peer:+.3f} across breaths.  The b{K-1} chance CE is "
                      f"NOT caused by cell-MP over-smoothing -> look elsewhere (e.g. "
                      f"a last-breath gate/marker bug or CELL_RENORM interaction).")
    print(f"  ** {sm_verdict} **")
    print(f"  {sm_detail}")
    print()

    # ==========================================================================
    # SECTION B — VALUE-BASIN / MODE-COLLAPSE
    # ==========================================================================
    print("=" * 66)
    print("SECTION B — VALUE-BASIN / MODE-COLLAPSE")
    print("  true values: batch.gold  (B,49) int 1..N_MAX, field KenKenBatch.gold")
    print(f"  N_MAX={N_MAX}  uniform entropy = {np.log(N_MAX):.4f} nats  "
          f"true_entropy = {true_entropy_sum / batch_count:.4f} nats")
    print("=" * 66)

    true_ent_avg = true_entropy_sum / batch_count
    ln_n         = np.log(N_MAX)          # 1.9459 nats for N_MAX=7 — uniform ceiling

    print(f"  {'breath':>6}  {'pred_entropy':>12}  {'top_value_frac':>14}  "
          f"{'kl_pred_true':>12}")
    print(f"  {'':>6}  {'(nats)':>12}  {'':>14}  {'(nats)':>12}")
    print("-" * 66)
    for k in range(K):
        pe  = pred_entropy_sum[k]  / batch_count
        tvf = top_value_frac_sum[k] / batch_count
        kl  = kl_pred_true_sum[k]  / batch_count
        print(f"  {k:>6}  {pe:>12.4f}  {tvf:>14.4f}  {kl:>12.4f}")
    print("-" * 66)
    print(f"  {'ref':>6}  true_entropy={true_ent_avg:.4f}  "
          f"true_top_frac={true_counts.max() / (total_valid + 1e-12):.4f}  "
          f"uniform_H={ln_n:.4f}")
    print()

    # ---- basin verdict -------------------------------------------------------
    pe_first  = pred_entropy_sum[0]  / batch_count
    pe_last   = pred_entropy_sum[K-1] / batch_count
    tvf_first = top_value_frac_sum[0]  / batch_count
    tvf_last  = top_value_frac_sum[K-1] / batch_count
    kl_first  = kl_pred_true_sum[0]  / batch_count
    kl_last   = kl_pred_true_sum[K-1] / batch_count

    pe_delta  = pe_last  - pe_first
    tvf_delta = tvf_last - tvf_first
    kl_delta  = kl_last  - kl_first

    print("VERDICT (value-basin)")
    print(f"  pred_entropy:   b0={pe_first:.4f}  ->  b{K-1}={pe_last:.4f}  "
          f"(delta={pe_delta:+.4f};  true_entropy={true_ent_avg:.4f})")
    print(f"  top_value_frac: b0={tvf_first:.4f}  ->  b{K-1}={tvf_last:.4f}  "
          f"(delta={tvf_delta:+.4f})")
    print(f"  kl_pred_true:   b0={kl_first:.4f}  ->  b{K-1}={kl_last:.4f}  "
          f"(delta={kl_delta:+.4f})")
    print()

    # Three-way classification: basin vs uniform-collapse vs neither.
    falling_entropy  = pe_delta < -0.05
    below_true       = pe_last < true_ent_avg - 0.10
    rising_top_frac  = tvf_delta > 0.05
    high_top_frac    = tvf_last > 0.30
    rising_kl        = kl_delta > 0.05
    near_uniform_h   = pe_last > ln_n - 0.15     # close to uniform ceiling
    low_top_frac     = tvf_last < 1.0 / N_MAX + 0.05

    if (falling_entropy or below_true) and (rising_top_frac or high_top_frac) and rising_kl:
        basin_verdict = "BASIN / MODE-COLLAPSE CONFIRMED"
        basin_detail  = (
            f"pred_entropy FALLS {pe_delta:+.3f} (b0={pe_first:.3f} -> b{K-1}={pe_last:.3f}), "
            f"ending {'BELOW' if below_true else 'near'} true_entropy={true_ent_avg:.3f}.  "
            f"top_value_frac RISES {tvf_delta:+.3f} to {tvf_last:.3f} (>0.30) and "
            f"kl_pred_true RISES {kl_delta:+.3f}.  "
            f"The iteration is drifting toward high-prior value basins regardless of the puzzle."
        )
    elif near_uniform_h and low_top_frac and pe_last >= true_ent_avg - 0.05:
        basin_verdict = "BASIN REFUTED — OVER-SMOOTHING-TO-CHANCE (uniform)"
        basin_detail  = (
            f"pred_entropy is HIGH ({pe_last:.3f} near uniform ceiling {ln_n:.3f}) "
            f"and top_value_frac is LOW ({tvf_last:.3f} ~1/N).  "
            f"This is NOT a value-basin — it is over-smoothing to a near-uniform "
            f"distribution.  Cross-check with Section A peer_cos for confirmation."
        )
    elif not (falling_entropy or below_true) and not (rising_top_frac and high_top_frac):
        basin_verdict = "BASIN REFUTED"
        basin_detail  = (
            f"pred_entropy stays at {pe_last:.3f} (>= true_entropy={true_ent_avg:.3f} - 0.10) "
            f"and top_value_frac stays at {tvf_last:.3f} (not strongly dominated).  "
            f"No evidence of value-basin drift — predictions track the true value mix."
        )
    else:
        basin_verdict = "BASIN AMBIGUOUS"
        basin_detail  = (
            f"Mixed signals: pred_entropy={pe_last:.3f} (true={true_ent_avg:.3f}), "
            f"top_value_frac={tvf_last:.3f}, kl={kl_last:.3f}.  "
            f"Some basin pressure may be present; check with more PROBE_BATCHES."
        )

    print(f"  ** {basin_verdict} **")
    print(f"  {basin_detail}")
    print()


if __name__ == "__main__":
    main()
