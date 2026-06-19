"""analyze_search_guidance_gate.py — PUCT-search viability gate for the graph-coloring deducer.

GOAL
----
Decide whether a neural-guided search tier (PUCT over the deducer) is worth
building — BEFORE building any tree.  Tests three conditions on the HARD band
(deduction_depth >= 3, the DSATUR-backtrack-heavy regime where cell_acc ~0.51):

  G1 — POLICY-RANKS-GOLD
      At wrong/ambiguous vertices, does the color softmax place the gold color
      above the uniform prior (1/k = 0.333)?  If yes, the policy has exploitable
      signal for search branching.

  G2 — ENTROPY-FLAGS-AMBIGUITY
      Does per-vertex prediction entropy (or 1-margin) predict WHERE the
      deducer is wrong?  AUC of (entropy -> is-wrong) > ~0.60 means the
      residual ambiguous set is identifiable — search can focus effort there.

  G4 — CLAMP-TO-GOLD / OFF-DISTRIBUTION
      The deducer trained on all-zero input_cells (no givens).  If we clamp a
      few vertices to their gold colors and re-run, does cell_acc on the
      REMAINING unclamped valid vertices improve?
        PASS: clamping gold improves rest  -> off-distribution is manageable,
              clamping propagates,          -> search viable.
        FAIL: clamping degrades / no help  -> off-distribution breaks the
              deducer -> search futile (the v106 negative recurs), fix
              value/policy/calibration first.
      Sweeps n_clamp in {1, 2, 4}.
      Also checks: does the calibration head (scalar per breath) actually
      correlate with fraction-correct on partial (clamped) states?

VERDICT
-------
  GATE PASS  if G1 (gold prob > 1/k), G2 (AUC > 0.60), G4 (clamp helps).
  GATE FAIL  otherwise (with sub-gate breakdown).

GPU-FREE BUILD
--------------
  ast.parse always runs at import time.
  CPU selftest (synthetic) runs when FG_CKPT not set or SELFTEST=1.

RUN COMMAND (AMD GPU)
---------------------
  DEV=AMD FG_CKPT=.cache/fg_ckpts/fg_coloring_k16/fg_coloring_k16_final.safetensors \\
      .venv/bin/python3 scripts/analyze_search_guidance_gate.py
"""
from __future__ import annotations

import ast
import os
import sys

_THIS_FILE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(os.path.dirname(_THIS_FILE)))

import numpy as np

# ---------------------------------------------------------------------------
# AST parse gate (always runs, even on CPU)
# ---------------------------------------------------------------------------

def _ast_parse_ok() -> bool:
    with open(_THIS_FILE) as f:
        src = f.read()
    try:
        ast.parse(src)
        return True
    except SyntaxError as e:
        print(f"[ast.parse] FAILED: {e}", flush=True)
        return False


# ---------------------------------------------------------------------------
# AUC helper (no sklearn needed — trapezoidal rule over sorted thresholds)
# ---------------------------------------------------------------------------

def _auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Binary AUC-ROC: scores = higher means "positive"; labels in {0, 1}."""
    assert scores.shape == labels.shape
    n = len(scores)
    if n == 0:
        return 0.5
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Count: for each (pos, neg) pair, does pos score > neg score?
    # Efficient O(n log n) via sort.
    pos_scores = np.sort(scores[pos])
    neg_scores = np.sort(scores[neg])
    # Searchsorted: for each neg score, how many pos_scores > it?
    concordant = n_pos * n_neg - np.searchsorted(pos_scores, neg_scores, side="right").sum()
    return float(concordant) / float(n_pos * n_neg)


# ---------------------------------------------------------------------------
# CPU selftest (no GPU, no checkpoint)
# ---------------------------------------------------------------------------

def _selftest() -> bool:
    """Synthetic CPU selftest verifying G1/G2/G4 logic without GPU or checkpoint.

    Returns True if all assertions pass.
    """
    print("\n--- CPU SELFTEST ---", flush=True)
    passed = True

    # --- G1 selftest ---
    # Synthetic: 100 wrong vertices; gold color = 0.
    # Scenario A: policy ranks gold high (gold_prob ~ 0.5 >> 1/3).
    rng = np.random.RandomState(42)
    k = 3
    n = 100
    # logits that put gold=0 on top for most vertices
    logits_good = rng.randn(n, k).astype(np.float32)
    logits_good[:, 0] += 1.5           # gold = color 0, boosted
    probs_good  = np.exp(logits_good - logits_good.max(axis=1, keepdims=True))
    probs_good /= probs_good.sum(axis=1, keepdims=True)
    gold_prob_good = probs_good[:, 0].mean()   # should be > 1/k
    g1_pass_a = gold_prob_good > (1.0 / k)
    print(f"  G1 self-A: gold_prob={gold_prob_good:.3f} > 1/k={1/k:.3f} -> "
          f"{'PASS' if g1_pass_a else 'FAIL'}", flush=True)
    if not g1_pass_a:
        passed = False

    # Scenario B: random policy (gold_prob ~ 1/k) — should NOT pass if threshold applied.
    logits_rand = rng.randn(n, k).astype(np.float32)
    probs_rand  = np.exp(logits_rand - logits_rand.max(axis=1, keepdims=True))
    probs_rand /= probs_rand.sum(axis=1, keepdims=True)
    gold_prob_rand = probs_rand[:, 0].mean()
    g1_pass_b = gold_prob_rand > (1.0 / k)   # random should be ~= 1/k, close either way
    print(f"  G1 self-B: random gold_prob={gold_prob_rand:.3f} (expect ~{1/k:.3f}; "
          f"pass={g1_pass_b} is irrelevant here — just checking numeric logic)", flush=True)

    # --- G2 selftest ---
    # Synthetic: entropy separates wrong from right.
    n_right = 150
    n_wrong = 50
    # Right vertices: low entropy (near-deterministic prediction).
    entropy_right = rng.uniform(0.0, 0.3, size=n_right)
    # Wrong vertices: high entropy (uncertain).
    entropy_wrong = rng.uniform(0.8, 1.1, size=n_wrong)   # log(3) ~ 1.099
    all_entropy = np.concatenate([entropy_right, entropy_wrong])
    all_labels  = np.concatenate([np.zeros(n_right), np.ones(n_wrong)])
    auc = _auc_roc(all_entropy, all_labels)
    g2_pass = auc > 0.60
    print(f"  G2 self: entropy->is_wrong AUC={auc:.3f} > 0.60 -> "
          f"{'PASS' if g2_pass else 'FAIL'}", flush=True)
    if not g2_pass:
        passed = False

    # --- G4 selftest ---
    # Synthetic: simulate clamp-improves-rest.
    # Baseline acc on remaining vertices = 0.50.
    # Clamped (propagation works): acc improves to 0.65.
    baseline_acc = 0.50
    clamped_acc  = 0.65
    g4_pass = clamped_acc > baseline_acc * 1.01   # > 1% relative improvement
    print(f"  G4 self: clamped_acc={clamped_acc:.3f} > baseline_acc={baseline_acc:.3f}"
          f" -> {'PASS' if g4_pass else 'FAIL'}", flush=True)
    if not g4_pass:
        passed = False

    # --- AUC edge cases ---
    auc_all_pos = _auc_roc(np.array([0.9, 0.8]), np.array([1, 1]))
    assert auc_all_pos == 0.5, f"all-pos AUC should be 0.5, got {auc_all_pos}"
    auc_perfect  = _auc_roc(np.array([0.9, 0.1]), np.array([1, 0]))
    assert auc_perfect == 1.0, f"perfect AUC should be 1.0, got {auc_perfect}"
    print("  AUC edge-cases OK.", flush=True)

    print(f"--- CPU SELFTEST: {'ALL PASS' if passed else 'SOME FAILED'} ---\n",
          flush=True)
    return passed


# ---------------------------------------------------------------------------
# Model build / load (mirrors eval_coloring_bands.py exactly)
# ---------------------------------------------------------------------------

def cast_layers_fp32(model) -> None:
    from tinygrad import dtypes
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


_FG_PARAM_NAMES = [
    "fg_state_embed", "fg_position_embed", "fg_value_codebook",
    "fg_calib_head_w", "fg_calib_head_b", "fg_breath_embed", "fg_delta_gate",
]


def model_state_dict_fg(model) -> dict:
    sd = {"ln_f.g": model.ln_f_g, "ln_f.b": model.ln_f_b}
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out",
              "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b"):
        sd[f"shared.{a}"] = getattr(sw, a)
    for i, layer in enumerate(model.block.layers):
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            sd[f"phase{i}.{a}"] = getattr(layer, a)
    for nm in _FG_PARAM_NAMES:
        sd[nm] = getattr(model, nm)
    return sd


def load_ckpt(model, path: str) -> None:
    from tinygrad.nn.state import safe_load
    sd = safe_load(path)
    targets = model_state_dict_fg(model)
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
                missing.append(f"{name} (shape mismatch)")
                continue
        if src.dtype != dst.dtype:
            src = src.cast(dst.dtype)
        dst.assign(src).realize()
    if missing:
        print(f"  ckpt missing {len(missing)} keys: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}", flush=True)
    else:
        print(f"  ckpt loaded cleanly ({len(targets)} keys).", flush=True)


# ---------------------------------------------------------------------------
# Forward pass helper that returns numpy arrays for analysis
# ---------------------------------------------------------------------------

def run_forward_np(model, batch, spec, K: int):
    """Run factor_breathing_forward, return (probs_np, calib_np).

    probs_np  : (B, S, N) float32  — softmax of final-breath logits.
    calib_np  : (B,)      float32  — calibration sigmoid at final breath.
    """
    from mycelium.factor_graph_engine import factor_breathing_forward
    logits_history, calib_history = factor_breathing_forward(
        model, batch, spec, K=K)
    final_logits = logits_history[-1]                # (B, S, N)
    calib_final  = calib_history[-1]                 # (B,)

    # Softmax of final logits (float32 from engine already).
    logits_np = final_logits.realize().numpy()       # (B, S, N)
    e = np.exp(logits_np - logits_np.max(axis=-1, keepdims=True))
    probs_np  = e / (e.sum(axis=-1, keepdims=True) + 1e-12)
    calib_np  = calib_final.realize().numpy()        # (B,)
    return probs_np, calib_np


def run_clamped_forward_np(model, batch_orig, spec, K: int,
                            clamp_indices: list[int], gold_np: np.ndarray,
                            n_values: int):
    """Re-run forward with selected per-instance vertex indices clamped to gold.

    clamp_indices : list of vertex indices (0-based within s_max) to clamp,
                    applied IDENTICALLY across all instances in the batch.
    gold_np       : (B, S) int, gold values 1..N.

    Returns (probs_np, calib_np) with the same shape as run_forward_np.
    """
    from tinygrad import Tensor, dtypes
    import numpy as np

    B  = int(batch_orig.input_cells.shape[0])
    S  = int(batch_orig.cell_valid.shape[1])
    N  = n_values

    # Build new input_cells: copy original (all 0), set clamped slots to gold.
    ic_np = batch_orig.input_cells.realize().numpy().copy()   # (B, S)
    cv_np = batch_orig.cell_valid.realize().numpy()           # (B, S)
    for vidx in clamp_indices:
        # Only clamp real (valid) vertices.
        for b in range(B):
            if cv_np[b, vidx] > 0.5:
                ic_np[b, vidx] = int(gold_np[b, vidx])       # 1..N

    # Build new value_domain_mask: all-1 for real vertices EXCEPT clamped ones
    # which become a one-hot (only the gold color legal).
    vdm_np = batch_orig.value_domain_mask.realize().numpy().copy()  # (B, S, N)
    for vidx in clamp_indices:
        for b in range(B):
            if cv_np[b, vidx] > 0.5:
                color_idx = int(gold_np[b, vidx]) - 1    # 0..N-1
                vdm_np[b, vidx, :] = 0.0
                vdm_np[b, vidx, color_idx] = 1.0

    # Wrap in new tensors.
    new_ic  = Tensor(ic_np.astype(np.int32),   dtype=dtypes.int).contiguous().realize()
    new_vdm = Tensor(vdm_np.astype(np.float32), dtype=dtypes.float).contiguous().realize()

    # Build a lightweight proxy batch that inherits everything else.
    class _ClampedBatch:
        def __init__(self, orig, ic, vdm):
            self.input_cells       = ic
            self.cell_valid        = orig.cell_valid
            self.value_domain_mask = vdm
            self.gold              = orig.gold
            self.membership        = orig.membership
            self.latent_type       = orig.latent_type
            self.factor_inlet      = None
            self.deduction_depth   = orig.deduction_depth

    clamped_batch = _ClampedBatch(batch_orig, new_ic, new_vdm)
    return run_forward_np(model, clamped_batch, spec, K)


# ---------------------------------------------------------------------------
# Gate metrics (CPU numpy — call after materializing tensors)
# ---------------------------------------------------------------------------

def compute_g1(probs_np: np.ndarray, gold_np: np.ndarray,
               cell_valid_np: np.ndarray, k: int,
               ) -> dict:
    """G1 — POLICY-RANKS-GOLD at wrong vertices.

    probs_np       : (B, S, k) softmax probabilities.
    gold_np        : (B, S) int, gold values 1..k (0 = pad).
    cell_valid_np  : (B, S) float, 1 = real vertex.

    Returns a dict with the key scalars.
    """
    B, S, _ = probs_np.shape
    gold_probs_at_wrong  = []
    gold_ranks_at_wrong  = []   # rank 0 = top-1
    topk_hit_at_wrong    = []   # 1 if gold is top-1 at wrong vertex

    for b in range(B):
        for s in range(S):
            if cell_valid_np[b, s] < 0.5:
                continue
            g_val = int(gold_np[b, s])
            if g_val < 1 or g_val > k:
                continue
            g_idx = g_val - 1                          # 0..k-1
            pred_idx = int(probs_np[b, s].argmax())
            if pred_idx != g_idx:                       # wrong vertex
                prob_gold = float(probs_np[b, s, g_idx])
                rank_gold = int((probs_np[b, s] > prob_gold).sum())  # 0=best
                gold_probs_at_wrong.append(prob_gold)
                gold_ranks_at_wrong.append(rank_gold)
                topk_hit_at_wrong.append(1 if rank_gold == 0 else 0)

    n_wrong = len(gold_probs_at_wrong)
    if n_wrong == 0:
        return {
            "n_wrong_vertices": 0,
            "mean_gold_prob":   float("nan"),
            "uniform_baseline": 1.0 / k,
            "top1_hit_rate":    float("nan"),
            "rank_dist":        {},
            "pass":             False,
        }

    mean_gold_prob = float(np.mean(gold_probs_at_wrong))
    top1_hit       = float(np.mean(topk_hit_at_wrong))
    # Rank distribution
    rank_dist: dict[int, int] = {}
    for r in gold_ranks_at_wrong:
        rank_dist[r] = rank_dist.get(r, 0) + 1

    return {
        "n_wrong_vertices": n_wrong,
        "mean_gold_prob":   mean_gold_prob,
        "uniform_baseline": 1.0 / k,
        "top1_hit_rate":    top1_hit,
        "rank_dist":        rank_dist,
        "pass":             mean_gold_prob > (1.0 / k),  # strictly above chance
    }


def compute_g2(probs_np: np.ndarray, gold_np: np.ndarray,
               cell_valid_np: np.ndarray, k: int,
               ) -> dict:
    """G2 — ENTROPY-FLAGS-AMBIGUITY.

    Computes AUC of (per-vertex entropy -> is_wrong) over valid vertices.
    """
    B, S, _ = probs_np.shape
    entropies = []
    is_wrong  = []

    log_k = np.log(float(k)) + 1e-12
    for b in range(B):
        for s in range(S):
            if cell_valid_np[b, s] < 0.5:
                continue
            g_val = int(gold_np[b, s])
            if g_val < 1 or g_val > k:
                continue
            g_idx    = g_val - 1
            pred_idx = int(probs_np[b, s].argmax())
            prob_vec = probs_np[b, s].clip(1e-12, 1.0)
            ent      = float(-np.sum(prob_vec * np.log(prob_vec)))   # nats
            ent_norm = ent / log_k                                    # 0..1
            wrong    = 1 if pred_idx != g_idx else 0

            entropies.append(ent_norm)
            is_wrong.append(wrong)

    entropies_np = np.array(entropies, dtype=np.float32)
    is_wrong_np  = np.array(is_wrong,  dtype=np.int32)

    n_valid  = len(entropies_np)
    n_wrong  = int(is_wrong_np.sum())
    n_right  = n_valid - n_wrong

    if n_valid == 0 or n_wrong == 0:
        return {
            "n_valid_vertices": n_valid,
            "n_wrong": n_wrong,
            "n_right": n_right,
            "mean_entropy_wrong": float("nan"),
            "mean_entropy_right": float("nan"),
            "auc": 0.5,
            "pass": False,
        }

    mean_ent_wrong = float(entropies_np[is_wrong_np == 1].mean())
    mean_ent_right = float(entropies_np[is_wrong_np == 0].mean()) if n_right > 0 else float("nan")
    auc = _auc_roc(entropies_np, is_wrong_np)

    return {
        "n_valid_vertices": n_valid,
        "n_wrong": n_wrong,
        "n_right": n_right,
        "mean_entropy_wrong": mean_ent_wrong,
        "mean_entropy_right": mean_ent_right,
        "auc": auc,
        "pass": auc > 0.60,
    }


def compute_g4_clamp(
    model, batches_hard: list, spec, K: int, k: int,
    n_clamp_list: list[int],
) -> dict:
    """G4 — CLAMP-TO-GOLD / OFF-DISTRIBUTION.

    For each n_clamp in n_clamp_list:
      1. Run baseline (unclamped) forward on hard instances.
      2. Clamp n_clamp vertices to gold, re-run.
      3. Measure cell_acc on the REMAINING (unclamped) valid vertices.
      4. Also track calibration correlation with fraction-correct on partials.

    batches_hard : list of GraphColoringBatch objects (hard band instances only).
    Returns a results dict with per-n_clamp breakdown.
    """
    import random

    rng = random.Random(0)
    results_per_nclamp: dict[int, dict] = {}

    for n_clamp in n_clamp_list:
        baseline_accs   = []
        clamped_accs    = []
        calib_vals      = []     # calibration value at final breath (clamped)
        frac_correct    = []     # fraction correct on unclamped valid vertices (clamped run)

        for batch in batches_hard:
            gold_np       = batch.gold.realize().numpy()       # (B, S) int
            cell_valid_np = batch.cell_valid.realize().numpy() # (B, S)

            B = int(gold_np.shape[0])
            S = int(gold_np.shape[1])

            # Baseline forward (unclamped).
            probs_base, _ = run_forward_np(model, batch, spec, K)
            pred_base = probs_base.argmax(axis=-1) + 1        # (B, S) 1..N

            # Choose n_clamp vertex indices uniformly (same set for all instances
            # in the batch, from the union of valid indices).
            # Gather globally valid vertex indices (valid in ALL instances in batch).
            global_valid = np.ones(S, dtype=bool)
            for b in range(B):
                global_valid &= (cell_valid_np[b] > 0.5)
            valid_idx = np.where(global_valid)[0].tolist()
            if len(valid_idx) < n_clamp + 1:
                # Not enough valid vertices to clamp and still have remainder.
                continue

            clamp_idx = rng.sample(valid_idx, n_clamp)
            remainder_idx = [i for i in valid_idx if i not in set(clamp_idx)]

            # Clamped forward.
            probs_clamp, calib_clamp = run_clamped_forward_np(
                model, batch, spec, K, clamp_idx, gold_np, k)
            pred_clamp = probs_clamp.argmax(axis=-1) + 1      # (B, S) 1..N

            # Compute acc on REMAINDER vertices only.
            for b in range(B):
                if len(remainder_idx) == 0:
                    continue
                r_idx = np.array(remainder_idx)

                # Baseline acc on remainder.
                base_eq   = (pred_base[b, r_idx] == gold_np[b, r_idx]).astype(float)
                base_acc  = float(base_eq.mean())

                # Clamped acc on remainder.
                clamp_eq  = (pred_clamp[b, r_idx] == gold_np[b, r_idx]).astype(float)
                clamp_acc = float(clamp_eq.mean())

                baseline_accs.append(base_acc)
                clamped_accs.append(clamp_acc)
                calib_vals.append(float(calib_clamp[b]))
                frac_correct.append(clamp_acc)   # fraction correct on partials

        if not baseline_accs:
            results_per_nclamp[n_clamp] = {
                "n_instances": 0,
                "baseline_acc": float("nan"),
                "clamped_acc": float("nan"),
                "delta": float("nan"),
                "calib_frac_corr": float("nan"),
                "pass": False,
            }
            continue

        mean_base   = float(np.mean(baseline_accs))
        mean_clamp  = float(np.mean(clamped_accs))
        delta       = mean_clamp - mean_base

        # Calibration vs fraction-correct Pearson correlation.
        calib_arr = np.array(calib_vals, dtype=np.float32)
        frac_arr  = np.array(frac_correct, dtype=np.float32)
        if calib_arr.std() > 1e-6 and frac_arr.std() > 1e-6:
            calib_corr = float(np.corrcoef(calib_arr, frac_arr)[0, 1])
        else:
            calib_corr = float("nan")

        results_per_nclamp[n_clamp] = {
            "n_instances":     len(baseline_accs),
            "baseline_acc":    mean_base,
            "clamped_acc":     mean_clamp,
            "delta":           delta,
            "calib_frac_corr": calib_corr,
            "pass":            delta > 0.005,  # >0.5pp improvement = meaningful
        }

    # Overall G4 pass: at least ONE n_clamp level shows improvement.
    any_pass = any(r.get("pass", False) for r in results_per_nclamp.values()
                   if r["n_instances"] > 0)
    return {
        "per_nclamp": results_per_nclamp,
        "pass": any_pass,
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

SEP = "=" * 65


def _print_g1(r: dict) -> None:
    print(f"\n{SEP}", flush=True)
    print("  G1 — POLICY-RANKS-GOLD (at wrong/ambiguous vertices)", flush=True)
    print(SEP, flush=True)
    print(f"  n_wrong_vertices  : {r['n_wrong_vertices']}", flush=True)
    print(f"  mean_gold_prob    : {r['mean_gold_prob']:.4f}  "
          f"(uniform baseline 1/k={r['uniform_baseline']:.4f})", flush=True)
    print(f"  top-1 hit rate    : {r['top1_hit_rate']:.4f}  "
          f"(gold is argmax at wrong vertex)", flush=True)
    rd = r["rank_dist"]
    for rk in sorted(rd.keys()):
        pct = 100.0 * rd[rk] / max(r["n_wrong_vertices"], 1)
        print(f"    rank={rk}: {rd[rk]} ({pct:.1f}%)", flush=True)
    verdict = "PASS" if r["pass"] else "FAIL"
    print(f"  G1 verdict: {verdict}  "
          f"(pass if mean_gold_prob > 1/k={r['uniform_baseline']:.4f})", flush=True)


def _print_g2(r: dict) -> None:
    print(f"\n{SEP}", flush=True)
    print("  G2 — ENTROPY-FLAGS-AMBIGUITY", flush=True)
    print(SEP, flush=True)
    print(f"  n_valid_vertices  : {r['n_valid_vertices']}", flush=True)
    print(f"  n_wrong / n_right : {r['n_wrong']} / {r['n_right']}", flush=True)
    print(f"  mean_entropy(wrong): {r['mean_entropy_wrong']:.4f}  "
          f"(normalised, 0..1)", flush=True)
    print(f"  mean_entropy(right): {r['mean_entropy_right']:.4f}", flush=True)
    print(f"  AUC(entropy->wrong): {r['auc']:.4f}  (threshold 0.60)", flush=True)
    verdict = "PASS" if r["pass"] else "FAIL"
    print(f"  G2 verdict: {verdict}  (pass if AUC > 0.60)", flush=True)


def _print_g4(r: dict) -> None:
    print(f"\n{SEP}", flush=True)
    print("  G4 — CLAMP-TO-GOLD / OFF-DISTRIBUTION", flush=True)
    print(SEP, flush=True)
    for nc, d in r["per_nclamp"].items():
        ni = d.get("n_instances", 0)
        if ni == 0:
            print(f"  n_clamp={nc}: no instances (too few valid vertices)", flush=True)
            continue
        print(f"  n_clamp={nc}:", flush=True)
        print(f"    n_instances     : {ni}", flush=True)
        print(f"    baseline_acc    : {d['baseline_acc']:.4f}", flush=True)
        print(f"    clamped_acc     : {d['clamped_acc']:.4f}", flush=True)
        print(f"    delta           : {d['delta']:+.4f}  "
              f"({'PASS' if d['pass'] else 'FAIL'} >0.005 threshold)", flush=True)
        print(f"    calib_frac_corr : {d['calib_frac_corr']:.4f}  "
              f"(calib vs frac-correct on partial state)", flush=True)
    verdict = "PASS" if r["pass"] else "FAIL"
    print(f"  G4 verdict: {verdict}  "
          f"(pass if ANY n_clamp level shows delta > 0.005)", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import gc
    from tinygrad import Tensor, Device
    from tinygrad.helpers import getenv

    CKPT = getenv(
        "FG_CKPT",
        ".cache/fg_ckpts/fg_coloring_k16/fg_coloring_k16_final.safetensors",
    )
    K           = int(getenv("FG_K_MAX", getenv("K", "16")))
    EVAL_BATCH  = int(getenv("EVAL_BATCH", getenv("BATCH", "8")))
    SEED        = int(getenv("SEED", "42"))
    N_INSTANCES = int(getenv("FG_N_INSTANCES", "8000"))
    S_MAX       = int(getenv("FG_S_MAX", "49"))
    N_VALUES    = int(getenv("FG_N_VALUES", "3"))
    # deduction_depth >= HARD_DEPTH_THRESH are the hard instances.
    HARD_THRESH = int(getenv("HARD_DEPTH_THRESH", "3"))
    # How many vertices to clamp in G4.
    N_CLAMP_LIST = [int(x) for x in getenv("N_CLAMP_LIST", "1,2,4").split(",")]

    print("=== analyze_search_guidance_gate.py — PUCT viability gate ===",
          flush=True)
    print(f"device={Device.DEFAULT}  ckpt={CKPT}", flush=True)
    print(f"K={K}  EVAL_BATCH={EVAL_BATCH}  seed={SEED}  "
          f"n_instances={N_INSTANCES}  s_max={S_MAX}  k={N_VALUES}  "
          f"hard_depth>={HARD_THRESH}  n_clamp_list={N_CLAMP_LIST}", flush=True)
    print(flush=True)

    # ---- build model (mirrors eval_coloring_bands.py) ----------------------
    from mycelium import Config, BreathingTransformer
    from mycelium.loader import _load_state, load_breathing
    from mycelium.factor_graph_engine import (
        FactorGraphSpec,
        attach_factor_graph_params,
        FG_HYP_MASK,
    )
    from mycelium.factor_masks import attach_factor_hyperbolic_params
    from mycelium.graph_coloring_data import GraphColoringLoader

    spec = FactorGraphSpec(
        s_max=S_MAX,
        n_values=N_VALUES,
        n_factor_types=1,
        n_heads=16,
        k_max=K,
        has_factor_inlet=False,
    )

    print("loading Pythia-410M -> BreathingTransformer...", flush=True)
    cfg = Config()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    gc.collect()
    cast_layers_fp32(model)
    attach_factor_graph_params(model, hidden=cfg.hidden, spec=spec)

    if FG_HYP_MASK:
        print("[FG_HYP_MASK=1] building coloring anchor tables...", flush=True)
        _ref_loader = GraphColoringLoader(
            n_instances=N_INSTANCES, s_max=S_MAX, k_colors=N_VALUES,
            batch_size=EVAL_BATCH, seed=SEED,
        )
        _ref_batch = _ref_loader.sample_batch()
        _mem_np = _ref_batch.membership.realize().numpy()
        _lt_np  = _ref_batch.latent_type.realize().numpy()
        attach_factor_hyperbolic_params(
            model, n_heads=spec.n_heads, n_factor_types=spec.n_factor_types,
            s_max=spec.s_max, membership_np=_mem_np, latent_type_np=_lt_np,
        )
        del _ref_loader, _ref_batch, _mem_np, _lt_np
        print("  coloring hyperbolic params attached (frozen).", flush=True)

    Device[Device.DEFAULT].synchronize()

    print(f"loading checkpoint: {CKPT}", flush=True)
    load_ckpt(model, CKPT)

    # ---- reconstruct IDENTICAL test split ----------------------------------
    print(f"\nreconstructing test split "
          f"(n_instances={N_INSTANCES}, seed={SEED})...", flush=True)
    loader = GraphColoringLoader(
        n_instances=N_INSTANCES, s_max=S_MAX, k_colors=N_VALUES,
        batch_size=EVAL_BATCH, seed=SEED,
    )
    n_test = len(loader.test_records)
    print(f"  test set: {n_test} instances", flush=True)

    # ---- collect hard instances (depth >= HARD_THRESH) ---------------------
    Tensor.training = False

    # Accumulators for G1/G2 across all batches.
    all_probs_list: list[np.ndarray] = []
    all_gold_list:  list[np.ndarray] = []
    all_valid_list: list[np.ndarray] = []

    # Hard batches (whole batch object) for G4.
    hard_batches: list = []

    n_hard_instances = 0
    n_total          = 0

    print("\npass 1: collecting G1/G2 data (all test instances)...", flush=True)

    for batch in loader.iter_eval(batch_size=EVAL_BATCH):
        probs_np, _ = run_forward_np(model, batch, spec, K)

        gold_np  = batch.gold.realize().numpy()            # (B, S)
        valid_np = batch.cell_valid.realize().numpy()      # (B, S)

        all_probs_list.append(probs_np)
        all_gold_list.append(gold_np)
        all_valid_list.append(valid_np)

        # Identify hard instances in this batch.
        batch_hard_mask = [d >= HARD_THRESH for d in batch.deduction_depth]
        n_hard_in_batch = sum(batch_hard_mask)
        n_hard_instances += n_hard_in_batch
        n_total          += EVAL_BATCH

        if n_hard_in_batch > 0:
            hard_batches.append(batch)

    print(f"  total instances scanned : {n_total}  "
          f"(n_hard depth>={HARD_THRESH}: {n_hard_instances})", flush=True)

    # ---- G1 ----------------------------------------------------------------
    print("\ncomputing G1 (POLICY-RANKS-GOLD)...", flush=True)
    all_probs = np.concatenate(all_probs_list, axis=0)    # (N_test, S, k)
    all_gold  = np.concatenate(all_gold_list,  axis=0)    # (N_test, S)
    all_valid = np.concatenate(all_valid_list, axis=0)    # (N_test, S)
    g1_result = compute_g1(all_probs, all_gold, all_valid, N_VALUES)
    _print_g1(g1_result)

    # ---- G2 ----------------------------------------------------------------
    print("\ncomputing G2 (ENTROPY-FLAGS-AMBIGUITY)...", flush=True)
    g2_result = compute_g2(all_probs, all_gold, all_valid, N_VALUES)
    _print_g2(g2_result)

    # Free the large arrays before G4 (G4 does per-batch forward passes).
    del all_probs_list, all_gold_list, all_valid_list
    del all_probs, all_gold, all_valid
    gc.collect()

    # ---- G4 ----------------------------------------------------------------
    if n_hard_instances == 0:
        print(f"\nG4 SKIPPED: no hard instances (depth>={HARD_THRESH}) found in "
              f"test set.  Try lowering HARD_DEPTH_THRESH.", flush=True)
        g4_result = {"per_nclamp": {}, "pass": False}
    else:
        print(f"\ncomputing G4 (CLAMP-TO-GOLD) on {len(hard_batches)} hard batches...",
              flush=True)
        print(f"  n_clamp_list = {N_CLAMP_LIST}", flush=True)
        g4_result = compute_g4_clamp(
            model, hard_batches, spec, K, N_VALUES, N_CLAMP_LIST)
    _print_g4(g4_result)

    # ---- VERDICT -----------------------------------------------------------
    g1_pass = g1_result["pass"]
    g2_pass = g2_result["pass"]
    g4_pass = g4_result["pass"]

    print(f"\n{SEP}", flush=True)
    print("  SEARCH GUIDANCE GATE — VERDICT", flush=True)
    print(SEP, flush=True)
    print(f"  G1 (policy > chance)   : {'PASS' if g1_pass else 'FAIL'}", flush=True)
    print(f"  G2 (entropy flags AUC) : {'PASS' if g2_pass else 'FAIL'}", flush=True)
    print(f"  G4 (clamp-gold helps)  : {'PASS' if g4_pass else 'FAIL'}", flush=True)
    print(flush=True)

    if g1_pass and g2_pass and g4_pass:
        print("  *** GATE PASS — build the residual-search tier (PUCT) ***", flush=True)
        print("  Policy has signal above chance (G1), entropy identifies", flush=True)
        print("  ambiguous vertices (G2), and clamping gold propagates to", flush=True)
        print("  improve unclamped remainder (G4). Off-distribution is OK.", flush=True)
    else:
        reasons = []
        if not g1_pass:
            reasons.append("G1 FAIL: policy does NOT rank gold above chance — "
                           "branching on the policy gives no edge over random; "
                           "fix the readout before search.")
        if not g2_pass:
            reasons.append(
                f"G2 FAIL: entropy does NOT flag wrong vertices "
                f"(AUC={g2_result['auc']:.3f} <= 0.60) — "
                "cannot identify the residual ambiguous set; "
                "search cannot focus effort.")
        if not g4_pass:
            reasons.append("G4 FAIL: clamping gold does NOT help (or hurts) the "
                           "unclamped remainder — off-distribution breaks the deducer "
                           "(v106 negative recurs); fix value/calibration before search.")
        print("  *** GATE FAIL — do NOT build PUCT tier yet ***", flush=True)
        for r in reasons:
            print(f"  -> {r}", flush=True)

    print(SEP, flush=True)
    print(flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ok = _ast_parse_ok()
    print(f"[ast.parse] astparse_ok={ok}", flush=True)
    if not ok:
        sys.exit(1)

    # Run CPU selftest unless SELFTEST=0 explicitly.
    from tinygrad.helpers import getenv
    selftest = int(getenv("SELFTEST", "1"))
    if selftest:
        st_ok = _selftest()
        print(f"[selftest] selftest_ok={st_ok}", flush=True)
        if not st_ok:
            sys.exit(1)

    # If no checkpoint or SELFTEST_ONLY=1, stop after selftest.
    selftest_only = int(getenv("SELFTEST_ONLY", "0"))
    ckpt_env = os.environ.get("FG_CKPT", "")
    ckpt_default = ".cache/fg_ckpts/fg_coloring_k16/fg_coloring_k16_final.safetensors"
    ckpt_path = ckpt_env if ckpt_env else ckpt_default
    if selftest_only:
        print("[analyze_search_guidance_gate] SELFTEST_ONLY=1 — skipping GPU run.",
              flush=True)
        sys.exit(0)

    if not os.path.exists(ckpt_path):
        print(f"[analyze_search_guidance_gate] checkpoint not found: {ckpt_path}",
              flush=True)
        print("  Set FG_CKPT=<path> or run with SELFTEST_ONLY=1 for CPU-only mode.",
              flush=True)
        sys.exit(0)

    main()
