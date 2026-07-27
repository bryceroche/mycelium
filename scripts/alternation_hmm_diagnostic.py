"""alternation_hmm_diagnostic.py — the #78/#79 polarization/telegraph read
(2026-07-27, ordered pre-Fire-2). Substrate: the r-loop's STARTING anatomy
(pretrained layers on fg tokens, pad32). Hypotheses pinned blind at
registration: STATIC polarization vs TELEGRAPH regime-switching vs NULL.
Both meters banked (entropy + JSD); the iaf_v3 artifact's own caveat
governs comparisons (their instrument WAS entropy -> entropy-to-entropy
is the lawful lineage comparison; K=2 modal, dwell structure = prior)."""
import sys, os, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("V200_TASK", "1")
import numpy as np
from tinygrad import Tensor, dtypes
from mycelium.llama_loader import load_llama_weights, SMOLLM2_1_7B_CFG, attach_llama_layers
from mycelium.factor_graph_v200 import (attach_fg_params_v200, fg_breathing_forward_v200r,
    fg_v200_empty_taps, _embed_fg_tokens_v200, V200_N_LATENTS, V200_N_VAR_LAT,
    V200_N_DIGITS, V200_N_MAX, V200_F_MAX, V200_WAIST_DIM)
from mycelium.factor_graph_data_v107 import FactorGraphLoaderV107

K = 8
class M: pass
model = M()
attach_llama_layers(model, n_layers=4, sd=load_llama_weights(), cfg=SMOLLM2_1_7B_CFG, layer_offset=0)
attach_fg_params_v200(model, n_latents=V200_N_LATENTS, n_var_lat=V200_N_VAR_LAT, k_max=K,
    n_digits=V200_N_DIGITS, n_max=V200_N_MAX, f_max=V200_F_MAX, stage2a_waist=True, waist_dim=V200_WAIST_DIM)
loader = FactorGraphLoaderV107(".cache/factor_graph_test.jsonl", batch_size=8,
    n_max=V200_N_MAX, f_max=V200_F_MAX, k_max=K, n_heads=16, seed=0)

ent_seq, jsd_seq = [], []   # k-major sequences over (breath, layer)
n_batches = 3
prev_flat = None
E = np.zeros((n_batches, K, 4)); J = np.full((n_batches, K, 4), np.nan)
for bi in range(n_batches):
    b = loader.sample_batch(step=bi)
    fg = _embed_fg_tokens_v200(model, b["domain_init"], b["node_kinds"], V200_N_MAX, V200_F_MAX).cast(dtypes.float).realize()
    d = fg.numpy(); B, T0, H = d.shape
    dp = np.zeros((B, 32, H), np.float32); dp[:, :T0] = d
    m = b["staging_mask"].numpy()
    mp = np.full((B, m.shape[1], 32, 32), -1e4, np.float32)
    mp[:, :, :T0, :T0] = m
    mp[:, :, np.arange(T0, 32), np.arange(T0, 32)] = 0.0
    taps = fg_v200_empty_taps()
    fg_breathing_forward_v200r(model, Tensor(dp, dtype=dtypes.float), b["node_kinds"], K=K,
        n_max=V200_N_MAX, f_max=V200_F_MAX, n_var_lat=V200_N_VAR_LAT, n_digits=V200_N_DIGITS,
        training=False, breath_masks=Tensor(mp, dtype=dtypes.float), taps=taps)
    prev = None
    for k in range(K):
        for li in range(4):
            w = taps["r_sa_weights"][k][li].numpy()[:, :, :T0, :]   # (B, nh, T0, 32) real queries
            p = w / (w.sum(-1, keepdims=True) + 1e-12)
            ent = -(p * np.log(p + 1e-12)).sum(-1)                  # (B, nh, T0)
            E[bi, k, li] = float(ent.mean())
            flat = p.mean(axis=(0, 1))                              # (T0, 32) mean attention profile
            if prev is not None:
                mmid = 0.5 * (flat + prev)
                jsd = 0.5 * ((flat * np.log((flat + 1e-12) / (mmid + 1e-12))).sum(-1).mean()
                           + (prev * np.log((prev + 1e-12) / (mmid + 1e-12))).sum(-1).mean())
                J[bi, k, li] = float(jsd)
            prev = flat
Em = E.mean(0); Jm = np.nanmean(J, axis=0)
print("[entropy grid (breath x layer)]")
for k in range(K):
    print(f"  k={k}: " + " ".join(f"{Em[k,li]:.3f}" for li in range(4)))
print("[JSD grid (vs previous step in sequence)]")
for k in range(K):
    print(f"  k={k}: " + " ".join(f"{Jm[k,li]:.3f}" if np.isfinite(Jm[k,li]) else "  -- " for li in range(4)))
seq = Em.reshape(-1)                     # 32-point k-major sequence
lag1 = float(np.corrcoef(seq[:-1], seq[1:])[0, 1])
layer_means = Em.mean(0); layer_stds = E.reshape(-1, K, 4).std(axis=(0, 1))
sep = float((layer_means.max() - layer_means.min()) / (layer_stds.mean() + 1e-8))
med = np.median(seq); states = (seq > med).astype(int)
runs = np.diff(np.where(np.concatenate(([1], np.diff(states) != 0, [1])))[0])
print(f"\n[reads] lag-1 autocorr {lag1:+.3f} | layer-separation {sep:.2f} sd | dwell runs (median-split): {list(runs)}")
verdict = ("TELEGRAPH-LIKE (alternating)" if lag1 < -0.3 else
           ("STATIC POLARIZATION (per-layer separation, no alternation)" if sep > 2.0 and lag1 > -0.3 else
            "NULL / UNCLASSIFIED"))
print(f"[verdict vs blind bins] {verdict}  (lineage prior: K=2 modal, entropy meter — entropy-to-entropy comparison lawful)")
json.dump({"entropy_grid": Em.tolist(), "jsd_grid": np.where(np.isfinite(Jm), Jm, -1).tolist(),
           "lag1": lag1, "layer_sep_sd": sep, "dwell_runs": [int(x) for x in runs],
           "verdict": verdict, "SCOPE": "r-loop STARTING anatomy (pre-Fire-2, untrained gates)"},
          open(".cache/alternation_hmm_diag.json", "w"), indent=1)
print("[diag] banked -> .cache/alternation_hmm_diag.json")
