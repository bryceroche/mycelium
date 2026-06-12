"""#237 specialization disambiguators (§1A.E.11, pinned before step-500/1000 reads).

Computes, per dense checkpoint, on the SAME fixed diag batch as the #237 driver:

  1. THINK-attention vs quotient-graph correlation — mean inter-latent
     attention matrix A (over batch, heads, L0-L3 layers, breaths; diagonal
     excluded) vs the static quotient-graph adjacency Q implied by partition
     1a AS BUILT (mod-4 ownership; per-token<->per-token factor edges omitted).
     Spearman rho over off-diagonal entries + latent-relabeling permutation
     null (1000 perms). Pre-committed bands (E.11): ALIGNED = rho >= 0.5 AND
     > 99th pct of null; NON-TOPOLOGICAL = rho < 0.2 OR within null;
     0.2 <= rho < 0.5 = AMBIGUOUS (treated as Branch B).

  2. Top-10 dim-set overlap across latents at post-THINK (same site as the §7
     concentration metric): per latent, top-10 energy dims (mean over batch);
     mean pairwise overlap across the 32 latents per breath; overlap with the
     pooled all-latent top-10; cross-breath dim-identity stability.
     Pre-committed bands (E.11): SINK >= 7/10 pairwise with stable dims
     across breaths; SPECIALIZATION <= 3/10; middle = MIXED.

Runs on CPU by design (DEV=CPU) so it can execute while the GPU trains —
the THINK forward is tiny (32-latent sequences). Loads the run's dense
checkpoints; the topology mask is attached by attach_fg_params_v200
(deterministic buffer) exactly as in training.

Usage:
  DEV=CPU .venv/bin/python scripts/diag_v237_specialization.py            # all 237 ckpts
  DEV=CPU STEPS_LIST=0,200,500 .venv/bin/python scripts/diag_v237_specialization.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from scipy.stats import spearmanr

from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

from mycelium.llama_loader import (
    attach_llama_layers, load_llama_weights,
    LLAMA_3_2_1B_CFG, SMOLLM2_1_7B_CFG,
)
from mycelium.factor_graph_v200 import (
    attach_fg_params_v200, fg_v200_state_dict, fg_breathing_forward_v200,
    fg_v200_empty_taps, V200_MASK_FAMILIES,
    V200_K_MAX, V200_N_MAX, V200_F_MAX, V200_N_VAR_LAT, V200_N_DIGITS,
    V200_STAGE2A_WAIST, V200_WAIST_DIM,
)
from mycelium.factor_graph_data_v107 import FactorGraphLoaderV107
from mycelium.provenance import make_provenance

RUN_TAG   = os.environ.get("RUN_TAG", "237")   # "237" (frozen control) or "237_5"
CKPT_STEM = {"237": "v200_perceiver_237_mask1a", "237_5": "v200_perceiver_237_5_substrate", "238": "v200_perceiver_238_write8"}[RUN_TAG]
CKPT_GLOB = f".cache/v200_perceiver_ckpts/{CKPT_STEM}_step*.safetensors"
OUT_DIR   = ".cache/v200_smoke"
SEED      = 42   # driver SEED; diag loader uses SEED+2 = 44 (same as driver val_loader)

# E.11 pre-committed bands
RHO_ALIGNED       = 0.5
RHO_NONTOPO       = 0.2
NULL_PCT          = 99.0
OVERLAP_SINK      = 7.0
OVERLAP_SPECIAL   = 3.0
N_PERMS           = 1000


class _Obj:
    pass


def build_quotient_adjacency(L: int = 32, n_op: int = 4) -> np.ndarray:
    """Static quotient-graph adjacency Q implied by partition 1a AS BUILT.

    - per-token latent i (0..23) <-> per-op latent 24+(i mod 4)  [mod-4 ownership]
    - global latents 28..31 <-> everyone (incl. each other)
    - per-token <-> per-token: 0 (factor-share edges omitted — the mod-4
      build carries no factor annotation; documented E.11 caveat)
    - per-op <-> per-op: 0
    Diagonal: 0 (self-attention excluded from the comparison).
    """
    Q = np.zeros((L, L), dtype=np.float64)
    for i in range(24):
        j = 24 + (i % n_op)
        Q[i, j] = 1.0
        Q[j, i] = 1.0
    for g in range(28, 32):
        Q[g, :] = 1.0
        Q[:, g] = 1.0
    np.fill_diagonal(Q, 0.0)
    return Q


def mean_think_attention(taps: dict) -> np.ndarray:
    """Mean inter-latent attention (L, L) over batch, heads, layers, breaths."""
    mats = []
    for sa_k in taps["sa_weights"]:
        for w_t in sa_k:
            w = np.asarray(w_t.numpy(), dtype=np.float64)   # (B, nh, L, L)
            mats.append(w.mean(axis=(0, 1)))
    A = np.stack(mats, axis=0).mean(axis=0)
    np.fill_diagonal(A, 0.0)
    return A


def quotient_correlation(A: np.ndarray, Q: np.ndarray, n_perms: int = N_PERMS,
                         seed: int = 0) -> dict:
    """All-32 ρ plus PER-MASK-FAMILY ρ (per-group discipline applied to the
    alignment read — refinement pinned BEFORE any ρ was ever computed).

    The PRIMARY routing statistic is the per_token family ρ: global latents
    SHOULD attend broadly (their Q rows are all-ones → Spearman degenerate,
    which is the formal confirmation ρ is the wrong lens for them — they are
    reported as structurally exempt; their read is entropy). Folding global
    rows into one ρ could drag a real per-token alignment below threshold
    and misroute Branch C as Branch B.

    Null: random relabeling of all 32 latents (full row+col permutation);
    every statistic (all-32 and per-family) is computed identically on the
    real and permuted matrices within one permutation loop.
    """
    L = A.shape[0]
    off = ~np.eye(L, dtype=bool)

    fam_masks = {}
    for name, lo, hi in V200_MASK_FAMILIES:
        m = np.ones((hi - lo, L), dtype=bool)
        for r in range(lo, hi):
            m[r - lo, r] = False
        fam_masks[name] = (slice(lo, hi), m)

    def _stats(M: np.ndarray) -> dict:
        s = {"all": float(spearmanr(M[off], Q[off]).correlation)}
        for name, (rows, m) in fam_masks.items():
            q = Q[rows][m]
            if np.ptp(q) == 0:
                s[name] = None   # degenerate (attends-all family) — exempt
            else:
                s[name] = float(spearmanr(M[rows][m], q).correlation)
        return s

    real = _stats(A)
    rng = np.random.default_rng(seed)
    null: dict = {k: [] for k in real}
    for _ in range(n_perms):
        perm = rng.permutation(L)
        for k, v in _stats(A[perm][:, perm]).items():
            if v is not None:
                null[k].append(v)

    out = {"families": {}, "bands_pre_committed": (
        "PRIMARY=per_token family rho: ALIGNED>=0.5 & >null99; "
        "NON_TOPO<0.2 | within null; else AMBIGUOUS. global exempt "
        "(Q rows constant — attends-all by construction; read via entropy)."
    )}
    for k, rho in real.items():
        if rho is None:
            out["families"][k] = {"rho": None,
                                  "note": "degenerate Q (attends-all family); exempt from rho"}
            continue
        npct = float(np.percentile(null[k], NULL_PCT))
        above = bool(rho > npct)
        if rho >= RHO_ALIGNED and above:
            band = "ALIGNED"
        elif rho < RHO_NONTOPO or not above:
            band = "NON_TOPOLOGICAL"
        else:
            band = "AMBIGUOUS"
        out["families"][k] = {"rho": rho, "null_pct99": npct,
                              "null_mean": float(np.mean(null[k])),
                              "above_null": above, "band": band}
    # Primary routing statistic (E.11 as refined): the per_token family
    pt = out["families"]["per_token"]
    out["spearman_rho"] = pt["rho"]
    out["null_pct99"] = pt["null_pct99"]
    out["above_null"] = pt["above_null"]
    out["band"] = pt["band"]
    out["all32_rho_secondary"] = out["families"]["all"]["rho"]
    return out


def top10_overlap_stats(taps: dict) -> dict:
    """Top-10 dim-set overlap across latents at post-THINK, per breath."""
    K = len(taps["wb_post_think"])
    per_breath_pairwise = []
    per_breath_pooled = []
    topsets_per_breath = []   # K x list of 32 frozensets
    for k in range(K):
        z = np.asarray(taps["wb_post_think"][k].numpy(), dtype=np.float32)  # (B, L, H)
        energy = (z.astype(np.float64) ** 2).mean(axis=0)                    # (L, H)
        L = energy.shape[0]
        tops = [frozenset(np.argpartition(energy[l], -10)[-10:].tolist()) for l in range(L)]
        topsets_per_breath.append(tops)
        overlaps = [len(tops[i] & tops[j]) for i in range(L) for j in range(i + 1, L)]
        per_breath_pairwise.append(float(np.mean(overlaps)))
        pooled_energy = energy.mean(axis=0)
        pooled = frozenset(np.argpartition(pooled_energy, -10)[-10:].tolist())
        per_breath_pooled.append(float(np.mean([len(t & pooled) for t in tops])))
    # cross-breath dim-identity stability: per latent, overlap of its top-10
    # between consecutive breaths, averaged over latents and breath pairs
    stab = []
    for k in range(K - 1):
        L = len(topsets_per_breath[k])
        stab.append(float(np.mean([
            len(topsets_per_breath[k][l] & topsets_per_breath[k + 1][l])
            for l in range(L)
        ])))
    mean_pairwise = float(np.mean(per_breath_pairwise))
    mean_stab = float(np.mean(stab)) if stab else float("nan")
    if mean_pairwise >= OVERLAP_SINK and mean_stab >= OVERLAP_SINK:
        band = "SINK_RECRUITMENT"
    elif mean_pairwise <= OVERLAP_SPECIAL:
        band = "HEALTHY_SPECIALIZATION"
    else:
        band = "MIXED"
    # which families share dims (MIXED companion signal)
    fam_shared = {}
    last_tops = topsets_per_breath[-1]
    for name, lo, hi in V200_MASK_FAMILIES:
        fam_sets = last_tops[lo:hi]
        ov = [len(fam_sets[i] & fam_sets[j])
              for i in range(len(fam_sets)) for j in range(i + 1, len(fam_sets))]
        fam_shared[name] = float(np.mean(ov)) if ov else float("nan")
    # Dim IDENTITIES (added before the final GPU sweep): the modal shared set
    # — for cross-checkpoint comparison and the latent_init/W_expand source
    # attribution (step-500 check: shared dims 8/10 latent_init common mode,
    # 5/10 W_expand cols, 1/10 breath_embed)
    from collections import Counter
    cnt = Counter(d for t in last_tops for d in t)
    modal_shared = sorted(d for d, _ in cnt.most_common(10))
    return {
        "mean_pairwise_overlap_of10": mean_pairwise,
        "per_breath_pairwise": per_breath_pairwise,
        "mean_overlap_with_pooled_top10": float(np.mean(per_breath_pooled)),
        "cross_breath_dim_stability_of10": mean_stab,
        "within_family_overlap_final_breath": fam_shared,
        "modal_shared_top10_dims_final_breath": modal_shared,
        "band": band,
        "bands_pre_committed": "SINK>=7/10 pairwise & >=7/10 stable; SPECIALIZATION<=3/10; else MIXED",
    }


def main() -> None:
    K = V200_K_MAX
    steps_env = os.environ.get("STEPS_LIST", "")
    ckpts = sorted(glob.glob(CKPT_GLOB),
                   key=lambda p: int(re.search(r"step(\d+)\.safetensors", p).group(1)))
    if steps_env:
        wanted = {int(s) for s in steps_env.split(",")}
        ckpts = [c for c in ckpts
                 if int(re.search(r"step(\d+)\.safetensors", c).group(1)) in wanted]
    assert ckpts, f"no checkpoints matched {CKPT_GLOB}"
    print(f"[diag-237] {len(ckpts)} ckpts; device DEV={os.environ.get("DEV", "")}")

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    # Model skeleton (mirrors the #237 driver)
    llama32_path = ".cache/llama-3.2-1b-weights/model.safetensors"
    use_llama32 = os.path.exists(llama32_path)
    if use_llama32:
        sd = safe_load(llama32_path)
        cfg = LLAMA_3_2_1B_CFG
    else:
        sd = load_llama_weights()
        cfg = SMOLLM2_1_7B_CFG
    model = _Obj()
    attach_llama_layers(model, n_layers=4, sd=sd, cfg=cfg)
    del sd
    attach_fg_params_v200(
        model, n_latents=32, n_var_lat=V200_N_VAR_LAT, k_max=K,
        n_digits=V200_N_DIGITS, n_max=V200_N_MAX, f_max=V200_F_MAX,
        stage2a_waist=True, waist_dim=V200_WAIST_DIM,   # #237 trained with the waist ON — explicit, not env-inherited (sweep-crash fix)
    )
    for layer in model.llama_layers:
        for p in layer.parameters():
            if p.dtype != dtypes.float:
                p.assign(p.cast(dtypes.float)).realize()

    # Fixed diag batch — IDENTICAL construction to the driver's val_loader
    # (seed SEED+2) so per-checkpoint numbers are same-batch comparable.
    val_loader = FactorGraphLoaderV107(
        ".cache/factor_graph_test.jsonl", batch_size=8,
        difficulty_filter=None, curriculum=False,
        n_max=V200_N_MAX, f_max=V200_F_MAX, k_max=K, n_heads=16, seed=SEED + 2,
    )
    batch = next(val_loader.iter_eval())
    domain_init, node_kinds = batch["domain_init"], batch["node_kinds"]

    Q = build_quotient_adjacency()
    targets = fg_v200_state_dict(model)
    results = []
    for ck in ckpts:
        step = int(re.search(r"step(\d+)\.safetensors", ck).group(1))
        sd_ck = safe_load(ck)
        loaded = 0
        for name, dst in targets.items():
            if name in sd_ck:
                src = sd_ck[name].to(dst.device).realize()
                if src.dtype != dst.dtype:
                    src = src.cast(dst.dtype)
                dst.assign(src).realize()
                loaded += 1
        del sd_ck
        was_training = Tensor.training
        Tensor.training = False
        taps = fg_v200_empty_taps()
        fg_breathing_forward_v200(
            model, domain_init, node_kinds, K=K,
            n_max=V200_N_MAX, f_max=V200_F_MAX,
            training=False,
            stage2a_waist=True,   # #237 trained config — explicit, never env-inherited
            taps=taps,
        )
        Tensor.training = was_training

        A = mean_think_attention(taps)
        qc = quotient_correlation(A, Q, seed=step)
        ov = top10_overlap_stats(taps)
        entry = {"step": step, "ckpt": ck, "params_loaded": loaded,
                 "quotient_correlation": qc, "top10_overlap": ov,
                 "attention_matrix_mean": A.tolist()}
        results.append(entry)
        fam_str = "  ".join(
            f"{k}={v['rho']:+.3f}" if v.get("rho") is not None else f"{k}=exempt"
            for k, v in qc["families"].items())
        print(f"[step {step:5d}] PRIMARY per_token rho={qc['spearman_rho']:+.4f} "
              f"(null99={qc['null_pct99']:+.4f}) band={qc['band']:15s} | "
              f"overlap={ov['mean_pairwise_overlap_of10']:.2f}/10 "
              f"stab={ov['cross_breath_dim_stability_of10']:.2f}/10 "
              f"band={ov['band']}", flush=True)
        print(f"    per-family rho: {fam_str}", flush=True)

    out_path = os.path.join(OUT_DIR, f"specialization_{RUN_TAG}.json")
    with open(out_path, "w") as f:
        json.dump({
            "run_id": RUN_TAG,
            # Probes assert the config they measured (tripwire-9 recurrence
            # fix, strong form): binding-time bugs become self-identifying.
            "measured_config": {"stage2a_waist": True, "K": K,
                                "n_latents": 32, "topology_mask": "1a"},
            "e11_bands": {
                "rho_aligned": RHO_ALIGNED, "rho_nontopo": RHO_NONTOPO,
                "overlap_sink": OVERLAP_SINK, "overlap_special": OVERLAP_SPECIAL,
            },
            "quotient_adjacency_note": (
                "static partition-1a-as-built Q: mod-4 ownership + global rows; "
                "per-token<->per-token factor edges omitted (E.11 caveat)"
            ),
            "diag_batch": "val_loader seed 44, first iter_eval batch (same as driver)",
            "checkpoints": results,
        }, f, indent=2)
    # arch_version / metric_sha: same derivation as the #237 driver
    try:
        sha8 = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=_PROJECT_ROOT, timeout=5).stdout.strip()[:8]
        if subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                          text=True, cwd=_PROJECT_ROOT, timeout=5).stdout.strip():
            sha8 += "-dirty"
    except Exception:
        sha8 = "unknown"
    try:
        msha = subprocess.run(
            ["git", "hash-object", os.path.join(_PROJECT_ROOT, "mycelium", "factor_graph_v200.py")],
            capture_output=True, text=True, cwd=_PROJECT_ROOT, timeout=5).stdout.strip()
    except Exception:
        msha = "unknown"
    prov = make_provenance(
        metric="specialization_disambiguators_e11", units="spearman rho / overlap of 10",
        shape=[len(results)], ckpt="dense-ckpt-sweep",
        split="smoke-eval", seed=SEED, step=results[-1]["step"],
        env_vars={"STEPS_LIST": steps_env or "all", "DEV": os.environ.get("DEV", ""),
                  "measured_stage2a_waist": "True"},
        output_path=os.path.abspath(out_path), key="checkpoints",
        arch_version=f"v200-{sha8}-K8_L32_prenorm5_seamthree_gate-2_mask1a",
        metric_sha=msha,
    )
    with open(out_path.replace(".json", ".provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
