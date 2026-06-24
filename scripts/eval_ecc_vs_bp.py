#!/usr/bin/env python3
"""eval_ecc_vs_bp.py — head-to-head BER/FER: the deducer vs classical BP, SNR-stratified.

The §8.1 ECC eval. Loads a TRAINED factor-graph deducer checkpoint (FG_TASK=ecc) and
decodes the SAME SNR-stratified held-out BCH(31,16) transmissions with BOTH:
  * the DEDUCER at K breaths (the learned BP decoder; argmax+1 == gold on bit-cells),
  * the CLASSICAL baseline: VERIFIED sum-product + normalized min-sum from
    scripts/frontier_ecc_bp_gate.py at a FIXED 16 iterations on the SAME instances,
    so the deducer BER/FER vs min-sum-at-K=16 is APPLES-TO-APPLES.

Reuses (single source of truth):
  - mycelium.ecc_data.ECCLoader  : the SAME held-out eval set (fixed seed) the trainer
    uses, so the deducer is evaluated on exactly the instances it never trained on.
  - frontier_ecc_bp_gate.bp_decode_batch / ber_fer : the VERIFIED classical decoders +
    metric, byte-identical to the kill-gate.

This script RUNS THE ENGINE (loads Pythia + the ckpt), so it is GPU-side — the human
fires it after training. It is NOT part of the CPU static checks.

Run (after training):
  RESUME_FROM=.cache/fg_ckpts/<run>/fg_ecc_final.safetensors \
    K=16 .venv/bin/python scripts/eval_ecc_vs_bp.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv

from mycelium import Config, BreathingTransformer
from mycelium.loader import _load_state, load_breathing
from mycelium.factor_graph_engine import (
    FactorGraphSpec, attach_factor_graph_params, factor_breathing_forward,
    attach_factor_lora_params,
)
from mycelium.ecc_data import ECCLoader, values_to_bits, N_BITS

# Reuse the trainer's checkpoint loader + fp32 cast (no duplication).
from scripts.factor_graph_train import load_ckpt, cast_layers_fp32
# Reuse the VERIFIED classical decoders + metric from the kill-gate.
from frontier_ecc_bp_gate import bp_decode_batch, ber_fer


def main():
    K = int(getenv("FG_K_MAX", getenv("K", "16")))
    BP_ITERS = int(getenv("BP_ITERS", "16"))          # classical BP at the SAME budget
    MS_NORM = float(getenv("MS_NORM", "0.8"))         # normalized min-sum factor
    EVAL_BATCH = int(getenv("EVAL_BATCH", "8"))
    SEED = int(getenv("SEED", "42"))
    H_KIND = getenv("ECC_H_KIND", "min").strip().lower()
    eval_snrs = tuple(float(s) for s in
                      getenv("ECC_EVAL_SNRS", "3,4,5,6,7").split(",") if s.strip())
    n_eval_per_snr = int(getenv("ECC_EVAL_PER_SNR", "200"))
    RESUME_FROM = getenv("RESUME_FROM", "")
    PYTHIA_INIT = int(getenv("PYTHIA_INIT", "1")) > 0
    assert RESUME_FROM, "set RESUME_FROM=<ckpt> (the trained FG_TASK=ecc deducer)"

    print(f"=== ECC head-to-head: deducer(K={K}) vs sum-product/min-sum(@{BP_ITERS}) ===")
    print(f"ckpt={RESUME_FROM}  H_kind={H_KIND}  SNRs={list(eval_snrs)}  "
          f"n/SNR={n_eval_per_snr}")

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    # ---- backbone + ECC spec + fg params + ckpt.
    cfg = Config()
    if PYTHIA_INIT:
        sd = _load_state(); model = load_breathing(cfg, sd=sd); del sd
    else:
        model = BreathingTransformer(cfg)
    cast_layers_fp32(model)
    # ECC FIXES: the eval spec MUST match the trained ckpt's toggles (the engine reads
    # spec.reinject_input / spec.lora_rank). Read the SAME env flags the trainer used so
    # Arm A (reinject only) and Arm B (reinject + lora) decode with the right forward.
    FG_ECC_REINJECT = int(getenv("FG_ECC_REINJECT", "0")) > 0
    FG_LORA_RANK = int(getenv("FG_LORA_RANK", "0"))
    spec = FactorGraphSpec(s_max=49, n_values=2, n_factor_types=1,
                           n_heads=cfg.n_heads, k_max=K, has_factor_inlet=False,
                           continuous_input=True,
                           reinject_input=bool(FG_ECC_REINJECT),
                           lora_rank=int(FG_LORA_RANK))
    attach_factor_graph_params(model, hidden=cfg.hidden, spec=spec)
    if FG_LORA_RANK > 0:
        attach_factor_lora_params(model, hidden=cfg.hidden, spec=spec, rank=FG_LORA_RANK)
    load_ckpt(model, RESUME_FROM)
    if FG_ECC_REINJECT or FG_LORA_RANK > 0:
        print(f"  [ECC fixes] reinject_input={spec.reinject_input} "
              f"lora_rank={spec.lora_rank}")

    # ---- the SAME held-out eval set the trainer uses (fixed seed -> identical).
    loader = ECCLoader(H_kind=H_KIND, batch_size=EVAL_BATCH, seed=SEED,
                       eval_snrs=eval_snrs, n_eval_per_snr=n_eval_per_snr)
    H = loader.H

    Tensor.training = False

    # Accumulators per SNR: bits-decoded, bit-errors, frames, frame-errors for each
    # of the three decoders (deducer / sum-product / min-sum).
    per_snr = {s: {d: dict(bit_err=0, n_bits=0, frame_err=0, n_frames=0)
                   for d in ("deducer", "sum", "ms")} for s in eval_snrs}

    for eb in loader.iter_eval(batch_size=EVAL_BATCH):
        snrs = eb.snr_db                                # per-instance SNR (python list)
        gold_np = eb.gold.numpy().astype(np.int32)      # (B, 49)
        cv_np = eb.cell_valid.numpy()                   # (B, 49)
        gold_bits = values_to_bits(gold_np[:, :N_BITS])  # (B, 31) the transmitted bits

        # The channel LLR (the deducer's continuous input) IS the classical decoder's
        # input — same numbers, so the comparison is exact.
        llr_np = eb.cont_input.numpy()[:, :N_BITS]      # (B, 31)

        # ---- DEDUCER decode.
        logits_history, _ = factor_breathing_forward(model, eb, spec, K=K)
        pred_val = (logits_history[-1].argmax(axis=-1) + 1).numpy().astype(np.int32)
        pred_bits = values_to_bits(pred_val[:, :N_BITS])  # (B, 31)

        # ---- CLASSICAL decode on the SAME LLRs (sum-product + normalized min-sum).
        sp_hard, _, _, _ = bp_decode_batch(H, llr_np, BP_ITERS, "sum")
        ms_hard, _, _, _ = bp_decode_batch(H, llr_np, BP_ITERS, "ms", ms_norm=MS_NORM)

        B = gold_bits.shape[0]
        for b in range(B):
            s = float(snrs[b])
            if s not in per_snr:                       # padding-repeat from iter_eval
                continue
            for dname, dec in (("deducer", pred_bits[b]),
                               ("sum", sp_hard[b]), ("ms", ms_hard[b])):
                err = (dec != gold_bits[b])
                acc = per_snr[s][dname]
                acc["bit_err"] += int(err.sum())
                acc["n_bits"] += N_BITS
                acc["frame_err"] += int(err.any())
                acc["n_frames"] += 1

    # ---- report.
    print(f"\n{'SNR':>5} | {'deducer BER':>12} {'FER':>8} | "
          f"{'sum-prod BER':>12} {'FER':>8} | {'min-sum BER':>12} {'FER':>8}")
    print("-" * 80)
    for s in eval_snrs:
        row = [f"{s:>5.1f} |"]
        for d in ("deducer", "sum", "ms"):
            a = per_snr[s][d]
            ber = a["bit_err"] / max(a["n_bits"], 1)
            fer = a["frame_err"] / max(a["n_frames"], 1)
            row.append(f"{ber:>12.4e} {fer:>8.4f} |")
        print(" ".join(row))
    print("\n(deducer-at-K vs min-sum-at-16 on the SAME held-out instances; lower=better)")


if __name__ == "__main__":
    main()
