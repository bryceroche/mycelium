"""crossover_probe_capture.py — the altitude-crossover probe's CAPTURE pass
(2026-07-25). One GPU pass over the June engine (fg_kenken_k16_reg, its own
domain, K=16) banking per-breath readout reps + all three altitude families'
labels; every probe fit/read after this is a zero-GPU replay (the graph-bank
pattern applied to latents).

Signatures A/B/C + the fourth bin, t95 operationalization, and riders are
pre-registered in the ledger (2026-07-25) — this pass READS NOTHING; it only
banks. Banks -> .cache/crossover_capture_k16.npz:
  reps        (K, N_inst, S, H) fp16   per-breath readout-LN reps
  gold        (N_inst, S)              SOLUTION family target (cell values)
  input_cells (N_inst, S)              SURFACE family target (clue/given token)
  cage_op     (N_inst, S)              SCHEMA family target (per-cell cage op)
  cage_size   (N_inst, S)              SCHEMA family target (per-cell cage size)
  is_given    (N_inst, S)              SCHEMA/SURFACE auxiliary
  cell_valid  (N_inst, S)              mask
  band, n, deduction_depth (N_inst)    strata for per-stratum reads
  settle      (N_inst)                 settle breath from logits (trajectory aux)
"""
import sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from diag_kenken_granularity_probe import (
    build_kenken_spec, build_kenken_deducer_model,
    run_kenken_per_breath_forward, stack_records,
    _sample_balanced_records, extract_granularity_features,
    _settle_breath_from_logits,
)

K = int(os.environ.get("K", "16"))
TOTAL = int(os.environ.get("TOTAL", "240"))
BATCH = int(os.environ.get("EVAL_BATCH", "8"))
CKPT = os.environ.get("FG_CKPT",
    ".cache/fg_ckpts/fg_kenken_k16_reg/fg_kenken_k16_reg_final.safetensors")
CURR = ".cache/kenken_test_curriculum.jsonl"
N_CAGES_MAX = 41

spec = build_kenken_spec(K)
model = build_kenken_deducer_model(spec, CKPT, seed=0)
recs = _sample_balanced_records(CURR, ["g10", "g20", "g30", "g40"], TOTAL,
                                [5, 6, 7], seed=0)
print(f"[capture] {len(recs)} puzzles, K={K}, batch={BATCH}", flush=True)

reps_b, golds, inputs, ops, sizes, given, valid, settle_b = [], [], [], [], [], [], [], []
band, nn, depth = [], [], []
import time
t0 = time.time()
for s0 in range(0, len(recs), BATCH):
    kb = stack_records(recs[s0:s0 + BATCH], N_CAGES_MAX)
    reps, logits_hist = run_kenken_per_breath_forward(model, kb, spec, K)
    feats = extract_granularity_features(kb, spec)
    settle_b.append(_settle_breath_from_logits(logits_hist))
    reps_b.append(reps.astype(np.float16))
    golds.append(kb.gold.numpy())
    inputs.append(kb.input_cells.numpy())
    cid = kb.cell_cage_id.numpy()                        # (b, S)
    cop = kb.cage_op.numpy(); csz = kb.cage_size.numpy() # (b, n_cages)
    ops.append(np.take_along_axis(cop, cid, axis=1))
    sizes.append(np.take_along_axis(csz, cid, axis=1))
    given.append(feats["is_given"]); valid.append(feats["cell_valid"])
    band.append(feats["band"]); nn.append(feats["n"])
    depth.append(feats["deduction_depth"])
    print(f"[capture] {s0 + reps.shape[1]}/{len(recs)} ({time.time()-t0:.0f}s)", flush=True)

np.savez_compressed(".cache/crossover_capture_k16.npz",
    reps=np.concatenate(reps_b, axis=1),
    gold=np.concatenate(golds), input_cells=np.concatenate(inputs),
    cage_op=np.concatenate(ops), cage_size=np.concatenate(sizes),
    is_given=np.concatenate(given), cell_valid=np.concatenate(valid),
    band=np.concatenate([np.atleast_1d(b) for b in band]),
    n=np.concatenate([np.atleast_1d(x) for x in nn]),
    deduction_depth=np.concatenate([np.atleast_1d(d) for d in depth]),
    settle=np.concatenate(settle_b))
print("[capture] banked -> .cache/crossover_capture_k16.npz", flush=True)
