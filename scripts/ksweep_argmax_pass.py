"""ksweep_argmax_pass.py — one small re-pass banking the per-breath argmax
value history (the engine's own readout at breath k IS the K=k answer).
Same 240 puzzles (seed-0 determinism), K=16. Banks (K, N, S) int8."""
import sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from diag_kenken_granularity_probe import (build_kenken_spec,
    build_kenken_deducer_model, stack_records, _sample_balanced_records,
    run_kenken_per_breath_forward)
spec = build_kenken_spec(16)
model = build_kenken_deducer_model(spec,
    ".cache/fg_ckpts/fg_kenken_k16_reg/fg_kenken_k16_reg_final.safetensors", 0)
recs = _sample_balanced_records(".cache/kenken_test_curriculum.jsonl",
                                ["g10","g20","g30","g40"], 240, [5,6,7], seed=0)
outs = []
for s0 in range(0, len(recs), 8):
    kb = stack_records(recs[s0:s0+8], 41)
    _, logits_hist = run_kenken_per_breath_forward(model, kb, spec, 16)
    outs.append(np.stack([np.argmax(l.numpy(), axis=-1) for l in logits_hist]))
    print(f"[argmax] {s0+outs[-1].shape[1]}/240", flush=True)
np.savez_compressed(".cache/ksweep_argmax.npz", pred=np.concatenate(outs, axis=1).astype(np.int8))
print("[argmax] banked -> .cache/ksweep_argmax.npz")
