# NEXT SESSION — start here (handoff, 2026-07-07 night)

Cold-start entry point. Read this first; it points to everything else.

## Where we are (one paragraph)
**The Alternator loop is closed gold-free** (Brick-C: retention 0.52, 8/57 recovered)
and **the math expansion is underway with real failures to study**. The algebra
pipeline (arith3 registry + band-labeled generator + the §11 two-bank head) answers
end-to-end through GENUINE search; on the teeth corpus (obliques, letter shuffle,
distractors, irrelevant subsystems): ANSWER 121/300, **the SILENT class is born (14;
KenKen had zero in seven points), detectable 0.92**, DETECT_multi live (43).
Prediction #2-algebra resolved BOTH ways: inversion confirmed; the chain>>coupled
ordering refuted by MULTI-ERROR DENSITY (2nd sighting of "single-error-regime
predictions break at multi-error density" — watch for the 3rd). Training is
UNCONVERGED (floors). Spec §10-§12 of docs/phase1_skeleton_spec.md hold the whole
chapter; the build log there is the ground truth.

## START HERE next session (in order)
1. **Tier-0 incumbent measurements (spec §12 — ZERO GPU):** per-field AUC + post-
   temperature ECE of slot_confidence on the banked artifacts; THE question: does
   the free confidence signal spot the 14 algebra silents (the high-confidence-error
   subset is where the entropy null is blind)? Then the withholding-cost curves
   (KenKen first — conditional prediction needs the AUC first).
2. **More training/data on the teeth corpus** (loss 5.3 and falling at 16k — cheap
   headroom before any architecture talk).
3. **Size-controlled factorization read** (fac-exact vs band at FIXED n_vars) before
   touching the v0 "independent axes" claim.
4. Then: tier-3 transplant onto the silents; multi-round retransmission; ledger
   re-parse (frontier rank #1).

## Assets in hand (don't rebuild)
- `scripts/kenken_nl_gen.py` — NL generator; span-SET gold; round-trip = generation
  gate (labeling bugs die at generation). 4,140/4,140 samples passed to date.
- `scripts/phase1_delta_head.py` — precompute / train (RESUME=1) / eval (per-width) /
  `--errors` taxonomy / `--blame` sweep / `--capture` waist silhouettes. Ckpt:
  `.cache/phase1_delta_head.safetensors`. Corpora: `.cache/kenken_nl_{train,test}.jsonl`.
- `scripts/phase1_residency_smoke.py` — 2.9 GB co-resident, ~21 GB headroom, trunk JIT
  replay 0.34s/batch, no AM hazards (§5 budget line).
- `scripts/capture_silhouette_trajectories.py` — capture-once schema; `--beliefs`
  recompute (validated 1.0000 argmax match from fp16 residuals).
- Deducer: `fg_kenken_k16_reg` (cell 0.80/puzzle 0.35) — the healthy KenKen ckpt;
  `fg_kk_2k*` are hidden=2048 (incompatible); loader hard-errors on mismatch in eval.

## Do NOT redo (measured this arc — don't re-litigate)
- Silhouette-level linearity (REFUTED: ratio 0.72 — compose problems, not silhouettes).
- Residual-space temporal banding (none; beliefs are the envelope, residual the carrier).
- Delete-one blame at multi-error density (recall 0.034; single-error-regime tool).
- Canonical-order slots (circular grid-sort; TEXT-ORDER slots + decoder canonicalization).
- One-hop sentence counting (give the sentence-index embedding; don't ask attention to count).

## The honest open risks (keep in the crosshairs)
- The 100%-detectable EXPIRATION CONDITION (above) — re-measure, don't assume.
- Detection ≠ correction: the retransmission loop is unproven (Brick-C's whole job).
- 1B-class comprehension under paraphrase diversity — template knobs are the instrument.
- Job B (the engine band on real math) remains existential and dated, not dropped.

— end handoff. The parser chapter is a clean two-day arc; everything committed + pushed.
