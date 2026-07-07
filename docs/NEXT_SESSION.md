# NEXT SESSION — start here (handoff, 2026-07-07 night)

Cold-start entry point. Read this first; it points to everything else.

## Where we are (one paragraph)
**Both domains run measured repair loops.** KenKen composed stack (two-checkpoint,
fully gold-free): **47%** recovery. Algebra composed stack: **24%** — BUT with an
asterisk (below). The day resolved ~10 registered predictions: tier-0 completed at
ZERO params (min-combination AUC 0.812 on 78 silents; trained head unbuilt, kill =
beat 0.812, re-arm condition written); factorization survived size-controlled
(partial corr +0.061); withholding = the domain-general third repair channel (sparse
flip REFUTED — 4th density-regime sighting; §6 now demands the ARITHMETIC: "what
does the k-th withheld factor actually hit at measured density x AUC"); the
blank-pass tax was registered as headroom and collected same-day (42% -> 47%).
Spec §10-§12 build logs are ground truth. Parser convergence run FIRED overnight
(RESUME, teeth corpus — the multiplicative lever: every fac-exact point lifts all
stack numbers).

## START HERE next session (in order)
1. **THE 24% ASTERISK — deployable-flags ablation, FIRST FIRE.** The algebra stack's
   span-level suspect flags located suspects via GOLD factor spans (mildly oracle);
   KenKen's 47% is fully gold-free — the two headlines are not yet the same kind of
   number. Deployable path: spans from the parser's OWN mention/attention predictions
   (machinery exists — the mention head supervises it). The dual-granularity ablation
   (field-only vs both) doubles as the leakage bound. INFORMAL PREDICTION (relay,
   registered): deployable lands within 3-5 points of 24% — the clean field-level
   channel plausibly carries most of the repair signal (structure beating
   localization, again). Do NOT quote 24% anywhere without this resolved.
2. **THE OVERNIGHT RUN NEEDS A REDO WITH HYGIENE — read before trusting it.**
   +32k steps at CONSTANT lr 3e-4 went unstable late (loss 4.17 @16k -> 7.10 @32k;
   the SAVED ckpt is post-spike — the quick trainer has no LR decay and no
   best-ckpt selection). Eval moved MIXED: ANSWER 587 -> 606 (+19), query up
   strongly (0.73 -> 0.86 band 2), but graph-solve 485 -> 453 (-32). Morning fix:
   resume-retrain with LR decay (3e-4 -> 3e-5 cosine or halving) + periodic ckpt
   + pick-best-by-eval, THEN re-run the stacks on the honest best parser (also
   re-prep/retrain the NACK retransmitter against its failure distribution).
3. **Multi-round** against honest numbers (frame registered: declining per-round
   recovery, asymptote ~ decodable-ceiling share; violation = ledger-conditioning
   early = the 46.7% frontier's cheapest probe).

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
