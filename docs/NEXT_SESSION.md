# NEXT SESSION — start here (handoff, 2026-07-06)

Cold-start entry point. Read this first; it points to everything else.

## Where we are (one paragraph)
The **Phase-1 skeleton is BUILT and measured** (2026-07-05/06, spec + build log:
`docs/phase1_skeleton_spec.md`). Frozen Llama-3.2-1B L0–L3 + a 3.2M-param slot delta
head + parse-side 512→128 Matryoshka waist parses KenKen-in-words at **factor-exact
0.748** (op 0.944 / target 0.865 / member F1 0.891; 3,060-sample corpus). The
**error taxonomy: 60/60 parse errors symbolically detectable, ZERO silent** (44 UNSAT
+ 16 malformed) — the NACK-recoverable ceiling is 100% at an operating point where
one-shot solve is 0%. The **blame sweep honest negative**: delete-one-factor re-solve
= precision 1.000 / recall 0.034 (single-error-regime tool; at ~5 wrong factors per
parse symbolic localization alone fails) → the confidence-ordered **add-back sweep**
and the deducer's **soft-solve suspicion field** are now REQUIRED by measurement, not
taste. Matryoshka: width 128 ≈ 512 on every head — the parse signal is intrinsically
low-dimensional. Earlier in the arc (07-04/05): the waist-vs-tap split landed in docs;
the deduce-side silhouette look refuted silhouette-linearity (compose PROBLEMS) and
found temporal structure lives in BELIEF space (late-JSD → wrong-cell AUC 0.687).

## START HERE next session (in order)
1. **Memmap dataloader + the 40k membership push.** Trunk states for the curriculum
   corpus won't fit RAM as npz (~84 GB fp16) — store fp16 raw `.npy` + `np.memmap`
   batches (fp16 round-trip proven byte-safe for argmax downstream). Frozen trunk =
   pay precompute once; do NOT switch to live forwards (couples head training to
   trunk throughput). Target: membership exactness (present in 100% of failures).
2. **Re-run the taxonomy at every accuracy checkpoint** (`--errors`): track
   detectable-fraction AS A FUNCTION OF factor-exactness. The 100% has an expiration
   condition — as membership improves, the error mix rotates toward target/op (the
   coherent-misreading class) and a SILENT class can be BORN. Watch for it.
3. **Brick-A** (the zero-LoRA null): notebook/NACK-conditioned re-parse through the
   SAME frozen trunk. Then **Brick-C-v0** with the ADD-BACK sweep as localization
   (skeleton = rows/cols/givens, SAT by construction; add cages confidence-first;
   blame insertions that break SAT). Detection ≠ correction — Brick-C's bar is that
   NACK-conditioned re-parse FIXES flagged regions vs the no-NACK control.
4. **The Job-B milestone gate is due at Brick-A completion** (deferred WITH A DATE):
   decisions-per-problem over MATH-500-style problems via the search tier — map the
   calculator band vs the engine band. Local only, no API calls.

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
