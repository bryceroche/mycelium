# Empirical Status: v45-v95 GSM8K push (Archived)

These results were the pre-v98 empirical state of the project.
They are preserved as historical context. The current empirical
status lives in `README.md` §3 (and `paper/outline.md`).

Archived 2026-06-01 after the v98 Sudoku pivot.

This document corresponds to README §14 in its pre-pivot form (dated
2026-05-27). It describes the L4_MIXED / WaistController / DAG paradigm
arc that culminated in the GSM8K plateau, immediately before the v98
Sudoku breakthrough validated the factor-graph framing.

---

## Empirical Status (May 27, 2026)

The architecture above (see `vision_v1_to_v95.md`) is the design. The
empirical evolution has surfaced what's load-bearing and what's scaffolding.
This section is the honest current-state record at the moment v98 began.

**Two paradigms, two champions, plus a third paradigm bridging to GSM8K.**

**Misaligned-decode paradigm champion: v45 reg take 3 step 1000** = **96 / 94 / 93** on L4_MIXED eval (A=1 / A=4 / A=8). File: `.cache/l4_mixed_ckpts/v45_reg_take3_step1000.safetensors`. Warm-started from v24c step 500 (96/94/91) with the regularization stack below.

**Rep-space-thinking paradigm champions (May 19):** v55 step 500 = 89% aligned on L4 (K=2 breaths); v56 step 500 = 89% segmented on L4.5 (K=3); v59 step 1500 = 45% segmented on L4.7 (K=4) with PERFECT step decomposition (per-step accuracy ~78% — breaks the 4-layer 65% arithmetic ceiling). The K-axis is doing real work: at K=3 breaths SPECIALIZE — breath-1 decodes division, breath-2 decodes subtraction on the same problem. The segmented–aligned gap of +14 pt at K=3 is the proof.

**DAG paradigm (v77+, bridge to GSM8K, in progress):** instead of emitting a numeric answer, the final breath emits a SymPy-executable DAG (`x0 = 50 / 60 ; x1 = x0 * 12 ; answer = x1`). SymPy executes the arithmetic. The model only needs to learn STRUCTURAL correctness — operand binding, op selection, dependency graph — not multi-digit arithmetic. K=7 breaths × 7 Haiku-distilled supervision layers (L0: paraphrase → L6: pure DAG). This separates the orchestration problem (which the model handles well — v59 perfect decomposition) from the arithmetic-execution ceiling.

**Waist load-bearing — VALIDATED (May 25):** the WaistController paradigm raises the question: is the 512d waist actually carrying the reasoning, or is the decoder just bypassing it via prompt cross-attn? Direct ablation answers this: with the waist zeroed, CE rises +3.77 on average across 7 breaths (+5.6/+5.9 at L5/L6). With breath-0's waist substituted for breath-k's, CE rises +4.59 — so waists are breath-SPECIFIC, not interchangeable. The architecture is doing what it was designed to do; the WaistController capacity (4 cross-attn layers) is right-sized.

**Regularization stack (validated 2026-05-18):**
- `STOCH_DEPTH_P=0.10` — per-breath Bernoulli drop with ResNet-style 1/(1-p) scaling. Mask-gen guarantees ≥1 active-breath kept per step (no all-dropped catastrophe); skipped entirely at n_loops=1 where SD has no meaning.
- `LABEL_SMOOTHING=0.1` — applied to main answer-CE only. Training-only (eval CE gated on `Tensor.training` so reported val loss stays comparable across runs).
- `WEIGHT_DECAY=0.05` — bumped from the 0.01 default. AdamW typical range for this scale.

**STAGE2_NOTEBOOK bug (caught 2026-05-18):** The v40-era inference code added per-token notebook reads/writes inside `cached_generate_batch`'s Stage 2 decode JIT, intending to "mirror training." But training has no autoregressive decode — adding ~240 OOD notebook updates per problem produced low teacher-forced val loss + 0% generation acc + garbage output ("M 1 1lezIntrIntr..."). Fixed by gating the Stage 2 notebook ops behind `STAGE2_NOTEBOOK` env var (default 0 → restores v24c-compatible behavior). v45 takes 1 and 2 collapsed because of this; take 3 succeeded once the gate was in.

**What ablations established:**
- π-cycled RoPE (per-head phase offset) is the clearly load-bearing closed-loop component: −73 points if ablated, vs −0 to −6 for the others at 150 steps.
- Sine-baseline temperature (2.0 → 0.7 cosine half-period) is load-bearing for warm-start stability.
- Regularization stack (v45, above) — load-bearing for breaking past v24c's overfitting wall and compressing the depth-gradient.
- Integration, notebook, controller decisions (temperature/gate/step_mult), step-size adaptivity — measured as decorative on converged ARITH_HARD. The controller specifically learned `f(breath_idx)`, not `f(rep)`: an open-loop schedule, problem-blind.

**Verification probe — definitive negative:** MLP probes on trained reps (up to 17M params, 1000 steps) cannot distinguish correct from wrong answers (AUC 0.29, anti-correlated). Verification information is not present in the reps as trained. The "7/7 closed feedback loop" is correctly wired but most of it doesn't yet have a job that benefits the loss.

**v80 ladder iterations + DISCOVERY (May 25-27):** 10 sub-versions of v80 chased per-breath CE smoothness via Haiku template tuning ($200+ spent). CV dropped from v1's 1.34 to v3's 0.47 best. But two diagnostics in the final week reframed the project:

1. **Eval bug — missing notebook_pool_mask** silently invalidated 9 iterations of accuracy measurements. After fix, v80_prod_step400 went from 0% → 28.3% DAG parse rate, 1.7% accuracy. The CE chase had been on a model that actually worked under teacher forcing; we just couldn't measure it.
2. **FOUR-mask training requirement** discovered during v81 Phase 2 masking audit. Beyond kv_mask + notebook_pool_mask, we ALSO need main_attn_mask (main self-attn at answer-span KV) and embed_mask (zero input embeddings at answer-span). All 4 are required for clean train/eval consistency. v80 era trained with only 2 of 4.

**v81 paradigm (current, May 27): same task per breath at different granularity.** Architectural insight: the 4 transformer layers are SHARED across K=7 breaths. Different tasks per breath = 7-fold gradient conflict. v81 makes every breath do the SAME job — coarse-to-fine refinement of a multi-list DAG representation. IB clustering on Pythia embeddings yields a 32-leaf codebook tree (4 ops × ~8 sub-clusters). Each breath emits 4 parallel lists — ops, types_path, args1, args2 — with each breath filling/refining ONE list more than the previous. Multi-head WaistController (4 heads, one per list). Full 4-mask training enforced. v81 prod fires now with BATCH=4, FIXED_LEN=256 (post-perf-tuning) — ETA ~5h to step 2000.

**The lesson the empirical curriculum keeps re-establishing:** "train on what you evaluate on" — but verify it BEFORE chasing metrics. Lookahead leaks between training and eval can silently invalidate hundreds of dollars of CE iterations. Always run a masking audit BEFORE celebrating CE drops. v45's regularization stack, v55's per-breath specialization, v79's causal masks, v80's format-ladder, v81's four-mask training are all expressions of the same principle — match the training signal to the target geometry, verify there are no leaks, then the architecture does its job.

**Architectural triad preserved across all paradigms.** Each successive paradigm (multi-cycle → rep-space-thinking → DAG → multi-list parallel) keeps the breathing + π-cycled-RoPE + WaistController triad and reshapes only the supervision interface. The codebook (V78_HEAD_CODEBOOK_N=32 now, was 12) now natively aligns with the IB cluster tree leaves — internal representation and supervision targets share the same indices.

A 127M model that breathes, refines coarse-to-fine across 7 cycles, emits an integer-encoded DAG, and hands off to SymPy. ~7 months to December 25.

---

## Coda — what came next

Within two days of this snapshot, v82-v97 added 17 more architectural
variants on the WaistController paradigm. None broke the GSM8K plateau
(0-1.7% on real problems). On 2026-05-29 the strategic pivot to Sudoku
validated the factor-graph framing of the breathing transformer:
**v98 hit 79% puzzle accuracy on easy Sudoku, 97.65% cell accuracy.**
The architectural arc described above ended there; the post-v98
empirical state is in `README.md` §3 and `paper/outline.md`.
