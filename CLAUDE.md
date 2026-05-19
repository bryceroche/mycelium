# Mycelium v4: The Breathing Transformer — Agent Brief

**Author:** Bryce + Claude · **Deadline:** Dec 25, 2026 · **Target:** MATH-500
**Platform:** Shadow Glass (AMD 7900 XTX, 24GB) · tinygrad + AM driver · no ROCm

For the full conceptual writeup, see `README.md`. This file is the agent-facing brief: the architecture compressed around the seven components that form the closed feedback loop, plus the rules that govern editing this codebase.

---

## 1. The Architecture in One Paragraph

A small transformer (4 specialized layers from Pythia-410M L0-3, h=1024, 16 heads, ~127M params total) that **loops its own layers** to refine each problem. Each loop is one expand-collapse breath shaped as a sine wave; the 16 heads scan 16 different phase angles in parallel like BirdNET identifying multiple bird species in one clip; a running integral accumulates evidence across breaths; a controller decides when enough has been accumulated. All reasoning stays in 1024d representation space — tokens are generated **once** at the end, never between breaths (the "copy machine principle": photocopying through tokens destroys signal, observed empirically as "had had had" collapse after 2 autoregressive loops).

---

## 2. The Seven Components of the Closed Feedback Loop

The breathing transformer's reasoning is an **irreducible closed feedback loop** of seven components. Each amplifies the others; remove any one and the system degrades. As of 2026-05-10, all seven are implemented and wired together — this is the architecture's first complete realization.

| # | Component | What it does | Why it's necessary | Implementation |
|---|---|---|---|---|
| 1 | **Rotation** (π-cycled RoPE) | Per-head phase offsets rotate the attention geometry each breath, so each loop sees the problem from a different angle | Provides **independent observations**. Without it, every breath sees the same view and integration accumulates redundant info. Geometric, not learned — gradient descent cannot erase it. | `breathing.py: RoPE` |
| 2 | **Integration** | Gated running integral across breaths; controller-emitted gate weights novel observations high, redundant ones low | Makes observations **cumulative**. Rotation without integration is amnesia — each breath's insight is forgotten when the next begins. Bayesian evidence combination over independent angles. | `breathing.py: BreathingBlock.breathe` |
| 3 | **Notebook** | 512d pages written after each breath, persisting across both inner loops and outer execution cycles; tree-structured attention over ancestors/siblings/children | Provides **memory across breaths and cycles**. Without it the controller has no basis for comparing "now" against "three breaths ago" — cannot detect convergence or track factorization evolution. | `controller.py: Notebook` |
| 4 | **Lookup table** | 16×1024 cosine matcher storing prime operations (add/sub/mul/div/fraction/compare/combine/sequential...) each with pattern, resonant angle, subtraction mask, confidence threshold; plus a 16×16 coupling matrix | Provides the **reference library and target map**. Without it the controller adapts on energy signals alone ("something is changing") without knowing **what** to look for. Transforms blind search into guided search. Empirically validated 2026-05-10: trained 16×1024 table hits 100% op classification. | `lookup_table.py: LookupTable` |
| 5 | **Controller** | ~40M conductor thinking in 512d. State reader (Perceiver, 1024d→512d) + notebook attention + decision heads emitting `{temperature, gate, stop_logit, step_mult}`. Trained by REINFORCE + lookup-CE + stop calibration on a **separate optimizer**. | Provides **adaptive feedback**. Without it rotation is uniform, temperature is fixed, stopping is arbitrary. The intelligence of the loop. | `controller.py: Controller` |
| 6 | **Temperature modulation** | Controller scalar × sine baseline. Warm = broad/coarse attention; cool = sharp/fine attention. | Controls **resolution at each angle**. Without it every breath has the same precision — wastes early breaths on unnecessary precision and starves late breaths of needed precision. | `breathing.py: BreathingLayer (temp_mult)` |
| 7 | **Step size / rotation rate** | Controller emits `step_mult` adjusting the π/max_loops baseline step | Determines **spectral coverage efficiency**. Nyquist: to resolve two primes separated by Δθ, the rotation step must be ≤ Δθ/2. Too large misses closely-spaced modes; too small wastes breaths. | `breathing.py: breathe_controlled` |

### The Loop, End-to-End

The closed loop is invoked via `model.breathe_controlled(tokens, max_loops, notebook)`. Per breath:

1. Controller reads the running integrated rep
2. → writes a 512d page into the notebook
3. → notebook attention refines the page over all prior pages
4. → lookup table matches the page against its 16 prime entries, returning match weights + confidence
5. → decision heads emit `{temperature, gate, stop_logit, step_mult}` for the next breath
6. → next breath rotates at the controller's adaptive phase, runs at the controller's temperature, integrates weighted by the controller's gate

The loop **terminates** when both: (a) the integral has stabilized (Lyapunov criterion — new breaths add negligible information), and (b) the spectral residual is noise (all significant primes identified and subtracted).

### Gradient Separation Is Enforced by Construction

**The controller's gradient NEVER flows through the transformer.** Three days of v1-v3 evidence proved any such path collapses to one basin.

- `model.parameters()` returns transformer + lookup_table params only. Trained on main CE + small joint lookup-aux CE.
- `controller_train_step` uses a separate optimizer over `model.controller_parameters()`. Loss is per-breath lookup-CE + stop calibration. Gradients that reach transformer params are discarded by the next `main_opt.zero_grad()`.
- Verified on a 5-step joint smoke: 0/39 transformer params changed in controller training; 61/62 controller params changed.

**Do not introduce any code path that lets controller gradients reach transformer weights.** This is the single most important architectural rule.

---

## 3. Empirical Status (as of 2026-05-19)

**Two champions in two paradigms.**

- **Misaligned-decode paradigm (multi-cycle eval via ln_f + embed_out):** v45 reg take 3 step 1000 = **96 / 94 / 93 on L4_MIXED (A=1 / A=4 / A=8)**. File: `.cache/l4_mixed_ckpts/v45_reg_take3_step1000.safetensors`.
- **Rep-space-thinking paradigm (K breaths once, decode via WaistController):**
  - v55 step 500 = **89.0% aligned on L4 (K=2)**. File: `.cache/l4_ckpts/v55_controller_codebook_step500.safetensors`.
  - v56 step 500 = **89.0% segmented on L4.5 (K=3)**, aligned only 75%. File: `.cache/l4_5_ckpts/v56_controller_codebook_l4_5_step500.safetensors`.
  - The K=3 segmented–aligned gap (+14 pt) is the proof that **per-breath waist specialization is real**: at K=3 breath-1 decodes division and breath-2 decodes subtraction on the same problem, and the last-breath-only "aligned" eval reads the wrong op when generating step-1 tokens. At K=2 breaths converge and aligned ≈ segmented (89/90).

Trajectory:
- **L3-spaced:** 70% at A=8 vs 65% at A=1 (depth helps). 65% is the 4-layer arithmetic ceiling.
- **L4 v4 (May 12):** 43% pure L4 from arith_mixed_v6 warm-start. Step 1500 was the headline.
- **L4_BORROW v1 overnight (May 13):** 80% on L4_BORROW eval (cascade-heavy), but only 32% on pure L4 — narrow-curriculum trade-off. Catastrophic forgetting between L4_BORROW and standard L4.
- **L4_MIXED v1 (May 13):** broadest distribution (6 standard + 3 cascade variants). Step 1500 = **66/67/65** on L4_MIXED eval. +20 vs v4 baseline. Best ckpt to date.
- **v24c dual notebook (May 15):** DUAL notebook (REPLACE + ACCUMULATE 512d, random 0.02 init, attn-pool write source). Step 500 = **96/94/91** on L4_MIXED — first time all loop counts in 90s. Overfits past step 1000.
- **v45 reg take 3 (May 18):** warm-start from v24c step 500 + reg stack (`STOCH_DEPTH_P=0.10`, `LABEL_SMOOTHING=0.1`, `WEIGHT_DECAY=0.05`). Step 1000 = **96/94/93** — ties v24c at A=1/4, **+2 at A=8**. Misaligned-decode champion. 200-step continuation showed step 1000 is at/near local peak.
- **STAGE2_NOTEBOOK inference-path fix (May 18):** v45 takes 1+2 collapsed (~0% gen acc, val loss fine) because `cached_generate_batch` Stage 2 was updating notebook per generated token — train/eval mismatch from the v40 work. Gated behind `STAGE2_NOTEBOOK` env var (default 0). v24c step 500 ckpt eval'd on current code recovered to its training-time 96/94/91.
- **v54 controller paradigm (May 19):** K=2 inner breaths, REPLACE notebook, BFIELD_WAIST=512, WaistController (1 cross-attn block) as the sole text-supervision conduit. v46b L4.5 step 750 warm-start, 500 steps L4. Aligned eval = **85%**. Demonstrates that all reasoning can happen in 512d rep space with token decode only via the controller.
- **v55 controller + codebook (May 19):** v54 + WAIST_CODEBOOK_N=64 (16 heads × 4 ops, values zero-init). Aligned eval = **89%**, val_loss 0.013/0.009 (vs v54 0.021/0.019).
- **v56 K=3 on L4.5 (May 19):** v55 architecture extended to K=3 from v55 step 500. Aligned 75%, segmented **89%**. The +14 pt segmented–aligned gap (vs v55 K=2's +1 pt) proves the K-axis is doing real work — breaths specialize at K=3 in a way they didn't at K=2. Eval script: `scripts/eval_ckpt_controller_segmented.py`. JIT'd per-breath training path added the same day (flat 0.95s/step vs eager's linear growth).

### What the ablations established

| Component | Status | Load-bearing? |
|---|---|---|
| Rotation (per-head π-cycled) | Validated | YES (−73 pts if ablated; only clearly load-bearing piece) |
| Integration (gated running integral) | Validated | Decorative at converged ARITH_HARD; untested in multi-step |
| Notebook | Validated | Decorative on single-cycle data |
| Lookup table | 100% op classification | Useful as supervised target, decorative as live signal |
| Controller (temperature/gate/step_mult/stop) | Validated | Learned `f(breath_idx)` not `f(rep)` — open-loop schedule, problem-blind |
| Temperature modulation (sine baseline 2.0→0.7) | Validated | Load-bearing for warm-start stability |
| Step size (controller-emitted) | Validated | Decorative (no problem-dependent signal) |

The "7/7 closed feedback loop" framing is the canonical vision. Empirically only **rotation + sine temperature** are clearly load-bearing. The controller and downstream components remain as scaffolding — they're correctly wired, they just haven't yet found a job that benefits the loss.

### Verification probe — definitive negative

MLP probes on v6 reps (shallow 1024→512→1; deep 8192-concat→2048→512→1, 17M params, 1000 steps) BOTH failed. Test acc 34%, AUC 0.29 — anti-correlated. Verification info is NOT in the trained reps. The "deeper fix" — training calibration as a transformer objective rather than a frozen probe — is the current direction (`calibration_train_step`).

---

## 4. Specifications (Tight)

- **Init:** Pythia-410M L0-3. Take attn+FFN weights + token embeddings (50304×1024) + untied output head. Phase-specific copies of Q, K, FFN gate (one per breath phase). Shared V, O, FFN up/down, norms.
- **Dimensions:** h=1024, 16 heads × head_dim 64, FFN 4096, vocab 50304, max seq 512, 4 layers/breath, max 8 loops.
- **Parameters:** ~35.7M transformer processing + 51.5M token embeddings + 51.5M untied output head = ~139M (training log reports 134.5M `trainable params` from `collect_params()`; minor delta from vocab_active=50277 used in output). Separate ~6.6M controller (Step C; spec target ~40M at Step E) trained on its own optimizer. With 8 loops → ~286M effective processing capacity.
- **Memory:** ~5GB mixed precision, ~19GB headroom. KV cache sized to actual seq length (not model max): cache_max_len=32 for L3-spaced → ~2GB at B=100.
- **Platform:** AMD 7900 XTX, tinygrad, AM driver (working as of 2026-05-11 — see §6), Ubuntu 24.04. No ROCm, no CUDA, no PyTorch.

---

## 5. Editing Rules

- **Never** create a code path where controller gradients can reach transformer weights. Use separate optimizers and verify with the parameter-change smoke if in doubt.
- **No mid-breath token generation.** All reasoning stays in 1024d hidden states / 512d pages. Tokens are generated once at the end. (Empirical: "had had had" within 2 loops if violated.)
- **Diversity must be structural, not learned.** π-cycling, per-head phase offsets, sine temperature — all geometric. Every v1-v3 learned diversity mechanism (scales, soft tokens, codebooks, fingerprints) collapsed to constant within one epoch.
- **Digit-spaced generation only** for arithmetic. Whole-number BPE tokens force memorization.
- **Bryce wants root-cause perf fixes**, not workarounds, when perf is the bottleneck. (See memory.)
- **KV cache invariants:** size to actual seq length; pad eval batches to fixed batch_size so compiled graphs match; compile once during first eval, replay for the rest of the run.

---

## 6. Current Work In Progress

- **v56 K=3 segmented on L4.5 = 89% (May 19):** v55 architecture extended to K=3 inner breaths from v55 step 500. Aligned (last-breath) eval 75%, **segmented (breath-k decodes step-k) 89%**. The +14 pt gap is hard evidence the K-axis carries real specialization: on the same problem, breath-1 decodes division (`2 8 2 / 2 = 1 4 1`) and breath-2 decodes subtraction — and the aligned eval reads the wrong op when generating step-1 tokens. At K=2 (v55) breaths converged (+1 gap); at K=3 they specialized (+14 gap). Eval script: `scripts/eval_ckpt_controller_segmented.py`. JIT'd per-breath training path: `mycelium/l3_training.py:_compile_jit_per_breath_step` (flat 0.95s/step vs eager's linear growth from 1.8 → 8.2s).
- **v55 controller + waist codebook = 89% aligned on L4 (May 19):** First rep-space-thinking result. K=2 inner breaths, REPLACE-only notebook, BFIELD_WAIST=512 end-of-breath. A 1-block cross-attn `WaistController` reads the compressed 512d waist, cross-attends to the prompt embedding, and decodes via tied `embed_out` → vocab logits — that is the only text-supervision conduit. Per-breath supervision: breath k decodes step-k's gen_target. v55 adds a 64-entry codebook at the waist (keys randn × 0.02, values zero-init). vs v54 (codebook OFF): **+4 pt aligned (85 → 89)**, val_loss 0.021/0.019 → 0.013/0.009. Ckpt: `.cache/l4_ckpts/v55_controller_codebook_step500.safetensors`.
- **v46 hybrid-heads quadrature (May 18, decorative):** split the 16 heads, 8 keep PER_HEAD_PITCH pitch `l·π/64`, 8 add `π/2`. Cheapest test of the photon/quadrature idea (item #10, #15). Validated as decorative in v46/v47/v47b warm-start collapses (see `memory/project_2026_05_18_session_synthesis.md`). v46b step 750 (92/92/88 on L4.5) remains a strong intermediate ckpt — used as the warm-start for v54/v55.
- **Regularization stack validated (v45, May 18):** `STOCH_DEPTH_P=0.10` + `LABEL_SMOOTHING=0.1` + `WEIGHT_DECAY=0.05`. l3_train.py mask-gen ensures ≥1 active breath kept per step (skip SD when n_loops<2). Gates the integral-contribution scaling in `BreathingBlock.breathe()` and `BreathingTransformer.breathe_with_lookup()` only when `Tensor.training`. Now standard.
- **STAGE2_NOTEBOOK env var (May 18):** gates per-token notebook updates in `cached_generate_batch`'s Stage 2 decode JIT. Default 0 matches v24c-era training. Set to 1 only for models explicitly trained with this mode. Bug found by direct ckpt-eval diagnostic (`scripts/diag_v24c_eval.py`).
- **AM driver (working since 2026-05-11):** `DEV='PCI+AMD'` works after Secure Boot off + `vm.compact_unevictable_allowed=0`. `scripts/setup_am_driver.sh` installs both. See `memory/project_am_driver_state.md`.

---

## 7. What We Carry Forward (and What We Left Behind)

**Forward:** expand-collapse breathing pattern, π-cycled attention, equal-reward decomposition, number augmentation, gradient separation, differentiable lookup table (from project origins), the copy machine principle, the JIT-fused KV cache.

**Left behind:** Llama 1B (replaced by Pythia-410M L0-3), LoRA atoms and continuous scales, the straight-through gradient estimator, soft token diversity mechanisms, PyTorch/ROCm, Windows.

A 127M model that breathes, alternates, integrates, and factorizes. ~7 months to December 25.

---

## 8. Open Research Threads (Laundry List, as of 2026-05-18)

Active design ideas across the project. Each tagged with status: ✅ validated, 🟡 partial/stuck, 🆕 new/unstarted, 🔴 known broken.

1. **✅ Centroid injection at the waist (partial)** — Validated in v55 via `WAIST_CODEBOOK_N=64` (zero-init values) + WaistController decode. +4 pt aligned over v54. CFG α and LoRA facets of the unified design are still unrealized; only the centroid-injection facet has earned a clean win so far. Design captured in `memory/project_unified_waist_harness.md`.

2. **🟡 E & B waves perpendicular** (π-cycled RoPE + expand-collapse) — E field works brilliantly (rotation + per-head pitch). B field tried in v38 (decorative), v39 (A=8 collapsed via info chain through repeated bottlenecks), v40 (failed in combination). Not phase-locked yet — see item 15.

3. **🟡 Per-(layer, op, head) 256-entry lookup** — Infrastructure ready: `N_LOOKUP_ENTRIES` env var, `extract_per_op_layer_head_centroids.py` captures per-head 64d via W_O column blocks. Blocked by centroid quality — see item below on +/- blindspot.

4. **🟡 Compression-waist harness for centroid injection** — Designed: supervised init from pre-extracted centroids, aux op-CE supervision, conditional dropout for CFG. v38/v39 had pieces; no full working integration yet.

5. **✅ Overfitting / memorization control** — Validated 2026-05-18 by v45. Stack: `STOCH_DEPTH_P=0.10` (per-breath Bernoulli drop with ResNet-style 1/(1-p) scaling, skip at n=1, ≥1 kept safeguard at n≥2) + `LABEL_SMOOTHING=0.1` (training-only, eval CE gated on `Tensor.training`) + `WEIGHT_DECAY=0.05`. From v24c step 500 (96/94/91), v45 step 1000 = 96/94/**93** — ties at shallow, **+2 at A=8**, depth-spread compressed 5pt → 3pt. Reg makes deep loops MORE useful, not less. Dropout still blocked by tinygrad JIT (`Tensor.rand_like`) but stoch depth subsumes it for this architecture.

6. **✅ All thinking in rep space, autoregressive decode only at the end** — Realized via the WaistController paradigm (v54: 85% aligned, v55: 89% aligned). K inner breaths run once on the prompt + currently-emitted tokens; a 1-block cross-attn `WaistController` reads the compressed 512d waist and decodes via tied `embed_out` for one token at a time. No mid-breath token generation. Per-breath supervision via per-step gen_target. Eval script: `scripts/eval_ckpt_controller_l4.py`. Standing extension: stage-2 KV cache reuse for inference speedup, K=3/K=4 for L4.5/L4.7.

7. **🟡 BirdNET parallelism — classification + computation across 16 heads in parallel** — Per-head pitch (v23a) gives 64 distinct (layer, head) angular positions. Conceptual framework partially supported; per-head computation specialization not yet validated as load-bearing.

8. **✅ REPLACE notebook (not accumulate)** — Validated in v54/v55 (NOTEBOOK_V24=1, NOTEBOOK_ACCUMULATE_ENABLED=0, NOTEBOOK_DUAL=1). REPLACE-only works in the K-breath WaistController paradigm. The breath-N output replaces the page each step, so the controller reads the most-recent compressed waist rather than the running integral.

9. **🆕 Recursive hierarchical IB → tree-structured lookup → MCTS traversal** — New direction. Builds on existing IB plateau analysis. Currently flat 256-entry lookup; tree-structured would enable coarse-to-fine MCTS at inference. Substantial design + implementation work.

10. **🆕 Photon (not helix) — E and B zero-crossings TOGETHER** — Reframing item 2. Requires sinusoidal amplitude co-modulation of both rotation magnitude and compression magnitude, phase-locked. See item 15 for mechanism brainstorm.

11. **🆕 Diffusion analogy — coarse-to-fine denoising in parallel with SNR awareness** — Partially aligned (SINE_TEMP IS a noise schedule, breath_time_embed IS diffusion-step conditioning). Not formally integrated. Could reframe: high temp = noisy, low temp = denoised, integral over breaths = denoising steps.

12. **✅ Curriculum learning works** — Validated: ARITH → L3 → L4 → L4_MIXED → L4.5 chain. v24c warm-started from v24b chain reached 96%. Recent cold-start ARITH iterations are REGRESSION from curriculum. Lesson: always warm-start from a working checkpoint.

13. **✅ π-cycled RoPE works brilliantly** — Validated as load-bearing. PER_HEAD_PITCH=1 (frozen per-(layer, head) offsets at l·π/64) gives 64 unique angular positions. Head-collision resonance discovery (v22 → v23a) refined this.

14. **🟡 Compression on the last layer** — Tried in v39 (end-of-breath waist after L3, enforced 512d, sin mod). A=8 collapsed (-51pt vs v38) from info-chain leakage through repeated bottlenecks. The Stage 2 decode fix (item 6) might rescue this — eval-time 8 breaths × decode cycles compounds the bottleneck, but ONCE-then-decode wouldn't.

15. **🆕 Mechanism for E&B zero-crossing co-oscillation** — Open brainstorm. Currently: E (rotation angle) accumulates monotonically across breaths (period = `max_loops` breaths); B (compression) fires every breath identically. They're at different frequencies and not phase-locked. To phase-lock with zero crossings: both need to follow `sin(l · π / max_loops)`-style amplitude envelopes that peak in mid-breath sequence and go to zero at endpoints. But "E magnitude" needs definition — rotation accumulates phase, not amplitude. Candidate: modulate temperature × rotation EFFECT via sine envelope, so the "energy" of rotation oscillates (high at peak, low at zero-crossings) while the angle still advances monotonically.

---

### Cross-cutting issue surfaced 2026-05-17:

**+/- blindspot in cold-start ARITH.** Both v36 (50/51/49 final) and v38 (61/59/58 final) — independent architectures, independent training runs — converged to × ÷ specialists with NEAR-ZERO accuracy on + and -.

- v38 per-op extraction: {+ : 0, − : 1, × : 175, ÷ : 139} of ~200 each
- v36 per-op extraction: {+ : 2, − : 4, × : 175, ÷ : 118} of ~200 each
- Mean accuracy 50-60% on uniform ARITH masks total failure on half the ops.

**Root cause hypothesis:** multi-digit + and − require carry propagation (sequential dependency between digit predictions). The model finds a local minimum that handles × and ÷ patterns (less carry-dependent) and fails to learn carry chains.

**v24c does NOT have this blindspot** because L4_MIXED training had + and − embedded in multi-cycle word problems with explicit intermediate-result supervision — the model couldn't take the × ÷ shortcut.

**Strategic implication:** stop cold-starting from Pythia on bare ARITH. Build forward from v24c on richer distributions (L4.5, GSM8K).
