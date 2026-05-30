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

## 3. Empirical Status (as of 2026-05-29)

**v98 Sudoku is the breakthrough.** After 17 GSM8K architectural variants plateaued at 0-1.7%, a strategic pivot to Sudoku validated the breathing-transformer paradigm as approximate joint MAP inference on a factor graph. **79.0% puzzle accuracy on easy, 97.65% cell accuracy, 87M parameters, ~4h training on AMD 7900 XTX.** See §3a below for the full breakdown. The earlier GSM8K work (v45/v55/v56/v59/v77-v81) is preserved here for context.

**Three paradigm tracks, three champions.**

- **Misaligned-decode paradigm (multi-cycle eval via ln_f + embed_out):** v45 reg take 3 step 1000 = **96 / 94 / 93 on L4_MIXED (A=1 / A=4 / A=8)**. File: `.cache/l4_mixed_ckpts/v45_reg_take3_step1000.safetensors`.
- **Rep-space-thinking paradigm (K breaths once, decode via WaistController):**
  - v55 step 500 = **89.0% aligned on L4 (K=2)**. File: `.cache/l4_ckpts/v55_controller_codebook_step500.safetensors`.
  - v56 step 500 = **89.0% segmented on L4.5 (K=3)**, aligned only 75%. File: `.cache/l4_5_ckpts/v56_controller_codebook_l4_5_step500.safetensors`.
  - **v59 step 1500 = 45% segmented on L4.7 (K=4)**, with PERFECT step decomposition. File: `.cache/l4_7_ckpts/v59_continue_l4_7_step1500.safetensors`. Per-step accuracy ~78% — breaks the 4-layer 65% arithmetic ceiling. Errors are arithmetic execution within a step, not orchestration.
  - The K=3 segmented–aligned gap (+14 pt) is the proof that **per-breath waist specialization is real**: at K=3 breath-1 decodes division and breath-2 decodes subtraction on the same problem, and the last-breath-only "aligned" eval reads the wrong op when generating step-1 tokens. At K=2 breaths converge and aligned ≈ segmented (89/90).
- **DAG paradigm (v77+, bridge to GSM8K):** instead of having the model emit a numeric answer, the FINAL breath emits a SymPy-executable DAG (`x0 = 50 / 60 ; x1 = x0 * 12 ; answer = x1`). SymPy executes the arithmetic. The model only needs to learn STRUCTURAL correctness — operand binding, operator selection, dependency graph — not multi-digit arithmetic execution. K=7 breaths × 7 supervision layers (L0..L6). **Waist load-bearing validated** (Δzero=+3.77, Δswap=+4.59 on v79 step 1000). **v80_prod_step400 post-mask-fix: 28% DAG parse, 1.7% accuracy on 60 problems** — first non-zero real-task accuracy.
- **v81 paradigm (May 27): same task per breath at different granularity.** Architectural insight: the 4 transformer layers are SHARED across K=7 breaths. If each breath has a different task, the 7-fold gradient pulls weights in different directions. v81 makes every breath do the SAME job — coarse-to-fine refinement of a multi-list DAG representation. (1) IB clustering on Pythia embeddings of GSM8K L2 step descriptions yields a 32-leaf cluster tree (4 ops × 8 sub-clusters avg). (2) Each breath emits 4 parallel lists — ops, types_path, args1, args2. (3) Multi-head WaistController (4 heads, one per list) at the final breath. (4) FOUR masks required for clean training. Plateaued at ~3% GSM8K despite clean training infrastructure — the bottleneck was the compressed 512d waist → AR decode interface, not the breathing/factor structure.
- **v98 Sudoku paradigm (May 29): factor graph inference paradigm — THE BREAKTHROUGH.** Pythia-410M L0-L3 backbone (same as before), K=20 breaths, per-head attention masks encoding factor topology (heads 0-4: row, 5-9: col, 10-14: box, 15: global), per-breath weighted CE on cell predictions, constraint energy loss, calibration head with detached target, learnable per-breath delta gate, state-embed/digit-codebook aligned init. **Final eval (K=20, n=200): easy 97.65% cell / 79.0% puzzle, medium 83.33% cell / 6.5% puzzle, hard 76.16% cell / 0.0% puzzle.** The constraint energy curve (21.0 at K=1 → 0.71 at K=20, geometric decay rate ~0.5×/3 breaths) IS the mathematical signature of loopy BP convergence — the central empirical claim of the project. See `memory/project_factor_graph_framing.md` and `paper/outline.md`. Files: `mycelium/sudoku.py`, `.cache/sudoku_ckpts/v98_prod_final.safetensors`.

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

- **v98 Sudoku — DONE (May 29):** prod training to step 5000 complete. Final ckpt `.cache/sudoku_ckpts/v98_prod_final.safetensors`. K-sweep at K ∈ {1,3,5,8,12,15,18,20} done (`.cache/v98_ksweep/K*.log`). Per-breath convergence diagnostic done (`.cache/v98_per_breath_convergence.json`). Paper outline drafted (`paper/outline.md`) with all real numbers and the central energy decay table. Six-component architectural recipe documented (`memory/project_v98_sudoku_validates_paradigm.md`). Factor graph framing memorialized (`memory/project_factor_graph_framing.md`).
- **v99 factor graph paradigm — DONE, NEGATIVE RESULT (May 29):** synthetic arithmetic factor graphs (50K train + 5K test) + bipartite breathing transformer (K=10 ceiling due to AMD JIT capacity, soft factor-type-embedding-on-edges, moment-matching constraint energy). 2000 steps prod training. **Result: cell_acc 9% on easy, flat across K∈{1,2,5,10}, flat across DAG depths 2-7. Energy decays geometrically like Sudoku's (4.7M→2.5M from K=1→K=2), but accuracy stays at chance-floor.** The model is doing BP — and converging to a uniform-distribution wrong fixed point. Moment-matching energy has a trivial low-energy attractor (broad distributions satisfy var_c ≈ var_a + var_b when var_a is small). The architecture works mechanically; the breathing rhythm is wrong for this topology. See `memory/project_musical_keys_topology.md`.
- **Musical keys framework (May 29 — NEW conceptual framing):** the breathing transformer's "instrument" is universal (Pythia + K iter + masked attn), but each problem class is in a different **key**, requiring a different breathing **rhythm** matched to its topology's symmetry. Sudoku=cyclic key (rotational breathing works), arithmetic DAG=directional key (rotation collapses), verification=cadence key (alternating direction), multi-modal=modulation. v99's K=1≈K=10 outcome is the directional-key failure mode for rotational breathing. v100 = directional-key with matched rhythm.
- **JPEG codec mental model:** each breath = Transform (attention basis rotation, key-dependent: π-cycled RoPE for cyclic, factor-aligned mask for directional) → Quantize (waist 1024→512, deliberate lossy step that forces commitment) → Encode (notebook carries survivors to next breath) → Psychoacoustic model (next-breath CE is the LEARNED equivalent of MP3's perceptual model — "what does the next breath need?"). Complements musical keys: keys tell you the RHYTHM, codec tells you the COMPRESSION. Together: rhythm × compression = the architecture.
- **JPEG codec — implementation status (May 30):** v98 Sudoku and v100 factor graph are **3-of-4 codec architectures** — they have Transform + Encode (delta_gate) + Psychoacoustic (per-breath CE) but **NO explicit Quantize step**. Residual stream stays at 1024d through all breaths. v54-v95 architectures had the waist (WaistController paradigm); it was dropped when v98 removed AR decode. Suspected to be load-bearing accuracy left on the table. v101 (next) adds the per-breath waist back to test the hypothesis: codec's Quantize step IS the commitment mechanism. If v101 > v100 on factor graphs, the codec framework's compression step is validated for the directional key.
- **v100 spec (next):** topological staging masks (breath k sees up to DAG depth k, mask GROWS across breaths, info has to be earned by waiting for predecessor breaths) + aligned init for 100-way codebook (state_embed[i] = digit_codebook[i] — the v98 unlock v99 missed) + hard head specialization (heads 0-3 add only, 4-7 sub only, 8-11 mul only, 12-15 div only — drop soft factor-type embed) + factor-execute auxiliary loss (direct supervision on factor node hidden states) + replace moment-matching with exact KL on convolved distributions (eliminate the uniform attractor).
- **v99 optimal prefill design (May 29):** memorialized in `memory/project_v99_optimal_prefill_design.md`. Multi-resolution prefill (final residual + last-breath expansion 1024→2048 + trajectory waists + factor-decomposed views + per-position calibration). For tasks requiring AR decode (GSM8K), compress for thinking, expand for speaking. Explains v80 GSM8K failure: decode read from compressed 512d waist directly, no expansion, no trajectory. v99 inherits the breathing inner loop from v98; the expansion + multi-source prefill is added when we need AR output.
- **v81 abandoned (replaced by v98 + v99):** v81 prod completed but plateaued at <5% on GSM8K. The 4-mask discovery is preserved as a portable lesson; the multi-list refinement architecture is preserved as inspiration for v98's same-task-per-breath principle.
- **FOUR-mask training requirement — DISCOVERED (May 27):** Sonnet's masking audit during v81 Phase 2 found that **two more masks were missing from training**, on top of the kv_mask + notebook_pool_mask we knew about. The full mask set: (1) `kv_mask` — WaistController cross-attn, (2) `notebook_pool_mask` — breathe_with_lookup notebook pool, (3) `main_attn_mask` — main self-attn KV at answer-span (was leaking through self-attn even with causal triangular mask), (4) `embed_mask` — zero input embeddings at answer-span (residual stream was carrying answer info regardless of attention masking). Without all 4, the model gets train/eval mismatch and degenerate AR output. v80 prod ckpts were trained with only 2 of 4. v81 trains with all 4. See `memory/project_v81_four_mask_discovery.md`.
- **Eval bug fixed (May 26):** `eval_v77_dag.py` was missing `notebook_pool_mask` and using a growing `kv_mask` (covering generated tokens). This silently invalidated all v77+ eval accuracy measurements — the model HAD learned DAG structure under teacher forcing but autoregressive eval was so OOD it produced degenerate "x1 = x1 = ..." cascade. **v80_prod_step400 went 0% → 28.3% DAG parse rate, 1.7% accuracy** after the fix. The CE-chasing v80 iterations were on a model that actually worked; we just couldn't measure it.
- **IB clustering — DISCOVERS natural codebook (May 27):** `scripts/diag_ib_clustering.py` does hierarchical K-means on Pythia embeddings of L2 NL step descriptions, per OP. With min_size=150/max_depth=3/max_k=3 yields **32 leaves across 4 op families** — naturally aligns with `V78_HEAD_CODEBOOK_N=32`. Each leaf has a sample-representative cluster (e.g. "Convert minutes to hours" → leaf DIV.0.1, "Find half of N" → leaf DIV.0.2.1). OP-constrained nearest-centroid assignment of new problems is deterministic and 100% OP-correct by construction. Tree + centroids saved to `.cache/ib_tree.json` + `.cache/ib_centroids.npz`.
- **v80 prod v3 LOCK (May 26):** $95 Haiku regen produced 4355 train + 743 test records. Step 400 was the best ckpt (loss 3.83) before the eval bug fix surfaced. The whole v80 v1→v8 iteration arc (10 sub-versions, ~$200 in Haiku) was chasing per-breath CE smoothness — never measured DAG correctness until the eval-bug discovery.
- **v8 v3 architectural insight (May 27):** Progressive DAG refinement supervision — every breath outputs the same DAG skeleton with one more piece filled in (skeleton → OPs → first arg → all args → cluster ID → operator symbols → final DAG). Smoke step 70: spread 1.28 CE (vs v3's 2.66, v6's 0.69 best). The 4 shared transformer layers learn ONE task — fill in the next refinement piece — across all 7 breaths. This insight motivated v81.
- **Diagnostics built (May 25-27):** `diag_waist_zero_ablation.py` (Δzero/Δswap analysis), `diag_jsd_attention_boundaries.py` (Panama-hat segmentation discovery — found breaths attend to SAME prompt regions; specialization is in Q transformation, not K/V reading), `diag_ib_clustering.py` + `diag_ib_tree_export.py` (codebook structure discovery), `diag_v81_masking_audit.py` (4-mask audit).
- **v59 K=4 on L4.7 + PERFECT decomposition (May 20, pre-DAG era):** v58 wider controller + 1500 more steps. Segmented eval = 45% on L4.7, per-step accuracy ~78%, broke the 4-layer arithmetic 65% ceiling. Motivated the move to DAG paradigm (separates orchestration from arithmetic via SymPy).
- **AM driver (working since 2026-05-11):** `DEV='PCI+AMD'` works after Secure Boot off + `vm.compact_unevictable_allowed=0`. See `memory/project_am_driver_state.md` and `memory/reference_tinygrad_am_quirks.md`. AMD JIT capacity limits force compromises in v81: bias-only per-head MLP, per-head CE only at final breath.

---

## 7. What We Carry Forward (and What We Left Behind)

**Forward:** expand-collapse breathing pattern, π-cycled attention, equal-reward decomposition, number augmentation, gradient separation, differentiable lookup table (from project origins), the copy machine principle, the JIT-fused KV cache.

**Left behind:** Llama 1B (replaced by Pythia-410M L0-3), LoRA atoms and continuous scales, the straight-through gradient estimator, soft token diversity mechanisms, PyTorch/ROCm, Windows.

A 127M model that breathes, alternates, integrates, and factorizes. ~7 months to December 25.

---

## 8. Open Research Threads (Laundry List, as of 2026-05-26)

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

16. **✅ Waist is load-bearing AND breath-specific** — Validated 2026-05-25 by `diag_waist_zero_ablation.py` on v79 step 1000. Δzero=+3.77 (controller doesn't bypass waist), Δswap=+4.59 (waists are breath-distinct, not interchangeable). 4-layer WaistController is right-sized — beefing up risks bypass. The L4→L5 CE cliff is an ENTROPY problem (target format), not a structural problem. See `memory/project_waist_load_bearing_validated.md`.

17. **🆕 Format-ladder rigidification (v80)** — v78c's per-breath CE shows a cliff: L0-L4 ≈ 5, L5-L6 ≈ 2 — the 3.5-point jump at L4→L5 corresponds to the verbal→symbolic phase transition in the target format. Plan: rebuild L0-L6 with PROGRESSIVELY tighter templates (paraphrase → slot template → step grammar → symbolic skeleton → semantic equations → generic equations → DAG) so per-token entropy drops smoothly ~0.5 CE per layer. Each layer strips one variability axis. Gated by the 97-problem smoke training before committing $57 to the full regen.

18. **✅ JSD + Panama hats — validated diagnostic, not supervision (May 25)** — `diag_jsd_attention_boundaries.py` ran on v79 step 1000 with 50 problems. Result: **breaths attend to the SAME prompt regions** (max breath-side JSD = 0.07 vs log(2)=0.69 ceiling). Decode-side JSD ~0.26 (within-breath gaze shifts work). The waist-zero Δswap=+4.59 result tells us breaths produce different waist CONTENT, but attention pattern over prompt is shared. Implication: per-breath specialization is in Q-side transformation, not K/V reading. Memory `project_jsd_attention_diagnostic.md`.

19. **✅ IB clustering — natural codebook taxonomy (May 27)** — Hierarchical K-means on Pythia embeddings of L2 NL step descriptions yields a 32-leaf tree per OP (4 ops × ~8 sub-clusters), 1:1 aligned with `V78_HEAD_CODEBOOK_N=32`. Each leaf = semantic cluster (e.g. "Convert minutes to hours" → DIV.0.1, "Find half of N" → DIV.0.2.1). Replaces hand-picked op_role vocabularies. OP-constrained nearest-centroid assignment is deterministic. Tree at `.cache/ib_tree.json`. Critical lesson: data-driven taxonomy > intuited categories.

20. **✅ Same task per breath, coarse-to-fine (v8 v3 → v81)** — The 4 transformer layers are SHARED across K=7 breaths. If each breath has a different task, the 7-fold gradient pulls weights in different directions. v8 v3 (every breath outputs same DAG skeleton, one more piece filled in) showed CE spread 1.28 — tightest ever. v81 generalizes this to multi-list parallel supervision: ops, types_path, args1, args2 each fill progressively across breaths. The 4 looped layers learn ONE skill — coarse-to-fine refinement — at all 7 breaths. Memory: `project_v81_same_task_principle.md`.

21. **✅ FOUR-mask training requirement (May 27)** — Train/eval consistency requires all 4 of: kv_mask (WaistController cross-attn), notebook_pool_mask (notebook pool), main_attn_mask (main self-attn at answer-span KV), embed_mask (input embedding zeroing at answer-span). Missing ANY of them allows training-time leakage that produces degenerate AR output at eval. v80 era trained with only 2 of 4. v81+ trains with all 4. Audit script: `scripts/diag_v81_masking_audit.py`. Memory: `project_v81_four_mask_discovery.md`.

22. **🆕 Multi-head WaistController** — v81's 4 parallel output heads (ops/types/args1/args2) sharing the cross-attn backbone. Currently bias-only per-head MLP (full 2-layer MLP exceeded AMD JIT capacity; per-head CE only at final breath for the same reason). Performance comparable to v8 v3 in smoke (pb_ce 4.23 at L6 step 69). Full per-head MLP capacity recovery deferred to AMD JIT compile-budget research (e.g. per-breath graph decomposition like Sonnet's KV cache fused-JIT work).

---

### Cross-cutting issue surfaced 2026-05-17:

**+/- blindspot in cold-start ARITH.** Both v36 (50/51/49 final) and v38 (61/59/58 final) — independent architectures, independent training runs — converged to × ÷ specialists with NEAR-ZERO accuracy on + and -.

- v38 per-op extraction: {+ : 0, − : 1, × : 175, ÷ : 139} of ~200 each
- v36 per-op extraction: {+ : 2, − : 4, × : 175, ÷ : 118} of ~200 each
- Mean accuracy 50-60% on uniform ARITH masks total failure on half the ops.

**Root cause hypothesis:** multi-digit + and − require carry propagation (sequential dependency between digit predictions). The model finds a local minimum that handles × and ÷ patterns (less carry-dependent) and fails to learn carry chains.

**v24c does NOT have this blindspot** because L4_MIXED training had + and − embedded in multi-cycle word problems with explicit intermediate-result supervision — the model couldn't take the × ÷ shortcut.

**Strategic implication:** stop cold-starting from Pythia on bare ARITH. Build forward from v24c on richer distributions (L4.5, GSM8K).
