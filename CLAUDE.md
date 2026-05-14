# Mycelium v4: The Breathing Transformer — Agent Brief

**Author:** Bryce + Claude · **Deadline:** Sep 1, 2026 · **Target:** MATH-500
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

## 3. Empirical Status (as of 2026-05-13)

**Best ckpt for pure L4: L4_MIXED v1 step 1500 = 66 / 67 / 65 (A=1 / A=4 / A=8).**

Trajectory:
- **L3-spaced:** 70% at A=8 vs 65% at A=1 (depth helps). 65% is the 4-layer arithmetic ceiling.
- **L4 v4 (May 12):** 43% pure L4 from arith_mixed_v6 warm-start. Step 1500 was the headline.
- **L4_BORROW v1 overnight (May 13):** 80% on L4_BORROW eval (cascade-heavy), but only 32% on pure L4 — narrow-curriculum trade-off. Catastrophic forgetting between L4_BORROW and standard L4.
- **L4_MIXED v1 (May 13):** broadest distribution (6 standard + 3 cascade variants). Step 1500 = **66/67/65** on L4_MIXED eval. +20 vs v4 baseline. Best ckpt to date.
- **L4_MIXED v2 / `ROTATION_PERIOD=4` (May 13):** closed-cycle RoPE (period 4, 50% per-breath overlap). 68/64/60 at step 1500 — partial LOSE. Depth got *worse* (A=8 −5 vs v1). Closure alone isn't sufficient without per-breath training pressure for verification.
- **L4_MIXED v3/v4 calibration (May 13):** added `ConfidenceHead` + BCE(conf, argmax-correct) per breath. v3 (single-cycle encoding): 1/1/1 at step 250 — catastrophic forgetting from encoding mismatch. v4 (multi-cycle encoding, digit-only mask): 0/1/0 — same catastrophic forgetting because non-digit positions got no training signal. v5 (multi-cycle + full target mask): in progress.

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

- **Calibration training (L4_MIXED v5):** `calibration_train_step` adds a `ConfidenceHead` (1024→256→1 MLP, sigmoid) at each step's "=" position. Per-breath BCE(conf, argmax-correct) supervises the head to predict its own correctness — the verification objective the probe found was missing from frozen reps. Multi-cycle encoding (matches eval distribution), full target mask (every target token supervised, fixing v4's digit-only failure). `CALIBRATION_MODE=1` env var routes the training step. Currently running from L4_MIXED v1 step 1500 warm-start. Decision point at step 250 eval.
- **`ROTATION_PERIOD` env var:** closed-cycle RoPE rotation. Default 0 preserves existing behavior (`loop_phase = l * π/max_loops`); `=N` switches to `l * 2π/N` (period N breaths, full cycle returns to start). v2 with `=4` showed depth hurt — closure alone isn't sufficient. Available for future experiments once calibration baseline lands.
- **AM driver (working since 2026-05-11):** `DEV='PCI+AMD'` works after Secure Boot off + `vm.compact_unevictable_allowed=0`. `scripts/setup_am_driver.sh` installs both. See `memory/project_am_driver_state.md`.

---

## 7. What We Carry Forward (and What We Left Behind)

**Forward:** expand-collapse breathing pattern, π-cycled attention, equal-reward decomposition, number augmentation, gradient separation, differentiable lookup table (from project origins), the copy machine principle, the JIT-fused KV cache.

**Left behind:** Llama 1B (replaced by Pythia-410M L0-3), LoRA atoms and continuous scales, the straight-through gradient estimator, soft token diversity mechanisms, PyTorch/ROCm, Windows.

A 127M model that breathes, alternates, integrates, and factorizes. Four months to September 1.
