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

## 3. Empirical Status (as of 2026-05-11)

- **L3-spaced training:** 8-loop accuracy 70% vs 1-loop 65% (more thinking helps). Loss gap closed 73% (0.77 → 0.20 nats). The 65% ceiling is arithmetic precision (4-layer model limit), not a breathing limitation.
- **L4-spaced:** 10% with rotation + integration only. The full 7/7 loop is now ready for L4 — this is where the controller has the most to contribute (deciding *which step's operation comes next*).
- **Looping signal survives:** centered cross-problem cosine -0.05 (orthogonal) through 4 loops, effective rank 15-16, signal norm grows 3.9 → 6.4 across 8 loops.
- **Lookup table validated:** trained 16×1024 cosine table hits 100% op classification on standalone test.
- **Digit-by-digit generation:** "1 7 0 - 1 3 2 = 3 8" jumped 71% → 87.5% on peek samples (BPE single-token "170" forces memorization; per-digit forces computation).
- **Inference engine:** JIT-fused KV cache, 42.8× speedup (268s → 6.3s for N=100, LOOPS=8), bit-for-bit identical outputs. Compile once, replay forever. Eval is no longer a bottleneck.

---

## 4. Specifications (Tight)

- **Init:** Pythia-410M L0-3. Take attn+FFN weights + embeddings (50304×1024, tied with output). Phase-specific copies of Q, K, FFN gate (one per breath phase). Shared V, O, FFN up/down, norms.
- **Dimensions:** h=1024, 16 heads × head_dim 64, FFN 4096, vocab 50304, max seq 512, 4 layers/breath, max 8 loops.
- **Parameters:** 35.7M transformer processing + 40M controller + 51.5M embeddings = ~127M total. With 8 loops → ~286M effective processing capacity.
- **Memory:** ~5GB mixed precision, ~19GB headroom. KV cache sized to actual seq length (not model max): cache_max_len=32 for L3-spaced → ~2GB at B=100.
- **Platform:** AMD 7900 XTX, tinygrad, AM driver (in progress — see §6), Ubuntu 24.04. No ROCm, no CUDA, no PyTorch.

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

- **AM driver bringup (working as of 2026-05-11):** `DEV='PCI+AMD'` smoke test passes after disabling Secure Boot (lockdown was the EPERM root cause) and adding `vm.compact_unevictable_allowed=0`. `scripts/setup_am_driver.sh` now installs both fixes plus a corrected unbind-service ordering (polls for the driver symlink instead of racing against udev). Re-run the script once to activate the updated unit + sysctl. We were on KFD/ROCm before this; AM is the spec's stated stack. See `memory/project_am_driver_state.md`.
- **Full 7/7 closed loop ready for empirical validation on L4-spaced** — first run with all components active.

---

## 7. What We Carry Forward (and What We Left Behind)

**Forward:** expand-collapse breathing pattern, π-cycled attention, equal-reward decomposition, number augmentation, gradient separation, differentiable lookup table (from project origins), the copy machine principle, the JIT-fused KV cache.

**Left behind:** Llama 1B (replaced by Pythia-410M L0-3), LoRA atoms and continuous scales, the straight-through gradient estimator, soft token diversity mechanisms, PyTorch/ROCm, Windows.

A 127M model that breathes, alternates, integrates, and factorizes. Four months to September 1.
