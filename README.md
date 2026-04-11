# Mycelium — Breathing Models

Differentiable recurrent reasoning for small language models. A frozen base LLM that can't chain reasoning internally learns to chain through external differentiable compression — thinking in a loop where each pass rewires its own attention via state-conditioned LoRA.

**Lead:** Bryce Roche · **Target:** MATH-500 · **Deadline:** May 22, 2026

---

## Proven Results

| Task | Base | With Breathing | Notes |
|---|---|---|---|
| Single-step arithmetic | 70% | **100%** | Llama 1B, 64-float state |
| Two-step arithmetic | 0% | **94.8%** | Page-based + target-cos contrastive |
| Three-step arithmetic | 0% | **83.4%** | Page-based + contrastive, warm-started |
| L2 word ops | 0.6% | **53.4%** | CoT targets + pass-conditioned hypernetwork |
| L3 named qty (single LoRA) | 18.8% | **88.6%** | CoT + warm start from L2 |
| L3 named qty (dual LoRA) | 18.8% | **96.0%** | Dual LoRA verification (+7.4 pts over single) |
| L4 two-step word problems | 40.8% | **100.0%** | Dual LoRA, warm from L3, 1 epoch |
| **GSM8K** | **2.2%** | **17.8%** | **Dual LoRA, 5 passes, curriculum L0→L4→GSM8K** |

Two-step 94.8% → 97.4% effective per-step (up from 70% base).
Three-step 83.4% → 93.8% effective per-step (cube root).
L2 word ops: 53.4% from 0.6% baseline — CoT targets were the breakthrough (12.2% with terse targets).
L3 dual LoRA: 96.0% vs 88.6% single LoRA — verification templates catch errors forward-only misses.
L4 two-step WP: 100.0% in 1 epoch — model generalizes from L3 to diverse two-step word problems.
GSM8K: 17.8% from 2.2% baseline (8.1x). Frozen 1B base model + 110M learned params. Blend ≈ 0.65 — model uses heavy verification on hard problems.

---

## Architecture (v20.1 — State-Conditioned LoRA)

```
state (64) ──┐
strategy(512)┴──→ HYPERNETWORK ──→ 256 LoRA scales
                                       │
                                       ▼
              [problem] → Llama 16L (frozen, additive LoRA on Q,K,V,O)
                              │ all-layer hidden states
                              ▼
                       7-LAYER PERCEIVER (~105M)
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            64-float state delta   512-float strategy
                    │
                    ▼
            normalize(state + delta) * √64    (loop ×3)
```

- **Llama 3.2 1B base** (frozen). Not instruct — instruct already chains in one shot.
- **64-float state** on the hypersphere — tight enough to force incremental thinking.
- **512-float strategy** side channel — ephemeral, feeds only the hypernetwork.
- **Additive LoRA** — no hooks, no weight modification: `q = W_q x + (x B^T) · scales · A^T`.
- **7-layer perceiver compressor** reads all 16 Llama layers with pass-conditioned attention.

---

## Next Direction (v21 — Page-Based State Accumulation)

Single overwriting state has amnesia — each pass partially erases the previous one through hypersphere rotation. The fix: **append, don't overwrite.**

```
Cycle 1: compress → 64-float page → append
Cycle 2: compress → 64-float page → append
Cycle 3: compress → 64-float page → append
                                                          ↓
Hypernetwork cross-attends over ALL pages → LoRA scales
```

- **No amnesia.** Page 1 is preserved exactly through pass N.
- **Variable-length thinking is free.** Cross-attention handles 2 pages or 8.
- **Frequency bands emerge naturally.** Each cycle encodes a different level of detail.
- **Free interpretability.** Attention weights show which past cycles drove each decision.
- **Hybrid path baked in.** Pages → pseudo-tokens for generation (LoRA off).

Per-pass bottleneck stays at 64 floats. ~+800K params total. See `plan/page_state_handoff.md`.

**Smoke test (April 2026):** Two-step arithmetic, warm-started from v20.1 — pages hit **86.2%** (vs 85.4% v20.1 baseline). No regression. Architecture preserved.

**But the pages don't encode anything.** A cosine-similarity diagnostic on the 86.2% checkpoint revealed that last pages are essentially constant across problems (same-answer cos sim 1.0000, diff-answer 0.9998, delta 0.0002; 28/64 dims dead). The entire breathing architecture collapsed into a learned static LoRA — one good configuration applied to every input. The 85.4%/86.2% came from the LoRA→generation path, not from per-problem thinking. That's why every readout head (log-mag, digit) failed on L0 arithmetic: there's nothing in the pages to read.

---

## Curriculum: L0 → GSM8K (PROVEN)

Complete stepping stones curriculum, each level warm-started from the previous:

```
L0: single-step arithmetic (70% → 100%)     ✓
L1: two-step arithmetic (0% → 94.8%)        ✓  target-cos contrastive
L2: word ops (0.6% → 53.4%)                 ✓  CoT targets breakthrough
L3: named quantities (18.8% → 96.0%)        ✓  dual LoRA verification (+7.4 pts)
L4: two-step word problems (40.8% → 100%)   ✓  1 epoch, instant generalization
GSM8K: (2.2% → 17.8%)                       ✓  8.1x, 5 passes, blend ≈ 0.65
```

Key findings across levels:
- **CoT targets** matching base model's natural style (L2: 12.2% → 53.4%)
- **Dual LoRA verification** helps most on hard/unseen problems (blend adapts to difficulty)
- **Curriculum warm-starting** enables each level to build on the previous
- **Easy problems don't need per-problem pages** (L4: page_cos=1.0 is correct behavior)
- **Hard problems use heavy verification** (GSM8K: blend ≈ 0.65 vs L4: blend ≈ 0.25)

---

## Next: Three Fixes for the GSM8K Ceiling (v22.3)

GSM8K plateaued at 17.8% — three root causes identified, three targeted fixes:

**1. Gradient scaling per cycle.** Early cycles get weak gradient (attenuated through later cycles). Fix: scale gradient inversely to distance from loss. `page = scale_gradient(page, num_passes - pass_num)`. One line, no architecture change.

**2. Fresh data every epoch.** 20K problems memorized by epoch 3 (ans_loss → 0.0000). Fix: procedurally generate new problems each epoch. For GSM8K: augment with number/name swaps.

**3. Fill the L4→L5 gap.** L4 (100%) → GSM8K (17.8%) is a cliff. Fix: intermediate levels.

```
L4:    2-step, [1-200]            → 100% ✓
L4.5:  2-step, [1-2000]           → ??? (bigger numbers)
L4.7:  3-step, [1-5000]           → ??? (more steps)
L4.9:  GSM8K easy (2-3 step)      → ??? (real formatting)
L5:    Full GSM8K                  → 17.8% → ???
```

See `plan/three_fixes_handoff.md` for implementation details.

---

## Dual LoRA Verification (v22 — PROVEN)

Two sets of LoRA templates blended by a learned sigmoid weight per pass:

- **Forward templates** — narrow, sequential attention for computation
- **Verify templates** — broad, relational attention for consistency checking
- **Blend weight** — learned sigmoid, starts ~0.15, climbs to ~0.30 over training

```
LoRA term = (1-blend)·q_forward + blend·q_verify
```

**Result: 96.0% on L3 (vs 88.6% single LoRA) — +7.4 points.** Verification is a generalization tool: it helps most on unseen problems, less needed on memorized ones. The blend trajectory shows the model discovering verification's value over training.

Key finding: the confidence head needs per-pass correctness training (not always target=1.0). Currently broken for dynamic stopping but the fixed-pass dual LoRA result is proven.

See `plan/dual_lora_verification.md`, `plan/morning_handoff.md`.

---

## Repo Layout

```
src/
  thinking_model.py            # main model
  all_layer_perceiver.py       # 7-layer perceiver, dual heads (state + strategy)
  state_conditioned_lora.py    # additive LoRA + hypernetwork
  pseudo_token_head.py         # soft-prompt head for hybrid generation
scripts/
  train_thinking.py            # arithmetic + GSM8K training
  train_gsm8k_hybrid.py        # LoRA thinking + pseudo-token generation
plan/
  page_state_handoff.md        # v21 architecture spec
checkpoints/
  three_step_best.pt           # 73.6% three-step (warm-start source)
```

---

## Architecture Evolution

```
v15  Text-based [EXPAND]/[COLLAPSE]    →  not differentiable, abandoned
v16  SmolLM2-135M latent bottleneck    →  80.4% two-step ✓
v17  Llama 3.2 1B engine swap          →  richer hidden states
v18  No text generation while thinking →  forward passes only
v19  64-float bottleneck + 7L perceiver
v20  State-conditioned LoRA            →  53% two-step
v20.1 Side channel + additive LoRA     →  85.4% two-step, 73.6% three-step ✓
v21  Page-based state accumulation     →  86.2% two-step ✓ (but pages are constant)
v21.2 Target-cosine contrastive        →  94.8% two-step, 83.4% three-step ✓ (but pages copy)
v21.3 Pass-conditioned hypernetwork     →  pages differentiate ✓ (p2v3=0.30)
v21.4 Stepping stones L2               →  53.4% word ops ✓ (CoT targets)
v21.5 Stepping stones L3               →  88.6% single LoRA ✓
v22  Dual LoRA (forward + verify)       →  96.0% L3 ✓ (+7.4 pts over single)
v22.1 L4 two-step word problems        →  100.0% ✓ (1 epoch, instant generalization)
v22.2 GSM8K dual LoRA                  →  17.8% ✓ (8.1x over 2.2% baseline, 5 passes)
v22.3 Three fixes (grad scale + fresh data + gap fill)  →  NEXT
```

See `CLAUDE.md` for full project context, known bugs, and training setup.
