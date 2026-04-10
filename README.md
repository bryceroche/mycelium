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
| GSM8K (hybrid, epoch 1) | 6.2% | 6.6% | Initial result, training in progress |

Two-step 94.8% → 97.4% effective per-step (up from 70% base).
Three-step 83.4% → 93.8% effective per-step (cube root).
L2 word ops: 53.4% from 0.6% baseline — CoT targets were the breakthrough (12.2% with terse targets).
L3 dual LoRA: 96.0% vs 88.6% single LoRA — verification templates catch errors forward-only misses.

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

## Current Direction (v21.4 → L3 + Dual LoRA Verification)

Two proven failure modes in page-based breathing, both solved:

1. **Fixed-point collapse** — pages constant across problems. Fix: target-cosine contrastive loss (self-stabilizing at cos=0.7). → 94.8% two-step, 83.4% three-step.
2. **Page copying** — pages 2-3 identical within each problem. Fix: pass-conditioned hypernetwork (pass embedding breaks the circular copy loop). → p2v3 dropped from 1.000 to 0.30.

Key finding: three-step arithmetic doesn't need multi-pass thinking. The model gets 83.4% with one effective pass. Multi-pass is the right architecture for HARDER problems where different passes need different cognitive operations.

**L2 word ops (53.4%):** CoT targets matching the base model's natural style were the breakthrough fix. Terse answer targets ("143") caused number-spam; CoT targets ("the square of 8 = 64. 64 plus 79 = 143. The answer is 143.") jumped accuracy from 12.2% to 53.4%.

**Next: L3 named quantities + dual LoRA verification.**

```
L2: "half of 48 plus 48"                   → 53.4% ✓ (CoT targets)
L3: "Jamie had 56 cookies and gave 2 away" → NEXT (named quantities, warm from L2)
L4: 2-step word problems, small numbers     → easy GSM8K style (4-6 passes)
L5: Full GSM8K                             → complex multi-step (6-12 passes)
```

See `plan/morning_handoff.md` for implementation order.

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
v22.1 L4 two-step word problems        →  NEXT
```

See `CLAUDE.md` for full project context, known bugs, and training setup.
