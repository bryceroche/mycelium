# Mycelium — Breathing Models

Differentiable recurrent reasoning for small language models. A frozen base LLM that can't chain reasoning internally learns to chain through external differentiable compression — thinking in a loop where each pass rewires its own attention via state-conditioned LoRA.

**Lead:** Bryce Roche · **Target:** MATH-500 · **Deadline:** May 22, 2026

---

## Proven Results

| Task | Base | With Breathing | Notes |
|---|---|---|---|
| Single-step arithmetic | 70% | **100%** | Llama 1B, 64-float state |
| Two-step arithmetic | 0% | **85.4% ± 0.0%** | 5 seeds, side channel + additive LoRA |
| Three-step arithmetic | 0% | **73.6% ± 0.0%** | 5 seeds, warm-started from two-step |
| GSM8K (hybrid, epoch 1) | 6.2% | 6.6% | Initial result, training in progress |

Two-step 85.4% → 92.4% effective per-step (up from 70% base).
Three-step 73.6% → 90.1% effective per-step (cube root).

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

---

## After GSM8K (v22 — Dual LoRA Verification Mirror)

Two sets of LoRA templates blended by a learned sigmoid weight:

- **Forward templates** — narrow, sequential attention for computation
- **Verify templates** — broad, relational attention for consistency checking
- **Blend weight** — smooth sigmoid trajectory, early cycles compute, later cycles verify

```
LoRA term = (1-blend)·q_forward + blend·q_verify
```

The model rotates from building an answer to checking it — the geometric mirror of computation on the same hypersphere. Re-enables the confidence head with a correctness signal: easy problems verify in 2 cycles, hard ones in 8. Adds ~1.1M params. See `plan/dual_lora_verification.md`.

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
v21  Page-based state accumulation     →  86.2% two-step ✓ (smoke test)
v22  Dual LoRA (forward + verify mirror) →  PLANNED (post-GSM8K)
```

See `CLAUDE.md` for full project context, known bugs, and training setup.
