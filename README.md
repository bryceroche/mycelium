# Mycelium — The Breathing Transformer

A 140M transformer that loops its own layers, breathing through each problem until it finds the resonant decomposition. Not a frozen LLM with external scaffolding — a model that breathes natively. Reasoning happens in representation space, not token space.

**The thesis:** decomposition is everything. The model doesn't become smarter — it learns to break hard problems into pieces it can already solve.

**Lead:** Bryce Roche · **Target:** MATH-500 · **Deadline:** September 1, 2026

---

## Results

### v1 — Llama 3.2 1B + LoRA Atoms (completed)

| Task | Base Llama | Breathing | Notes |
|------|-----------|-----------|-------|
| L3 (1-step arithmetic) | 18.8% | **100%** | 3 epochs from scratch |
| L4 (2-step word problems) | 6.0% | **99.5%** | Genuine 2-cycle decomposition |
| L4.5 (3-step word problems) | — | **99.5%** | All 3 cycles contribute |
| GSM8K | 2.2% | **22%** | Multi-step, beats 17.8% CoT ceiling |

### v4 Validation — Pythia-160M Looping (completed)

| Finding | Status | Key Number |
|---------|--------|------------|
| Representations improve with looping | **Proven** | Signal grows 7x across 8 loops |
| Per-problem diversity holds | **Proven** | Effective rank: 16.0 → 16.6 (stable) |
| SNR increases with loops | **Proven** | 0.114 → 0.127 (L0-3) |
| Copy Machine Principle | **Proven** | Mid-breath generation destroys signal |
| L0-3 best layer selection | **Proven** | Better stability than full 12-layer model |
| Generation needs fine-tuning | **Understood** | DC component overwhelms frozen gen head |

---

## How It Works

### The Breathing Loop (v4)

```
BREATHE in representation space (no tokens generated):
  4 Pythia layers × N loops
  π-cycled attention (12 heads, 12 phase angles)
  Sine-wave temperature (expand → compress → expand → ...)
  Controller reads hidden states between breaths

SPEAK once (tokens generated only at the end):
  Generate from the final, refined representation
  One copy from the original — not copies of copies
```

### Architecture

```
Component                        Params      Role
──────────────────────────────────────────────────────────────
Pythia-160M L0-3 (fine-tuned)     ~58M       Breathe (4 layers × N loops)
Controller                        ~80M       Observe + plan + stop
Differentiable Lookup Table         ~1M       Prime factorization of structure
──────────────────────────────────────────────────────────────
Total:                            ~140M
```

### Key Discoveries

- **The Copy Machine Principle:** Autoregressive generation between breaths degrades like photocopies — each token is lossy compression that compounds errors. Breathing in representation space avoids this entirely. Hidden states are the original painting; tokens are photocopies.
- **Equal-reward decomposition:** Each target worth 1/N reward. The ONLY incentive structure that produces genuine multi-step reasoning.
- **π-cycled attention:** Structural diversity immune to gradient collapse. Solves the diversity problem that killed v1-v3.
- **DC component insight:** 99.8% of hidden state variance is a shared direction across all inputs. Raw cosine similarity is misleading — always mean-subtract before measuring diversity.

---

## Current Status: v4 Build

The looping thesis is **empirically validated**. Representations get richer with every breath. The only gap is the generation head's calibration for looped representations — a precise, trainable objective.

```
v4 Validation:  ✓ Complete (Pythia looping experiments, May 6-7 2026)
Phase 0:        Loop consistency training          ← NEXT
Phase 1:        Learn to breathe (L3-L4.5)
Phase 2:        Controller + lookup table
Phase 3:        GSM8K push (>22%)
Phase 4:        MATH-500
```

**Platform:** Shadow Glass (AMD 7900 XTX 24GB, tinygrad + AM driver)

---

See `CLAUDE.md` for full technical context, `plan/pre_shadow_glass_summary.md` for the complete validation results and Day 1 plan, and `plan/mycelium_v4_final_architecture.md` (in Downloads) for the full architecture spec.
