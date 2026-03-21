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

## Next: 64-Atom LoRA (v24 — Anonymous Cognitive Atoms)

Replace named LoRA templates with 64 anonymous rank-6 atoms (~100M params),
independently scaled by a 10M-param hypernetwork. The model discovers its own
cognitive decomposition. No softmax, no mode collapse, no naming.

- 64 rank-6 LoRA atoms (each: A[d_model, 6], B[6, proj_dim] per layer per proj)
- AtomHypernetwork: 2-layer page attention + pass embed → 64 tanh scales
- Symmetric capacity: 105M compress (perceiver) ≈ 100M expand (atoms)
- Strategy removed — perceiver outputs page only
- Post-training analysis reveals what each atom learned

**Why atoms over named modes?** Quad LoRA (v23) tested at 92.5% L3 (vs 96% dual
baseline). The 4-way softmax caused mode collapse — COMPUTE dominated, killing
three modes with zero gradient. Entropy regularization prevented collapse but also
prevented differentiation. 64 anonymous atoms with tanh scaling avoid this entirely:
no competition, no collapse, no human prior on what the modes should be.

See `plan/atom_lora_handoff.md` for architecture spec.

## Fourier Pass Encoding (v24.1 — IMPLEMENTED)

Replace discrete pass embedding (`nn.Embedding`) with continuous Fourier features.
The hypernetwork learns atom scales as smooth waveforms across passes — atoms
activate/deactivate in rhythmic patterns rather than arbitrary jumps.

- **Smooth interpolation:** consecutive passes have related encodings
- **No max_passes limit:** works for any pass number
- **Periodicity as prior:** encourages rhythmic thinking patterns
- **Generalizes to unseen pass counts:** train with 5 passes, eval with 8

Deployed in GSM8K training: 13.3% at epoch 3 (from 3.0% baseline), climbing.

## Pi-Harmonic Page Encoding (v24.1b — IMPLEMENTED)

Add frequency identity to pages using pi-harmonic basis — the same math as DCT
(JPEG compression). Each of 64 dims is a sine/cosine pair at multiples of π.

```
dims 0-1:   1st harmonic of π (coarsest — problem type)
dims 30-31: 16th harmonic (mid — operations)
dims 62-63: 32nd harmonic (finest — exact values)
```

- **Pi-harmonic, not transformer-style:** `freqs = k * π / d` (orthogonal basis, not arbitrary 10000)
- **32 frequency pairs × 2 = 64 dims:** natural coarse-to-fine structure
- **Zero parameters:** fixed buffer, deterministic
- **Applied after hypersphere normalization:** content first, then structure

## Apéry-Weighted Wavelet Init

Wavelet level weights start with 1/k³ power law (coarse > fine). Total power
converges to Apéry's constant ζ(3) ≈ 1.202. Physically motivated prior — natural
signals have power that decays with frequency.

See `plan/fourier_pass_encoding.md` for full spec.

## Perceiver Skip Connection (v24.2 — DESIGNED)

Give the perceiver private memory across passes. Save mid-layer query states
(4096 floats after layer 4) and let future passes cross-attend to them.

- **Private notebook:** 4096 floats/pass (64x richer than public 64-float pages)
- **Second gradient path:** loss → skip_attn → earlier perceiver (shorter than page chain)
- **No bottleneck bypass:** hypernetwork only sees compressed pages
- **Minimal cost:** one cross-attention layer between perceiver layers 4 and 5

## Haar Wavelet Preprocessing (v24.3 — IMPLEMENTED)

Apply Haar wavelet transform to hidden states before the perceiver reads them.
2x compression + frequency structure, all on GPU.

```
Raw:      100 tokens × 2048 per layer → perceiver (must learn to compress)
Wavelet:  ~49 coefficients × 2048 per layer → perceiver (pre-compressed)
```

- **2x fewer perceiver input tokens** — 4x faster attention (quadratic)
- **Frequency structure** — coarsest coefficients first, detail later
- **Position-preserving** — unlike FFT, wavelet keeps spatial localization
- **Pure PyTorch** — stays on GPU, supports autograd, no pywt dependency
- **Learned level weights** — model decides importance of each resolution

Connection to full wave architecture:
```
wavelet input → perceiver → Fourier page → Fourier pass → atom expansion
```

First full wave run (epoch 1): gen=10.9% from 2.7% baseline (4x improvement).
See `plan/wavelet_preprocessing.md` for implementation details.

## Page Cache + Replay Buffer (v24.4 — IMPLEMENTED)

Cache graduated cycle pages to focus training compute on later cycles where
accuracy is lowest. Training gets faster as more cycles graduate.

```
Epoch 1:  graduated=0 → all 5 cycles every step → 5 forwards/step
Epoch 5:  graduated=2 → cache cycles 1-2       → avg 3.3 forwards/step
Epoch 12: graduated=4 → cache cycles 1-4       → avg 1.7 forwards/step (2.8x speedup)
```

Key mechanisms:
- **Per-step accuracy measurement** — track each cycle's contribution
- **Graduation threshold** — per-step acc > 90% or stable for 2 epochs
- **Replay buffer** — cache pages from multiple epochs, train later cycles
  against varying quality earlier thinking (robustness)
- **Probabilistic loading** — 70% from cache, 20% frontier-1, 10% full runs

Memory footprint: < 200 MB CPU-resident (7,473 GSM8K × 5 epochs × 64 floats).
Training accelerates as the model improves — virtuous cycle.

See `plan/page_cache_replay.md` for full design.

## Entropy Flow + Surprise Detection (v24.5 — NEXT)

Track entropy of thinking across cycles. Measure surprise (unexpected entropy
drops). Add smoothness signal to confidence head — "was my thinking smooth?"

```
Good thinking (smooth):  100% → 80% → 60% → 40% → 20% → stop (trust it)
Bad thinking (choppy):   100% → 95% → 30% → 80% → 15% → stop (suspicious)
```

Three components:
- **EntropyTracker** — page entropy, page deltas, atom entropy (0 params)
- **SurpriseDetector** — |actual - expected| / std, triggers MCMC retry
- **EntropyFlowConfidence** — GRU tracks page dynamics → (confidence, smoothness)

Decision quadrants:
| | Smooth | Choppy |
|---|---|---|
| High conf | STOP (trust) | SUSPICIOUS (retry) |
| Low conf | CONTINUE | RETRY (lost thread) |

Parameter cost: ~216K (GRU ~200K, heads ~16K). Negligible.

See `plan/entropy_flow_handoff.md` for full design.

---

## Three Fixes for the GSM8K Ceiling (v22.3 — BUILT)

GSM8K plateaued at 17.8% — three root causes identified, three targeted fixes:

**1. Gradient scaling per cycle.** Earlier cycles get amplified gradient (capped at 4x).

**2. Fresh data every epoch.** Procedural levels regenerate; GSM8K augments via number swaps.

**3. Fill the L4→L5 gap.** Intermediate levels L4.5 → L4.7 → L4.9 bridge to full GSM8K.

```
L4:    2-step, [1-200]            → 100% ✓
L4.5:  2-step, [1-2000]           → ??? (bigger numbers)
L4.7:  3-step, [1-5000]           → ??? (more steps)
L4.9:  GSM8K easy (2-3 step)      → ??? (real formatting)
L5:    Full GSM8K                  → 17.8% → ???
```

Infrastructure built: `scripts/train_three_fixes.py`, `scripts/datasets_L45_L47.py`, `scripts/datasets_L49_gsm8k.py`. Three fixes carry forward into the v24 atom architecture (grad scaling, fresh data, gap fill are orthogonal to the LoRA structure).

---

## Dual LoRA Verification (v22 — PROVEN)

Two sets of LoRA templates blended by a learned sigmoid weight per pass. The model naturally adapts verification intensity to problem difficulty (blend ~0.25 easy, ~0.65 GSM8K).

**Result: 96.0% on L3 (vs 88.6% single LoRA) — +7.4 points.** Evolved into 64-atom LoRA (above) via quad LoRA (v23, tested but superseded).

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
v22.3 Three fixes (grad scale + fresh data + gap fill)  →  BUILT
v23   Four-mode LoRA (parse + compute + verify + answer) →  blend stayed uniform, mode collapse risk
v24   64-Atom LoRA (anonymous atoms, tanh, no softmax)  →  93% L3, 91% L4 ✓
v24.1 Fourier pass encoding                            →  13.3% GSM8K epoch 3 (climbing)
v24.2 Perceiver skip connection                        →  DESIGNED
v24.3 Haar wavelet preprocessing                       →  10.9% GSM8K epoch 1 (4x baseline) ✓
v24.4 Page cache + replay buffer                       →  IMPLEMENTED
v24.5 Entropy flow + surprise + smoothness confidence  →  NEXT
```

See `CLAUDE.md` for full project context, known bugs, and training setup.
