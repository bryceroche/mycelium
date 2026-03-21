# CLAUDE.md — Mycelium Breathing Models

## Project Overview

Mycelium builds differentiable recurrent reasoning for small language models. A frozen base LLM that can't chain reasoning internally learns to chain through external compression — thinking in a loop where each pass rewires its own attention via state-conditioned LoRA.

**Lead:** Bryce (Manhattan Beach, CA) · **Target:** MATH-500 · **Deadline:** May 22, 2026
**Infrastructure:** AWS EC2 g5.xlarge (A10G 24GB), S3 bucket mycelium-data-v7

---

## Architecture (v24 — 64-Atom LoRA)

```
STATE PAGES (64 floats/pass, accumulated)
     │
     ▼
ATOM HYPERNETWORK (10M params)
  2-layer cross-attention over pages + Fourier pass encoding → 64 tanh scales
     │
     ▼
64 RANK-6 LORA ATOMS (~100M params)
  q = W_q x + Σᵢ (scaleᵢ · (x @ Bᵢᵀ) @ Aᵀᵢ)
     │
     ▼
LLAMA 3.2 1B BASE (frozen) → all-layer hidden states
     │
     ▼
HAAR WAVELET PREPROCESS (2x compression, frequency structure)
     │
     ▼
7-LAYER PERCEIVER (~105M) → 64-float page → append to pages
```

**Key components:**
- **Llama 3.2 1B base** (frozen, NOT instruct — base can't chain, instruct already can)
- **64-float pages** on hypersphere — tight bottleneck forces incremental thinking
- **64 anonymous LoRA atoms** — model discovers its own cognitive decomposition
- **Tanh scaling** — no softmax competition, no mode collapse
- **Haar wavelet preprocessing** — 2x input compression, 4x faster attention
- **Fourier pass encoding** — smooth rhythmic atom activation patterns
- **Pi-harmonic page encoding** — frequency identity per dimension (coarse→fine)
- **Apéry-weighted wavelet init** — 1/k³ power law (ζ(3) ≈ 1.202)

---

## Proven Results

| Task | Base | Breathing | Notes |
|------|------|-----------|-------|
| Single-step arithmetic | 70% | **100%** | Llama 1B, 64-float state |
| Two-step arithmetic | 0% | **94.8%** | Page-based + target-cos contrastive |
| Three-step arithmetic | 0% | **83.4%** | Warm-started, 93.8% effective per-step |
| L2 word ops | 0.6% | **53.4%** | CoT targets (12.2% with terse targets) |
| L3 named qty (single) | 18.8% | **88.6%** | CoT + warm from L2 |
| L3 named qty (dual LoRA) | 18.8% | **96.0%** | +7.4 pts over single |
| L3 named qty (atom) | 4.5% | **93.0%** | 64 atoms, warm perceiver |
| L4 two-step WP | 40.8% | **100.0%** | 1 epoch, instant generalization |
| L4 word problems (atom) | 57.0% | **91.0%** | 64 atoms, warm from L3 |
| **GSM8K (dual LoRA)** | **2.2%** | **17.8%** | **8.1x, curriculum L0→L4→GSM8K** |
| GSM8K (atom, ep3) | 3.0% | 13.3% | 64 atoms + Fourier, climbing |

**Key finding:** 85.4% two-step with zero variance across 5 seeds (92.4% effective per-step).

---

## Curriculum: L0 → GSM8K

```
L0: single-step arithmetic (70% → 100%)     ✓
L1: two-step arithmetic (0% → 94.8%)        ✓  target-cos contrastive
L2: word ops (0.6% → 53.4%)                 ✓  CoT targets breakthrough
L3: named quantities (18.8% → 96.0%)        ✓  dual LoRA verification
L4: two-step word problems (40.8% → 100%)   ✓  instant generalization
GSM8K: (2.2% → 17.8%)                       ✓  8.1x improvement
```

---

## What to Do Next (Priority)

### 1. Entropy Flow + Surprise Detection (v24.5 — NEXT)
Track entropy across cycles. Good thinking = steady reduction (smooth current flow). Bad thinking = choppy jumps (sparks). GRU-based confidence head outputs (confidence, smoothness). ~216K params.

### 2. Three Fixes for GSM8K Ceiling (v22.3 — BUILT)
- Gradient scaling per cycle (earlier = amplified, capped 4x)
- Fresh data every epoch (procedural regen + GSM8K number-swap augmentation)
- Fill L4→L5 gap (L4.5/L4.7/L4.9 intermediate levels)

### 3. Perceiver Skip Connection (v24.2 — DESIGNED)
Private 4096-float memory across passes (mid-layer query states). Second gradient path to earlier cycles.

### 4. MATH-500 Benchmark (May 22 deadline)

---

## Design Decisions That Matter

- **Base model, not Instruct.** Instruct chains at 42% GSM8K. Base model: 0% chained, 70% single-step. Architecture provides the missing chaining.
- **64-float bottleneck.** Tight enough to force incremental thinking, loose enough for value + context + intent.
- **Hypersphere normalization.** `state = normalize(state + delta) * √64`. Constant magnitude, changing direction.
- **Additive LoRA.** `q = W_q @ x + (x @ B.T) * scales @ A.T`. No hooks, clean gradient flow.
- **Separate LRs.** Perceiver: 1e-4, LoRA templates: 1e-3 (10x). 4530x gradient imbalance compensation.
- **Llama frozen.** Unfreezing at 1e-6 caused 40% volatile cap. Frozen: stable 53%.
- **Random hypersphere init.** Implicit augmentation — model learns task, not trajectory. 85.4% vs 59.6% learned init.
- **CoT targets.** Match base model's natural completion style. Terse targets cause number-spam.

---

## Known Bugs & Gotchas

| Bug | Impact |
|-----|--------|
| DataCollatorForLanguageModeling overwrites label masking | Training fails silently |
| Temperature > 0 at eval | Massive baseline shift |
| Small eval sets (N<100) | Misleading accuracy |
| GQA K,V dimensions | (4, 512) not (4, 2048) — 8 KV heads |
| Answer extraction edge cases | Strip periods, check \boxed{}, ####, last-number fallback |
| Probe-only training | Caps at 2.8% — need answer loss too |
| Learned initial state | Overfits to one trajectory (59.6% vs 85.4%) |
| Bias injection to embeddings | Corrupts pretrained representations, mode collapse |
| Softmax mode collapse | One mode dominates, kills others with zero gradient |
| GSM8K overfitting | Answer loss 0.39, accuracy plateaus — need fresh data |
| L4→L5 cliff | Too large for curriculum alone — need intermediate levels |

---

## AWS Setup

```bash
# SSH (IP changes — check AWS console)
ssh -i ~/.ssh/mycelium-key.pem ubuntu@<IP>

# Always use tmux
tmux new -s train
python scripts/train_thinking.py --config configs/thinking_gsm8k.yaml

# Monitor
tail -f logs/train_*.log
```

Instance: g5.xlarge (A10G 24GB, ~$1/hr)

---

## Architecture Evolution

```
v15   Text [EXPAND]/[COLLAPSE]           → not differentiable, abandoned
v16   SmolLM2-135M latent bottleneck     → 80.4% two-step ✓
v17   Llama 3.2 1B engine                → richer hidden states
v19   64-float bottleneck + perceiver
v20   State-conditioned LoRA             → 53% two-step
v20.1 Side channel + additive LoRA       → 85.4% two-step, 73.6% three-step ✓
v21   Page-based accumulation            → 86.2% (but pages constant)
v21.2 Target-cosine contrastive          → 94.8% two-step, 83.4% three-step ✓
v21.3 Pass-conditioned hypernetwork      → pages differentiate ✓
v21.4 L2 word ops                        → 53.4% ✓ (CoT breakthrough)
v22   Dual LoRA verification             → 96.0% L3 ✓ (+7.4 pts)
v22.2 GSM8K curriculum                   → 17.8% ✓ (8.1x)
v23   Four-mode LoRA                     → mode collapse, superseded
v24   64-Atom LoRA                       → 93% L3, 91% L4 ✓
v24.1 Fourier pass encoding              → 13.3% GSM8K ep3, climbing
v24.1b Pi-harmonic page encoding         → frequency identity per dim
v24.3 Haar wavelet preprocessing         → 2x compression, 4x attention speedup ✓
v24.4 Page cache + replay buffer         → up to 2.8x training speedup
v24.5 Entropy flow + smoothness          → NEXT
```

---

## Failed Experiments (Don't Repeat)

| Experiment | Result | Why |
|------------|--------|-----|
| Text [EXPAND]/[COLLAPSE] | Matched vanilla | Not differentiable |
| SmolLM2-135M on word problems | 0% | Can't parse "half as many" |
| Llama instruct with breathing | 11% (vs 42% base) | Instruct already chains |
| Bias injection to embeddings | Mode collapse | Corrupts representations |
| Unfreezing Llama at 1e-6 | 40% volatile | Destabilizes training |
| Efficiency penalty on passes | 1-pass collapse | Model games it |
| Probe-only training | 2.8% | Need answer loss |
| Learned initial state | 59.6% | Random init better (85.4%) |
| Margin contrastive | 91.6% or 68.8% | Batch-order dependent |
| SupCon temperature 0.1-0.3 | 49-67% | Overshoots (page_cos→0.02) |
| Terse answer targets | 12.2% | Number-spam from spillover |
| Quad LoRA 4-way softmax | 92.5% (vs 96%) | Mode collapse |

---

## File Structure

```
src/
  thinking_model.py              # Main model
  all_layer_perceiver.py         # 7-layer perceiver
  state_conditioned_lora.py      # Additive LoRA + hypernetwork
scripts/
  train_thinking.py              # Main training
  train_three_fixes.py           # GSM8K ceiling fixes
plan/
  fourier_pass_encoding.md       # v24.1 Fourier + pi-harmonic design
  entropy_flow_handoff.md        # v24.5 smoothness confidence
  wavelet_preprocessing.md       # v24.3 Haar wavelet
  page_cache_replay.md           # v24.4 training acceleration
```
