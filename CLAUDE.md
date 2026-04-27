# CLAUDE.md — Mycelium Breathing Models

## Project Overview

Mycelium builds differentiable recurrent reasoning for small language models. A frozen base LLM that can't chain reasoning internally learns to chain through external compression — thinking in a loop where each pass rewires its own attention via state-conditioned LoRA.

**Lead:** Bryce (Manhattan Beach, CA) · **Target:** MATH-500 · **Deadline:** May 22, 2026
**Infrastructure:** AWS EC2 g5.xlarge (A10G 24GB), S3 bucket mycelium-data-v7

---

## Architecture (v25 — Per-Cycle Targets + Page Delta)

```
STATE PAGES (64 floats/pass, accumulated)
MESSAGES (16 floats/pass, direct from last layer)
     │
     ▼
ATOM HYPERNETWORK (10M params)
  2-layer cross-attention over pages + messages → 64 tanh scales
     │
     ▼
64 RANK-6 LORA ATOMS (~100M params)
  q = W_q x + Σᵢ (scaleᵢ · (x @ Bᵢᵀ) @ Aᵀᵢ)
     │
     ▼
LLAMA 3.2 1B BASE (frozen) + text-injected previous results
     │
     ▼
HAAR WAVELET PREPROCESS (2x compression, frequency structure)
     │
     ▼
7-LAYER PERCEIVER (~105M) → 64-float page → append to pages
     │
     ▼
ANSWER HEAD reads PAGE DELTA (page_n - page_{n-1}) for cycle 2+
```

**Key components:**
- **Llama 3.2 1B base** (frozen, NOT instruct — base can't chain, instruct already can)
- **64-float pages** on hypersphere — tight bottleneck forces incremental thinking
- **64 anonymous LoRA atoms** — model discovers its own cognitive decomposition
- **Per-cycle intermediate targets** — each cycle predicts ONE intermediate result
- **Page delta for answer head** — cycle 2+ reads (page_n - page_{n-1}), not raw page (prevents copying)
- **Text injection** — cycle 2+ gets "Step 1 result: 160\n" prepended as actual text tokens
- **Cycle message generator** — 16-float direct signal from last layer, bypasses perceiver
- **Hybrid loss** — gen loss powers learning, answer head shapes correctness (flipped weights per cycle)
- **Natural sentence gen targets** — full sentences with embedded computation per cycle
- **Tanh scaling** — no softmax competition, no mode collapse
- **Haar wavelet preprocessing** — 2x input compression, 4x faster attention
- **Pi-harmonic page encoding** — frequency identity per dimension (coarse→fine)

---

## Proven Results

| Task | Base | Breathing | Notes |
|------|------|-----------|-------|
| Single-step arithmetic | 70% | **100%** | Llama 1B, 64-float state |
| Two-step arithmetic | 0% | **94.8%** | Page-based + target-cos contrastive |
| Three-step arithmetic | 0% | **83.4%** | Warm-started, 93.8% effective per-step |
| L2 word ops | 0.6% | **53.4%** | CoT targets (12.2% with terse targets) |
| L3 named qty (dual LoRA) | 18.8% | **96.0%** | +7.4 pts over single |
| L3 named qty (atom) | 4.5% | **93.0%** | 64 atoms, warm perceiver |
| **L4 per-cycle (v25)** | — | **89.0%** | **Per-cycle targets + page delta, climbing** |
| L4 word problems (atom) | 57.0% | **91.0%** | 64 atoms, warm from L3 |
| **GSM8K (dual LoRA)** | **2.2%** | **17.8%** | **8.1x, curriculum L0→L4→GSM8K** |
| GSM8K (atom, ep3) | 3.0% | 13.3% | 64 atoms + Fourier, climbing |

**Key finding (v25):** Per-cycle targets + page delta = 89% on L4 2-step word problems (cycle 1: 94%, cycle 2: 89%). The page delta was the critical fix — without it, cycle 2 copies cycle 1's answer (60% of errors were exact copies).

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

### 1. Per-Cycle Curriculum L4.5 → L4.7 → L4.9 → GSM8K (v25 — IN PROGRESS)
L4 at 89% with per-cycle targets + page delta. Extend to 3-5 step problems.

```
L4.5: 3 steps, warm from L4, target >80%
L4.7: 4 steps, warm from L4.5, target >70%
L4.9: 5 steps, warm from L4.7, target >50%
GSM8K: variable steps, warm from L4.9, target >30%
```

Key settings: hybrid loss (gen=1.0/ah=0.5 for cycle 1, gen=0.1/ah=5.0 for cycle 2+), page delta for answer head, text injection for cycle 2+, natural sentence gen targets, lr_scale=0.7 when resuming.

### 2. GSM8K Per-Cycle Decomposition
Use Claude API to annotate 7,473 GSM8K problems with per-step intermediates. Script ready: `scripts/annotate_gsm8k_cycles.py`.

### 3. MATH-500 Benchmark (May 22 deadline)

---

## Design Decisions That Matter

- **Base model, not Instruct.** Instruct chains at 42% GSM8K. Base model: 0% chained, 70% single-step. Architecture provides the missing chaining.
- **64-float bottleneck.** Tight enough to force incremental thinking, loose enough for value + context + intent.
- **Hypersphere normalization.** `state = normalize(state + delta) * √64`. Constant magnitude, changing direction.
- **Additive LoRA.** `q = W_q @ x + (x @ B.T) * scales @ A.T`. No hooks, clean gradient flow.
- **Separate LRs.** Perceiver: 1e-4, LoRA templates: 1e-3 (10x). 4530x gradient imbalance compensation.
- **Llama frozen.** Unfreezing at 1e-6 caused 40% volatile cap. Frozen: stable 53%.
- **Random hypersphere init.** Implicit augmentation — model learns task, not trajectory. 85.4% vs 59.6% learned init.
- **Per-cycle targets, not CoT.** Each cycle predicts ONE intermediate. CoT makes cycles redundant.
- **Page delta for cycle 2+.** Answer head reads (page_n - page_{n-1}). Without this, cycle 2 copies cycle 1's answer (60% of errors). One-line fix: 5% → 89%.
- **Text injection.** Cycle 2+ gets previous answer as actual text tokens ("Step 1 result: 160\n"). Llama needs text, not continuous vectors.
- **Hybrid loss with flipped weights.** Cycle 1: gen=1.0, ah=0.5 (gen drives parsing). Cycle 2+: gen=0.1, ah=5.0 (answer head dominates correctness).
- **Natural sentence gen targets.** Full sentences with embedded computation, not bare numbers. Llama needs to BREATHE — expand in natural language, collapse through bottleneck.

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
| Cycle 2 copies cycle 1 without page delta | 60% of errors are exact copies — use delta |
| Answer head on raw page for cycle 2+ | Reads persisted cycle 1 info, not new computation |
| Final accuracy eval reading wrong page | Must read last SUPERVISED cycle's delta, not last pass |

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
v24.5 Entropy flow + smoothness          → parked
v25   Per-cycle intermediate targets      → 89% L4 ✓ (page delta breakthrough)
v25.1 Cycle message generator            → 16-float direct signal, bypasses perceiver
v25.2 Text injection                     → previous answer as text tokens for cycle 2+
v25.3 Page delta answer head             → read page_n - page_{n-1}, not raw page ✓
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
| Per-cycle answer head loss only | 0-1% | Gradient too weak (0.0002 to atoms) |
| Bare number gen targets ("97") | cycle 2 = 3-5% | LLM guesses plausible numbers |
| Equation gen targets ("160-63=97") | cycle 2 = 3-5% | LLM learns equation format, not computation |
| Cycle message (16 floats) | cycle 2 = 3-5% | Right info, wrong format (not text) |
| Raw page answer head for cycle 2 | cycle 2 = 3-5% | Copies cycle 1's answer (60% exact copies) |

---

## File Structure

```
scripts/
  atom_lora.py                   # Model: AtomLoRAModel, AnswerHead, CycleMessageGenerator, AtomHypernetwork
  train_per_cycle.py             # v25 training: per-cycle targets + page delta + text injection
  generate_per_cycle_data.py     # Data gen: L3-L4.9 with cycle_targets + cycle_gen_targets
  annotate_gsm8k_cycles.py       # Claude API: decompose GSM8K into per-step intermediates
  diag_cycle2.py                 # Diagnostic: print correct/wrong cycle 2 predictions
  train_atom_lora.py             # v24 training (legacy, CoT-based)
plan/
  per_cycle_targets_handoff.md   # v25 design: per-cycle intermediate targets
  cycle_message_handoff.md       # v25.1 design: 16-float message channel
  fourier_pass_encoding.md       # v24.1 Fourier + pi-harmonic design
  wavelet_preprocessing.md       # v24.3 Haar wavelet
data/
  per_cycle/                     # Generated JSONL: L3-L4.9 train/eval with cycle_targets
```
