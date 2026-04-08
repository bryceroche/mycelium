# CLAUDE.md — Mycelium Breathing Models

## Project Overview

Mycelium is a multi-month research project building differentiable recurrent reasoning architectures for small language models. The core idea: a small model that can't chain reasoning internally learns to chain through external differentiable compression. The model thinks in a loop — each pass processes the problem, compresses its understanding into a tight state vector, and uses that state to think DIFFERENTLY on the next pass through state-conditioned LoRA.

**Lead researcher:** Bryce (Manhattan Beach, CA)
**Target benchmark:** MATH-500 (deadline: May 22, 2026)
**Infrastructure:** AWS EC2 g5.xlarge (A10G 24GB), S3 bucket mycelium-data-v7

---

## Current Architecture (v20 — State-Conditioned LoRA)

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  STATE (64 floats on hypersphere)                                │
│  STRATEGY (512 floats, ephemeral)                                │
│     │                                                            │
│     ├──→ HYPERNETWORK (576 → 512 → 256 LoRA scales)              │
│     │         │                                                  │
│     │         ▼                                                  │
│     │    LoRA applied as additive term (no hooks):               │
│     │    q = W_q @ x + (A @ diag(scales) @ B) @ x               │
│     │    Applied to Q,K,V,O at all 16 layers                    │
│     │         │                                                  │
│     │         ▼                                                  │
│     │    [problem tokens] → Llama 1-16 (WITH LoRA)               │
│     │         │          │         │              │              │
│     │       (all 16 layer hidden states saved)                   │
│     │         │          │         │              │              │
│     │         ▼          ▼         ▼              ▼              │
│     │    7-LAYER PERCEIVER COMPRESSOR (105M params)               │
│     │    reads ALL layers, pass-conditioned attention             │
│     │         │                    │                             │
│     │         ▼                    ▼                             │
│     │    64-float state delta    512-float strategy               │
│     │         │                    │                             │
│     │         ▼                    └──→ feeds hypernetwork        │
│     └──→ state = normalize(state + delta) * √64                  │
│               │                                                  │
│               ├──→ SymPy Probe → per-pass gradient               │
│               │                                                  │
│               ├──→ ConfidenceHead → ready? (disabled, fixed 3)   │
│               │         │                                        │
│               │     YES ▼                                        │
│               │    GENERATE ANSWER                               │
│               │                                                  │
│               └──→ NO: loop back                                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Params | Description |
|-----------|--------|-------------|
| Llama 3.2 1B Base | 1.23B | `unsloth/Llama-3.2-1B` (BASE, not instruct). Frozen. Processes problem with LoRA-modified attention. |
| 7-Layer Perceiver | ~105M | d_perceiver=1024, 4 queries, reads ALL 16 Llama layers via pass-conditioned layer gate. Outputs state delta (64 floats) AND strategy (512 floats). |
| State-Conditioned LoRA | ~1.1M | Learned A/B templates (rank 4) for Q,K,V,O at all 16 layers. Hypernetwork takes state+strategy (576 floats) → 256 LoRA scales. Templates modulated by scales. **K,V templates are (4, 512) not (4, 2048) due to Llama's GQA (8 KV heads).** |
| SymPy Probe | ~65 | Linear(64, 1). Reads state, predicts intermediate value. SymPy precomputes targets. Per-pass gradient signal. |
| ConfidenceHead | ~2.1K | Currently disabled. Passes fixed at 3. Will re-enable after accuracy climbs. |

### Key Design Decisions

- **Base model, not Instruct.** Instruct already chains reasoning in one shot (42% GSM8K). Base model CANNOT chain (0% on chained arithmetic) but knows individual operations (70%). The architecture provides the chaining the base model lacks.
- **64-float bottleneck.** Tight enough to force incremental thinking (can't encode full solution), loose enough for value + context + intent per pass.
- **512-float strategy (side channel).** Ephemeral, doesn't accumulate, feeds only into hypernetwork. Can't bypass bottleneck. Gives hypernetwork 9x more information than state alone.
- **Hypersphere normalization.** `state = normalize(state + delta) * √64`. Constant magnitude, changing direction. Each pass is a rotation. No explosion/collapse.
- **LoRA as additive term.** `q = W_q @ x + (x @ B.T) * scales @ A.T`. No hooks, no weight modification. Clean, differentiable, fast.
- **Separate learning rates.** Perceiver: 1e-4, LoRA templates: 1e-3 (10x), Hypernetwork: 1e-3 (10x). There's a 4530x gradient imbalance between perceiver and templates — the higher template LR compensates.
- **Transformer stays FROZEN.** Unfreezing at 1e-6 caused instability (40% cap, volatile). Frozen: stable 53%.

---

## Proven Results

```
Task                        Baseline    With Breathing    Architecture
─────────────────────────────────────────────────────────────────────
Single-step arithmetic      70%         100%              Llama 1B, 64-float, LoRA
Two-step arithmetic         0%          85.4%  ← BEST    Llama 1B, side channel, additive LoRA, answer loss
  (robust eval)                         ±0.0% (5 seeds)  (92.4% effective per-step)
Two-step arithmetic         0%          53%               Llama 1B, 64-float, LoRA (probe only, no answer loss)
  (theoretical ceiling)                 (49%)             (53% > 49% → LoRA improves per-step)
Three-step arithmetic       0%          73.6% ← BEST     Llama 1B, side channel, additive LoRA, warm start from 85.4%
  (robust eval)                         ±0.0% (5 seeds)  (90.1% effective per-step, cube root)
Three-step arithmetic       0%          52%               SmolLM2-135M, 64-float, pseudo-tokens
Two-step arithmetic         0%          83%               Llama 1B, 512-float, pseudo-tokens (earlier arch)
Two-step arithmetic         0%          80.4%             SmolLM2-135M, 32-float, pseudo-tokens
```

Key finding: 85.4% on two-step with zero variance across 5 seeds. Effective per-step
accuracy: 92.4% (up from 70% base). The three changes that broke the 53% ceiling:
1. Answer loss (was missing entirely — probe alone capped at 2.8%)
2. Side channel (576 floats feeding hypernetwork instead of 64)
3. Additive LoRA (clean, fast, fully differentiable, post-LoRA hidden states)

---

## What's Being Built Right Now

### Page-Based State Accumulation (NEXT — v21)

Single overwriting state has amnesia: each thinking pass partially erases the
previous one through hypersphere rotation. Page-based architecture: each pass
appends a 64-float "page" to a notebook, nothing is overwritten, hypernetwork
attends over all pages with cross-attention to generate LoRA scales.

```
state_pages = []
for pass_num in range(max_passes):
    if state_pages:
        lora_scales, attn = page_hypernetwork(state_pages, strategy)
    else:
        lora_scales = zeros(256)
    apply_lora(lora_scales)
    hs = llama(problem, output_hidden_states=True).hidden_states[1:]
    remove_lora()
    page, strategy = perceiver(hs, pass_num)
    page = normalize(page) * sqrt(64)  # per-page normalization
    state_pages.append(page)            # APPEND, not overwrite
# generation: pages → pseudo_tokens via cross-attention, LoRA OFF
```

What this fixes:
- **No amnesia.** Page 1 is preserved exactly through pass N. Hypernet's
  cross-attention retrieves it directly. No rotation dilution.
- **Variable-length thinking is free.** Easy: 2 pages (128 floats). Hard: 8
  pages (512 floats). Attention handles any sequence length.
- **Frequency bands emerge naturally.** Each pass sees different LoRA-modified
  hidden states, so each page captures a different level of detail.
- **Free interpretability.** Attention weights over pages = which thinking
  cycles mattered for each subsequent decision.
- **Hybrid path baked in.** Pages → pseudo-tokens for generation (LoRA OFF),
  pages → LoRA for thinking. Best of both architectures.

Per-pass bottleneck stays at 64 floats. Still tight. Still incremental. But
nothing is lost. Plan doc: `plan/page_state_handoff.md`.

**Status:** Architecture defined. Implementation next.

### GSM8K Hybrid (LoRA thinking + pseudo-token generation) — INITIAL RESULT

The arithmetic-trained LoRA destroys language generation on GSM8K — it
collapses into "Answer 1600\nAnswer 1600..." spam, learned from being trained
to "emit a number, immediately, then stop." Diagnosis confirmed by inspection.

Fix: hybrid path. Thinking passes use LoRA (powerful attention rewiring).
Generation uses pseudo-tokens prepended to embeddings, LoRA OFF (gentle state
injection that doesn't override the language head). Trained from scratch on
GSM8K (not warm-started from arithmetic) on full reasoning traces.

- Real verified baseline (3-shot, no thinking, 500 problems): **6.2%**
- Hybrid epoch 1: **6.6%** (+0.4 points, beats baseline by ~0.4 σ — needs more)
- Old LoRA-only path: 5.0% (worse than baseline)

**Status:** Epoch 1 promising, need more epochs and seeds for confidence.
Checkpoint: `gsm8k_hybrid_best.pt`. Page-based architecture (above) is the
next major step.

### Three-Step Arithmetic — PROVEN

Two-step achieved 85.4% (92.4% effective per-step). Three-step warm-started from two-step checkpoint.
- 73.6% ± 0.0% across 5 seeds. 90.1% effective per-step (cube root).
- Previous best: 52% (SmolLM2). Blew past it by epoch 2.
- Surpassed theoretical ceiling of 78.9% (92.4%^3) in effective per-step terms.

**Status:** Proven. Checkpoint: `three_step_best.pt` (epoch 7, 74.2% peak → 73.6% robust).

### Side Channel (state + strategy) — PROVEN

The previous 53% used only 64-float state to drive LoRA. The hypernetwork was starving — 64 floats → 256 scales meant ~0.25 floats per scale. The compressor now outputs TWO signals:

```
STATE (64 floats):    Content (numbers, values). Accumulates on hypersphere.
STRATEGY (512 floats): Meta-information (what to focus on next). Ephemeral, consumed each pass.
```

The hypernetwork reads both (576 floats) for richer LoRA generation. Strategy can't bypass bottleneck because it doesn't persist across passes.

**Status:** Proven. 85.4% two-step (up from 53% without side channel).

### LoRA as Additive Term — PROVEN

Previous implementation used forward hooks to apply/remove LoRA every pass. Slow (~13 min per epoch). Changed to:

```python
q = layer.q_proj(hidden) + (hidden @ B.T) * scales @ A.T
```

No hooks. No weight modification. Just parallel additive computation. ~4 min per epoch on two-step.

**Status:** Proven. Monkey-patched forwards ensure output_hidden_states returns post-LoRA hidden states.

---

## Training Setup

### Data
- Two-step arithmetic: procedural generation, `(a op1 b) op2 c`
- Three-step arithmetic: procedural generation, `((a op1 b) op2 c) op3 d`
- GSM8K: `data/gsm8k_easy.jsonl` (4,232 problems), full GSM8K train (7,473)
- SymPy probe targets: precomputed intermediate values per step

### Training Loop
```python
for pass_num in range(3):  # fixed 3 passes
    # Apply LoRA (additive, from state + strategy)
    # Forward through Llama (frozen, output_hidden_states=True)
    # Perceiver compresses all 16 layers → state_delta (64) + strategy (512)
    # State: accumulate on hypersphere
    # Probe: MSE(probe(state), sympy_target) → per-pass gradient
    
# Final: teacher-forced answer loss
# Total: answer_loss + 0.5 * probe_loss
# Gradient clipping: max_norm=1.0
```

### Optimizer
```python
AdamW([
    {'params': perceiver_params, 'lr': 1e-4},
    {'params': lora_template_params, 'lr': 1e-3},   # 10x (gradient imbalance fix)
    {'params': hypernetwork_params, 'lr': 1e-3},     # 10x
    {'params': probe_params, 'lr': 1e-4},
])
# Transformer is FROZEN — not in optimizer
```

---

## File Structure (Expected)

```
src/
  thinking_model.py              # ThinkingModel (main class)
  all_layer_perceiver.py         # 7-layer perceiver, two output heads (state + strategy)
  state_conditioned_lora.py      # Templates + hypernetwork, additive LoRA
  confidence_head.py             # Readiness judge (currently disabled)
  
scripts/
  train_thinking.py              # Main training script
  eval_thinking.py               # Accuracy, ablation, probe analysis
  generate_arithmetic.py         # Procedural arithmetic data generation

data/
  gsm8k_easy.jsonl               # 4,232 easy GSM8K problems
  
checkpoints/
  phase1_best.pt                 # Best two-step checkpoint (53%)
  
logs/
  train_*.log                    # Training logs
```

---

## Known Bugs & Gotchas

```
Bug #26/#27: DataCollatorForLanguageModeling overwrites label masking
Bug #35: Temperature must be 0 for eval. 0.7 caused massive baseline shift.
Bug #37: Small eval sets lie. Always N≥100.
Bug #38: Always verify baseline on SAME problem set with SAME prompt format.

GQA: Llama 3.2 1B uses Grouped Query Attention (8 KV heads, 32 Q heads).
     K,V projections are (2048, 512) not (2048, 2048).
     LoRA templates for K,V must be (rank, 512) not (rank, 2048).

Tokenizer: Llama 3.2 uses its own tokenizer. Don't mix with SmolLM2's.

Answer extraction: Strip trailing periods ("72." → "72"). Check for \boxed{}, 
     ####, and last-number-in-text fallbacks. We've been bitten by extraction 
     bugs 3+ times.

Prompt format: Instruct model needs chat template + "Solve step by step" for 42%.
     Base model on GSM8K (verified 500 problems):
       - zero-shot raw "Problem: ... Answer:": 4-5% (matches old CLAUDE.md note)
       - 3-shot prompted with "The answer is X" extraction: 6.2% ← REAL BASELINE
     Inspecting 30 problems gave a misleading 16.7% — first 8 GSM8K problems
     are the famous easy canonical ones (Natalia clips, Weng babysitting). Bug
     #37 again. Always verify on N≥500 before trusting a baseline.
     For arithmetic: simple completion format "48 / 2 =" works at 70%.

Gradient imbalance: Perceiver gets 4530x more gradient than LoRA templates.
     Fix: 10x higher LR for templates (1e-3 vs 1e-4).

Unfreezing Llama: Caused instability even at 1e-6 LR. Keep frozen until 
     the architecture is proven on word problems.

Efficiency penalty: 0.01 * num_passes caused model to always stop at 1 pass.
     Removed. Passes fixed at 3. Re-add confidence head later.

Random hypersphere init: ALWAYS use random init during training, NOT learned init.
     Random init forces the model to work from ANY starting point on the
     hypersphere — implicit data augmentation. Every problem is seen from
     multiple "angles" across training. The model learns the TASK, not a
     specific TRAJECTORY. Learned init overfits to one path (59.6% vs 85.4%).
     Once trained with random init, the model has zero eval variance across
     seeds — robust eval confirmed 85.4% ± 0.0% across 5 seeds.

Probe-only training caps low: Training with only probe loss (MSE on state →
     intermediate values) capped at 2.8%. Must include teacher-forced answer
     loss. The probe teaches the state WHAT to encode. The answer loss teaches
     the system HOW to generate from that state. Both are required:
     loss = answer_loss + 0.5 * probe_loss.

State conditioning: Adding current_state as input to perceiver queries HURT
     performance (50% vs 53%). The pass embedding alone is sufficient. Reverted.

Bias injection: Adding state as bias to embeddings CORRUPTED pretrained 
     representations. Model mode-collapsed (output "30" for everything). 
     Don't add bias to embeddings — use pseudo-tokens or LoRA only.
```

---

## AWS Setup

```bash
# SSH to VM
ssh -i ~/.ssh/mycelium-key.pem ubuntu@<IP>

# The IP changes — check AWS console for current g5.xlarge
# Instance type: g5.xlarge (A10G 24GB, ~$1/hr)
# OS: Ubuntu 24
# Python: 3.x with torch, transformers, sympy

# S3 bucket
aws s3 ls s3://mycelium-data-v7/

# Project directory on VM
cd ~/mycelium

# Training with tmux (ALWAYS use tmux — session loss caused significant context loss before)
tmux new -s train
python scripts/train_thinking.py --config configs/thinking_gsm8k.yaml

# Monitor
tail -f logs/train_*.log
```

---

## What to Do Next (Priority Order)

### 1. GSM8K Word Problems (IN PROGRESS)

Generalization test: arithmetic → natural language word problems.
Base model baseline: 4-5%. Target: >10%. Paper territory: >20%.
Training from three-step 73.6% checkpoint.

### 2. MATH-500 Benchmark (May 22 deadline)

### 3. Dual LoRA: Forward Computation + Verification Mirror (v22)

Two sets of LoRA templates — `forward` (computation: narrow, sequential
attention) and `verify` (consistency-checking: broad, relational attention) —
blended by a learned sigmoid weight per cycle. The hypernetwork outputs
`(forward_scales, verify_scales, blend)`, and the additive LoRA term is
`(1-blend) * q_forward + blend * q_verify`. The model gradually rotates from
building an answer to checking it — the geometric mirror of computation on
the same hypersphere.

The blend weight traces a natural sigmoid arc across cycles: early passes are
near-pure forward (parse, compute), later passes near-pure verify (check
intermediates, confirm relationships). Reflection point ≈ blend 0.5.

This is also where the confidence head finally earns its keep: with pages +
blend history as input, it can learn "don't stop until verification has
happened," trained against a correctness signal (no efficiency penalty —
which broke it last time). Easy problems verify in 2 cycles, hard ones in 8.

Adds ~1.1M params (second template set). Plan doc: `plan/dual_lora_verification.md`.

**Status:** Architecture defined. Phase 2 — after GSM8K baseline with single LoRA.

### 4. Boltzmann Exploration for Pass Count

Currently passes are fixed at 3. Instead of a hard confidence threshold, use
Boltzmann (softmax) sampling over a learned "continue/stop" distribution to
decide whether to take another pass. Temperature annealing during training:
start hot (explore many pass counts) → cool down (exploit optimal count per
problem). This lets the model learn variable-depth reasoning — easy problems
get 1-2 passes, hard problems get 5+. Avoids the efficiency penalty collapse
(always stopping at 1) that killed the previous confidence head attempt.

### 5. Attention Residuals Across Passes (5+ Pass Scaling)

At 5+ passes, the state bottleneck may lose information from early passes.
Add cross-pass attention residuals: each pass's perceiver attends not just to
the current Llama hidden states but also to a buffer of compressed
representations from prior passes. Like a "memory of memories" — the state
vector carries the compressed result, but the residual connection preserves
richer structure for the perceiver to query. This prevents information decay
in deep reasoning chains without widening the 64-float bottleneck.

---

## Architecture Evolution (How We Got Here)

```
v10-v15: Text-based compression ([EXPAND]/[COLLAPSE] tags) → abandoned (not differentiable)
v16:     Latent-space bottleneck (SmolLM2-135M, pseudo-tokens) → 80.4% two-step ✓
v17:     Engine swap to Llama 3.2 1B (richer hidden dim 2048)
v18:     Integrated thinking (no text generation during thinking passes)
v19:     64-float tight bottleneck + 7-layer perceiver + all-layer reading
v20:     State-conditioned LoRA (state rewires attention, not just pseudo-tokens) → 53% ✓
v20.1:   Side channel (512-float strategy) + additive LoRA + answer loss → 85.4% ✓
v20.2:   Three-step arithmetic (warm start from v20.1) → TRAINING NOW
```

Key insight at each pivot:
- v15→v16: Text compression isn't differentiable. Latent compression is.
- v16→v17: SmolLM2-135M can't parse word problems. Llama 1B can.
- v17→v18: Generating text at each breath is slow. Forward passes are cheap.
- v18→v19: Model must be forced to think incrementally. Tight bottleneck.
- v19→v20: Pseudo-tokens get ignored (4/104 = 3.8%). LoRA rewires attention — can't be ignored.
- v20→v20.1: 64 floats starved the hypernetwork. Side channel gives 9x more info.

---

## Experiments That Failed (Don't Repeat)

| Experiment | Result | Why It Failed |
|------------|--------|---------------|
| Text-based [EXPAND]/[COLLAPSE] | Matched or hurt vanilla | Not differentiable |
| SmolLM2-135M on word problems | 0% | Can't parse "half as many" |
| Llama 1B base on GSM8K (raw) | 4-5% | Format issue, not capability |
| Llama 1B instruct with breathing | 11% (vs 42% baseline) | Instruct already chains; breathing hurts |
| Bias injection into embeddings | 0%, mode collapse | Corrupts pretrained representations |
| State conditioning in perceiver | 50% (vs 53%) | Added noise, pass embed sufficient |
| Unfreezing Llama at 1e-6 | 40% volatile (vs 53%) | Destabilizes training |
| Efficiency penalty on passes | 1-pass collapse | Model games it, always stops early |
| Alternating EXPAND/COLLAPSE breaths | 3% on GSM8K | SmolLM2 couldn't parse word problems |
| Probe-only training (no answer loss) | 2.8% (vs 85.4%) | Probe teaches WHAT to encode, not HOW to generate |
| Learned initial state (nn.Parameter) | 59.6% (vs 85.4%) | Overfits to one trajectory; random init = implicit augmentation |
