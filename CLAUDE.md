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
Two-step arithmetic         0%          94.8%  ← BEST    Llama 1B, page-based, target-cos contrastive
Two-step arithmetic         0%          85.4%             Llama 1B, side channel, additive LoRA, answer loss
  (robust eval)                         ±0.0% (5 seeds)  (92.4% effective per-step)
Two-step arithmetic         0%          53%               Llama 1B, 64-float, LoRA (probe only, no answer loss)
  (theoretical ceiling)                 (49%)             (53% > 49% → LoRA improves per-step)
Three-step arithmetic       0%          83.4% ← BEST     Llama 1B, page-based, target-cos contrastive (1 effective pass)
Three-step arithmetic       0%          73.6%             Llama 1B, side channel, additive LoRA, warm start from 85.4%
  (robust eval)                         ±0.0% (5 seeds)  (90.1% effective per-step, cube root)
L2 word ops                 0.6%        53.4%             CoT targets + pass-conditioned hypernetwork
L3 named qty (dual LoRA)    18.8%       96.0% ← BEST     Dual LoRA verification (+7.4 pts over single)
L3 named qty (single LoRA) 18.8%       88.6%             CoT targets + warm start from L2
L4 two-step word problems   40.8%       100.0% ← BEST     Dual LoRA, warm from L3, 1 epoch
GSM8K (dual LoRA)           2.2%        17.8%  ← BEST     Dual LoRA, 5 passes, curriculum L0→L4→GSM8K (8.1x)
GSM8K (hybrid, epoch 1)     6.2%        6.6%              Old hybrid path (LoRA thinking + pseudo-token gen)
L2 word ops (terse targets) 0.6%        12.2%             Terse "143" targets → number-spam (format bug)
Three-step w/ pass-cond     0%          55.4%             Pass-conditioned hypernetwork (pages differentiate but accuracy drops)
Three-step arithmetic       0%          52%               SmolLM2-135M, 64-float, pseudo-tokens
Two-step arithmetic         0%          83%               Llama 1B, 512-float, pseudo-tokens (earlier arch)
Two-step arithmetic         0%          80.4%             SmolLM2-135M, 32-float, pseudo-tokens
```

Key finding: 85.4% on two-step with zero variance across 5 seeds. Effective per-step
accuracy: 92.4% (up from 70% base). The three changes that broke the 53% ceiling:
1. Answer loss (was missing entirely — probe alone capped at 2.8%)
2. Side channel (576 floats feeding hypernetwork instead of 64)
3. Additive LoRA (clean, fast, fully differentiable, post-LoRA hidden states)

L2 word ops breakthrough: CoT targets matching the base model's natural style
("half of 48 = 24. 24 plus 48 = 72. The answer is 72.") jumped accuracy from
12.2% (terse targets) to 53.4%. The base model already shows chain-of-thought
capability in its completions — training targets must match this natural style.

GSM8K result: 17.8% from 2.2% baseline (8.1x) using full curriculum warm-start
(L0→L1→L2→L3→L4→GSM8K), dual LoRA verification with 5 thinking passes.
Blend ≈ 0.65 on GSM8K (heavy verification) vs ≈ 0.25 on easy problems —
the model naturally adapts verification intensity to problem difficulty.

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

### Pass-Conditioned Hypernetwork (v21.3 — PROVEN)

The page diagnostic on the 83.4% three-step checkpoint revealed pages 2-3 were
identical (cosine 1.0000). A circular copy loop: same pages → same LoRA → same
hidden states → same perceiver output → same pages. No page-level penalty could
break it (tried anti-copying at λ=1.0 — pages stayed at cos 1.0000).

**Fix: inject pass_num into the hypernetwork.** The hypernetwork receives a
learned pass embedding so different passes produce different LoRA even with
identical pages. Breaks the symmetry at the INPUT (LoRA generation), not the
OUTPUT (pages).

```python
# In PageAttentionHypernetwork.__init__:
self.pass_embed = nn.Embedding(max_passes, 256)

# In compute_scales:
pass_emb = self.pass_embed(pass_num)
combined = torch.cat([page_summary, strategy, pass_emb], dim=-1)
scales = self.combine(combined)
```

Result: p2v3 dropped from 1.0000 → 0.30 on epoch 1. Pages are genuinely
different. The architectural fix worked where loss-based penalties could not.

However, for three-step arithmetic the model gets 83.4% with one effective pass
(static pages). Forcing three different passes makes each weaker (55.4% peak).
Multi-pass thinking is the right architecture for HARDER problems (word problems)
where different passes genuinely need different cognitive operations (parse vs
compute vs verify). Three-step arithmetic doesn't need it.

**Status:** Proven (breaks page copying). Carry forward to stepping stones.

### Contrastive Page Loss (v21.2 → v21.3)

Running a page-content diagnostic on the 86.2% `two_step_pages_best.pt`
checkpoint revealed the pages were constant:

```
Last-page cosine similarity (200 problems):
  same answer : 1.0000
  diff answer : 0.9998
  delta       : 0.0002   ← pages are constant across problems
  per-dim std : 0.0127 mean, 28/64 dims have std < 0.01
```

The whole architecture had collapsed to a learned static LoRA — one good
configuration applied to every input. The "thinking loop" was an illusion.

**Two proven failure modes:**

1. **Fixed-point collapse** — all pages constant across all problems.
   Discovered at 85.4% checkpoint (page cosine 0.9998 across problems).

2. **Page copying** — page 1 constant, pages 2-3 identical within each
   problem. Discovered at 83.4% three-step checkpoint (page 2-3 cos 1.0000).
   Pages 2&3 learned identical LoRA configs — three passes collapsed to one.

**Evolution of contrastive losses:**

- **v21.2a Margin contrastive** (last page only): 91.6% two-step but fragile
  (68.8% on same config, batch-order dependent). One-sided ReLU, knife-edge λ.

- **v21.2b Target-cosine** (last page, then all pages): Stable but requires
  choosing target_cos (0.4 → overshooting, 0.7 → better). Best: 83.4% three-step
  with target=0.7 on last page. Per-page variant with within_target=0.3 failed:
  81.6→71.0% (within-term too weak at λ=0.05 against answer loss gradient).

- **v21.3 SupCon + anti-copying** (current): Two principled constraints, no
  arbitrary targets. Supervised contrastive (SupCon) on ALL pages — temperature
  controls geometry, not a target cosine. Soft quadratic anti-copying penalty
  above cos=0.7 between pages within a problem — free below 0.7, the model
  discovers its own within-problem geometry.

```python
def breathing_contrastive_loss(all_pages, gold_answers, temperature=0.1):
    loss = 0.0
    # Term 1: per-page SupCon (same answer → together, diff → apart)
    for page in all_pages:
        loss += supervised_contrastive(page, gold_answers, temperature)
    # Term 2: anti-copying (free below 0.7, quadratic above)
    for i, j in pairs:
        within_cos = cos(page_i, page_j)
        loss += relu(within_cos - 0.7)² .mean()
    return loss / len(all_pages)

total_loss = generation_loss + 0.05 * breathing_contrastive_loss(all_pages, gold)
```

Key design choices:
- **SupCon over target-cosine**: no target to choose, geometry emerges from data
- **Anti-copying threshold 0.7**: matches empirical cross-problem sweet spot,
  ungameable, model is free below 0.7 to discover its own page geometry
- **λ=0.05 constant**: SupCon is self-regulating, no schedule needed
- **Temperature=0.1**: standard SupCon, controls cluster tightness

Findings doc: `plan/fixed_point_collapse_findings.md`.
Loss: `src/contrastive_page_loss.py`. Script: `scripts/train_three_step_contrastive.py`.
Diagnostics: `scripts/diag_pages.py`, `scripts/diag_three_pages.py`.

**Status:** SupCon + anti-copying recipe ready. Training next.

### Log-Answer Head (v21.1 — DEFERRED pending pages that actually encode)

Every GSM8K generation attempt collapsed to one of two failure modes:
LoRA-only ran into "emit number immediately" (arithmetic-training spillover),
and hybrid-short-completion ran into "The answer is X. The answer is X. ..."
number-spam (training target was 6 tokens, so the model learned to skip
reasoning entirely). The thinking chain works fine — the readout is broken.

**Pivot: delete the readout problem.** Train the last page to encode the
answer directly, read it out with two tiny linear heads:

```
last_page (64 floats)
    ├──  Linear(64, 1) → log10(|ans| + 1)
    └──  Linear(64, 1) → P(ans >= 0)  (sigmoid)

answer = sign * (10^log_mag - 1)
loss   = MSE(log_mag, log10(|gold|+1)) + 0.1 * BCE(sign_logit, gold >= 0)
```

No generation. No tokenized completions. No teacher forcing. 130 new params.

Why this is different from the previous probe-only attempt (capped at 2.8%):
the old probe used raw values (loss exploded to 1.26B), trained on
intermediates not finals, and competed with a generation loss. The log-head
runs in bounded log space (0→6 for GSM8K), trains only on the final answer,
and is the *primary* objective with no generation loss to fight.

Two eval metrics: **exact** (rounded equality) and **tol1%** (within 1% of
gold — directional correctness even when integer rounding fails on large
answers). If exact lags tol1%, the pages are right and we need output
precision (mantissa + exponent upgrade, not an architecture change).

PageToTokens kept as fallback for MATH-500 (non-numeric answers).
Plan doc: `plan/log_answer_head.md`. Script: `scripts/train_gsm8k_answerhead.py`.

**Status:** Built. Warm-starts from `two_step_pages_best.pt` (v21 86.2% two-step).

### GSM8K Hybrid (LoRA thinking + pseudo-token generation) — ARCHIVED

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

CoT targets: Always train on chain-of-thought targets that match the base
     model's natural completion style. Terse targets ("143") cause number-spam
     from arithmetic training spillover. CoT targets ("half of 48 = 24. 24
     plus 48 = 72. The answer is 72.") jump L2 from 12.2% to 53.4%.
     Check base model generations FIRST to see what style it naturally produces.
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

### 1. L3 Named Quantities (NEXT)

L2 word ops achieved 53.4% with CoT targets. Next level:

```
L2: "half of 48 plus 48"                   → 53.4% ✓ (CoT targets)
L3: "Jamie had 56 cookies and gave 2 away" → NEXT (named quantities)
L4: 2-step word problems, small numbers     (easy GSM8K style, 4-6 passes)
L5: Full GSM8K                             (complex multi-step, 6-12 passes)
```

Recipe (proven on L2): 20K problems, CoT targets matching base model style,
target-cos contrastive (0.7), pass-conditioned hypernetwork, patience=2,
warm-start from L2 CoT best checkpoint (53.4%).

If L3 fails (<20%): frozen Llama may not parse narrative context ("Jamie had
56 cookies" is harder than "56 minus 2"). Consider unfreezing at 1e-7.

Plan docs: `plan/stepping_stones_handoff.md`, `plan/morning_handoff.md`.

### 2. Dual LoRA: Forward Computation + Verification Mirror (v22)

Two sets of LoRA templates — `forward` (computation: narrow, sequential
attention) and `verify` (consistency-checking: broad, relational attention) —
blended by a learned sigmoid weight per pass. The hypernetwork outputs
`(forward_scales, verify_scales, blend)`, and the additive LoRA term is
`(1-blend) * q_forward + blend * q_verify`. The model gradually rotates from
building an answer to checking it.

Even on easy problems, verification catches the ~2.6% per-step errors:
```
Without verification: 97.4% per-step
With verification:    potentially 99%+ per-step (errors caught before output)
```

The confidence head reads pages + blend history, trained with correctness
signal (no efficiency penalty). Learns "don't stop until verification has
happened (blend > 0.5)." Easy problems: 2 passes. Hard: 8.

```python
# Blended application (additive, no hooks):
q_forward = (hidden @ A_forward.T) * forward_scales @ B_forward
q_verify = (hidden @ A_verify.T) * verify_scales @ B_verify
q_lora = (1 - blend) * q_forward + blend * q_verify
q = layer.q_proj(hidden) + q_lora
```

Adds ~1.1M params (second template set). Plan docs: `plan/dual_lora_verification.md`,
`plan/morning_handoff.md`.

**Status:** PROVEN. 96.0% on L3 (vs 88.6% single LoRA, +7.4 pts).
Blend trajectory: 0.15→0.30 over 8 epochs — model discovers verification helps.
Verification is a generalization tool (helps most at epoch 1, less when memorized).
Confidence head needs per-pass correctness training for dynamic stopping.
Checkpoint: `dual_lora_L3_best.pt` (epoch 5, 96.0%).

### 3. MATH-500 Benchmark (May 22 deadline)

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
v20.2:   Three-step arithmetic (warm start from v20.1) → 73.6% three-step ✓
v21:     Page-based state accumulation → 86.2% two-step ✓ (but pages constant)
v21.2:   Target-cosine contrastive → 94.8% two-step, 83.4% three-step ✓
v21.3:   Pass-conditioned hypernetwork → pages differentiate ✓ (p2v3=0.30)
v21.4:   Stepping stones L2 word ops → 53.4% ✓ (CoT targets breakthrough)
v21.5:   Stepping stones L3 named qty → 88.6% single LoRA ✓
v22:     Dual LoRA (forward + verify) → 96.0% L3 ✓ (+7.4 pts, verification proven)
v22.1:   L4 two-step word problems → 100.0% ✓ (1 epoch, instant generalization)
v22.2:   GSM8K dual LoRA → 17.8% ✓ (8.1x over 2.2%, 5 passes, blend ≈ 0.65)
```

Key insight at each pivot:
- v15→v16: Text compression isn't differentiable. Latent compression is.
- v16→v17: SmolLM2-135M can't parse word problems. Llama 1B can.
- v17→v18: Generating text at each breath is slow. Forward passes are cheap.
- v18→v19: Model must be forced to think incrementally. Tight bottleneck.
- v19→v20: Pseudo-tokens get ignored (4/104 = 3.8%). LoRA rewires attention — can't be ignored.
- v20→v20.1: 64 floats starved the hypernetwork. Side channel gives 9x more info.
- v21.3→v21.4: Terse targets cause number-spam. CoT targets match base model's natural style.

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
| Target-cos contrastive (last page only) | 83.4% three-step | Pages 2&3 identical (cos=1.0) — last-page-only contrastive doesn't prevent page copying |
| Per-page target-cos with within_target=0.3 | 81.6→71.0% | Within-term too weak at λ=0.05; answer loss gradient overwhelms it, accuracy degrades |
| Margin contrastive (λ=0.3) | 91.6% or 68.8% | One-sided ReLU, batch-order dependent, knife-edge λ — not reproducible |
| SupCon (temperature 0.1-0.3) | 49-67% | Overshoots — pushes page_cos to 0.02-0.09, way past 0.7 sweet spot |
| Forced multi-pass on 3-step arith | 55.4% (vs 83.4%) | Problem doesn't need 3 passes. One effective pass suffices. Architecture correct, problem too easy. |
| Terse answer targets on L2 word ops | 12.2% (vs 53.4% CoT) | Model learns number-spam from arithmetic training spillover. CoT targets matching base model's natural completion style fix it. |
