# CLAUDE.md — Mycelium Breathing Models

## Project Overview

Mycelium builds differentiable recurrent reasoning for small language models. A frozen base LLM that can't chain reasoning internally learns to chain through external compression — thinking in a loop where each pass rewires its own attention via state-conditioned LoRA. The core thesis: **decomposition is everything.** Each thinking cycle matches one pattern from a learned library, computes one result, and records it in a growing notebook. The model breathes: EXPAND (think in natural language) → COLLAPSE (compress to 64 floats) → repeat.

**Lead:** Bryce (Manhattan Beach, CA) · **Target:** MATH-500 · **Deadline:** July 1, 2026
**Infrastructure:** AWS EC2 g5.xlarge (A10G 24GB)

---

## Architecture (v27.0 — Unified BreathingController)

```
Problem text + text injection ("Step 1: 240\nStep 2: 80")
  │
  ▼
LLAMA 3.2 1B BASE (frozen, atom-modified attention)
  │
  ├── all 16 layers of hidden states
  │     │
  │     └── BREATHING CONTROLLER (~190M, reads ALL layers)
  │           │
  │           ├── reads current hidden states (what Llama just computed)
  │           ├── cross-attends over history (previous pages + hidden pools)
  │           │
  │           ├── PAGE HEAD → 64-float page → Mobius → append to notebook
  │           │                (what was understood — the record)
  │           │
  │           ├── SCALE HEAD → 64 atom scales → applied to NEXT cycle
  │           │                (what to think about next — the plan)
  │           │
  │           └── FOCUS HEAD → 64 Mobius focus floats
  │                            (where on the sphere to record)
  │
  └── logits → GENERATION → "240 episodes #### 240</s>"
                              (the output, scored by extraction)
```

**Two pillars:**
- **BreathingController (~190M):** OBSERVE — reads Llama's hidden states, produces page + scales from shared understanding
- **64 LoRA Atoms (~82M):** BREATHE — modify how Llama attends, Fourier-initialized, tanh-scaled

**Why unified?** The old separate perceiver (105M) + hypernetwork (101M) failed: the hypernetwork produced IDENTICAL scales for every problem at cycle 1 (constant function) because it read compressed pages, not raw hidden states. The controller reads hidden states directly — different problems always produce different scales.

**Key components:**
- **Llama 3.2 1B base** (frozen, NOT instruct — base can't chain, instruct already can)
- **Growing notebook** — each cycle appends a fresh 64-float page, nothing overwritten
- **64 anonymous LoRA atoms** (Fourier-initialized, orthogonal) — the pattern library
- **Per-cycle intermediate targets** — each cycle predicts ONE intermediate result
- **Text injection** (cumulative) — all previous results as actual text tokens
- **Generation-only loss** — no answer head, gen cross-entropy is the only training signal
- **#### marker format** — each cycle ends with `#### {number}</s>` for clean extraction
- **EOS weighted 5x** — teaches clean breath boundaries
- **Three-tier gating** — correct=1.0, wrong=0.1, copying consumed target=0.0
- **Number augmentation** — randomize all numbers each epoch (anti-memorization)
- **Gen target dropout 15%** — mask equation results (anti-copying)
- **Label smoothing 0.05** — prevents gen_loss collapse
- **Scale diversity loss** — penalizes similar scales between cycles (target cos < 0.3)
- **Mobius transform** — conformal warp for page diversity on hypersphere
- **Tanh scaling with hard clamp [-3, 3]** — no softmax, no mode collapse

---

## Three Principles

### 1. Decomposition Is Everything
Each cycle handles one piece. "160 - 63" is easy. Knowing to do "160 - 63" is the hard part. The model learns to break hard problems into easy pieces.

### 2. Patterns Are the Vocabulary of Decomposition
64 atoms BLEND continuously — an infinite space of attention modifications. Each point is a unique pattern recognizer. The controller navigates to the right blend per cycle.

### 3. The System Observes Itself
One controller reads what Llama computed (OBSERVE), records understanding (page), and plans the next step (scales). Record and plan come from the same act of self-reflection.

---

## Proven Results

| Task | Base | Breathing | Notes |
|------|------|-----------|-------|
| Single-step arithmetic | 70% | **100%** | Llama 1B, 64-float state |
| Two-step arithmetic | 0% | **94.8%** | Target-cos contrastive |
| Three-step arithmetic | 0% | **83.4%** | 93.8% per-step |
| L2 word ops | 0.6% | **53.4%** | CoT targets breakthrough |
| L3 named qty | 18.8% | **96.0%** | Dual LoRA verification |
| L4 two-step word | 6.0% | **91.0%** | Page delta breakthrough |
| L4.5 three-step word | — | **94.5%** | Growing notebook |
| GSM8K | 2.2% | **training** | BreathingController, xpass_cos=-0.2 |

---

## Curriculum

```
L0: single-step arithmetic (70% → 100%)     ✓
L1: two-step arithmetic (0% → 94.8%)        ✓
L2: word ops (0.6% → 53.4%)                 ✓
L3: named quantities (18.8% → 96.0%)        ✓
L4: two-step word (6% → 91%)                ✓
L4.5: three-step word (94.5%)               ✓
GSM8K: (2.2% → training)                    → IN PROGRESS (BreathingController)
```

---

## What to Do Next (Priority)

### 1. GSM8K with BreathingController (IN PROGRESS)
Unified controller training from scratch. xpass_cos = -0.2 (scales genuinely diverse).
10 min/epoch (53% faster than old architecture). Watching gen_loss drop.

### 2. Confidence Head + Variable Cycles
Train confidence to decide when to stop. Enable 3-8 cycles per problem.

### 3. Gentle Training Wheel Removal
Blend teacher and autonomous decompositions. Increase autonomy based on competence.

### 4. MATH-500 (July 1 deadline)

---

## Design Decisions

- **Unified controller, not separate perceiver+hypernetwork.** Separate hypernetwork collapsed to constant function. Controller reads hidden states directly — can't collapse.
- **APPEND pages, don't blend.** Residual gate caused convergence (page_cos 0.91 at depth 3). Fresh independent pages.
- **Base model, not Instruct.** Instruct chains at 42%. Base: 0% chained. Architecture provides chaining.
- **Fourier atom init.** Orthogonal basis functions. Multi-scale pattern library.
- **Per-cycle targets, not CoT.** CoT makes cycles redundant. Per-cycle targets force each cycle to do one job.
- **Generation-only (no answer head).** Five answer head versions all failed on GSM8K (4% peak).
- **Scale diversity loss.** Direct pressure to differentiate cycles. Without it, hypernetwork produces identical scales.
- **Number augmentation.** Randomize all numbers each epoch. Prevents memorization.
- **Label smoothing 0.05.** Prevents gen_loss collapse.
- **No graduation/detach.** Every scheme destabilized training. Plain full-weight training works best.
- **64-float bottleneck.** Forces incremental thinking. Notebook GROWS (8 cycles = 512 total).

---

## Loop-Alive Checklist (EVERY training script)

```
□ scale_reg active (hard clamp [-3, 3])
□ No residual gate (APPEND pages, don't blend)
□ EOS weighted 5x in generation loss
□ #### marker in gen targets for extraction
□ --augment flag for GSM8K (anti-memorization)
□ scale_diversity_loss active (target_cos=0.3)
□ Controller reads ALL 16 Llama layers
```

---

## Known Bugs & Gotchas

| Bug | Impact |
|-----|--------|
| Temperature > 0 at eval | Baseline shift |
| Small eval sets (N<100) | Misleading accuracy |
| Answer extraction edge cases | Strip periods, \boxed{}, ####, last-number |
| Cycle 2 copies cycle 1 | Three-tier gating: copying=0.0 reward |
| Generation rambles past correct answer | EOS 5x weight + #### marker format |
| gen_loss collapse without augmentation | 0.07 train, 7% eval = memorization |
| Separate hypernetwork constant function | Scales identical all problems → use controller |
| Bypass/message/text-context collapsed | All intermediaries → constant. Controller reads direct |
| Extracted numbers overflow torch.long | Clamp to [-999999999, 999999999] |
| NoneType in eval prev_preds | Guard with `if val is not None else 0` |

---

## AWS Setup

```bash
aws ec2 start-instances --instance-ids i-08c1c295a4113a908
ssh -i ~/.ssh/mycelium-key.pem ubuntu@<IP>  # IP changes on restart
cd ~/mycelium && tmux new -s train
python3 scripts/train_per_cycle.py --level gsm8k --data_dir data/per_cycle \
  --epochs 50 --num_passes 3 --patience 30 --batch_size 4 \
  --warm_from checkpoints/per_cycle_gsm8k_best.pt --lr_scale 0.5 --augment
```

Instance: g5.xlarge (A10G 24GB, ~$1/hr)

---

## Architecture Evolution

```
v15   Text [EXPAND]/[COLLAPSE]           → not differentiable, abandoned
v16   SmolLM2-135M latent bottleneck     → 80.4% two-step ✓
v17   Llama 3.2 1B engine               → richer hidden states
v20   State-conditioned LoRA             → 85.4% two-step ✓
v21   Page-based accumulation            → pages constant (cosine 0.9998)
v21.2 Target-cosine contrastive          → 94.8% two-step ✓
v22   Dual LoRA verification             → 96.0% L3 ✓
v22.2 GSM8K curriculum                   → 17.8% ✓
v24   64-Atom LoRA + Fourier init        → 93% L3, 91% L4 ✓
v25   Per-cycle targets                  → 89% L4 ✓
v25.4 Growing notebook (remove gate)     → fresh pages, no degradation ✓
v26.0 Generation-only (drop answer head) → no gradient conflict
v26.1 Number augmentation + dropout      → anti-memorization
v27.0 BreathingController (unified)      → scales diverse (xpass_cos=-0.2), 53% faster
```

---

## Failed Experiments (Don't Repeat)

| What | Result | Lesson |
|------|--------|--------|
| Llama instruct + breathing | 11% | Already chains — use base |
| Quad LoRA softmax | Mode collapse | Use tanh, no competition |
| Answer head (5 versions on GSM8K) | 4% peak, hurt gen | Remove — generation IS the output |
| Residual gate blending | cos(2,3)=0.91 | Pages converge → APPEND instead |
| Dynamic detach/graduation | Crashes, collapse | Disrupts equilibrium → don't use |
| Full trifecta regularization | gen_loss 1.78 | Too strong → reduce dropout/smoothing |
| Separate hypernetwork | Constant function | Reads pages not hidden states → controller |
| Bypass (512 floats) | Collapsed to constant | Intermediary without training signal → controller |
| Message generator | Collapsed to constant | Same problem → controller reads direct |
| Text context for hypernetwork | Ignored | No pressure to use it → diversity loss |

---

## File Structure

```
scripts/
  atom_lora.py                   # Model: AtomLoRAModel, BreathingController, LoRAAtoms
  train_per_cycle.py             # Training: gen-only + augmentation + controller
  generate_per_cycle_data.py     # Data gen: L3-L4.9 with cycle_targets + gen_targets
  parse_gsm8k.py                 # Parse GSM8K built-in step annotations
  annotate_gsm8k_cycles.py       # Claude API: decompose GSM8K problems
  diag_laplace.py                # Diagnostic: page trajectory analysis (FFT, convergence)
  diag_debug_cycles.py           # Diagnostic: full chain inspection per problem
plan/
  project_outline.md             # Core thesis
  breathing_controller_handoff.md # Current architecture
  generation_only_handoff.md     # Drop answer head
  augmentation_dropout_handoff.md # Anti-memorization
  per_cycle_targets_handoff.md   # Per-cycle intermediate targets
  append_pages_handoff.md        # Growing notebook
  confidence_autonomy_handoff.md # Confidence head (next feature)
  mobius_transform_handoff.md    # Page diversity
  isotropic_confidence_handoff.md # Isotropic regularizer
src/
  contrastive_page_loss.py       # Cross-problem page diversity loss
data/
  per_cycle/                     # Generated JSONL: L3-L4.9 + GSM8K
```
