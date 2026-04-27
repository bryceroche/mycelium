# CLAUDE.md — Mycelium Breathing Models

## Project Overview

Mycelium builds differentiable recurrent reasoning for small language models. A frozen base LLM that can't chain reasoning internally learns to chain through external compression — thinking in a loop where each pass rewires its own attention via state-conditioned LoRA. The core thesis: **decomposition is everything.** Each thinking cycle matches one pattern from a learned library, computes one result, and records it in a growing notebook. The model breathes: EXPAND (think in natural language) → COLLAPSE (compress to 64 floats) → repeat.

**Lead:** Bryce (Manhattan Beach, CA) · **Target:** MATH-500 · **Deadline:** July 1, 2026
**Infrastructure:** AWS EC2 g5.xlarge (A10G 24GB), S3 bucket mycelium-data-v7

---

## Architecture (v25.4 — Growing Notebook)

```
GROWING NOTEBOOK: [page_1, page_2, ..., page_N]  (64 floats each, APPENDED)
MESSAGES: [msg_1, msg_2, ..., msg_N]  (16 floats each, direct from last layer)
     │
     ▼
ATOM HYPERNETWORK (10M params)
  2-layer cross-attention over ALL pages + messages → 64 tanh scales
     │
     ▼
64 RANK-6 LORA ATOMS (~82M params, Fourier-initialized, orthogonal)
  q = W_q x + Σᵢ (scaleᵢ · (x @ Bᵢᵀ) @ Aᵢᵀ)
     │
     ▼
LLAMA 3.2 1B BASE (frozen) + text-injected previous results
  "Step 1 result: 160\nStep 2 result: 97\n[problem text]"
     │
     ▼
HAAR WAVELET PREPROCESS (2x compression, frequency structure)
     │
     ▼
7-LAYER PERCEIVER (~105M) → FRESH 64-float page → APPEND to notebook
     │
     ▼
ANSWER HEAD reads each cycle's OWN page directly (no delta, no blending)
GENERATION produces natural sentences with embedded computation per cycle
```

**Critical design: APPEND, don't blend.** Each cycle's perceiver outputs a FRESH page appended to the notebook. No residual gate. No blending. No delta. The hypernetwork reads the full notebook via attention — like flipping through a real notebook. The answer head reads each cycle's own page directly — same quality at every depth.

**Key components:**
- **Llama 3.2 1B base** (frozen, NOT instruct — base can't chain, instruct already can)
- **Growing notebook** — each cycle appends a fresh 64-float page, nothing overwritten
- **64 anonymous LoRA atoms** (Fourier-initialized, orthogonal) — the pattern library
- **Per-cycle intermediate targets** — each cycle predicts ONE intermediate result
- **Text injection** (cumulative) — all previous results as actual text tokens
- **Cycle message generator** — 16-float direct signal from last layer, bypasses perceiver
- **Hybrid loss** — gen loss powers learning (1000x gradient), answer head verifies correctness
- **Natural sentence gen targets** — full expansion per cycle (the model BREATHES)
- **Tanh scaling with hard clamp [-3, 3]** — no softmax, no mode collapse, no saturation
- **Haar wavelet preprocessing** — 2x input compression, 4x faster attention
- **Pi-harmonic page encoding** — frequency identity per dimension (coarse→fine)
- **skip_pass_embed=True** — hypernetwork reads pages, can't shortcut via pass number

---

## Three Principles

### 1. Decomposition Is Everything
Each cycle handles one piece. "160 - 63" is easy. Knowing to do "160 - 63" is the hard part. The model learns to break hard problems into easy pieces.

### 2. Patterns Are the Vocabulary of Decomposition
64 atoms BLEND continuously — an infinite space of attention modifications. Each point is a unique pattern recognizer. The hypernetwork navigates to the right blend per cycle.

### 3. Match the Largest Pattern That Fits (Panama Hats)
"Half as many clips in May" is ONE pattern, not five words. Larger patterns = fewer cycles = less error compounding. Fourier atom init: low-frequency atoms for large patterns, high-frequency for fine detail.

---

## Proven Results

| Task | Base | Breathing | Notes |
|------|------|-----------|-------|
| Single-step arithmetic | 70% | **100%** | Llama 1B, 64-float state |
| Two-step arithmetic | 0% | **94.8%** | Target-cos contrastive |
| Three-step arithmetic | 0% | **83.4%** | 93.8% per-step |
| L2 word ops | 0.6% | **53.4%** | CoT targets breakthrough |
| L3 named qty | 18.8% | **96.0%** | Dual LoRA verification |
| **L4 per-cycle** | 6.0% | **91.0%** | **Page delta breakthrough** |
| **L4.5 per-cycle** | — | **training** | **Growing notebook, cycle 2 = 85.5% ep1** |
| GSM8K | 2.2% | **17.8%** | 8.1x, curriculum L0→L4→GSM8K |

---

## Six Breakthroughs

| # | Fix | Impact | Detail |
|---|-----|--------|--------|
| 1 | Per-cycle intermediate targets | Cycles non-redundant | Each cycle predicts ONE intermediate number |
| 2 | Hybrid generation loss | 1000x gradient to atoms | Gen loss = engine, answer head = verification |
| 3 | Page delta for answer head | 5% → 89% cycle 2 | Isolates new computation from persisted copy |
| 4 | Text injection (cumulative) | Llama reads intermediates | "Step 1 result: 160\n" as actual text |
| 5 | Natural sentence gen targets | Full expansion per cycle | The model BREATHES (inhale in language, exhale through bottleneck) |
| 6 | Growing notebook (remove gate) | 85.5% cycle 2 on epoch 1 | Fresh pages, no convergence, no degradation |

---

## Curriculum

```
L0: single-step arithmetic (70% → 100%)     ✓
L1: two-step arithmetic (0% → 94.8%)        ✓  target-cos contrastive
L2: word ops (0.6% → 53.4%)                 ✓  CoT targets
L3: named quantities (18.8% → 96.0%)        ✓  dual LoRA verification
L4: two-step word (6% → 91%)                ✓  per-cycle + page delta
L4.5: three-step word                       → IN PROGRESS (growing notebook)
L4.7: four-step word                        → NEXT
L4.9: five-step word                        → NEXT
GSM8K: (2.2% → 17.8% → target >30%)        → after curriculum
```

---

## What to Do Next (Priority)

### 1. L4.5 with Growing Notebook (IN PROGRESS)
Cycle 2 hit 85.5% on epoch 1. Watching cycle 3 — should learn now that pages don't converge.

### 2. Push through L4.7 → L4.9
Each level adds one cycle. Notebook grows. Pattern library expands.

### 3. GSM8K Decomposition
Run `scripts/annotate_gsm8k_cycles.py` (Claude API) for 7,473 problems.

### 4. Confidence Head + Variable Cycles
Train confidence to decide when to stop. Enable 3-8 cycles per problem for GSM8K.

### 5. Gentle Training Wheel Removal (Phase 2)
Blend teacher and autonomous decompositions. Increase autonomy based on competence (0% → 20% → 50% → 80% → 100%).

### 6. MATH-500 (July 1 deadline)

---

## Design Decisions

- **APPEND pages, don't blend.** Residual gate caused convergence (page_cos 0.91 at depth 3). Removing it: fresh independent pages. Hypernetwork reads full history via attention.
- **Base model, not Instruct.** Instruct chains at 42%. Base: 0% chained. Architecture provides chaining.
- **Fourier atom init.** 384 orthogonal basis functions. 45/64 active. Multi-scale pattern library.
- **Per-cycle targets, not CoT.** CoT makes cycles redundant. Per-cycle targets force each cycle to do one job.
- **Hybrid loss (flipped per cycle).** Cycle 1: gen=1.0, ah=0.5. Cycle 2+: gen=0.1, ah=5.0.
- **Natural sentences.** Each cycle BREATHES — full inhale (natural language), full exhale (compress to 64 floats).
- **Text injection (cumulative).** Cycle N sees ALL previous results as text. Llama needs TEXT, not vectors.
- **Hard clamp [-3, 3].** Tanh saturation permanently solved. Previous: sreg=49,944. After: 0.2.
- **skip_pass_embed=True.** Without it, hypernetwork uses pass number as shortcut, ignoring pages.
- **Separate LRs.** Perceiver: 1e-4, atoms: 1e-4, hypernetwork: 1e-3.
- **No graduation/detach.** Every scheme destabilized training. Plain full-weight training works best.
- **64-float bottleneck.** Forces incremental thinking. Notebook GROWS (8 cycles = 512 total).
- **Smooth transitions only.** No discrete hyperparameter jumps. Sigmoid ramps, not if-statements.

---

## Loop-Alive Checklist (EVERY training script)

```
□ skip_pass_embed=True
□ scale_reg active (hard clamp [-3, 3])
□ lam_answer_head ≥ 1.0
□ skip connections / direct_path in hypernetwork
□ No residual gate (APPEND pages, don't blend)
```

---

## Known Bugs & Gotchas

| Bug | Impact |
|-----|--------|
| DataCollatorForLanguageModeling overwrites label masking | Fails silently |
| Temperature > 0 at eval | Baseline shift |
| Small eval sets (N<100) | Misleading accuracy |
| GQA K,V dims = (4, 512) not (4, 2048) | 8 KV heads |
| Answer extraction edge cases | Strip periods, \boxed{}, ####, last-number |
| Softmax mode collapse | One mode dominates (quad LoRA failure) |
| Residual gate at depth 3+ | Pages converge, delta = noise → REMOVED |
| Cycle 2 copies cycle 1 without delta/fresh pages | 60% of errors exact copies |
| Generation rambles past correct answer | Extract FIRST equation, not last number |
| Discrete graduation thresholds | Destabilize equilibrium → don't use |
| SymPy through LLM generation | Contamination → use separate decoder |
| Text injection only last step (not cumulative) | Cycle 3 loses step 1's result |

---

## AWS Setup

```bash
ssh -i ~/.ssh/mycelium-key.pem ubuntu@<IP>  # IP changes, check AWS console
tmux new -s train
python scripts/train_per_cycle.py --level L4.5
tail -f logs/train_per_cycle_*.log
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
v21.2 Target-cosine contrastive          → 94.8% two-step ✓ (pages alive)
v22   Dual LoRA verification             → 96.0% L3 ✓
v22.2 GSM8K curriculum                   → 17.8% ✓ (8.1x)
v23   Four-mode LoRA                     → mode collapse, abandoned
v24   64-Atom LoRA + Fourier init        → 93% L3, 91% L4 ✓
v25   Per-cycle targets                  → 89% L4 ✓ (page delta breakthrough)
v25.1 Cycle message (16-float bypass)    → direct signal, no perceiver
v25.2 Text injection (cumulative)        → Llama reads intermediates natively ✓
v25.3 Natural sentence gen targets       → full expansion, the model breathes ✓
v25.4 Growing notebook (remove gate)     → fresh pages, no degradation ✓
```

---

## Failed Experiments (Don't Repeat)

| What | Result | Lesson |
|------|--------|--------|
| Text [EXPAND]/[COLLAPSE] | Matched vanilla | Must be differentiable |
| Llama instruct + breathing | 11% | Already chains — use base |
| Unfreezing Llama 1e-6 | 40% volatile | Keep frozen |
| Quad LoRA softmax | Mode collapse | Use tanh, no competition |
| Answer head loss only (no gen) | 0.0002 gradient | Need hybrid gen+head loss |
| Bare number targets ("97") | 3-5% cycle 2 | LLM guesses, doesn't compute |
| Equation targets ("160-63=97") | 3-5% cycle 2 | Format, not computation |
| Raw page for cycle 2+ answer | 3-5% cycle 2 | Copies cycle 1 (60% exact) |
| SymPy through LLM head | Contamination | Separate decoder needed |
| SymPy decoder (premature) | 0% | Pages not rich enough yet |
| Dynamic detach at 90% | Crashes | Disrupts equilibrium |
| Loss skip for graduated | Collapse to 0% | No anchor |
| Residual gate blending | cos(2,3)=0.91 | Pages converge → APPEND instead |
| Message in answer head | 0.5% after 6 ep | Fresh head can't learn dual input |

---

## File Structure

```
scripts/
  atom_lora.py                   # Model: AtomLoRAModel, AnswerHead, CycleMessageGenerator
  train_per_cycle.py             # v25 training: per-cycle + hybrid loss + growing notebook
  generate_per_cycle_data.py     # Data gen: L3-L4.9 with cycle_targets + gen_targets
  annotate_gsm8k_cycles.py       # Claude API: decompose GSM8K problems
  diag_cycle2.py                 # Diagnostic: correct/wrong predictions
  diag_page_variance.py          # Diagnostic: per-dim variance
plan/
  project_outline.md             # Core thesis: decomposition through pattern matching
  per_cycle_targets_handoff.md   # v25: per-cycle intermediate targets
  append_pages_handoff.md        # v25.4: growing notebook
  cycle_message_handoff.md       # 16-float message channel
  confidence_autonomy_handoff.md # Confidence head + training wheel removal
  fourier_init_handoff.md        # Fourier atom init
  sympy_decoder_handoff.md       # SymPy decoder (parked)
data/
  per_cycle/                     # Generated JSONL: L3-L4.9
```
