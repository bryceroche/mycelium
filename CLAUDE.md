# CLAUDE.md — Mycelium v4

## Project Overview

Mycelium builds differentiable recurrent reasoning for small language models. A 140M breathing transformer loops 4 layers from Pythia-160M with π-cycled attention, reasoning in representation space and generating tokens only once at the end. The core thesis: **decomposition is everything.**

**Lead:** Bryce (Manhattan Beach, CA) · **Target:** MATH-500 · **Deadline:** September 1, 2026
**Infrastructure:** Shadow Glass (AMD 7900 XTX 24GB, tinygrad + AM driver) · AWS EC2 g5.xlarge (A10G 24GB) for validation

---

## Architecture (v4 — The Breathing Transformer)

```
BREATHING LOOP:

  BREATHE (representation space, no token generation):
    4 layers (Pythia-160M L0-3) × N loops
    π-cycled attention: 12 heads at 12 phase offsets, rotated each loop
    Sine-wave temperature: RISE → PEAK → FALL → TROUGH
    Integration: gated running integral across breaths
    Controller reads hidden states → pages, temperature, phase, stop decision

  SPEAK (once, at the end):
    Generate tokens from final integrated representation
    One copy from a refined original — NOT copies of copies

  STOP when: integral stabilizes (Lyapunov criterion)
```

**Components:**
- **Pythia-160M L0-3 (fine-tuned for looping):** 4 layers with partial weight sharing. Unique Q, K, FFN gate per phase; shared V, FFN basis, norms. ~20M phase-specific + ~6M shared + ~39M embeddings.
- **Controller (~80M):** Reads 768d transformer hidden states in 512d thinking space. Produces pages, temperature modulation, phase angle, integration gate, stop signal. Gets gradient via REINFORCE — never through transformer.
- **Differentiable Lookup Table:** 8-12 prime entries at 768d. Spectral factorization of problem structure. Coupling matrix determines tree shape.
- **No chain-of-thought tokens.** All reasoning in representation space. The Copy Machine Principle prohibits mid-breath generation.

**Cardinal rules:**
1. Controller gradient NEVER flows through transformer
2. No token generation between breaths (Copy Machine Principle)
3. Diversity is structural (π cycling), not learned

---

## Results

### v1 (completed)
```
L3 (1-step):   100.0%   ✓ (from scratch)
L4 (2-step):    99.5%   ✓ (genuine 2-step decomposition)
L4.5 (3-step):  99.5%   ✓ (genuine 3-step decomposition)
GSM8K:           22%     ✓ (multi-step, beats 17.8% CoT ceiling)
```

### v4 Validation (completed — May 6-7, 2026)
```
Looping enriches representations:     ✓ PROVEN
  Signal grows 7x across 8 loops (L0-3)
  Effective rank holds: 16.0 → 16.6
  SNR actually INCREASES: 0.114 → 0.127

Generation requires fine-tuning:       ✓ UNDERSTOOD
  DC component grows linearly → overwhelms generation head
  Fine-tuning target: teach gen head to extract signal from looped repr

Copy Machine Principle:                ✓ PROVEN
  Autoregressive mid-loop generation destroys signal
  Representation-space reasoning preserves it

L0-3 best layer selection:             ✓ PROVEN
  Better SNR stability than full 12-layer model
  L11 is toxic for looping (8x norm explosion)
```

### v4 Training (in progress)
```
Phase 0: Loop consistency training     ← NEXT (Shadow Glass Day 1)
Phase 1: Learn to breathe (L3-L4.5 curriculum)
Phase 2: Controller + lookup table
Phase 3: GSM8K push (>22%)
Phase 4: MATH-500
```

---

## Key Design Decisions

- **Equal-reward (1/N per target):** The ONLY way to maximize reward is to decompose. Proven on L4/L4.5.
- **Copy Machine Principle:** No token generation between breaths. Reasoning in representation space is not just efficient — it's necessary. Empirically proven: hidden states survive looping, autoregressive generation doesn't.
- **π-cycled attention:** Per-head phase offsets provide structural diversity that gradient descent cannot erase. Solves the v1-v3 diversity collapse problem.
- **L0-3 from Pythia-160M:** Best loop stability of any layer selection. SNR increases with loops. L11 is toxic (norm explosion).
- **Separate backward passes:** Controller gradient via REINFORCE, never through transformer. The gen_loss landscape has one dominant basin for any controller path through the LLM.
- **DC component management:** The shared direction in hidden state space grows linearly with loops. The generation head must learn to extract per-problem signal from this. This is THE fine-tuning objective.

---

## File Structure

```
scripts/
  # v4 validation
  validate_looping.py                    # Day 1-2: looping experiments
  smoke_test_breathe_then_speak.py       # Copy Machine Principle validation
  diag_loop_collapse.py                  # Root cause diagnostics (norms, cosine, entropy, rank)

  # v2/v1 (reference)
  controller.py                          # v2 BreathingController (reference)
  atom_lora.py                           # v1 model (AtomLoRAModel, LoRAAtoms, baking)
  train_per_cycle.py                     # v1 training loop
  generate_per_cycle_data.py             # Data gen: L3-L4.9 procedural
  smoke_test_controller.py               # v2 Phase 0 smoke test
  diag_*.py                              # v1/v2 diagnostics

plan/
  pre_shadow_glass_summary.md            # v4 complete summary + Day 1 plan
  mycelium_v2_master_rebuild_handoff.md  # v2 architecture (superseded by v4)
  v4_validation_handoff.md               # Validation experiment plan
  + design documents

data/
  per_cycle/                             # L3-L4.9 + GSM8K JSONL
  looping_validation_results.json        # Validation experiment results
```

---

## Infrastructure

### Shadow Glass (primary — arriving May 8, 2026)
```
AMD 7900 XTX (24GB GDDR6, 960 GB/s, ~120 TFLOPS FP16)
tinygrad + AM driver (no ROCm, no PyTorch)
~4GB VRAM for 140M model, ~20GB headroom
```

### AWS (validation, backup)
```
aws ec2 start-instances --instance-ids i-08c1c295a4113a908
ssh -i ~/.ssh/mycelium-key.pem ubuntu@<IP>
cd ~/mycelium && tmux new -s train
```

---

## Diagnostics

| Diagnostic | What It Reveals | Key Lesson |
|-----------|----------------|------------|
| Centered cross-problem cosine | Per-problem diversity | Raw cosine is MISLEADING (99.8% DC) — always center first |
| Effective rank (SVD) | Dimensionality of repr | Must hold across loops (target: 16+) |
| DC norm | Shared component magnitude | Grows linearly with loops — gen head's enemy |
| Signal norm | Per-problem component | GROWS 7x over 8 loops — looping works |
| SNR (signal/DC) | Can gen head distinguish problems? | L0-3 SNR increases — best layer selection |
| Attention entropy | Degenerate attention patterns | Increases with loops (uniform = homogeneity) |
| Per-head contribution Gini | Inbreeding detection | Low = healthy diversity; high = heads collapsed |

---

## Failed Experiments (Don't Repeat)

### v1-v3 (Llama-based)
| What | Result | Lesson |
|------|--------|--------|
| Answer head (5 versions) | 4% peak | Generation IS the output |
| Atoms at [-3,3] | 14.6% | Too loud, corrupts arithmetic. Use [-0.5,0.5] |
| Q,K-only atoms | 0.4% | WORSE than vanilla. V,O does the work |
| No decomposition incentive | Cycle 2=6%, final=0% | Without 1/N reward, model one-shots |
| Routing gradient through Llama | 500x attenuation | Use REINFORCE/ST estimator |
| clamp + tanh on scales | Dead gradient | Use smooth activation only |
| Loading collapsed controller weights | Still saturated | Always fresh init |

### v4 Validation (Pythia looping)
| What | Result | Lesson |
|------|--------|--------|
| Frozen-weight looping + generation | "had had had" | Generation head needs fine-tuning for looped repr |
| RMSNorm between loops | No help | Preserves DC component — use LayerNorm |
| Mean subtraction between loops | Changes attractor only | DC is a direction, not a bias |
| Layer selections with L11 | Worse SNR | L11's norm explosion is toxic for iteration |
| Autoregressive generation between loops | Signal destroyed | Copy Machine Principle — breathe in repr space |
| DC subtraction before gen head | No change | DC is cross-problem direction, not per-token bias |
