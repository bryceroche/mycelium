# Handoff: Fresh Curriculum with Full Wave Architecture

## One-Sentence Summary

Train the complete v24.5 architecture from scratch through L3 → L4 → L5 (GSM8K) with pi-harmonic page encoding, Apéry-weighted wavelets, 64 atoms, entropy flow monitoring, and all training optimizations. No warm start from old encoding — clean slate with the new frequency basis. Deadline: June 1, 2026.

---

## Why Fresh Start

The pi-harmonic page encoding uses a completely different frequency basis than the old Fourier encoding:

```
Old:  freqs = exp(-i * log(10000) / d)   (transformer-style, geometric spacing)
New:  freqs = k * π / page_size           (DCT-like, arithmetic spacing, orthogonal)
```

Warm-starting from old checkpoints caused catastrophic regression (24% → 16%) because the perceiver and hypernetwork learned to read/write in the old frequency space. The new encoding is a different language — everything must relearn.

---

## The Complete Architecture (v24.5)

Everything we've built, from scratch:

```
INPUT PROCESSING:
  ✓ Haar wavelet preprocessing (2x compression, Apéry-weighted levels)
  
EXPAND (per thinking cycle):
  ✓ 64 rank-6 LoRA atoms (~82M params, Tanh scaled, no softmax)
  
CONTROL:
  ✓ 10M atom hypernetwork
  ✓ Fourier pass encoding (smooth temporal waveforms)
  ✓ 2-layer page attention (reads all accumulated pages)
  ✓ Pass-conditioned (different atoms per cycle)

COLLAPSE:
  ✓ 7-layer Perceiver compressor (~105M params)
  ✓ Pi-harmonic page encoding (DCT-like orthogonal frequency basis)
  ✓ 64-float pages, append-only notebook
  ✓ Per-page hypersphere normalization

MONITORING:
  ✓ Entropy tracker (per-cycle information change)
  ✓ Surprise detector (unexpected entropy drops)
  ✓ Atom spectrogram logging (activation patterns across passes)

STOPPING:
  ✓ GRU confidence head (confidence + smoothness)
  ✓ Four-quadrant decision logic

EXTRACTION:
  ✓ Answer head (sign + length + per-digit prediction)
  ✓ Generation path (CoT trace via pseudo-tokens, LoRA OFF)

TRAINING:
  ✓ Contrastive page loss (target-cos 0.7, per-page)
  ✓ Anti-copying (soft quadratic above 0.7)
  ✓ Gradient scaling per cycle (capped at 4x)
  ✓ Fresh data per epoch (procedural generation)
  ✓ All regularization (dropout 0.1, weight decay, page noise 0.05, atom dropout 10%)
  ✓ CoT targets (full reasoning traces)
  ✓ Early stopping with patience
```

---

## Training Recipe

### Loss

```python
total_loss = (generation_loss              # 1.0 weight — CoT reasoning trace
              + 1.0 * answer_head_loss     # 1.0 weight — digit extraction (anti-collapse)
              + 0.05 * contrastive_loss    # per-page target-cos 0.7
              + 0.1 * confidence_loss      # correctness + smoothness
              + 0.1 * smoothness_loss)     # even entropy flow
```

Key: answer_head_loss at 1.0 (equal to generation loss). This was the breakthrough that pushed GSM8K to 22%. The answer head provides direct gradient to pages demanding per-problem differentiation. This is the primary anti-collapse mechanism.

### Optimizer

```python
optimizer = AdamW([
    {'params': perceiver_params, 'lr': 1e-4, 'weight_decay': 0.01},
    {'params': atom_A_params, 'lr': 1e-4, 'weight_decay': 0.05},      # slow (large param set)
    {'params': atom_B_params, 'lr': 1e-4, 'weight_decay': 0.05},      # slow (large param set)
    {'params': hypernetwork_params, 'lr': 1e-3, 'weight_decay': 0.1},  # fast (control layer)
    {'params': answer_head_params, 'lr': 1e-3, 'weight_decay': 0.01},
    {'params': confidence_params, 'lr': 1e-3, 'weight_decay': 0.01},
    {'params': page_to_tokens_params, 'lr': 1e-4, 'weight_decay': 0.01},
])
# Llama: FROZEN
# Atoms: 1e-4 (slow — learned stability lesson)
# Hypernetwork: 1e-3 (fast — adapts to atoms)
```

### Atom Initialization

```python
# Atom templates: small random init
atom_A = nn.Parameter(torch.randn(...) * 0.01)
atom_B = nn.Parameter(torch.randn(...) * 0.01)

# Hypernetwork final layer bias: 0.05 (gentle activation, not too noisy)
hypernetwork.scale_net[-2].bias.data.fill_(0.05)
# tanh(0.05) ≈ 0.05 → atoms barely active but gradient flows
```

### Regularization

```python
# All four layers active:
# 1. Dropout 0.1 in hypernetwork MLP
# 2. Per-component weight decay (perceiver 0.01, atoms 0.05, hypernetwork 0.1)
# 3. Page noise 0.05 during training
# 4. Atom dropout 10% during training
```

---

## Curriculum

### L3: Named Quantities (Week 1 Start)

```bash
python3 scripts/train_atom_lora.py \
    --level L3 \
    --epochs 8 \
    --patience 5 \
    --batch_size 8 \
    --passes 3 \
    --lam 0.05 \
    --lam_answer 1.0 \
    --lam_conf 0.1 \
    --fresh_data \
    --use_wavelet \
    --use_pi_harmonic
    # No --warm flag — fresh start
```

```
Data:       20K per epoch, fresh procedural generation
Passes:     3 (L3 is simple, doesn't need more)
Target:     >85% gen_acc (dual LoRA got 96%, atom should match)
Move on:    when gen_acc plateaus for 3 epochs or hits 90%
Checkpoint: save as checkpoints/atom_piharmonic_L3_best.pt
```

### L4: Two-Step Word Problems

```bash
python3 scripts/train_atom_lora.py \
    --level L4 \
    --warm checkpoints/atom_piharmonic_L3_best.pt \
    --epochs 6 \
    --patience 3 \
    --batch_size 8 \
    --passes 3 \
    --lam 0.05 \
    --lam_answer 1.0 \
    --fresh_data \
    --use_wavelet \
    --use_pi_harmonic
```

```
Data:       20K per epoch, fresh
Passes:     3
Target:     >90% (dual LoRA got 100%, aim high)
Move on:    when gen_acc plateaus or hits 95%
Checkpoint: save as checkpoints/atom_piharmonic_L4_best.pt
```

### L5: GSM8K

```bash
python3 scripts/train_atom_lora.py \
    --level L5 \
    --warm checkpoints/atom_piharmonic_L4_best.pt \
    --epochs 20 \
    --patience 5 \
    --batch_size 8 \
    --passes 5 \
    --lam 0.01 \
    --lam_answer 1.0 \
    --fresh_data \
    --use_wavelet \
    --use_pi_harmonic \
    --eval_size 100
```

```
Data:       7,473 GSM8K + augmentation per epoch
Passes:     5 (GSM8K needs more thinking)
Target:     >22% (previous best), stretch >30%
Eval:       100 problems for quick checks, 500 for final validation
Checkpoint: save as checkpoints/atom_piharmonic_L5_best.pt
```

---

## What to Monitor Per Level

```
EVERY LEVEL:
  1. gen_acc:       generation accuracy (primary metric)
  2. head_acc:      answer head accuracy (should climb WITH gen_acc)
  3. page_cos:      page differentiation (should be <0.95, NOT 1.0)
  4. active_atoms:  how many atoms have |scale| > 0.1
  5. scale_std:     atom scale differentiation
  6. ans_loss:      should decrease but NOT to 0.0000 (overfitting signal)
  7. entropy_flow:  per-cycle entropy drops (should be smooth)

L3 SPECIFIC:
  - Expect few atoms active (8-15) — simple problems
  - page_cos may be high (0.9+) — similar problems, that's okay
  - head_acc should climb faster than on GSM8K — small answer range [1,200]

L4 SPECIFIC:
  - Should be quick (1-2 epochs to high accuracy)
  - If page_cos=1.0: the answer_head_loss at 1.0 should prevent this
  - If not: increase lam_answer to 2.0

L5 SPECIFIC:
  - Expect more atoms active (30-40) — harder problems
  - page_cos should differentiate naturally (diverse problem structures)
  - Watch for overfitting: ans_loss → 0 but gen_acc plateauing
  - Blend of active atoms across passes should show temporal structure
  - Run atom spectrogram diagnostic every 3 epochs
```

---

## When to Advance Levels

```
L3 → L4:  gen_acc > 85% OR plateaued for 3 epochs
L4 → L5:  gen_acc > 90% OR plateaued for 3 epochs
L5:       train until patience expires, save best checkpoint
```

Don't over-train on any level. The curriculum is about building warm-start representations, not maximizing accuracy at each step. L3 and L4 are stepping stones to L5.

---

## Page Cache + Replay Buffer (Enable on L5)

Once training on L5, enable the page cache to speed up later epochs:

```python
# After epoch 3 on L5, run per-step accuracy diagnostic
per_step_acc = measure_per_step_accuracy(model, eval_data, max_passes=5)

# If any pass is >90% accurate, cache it
# Expected: passes 1-2 might graduate by epoch 5-6
# Cache graduated passes, focus compute on later passes
```

Don't enable the cache on L3/L4 — those levels are fast enough already. The cache matters on L5 where 5 passes × 7473 problems × 20 epochs is expensive.

---

## Results to Beat

```
Previous bests (all with old encoding):
  L3: 96.0% (dual LoRA)
  L4: 100% (dual LoRA)
  L5: 22.0% (64 atoms, old Fourier, answer_head_loss=1.0)

Targets with pi-harmonic (fresh curriculum):
  L3: >85% (matching range, atoms start from scratch)
  L4: >90% 
  L5: >25% (beat previous best with principled encoding)
  Stretch L5: >30% (with page cache, more epochs, filled L4-L5 gap)
```

---

## Key Lessons (Don't Repeat)

```
1.  CoT targets, not terse numbers (number-spam)
2.  answer_head_loss at 1.0 — primary anti-collapse mechanism
3.  Atom LR at 1e-4 (slow), hypernetwork at 1e-3 (fast)
4.  Bias init 0.05 on hypernetwork final layer (gentle atom activation)
5.  Don't warm-start from checkpoints with different encoding basis
6.  Overfitting starts epoch 3-4 — watch ans_loss vs gen_acc divergence
7.  Fresh data per epoch — never see the same problem twice
8.  High page_cos is okay on easy problems — don't force differentiation
9.  page_cos=1.0 is NOT okay — means fixed-point collapse
10. The model ALWAYS finds shortcuts — every unconstrained path will be exploited
11. Print 10 generations before trusting accuracy numbers
12. Eval on 100 problems for quick checks, 500 for final validation
13. Always use tmux on the VM
14. Verify baseline on same eval set with same prompt format
```

---

## Infrastructure

```
VM:          AWS g5.xlarge (A10G 24GB, ~$1/hr)
SSH:         ssh -i ~/.ssh/mycelium-key.pem ubuntu@<check AWS for IP>
Project:     ~/mycelium on VM
Logs:        ~/mycelium/logs/
Checkpoints: ~/mycelium/checkpoints/
Always:      tmux new -s train (or tmux attach -t train)
Monitor:     tail -f ~/mycelium/logs/<current_run>.log
Git:         commit after each level completes
```

---

## Timeline

```
Week 1 (Apr 15-21):   L3 → L4 curriculum with pi-harmonic
Week 2 (Apr 22-28):   L5 (GSM8K) push to >25%
Week 3 (Apr 29-May 5): Fill L4→L5 gap, page cache, target >30%
Week 4 (May 6-12):    Unfreeze Llama, tree search, target >35%
Week 5 (May 13-19):   MATH-500 preparation
Week 6 (May 20-Jun 1): MATH-500 evaluation + writeup
Deadline: June 1, 2026
```
