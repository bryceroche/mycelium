# Handoff: Strategy Channel Architecture

## Session Summary (April 4, 2026)

Added a **strategy channel** to feed the hypernetwork with richer signal. The bottleneck was identified: 64 floats → 256 LoRA scales = 0.25 floats per scale (starving).

**Solution:** Add 512-float strategy vector alongside the 64-float state.
- State (64): accumulates on hypersphere, persistent memory
- Strategy (512): overwrites each pass, attention instructions

Now: 576 floats → 256 scales = 2.25 floats per scale (well-fed).

---

## What Works

### Baseline: 53% on Two-Step Arithmetic
- Script: `scripts/train_arithmetic_probe.py`
- Architecture: hooks-based LoRA + 64-float state + probe supervision
- Training: ~4.2 it/s on A10G

### Hooks-Based LoRA
- File: `src/lora_hooks.py`
- Applies LoRA via forward hooks on Q, K, V, O projections
- Proven to work - gradients flow correctly

---

## What Doesn't Work

### Clean LoRA (Custom Forward Pass)
- Attempted to avoid hooks by rewriting the entire forward pass
- Result: 0% accuracy, numerical instability
- Problem: Manual attention computation differs from HF implementation
- **Don't pursue this path** - hooks are slower but they work

---

## New Files

```
src/compressor_v3.py           # Two-headed: state (64) + strategy (512)
src/state_conditioned_lora_v3.py  # Hypernetwork accepts state + strategy
scripts/train_strategy_v3.py   # Training with hooks + strategy channel
```

---

## Architecture

```
Pass N:
  state (64) + strategy (512)
         |
         v
  HYPERNETWORK (576 → 256 scales)
         |
         v
  LoRA hooks modify Q, K, V, O
         |
         v
  Llama forward pass
         |
         v
  COMPRESSOR (hidden states → state_delta + new_strategy)
         |
         v
  state = normalize(state + delta) * sqrt(64)  [hypersphere]
  strategy = new_strategy  [overwrites]
```

---

## Next Steps

1. **Run strategy channel training to completion**
   ```bash
   ssh mycelium-vm
   cd /home/ubuntu/mycelium
   PYTHONUNBUFFERED=1 python scripts/train_strategy_v3.py
   ```

2. **Compare to 53% baseline**
   - If > 53%: strategy channel helps hypernetwork
   - If ≈ 53%: bottleneck is elsewhere (maybe compressor)
   - If < 53%: debug - strategy might be noise

3. **If strategy channel works, scale to GSM8K**
   - Same architecture, harder problems
   - May need more breaths (3-5 instead of 2)

---

## Key Insight

The hypernetwork was starving. Each LoRA scale had only 0.25 floats of input information. With the strategy channel, each scale has 2.25 floats - enough signal to make nuanced per-layer decisions about how to modify attention.

The state (64 floats) remains the bottleneck for **memory** - what persists across passes. The strategy (512 floats) provides rich **instructions** - how to configure attention this pass.

---

## VM Details

```
Instance: mycelium-phase0 (i-08c1c295a4113a908)
IP: 18.234.34.226 (may change on restart)
SSH: ssh -i ~/.ssh/mycelium-key.pem ubuntu@<IP>
     or: ssh mycelium-vm (if ~/.ssh/config is set)
```

Start VM:
```bash
aws ec2 start-instances --instance-ids i-08c1c295a4113a908
```

---

## Training Results So Far

| Experiment | Best Accuracy | Notes |
|------------|---------------|-------|
| Baseline (no strategy) | 53% | hooks + probe supervision |
| Comm loop + unfrozen | 40% | volatile, unstable |
| State conditioning only | 50% | below baseline |
| Clean LoRA (no hooks) | 0% | numerical issues |
| Strategy channel | TBD | killed before completion |
