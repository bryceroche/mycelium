# Handoff: Warm Start the Breathing Controller

## One-Sentence Summary

Transfer what we can from the old perceiver and hypernetwork checkpoints into the new controller, and use a gentle learning rate warmup so the controller learns to use its new architecture without destroying the transferred knowledge.

---

## The Problem: Training From Scratch Is Too Slow

```
Old system (warm from L4.5):   gen=0.37 at epoch 1 (already knew format)
New controller (random init):  gen=2.25 at epoch 6 (still learning tokens)

The controller is 190M params of random weights.
It's relearning everything the perceiver spent 50+ epochs mastering:
  - How to read Llama's 16 layers
  - How to weight different layers
  - How to compress to meaningful 64-float pages
  - GSM8K sentence structure
  - The #### format
  - When to produce EOS

All of this was KNOWN by the old system. We threw it away.
```

## What Can Transfer

The old perceiver and the new controller's page head do the SAME JOB:

```
Old perceiver:          Llama hidden states → cross-attention → 64-float page
Controller page path:   Llama hidden states → current_project → trunk → page_head → 64-float page
```

The layer_weights, the cross-attention patterns, and the compression learned by the perceiver are partially transferable. Not a perfect match (different architecture) but better than random init.

The old hypernetwork's cross-attention over pages is also partially transferable to the controller's history_attn. The hypernetwork learned to attend over notebook entries — the controller does the same thing with richer entries.

## Warm Start Strategy

### Step 1: Transfer Perceiver Layer Weights

```python
# The perceiver learned which Llama layers matter most
# Transfer this knowledge to the controller
if 'perceiver.layer_weights' in old_checkpoint:
    controller.layer_weights.data = old_checkpoint['perceiver.layer_weights']
    print("Transferred layer weights from perceiver")
```

### Step 2: Transfer What Fits, Random Init the Rest

```python
def partial_warm_start(controller, old_checkpoint):
    """
    Transfer compatible weights from old perceiver + hypernetwork.
    Leave incompatible weights at random init.
    Log what transferred and what didn't.
    """
    transferred = 0
    skipped = 0
    
    controller_state = controller.state_dict()
    
    for name, param in old_checkpoint.items():
        # Map old names to new names
        new_name = map_old_to_new(name)
        
        if new_name is None:
            skipped += 1
            continue
        
        if new_name in controller_state:
            if controller_state[new_name].shape == param.shape:
                controller_state[new_name].copy_(param)
                transferred += 1
            else:
                # Shape mismatch — partial copy if possible
                if try_partial_copy(controller_state[new_name], param):
                    transferred += 1
                else:
                    skipped += 1
        else:
            skipped += 1
    
    controller.load_state_dict(controller_state)
    print(f"Transferred {transferred} params, skipped {skipped}")


def map_old_to_new(old_name):
    """Map perceiver/hypernetwork param names to controller names."""
    mapping = {
        # Perceiver layer weights → controller layer weights
        'perceiver.layer_weights': 'layer_weights',
        
        # Perceiver cross-attention → controller history attention
        # (partial match — shapes may differ)
        'perceiver.cross_attn': 'history_attn_layers.0.attn',
        
        # Hypernetwork scale MLP → controller scale head
        # (partial match — input dim differs)
        'hypernetwork.scale_mlp': 'scale_head',
    }
    
    for old_prefix, new_prefix in mapping.items():
        if old_name.startswith(old_prefix):
            suffix = old_name[len(old_prefix):]
            return new_prefix + suffix
    
    return None  # no mapping found
```

### Step 3: Transfer Atom Weights (CRITICAL)

The 64 LoRA atoms (82M params) were trained with the old system. They should transfer directly — the atoms modify Llama's attention, which doesn't depend on whether the perceiver or controller produces the scales.

```python
# Atoms transfer directly — same architecture, same job
for name, param in old_checkpoint.items():
    if name.startswith('atoms.') or name.startswith('lora_'):
        if name in new_model.state_dict():
            new_model.state_dict()[name].copy_(param)
            print(f"Transferred atom: {name}")
```

### Step 4: DON'T Transfer Answer Head

We removed the answer head. Don't transfer it. The generation path is the output now.

---

## Gentle LR Ramp-Up

The transferred weights are from a DIFFERENT architecture. Hitting them with full LR immediately might destroy the useful patterns. Ramp up gently:

```python
class WarmupScheduler:
    """
    Gentle LR ramp-up over the first N epochs.
    
    Epoch 1-3:   low LR (let transferred weights stabilize)
    Epoch 4-8:   ramp up (controller learns its new architecture)
    Epoch 8+:    full LR (learning at full speed)
    """
    def __init__(self, optimizer, base_lr, warmup_epochs=8):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup: 10% → 100% over warmup_epochs
            factor = 0.1 + 0.9 * (epoch / self.warmup_epochs)
        else:
            factor = 1.0
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr * factor
        
        return self.base_lr * factor

# Usage:
scheduler = WarmupScheduler(optimizer, base_lr=1e-4, warmup_epochs=8)

for epoch in range(num_epochs):
    current_lr = scheduler.step(epoch)
    print(f"Epoch {epoch}: LR = {current_lr:.6f}")
    train_one_epoch(...)
```

The warmup schedule:

```
Epoch 1:   LR = 1e-5  (10% — stabilize transferred weights)
Epoch 2:   LR = 2e-5  (21%)
Epoch 3:   LR = 3e-5  (33%)
Epoch 4:   LR = 4.5e-5 (44%)
Epoch 5:   LR = 5.6e-5 (56%)
Epoch 6:   LR = 6.8e-5 (68%)
Epoch 7:   LR = 7.9e-5 (79%)
Epoch 8:   LR = 1e-4  (100% — full speed)
Epoch 9+:  LR = 1e-4  (full speed)
```

### Per-Component LR Groups

The transferred components should have LOWER LR than the fresh components:

```python
# Transferred weights: gentle (don't destroy what they know)
# Fresh weights: aggressive (need to learn fast)
param_groups = [
    {
        'params': controller.transferred_params(),  # layer_weights, etc.
        'lr': 1e-5,  # gentle — preserve transferred knowledge
        'name': 'transferred'
    },
    {
        'params': controller.fresh_params(),  # trunk, heads, etc.
        'lr': 1e-4,  # aggressive — learn new architecture fast
        'name': 'fresh'
    },
    {
        'params': model.atoms.parameters(),  # transferred from old checkpoint
        'lr': 3e-5,  # moderate — atoms are trained but need to adapt to controller
        'name': 'atoms'
    },
]

optimizer = AdamW(param_groups, weight_decay=0.01)
```

The atoms get moderate LR — they were trained with the old perceiver/hypernetwork and need to adapt to the controller's different scale distribution. Not random init (so not full LR) but not fully converged with the new controller (so not minimal LR).

---

## Expected Trajectory

```
WITHOUT warm start (current):
  Epoch 1:  gen=2.35  (random init, learning from zero)
  Epoch 6:  gen=2.25  (barely moved, still learning tokens)
  Epoch ?:  gen=0.37  (when it reaches old system's start — unknown)

WITH warm start + gentle ramp:
  Epoch 1:  gen=1.0-1.5?  (transferred weights provide head start)
  Epoch 3:  gen=0.5-0.8?  (controller adapts to new architecture)
  Epoch 8:  gen=0.3-0.5?  (full LR, learning at full speed)
  Epoch 15: gen<0.2?      (converging on GSM8K patterns)

The warm start should skip the "learning basic format" phase entirely.
The perceiver already knew how to produce pages.
The atoms already knew how to steer attention.
The controller just needs to learn to coordinate them through its new trunk.
```

---

## What Transfers vs What's Fresh

```
TRANSFERS (from old checkpoint):
  layer_weights:        which Llama layers matter (perceiver learned this)
  atom templates:       82M of attention patterns (proven on L4.5)
  cross-attention:      how to attend over notebook (partial, shape may differ)

FRESH (random init, needs to learn):
  trunk:                shared understanding → page + scales
  page_head:            compressed representation of understanding
  scale_head:           atom selection from understanding
  focus_head:           Möbius focus point
  current_project:      how to read current hidden states
  history_entry_project: how to combine page + hidden for history

ADAPTS (transferred but needs adjustment):
  atoms:                trained for old system, need to adapt to controller scales
```

---

## Implementation Steps

```python
# 1. Load old checkpoint
old_ckpt = torch.load('checkpoints/per_cycle_gsm8k_best.pt')

# 2. Build new model with controller
model = build_model_with_controller(...)

# 3. Transfer atom weights (direct copy)
transfer_atoms(model, old_ckpt)

# 4. Transfer perceiver knowledge to controller (partial)
partial_warm_start(model.controller, old_ckpt)

# 5. Set up per-component LR groups
optimizer = setup_optimizer_groups(model)

# 6. Set up warmup scheduler
scheduler = WarmupScheduler(optimizer, base_lr=1e-4, warmup_epochs=8)

# 7. Train with warmup
for epoch in range(50):
    lr = scheduler.step(epoch)
    train_one_epoch(model, optimizer, ...)
```

---

## What to Monitor

```
1. Gen loss at epoch 1:
   Random init: 2.35 (current)
   Warm start target: < 1.5 (head start from transferred knowledge)
   If still > 2.0: transfer didn't help, weights incompatible

2. Gen loss trajectory:
   Should drop faster than random init
   Target: gen < 1.0 by epoch 10 (vs epoch 60+ from scratch)

3. Atom adaptation:
   Atoms were trained for old perceiver scales
   Controller produces different scales (xpass_cos = -0.2 vs old 0.999)
   Monitor: do atoms adapt to the new scale distribution?
   If accuracy drops then recovers: atoms adapting (good)
   If accuracy drops and stays down: atoms can't adapt (reduce atom LR)

4. Page quality:
   Transferred layer_weights should give reasonable pages immediately
   Monitor: page_cos at epoch 1 (should be < 0.85 from transferred weights)
   The perceiver's layer weighting helps even with new page_head

5. Scale diversity:
   Should maintain xpass_cos < 0 (the controller naturally differentiates)
   Warm start shouldn't destroy this — hidden states still differ per problem
   If xpass_cos collapses to 0.999: transferred weights pulling back to old behavior
```

---

## Fallback

```
If warm start doesn't help (gen still > 2.0 at epoch 1):
  The architectures are too different for weight transfer.
  Fall back to random init with full LR.
  Accept the 70+ epoch convergence time.
  
  But at 10 min/epoch, 70 epochs = 12 hours.
  Overnight run. Not ideal but workable.

If warm start causes instability (loss spikes, accuracy crashes):
  Reduce ALL learning rates by 2x.
  The transferred weights are being destroyed too fast.
  More gentle ramp: warmup_epochs = 15 instead of 8.
```

---

## What NOT to Do

```
- Do NOT transfer the old hypernetwork's scale_mlp directly.
  The input dimension changed (96 → 2048). The weights are incompatible.
  The scale_head is fresh init.

- Do NOT use lr_scale < 0.5 for the entire model.
  The FRESH components (trunk, heads) need full LR.
  Only the TRANSFERRED components need gentle LR.
  Per-component LR groups handle this.

- Do NOT skip the warmup.
  Transferred + fresh weights at the same LR = chaos.
  The warmup lets transferred weights stabilize while fresh weights catch up.

- Do NOT transfer the answer head.
  We removed it. Generation is the output now.
  
- Do NOT transfer the bypass or message generator.
  We removed them. The controller reads hidden states directly.
```
