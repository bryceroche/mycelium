# Breathing Models — Next Steps & Future Architecture

## Current Status (April 2026)

```
Task                  Base model    Breathing       Per-step    Variance
──────────────────────────────────────────────────────────────────────────
Single-step           70%           100%            100%        —
Two-step              0%            85.4% ±0.0%     92.4%      zero
Three-step            0%            73.6% ±0.0%     90.3%      zero
GSM8K word problems   4-5%          ???             ???        IN PROGRESS
```

Architecture: Llama 3.2 1B (frozen) + 7-layer Perceiver (105M) + State-Conditioned LoRA (1.1M) + SymPy Probe + 512-float Strategy Side Channel. 64-float state on hypersphere. Additive LoRA (no hooks). Separate LRs (10x for templates). Answer loss + probe loss.

---

## Priority 1: GSM8K (NOW)

Train the proven architecture on GSM8K word problems. No architectural changes. The base model can do arithmetic (70%) and understands language ("half of 48 is 24"). The architecture chains operations. The question: does it chain word problem reasoning?

**Target:** Thinking accuracy > 4-5% (single-shot baseline).
**Stretch target:** >20% (paper territory).
**Deadline:** MATH-500 by April 22.

---

## Priority 2: Perceiver Memory Buffer (WHEN SCALING TO 5+ PASSES)

### The Problem

At 3 passes, the perceiver only sees the current pass's hidden states. The 64-float state carries forward information but is lossy. At 10 passes for hard problems, early pass information is lost — the "amnesia problem" identified in Kimi's Attention Residuals paper (April 2026).

### The Solution

Give the perceiver a memory buffer of previous passes' internal representations. Not just the 64-float state output but the rich 1024-dim query states from each pass's perceiver processing.

```
Current:
  Pass 3 perceiver attends over: [Llama hidden states from pass 3]
  
With memory buffer:
  Pass 3 perceiver attends over: [Llama hidden states from pass 3]
                                 + [perceiver queries from pass 1]
                                 + [perceiver queries from pass 2]
```

### Implementation

```python
class AllLayerPerceiver(nn.Module):
    def __init__(self, ...):
        # ... existing init ...
        self.memory_buffer = []  # stores query states from previous passes
    
    def forward(self, all_layer_hidden_states, pass_num):
        # ... existing perceiver processing through 7 layers ...
        # After layer 4 (midpoint), store query states
        midpoint_queries = queries.detach().clone()  # (batch, 4, 1024)
        
        # Cross-attend to memory buffer (previous passes' query states)
        if len(self.memory_buffer) > 0:
            memory = torch.cat(self.memory_buffer, dim=1)  # (batch, N*4, 1024)
            memory_attended, _ = self.memory_cross_attn(
                query=queries, key=memory, value=memory
            )
            queries = self.memory_norm(queries + memory_attended)
        
        # ... remaining perceiver layers ...
        
        # Store for future passes
        self.memory_buffer.append(midpoint_queries)
        
        return state_delta, strategy
    
    def reset_memory(self):
        """Call at the start of each new problem."""
        self.memory_buffer = []
```

### Memory Budget

```
Per pass: 4 queries × 1024 dim = 4,096 floats (16KB)
10 passes: 40,960 floats (160KB)
Negligible memory cost. Rich information for selective retrieval.
```

### What This Enables

```
Without buffer (current):
  Pass 10 has: 64 floats from pass 9 (accumulated state)
  If pass 1 found something crucial, it's gone

With buffer:
  Pass 10 has: 64 floats (state) + direct attention to ALL previous passes
  "What did I find at pass 1?" → attend to pass 1's query states → retrieve
```

### When to Implement

After GSM8K works at 3 passes. When we scale to 5+ passes on harder problems and accuracy plateaus, the memory buffer is the first thing to try. It's a small change to the existing perceiver — one extra cross-attention layer and a list to store query states.

---

## Priority 3: LoRA Bandwidth Constraints (ANYTIME)

### The Problem

LoRA scales are bounded by Tanh to [-1, 1] but this is still a wide range. Large-magnitude scales can destabilize the inner transformer's pretrained representations.

### The Solution

Add a learnable bandwidth parameter that constrains how much the LoRA can modify attention:

```python
# In StateConditionedLoRA:
self.bandwidth = nn.Parameter(torch.tensor(0.1))  # starts small

def forward(self, state, strategy):
    raw_scales = self.scales_net(combined)        # (batch, 256)
    scales = torch.tanh(raw_scales) * self.bandwidth  # bounded modification
    # bandwidth starts at 0.1: LoRA can only perturb attention by ±10%
    # bandwidth can grow during training if the model needs more range
```

### Annealing Schedule (Alternative to Learned)

```
Epoch 1-5:   bandwidth = 0.05 (barely any modification, safe)
Epoch 5-10:  bandwidth = 0.10
Epoch 10-15: bandwidth = 0.15
Epoch 15+:   bandwidth = 0.20 (earned the right to modify more)
```

### Why This Helps

- Prevents catastrophic LoRA modifications during early training
- Forces the model to learn SUBTLE attention changes first
- The bandwidth is a diagnostic: if the model pushes bandwidth to 0.3+, it needs large modifications. If 0.05 is sufficient, the modifications are fine-grained.

### When to Implement

Could add anytime as a safety feature. Most useful if we see instability when unfreezing Llama or scaling to harder problems.

---

## Priority 4: Unfreeze Llama (AFTER GSM8K PROVEN)

### The Problem

Per-step accuracy is 90% on arithmetic, limited by frozen Llama's pretrained capabilities. Each 1% improvement in per-step compounds through the chain.

### Previous Attempt

Unfreezing at 1e-6 LR caused instability (40% cap, volatile training). But that was BEFORE the side channel and answer loss. The architecture is much stronger now.

### The Plan

```
Step 1: Prove GSM8K works with frozen Llama (establish baseline)
Step 2: Unfreeze at 1e-7 (gentler than before)
Step 3: Combine with LoRA bandwidth constraint (prevents instability)
Step 4: Monitor per-step accuracy — does it climb above 90%?
```

### Expected Impact

```
If per-step goes 90% → 95%:
  Two-step:   90.2% → 95%²  = 90.2%
  Three-step: 73.6% → 95%³  = 85.7%
  Five-step:  59.0% → 95%⁵  = 77.4%

The compounding is dramatic at higher step counts.
```

---

## Priority 5: Boltzmann Exploration (WHEN GSM8K PLATEAUS)

### The Problem

Harder problems may have multiple valid decomposition paths. The current deterministic thinking loop commits to one path. If that path hits a dead end, there's no recovery.

### The Solution

The perceiver outputs a mean delta and variance. The actual delta is sampled from a Boltzmann distribution:

```python
mean_delta, log_var = perceiver(hidden_states, pass_num)
std = torch.exp(0.5 * log_var)
delta = mean_delta + std * torch.randn_like(std) * temperature

# Temperature annealing across passes:
# Pass 1 (T=1.0): explore different thinking directions
# Pass 5 (T=0.1): commit to the best path
```

### Three Applications

1. **Exploration during thinking:** sample different state rotations per pass
2. **Energy landscape:** confidence head defines energy, Boltzmann samples low-energy states
3. **Strategy selection:** generate K candidate LoRA configs, select via Boltzmann

### When to Implement

When GSM8K accuracy plateaus and we suspect the model is stuck in local minima. The plateau itself is the signal that exploration is needed.

---

## Priority 6: Outer Transformer (LONG-TERM)

### The Vision

Replace all hand-designed components (perceiver, hypernetwork, confidence head, strategy channel) with a single small transformer that processes the thinking sequence. Each thinking pass is a "token" in the outer sequence.

```
INNER TRANSFORMER: Llama 3.2 1B (processes language)
OUTER TRANSFORMER: Small (25-50M params, processes thinking)
```

### Why Not Now

- Cold start problem (outer transformer learns 4 roles from random weights)
- Current components are proven and debuggable
- Sequence length (3-10 passes) is too short for transformer to shine
- Would spend weeks matching current 73.6%

### The Distillation Path

```
Phase 1: Hand-designed components (NOW — proven, 73.6%)
Phase 2: Push to GSM8K and MATH with current architecture
Phase 3: Distill current components into outer transformer
         - Run current system on 10K+ problems
         - Record per-pass: state, strategy, LoRA scales, confidence
         - Train outer transformer to MIMIC these outputs
Phase 4: Fine-tune outer transformer end-to-end (surpass teacher)
```

The key insight: the hand-designed components are the TEACHER. The outer transformer is the STUDENT that learns from a working system, not from scratch. This solves the cold start problem.

### When to Implement

After the current architecture has been pushed to its limits on MATH. The distillation data comes for free — we just log the intermediate outputs during normal training and evaluation.

---

## Implementation Priority Summary

```
┌──────────┬─────────────────────────────────┬─────────────────────────────────┐
│ Priority │           What                  │           When                  │
├──────────┼─────────────────────────────────┼─────────────────────────────────┤
│ P1       │ GSM8K word problems             │ NOW                             │
├──────────┼─────────────────────────────────┼─────────────────────────────────┤
│ P2       │ Perceiver memory buffer         │ When scaling to 5+ passes       │
├──────────┼─────────────────────────────────┼─────────────────────────────────┤
│ P3       │ LoRA bandwidth constraints      │ Anytime (safety feature)        │
├──────────┼─────────────────────────────────┼─────────────────────────────────┤
│ P4       │ Unfreeze Llama (1e-7)           │ After GSM8K baseline proven     │
├──────────┼─────────────────────────────────┼─────────────────────────────────┤
│ P5       │ Boltzmann exploration           │ When GSM8K accuracy plateaus    │
├──────────┼─────────────────────────────────┼─────────────────────────────────┤
│ P6       │ Outer transformer (distilled)   │ After MATH, long-term           │
└──────────┴─────────────────────────────────┴─────────────────────────────────┘
```

Each priority is triggered by a specific signal — not a calendar date. The architecture evolves in response to what the experiments tell us.

---

## MATH-500 Deadline: April 22, 2026

```
Path: GSM8K (prove word problems) → scale passes → MATH L1-L3 → MATH L4-L5 → MATH-500 eval
Time remaining: ~16 days
```
