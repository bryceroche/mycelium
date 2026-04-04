# Handoff: Side Channel — State + Strategy Architecture

## One-Sentence Summary

The compressor outputs TWO signals: a 64-float state (WHAT was learned, accumulates on hypersphere) and a 512-float strategy (HOW to think next, ephemeral, consumed by hypernetwork each pass). The LoRA hypernetwork reads both (576 floats) to generate rich, nuanced attention modifications. The bottleneck stays tight at 64 floats — the strategy can't bypass it because it doesn't accumulate.

---

## Why This Fixes the Communication Gap

Before: 64 floats was the ONLY link between compressor and hypernetwork. Each of the 256 LoRA scales was informed by ~0.25 floats of input. The hypernetwork was starving — not enough information to generate nuanced per-layer attention modifications. Templates grew 140% but the scales were coarse.

After: 576 floats (64 state + 512 strategy) feed the hypernetwork. Each LoRA scale is informed by ~2.25 floats. The compressor can tell the hypernetwork exactly what it found and what to look for next.

---

## The Two Signals

```
STATE (64 floats):
  - Content: numbers, values, intermediate results
  - Accumulates on hypersphere: state = normalize(state + delta) * √64
  - Persists across ALL passes — this is memory
  - Read at generation time to condition the final answer
  - THE BOTTLENECK — forces bite-sized chunks per pass

STRATEGY (512 floats):
  - Meta-information: what to focus on, what was confusing, what's next
  - Ephemeral: generated fresh each pass, consumed immediately, then gone
  - Does NOT accumulate — no cross-pass memory
  - Feeds ONLY into the hypernetwork → LoRA scales → attention modification
  - Cannot bypass bottleneck because:
    1. Doesn't persist (overwritten each pass)
    2. Consumed by hypernetwork (becomes attention weights, not recoverable content)
```

---

## Why 512 Strategy Floats Can't Bypass the 64-Float Bottleneck

The strategy is a one-way valve into attention modification:

```
Content through state:     48 → state → persists → read at generation → "48"
Content through strategy:  48 → strategy → LoRA scale → attention shifts slightly
                           Model can't reconstruct "48" from an attention shift
```

Even if the compressor encodes "48" in the strategy, it becomes a modification to attention weights, not a recoverable number. And it's gone next pass. The ONLY way to carry "48" from pass 1 to pass 3 is through the 64-float state.

---

## Architecture Changes

### Compressor: Two Output Heads

```python
# In AllLayerPerceiver, replace single output with two heads:

# Remove:
self.project_out = nn.Linear(d_perceiver, state_size // num_queries)

# Add:
self.project_state = nn.Linear(d_perceiver, state_size // num_queries)     # → 16 per query → 64 total
self.project_strategy = nn.Linear(d_perceiver, strategy_size // num_queries) # → 128 per query → 512 total

def forward(self, all_layer_hidden_states, pass_num):
    # ... existing perceiver processing (7 layers, unchanged) ...
    
    # Two outputs from the same processed queries
    state_delta = self.project_state(queries)       # (batch, 4, 16)
    strategy = self.project_strategy(queries)        # (batch, 4, 128)
    
    return state_delta.flatten(start_dim=1), strategy.flatten(start_dim=1)
    # Returns: (batch, 64), (batch, 512)
```

The perceiver's 7 layers of processing are shared. Only the final projection branches. Minimal parameter increase (~0.5M for the strategy projection).

### Hypernetwork: Reads State + Strategy

```python
# In StateConditionedLoRA, update input size:

# Remove:
self.state_to_scales = nn.Sequential(
    nn.Linear(64, 256),
    nn.Tanh(),
)

# Add:
self.scales_net = nn.Sequential(
    nn.Linear(64 + 512, 512),   # 576 → 512
    nn.GELU(),
    nn.Linear(512, 256),         # 512 → 256 LoRA scales
    nn.Tanh(),
)

def forward(self, state, strategy):
    combined = torch.cat([state, strategy], dim=-1)  # (batch, 576)
    all_scales = self.scales_net(combined)             # (batch, 256)
    # ... rest unchanged ...
```

The hypernetwork is now a 2-layer MLP (576 → 512 → 256) instead of a single linear layer (64 → 256). More capacity to map the rich input to nuanced LoRA scales. ~400K additional params.

### Thinking Loop: Pass Strategy to Hypernetwork

```python
def think(self, problem_text, max_passes=10, confidence_threshold=0.8):
    state = initialize_on_hypersphere(self.state_size)
    strategy = torch.zeros(1, 512, device=self.device)  # no strategy for first pass
    
    for pass_num in range(max_passes):
        # Hypernetwork reads BOTH state and strategy
        self.apply_lora(state, strategy)
        
        # Forward through Llama with LoRA
        outputs = self.transformer(input_ids=prompt_ids, output_hidden_states=True)
        all_layer_hidden = list(outputs.hidden_states[1:])
        
        self.remove_lora()
        
        # Compressor outputs BOTH state delta and strategy
        state_delta, strategy = self.compressor(all_layer_hidden, pass_num)
        #              ↑ accumulates                    ↑ ephemeral (overwritten)
        
        # State accumulates on hypersphere
        state = state + state_delta
        state = F.normalize(state, dim=-1) * math.sqrt(self.state_size)
        
        # Strategy is fresh — used by hypernetwork on NEXT pass, then replaced
        # (no accumulation, no normalization)
        
        # Confidence reads state only (not strategy)
        conf = self.confidence(state)
        if conf.item() > confidence_threshold:
            break
    
    return state, strategy  # final state for generation, final strategy for final LoRA
```

### Generation: Uses Final State + Strategy

```python
def generate_answer(self, problem_text, state, strategy):
    self.apply_lora(state, strategy)  # final LoRA from both signals
    output = self.transformer.generate(input_ids=prompt_ids, max_new_tokens=512)
    self.remove_lora()
    return self.tokenizer.decode(output[0], skip_special_tokens=True)
```

---

## Parameter Changes

```
Before:
  Hypernetwork: Linear(64, 256)        = 16K params
  Compressor output: Linear(1024, 16)  = 16K params

After:
  Hypernetwork: Linear(576, 512) + Linear(512, 256) = ~425K params
  Compressor state head: Linear(1024, 16)            = 16K params (unchanged)
  Compressor strategy head: Linear(1024, 128)         = 131K params (new)

Total new params: ~540K (trivial)
Total architecture: still ~106M (perceiver dominates)
```

---

## What to Monitor

```
1. Strategy vector norms: are they meaningful? (not all zeros, not random)
2. Strategy diversity across passes: cos_sim(strategy_1, strategy_2) should be LOW
   (different passes produce different strategies)
3. LoRA scale diversity: should INCREASE vs previous run
   (richer input → more nuanced scales)
4. Two-step accuracy: target >53% (previous best)
5. Template growth: should accelerate (hypernetwork gives better gradient signal)
```

---

## What NOT to Change

```
- State size stays at 64 floats (tight bottleneck)
- Hypersphere normalization stays (state only, not strategy)
- 7-layer perceiver stays (shared processing, two output heads)
- LoRA rank stays at 4
- Probe stays (per-pass gradient)
- Transformer stays FROZEN (unfreezing caused instability)
- Template LR stays at 10x (1e-3)
```

---

## Expected Impact

The previous 53% was limited by the hypernetwork receiving only 64 floats. With 576 floats of input (state + strategy), the hypernetwork can generate much more targeted LoRA modifications. The compressor can say "I found a number at position 5, the operation is division, focus on the denominator next" instead of just "here's 64 floats of compressed everything."

If this pushes past 53% on two-step, we move to three-step with confidence that the communication channel is working.
