# Handoff: Close the Communication Loop + Unfreeze Llama

## Two Changes

### 1. Perceiver gets current state as input

The perceiver doesn't know what's already encoded in the state. It might re-extract "48" every pass instead of extracting NEW information. Fix: condition the perceiver queries on the current state.

```python
# In AllLayerPerceiver.__init__, add:
self.state_project = nn.Linear(state_size, d_perceiver)  # 64 → 1024

# In forward(), change query conditioning:
pass_context = self.pass_embed(pass_num)
state_context = self.state_project(current_state)  # NEW: what's already encoded
queries = self.queries + pass_context + state_context
# Queries now know: "I'm on pass 2, state already has 48, extract something NEW"
```

Update the forward signature: `forward(self, all_layer_hidden_states, pass_num, current_state)`

Update the thinking loop to pass current state to the perceiver.

### 2. Unfreeze Llama with tiny LR

Per-step accuracy is the binding constraint (70% ceiling → 49% two-step). Unfreeze Llama so it can improve its arithmetic through training.

```python
optimizer = AdamW([
    {'params': perceiver.parameters(), 'lr': 1e-4},
    {'params': lora_templates.parameters(), 'lr': 1e-3},
    {'params': hypernetwork.parameters(), 'lr': 1e-3},
    {'params': transformer.parameters(), 'lr': 1e-6},  # NEW: 1000x smaller
    {'params': probe_head.parameters(), 'lr': 1e-4},
])
```

### What to monitor

- Per-step accuracy: does it climb above 70%? (unfreezing helps)
- Two-step accuracy: does it break past 49%? (both changes help)
- Probe error per pass: does pass 2 probe improve faster now? (state conditioning helps perceiver extract new info)
- Template norms: still growing? (gradient path now includes unfrozen Llama)

### Don't change anything else

Same 64-float bottleneck. Same 7-layer perceiver. Same LoRA rank 4. Same SymPy probe. Same hypersphere normalization. Just close the communication loop and let Llama learn.
