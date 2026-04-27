# Handoff: Hypernetwork Text Context — Breaking the Fixed Point

## One-Sentence Summary

Feed the hypernetwork a small vector encoding what previous cycles computed (extracted numbers from text injection), giving it the ONE input that's genuinely different per cycle and breaking the scales cos=0.999 fixed-point collapse.

---

## The Problem: Fixed-Point Collapse

The Laplace diagnostic revealed the exact failure:

```
Cycle 1→2:  scales cos=0.978  →  raw_page cos=0.941
Cycle 2→3:  scales cos=0.999  →  raw_page cos=0.999  →  page cos=1.000

The hypernetwork AMPLIFIES similarity:
  Input similarity 0.94  →  output similarity 0.978  (contracted!)
  Input similarity 0.999 →  output similarity 0.999  (locked in!)

The hypernetwork is a CONTRACTING map.
It pulls scales toward a fixed point.
By cycle 3, everything is identical to cycle 2.
```

The root cause: the hypernetwork's inputs (pages + messages) are too similar across cycles. Similar inputs → similar cross-attention → similar scales. The hypernetwork has NO divergent signal to differentiate cycles.

---

## The Fix: Text Context Input

The text injection already provides genuinely different content per cycle:

```
Cycle 1: no previous results
Cycle 2: "Step 1 result: 240"
Cycle 3: "Step 1 result: 240\nStep 2 result: 80"
```

But the hypernetwork doesn't read the text injection — it only reads pages and messages. The text injection goes to LLAMA (as tokens), not to the hypernetwork (as features).

The fix: encode the text injection content as a small float vector and feed it to the hypernetwork alongside pages and messages.

```
BEFORE:
  hypernetwork reads: pages (similar) + messages (similar) → scales (similar)
  
AFTER:
  hypernetwork reads: pages (similar) + messages (similar) + context (DIFFERENT!)
  → scales (DIFFERENT!)
  
The context is the divergent signal that breaks the fixed point.
```

---

## Implementation

### Context Encoder

```python
def encode_text_context(prev_results, max_steps=8):
    """
    Encode previous cycle results as a fixed-size float vector.
    
    Non-differentiable (values come from extraction).
    But doesn't need to be — it's an INPUT that breaks symmetry,
    not a PATH that needs gradient flow.
    Same pattern as text injection to Llama.
    
    Args:
        prev_results: list of extracted numbers from previous cycles
                      e.g., [240] at cycle 2, [240, 80] at cycle 3
        max_steps: maximum number of previous results to encode
    
    Returns:
        context: (max_steps * 2,) float tensor
                 pairs of (step_exists, normalized_value)
    """
    context = torch.zeros(max_steps * 2)
    
    for i, val in enumerate(prev_results):
        if i < max_steps:
            context[i * 2] = 1.0                  # this step exists
            context[i * 2 + 1] = float(val) / 1000.0  # normalized value
    
    return context  # (16,) for max_steps=8

# Examples:
# Cycle 1: prev_results = []
#   context = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#   "nothing computed yet"
#
# Cycle 2: prev_results = [240]
#   context = [1.0, 0.24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#   "one step done: 240"
#
# Cycle 3: prev_results = [240, 80]
#   context = [1.0, 0.24, 1.0, 0.08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#   "two steps done: 240, 80"
#
# These are CLEARLY different → hypernetwork can differentiate!
```

### Hypernetwork Integration

```python
class AtomHypernetwork(nn.Module):
    def __init__(self, hidden_dim=1024, num_atoms=64, 
                 context_dim=16, ...):  # max_steps=8 → 16 floats
        super().__init__()
        
        # Existing: cross-attention over pages + messages
        self.page_attn = ...
        self.msg_attn = ...
        
        # NEW: project context to hidden dim
        self.context_project = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim),  # project to same dim as page features
        )
        
        # Existing: scale and focus heads
        # Input dim increases by hidden_dim (context features added)
        self.scale_head = nn.Linear(hidden_dim * 2, num_atoms)  # page_features + context
        self.focus_head = nn.Linear(hidden_dim * 2, num_atoms)  # for Möbius
    
    def forward(self, notebook, messages, text_context):
        """
        notebook:      list of pages [(batch, 64), ...]
        messages:      list of messages [(batch, 32), ...]
        text_context:  (batch, 16) — encoded previous results
        """
        # Cross-attend over notebook (existing)
        page_features = self.attend_over_pages(notebook, messages)  # (batch, hidden_dim)
        
        # Project text context (NEW)
        context_features = self.context_project(text_context)  # (batch, hidden_dim)
        
        # Combine page features with context (NEW)
        combined = torch.cat([page_features, context_features], dim=-1)  # (batch, 2*hidden_dim)
        
        # Generate scales from combined features
        scales = torch.tanh(self.scale_head(combined))
        scales = torch.clamp(scales, -3.0, 3.0)
        
        # Generate Möbius focus
        focus = self.focus_head(combined)
        
        return scales, focus
```

### Training Loop Integration

```python
def think_one_pass(self, problem_ids, notebook, focus_points, messages,
                    cycle, prev_results):
    """One breathing cycle with text context for the hypernetwork."""
    
    # Encode previous results as context vector
    text_context = encode_text_context(prev_results)  # (16,)
    text_context = text_context.unsqueeze(0).expand(batch_size, -1)  # (B, 16)
    text_context = text_context.to(device)
    
    # Hypernetwork reads pages + messages + context
    scales, focus = self.hypernetwork(notebook, messages, text_context)
    
    # Rest of the cycle unchanged...
    self.apply_lora(scales)
    outputs = self.llama(full_input, output_hidden_states=True)
    self.remove_lora()
    
    raw_page = self.perceiver(outputs.hidden_states)
    page = F.normalize(raw_page, dim=-1) * math.sqrt(64)
    warped_page = self.mobius(page, focus)
    
    notebook.append(warped_page)
    focus_points.append(focus)
    
    gen_logits = self.generate(outputs, problem_ids)
    message = self.message_generator(outputs.hidden_states[-1])
    messages.append(message)
    
    return warped_page, focus, gen_logits, message
```

---

## Why This Isn't skip_pass_embed

We removed skip_pass_embed because the cycle number was a SHORTCUT — the hypernetwork used "you are cycle 3" to produce fixed cycle-3 scales without reading the pages at all. It bypassed the notebook entirely.

The text context is fundamentally different:

```
skip_pass_embed:     "you are cycle 3"
                     No content. No computation history.
                     The hypernetwork ignores the notebook.
                     Shortcut: cycle_3 → fixed_scales_3

text_context:        "results so far: 240, 80"  
                     Actual computed values. Computation history.
                     The hypernetwork uses the notebook AND the context.
                     Reasoning: "240 and 80 computed → what's next?"
```

The text context carries WHAT was computed, not WHICH cycle. On different problems, the same cycle number produces different contexts:

```
Problem A, cycle 3: context = [1.0, 0.240, 1.0, 0.080, ...]
Problem B, cycle 3: context = [1.0, 0.048, 1.0, 0.024, ...]

Same cycle number. Different context. Different scales.
The hypernetwork reasons about CONTENT, not POSITION.
```

---

## Why Non-Differentiable Is Fine

The context values come from regex extraction of generated text. Not differentiable. But:

```
Text injection to Llama:
  extracted number → argmax → tokens → Llama reads
  Non-differentiable. Works. Gradient flows through Llama's attention.

Text context to hypernetwork:
  extracted number → float vector → hypernetwork reads  
  Non-differentiable. Should work. Gradient flows through hypernetwork's scales.

Same pattern:
  Non-diff content provides the SIGNAL (what was computed)
  Differentiable reader provides the GRADIENT (how to use it)
  The content breaks symmetry. The gradient does the learning.
```

The hypernetwork's gradient flows through:
```
loss → gen_logits → Llama → atoms → scales → hypernetwork → scale_head
                                                              ↑
                                              context enters HERE (non-diff input)
                                              gradient flows through scale_head (diff)
```

The context is a CONDITIONING input, not a gradient path. Like a label in conditional generation — the label itself isn't differentiable but the network that reads it is.

---

## Expected Impact

```
BEFORE (no context):
  Cycle 2 hypernetwork input: pages (similar) + messages (similar)
  Cycle 3 hypernetwork input: pages (similar) + messages (similar)
  → scales cos(2,3) = 0.999 (identical)
  → pages cos(2,3) = 1.000 (fixed point)
  → cycle 3 copies cycle 2

AFTER (with context):
  Cycle 2 input: pages + messages + [1.0, 0.240, 0, 0, ...]
  Cycle 3 input: pages + messages + [1.0, 0.240, 1.0, 0.080, ...]
  → scales cos(2,3) should DIVERGE (different context → different scales)
  → pages cos(2,3) should drop (different attention → different hidden states)
  → cycle 3 computes something NEW
```

The context is the ONLY input that's guaranteed to be different per cycle (assuming cycle 1 produces a correct extraction). Even if pages and messages are identical, the context differentiates.

---

## Parameter Cost

```
context_project:     Linear(16, 128) + Linear(128, 1024) = 134K
scale_head change:   Linear(1024, 64) → Linear(2048, 64) = +64K  
focus_head change:   Linear(1024, 64) → Linear(2048, 64) = +64K
───────────────────
Total new:           ~262K

Negligible compared to the 101M hypernetwork.
```

---

## What to Monitor

```
1. Scales cosine between cycles (THE key metric):
   BEFORE: cos(2,3) = 0.999 (fixed point)
   TARGET: cos(2,3) < 0.5 (differentiated)
   Run: diag_laplace.py every 5 epochs

2. Raw page cosine between cycles:
   BEFORE: cos(2,3) = 0.999
   TARGET: cos(2,3) < 0.7 (follows from different scales)

3. DC ratio:
   BEFORE: 0.90 (90% constant — loop barely breathing)
   TARGET: < 0.5 (each cycle contributes new info)

4. Cycle 2 generation:
   BEFORE: copies cycle 1's output verbatim
   TARGET: generates DIFFERENT text with DIFFERENT number

5. Per-cycle accuracy:
   cycle_1 should stay at 30%+
   cycle_2 should climb past 15%
   cycle_3 should start contributing
```

---

## What NOT to Do

```
- Do NOT make the context differentiable (yet).
  Start simple. Extraction-based context breaks the fixed point.
  Differentiability is a refinement for later if needed.

- Do NOT encode cycle number in the context.
  The context carries WHAT was computed, not WHICH cycle.
  The number of filled slots implicitly indicates cycle depth.
  That's content-based, not position-based.

- Do NOT remove pages/messages from the hypernetwork input.
  The context SUPPLEMENTS pages and messages.
  Pages carry compressed understanding. Messages carry direct signal.
  Context carries computation history. All three are needed.

- Do NOT increase context_dim beyond 16.
  8 steps × 2 values = 16 floats is enough.
  More dims = more noise, no more signal.
  The signal is in the VALUES (240, 80), not in high-dimensional features.

- Do NOT normalize the context values to [0,1].
  Divide by 1000.0 for rough scaling, but don't force [0,1].
  GSM8K answers range 0-100K. Division by 1000 gives 0-100 range.
  Good enough for the hypernetwork to learn from.
```
