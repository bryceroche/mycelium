# Handoff: Scale Diversity Loss + 512-Float Differentiable Bypass

## One-Sentence Summary

Add a direct loss penalizing similar atom scales between cycles (the DEMAND to differentiate) alongside a 512-float differentiable bypass from Llama's hidden states to the hypernetwork (the MECHANISM to differentiate correctly). The loss says "be different." The bypass says "here's HOW to be different."

---

## The Root Cause: No Pressure to Differentiate

The Laplace diagnostic revealed scales cos(2,3) = 0.999. Every architectural fix gave the hypernetwork the ABILITY to differentiate (Möbius, text context, 101M capacity). The hypernetwork ignored them all. Why?

```
Because producing identical scales is OPTIMAL for the current loss.

Cycle 1: generates correct answer 54% of the time → gen_loss happy
Cycle 2: copies cycle 1 → gen_loss equally happy (same text = low loss)
Cycle 3: copies again → gen_loss still happy

The generation loss doesn't care whether cycles differentiate.
The hypernetwork CORRECTLY learns: "same scales every cycle = optimal."
The fixed point is a LOCAL MINIMUM that satisfies all existing losses.
```

Every fix so far addressed ABILITY, not DEMAND:

```
Fix                      What it provided          Did it help?
───────────────────────────────────────────────────────────────
101M hypernetwork        capacity to differentiate  page_cos 0.90→0.61, but scales still 0.999
Möbius transform         geometric diversity        pages moved but raw content identical
Text context             divergent input            hypernetwork ignored it (no pressure to use)
Isotropic regularizer    dimensional spread         fixes dead dims, not cycle similarity
Contrastive loss         inter-problem diversity    different problems ≠ different cycles
```

The missing piece: a LOSS that directly penalizes scale similarity between cycles.

---

## Part 1: Scale Diversity Loss

### The Fix

```python
def scale_diversity_loss(all_scales, target_cos=0.3):
    """
    Directly penalize similar atom scales between consecutive cycles.
    
    This is the DEMAND that makes the hypernetwork differentiate.
    Without it, the hypernetwork has no reason to produce different scales.
    With it, the gradient says: "your cycle 2 scales are too similar
    to cycle 1 — CHANGE THEM."
    
    Args:
        all_scales: list of (batch, 64) scale tensors, one per cycle
        target_cos: maximum allowed cosine before penalty kicks in
                    0.3 = scales must be meaningfully different
                    
    Returns:
        scalar loss — penalizes cos(scales_i, scales_{i+1}) above target_cos
    """
    if len(all_scales) < 2:
        return torch.tensor(0.0, device=all_scales[0].device)
    
    diversity_loss = 0.0
    num_pairs = 0
    
    for i in range(len(all_scales) - 1):
        cos = F.cosine_similarity(
            all_scales[i], all_scales[i + 1], dim=-1
        ).mean()
        
        # Penalize cosine above target (scales too similar)
        diversity_loss += F.relu(cos - target_cos)
        num_pairs += 1
    
    return diversity_loss / max(num_pairs, 1)
```

### Why target_cos = 0.3

```
cos = 1.0:    identical scales (current — fixed point)
cos = 0.7:    very similar (still mostly same attention)
cos = 0.5:    moderately different (some differentiation)
cos = 0.3:    meaningfully different (good target)
cos = 0.0:    orthogonal (maximum differentiation)
cos = -1.0:   opposite (too extreme)

0.3 means: "scales should share at most 30% of their direction."
This forces genuinely different atom blends per cycle
without demanding they be completely orthogonal.
```

### Weight: 0.1

```
diversity_weight = 0.01:   too weak — doesn't break the 0.999 fixed point
diversity_weight = 0.1:    gentle nudge — breaks fixed point, doesn't dominate
diversity_weight = 0.5:    moderate — risk of forcing random diversity
diversity_weight = 1.0:    too strong — random different scales, hurts accuracy

Start at 0.1. The generation loss (~0.8-1.0) is the primary signal.
The diversity loss at 0.1 is a correction: "be different" but
"being correct is still 10x more important than being different."

If scales cos doesn't drop below 0.7:  increase to 0.2
If accuracy drops:                      decrease to 0.05
```

### What the Gradient Says

```
BEFORE (no diversity loss):
  Hypernetwork: "produce scales that minimize gen_loss"
  Result: same scales every cycle (copying minimizes gen_loss)

AFTER (with diversity loss):
  Hypernetwork: "produce scales that minimize gen_loss AND differ from last cycle"
  Result: must find DIFFERENT scales that ALSO produce correct generation
  
  The hypernetwork reaches for whatever input helps differentiate:
  - bypass vectors (rich, different per cycle)
  - text context (extracted numbers, different per cycle)
  - pages (currently similar, but diversity pressure helps)
  - messages (currently similar, but diversity pressure helps)
```

---

## Part 2: 512-Float Differentiable Bypass

### Why the Bypass Matters NOW

The diversity loss says "be different." But being RANDOMLY different is bad — the scales should be different because the COMPUTATION requires different attention, not because we forced arbitrary diversity.

The bypass provides the INFORMATION to be different correctly:

```
Diversity loss alone:          different but RANDOM (arbitrary attention changes)
Bypass alone:                  CAN be different but WON'T (no pressure, proven to fail)
Diversity loss + bypass:       different AND INFORMED (correct different attention)

The diversity loss breaks the fixed point.
The bypass tells the hypernetwork HOW to differentiate correctly.
Together: "you MUST be different (loss) and here's WHY (bypass)."
```

### Architecture

```python
class DifferentiableBypass(nn.Module):
    """
    512-float differentiable projection from Llama's hidden states
    directly to the hypernetwork, bypassing the 64-float compressor.
    
    Creates a real recurrent gradient: atoms at cycle 1 get gradient
    from cycle 2's loss flowing backward through the bypass.
    
    The bypass carries RICH information that the compressor discards:
    specific numbers, computation details, fine-grained context.
    The compressor keeps pattern type. The bypass keeps content.
    """
    def __init__(self, hidden_dim=2048, bypass_dim=512):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, bypass_dim),
            nn.LayerNorm(bypass_dim),
            nn.GELU(),
        )
    
    def forward(self, hidden_pool):
        """
        hidden_pool: (batch, 2048) — mean-pooled Llama last layer
        returns:     (batch, 512) — bypass vector
        
        Fully differentiable: gradient flows from hypernetwork scales
        through this projection back to Llama's hidden states and
        ultimately to the atoms that shaped those hidden states.
        """
        return self.project(hidden_pool)  # (batch, 512)
```

### Gradient Flow

```
The REAL RECURRENT GRADIENT path:

Cycle 2 loss
  → gen_logits (cycle 2)
    → Llama (with cycle 2 atoms)
      → atom scales (from hypernetwork)
        → hypernetwork reads bypass vectors from cycle 1
          → bypass = project(cycle 1 hidden states)
            → cycle 1 hidden states
              → Llama (with cycle 1 atoms)
                → cycle 1 atom scales
                  → GRADIENT REACHES CYCLE 1 ATOMS!

Cycle 1 atoms learn:
  "My attention at cycle 1 produced hidden states that,
   through the bypass, led to cycle 2 scales that led to
   wrong cycle 2 generation. I should attend DIFFERENTLY
   so the bypass carries better information for cycle 2."

This is GENUINE RECURRENCE. Not shared weights — gradient flow across cycles.
```

### Hypernetwork Integration

```python
class AtomHypernetwork(nn.Module):
    def __init__(self, page_dim=64, message_dim=32, bypass_dim=512,
                 hidden_dim=1024, num_atoms=64):
        super().__init__()
        
        # Cross-attention over notebook entries
        # Each entry: page (64) + message (32) + bypass (512) = 608 per cycle
        self.entry_project = nn.Linear(page_dim + message_dim + bypass_dim, hidden_dim)
        
        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True
        )
        self.query = nn.Parameter(torch.randn(4, hidden_dim))
        
        # Scale generation
        self.scale_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_atoms),
        )
        
        # Möbius focus
        self.focus_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, page_dim),
        )
    
    def forward(self, pages, messages, bypass_vectors):
        """
        pages:          list of (batch, 64) — notebook pages
        messages:       list of (batch, 32) — cycle memos
        bypass_vectors: list of (batch, 512) — rich detail (DIFFERENTIABLE)
        
        Returns: scales (batch, 64), focus (batch, 64)
        """
        # Combine all inputs per notebook entry
        entries = []
        for page, msg, bypass in zip(pages, messages, bypass_vectors):
            combined = torch.cat([page, msg, bypass], dim=-1)  # (batch, 608)
            projected = self.entry_project(combined)  # (batch, hidden_dim)
            entries.append(projected)
        
        # Stack entries as key/value sequence
        kv = torch.stack(entries, dim=1)  # (batch, num_cycles, hidden_dim)
        
        # Cross-attention: queries attend over entries
        q = self.query.unsqueeze(0).expand(kv.size(0), -1, -1)
        attn_out, _ = self.cross_attn(q, kv, kv)  # (batch, 4, hidden_dim)
        
        # Pool queries
        pooled = attn_out.mean(dim=1)  # (batch, hidden_dim)
        
        # Generate scales and focus
        scales = torch.tanh(self.scale_mlp(pooled))
        scales = torch.clamp(scales, -3.0, 3.0)
        focus = self.focus_mlp(pooled)
        
        return scales, focus
```

### Data Flow (Complete)

```
Problem text + text injection ("Step 1: 240\nStep 2: 80")
  │
  ▼
Llama (atom-modified attention)
  │
  ├── hidden_states[-1] (last layer, 2048-dim)
  │     │
  │     ├── mean_pool → DifferentiableBypass → 512 floats ──→ HYPERNETWORK
  │     │                                                       (rich detail,
  │     │                                                        DIFF gradient)
  │     │
  │     ├── mean_pool → MessageGenerator → 32 floats ──→ HYPERNETWORK
  │     │                                                  (quick memo)
  │     │
  │     └── generation head → "240 episodes #### 240</s>"
  │                           (THE output, scored by extraction)
  │
  └── all 16 layers → Perceiver → 64-float page ──→ NOTEBOOK
                                                       │
                                    Möbius(page, focus) │
                                                       ▼
                                                  HYPERNETWORK
                                                  (pattern type,
                                                   notebook history)

Hypernetwork reads THREE channels per cycle:
  Page (64):     what TYPE of step (from compressor)
  Message (32):  quick memo (existing bypass, light)  
  Bypass (512):  what SPECIFIC computation (new bypass, heavy, DIFF)
  
Total input per notebook entry: 608 floats
The bypass is the dominant signal — 84% of the input.
```

---

## Combined Training Loop

```python
def train_step(model, problem_ids, cycle_targets, cycle_gen_targets,
               final_answer, final_accuracy, num_cycles, tokenizer):
    notebook = []
    messages = []
    bypass_vectors = []
    all_scales = []  # collect for diversity loss
    total_loss = 0.0
    available_targets = list(cycle_targets)
    prev_results = []
    
    for cycle in range(num_cycles):
        # Hypernetwork reads notebook + messages + bypass
        scales, focus = model.hypernetwork(notebook, messages, bypass_vectors)
        all_scales.append(scales)  # save for diversity loss
        
        # Apply atoms, run Llama
        model.apply_lora(scales)
        outputs = model.llama(full_input, output_hidden_states=True)
        model.remove_lora()
        
        # Perceiver → page → Möbius → notebook
        raw_page = model.perceiver(outputs.hidden_states)
        page = F.normalize(raw_page, dim=-1) * math.sqrt(64)
        if cycle > 0:
            warped_page = model.mobius(page, focus)
        else:
            warped_page = page
        notebook.append(warped_page)
        
        # Message (existing light bypass)
        message = model.message_generator(outputs.hidden_states[-1])
        messages.append(message)
        
        # Bypass (NEW heavy differentiable bypass)
        hidden_pool = outputs.hidden_states[-1].mean(dim=1)
        bypass = model.bypass(hidden_pool)
        bypass_vectors.append(bypass)
        
        # Generation
        gen_logits = model.generate(outputs, problem_ids)
        gen_loss = weighted_generation_loss(
            gen_logits, cycle_gen_targets[cycle], tokenizer, eos_weight=5.0
        )
        
        # Extraction for gating and text injection
        with torch.no_grad():
            generated_text = tokenizer.decode(gen_logits.argmax(-1)[0])
            predicted_number = extract_answer(generated_text)
        
        # Three-tier gating: correct=1.0, wrong=0.1, copying=0.0
        consumed_numbers = [t for t in cycle_targets if t not in available_targets]
        
        if predicted_number is not None and predicted_number in consumed_numbers:
            gen_weight = 0.0   # copying consumed target — ZERO reward
        elif predicted_number is not None and (
            predicted_number in available_targets or predicted_number == final_answer
        ):
            gen_weight = 1.0   # correct new target — full reward
            # consume the target
            if predicted_number in available_targets:
                available_targets.remove(predicted_number)
        else:
            gen_weight = 0.1   # wrong but trying — reduced reward
        
        # Apply smooth fading
        teacher_weight = per_cycle_target_weight(final_accuracy, cycle, num_cycles)
        total_loss += teacher_weight * gen_weight * gen_loss
        
        # Text injection for next cycle
        prev_results.append(predicted_number if predicted_number else 0)
    
    # === SCALE DIVERSITY LOSS (the DEMAND) ===
    total_loss += 0.1 * scale_diversity_loss(all_scales, target_cos=0.3)
    
    # Existing regularizers
    total_loss += 0.01 * isotropic_reg(raw_pages)
    total_loss += 0.1 * confidence_entropy_loss(notebook, final_answer)
    
    return total_loss
```

---

## Why Both Together

```
DEMAND (diversity loss):
  "Your scales at cycle 2 must differ from cycle 1."
  Breaks the fixed point. Escapes the local minimum.
  Without this, the hypernetwork ignores all bypass information.

MECHANISM (512 bypass):
  "Here's cycle 1's rich hidden states so you know WHAT was computed."
  Enables INFORMED differentiation (not random).
  Without this, the hypernetwork differentiates randomly (hurts accuracy).

TOGETHER:
  "You MUST differentiate (loss) and here's HOW (bypass)."
  The loss breaks the fixed point.
  The bypass guides toward correct differentiation.
  
  Diversity loss without bypass: random different scales → hurts accuracy
  Bypass without diversity loss: informed but ignored → fixed point persists
  Both: informed AND forced → correct differentiation → cycles contribute
```

---

## Parameter Cost

```
Component          Before     After      Change
──────────────────────────────────────────────────
Bypass               —         5M        NEW (3-layer 2048→1024→512)
Hypernetwork entry   —        +0.6M     entry_project handles 608 dims not 96
Diversity loss       —         0         pure math, no params
──────────────────────────────────────────────────
Total new:                    ~5.6M
Total trainable:              ~300M     (from 294M)
```

---

## What to Monitor

```
1. Scales cosine (THE key metric):
   BEFORE: cos(1,2)=0.978, cos(2,3)=0.999 (fixed point)
   TARGET: cos(1,2)<0.5, cos(2,3)<0.5 (differentiated)
   If still >0.7 after 5 epochs: increase diversity weight to 0.2

2. DC ratio (Laplace diagnostic):
   BEFORE: 0.90 (90% constant — loop dead after cycle 1)
   TARGET: <0.5 (each cycle contributes new computation)

3. Cycle 2 generation:
   BEFORE: copies cycle 1 verbatim
   TARGET: generates DIFFERENT text with DIFFERENT number
   Check: extract cycle 2's #### number vs cycle 1's

4. Accuracy trajectory:
   Cycle 1 should stay at 30%+ (diversity doesn't hurt cycle 1)
   Cycle 2 should climb past 15% (differentiation enables new computation)
   Final should climb past 13% (both cycles contributing)

5. Bypass vs page contribution:
   Monitor hypernetwork attention: does it attend more to bypass or pages?
   If bypass dominates: pages might be unnecessary (simplification later)
   If both contribute: complementary channels working

6. Generation accuracy vs diversity:
   If accuracy drops when diversity is added: weight too high (reduce 0.1→0.05)
   If accuracy climbs AND scales diverge: working perfectly
   The sweet spot: different scales that produce correct generation
```

---

## Expected Outcome

```
Epoch 1-3:   diversity loss forces scales apart (cos drops from 0.999)
             accuracy might dip temporarily (hypernetwork adjusting)
             
Epoch 5-10:  scales stabilized at cos < 0.5
             bypass provides correct differentiation signal
             cycle 2 starts generating NEW text (not copies)
             accuracy recovers and climbs past 13%

Epoch 15-20: cycles genuinely differentiate
             cycle 1: parses/first compute (30%+)
             cycle 2: second compute (20%+)
             cycle 3: third compute or final answer
             final accuracy approaches 20%+
             
The model breathes differently each cycle — because it MUST (loss)
and it KNOWS HOW (bypass).
```

---

## What NOT to Do

```
- Do NOT add diversity loss without the bypass.
  Random differentiation hurts accuracy.
  The bypass provides informed differentiation.
  Both are needed together.

- Do NOT set diversity weight above 0.3.
  The generation loss (~0.8) is the primary signal.
  Diversity at 0.1 is a correction, not a command.
  Too high → random scales that don't compute correctly.

- Do NOT set target_cos below 0.1.
  Forcing near-orthogonal scales is too extreme.
  The scales should be different, not opposite.
  0.3 means "meaningfully different but can share some structure."

- Do NOT remove the Möbius transform.
  The diversity loss operates on SCALES.
  The Möbius operates on PAGES.
  Both contribute to cycle differentiation at different levels.
  Scale diversity → different atom blends → different attention → different pages.
  Möbius → different page positions → different hypernetwork reading.

- Do NOT remove existing regularizers.
  Isotropic: prevents dimension collapse (different problem)
  Contrastive: prevents inter-problem similarity (different problem)
  Diversity: prevents inter-CYCLE similarity (THIS problem)
  All three address different collapse modes. Keep all three.
```
