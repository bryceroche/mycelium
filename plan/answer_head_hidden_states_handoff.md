# Handoff: Answer Head Reads Hidden States + EOS Weight

## One-Sentence Summary

The answer head reads Llama's hidden states (where the numbers actually live) instead of just the 64-float page (which encodes pattern type for the hypernetwork), and the EOS token is weighted 5x in generation loss to teach clean one-step-per-cycle breathing.

---

## The Core Insight

The 64-float page encodes WHICH PATTERN was matched — not the number that was computed. The page is compressed FOR THE HYPERNETWORK (so the next cycle knows what kind of step was done). The actual numbers live in Llama's hidden states, which the GENERATION reads directly.

```
Page (64 floats):              "subtraction pattern applied" (for hypernetwork)
Hidden states (2048 floats):   "160 - 63 = 97" (the actual computation)

Answer head reading page:      4% accuracy  (number isn't there)
Generation reading hidden:     52% accuracy (number IS there)

The answer head was reading the wrong representation.
```

This explains every answer head failure:

```
900K head reading page:          collapsed to "10"
7.7M head reading page:          4% (more capacity, still wrong input)
Message (32 floats from hidden): too compressed, lost the number
7.7M head reading page + hidden: should match generation's 52%
```

---

## Part 1: Answer Head Reads Hidden States

### Architecture

```python
class AnswerHead(nn.Module):
    """
    Reads Llama's mean-pooled last-layer hidden states (2048 dim)
    plus the page (64 dim) plus cycle embedding (256 dim).
    
    The hidden states contain the actual computed numbers.
    The page provides cycle context (what pattern was matched).
    The cycle embedding tells which depth we're at.
    """
    def __init__(self, hidden_dim=2048, page_size=64, max_cycles=12,
                 cycle_embed_dim=256, transform_dim=1024, 
                 encoder_dim=1024, num_layers=6, max_digits=8):
        super().__init__()
        
        # Per-cycle embedding
        self.cycle_embed = nn.Embedding(max_cycles, cycle_embed_dim)
        
        # Transform: hidden + page + cycle → encoder input
        # NO bottleneck — full width preserved
        input_dim = hidden_dim + page_size + cycle_embed_dim  # 2048+64+256 = 2368
        self.transform = nn.Sequential(
            nn.Linear(input_dim, transform_dim),   # 2368 → 1024 (no bottleneck)
            nn.LayerNorm(transform_dim),
            nn.GELU(),
        )
        
        # Deep encoder
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(transform_dim if i == 0 else encoder_dim, encoder_dim),
                nn.LayerNorm(encoder_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
        self.encoder = nn.Sequential(*layers)
        
        # Digit prediction heads
        self.sign_head = nn.Linear(encoder_dim, 2)
        self.length_head = nn.Linear(encoder_dim, max_digits)
        self.digit_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_dim, 256),
                nn.GELU(),
                nn.Linear(256, 10),
            )
            for _ in range(max_digits)
        ])
    
    def forward(self, page, hidden_pool, cycle_num):
        """
        page:        (batch, 64)   — pattern context from perceiver
        hidden_pool: (batch, 2048) — mean-pooled Llama last layer (where numbers live)
        cycle_num:   int           — which cycle
        """
        cycle_ctx = self.cycle_embed(
            torch.tensor(cycle_num, device=page.device)
        ).expand(page.size(0), -1)
        
        combined = torch.cat([hidden_pool, page, cycle_ctx], dim=-1)  # 2368
        transformed = self.transform(combined)  # 1024
        encoded = self.encoder(transformed)     # 1024
        
        sign = self.sign_head(encoded)
        length = self.length_head(encoded)
        digits = [head(encoded) for head in self.digit_heads]
        
        return sign, length, digits
```

### Data Flow

```
Llama forward pass (atom-modified attention)
  │
  ├── hidden_states[-1]  (last layer, full 2048-dim)
  │     │
  │     ├── mean_pool → hidden_pool (2048) ──→ ANSWER HEAD (extract number)
  │     │
  │     ├── mean_pool → message_gen → 32-float message (for hypernetwork)
  │     │
  │     └── generation head → text output ("97 cookies remaining</s>")
  │
  └── all layers → perceiver → 64-float page ──→ HYPERNETWORK (select next atoms)

Two consumers of the same hidden states:
  Generation: reads hidden states → produces correct text (52%)
  Answer head: reads hidden states → should extract correct number (target: 52%)

One consumer of the page:
  Hypernetwork: reads page → selects atom blend for next cycle
```

### Why No Bottleneck in Transform

```
WRONG:  2368 → 512 → 1024 encoder
        The 512 bottleneck might discard number information
        The number could be spread across many of the 2048 hidden dims
        Compressing to 512 might lose it

RIGHT:  2368 → 1024 → 1024 encoder
        No bottleneck — full width preserved through transform
        The encoder sees rich representation
        ~2M more params, negligible cost
```

### What the Page vs Hidden States Encode

```
The page (64 floats) — PATTERN information:
  "This cycle applied a subtraction pattern"
  "This cycle parsed the first quantity"
  "This cycle handled a fraction operation"
  Useful for: hypernetwork (what to do next)
  NOT useful for: extracting the specific number computed

The hidden states (2048 floats) — COMPUTATION information:
  "The result is 97"
  "The operands were 160 and 63"
  "The operation was subtraction"
  Useful for: extracting the specific number
  Also useful for: generation (which reads these directly)
```

The answer head gets BOTH — the page tells it what KIND of number to expect (a parsed quantity? a computed result? a final answer?) and the hidden states tell it WHAT the number actually is.

---

## Part 2: EOS Token Weight

### The Problem

The model generates correct text then rambles past it:

```
Without EOS weight: "20 + 40 = 60 total toys. He buys 2*60 = 120..."
                     Generation correct at "60" but extraction grabs "120"
```

The EOS token is in the generation target but the model doesn't learn to produce it strongly enough. Regular tokens and EOS get equal weight in cross-entropy.

### The Fix

Weight EOS 5x higher in the generation loss:

```python
def weighted_generation_loss(logits, targets, tokenizer, eos_weight=5.0):
    """
    Cross-entropy with higher weight on EOS token.
    
    The model learns: stopping at the right place is 5x more important
    than getting any individual word right.
    """
    # Create per-token weights
    token_weights = torch.ones(tokenizer.vocab_size, device=logits.device)
    token_weights[tokenizer.eos_token_id] = eos_weight
    
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        weight=token_weights,
        ignore_index=-100,
    )
```

### What the Model Learns

```
"97 cookies remaining.</s>"       → low loss (EOS at right place, 5x reward)
"97 cookies remaining. He then"   → high loss (missed EOS, 5x penalty)
"97 cookies</s>"                  → moderate (EOS too early, content truncated)
```

One breath = one step = EOS. The model learns precise breath boundaries.

### EOS Fades With Smooth Training Wheel Removal

The EOS weight is part of the generation loss. As accuracy climbs and intermediate targets fade via smooth sigmoid, the EOS emphasis fades naturally:

```
At 50% accuracy:  gen_weight ≈ 1.0  → effective EOS weight = 5.0 (strong)
At 80% accuracy:  gen_weight ≈ 0.5  → effective EOS weight = 2.5 (moderate)
At 90% accuracy:  gen_weight ≈ 0.05 → effective EOS weight = 0.25 (fading)
At 95% accuracy:  gen_weight ≈ 0.01 → effective EOS weight = 0.05 (gone)

Everything fades on the same smooth sigmoid.
By the time the model is autonomous, it has internalized when to stop.
The training wheels dissolve naturally — including EOS emphasis.
```

---

## Combined Training Step

```python
def train_step(model, problem_ids, cycle_targets, final_answer, 
               final_accuracy, num_cycles, tokenizer):
    notebook = []
    total_loss = 0.0
    available_targets = list(cycle_targets)
    
    for cycle in range(num_cycles):
        page, hidden_pool, gen_output = model.think_one_pass(
            problem_ids, notebook, cycle
        )
        notebook.append(page)
        
        # Smooth fading weight for intermediate targets
        teacher_weight = per_cycle_target_weight(final_accuracy, cycle, num_cycles)
        
        # === ANSWER HEAD reads hidden states (not just page) ===
        if cycle == num_cycles - 1:
            ah_loss = answer_head_loss(page, hidden_pool, final_answer, cycle)
            total_loss += 5.0 * ah_loss
        else:
            # Flexible loss with consumption
            losses = []
            for i, target in enumerate(available_targets):
                losses.append((answer_head_loss(page, hidden_pool, target, cycle), i))
            losses.append((answer_head_loss(page, hidden_pool, final_answer, cycle), -1))
            best_loss, best_idx = min(losses, key=lambda x: x[0].item())
            total_loss += teacher_weight * best_loss
            if best_idx >= 0:
                available_targets.pop(best_idx)
        
        # === GENERATION with EOS weight ===
        # Per-problem conditional gating
        per_problem_ah = answer_head_loss_per_sample(
            page, hidden_pool, target, cycle
        )
        confidence = torch.sigmoid(-per_problem_ah + 15.0)
        gen_weight = 0.1 * confidence  # per-problem gate
        
        gen_loss = weighted_generation_loss(
            gen_output, gen_target_with_eos, tokenizer, eos_weight=5.0
        )
        
        if cycle == 0:
            total_loss += 1.0 * gen_loss  # cycle 1: gen always active
        else:
            total_loss += (gen_weight * gen_loss).mean()  # per-problem gated
    
    # Regularizers (unchanged)
    total_loss += 0.01 * isotropic_reg(raw_pages)
    total_loss += 0.1 * confidence_entropy_loss(notebook, final_answer)
    total_loss += 0.05 * contrastive_loss(notebook)
    
    return total_loss
```

---

## What to Monitor

```
1. Answer head vs generation accuracy gap:
   BEFORE: AH=4%, GEN=52% (13x gap — head can't read pages)
   TARGET: AH=40%+, GEN=50%+ (gap closes — head reads hidden states)
   
2. EOS placement:
   Average tokens before EOS per cycle
   Target: 10-30 tokens (one natural sentence)
   If EOS isn't learned: increase eos_weight to 10.0
   If EOS is too aggressive: decrease to 3.0

3. Per-problem conditional gate distribution:
   Mean confidence across batch per epoch
   Target: 0.3-0.7 (some problems gated, some not)
   If all ≈ 0.0: center too strict (raise from 15)
   If all ≈ 1.0: center too generous (lower from 15)

4. ah_loss trajectory:
   Should drop faster than before (hidden states are better input)
   If ah_loss plateaus above 12: transform might need more capacity
```

---

## What NOT to Do

```
- Do NOT remove the page from the answer head input.
  The page provides cycle context (what kind of step this is).
  The hidden states provide the number.
  Both are useful. The page is supplementary, not primary.

- Do NOT remove the answer head entirely.
  Generation at 52% proves the model computes.
  But generation loss (cross-entropy on tokens) is weak per-number signal.
  The answer head provides strong per-digit gradient.
  Once the head reads hidden states, it provides correct gradient.

- Do NOT set EOS weight above 10.0.
  Too high → model learns to produce EOS immediately (truncated output)
  5.0 is the sweet spot: strong enough to learn boundaries,
  not so strong that it dominates content.

- Do NOT add a bottleneck in the transform.
  2368 → 1024 (no bottleneck, preserves information)
  NOT 2368 → 512 → 1024 (bottleneck might discard numbers)
```
