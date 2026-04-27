# Handoff: Soft-Token Answer Head (Reads Generation Output)

## One-Sentence Summary

Replace the answer head's input from hidden states/pages (which it can't decode — 4%) to the generation's own logits via differentiable soft tokens (which contain the correct number — generation at 56%). The answer head reads what the model SAYS, not what it THINKS.

---

## The Core Problem

The answer head has been asked to read three different representations, and failed at all of them:

```
Attempt 1: Read 64-float page             → collapsed to "10" (0-2%)
Attempt 2: Read page + hidden states       → 4% peak, never improved
Attempt 3: Read hidden states directly     → 4% peak, 32 epochs flat

Meanwhile: generation reads hidden states  → 56% correct
           The number IS there, the answer head just can't find it
```

The generation head succeeds because it produces TEXT — "97 cookies remaining" — where the number is explicit. The answer head fails because it tries to decode a compressed representation where the number is implicit.

## The Insight

The answer head should read the generation's OUTPUT, not Llama's hidden states. The generation already does the hard work of producing the number as text. The answer head just needs to read it.

```
OLD (failed):
  Llama hidden states (2048 dim) → answer head → ???
  "Somewhere in these 2048 floats is the number 97. Good luck."

NEW:
  Llama hidden states → generation → "97 cookies remaining" → answer head reads this
  "The number 97 is right there in the text. Extract it."
```

## The Differentiability Challenge

Generated tokens come from argmax (discrete, not differentiable). But we need gradient to flow from the answer head loss back to the atoms. The fix: read SOFT TOKENS instead of hard tokens.

```
Hard tokens (not differentiable):
  logits → argmax → token_id → embedding → answer head
            ^^^^^^
            discrete, gradient blocked

Soft tokens (fully differentiable):
  logits → softmax → weighted average of ALL embeddings → answer head
            ^^^^^^^
            continuous, gradient flows
```

Soft tokens are a probability-weighted blend of all token embeddings. If the logits strongly favor "9", the soft embedding is ~99% the embedding of "9". The information is the same, but the path is differentiable.

## Architecture

```python
class SoftTokenAnswerHead(nn.Module):
    """
    Reads generation logits via differentiable soft tokens.
    Extracts the number from what the model GENERATES.
    
    Fully differentiable: gradient flows from digit prediction
    through soft tokens through gen logits to Llama to atoms.
    
    The generation at 56% proves the number is in the logits.
    This head just needs to read it.
    """
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, 
                 num_layers=3, max_digits=8, max_cycles=12):
        super().__init__()
        
        # Learnable token embeddings for soft-token reading
        # (separate from Llama's embeddings — this head has its own vocabulary)
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Per-cycle embedding
        self.cycle_embed = nn.Embedding(max_cycles, 64)
        
        # Reader: processes the soft-token sequence
        layers = []
        for i in range(num_layers):
            in_dim = embed_dim + 64 if i == 0 else hidden_dim  # +64 for cycle embed
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
        self.reader = nn.Sequential(*layers)
        
        # Digit prediction heads
        self.sign_head = nn.Linear(hidden_dim, 2)
        self.length_head = nn.Linear(hidden_dim, max_digits)
        self.digit_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.GELU(),
                nn.Linear(128, 10),
            )
            for _ in range(max_digits)
        ])
        
        self.max_digits = max_digits
    
    def forward(self, gen_logits, cycle_num):
        """
        gen_logits: (batch, seq_len, vocab_size) — raw generation output
        cycle_num:  int — which cycle
        
        Returns: sign, length, digit logits
        """
        # Soft tokens: differentiable approximation of generated text
        # Temperature controls sharpness (lower = closer to argmax)
        temperature = 0.5
        soft_probs = F.softmax(gen_logits / temperature, dim=-1)  # (B, seq, vocab)
        
        # Weighted average of token embeddings
        soft_embeds = soft_probs @ self.token_embed.weight  # (B, seq, embed_dim)
        
        # Mean pool over sequence length
        pooled = soft_embeds.mean(dim=1)  # (B, embed_dim)
        
        # Add cycle context
        cycle_ctx = self.cycle_embed(
            torch.tensor(cycle_num, device=pooled.device)
        ).expand(pooled.size(0), -1)  # (B, 64)
        
        combined = torch.cat([pooled, cycle_ctx], dim=-1)  # (B, embed_dim + 64)
        
        # Read and predict
        hidden = self.reader(combined)  # (B, hidden_dim)
        
        sign = self.sign_head(hidden)
        length = self.length_head(hidden)
        digits = [head(hidden) for head in self.digit_heads]
        
        return sign, length, digits
```

## Gradient Flow

```
Complete differentiable path:

answer_head_loss
  → digit predictions
    → reader MLP
      → soft_embeds (weighted average of embeddings)
        → soft_probs (softmax of gen_logits)
          → gen_logits (Llama's output with LoRA-modified attention)
            → Llama attention (modified by atoms)
              → atom scales (from hypernetwork)
                → hypernetwork (reads notebook pages)

The answer head's "this should be 97 not 100" gradient flows
all the way back to the atoms: "change your attention so the
generation produces 97 instead of 100."

This is the gradient the answer head was SUPPOSED to provide
but couldn't when reading pages or hidden states.
```

## Why This Fixes the Cycle 2 Copying Problem

Cycle 2 copies cycle 1's generation. With the soft-token head + flexible loss with consumption:

```
Cycle 1 generates: "240 episodes"
  → soft tokens → answer head predicts 240
  → matches target[0] → CONSUMED ✓

Cycle 2 generates: "240 episodes" (copying!)
  → soft tokens → answer head predicts 240
  → tries target[0] → ALREADY CONSUMED
  → tries target[1] → doesn't match 240
  → tries final → doesn't match 240
  → HIGH LOSS → gradient says "predict something different!"

Gradient flows through soft tokens → gen logits → atoms:
  "At cycle 2, your attention produced text that repeats cycle 1.
   Change your attention to produce DIFFERENT text with a DIFFERENT number."
```

The consumption mechanism makes copying directly visible and penalized through the generation path. The atoms get gradient to differentiate their attention between cycles.

## Temperature in Soft Tokens

The temperature parameter controls how "sharp" the soft tokens are:

```
temperature = 1.0:   soft, blurry tokens (gradient flows easily but signal is diffuse)
temperature = 0.5:   medium sharpness (good balance)
temperature = 0.1:   nearly discrete (like argmax but still differentiable)

Start at 0.5. If gradient is too weak (head doesn't learn): lower to 0.3.
If gradient is unstable: raise to 0.8.
```

Lower temperature = sharper soft tokens = answer head sees clearer text = but gradient is spikier. Higher temperature = smoother gradient = but answer head sees blurrier text.

0.5 is a good starting point. The soft tokens are sharp enough that "97" produces mostly the embeddings for "9" and "7", but smooth enough that gradient flows cleanly.

## Data Flow (Complete Architecture)

```
Problem text + text injection
  │
  ▼
Llama (atom-modified attention)
  │
  ├── hidden_states → perceiver → 64-float page → APPEND to notebook
  │                                                    │
  │                                                    ▼
  │                                              HYPERNETWORK
  │                                              (reads notebook → atom scales)
  │
  ├── last_layer → message_gen → 32-float message (for hypernetwork)
  │
  └── logits → GENERATION → "97 cookies remaining</s>"
                    │
                    ├── text output (for extraction / text injection to next cycle)
                    │
                    └── soft tokens → ANSWER HEAD → digit prediction
                                      (reads what the model SAYS)
                                      (gradient flows back to atoms)
```

Two paths diverge from Llama's output:
- **Page path**: hidden states → perceiver → page → hypernetwork (for next cycle)
- **Generation path**: logits → text + soft tokens → answer head (for correctness)

The page serves the HYPERNETWORK (what to do next).
The generation serves the ANSWER HEAD (was the number correct).
Each path gets the representation it needs.

## Parameter Count

```
SoftTokenAnswerHead:
  token_embed: vocab_size × 256 = 32K × 256 ≈ 8.2M
  cycle_embed: 12 × 64 = 768
  reader: 3 layers × (512 × 512) ≈ 0.8M
  digit heads: 8 × (512×128 + 128×10) ≈ 0.5M
  sign + length: ~5K
  ────────────────
  Total: ~9.5M

Dominated by the token embedding (8.2M).
Could reduce embed_dim to 128 for ~4.5M total.
Or share Llama's embeddings (frozen, no extra params).
```

### Option: Share Llama's Token Embeddings

```python
# Instead of learning new embeddings:
self.token_embed = nn.Embedding(vocab_size, embed_dim)  # 8.2M params

# Share Llama's frozen embeddings (0 extra params):
self.token_embed = llama.embed_tokens  # frozen, already knows token meanings
# Then project from Llama's 2048 dim to our 256 dim:
self.embed_project = nn.Linear(2048, 256)  # 0.5M params

# The soft tokens use Llama's understanding of what each token means.
# Much richer than learned-from-scratch embeddings.
```

Sharing Llama's embeddings is cleaner — Llama already knows that "9" and "7" are digits and "cookies" is a noun. The answer head inherits this knowledge for free.

## Training Integration

```python
def train_step(model, problem_ids, cycle_targets, final_answer, num_cycles):
    notebook = []
    total_loss = 0.0
    available_targets = list(cycle_targets)
    
    for cycle in range(num_cycles):
        # Forward pass: get page, generation logits, text
        page, gen_logits, gen_text = model.think_and_generate(
            problem_ids, notebook, cycle
        )
        notebook.append(page)
        
        # GENERATION LOSS (with EOS weight)
        gen_target_tokens = prepare_gen_target(cycle_gen_targets[cycle], tokenizer)
        gen_loss = weighted_generation_loss(gen_logits, gen_target_tokens, 
                                            tokenizer, eos_weight=5.0)
        
        # ANSWER HEAD reads generation logits (soft tokens)
        sign, length, digits = model.soft_answer_head(gen_logits, cycle)
        
        # Flexible loss with consumption
        if cycle == num_cycles - 1:
            ah_loss = digit_loss(sign, length, digits, final_answer)
            total_loss += 5.0 * ah_loss
        else:
            losses = []
            for i, target in enumerate(available_targets):
                losses.append((digit_loss(sign, length, digits, target), i))
            losses.append((digit_loss(sign, length, digits, final_answer), -1))
            
            best_loss, best_idx = min(losses, key=lambda x: x[0].item())
            
            # Per-problem conditional gating on gen loss
            confidence = torch.sigmoid(-best_loss + center)
            gated_gen = confidence * gen_loss
            
            teacher_weight = per_cycle_target_weight(final_accuracy, cycle, num_cycles)
            total_loss += teacher_weight * best_loss  # answer head loss
            
            if cycle == 0:
                total_loss += 1.0 * gen_loss      # cycle 1: gen always active
            else:
                total_loss += gated_gen.mean()     # cycle 2+: gated by correctness
            
            if best_idx >= 0:
                available_targets.pop(best_idx)
    
    # Regularizers
    total_loss += 0.01 * isotropic_reg(raw_pages)
    total_loss += 0.1 * confidence_entropy_loss(notebook, final_answer)
    
    return total_loss
```

## What to Monitor

```
1. Answer head accuracy vs generation accuracy:
   BEFORE: AH=4%, GEN=56% (13x gap)
   TARGET: AH tracks GEN closely (both reading same logits)
   If AH ≈ GEN: soft token reading works
   If AH << GEN: temperature might be wrong, adjust

2. Cycle 2 copying:
   BEFORE: cycle 2 copies cycle 1's text (0% cycle 2 gen accuracy)
   TARGET: cycle 2 produces DIFFERENT text (consumption penalizes copies)
   Monitor: what text does cycle 2 generate? Same as cycle 1 or new?

3. Generation quality:
   BEFORE: gen declined 56% → 46% (answer head gradient hurting)
   TARGET: gen stable or improving (answer head gradient now aligned)
   The soft-token head gradient should HELP generation, not hurt it

4. Temperature sensitivity:
   If AH accuracy is low: lower temperature (sharper soft tokens)
   If training is unstable: raise temperature (smoother gradient)
   Start at 0.5, adjust based on results

5. Gradient norm through soft tokens:
   The gradient should flow from digit loss through softmax to logits
   If gradient is near zero: temperature too high or softmax too uniform
   If gradient explodes: temperature too low
```

## Expected Outcome

```
The generation produces correct text 56% of the time.
The soft-token answer head reads that same text.
The answer head should reach 40-50% accuracy quickly
(reading text is MUCH easier than decoding hidden states).

Once the answer head works:
  → conditional gen gating works properly
  → consumption penalizes cycle 2 copying
  → gradient flows to atoms: "differentiate your attention"
  → cycle 2 starts producing different (correct) text
  → final accuracy climbs past 17.8% ceiling
```

## What NOT to Do

```
- Do NOT keep the old hidden-states answer head.
  It never learned (4% after 32 epochs) and its gradient
  hurt generation (56% → 46%). Remove it entirely.

- Do NOT use hard tokens (argmax).
  Not differentiable. The whole point is gradient flow
  from answer head through generation to atoms.

- Do NOT set temperature below 0.1.
  Nearly discrete — gradient vanishes through near-one-hot softmax.
  0.3-0.8 is the useful range.

- Do NOT remove the generation loss.
  The generation loss teaches the model to produce correct text.
  The answer head loss teaches it to produce correct NUMBERS in that text.
  Both are needed. The answer head piggybacks on the generation.

- Do NOT worry about the extra compute.
  Soft tokens = one matrix multiply (softmax @ embeddings).
  Negligible compared to Llama's forward pass.
```
