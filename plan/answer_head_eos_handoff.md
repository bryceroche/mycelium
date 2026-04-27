# Handoff: 40M Answer Head + EOS Token

## One-Sentence Summary

Scale the answer head to 40M parameters (matching the importance of its role as the correctness gatekeeper) and add an EOS token so the model learns when to stop generating within each cycle — full expansion, learned stopping, no hardcoded token limits.

---

## Part 1: 40M Answer Head

### Why 40M

The answer head reads EVERY cycle's page and must decode numbers across GSM8K's full range (0 to 999,999+). It's the keystone of the entire training loop — the conditional gen loss gates on head_loss, the flexible loss matches intermediates through the head, and the per-cycle accuracy is measured through the head. When the head collapses, EVERYTHING collapses.

```
Current component ratios:
  Perceiver:     105M  (compress)
  Hypernetwork:  101M  (decide)
  Atoms:          82M  (expand)
  Answer head:   0.9M  (READ EVERY CYCLE'S OUTPUT — tiny!)
  
The answer head is 100x smaller than the components it reads.
It collapsed to predicting "0" or "10" on GSM8K.
Generation at 16% proves the model CAN compute.
The answer head can't READ the computation.

Proposed:
  Answer head:    40M  (real capacity, ~half an atom set)
```

The answer head's job is arguably as important as the hypernetwork's — the hypernetwork decides WHAT to compute, the answer head verifies WHETHER it computed correctly. Both decisions drive the training loop. Both deserve serious capacity.

### Architecture

```python
class ScaledAnswerHead(nn.Module):
    """
    40M parameter answer head.
    
    Reads a 64-float page and predicts:
      - Sign (positive/negative)
      - Number of digits (1-8)
      - Each digit (0-9) independently
    
    Per-cycle embeddings so it learns different reading 
    patterns at different depths.
    """
    def __init__(self, page_size=64, max_cycles=12, hidden=1024, 
                 num_layers=6, max_digits=8):
        super().__init__()
        
        # Per-cycle embedding (which cycle am I reading?)
        self.cycle_embed = nn.Embedding(max_cycles, 256)
        
        # Deep encoder: page (64) + cycle_embed (256) = 320 → 1024
        layers = []
        in_dim = page_size + 256  # 320
        for i in range(num_layers):
            out_dim = hidden  # 1024
            layers.extend([
                nn.Linear(in_dim if i == 0 else hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
        self.encoder = nn.Sequential(*layers)
        
        # Prediction heads
        self.sign_head = nn.Linear(hidden, 2)       # positive / negative
        self.length_head = nn.Linear(hidden, max_digits)  # how many digits
        self.digit_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, 256),
                nn.GELU(),
                nn.Linear(256, 10),  # 0-9
            )
            for _ in range(max_digits)
        ])
        
        self.max_digits = max_digits
    
    def forward(self, page, cycle_num):
        """
        page: (batch, 64)
        cycle_num: int
        returns: sign_logits, length_logits, digit_logits
        """
        cycle_ctx = self.cycle_embed(
            torch.tensor(cycle_num, device=page.device)
        ).expand(page.size(0), -1)  # (batch, 256)
        
        combined = torch.cat([page, cycle_ctx], dim=-1)  # (batch, 320)
        hidden = self.encoder(combined)  # (batch, 1024)
        
        sign = self.sign_head(hidden)    # (batch, 2)
        length = self.length_head(hidden)  # (batch, max_digits)
        digits = [head(hidden) for head in self.digit_heads]  # list of (batch, 10)
        
        return sign, length, digits

# Parameter count:
# Encoder: 6 layers × (1024 × 1024 + 1024 + 1024) ≈ 6.3M per layer × 6 = 37.8M
# First layer: 320 × 1024 = 0.3M
# Heads: sign 2K + length 8K + digits 8 × (1024×256 + 256×10) ≈ 2.1M
# Cycle embed: 12 × 256 = 3K
# Total: ~40M
```

### Why Not Smaller

```
900K (current):   collapsed on GSM8K, predicts "0" or "10"
5M:               might work for simple problems, uncertain for GSM8K range
10M:              probably works but why risk another round of "too small"?
40M:              definitive capacity, no more answer head bottlenecks

The cost of being too small: another failed training run (42 min/epoch × 50 epochs)
The cost of being too big: a few MB of VRAM and slightly longer backward pass
The answer head runs ONCE per cycle (one linear pass). It's not the compute bottleneck.
```

### Updated Component Ratios

```
Component          Before     After      Role
──────────────────────────────────────────────────
Perceiver          105M       105M       compress
Hypernetwork       101M       101M       decide
Atoms               82M        82M       expand
Answer Head        0.9M        40M       READ (the gatekeeper)
Confidence          2.5M       2.5M      stop
Message             1.1M       1.1M      bypass
──────────────────────────────────────────────────
Total trainable:   ~297M      ~336M      (+13%)
```

Four substantial components now: perceiver (105M), hypernetwork (101M), atoms (82M), answer head (40M). The readout is no longer a bottleneck.

---

## Part 2: EOS Token for Learned Generation Stopping

### The Problem

The model generates correct text then RAMBLES past it:

```
Correct:  "20 + 40 = 60 total toys."
Actual:   "20 + 40 = 60 total toys. He buys 2*60 = 120 more..." (rambling)

The extraction grabs 120 (wrong) instead of 60 (correct).
```

We don't want to hardcode max_new_tokens because we want FULL EXPANSION per cycle. Some steps need 10 tokens ("48/2 = 24"), some need 30 ("Natalia sold half as many clips in May as in April. 48/2 = 24 clips in May."). The model should breathe fully — but know when the breath ends.

### The Solution: Learned EOS

Add an EOS token to the generation target. The model learns to produce EOS after completing one step. Full expansion, learned stopping:

```
Training target:  "20 + 40 = 60 total toys.</s>"
                                              ^^^^ EOS token

The model learns:
  "20 + 40 = 60 total toys." → next token = </s> (STOP)
  
  NOT:
  "20 + 40 = 60 total toys." → next token = " He" (RAMBLE)
```

### Implementation

```python
# In data preparation: append EOS to every cycle's generation target
def prepare_gen_target(gen_text, tokenizer):
    """Add EOS token to generation target."""
    # gen_text: "20 + 40 = 60 total toys."
    # target:   "20 + 40 = 60 total toys.</s>"
    tokens = tokenizer.encode(gen_text) + [tokenizer.eos_token_id]
    return tokens

# In generation (inference): stop when EOS is produced
def generate_one_cycle(model, page, problem_ids, tokenizer, max_tokens=50):
    """Generate until EOS or max_tokens."""
    generated = []
    for step in range(max_tokens):
        logits = model.generate_next_token(page, problem_ids, generated)
        next_token = logits.argmax(-1)
        
        if next_token == tokenizer.eos_token_id:
            break  # model says "I'm done with this breath"
        
        generated.append(next_token)
    
    return tokenizer.decode(generated)

# In extraction: take text BEFORE EOS, extract answer from that
def extract_answer(generated_text):
    """Extract the number from the FIRST equation in generated text."""
    # The model generated up to EOS — no rambling past it
    match = re.search(r'(\d+)\s*[+\-*/]\s*(\d+)\s*=\s*(\d+)', generated_text)
    if match:
        return int(match.group(3))
    # Fallback: last number in text
    numbers = re.findall(r'[-]?\d+', generated_text)
    return int(numbers[-1]) if numbers else None
```

### Why EOS Is Better Than Max Tokens

```
Max tokens = 30:
  Short step "48/2 = 24" → fine (12 tokens)
  Long step "Natalia sold half as many clips..." → TRUNCATED (needs 35 tokens)
  The model can't fully expand on complex steps.
  We're forcing partial expansion — the opposite of breathing.

EOS token:
  Short step → "48/2 = 24.</s>" (stops at 12 tokens)
  Long step → "Natalia sold half as many clips in May as in April. 48/2 = 24 clips.</s>" (stops at 35 tokens)
  Full expansion regardless of length. Learned stopping.
  
EOS = the model learns the boundary of one breath.
Max tokens = we impose an artificial boundary.
```

### EOS Handles the Multi-Step Problem

The model currently generates multiple steps in one cycle:

```
Without EOS:  "12 * 20 = 240 episodes. Cesar watched 1/3 of 240 = 80..."
              Two steps crammed into one cycle. Extraction confused.

With EOS:     "12 * 20 = 240 episodes.</s>"
              One step. Clean. The model learns that one breath = one step.
              If it wants to compute more, it does it in the NEXT cycle.
```

The EOS token naturally enforces one-step-per-cycle without hardcoding it. The generation target for each cycle ends with EOS after ONE step. The model learns: "produce one complete step, then stop." If the model CAN solve the whole problem in one step (Panama hat), it produces the final answer + EOS. If it needs more steps, each cycle produces one step + EOS.

### EOS + Flexible Loss

The EOS token and flexible loss work together:

```
Problem: "Jamie had 160, lost 63, gained 20. How many?"

Cycle 1 generates: "Jamie had 160 cookies.</s>"
  → extracted: 160 → matches target[0] → consumed ✓

Cycle 2 generates: "160 - 63 = 97 remaining.</s>"
  → extracted: 97 → matches target[1] → consumed ✓

Cycle 3 generates: "97 + 20 = 117 total.</s>"
  → extracted: 117 → matches final → done ✓

OR (Panama hat — big bite):

Cycle 1 generates: "Jamie had 160 - 63 + 20 = 117 cookies.</s>"
  → extracted: 117 → matches final! → done in one cycle ✓
  Confidence high → stop early
```

The EOS prevents "117 cookies. He then sold 50..." rambling. Whether the model takes small bites or big bites, each cycle ends cleanly at EOS.

---

## Combined: 40M Head + EOS Token

The two fixes address different failure modes:

```
40M answer head:   pages → correct numbers (fixes 2% → should be 16%+ matching gen)
EOS token:         generation → clean single-step output (fixes rambling)

Together:
  The model generates one clean step (EOS stops rambling)
  The answer head reads the page correctly (40M capacity)
  The conditional gen loss gates properly (head actually works)
  The flexible loss matches intermediates (extraction isn't confused)
```

---

## Training Changes

```python
# 1. Generation targets now include EOS
gen_target = "20 + 40 = 60 total toys." + tokenizer.eos_token
# The cross-entropy loss on generation naturally teaches EOS prediction

# 2. Answer head is 40M with 6-layer encoder
# Per-cycle embeddings, 1024 hidden dim, dropout 0.1
# Warm start: can't transfer from 900K head (different architecture)
# Fresh init with higher initial LR (1e-3) to learn quickly

# 3. Conditional gen loss uses per-problem head_loss from 40M head
# The gate should work properly now that the head has capacity

# 4. Generation extraction stops at EOS
# No more grabbing wrong numbers from rambling text
```

---

## What to Monitor

```
1. Answer head accuracy per cycle:
   BEFORE: 2% (collapsed to predicting 0/10)
   TARGET: match generation accuracy (~16%+)
   If head acc ≈ gen acc: the head can read pages properly

2. EOS placement:
   Does the model learn to produce EOS after one step?
   Check: how many tokens before EOS on average
   Target: 10-30 tokens (one natural sentence)
   If >50: EOS not learned, model still rambles
   If <5: EOS too aggressive, model truncates

3. Generation accuracy (extraction):
   BEFORE: 16% but rambling confused extraction
   TARGET: 20%+ with clean EOS-bounded extraction

4. Conditional gen loss gate:
   With 40M head, head_loss should be lower for correct problems
   The gate should actually discriminate (some problems gated, some not)
   Monitor: mean head_confidence per epoch (should be 0.3-0.7, not 0.0 or 1.0)
```

---

## What NOT to Do

```
- Do NOT limit max_new_tokens as the primary solution.
  The model must breathe fully. EOS is learned stopping.
  Keep a safety max (50-60 tokens) but the model should hit EOS first.

- Do NOT warm-start the 40M head from the 900K head.
  Different architecture (6 layers vs 2, 1024 vs 256 hidden).
  Fresh init. The head learns quickly with proper capacity.

- Do NOT remove the conditional gen loss.
  The 40M head makes the gate actually functional.
  The head can now discriminate correct from wrong.
  The gate prevents "pretty text, wrong number."

- Do NOT punish big bites separately.
  EOS handles this — each cycle produces one clean output.
  If the model takes a big correct bite, great (Panama hat).
  If it takes a big wrong bite, the flexible loss penalizes it.
  No need for a separate bite penalty.
```
