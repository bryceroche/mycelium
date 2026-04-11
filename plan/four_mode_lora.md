# Four-Mode LoRA: Parse, Compute, Verify, Answer

## One-Sentence Summary

Four sets of LoRA templates — each specialized for a different cognitive operation — blended via a 4-way softmax that naturally transitions from reading to computing to checking to answering across thinking passes.

---

## The Four Modes

```
PARSE:    "Read the problem, extract quantities and relationships"
          Attention focused on: words, names, numbers in context
          The model understands "half as many" means division
          Language-heavy, comprehension-focused

COMPUTE:  "Apply operations to extracted quantities"
          Attention focused on: numbers and operators
          The model executes 48 / 2 = 24
          Math-heavy, execution-focused

VERIFY:   "Check that the solution is internally consistent"
          Attention focused on: all quantities simultaneously
          The model checks 24 = 48/2 ✓, 72 = 48+24 ✓
          Broad, relational, holistic

ANSWER:   "Shape hidden states for clean answer extraction"
          Attention focused on: the final result
          The model makes the answer readable by the answer head
          Extraction-focused, answer-oriented
```

---

## Why Four Modes (Not Two or Three)

We have evidence each is genuinely different:

```
PARSE vs COMPUTE:
  L1 (pure arithmetic): 94.8%  — model computes well
  L2 (word operations):  53.4%  — model struggles to parse
  The gap IS the parsing gap. Different attention needed.

COMPUTE vs VERIFY:
  Blend=0.25 on easy problems (barely verifies) → 96%
  Blend=0.65 on GSM8K (heavy verification) → 17.8%
  The model naturally uses more verification on harder problems.
  Proven to add 7.4 points (88.6% → 96.0% on L3).

VERIFY vs ANSWER:
  Generation-based extraction: number-spam, format bugs, regex fragility
  Answer-head extraction: failed before (pages constant), now pages alive
  A dedicated ANSWER mode shapes hidden states specifically for extraction.
  Separates "checking my work" from "reporting my answer."
```

---

## Architecture

### Four Template Sets

```python
class QuadLoRA(nn.Module):
    def __init__(self, d_model=2048, rank=4, num_layers=16):
        proj_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        
        # Four sets of templates, one per cognitive mode
        self.templates = nn.ModuleDict({
            'parse': nn.ModuleDict({
                'A': nn.ParameterList([nn.Parameter(torch.randn(num_layers, d_model, rank) * 0.01) for _ in proj_names]),
                'B': nn.ParameterList([nn.Parameter(torch.randn(num_layers, rank, self._proj_dim(p, d_model)) * 0.01) for p in proj_names]),
            }),
            'compute': nn.ModuleDict({
                'A': nn.ParameterList([...]),
                'B': nn.ParameterList([...]),
            }),
            'verify': nn.ModuleDict({
                'A': nn.ParameterList([...]),
                'B': nn.ParameterList([...]),
            }),
            'answer': nn.ModuleDict({
                'A': nn.ParameterList([...]),
                'B': nn.ParameterList([...]),
            }),
        })
    
    def _proj_dim(self, proj_name, d_model):
        if proj_name in ['k_proj', 'v_proj']:
            return 512  # Llama GQA: 8 KV heads
        return d_model
```

### Hypernetwork With 4-Way Softmax

```python
class QuadHypernetwork(nn.Module):
    def __init__(self, page_summary_dim=1024, strategy_size=64, 
                 pass_embed_dim=256, num_scales=256):
        super().__init__()
        input_dim = page_summary_dim + strategy_size + pass_embed_dim
        
        # Generate scales for all four modes + blend weights
        self.scales_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 4 * num_scales + 4),
            # 4 × 256 = 1024 scales (256 per mode)
            # + 4 blend logits
        )
    
    def forward(self, page_summary, strategy, pass_embed):
        out = self.scales_net(torch.cat([page_summary, strategy, pass_embed], dim=-1))
        
        # Split into per-mode scales and blend weights
        parse_scales   = torch.tanh(out[:, 0:256])
        compute_scales = torch.tanh(out[:, 256:512])
        verify_scales  = torch.tanh(out[:, 512:768])
        answer_scales  = torch.tanh(out[:, 768:1024])
        
        # 4-way softmax blend (sums to 1.0)
        blend_logits = out[:, 1024:1028]
        blend = F.softmax(blend_logits, dim=-1)  # (batch, 4)
        # blend[:, 0] = parse weight
        # blend[:, 1] = compute weight
        # blend[:, 2] = verify weight
        # blend[:, 3] = answer weight
        
        return {
            'parse': parse_scales,
            'compute': compute_scales,
            'verify': verify_scales,
            'answer': answer_scales,
            'blend': blend,
        }
```

### Blended Application (Additive, No Hooks)

```python
def apply_quad_lora(hidden, layer_idx, proj_idx,
                     templates, scales, blend):
    """
    Four LoRA paths blended by softmax weights.
    """
    lora_out = torch.zeros_like(hidden @ templates['parse']['B'][proj_idx][layer_idx])
    
    for mode_idx, mode in enumerate(['parse', 'compute', 'verify', 'answer']):
        A = templates[mode]['A'][proj_idx][layer_idx]     # (d_model, rank)
        B = templates[mode]['B'][proj_idx][layer_idx]     # (rank, proj_dim)
        s = scales[mode][:, layer_idx*4:(layer_idx+1)*4]  # (batch, rank)
        
        mode_out = (hidden @ A) * s @ B                   # (batch, seq, proj_dim)
        lora_out = lora_out + blend[:, mode_idx:mode_idx+1].unsqueeze(1) * mode_out
    
    return lora_out  # added to original projection output
```

---

## Expected Blend Trajectory

### Easy Problem ("Jamie had 56 cookies, gave 2 away")

```
Pass 1: [0.7, 0.2, 0.05, 0.05]  → parse (quick, simple)
Pass 2: [0.1, 0.6, 0.2, 0.1]   → compute (56-2=54)
Pass 3: [0.05, 0.05, 0.1, 0.8]  → answer (extract 54)
Confidence: high → stop after 3 passes
```

### Hard Problem (5-step GSM8K)

```
Pass 1:  [0.8, 0.1, 0.05, 0.05]  → parse (extract all quantities)
Pass 2:  [0.5, 0.4, 0.05, 0.05]  → parse + compute (still extracting, starting to compute)
Pass 3:  [0.1, 0.7, 0.1, 0.1]   → compute (main calculation)
Pass 4:  [0.1, 0.5, 0.3, 0.1]   → compute + verify (computing while checking)
Pass 5:  [0.05, 0.1, 0.7, 0.15]  → verify (consistency check)
Pass 6:  [0.05, 0.05, 0.1, 0.8]  → answer (extract final number)
Confidence: high → stop after 6 passes
```

The model allocates MORE passes to harder problems AND uses different mode distributions. PARSE dominates early. COMPUTE dominates middle. VERIFY follows. ANSWER finishes. The trajectory is learned, not hardcoded.

---

## Answer Head (Works With ANSWER Mode)

The ANSWER LoRA shapes hidden states for clean extraction. The answer head reads the last page (produced from ANSWER-focused hidden states):

```python
class AnswerHead(nn.Module):
    def __init__(self, page_size=64, max_digits=6):
        super().__init__()
        self.sign_head = nn.Linear(page_size, 2)           # positive or negative
        self.length_head = nn.Linear(page_size, max_digits) # how many digits
        self.digit_heads = nn.ModuleList([
            nn.Linear(page_size, 10) for _ in range(max_digits)  # 0-9 per position
        ])
    
    def forward(self, last_page):
        sign = self.sign_head(last_page)          # (batch, 2)
        length = self.length_head(last_page)      # (batch, max_digits)
        digits = [h(last_page) for h in self.digit_heads]  # list of (batch, 10)
        return sign, length, digits
    
    def decode(self, last_page):
        sign, length, digits = self.forward(last_page)
        
        num_digits = length.argmax(dim=-1) + 1   # 1-indexed
        is_negative = sign.argmax(dim=-1) == 1
        
        # Build number from predicted digits
        number = 0
        for i, digit_logits in enumerate(digits):
            digit = digit_logits.argmax(dim=-1)
            number = number * 10 + digit
        
        # Mask to predicted length
        # ... (truncate to num_digits)
        
        return number * (-1 if is_negative else 1)
```

### Training the Answer Head

```python
def answer_head_loss(last_page, gold_answer):
    sign, length, digits = answer_head(last_page)
    
    # Parse gold into digits
    gold_sign = 0 if gold_answer >= 0 else 1
    gold_str = str(abs(gold_answer))
    gold_length = len(gold_str) - 1  # 0-indexed
    gold_digits = [int(d) for d in gold_str]
    
    loss = F.cross_entropy(sign, gold_sign)
    loss += F.cross_entropy(length, gold_length)
    for i, (digit_logits, gold_digit) in enumerate(zip(digits, gold_digits)):
        loss += F.cross_entropy(digit_logits, gold_digit)
    
    return loss
```

The answer head loss goes directly into the last page, which goes into the perceiver, which goes into the ANSWER-mode LoRA. The entire chain learns: "when in ANSWER mode, produce hidden states that the perceiver compresses into pages that the answer head can read as digits."

---

## Confidence Head Update

The confidence head reads the blend history and learns that the ANSWER mode signals readiness:

```python
class QuadConfidenceHead(nn.Module):
    def __init__(self, page_size=64, hidden=128):
        super().__init__()
        self.page_project = nn.Linear(page_size, hidden)
        self.blend_project = nn.Linear(4, hidden)  # 4-way blend, not 1
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, hidden))
        self.output = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, state_pages, blend_history):
        pages = torch.stack(state_pages, dim=1)
        pages_proj = self.page_project(pages)
        
        # Blend history: (batch, num_passes, 4)
        blends = torch.stack(blend_history, dim=1)
        blend_proj = self.blend_project(blends)
        pages_proj = pages_proj + blend_proj
        
        batch_size = pages.size(0)
        q = self.query.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.attn(query=q, key=pages_proj, value=pages_proj)
        return self.output(attended.squeeze(1))
```

The confidence head learns: "stop when ANSWER mode has dominated a pass AND the pages look confident." It can see the full 4-way blend trajectory and learn patterns like "don't stop if no VERIFY pass has happened yet."

---

## Combined Loss

```python
total_loss = (generation_loss           # CoT reasoning trace (trains PARSE + COMPUTE)
              + 0.05 * contrastive_loss  # prevents page collapse
              + 0.1 * confidence_loss    # trains stopping decision
              + 0.3 * answer_head_loss)  # trains ANSWER mode + digit extraction

# The answer head loss is weighted higher because it provides
# direct gradient to the ANSWER templates through a short path:
# answer_loss → digit_heads → last_page → perceiver → ANSWER LoRA
```

---

## Parameter Budget

```
Component                    Params      % of total
────────────────────────────────────────────────────
Llama 3.2 1B (frozen)        1.23B       91.3%
7-Layer Perceiver            105M        7.8%
PARSE LoRA templates         1.1M        0.08%
COMPUTE LoRA templates       1.1M        0.08%
VERIFY LoRA templates        1.1M        0.08%
ANSWER LoRA templates        1.1M        0.08%  ← NEW
Quad Hypernetwork            ~700K       0.05%
Answer Head                  ~4K         0.00%  ← NEW
Confidence Head              ~20K        0.00%
Page-to-Tokens               ~550K       0.04%
────────────────────────────────────────────────────
Total:                       ~1.34B
Total LoRA:                  ~4.4M       (0.33%)
New params (ANSWER + head):  ~1.1M       (< 0.1%)
```

---

## Training Plan

```
Phase 1: Train QuadLoRA on L3 (named quantities)
         Compare: 4-mode vs 2-mode (96.0% baseline)
         Does the PARSE mode help word problems?
         Does the ANSWER mode enable digit extraction?

Phase 2: If Phase 1 improves → train through L4 → L4.5 → L4.7 → L4.9 → L5
         Each level warm-starts from previous
         The model should naturally shift its blend trajectory for harder problems

Phase 3: Full GSM8K with answer head as primary extraction
         No more generation + regex extraction
         Answer falls out of the last page through ANSWER-focused compression
```

---

## What to Monitor

```
1. Blend trajectory per level:
   Do harder problems use more PARSE passes? More VERIFY passes?
   Does ANSWER always dominate the final pass?

2. Mode specialization:
   Do the four template sets DIVERGE during training?
   template_cos(PARSE_A, COMPUTE_A) should decrease — they should specialize

3. Answer head accuracy vs generation extraction:
   Does digit prediction from the last page match or beat regex extraction?
   If yes: answer head becomes primary, generation becomes optional

4. Per-mode gradient norms:
   Are all four template sets getting meaningful gradient?
   ANSWER gets direct gradient from answer_head_loss
   PARSE/COMPUTE get gradient from generation_loss
   VERIFY gets gradient when blend is high
```

---

## The Thinking Trajectory as a Story

```
Human solving a word problem:
  1. READ the problem carefully          → PARSE
  2. FIGURE OUT the math                 → COMPUTE
  3. CHECK the answer makes sense        → VERIFY
  4. WRITE DOWN the final answer         → ANSWER

Our model:
  1. PARSE LoRA → attend to language     → compress to page
  2. COMPUTE LoRA → attend to math       → compress to page
  3. VERIFY LoRA → attend to consistency → compress to page
  4. ANSWER LoRA → attend to result      → compress to page → answer head reads it

Same cognitive arc. Four modes. Learned blend. Dynamic passes.
```
