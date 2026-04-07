# Handoff: Page-Based State Accumulation

## One-Sentence Summary

Each thinking cycle compresses to 64 floats and APPENDS to a growing notebook of pages. Nothing is overwritten. The hypernetwork attends over ALL pages to generate LoRA scales. The per-cycle bottleneck stays tight (64 floats). The accumulated memory grows richer with each cycle.

---

## Why This Is Better

```
BEFORE (single state, overwritten):
  Cycle 1: compress → 64 floats
  Cycle 2: compress → normalize(old + delta) → 64 floats (cycle 1 partially lost)
  Cycle 3: compress → normalize(old + delta) → 64 floats (cycle 1 mostly gone)
  
  Problem: each cycle overwrites previous. Amnesia through accumulation.
  Problem: hypersphere rotation dilutes early information.

AFTER (pages, appended):
  Cycle 1: compress → 64 floats → append page 1
  Cycle 2: compress → 64 floats → append page 2
  Cycle 3: compress → 64 floats → append page 3
  
  Full state: [page1, page2, page3] = 192 floats, nothing lost.
  Each page preserved exactly. No overwriting. No dilution.
```

---

## Architecture Changes

### Perceiver: Same Output, Different Storage

The perceiver still outputs 64-float state delta and 512-float strategy. But the state delta is now a PAGE that gets appended, not accumulated:

```python
# In the thinking loop:
state_pages = []

for pass_num in range(max_passes):
    # ... LoRA applied, Llama forward, perceiver compresses ...
    
    state_delta, strategy = perceiver(all_layer_hidden, pass_num)
    
    # Normalize this page on its own hypersphere
    page = F.normalize(state_delta, dim=-1) * math.sqrt(64)
    state_pages.append(page)  # APPEND, not overwrite
    
    # Hypernetwork attends over ALL pages + current strategy
    lora_scales = hypernetwork(state_pages, strategy)
    
    # ... apply LoRA, continue loop ...
```

### Hypernetwork: Attention Over Pages

The hypernetwork no longer reads a flat state vector. It attends over the sequence of pages like a transformer reads tokens:

```python
class PageAttentionHypernetwork(nn.Module):
    def __init__(self, page_size=64, strategy_size=512, num_scales=256):
        super().__init__()
        
        # Project pages to attention space
        self.page_project = nn.Linear(page_size, 256)
        
        # Learned query for attending over pages
        self.page_query = nn.Parameter(torch.randn(4, 256))  # 4 query heads
        
        # Cross-attention: queries attend over pages
        self.page_attn = nn.MultiheadAttention(256, num_heads=4, batch_first=True)
        self.page_norm = nn.LayerNorm(256)
        
        # Combine page summary with strategy
        # 4 queries × 256 dim = 1024 from pages + 512 from strategy
        self.combine = nn.Sequential(
            nn.Linear(1024 + strategy_size, 512),
            nn.GELU(),
            nn.Linear(512, num_scales),
            nn.Tanh(),
        )
    
    def forward(self, state_pages, strategy):
        """
        state_pages: list of tensors, each (batch, 64)
        strategy: (batch, 512) from latest cycle
        returns: (batch, 256) LoRA scales
        """
        # Stack pages into sequence
        pages = torch.stack(state_pages, dim=1)  # (batch, num_pages, 64)
        pages_proj = self.page_project(pages)     # (batch, num_pages, 256)
        
        # Attend over pages
        batch_size = pages.size(0)
        queries = self.page_query.unsqueeze(0).expand(batch_size, -1, -1)
        
        attended, attn_weights = self.page_attn(
            query=queries, key=pages_proj, value=pages_proj
        )  # (batch, 4, 256)
        attended = self.page_norm(attended)
        
        # Flatten query outputs
        page_summary = attended.flatten(start_dim=1)  # (batch, 1024)
        
        # Combine with strategy
        combined = torch.cat([page_summary, strategy], dim=-1)  # (batch, 1536)
        scales = self.combine(combined)  # (batch, 256)
        
        return scales, attn_weights  # return weights for diagnostics
```

### Confidence Head: Also Reads Pages

```python
class PageConfidenceHead(nn.Module):
    def __init__(self, page_size=64, hidden=128):
        super().__init__()
        self.page_project = nn.Linear(page_size, hidden)
        # Attend over pages, produce single confidence scalar
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, hidden))
        self.output = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, state_pages):
        pages = torch.stack(state_pages, dim=1)
        pages_proj = self.page_project(pages)
        batch_size = pages.size(0)
        q = self.query.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.attn(query=q, key=pages_proj, value=pages_proj)
        return self.output(attended.squeeze(1))
```

### Generation: Pseudo-Tokens From Pages

For the generation pass, convert accumulated pages into pseudo-tokens:

```python
class PageToTokens(nn.Module):
    def __init__(self, page_size=64, d_model=2048, max_tokens=8):
        super().__init__()
        self.page_project = nn.Linear(page_size, 256)
        self.attn = nn.MultiheadAttention(256, num_heads=4, batch_first=True)
        self.queries = nn.Parameter(torch.randn(max_tokens, 256))
        self.output_project = nn.Linear(256, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, state_pages):
        pages = torch.stack(state_pages, dim=1)
        pages_proj = self.page_project(pages)
        batch_size = pages.size(0)
        q = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.attn(query=q, key=pages_proj, value=pages_proj)
        tokens = self.output_project(attended)
        return self.norm(tokens)  # (batch, max_tokens, 2048)
```

---

## What We No Longer Need

```
REMOVED:
  - Hypersphere normalization of accumulated state (each page normalizes independently)
  - Perceiver memory buffer (pages ARE the memory)
  - State conditioning in perceiver (pages replace this)
  - Fixed-size state input to hypernetwork (attention handles variable length)

KEPT:
  - 64-float per-cycle bottleneck (tight compression, unchanged)
  - 7-layer perceiver compressor (reads all Llama layers, unchanged)
  - Strategy channel (512 floats, ephemeral, unchanged)
  - State-conditioned LoRA templates (rank 4, unchanged)
  - Additive LoRA application (no hooks, unchanged)
  - SymPy probe (disabled for GSM8K, available for arithmetic)
  - Deep supervision (every page's state can be evaluated)
```

---

## The Full Thinking Loop

```
state_pages = []
strategy = zeros(512)

for pass_num in range(max_passes):
    # 1. Hypernetwork attends over all pages + strategy → LoRA scales
    if len(state_pages) > 0:
        lora_scales, attn_weights = hypernetwork(state_pages, strategy)
    else:
        lora_scales = zeros(256)  # first pass: no state, default attention
    
    # 2. Apply LoRA (additive)
    apply_lora(lora_scales)
    
    # 3. Llama forward (WITH LoRA)
    outputs = llama(problem_tokens, output_hidden_states=True)
    all_layer_hidden = outputs.hidden_states[1:]  # 16 layers, post-LoRA
    
    # 4. Remove LoRA
    remove_lora()
    
    # 5. Perceiver compresses → page + strategy
    page, strategy = perceiver(all_layer_hidden, pass_num)
    page = F.normalize(page, dim=-1) * sqrt(64)  # normalize THIS page
    state_pages.append(page)
    
    # 6. Confidence check (reads all pages)
    conf = confidence_head(state_pages)
    if conf > threshold:
        break

# 7. Generate: convert pages to pseudo-tokens, LoRA OFF
pseudo_tokens = page_to_tokens(state_pages)
answer = llama.generate(pseudo_tokens + problem_tokens)  # no LoRA, gentle injection
```

---

## What Emerges Naturally

### Frequency Bands Without Engineering

```
Cycle 1 page: coarse information (problem type, main numbers)
Cycle 2 page: medium information (first operation, intermediate)
Cycle 3 page: fine information (computation, result)
Cycle 4 page: finest information (verification, corrections)

Each page IS a frequency band. We don't need to engineer band structure.
The perceiver naturally encodes different information at different cycles
because each cycle sees different LoRA-modified hidden states.
```

### No Amnesia

```
Cycle 5 needs something from cycle 1?
Hypernetwork's attention over pages retrieves it directly.
No lossy accumulation. No rotation dilution.
Page 1 is exactly as it was when cycle 1 wrote it.
```

### Variable-Length Thinking

```
Easy problem: 2 pages (128 floats total). Hypernetwork attends over 2 pages.
Hard problem: 8 pages (512 floats total). Hypernetwork attends over 8 pages.
The architecture scales naturally. No fixed state size.
```

### Attention Weights as Diagnostics

```
The hypernetwork's attention weights over pages tell us:
"For cycle 4, the hypernetwork attended mostly to page 1 and page 3"
→ cycle 1 (parsing) and cycle 3 (computation) were most relevant
→ cycle 2 (intermediate) was less needed

This is FREE interpretability. We can see which thinking cycles
matter for each subsequent decision.
```

---

## Parameter Changes

```
REMOVED:
  - Old hypernetwork: Linear(576, 512) + Linear(512, 256) = ~425K

ADDED:
  - Page attention hypernetwork: ~650K
    page_project: Linear(64, 256) = 16K
    page_attn: MultiheadAttention(256, 4) = 263K
    combine: Linear(1536, 512) + Linear(512, 256) = ~900K
  - Page confidence head: ~20K
  - Page-to-tokens (generation): ~550K

NET CHANGE: ~+800K params (trivial vs 105M perceiver)
```

---

## Training

Same as before, with one change — the answer loss flows through pseudo-tokens at generation:

```python
def train_step(model, problem, gold_answer):
    # Thinking passes (LoRA ON, no generation)
    state_pages, strategy = model.think(problem, max_passes=3)
    
    # Generate with pseudo-tokens (LoRA OFF)
    pseudo_tokens = model.page_to_tokens(state_pages)
    outputs = model.generate_teacher_forced(pseudo_tokens, problem, gold_answer)
    
    answer_loss = outputs.loss
    
    # Deep supervision: each intermediate page set tries to generate
    intermediate_loss = 0
    for i in range(1, len(state_pages)):
        partial_pages = state_pages[:i+1]
        partial_tokens = model.page_to_tokens(partial_pages)
        partial_out = model.generate_teacher_forced(partial_tokens, problem, gold_answer)
        weight = (i + 1) / len(state_pages)
        intermediate_loss += weight * partial_out.loss
    
    loss = answer_loss + 0.3 * intermediate_loss
    loss.backward()
```

### Separate LRs (same as before)

```python
optimizer = AdamW([
    {'params': perceiver_params, 'lr': 1e-4},
    {'params': lora_template_params, 'lr': 1e-3},
    {'params': hypernetwork_params, 'lr': 1e-3},
    {'params': page_to_tokens_params, 'lr': 1e-4},
])
# Transformer FROZEN
```

---

## What to Monitor

```
1. Page norms: are all pages similar magnitude? (they should be, per-page normalization)
2. Hypernetwork attention weights: which pages does it attend to at each cycle?
3. Page cosine similarity: are consecutive pages different? (should be <0.7)
4. Accuracy trajectory: should climb past 6.2% GSM8K baseline
5. Generation quality: does LoRA-OFF + pseudo-tokens produce coherent text?
6. Per-cycle accuracy (deep supervision): does accuracy improve with more pages?
```

---

## What NOT to Do

```
- Do NOT accumulate pages into a single state vector. Append only.
- Do NOT apply LoRA during generation. Pseudo-tokens only.
- Do NOT use a fixed-size input to the hypernetwork. Attention handles variable pages.
- Do NOT normalize across pages. Each page normalizes independently.
- Do NOT skip the first-pass zero state. First cycle has no pages, LoRA scales = 0.
```
