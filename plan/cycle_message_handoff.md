# Handoff: Cycle Message — Direct Signal Bypassing the Bottleneck

## One-Sentence Summary

Add a 16-float message alongside the 64-float page that passes between cycles WITHOUT compression. The page carries compressed multi-layer understanding through the perceiver bottleneck. The message carries a direct signal from Llama's last layer — a post-it note that says "the key number was 160" without squeezing through the 64-float bottleneck.

---

## Why

Cycle 2 is stuck at 3-5% because it can't reliably read cycle 1's intermediate result from the compressed page. The perceiver squeezes everything into 64 floats, and the number "160" is buried in those dimensions alongside problem type, operation, context, and everything else.

The message provides a DIRECT CHANNEL:

```
Current (page only):
  Cycle 1 → perceiver → 64-float page → cycle 2 must decode "160" from compressed dims
  The number is buried. Cycle 2 can't find it reliably.

With message:
  Cycle 1 → perceiver → 64-float page (compressed understanding)
  Cycle 1 → message_net → 16-float message (direct "160" signal)
  Cycle 2 reads BOTH. The page says "subtraction problem." The message says "160."
```

---

## Architecture

### Message Generator

A simple projection from Llama's last hidden layer to 16 floats. Bypasses the perceiver entirely.

```python
class CycleMessageGenerator(nn.Module):
    """
    Generates a small message from Llama's last layer output.
    Bypasses the perceiver compression bottleneck entirely.
    
    The message is a POST-IT NOTE — quick, direct, uncompressed.
    The page is the FORMAL RECORD — rich, compressed, multi-layer.
    """
    def __init__(self, d_model=2048, message_dim=16):
        super().__init__()
        self.message_dim = message_dim
        
        # Mean-pool last layer → project to small message
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, message_dim),
        )
    
    def forward(self, last_layer_hidden):
        """
        last_layer_hidden: (batch, seq_len, d_model) — Llama's final layer
        returns: (batch, message_dim) — small direct signal
        """
        pooled = last_layer_hidden.mean(dim=1)  # (batch, d_model)
        message = self.net(pooled)               # (batch, 16)
        return message
```

### Hypernetwork Reads Pages + Messages

```python
class AtomHypernetwork(nn.Module):
    def __init__(self, page_size=64, message_dim=16, num_atoms=64, ...):
        super().__init__()
        
        # Page reading (existing — attention over accumulated pages)
        self.page_project = nn.Linear(page_size, 512)
        # ... existing page attention layers ...
        self.summary_project = nn.Linear(2048, 1024)
        
        # NEW: Message reading (simple — concatenate and project)
        self.message_project = nn.Sequential(
            nn.Linear(message_dim * 12, 256),  # up to 12 messages (max cycles)
            nn.GELU(),
            nn.Linear(256, 256),
        )
        
        # Direct path (existing)
        self.direct_path = nn.Sequential(
            nn.Linear(page_size, 256),
            nn.GELU(),
            nn.Linear(256, num_atoms),
        )
        
        # Scale net now takes page_summary + message_summary
        # 1024 (page) + 256 (message) = 1280
        self.scale_net = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, num_atoms),
        )
        
        self.blend = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, state_pages, messages=None):
        batch_size = state_pages[0].size(0) if state_pages else 1
        device = state_pages[0].device if state_pages else 'cpu'
        
        # Read pages (existing)
        page_summary = self.read_pages(state_pages)  # (batch, 1024)
        
        # Read messages (new)
        if messages and len(messages) > 0:
            # Pad to max_messages length
            max_msg = 12
            msg_list = messages + [torch.zeros_like(messages[0])] * (max_msg - len(messages))
            msg_cat = torch.cat(msg_list[:max_msg], dim=-1)  # (batch, 16 * 12)
            msg_summary = self.message_project(msg_cat)       # (batch, 256)
        else:
            msg_summary = torch.zeros(batch_size, 256, device=device)
        
        # Combine page + message summaries
        combined = torch.cat([page_summary, msg_summary], dim=-1)  # (batch, 1280)
        context_logits = self.scale_net(combined)
        
        # Direct path (existing)
        if state_pages:
            direct_logits = self.direct_path(state_pages[-1])
        else:
            direct_logits = torch.zeros(batch_size, self.num_atoms, device=device)
        
        # Blend
        blend = torch.sigmoid(self.blend)
        logits = blend * direct_logits + (1 - blend) * context_logits
        
        # Hard clamp + tanh (existing)
        logits = torch.clamp(logits, -3.0, 3.0)
        return torch.tanh(logits)
```

### Integration Into Thinking Loop

```python
def think_one_pass(self, problem_ids, state_pages, messages, cycle):
    """
    One thinking cycle producing BOTH a page and a message.
    """
    # 1. Hypernetwork reads pages + messages → atom scales
    atom_scales = self.hypernetwork(state_pages, messages)
    
    # 2. Apply atoms, run Llama
    self.apply_lora(atom_scales)
    outputs = self.llama(problem_ids, output_hidden_states=True)
    self.remove_lora()
    
    # 3. Perceiver compresses → page (64 floats, bottleneck)
    page = self.perceiver(outputs.hidden_states, state_pages)
    
    # 4. Message generator → message (16 floats, NO bottleneck)
    message = self.message_generator(outputs.hidden_states[-1])
    
    # 5. Residual gate on page (existing)
    if len(state_pages) > 0:
        page = self.residual_gate(page, state_pages[-1])
    page = F.normalize(page, dim=-1) * math.sqrt(64)
    
    # 6. Pi-harmonic encoding on page (existing)
    page = self.pi_encoding.apply(page, cycle)
    
    # Append both
    state_pages.append(page)
    messages.append(message)
    
    return page, message


def solve(self, problem_ids, max_cycles=8):
    """Full solve with pages + messages."""
    state_pages = []
    messages = []
    
    for cycle in range(max_cycles):
        page, message = self.think_one_pass(
            problem_ids, state_pages, messages, cycle
        )
        
        # Confidence check
        if cycle >= 1:
            conf, smooth = self.confidence_head(state_pages)
            if conf > 0.9:
                break
    
    return self.answer_head.decode(state_pages[-1])
```

---

## Why This Helps Cycle 2

```
WITHOUT message:
  Cycle 1: extracts 160, compresses into page dims somewhere
  Cycle 2: reads page, tries to find 160 among 64 compressed dims
           Can't reliably decode → falls back to guessing → stuck at 3%

WITH message:
  Cycle 1: extracts 160, compresses into page AND sends message
           Message carries direct signal "160" through 16 uncompressed floats
  Cycle 2: reads page (context) + message (explicit intermediate)
           Message clearly carries cycle 1's key finding
           Can focus on COMPUTING with 160 instead of FINDING 160
           The hard work shifts from decoding to computing
```

The page still carries the rich compressed context (problem type, operation, structure). The message carries the specific intermediate result that cycle 2 needs to build on. Two channels, complementary roles.

---

## Gradient Flow

```
Page gradient (long path, attenuated):
  loss → page → perceiver → all 16 layers (compressed)
  
Message gradient (short path, direct):
  loss → message → message_net → last layer only (uncompressed)
  
The message gives the SHORTEST gradient path from cycle 2's loss
back to cycle 1's computation. Just: loss → hypernetwork → message → message_net.
Three steps. No perceiver. No compression. No attenuation.
```

This is why the message might break the cycle 2 plateau. The gradient telling cycle 1 "your intermediate was wrong, encode 160 not 150" flows through the MESSAGE, not through the compressed page. The gradient is direct and strong.

---

## Why 16 Floats (Not Larger)

```
Message = 16 floats:  enough to encode a number + metadata
                      small relative to page (64) — doesn't compete
                      the page is the primary channel
                      the message is supplementary

Message = 64 floats:  same size as page — would compete
                      the model might use message instead of page
                      the bottleneck becomes meaningless
                      
Message = 4 floats:   too small — can't encode a useful signal
                      might as well not have it
```

16 is the sweet spot. Enough to carry "the number is 160 and I'm fairly sure" but not enough to bypass the bottleneck entirely. The page must still do the heavy lifting of compressed understanding. The message just carries the headline.

---

## Why This Is NOT the Old Strategy Vector

```
Strategy (removed):
  Produced by: perceiver (SAME compression path as page)
  Size: 64 floats (SAME as page)
  Content: redundant with page (perceiver outputs both)
  Result: carried no new information

Message (new):
  Produced by: message_net (BYPASSES perceiver entirely)
  Size: 16 floats (smaller, complementary)
  Content: direct from Llama's last layer (uncompressed)
  Result: carries DIFFERENT information through DIFFERENT path
```

The strategy was redundant because it went through the same compression. The message is useful because it SKIPS compression.

---

## Training

No change to the training loop structure. The message is just an additional input to the hypernetwork:

```python
def train_step(model, problem_ids, cycle_targets, cycle_gen_targets):
    state_pages = []
    messages = []
    total_loss = 0.0
    
    for cycle, (target_num, target_text) in enumerate(
        zip(cycle_targets, cycle_gen_targets)
    ):
        page, message = model.think_one_pass(
            problem_ids, state_pages, messages, cycle
        )
        
        # Generation loss (engine)
        gen_loss = model.generation_loss(page, problem_ids, target_text)
        
        # Answer head loss (shaping, higher weight for cycle 2+)
        head_loss = model.answer_head.loss(page, target_num)
        head_weight = 0.5 if cycle == 0 else 3.0
        
        total_loss += gen_loss + head_weight * head_loss
    
    total_loss += 0.05 * contrastive_loss(state_pages)
    total_loss += 0.1 * model.get_scale_reg()
    
    return total_loss / len(cycle_targets)
```

---

## Parameter Cost

```
CycleMessageGenerator:
  LayerNorm(2048):        4,096 params
  Linear(2048, 256):      524,288 params
  Linear(256, 16):        4,096 params
  Total:                  ~533K params

Message reading in hypernetwork:
  Linear(192, 256):       49,152 params
  Linear(256, 256):       65,536 params
  Total:                  ~115K params

Scale net input change:   1280 vs 1024 (minor reshape)

Grand total new:          ~650K params (negligible vs 198M trainable)
```

---

## What to Monitor

```
1. Message content:
   Print message values for different problems.
   Do messages DIFFER across problems? (should — different intermediates)
   Do messages carry number-like information? (hopefully)

2. Cycle 2 accuracy:
   The ONLY metric that matters. Does it climb past 5%?
   If yes: the message is providing the direct channel cycle 2 needed.
   If no: the message isn't carrying useful information (debug what it encodes).

3. Message vs page contribution:
   Can cycle 2 solve problems using message alone? (ablation)
   Can cycle 2 solve problems using page alone? (current, stuck at 3%)
   If message alone > page alone: the direct channel is more useful than compression.

4. Gradient norms through message path:
   Should be strong (short path, no compression attenuation).
   Compare to gradient through page path.
   Message gradient >> page gradient confirms the design.
```

---

## What NOT to Do

```
- Do NOT make the message larger than 16 floats.
  If the message is too large, it replaces the page and the bottleneck is meaningless.
  16 floats is a post-it note, not a second notebook.

- Do NOT compress the message through the perceiver.
  The whole point is that the message BYPASSES compression.
  It goes directly from Llama's last layer to the hypernetwork.

- Do NOT add the message to the page.
  They're separate channels read separately by the hypernetwork.
  Adding them would blend the compressed and uncompressed signals.

- Do NOT use the message for the answer head.
  The answer head reads the PAGE (compressed understanding).
  The message informs the HYPERNETWORK (atom selection).
  Different consumers, different channels.

- Do NOT accumulate messages with residual connections.
  Messages are ephemeral — each cycle produces a fresh one.
  Pages persist (residual gate). Messages are one-shot post-it notes.
  This prevents the message channel from growing to dominate the page.
```

---

## The Two-Channel Architecture

```
COMPRESSED CHANNEL (page, 64 floats):
  Path:     all 16 layers → perceiver → bottleneck → page
  Content:  rich, multi-layer, compressed understanding
  Persists: yes (residual gate, append-only notebook)
  Read by:  hypernetwork (attention), answer head, confidence head

DIRECT CHANNEL (message, 16 floats):
  Path:     last layer → mean pool → Linear → message
  Content:  specific, single-layer, uncompressed signal
  Persists: no (ephemeral, one cycle only, accumulated as list)
  Read by:  hypernetwork only (concatenated with page summary)

The page is the MEMORY. The message is the MEMO.
The page carries everything. The message carries the headline.
Together they give cycle 2 both context AND specific intermediates.
```
