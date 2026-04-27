# Handoff: The Breathing Controller — One Self-Observing Network

## One-Sentence Summary

Replace the separate perceiver (105M), hypernetwork (101M), and bypass (5M) with a single unified BreathingController (~130M) that reads Llama's hidden states and produces BOTH the 64-float page (what was understood) AND the 64 atom scales (what to do next) from shared understanding.

---

## The Insight: The System Observes Itself

The breathing loop is a cycle of thinking and self-reflection:

```
BREATHE:   Llama thinks (with atom-modified attention)
OBSERVE:   Controller watches what Llama computed
RECORD:    Write a page (what was understood)
PLAN:      Choose atom scales (what to think about next)
BREATHE:   Llama thinks again (with new atoms)
```

The record and the plan come from the SAME act of observation. Currently they come from SEPARATE networks (perceiver and hypernetwork) that read the same signal independently and might disagree. A single controller ensures consistency.

---

## Why the Separation Failed

The debug diagnostic revealed:

```
Perceiver (105M):     reads Llama hidden states → produces pages      ✓ (pages healthy)
Hypernetwork (101M):  reads pages → produces scales                   ✗ (constant function!)
Bypass (5M):          reads hidden states → feeds hypernetwork         ✗ (collapsed to constant)
Message (1.1M):       reads hidden states → feeds hypernetwork         ✗ (collapsed to constant)
```

The perceiver succeeds because it reads Llama's hidden states DIRECTLY. The hypernetwork fails because it reads DOWNSTREAM COMPRESSIONS that have lost the content. Every intermediary we built (bypass, message, text context) collapsed to a constant.

The root cause: the hypernetwork doesn't read Llama's hidden states. It reads summaries of summaries. The solution isn't another intermediary — it's giving the hypernetwork DIRECT access to the same signal the perceiver reads.

And if both the perceiver and hypernetwork read the same hidden states, why have two separate networks?

---

## The Unified Architecture

```python
class BreathingController(nn.Module):
    """
    One network that observes Llama's thinking and produces:
      1. A 64-float page (what was understood — the record)
      2. 64 atom scales (what to do next — the plan)
    
    Both outputs come from SHARED understanding of what Llama computed.
    The controller reads all 16 Llama layers (like the perceiver did)
    AND attends over previous cycles (like the hypernetwork did).
    
    One act of self-reflection. Not two separate readings.
    """
    def __init__(self, num_layers=16, hidden_dim=2048, 
                 internal_dim=1024, num_heads=8,
                 page_dim=64, num_atoms=64, max_cycles=12):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.internal_dim = internal_dim
        self.page_dim = page_dim
        
        # --- READ CURRENT: what did Llama just compute? ---
        
        # Learned weighted combination of all 16 Llama layers
        # (the perceiver's multi-layer reading ability)
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # Project weighted hidden states to internal dimension
        self.current_project = nn.Sequential(
            nn.Linear(hidden_dim, internal_dim),
            nn.LayerNorm(internal_dim),
            nn.GELU(),
        )
        
        # --- READ HISTORY: what have previous cycles understood? ---
        
        # Project each history entry (page + hidden_pool) to internal dim
        self.history_entry_project = nn.Linear(
            page_dim + hidden_dim, internal_dim
        )
        
        # Cross-attention: current understanding queries history
        self.history_attn_layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(
                    embed_dim=internal_dim, num_heads=num_heads, batch_first=True
                ),
                'norm1': nn.LayerNorm(internal_dim),
                'ff': nn.Sequential(
                    nn.Linear(internal_dim, internal_dim * 4),
                    nn.GELU(),
                    nn.Linear(internal_dim * 4, internal_dim),
                ),
                'norm2': nn.LayerNorm(internal_dim),
            })
            for _ in range(4)  # 4 layers of cross-attention over history
        ])
        
        # --- SHARED TRUNK: integrate current + history ---
        
        self.trunk = nn.Sequential(
            nn.Linear(internal_dim * 2, internal_dim),
            nn.LayerNorm(internal_dim),
            nn.GELU(),
            nn.Linear(internal_dim, internal_dim),
            nn.LayerNorm(internal_dim),
            nn.GELU(),
            nn.Linear(internal_dim, internal_dim),
            nn.LayerNorm(internal_dim),
            nn.GELU(),
        )
        
        # --- TWO HEADS: record + plan from shared understanding ---
        
        # Page head: "what did I understand?" (the record)
        self.page_head = nn.Sequential(
            nn.Linear(internal_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, page_dim),  # → 64 floats
        )
        
        # Scale head: "what should I think about next?" (the plan)
        self.scale_head = nn.Sequential(
            nn.Linear(internal_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_atoms),  # → 64 scales
        )
        
        # Möbius focus head (optional, for page diversity)
        self.focus_head = nn.Sequential(
            nn.Linear(internal_dim, 256),
            nn.GELU(),
            nn.Linear(256, page_dim),  # → 64 focus floats
        )
    
    def forward(self, hidden_states_all_layers, history_pages, history_hiddens):
        """
        hidden_states_all_layers: list of 16 tensors, each (batch, seq, 2048)
        history_pages:            list of previous cycle pages [(batch, 64), ...]
        history_hiddens:          list of previous cycle hidden pools [(batch, 2048), ...]
        
        Returns:
            page:   (batch, 64) — what was understood (the record)
            scales: (batch, 64) — what to do next (the plan)
            focus:  (batch, 64) — Möbius focus for page warping
        """
        
        # --- READ CURRENT ---
        
        # Weighted combination of all Llama layers
        weights = F.softmax(self.layer_weights, dim=0)
        combined_hidden = sum(
            w * h.mean(dim=1)  # mean pool over sequence
            for w, h in zip(weights, hidden_states_all_layers)
        )  # (batch, 2048)
        
        current = self.current_project(combined_hidden)  # (batch, 1024)
        
        # --- READ HISTORY ---
        
        if len(history_pages) > 0:
            # Build history sequence: each entry is page + hidden_pool
            history_entries = []
            for page, hidden in zip(history_pages, history_hiddens):
                entry = torch.cat([page, hidden], dim=-1)  # (batch, 64+2048)
                projected = self.history_entry_project(entry)  # (batch, 1024)
                history_entries.append(projected)
            
            history_seq = torch.stack(history_entries, dim=1)  # (batch, N, 1024)
            
            # Cross-attention: current queries history (4 layers)
            query = current.unsqueeze(1)  # (batch, 1, 1024)
            for layer in self.history_attn_layers:
                # Cross-attention
                attn_out, _ = layer['attn'](query, history_seq, history_seq)
                query = layer['norm1'](query + attn_out)
                # Feed-forward
                ff_out = layer['ff'](query)
                query = layer['norm2'](query + ff_out)
            
            history_ctx = query.squeeze(1)  # (batch, 1024)
        else:
            # Cycle 1: no history — but we STILL have current hidden states!
            # This is why the constant-function problem disappears.
            history_ctx = torch.zeros_like(current)
        
        # --- SHARED TRUNK ---
        
        shared = self.trunk(
            torch.cat([current, history_ctx], dim=-1)
        )  # (batch, 1024)
        
        # --- TWO HEADS ---
        
        # The record: what was understood
        page = self.page_head(shared)  # (batch, 64)
        page = F.normalize(page, dim=-1) * math.sqrt(64)  # hypersphere
        
        # The plan: what to do next
        scales = self.scale_head(shared)  # (batch, 64)
        scales = torch.tanh(scales)
        scales = torch.clamp(scales, -3.0, 3.0)
        
        # Möbius focus (for page diversity on the sphere)
        focus = self.focus_head(shared)  # (batch, 64)
        
        return page, scales, focus
```

---

## The Breathing Loop (Simplified)

```python
def solve(model, problem_ids, tokenizer, max_cycles=8, conf_threshold=0.85):
    """
    The breathing loop with unified controller.
    
    Each cycle:
      1. BREATHE: Llama reads problem with atom-modified attention
      2. OBSERVE: Controller reads what Llama computed
      3. RECORD:  Controller writes a page (understanding)
      4. PLAN:    Controller chooses scales (next attention pattern)
      5. Repeat with new scales
    """
    notebook_pages = []
    notebook_hiddens = []
    prev_results = []
    
    # Initial scales: controller reads problem WITHOUT history
    # This produces PROBLEM-SPECIFIC scales at cycle 1 (not constant!)
    # We need a first forward pass to get hidden states
    
    # Cycle 0: initial Llama pass with default scales
    initial_scales = torch.zeros(1, 64, device=device)  # no modification
    
    for cycle in range(max_cycles):
        if cycle == 0:
            scales = initial_scales  # first pass: no atom modification
        # else: scales from previous cycle's controller output
        
        # 1. BREATHE: Llama forward with atom-modified attention
        text_ctx = format_text_injection(prev_results)
        input_ids = tokenize(text_ctx + problem_text)
        
        model.apply_lora(scales)
        outputs = model.llama(input_ids, output_hidden_states=True)
        model.remove_lora()
        
        # 2. OBSERVE + RECORD + PLAN: controller reads hidden states
        page, next_scales, focus = model.controller(
            outputs.hidden_states,     # all 16 layers of current thinking
            notebook_pages,            # history of understanding
            notebook_hiddens,          # history of hidden states
        )
        
        # 3. Apply Möbius warp to page
        if cycle > 0:
            page = model.mobius(page, focus)
        
        # 4. Store in notebook
        hidden_pool = outputs.hidden_states[-1].mean(dim=1)
        notebook_pages.append(page)
        notebook_hiddens.append(hidden_pool)
        
        # 5. Generate text (the output)
        gen_logits = model.generate(outputs, input_ids)
        generated_text = tokenizer.decode(gen_logits.argmax(-1)[0])
        predicted = extract_answer(generated_text)
        prev_results.append(predicted if predicted else 0)
        
        # 6. Check confidence
        if cycle >= 1:
            conf = model.confidence_head(notebook_pages)
            if conf > conf_threshold:
                break
        
        # 7. Use NEXT cycle's scales (from this cycle's controller)
        scales = next_scales
    
    return predicted
```

---

## Cycle 1: The Constant Function Problem Vanishes

```
BEFORE (separate perceiver + hypernetwork):
  Cycle 1: perceiver reads hidden states → page (fine)
           hypernetwork reads EMPTY notebook → CONSTANT scales (broken!)
  
  The hypernetwork has NO input at cycle 1. Same scales for every problem.
  "Natalia sold clips" gets the same attention as "Mark has toys."

AFTER (unified controller):
  Cycle 1: controller reads hidden states → page AND scales (both informed!)
  
  Even with empty history, the controller reads CURRENT hidden states.
  "Natalia sold clips" → different hidden states → different scales
  "Mark has toys" → different hidden states → different scales
  
  The constant function problem is IMPOSSIBLE.
  The hidden states are always problem-specific.
```

---

## Why Intermediaries Collapsed

Every bypass and intermediary we built collapsed to a constant:

```
Message (32 floats):     collapsed — cos 0.998 between cycles
Bypass (512 floats):     collapsed — cos 0.999 between cycles  
Text context (16 floats): ignored — hypernetwork didn't use it
```

WHY did they all collapse? Because they were SEPARATE PROJECTIONS with no direct training signal. The generation loss trains Llama → atoms. The diversity loss trains scales. But nothing directly trains the bypass to carry useful content. The bypass projection learns "output a constant" because that minimizes its own implicit regularization (weight decay pushes toward zero → constant output).

The controller doesn't have this problem because:

```
The page head is trained by:     generation loss (text must be correct)
The scale head is trained by:    generation loss (atoms must steer correctly)
                                 diversity loss (scales must differ)
BOTH heads share the trunk:      gradient from both flows through shared understanding

The trunk can't collapse to a constant because the SCALE head demands
problem-specific output (different problems need different atoms).
And the scale head is directly trained — it's not a disconnected projection.
```

---

## Data Flow (Complete)

```
Problem text + text injection ("Step 1: 240")
  │
  ▼
Llama 3.2 1B (frozen, atom-modified attention)
  │
  ├── all 16 layers of hidden states
  │     │
  │     └── BreathingController reads ALL layers
  │           │
  │           ├── reads current hidden states (what Llama just computed)
  │           ├── attends over history (previous pages + hidden states)
  │           │
  │           ├── PAGE HEAD → 64-float page → Möbius → append to notebook
  │           │                (what was understood — the record)
  │           │
  │           ├── SCALE HEAD → 64 atom scales → applied to NEXT cycle's Llama
  │           │                (what to think about next — the plan)
  │           │
  │           └── FOCUS HEAD → 64 Möbius focus floats
  │                            (where on the sphere to record)
  │
  └── logits → GENERATION → "240 episodes #### 240</s>"
                              (the output, scored by extraction)
```

---

## Parameter Count

```
BEFORE (separate components):
  Perceiver:      105M
  Hypernetwork:   101M
  Bypass:           5M
  Message:        1.1M
  Text context:   0.3M
  ──────────────────
  Total:          212M

AFTER (unified controller):
  BreathingController:
    layer_weights:           16 params (negligible)
    current_project:         2048 × 1024 = 2.1M
    history_entry_project:   2112 × 1024 = 2.2M
    history_attn (4 layers): 4 × (1024² × 3 + 1024 × 4096 × 2) = ~50M
    trunk (3 layers):        3 × (1024² + 1024) = ~3.2M
    page_head:               1024→512→256→64 = 0.7M
    scale_head:              1024→512→256→64 = 0.7M
    focus_head:              1024→256→64 = 0.3M
  ──────────────────
  Total:          ~60M

  Or scaled up to ~130M with wider layers / more attention depth.
  Still LESS than the old 212M.
```

### Recommended Size: ~130M

```
Scale up for capacity:
  internal_dim:  1024 → 1536
  history_attn:  4 layers → 6 layers
  trunk:         3 layers → 4 layers
  
  Gives ~130M — comparable to the old perceiver alone,
  but doing BOTH jobs with shared understanding.
```

### Complete Architecture

```
Component                    Params      Role
──────────────────────────────────────────────────────────────
Llama 3.2 1B (frozen)        1,230M      Thinks (reads with modified attention)
BreathingController           130M       Observes (reads hidden states → page + scales)
64 LoRA Atoms (rank 6)         82M       Pattern library (attention modifications)
Confidence Head               2.5M       When to stop breathing
Möbius Transform                65K       Page diversity on hypersphere
──────────────────────────────────────────────────────────────
Total:                       ~1.44B
Trainable:                    ~215M      (was 300M — simpler AND smaller)
Frozen:                       1.23B
```

Two trainable pillars:
```
BreathingController (130M): OBSERVE — read what Llama computed, record + plan
Atoms (82M):                BREATHE — modify how Llama attends

The controller decides WHAT to attend to.
The atoms implement HOW to attend.
Two components. Two jobs. Clear separation.
```

---

## REMOVED Components

```
REMOVED          Params    Why
──────────────────────────────────────────────────────────
Perceiver        105M      Merged into controller (reads all 16 layers)
Hypernetwork     101M      Merged into controller (produces scales)
Bypass             5M      Redundant (controller reads hidden states directly)
Message          1.1M      Redundant (controller reads hidden states directly)
Text context     0.3M      Redundant (controller reads hidden states directly)
──────────────────────────────────────────────────────────
Total removed:   212M      Replaced by 130M controller
```

---

## The Elegant Picture

```
BEFORE (many components, many workarounds):
  Llama → Perceiver → page ────────────→ Hypernetwork → scales
  Llama → Bypass (collapsed) ───────────→ Hypernetwork
  Llama → Message (collapsed) ──────────→ Hypernetwork  
  Extraction → Text context (ignored) ──→ Hypernetwork
  
  Four inputs to the hypernetwork. Three collapsed. One too compressed.
  The hypernetwork was blind. It produced constant scales.

AFTER (one component, no workarounds):
  Llama → BreathingController → page + scales
  
  One reader. One understanding. Two outputs.
  The controller sees what Llama sees.
  It can't be blind — it reads the raw signal.
  It can't collapse — the scale head is directly trained.
  
  The system observes itself.
  Each breath: think → observe → record + plan → think again.
```

---

## What to Monitor

```
1. Scales diversity across problems at cycle 1:
   BEFORE: identical for ALL problems (constant function)
   TARGET: different for different problems (problem-specific)
   This is the PRIMARY test — the constant function must break.

2. Scales diversity across cycles:
   BEFORE: cos(1,2)=0.978, cos(2,3)=0.999 (fixed point)
   TARGET: cos(1,2)<0.7, cos(2,3)<0.7 (differentiated)

3. DC ratio (Laplace diagnostic):
   BEFORE: 0.90 (loop dead after cycle 1)
   TARGET: <0.5 (each cycle contributes)

4. Generation accuracy:
   cycle_1 should stay at 50%+ (controller provides good scales)
   cycle_2 should climb past 15% (controller differentiates cycles)
   The controller's shared understanding should help BOTH cycles.

5. Page-scale consistency:
   Are the page and scales telling the same story?
   If the page says "subtraction done" the scales should select
   atoms for the NEXT operation, not repeat subtraction.
   Shared trunk ensures this consistency.
```

---

## What NOT to Do

```
- Do NOT keep the perceiver alongside the controller.
  The controller IS the perceiver (reads all 16 layers, produces page).
  Keeping both wastes params and creates inconsistency.

- Do NOT keep the hypernetwork alongside the controller.
  The controller IS the hypernetwork (reads history, produces scales).
  Same argument — one reader, one understanding.

- Do NOT keep the bypass, message, or text context.
  The controller reads hidden states directly.
  All intermediaries are redundant.

- Do NOT make the controller too small.
  130M is right — it replaces 212M of separate components.
  The controller does the MOST IMPORTANT job in the system.
  Don't starve it. Two pillars: controller (130M) + atoms (82M).

- Do NOT remove the Möbius transform.
  The controller produces a focus vector alongside page and scales.
  The Möbius warps pages for notebook diversity.
  This is cheap (65K) and addresses a real problem (page clustering).
  The controller produces the focus — it's a natural third output.

- Do NOT remove the diversity loss immediately.
  Keep it at 0.05 as insurance while validating the controller.
  If scales naturally diversify (because hidden states differ per problem),
  the diversity loss can be removed later.
```
