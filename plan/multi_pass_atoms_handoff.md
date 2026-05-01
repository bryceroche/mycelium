# Handoff: Multi-Pass Atoms with Interleaved Observation

## One-Sentence Summary

N sequential sets of atoms applied through N Llama passes per breathing cycle, with the controller OBSERVING each pass's output before deciding the next pass's blend — creating two nested loops of self-observation (fast: within a cycle across atom layers, slow: across cycles for problem decomposition), all made cheap by caching base KV + the proven "math mode" transformation.

---

## The Architecture in One Picture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          BREATHING LOOP                                  │
│                                                                          │
│  CYCLE 0: COMPREHEND + CACHE                                             │
│    Vanilla Llama reads problem → cache base KV                           │
│    Apply L4.5 atoms (proven math mode) → cache math-mode KV              │
│    Controller reads → initial scales for cycle 1                         │
│                                                                          │
│  ┌─── CYCLE 1 ────────────────────────────────────────────────────┐     │
│  │                                                                 │     │
│  │  ┌─ INNER LOOP (fast timescale: coordinate atom layers) ─────┐ │     │
│  │  │                                                            │ │     │
│  │  │  Controller reads hidden states → scales_1                 │ │     │
│  │  │  Pass 1: atom_set_1 + scales_1 → Llama (math-mode KV)     │ │     │
│  │  │                    ↓                                       │ │     │
│  │  │  Controller reads hidden_states_1 → scales_2  (INFORMED)   │ │     │
│  │  │  Pass 2: atom_set_2 + scales_2 → Llama (math-mode KV)     │ │     │
│  │  │                    ↓                                       │ │     │
│  │  │  Controller reads hidden_states_2 → scales_3  (INFORMED)   │ │     │
│  │  │  Pass 3: atom_set_3 + scales_3 → Llama → GENERATE         │ │     │
│  │  │                                                            │ │     │
│  │  └────────────────────────────────────────────────────────────┘ │     │
│  │                                                                 │     │
│  │  Controller reads final hidden states                           │     │
│  │    → page (notebook record)                                     │     │
│  │    → confidence (keep cycling?)                                 │     │
│  │    → scales for cycle 2, pass 1                                 │     │
│  │                                                                 │     │
│  └─────────────────────────────────────────────────────────────────┘     │
│           ↓ text injection + notebook update                             │
│  ┌─── CYCLE 2 (same inner loop structure) ─────────────────────────┐    │
│  │  ...                                                             │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│           ↓ confidence > threshold? → STOP                               │
│                                                                          │
│  OUTER LOOP (slow timescale: decompose the problem into steps)           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Two Nested Loops of Self-Observation

The controller uses the SAME mechanism at two timescales:

```
INNER LOOP (within a cycle — coordinates atom layers):
  Observe pass 1 → "pass 1 found the quantities"
  Decide:        → "pass 2 should focus on the operation" 
  Observe pass 2 → "pass 2 set up the equation"
  Decide:        → "pass 3 should complete the computation"
  
  Like tasting between each ingredient addition while cooking.

OUTER LOOP (across cycles — coordinates decomposition):
  Observe cycle 1 → "cycle 1 computed 48/2=24"
  Decide:         → "cycle 2 should add 48+24"
  Observe cycle 2 → "cycle 2 computed 48+24=72"
  Decide:         → "confidence high, stop cycling"
  
  Like completing one paragraph before planning the next.

SAME MECHANISM: read hidden states → trunk → produce scales/page/confidence
TWO TIMESCALES: fast (atom layers) and slow (breathing cycles)
```

---

## Why Interleaved, Not All-At-Once

```
ALL-AT-ONCE (blind planning):
  Controller → [scales_1, scales_2, scales_3] → run all passes
  
  scales_2 decided BEFORE pass 1 runs.
  The controller GUESSES what pass 1 will do.
  Like writing a recipe without tasting between steps.

INTERLEAVED (informed planning):
  Controller → scales_1 → run pass 1 → OBSERVE
  Controller reads pass 1 output → scales_2 → run pass 2 → OBSERVE
  Controller reads pass 2 output → scales_3 → run pass 3 → generate
  
  Each decision is INFORMED by actual computation.
  The controller ADAPTS based on what each pass produced.
  Like cooking with tasting between each addition.
```

The cost of interleaving is negligible — the controller (190M) is ~15% of Llama (1.23B). Running the controller N extra times per cycle adds ~5% overhead. The Llama passes dominate.

---

## The Principle: Capacity Without Prescription

We don't label what each pass does. We provide:

```
PROVIDE:
  - N atom sets (N × 82M, different random init)
  - N sequential Llama passes with cached KV
  - Interleaved controller observation between passes
  - Cycle and pass awareness (embeddings, not constraints)
  - Previous scales as context (what was just done)
  - Full gradient flow through both loops

DON'T PROVIDE:
  - Labels ("parse layer", "compute layer")
  - Constraints on what each pass handles
  - Prescribed roles or ordering
  - Any assumption about emergent specialization
```

Gradient discovers what each pass should do. Maybe pass 1 handles language understanding. Maybe pass 2 handles arithmetic. Maybe it's something we never imagined. We observe after training. We don't prescribe before.

---

## Three-Tier KV Caching

### Tier 1: Base KV (vanilla Llama, computed once)

```
K_base = W_k @ x    for all problem tokens
V_base = W_v @ x    for all problem tokens

Computed at cycle 0. Reused everywhere. This is the expensive part.
```

### Tier 2: Math-Mode KV (base + L4.5 atoms, computed once)

```
K_math = K_base + L4.5_lora_delta_k
V_math = V_base + L4.5_lora_delta_v

The L4.5 atoms (proven at 99.5% on procedural math) transform Llama
from "multiple choice mode" into "math computation mode."

This transformation is ALWAYS applied. It's the foundation.
Computed once at cycle 0, right after base KV. Cached and reused.
```

### Tier 3: Per-Pass Deltas (new atom sets, computed per pass)

```
K_pass_n = K_math + atom_set_n_delta_k    (cheap! rank-6 delta)
V_pass_n = V_math + atom_set_n_delta_v    (cheap! rank-6 delta)

Each pass's atom set adds its OWN delta on top of the cached math-mode KV.
Different delta per pass. Different atom set. Different blend.
Only the delta is computed — the base + math-mode are cached.
```

```
The layered cache:
  Tier 1 (base):       W_k @ x                          — expensive, cached
  Tier 2 (math mode):  base + L4.5 delta                — computed once, cached
  Tier 3 (per pass):   math-mode + new atom delta        — cheap, per pass

  Pass 1: Tier 2 (cached) + atom_set_1 delta = 0.03 cost
  Pass 2: Tier 2 (cached) + atom_set_2 delta = 0.03 cost
  Pass 3: Tier 2 (cached) + atom_set_3 delta = 0.03 cost
```

### Cost Analysis

```
3-step problem, N=3 passes:

Without any cache:
  Cycle 0: 1 full forward                                    = 1.0
  3 cycles × 3 passes × 1 full forward                       = 9.0
  Total:                                                      = 10.0

With 3-tier cache:
  Cycle 0: 1 full forward + L4.5 delta → cache tiers 1+2     = 1.3
  3 cycles × 3 passes × tier-3 delta only                     = 3 × 3 × 0.03 = 0.27
  3 cycles × N controller reads (interleaved)                 = 3 × 3 × 0.05 = 0.45
  Total:                                                      = 2.0

  5x CHEAPER than no cache. With 3x richer processing per step.

For comparison:
  Current N=1, no cache:  4.0 cost units
  Multi-pass N=3, cached: 2.0 cost units  ← HALF the cost AND 3x richer!
```

---

## L4.5 Atoms as Frozen Math-Mode Foundation

The L4.5 atoms scored 99.5% on procedural math. They KNOW how to reprogram Llama from text completion to math computation (the V,O discovery). They are PROVEN.

```
L4.5 atoms:  FROZEN (or very low LR)
             Always applied as Tier 2 cache
             Provide the "math mode" foundation
             Never change — stable base for everything built on top

New atom sets: TRAINABLE
              Applied as Tier 3 per-pass deltas
              Handle GSM8K's language diversity ON TOP of math mode
              Each set discovers its role through gradient
```

```
L4.5 atoms:    "Llama, you are a math engine" (always on, proven, cached)
New set 1:     "for THIS specific problem, focus on..." (trainable, per-pass)
New set 2:     "refine by..." (trainable, per-pass)
New set 3:     "finalize by..." (trainable, per-pass)

The foundation doesn't change. The refinements adapt per problem.
```

### Risk: L4.5 atoms were trained at [-3, 3]

The L4.5 atoms' weights were optimized for loud application. Two options:

```
Option A: Apply L4.5 atoms at THEIR trained scale (loud math mode)
          New atom sets at [-0.4] (gentle refinement)
          Strong foundation + gentle refinement

Option B: Apply L4.5 atoms at [-0.4] (same as new atoms)
          Eval showed 20.3% with this clamping — it partially works
          But might lose some math-mode reprogramming

Start with Option A. The L4.5 atoms provide strong math mode.
New atoms provide gentle problem-specific refinement.
If arithmetic corruption returns, try Option B.
```

---

## Controller with Full Awareness

```python
class BreathingController(nn.Module):
    """
    Observes each pass's hidden states and produces the NEXT pass's blend.
    Runs between every pass (interleaved observation).
    
    Awareness:
      - Hidden states (what Llama just computed — PRIMARY input)
      - Notebook history (what previous cycles understood)
      - Cycle number (which breath we're on)
      - Pass number (which atom layer within this breath)
      - Previous scales (what atom blend was just used)
    
    No prescribed roles. Full context. Gradient decides.
    """
    def __init__(self, num_atoms=64, internal_dim=1024, page_dim=64,
                 max_cycles=12, max_passes=4, hidden_dim=2048):
        super().__init__()
        
        # Read current hidden states (all 16 layers)
        self.layer_weights = nn.Parameter(torch.ones(16) / 16)
        self.current_project = nn.Sequential(
            nn.Linear(hidden_dim, internal_dim),
            nn.LayerNorm(internal_dim),
            nn.GELU(),
        )
        
        # Read notebook history (cross-attention over previous entries)
        self.history_entry_project = nn.Linear(page_dim + hidden_dim, internal_dim)
        self.history_attn = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(internal_dim, 8, batch_first=True),
                'norm1': nn.LayerNorm(internal_dim),
                'ff': nn.Sequential(
                    nn.Linear(internal_dim, internal_dim * 4),
                    nn.GELU(),
                    nn.Linear(internal_dim * 4, internal_dim),
                ),
                'norm2': nn.LayerNorm(internal_dim),
            })
            for _ in range(4)
        ])
        
        # Awareness embeddings (context, not constraints)
        self.cycle_embed = nn.Embedding(max_cycles, 64)
        self.pass_embed = nn.Embedding(max_passes, 64)
        self.prev_scale_project = nn.Linear(num_atoms, 64)
        
        # Shared trunk
        # Input: current (1024) + history (1024) + cycle (64) + pass (64) + prev_scales (64)
        trunk_input = internal_dim * 2 + 64 * 3  # 2240
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input, internal_dim),
            nn.LayerNorm(internal_dim),
            nn.GELU(),
            nn.Linear(internal_dim, internal_dim),
            nn.LayerNorm(internal_dim),
            nn.GELU(),
            nn.Linear(internal_dim, internal_dim),
            nn.LayerNorm(internal_dim),
            nn.GELU(),
        )
        
        # Output heads
        self.scale_head = nn.Sequential(
            nn.Linear(internal_dim, 512),
            nn.GELU(),
            nn.Linear(512, num_atoms),
        )
        
        self.page_head = nn.Sequential(
            nn.Linear(internal_dim, 512),
            nn.GELU(),
            nn.Linear(512, page_dim),
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(internal_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        
        self.focus_head = nn.Sequential(
            nn.Linear(internal_dim, 256),
            nn.GELU(),
            nn.Linear(256, page_dim),
        )
    
    def forward(self, hidden_states_all_layers, history_pages, history_hiddens,
                cycle_num, pass_num, prev_scales=None):
        """
        Called BETWEEN each pass (interleaved observation).
        Reads what the last pass produced. Decides what the next pass should do.
        """
        # Read current hidden states
        weights = F.softmax(self.layer_weights, dim=0)
        combined = sum(w * h.mean(dim=1) for w, h in 
                      zip(weights, hidden_states_all_layers))
        current = self.current_project(combined)
        
        # Read history (cross-attention over notebook)
        if len(history_pages) > 0:
            entries = [self.history_entry_project(torch.cat([p, h], dim=-1))
                      for p, h in zip(history_pages, history_hiddens)]
            history_seq = torch.stack(entries, dim=1)
            query = current.unsqueeze(1)
            for layer in self.history_attn:
                attn_out, _ = layer['attn'](query, history_seq, history_seq)
                query = layer['norm1'](query + attn_out)
                query = layer['norm2'](query + layer['ff'](query))
            history_ctx = query.squeeze(1)
        else:
            history_ctx = torch.zeros_like(current)
        
        # Awareness context
        cycle_ctx = self.cycle_embed(torch.tensor(cycle_num, device=current.device))
        cycle_ctx = cycle_ctx.expand(current.size(0), -1)
        
        pass_ctx = self.pass_embed(torch.tensor(pass_num, device=current.device))
        pass_ctx = pass_ctx.expand(current.size(0), -1)
        
        if prev_scales is not None:
            prev_ctx = self.prev_scale_project(prev_scales)
        else:
            prev_ctx = torch.zeros(current.size(0), 64, device=current.device)
        
        # Combine everything
        combined = torch.cat([current, history_ctx, cycle_ctx, pass_ctx, prev_ctx], dim=-1)
        shared = self.trunk(combined)
        
        # Outputs
        scales = torch.tanh(self.scale_head(shared)).clamp(-0.4, 0.4)
        page = F.normalize(self.page_head(shared), dim=-1) * math.sqrt(64)
        confidence = self.confidence_head(shared)
        focus = self.focus_head(shared)
        
        return scales, page, confidence, focus
```

---

## The Interleaved Cycle

```python
def think_one_cycle(model, controller, atom_sets, math_mode_kv,
                     notebook, cycle_num, prev_results, tokenizer):
    """
    One breathing cycle with interleaved observation.
    
    The controller observes each pass's output before deciding the next.
    Like tasting between each ingredient addition.
    """
    text_ctx = format_text_injection(prev_results)
    cycle_input = tokenize(text_ctx + problem_text)
    
    current_hidden = notebook.last_hidden if notebook.size > 0 else vanilla_hidden
    prev_scales = None
    
    # === INNER LOOP: N passes with interleaved observation ===
    for pass_idx in range(len(atom_sets)):
        
        # OBSERVE: controller reads current state → decides this pass's blend
        scales, page, confidence, focus = controller(
            current_hidden,
            notebook.pages, notebook.hiddens,
            cycle_num=cycle_num,
            pass_num=pass_idx,
            prev_scales=prev_scales,
        )
        
        # ACT: apply this pass's atoms with the decided blend
        clamped_scales = scales.clamp(-0.4, 0.4)
        atom_sets[pass_idx].apply_to_llama(model, clamped_scales)
        
        outputs = model.llama(
            cycle_input,
            output_hidden_states=True,
            past_key_values=math_mode_kv,  # Tier 2 cache (math mode always on)
        )
        
        atom_sets[pass_idx].remove_from_llama(model)
        
        # Update state for next observation
        current_hidden = outputs.hidden_states
        prev_scales = clamped_scales
    
    # === GENERATE from final pass ===
    gen_logits = model.generate(outputs, cycle_input)
    generated = tokenizer.decode(gen_logits.argmax(-1)[0])
    predicted = extract_answer(generated)
    
    # === RECORD: controller produces page + confidence for outer loop ===
    final_scales, page, confidence, focus = controller(
        current_hidden,
        notebook.pages, notebook.hiddens,
        cycle_num=cycle_num,
        pass_num=len(atom_sets),  # "post-generation" pass
        prev_scales=prev_scales,
    )
    
    warped_page = mobius(page, focus)
    hidden_pool = current_hidden[-1].mean(dim=1)
    notebook.append(warped_page, hidden_pool)
    
    return gen_logits, predicted, confidence, final_scales
```

---

## The Full Solve Loop

```python
def solve(model, controller, atom_sets, l45_atoms, problem_ids,
          tokenizer, max_cycles=8, conf_threshold=0.85):
    """
    Solve by breathing: N interleaved passes (depth) × C cycles (length).
    Math-mode cached. Controller observes between every pass and cycle.
    """
    notebook = Notebook()
    prev_results = []
    
    # === CYCLE 0: COMPREHEND + BUILD CACHE ===
    
    # Tier 1: base KV
    vanilla_out = model.llama(problem_ids, output_hidden_states=True)
    base_kv = cache_base_kv(vanilla_out)
    
    # Tier 2: math-mode KV (L4.5 atoms, proven, frozen)
    l45_atoms.apply_to_llama(model)
    math_out = model.llama(problem_ids, output_hidden_states=True,
                            past_key_values=base_kv)
    l45_atoms.remove_from_llama(model)
    math_mode_kv = cache_kv(math_out)  # cached for all future passes
    
    # Initial controller observation
    initial_scales, page_0, _, _ = controller(
        math_out.hidden_states, [], [],
        cycle_num=0, pass_num=0, prev_scales=None
    )
    notebook.append(page_0, math_out.hidden_states[-1].mean(dim=1))
    
    # === OUTER LOOP: breathing cycles ===
    for cycle in range(max_cycles):
        
        gen_logits, predicted, confidence, next_scales = think_one_cycle(
            model, controller, atom_sets, math_mode_kv,
            notebook, cycle_num=cycle, 
            prev_results=prev_results, tokenizer=tokenizer
        )
        
        prev_results.append(predicted if predicted else 0)
        
        # LENGTH control: should we keep cycling?
        if cycle >= 1 and confidence > conf_threshold:
            break
    
    return predicted
```

---

## Gradient Flow (Both Loops)

```
INNER LOOP gradient (within a cycle):

gen_loss → pass 3 output
         → atom_set_3 scales → controller (read hidden_states_2) → trunk
         → hidden_states_2 → atom_set_2 scales → controller (read hidden_states_1) → trunk
         → hidden_states_1 → atom_set_1 scales → controller (read initial state) → trunk

All atom sets get gradient. All controller decisions get gradient.
The controller learns: "my scales_1 choice affected hidden_states_1
which affected my scales_2 choice which affected the final generation."

OUTER LOOP gradient (across cycles):

cycle 2 gen_loss → cycle 2 controller reads notebook
                 → notebook contains cycle 1's page + hidden states
                 → cycle 1's atom sets get cross-cycle gradient

Atoms at cycle 1 learn: "my output affected cycle 2's success."
Confirmed working: gradient norms 0.27-0.31 (meaningful).

COMBINED: the gradient flows through BOTH nested loops.
Inner loop: how to coordinate atom layers within a step.
Outer loop: how to coordinate steps across the problem.
The controller is the connective tissue at both timescales.
```

---

## Parameter Budget

```
Component                    Params     Role
──────────────────────────────────────────────────────────────
Llama 3.2 1B (frozen)        1,230M    Thinks
L4.5 Atoms (frozen)            82M     Math mode foundation (cached in Tier 2)
Atom Set 1 (trainable)         82M     First refinement pass
Atom Set 2 (trainable)         82M     Second refinement pass
(Atom Set 3 if N=3)           (82M)    (Third refinement pass)
Controller (trainable)         190M    Observes + decides (both loops)
Confidence Head (in controller) incl.  Length control
Möbius (in controller)          incl.  Page diversity
──────────────────────────────────────────────────────────────
N=2: Total trainable = 354M    Total with frozen = 1,666M
N=3: Total trainable = 436M    Total with frozen = 1,748M

All fit in 24GB VRAM.
```

---

## Training

```python
def train_step(model, controller, atom_sets, l45_atoms, 
               problem_ids, cycle_gen_targets, kv_caches, 
               num_cycles, tokenizer):
    
    notebook = Notebook()
    total_loss = 0.0
    prev_results = []
    available_targets = list(cycle_targets)
    
    # Build Tier 1+2 cache
    vanilla_out = model.llama(problem_ids, output_hidden_states=True)
    base_kv = cache_base_kv(vanilla_out)
    
    l45_atoms.apply_to_llama(model)
    math_out = model.llama(problem_ids, past_key_values=base_kv,
                            output_hidden_states=True)
    l45_atoms.remove_from_llama(model)
    math_mode_kv = cache_kv(math_out)
    
    # Initial observation
    scales, page_0, _, _ = controller(
        math_out.hidden_states, [], [],
        cycle_num=0, pass_num=0, prev_scales=None
    )
    notebook.append(page_0, math_out.hidden_states[-1].mean(dim=1))
    
    for cycle in range(num_cycles):
        cycle_input = tokenize(format_text_injection(prev_results) + problem_text)
        
        # Inner loop: N interleaved passes
        current_hidden = notebook.last_hidden
        prev_scales = None
        
        for pass_idx in range(len(atom_sets)):
            scales, page, conf, focus = controller(
                current_hidden, notebook.pages, notebook.hiddens,
                cycle_num=cycle, pass_num=pass_idx, prev_scales=prev_scales
            )
            
            outputs = atom_sets[pass_idx].forward_with_cache(
                model, cycle_input, scales, math_mode_kv
            )
            current_hidden = outputs.hidden_states
            prev_scales = scales
        
        # Generate + loss
        gen_logits = model.generate(outputs, cycle_input)
        gen_loss = weighted_generation_loss(
            gen_logits, cycle_gen_targets[cycle], tokenizer, eos_weight=5.0
        )
        
        # Four-tier gating
        with torch.no_grad():
            text = tokenizer.decode(gen_logits.argmax(-1)[0])
            predicted = extract_answer(text)
            self_consistent = check_computation_correct(text)
        
        consumed = [t for t in cycle_targets if t not in available_targets]
        if predicted in consumed:
            gen_weight = 0.0
        elif predicted in available_targets or predicted == final_answer:
            gen_weight = 1.0
            if predicted in available_targets:
                available_targets.remove(predicted)
        elif self_consistent:
            gen_weight = 1.0
        else:
            gen_weight = 0.2
        
        teacher_weight = per_cycle_target_weight(final_accuracy, cycle, num_cycles)
        total_loss += teacher_weight * gen_weight * gen_loss
        
        # Record
        _, page, conf, focus = controller(
            current_hidden, notebook.pages, notebook.hiddens,
            cycle_num=cycle, pass_num=len(atom_sets), prev_scales=prev_scales
        )
        notebook.append(mobius(page, focus), current_hidden[-1].mean(dim=1))
        prev_results.append(predicted if predicted else 0)
    
    # Regularizers
    total_loss += 0.01 * isotropic_reg(raw_pages)
    total_loss += 0.1 * confidence_entropy_loss(notebook, final_answer)
    
    return total_loss
```

---

## Discovery Protocol

```
Step 1: N=2 on L3 → L4 → L4.5 (with cached L4.5 math mode)
        Validate: multi-pass works, curriculum transfers
        Target: 99%+ (match single-pass results)

Step 2: N=2 on single-step GSM8K
        Key test: does the second pass help with language diversity?
        Target: beat 20.3% single-pass ceiling

Step 3: If N=2 helps, try N=3
        Diminishing returns → stop

Step 4: Best N on multi-step GSM8K
        Full system: depth × length with interleaved observation
        Target: beat 17.8% CoT ceiling

Step 5: Observe what emerged
        Per-pass atom activation patterns
        What did gradient discover?
        DON'T label — observe
```

---

## What to Monitor

```
1. Inner loop observation value:
   Does the controller produce DIFFERENT scales after observing each pass?
   If scales change between observations: interleaving helps
   If scales are the same regardless: all-at-once would suffice

2. Per-pass hidden state delta:
   How much does each pass change the representation?
   Healthy: each pass makes meaningful changes
   Unhealthy: only one pass matters, others are dead

3. Math-mode cache validity:
   Does the L4.5 foundation help GSM8K?
   Compare: with L4.5 cache vs without
   If L4.5 cache helps: the math-mode foundation transfers

4. Interleaved vs all-at-once accuracy:
   Does interleaved observation beat all-at-once?
   If yes: the controller benefits from mid-cycle feedback
   If no: simplify to all-at-once (less overhead)

5. Per-pass atom specialization:
   Do different atom sets activate different atoms?
   DON'T label what each set does — just observe whether they differ

6. Confidence calibration:
   Does the confidence head stop at the right time?
   Easy problems: 1-2 cycles
   Hard problems: 4-5 cycles

7. KV cache speedup:
   Verify theoretical 5x speedup is real
   Tier 2 (math-mode) cache hit rate
```

---

## What NOT to Do

```
- Do NOT label the passes or atom sets
- Do NOT constrain what each set can learn
- Do NOT use same random init for all new atom sets (symmetry breaking)
- Do NOT unfreeze L4.5 atoms initially (proven foundation, preserve it)
- Do NOT remove the KV cache (makes everything affordable)
- Do NOT skip the curriculum (L3 → L4 → L4.5 → GSM8K)
- Do NOT conflate inner loop (atom layers) with outer loop (breathing cycles)
- Do NOT prescribe cycle or pass awareness as constraints
  (they're embeddings = information, not hard constraints)
```

---

## The Elegant Picture

```
A system that observes itself at two timescales:

FAST (within each breath):
  Think with atoms → observe → adjust atoms → think again → observe → adjust
  Each pass refines the previous pass's work.
  The controller coordinates atom layers through observation.

SLOW (across breaths):
  Complete one step → record → plan next step → complete → record → plan
  Each cycle builds on previous cycles.
  The controller coordinates decomposition through the notebook.

FOUNDATION (always on):
  L4.5 atoms cached as math mode.
  Llama is ALWAYS a math engine, never a multiple-choice bot.
  Proven. Frozen. Cached. The bedrock everything builds on.

REFINEMENT (learned per problem):
  N atom sets discover what to do through gradient.
  The controller observes and adapts at every step.
  No prescribed roles. Emergent specialization.

Two loops. One controller. One mechanism. Two timescales.
The system breathes, observes, and adapts.
Depth × Length. Quality × Decomposition.
Capacity without prescription. Structure without constraints.
Gradient discovers. We observe.
```
