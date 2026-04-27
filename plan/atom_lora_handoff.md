# Handoff: 64-Atom LoRA Architecture

## One-Sentence Summary

Replace named LoRA templates (parse/compute/verify/answer) with 64 anonymous rank-6 atoms (~100M params), independently scaled by a 10M-param hypernetwork. The model discovers its own cognitive decomposition. No softmax, no mode collapse, no naming. Symmetric capacity: 105M compression (perceiver), 100M expansion (atoms).

---

## Why This Architecture

### The Problem With Named Modes

We hand-designed 4 modes (parse, compute, verify, answer) based on how HUMANS think. But:
- The model might discover different decompositions that don't map to human categories
- The 4-way softmax caused mode collapse (one mode dominated, three died)
- Entropy regularization prevented collapse but also prevented differentiation
- L3 problems didn't need 4 modes — most blend was uniform at 0.25 each
- We were imposing structure the model didn't need

### The Solution: Anonymous Atoms

64 rank-6 LoRA atoms. Each is an independent direction of attention modification. The hypernetwork outputs 64 scalars controlling each atom independently. No grouping. No softmax. No naming. No competition between atoms.

```
Named modes (old):     softmax([parse, compute, verify, answer]) → one wins, three lose
Anonymous atoms (new):  tanh([atom_1, atom_2, ..., atom_64]) → all contribute independently
```

The model discovers what each atom does through training. Easy problems activate 5-10 atoms. Hard problems activate 30-40. We inspect what each atom learned AFTER training, not before.

### Symmetric Capacity

```
BEFORE:                              AFTER:
  Compress: 105M (perceiver)           Compress: 105M (perceiver)
  Expand:   4.4M (4 templates)         Expand:   100M (64 atoms)
  Ratio:    24:1 (massively            Ratio:    ~1:1 (balanced)
             imbalanced)
```

The expand-collapse cycle now has equal capacity on both sides. The model can compress richly AND expand richly.

---

## Architecture

### 64 Rank-6 LoRA Atoms (~100M params)

Each atom is a rank-6 modification to one attention projection at one layer:

```python
class LoRAAtoms(nn.Module):
    def __init__(self, d_model=2048, rank=6, num_atoms=64, num_layers=16):
        super().__init__()
        self.num_atoms = num_atoms
        self.rank = rank
        self.num_layers = num_layers
        
        proj_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        
        # 64 atoms, each with A and B matrices for each layer and projection
        # A: (d_model, rank), B: (rank, proj_dim)
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        
        for proj_idx, proj_name in enumerate(proj_names):
            proj_dim = 512 if proj_name in ['k_proj', 'v_proj'] else d_model  # GQA
            self.A[proj_name] = nn.Parameter(
                torch.randn(num_atoms, num_layers, d_model, rank) * 0.01
            )
            self.B[proj_name] = nn.Parameter(
                torch.randn(num_atoms, num_layers, rank, proj_dim) * 0.01
            )
    
    def apply(self, hidden, layer_idx, proj_name, atom_scales):
        """
        hidden:      (batch, seq, d_model)
        atom_scales: (batch, 64) — one scalar per atom
        returns:     (batch, seq, proj_dim) — additive LoRA output
        """
        A = self.A[proj_name][:, layer_idx]  # (64, d_model, rank)
        B = self.B[proj_name][:, layer_idx]  # (64, rank, proj_dim)
        
        # Compute each atom's contribution
        # hidden @ A_i → (batch, seq, rank) for each atom
        # scale and sum across atoms
        lora_out = torch.zeros(
            hidden.size(0), hidden.size(1), B.size(-1), 
            device=hidden.device, dtype=hidden.dtype
        )
        
        for i in range(self.num_atoms):
            scale = atom_scales[:, i:i+1].unsqueeze(1)  # (batch, 1, 1)
            contribution = (hidden @ A[i]) @ B[i]        # (batch, seq, proj_dim)
            lora_out = lora_out + scale * contribution
        
        return lora_out
```

Note: the naive per-atom loop above is for clarity. Production code should batch this:

```python
# Batched version (much faster):
# hidden: (batch, seq, d_model)
# A: (64, d_model, rank) → einsum

# All atoms at once:
# (batch, seq, d_model) @ (64, d_model, rank) → (batch, 64, seq, rank)
projections = torch.einsum('bsd,adr->basr', hidden, A)

# Scale each atom: (batch, 64, seq, rank) * (batch, 64, 1, 1)
scaled = projections * atom_scales.unsqueeze(-1).unsqueeze(-1)

# Sum across atoms and project: (batch, seq, rank) @ (64, rank, proj_dim)
# → sum over atoms → (batch, seq, proj_dim)
lora_out = torch.einsum('basr,arp->bsp', scaled, B)
```

### 10M Hypernetwork (The Brain)

Reads accumulated pages + strategy + pass number. Outputs 64 scalars controlling each atom independently.

```python
class AtomHypernetwork(nn.Module):
    def __init__(self, page_size=64, strategy_size=64, 
                 pass_embed_dim=512, num_atoms=64):
        super().__init__()
        
        # === Page reading (2-layer attention over accumulated pages) ===
        self.page_project = nn.Linear(page_size, 512)
        
        self.page_query = nn.Parameter(torch.randn(4, 512))
        self.page_attn_1 = nn.MultiheadAttention(512, 8, batch_first=True)
        self.page_norm_1 = nn.LayerNorm(512)
        self.page_ffn_1 = nn.Sequential(
            nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, 512)
        )
        self.page_ffn_norm_1 = nn.LayerNorm(512)
        
        self.page_attn_2 = nn.MultiheadAttention(512, 8, batch_first=True)
        self.page_norm_2 = nn.LayerNorm(512)
        self.page_ffn_2 = nn.Sequential(
            nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, 512)
        )
        self.page_ffn_norm_2 = nn.LayerNorm(512)
        
        # 4 queries × 512 dim = 2048 flattened
        self.summary_project = nn.Linear(2048, 1024)
        
        # === Strategy + pass integration ===
        self.strategy_project = nn.Linear(strategy_size, 512)
        self.pass_embed = nn.Embedding(20, 512)
        
        # === Deep MLP: 1024 + 512 + 512 = 2048 → 64 atom scales ===
        self.scale_net = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, num_atoms),
            nn.Tanh(),  # bounded [-1, 1], no softmax, no competition
        )
    
    def forward(self, state_pages, strategy, pass_num):
        """
        state_pages: list of (batch, 64) tensors
        strategy: (batch, 64)
        pass_num: int
        returns: (batch, 64) atom scales
        """
        batch_size = state_pages[0].size(0) if len(state_pages) > 0 else strategy.size(0)
        device = strategy.device
        
        # Read pages via 2-layer attention
        if len(state_pages) > 0:
            pages = torch.stack(state_pages, dim=1)       # (batch, N, 64)
            pages_proj = self.page_project(pages)          # (batch, N, 512)
            
            queries = self.page_query.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Layer 1
            att1, _ = self.page_attn_1(query=queries, key=pages_proj, value=pages_proj)
            queries = self.page_norm_1(queries + att1)
            queries = self.page_ffn_norm_1(queries + self.page_ffn_1(queries))
            
            # Layer 2
            att2, _ = self.page_attn_2(query=queries, key=pages_proj, value=pages_proj)
            queries = self.page_norm_2(queries + att2)
            queries = self.page_ffn_norm_2(queries + self.page_ffn_2(queries))
            
            page_summary = self.summary_project(queries.flatten(1))  # (batch, 1024)
        else:
            page_summary = torch.zeros(batch_size, 1024, device=device)
        
        # Strategy + pass
        strat = self.strategy_project(strategy)          # (batch, 512)
        pass_ctx = self.pass_embed(
            torch.tensor(pass_num, device=device)
        ).unsqueeze(0).expand(batch_size, -1)            # (batch, 512)
        
        # Combine and generate atom scales
        combined = torch.cat([page_summary, strat, pass_ctx], dim=-1)  # (batch, 2048)
        atom_scales = self.scale_net(combined)            # (batch, 64)
        
        return atom_scales
```

### Integration Into Thinking Loop

```python
def think_one_pass(self, problem_ids, state_pages, strategy, pass_num):
    # Hypernetwork generates 64 atom scales
    atom_scales = self.hypernetwork(state_pages, strategy, pass_num)
    
    # Apply 64-atom LoRA to all layers (additive, no hooks)
    # Each layer's Q,K,V,O projections get:
    #   output = W @ x + atoms.apply(x, layer_idx, proj_name, atom_scales)
    self.apply_atom_lora(atom_scales)
    
    # Forward through Llama with atom-modified attention
    outputs = self.llama(problem_ids, output_hidden_states=True)
    all_layer_hidden = list(outputs.hidden_states[1:])
    
    # Remove LoRA
    self.remove_atom_lora()
    
    # Perceiver compresses all 16 layers → page + strategy
    page, strategy = self.perceiver(all_layer_hidden, pass_num)
    page = F.normalize(page, dim=-1) * math.sqrt(64)
    
    # Gradient scaling for earlier cycles
    grad_scale = min(float(max_passes - pass_num), 4.0)
    page = scale_gradient(page, grad_scale)
    
    return page, strategy, atom_scales
```

---

## No Softmax, No Mode Collapse

The critical design choice: atom scales use Tanh, not Softmax.

```
Softmax: atoms COMPETE. Activating atom 1 suppresses atom 2.
         Leads to mode collapse — one atom wins, 63 die.
         Requires entropy regularization to prevent.

Tanh:    atoms are INDEPENDENT. Activating atom 1 doesn't affect atom 2.
         Each atom contributes additively. No competition.
         No mode collapse possible. No entropy regularization needed.
         
         scale = tanh(logit)
         scale = -1: atom actively reverses this attention modification
         scale =  0: atom inactive
         scale = +1: atom fully active
```

The model naturally learns sparsity — most atoms at near-zero for easy problems (only 5-10 active), most atoms active for hard problems (30-40 active). The sparsity emerges from training, not from architectural constraint.

---

## Parameter Budget

```
Component                    Params      % of total    Role
───────────────────────────────────────────────────────────────
Llama 3.2 1B (frozen)        1,230M      89.7%         The thinker
7-Layer Perceiver             105M       7.7%          Compress (COLLAPSE)
64 LoRA Atoms (rank 6)        100M       7.3%          Expand (EXPAND)
Atom Hypernetwork              10M       0.7%          Control (which atoms)
Page-to-Tokens                0.5M       0.04%         Generation bridge
Answer Head                   4K         —             Digit extraction
Confidence Head               20K        —             Stop decision
───────────────────────────────────────────────────────────────
Total:                       ~1.45B
Trainable:                   ~216M      (14.9%)

Symmetry: 105M compress ≈ 100M expand ≈ 10M control
```

---

## What We No Longer Need

```
REMOVED:
  - Named templates (parse/compute/verify/answer)
  - Softmax blend (4-way or 8-way)
  - Entropy regularization (no competition → no collapse)
  - Blend history tracking (no blend weights)
  - Mode-specific gradient debugging
  - Strategy side channel (64-float ephemeral signal — redundant with pages)
    The hypernetwork attends over ALL pages. The strategy added nothing
    the pages don't already provide. The perceiver now outputs ONLY a page.

KEPT:
  - 7-layer Perceiver (simplified — outputs page only, no strategy)
  - Page-based state (append, no overwrite)
  - 64-float pages (one per pass, accumulated)
  - Pass-conditioned hypernetwork
  - Contrastive page loss (target-cos 0.7)
  - Anti-copying (soft quadratic at 0.7)
  - Gradient scaling per cycle (capped at 4x)
  - Answer head (sign + length + digits)
  - Confidence head (reads pages, correctness-trained)
  - CoT targets for generation
  - Fresh data per epoch
```

## Perceiver Change: Strategy Removed

```python
# BEFORE:
class Perceiver:
    def __init__(self):
        self.project_state = nn.Linear(d_perceiver, 16)    # per query → 64 total
        self.project_strategy = nn.Linear(d_perceiver, 16)  # per query → 64 total
    
    def forward(self, hidden_states, pass_num):
        # ... 7 layers of attention ...
        page = cat([project_state(q) for q in queries])       # 64 floats
        strategy = cat([project_strategy(q) for q in queries]) # 64 floats
        return page, strategy

# AFTER:
class Perceiver:
    def __init__(self):
        self.project_page = nn.Linear(d_perceiver, 16)  # per query → 64 total
    
    def forward(self, hidden_states, pass_num):
        # ... 7 layers of attention ...
        page = cat([project_page(q) for q in queries])  # 64 floats
        return page  # that's it. No strategy.
```

The hypernetwork input changes correspondingly:

```python
# BEFORE: page_summary (1024) + strategy (512→64) + pass_embed (512) = 1600
# AFTER:  page_summary (1024) + pass_embed (512) = 1536

# The hypernetwork just reads pages + pass number. No strategy.
```

---

## Future: Binary Tree Search (Inference-Time Only)

At inference time, we can explore MULTIPLE thinking strategies by branching:

```
TRAINING:   linear thinking (one path per pass, standard backprop)
INFERENCE:  optionally branched (K paths at pass 1, prune, continue linear)

Pass 1: branch into K paths
  Path A: hypernetwork generates scales_A → Llama → page_1A
  Path B: hypernetwork generates scales_B → Llama → page_1B
  Path C: hypernetwork generates scales_C → Llama → page_1C
  Path D: hypernetwork generates scales_D → Llama → page_1D
  
  Evaluate: confidence head picks the best path
  Prune: keep best, discard rest
  
Pass 2-N: linear (continue from best path)
```

The branching comes from learned perturbation vectors in the hypernetwork:

```python
class BranchingHypernetwork(AtomHypernetwork):
    def __init__(self, ..., max_branches=16):
        super().__init__(...)
        self.branch_vectors = nn.ParameterList([
            nn.Parameter(torch.randn(num_atoms) * 0.1)
            for _ in range(max_branches - 1)
        ])
    
    def forward_branched(self, state_pages, pass_num, K=4):
        base_scales = self.forward(state_pages, pass_num)
        branches = [base_scales]
        for k in range(K - 1):
            branches.append(torch.tanh(base_scales + self.branch_vectors[k]))
        return branches
```

Test-time compute scaling:

```
K=1:   standard linear thinking (baseline cost, baseline accuracy)
K=4:   explore 4 strategies (4x cost at pass 1, higher accuracy)
K=16:  explore 16 strategies (16x cost at pass 1, highest accuracy)
```

No retraining needed. The atom hypernetwork already generates scales. Branching generates MULTIPLE scale sets and uses the confidence head to pick the best.

Implementation priority: AFTER the 64-atom architecture is proven on GSM8K. This is an inference-time optimization, not an architecture change.

---

## Training

### Loss

```python
total_loss = (generation_loss                      # CoT trace (main signal)
              + 0.05 * contrastive_loss            # prevents page collapse
              + 0.3 * answer_head_loss             # trains digit extraction
              + 0.1 * confidence_loss)             # trains stopping decision
```

No entropy regularization. No blend tracking. Simpler than the quad LoRA setup.

### Optimizer

```python
optimizer = AdamW([
    {'params': perceiver_params, 'lr': 1e-4},
    {'params': atom_A_params, 'lr': 1e-3},         # 10x (gradient imbalance)
    {'params': atom_B_params, 'lr': 1e-3},         # 10x
    {'params': hypernetwork_params, 'lr': 1e-3},   # 10x
    {'params': answer_head_params, 'lr': 1e-3},
    {'params': confidence_params, 'lr': 1e-3},
    {'params': page_to_tokens_params, 'lr': 1e-4},
])
# Transformer FROZEN
```

### Warm Start

Cannot warm-start atom templates from quad LoRA (completely different structure). Start atoms from small random init (0.01). CAN warm-start perceiver from previous checkpoints. Hypernetwork re-initializes (new output dimension).

This means: re-run the curriculum from L2 or L3. The perceiver carries forward. The atoms and hypernetwork train from scratch on the stepping stones.

### Curriculum

```
L3 (named quantities):   20K fresh per epoch, early stopping patience=5
L4 (two-step word):       20K fresh per epoch
L4.5-L4.9 (stepping):    20K fresh per epoch
L5 (GSM8K):              7.5K + augmentation per epoch
```

---

## What to Monitor

```
1. Atom activation sparsity:
   How many of the 64 atoms have |scale| > 0.1?
   Easy problems: expect 5-15 active atoms
   Hard problems: expect 30-50 active atoms
   If all 64 active on easy problems: atoms not specializing

2. Atom scale variance across passes:
   Do different passes activate different atoms?
   cos(scales_pass1, scales_pass3) should be < 0.8
   If identical across passes: pass conditioning not working

3. Atom clustering after training:
   Cluster the 64 A/B template pairs by similarity
   Do natural groups emerge? (some atoms similar to each other)
   These emergent groups ARE the cognitive modes — discovered, not designed

4. Per-atom activation patterns:
   Which atoms activate on which problem types?
   Atom 17 always active on division problems → it learned "division attention"
   Atom 42 only active on 5-step problems → it learned "long-chain attention"

5. Standard metrics:
   gen_acc, head_acc, page_cos, contrastive_loss, answer_head_loss
```

---

## Post-Training Analysis: Discovering the Cognitive Modes

After training, we can reverse-engineer what the model discovered:

```python
def analyze_atoms(model, problems_by_type):
    """Discover what each atom does by analyzing activation patterns."""
    atom_profiles = defaultdict(list)
    
    for problem_type, problems in problems_by_type.items():
        for problem in problems:
            state_pages, atom_scales_per_pass = model.think(problem)
            for pass_num, scales in enumerate(atom_scales_per_pass):
                atom_profiles[(problem_type, pass_num)].append(scales)
    
    # Cluster atoms by their activation patterns
    # Atoms that co-activate on similar problem types form natural groups
    # These groups ARE the cognitive modes — but discovered, not designed
    
    # Example findings:
    # Atoms 3, 17, 42: always active on pass 1 → "parsing cluster"
    # Atoms 8, 15, 29: active on arithmetic passes → "computation cluster"  
    # Atoms 5, 22: only active on hard problems → "deep reasoning cluster"
    # Atoms 11, 38: active on final pass → "answer extraction cluster"
```

The discovered modes might map to parse/compute/verify/answer. Or they might reveal something entirely different — modes we didn't think of. That's the value of anonymous atoms: the model tells US its cognitive architecture.

---

## Why This Will Work

```
1. Symmetric capacity (105M compress ≈ 100M expand) — balanced architecture
2. No softmax → no mode collapse — the failure mode that killed quad LoRA is gone
3. Independent atoms → natural sparsity — easy=few atoms, hard=many atoms
4. 10M hypernetwork — real capacity to make per-problem, per-pass decisions
5. Tanh scaling → each atom is independently useful — no competition
6. Post-training analysis → we discover what the model learned — interpretable
7. Same training recipe (CoT + contrastive + answer head) — proven components
```
