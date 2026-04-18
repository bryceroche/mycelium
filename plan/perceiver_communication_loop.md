# Handoff: Close the Communication Loop + Residual Pages

## One-Sentence Summary

The perceiver is BLIND to previous pages — it compresses Llama's hidden states without knowing what's already been captured. Close the communication loop by feeding accumulated pages into the perceiver. Add residual connections so information persists across cycles instead of draining. Together these should shift eigenvalues toward 1.0 and give the loop actual memory.

---

## The Problem

The Jacobian diagnostic on GSM8K shows:

```
Spectral radius: 0.014 - 0.250 (all eigenvalues contracting)
Information retention after 5 passes: 0.25^4 ≈ 0.004 (effectively zero)
```

The loop is alive but LEAKY. Information drains out each cycle. By pass 5, the model has forgotten what pass 1 discovered. Multi-step reasoning can't work if early steps are forgotten.

Root cause: the perceiver has no access to previous pages. It re-compresses from scratch every cycle, potentially re-extracting the same information rather than building on previous understanding.

```
Current information flow:
  pages → hypernetwork → atom scales → LoRA → Llama → hidden states → perceiver → new page
                                                                        ↑
                                                              BLIND (doesn't see pages)

The perceiver doesn't know what's already been compressed.
It might re-extract "the numbers are 48 and 24" every single cycle
instead of building: "pass 1: numbers are 48,24" → "pass 2: compute 48/2=24" → "pass 3: sum is 72"
```

---

## Fix 1: Perceiver Receives Current State (Primary)

Feed the accumulated pages into the perceiver as additional context. The perceiver sees BOTH the hidden states from Llama AND the previous pages. It can make informed compression decisions — "I already know the numbers, so this pass I'll focus on the computation."

### Implementation

```python
class Perceiver(nn.Module):
    def __init__(
        self,
        d_model: int = 2048,
        d_perceiver: int = 1024,
        page_size: int = 64,
        num_queries: int = 4,
        num_layers: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Existing components (unchanged)
        self.queries = nn.Parameter(torch.randn(num_queries, d_perceiver) * 0.02)
        self.hidden_project = nn.Linear(d_model, d_perceiver)
        # ... existing perceiver layers ...
        self.project_page = nn.Linear(d_perceiver, page_size // num_queries)
        
        # NEW: Project pages into perceiver's key/value space
        self.page_to_kv = nn.Linear(page_size, d_perceiver)
        
        # NEW: Learnable gate for page influence (starts at 0.5)
        self.page_gate = nn.Parameter(torch.tensor(0.5))
        
        # Existing direct path (gradient highway)
        self.direct_pool_norm = nn.LayerNorm(d_model)
        self.direct_project = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, page_size),
        )
        self.blend = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, all_layer_hidden, state_pages=None):
        """
        all_layer_hidden: list of 16 × (batch, seq_len, d_model)
        state_pages: list of (batch, page_size) — accumulated pages from previous passes
        returns: page (batch, page_size)
        """
        # Project hidden states to perceiver dim
        hidden_tokens = self.prepare_hidden_tokens(all_layer_hidden)
        # hidden_tokens: (batch, num_tokens, d_perceiver)
        
        # NEW: If we have previous pages, add them as additional key/value tokens
        if state_pages and len(state_pages) > 0:
            pages_stacked = torch.stack(state_pages, dim=1)  # (batch, num_pages, page_size)
            pages_projected = self.page_to_kv(pages_stacked)  # (batch, num_pages, d_perceiver)
            
            # Gate the page contribution
            gate = torch.sigmoid(self.page_gate)
            pages_projected = pages_projected * gate
            
            # Append page tokens to the hidden state tokens
            kv_tokens = torch.cat([hidden_tokens, pages_projected], dim=1)
        else:
            # First pass: no previous pages, just hidden states
            kv_tokens = hidden_tokens
        
        # ===== CONTEXTUAL PATH (perceiver with augmented input) =====
        queries = self.queries.unsqueeze(0).expand(kv_tokens.size(0), -1, -1)
        
        # Cross-attention: queries attend over hidden states + pages
        for layer in self.perceiver_layers:
            queries = layer(queries, kv_tokens)
        
        # Project to page
        page_parts = [self.project_page(queries[:, q, :]) for q in range(queries.size(1))]
        page_contextual = torch.cat(page_parts, dim=-1)
        
        # ===== DIRECT PATH (gradient highway) =====
        last_layer = all_layer_hidden[-1]
        pooled = last_layer.mean(dim=1)
        pooled = self.direct_pool_norm(pooled)
        page_direct = self.direct_project(pooled)
        
        # ===== BLEND =====
        blend = torch.sigmoid(self.blend)
        page = blend * page_direct + (1 - blend) * page_contextual
        
        # Normalize
        page = F.normalize(page, dim=-1) * math.sqrt(self.page_size)
        
        return page
```

### What This Changes

```
BEFORE: perceiver cross-attends over [hidden_state_tokens]
AFTER:  perceiver cross-attends over [hidden_state_tokens, page_tokens]

The perceiver now sees:
  - What Llama's attention produced this cycle (hidden states)
  - What was compressed in ALL previous cycles (pages)

It can ask: "What's in the hidden states that ISN'T already in my pages?"
This is INFORMED compression — focus on new information, don't re-extract old.
```

### Why This Helps the Jacobian

The page_to_kv projection creates a DIRECT path from state_pages into the perceiver's computation:

```
page_k → page_to_kv → perceiver_attention(kv includes page_k) → page_{k+1}

This is a SHORT gradient path that doesn't go through the full
hypernetwork → LoRA → Llama → perceiver chain.

The Jacobian gets a strong direct component.
```

---

## Fix 2: Residual Pages (Complement)

Add a residual connection from the previous page to the new page. Information persists across cycles by default — the perceiver only needs to output the DELTA (what's new this cycle).

### Implementation

```python
class ResidualPageGate(nn.Module):
    """Per-dimension blending of new and old page."""
    def __init__(self, page_size=64):
        super().__init__()
        self.gate = nn.Linear(page_size * 2, page_size)
    
    def forward(self, new_page, old_page):
        """
        gate ≈ 1: keep new_page (overwrite this dimension)
        gate ≈ 0: keep old_page (preserve this dimension)
        """
        combined = torch.cat([new_page, old_page], dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        blended = gate * new_page + (1 - gate) * old_page
        return blended
```

The per-dimension gate lets the model decide: "dimension 5 (coarse problem type) should be preserved from pass 1. Dimension 50 (current computation step) should be overwritten." Stable information persists. Evolving information updates.

### Effect on Eigenvalues

```
Without residual:
  page_{k+1} = f(page_k)
  Jacobian J = ∂f/∂page_k
  Eigenvalues < 0.25 (contracting — information drains)

With residual (weight 0.5):
  page_{k+1} = f(page_k) + 0.5 * page_k
  Jacobian J = ∂f/∂page_k + 0.5 * I
  Eigenvalues shift by +0.5!
  
  Old eigenvalue 0.10 → new eigenvalue 0.60 (stable!)
  Old eigenvalue 0.25 → new eigenvalue 0.75 (stable!)
  
Information stops draining. The loop has memory.
```

---

## Combined: Communication Loop + Residual

```python
def think_one_pass(self, problem, state_pages, pass_num):
    # 1. Hypernetwork reads pages → atom scales
    atom_scales = self.hypernetwork(state_pages)
    
    # 2. Apply atom LoRA, run Llama
    self.apply_lora(atom_scales)
    hidden_states = self.llama(problem, output_hidden_states=True)
    self.remove_lora()
    
    # 3. Perceiver compresses hidden states + READS PREVIOUS PAGES
    new_page = self.perceiver(hidden_states, state_pages=state_pages)
    
    # 4. Residual connection (preserve information)
    if len(state_pages) > 0:
        new_page = self.residual_gate(new_page, state_pages[-1])
        new_page = F.normalize(new_page, dim=-1) * math.sqrt(64)
    
    # 5. Add pi-harmonic encoding
    new_page = self.pi_encoding.apply(new_page, pass_num)
    
    return new_page
```

---

## What to Monitor

```
1. Jacobian eigenvalues:
   BEFORE: all < 0.25 (contracting, information drains)
   TARGET: several in range 0.5-1.0 (stable, information preserved)
   The residual should shift eigenvalues by ~0.5

2. Spectral radius:
   BEFORE: 0.014-0.250 (weak recurrence)
   TARGET: 0.5-1.2 (strong recurrence)

3. Page evolution across passes (same problem):
   BEFORE: pages change but don't accumulate
   AFTER: pages accumulate (each adds to previous)
   Cosine between page_k and page_{k+1} should be 0.6-0.8
   (similar but evolving, not identical and not random)

4. Information retention:
   Can the answer head read the correct answer at each pass?
   If pass 1-2 have the numbers but pass 5 doesn't → information drains
   With fixes: pass 5 retains info from pass 1 + adds computation

5. GSM8K accuracy:
   TARGET: >22% (beat previous all-time best)
   The 17% ceiling was likely caused by information drain.

6. page_gate value:
   Low (0.1): perceiver ignoring pages (bad)
   Mid (0.3-0.7): balancing hidden states and pages (good)
   High (0.9): perceiver ignoring hidden states (bad)
```

---

## Parameter Cost

```
Fix 1 (perceiver sees pages):
  page_to_kv: Linear(64, 1024) = 65,536 params
  page_gate: 1 param
  Total: ~66K

Fix 2 (residual gate):
  gate: Linear(128, 64) = 8,192 params
  Total: ~8K

Combined: ~74K params (negligible vs 198M trainable)
```

---

## Implementation Order

```
1. Add state_pages parameter to Perceiver.forward()
2. Add page_to_kv projection and page_gate
3. Concatenate page tokens with hidden tokens in cross-attention
4. Add ResidualPageGate module
5. Integrate into thinking loop
6. Train L3 fresh (verify Jacobian eigenvalue shift)
7. If eigenvalues near 0.5-1.0 → proceed to GSM8K
8. Target: >22% with information-preserving loop
```

---

## What NOT to Do

```
- Do NOT use a large residual weight (>0.8). Too much residual makes pages
  nearly identical across passes — a different form of collapse. 0.5 is the
  starting point. The learnable gate is better.

- Do NOT feed pages into perceiver WITHOUT the gate. The perceiver might
  just copy pages. The gate prevents this.

- Do NOT remove the direct path (gradient highway) from v24.7. That fix
  is still needed. This ADDS to it, not replaces it.

- Do NOT add pages to the perceiver's QUERY side. Pages should be in
  key/value (things read FROM), not query (things asked FOR).

- Do NOT normalize pages before feeding to perceiver. Raw values carry
  magnitude information useful for attention weighting.
```
