# Handoff: Gradient Coupling Skip Connections

## One-Sentence Summary

Add direct skip connections in the hypernetwork and perceiver that bypass the softmax bottlenecks, giving gradient a clean highway from page_{k+1} back to page_k. The Jacobian should jump from ~1e-6 to meaningful values, making the thinking loop genuinely recurrent.

---

## Why

The current gradient path from page_{k+1} to page_k goes through TWO softmax bottlenecks — one in the hypernetwork's page attention, one in the perceiver's cross-attention. Each softmax attenuates gradient by orders of magnitude. Two in series gives ~1e-6, which is what the Jacobian diagnostic shows.

The result: the loop appears functional (pages differ across problems from contrastive loss) but isn't genuinely recurrent (page_k doesn't influence page_{k+1} through gradient flow). The contrastive loss does static problem→page mapping, not dynamic recurrence.

Skip connections give gradient a direct route that bypasses the softmaxes. The attention paths are preserved for contextual richness, but the gradient can flow cleanly through the skip paths.

---

## Current Gradient Path (Attenuated)

```
page_k 
  → Q/K/V projections
  → attention_weights = softmax(Q @ K.T / sqrt(d))    [gradient DIES HERE]
  → weighted sum of V
  → page_summary (1024-dim)
  → MLP with dropout
  → raw_logits
  → tanh(logits)
  → atom_scales
  → LoRA × scales
  → modified Llama attention
  → hidden_states
  → perceiver cross-attention (softmax again)          [gradient DIES HERE]
  → page_{k+1}

Two softmaxes in series = gradient ≈ 1e-6
```

## New Gradient Path (Direct Highway)

```
page_k 
  ├─ attention path (unchanged, preserves context)    [gradient ≈ 1e-6]
  └─ DIRECT PATH: linear → GELU → linear              [gradient ≈ 1.0]
  
  blended atom_scales
  → LoRA × scales
  → modified Llama attention
  → hidden_states
  ├─ perceiver attention (unchanged, preserves context) [gradient ≈ 1e-6]
  └─ DIRECT PATH: mean-pool last layer → linear        [gradient ≈ 1.0]
  
  blended page_{k+1}

Direct path gradient ≈ 1e-2 to 1e-1 (clean flow)
```

---

## Implementation

### 1. AtomHypernetwork with Direct Path

```python
class AtomHypernetwork(nn.Module):
    def __init__(
        self,
        page_size: int = 64,
        num_atoms: int = 64,
        hidden_dim: int = 1024,
        attention_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.page_size = page_size
        self.num_atoms = num_atoms
        
        # ===== DIRECT PATH (new — clean gradient highway) =====
        # Reads last page only, direct linear transformation
        self.direct_path = nn.Sequential(
            nn.Linear(page_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_atoms),
        )
        
        # ===== CONTEXTUAL PATH (existing — attention over all pages) =====
        self.page_project = nn.Linear(page_size, hidden_dim)
        
        self.page_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        self.context_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_atoms),
        )
        
        # ===== BLEND (learnable mixing of direct + contextual) =====
        # Initialize to 0.5 (balanced). Model can shift emphasis during training.
        self.blend = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, state_pages):
        """
        state_pages: list of (batch, page_size) tensors, one per pass so far
        returns: atom_scales (batch, num_atoms)
        """
        # Stack pages: (batch, num_pages, page_size)
        pages_stacked = torch.stack(state_pages, dim=1)
        last_page = pages_stacked[:, -1, :]  # (batch, page_size)
        
        # ===== DIRECT PATH =====
        # Clean gradient from last_page directly to logits
        direct_logits = self.direct_path(last_page)  # (batch, num_atoms)
        
        # ===== CONTEXTUAL PATH =====
        # Project pages to hidden dim
        pages_proj = self.page_project(pages_stacked)  # (batch, num_pages, hidden_dim)
        
        # Self-attention over pages (hypernetwork reads the notebook)
        attn_out, _ = self.page_attention(
            query=pages_proj,
            key=pages_proj,
            value=pages_proj,
        )
        attn_out = self.attention_norm(attn_out + pages_proj)
        
        # Summarize by taking the last position (most recent after attention)
        page_summary = attn_out[:, -1, :]  # (batch, hidden_dim)
        
        # Generate contextual logits
        context_logits = self.context_mlp(page_summary)  # (batch, num_atoms)
        
        # ===== BLEND =====
        blend = torch.sigmoid(self.blend)  # scalar in (0, 1)
        logits = blend * direct_logits + (1 - blend) * context_logits
        
        # Tanh for bounded atom scales
        atom_scales = torch.tanh(logits)
        
        # Expose raw logits for regularization (prevents tanh saturation)
        self.last_raw_logits = logits
        
        return atom_scales
```

### 2. Perceiver with Direct Path

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
        
        # ===== CONTEXTUAL PATH (existing perceiver) =====
        self.queries = nn.Parameter(torch.randn(num_queries, d_perceiver) * 0.02)
        # ... existing perceiver layers ...
        self.project_page = nn.Linear(d_perceiver, page_size // num_queries)
        
        # ===== DIRECT PATH (new — clean gradient from hidden states to page) =====
        # Mean-pool the last layer's hidden states, project to page_size
        self.direct_pool_norm = nn.LayerNorm(d_model)
        self.direct_project = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, page_size),
        )
        
        # ===== BLEND =====
        self.blend = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, all_layer_hidden):
        """
        all_layer_hidden: list of 16 tensors, each (batch, seq_len, d_model)
        returns: page (batch, page_size)
        """
        # ===== CONTEXTUAL PATH =====
        # Concatenate all layers, run through perceiver attention
        # ... existing perceiver forward pass produces page_contextual ...
        page_contextual = self.existing_perceiver_forward(all_layer_hidden)
        # page_contextual: (batch, page_size)
        
        # ===== DIRECT PATH =====
        # Mean-pool the last layer, project to page
        last_layer = all_layer_hidden[-1]  # (batch, seq_len, d_model)
        pooled = last_layer.mean(dim=1)     # (batch, d_model)
        pooled = self.direct_pool_norm(pooled)
        page_direct = self.direct_project(pooled)  # (batch, page_size)
        
        # ===== BLEND =====
        blend = torch.sigmoid(self.blend)
        page = blend * page_direct + (1 - blend) * page_contextual
        
        # Normalize to hypersphere (existing behavior)
        page = F.normalize(page, dim=-1) * math.sqrt(self.page_size)
        
        return page
```

---

## Why This Works

**The softmax isn't broken — it's just gradient-lossy.** Softmax is great for attention (concentrates focus) but terrible for gradient flow (most paths multiplied by near-zero attention weights). By ADDING a direct path that doesn't go through softmax, we preserve the attention mechanism's benefits while giving gradient a clean route.

**The blend is learnable.** The model decides how much to rely on each path. Early in training, it might favor the direct path (stronger gradient, easier to learn). Late in training, it might favor the contextual path (richer information from all pages). We don't prescribe — the training objective decides.

**Skip connections are well-understood.** ResNets, transformers, and virtually every successful deep architecture uses skip connections specifically because they solve the gradient flow problem. We're applying the same principle at a higher level — across the thinking loop, not just within a single forward pass.

---

## What to Monitor

```
1. Jacobian magnitude:
   Before: ~1e-6 (essentially zero, loop is dead)
   Target: >1e-3 (meaningful gradient flow)
   Ideal: ~1e-2 to 1e-1 (strong recurrence)

2. Blend parameters:
   Track self.blend.sigmoid() in both modules over training.
   Early: probably ~0.5 (exploring)
   Late: might shift toward 0.3 (more context) or 0.7 (more direct)
   Either is fine — the model is choosing.

3. Accuracy on L5 (GSM8K):
   Previous best: 22%
   Target: beat that with genuine recurrence
   If accuracy climbs AND Jacobian is non-zero: multi-pass reasoning is real

4. Head accuracy:
   Should remain high (already at 94% on L3).
   The answer head validates that pages encode problem-specific info.

5. Page cosine similarity:
   Should stay in 0.4-0.7 range (differentiated but not orthogonal).
```

---

## Connection to Previous Fixes

```
Fix 1 (logit regularization):   prevents tanh saturation, preserves gradient through tanh
Fix 2 (skip_pass_embed):         removes pass_num shortcut, forces hypernetwork to read pages
Fix 3 (skip connections):        bypasses softmax, gives gradient a direct highway

All three are needed:
  - Without fix 1: tanh saturates, gradient dies at activation
  - Without fix 2: hypernetwork ignores pages, no functional coupling
  - Without fix 3: softmax attenuates, gradient dies at attention

Together: clean gradient from page_{k+1} back to page_k.
The loop becomes genuinely recurrent.
```

---

## Parameter Cost

```
Direct path in hypernetwork:
  Linear(64, 256):     16,384 params
  Linear(256, 64):     16,384 params
  LayerNorm overhead:  small
  Blend:               1 param
  Total: ~33K params

Direct path in perceiver:
  LayerNorm(2048):     4,096 params
  Linear(2048, 512):   1,048,576 params
  Linear(512, 64):     32,768 params
  Blend:               1 param
  Total: ~1.1M params

Total new parameters: ~1.1M (negligible vs 198M trainable total)
```

---

## Implementation Order

```
1. Modify AtomHypernetwork:
   - Add direct_path module
   - Add blend parameter
   - Modify forward to compute both paths and blend
   
2. Modify Perceiver:
   - Add direct_pool_norm and direct_project modules
   - Add blend parameter
   - Modify forward to compute both paths and blend
   
3. Restart L3 training:
   - Fresh random init on new modules (direct paths, blends)
   - Existing modules initialized from previous checkpoint
   - OR start fully fresh if easier
   
4. Run Jacobian diagnostic after epoch 1:
   - If Jacobian > 1e-3: SUCCESS, proceed to L4/L5
   - If Jacobian still < 1e-4: investigate blend values (maybe training prefers context)
   
5. Monitor blend parameters:
   - Log blend.sigmoid() every epoch
   - Understand which path the model prefers at each training stage
```

---

## What NOT to Do

```
- Do NOT force blend = 1.0 (direct only). We want BOTH paths available.
  The contextual path has valuable information we shouldn't lose.

- Do NOT skip the blend parameter. Hard-coding 0.5 prevents the model
  from shifting emphasis during training.

- Do NOT add direct paths inside the existing attention modules.
  The skip connection must BYPASS attention, not augment it.

- Do NOT remove the logit regularization or skip_pass_embed settings.
  All three fixes are needed together — they address different problems.

- Do NOT train from warm-start if the checkpoint has different architecture.
  Adding skip connections means the model structure changed. Fresh init
  for the new modules, or fully fresh training from random init.
```
