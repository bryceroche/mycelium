# Handoff: Möbius Transformation on the Hypersphere

## One-Sentence Summary

Apply a learned Möbius (conformal) transformation to each cycle's page, warping the hypersphere so each cycle's page lands in a DIFFERENT region — directly constructing page diversity instead of indirectly pressuring it through loss regularization.

---

## The Problem: Pages Cluster on the Sphere

Pages live on a 64-dimensional hypersphere (radius sqrt(64)). Despite contrastive loss and isotropic regularization, pages from different cycles cluster in the same neighborhood:

```
page_cos(cycle_1, cycle_2) = 0.75   (should be < 0.5)
page_cos(cycle_2, cycle_3) = 0.80   (should be < 0.5)

The perceiver outputs similar directions regardless of cycle.
The hypernetwork reads similar pages → generates similar atom scales.
Cycles don't differentiate → cycle 2 copies cycle 1.
```

Previous approaches pushed pages apart INDIRECTLY through the loss:
```
Contrastive loss:    "pages SHOULD be different" (loss penalty)
Isotropic reg:       "all dims SHOULD be active" (loss penalty)
Result:              page_cos improved 0.90 → 0.75, but not enough
```

The Möbius transformation constructs diversity DIRECTLY:
```
Möbius transform:    "pages ARE different" (geometric construction)
Each cycle warps the sphere differently → pages land in different regions
No loss needed — the geometry guarantees diversity.
```

---

## What Is a Möbius Transformation?

A Möbius transformation on the sphere is like a magnifying glass on a globe. It EXPANDS one region (you see more detail) while COMPRESSING the rest (you see less). The sphere is preserved — nothing is added or removed — but the distribution of emphasis shifts.

```
Without Möbius (uniform sphere):
  [........XXXX........]   ← pages clustered in one region
  
Cycle 1 Möbius (focus on region A):
  [..XXXXXXXX..........]   ← expanded around A, compressed elsewhere

Cycle 2 Möbius (focus on region B):
  [..........XXXXXXXX..]   ← expanded around B, compressed elsewhere

Pages land in DIFFERENT regions → cos(p1, p2) drops → diversity!
```

Key mathematical properties:
- **Conformal**: preserves angles LOCALLY (nearby distinctions preserved)
- **Bijective**: every point on the sphere maps to exactly one other point
- **Stays on the sphere**: no need to re-normalize after transformation
- **Parameterized by a focus point**: one vector controls the entire warp

---

## The Breathing Connection

The Möbius transformation IS the breathing metaphor made mathematical:

```
INHALE (expand):    Möbius magnifies the focus region
                    The model sees DETAIL where it's looking
                    "48 clips sold in April" — the specific number, the operation

EXHALE (compress):  Everything outside the focus compresses
                    Irrelevant context becomes summary
                    "Natalia... May..." — background, compressed

Each cycle focuses on something DIFFERENT:
  Cycle 1 focus: quantity extraction region → sees "48" in detail
  Cycle 2 focus: computation region → sees "48/2=24" in detail
  Cycle 3 focus: synthesis region → sees "48+24=72" in detail
```

The focus point IS what this cycle cares about. Different focus → different page → different cycle behavior. The Möbius transformation formalizes "each breath focuses on something different."

---

## Architecture

### The Möbius Module

```python
class MobiusTransform(nn.Module):
    """
    Möbius transformation on the unit sphere in R^n.
    
    Warps the sphere by expanding around a focus point and
    compressing away from it. Conformal (angle-preserving locally),
    bijective, and stays on the sphere.
    
    The focus point comes from the hypernetwork — each cycle
    gets a different focus, creating page diversity by construction.
    
    Math: For a point x on the unit sphere and focus a (inside the ball),
    the Möbius transformation is:
    
      T_a(x) = (1 - |a|²)(x - a) / |x - a|² + a
    
    This maps the sphere to itself, expanding around a.
    """
    
    def __init__(self, dim=64, max_focus_norm=0.7):
        super().__init__()
        self.dim = dim
        self.max_focus_norm = max_focus_norm
    
    def forward(self, page, focus):
        """
        page:  (batch, 64) — on the hypersphere (normalized)
        focus: (batch, 64) — from hypernetwork (will be constrained to ball)
        
        Returns: warped page on the hypersphere
        """
        # Constrain focus to be INSIDE the unit ball (|focus| < max_focus_norm)
        # The closer to the boundary, the stronger the warp
        focus_norm = focus.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        focus = focus * (self.max_focus_norm * torch.tanh(focus_norm) / focus_norm)
        
        # Normalize page to unit sphere for the transformation
        x = page / page.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        
        # Möbius transformation
        a = focus
        a_norm_sq = (a * a).sum(dim=-1, keepdim=True)  # |a|²
        diff = x - a                                      # x - a
        diff_norm_sq = (diff * diff).sum(dim=-1, keepdim=True).clamp(min=1e-8)  # |x-a|²
        
        transformed = (1.0 - a_norm_sq) * diff / diff_norm_sq + a
        
        # Re-normalize to unit sphere (should be close already, this is safety)
        transformed = F.normalize(transformed, dim=-1)
        
        # Scale back to our radius (sqrt(64) = 8.0)
        transformed = transformed * math.sqrt(self.dim)
        
        return transformed
    
    def inverse(self, warped_page, focus):
        """
        Inverse Möbius: unwarp a page back to canonical space.
        T_a^{-1} = T_{-a}
        """
        return self.forward(warped_page, -focus)
```

### Hypernetwork Outputs Focus + Scales

The hypernetwork reads the notebook and outputs TWO things per cycle:

```python
class AtomHypernetwork(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Existing: generates atom scales
        self.scale_head = nn.Linear(hidden_dim, 64)  # → 64 atom scales
        
        # NEW: generates Möbius focus point
        self.focus_head = nn.Linear(hidden_dim, 64)  # → 64-dim focus vector
    
    def forward(self, notebook, messages):
        # Cross-attention over notebook (existing)
        hidden = self.attend_over_pages(notebook, messages)
        
        # Atom scales (existing) — HOW Llama reads the problem
        scales = torch.tanh(self.scale_head(hidden))
        scales = torch.clamp(scales, -3.0, 3.0)
        
        # Möbius focus (NEW) — WHERE on the sphere this cycle's page lands
        focus = self.focus_head(hidden)  # unconstrained, Möbius module constrains it
        
        return scales, focus
```

Two outputs from one brain:
```
scales: steer WHAT Llama pays attention to (reading)
focus:  steer WHERE the page lands on the sphere (recording)

Atoms control the INHALE (what to read)
Focus controls the EXHALE (how to record)
```

### Integration in the Breathing Loop

```python
def think_one_pass(self, problem_ids, notebook, focus_points, cycle, prev_results):
    """One breathing cycle with Möbius-warped pages."""
    
    # 1. Hypernetwork reads notebook → atom scales + focus point
    scales, focus = self.hypernetwork(notebook, messages)
    
    # 2. Apply atoms to Llama, forward pass
    self.apply_lora(scales)
    outputs = self.llama(full_input, output_hidden_states=True)
    self.remove_lora()
    
    # 3. Perceiver compresses → raw page
    raw_page = self.perceiver(outputs.hidden_states)
    
    # 4. Normalize to sphere
    page = F.normalize(raw_page, dim=-1) * math.sqrt(64)
    
    # 5. NEW: Möbius warp — push page to this cycle's focus region
    warped_page = self.mobius(page, focus)
    
    # 6. Append WARPED page to notebook
    notebook.append(warped_page)
    focus_points.append(focus)
    
    # 7. Message (unchanged)
    message = self.message_generator(outputs.hidden_states[-1])
    
    # 8. Generation (unchanged — reads hidden states, not page)
    gen_logits = self.generate(outputs, problem_ids)
    
    return warped_page, focus, gen_logits
```

### The Hypernetwork Reads Warped Pages

The hypernetwork reads warped pages from the notebook. It doesn't need to explicitly unwarp — the cross-attention learns to read pages in different sphere regions. The focus points are implicitly encoded in where the pages land:

```
Notebook after 3 cycles:
  [warped_page_1 (region A), warped_page_2 (region B), warped_page_3 (region C)]

Hypernetwork cross-attention:
  Query: "what should cycle 4 do?"
  Keys: pages in regions A, B, C
  
  The attention NATURALLY differentiates because keys are in different regions.
  Page in region A gets different attention weight than page in region B.
  The hypernetwork doesn't need to know about Möbius — it just sees diverse keys.
```

Optionally, store focus points alongside pages for richer reading:

```python
# Option: concatenate focus with page for hypernetwork input
notebook_with_focus = [
    torch.cat([page, focus], dim=-1)  # (batch, 128) per entry
    for page, focus in zip(notebook, focus_points)
]
```

---

## Why This Fixes Our Problems

### Problem 1: Pages too similar (cos 0.75)

```
WITHOUT Möbius:
  perceiver → similar raw pages → normalize → similar pages (cos 0.75)
  
WITH Möbius:
  perceiver → similar raw pages → normalize → Möbius with DIFFERENT focus per cycle
  → pages in DIFFERENT sphere regions → cos drops (target: < 0.3)
  
The Möbius BREAKS the similarity at the last step.
Even if the perceiver produces similar raw pages,
different focus points push them to different regions.
```

### Problem 2: Cycle 2 copies cycle 1

```
WITHOUT Möbius:
  similar pages → hypernetwork reads similar inputs → similar atom scales
  → same attention → same generation → copying
  
WITH Möbius:
  different page regions → hypernetwork reads DISTINCT inputs
  → different atom scales → different attention → different generation
  
The diversity in pages PROPAGATES to diversity in atom blends.
The Möbius provides the initial symmetry-breaking kick.
```

### Problem 3: Hypernetwork can't differentiate cycles

```
WITHOUT Möbius:
  notebook = [page_A, page_A', page_A'']  (all in region A, cos ~0.75)
  hypernetwork cross-attention: all keys similar → uniform attention → same output
  
WITH Möbius:
  notebook = [page_A, page_B, page_C]  (different regions)
  hypernetwork cross-attention: keys are distinct → differentiated attention
  → different output per cycle
  
The hypernetwork NEEDS diverse inputs to produce diverse outputs.
The Möbius PROVIDES diverse inputs.
```

---

## Möbius Focus Strength

The max_focus_norm parameter controls how strong the warp is:

```
max_focus_norm = 0.0:  no warp (identity transform, current behavior)
max_focus_norm = 0.3:  gentle warp (pages slightly separated)
max_focus_norm = 0.5:  moderate warp (pages in different neighborhoods)
max_focus_norm = 0.7:  strong warp (pages in clearly different regions)
max_focus_norm = 0.9:  extreme warp (nearly all emphasis on one point)
```

Start at 0.5 — moderate warp. If page_cos doesn't drop below 0.5, increase to 0.7. If training is unstable, decrease to 0.3.

The focus norm is LEARNED per cycle (from the hypernetwork's focus_head). The model discovers how much warping each cycle needs. Early cycles might use light warp (broad overview), later cycles might use strong warp (focused computation). The strength adapts to the task.

---

## Parameter Cost

```
New parameters:
  focus_head:        Linear(hidden_dim, 64) ≈ 65K params
  MobiusTransform:   0 params (pure math, no learnable parameters)
  ──────────────────
  Total new:         ~65K

This is NEGLIGIBLE compared to the 101M hypernetwork.
The Möbius transformation is essentially free.
```

---

## Interaction with Existing Components

### Isotropic Regularizer
Applied to raw pages BEFORE Möbius warp. The isotropic reg shapes the perceiver's output distribution. The Möbius then warps it per cycle. Both work on different stages:

```
perceiver → raw_page → isotropic_reg shapes this → normalize → Möbius warps this → notebook
```

### Contrastive Loss
Applied to WARPED pages in the notebook. With Möbius, pages should be naturally diverse. The contrastive loss becomes less necessary but can remain as a safety net at low weight (0.01 instead of 0.05).

### Generation Path
UNAFFECTED. Generation reads Llama's hidden states, not the page. The Möbius transformation only changes what goes into the notebook (for the hypernetwork). The generation accuracy (56%) should not change.

### Text Injection
UNAFFECTED. Text injection uses extraction from generation, not from pages. The Möbius transformation is invisible to the text injection path.

### Smooth Fading
UNAFFECTED. The teacher target weights fade based on accuracy. The Möbius transformation doesn't change the loss structure — it changes the PAGE structure.

---

## What to Monitor

```
1. page_cos between cycles (THE key metric):
   BEFORE Möbius: 0.75 (too similar)
   TARGET: < 0.3 (genuinely different regions)
   If still > 0.5: increase max_focus_norm

2. Focus point diversity:
   cos(focus_1, focus_2) should be low (different cycles focus differently)
   If focuses are similar: hypernetwork isn't differentiating
   Might need focus diversity loss (same as page contrastive)

3. Atom scale diversity (xpass_cos):
   BEFORE: ~0.44 (moderately different)
   TARGET: < 0.3 (clearly different per cycle)
   Diverse pages → diverse scales should follow automatically

4. Generation accuracy:
   Should be UNCHANGED (generation doesn't read pages)
   If it changes: something is wrong with gradient flow

5. Training stability:
   Möbius can cause gradient spikes if focus is near the boundary (|a| → 1)
   The tanh constraint prevents this, but monitor gradient norms
   If unstable: decrease max_focus_norm
```

---

## What NOT to Do

```
- Do NOT apply Möbius to generation logits or hidden states.
  Only apply to PAGES. Pages serve the hypernetwork.
  Generation reads hidden states directly (unaffected).

- Do NOT let focus_norm reach 1.0.
  At |a| = 1 the Möbius transformation degenerates (maps everything to one point).
  The tanh constraint with max_focus_norm = 0.7 prevents this.

- Do NOT remove isotropic regularizer when adding Möbius.
  They work on different stages:
  Isotropic shapes RAW perceiver output (per-dim variance)
  Möbius warps NORMALIZED page (per-cycle focus region)
  Both still useful. Reduce contrastive loss weight instead (0.05 → 0.01).

- Do NOT unwarp pages before the hypernetwork reads them.
  The hypernetwork SHOULD read warped pages.
  The diversity in warped pages is what drives diverse atom scales.
  Unwarping would undo the benefit.

- Do NOT add Möbius to the first cycle.
  Cycle 1 has no previous pages — no focus to differentiate from.
  Apply Möbius starting from cycle 2:
    Cycle 1: raw page → notebook (no warp, baseline)
    Cycle 2+: raw page → Möbius(page, focus) → notebook (warped)
  This gives cycle 1 a "canonical" position and all subsequent cycles
  warp relative to it.
```

---

## The Elegant Picture

```
BEFORE (indirect diversity through loss):
  perceiver → page → loss says "be different" → slow, indirect pressure
  page_cos improves: 0.90 → 0.75 (not enough)

AFTER (direct diversity through geometry):
  perceiver → page → Möbius CONSTRUCTS different position per cycle
  page_cos: should drop to < 0.3 (different by construction)

The Möbius transformation doesn't ASK the model to be diverse.
It MAKES the model diverse through geometric construction.
Then the hypernetwork naturally produces different atom blends
because its inputs are genuinely different.
The cycles differentiate. The copying stops. The model breathes.

Each breath focuses on something different.
The focus IS the Möbius transformation.
The breathing IS conformal geometry.
```
