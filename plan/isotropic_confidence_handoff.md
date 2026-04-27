# Handoff: Isotropic Regularizer + Confidence Entropy

## One-Sentence Summary

Add two gentle regularizers that prevent collapse at different levels: an isotropic Gaussian constraint on pages (forces all 64 dimensions active, no dead dims) and entropy regularization on the confidence head (prevents "always stop at cycle N," enables variable stopping).

---

## Part 1: Isotropic Page Regularizer

### The Problem

Pages use only 10 of 64 dimensions. 68% of variance is concentrated in 10 dims. 54 dims are effectively dead. The contrastive loss pushes pages APART but doesn't ensure all dimensions are USED. Pages can be "different" while only varying in the same 10 dims.

```
Contrastive loss says:    "be different from other problems"
                          Satisfied with 10 varying dims (cos 0.77 ✓)
                          54 dead dims are invisible to contrastive

Isotropic reg says:       "use ALL dimensions equally"
                          Every dim must have variance ≈ 1
                          Zero dead dims by construction
                          Information spread across full 64-dim space
```

### Inspiration: JEPA's LEGA

Yann LeCun's JEPA architecture faced the same problem — representation collapse where all embeddings become identical. Their LEGA solution: force the embedding distribution toward an isotropic Gaussian. Information must spread evenly across all dimensions. No dimension can go dead because the regularizer explicitly penalizes low-variance dimensions.

### Implementation

```python
class IsotropicRegularizer(nn.Module):
    """
    Forces page distribution toward isotropic Gaussian.
    Three terms:
      1. Per-dim mean → 0 (pages centered)
      2. Per-dim variance → 1 (all dims active)
      3. Cross-dim correlation → 0 (dims encode independent info)
    
    Applied to RAW perceiver output BEFORE normalization.
    The normalization projects to the hypersphere afterward,
    but the directional spread inherits from the isotropic raw output.
    """
    
    def __init__(self, target_var=1.0, corr_weight=0.1):
        super().__init__()
        self.target_var = target_var
        self.corr_weight = corr_weight
    
    def forward(self, pages_batch):
        """
        pages_batch: (batch_size, 64) — RAW perceiver output, before normalization
        returns: scalar loss
        """
        if pages_batch.size(0) < 4:
            return torch.tensor(0.0, device=pages_batch.device)
        
        # Term 1: Per-dimension mean should be near 0
        # Prevents pages from having a shared bias direction
        dim_means = pages_batch.mean(dim=0)  # (64,)
        mean_loss = (dim_means ** 2).mean()
        
        # Term 2: Per-dimension variance should be near target_var
        # Forces ALL 64 dimensions to carry meaningful variance
        # This is the anti-dead-dimension term
        dim_vars = pages_batch.var(dim=0)  # (64,)
        var_loss = ((dim_vars - self.target_var) ** 2).mean()
        
        # Term 3: Cross-dimension correlation should be near 0
        # Forces dimensions to encode INDEPENDENT information
        # Prevents redundancy (two dims encoding the same thing)
        normalized = (pages_batch - dim_means) / (dim_vars.sqrt() + 1e-8)
        batch_size = pages_batch.size(0)
        correlation = (normalized.T @ normalized) / batch_size  # (64, 64)
        identity = torch.eye(64, device=pages_batch.device)
        off_diagonal = correlation - identity
        corr_loss = (off_diagonal ** 2).mean()
        
        return mean_loss + var_loss + self.corr_weight * corr_loss
```

### Where to Apply

Apply to raw perceiver output BEFORE hypersphere normalization:

```python
def think_one_pass(self, problem_ids, notebook, cycle):
    # ... atoms, Llama forward, hidden states ...
    
    # Perceiver compresses to raw page
    raw_page = self.perceiver(hidden_states)
    
    # Isotropic reg on RAW page (before normalization)
    # Collect raw_pages across batch for the regularizer
    self.raw_pages_buffer.append(raw_page)
    
    # THEN normalize to hypersphere
    page = F.normalize(raw_page, dim=-1) * math.sqrt(64)
    
    notebook.append(page)
    return page

# In the training loop, after all cycles:
raw_pages_all = torch.cat(model.raw_pages_buffer, dim=0)
iso_loss = model.isotropic_reg(raw_pages_all)
total_loss += 0.01 * iso_loss  # gentle weight
model.raw_pages_buffer.clear()
```

### Why 0.01 Weight

```
0.1:    too strong — fights the task, pages become uniform noise
0.01:   gentle — shapes the distribution without dominating the task loss
0.001:  too weak — negligible effect, dims stay dead

0.01 is the "nudge" — it says "spread your information" without saying
"I don't care about the task." The task losses (gen + answer head) still
dominate. The isotropic reg is a gentle geometric prior.
```

### Expected Effect

```
BEFORE (no isotropic reg):
  10/64 dims active (16%)
  per_dim_std: [0.51, 0.36, 0.36, ..., 0.016, 0.019, 0.025]
  Most dims near zero variance
  page_cos across problems: 0.77-0.89 (differ in only 10 dims)

AFTER (with isotropic reg):
  64/64 dims active (100%) — by construction
  per_dim_std: [≈1.0, ≈1.0, ≈1.0, ..., ≈1.0]  (uniform variance)
  All dims carry meaningful information
  page_cos across problems: should drop (more dims available for differentiation)
```

### Interaction With Contrastive Loss

The isotropic reg and contrastive loss are complementary, not redundant:

```
Contrastive:  "different problems → different pages" (INTER-problem diversity)
Isotropic:    "all dimensions must carry info" (INTRA-page spread)

Together:     different problems produce different pages that use ALL dimensions
              Contrastive ensures diversity. Isotropic ensures coverage.
```

Keep both. Contrastive at 0.05 weight (existing). Isotropic at 0.01 weight (new).

---

## Part 2: Confidence Head Entropy Regularization

### The Problem

The confidence head must learn to stop at DIFFERENT cycles for DIFFERENT problems:

```
Easy (2-step):  stop after cycle 2
Medium (3-step): stop after cycle 3
Hard (5-step):  stop after cycle 5

But without regularization, the head might collapse to:
  "always stop at cycle 3" regardless of difficulty
  
This is the same "reward hacking" the Oro paper found:
  their exit gate funneled all probability into one loop.
```

### The Oro Paper's Lesson

The Looped Language Model (Oro) added entropy regularization to their exit gate. Without it, the model "cheated" by funneling all exit probability into a single loop (usually the last one). Entropy reg forces the exit probability to spread across steps, enabling genuine variable-depth reasoning.

Same risk for our confidence head. Same fix.

### Implementation

```python
def confidence_entropy_loss(model, notebook, final_answer):
    """
    Train the confidence head with correctness signal + entropy regularization.
    
    Correctness: confidence should predict whether stopping HERE would give the right answer.
    Entropy: confidence should USE the full range of cycle counts, not collapse to one.
    """
    exit_probs = []
    correctness_loss = 0.0
    
    for k in range(1, len(notebook) + 1):
        # Would stopping at cycle k give the right answer?
        page_k = notebook[k - 1]
        pred = model.answer_head.decode(page_k, cycle=k-1)
        is_correct = float(pred == final_answer)
        
        # Confidence prediction at cycle k
        conf = model.confidence_head(notebook[:k])  # reads first k pages
        target = torch.tensor([[is_correct]], device=page_k.device)
        correctness_loss += F.binary_cross_entropy(conf, target)
        
        exit_probs.append(conf.squeeze())
    
    correctness_loss = correctness_loss / len(notebook)
    
    # Entropy regularization on exit probabilities across cycles
    # Prevent collapse to "always stop at cycle N"
    exit_dist = torch.stack(exit_probs)  # (num_cycles,)
    exit_dist = exit_dist / (exit_dist.sum() + 1e-8)  # normalize to distribution
    entropy = -(exit_dist * torch.log(exit_dist + 1e-8)).sum()
    
    # Maximize entropy (spread exit probability across cycles)
    entropy_loss = -entropy
    
    return correctness_loss + 0.01 * entropy_loss
```

### Why 0.01 Weight

We learned from quad LoRA that entropy reg can PREVENT differentiation:

```
Quad LoRA entropy at 0.1:  blend locked at [0.25, 0.25, 0.25, 0.25]
                           Entropy reg prevented any specialization

Confidence entropy at 0.01: gently discourages collapse
                            Still allows "hard problems → late stopping"
                            Prevents "ALL problems → same stopping point"
```

Light touch. Prevent collapse. Don't prevent differentiation.

### Expected Behavior

```
WITHOUT entropy reg:
  Easy problem:  conf = [0.1, 0.9, 0.95] → stops at cycle 2 ✓
  Hard problem:  conf = [0.1, 0.9, 0.95] → stops at cycle 2 ✗ (same pattern!)
  The head learns one pattern and applies it everywhere.

WITH entropy reg:
  Easy problem:  conf = [0.3, 0.9, 0.95] → stops at cycle 2 ✓
  Hard problem:  conf = [0.1, 0.2, 0.4, 0.8, 0.95] → stops at cycle 5 ✓
  Different problems trigger stopping at different depths.
```

---

## Combined Training Loop

```python
def train_step_with_regularizers(model, problem_ids, cycle_targets, 
                                  final_answer, final_accuracy, num_cycles):
    notebook = []
    raw_pages = []
    total_loss = 0.0
    available_targets = list(cycle_targets)
    
    for cycle in range(num_cycles):
        page, raw_page = model.think_one_pass(problem_ids, notebook, cycle)
        notebook.append(page)
        raw_pages.append(raw_page)
        
        # Flexible loss with consumption (existing)
        teacher_weight = per_cycle_target_weight(final_accuracy, cycle, num_cycles)
        
        if cycle == num_cycles - 1:
            total_loss += 5.0 * answer_head_loss(page, final_answer, cycle)
        else:
            losses = [(answer_head_loss(page, t, cycle), i) 
                      for i, t in enumerate(available_targets)]
            losses.append((answer_head_loss(page, final_answer, cycle), -1))
            best_loss, best_idx = min(losses, key=lambda x: x[0].item())
            total_loss += teacher_weight * best_loss
            if best_idx >= 0:
                available_targets.pop(best_idx)
    
    # === NEW: Isotropic regularizer on raw pages ===
    raw_pages_batch = torch.stack(raw_pages, dim=0)  # (num_cycles * batch, 64)
    # Reshape to combine cycles and batch
    raw_pages_flat = raw_pages_batch.view(-1, raw_pages_batch.size(-1))
    iso_loss = model.isotropic_reg(raw_pages_flat)
    total_loss += 0.01 * iso_loss
    
    # === NEW: Confidence head with entropy reg ===
    conf_loss = confidence_entropy_loss(model, notebook, final_answer)
    total_loss += 0.1 * conf_loss  # confidence loss weight
    # (entropy reg is 0.01 INSIDE confidence_entropy_loss)
    
    # Existing regularizers
    total_loss += 0.05 * contrastive_loss(notebook)
    total_loss += 0.1 * model.get_scale_reg()
    
    # Notebook diversity (existing)
    for i in range(len(notebook)):
        for j in range(i+1, len(notebook)):
            cos = F.cosine_similarity(notebook[i], notebook[j], dim=-1)
            total_loss += 0.1 * F.relu(cos - 0.3).mean()
    
    return total_loss
```

---

## Parameter Cost

```
Isotropic regularizer:     0 params (computed from raw pages, no learnable parameters)
Confidence entropy reg:    0 params (computed from confidence outputs, no new parameters)

Total new parameters:      ZERO
Total compute cost:        negligible (matrix multiply on 64-dim pages)
```

Both regularizers are FREE — no new parameters, minimal compute. They shape the geometry of the representation space through the loss function, not through architecture changes.

---

## What to Monitor

```
1. Per-dimension variance (isotropic working?):
   BEFORE: 10/64 dims active, variance concentrated
   AFTER:  64/64 dims active, variance ≈ 1.0 per dim
   Run: diag_page_variance.py every 5 epochs

2. Page cosine across problems (diversity improving?):
   BEFORE: 0.77-0.89
   AFTER:  should drop (more dims available for differentiation)

3. Confidence distribution across cycles (entropy working?):
   Easy problems should stop early (cycle 2-3)
   Hard problems should stop late (cycle 5-8)
   If all stop at same cycle: entropy reg too weak, increase to 0.02
   If none ever stop: entropy reg too strong, decrease to 0.005

4. Dead dimensions (the key metric):
   Count dims with variance < 0.1
   BEFORE: 54 dead dims
   TARGET: 0 dead dims
```

---

## What NOT to Do

```
- Do NOT set isotropic weight above 0.05.
  Too strong → pages become uniform noise, task information destroyed.
  0.01 is a gentle nudge toward good geometry.

- Do NOT set confidence entropy weight above 0.05.
  Too strong → confidence head outputs 0.5 everywhere, never decides.
  0.01 prevents collapse without preventing decisions.
  We learned this from quad LoRA entropy reg (0.1 locked blend at uniform).

- Do NOT apply isotropic reg to NORMALIZED pages.
  Apply to RAW perceiver output, before normalization.
  Normalized pages are on the hypersphere — can't have per-dim variance = 1.
  Raw pages are unconstrained — isotropic reg shapes them freely.

- Do NOT remove contrastive loss when adding isotropic reg.
  They're complementary:
  Contrastive: inter-problem diversity (different problems → different pages)
  Isotropic: intra-page coverage (all dims active, no redundancy)
  Both needed. Both at low weight (0.05 + 0.01).
```
