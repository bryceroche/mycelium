# Handoff: Stepping Stones L2-L5 with Dynamic Passes

## One-Sentence Summary

Move to word problems with the full architecture: pass-conditioned hypernetwork, target-cosine contrastive per page, confidence head for dynamic stopping (1-12 passes), frozen Llama. Let the problems demand the number of thinking passes, don't hardcode them.

---

## Where We Are

```
Task              Accuracy    Per-step    Pages         Passes Used
─────────────────────────────────────────────────────────────────────
L0 single-step    100%        100%        constant      1 (sufficient)
L1 two-step       94.8%       97.4%       differentiated  1 effective
L2 three-step     83.4%       94.1%       1 effective   1 effective
L2-L5 word probs  ???         ???         ???           ??? (NEXT)
```

Key lesson: three-step arithmetic didn't need three passes. The model solved it in one effective pass at 83.4%. Forcing multi-pass hurt accuracy. The architecture should let the PROBLEM determine how many passes are needed.

---

## Stepping Stones Curriculum

```
L2: "half of 48 plus 48"                    (word operations over numerals)
    New challenge: parse "half of" into ÷2
    Expected passes: 2-3 (parse + compute)

L3: "Jamie had 56 cookies and gave 2 away"  (named quantities in sentences)
    New challenge: extract quantities from narrative
    Expected passes: 3-4 (parse + extract + compute)

L4: 2-step word problems, small numbers      (easy GSM8K style)
    New challenge: multi-step reasoning in language
    Expected passes: 4-6 (parse + plan + compute + verify?)

L5: Full GSM8K                               (the target)
    New challenge: complex multi-step word problems
    Expected passes: 6-12

All levels: positive integer answers in [1, 200]
Data: 20K problems per level (prevents overfitting)
```

---

## Architecture (Unchanged Core + Re-enabled Confidence Head)

### Components

```
Llama 3.2 1B (frozen):           1.23B   — the thinker
7-Layer Perceiver:               ~105M   — compresses all 16 layers to 64-float pages
Pass-Conditioned Hypernetwork:   ~1.5M   — generates LoRA scales from pages + strategy + pass_num
State-Conditioned LoRA (rank 4): ~1.1M   — modifies attention per pass
Confidence Head:                 ~20K    — reads pages, decides when to stop (RE-ENABLED)
PageToTokens:                    ~550K   — converts pages to pseudo-tokens for generation

Total new: ~108M (8.8% of model)
Bottleneck: 64 floats per page, appended not overwritten
Strategy: 64 floats (ephemeral, shrunk from 512)
```

### Dynamic Passes

```python
def solve(self, problem, min_passes=1, max_passes=12, confidence_threshold=0.85):
    state_pages = []
    strategy = torch.zeros(1, 64, device=device)
    
    for pass_num in range(max_passes):
        # Generate pass-conditioned LoRA
        scales = self.hypernetwork(state_pages, strategy, pass_num)
        
        # Think: Llama forward with LoRA
        self.apply_lora(scales)
        outputs = self.llama(problem_ids, output_hidden_states=True)
        self.remove_lora()
        
        # Compress: perceiver reads all layers → page + strategy
        page, strategy = self.perceiver(outputs.hidden_states[1:], pass_num)
        page = F.normalize(page, dim=-1) * math.sqrt(64)
        state_pages.append(page)
        
        # Check confidence (after minimum passes)
        if pass_num >= min_passes - 1:
            conf = self.confidence_head(state_pages)
            if conf > confidence_threshold:
                break
    
    # Generate: pseudo-tokens from pages, LoRA OFF
    pseudo_tokens = self.page_to_tokens(state_pages)
    answer = self.llama.generate(pseudo_tokens + problem_ids)
    return answer, len(state_pages), conf
```

### Confidence Head (Re-enabled, Correctness-Trained)

```python
class PageConfidenceHead(nn.Module):
    def __init__(self, page_size=64, hidden=128):
        super().__init__()
        self.page_project = nn.Linear(page_size, hidden)
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

Training the confidence head with CORRECTNESS signal (not efficiency penalty):

```python
def train_confidence(model, problem, gold_answer, max_passes=12):
    """
    At each pass, check: would the current pages produce the right answer?
    Confidence target = 1.0 if yes, 0.0 if no.
    The head learns WHEN the thinking is sufficient.
    """
    state_pages = []
    confidence_loss = 0.0
    
    for pass_num in range(max_passes):
        # Think one pass
        page, strategy = model.think_one_pass(problem, state_pages, pass_num)
        state_pages.append(page)
        
        # Would current pages produce the right answer?
        with torch.no_grad():
            pseudo_tokens = model.page_to_tokens(state_pages)
            generated = model.generate(pseudo_tokens, problem)
            is_correct = (extract_answer(generated) == gold_answer)
        
        # Train confidence to predict correctness
        predicted_conf = model.confidence_head(state_pages)
        target = torch.tensor([[float(is_correct)]], device=device)
        confidence_loss += F.binary_cross_entropy(predicted_conf, target)
    
    return confidence_loss / max_passes
```

---

## Training Recipe

### Loss

```python
total_loss = (generation_loss 
              + 0.05 * target_cos_contrastive_loss  # per-page, target=0.7
              + 0.05 * anti_copy_loss               # soft quadratic above 0.7
              + 0.1 * confidence_loss)               # correctness-based stopping
```

### Answer Extraction

For stepping stones L2-L4 (answers in [1, 200]), use the generation path:
- LoRA OFF during generation
- Pseudo-tokens from pages prepended to prompt
- Base model generates reasoning + answer
- Extract last number from generation

### Early Stopping

Peak accuracy typically at epoch 2-4 then overfitting. Use patience-based early stopping:

```
patience = 2 epochs
Save best checkpoint by eval accuracy
Stop when accuracy doesn't improve for 2 consecutive epochs
```

### Data Per Level

```
20K training problems per level (prevents overfitting)
500 eval problems per level
Positive integer answers in [1, 200]
Procedurally generated for L2-L4, real GSM8K for L5
```

### Optimizer

```python
AdamW([
    {'params': perceiver_params, 'lr': 1e-4},
    {'params': lora_template_params, 'lr': 1e-3},       # 10x
    {'params': hypernetwork_params, 'lr': 1e-3},         # 10x
    {'params': confidence_head_params, 'lr': 1e-3},      # 10x
    {'params': page_to_tokens_params, 'lr': 1e-4},
])
# Transformer FROZEN
```

### Warm Start

From the best three-step checkpoint (83.4%). The perceiver, LoRA templates, and most of the hypernetwork carry forward. The pass embedding and combine layer may re-initialize due to shape changes — that's fine, they train fast.

---

## What to Monitor

### Per Level

```
accuracy:          does the model solve word problems?
avg_passes:        how many passes does the model use? (should increase with difficulty)
confidence_curve:  does confidence rise with more passes? (should be sigmoid-shaped)
page_cos:          differentiation across problems (should be 0.7-0.9)
p_i_v_j:           within-problem page similarity (should be < 0.9, not copying)
```

### The Key Diagnostic: Pass Count vs Problem Difficulty

```
If L2 uses 2 passes and L4 uses 5 passes → model allocates compute to difficulty ✓
If L2 uses 6 passes and L4 uses 6 passes → confidence head isn't working
If all problems use 1 pass → model is still doing one-pass thinking
If all problems use 12 passes → confidence never fires, threshold too high
```

### Where It Breaks

```
L2 fails: model can't parse "half of" → parsing is the bottleneck → maybe unfreeze Llama
L3 fails: model can't extract from narrative → language understanding gap → unfreeze Llama
L4 fails: model can't chain word-problem steps → architecture gap → more passes needed
L5 fails: full GSM8K too hard → need stepping stones between L4 and L5
```

Each failure tells us exactly what to fix.

---

## What NOT to Do

```
- Do NOT hardcode pass count. Let confidence head decide (1-12).
- Do NOT add efficiency penalty. Model should think as long as it needs.
- Do NOT force page diversity beyond the contrastive + anti-copy loss.
  If the model uses 1 pass for easy problems, that's correct behavior.
- Do NOT unfreeze Llama yet. Establish frozen baselines first.
  Unfreeze only if parsing is the identified bottleneck.
- Do NOT train more than 4 epochs per level. Early stopping prevents overfitting.
- Do NOT use fewer than 20K problems per level. Overfitting is the recurring enemy.
```

---

## Expected Progression

```
L2 ("half of 48 plus 48"):
  Baseline: ~70% (model knows "half of X" and addition)
  Target:   >80% with 2-3 passes
  If fails:  parsing bottleneck → unfreeze Llama

L3 ("Jamie had 56 cookies and gave 2 away"):
  Baseline: ~40% (narrative parsing is harder)
  Target:   >50% with 3-4 passes
  If fails:  language understanding gap → unfreeze Llama

L4 (easy 2-step word problems):
  Baseline: ~20% (multi-step reasoning in language)
  Target:   >30% with 4-6 passes
  If fails:  architecture gap → examine page content, maybe more passes

L5 (full GSM8K):
  Baseline: 6.2% (established)
  Target:   >10% with 6-12 passes
  Stretch:  >20% (paper territory)
```

---

## The Scoreboard (Updated)

```
Task              Base Model    Breathing       Passes    Variance
──────────────────────────────────────────────────────────────────
Single-step       70%           100%            1         —
Two-step          0%            94.8% ±0.0%     1 eff.    zero
Three-step        0%            83.4%           1 eff.    pending
L2 word ops       ???           ???             ???       NEXT
L3 named qty      ???           ???             ???       NEXT
L4 easy word      ???           ???             ???       NEXT
L5 GSM8K          4-5%          ???             ???       NEXT
```

---

## MATH-500 Deadline: April 22, 2026

```
Path: L2 → L3 → L4 → L5 (GSM8K) → MATH
Time: ~14 days
Each level: 1-2 days (20K data gen + 4 epochs + eval + diagnostic)
```
