# Handoff: Contrastive Page Loss — Breaking the Fixed-Point Collapse

## The Problem

The pages are constant. Diagnostic on the 86.2% two-step checkpoint:

```
Same-answer cosine similarity:  1.0000
Diff-answer cosine similarity:  0.9998
Delta:                          0.0002 (essentially zero)
28/64 dimensions have std < 0.01 (dead)
```

The entire architecture collapsed to a learned static LoRA. The perceiver outputs the same page regardless of input. The 85.4% accuracy came from one good LoRA configuration, not from per-problem thinking. The breathing loop is an illusion — the model found a shortcut.

## The Fix: Contrastive Page Loss

Don't ask the pages to predict the answer. Ask them to be DIFFERENT for different problems. Directly attack the fixed-point collapse.

```
total_loss = generation_loss + λ * contrastive_page_loss

generation_loss:    "get the right answer" (drives accuracy, proven at 85%)
contrastive_loss:   "pages for different problems must be different" (breaks fixed point)
```

The model can't satisfy both with constant pages. The contrastive loss FORCES per-problem variation. The generation loss FORCES correct answers. The only solution satisfying both: per-problem pages that contribute to correct answers. Genuine thinking.

## Implementation

```python
def contrastive_page_loss(last_pages, gold_answers):
    """
    last_pages:   (batch_size, 64) — last page for each problem in batch
    gold_answers: (batch_size,) — gold answer for each problem
    
    Push apart pages with different answers.
    Pull together pages with same answers.
    """
    # Normalize pages for cosine similarity
    normed = F.normalize(last_pages, dim=-1)  # (batch, 64)
    sim_matrix = normed @ normed.T            # (batch, batch)
    
    # Which pairs share the same answer?
    same_answer = (gold_answers.unsqueeze(0) == gold_answers.unsqueeze(1)).float()
    
    # Pull together: same-answer pages should be similar
    pos_loss = (1.0 - sim_matrix) * same_answer
    
    # Push apart: different-answer pages should be dissimilar (margin 0.2)
    neg_loss = F.relu(sim_matrix - 0.2) * (1.0 - same_answer)
    
    # Average over all pairs
    num_pos = same_answer.sum().clamp(min=1)
    num_neg = (1.0 - same_answer).sum().clamp(min=1)
    
    return pos_loss.sum() / num_pos + neg_loss.sum() / num_neg
```

### Integration Into Training Loop

```python
def train_step(model, batch_problems, batch_answers):
    all_last_pages = []
    all_gen_losses = []
    
    for problem, answer in zip(batch_problems, batch_answers):
        # Thinking passes
        state_pages, strategy = model.think(problem, max_passes=3)
        last_page = state_pages[-1]
        all_last_pages.append(last_page)
        
        # Generation loss (teacher-forced, full reasoning trace)
        pseudo_tokens = model.page_to_tokens(state_pages)
        gen_loss = model.generate_teacher_forced(pseudo_tokens, problem, answer)
        all_gen_losses.append(gen_loss)
    
    # Stack for contrastive computation
    last_pages = torch.stack(all_last_pages, dim=0)  # (batch, 64)
    answers = torch.tensor(batch_answers, device=device)
    
    # Combined loss
    generation_loss = torch.stack(all_gen_losses).mean()
    contrastive_loss = contrastive_page_loss(last_pages, answers)
    
    total_loss = generation_loss + 0.3 * contrastive_loss
    total_loss.backward()
```

### Why Contrastive Works Where Head-Based Losses Failed

```
Log-head failed:    gradient from Linear(64,1) too weak. Model predicts mean.
Digit-head failed:  gradient from Linear(64,30) too weak. Model predicts digit frequencies.
Contrastive works:  gradient is O(batch²). Every problem pushes against every other problem.
                    BATCH-LEVEL signal, not per-sample signal.
                    Directly penalizes the exact pathology (constant pages).
```

The contrastive loss doesn't need to know the answer format, the answer range, or the number of digits. It just needs to know "are these two problems the same or different?" Universal across arithmetic, word problems, and MATH.

## What Should Happen

### During Training

```
Epoch 1: contrastive loss is HIGH (pages are still near-constant from warm start)
         generation loss starts at ~85% baseline level
         pages begin to spread apart on the hypersphere
         
Epoch 2-3: contrastive loss drops (pages are differentiating)
           generation loss may temporarily increase (model can't use fixed LoRA anymore)
           the model must learn per-problem LoRA configurations
           
Epoch 4+: contrastive loss stabilizes (pages are well-separated)
          generation loss recovers and potentially exceeds baseline
          pages now carry per-problem information
          the breathing loop is genuinely thinking
```

There may be a temporary accuracy DIP as the model transitions from "one good static LoRA" to "per-problem dynamic LoRA." That's expected — the model is unlearning a shortcut. If accuracy recovers above 85%, the dynamic approach is strictly better.

### Diagnostic After Training

Re-run the page similarity diagnostic:

```
Target:
  same-answer cosine similarity:  > 0.9 (similar problems → similar pages)
  diff-answer cosine similarity:  < 0.5 (different problems → different pages)
  delta:                          > 0.4 (clear separation)
  per-dim std:                    > 0.1 across most dims (no dead dims)
```

If the pages differentiate, everything downstream works: the hypernetwork reads per-problem pages, generates per-problem LoRA, the model thinks differently per problem. The log-head or digit-head would now have actual signal to read.

## Hyperparameters

```
λ (contrastive weight):   start at 0.3, tune if needed
                          too low: pages stay constant
                          too high: pages differentiate but accuracy drops
                          (the model spends capacity on being different rather than being correct)

margin (neg_loss):        0.2 (different-answer pages should have cosine sim < 0.2)
                          tighter margin = more separation = stronger regularization
                          
batch_size:               at least 8 for meaningful contrastive signal
                          larger batch = more negative pairs = stronger gradient
                          batch_size=16 ideal (240 negative pairs per batch)
```

## What NOT to Do

```
- Do NOT remove the generation loss. Contrastive alone gives differentiation but not accuracy.
- Do NOT set λ too high. The model needs to prioritize correct answers, not just different pages.
- Do NOT apply contrastive to ALL pages. Apply to LAST page only (that's where the answer should be).
  Earlier pages should be free to share structure across problems.
- Do NOT expect immediate accuracy improvement. Expect a DIP then recovery as the model transitions.
- Do NOT panic if epoch 1 accuracy drops to 50-60%. The model is unlearning the fixed-LoRA shortcut.
```

## After Contrastive Works

Once pages carry per-problem information (verified by diagnostic):

1. **Re-test output heads.** The log-head or digit-head should now work because pages have actual signal.
2. **Climb stepping stones.** L2→L3→L4→L5 with per-problem pages.
3. **The answer-falls-out-of-last-page dream becomes real.** Pages encode the answer because they must (contrastive) and they do (generation loss trains them). A small head reads it out.

## The Larger Lesson

The architecture was always capable of per-problem thinking. It just needed a reason to do it. The generation loss alone wasn't enough — the model found a cheaper solution (static LoRA). The contrastive loss makes the cheap solution impossible. The only remaining solution is the one we designed for: genuine per-problem thinking through differentiable compression.
