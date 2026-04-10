# Handoff: Per-Page Contrastive + Anti-Copying Loss

## One-Sentence Summary

Apply supervised contrastive loss to EVERY page (not just the last) to prevent fixed-point collapse at every pass, plus a soft quadratic anti-copying penalty to prevent page duplication within a problem. Two proven failure modes closed with minimal assumptions about page geometry.

---

## The Two Proven Failure Modes

```
Failure 1 — Fixed-point collapse:
  All pages constant across all problems.
  Discovered at 85.4% checkpoint: page cosine 0.9998 across problems.
  Fix: supervised contrastive per page.

Failure 2 — Page copying:
  Page 1 constant, pages 2-3 identical within each problem.
  Discovered at 82.6% checkpoint: page 2-3 cosine 1.0000.
  Fix: soft quadratic anti-copying penalty.
```

Both are proven shortcuts the model takes when unconstrained. Both must be closed.

---

## The Loss

```python
def breathing_contrastive_loss(all_pages, gold_answers, temperature=0.1):
    """
    all_pages: list of N tensors, each (batch, 64) — one per thinking pass
    gold_answers: (batch,) — integer gold answers
    temperature: controls cluster tightness in supervised contrastive
    """
    loss = 0.0
    
    # === TERM 1: Per-page supervised contrastive ===
    # Each page independently must differentiate across problems.
    # Same answer → pull together. Different answer → push apart.
    # No target cosine — temperature controls geometry.
    for page in all_pages:
        loss += supervised_contrastive(page, gold_answers, temperature)
    
    # === TERM 2: Anti-copying (soft quadratic above 0.7) ===
    # Pages within a problem must not be identical.
    # Free below 0.7 cosine. Quadratic cost above.
    # The model discovers its own within-problem geometry.
    for i in range(len(all_pages)):
        for j in range(i + 1, len(all_pages)):
            page_i = F.normalize(all_pages[i], dim=-1)
            page_j = F.normalize(all_pages[j], dim=-1)
            within_cos = (page_i * page_j).sum(dim=-1)  # (batch,)
            loss += (F.relu(within_cos - 0.7) ** 2).mean()
    
    return loss / len(all_pages)


def supervised_contrastive(pages, gold_answers, temperature=0.1):
    """
    Standard supervised contrastive (SupCon) loss.
    Positives: same answer. Negatives: different answer.
    """
    normed = F.normalize(pages, dim=-1)
    sim = normed @ normed.T / temperature  # (batch, batch)
    
    # Same-answer mask (exclude self-pairs)
    same = (gold_answers.unsqueeze(0) == gold_answers.unsqueeze(1)).float()
    self_mask = 1.0 - torch.eye(same.size(0), device=same.device)
    same = same * self_mask
    
    # Log-softmax over each row
    # Subtract max for numerical stability
    sim_max = sim.max(dim=-1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim - sim_max) * self_mask
    log_prob = (sim - sim_max) - torch.log(exp_sim.sum(dim=-1, keepdim=True) + 1e-8)
    
    # Average log probability of positive (same-answer) pairs
    num_positives = same.sum(dim=-1).clamp(min=1)
    pos_log_prob = (log_prob * same).sum(dim=-1) / num_positives
    
    return -pos_log_prob.mean()
```

---

## Design Decisions

### Why supervised contrastive instead of target-cosine?

```
Target-cosine (previous):
  - Required choosing a target (0.4? 0.7? arbitrary)
  - Different tasks had different sweet spots
  - Target too low → overshoot → accuracy collapse
  - Target too high → undershoot → near-collapse

Supervised contrastive (new):
  - No target cosine to choose
  - Temperature is the only knob (controls tightness)
  - The geometry emerges from the data
  - Same answer → clustered. Different answer → separated. 
  - HOW clustered and HOW separated → learned, not prescribed.
```

### Why soft quadratic at 0.7 for anti-copying?

```
Cosine   Meaning              Penalty
──────────────────────────────────────
0.0      orthogonal           0 (free)
0.5      quite different      0 (free)
0.7      meaningfully similar 0 (free — this is the threshold)
0.8      very similar         0.01 (gentle)
0.9      nearly identical     0.04 (moderate)
0.95     barely different     0.0625 (strong)
1.0      copying              0.09 (maximum)

Below 0.7: model is completely free — pages can be as different as they want
Above 0.7: quadratic cost increases — model pays more for similar pages
At 1.0:    strong penalty — copying is expensive

The 0.7 threshold matches the empirical cross-problem sweet spot.
"Your passes should be at least as different as different problems are."
```

### Why not learned thresholds?

```
Risk: the model learns to output threshold=1.0 (disable anti-copying)
Complexity: extra network, extra training signal, extra debugging
The fixed threshold is ungameable, simple, and proven at 0.7.
If it hurts specific problem types, we add learned thresholds later.
```

---

## Integration Into Training

```python
total_loss = generation_loss + lambda_contrastive * breathing_contrastive_loss(
    all_pages, gold_answers, temperature=0.1
)
```

### Hyperparameters

```
lambda_contrastive: 0.05 (same as before — the loss is self-regulating)
temperature:        0.1 (standard for supervised contrastive)
anti_copy_threshold: 0.7 (soft quadratic kicks in above this)
```

### What's Different From Previous Contrastive

```
Previous (target-cosine):
  - Applied to LAST page only
  - Target cosine 0.4 → overshooting problems
  - Changed to 0.7 → better but still arbitrary
  - No within-problem constraint → page copying

New (supervised contrastive + anti-copying):
  - Applied to ALL pages (page 1, 2, 3 each independently)
  - No target cosine (temperature controls geometry)
  - Soft anti-copying prevents page duplication
  - Two principled constraints, no arbitrary targets
```

---

## Expected Behavior

### Page 1 (was constant at cos=0.995):

Now gets its own contrastive loss. Must differentiate across problems. Should start encoding problem-specific parsing information — different problems should activate different page 1 representations.

### Pages 2-3 (were identical at cos=1.000):

Anti-copying penalty forces them apart. Each must encode something the other doesn't. The model discovers what: maybe page 2 = intermediate, page 3 = final. Or page 2 = computation, page 3 = verification. We don't prescribe — we prevent copying and let the model decide.

### Overall accuracy:

May dip initially as the model unlearns three shortcuts simultaneously (constant page 1, copied pages 2-3). Should recover and exceed 82.6% as genuine three-pass thinking is more powerful than one-pass-with-copies.

---

## What to Monitor

```
1. Per-page cosine:     page_1_cos, page_2_cos, page_3_cos across problems
                        All should show differentiation (not 0.995)
                        
2. Within-problem cos:  cos(page1, page2), cos(page2, page3), cos(page1, page3)
                        Should be below 0.95 (not copying)
                        May settle anywhere in [0.3, 0.9] — that's fine
                        
3. Accuracy:            should recover above 82.6% after initial dip

4. Loss components:     contrastive, anti_copy, generation
                        contrastive should decrease (pages differentiating)
                        anti_copy should decrease (pages not copying)
                        generation should decrease (answers improving)
```

---

## What NOT to Do

```
- Do NOT add within-problem target cosine. Let the model discover the geometry.
- Do NOT remove the anti-copying term. The model WILL copy if allowed (proven).
- Do NOT set lambda > 0.1. The contrastive loss is self-regulating; strong lambda overshoots.
- Do NOT set temperature too low (<0.05). Very tight clusters are hard to learn.
- Do NOT expect immediate improvement. Expect a dip as three shortcuts unlearn simultaneously.
- Do NOT change the 0.7 anti-copying threshold without evidence it's wrong.
```

---

## After This Works

If three-step breaks 82.6% with per-page contrastive + anti-copying:

1. Run page diagnostic — verify all 3 pages are differentiated AND different from each other
2. 5-seed robust eval — confirm stability
3. Move to stepping stones (L2-L4) toward GSM8K
4. Same loss recipe carries forward — it's task-agnostic
