# Handoff: Log-Answer Head (v21.1)

## The Pivot

Stop training the model to *generate* the answer. Train it to *compute* the answer directly into the last page, then read it out with a tiny linear head.

```
Think → compress → think → compress → think → compress
                                                  ↓
                                         last page (64 floats)
                                                  ↓
                          ┌──── Linear(64, 1) → log10(|ans| + 1)
                          └──── Linear(64, 1) → P(ans >= 0)  (sigmoid)
                                                  ↓
                                    sign × (10^log_mag - 1) → answer
```

## Why

Every GSM8K experiment so far fought generation-side problems:
- Arithmetic-trained LoRA collapsed to "emit a number immediately"
- Short-completion v21 collapsed to "The answer is X. The answer is X. ..."
- Generation distribution narrows under fine-tuning; reasoning capability is lost

But the **thinking chain (Llama + LoRA + perceiver + pages) works fine.** It crushed arithmetic (85.4% two-step, 73.6% three-step, 86.2% with pages). The pages contain useful computation. What fails is the readout — asking a base model to write structured reasoning text under a fine-tuning signal that only rewards the final token.

**Delete the readout problem.** Train the last page to encode the answer directly. One `Linear(64, 1)` in log space, one `Linear(64, 1)` for sign.

## Why This Is Different From the Failed Probe-Only Attempt

CLAUDE.md records: "Probe-only training capped at 2.8%." That was:
- **Raw values** — loss hit 1.26B on GSM8K from large answers
- **Per-step probe** — trying to predict intermediates, not the final answer
- **No structural help** — the probe was a diagnostic, not the primary objective

The log-head version is:
- **log10 space** — answers 1→1M become 0→6. Bounded, stable, learnable
- **Final-answer only** — direct gradient path from the 64 floats that matter
- **Primary training objective** — no generation loss to compete with
- **Sign + magnitude decoupled** — rare negatives handled cleanly

## Architecture

### LogAnswerHead (src/log_answer_head.py)

```python
class LogAnswerHead(nn.Module):
    def __init__(self, page_size=64):
        super().__init__()
        self.log_mag_head = nn.Linear(page_size, 1)
        self.sign_head    = nn.Linear(page_size, 1)

    def forward(self, last_page):
        return self.log_mag_head(last_page).squeeze(-1), \
               self.sign_head(last_page).squeeze(-1)

    def compute_loss(self, last_page, gold):
        log_mag, sign_logit = self.forward(last_page)
        mag_loss  = F.mse_loss(log_mag, torch.log10(gold.abs() + 1))
        sign_loss = F.binary_cross_entropy_with_logits(sign_logit, (gold >= 0).float())
        return mag_loss + 0.1 * sign_loss

    def decode(self, last_page):
        log_mag, sign_logit = self.forward(last_page)
        mag = 10**log_mag - 1
        sign = torch.where(torch.sigmoid(sign_logit) >= 0.5, 1.0, -1.0)
        return sign * mag
```

Total params: `2 × (64 + 1) = 130`. Trivial.

### Training Loop

```python
state_pages = run_thinking(question, num_passes=3)
last_page = state_pages[-1]  # (batch, 64)
loss = answer_head.compute_loss(last_page, gold)
loss.backward()
```

No generation forward. No completion tokenization. No teacher forcing. One backward through the 3 thinking forwards + the 130-param head.

### Eval

```python
pred = answer_head.decode(last_page).item()
pred = round(pred) if gold.is_integer() else pred
exact = (pred == gold)
```

Two metrics reported:
- **exact**: rounded equality (the headline accuracy)
- **tol1%**: `|pred - gold| / max(|gold|, 1) ≤ 0.01` — "directionally correct even when rounding fails"

If exact is low but tol1% is high, the pages are computing correctly and we just need output precision (mantissa + exponent head, or a finer decoder).

## Warm-Start

```
two_step_pages_best.pt  (86.2% two-step)
      ↓
      compressor       ✓ loaded
      hypernet         ✓ loaded (incl. LoRA templates)
      answer_head      ✗ fresh (didn't exist before)
```

## Expected Behavior

### Epoch 1
The answer head is random. Loss will start high (~2-4 in log-mag space). Exact accuracy will be near zero because small random predictions round to 0 or 1. Tol1% may also be near zero. **Do not panic.** The pages need to learn to encode the answer, and the head needs to learn to read them. Both start from scratch.

### Epoch 2-3
Loss should drop sharply. Exact accuracy will climb — hopefully past 6.2% baseline. Tol1% will climb faster than exact (precision lags correctness).

### Failure modes to watch
- **Loss plateaus high (>2):** pages aren't encoding the answer. Check page diversity, hypernet gradient magnitudes.
- **Loss low but exact ≈ 0:** head is predicting the mean. Check log_mag variance across batch.
- **Exact low but tol1% high:** pages are right, head precision insufficient. Add mantissa/exponent head.
- **Sign loss high:** should be trivial (>95% of GSM8K answers are positive). If high, something is pathologically wrong.

## What Stays the Same

- Pages architecture (append-only, per-page normalized)
- PageAttentionHypernetwork driving LoRA
- Llama 3.2 1B frozen
- 7-layer perceiver compressor
- 3 thinking passes
- Random hypersphere init during training
- Warm-start from two-step pages checkpoint

## What Changes

- No `PageToTokens` used in training (kept in codebase as fallback for MATH-500)
- No generation forward in `forward_train`
- No tokenization of completions
- Training target: `(log_mag, sign)` instead of token sequence
- Eval: decode from last page, no `model.generate()`

## Parameter Budget

```
Component                   Params      Change
────────────────────────────────────────────────
Llama 3.2 1B (frozen)       1.23B       —
7-Layer Perceiver           105M        —
PageAttentionHypernet       ~650K       —
LoRA templates              1.1M        —
LogAnswerHead               130         NEW (+130 params)
PageToTokens                ~550K       kept as fallback (not trained)
────────────────────────────────────────────────
Total trainable             ~107M       same as v21 minus page_to_tokens
```

## Known Limitations (For Later)

1. **MATH-500:** Non-numeric answers (`\frac{3}{4}`, `\sqrt{2}`) can't be log-decoded. When we pivot to MATH-500, add a mode switch: numeric problems → LogAnswerHead, non-numeric → PageToTokens generation fallback. Architecture supports both.

2. **Precision at large answers:** log-space error of 0.01 at magnitude 6 = ~2% relative error. If exact is capped while tol1% is high, upgrade to mantissa (32 floats) + exponent (32 floats) from the 64-float page, decoded as `mantissa × 10^exponent`. Not needed until we see the symptom.

3. **Interpretability:** We lose step-by-step reasoning as an artifact. The pages are still there for inspection (and the attention weights over pages show which past cycles drove each LoRA decision) but we can no longer read "what the model was thinking" in natural language. Trade-off accepted.

4. **Probe heads for intermediates (future):** If the last page learns to encode the final answer, the earlier pages should naturally encode intermediate computations. We can validate this by attaching secondary probes at each page position and checking whether they decode intermediate values (parseable from GSM8K's `<<a+b=c>>` annotations). Free interpretability without a separate training objective.

## Speed

No generation forward. Memory pressure drops dramatically — we can likely fit `batch_size=12–16` at 3 thinking passes without issue. Expected: ~15-20 min/epoch (vs ~25-40 min with generation).

## What This Is Not

This is not "the answer is already in the pages." It is "point the pages at a cleaner target and let them learn to put the answer there." The compute chain (Llama → perceiver → LoRA → pages) is unchanged; the supervision signal is cleaner. That's the whole pitch.
