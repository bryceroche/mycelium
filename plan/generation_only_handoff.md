# Handoff: Drop the Answer Head — Generation-Only Training

## One-Sentence Summary

Remove the answer head entirely. The generation IS the output, the training signal, and the metric. Each cycle generates a natural sentence ending with `#### {answer}</s>` — a standard marker that makes extraction trivial and scales to MATH-500.

---

## Why Drop the Answer Head

Five attempts on GSM8K. Zero success:

```
900K reading pages:              collapsed to "10"            ✗
7.7M reading pages:              4%, never improved           ✗
10.9M reading hidden states:     4%, hurt generation (56%→46%) ✗
7.7M reading hidden states:      4%, 32 epochs flat           ✗
916K reading digit logits:       declining after epoch 2       ✗

Meanwhile generation:            52-56% the whole time         ✓
```

The answer head wasn't just failing — it was actively hurting. Its wrong gradient (96% incorrect predictions) pulled atoms away from correct generation. Generation DECLINED from 56% to 46% because of answer head interference.

Correct weak signal beats strong wrong signal. The generation loss provides correct gradient. The answer head provides wrong gradient. Remove the wrong signal.

---

## The New Architecture (No Answer Head)

```
Problem text + text injection
  │
  ▼
Llama (atom-modified attention)
  │
  ├── hidden_states → perceiver → 64-float page → APPEND to notebook
  │                                                    │
  │                                                    ▼
  │                                              HYPERNETWORK
  │                                              (reads notebook → atom scales)
  │
  ├── last_layer → message_gen → 32-float message (for hypernetwork)
  │
  └── logits → GENERATION → "160 - 63 = 97 remaining. #### 97</s>"
                    │                                   ^^^^^^^^^^^^
                    │                                   standard answer marker + EOS
                    │
                    ├── gen_loss (cross-entropy, differentiable — THE training signal)
                    │
                    └── extraction (regex on ####, non-differentiable — THE metric)

NO ANSWER HEAD. No digit classification. No hidden state decoding.
One output path: generation. One extraction method: #### marker.
```

### What's Removed

```
- Answer head module (all versions — pages, hidden states, digit logits)
- Answer head loss (was fighting generation loss)
- Answer head gradient (was corrupting atom learning)
- Conditional gen gating based on answer head confidence
- Per-problem head_loss computation
- All the complexity of head_loss center tuning (5.0, 15.0, etc.)
```

### What's Simplified

```
- One training signal: generation cross-entropy
- One output path: generated text
- One extraction method: regex on #### marker
- One gating mechanism: extraction-based (non-differentiable scalar weight)
- Fewer parameters: save 1-10M depending on which head was active
- No gradient conflict between head and generation
```

---

## Generation Target Format

Each cycle's generation target ends with `#### {answer}</s>`:

```
Cycle 1: "Jamie had 160 cookies. #### 160</s>"
Cycle 2: "He gave away 63. 160 - 63 = 97 remaining. #### 97</s>"
Cycle 3: "He got 20 more. 97 + 20 = 117 total. #### 117</s>"
```

The format:
```
{natural sentence with computation}. #### {intermediate_number}</s>
                                      ^^^^                    ^^^^
                                      answer marker           end of breath
```

The model learns to produce:
1. A natural sentence (full expansion — the inhale)
2. The computation embedded in the sentence ("160 - 63 = 97")
3. The answer marker `####` followed by the extracted number
4. EOS to end the breath

### Data Preparation

```python
def format_gen_target(gen_text, intermediate_number, tokenizer):
    """
    Append #### marker and EOS to each cycle's generation target.
    
    Input:  "He gave away 63. 160 - 63 = 97 remaining."
    Output: "He gave away 63. 160 - 63 = 97 remaining. #### 97</s>"
    """
    target = f"{gen_text} #### {intermediate_number}{tokenizer.eos_token}"
    return tokenizer.encode(target)

# For GSM8K data:
for problem in gsm8k_data:
    for cycle, (gen_text, intermediate) in enumerate(
        zip(problem['cycle_gen_targets'], problem['cycle_targets'])
    ):
        problem['cycle_gen_targets'][cycle] = format_gen_target(
            gen_text, intermediate, tokenizer
        )
```

### Extraction

```python
def extract_answer(generated_text):
    """
    Extract the number after #### marker.
    Trivial, unambiguous, one regex.
    """
    match = re.search(r'####\s*([-]?\d+)', generated_text)
    if match:
        return int(match.group(1))
    return None

def extract_answer_math500(generated_text):
    """
    For MATH-500: extract LaTeX expression after ####.
    Same marker, different content type.
    """
    match = re.search(r'####\s*(.+?)$', generated_text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None
```

### Why #### Format

```
1. GSM8K already uses it: "#### 72" is the standard GSM8K answer format
   The model learns a format it may have seen in pretraining

2. Unambiguous: #### never appears in natural math text
   No confusion between "97 cookies" and the actual answer

3. Scales to MATH-500: "#### \frac{3}{4}" or "#### \sqrt{2}"
   Same marker, different answer types

4. Trivial extraction: one regex, always works
   No "grab first equation" vs "grab last number" ambiguity

5. The model learns it: #### is just tokens, generated like any other text
   The EOS weight ensures stopping right after
```

---

## Training Loop

```python
def train_step(model, problem_ids, cycle_targets, cycle_gen_targets,
               final_answer, final_accuracy, num_cycles, tokenizer):
    """
    Generation-only training. No answer head.
    
    The generation loss IS the training signal.
    The extraction IS the metric.
    The gating IS based on extraction correctness.
    """
    notebook = []
    total_loss = 0.0
    available_targets = list(cycle_targets)  # for flexible matching
    prev_results = []
    
    for cycle in range(num_cycles):
        # Forward pass
        page, gen_logits = model.think_and_generate(
            problem_ids, notebook, cycle, prev_results
        )
        notebook.append(page)
        
        # Generation loss with EOS weight (THE training signal)
        gen_target = cycle_gen_targets[cycle]  # includes #### and EOS
        gen_loss = weighted_generation_loss(
            gen_logits, gen_target, tokenizer, eos_weight=5.0
        )
        
        # Extract prediction (non-differentiable — for gating and metrics)
        with torch.no_grad():
            generated_tokens = gen_logits.argmax(dim=-1)
            generated_text = tokenizer.decode(generated_tokens[0])
            predicted_number = extract_answer(generated_text)
        
        # Flexible matching with consumption
        matched = False
        if predicted_number is not None:
            # Check against available targets
            for i, target in enumerate(available_targets):
                if predicted_number == target:
                    available_targets.pop(i)
                    matched = True
                    break
            # Check against final answer
            if not matched and predicted_number == final_answer:
                matched = True
        
        # Gate generation loss by correctness
        # Correct → full reward. Wrong → reduced (still learns language).
        smooth_fading_weight = per_cycle_target_weight(
            final_accuracy, cycle, num_cycles
        )
        
        if cycle == 0:
            # Cycle 1: gen always active (learning to parse)
            total_loss += smooth_fading_weight * 1.0 * gen_loss
        else:
            if matched:
                total_loss += smooth_fading_weight * 1.0 * gen_loss
            else:
                total_loss += smooth_fading_weight * 0.3 * gen_loss
        
        # Text injection for next cycle
        if predicted_number is not None:
            prev_results.append(predicted_number)
        else:
            prev_results.append(0)  # fallback
    
    # Final cycle must match final answer (always full weight)
    # This is enforced through the generation target which includes
    # the final answer after ####
    
    # Regularizers
    total_loss += 0.01 * isotropic_reg(raw_pages)
    total_loss += 0.1 * confidence_entropy_loss(notebook, final_answer)
    total_loss += 0.05 * contrastive_loss(notebook)
    
    return total_loss
```

---

## Gating: Reward Correct, Suppress Wrong

The gating uses extraction (non-differentiable) to scale the generation loss (differentiable):

```
Cycle generates: "160 - 63 = 97 remaining. #### 97</s>"
Extract: 97
Target: 97
Match: YES → gen_loss × 1.0 (full reward)

Cycle generates: "160 - 63 = 100 remaining. #### 100</s>"
Extract: 100
Target: 97
Match: NO → gen_loss × 0.3 (reduced reward, still learns language)
```

The 0.3 weight for wrong answers (not 0.0) is important — the model still gets gradient for language patterns, sentence structure, and EOS placement even when the number is wrong. It just gets LESS reward. The gradient says "your language is fine but the number is wrong."

```
Wrong number, weight 0.0:  model gets NO gradient → forgets language patterns
Wrong number, weight 0.3:  model gets SOME gradient → retains language, number improves
Wrong number, weight 1.0:  model gets FULL gradient → no pressure to fix number
```

0.3 is the sweet spot — enough to maintain language ability, little enough to pressure correct computation.

---

## Accuracy Measurement

```python
def evaluate(model, eval_data, tokenizer, max_cycles=3):
    """
    Evaluate by generation extraction.
    No answer head. Just generate and extract.
    """
    correct = 0
    total = 0
    
    for problem in eval_data:
        notebook = []
        prev_results = []
        
        for cycle in range(max_cycles):
            page, gen_logits = model.think_and_generate(
                problem['input_ids'], notebook, cycle, prev_results
            )
            notebook.append(page)
            
            # Generate text
            generated = tokenizer.decode(gen_logits.argmax(-1)[0])
            predicted = extract_answer(generated)
            
            if predicted is not None:
                prev_results.append(predicted)
            
            # Check confidence head for early stopping
            if model.confidence_head(notebook) > 0.85 and cycle >= 1:
                break
        
        # Final answer is last extraction
        if predicted == problem['final_answer']:
            correct += 1
        total += 1
    
    return correct / total
```

---

## Expected Impact

```
BEFORE (with answer head):
  Generation: 56% → 46% (DECLINED due to answer head interference)
  Answer head: 4% (every version failed on GSM8K)
  Final accuracy: 3-4% (measured by broken answer head)

AFTER (generation only):
  Generation: should recover to 56%+ (no conflicting gradient)
  No answer head gradient corruption
  Final accuracy: measured by extraction (should reflect true 56% on cycle 1)
  
  If cycle 2 copying resolves: final could reach 30%+
  (56% cycle 1 × 50%+ cycle 2 = 28%+ on 2-step problems)
```

---

## Component Ratios (Updated)

```
Component                    Params      Role
──────────────────────────────────────────────────────────────
Llama 3.2 1B (frozen)        1,230M      Comprehends
7-Layer Perceiver             105M       Compresses (page for hypernetwork)
64 LoRA Atoms (rank 6)         82M       Pattern library
Atom Hypernetwork             101M       The brain (selects atom blends)
Confidence Head               2.5M       Variable stopping
Message Generator             1.1M       32-float direct bypass
REMOVED: Answer Head            0M       REMOVED — generation handles it
──────────────────────────────────────────────────────────────
Total trainable:              ~292M
Frozen:                       1.23B
```

Three pillars remain: perceiver (105M), hypernetwork (101M), atoms (82M). No answer head competing for gradient. Clean architecture.

---

## Scales to MATH-500

```
GSM8K generation target:
  "He gave away 63. 160 - 63 = 97 remaining. #### 97</s>"
  Extraction: regex for integer after ####

MATH-500 generation target:
  "Dividing by 4: 3/4 of the total. #### \frac{3}{4}</s>"
  Extraction: regex for LaTeX after ####, compare with SymPy

Same architecture. Same #### format. Same EOS.
Only the extraction parser changes (integer → LaTeX → SymPy comparison).
```

---

## What NOT to Do

```
- Do NOT bring back the answer head.
  Five attempts, zero success on GSM8K. Each one hurt generation.
  The generation works. Trust it.

- Do NOT use 0.0 weight for wrong answers.
  The model needs language gradient even when numbers are wrong.
  0.3 maintains language ability while pressuring correct computation.

- Do NOT skip the #### marker.
  Without it, extraction is ambiguous ("97 cookies" vs "63 away" — which number?)
  With it, extraction is trivial and unambiguous.

- Do NOT skip the EOS token or its 5x weight.
  EOS teaches breath boundaries. Without it, the model rambles.
  The #### marker + EOS = clean, bounded, extractable output.

- Do NOT worry about non-differentiable gating.
  The GATE is non-differentiable (extraction + comparison).
  The LOSS is differentiable (cross-entropy on generation).
  The gate scales the loss. The gradient flows through the loss.
  This is standard practice (like REINFORCE-style scalar rewards).
```
