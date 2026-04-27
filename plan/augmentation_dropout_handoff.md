# Handoff: Number Augmentation + Generation Target Dropout

## One-Sentence Summary

Randomize all numbers in GSM8K problems every epoch (forcing the model to compute, not memorize) AND randomly mask numbers in generation targets (forcing the model to derive answers from computation, not copy from the target text).

---

## The Problem: Memorization Not Computation

```
Epoch 1:  "Natalia sold 48 clips. #### 48"  → model sees 48, outputs 48
Epoch 5:  "Natalia sold 48 clips. #### 48"  → model memorized "48 → 48"
Epoch 10: gen_loss = 0.07 (perfect reproduction), eval = 7% (can't generalize)

The model learned to COPY text patterns, not COMPUTE.
Training accuracy goes up. Eval accuracy goes down. Classic overfitting.
```

Decomposition into bite-sized pieces makes this WORSE — each piece is short and easy to memorize. "Had X cookies. #### X" is just a copy operation. The model never needs to actually parse or compute.

---

## Part 1: Number Augmentation

### Concept

Every epoch, replace all numbers in each problem with random alternatives. Recompute all intermediates and the final answer. The problem STRUCTURE stays the same but the NUMBERS change:

```
Epoch 1: "Natalia sold 48 clips in April, half as many in May. Total?"
         Steps: [48, 24, 72]
         
Epoch 2: "Natalia sold 76 clips in April, half as many in May. Total?"
         Steps: [76, 38, 114]
         
Epoch 3: "Natalia sold 22 clips in April, half as many in May. Total?"
         Steps: [22, 11, 33]

Same structure. Different numbers. Must COMPUTE each time.
```

### Implementation

```python
import random
import re

def augment_gsm8k_numbers(problem, rng=None):
    """
    Replace all numbers in a GSM8K problem with random alternatives.
    Recompute intermediates and final answer from the operations.
    
    The problem structure (operations, relationships) stays the same.
    Only the specific numbers change.
    """
    if rng is None:
        rng = random.Random()
    
    question = problem['question']
    original_steps = problem['cycle_gen_targets']  # ["48 clips...", "48/2=24...", "48+24=72..."]
    original_targets = problem['cycle_targets']     # [48, 24, 72]
    operations = problem.get('operations', None)    # [extract, divide_by_2, add]
    
    # Find the base numbers in the problem (the ones stated in the question)
    base_numbers = extract_base_numbers(question)
    
    # Generate replacement numbers with constraints
    new_numbers = {}
    for num in base_numbers:
        # Scale within reasonable range (50% to 200% of original)
        low = max(2, num // 2)
        high = max(10, num * 2)
        new_num = rng.randint(low, high)
        
        # Apply divisibility constraints from operations
        if needs_even(num, operations):
            new_num = new_num - (new_num % 2)  # make even
        if needs_divisible_by(num, operations, 3):
            new_num = new_num - (new_num % 3)  # divisible by 3
        
        new_numbers[num] = new_num
    
    # Rewrite the question with new numbers
    new_question = replace_numbers_in_text(question, new_numbers)
    
    # Recompute all intermediates by replaying the operations
    new_targets = recompute_chain(original_targets, operations, new_numbers)
    
    # Rewrite generation targets with new numbers
    new_gen_targets = rewrite_gen_targets(original_steps, new_numbers, new_targets)
    
    return {
        'question': new_question,
        'cycle_targets': new_targets,
        'cycle_gen_targets': new_gen_targets,
        'final_answer': new_targets[-1],
        'operations': operations,  # unchanged
    }


def extract_base_numbers(question):
    """Find all numbers in the question text."""
    numbers = re.findall(r'\b(\d+)\b', question)
    return [int(n) for n in numbers]


def replace_numbers_in_text(text, number_map):
    """Replace each occurrence of old numbers with new numbers in text."""
    # Sort by length descending to avoid partial replacements
    # (replace "160" before "16")
    for old_num in sorted(number_map.keys(), key=lambda x: -len(str(x))):
        new_num = number_map[old_num]
        text = text.replace(str(old_num), str(new_num))
    return text


def recompute_chain(original_targets, operations, number_map):
    """
    Recompute the chain of intermediates with new base numbers.
    
    operations: list of (op_type, operand_source) tuples
    e.g., [('extract', 0), ('divide', 2), ('add', None)]
    """
    if operations is None:
        # Fallback: simple number replacement in targets
        return [number_map.get(t, t) for t in original_targets]
    
    new_targets = []
    for i, (op_type, operand) in enumerate(operations):
        if op_type == 'extract':
            # First cycle: extract a base number
            result = number_map.get(original_targets[i], original_targets[i])
        elif op_type == 'add':
            result = new_targets[-1] + number_map.get(operand, operand)
        elif op_type == 'subtract':
            result = new_targets[-1] - number_map.get(operand, operand)
        elif op_type == 'multiply':
            result = new_targets[-1] * number_map.get(operand, operand)
        elif op_type == 'divide':
            divisor = number_map.get(operand, operand)
            result = new_targets[-1] // divisor if divisor != 0 else 0
        elif op_type == 'half':
            result = new_targets[-1] // 2
        elif op_type == 'double':
            result = new_targets[-1] * 2
        elif op_type == 'triple':
            result = new_targets[-1] * 3
        else:
            result = original_targets[i]  # unknown op, keep original
        
        new_targets.append(result)
    
    return new_targets


def rewrite_gen_targets(original_gen_targets, number_map, new_targets):
    """Rewrite generation targets with new numbers and results."""
    new_gen_targets = []
    
    # Build complete number map (base numbers + computed results)
    full_map = dict(number_map)
    # Add original→new target mappings
    # (handled by replacing in text)
    
    for i, gen_text in enumerate(original_gen_targets):
        new_text = replace_numbers_in_text(gen_text, full_map)
        # Update the #### marker with the new target
        new_text = re.sub(
            r'####\s*[-]?\d+',
            f'#### {new_targets[i]}',
            new_text
        )
        new_gen_targets.append(new_text)
    
    return new_gen_targets
```

### Simplified Version (Start Here)

The full version with operation tracking is complex. Start with a simpler approach — just randomize the base numbers and recompute the chain using the GSM8K annotation format:

```python
def simple_augment(problem, rng):
    """
    Simple augmentation: scale all numbers by a random factor.
    The relationships between numbers are preserved.
    """
    scale = rng.uniform(0.5, 2.0)
    
    question = problem['question']
    base_numbers = extract_base_numbers(question)
    
    # Scale all numbers by the same factor (preserves ratios)
    number_map = {}
    for num in base_numbers:
        new_num = max(1, round(num * scale))
        number_map[num] = new_num
    
    new_question = replace_numbers_in_text(question, number_map)
    
    # Scale all targets by the same factor
    new_targets = [max(1, round(t * scale)) for t in problem['cycle_targets']]
    
    # Rewrite gen targets
    new_gen_targets = []
    for gen_text in problem['cycle_gen_targets']:
        new_text = replace_numbers_in_text(gen_text, number_map)
        new_gen_targets.append(new_text)
    
    # Fix #### markers
    for i in range(len(new_gen_targets)):
        new_gen_targets[i] = re.sub(
            r'####\s*[-]?\d+',
            f'#### {new_targets[i]}',
            new_gen_targets[i]
        )
    
    return {
        'question': new_question,
        'cycle_targets': new_targets,
        'cycle_gen_targets': new_gen_targets,
        'final_answer': new_targets[-1],
    }
```

The simple version scales ALL numbers by the same factor. This preserves ratios ("half as many" still works) and additive relationships scale proportionally. Not perfect for all problem types but covers 80% of cases and is trivial to implement.

### Integration with Training Loop

```python
def get_batch(dataset, batch_size, epoch, rng):
    """Each epoch, augment numbers differently."""
    epoch_rng = random.Random(epoch * 12345)  # deterministic per epoch
    
    batch = []
    for problem in random.sample(dataset, batch_size):
        # Augment with epoch-specific random numbers
        augmented = simple_augment(problem, epoch_rng)
        batch.append(augmented)
    
    return batch
```

Each epoch uses a different random seed → different numbers → different problems. The model sees the same STRUCTURES but never the same NUMBERS twice.

---

## Part 2: Generation Target Dropout

### Concept

Randomly mask numbers in the generation target so the model can't just copy from the target text. It must derive the answer from actual computation:

```
Full target:     "He gave away 63. 160 - 63 = 97 remaining. #### 97</s>"
                  The model can copy "97" from the equation

Masked target:   "He gave away 63. 160 - 63 = ___ remaining. #### 97</s>"
                  The model MUST compute 160-63 to produce "97"
                  It can't copy from the masked equation
```

### Implementation

```python
def dropout_gen_target(gen_text, target_number, dropout_prob=0.3, rng=None):
    """
    Randomly mask computed results in the generation target.
    Keep the #### marker intact (that's what we extract from).
    
    The model must compute the answer rather than copy it
    from earlier in the generation target.
    
    dropout_prob: probability of masking each intermediate result
    """
    if rng is None:
        rng = random.Random()
    
    if rng.random() > dropout_prob:
        return gen_text  # no dropout this time
    
    # Find equation results: "X op Y = Z" → mask Z
    def mask_result(match):
        a, op, b, eq, result = match.groups()
        if rng.random() < 0.5:
            # Mask the result — model must compute
            return f"{a} {op} {b} {eq} ___"
        else:
            return match.group(0)  # keep original
    
    masked = re.sub(
        r'(\d+)\s*([+\-*/])\s*(\d+)\s*(=)\s*(\d+)',
        mask_result,
        gen_text
    )
    
    # NEVER mask the #### marker — that's our extraction target
    # The #### number must always be present and correct
    
    return masked


def dropout_gen_target_v2(gen_text, target_number, dropout_prob=0.3, rng=None):
    """
    Simpler version: sometimes replace the ENTIRE natural sentence
    with just the #### marker.
    
    Full:    "He gave away 63. 160 - 63 = 97 remaining. #### 97</s>"
    Dropped: "#### 97</s>"
    
    The model must learn to produce the right number after ####
    WITHOUT the scaffold of the natural sentence.
    This is stronger dropout — the whole "hint" is removed.
    """
    if rng is None:
        rng = random.Random()
    
    if rng.random() < dropout_prob:
        # Drop everything except the #### marker
        return f"#### {target_number}</s>"
    else:
        return gen_text
```

### Two Levels of Dropout

```
Level 1 (v1): Mask equation results within the sentence
  "160 - 63 = ___ remaining. #### 97"
  The model sees the equation structure but must compute the result
  Gentle — most of the hint remains

Level 2 (v2): Drop the entire sentence, keep only ####
  "#### 97"
  The model must produce the right number with no text scaffold
  Aggressive — forces pure computation
  
Start with Level 1 at 30% dropout.
If model handles it, increase to 50% or add Level 2.
```

### Integration

```python
def prepare_training_batch(dataset, batch_size, epoch):
    """Augment numbers + dropout generation targets."""
    epoch_rng = random.Random(epoch * 12345)
    dropout_rng = random.Random(epoch * 67890)
    
    batch = []
    for problem in random.sample(dataset, batch_size):
        # Step 1: Randomize numbers (anti-memorization)
        augmented = simple_augment(problem, epoch_rng)
        
        # Step 2: Dropout generation targets (anti-copying)
        for i in range(len(augmented['cycle_gen_targets'])):
            augmented['cycle_gen_targets'][i] = dropout_gen_target(
                augmented['cycle_gen_targets'][i],
                augmented['cycle_targets'][i],
                dropout_prob=0.3,
                rng=dropout_rng,
            )
        
        batch.append(augmented)
    
    return batch
```

---

## Combined Effect

```
WITHOUT augmentation or dropout:
  Epoch 1: "Natalia sold 48 clips. 48/2=24. 48+24=72. #### 72"
  Epoch 5: same text → memorized → gen_loss=0.07 → eval=7%

WITH number augmentation:
  Epoch 1: "Natalia sold 48 clips. 48/2=24. 48+24=72. #### 72"
  Epoch 2: "Natalia sold 76 clips. 76/2=38. 76+38=114. #### 114"
  → Can't memorize numbers. Must parse and compute each time.

WITH augmentation + dropout:
  Epoch 1: "Natalia sold 48 clips. 48/2=___. 48+___=72. #### 72"
  Epoch 2: "Natalia sold 76 clips. 76/2=38. #### 114"
  → Can't memorize numbers AND can't copy from equation text.
  → Must actually compute. No shortcut.
```

---

## Expected Impact

```
BEFORE (no augmentation):
  gen_loss drops to 0.07 → model memorizes training text
  eval accuracy peaks at 13% then DECLINES (overfitting)
  
AFTER (with augmentation + dropout):
  gen_loss stays higher (~0.15-0.20) because every epoch has new numbers
  But eval accuracy should KEEP CLIMBING because the model generalizes
  The model learns COMPUTATION, not MEMORIZATION
  
  gen_loss stops being a reliable metric — it never reaches 0.01
  because the targets change every epoch. That's fine.
  eval accuracy is the real metric.
```

---

## Hyperparameters

```
Number augmentation:
  scale_range: (0.5, 2.0)     # scale all numbers by random factor
  per_epoch: True              # new numbers every epoch
  seed: epoch * 12345          # deterministic per epoch (reproducible)

Generation target dropout:
  dropout_prob: 0.3            # 30% of targets get masked
  level: "equation_result"     # mask "= Z" in "X op Y = Z"
  never_mask: "####"           # NEVER mask the extraction marker
  
Start conservative (30% dropout).
If gen_loss still drops fast (memorizing): increase to 50%.
If gen_loss is too high (can't learn): decrease to 15%.
```

---

## What to Monitor

```
1. Gen loss trajectory:
   WITHOUT augmentation: drops to 0.07 (memorization)
   WITH augmentation: should plateau at 0.15-0.20 (can't memorize, always learning)
   If it still drops to 0.07: augmentation isn't working, numbers are predictable

2. Eval accuracy trajectory:
   WITHOUT augmentation: peaks at 13% then declines (overfitting)
   WITH augmentation: should keep climbing past 13% (generalizing)
   If it still peaks and declines: deeper problem than memorization

3. Train vs eval gap:
   Large gap (train 90%, eval 10%): overfitting → increase augmentation
   Small gap (train 30%, eval 25%): generalizing → augmentation working

4. Number diversity:
   Check: does the model produce different numbers on augmented versions
   of the same problem? If it always produces "48" regardless of input,
   it's memorizing structure not computing.
```

---

## What NOT to Do

```
- Do NOT mask the #### marker.
  That's the extraction target. It must always be present and correct.
  Mask the natural sentence, never the marker.

- Do NOT augment eval data.
  Eval must be stable across epochs for comparison.
  Only augment training data.

- Do NOT use the same random seed every epoch.
  The whole point is different numbers each epoch.
  Each epoch gets a unique seed: epoch * 12345.

- Do NOT scale numbers to unreasonable ranges.
  "Natalia sold 0 clips" or "Natalia sold 9999999 clips" breaks realism.
  Keep within 50%-200% of original (scale 0.5 to 2.0).

- Do NOT dropout more than 50% of targets.
  The model still needs SOME supervision to learn the format.
  Too much dropout = no signal = random generation.
```
