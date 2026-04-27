# Handoff: Two-Pass Cycle 1

## One-Sentence Summary

Run Llama once without LoRA (vanilla comprehension), let the controller read those hidden states to produce informed initial scales, then run Llama again WITH those scales for the actual generation. The same mechanism that makes cycle 2 work, applied to cycle 1.

---

## The Problem: Cycle 1 Is Blind

The debug revealed the root cause of cycle 1's failure:

```
Cycle 1: prev_scales = None → no LoRA → vanilla Llama base
         Generates: "A) $100 B) $120 C) $140 D) $160 E) $180"
         Multiple choice format. No #### markers. No computation.
         
Cycle 2: controller reads cycle 1 hidden states → produces scales → LoRA active
         Generates: "40+20=60 #### 60" ← CORRECT!
         Perfect GSM8K format. Correct math.

Cycle 2 WORKS because the controller reads hidden states before choosing scales.
Cycle 1 FAILS because the controller has nothing to read — no previous hidden states.
```

The controller needs to observe Llama's comprehension BEFORE deciding which atoms to use. At cycle 2+, it reads previous hidden states. At cycle 1, there are no previous hidden states. The fix: give the controller hidden states at cycle 1 by running Llama once first.

---

## The Fix: Two Passes at Cycle 1

```
CYCLE 1 (two passes):
  Pass 1: Llama reads problem WITHOUT LoRA (vanilla comprehension)
          → hidden states capture Llama's natural understanding
          → controller reads these → produces INFORMED initial scales
          
  Pass 2: Llama reads problem WITH LoRA (atom-modified attention)
          → generation produces correct GSM8K format
          → controller reads THESE hidden states for cycle 2 scales

CYCLE 2+ (one pass each, same as before):
  Llama reads with LoRA → generation → controller → next scales
```

```
BEFORE (cycle 1 blind):
  [nothing] → controller → zero scales → Llama → garbage
  
AFTER (cycle 1 informed):
  Llama (vanilla) → controller → informed scales → Llama (with LoRA) → correct!
  
The controller sees comprehension BEFORE deciding.
Same mechanism as cycle 2. Applied to cycle 1.
```

---

## Implementation

```python
def solve(model, problem_ids, text_injection, tokenizer, max_cycles=8):
    """
    Breathing loop with two-pass cycle 1.
    
    Cycle 1 gets two Llama forward passes:
      Pass 1: vanilla (get hidden states for controller)
      Pass 2: with LoRA (actual generation)
    
    Cycles 2+ get one pass each (same as before).
    """
    notebook_pages = []
    notebook_hiddens = []
    prev_results = []
    
    # ========================================
    # CYCLE 1: TWO-PASS (informed initial scales)
    # ========================================
    
    # Pass 1: Vanilla Llama reads the problem
    # No LoRA — just Llama's natural comprehension
    input_ids = tokenize(problem_text)
    outputs_vanilla = model.llama(
        input_ids, 
        output_hidden_states=True
    )
    hidden_pool_vanilla = outputs_vanilla.hidden_states[-1].mean(dim=1)
    
    # Controller observes vanilla comprehension → initial scales
    # The controller sees HOW Llama naturally reads this problem
    # "This is about selling clips" vs "This is about buying toys"
    # → different hidden states → different scales
    page_0, initial_scales, focus_0 = model.controller(
        hidden_states_all_layers=outputs_vanilla.hidden_states,
        history_pages=[],       # no history yet
        history_hiddens=[],     # no history yet
    )
    
    # Store pass 1's observation in notebook
    notebook_pages.append(page_0)
    notebook_hiddens.append(hidden_pool_vanilla)
    
    # Pass 2: Llama reads WITH LoRA (atom-modified attention)
    model.apply_lora(initial_scales)
    outputs_lora = model.llama(
        input_ids,
        output_hidden_states=True
    )
    model.remove_lora()
    hidden_pool_lora = outputs_lora.hidden_states[-1].mean(dim=1)
    
    # Controller observes LoRA-modified comprehension → page + next scales
    page_1, scales_for_cycle2, focus_1 = model.controller(
        hidden_states_all_layers=outputs_lora.hidden_states,
        history_pages=notebook_pages,       # [page_0]
        history_hiddens=notebook_hiddens,    # [hidden_vanilla]
    )
    
    # Möbius warp page_1
    page_1 = model.mobius(page_1, focus_1)
    
    # Update notebook
    notebook_pages.append(page_1)
    notebook_hiddens.append(hidden_pool_lora)
    
    # Generate from pass 2 (with LoRA — correct format!)
    gen_logits_1 = model.generate(outputs_lora, input_ids)
    generated_text = tokenizer.decode(gen_logits_1.argmax(-1)[0])
    predicted = extract_answer(generated_text)
    prev_results.append(predicted if predicted else 0)
    
    # Next cycle's scales come from controller's observation of pass 2
    current_scales = scales_for_cycle2
    
    # ========================================
    # CYCLES 2+ (one pass each, same as before)
    # ========================================
    
    for cycle in range(1, max_cycles):
        # Text injection with previous results
        text_ctx = format_text_injection(prev_results)
        cycle_input_ids = tokenize(text_ctx + problem_text)
        
        # Llama forward with LoRA
        model.apply_lora(current_scales)
        outputs = model.llama(
            cycle_input_ids,
            output_hidden_states=True
        )
        model.remove_lora()
        hidden_pool = outputs.hidden_states[-1].mean(dim=1)
        
        # Controller observes → page + next scales
        page, next_scales, focus = model.controller(
            hidden_states_all_layers=outputs.hidden_states,
            history_pages=notebook_pages,
            history_hiddens=notebook_hiddens,
        )
        
        # Möbius warp
        page = model.mobius(page, focus)
        
        # Update notebook
        notebook_pages.append(page)
        notebook_hiddens.append(hidden_pool)
        
        # Generate
        gen_logits = model.generate(outputs, cycle_input_ids)
        generated_text = tokenizer.decode(gen_logits.argmax(-1)[0])
        predicted = extract_answer(generated_text)
        prev_results.append(predicted if predicted else 0)
        
        # Confidence check
        if cycle >= 1:
            conf = model.confidence_head(notebook_pages)
            if conf > 0.85:
                break
        
        current_scales = next_scales
    
    return predicted
```

---

## Training Loop

```python
def train_step(model, problem_ids, cycle_targets, cycle_gen_targets,
               final_answer, final_accuracy, num_cycles, tokenizer):
    """
    Training with two-pass cycle 1.
    
    Cycle 1 has two forward passes — both contribute to the loss.
    Pass 1 (vanilla) provides observation for the controller.
    Pass 2 (LoRA) provides the actual generation.
    Only pass 2's generation gets the gen_loss.
    """
    notebook_pages = []
    notebook_hiddens = []
    all_scales = []
    total_loss = 0.0
    available_targets = list(cycle_targets)
    prev_results = []
    
    # ========================================
    # CYCLE 1: TWO-PASS
    # ========================================
    
    # Pass 1: vanilla (no LoRA, no generation loss)
    outputs_vanilla = model.llama(problem_ids, output_hidden_states=True)
    hidden_pool_vanilla = outputs_vanilla.hidden_states[-1].mean(dim=1)
    
    page_0, initial_scales, focus_0 = model.controller(
        outputs_vanilla.hidden_states, [], []
    )
    all_scales.append(initial_scales)
    
    notebook_pages.append(page_0)
    notebook_hiddens.append(hidden_pool_vanilla.detach())  # detach vanilla pass
    
    # Pass 2: with LoRA (generation loss applies here)
    model.apply_lora(initial_scales)
    outputs_lora = model.llama(problem_ids, output_hidden_states=True)
    model.remove_lora()
    hidden_pool_lora = outputs_lora.hidden_states[-1].mean(dim=1)
    
    page_1, scales_for_next, focus_1 = model.controller(
        outputs_lora.hidden_states, notebook_pages, notebook_hiddens
    )
    all_scales.append(scales_for_next)
    
    page_1 = model.mobius(page_1, focus_1)
    notebook_pages.append(page_1)
    notebook_hiddens.append(hidden_pool_lora)
    
    # Generation loss for cycle 1 (from pass 2 only)
    gen_logits_1 = model.generate(outputs_lora, problem_ids)
    gen_loss_1 = weighted_generation_loss(
        gen_logits_1, cycle_gen_targets[0], tokenizer, eos_weight=5.0
    )
    
    # Extraction + gating
    with torch.no_grad():
        text_1 = tokenizer.decode(gen_logits_1.argmax(-1)[0])
        pred_1 = extract_answer(text_1)
    
    # Flexible loss with consumption
    matched = check_and_consume(pred_1, available_targets, final_answer)
    gen_weight = 1.0 if matched else 0.1
    
    teacher_weight = per_cycle_target_weight(final_accuracy, 0, num_cycles)
    total_loss += teacher_weight * gen_weight * gen_loss_1
    
    prev_results.append(pred_1 if pred_1 else 0)
    current_scales = scales_for_next
    
    # ========================================
    # CYCLES 2+ (one pass each)
    # ========================================
    
    for cycle in range(1, num_cycles):
        text_ctx = format_text_injection(prev_results)
        cycle_input = tokenize(text_ctx + problem_text)
        
        model.apply_lora(current_scales)
        outputs = model.llama(cycle_input, output_hidden_states=True)
        model.remove_lora()
        hidden_pool = outputs.hidden_states[-1].mean(dim=1)
        
        page, next_scales, focus = model.controller(
            outputs.hidden_states, notebook_pages, notebook_hiddens
        )
        all_scales.append(next_scales)
        
        page = model.mobius(page, focus)
        notebook_pages.append(page)
        notebook_hiddens.append(hidden_pool)
        
        gen_logits = model.generate(outputs, cycle_input)
        gen_loss = weighted_generation_loss(
            gen_logits, cycle_gen_targets[cycle], tokenizer, eos_weight=5.0
        )
        
        with torch.no_grad():
            text = tokenizer.decode(gen_logits.argmax(-1)[0])
            pred = extract_answer(text)
        
        consumed = [t for t in cycle_targets if t not in available_targets]
        if pred in consumed:
            gen_weight = 0.0   # copying — zero reward
        elif check_and_consume(pred, available_targets, final_answer):
            gen_weight = 1.0   # correct — full reward
        else:
            gen_weight = 0.1   # wrong — reduced
        
        teacher_weight = per_cycle_target_weight(final_accuracy, cycle, num_cycles)
        total_loss += teacher_weight * gen_weight * gen_loss
        
        prev_results.append(pred if pred else 0)
        current_scales = next_scales
    
    # Regularizers
    total_loss += 0.05 * scale_diversity_loss(all_scales, target_cos=0.3)
    total_loss += 0.01 * isotropic_reg(raw_pages)
    total_loss += 0.1 * confidence_entropy_loss(notebook_pages, final_answer)
    
    return total_loss
```

---

## Cost Analysis

```
BEFORE (one pass per cycle, but cycle 1 blind):
  3 cycles × 1 Llama forward = 3 forwards
  Cycle 1 produces garbage → wasted
  Effective: 2 useful forwards (cycles 2-3)

AFTER (two-pass cycle 1):
  Cycle 1: 2 Llama forwards (vanilla + LoRA)
  Cycles 2-3: 1 forward each
  Total: 4 forwards for 3 cycles
  
  +1 forward compared to before
  But cycle 1 now WORKS → 3 useful forwards instead of 2
  Net: +33% compute for +50% useful output
```

At 10 min/epoch, the extra forward adds ~2-3 min → ~12-13 min/epoch. Still faster than the old 15 min with separate perceiver + hypernetwork.

---

## What Changes About the Notebook

The notebook now has an EXTRA entry from pass 1:

```
BEFORE (3 cycles):
  notebook = [page_1, page_2, page_3]  (3 entries)

AFTER (3 cycles with two-pass cycle 1):
  notebook = [page_0, page_1, page_2, page_3]  (4 entries)
  
  page_0: from vanilla pass (what Llama naturally understands)
  page_1: from LoRA pass (what Llama sees with atoms)
  page_2: cycle 2's observation
  page_3: cycle 3's observation
```

Page_0 is interesting — it captures Llama's NATURAL comprehension before any atom modification. The controller at cycle 2 can compare page_0 (natural) vs page_1 (modified) to understand what the atoms CHANGED. This is genuine self-reflection: "how did the atoms alter my understanding?"

---

## Why This Fixes Cycle 1

```
BEFORE:
  Controller at cycle 1: reads nothing → produces fixed scales
  Same scales for "Natalia sold clips" and "Mark has toys"
  
AFTER:
  Controller at cycle 1: reads Llama's vanilla comprehension
  "Natalia sold clips" → specific hidden states → specific scales
  "Mark has toys" → different hidden states → different scales
  
  Problem-specific scales from cycle 1.
  The constant-function problem is eliminated.
```

And cycle 2 already WORKS (proven by the debug: "40+20=60 #### 60" correct). The two-pass just applies the same working mechanism to cycle 1.

---

## Cycle 3 Scale Collapse (Separate Fix)

The debug also showed cycle 3 scales collapsing to near-zero (norm 0.08 vs 3.4). This is separate from the cycle 1 issue. The controller learns to turn off LoRA at cycle 3.

Fix: minimum scale norm or stronger scale_reg:

```python
# After controller produces scales:
scales = model.controller.scale_head(shared)
scales = torch.tanh(scales)
scales = torch.clamp(scales, -3.0, 3.0)

# Ensure minimum scale activity (prevent collapse to zero)
scale_norm = scales.norm(dim=-1, keepdim=True)
min_norm = 1.0  # minimum scale norm
if scale_norm < min_norm:
    scales = scales * (min_norm / (scale_norm + 1e-8))
```

Or simpler — increase scale_reg weight to penalize near-zero scales more aggressively. The current scale_reg pushes scales toward [-3, 3] range. If it also penalizes near-zero, cycle 3 can't collapse:

```python
def scale_reg_with_minimum(scales, min_norm=1.0):
    """Penalize scales outside [-3,3] AND scales near zero."""
    clamp_penalty = F.relu(scales.abs() - 3.0).mean()
    collapse_penalty = F.relu(min_norm - scales.norm(dim=-1)).mean()
    return clamp_penalty + collapse_penalty
```

---

## What to Monitor

```
1. Cycle 1 generation format:
   BEFORE: "A) $100 B) $120 C) $140" (multiple choice garbage)
   TARGET: "Natalia sold 48 clips #### 48</s>" (GSM8K format)
   This is the PRIMARY test — does two-pass fix cycle 1?

2. Cycle 1 scale diversity across problems:
   BEFORE: identical for ALL problems (constant function)
   TARGET: different for different problems (informed by comprehension)

3. Gen loss trajectory:
   BEFORE: stuck at 2.20-2.25 (cycle 1 garbage dragging it up)
   TARGET: should drop faster (cycle 1 now contributes correct generation)

4. Cycle 3 scale norm:
   BEFORE: 0.08 (collapsed to near-zero)
   TARGET: > 1.0 (active atoms)

5. Accuracy:
   Cycle 1: should jump immediately (no more garbage)
   Cycle 2: should stay at current level (already works)
   Final: should improve significantly (cycle 1 + cycle 2 both contributing)
```

---

## Expected Impact

```
BEFORE (cycle 1 blind):
  Cycle 1: garbage (multiple choice) → 0% useful
  Cycle 2: correct ("40+20=60 #### 60") → working
  Cycle 3: collapsed (scales near zero) → 0% useful
  Gen loss: stuck at 2.20 (dragged up by garbage cycle 1)
  Final: 10% (only cycle 2 contributing)

AFTER (cycle 1 informed):
  Cycle 1: correct (controller saw comprehension) → should be ~50%+ (like old gen)
  Cycle 2: correct (already works) → should stay ~50%+
  Cycle 3: active (min scale norm) → should contribute
  Gen loss: should drop below 1.5 quickly (no more garbage cycles)
  Final: if cycle 1 and 2 both at 50% → final could reach 25%+
```

---

## What NOT to Do

```
- Do NOT skip pass 1 to save compute.
  Pass 1 provides the comprehension the controller needs.
  Without it, cycle 1 is blind → garbage → everything downstream suffers.
  One extra forward pass is cheap compared to garbage cycle 1.

- Do NOT apply generation loss to pass 1.
  Pass 1 is vanilla Llama (no LoRA) — its generation is multiple-choice garbage.
  Only pass 2 (with LoRA) gets generation loss.
  Pass 1 is OBSERVATION ONLY.

- Do NOT detach pass 1 hidden states from the graph.
  The gradient should flow: cycle 2 loss → scales → controller → 
  pass 1 hidden states → Llama embedding. This creates the deepest
  recurrent gradient path. Detaching would lose it.
  
  Exception: detach if OOM. The extra backward through pass 1 uses memory.

- Do NOT store pass 1's generation output.
  It's garbage (multiple choice). Don't extract from it.
  Don't inject it as text. Don't use it for anything except
  providing hidden states to the controller.
```
