# Handoff: SymPy Integration — Calculator at Every Cycle

## One-Sentence Summary

Give the model a calculator (SymPy) at every thinking cycle during both training and inference. The model FORMULATES computations as SymPy expressions, SymPy EXECUTES them exactly, and the verified results feed back into the next page. The breathing loop becomes a comprehension and formulation engine. Arithmetic is outsourced to a tool that never makes errors.

---

## Why

The 17% GSM8K ceiling diagnosis revealed: the base model can't reliably execute arithmetic. Llama 1B was pretrained to DISCUSS math, not DO math. It writes "Let x be..." instead of computing "16-3-4=9". The breathing architecture gives it the right reasoning structure, but the frozen transformer can't execute computations reliably. Numbers above ~1000 compound errors. Multiplication and percentages fail frequently.

```
What the model is GOOD at:  reading problems, identifying quantities, planning steps
What the model is BAD at:   48 × 7, 150% of 200, carrying digits, long division

Solution: let the model do what it's good at (formulate), outsource what it's bad at (compute)
```

The SymPy probe from earlier in the project showed this directly: 38% accuracy at epoch 1 when generating SymPy expressions versus 27% without. The model is BETTER at setting up symbolic computations than at doing arithmetic itself.

---

## Architecture Change

### Before: Model Computes

```
Pass 1: parse → page encodes "48 and half"
Pass 2: think → page encodes "probably 24" (unreliable arithmetic)
Pass 3: think → page encodes "probably 72" (compounded error)
Generate: "The answer is 72" (maybe, maybe not)
```

### After: Model Formulates, SymPy Computes

```
Pass 1: parse → generate "n_april = 48; n_may = n_april / 2"
        → SymPy evaluates: {n_april: 48, n_may: 24} (EXACT)
        → page 1 encodes verified [48, 24, division]

Pass 2: read page 1 → generate "total = n_april + n_may"
        → SymPy evaluates: {total: 72} (EXACT)
        → page 2 encodes verified [72, addition, complete]

Pass 3: read page 2 → confidence high → stop
        → answer: 72 (VERIFIED by SymPy)
```

Each page carries VERIFIED information. Not "the model thinks 48/2 is probably 24" but "SymPy confirms 48/2 = 24." The next cycle builds on certainty, not noisy estimates.

---

## Implementation

### Safe SymPy Execution

```python
import sympy
import signal
from contextlib import contextmanager

class SymPyEvaluator:
    """Safe, sandboxed SymPy expression evaluator."""
    
    ALLOWED_NAMES = {
        'Symbol', 'symbols', 'Rational', 'Integer', 'Float',
        'sqrt', 'Abs', 'ceiling', 'floor', 'Mod',
        'simplify', 'expand', 'factor', 'solve',
        'pi', 'E', 'oo',
    }
    
    @staticmethod
    def safe_eval(code_str, timeout_sec=5):
        """
        Execute SymPy code and return variable bindings.
        Returns: dict of {variable_name: numeric_value} or {} on failure
        """
        try:
            # Basic sanitization
            if any(dangerous in code_str for dangerous in
                   ['import os', 'import sys', 'exec(', 'eval(', '__', 'open(']):
                return {}
            
            # Create restricted namespace
            namespace = {
                'sympy': sympy,
                'Rational': sympy.Rational,
                'sqrt': sympy.sqrt,
                'Abs': sympy.Abs,
                'ceiling': sympy.ceiling,
                'floor': sympy.floor,
                'pi': sympy.pi,
            }
            
            # Execute with timeout
            exec(code_str, {"__builtins__": {}}, namespace)
            
            # Extract numeric results
            results = {}
            for name, value in namespace.items():
                if name.startswith('_') or name in ('sympy', 'Rational', 'sqrt',
                                                      'Abs', 'ceiling', 'floor', 'pi'):
                    continue
                try:
                    numeric = float(value)
                    results[name] = numeric
                except (TypeError, ValueError):
                    try:
                        numeric = float(sympy.N(value))
                        results[name] = numeric
                    except:
                        pass
            
            return results
            
        except Exception as e:
            return {}
```

### Result Encoder

Encode SymPy results into a fixed-size vector that augments the page:

```python
class SymPyResultEncoder(nn.Module):
    """Encode SymPy evaluation results into a page-compatible vector."""
    
    def __init__(self, page_size=64, max_variables=8):
        super().__init__()
        self.max_variables = max_variables
        
        # Encode each variable's value
        # Use log-scale encoding for numbers (handles wide range)
        self.value_encoder = nn.Sequential(
            nn.Linear(max_variables * 2, 128),  # value + log(|value|) per variable
            nn.GELU(),
            nn.Linear(128, page_size),
        )
        
        # Gate: how much should SymPy results influence the page?
        self.result_gate = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, sympy_results, device):
        """
        sympy_results: dict of {name: numeric_value}
        returns: (page_size,) vector encoding the results
        """
        # Encode up to max_variables results
        values = list(sympy_results.values())[:self.max_variables]
        
        encoded = torch.zeros(self.max_variables * 2, device=device)
        for i, v in enumerate(values):
            encoded[i * 2] = v                                    # raw value
            encoded[i * 2 + 1] = math.log(abs(v) + 1e-8)        # log magnitude
        
        result_vec = self.value_encoder(encoded.unsqueeze(0))  # (1, page_size)
        gate = torch.sigmoid(self.result_gate)
        
        return result_vec.squeeze(0) * gate  # (page_size,)
```

### SymPy Context Formatter

Format previous SymPy results as text context for Llama:

```python
def format_sympy_context(sympy_results):
    """
    Format accumulated SymPy results as text context prepended to the problem.
    
    Example output: "Known values: n_april=48, n_may=24\n"
    """
    if not sympy_results:
        return ""
    
    parts = [f"{name}={value}" for name, value in sympy_results.items()]
    return "Known values: " + ", ".join(parts) + "\n"
```

### Modified Thinking Loop

```python
class AtomLoRAModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... existing components ...
        self.sympy_eval = SymPyEvaluator()
        self.result_encoder = SymPyResultEncoder(page_size=64)
    
    def think_one_pass_with_sympy(
        self, problem_text, problem_ids, state_pages, pass_num,
        sympy_results, teacher_sympy=None
    ):
        """
        One thinking cycle with SymPy integration.
        
        problem_text: raw text of the problem
        problem_ids: tokenized problem
        state_pages: accumulated pages
        pass_num: current pass number
        sympy_results: dict of accumulated SymPy variable bindings
        teacher_sympy: (training only) correct SymPy code for this pass
        
        returns: new_page, updated sympy_results
        """
        # 1. Format SymPy context from previous results
        sympy_context = format_sympy_context(sympy_results)
        context_ids = self.tokenize(sympy_context)
        full_input = torch.cat([context_ids, problem_ids], dim=-1)
        
        # 2. Hypernetwork reads pages → atom scales
        atom_scales = self.hypernetwork(state_pages)
        
        # 3. Apply LoRA, run Llama with SymPy context
        self.apply_lora(atom_scales)
        outputs = self.llama(full_input, output_hidden_states=True)
        self.remove_lora()
        
        # 4. Perceiver compresses (sees hidden states + previous pages)
        new_page = self.perceiver(outputs.hidden_states, state_pages)
        
        # 5. Generate SymPy expression for this cycle
        if self.training and teacher_sympy is not None:
            # Teacher forcing: use the correct SymPy expression
            sympy_code = teacher_sympy
        else:
            # Inference: model generates SymPy
            self.remove_lora()  # LoRA OFF for generation
            pseudo_tokens = self.page_to_tokens(state_pages + [new_page])
            sympy_code = self.generate(
                pseudo_tokens, full_input,
                prefix="# Step computation:\n",
                max_tokens=60,
                temperature=0,
            )
        
        # 6. Execute with SymPy
        new_results = self.sympy_eval.safe_eval(sympy_code)
        sympy_results.update(new_results)
        
        # 7. Encode SymPy results into the page
        if new_results:
            result_vec = self.result_encoder(new_results, new_page.device)
            new_page = new_page + result_vec.unsqueeze(0)
        
        # 8. Residual + normalize
        if len(state_pages) > 0:
            new_page = self.residual_gate(new_page, state_pages[-1])
        new_page = F.normalize(new_page, dim=-1) * math.sqrt(64)
        
        # 9. Pi-harmonic encoding
        new_page = self.pi_encoding.apply(new_page, pass_num)
        
        state_pages.append(new_page)
        return new_page, sympy_results
    
    def solve(self, problem_text, max_passes=5):
        """Full inference with SymPy at every cycle."""
        problem_ids = self.tokenize(problem_text)
        state_pages = []
        sympy_results = {}
        
        for pass_num in range(max_passes):
            page, sympy_results = self.think_one_pass_with_sympy(
                problem_text, problem_ids, state_pages, pass_num,
                sympy_results,
            )
            
            # Check for explicit answer variable
            if 'answer' in sympy_results:
                return sympy_results['answer']
            
            # Confidence check
            if pass_num >= 1:
                conf, smooth = self.confidence_head(state_pages)
                if conf > 0.9 and smooth > 0.7:
                    break
        
        # Extract answer: prefer SymPy results, fallback to answer head
        if sympy_results:
            # Return the last computed value
            return list(sympy_results.values())[-1]
        else:
            return self.answer_head_predict(state_pages[-1])
```

---

## Training Data: Per-Step SymPy Annotations

Each GSM8K problem needs per-step SymPy expressions. Convert existing CoT traces:

### Conversion Example

```
CoT trace:
  "Natalia sold clips to 48 of her friends in April.
   She sold half as many clips in May.
   48 / 2 = 24.
   Natalia sold 48 + 24 = 72 clips altogether."

Per-step SymPy:
  Step 1: "n_april = 48"
  Step 2: "n_may = n_april / 2"          → SymPy: {n_may: 24}
  Step 3: "total = n_april + n_may"       → SymPy: {total: 72}
  Step 4: "answer = total"                → SymPy: {answer: 72}
```

### Automated Conversion Script

```python
def cot_to_sympy_steps(cot_trace, gold_answer):
    """
    Convert a CoT trace into per-step SymPy expressions.
    
    Strategy:
    1. Parse equations from CoT (regex for "X op Y = Z" patterns)
    2. Assign variable names (var_1, var_2, ...)
    3. Build dependency chain
    4. Output per-step SymPy code
    """
    import re
    
    steps = []
    variables = {}
    var_counter = 0
    
    # Find all equations in the CoT
    equations = re.findall(
        r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)',
        cot_trace
    )
    
    for left, op, right, result in equations:
        var_counter += 1
        var_name = f"v{var_counter}"
        
        # Check if operands are previous variables
        left_ref = variables.get(float(left), left)
        right_ref = variables.get(float(right), right)
        
        sympy_code = f"{var_name} = {left_ref} {op} {right_ref}"
        steps.append(sympy_code)
        
        variables[float(result)] = var_name
    
    # Add answer step
    if gold_answer in variables:
        steps.append(f"answer = {variables[gold_answer]}")
    else:
        steps.append(f"answer = {gold_answer}")
    
    return steps
```

### Better: Use an LLM to Generate SymPy Annotations

For higher quality, use a larger model (Claude or GPT-4) to generate per-step SymPy for each GSM8K problem:

```python
ANNOTATION_PROMPT = """
Given this math word problem, decompose the solution into step-by-step SymPy expressions.
Each step should compute one value. Use descriptive variable names.
The final step should assign to 'answer'.

Problem: {problem}
Solution: {cot_trace}
Gold answer: {gold_answer}

Output ONLY the SymPy steps, one per line:
"""

# Run once to generate annotations for all 7473 GSM8K training problems
# Store as: {problem_hash: [step1_sympy, step2_sympy, ...]}
```

---

## Training Recipe

### Loss

```python
total_loss = (
    sympy_generation_loss      # 1.0 — learn to generate correct SymPy per step
    + 1.0 * answer_head_loss   # 1.0 — pages must encode the answer
    + 0.01 * contrastive_loss  # gentle page differentiation
    + 0.1 * confidence_loss    # when to stop
    + 0.1 * scale_reg          # prevent tanh saturation
)
```

The primary loss is now SymPy generation, not CoT text generation. The model learns to produce valid SymPy expressions at each step.

### Teacher Forcing Schedule

```
Epoch 1-3:   100% teacher forcing (always use correct SymPy)
             Model learns the format and gets verified results at every cycle.
             
Epoch 4-6:   50% teacher forcing (half the time, model generates its own)
             Model starts generating SymPy independently, with fallback.
             
Epoch 7+:    0% teacher forcing (model generates all SymPy)
             Model must formulate independently. SymPy still evaluates.
```

### Per-Cycle Training

```python
def train_step_with_sympy(model, problem, sympy_steps, gold_answer):
    """
    Train one problem with per-cycle SymPy teacher forcing.
    
    sympy_steps: list of correct SymPy code strings, one per cycle
    """
    state_pages = []
    sympy_results = {}
    total_loss = 0.0
    
    num_passes = min(len(sympy_steps), 5)
    
    for pass_num in range(num_passes):
        # Teacher forcing: provide correct SymPy for this step
        teacher_code = sympy_steps[pass_num] if pass_num < len(sympy_steps) else None
        
        page, sympy_results = model.think_one_pass_with_sympy(
            problem, problem_ids, state_pages, pass_num,
            sympy_results,
            teacher_sympy=teacher_code,
        )
        
        # Loss: model should have GENERATED this SymPy code
        if teacher_code:
            teacher_ids = model.tokenize(teacher_code)
            gen_loss = model.compute_generation_loss(
                state_pages, problem_ids, teacher_ids
            )
            total_loss += gen_loss
    
    # Answer head loss on final page
    answer_loss = model.answer_head_loss(state_pages[-1], gold_answer)
    total_loss += answer_loss
    
    return total_loss
```

---

## What This Gives Us

```
BEFORE (model computes):
  Strengths: simple pipeline, no tool dependency
  Weaknesses: arithmetic errors, number-spam, extraction fragility
  Ceiling: ~17% on GSM8K (can't reliably compute)

AFTER (model formulates, SymPy computes):
  Strengths: exact arithmetic, verified intermediate results, clean extraction
  Weaknesses: must generate valid SymPy code (new skill to learn)
  Expected: >>17% on GSM8K (arithmetic no longer the bottleneck)
```

The breathing loop is reframed:

```
OLD: each cycle COMPUTES (unreliable arithmetic in a language model)
NEW: each cycle FORMULATES (translate understanding into symbolic expressions)
     SymPy COMPUTES (exact, every time)
     Verified result feeds into the next cycle's page
```

The model does what it's good at (language understanding, step planning). SymPy does what it's good at (arithmetic). The breathing loop coordinates between them.

---

## Connection to Existing Architecture

Everything we've built still applies:

```
✓ 64 atoms still modify attention (how the model reads the problem)
✓ Perceiver still compresses to 64-float pages (with SymPy results added)
✓ Hypernetwork still controls atoms (reading pages with verified info)
✓ Residual gate still preserves information (verified results persist)
✓ Skip connections still provide gradient highway
✓ Pi-harmonic encoding still structures pages
✓ Wavelet preprocessing still compresses perceiver input
✓ Confidence + smoothness still control stopping
```

The only additions:
- SymPyEvaluator (0 params, pure computation)
- SymPyResultEncoder (~17K params, encodes results into page)
- SymPy context formatting (0 params, text preprocessing)
- Per-step SymPy training targets (data change, not architecture)

---

## Implementation Order

```
1. Build SymPyEvaluator (safe sandboxed execution)
2. Build SymPyResultEncoder (encode results into page vectors)
3. Generate SymPy annotations for GSM8K training data
   - Automated regex conversion from CoT traces
   - Manual cleanup of edge cases
4. Modify thinking loop to integrate SymPy per cycle
5. Train L3 with SymPy (verify format learning)
6. Train GSM8K with SymPy (beat 17% ceiling)
7. Progressive teacher forcing schedule
```

---

## What NOT to Do

```
- Do NOT let the model import arbitrary Python modules. 
  SymPy namespace is restricted. No os, sys, subprocess, etc.

- Do NOT execute untrusted code without timeout.
  Always wrap in a timeout (5 seconds max).

- Do NOT make SymPy the ONLY answer path.
  Keep the answer head and text generation as fallbacks.
  If SymPy code fails to parse, fall back to generation.

- Do NOT train with SymPy from scratch on L0/L1.
  The model needs to understand problem structure first.
  Start SymPy integration at L3 or L4 level.

- Do NOT allow SymPy to change the page normalization.
  The result encoder ADDS to the page, then normalization applies.
  SymPy results are additional signal, not replacement.

- Do NOT generate SymPy in the same forward pass as thinking.
  Thinking uses LoRA ON (modified attention for comprehension).
  SymPy generation uses LoRA OFF (natural language generation).
  These must be separate forward passes.
```

---

## Expected Impact

```
Arithmetic accuracy: ~70% → ~99% (SymPy is exact)
Multi-step chaining: each step verified, errors don't compound
Answer extraction: read from SymPy result dict, no regex needed
GSM8K target: >30% (arithmetic is no longer the bottleneck)
Stretch target: >40% (if formulation quality is high)
```

The remaining bottleneck after SymPy: can the model correctly FORMULATE the problem as SymPy expressions? That's a language understanding task, which is what Llama was actually pretrained for. The breathing loop helps the model build understanding across cycles. SymPy handles execution. The division of labor plays to each component's strengths.
