# Handoff: SymPy Decoder — Separated Comprehension and Computation

## One-Sentence Summary

Build a small dedicated SymPy decoder (2-layer transformer, ~5M params, tiny math vocabulary) that reads pages and emits symbolic expressions. The LLM comprehends. The decoder formulates. SymPy computes. No contamination between English and SymPy syntax because they're completely separate systems communicating through the 64-float page.

---

## Why Separate

The previous SymPy integration tried to make the LLM generate BOTH English CoT AND SymPy code through the same generation head. This caused:

1. **Contamination:** "The values, v1=The answer" — SymPy syntax leaking into English
2. **Page collapse:** The generation loss (CoT text) dominated the answer head loss, killing page diversity
3. **Format confusion:** The model didn't know when to output English vs Python

The fix: two separate output systems that never interfere.

```
BEFORE (contaminated):
  One head → sometimes English, sometimes SymPy → confused

AFTER (clean):
  LLM generation head → English only ("The answer is 72")
  SymPy decoder head → SymPy only ("answer = v1 + v2")
  They share the PAGE as interface but nothing else
```

---

## Architecture

### SymPy Decoder (~8M params)

A small transformer decoder that reads a page as 64 TOKENS (one per dimension) and emits a SymPy expression. The decoder cross-attends over page dimensions, selectively querying specific frequency bands for specific token predictions.

**Why 64 tokens instead of 1 vector:** The original decoder read the page through a pinhole — one linear projection collapsing 64 floats into a single 256-dim vector. The decoder couldn't selectively attend to specific page dimensions. With the page as 64 tokens, the decoder's cross-attention can query: "to predict the operator, look at mid-frequency dims 20-30" and "to predict the number, look at high-frequency dims 50-60."

```python
import torch
import torch.nn as nn

class SymPyDecoder(nn.Module):
    """
    Decoder that reads a 64-float page as 64 separate tokens
    and emits a SymPy expression.
    
    Key change from v1: page enters as a SEQUENCE of 64 tokens,
    not a single vector. The decoder cross-attends over page dimensions,
    enabling selective reading of specific frequency bands.
    
    3-layer transformer decoder, 256 dim, ~8M params.
    """
    
    def __init__(self, page_size=64, d_model=256, nhead=4, 
                 num_layers=3, max_tokens=40):
        super().__init__()
        self.d_model = d_model
        self.max_tokens = max_tokens
        self.page_size = page_size
        
        # Math-specific vocabulary
        self.vocab = SymPyVocab()
        vocab_size = len(self.vocab)
        
        # Token embedding (for decoder's own tokens)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_tokens, d_model)
        
        # === PAGE AS 64 TOKENS (key change) ===
        self.page_dim_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.page_pos_embed = nn.Embedding(page_size, d_model)
        self.page_norm = nn.LayerNorm(d_model)
        
        # Project accumulated SymPy results as additional memory tokens
        self.result_project = nn.Sequential(
            nn.Linear(page_size, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # 3-layer transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output = nn.Linear(d_model, vocab_size)
        self.norm = nn.LayerNorm(d_model)
    
    def prepare_page_memory(self, page, result_embedding=None):
        """Convert page to 64 memory tokens for cross-attention."""
        device = page.device
        page_tokens = page.unsqueeze(-1)  # (batch, 64, 1)
        page_embedded = self.page_dim_embed(page_tokens)  # (batch, 64, d_model)
        positions = torch.arange(self.page_size, device=device)
        page_embedded = page_embedded + self.page_pos_embed(positions)
        page_embedded = self.page_norm(page_embedded)
        
        if result_embedding is not None:
            result_mem = self.result_project(result_embedding).unsqueeze(1)
            memory = torch.cat([page_embedded, result_mem], dim=1)
        else:
            memory = page_embedded
        return memory
    
    def forward(self, page, result_embedding=None, target_tokens=None):
        device = page.device
        memory = self.prepare_page_memory(page, result_embedding)
        
        if target_tokens is not None:
            seq_len = target_tokens.size(1)
            tgt = self.embed(target_tokens) + self.pos_embed(
                torch.arange(seq_len, device=device))
            tgt = self.norm(tgt)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=device)
            out = self.decoder(tgt, memory, tgt_mask=causal_mask)
            return self.output(out)
        else:
            return self.generate(memory, device)
    
    def generate(self, memory, device, temperature=0.0, min_tokens=5):
        """Autoregressive generation with EOS blocking for first min_tokens."""
        tokens = [self.vocab.bos_id]
        for step in range(self.max_tokens):
            token_ids = torch.tensor([tokens], device=device)
            tgt = self.embed(token_ids) + self.pos_embed(
                torch.arange(len(tokens), device=device))
            tgt = self.norm(tgt)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                len(tokens), device=device)
            out = self.decoder(tgt, memory, tgt_mask=causal_mask)
            logits = self.output(out[:, -1, :])
            if step < min_tokens:
                logits[:, self.vocab.eos_id] = -1e9
            if temperature == 0:
                next_token = logits.argmax(dim=-1).item()
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)
            if next_token == self.vocab.eos_id:
                break
        return self.vocab.decode(tokens[1:-1])
    
    def forward_scheduled_sampling(self, page, target_tokens,
                                    result_embedding=None, sample_rate=0.0):
        """
        Scheduled sampling: with probability sample_rate, use decoder's
        own prediction instead of teacher token. Prevents collapse to
        safe default at generation time.
        
        sample_rate=0.0: pure teacher forcing (early training)
        sample_rate=0.2: 80% teacher, 20% self (mid training)
        sample_rate=0.5: half and half (late training)
        """
        device = page.device
        memory = self.prepare_page_memory(page, result_embedding)
        batch_size, seq_len = target_tokens.shape
        input_tokens = target_tokens[:, :1]  # BOS
        all_logits = []
        
        for t in range(1, seq_len):
            tgt = self.embed(input_tokens) + self.pos_embed(
                torch.arange(input_tokens.size(1), device=device))
            tgt = self.norm(tgt)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                input_tokens.size(1), device=device)
            out = self.decoder(tgt, memory, tgt_mask=causal_mask)
            logits = self.output(out[:, -1:, :])
            all_logits.append(logits)
            
            if self.training and torch.rand(1).item() < sample_rate:
                next_token = logits.squeeze(1).argmax(dim=-1, keepdim=True)
            else:
                next_token = target_tokens[:, t:t+1]
            input_tokens = torch.cat([input_tokens, next_token], dim=1)
        
        return torch.cat(all_logits, dim=1)
```

**Page cross-attention enables frequency-selective reading:**

```
Predicting "v1":     cross-attend to low-freq dims 0-15 (which variable?)
Predicting "=":      no page attention needed (always "=")
Predicting "48":     cross-attend to high-freq dims 40-63 (exact number)
Predicting "/":      cross-attend to mid-freq dims 16-40 (operation type)
Predicting "2":      cross-attend to high-freq dims 40-63 (exact operand)
```

**Scheduled sampling ramp (competence-based, not epoch-based):**

```
dec_loss > 1.0:   sample_rate = 0.0  (still learning format)
dec_loss 0.5-1.0: sample_rate = 0.2  (format learned, start self-reliance)
dec_loss 0.2-0.5: sample_rate = 0.5  (mostly competent)
dec_loss < 0.2:   sample_rate = 0.8  (self-reliant, teacher as safety net)
```

### Math-Specific Vocabulary (~50 tokens)

```python
class SymPyVocab:
    """
    Tiny vocabulary for mathematical expressions.
    ~50 tokens total. NOT the LLM's 128K vocabulary.
    """
    
    def __init__(self):
        self.tokens = [
            # Special
            '<bos>', '<eos>', '<pad>',
            
            # Digits
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            
            # Decimal point
            '.',
            
            # Operators
            '+', '-', '*', '/', '//', '%', '**',
            
            # Parentheses
            '(', ')',
            
            # Assignment
            '=',
            
            # Separators
            ';', ',',
            
            # Variables (up to 12 intermediates)
            'v1', 'v2', 'v3', 'v4', 'v5', 'v6',
            'v7', 'v8', 'v9', 'v10', 'v11', 'v12',
            
            # Special variable
            'answer',
            
            # Common functions
            'abs', 'max', 'min', 'round',
            
            # Common constants
            'Rational',
        ]
        
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_token = {i: t for i, t in enumerate(self.tokens)}
        
        self.bos_id = self.token_to_id['<bos>']
        self.eos_id = self.token_to_id['<eos>']
        self.pad_id = self.token_to_id['<pad>']
    
    def __len__(self):
        return len(self.tokens)
    
    def encode(self, expression_str):
        """
        Tokenize a SymPy expression string.
        
        "v1 = 48; v2 = v1 / 2" → [v1, =, 4, 8, ;, v2, =, v1, /, 2]
        """
        tokens = []
        i = 0
        expr = expression_str.strip()
        
        while i < len(expr):
            # Skip whitespace
            if expr[i] == ' ':
                i += 1
                continue
            
            # Try multi-char tokens first (v10, v11, v12, //, **, answer, etc.)
            matched = False
            for length in [6, 3, 2]:  # 'answer', 'v10', 'v1', '//', '**'
                candidate = expr[i:i+length]
                if candidate in self.token_to_id:
                    tokens.append(self.token_to_id[candidate])
                    i += length
                    matched = True
                    break
            
            if not matched:
                # Single character token
                char = expr[i]
                if char in self.token_to_id:
                    tokens.append(self.token_to_id[char])
                i += 1
        
        return tokens
    
    def decode(self, token_ids):
        """Convert token IDs back to a SymPy expression string."""
        tokens = [self.id_to_token.get(t, '?') for t in token_ids
                  if t not in (self.bos_id, self.eos_id, self.pad_id)]
        
        # Join with smart spacing
        result = []
        for t in tokens:
            if t in ('=', '+', '-', '*', '/', ';'):
                result.append(f' {t} ')
            else:
                result.append(t)
        
        return ''.join(result).strip()
```

---

## Integration Into Thinking Loop

```python
def think_one_pass_with_decoder(self, problem_text, problem_ids, 
                                 state_pages, pass_num, sympy_results,
                                 teacher_sympy_tokens=None):
    """
    One thinking cycle with separated comprehension and computation.
    
    COMPREHENSION: Llama reads problem with atom-modified attention
    COLLAPSE:      Perceiver compresses to page
    FORMULATION:   SymPy decoder reads page → emits expression
    COMPUTATION:   SymPy evaluates expression exactly
    FEEDBACK:      Result encoded back into page
    """
    
    # 1. Format SymPy context from previous results
    sympy_context = format_sympy_context(sympy_results)
    context_ids = self.tokenize(sympy_context)
    full_input = torch.cat([context_ids, problem_ids], dim=-1)
    
    # 2. COMPREHENSION: Hypernetwork reads pages → atom scales
    atom_scales = self.hypernetwork(state_pages)
    
    # 3. EXPAND: Apply atom LoRA, run Llama
    self.apply_lora(atom_scales)
    outputs = self.llama(full_input, output_hidden_states=True)
    self.remove_lora()
    
    # 4. COLLAPSE: Perceiver compresses (sees hidden states + previous pages)
    new_page = self.perceiver(outputs.hidden_states, state_pages)
    
    # 5. FORMULATION: SymPy decoder reads page → expression
    result_embedding = self.result_encoder(sympy_results) if sympy_results else None
    
    if self.training and teacher_sympy_tokens is not None:
        # Teacher forcing: provide correct tokens, get loss
        sympy_logits = self.sympy_decoder(
            new_page, result_embedding, target_tokens=teacher_sympy_tokens
        )
        # Loss computed externally
    else:
        # Inference: generate expression
        sympy_code = self.sympy_decoder(new_page, result_embedding)
    
    # 6. COMPUTATION: SymPy evaluates
    if not self.training:
        new_results = self.sympy_eval.safe_eval(sympy_code)
        sympy_results.update(new_results)
    elif teacher_sympy_tokens is not None:
        # During training, evaluate the TEACHER expression (always correct)
        teacher_code = self.sympy_decoder.vocab.decode(teacher_sympy_tokens[0].tolist())
        new_results = self.sympy_eval.safe_eval(teacher_code)
        sympy_results.update(new_results)
    
    # 7. FEEDBACK: Encode SymPy result into page
    if new_results:
        result_vec = self.result_encoder(new_results, new_page.device)
        new_page = new_page + result_vec.unsqueeze(0)
    
    # 8. RESIDUAL + NORMALIZE
    if len(state_pages) > 0:
        new_page = self.residual_gate(new_page, state_pages[-1])
    new_page = F.normalize(new_page, dim=-1) * math.sqrt(64)
    
    # 9. Pi-harmonic encoding
    new_page = self.pi_encoding.apply(new_page, pass_num)
    
    state_pages.append(new_page)
    return new_page, sympy_results, sympy_logits if self.training else None
```

---

## Training

### Loss (Clean Separation)

```python
# Three loss terms operating on three DIFFERENT outputs:

# 1. SymPy decoder loss — teach decoder to formulate correctly
#    Operates on: sympy_decoder output (tiny vocab)
#    Gradient to: sympy_decoder → page → perceiver → atoms
sympy_loss = F.cross_entropy(
    sympy_logits.view(-1, vocab_size),
    teacher_sympy_tokens.view(-1),
    ignore_index=pad_id,
)

# 2. Answer head loss — teach pages to encode the answer
#    Operates on: answer_head output (digit classification)
#    Gradient to: answer_head → page → perceiver → atoms
answer_loss = answer_head_loss(state_pages[-1], gold_answer)

# 3. Contrastive loss — prevent page collapse
#    Operates on: page cosine similarities
#    Gradient to: page → perceiver → atoms
contrastive_loss = target_cosine_contrastive(pages, gold_answers, target=0.7)

# 4. Scale regularization — prevent tanh saturation
scale_reg = (raw_logits ** 2).mean()

# Combined:
total_loss = (1.0 * sympy_loss          # primary: learn to formulate
              + 1.0 * answer_loss        # secondary: pages encode answer
              + 0.05 * contrastive_loss   # insurance: prevent collapse
              + 0.1 * scale_reg)          # insurance: keep gradient alive

# NOTE: NO generation_loss during training with SymPy decoder.
# The LLM generation path is only used at final answer time.
# This eliminates the contamination problem entirely.
```

### Why No Generation Loss

This is the key insight. The generation loss (CoT cross-entropy on LLM output) was the dominant gradient signal that killed page diversity. By removing it and replacing with sympy_loss, the page diversity is preserved:

```
BEFORE:
  generation_loss (huge): "generate correct English text" → doesn't need diverse pages
  answer_head_loss (small): "encode answer in page" → needs diverse pages
  generation_loss wins → pages collapse

AFTER:
  sympy_loss (moderate): "formulate correct SymPy" → needs diverse pages (different problems need different formulas)
  answer_head_loss (equal): "encode answer in page" → needs diverse pages
  BOTH losses want diverse pages → pages differentiate
```

The SymPy decoder INHERENTLY needs per-problem pages because "v1 = 48; v2 = v1 / 2" is different from "v1 = 200; v2 = v1 * 0.15". The formulation is problem-specific. The page must encode problem-specific information for the decoder to formulate correctly. Page diversity falls out naturally.

### Teacher Forcing Schedule

```
Phase 1 (epoch 1-5):   100% teacher forcing
  The decoder learns the vocabulary and format.
  Always receives correct SymPy tokens.
  SymPy evaluates the correct expression → verified results feed back.
  
Phase 2 (epoch 6-10):  50% teacher forcing
  Half the time decoder generates its own expression.
  If SymPy eval fails → no result feeds back (graceful degradation).
  If SymPy eval succeeds → verified result feeds back.
  
Phase 3 (epoch 11+):   0% teacher forcing
  Decoder generates all expressions independently.
  Model has learned the format and can formulate on its own.
```

### Adaptive Data Weighting (AdaBoost-Inspired)

After each epoch, reweight training problems by difficulty. Problems the model solved correctly get downweighted. Problems it got wrong get upweighted. Training compute flows to the model's frontier — no wasted epochs on easy problems.

```python
# After each epoch eval:
for i, (problem, was_correct) in enumerate(eval_results):
    if was_correct:
        sample_weights[i] *= 0.5   # solved — see less often
    else:
        sample_weights[i] *= 2.0   # failed — see more often

# Normalize weights
sample_weights = sample_weights / sample_weights.sum() * len(problems)

# Use weighted sampling in the next epoch's dataloader
sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(problems))
```

This directly fights overfitting: easy problems fade from training before the model memorizes them. Hard problems get repeated until the model learns them. Ten lines of code, proven technique (AdaBoost is 30 years old), directly addresses the overfitting pattern we've seen at every difficulty level.

### Training Data: Per-Step SymPy Annotations

Each GSM8K problem needs per-step SymPy expressions in the decoder's vocabulary:

```
Problem: "Natalia sold clips to 48 of her friends in April.
          She sold half as many clips in May.
          How many clips did Natalia sell altogether?"

Per-step SymPy (in decoder vocabulary):
  Step 1: "v1 = 4 8"                          → SymPy: {v1: 48}
  Step 2: "v2 = v1 / 2"                       → SymPy: {v2: 24}
  Step 3: "answer = v1 + v2"                   → SymPy: {answer: 72}

Tokenized (for teacher forcing):
  Step 1: [v1, =, 4, 8, <eos>]
  Step 2: [v2, =, v1, /, 2, <eos>]
  Step 3: [answer, =, v1, +, v2, <eos>]
```

Generate annotations using a larger model (Claude/GPT-4) once for all 7,473 GSM8K problems. Store as JSON:

```json
{
  "problem": "Natalia sold clips...",
  "gold_answer": 72,
  "sympy_steps": [
    "v1 = 48",
    "v2 = v1 / 2",
    "answer = v1 + v2"
  ]
}
```

---

## Full Solve Pipeline

```python
def solve(self, problem_text, max_passes=8):
    """
    Full inference with separated comprehension and computation.
    """
    problem_ids = self.tokenize(problem_text)
    state_pages = []
    sympy_results = {}
    
    for pass_num in range(max_passes):
        # Think one pass (comprehend → collapse → formulate → compute → feedback)
        page, sympy_results, _ = self.think_one_pass_with_decoder(
            problem_text, problem_ids, state_pages, pass_num, sympy_results
        )
        
        # Check if SymPy produced an answer
        if 'answer' in sympy_results:
            return int(sympy_results['answer'])
        
        # Confidence check
        if pass_num >= 1:
            conf, smooth = self.confidence_head(state_pages)
            if conf > 0.9 and smooth > 0.7:
                break
    
    # Fallback extraction (in order of preference):
    # 1. Last SymPy result
    if sympy_results:
        return int(list(sympy_results.values())[-1])
    
    # 2. Answer head reads last page
    head_answer = self.answer_head.decode(state_pages[-1])
    if head_answer is not None:
        return head_answer
    
    # 3. LLM generation with "The answer is " prefix (last resort)
    pseudo_tokens = self.page_to_tokens(state_pages)
    generated = self.llama.generate(
        pseudo_tokens + problem_ids,
        max_new_tokens=20,
    )
    return extract_number(generated)
```

---

## Parameter Budget

```
Component                    Params      Role
─────────────────────────────────────────────────────────────
Llama 3.2 1B (frozen)        1,230M      Comprehension
7-Layer Perceiver             105M       Collapse (compress)
64 LoRA Atoms                  82M       Expand (rewire attention)
Atom Hypernetwork              10M       Control (which atoms)
SymPy Decoder (NEW)            ~5M       Formulation (page → SymPy)
SymPy Result Encoder           17K       Feedback (results → page)
Page-to-Tokens                0.5M       Generation bridge (fallback)
Answer Head                    4K        Digit extraction (fallback)
Confidence Head               79K        Stop decision
Residual Gate                  8K        Memory persistence
Skip connections              ~1.1M      Gradient highway
─────────────────────────────────────────────────────────────
Total:                        ~1.43B
Trainable:                    ~203M     (14.2%)
New (SymPy decoder):           ~5M      (0.35%)
```

---

## Component Responsibilities (One Job Each)

```
Llama:          COMPREHENDS (reads problem with modified attention)
Atoms:          FOCUS (control what Llama attends to)
Hypernetwork:   DECIDE (which atoms to activate based on pages)
Perceiver:      COMPRESS (collapse understanding to 64 floats)
SymPy Decoder:  FORMULATE (translate page to symbolic math)
SymPy:          COMPUTE (exact arithmetic, zero parameters)
Result Encoder: FEEDBACK (verified results back into pages)
Residual Gate:  REMEMBER (information persists across cycles)
Confidence:     STOP (when thinking is sufficient)
Answer Head:    EXTRACT (digits from last page, fallback)
Pattern DB:     RECALL (what worked before, future)
```

No component does two jobs. No contamination possible.

---

## Implementation Order

```
Phase 1: Establish clean baseline (week 1)
  - Go back to pure CoT v24.8 architecture
  - All five loop-alive fixes active
  - Fresh perceiver, warm atoms/hypernetwork
  - Target: match 17% on GSM8K with clean extraction
  - Fill curriculum gap (L4.5, L4.7, L4.9)

Phase 2: Build SymPy decoder (week 2)
  - Implement SymPyVocab (~50 tokens)
  - Implement SymPyDecoder (2-layer transformer, 256 dim)
  - Generate SymPy annotations for L3/L4 problems
  - Train decoder on L3 with teacher forcing
  - Verify: decoder correctly formulates 80%+ of L3 problems from pages
  
Phase 3: Integrate on GSM8K (week 3-4)
  - Generate SymPy annotations for GSM8K (use Claude/GPT-4)
  - Train full pipeline: comprehension → formulation → computation
  - Progressive teacher forcing schedule
  - Target: >25% on GSM8K (arithmetic no longer bottleneck)

Phase 4: Advanced features (week 5-8)
  - Enable pattern memory (accumulate verified templates)
  - MCMC/tree search at inference (retry bad formulations)
  - Unfreeze Llama at 1e-7 if comprehension is bottleneck
  - MATH-500 evaluation
  - Target: >35% GSM8K, meaningful MATH-500 score
```

---

## What NOT to Do

```
- Do NOT share vocabulary between SymPy decoder and LLM.
  The LLM has 128K tokens. The decoder has ~50. They're different systems.
  Sharing causes contamination (the original problem).

- Do NOT train the LLM generation head on SymPy expressions.
  The LLM generation path is ONLY for final English output.
  All SymPy formulation goes through the dedicated decoder.

- Do NOT skip teacher forcing in early epochs.
  The decoder needs to learn the format before generating independently.
  100% teacher forcing for epochs 1-5 minimum.

- Do NOT make the decoder too large (>10M params).
  The vocabulary is tiny (~50 tokens). The expressions are short (<40 tokens).
  A 2-layer, 256-dim transformer is sufficient. Larger = slower + overfitting.

- Do NOT remove the answer head or generation fallback.
  The SymPy decoder might fail (invalid expression, timeout).
  The answer head and LLM generation are safety nets.

- Do NOT forget the five loop-alive fixes:
  □ skip_pass_embed=True
  □ scale_reg=0.1
  □ lam_answer=1.0
  □ skip connections (direct_path)
  □ residual_gate
  
  EVERY training script must include ALL of these. Checklist, not optional.
```

---

## The Vision

```
The breathing loop COMPREHENDS (builds understanding across cycles)
The SymPy decoder FORMULATES (translates understanding to math)
SymPy COMPUTES (executes math exactly)
The page CARRIES (compressed understanding + verified results)

The model does what language models are good at: understanding language.
SymPy does what computers are good at: computing arithmetic.
The 64-float page is the universal interface between them.

Each cycle adds one verified step. Pages accumulate certainty.
The model builds the solution incrementally, one exact computation at a time.
```
