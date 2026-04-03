# Breathing Models — Integrated Thinking Architecture Handoff

## One-Sentence Summary

Insert a 7-layer Perceiver compression engine (108M params) AFTER the final transformer layer, reading from ALL 16 layers with pass-conditioned attention. The model thinks in a loop — each pass processes through Llama's 16 layers, the Perceiver reads all layers and squeezes to a TIGHT 64-float state vector, accumulated via residual connection. 108M parameters deciding what goes on a 64-float sticky note. A confidence head decides when to stop thinking. Only the final pass generates text.

---

## The Architecture

```
                    ┌──────────────────────────────────────────────────────┐
                    │                                                      │
                    ▼                                                      │
[problem tokens] + [4 state pseudo-tokens]                                │
        │                                                                 │
        ▼                                                                 │
   Layer 1 → Layer 2 → ... → Layer 16                                     │
        │         │                │                                      │
        │         │    (all layers saved)                                 │
        ▼         ▼                ▼                                      │
   ┌──────────────────────────────────┐                                   │
   │  7-LAYER PERCEIVER (108M params) │                                   │
   │                                  │                                   │
   │  Pass-conditioned layer gate:    │                                   │
   │  "Which Llama layers matter      │                                   │
   │   for THIS thinking pass?"       │                                   │
   │                                  │                                   │
   │  7× (cross-attn + self-attn      │                                   │
   │      + FFN)                      │                                   │
   │                                  │                                   │
   │  4 queries → 64 floats           │ ← TIGHT BOTTLENECK               │
   └──────────────────────────────────┘                                   │
        │                                                                 │
        ▼                                                                 │
   state = state + alpha * compressed   ← RESIDUAL (not replace)         │
        │                                                                 │
        ├──→ ConfidenceHead(state) → ready?                               │
        │         │                                                       │
        │     NO  │  YES                                                  │
        │         │                                                       │
        │         ▼                                                       │
        │    GENERATE ANSWER (final pass, text output)                    │
        │                                                                 │
        └─────────────────────────────────────────────────────────────────┘
              (loop back: inject state, run transformer again)
```

### Why 64 Floats (Not 32, Not 512)

```
32 floats:   too tight for word problems (can encode a value but not its context)
64 floats:   sweet spot — "24 AND it's May sales AND I need the total next"
             Values + context + intent. Still forces 3-5 passes for multi-step problems.
128 floats:  too loose — model might encode full solution in 2 passes
512 floats:  no compression pressure at all

64 floats = 2,048 bits
A GSM8K step needs ~47 meaningful bits (number + operation + context)
64 floats gives enough precision but forces incremental thinking
```

### Why 108M Params for the Perceiver

The asymmetry is the point: massive capacity to DECIDE (108M params), tiny capacity to STORE (64 floats). The perceiver thinks HARD about what matters before squeezing through the straw. Like a brilliant editor who can only write a one-sentence summary — the quality of that sentence depends on how good the editor is, not how long the sentence is.

### The All-Layer Reading

The perceiver doesn't just read Llama's final layer. It reads ALL 16 layers with learned, pass-conditioned importance:

```
Pass 1 (parsing):   layer gate focuses on layers 1-8 (basic features, number extraction)
Pass 3 (reasoning): layer gate focuses on layers 8-12 (relationships, structure)  
Pass 5 (solving):   layer gate focuses on layers 12-16 (answer-oriented features)
```

The perceiver learns which layers contain what it needs for each stage of thinking. Early passes parse. Later passes reason. The layer gate creates this naturally.

### Bottleneck Ramp (Later, After Core Result)

```
Stage 1:  64 floats,  4 queries   (must think incrementally) ← START HERE
Stage 2:  128 floats, 8 queries   (more per pass)
Stage 3:  256 floats, 16 queries  (efficient compression)
Stage 4:  512 floats, 32 queries  (full capacity, adaptive passes)

Each stage: prove it works, then loosen. Don't loosen until accuracy plateaus.
```

### What Happens Each Pass

```
Pass 1:  [zeros state] + [problem] → 16 layers → perceiver reads all layers → state += delta
         State encodes: "48 clips in April, half as many in May, need total"
         Confidence: 0.15 (parsed the problem, haven't computed anything)
         
Pass 2:  [state] + [problem] → 16 layers → perceiver reads all layers → state += delta
         State encodes: "May = 24 clips (half of 48), need April + May"
         Confidence: 0.4 (computed one step, know what's next)
         
Pass 3:  [state] + [problem] → 16 layers → perceiver reads all layers → state += delta
         State encodes: "48 + 24 = 72 total clips"
         Confidence: 0.85 (have the answer!)
         
Pass 4:  [state] + [problem] → 16 layers → confident!
         → GENERATE: "Natalia sold 48 + 24 = 72 clips. \boxed{72}"
```

Each pass sips through the 64-float straw. Value + context + intent per pass. The 7-layer perceiver reads all Llama layers to decide what's most important for THIS pass, then squeezes it into 64 floats. The residual state accumulates understanding across passes.

---

## Why This Is Better Than Everything Before

### vs External Compression Loop (our v16 approach)
```
Before: generate 512 tokens → compress text → generate 512 tokens → ...
        Slow: autoregressive generation at every breath
        Lossy: text compression loses information
        
Now:    forward pass (no generation) → compress hidden states → loop
        Fast: forward passes without generation are cheap
        Rich: compresses the model's full internal representation, not text
        10 thinking passes ≈ cost of generating ~200 tokens
```

### vs Hourglass Bottleneck (our v4 approach)
```
Before: bottleneck IN THE MIDDLE of layers → distribution mismatch
        Layers after bottleneck received out-of-distribution input
        
Now:    bottleneck AFTER all layers → no distribution mismatch
        All 16 layers run normally. Compression happens after the model
        has built its best representation. Only the state injection at the
        INPUT is non-standard (pseudo-tokens).
```

### vs Text-Based Breathing (our v10-v15 approach)
```
Before: [EXPAND] tags, [COLLAPSE] tags, token budgets, countdown markers
        Model had to learn a format AND learn to compress
        Text is discrete — no gradients through compression
        
Now:    No tags. No format. No text compression.
        Compression is continuous (512 floats). Gradients flow through.
        The model just does forward passes. The architecture handles everything.
```

---

## The Five Components

### 1. Transformer: Llama 3.2 1B-Instruct (THE THINKER)

```
Model:          meta-llama/Llama-3.2-1B-Instruct (via unsloth)
Parameters:     1.23B
Hidden dim:     2048
Layers:         16
Context:        128K tokens
Chat template:  Required (use tokenizer.apply_chat_template)

Why Instruct: Gets ~35% on GSM8K single-shot with CoT prompting.
              Strong gradient signal from day one.
              Base model gets 4% — insufficient signal.
```

### 2. AllLayerPerceiver: 7 Layers, 108M Params → 64 Floats (THE COMPRESSOR)

```python
class AllLayerPerceiver(nn.Module):
    def __init__(self, 
                 num_transformer_layers=16,
                 d_transformer=2048,
                 d_perceiver=1024,
                 num_queries=4,
                 num_perceiver_layers=7,
                 state_size=64,
                 max_passes=20):
        super().__init__()
        
        # Learned queries (4 queries, each in perceiver space)
        self.queries = nn.Parameter(torch.randn(num_queries, d_perceiver))
        self.pass_embed = nn.Embedding(max_passes, d_perceiver)
        
        # Pass-conditioned layer gate
        # Learns which of Llama's 16 layers to focus on per pass
        self.layer_gate = nn.Sequential(
            nn.Linear(d_perceiver, 64),
            nn.ReLU(),
            nn.Linear(64, num_transformer_layers),
            nn.Softmax(dim=-1),
        )
        
        # Project from Llama's space to perceiver's internal space
        self.input_project = nn.Linear(d_transformer, d_perceiver)  # 2048 → 1024
        
        # 7-layer perceiver stack
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(
                    d_perceiver, num_heads=8,
                    kdim=d_perceiver, vdim=d_perceiver,
                    batch_first=True,
                ),
                'cross_norm': nn.LayerNorm(d_perceiver),
                'self_attn': nn.MultiheadAttention(
                    d_perceiver, num_heads=8, batch_first=True,
                ),
                'self_norm': nn.LayerNorm(d_perceiver),
                'ffn': nn.Sequential(
                    nn.Linear(d_perceiver, d_perceiver * 4),
                    nn.GELU(),
                    nn.Linear(d_perceiver * 4, d_perceiver),
                ),
                'ffn_norm': nn.LayerNorm(d_perceiver),
            })
            for _ in range(num_perceiver_layers)
        ])
        
        # Final tight projection: 1024 → 16 per query → 64 total
        self.project_out = nn.Linear(d_perceiver, state_size // num_queries)
    
    def forward(self, all_layer_hidden_states, pass_num):
        """
        all_layer_hidden_states: list of 16 tensors, each (batch, seq_len, 2048)
        pass_num: which thinking pass
        returns: (batch, 64) state delta
        """
        batch_size = all_layer_hidden_states[0].size(0)
        
        # Pass-conditioned queries
        pass_context = self.pass_embed(torch.tensor(pass_num, device=self.queries.device))
        queries = (self.queries + pass_context).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Pass-conditioned layer importance
        layer_weights = self.layer_gate(pass_context)  # (16,) softmax weights
        
        # Weighted combination of ALL transformer layers
        stacked = torch.stack(all_layer_hidden_states, dim=0)  # (16, batch, seq, 2048)
        weights = layer_weights.view(16, 1, 1, 1)
        combined = (stacked * weights).sum(dim=0)  # (batch, seq, 2048)
        
        # Project to perceiver dimension
        kv = self.input_project(combined)  # (batch, seq, 1024)
        
        # 7 layers of deep compression processing
        for layer in self.layers:
            # Cross-attend: queries read from transformer representations
            attended, _ = layer['cross_attn'](query=queries, key=kv, value=kv)
            queries = layer['cross_norm'](queries + attended)
            
            # Self-attend: queries coordinate with each other
            refined, _ = layer['self_attn'](query=queries, key=queries, value=queries)
            queries = layer['self_norm'](queries + refined)
            
            # FFN: nonlinear processing
            queries = layer['ffn_norm'](queries + layer['ffn'](queries))
        
        # Project to tight bottleneck
        state_delta = self.project_out(queries)  # (batch, 4, 16)
        return state_delta.flatten(start_dim=1)  # (batch, 64)
```

Parameter breakdown:
```
Queries + embeddings:            ~24K
Layer gate:                      ~66K
Input projection (2048→1024):    ~2.1M
7 perceiver layers:
  cross_attn (1024, 8 heads) × 7:  ~22.0M
  self_attn (1024, 8 heads) × 7:   ~22.0M
  FFN (1024→4096→1024) × 7:        ~58.7M
  LayerNorms × 7:                   ~43K
Output projection (1024→16):     ~16K
──────────────────────────────────────
Total:                           ~105M
```

### 3. StateInjector: 64 Floats → 4 Pseudo-Tokens (THE NOTEBOOK READER)

```python
class StateInjector(nn.Module):
    def __init__(self, state_size=64, d_model=2048, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.chunk_size = state_size // num_tokens  # 16
        self.project = nn.Linear(self.chunk_size, d_model)  # 16 → 2048
        self.norm = nn.LayerNorm(d_model)
        self.position_embed = nn.Parameter(torch.randn(num_tokens, d_model))

    def forward(self, state_vector):
        """
        state_vector: (batch, 64)
        returns: (batch, 4, 2048) pseudo-tokens to prepend to input
        """
        chunks = state_vector.reshape(-1, self.num_tokens, self.chunk_size)
        tokens = self.project(chunks) + self.position_embed
        return self.norm(tokens)
```

4 pseudo-tokens — minimal footprint. Each encodes 16 floats projected up to 2048 dimensions. The expansion (16 → 2048) means no information is lost at injection — the bottleneck is in the 64 floats, not the projection.

### 4. ConfidenceHead: 64 Floats → Scalar (THE READINESS JUDGE)

```python
class ConfidenceHead(nn.Module):
    def __init__(self, state_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, state_vector):
        """
        state_vector: (batch, 64)
        returns: (batch, 1) confidence between 0 and 1
        """
        return self.net(state_vector)
```

Tiny: ~2.1K parameters. Reads the 64-float state and judges: "do I have enough information to answer?"

### 5. The Residual State Update (THE MEMORY)

```python
# NOT this (replace):
state = compressor(hidden)

# THIS (accumulate):
state = state + alpha * compressor(hidden)
```

`alpha` is a learnable scalar (initialized to 0.1). The state ACCUMULATES across passes. Each pass ADDS to what's known, doesn't overwrite. Gradients flow through the addition directly — no vanishing gradient through the residual connection.

---

## The Full Model

```python
class ThinkingModel(nn.Module):
    def __init__(self, model_name="unsloth/Llama-3.2-1B-Instruct"):
        super().__init__()
        self.transformer = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.compressor = AllLayerPerceiver(
            num_transformer_layers=16,
            d_transformer=2048,
            d_perceiver=1024,
            num_queries=4,
            num_perceiver_layers=7,
            state_size=64,
        )
        self.injector = StateInjector(state_size=64, d_model=2048, num_tokens=4)
        self.confidence = ConfidenceHead(state_size=64)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # learnable residual weight
        self.state_size = 64

    def think(self, problem_text, max_passes=10, confidence_threshold=0.8):
        """
        Think about a problem in multiple passes.
        Each pass: forward through ALL 16 transformer layers,
        Perceiver reads ALL layers with pass-conditioned attention,
        compresses to 64 floats, accumulates into residual state.
        """
        # Tokenize problem with chat template
        messages = [{"role": "user", "content": problem_text}]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        prompt_ids = torch.tensor([prompt], device=self.device)
        prompt_embeds = self.transformer.get_input_embeddings()(prompt_ids)

        state = torch.zeros(1, self.state_size, device=self.device)
        all_states = [state]
        confidences = []

        for pass_num in range(max_passes):
            # Inject state as pseudo-tokens
            state_tokens = self.injector(state)  # (1, 4, 2048)

            # Concatenate: [state_tokens, problem_embeds]
            input_embeds = torch.cat([state_tokens, prompt_embeds], dim=1)

            # Forward through ALL 16 layers, collecting hidden states from each
            outputs = self.transformer(
                inputs_embeds=input_embeds,
                output_hidden_states=True,
            )
            # outputs.hidden_states is a tuple of 17 tensors (embedding + 16 layers)
            all_layer_hidden = list(outputs.hidden_states[1:])  # skip embedding, keep 16 layers

            # Perceiver reads ALL layers, compresses to 64 floats
            state_delta = self.compressor(all_layer_hidden, pass_num)  # (1, 64)

            # Residual update: accumulate, don't replace
            state = state + self.alpha * state_delta
            all_states.append(state)

            # Check confidence
            conf = self.confidence(state)  # (1, 1)
            confidences.append(conf.item())

            if conf.item() > confidence_threshold:
                break

        return state, all_states, confidences

    def generate_answer(self, problem_text, state):
        """
        Final pass: generate text answer using accumulated state.
        """
        messages = [{"role": "user", "content": problem_text}]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        prompt_ids = torch.tensor([prompt], device=self.device)
        prompt_embeds = self.transformer.get_input_embeddings()(prompt_ids)

        state_tokens = self.injector(state)
        input_embeds = torch.cat([state_tokens, prompt_embeds], dim=1)

        output = self.transformer.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=512,
            do_sample=False,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def solve(self, problem_text, max_passes=10):
        """
        Full pipeline: think → generate answer.
        """
        state, all_states, confidences = self.think(problem_text, max_passes)
        answer = self.generate_answer(problem_text, state)
        return {
            'answer': answer,
            'num_passes': len(confidences),
            'confidences': confidences,
            'final_state_norm': state.norm().item(),
        }
```

---

## Training

### The Training Loop

```python
def train(model, problems, optimizer, max_passes=10):
    for problem, gold_answer in problems:
        optimizer.zero_grad()

        # Think (no text generation, just forward passes + compression)
        state, all_states, confidences = model.think(problem, max_passes)

        # Generate answer from final state
        # Teacher-forced: compute loss against gold answer tokens
        messages = [{"role": "user", "content": problem}]
        prompt = model.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        prompt_ids = torch.tensor([prompt], device=device)
        prompt_embeds = model.transformer.get_input_embeddings()(prompt_ids)

        state_tokens = model.injector(state)
        input_embeds = torch.cat([state_tokens, prompt_embeds], dim=1)

        answer_ids = model.tokenizer.encode(gold_answer, return_tensors="pt").to(device)
        full_ids = torch.cat([prompt_ids, answer_ids], dim=1)

        outputs = model.transformer(
            inputs_embeds=input_embeds,
            labels=full_ids,
        )
        answer_loss = outputs.loss

        # Confidence loss: confidence should be high when answer would be correct
        # and low when more thinking is needed
        # Use the auxiliary prediction at intermediate states
        confidence_loss = 0.0
        for i, (s, c) in enumerate(zip(all_states[1:], confidences)):
            # Could the model answer correctly from this state?
            # Approximate: confidence should increase over passes
            target_conf = min(i / max_passes + 0.1, 0.95)
            confidence_loss += F.mse_loss(torch.tensor(c), torch.tensor(target_conf))
        confidence_loss /= len(confidences)

        # Efficiency bonus: fewer passes for correct answers
        num_passes = len(confidences)
        efficiency = 0.01 * num_passes  # small penalty for many passes

        # Total loss
        loss = answer_loss + 0.1 * confidence_loss + efficiency
        loss.backward()

        # Gradient clipping (important — large norms observed)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
```

### Deep Supervision (Solves Vanishing Gradient)

At every thinking pass, try to predict the answer from the CURRENT state. This gives every pass direct gradient, not just the last one:

```python
def train_with_deep_supervision(model, problem, gold_answer):
    state, all_states, confidences = model.think(problem, max_passes=10)
    
    total_loss = 0.0
    for i, intermediate_state in enumerate(all_states[1:]):  # skip zeros state
        # Each intermediate state tries to predict the answer
        state_tokens = model.injector(intermediate_state)
        input_embeds = torch.cat([state_tokens, prompt_embeds], dim=1)
        outputs = model.transformer(inputs_embeds=input_embeds, labels=answer_ids)
        
        # Weight increases with pass number (later passes should be better)
        weight = (i + 1) / len(all_states)
        total_loss += weight * outputs.loss
    
    total_loss.backward()
```

Every pass gets gradient. Earlier passes get lighter weight (they're expected to be worse). Later passes get heavier weight (they should be close to the answer). No vanishing gradient because each pass has its own direct loss.

### Training Phases

```
Phase 1: Freeze transformer, train compression head + injector + confidence (5-10 epochs)
  The transformer runs normally (it already gets 35% on GSM8K)
  The compression head learns to extract useful state
  The confidence head learns when the state is sufficient
  Low risk: can't hurt the transformer's existing capability

Phase 2: Unfreeze transformer, end-to-end (10-20 epochs)
  Now the transformer adapts to USE the state information
  The compression head adapts to what the transformer NEEDS
  Co-evolution: better thinking ↔ better compression

Start with max_passes=3 (minimal vanishing gradient)
Increase to 5, then 10 as training progresses
```

### State Scale Warmup (Same Proven Schedule)

```
Epoch 1-2:   scale = 0.1
Epoch 3-4:   scale = 0.3
Epoch 5-6:   scale = 0.5
Epoch 7-8:   scale = 0.7
Epoch 9-10:  scale = 1.0
```

---

## Data

```
Training:   GSM8K train split (~7,473 problems + gold answers)
            Use Instruct chat template for prompting
            CoT format for generating training signal
            
Eval:       GSM8K test split (1,319 problems)

Never:      MATH-500 (final evaluation only)
```

No oracle collapses. No THINK/COMPUTE pairs. No breath-structured data. Just problems and answers. The architecture handles the rest.

---

## What to Monitor

### Per Epoch
```
thinking_accuracy:    accuracy after N thinking passes + generation
single_shot_accuracy: same model, no thinking passes, just generate
zeros_accuracy:       thinking passes with zeros state (ablation)
avg_passes:           how many passes before confident (should decrease for easy problems)
avg_confidence:       confidence trend over training
```

### Per Pass (Deep Supervision Diagnostics)
```
pass_1_accuracy:  can the model answer from just 1 thinking pass?
pass_3_accuracy:  from 3 passes?
pass_5_accuracy:  from 5 passes?
pass_N_accuracy:  from N passes?

The accuracy should INCREASE with passes.
If it doesn't, the state isn't accumulating useful information.
```

### State Health
```
state_norm:      should grow steadily over passes (accumulating)
state_delta_norm: should be stable (each pass contributes similar amount)
alpha:           the learned residual weight (should be positive, moderate)
grad_norm:       should be nonzero at all passes (deep supervision ensures this)
```

### The Key Plot
```
X-axis: number of thinking passes
Y-axis: accuracy

Should show: accuracy INCREASES with more passes, then plateaus.
Easy problems: plateau early (2-3 passes)
Hard problems: plateau late (8-10 passes)
The confidence head should learn to match this curve.
```

---

## Validation Checkpoints

```
V1: Forward pass works with pseudo-tokens
    Model generates coherent text with 8 state pseudo-tokens prepended
    Timeline: step 1

V2: Gradients flow through all thinking passes
    grad_norm > 0 at compression head for ALL passes (not just last)
    Timeline: step 1 (deep supervision guarantees this)

V3: State accumulates meaningfully
    state_norm increases with passes (information accumulating)
    pass_3_accuracy > pass_1_accuracy (more thinking = better)
    Timeline: epoch 3-5

V4: Thinking matches single-shot
    thinking_accuracy ≈ single_shot_accuracy
    Compression isn't hurting (neutral)
    Timeline: epoch 5-8

V5: Thinking BEATS single-shot  ← THE RESULT
    thinking_accuracy > single_shot_accuracy
    The model solves problems it can't solve in one pass
    Timeline: epoch 8-15

V6: Confidence head works
    Easy problems: 2-3 passes, high confidence
    Hard problems: 8-10 passes, lower confidence
    Timeline: epoch 10-15

V7: Thinking accuracy > 45% on GSM8K
    (baseline is 35% single-shot)
    Timeline: epoch 15-20
```

---

## Parameter Count

```
Llama 3.2 1B-Instruct:  1.23B     (frozen Phase 1, fine-tuned Phase 2)
7-Layer Perceiver:       ~105M    (all-layer reading, pass-conditioned, 7× cross+self+FFN)
StateInjector:           ~0.13M   (projection + position embeds)
ConfidenceHead:          ~2.1K    (two small linear layers)
Alpha:                   1        (single learnable scalar)
────────────────────────────────
Total:                   ~1.34B
New parameters:          ~105M    (8% of total model)
Bottleneck:              64 floats
```

The perceiver is a SERIOUS compression engine. 105M parameters — nearly as many as the entire SmolLM2-135M model — dedicated to deciding what goes into 64 floats. The asymmetry is the architecture: 105M params of decision-making, 64 floats of storage.

---

## Why This Will Work

```
1. Tight bottleneck: 64 floats forces incremental thinking — model can't solve in one pass
2. Massive compressor: 105M param perceiver makes SMART decisions about those 64 floats
3. All-layer reading: perceiver accesses every level of Llama's representations
4. Pass-conditioned: perceiver focuses on different layers for different thinking stages
5. Strong baseline: Instruct model gets 35% on GSM8K (gradient signal exists)
6. Proven concept: tight bottleneck → 80.4% two-step (SmolLM2), 83% (Llama)
7. Residual state: gradients flow through addition (no vanishing gradient)
8. Deep supervision: every pass gets direct gradient (no distant signal)
9. Confidence head: adaptive compute (easy = few passes, hard = many passes)
10. Thinking is cheap: forward passes without generation cost ~1/50th of generation
11. Clean experiment: same model, same prompt, just with/without thinking passes
```

---

## Infrastructure

```
Model:          unsloth/Llama-3.2-1B-Instruct (bf16)
Compute:        AWS EC2 g5.xlarge (A10G 24GB)
Memory:         ~2.5GB model + ~1MB new params + activations for N passes
                With gradient checkpointing: 10 passes easily fits in 24GB
                32-float state = negligible memory overhead
Training:       HF Transformers + custom training loop
Time:           Phase 1 ~1-2 days, Phase 2 ~2-3 days
Cost:           ~$75-150 on AWS spot
```

### Files to Create
```
src/
  thinking_model.py         # ThinkingModel (the full architecture above)
  compression_head.py       # UPDATE: pass_embed instead of breath_embed
  state_injector.py         # Same as before
  confidence_head.py        # NEW: readiness judge

scripts/
  train_thinking.py         # Training with deep supervision
  eval_thinking.py          # Accuracy vs passes, ablation, confidence analysis
  visualize_thinking.py     # Plot accuracy vs num_passes curve

configs/
  thinking_gsm8k.yaml       # Hyperparameters
```

---

## What NOT to Do

```
- Do NOT start with 512-float state. Start at 64. The tight bottleneck forces incremental thinking.
- Do NOT increase bottleneck size until accuracy plateaus at current size.
- Do NOT use a small compression head. The 7-layer perceiver needs ~105M params to make smart decisions.
- Do NOT read from only the last transformer layer. The perceiver reads ALL 16 layers.
- Do NOT generate text during thinking passes. Forward pass only. Fast.
- Do NOT replace state. Accumulate with residual: state = state + alpha * delta.
- Do NOT skip deep supervision. Every pass needs direct gradient.
- Do NOT start with 10 passes. Start with 3, increase gradually.
- Do NOT use the base model. Instruct gets 35% — we need that gradient signal.
- Do NOT skip state scale warmup. Pseudo-tokens are still out-of-distribution.
- Do NOT forget gradient clipping (max_norm=1.0). Large grad norms observed.
- Do NOT train the transformer before the perceiver works (Phase 1 first).
```

---

## The Philosophical Core

This architecture implements something profound: the model THINKS before it SPEAKS.

Current LLMs generate tokens immediately. Each token is produced in one forward pass. The model gets one shot at each word.

This architecture decouples thinking from speaking. The model can process the problem 10 times internally — building up understanding, accumulating compressed insights — before producing a single token of output. When it finally speaks, it speaks from a place of accumulated understanding.

The number of thinking passes adapts to problem difficulty. Simple problems: think once or twice, answer quickly. Hard problems: think many times, answer carefully. The model allocates compute proportional to difficulty — not because we told it to, but because the confidence head learned when more thinking helps.

---

## Success Criterion

One number: **thinking accuracy > 35% on GSM8K.**

35% is the single-shot baseline. If thinking passes push accuracy above 35%, the compression carried useful information across passes and the model solved problems it couldn't solve in one forward pass.

That's the result. Everything else follows.

---

## MATH-500 Deadline: April 22, 2026

GSM8K first (prove thinking helps) → MATH (prove it scales to harder problems).
