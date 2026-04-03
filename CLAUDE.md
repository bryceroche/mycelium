# Mycelium v18 — Integrated Thinking (Llama 3.2 1B-Instruct)

A transformer that THINKS before it SPEAKS. Insert a 7-layer Perceiver compression engine (108M params) AFTER the final transformer layer, reading from ALL 16 layers with pass-conditioned attention. The model thinks in a loop — each pass processes through Llama's 16 layers, the Perceiver reads all layers and squeezes to a TIGHT 64-float state vector, accumulated via residual connection. 108M parameters deciding what goes on a 64-float sticky note.

> The model gets multiple forward passes to understand before producing a single token.
> Think hard, compress tight, speak from understanding.

---

Only train on the AWS VM — PEM key in ~/.ssh/mycelium-key.pem
Limit timeouts to less than 2 minutes
Always use vLLM + stage-based batching when possible, otherwise HF Transformers
Temperature = 0 (greedy) for all evals
MATH-500 NEVER in training data

---

## Current State (April 3, 2026)

```
Architecture:        Llama 3.2 1B-Instruct + 7-Layer AllLayerPerceiver (108M) + StateInjector + ConfidenceHead
Context window:      128K tokens
Compression:         64 floats (TIGHT bottleneck, residual accumulation)
Status:              ARCHITECTURE PIVOT — Integrated Thinking replaces external compression loop

PIVOT RATIONALE:
  External compression loop: generate 512 tokens → compress text → loop
    Slow: autoregressive generation at every breath
    Lossy: text compression loses information

  Integrated Thinking: forward pass (no generation) → compress hidden states → loop
    Fast: forward passes without generation are cheap (10 passes ≈ 200 tokens)
    Rich: compresses full internal representation, not text
    All-layer: Perceiver reads ALL 16 transformer layers

ARCHITECTURE:
  AllLayerPerceiver:  7 layers, 108M params, reads all 16 Llama layers
                      Pass-conditioned attention (which layers matter NOW?)
                      4 queries → 64 floats (TIGHT)
  StateInjector:      64 floats → 4 pseudo-tokens (simple projection)
  ConfidenceHead:     64 floats → scalar (when to stop thinking?)
  ResidualState:      state = state + alpha * delta (accumulate, don't replace)
  d_model:            2048 (Llama), 1024 (Perceiver internal)

WHY INSTRUCT (not BASE):
  Base model:    4% GSM8K single-shot — insufficient gradient signal
  Instruct:      35% GSM8K single-shot — strong signal from day one
  We need the model to already understand math — we're teaching it to THINK harder

SMOLLM2 PROOF-OF-CONCEPT (COMPLETE):
  Two-step arithmetic:   0% → 80.4% (32 floats, 4 queries)
  Three-step arithmetic: 0% → 52%   (64 floats, 4 queries)
  Ablation delta:        +90 points (real state vs zeros)

LLAMA TWO-STEP (COMPLETE):
  Two-step arithmetic:   0% → 83%
  Proves: tight bottleneck works at scale

Core principle:      Thinking happens in PASSES, not text.
                     The model processes the problem multiple times
                     internally before producing ANY output tokens.
                     Each pass compresses insights into 64 floats.
                     When confident, generate answer from accumulated state.

Target:              GSM8K thinking accuracy > 35% (single-shot baseline)
```

---

## The Thinking Cycle (NOT Breathing)

Each pass is one forward pass with state injection — NO text generation:

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

Critical architecture:
- **State vector**: 64 floats (TIGHT) — forces incremental thinking
- **Pseudo-tokens**: 4 learned embeddings prepended to input
- **All-layer reading**: Perceiver sees all 16 transformer layers, not just last
- **Pass-conditioned**: Layer gate focuses on different layers per pass
- **Residual accumulation**: state += alpha * delta (gradients flow through addition)
- **Confidence head**: Decides when to stop thinking and generate
- **No generation during thinking**: Forward passes only, FAST

The model loops until confident or max passes, then generates answer from final state.

---

## Why 64 Floats (The Tight Bottleneck)

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

The asymmetry is the point: **108M params to DECIDE, 64 floats to STORE**. Like a brilliant editor who can only write a one-sentence summary — the quality depends on how good the editor is, not how long the sentence is.

---

## Architecture Components

### AllLayerPerceiver (~108M params)
7-layer perceiver that reads ALL 16 Llama layers with pass-conditioned attention:

```python
# src/all_layer_perceiver.py
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

        # Learned queries (4 queries, perceiver space)
        self.queries = nn.Parameter(torch.randn(num_queries, d_perceiver))
        self.pass_embed = nn.Embedding(max_passes, d_perceiver)

        # Pass-conditioned layer gate — which of Llama's 16 layers to focus on
        self.layer_gate = nn.Sequential(
            nn.Linear(d_perceiver, 64),
            nn.ReLU(),
            nn.Linear(64, num_transformer_layers),
            nn.Softmax(dim=-1),
        )

        # Project from Llama (2048) to perceiver (1024)
        self.input_project = nn.Linear(d_transformer, d_perceiver)

        # 7-layer perceiver stack
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(d_perceiver, num_heads=8, batch_first=True),
                'cross_norm': nn.LayerNorm(d_perceiver),
                'self_attn': nn.MultiheadAttention(d_perceiver, num_heads=8, batch_first=True),
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

        # Final projection: 1024 → 16 per query → 64 total
        self.project_out = nn.Linear(d_perceiver, state_size // num_queries)

    def forward(self, all_layer_hidden_states, pass_num):
        # all_layer_hidden_states: list of 16 tensors, each (batch, seq_len, 2048)
        batch_size = all_layer_hidden_states[0].size(0)

        # Pass-conditioned queries
        pass_context = self.pass_embed(torch.tensor(pass_num, device=self.queries.device))
        queries = (self.queries + pass_context).unsqueeze(0).expand(batch_size, -1, -1)

        # Pass-conditioned layer importance
        layer_weights = self.layer_gate(pass_context)  # (16,) softmax

        # Weighted combination of ALL transformer layers
        stacked = torch.stack(all_layer_hidden_states, dim=0)  # (16, batch, seq, 2048)
        combined = (stacked * layer_weights.view(16, 1, 1, 1)).sum(dim=0)  # (batch, seq, 2048)

        # Project to perceiver dimension
        kv = self.input_project(combined)  # (batch, seq, 1024)

        # 7 layers of deep compression processing
        for layer in self.layers:
            attended, _ = layer['cross_attn'](query=queries, key=kv, value=kv)
            queries = layer['cross_norm'](queries + attended)
            refined, _ = layer['self_attn'](query=queries, key=queries, value=queries)
            queries = layer['self_norm'](queries + refined)
            queries = layer['ffn_norm'](queries + layer['ffn'](queries))

        # Project to tight bottleneck
        state_delta = self.project_out(queries)  # (batch, 4, 16)
        return state_delta.flatten(start_dim=1)  # (batch, 64)
```

### What the Layer Gate Learns

```
Pass 1 (parsing):   layer gate focuses on layers 1-8 (basic features, number extraction)
Pass 3 (reasoning): layer gate focuses on layers 8-12 (relationships, structure)
Pass 5 (solving):   layer gate focuses on layers 12-16 (answer-oriented features)
```

The perceiver learns which layers contain what it needs for each stage of thinking.

### StateInjector (~130K params)
Converts 64-float state vector to 4 pseudo-tokens:

```python
# src/state_injector.py
class StateInjector(nn.Module):
    def __init__(self, state_size=64, d_model=2048, num_tokens=4):
        self.project = nn.Linear(state_size // num_tokens, d_model)  # 16 → 2048
        self.position_embed = nn.Parameter(torch.randn(num_tokens, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, state_vector):
        # state_vector: (batch, 64)
        chunks = state_vector.reshape(-1, 4, 16)
        tokens = self.project(chunks) + self.position_embed
        return self.norm(tokens)  # (batch, 4, 2048)
```

### ConfidenceHead (~2K params)
Decides when to stop thinking and generate:

```python
# src/confidence_head.py
class ConfidenceHead(nn.Module):
    def __init__(self, state_size=64):
        self.net = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, state_vector):
        return self.net(state_vector)  # (batch, 1) confidence 0-1
```

### ThinkingModel — Full Loop
```python
# src/thinking_model.py
class ThinkingModel(nn.Module):
    def __init__(self, model_name="unsloth/Llama-3.2-1B-Instruct"):
        self.transformer = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.compressor = AllLayerPerceiver(...)
        self.injector = StateInjector(state_size=64, d_model=2048, num_tokens=4)
        self.confidence = ConfidenceHead(state_size=64)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # learnable residual weight

    def think(self, problem_text, max_passes=10, confidence_threshold=0.8):
        state = torch.zeros(1, 64, device=self.device)

        for pass_num in range(max_passes):
            state_tokens = self.injector(state)  # (1, 4, 2048)
            input_embeds = torch.cat([state_tokens, prompt_embeds], dim=1)

            # Forward through ALL 16 layers, collect hidden states
            outputs = self.transformer(inputs_embeds=input_embeds, output_hidden_states=True)
            all_layer_hidden = list(outputs.hidden_states[1:])  # 16 layers

            # Perceiver reads all layers, compresses to 64 floats
            state_delta = self.compressor(all_layer_hidden, pass_num)

            # Residual update: accumulate, don't replace
            state = state + self.alpha * state_delta

            if self.confidence(state).item() > confidence_threshold:
                break

        return state

    def generate_answer(self, problem_text, state):
        # Final pass: generate text from accumulated state
        state_tokens = self.injector(state)
        input_embeds = torch.cat([state_tokens, prompt_embeds], dim=1)
        return self.transformer.generate(inputs_embeds=input_embeds, max_new_tokens=512)
```

### Parameter Count
```
Llama 3.2 1B-Instruct:  1.23B   (frozen Phase 1, fine-tuned Phase 2)
AllLayerPerceiver:      ~108M   (7 layers, all-layer reading, pass-conditioned)
StateInjector:          ~130K   (4 tokens × 2048 d_model)
ConfidenceHead:         ~2K     (tiny MLP)
Alpha:                  1       (learnable scalar)
────────────────────────────────
Total:                  ~1.34B
New parameters:         ~108M   (8% of total model)
Bottleneck:             64 floats
```

---

## Two-Phase Training

### Phase 1: Freeze Transformer, Train Compression
```
Trainable:  AllLayerPerceiver + StateInjector + ConfidenceHead + alpha
Frozen:     Transformer (already gets 35% on GSM8K)
Duration:   5-10 epochs

The transformer runs normally (it already understands math).
The perceiver learns to extract useful state from all layers.
The confidence head learns when the state is sufficient.
Low risk: can't hurt the transformer's existing capability.
```

### Phase 2: End-to-End
```
Trainable:  Everything
Duration:   10-20 epochs

Now the transformer adapts to USE the state information.
The perceiver adapts to what the transformer NEEDS.
Co-evolution: better thinking ↔ better compression.

Start with max_passes=3, increase to 5, then 10.
```

### Deep Supervision (Solves Vanishing Gradient)
Every thinking pass tries to predict the answer from the CURRENT state:

```python
def train_with_deep_supervision(model, problem, gold_answer):
    state, all_states = model.think(problem, max_passes=10)

    total_loss = 0.0
    for i, intermediate_state in enumerate(all_states[1:]):
        state_tokens = model.injector(intermediate_state)
        outputs = model.transformer(inputs_embeds=..., labels=answer_ids)
        weight = (i + 1) / len(all_states)  # later passes weighted more
        total_loss += weight * outputs.loss

    total_loss.backward()
```

Every pass gets gradient. No vanishing gradient through the residual connection.

### State Scale Warmup (Proven Schedule)
```
Epoch 1-2:   scale = 0.1
Epoch 3-4:   scale = 0.3
Epoch 5-6:   scale = 0.5
Epoch 7-8:   scale = 0.7
Epoch 9-10:  scale = 1.0
```

---

## Why This Is Better Than Before

### vs External Compression Loop (v16)
```
Before: generate 512 tokens → compress text → generate 512 tokens → ...
        Slow: autoregressive generation at every breath
        Lossy: text compression loses information

Now:    forward pass (no generation) → compress hidden states → loop
        Fast: forward passes without generation are cheap
        Rich: compresses full internal representation, not text
        10 thinking passes ≈ cost of generating ~200 tokens
```

### vs Hourglass Bottleneck (v4)
```
Before: bottleneck IN THE MIDDLE of layers → distribution mismatch

Now:    bottleneck AFTER all layers → no distribution mismatch
        All 16 layers run normally. Compression happens after.
```

### vs Text-Based Breathing (v10-v15)
```
Before: [EXPAND] tags, [COLLAPSE] tags, token budgets
        Text is discrete — no gradients through compression

Now:    No tags. No format. Compression is continuous.
        Gradients flow through. Architecture handles everything.
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

No oracle collapses. No THINK/COMPUTE pairs. Just problems and answers.

---

## File Structure

```
src/
  __init__.py              # Package exports
  all_layer_perceiver.py   # 7-layer perceiver reading all transformer layers
  state_injector.py        # State → pseudo-tokens (64 → 4 tokens)
  confidence_head.py       # Readiness judge (when to stop thinking)
  thinking_model.py        # Full thinking loop

scripts/
  train_thinking.py        # Training with deep supervision
  eval_thinking.py         # Accuracy vs passes, ablation
  visualize_thinking.py    # Plot accuracy vs num_passes curve

configs/
  thinking_gsm8k.yaml      # Hyperparameters
```

---

## Infrastructure

### Models
```
ENGINE:
unsloth/Llama-3.2-1B-Instruct — thinking engine
  1.23B params, 16 layers, 2048 hidden dim, 128K context
  35% GSM8K single-shot with CoT (strong gradient signal)
  Use bfloat16

REFERENCE (SmolLM2 proof-of-concept):
HuggingFaceTB/SmolLM2-135M — proved tight bottleneck works
  80.4% two-step, 52% three-step, +90 point ablation delta
```

### VMs
```
AWS EC2 g5.xlarge (~$1/hr, A10G 24GB) — primary training
SSH: ssh -i ~/.ssh/mycelium-key.pem ubuntu@<IP>
```

---

## Milestones

```
PROOF-OF-CONCEPT (COMPLETE):
M0: SmolLM2 baseline: 0% GSM8K, 0% MATH-500
M1: Tight bottleneck works: 80.4% two-step, 52% three-step
M2: Llama two-step: 83% (tight bottleneck scales)

INTEGRATED THINKING (CURRENT):
M3: Build architecture
    AllLayerPerceiver, StateInjector, ConfidenceHead, ThinkingModel

M4: Validation checkpoints
    V1: Forward pass works with pseudo-tokens
    V2: Gradients flow through all passes (deep supervision)
    V3: State accumulates (pass_3 > pass_1 accuracy)
    V4: Thinking ≈ single-shot (neutral)
    V5: Thinking > single-shot (THE RESULT)

M5: GSM8K thinking accuracy > 35%
    Baseline is 35% single-shot. Beat it with thinking passes.

M6: GSM8K thinking accuracy > 45%
    +10 points from thinking.

M7: MATH L1-L3
    512-float state (loosened bottleneck for harder problems)

M8: THE HEADLINE — MATH L4-L5
    Non-zero accuracy where single-shot gets 0%
```

---

## What to Monitor

### Per Epoch
```
thinking_accuracy:    accuracy after N thinking passes + generation
single_shot_accuracy: same model, no thinking, just generate
zeros_accuracy:       thinking with zeros state (ablation)
avg_passes:           how many passes before confident
```

### Per Pass
```
pass_1_accuracy:  can answer from 1 pass?
pass_3_accuracy:  from 3 passes?
pass_5_accuracy:  from 5 passes?
Accuracy should INCREASE with passes.
```

### State Health
```
state_norm:        should grow (accumulating)
state_delta_norm:  should be stable
alpha:             learnable weight (should be positive, moderate)
grad_norm:         nonzero at all passes (deep supervision)
```

### The Key Plot
```
X-axis: number of thinking passes
Y-axis: accuracy

Accuracy should increase with passes, then plateau.
Easy problems: plateau early (2-3 passes)
Hard problems: plateau late (8-10 passes)
```

---

## Critical Rules

```
 1. Llama 3.2 1B-Instruct (NOT base) — need 35% baseline for gradient signal
 2. Use bfloat16 for Llama
 3. Start with 64-float bottleneck — TIGHT (don't loosen until accuracy plateaus)
 4. 7-layer perceiver reads ALL 16 transformer layers
 5. Pass-conditioned layer gate — perceiver learns which layers matter when
 6. Residual state: state = state + alpha * delta (ACCUMULATE, don't replace)
 7. Deep supervision: every pass gets direct gradient
 8. State scale warmup: 0.1 → 1.0 over first 5 epochs
 9. Start with max_passes=3, increase gradually to 10
10. Freeze transformer Phase 1, unfreeze Phase 2
11. NO text generation during thinking — forward passes only, FAST
12. Confidence head decides when to stop (adaptive compute)
13. Ablation (real state vs zeros) at every checkpoint
14. Temperature = 0 for all evaluations
15. N >= 100 problems for any accuracy measurement
16. MATH-500 NEVER in training data
17. Gradient clipping max_norm=1.0
18. Handoff doc before closing any session
```

---

## Bug List

```
CARRIED:
27. max_tokens=512 truncates before \boxed{}. Use 2048+.
28. parse_latex() on non-LaTeX corrupts. Only call if contains backslash.
35. Temperature must be 0 (greedy) for eval.
37. Small sample sizes lie. Always N>=100.

NEW FOR v18:
42. Use chat template for Instruct model — tokenizer.apply_chat_template required
43. output_hidden_states=True to get all 16 layers
44. hidden_states[0] is embedding layer — use hidden_states[1:] for 16 transformer layers
45. Gradient clipping essential — large norms observed with deep supervision
```

---

## Hardware

```
Training:    Llama 3.2 1B (~2.5GB bf16) + ~108M perceiver fits on A10G (24GB)
             With gradient checkpointing: 10 passes easily fits
Phase 1:     ~1-2 days on A10G
Phase 2:     ~2-3 days on A10G
Cost:        ~$75-150 on AWS spot
```

---

## Bottleneck Ramp (After Core Result)

```
Stage 1:  64 floats,  4 queries   (forces incremental thinking) ← START HERE
Stage 2:  128 floats, 8 queries   (more per pass)
Stage 3:  256 floats, 16 queries  (efficient compression)
Stage 4:  512 floats, 32 queries  (full capacity, adaptive passes)

Each stage: prove it works, then loosen. Don't loosen until accuracy plateaus.
```

---

## MATH-500 Deadline: April 22, 2026

GSM8K first (prove thinking helps) → MATH (prove it scales to harder problems).
Target: thinking accuracy > single-shot on L4-L5 problems.
The undeniable result: solving previously unsolvable problems through integrated thinking.
