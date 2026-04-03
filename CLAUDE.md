# Mycelium v19 — Asymmetric Hourglass (Llama 3.2 1B-Instruct)

A transformer that THINKS before it SPEAKS. Sandwich Llama between a lightweight **DECOMPRESSOR** (1.3M params) that projects state into input bias, and a heavy **COMPRESSOR** (120M params) that squeezes all 16 layer hidden states back into 64 floats. The transformer runs PRISTINE — no layer splitting, no architectural surgery.

> The intelligence is in COMPRESSION (what to keep), not DECOMPRESSION (how to project).
> 89x more params deciding what fits through 64 floats than deciding how to use them.

---

Only train on the AWS VM — PEM key in ~/.ssh/mycelium-key.pem
Limit timeouts to less than 2 minutes
Always use vLLM + stage-based batching when possible, otherwise HF Transformers
Temperature = 0 (greedy) for all evals
MATH-500 NEVER in training data

---

## Current State (April 3, 2026)

```
Architecture:        DECOMPRESSOR (7L, 105M) → Llama 16L (untouched) → COMPRESSOR (7L, 105M)
Context window:      128K tokens
Compression:         64 floats on HYPERSPHERE (radius √64 ≈ 8.0)
Status:              ARCHITECTURE REFINEMENT — Symmetric Hourglass

KEY INSIGHT:
  The transformer stays PRISTINE. No layer splitting. No architectural surgery.
  It processes exactly as Llama was pretrained to process.

  The DECOMPRESSOR modifies what goes INTO the transformer.
  The COMPRESSOR reads what comes OUT of the transformer.
  The transformer is sandwiched between them.

ARCHITECTURE:
  DECOMPRESSOR:    Simple MLP, ~1.3M params
                   64 floats → 512 → 2048 → residual stream bias
                   EASY job: just project faithfully

  TRANSFORMER:     Llama 3.2 1B-Instruct, 16 layers, UNTOUCHED
                   Runs exactly as pretrained

  COMPRESSOR:      7 layers, ~120M params
                   All 16 layer hidden states → compress → 64 floats
                   HARD job: must SELECT what matters
                   Pass-conditioned layer gate (which layers matter NOW?)

  HYPERSPHERE:     state = normalize(state + delta) * √64
                   No learnable alpha. Constant magnitude. Each pass is a rotation.

  ConfidenceHead:  64 floats → scalar (when to stop thinking?)

ASYMMETRY (intentional):
  Simple MLP IN (decompression), 16 layers THROUGH (transformer), 7 layers OUT (compression)
  The narrow point is 64 floats
  89x more params for compression than decompression — because compression is HARD

PROVEN RESULTS:
  Single-step arithmetic: 100% (L0)
  Two-step arithmetic:    54-57% (L1)
  Three-step arithmetic:  19-22% (L2)
  Hypersphere vs alpha:   Nearly identical performance

Core principle:      The transformer is SANDWICHED, not modified.
                     Decompressor: "expand state into something that changes how transformer processes"
                     Compressor: "squeeze the most important findings into 64 floats"
```

---

## The Thinking Cycle

```
┌───────────────────────────────────────────────────────────┐
│                                                           │
│  DECOMPRESSOR (7 layers, 105M)                            │
│  64 floats → expand → residual stream bias                │
│         │                                                 │
│         ▼                                                 │
│  [bias + problem tokens] → Llama layers 1-16 (untouched)  │
│         │                                                 │
│         ▼                                                 │
│  COMPRESSOR (7 layers, 105M)                              │
│  all 16 layer hidden states → compress → 64 floats        │
│         │                                                 │
│         ▼                                                 │
│  state = normalize(state + delta) * √64  ← HYPERSPHERE    │
│         │                                                 │
│         ├──→ Confidence → ready? ──→ GENERATE             │
│         │                                                 │
│         └──→ loop back to DECOMPRESSOR                    │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

Each thinking pass:
```
Pass 1:
  DECOMPRESSOR(state) → bias
  [problem + bias] → Layer 1 → 2 → ... → 16 → hidden states
  COMPRESSOR(all layer hidden states) → delta
  state = normalize(state + delta) * √64

Pass 2:
  DECOMPRESSOR(state) → bias
  [problem + bias] → Layer 1 → 2 → ... → 16 → hidden states
  COMPRESSOR(all layer hidden states) → delta
  state = normalize(state + delta) * √64

Pass 3:
  DECOMPRESSOR(state) → bias
  [problem + bias] → Layer 1 → 2 → ... → 16 → hidden states
  COMPRESSOR(all layer hidden states) → delta
  state = normalize(state + delta) * √64
  Confidence > threshold → GENERATE ANSWER
```

Critical architecture:
- **Transformer is PRISTINE**: Llama runs exactly as pretrained, 16 layers, no modification
- **Decompressor**: 7 layers expanding 64 floats into input bias
- **Compressor**: 7 layers squeezing all 16 layer hidden states into 64 floats
- **Hypersphere**: State lives on sphere of radius √64, each pass is a rotation
- **No alpha**: Magnitude is constant, perceiver just outputs direction
- **Symmetric capacity**: 105M params on each side of the hourglass

---

## Why 64 Floats (The Tight Bottleneck)

```
64 floats = the narrow point of the hourglass
105M params EXPANDING into that point
105M params COMPRESSING out of that point

The asymmetry is extreme:
  210M total params deciding what goes through 64 floats
  Like two brilliant editors collaborating on a one-sentence summary

32 floats:   too tight for word problems
64 floats:   sweet spot — value + context + intent
128 floats:  too loose — might encode full solution in 2 passes
512 floats:  no compression pressure
```

---

## Architecture Components

### Decompressor (~1.3M params)
Lightweight MLP that projects 64-float state into input bias.

The decompressor has the EASY job: just project 64 floats into 2048-dim bias.
No hard decisions about what to keep — just faithful translation.

```python
# src/decompressor.py
class Decompressor(nn.Module):
    def __init__(self, state_size=64, d_model=2048, d_hidden=512, max_passes=20):
        super().__init__()

        # Pass embedding: tells MLP which thinking pass we're on
        self.pass_embed = nn.Embedding(max_passes, state_size)  # 1.3K params

        # Simple MLP: 64 → 512 → 512 → 2048
        self.mlp = nn.Sequential(
            nn.Linear(state_size, d_hidden),    # 64 → 512
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),      # 512 → 512
            nn.GELU(),
            nn.Linear(d_hidden, d_model),       # 512 → 2048
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, state, pass_num, scale=1.0):
        # state: (batch, 64)
        # Returns: bias to add to input embeddings (batch, 1, 2048)

        pass_context = self.pass_embed(torch.tensor(pass_num, device=state.device))
        x = state + pass_context

        bias = self.mlp(x)
        bias = self.output_norm(bias)
        return bias.unsqueeze(1) * scale  # (batch, 1, 2048)
```

### Compressor (~120M params)
7-layer perceiver that reads ALL 16 transformer layers and compresses to 64 floats.

The compressor has the HARD job: read all 16 layers across full sequence,
decide what matters, and squeeze into 64 floats. This is why it needs 89x more params.

```python
# src/compressor.py
class Compressor(nn.Module):
    def __init__(self,
                 num_transformer_layers=16,
                 d_transformer=2048,
                 d_internal=1024,
                 num_queries=4,
                 num_layers=7,
                 state_size=64,
                 max_passes=20):
        super().__init__()

        # Learned queries
        self.queries = nn.Parameter(torch.randn(num_queries, d_internal) * 0.02)
        self.pass_embed = nn.Embedding(max_passes, d_internal)

        # Pass-conditioned layer gate
        self.layer_gate = nn.Sequential(
            nn.Linear(d_internal, 64),
            nn.ReLU(),
            nn.Linear(64, num_transformer_layers),
        )

        # Project transformer hidden states to internal dim
        self.input_project = nn.Linear(d_transformer, d_internal)

        # 7-layer perceiver stack
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(d_internal, num_heads=8, batch_first=True),
                'cross_norm': nn.LayerNorm(d_internal),
                'self_attn': nn.MultiheadAttention(d_internal, num_heads=8, batch_first=True),
                'self_norm': nn.LayerNorm(d_internal),
                'ffn': nn.Sequential(
                    nn.Linear(d_internal, d_internal * 4),
                    nn.GELU(),
                    nn.Linear(d_internal * 4, d_internal),
                ),
                'ffn_norm': nn.LayerNorm(d_internal),
            })
            for _ in range(num_layers)
        ])

        # Project to state size
        self.project_out = nn.Linear(d_internal, state_size // num_queries)  # 16

    def forward(self, all_layer_hidden_states, pass_num):
        # all_layer_hidden_states: list of 16 tensors, each (batch, seq, 2048)
        batch_size = all_layer_hidden_states[0].size(0)
        device = all_layer_hidden_states[0].device

        # Pass-conditioned queries
        pass_context = self.pass_embed(torch.tensor(pass_num, device=device))
        queries = (self.queries + pass_context).unsqueeze(0).expand(batch_size, -1, -1)

        # Pass-conditioned layer weights
        layer_logits = self.layer_gate(pass_context)  # (16,)
        layer_weights = F.softmax(layer_logits, dim=-1)

        # Weighted combination of ALL transformer layers
        stacked = torch.stack([h.float() for h in all_layer_hidden_states], dim=0)
        combined = (stacked * layer_weights.view(-1, 1, 1, 1)).sum(dim=0)

        # Pool and project
        pooled = combined.mean(dim=1)  # (batch, 2048)
        kv = self.input_project(pooled).unsqueeze(1)  # (batch, 1, d_internal)

        # 7 layers of compression
        for layer in self.layers:
            attended, _ = layer['cross_attn'](queries, kv, kv)
            queries = layer['cross_norm'](queries + attended)
            refined, _ = layer['self_attn'](queries, queries, queries)
            queries = layer['self_norm'](queries + refined)
            queries = layer['ffn_norm'](queries + layer['ffn'](queries))

        # Project to state
        state_chunks = self.project_out(queries)  # (batch, 4, 16)
        return state_chunks.flatten(start_dim=1)  # (batch, 64)
```

### ThinkingModel — Full Loop
```python
# src/thinking_model.py
class ThinkingModel(nn.Module):
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct"):
        self.transformer = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Asymmetric: lightweight decompressor, heavy compressor
        self.decompressor = Decompressor(state_size=64, d_model=2048, d_hidden=512)  # 1.3M
        self.compressor = Compressor(state_size=64, d_transformer=2048, num_layers=7)  # 120M
        self.confidence = ConfidenceHead(state_size=64)

        self.state_size = 64
        self.state_radius = math.sqrt(64)  # ≈ 8.0

    def init_state(self, batch_size=1):
        """Initialize state on hypersphere."""
        state = torch.randn(batch_size, self.state_size, device=self.device)
        return F.normalize(state, dim=-1) * self.state_radius

    def update_state(self, state, delta):
        """Rotate state on hypersphere."""
        return F.normalize(state + delta, dim=-1) * self.state_radius

    def think(self, problem_text, max_passes=10, confidence_threshold=0.8, scale=1.0):
        state = self.init_state()
        prompt_embeds = self.get_prompt_embeds(problem_text)

        for pass_num in range(max_passes):
            # DECOMPRESS: state → bias (EASY, just project)
            bias = self.decompressor(state, pass_num, scale=scale)  # (1, 1, 2048)

            # TRANSFORMER: [bias + problem] → all 16 layers
            input_embeds = prompt_embeds + bias  # bias modulates all positions
            outputs = self.transformer(inputs_embeds=input_embeds, output_hidden_states=True)
            all_layer_hidden = list(outputs.hidden_states[1:])  # 16 layers

            # COMPRESS: all layers → 64 floats (HARD, must select)
            delta = self.compressor(all_layer_hidden, pass_num)

            # HYPERSPHERE: rotate state
            state = self.update_state(state, delta)

            if self.confidence(state).item() > confidence_threshold:
                break

        return state

    def generate_answer(self, problem_text, state, scale=1.0):
        bias = self.decompressor(state, pass_num=0, scale=scale)
        input_embeds = self.get_prompt_embeds(problem_text) + bias
        return self.transformer.generate(inputs_embeds=input_embeds, max_new_tokens=512)
```

### Parameter Count
```
Llama 3.2 1B-Instruct:  1.23B   (frozen Phase 1, fine-tuned Phase 2)
Compressor:             ~120M   (7 layers, HARD job — selecting what matters)
Decompressor:           ~1.3M   (simple MLP, EASY job — just projecting)
ConfidenceHead:         ~2K     (tiny MLP)
────────────────────────────────
Total:                  ~1.35B
New parameters:         ~121M   (9% of total model, was 15% with symmetric)
Bottleneck:             64 floats on hypersphere
Asymmetry:              89x more params for compression than decompression
```

---

## Two-Phase Training

### Phase 1: Freeze Transformer, Train Hourglass
```
Trainable:  Decompressor + Compressor + ConfidenceHead
Frozen:     Transformer (already gets 35% on GSM8K)
Duration:   5-10 epochs

The transformer runs normally (it already understands math).
The decompressor learns to expand state into useful bias.
The compressor learns to extract useful state from all layers.
Low risk: can't hurt the transformer's existing capability.
```

### Phase 2: End-to-End
```
Trainable:  Everything
Duration:   10-20 epochs

Now the transformer adapts to USE the bias information.
Co-evolution: better decompression ↔ better processing ↔ better compression.
```

### Deep Supervision
Every thinking pass tries to predict the answer from the CURRENT state:
```python
for pass_num, intermediate_state in enumerate(all_states):
    bias = decompressor(intermediate_state, pass_num)
    outputs = transformer(inputs_embeds=prompt_embeds + bias, labels=answer_ids)
    weight = (pass_num + 1) / len(all_states)
    total_loss += weight * outputs.loss
```

### State Scale Warmup (Proven Schedule)
```
Epoch 1-2:   scale = 0.1   (bias barely affects transformer)
Epoch 3-4:   scale = 0.3
Epoch 5-6:   scale = 0.5
Epoch 7-8:   scale = 0.7
Epoch 9-10:  scale = 1.0   (full bias strength)
```

---

## Why This Is Better Than Before

### vs Prepended Pseudo-Tokens (v18)
```
Before: 4 pseudo-tokens prepended to input
        Transformer might attend to them, might ignore them
        Position 0-3 are "special" — distribution shift

Now:    Bias added to ALL input positions
        Every token in the problem is modulated by state
        No "special" positions — uniform influence
```

### vs StateInjector (~130K params)
```
Before: Simple projection: 64 → 4 tokens (~130K params)
        Had to compete for attention at positions 0-3

Now:    Simple MLP: 64 → 512 → 2048 (~1.3M params)
        Adds bias to ALL positions uniformly
        10x more capacity, much better influence mechanism
```

### Intentional Asymmetry
```
Before: ~108M compressor, ~130K injector (accidental asymmetry)

Now:    ~120M compressor, ~1.3M decompressor (intentional asymmetry)
        89x more params for compression than decompression
        Because compression is HARD (selecting what matters)
        Decompression is EASY (just projecting faithfully)
```

---

## File Structure

```
src/
  __init__.py           # Package exports
  decompressor.py       # Simple MLP state → bias (1.3M) — EASY job
  compressor.py         # 7-layer all-layer → state (120M) — HARD job
  confidence_head.py    # When to stop thinking
  thinking_model.py     # Full thinking loop

scripts/
  train_thinking.py     # Training with deep supervision
  eval_thinking.py      # Accuracy vs passes, ablation

configs/
  thinking_gsm8k.yaml   # Hyperparameters
```

---

## Milestones

```
PROOF-OF-CONCEPT (COMPLETE):
M0: SmolLM2 baseline: 0% GSM8K
M1: Tight bottleneck works: 80.4% two-step, 52% three-step
M2: Llama two-step: 83%
M3: Hypersphere constraint: works as well as alpha

ASYMMETRIC HOURGLASS (CURRENT):
M4: Build asymmetric architecture
    Decompressor (MLP, 1.3M), Compressor (7L, 120M)

M5: Validation checkpoints
    V1: Forward pass works with bias injection
    V2: Gradients flow through all passes
    V3: State accumulates (pass_3 > pass_1 accuracy)
    V4: Thinking ≈ single-shot (neutral)
    V5: Thinking > single-shot (THE RESULT)

M6: GSM8K thinking accuracy > 35% (single-shot baseline)

M7: GSM8K thinking accuracy > 45% (+10 points)

M8: MATH L1-L5 (capacity extension)
```

---

## Critical Rules

```
 1. Llama 3.2 1B-Instruct (NOT base) — need 35% baseline for gradient signal
 2. Use bfloat16 for Llama
 3. 64-float bottleneck on HYPERSPHERE (radius √64)
 4. Transformer stays PRISTINE — no layer splitting, no modification
 5. Decompressor: Simple MLP (1.3M), EASY job — just project state → bias
 6. Compressor: 7 layers (120M), HARD job — select from all 16 transformer layers → state
 7. Pass-conditioned layer gate in compressor
 8. Hypersphere update: state = normalize(state + delta) * √64
 9. NO learnable alpha — hypersphere handles magnitude
10. Deep supervision: every pass gets direct gradient
11. State scale warmup: 0.1 → 1.0 over first 5 epochs
12. Freeze transformer Phase 1, unfreeze Phase 2
13. Bias modulates ALL input positions, not just prepended tokens
14. Ablation (real state vs zeros) at every checkpoint
15. Temperature = 0 for all evaluations
16. N >= 100 problems for any accuracy measurement
17. MATH-500 NEVER in training data
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

FOR v19:
42. Use chat template for Instruct model
43. output_hidden_states=True to get all 16 layers
44. hidden_states[0] is embedding layer — use hidden_states[1:] for transformer layers
45. Gradient clipping essential with deep supervision
46. Bias should be ADDED to embeddings, not concatenated
```

---

## Hardware

```
Training:    Llama 1.23B + ~210M hourglass fits on A10G (24GB)
             With gradient checkpointing: 10 passes fits
Phase 1:     ~1-2 days on A10G
Phase 2:     ~2-3 days on A10G
Cost:        ~$75-150 on AWS spot
```

---

## MATH-500 Deadline: April 22, 2026

GSM8K first (prove thinking helps) → MATH (prove it scales).
Target: thinking accuracy > single-shot on L4-L5 problems.
The undeniable result: solving previously unsolvable problems through symmetric compression.
