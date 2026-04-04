# Breathing Models — Final Architecture Handoff

## One-Sentence Summary

The state vector doesn't whisper to the transformer through tokens — it REWIRES the transformer through state-conditioned LoRA. A 7-layer Perceiver compresses all 16 Llama layers into 64 floats. Those 64 floats generate scaling factors for learned LoRA templates, changing HOW the transformer attends at every layer on every thinking pass. Same problem, different eyes each pass.

---

## The Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  STATE (64 floats on hypersphere)                                │
│     │                                                            │
│     ├──→ HYPERNETWORK (64 → 64 LoRA scales)                     │
│     │         │                                                  │
│     │         ▼                                                  │
│     │    Scale learned A/B templates                             │
│     │    Apply LoRA to ALL 16 layers (Q, K, V, O)                │
│     │         │                                                  │
│     │         ▼                                                  │
│     │    [problem tokens] → Llama layers 1-16 (WITH LoRA)        │
│     │         │          │         │              │              │
│     │       (all 16 layer hidden states saved)                   │
│     │         │          │         │              │              │
│     │         ▼          ▼         ▼              ▼              │
│     │    7-LAYER PERCEIVER COMPRESSOR (105M params)               │
│     │    reads ALL layers, pass-conditioned attention             │
│     │         │                                                  │
│     │         ▼                                                  │
│     │    64-float state delta                                    │
│     │         │                                                  │
│     │         ▼                                                  │
│     └──→ state = normalize(state + delta) * √64  ← hypersphere  │
│               │                                                  │
│               ├──→ ConfidenceHead → ready?                       │
│               │         │                                        │
│               │     YES ▼                                        │
│               │    GENERATE ANSWER                               │
│               │    (Llama + final LoRA, generate text)            │
│               │                                                  │
│               └──→ NO: loop back to HYPERNETWORK                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### What Happens Each Pass

```
Pass 1:
  state = [initial point on hypersphere]
  → Hypernetwork: state → scales [0.9, 0.1, 0.3, 0.8] per layer
  → LoRA templates scaled: "focus on numbers and the question"
  → Llama processes problem with number-focused attention
  → Perceiver reads all 16 layers: "I see 48 clips, half as many, need total"
  → delta compressed to 64 floats
  → state rotates on hypersphere
  → Confidence: 0.2 (just parsed, not ready)

Pass 2:
  state = [rotated position]
  → Hypernetwork: state → scales [0.2, 0.9, 0.7, 0.1] per layer
  → LoRA templates scaled: "focus on operations and relationships"
  → Llama processes problem with computation-focused attention
  → Perceiver reads all 16 layers: "half of 48 is 24"
  → delta compressed, state rotates
  → Confidence: 0.5 (computed one step)

Pass 3:
  state = [rotated again]  
  → Hypernetwork: state → scales [0.3, 0.6, 0.5, 0.9] per layer
  → LoRA templates scaled: "focus on combining results"
  → Llama processes problem with answer-focused attention
  → Perceiver: "48 + 24 = 72"
  → Confidence: 0.9 (ready!)
  → GENERATE: "\boxed{72}"
```

The transformer reads the SAME problem tokens each pass. But it attends DIFFERENTLY because the LoRA weights change. Pass 1 notices the numbers. Pass 2 notices the operations. Pass 3 puts it together. Not because we told it to — because the state-conditioned LoRA learned these attention patterns through end-to-end training.

---

## The Five Components

### 1. Llama 3.2 1B-Instruct (THE THINKER)

```
Model:          meta-llama/Llama-3.2-1B-Instruct (via unsloth)
Parameters:     1.23B
Hidden dim:     2048
Layers:         16
Attention:      32 heads (8 KV heads, GQA)
Context:        128K tokens
GSM8K baseline: ~35% single-shot with CoT

The transformer processes the problem. Its attention weights are modified
per-pass by the state-conditioned LoRA. Same architecture, different
effective model each thinking pass.
```

### 2. State-Conditioned LoRA Hypernetwork (THE REWIRER)

The state vector generates scaling factors that modulate learned LoRA templates. The templates define KINDS of attention modification. The scales select the MIX for each pass.

```python
class StateConditionedLoRA(nn.Module):
    def __init__(self, d_model=2048, state_size=64, rank=4, 
                 num_layers=16, num_projections=4):
        super().__init__()
        # 4 projections: Q, K, V, O per layer
        self.num_layers = num_layers
        self.num_projections = num_projections  # Q, K, V, O
        self.rank = rank
        
        # Learned LoRA templates (shared across all passes)
        # These learn "useful ways to modify attention for math"
        self.A_templates = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, d_model, rank) * 0.01)
            for _ in range(num_projections)
        ])
        self.B_templates = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, rank, d_model) * 0.01)
            for _ in range(num_projections)
        ])
        
        # Hypernetwork: state → scaling factors
        # 64 floats → (16 layers × 4 projections × 4 rank) = 256 scales
        self.state_to_scales = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.Tanh(),  # scales bounded to [-1, 1]
        )
    
    def forward(self, state):
        """
        state: (batch, 64)
        returns: dict of per-layer LoRA modifications
        """
        # Generate all scales from state
        all_scales = self.state_to_scales(state)  # (batch, 256)
        all_scales = all_scales.reshape(-1, self.num_layers, self.num_projections, self.rank)
        # (batch, 16, 4, 4) = per-layer, per-projection, per-rank scales
        
        lora_mods = {}
        proj_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        
        for layer_idx in range(self.num_layers):
            lora_mods[layer_idx] = {}
            for proj_idx, proj_name in enumerate(proj_names):
                scales = all_scales[:, layer_idx, proj_idx, :]  # (batch, rank)
                A = self.A_templates[proj_idx][layer_idx]        # (2048, rank)
                B = self.B_templates[proj_idx][layer_idx]        # (rank, 2048)
                
                # Scale the rank dimensions
                # Each rank dimension is a "type" of attention modification
                # The state controls the mix
                lora_mods[layer_idx][proj_name] = {
                    'A': A,           # (2048, rank)
                    'B': B,           # (rank, 2048)
                    'scales': scales, # (batch, rank)
                }
        
        return lora_mods
```

Parameter count:
```
A templates: 4 projections × 16 layers × 2048 × 4 rank = 524K
B templates: 4 projections × 16 layers × 4 × 2048      = 524K
Hypernetwork: Linear(64 → 256)                           = 16K
Total: ~1.1M params
```

The templates are the VOCABULARY of thinking styles. 4 rank dimensions = 4 types of attention modification per projection. The state selects the mix. Different state = different attention = different thinking.

### 3. 7-Layer Perceiver Compressor (THE NOTE-TAKER)

Reads ALL 16 transformer layers with pass-conditioned attention. Compresses to 64 floats.

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
        
        self.queries = nn.Parameter(torch.randn(num_queries, d_perceiver))
        self.pass_embed = nn.Embedding(max_passes, d_perceiver)
        
        # Pass-conditioned layer gate
        self.layer_gate = nn.Sequential(
            nn.Linear(d_perceiver, 64),
            nn.ReLU(),
            nn.Linear(64, num_transformer_layers),
            nn.Softmax(dim=-1),
        )
        
        self.input_project = nn.Linear(d_transformer, d_perceiver)
        
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
        
        self.project_out = nn.Linear(d_perceiver, state_size // num_queries)
    
    def forward(self, all_layer_hidden_states, pass_num):
        batch_size = all_layer_hidden_states[0].size(0)
        
        pass_context = self.pass_embed(torch.tensor(pass_num, device=self.queries.device))
        queries = (self.queries + pass_context).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Pass-conditioned layer importance
        layer_weights = self.layer_gate(pass_context)
        stacked = torch.stack(all_layer_hidden_states, dim=0)
        weights = layer_weights.view(-1, 1, 1, 1)
        combined = (stacked * weights).sum(dim=0)
        
        kv = self.input_project(combined)
        
        for layer in self.layers:
            attended, _ = layer['cross_attn'](query=queries, key=kv, value=kv)
            queries = layer['cross_norm'](queries + attended)
            
            refined, _ = layer['self_attn'](query=queries, key=queries, value=queries)
            queries = layer['self_norm'](queries + refined)
            
            queries = layer['ffn_norm'](queries + layer['ffn'](queries))
        
        state_delta = self.project_out(queries)
        return state_delta.flatten(start_dim=1)
```

Parameter count: ~105M

### 4. ConfidenceHead (THE JUDGE)

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

    def forward(self, state):
        return self.net(state)
```

Parameter count: ~2.1K

### 5. Hypersphere State (THE MEMORY)

```python
# After each pass:
state = state + delta                              # accumulate
state = F.normalize(state, dim=-1) * math.sqrt(64) # project back to hypersphere

# Cosine similarity between consecutive states measures rotation
# ~0.7 = 46° rotation (meaningful change)
# ~0.99 = tiny rotation (pass barely changed anything)
# ~0.0 = 90° rotation (completely new direction)
```

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
        
        self.lora = StateConditionedLoRA(d_model=2048, state_size=64, rank=4, num_layers=16)
        self.compressor = AllLayerPerceiver(
            num_transformer_layers=16, d_transformer=2048,
            d_perceiver=1024, num_queries=4,
            num_perceiver_layers=7, state_size=64,
        )
        self.confidence = ConfidenceHead(state_size=64)
        self.state_size = 64

    def apply_lora(self, state):
        """Generate and apply state-conditioned LoRA to transformer."""
        lora_mods = self.lora(state)
        # Hook into transformer attention layers
        for layer_idx, mods in lora_mods.items():
            layer = self.transformer.model.layers[layer_idx].self_attn
            for proj_name, lora_params in mods.items():
                A = lora_params['A']           # (2048, 4)
                B = lora_params['B']           # (4, 2048)
                scales = lora_params['scales'] # (batch, 4)
                # Store for use in forward hook
                setattr(layer, f'{proj_name}_lora', (A, B, scales))

    def remove_lora(self):
        """Remove LoRA modifications after pass."""
        for layer in self.transformer.model.layers:
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if hasattr(layer.self_attn, f'{proj_name}_lora'):
                    delattr(layer.self_attn, f'{proj_name}_lora')

    def think(self, problem_text, max_passes=10, confidence_threshold=0.8):
        """Think in a loop: LoRA-modified forward passes + compression."""
        messages = [{"role": "user", "content": problem_text}]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        state = F.normalize(
            torch.randn(1, self.state_size, device=self.device), dim=-1
        ) * math.sqrt(self.state_size)
        
        all_states = [state]
        confidences = []

        for pass_num in range(max_passes):
            # Apply state-conditioned LoRA
            self.apply_lora(state)

            # Forward through transformer (WITH LoRA modifications)
            outputs = self.transformer(
                input_ids=prompt_ids,
                output_hidden_states=True,
            )
            all_layer_hidden = list(outputs.hidden_states[1:])  # 16 layers

            # Remove LoRA after pass
            self.remove_lora()

            # Perceiver compresses all layers to 64 floats
            delta = self.compressor(all_layer_hidden, pass_num)

            # Hypersphere: accumulate and normalize
            state = state + delta
            state = F.normalize(state, dim=-1) * math.sqrt(self.state_size)
            all_states.append(state)

            # Check confidence
            conf = self.confidence(state)
            confidences.append(conf.item())

            if conf.item() > confidence_threshold:
                break

        return state, all_states, confidences

    def generate_answer(self, problem_text, state):
        """Final pass: generate text with state-conditioned LoRA."""
        messages = [{"role": "user", "content": problem_text}]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        # Apply final LoRA
        self.apply_lora(state)

        output = self.transformer.generate(
            input_ids=prompt_ids,
            max_new_tokens=512,
            do_sample=False,
        )

        self.remove_lora()
        
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return answer

    def solve(self, problem_text, max_passes=10):
        """Full pipeline: think → generate."""
        state, all_states, confidences = self.think(problem_text, max_passes)
        answer = self.generate_answer(problem_text, state)
        return {
            'answer': answer,
            'num_passes': len(confidences),
            'confidences': confidences,
        }
```

---

## Training

### Training Loop with Deep Supervision

```python
def train(model, problems, optimizer, max_passes=5):
    for problem, gold_answer in problems:
        optimizer.zero_grad()

        state, all_states, confidences = model.think(problem, max_passes)

        # Deep supervision: every intermediate state tries to answer
        total_loss = 0.0
        for i, intermediate_state in enumerate(all_states[1:]):
            # Apply LoRA from this intermediate state
            model.apply_lora(intermediate_state)

            # Teacher-forced answer prediction
            prompt_ids = model.tokenizer.apply_chat_template(
                [{"role": "user", "content": problem}],
                add_generation_prompt=True, return_tensors="pt"
            ).to(device)
            answer_ids = model.tokenizer.encode(gold_answer, return_tensors="pt").to(device)
            
            full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
            outputs = model.transformer(input_ids=full_ids, labels=full_ids)
            
            model.remove_lora()

            # Weight increases with pass number
            weight = (i + 1) / len(all_states[1:])
            total_loss += weight * outputs.loss

        # Confidence loss
        confidence_loss = 0.0
        for i, conf in enumerate(confidences):
            target = min((i + 1) / max_passes + 0.1, 0.95)
            confidence_loss += F.mse_loss(
                torch.tensor(conf, device=device),
                torch.tensor(target, device=device),
            )
        confidence_loss /= len(confidences)

        # Efficiency penalty
        efficiency = 0.01 * len(confidences)

        loss = total_loss + 0.1 * confidence_loss + efficiency
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
```

### Training Phases

```
Phase 1: Freeze Llama, train LoRA templates + Perceiver + Confidence (5-10 epochs)
  Llama processes normally. LoRA templates learn useful attention modifications.
  Perceiver learns what to extract. Confidence learns when to stop.
  Low risk: can't hurt Llama's existing 35% capability.

Phase 2: Unfreeze Llama, end-to-end (10-20 epochs)
  Llama adapts to the LoRA modifications.
  Everything co-evolves: templates + perceiver + transformer.
```

### Curriculum

```
Start:    max_passes = 3 (minimal chain, strong gradients)
Epoch 5:  max_passes = 5
Epoch 10: max_passes = 7
Epoch 15: max_passes = 10

Start:    GSM8K easy (2-3 step problems)
Epoch 5:  Full GSM8K
Epoch 15: MATH L1-L3
```

---

## Parameter Count

```
Llama 3.2 1B-Instruct:       1.23B    (the thinker)
7-Layer Perceiver:            ~105M    (the compressor — reads all layers)
State-Conditioned LoRA:       ~1.1M    (the rewirer — changes attention)
ConfidenceHead:               ~2.1K    (the judge)
64-float state:               64       (the tight straw)
────────────────────────────────────────
Total:                        ~1.34B
New components:               ~106M    (8.6% of total)
Bottleneck:                   64 floats
```

The asymmetry:
```
COMPRESSOR:    105M params deciding what to remember (HARD job)
REWIRER:       1.1M params changing how to think (LIGHT job — just scale templates)
BOTTLENECK:    64 floats between them (TIGHT straw)
```

---

## Why This Architecture Is Different

### vs Pseudo-tokens (our previous approach)
```
Before: state → 4 pseudo-tokens → transformer ignores them (3.8% of input)
Now:    state → LoRA scales → transformer's attention weights change at every layer
        The state can't be ignored — it's IN the weights, not in the input
```

### vs Bias injection (failed approach)
```
Before: state → bias added to embeddings → corrupted pretrained representations
Now:    state → LoRA rank scaling → small, structured perturbation
        LoRA is designed for pretrained models — it works by construction
```

### vs Standard LoRA fine-tuning
```
Standard: fixed LoRA weights, same modification every forward pass
Ours:     state-CONDITIONED LoRA, different modification each pass
          The model becomes a different model on each thinking pass
```

### vs External compression loop (v16)
```
Before: generate text → compress text → generate text (slow, not differentiable)
Now:    forward pass → compress hidden states → modify LoRA → forward pass (fast, differentiable)
        No text generation during thinking. Just forward passes + LoRA updates.
```

---

## What to Monitor

### Per Epoch
```
thinking_accuracy:     accuracy after N thinking passes
single_shot_accuracy:  same model, no thinking passes (LoRA at zero)
delta:                 thinking - single_shot (THIS IS THE RESULT)
avg_passes:            average passes before confident
```

### Per Pass
```
pass_N_accuracy:       accuracy from intermediate state at pass N
                       Should INCREASE with N (more thinking = better)
lora_scale_magnitude:  how much is the state changing the attention?
cosine_sim:            rotation between consecutive states on hypersphere
```

### LoRA Diagnostics
```
template_norms:        are the A/B templates learning? (should grow from 0.01 init)
scale_distribution:    are scales diverse across passes? (different thinking styles)
scale_correlation:     do similar problems produce similar scales? (generalization)
```

### The Key Plot
```
X-axis: number of thinking passes
Y-axis: accuracy on GSM8K

Single-shot baseline (horizontal line at 35%)
Thinking curve (should rise above 35% with more passes)

The GAP between the curve and the baseline IS the contribution.
```

---

## Validation Checkpoints

```
V1: LoRA applies without breaking Llama
    Model still generates coherent text with state-conditioned LoRA
    Single-shot accuracy remains ~35% (LoRA at zero state = no modification)
    Timeline: step 1

V2: Gradients flow through LoRA → state → perceiver
    grad_norm > 0 on templates, hypernetwork, perceiver
    Timeline: step 1

V3: LoRA scales CHANGE between passes
    Cosine similarity of scale vectors between pass 1 and pass 3 < 0.9
    The model uses different attention patterns per pass
    Timeline: epoch 1-2

V4: Pass 3 accuracy > Pass 1 accuracy
    More thinking = better (the loop adds value)
    Timeline: epoch 3-5

V5: Thinking accuracy matches single-shot (35%)
    LoRA isn't hurting — neutral
    Timeline: epoch 5-8

V6: Thinking accuracy EXCEEDS single-shot  ← THE RESULT
    The model solves problems it can't solve in one pass
    Timeline: epoch 8-15

V7: Thinking accuracy > 45% on GSM8K
    +10 points over baseline — substantial improvement
    Timeline: epoch 15-20

V8: Confidence head works
    Easy problems: few passes, high confidence
    Hard problems: many passes, low initial confidence
    Timeline: epoch 10-15
```

---

## Infrastructure

```
Model:          unsloth/Llama-3.2-1B-Instruct (bf16)
Compute:        AWS EC2 g5.xlarge (A10G 24GB)
Memory:         ~2.5GB model + ~0.5GB perceiver + LoRA overhead
                Forward passes with output_hidden_states=True use more memory
                Gradient checkpointing essential for multi-pass training
Training:       HF Transformers + custom LoRA hooks
Data:           GSM8K train (7,473 problems + gold answers)
Time:           Phase 1 ~2-3 days, Phase 2 ~3-5 days
Cost:           ~$100-200 on AWS spot
```

### Files to Create
```
src/
  thinking_model.py              # ThinkingModel (full architecture)
  state_conditioned_lora.py      # StateConditionedLoRA hypernetwork
  all_layer_perceiver.py         # 7-layer perceiver compressor
  confidence_head.py             # Readiness judge
  lora_hooks.py                  # Forward hooks for applying/removing LoRA

scripts/
  train_thinking.py              # Training with deep supervision
  eval_thinking.py               # Accuracy vs passes, ablation
  visualize_lora_scales.py       # What attention patterns does each pass use?
  visualize_thinking.py          # Plot accuracy vs num_passes curve

configs/
  thinking_gsm8k.yaml            # Hyperparameters
```

---

## What NOT to Do

```
- Do NOT inject state as pseudo-tokens. The transformer ignores them.
- Do NOT inject state as embedding bias. It corrupts pretrained representations.
- Do NOT use a large decompressor. The hypernetwork is 1M params — that's enough.
- Do NOT start with 10 passes. Start with 3, increase gradually.
- Do NOT skip deep supervision. Every pass needs direct gradient.
- Do NOT use the base model. Instruct gets 35% — we need that gradient signal.
- Do NOT forget gradient clipping (max_norm=1.0).
- Do NOT train Llama before LoRA templates and perceiver are working (Phase 1 first).
- Do NOT make LoRA rank too high. Rank 4 = 4 thinking styles. Keep it small and interpretable.
- Do NOT forget to remove LoRA hooks after each pass to prevent accumulation.
```

---

## The Philosophical Core

The model THINKS BEFORE IT SPEAKS. And each time it thinks, it thinks DIFFERENTLY.

Not different input. Not different tokens. Different ATTENTION. The state vector learned from previous thinking rewires how the transformer attends to the problem. Numbers that were background become foreground. Operations that were implicit become focal points.

The confidence head learns when the accumulated thinking is sufficient. Easy problems: one or two passes of different attention patterns, then answer. Hard problems: many passes, each one shifting attention to a new aspect, gradually building understanding.

The 64-float bottleneck ensures each pass can only capture one bite. The hypersphere ensures each pass genuinely rotates to a new perspective. The LoRA ensures that new perspective translates into different attention. The perceiver ensures the most important findings are compressed into the next state.

Think. Compress. Rotate. Rewire. Think again. Until confident.

---

## Success Criterion

**Thinking accuracy > 35% on GSM8K.**

35% is single-shot. Above 35% means the thinking loop solved problems the model couldn't solve in one pass by rewiring its attention across multiple passes of the same problem.

---

## MATH-500 Deadline: April 22, 2026
