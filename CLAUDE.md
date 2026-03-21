# Mycelium v17 — Breathing Models (Llama 3.2 1B)

A transformer (Llama 3.2 1B) trained to reason in expand-collapse cycles with **differentiable latent-space compression**. The model solves problems beyond its single-shot capacity by taking small, verified steps, externalizing working memory through a learned 512-float state vector.

> Good expansion is expansion that's easy to collapse.
> Good collapse is collapse that sets up the next expansion.

---

Only train on the AWS VM — PEM key in ~/.ssh/mycelium-key.pem
Limit timeouts to less than 2 minutes
Always use vLLM + stage-based batching when possible, otherwise HF Transformers
Temperature = 0 (greedy) for all evals
MATH-500 NEVER in training data

---

## Current State (April 2, 2026)

```
Architecture:        Llama 3.2 1B + CompressionHead (64 queries, breath-conditioned) + StateInjector (8 pseudo-tokens)
Context window:      128K tokens
Compression:         Differentiable latent-space (512-float state vector)
Status:              ENGINE SWAP — Llama 3.2 1B replaces SmolLM2-135M

ENGINE SWAP RATIONALE:
  SmolLM2-135M proved the architecture (80.4% two-step, 52% three-step)
  but can't parse word problems — doesn't understand "half as many"
  Llama 3.2 1B: 1.23B params, 9T tokens, distilled from 8B/70B, ~44% GSM8K instruct

ARCHITECTURE (unchanged except d_model):
  CompressionHead:   64 queries (breath-conditioned) → cross-attend → 512 floats
  StateInjector:     512 floats → 8 pseudo-tokens (simple projection)
  Transformer sees:  [8 pseudo-tokens] + [BREATH 3/10] + [problem tokens]
  d_model:           2048 (was 576 for SmolLM2)

SMOLLM2 PROOF-OF-CONCEPT (COMPLETE):
  Two-step arithmetic:   0% → 80.4% (32 floats, 4 queries)
  Three-step arithmetic: 0% → 52%   (64 floats, 4 queries)
  Ablation delta:        +90 points (real state vs zeros)

Core principle:      Compression happens in LATENT SPACE, not text space.
                     The state vector is a continuous representation that
                     gradients can flow through from answer loss back
                     through every breath.

Next:                Retrain autoencoder for Llama hidden states (2048-dim)
                     → Arithmetic smoke tests → GSM8K with alternating breaths
```

---

## The Breathing Cycle (Latent-Space Version)

Each breath is one forward pass with state injection:

```
Input:   [state_tokens (8)] + [problem_tokens]
         |
         Llama 3.2 1B generates reasoning
         |
Output:  hidden_states -> CompressionHead -> 512-float state vector
         |
         state vector -> StateInjector -> 8 pseudo-tokens for next breath
```

Critical architecture:
- **State vector**: 512 floats (not text) — continuous, differentiable
- **Pseudo-tokens**: 8 learned embeddings prepended to input
- **Compression**: Perceiver-style cross-attention (queries attend to hidden states)
- **Fully differentiable**: Gradients flow from answer loss through state vector

The model loops until `\boxed{}` or max 20 breaths.

---

## Architecture Components

### CompressionHead (~8.5M params)
Breath-conditioned Perceiver encoder: hidden states + breath_num -> 512-float state vector

64 queries give fine-grained extraction. Breath embedding tells the compression head where it is.
At breath 1, ask "what's the setup?" At breath 5, ask "what progress and what's left?"

```python
# src/compression_head.py
class CompressionHead(nn.Module):
    def __init__(self, d_model=2048, state_size=512, num_queries=64, max_breaths=20):
        self.queries = nn.Parameter(torch.randn(num_queries, d_model))  # learned
        self.breath_embed = nn.Embedding(max_breaths, d_model)  # breath conditioning
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=8)
        self.norm = nn.LayerNorm(d_model)
        self.project = nn.Linear(d_model, state_size // num_queries)  # 2048 -> 8

    def forward(self, hidden_states, breath_num):
        # hidden_states: (batch, seq_len, 2048)
        # breath_num: (batch,) or scalar — which breath we're on (0-indexed)

        # Condition queries on breath number
        breath_context = self.breath_embed(breath_num)  # (batch, d_model) or (d_model,)
        conditioned_queries = self.queries + breath_context.unsqueeze(-2)  # broadcast add

        compressed, _ = self.cross_attn(query=conditioned_queries, key=hidden_states, value=hidden_states)
        compressed = self.norm(compressed)
        # compressed: (batch, 64, 2048)

        chunks = self.project(compressed)  # (batch, 64, 8)
        return chunks.flatten(start_dim=1)  # (batch, 512)
```

### StateInjector (~1M params)
Converts 512-float state vector to 8 pseudo-tokens (simple projection, transformer attention does the smart work)

```python
# src/state_injector.py
class StateInjector(nn.Module):
    def __init__(self, state_size=512, d_model=2048, num_tokens=8):
        self.project = nn.Linear(state_size // num_tokens, d_model)  # 64 -> 2048
        self.position_embed = nn.Parameter(torch.randn(num_tokens, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, state_vector, scale=1.0):
        # state_vector: (batch, 512)
        chunks = state_vector.reshape(-1, 8, 64)
        tokens = self.project(chunks) + self.position_embed
        return self.norm(tokens) * scale  # (batch, 8, 2048)
```

### StateDecoder (~8M params) — Phase 0 only
Reconstructs hidden states from state vector (autoencoder training)

```python
# src/state_decoder.py
class StateDecoder(nn.Module):
    def __init__(self, state_size=512, d_model=2048, max_seq_len=64):
        self.project = nn.Linear(state_size, max_seq_len * d_model)
        self.refine = nn.TransformerEncoder(...)  # 2 layers

    def forward(self, state_vector):
        # state_vector: (batch, 512)
        return reconstructed  # (batch, 64, 2048)
```

### BreathingModel — Full Recurrent Loop
```python
# src/breathing_model.py
class BreathingModel(nn.Module):
    def __init__(self):
        self.transformer = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            torch_dtype=torch.bfloat16,
        )
        self.compressor = CompressionHead(d_model=2048, state_size=512, num_queries=64)
        self.injector = StateInjector(state_size=512, d_model=2048, num_tokens=8)

    def breathe(self, problem_text, max_breaths=20):
        state = self.injector.get_empty_state()  # zeros (batch, 512)
        for breath_num in range(max_breaths):
            # Inject state as 8 pseudo-tokens
            state_tokens = self.injector(state)
            # Generate with transformer (sees [BREATH n/max] in prompt)
            hidden_states = self.forward_with_state(state_tokens, problem_text, breath_num)
            # Compress to new state (breath-conditioned queries)
            state = self.compressor(hidden_states, breath_num=breath_num)
            if has_boxed_answer: break
        return state, generated_text
```

### Parameter Count
```
Llama 3.2 1B:       1.23B (frozen in Phase 0, trainable in Phase 1-2)
CompressionHead:    ~8.5M (64 queries + breath embedding + attention + projection)
StateInjector:      ~1M (8 tokens × 2048 d_model)
StateDecoder:       ~8M   (Phase 0 only, discarded after)
Total trainable:    ~9.5M (Phase 0), ~1.24B (Phase 1-2)
```

---

## Two-Phase Training (Proven Recipe)

The original three-phase plan (freeze then unfreeze) turned out to be unnecessary.
The autoencoder + state scale warmup is sufficient for co-adaptation.

### Phase 0: Autoencoder Warmup — "Give the compressor a meaningful starting point"
```
Problem:    Cold start. Random compression head outputs garbage vectors.
Solution:   Train encoder+decoder on oracle collapses BEFORE any breathing.
Data:       Oracle-optimal collapses from GSM8K/MATH solutions
Objective:  Reconstruction loss (MSE on hidden states)
Duration:   2-4 hours on A10G (must re-run for Llama 2048-dim hidden states)

IMPORTANT FOR ENGINE SWAP:
- Pre-compute hidden states: run 206K collapses through frozen Llama 3.2 1B
- Memmap to disk (~80GB for 206K × 50 tokens × 2048 dims × 4 bytes)
- Then autoencoder trains on pure tensors — no transformer in the loop, fast

After Phase 0:
- Compression head outputs MEANINGFUL 512-float vectors
- Similar collapses -> similar state vectors (cosine similarity)
- t-SNE shows structure (variance ratio > 0.1)
- Decoder is DISCARDED — only encoder continues to breathing
```

Config: `configs/autoencoder.yaml`
Script: `scripts/train_autoencoder.py`

### Phase 1: End-to-End Breathing — "Co-evolve compression and generation"
```
Objective:  Full differentiable training through the state vector
Trainable:  Everything (CompressionHead + Transformer + StateInjector)
Initialized: CompressionHead from Phase 0 autoencoder
Loss:       Answer correctness (gradients flow through state!)
Duration:   4-8 hours per difficulty level

Key insight: State scale warmup (0.1 → 1.0 over 5 epochs) provides
the adjustment period. No frozen phase needed.

Why this works:
- Autoencoder gives good enough starting point
- Scale warmup gives transformer time to adjust to pseudo-tokens
- Compressor and transformer co-evolve from the start
- Proven by SmolLM2 two-step (80.4%) and three-step (52%) results
```

Config: `configs/gsm8k_easy.yaml`
Script: `scripts/train_gsm8k_easy.py`

---

## Why Latent-Space (Not Text-Based)

| Aspect | Text-Based (v10-v15) | Latent-Space (v16+) |
|--------|---------------------|-------------------|
| Compression | [COLLAPSE] text tokens | 512-float vector |
| Differentiable | No — text is discrete | Yes — gradients flow through |
| Gradient signal | None through compression | Full backprop through state |
| Cold start | Model must learn format | Autoencoder pretraining |
| Token budget | Explicit (100-200 tokens) | Implicit (512 floats) |
| Information density | Low (~200 tokens of text) | High (512 floats) |

**The key insight:** Text compression can't be trained end-to-end because text is discrete. You can't backprop through "x^2 - 5x + 6 = (x-2)(x-3)". But you CAN backprop through a 512-float vector that represents the same information.

---

## The Energy Landscape (Still Entropy-Based)

Token-level entropy still serves as the energy landscape:

```
H(t) = -sum p(token_i | context) * log p(token_i | context)
```

During generation, model entropy is HIGH (exploring).
During verified mathematical statements, entropy is LOW (committing).

**Difference from text-based:** We no longer need explicit [EXPAND]/[COLLAPSE] markers. The compression happens in latent space automatically. The entropy signal guides what gets preserved in the state vector.

### Training Signal (Phase 2)
```python
reward = answer_correct * compression_quality
       = (1 if \boxed{} matches gold else 0) * state_utilization

# Gradient flows:
# answer_loss -> transformer -> state_tokens -> injector -> state_vector -> compressor
```

---

## Data

### Oracle Collapses (Phase 0)
```
Source:     Strong model solutions segmented at natural joints
Format:     JSONL with collapse_text field
Location:   s3://mycelium-data-v7/breath_data/gsm8k_oracle_v1/breath_sequences.jsonl
Content:    Mathematical statements like "x^2 - 5x + 6 = (x-2)(x-3)"
Re-run through Llama 3.2 1B for 2048-dim hidden states
```

### Breath Sequences (Phase 1-2)
```
Source:     Qwen-0.5B or Llama-3.2-1B-Instruct correct solutions on GSM8K + MATH
Format:     Problem + state history -> next generation
```

---

## File Structure

```
src/
  __init__.py              # Package exports
  compression_head.py      # Perceiver encoder: hidden -> state (d_model=2048)
  state_injector.py        # State -> pseudo-tokens (d_model=2048)
  state_decoder.py         # Decoder for autoencoder (Phase 0 only)
  breathing_model.py       # Full recurrent loop (Llama 3.2 1B)

scripts/
  precompute_hidden.py     # Pre-compute Llama hidden states for autoencoder
  train_autoencoder.py     # Phase 0: autoencoder warmup
  train_two_step.py        # Two-step arithmetic smoke test
  train_three_step.py      # Three-step arithmetic smoke test
  train_alternating.py     # GSM8K with alternating EXPAND/COLLAPSE
  eval_breathing.py        # Evaluation

configs/
  autoencoder.yaml         # Phase 0 config (d_model: 2048)
  two_step.yaml            # Smoke test config
  three_step.yaml          # Smoke test config
  gsm8k_alternating.yaml   # Main GSM8K config
```

---

## Infrastructure

### Models
```
ENGINE:
meta-llama/Llama-3.2-1B (BASE, not Instruct) — production engine
  1.23B params, 9T tokens, 2048 hidden dim, 128K context
  Use bfloat16 for memory efficiency

REFERENCE (SmolLM2 proof-of-concept):
HuggingFaceTB/SmolLM2-135M — proved architecture works on arithmetic
  135M params, 576 hidden dim, 8K context
  Results: 80.4% two-step, 52% three-step
```

### Models on S3
```
REFERENCE (superseded by entropy-based landscape):
s3://mycelium-data-v7/models/e_edge_v1/                    # Gap difficulty (96.5% AUC, hidden states)
s3://mycelium-data-v7/models/e_node_base_v1/                # Joint quality (82%, hidden states)
s3://mycelium-data-v7/models/e_edge_text_v1/                # Text-based E_edge (78% AUC)
s3://mycelium-data-v7/models/grpo_lora_v1/                  # Wave coherence LoRA (reference)

FROZEN (read-only reference):
s3://mycelium-data/models/c1a_coarse_v6_aux_telegraph/      # Boundary detection F1=0.741
s3://mycelium-data/models/c1b_sequence_v5/                  # BP depth prediction
s3://mycelium-data/ib_cluster_to_type.json                  # 8 IB type mapping (3 expand + 5 collapse)

DON'T USE:
s3://mycelium-data-v7/models/cycle*_specialists/            # LoRA specialists (suppress reasoning)
```

### VMs
```
AWS EC2 g5.xlarge (~$1/hr, A10G 24GB) — primary training
AWS EC2 g5.48xlarge (~$16/hr) — large batch generation if needed
SSH: ssh -i ~/.ssh/mycelium-key.pem ubuntu@<IP>
```

---

## Milestones

```
SMOLLM2 PROOF-OF-CONCEPT (COMPLETE):
M0: COMPLETE — SmolLM2-135M single-shot baseline
    GSM8K: 0/1319 = 0.0%, MATH-500: 0/500 = 0.0%

M0.5: COMPLETE — Architecture built
    CompressionHead, StateInjector, StateDecoder, BreathingModel

M0.75: COMPLETE — Oracle collapse data (206K breaths)
    36K correct GSM8K solutions, oracle-optimal compression

M1: COMPLETE — Phase 0 autoencoder warmup (SmolLM2)
    5 epochs, loss 1.59 → 0.44
    t-SNE: structured (variance ratio 0.674)

M2: COMPLETE — Two-step arithmetic (LANDMARK RESULT)
    0% single-shot → 80.4% with breathing
    Ablation: real state 96% vs zeros 6% (+90 point delta)

M3: COMPLETE — Three-step arithmetic
    0% single-shot → 52% with breathing
    Core thesis proven: breathing extends capacity

LLAMA 3.2 1B ENGINE SWAP (CURRENT):
M4: Autoencoder retrain for Llama hidden states
    Pre-compute 206K collapses through Llama 3.2 1B → 2048-dim hidden states
    Retrain encoder+decoder, validate t-SNE structure
    Duration: 2-4 hours

M5: Arithmetic smoke tests (FAST)
    Two-step: expect >70% within 3-5 epochs (faster than SmolLM2)
    Three-step: expect >40% within 5-10 epochs
    Verify ablation delta still large

M6: GSM8K with alternating EXPAND/COLLAPSE breaths (THE MAIN EVENT)
    EXPAND: 512 tokens, parse and explore
    COLLAPSE: 64 tokens, compute and commit
    Target: >15% accuracy on easy GSM8K

M7: Full GSM8K
    512-float state, up to 10 breaths
    Autonomous decomposition: model decides where to split

M8: MATH L1-L3
    512-float state, 10+ breaths

M9: THE HEADLINE — MATH L4-L5
    Non-zero accuracy where single-shot gets 0%
    Capacity extension on competition math
```

---

## The Scoreboard

```
Task                    SmolLM2-135M    Llama-3.2-1B    Status
Two-step arithmetic     80.4%           ???             Smoke test
Three-step arithmetic   52%             ???             Smoke test
Easy GSM8K              3% (failed)     ???             Main event
Full GSM8K              —               ???             Target
MATH L4-5               —               ???             Headline
```

The SmolLM2 column proves the architecture. The Llama column proves the capability. Together they tell the story: differentiable latent compression works at multiple scales, and a capable engine unlocks word problems.

---

## Critical Rules

```
 1. Llama 3.2 1B BASE (not Instruct) — no instruction-following habits
 2. Use bfloat16 for Llama — designed for bf16, fp32 wastes memory
 3. Update tokenizer everywhere — Llama tokenizer != SmolLM2 tokenizer
 4. Final architecture: 64 queries → 512 floats → 8 pseudo-tokens (unchanged)
 5. d_model = 2048 for all compression components (was 576)
 6. Compression head is BREATH-CONDITIONED — queries shift based on breath number
 7. State scale warmup: 0.1 → 1.0 over first 5 epochs (enables co-adaptation)
 8. Breath counter [BREATH n/max] in prompt — transformer knows where it is
 9. Breath embedding in compression head — compressor knows where it is
10. Autoencoder warmup (Phase 0) MUST be retrained for Llama hidden states
11. Pre-compute hidden states before autoencoder training — faster, no transformer in loop
12. Train end-to-end from start — no frozen phase needed (proven by SmolLM2)
13. Gradients flow through state vector — this is the whole point
14. Ablation (real state vs zeros) at every major checkpoint
15. Temperature = 0 for all evaluations
16. N >= 100 problems for any accuracy measurement
17. MATH-500 NEVER in training data
18. Don't skip arithmetic smoke tests — verify architecture works with new engine first
19. Handoff doc before closing any session
```

---

## Bug List

```
CARRIED FROM v10:
27. max_tokens=512 truncates before \boxed{}. Use 2048+.
28. parse_latex() on non-LaTeX corrupts. Only call if contains backslash.
29. Implicit mult breaks function names. Protect reserved words.
30. Verification bottleneck: 29b != 29*b to naive parser.
31. LoRA specialists SUPPRESS reasoning. Don't use.
34. Sudden bottleneck -> collapse (no gradient signal). Fix: gradual training.

NEW FOR v16:
35. Temperature must be 0 (greedy) for eval. 0.7 caused 8->24% baseline shift.
36. vLLM process isolation kills global state. Run bottleneck BETWEEN calls, not inside.
37. Small sample sizes lie. N=20 -> 35%, N=100 -> 26% on same data. Always N>=100.
38. Always verify baseline on SAME problem set before comparing.

NEW FOR v17 (Llama swap):
39. Llama tokenizer is different — update ALL data pipelines, not just model loading.
40. Use BASE model, not Instruct. Instruct has instruction-following habits we don't want.
41. bf16 required for Llama — fp32 doubles memory usage for no benefit.
```

---

## Hardware

```
Training:    Llama 3.2 1B (~2.5GB bf16) + 9.5M params fits on A10G (24GB)
Phase 0:     ~2-4 hours on A10G (pre-compute hidden states + autoencoder)
Phase 1:     ~4-8 hours per difficulty level (end-to-end breathing)
Cost:        ~$75-150 on AWS spot for full curriculum (~3-5 days)
```

---

## Accelerated Curriculum

Llama 3.2 1B already understands math and language. The curriculum is faster because we're only teaching it to breathe, not arithmetic.

```
Day 1 morning:    Pre-compute Llama hidden states + retrain autoencoder (2-4 hours)
Day 1 afternoon:  Two-step arithmetic smoke test (expect 80%+ within 3-5 epochs)
Day 1 evening:    Three-step arithmetic smoke test (expect 50%+ within 5-10 epochs)
Day 2:            GSM8K with alternating EXPAND/COLLAPSE breaths (THE REAL TEST)
Day 3+:           Full GSM8K, then MATH
```

---

## MATH-500 Deadline: April 22, 2026

Engine swap → smoke tests → GSM8K → full GSM8K → MATH.
Target: breathing accuracy > single-shot on L4-L5 problems.
The undeniable result: solving previously unsolvable problems through differentiable compression.
