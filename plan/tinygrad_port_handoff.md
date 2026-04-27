# Handoff: Tinygrad Port of Mycelium

## One-Sentence Summary

Port the Mycelium breathing architecture from PyTorch to tinygrad, enabling training on AMD GPUs (7900 XTX) at ~2x the speed of our current AWS A10G setup. Test on AWS A10G first (NVIDIA CUDA backend), then deploy on local AMD hardware (HIP backend).

---

## Why Tinygrad

1. **AMD GPU access.** Tinygrad runs on AMD GPUs via HIP backend. PyTorch + ROCm is unreliable on consumer AMD cards. Tinygrad was built for this — the tinybox v1 shipped with 7900 XTX GPUs running tinygrad.

2. **2x faster training.** The 7900 XTX has 2x the FLOPS and 1.6x the memory bandwidth of the A10G. Tinygrad achieves full hardware utilization (61 TFLOPS measured on 7900 XTX). PyTorch + ROCm does not.

3. **$0/hr training.** Local hardware eliminates cloud costs. The $2,200 prebuilt pays for itself in ~2,200 hours vs AWS at $1/hr.

4. **Same code, two backends.** Tinygrad code runs on NVIDIA (CUDA) and AMD (HIP) without changes. Develop and test on AWS A10G, deploy on local 7900 XTX.

---

## Repo Structure

Keep PyTorch and tinygrad code SEPARATE to avoid confusion:

```
mycelium/
  scripts/                    # PyTorch code (existing, don't touch)
    atom_lora.py              #   model definition
    train_per_cycle.py        #   training loop
    generate_per_cycle_data.py
    parse_gsm8k.py
    ...
    
  tinygrad_port/              # NEW: tinygrad code goes here
    model.py                  #   model definition (tinygrad)
    train.py                  #   training loop (tinygrad)
    llama.py                  #   Llama loader (tinygrad, based on tinygrad/examples/llama.py)
    perceiver.py              #   7-layer perceiver (tinygrad)
    hypernetwork.py           #   100M atom hypernetwork (tinygrad)
    answer_head.py            #   answer head + confidence head (tinygrad)
    lora.py                   #   64 LoRA atoms + additive injection (tinygrad)
    losses.py                 #   all loss functions (tinygrad)
    data.py                   #   data loading (shared with PyTorch, minimal changes)
    utils.py                  #   normalization, isotropic reg, etc.
    convert_checkpoint.py     #   convert PyTorch checkpoint → tinygrad format
    validate_port.py          #   compare PyTorch vs tinygrad outputs
    README.md                 #   tinygrad-specific setup and usage
    
  data/                       # SHARED: training data (format-agnostic JSONL)
    per_cycle/
    gsm8k/
    
  checkpoints/                # SHARED: model weights
    pytorch/                  #   PyTorch .pt files
    tinygrad/                 #   tinygrad safetensor files
    
  plan/                       # SHARED: design docs
    tinygrad_port_handoff.md  #   this document
```

The data/ directory is shared — JSONL files don't care about the framework. Checkpoints are separate because the formats differ. The model code is completely separate.

---

## Porting Guide: PyTorch → Tinygrad

### Core API Mapping

```python
# ============ Tensor Creation ============
# PyTorch                           # tinygrad
import torch                        from tinygrad import Tensor, dtypes
torch.randn(64, 64)                 Tensor.randn(64, 64)
torch.zeros(64)                     Tensor.zeros(64)
torch.ones(64)                      Tensor.ones(64)
torch.tensor([1, 2, 3])             Tensor([1, 2, 3])
x.to(device)                        # automatic, set DEVICE env var
x.float()                           x.cast(dtypes.float32)
x.half()                            x.cast(dtypes.float16)
x.bfloat16()                        x.cast(dtypes.bfloat16)

# ============ Operations ============
x.matmul(y)                         x.matmul(y)   # or x @ y
x.sum(dim=-1)                       x.sum(axis=-1)
x.mean(dim=0)                       x.mean(axis=0)
x.var(dim=0)                        x.var(axis=0)
x.reshape(B, -1)                    x.reshape(B, -1)
x.transpose(0, 1)                   x.transpose(0, 1)  # or x.permute(...)
x.unsqueeze(0)                      x.unsqueeze(0)
x.squeeze(-1)                       x.squeeze(-1)
torch.cat([a, b], dim=-1)           a.cat(b, dim=-1)
torch.stack([a, b])                 Tensor.stack([a, b])
x.detach()                          x.detach()
x.contiguous()                      x.contiguous()

# ============ Activations ============
F.gelu(x)                           x.gelu()
F.relu(x)                           x.relu()
F.sigmoid(x)                        x.sigmoid()
F.tanh(x)                           x.tanh()
F.softmax(x, dim=-1)                x.softmax(axis=-1)
F.log_softmax(x, dim=-1)            x.log_softmax(axis=-1)

# ============ Layers ============
nn.Linear(64, 128)                  # Manual: W = Tensor.randn(128, 64); b = Tensor.zeros(128)
                                    # Or use tinygrad.nn: from tinygrad.nn import Linear
nn.LayerNorm(64)                    # tinygrad.nn.LayerNorm(64)
nn.Embedding(100, 64)               # tinygrad.nn.Embedding(100, 64)

# ============ Loss ============
F.cross_entropy(logits, target)     logits.sparse_categorical_crossentropy(target)
F.binary_cross_entropy(pred, tgt)   # Manual: -(tgt * pred.log() + (1-tgt) * (1-pred).log()).mean()
F.mse_loss(pred, target)            (pred - target).square().mean()
F.cosine_similarity(a, b, dim=-1)   # Manual: (a * b).sum(axis=-1) / (a.norm(axis=-1) * b.norm(axis=-1))

# ============ Optimizer ============
torch.optim.AdamW(params, lr=1e-3)  from tinygrad.nn.optim import AdamW
                                    AdamW(params, lr=1e-3)

# ============ Autograd ============
loss.backward()                     loss.backward()
optimizer.step()                    optimizer.step()
optimizer.zero_grad()               optimizer.zero_grad()

# ============ Save/Load ============
torch.save(state_dict, path)        from tinygrad.nn.state import safe_save, safe_load, get_state_dict
                                    safe_save(get_state_dict(model), path)
torch.load(path)                    safe_load(path)
```

### Key Differences to Watch

```
1. DEVICE MANAGEMENT
   PyTorch:  x = x.to('cuda')
   tinygrad: set DEVICE=CUDA (env var) or DEVICE=HIP for AMD
             Tensors are created on the active device automatically.

2. NO nn.Module
   PyTorch:  class MyModel(nn.Module): def __init__(self): ...
   tinygrad: plain Python classes. Parameters are just Tensor attributes.
             Use nn.state.get_parameters(model) to collect all params.

3. LAZY EVALUATION
   tinygrad tensors are LAZY — computation doesn't happen until .realize()
   or .numpy() is called. This enables kernel fusion but means you need
   to .realize() explicitly when you want results (e.g., for printing losses).

   loss_val = loss.realize().numpy()  # force computation
   
4. NO DataLoader
   Write your own batching. tinygrad doesn't have a DataLoader.
   Just load JSONL, batch with numpy, convert to Tensor.

5. GRADIENT CHECKPOINTING
   Not built in. May need manual implementation if OOM.
   Start without it — the 7900 XTX has 24GB, same as A10G.
```

---

## Porting Order (Component by Component)

### Phase 1: Core Model (2-3 hours)

Port the model components in isolation. Test each one independently.

```
1. lora.py — LoRA atoms + additive injection
   The core math: q = W_q @ x + sum(scale_i * (x @ B_i.T) @ A_i.T)
   Pure matrix operations. Straightforward port.
   Test: random input → compare output with PyTorch version.

2. perceiver.py — 7-layer perceiver compressor
   Cross-attention + self-attention + LayerNorm + MLP.
   Use tinygrad.nn.LayerNorm and manual attention.
   Test: random hidden states → compare 64-float page output.

3. hypernetwork.py — 100M atom hypernetwork
   Cross-attention over notebook pages → 64 tanh scales.
   Variable-length input (notebook grows).
   Test: random pages → compare 64 atom scales.

4. answer_head.py — answer head + confidence head
   Linear layers + GELU + classification heads.
   Test: random page → compare digit predictions.
```

### Phase 2: Llama Loading (2-3 hours)

```
5. llama.py — Load Llama 3.2 1B weights
   Reference: tinygrad/examples/llama.py (already exists!)
   Load safetensor weights directly (no HuggingFace dependency).
   Freeze all Llama params (don't include in optimizer).
   Test: tokenize "2+2=" → compare logits with PyTorch version.
```

### Phase 3: Full Model Assembly (2-3 hours)

```
6. model.py — Assemble the full breathing model
   Wire together: Llama + LoRA atoms + perceiver + hypernetwork + heads.
   Implement think_one_pass() with growing notebook.
   Test: full forward pass on one problem → compare with PyTorch.
```

### Phase 4: Training Loop (3-4 hours)

```
7. losses.py — All loss functions
   Generation loss (cross-entropy on tokens).
   Answer head loss (per-digit classification).
   Flexible loss with consumption.
   Isotropic regularizer.
   Confidence entropy regularizer.
   Contrastive loss.
   Scale regularizer.

8. train.py — Training loop
   Data loading (read JSONL, batch, tokenize).
   Per-cycle forward pass with text injection.
   Loss computation with smooth fading.
   Backward + optimizer step.
   Logging + checkpointing.

9. data.py — Data loading
   Read JSONL files (shared format with PyTorch).
   Tokenize with sentencepiece (same tokenizer).
   Batch and pad sequences.
   No DataLoader — simple Python generator.
```

### Phase 5: Validation (2-3 hours)

```
10. convert_checkpoint.py — Convert PyTorch checkpoint to tinygrad
    Load PyTorch state_dict, save as safetensor for tinygrad.
    Map parameter names between the two implementations.

11. validate_port.py — Verify outputs match
    Load same checkpoint in both frameworks.
    Run same input through both.
    Compare outputs (should be identical within float precision).
    Compare loss values on same batch.
    Compare gradient norms.
```

**Total estimated porting time: 2-3 days**

---

## Validation Checklist

Before switching to tinygrad for real training:

```
□ Forward pass outputs match PyTorch (within 1e-5)
□ Loss values match on same batch (within 1e-4)
□ Gradient norms are similar (within 10%)
□ One epoch produces similar accuracy trajectory
□ Memory usage fits in 24GB (A10G or 7900 XTX)
□ Training speed is measured:
    tinygrad on A10G: ??? min/epoch
    PyTorch on A10G:  42 min/epoch (known baseline)
□ Checkpoint save/load works
□ Text injection works correctly
□ Flexible loss with consumption works
□ Generation + answer head extraction work
```

---

## Llama Weight Loading

Tinygrad has a Llama example that loads weights directly from safetensors. No HuggingFace transformers dependency needed:

```python
# tinygrad Llama loading pattern (from tinygrad/examples/llama.py)
from tinygrad.nn.state import safe_load
from tinygrad import Tensor

# Load weights directly from safetensor files
weights = safe_load("/path/to/llama-3.2-1b/model.safetensors")

# Map weights to model attributes
model.tok_embeddings.weight = weights['model.embed_tokens.weight']
model.layers[0].attention.wq.weight = weights['model.layers.0.self_attn.q_proj.weight']
# ... etc

# Freeze all Llama params (exclude from optimizer)
llama_params = set(get_parameters(model.llama))
trainable_params = [p for p in get_parameters(model) if p not in llama_params]
optimizer = AdamW(trainable_params, lr=1e-3)
```

### Tokenizer

Use sentencepiece directly (no HuggingFace tokenizer dependency):

```python
import sentencepiece as sp
tokenizer = sp.SentencePieceProcessor()
tokenizer.Load("/path/to/llama-3.2-1b/tokenizer.model")
tokens = tokenizer.Encode("Jamie had 160 cookies")
text = tokenizer.Decode(tokens)
```

---

## Hardware Deployment Plan

```
PHASE 1: Port + Test on AWS A10G (no new hardware)
  - Port code to tinygrad
  - Test with DEVICE=CUDA on existing A10G instance
  - Compare: tinygrad vs PyTorch speed and accuracy
  - Cost: just AWS hours (same as current)

PHASE 2: Validate on AMD (optional, use son's PC or cheap cloud)
  - Test forward pass with DEVICE=HIP
  - Verify same outputs on AMD GPU
  - Don't need 24GB for validation — batch_size=1 might fit

PHASE 3: Buy + Deploy (after validation)
  - Purchase PRC ShadowGlass ($2,199) or equivalent 7900 XTX prebuilt
  - Install Ubuntu + tinygrad
  - Copy code + checkpoints
  - Train GSM8K locally: ~20 min/epoch, $0/hr
  
PHASE 4: Keep AWS as backup
  - Use AWS for burst capacity or parallel runs
  - Same tinygrad code runs on both (CUDA on AWS, HIP on local)
```

---

## Environment Setup

### AWS A10G (NVIDIA — for testing)

```bash
# On existing EC2 instance
pip install tinygrad --break-system-packages
# Set NVIDIA backend
export DEVICE=CUDA

# Test basic operation
python -c "from tinygrad import Tensor; print(Tensor.randn(64,64).sum().numpy())"

# Run validation
cd mycelium/tinygrad_port
python validate_port.py --pytorch-checkpoint ../checkpoints/pytorch/best.pt
```

### Local 7900 XTX (AMD — for training)

```bash
# On prebuilt PC after Ubuntu install
# Install AMD ROCm/HIP (tinygrad handles the rest)
sudo apt install rocm-hip-runtime

pip install tinygrad
export DEVICE=HIP

# Test
python -c "from tinygrad import Tensor; print(Tensor.randn(64,64).sum().numpy())"

# Train
cd mycelium/tinygrad_port
python train.py --data ../data/gsm8k/ --checkpoint ../checkpoints/tinygrad/warm_start.safetensor
```

---

## Checkpoint Conversion

PyTorch and tinygrad use different formats. We need a converter:

```python
# convert_checkpoint.py
import torch
from tinygrad import Tensor
from tinygrad.nn.state import safe_save

def convert_pytorch_to_tinygrad(pytorch_path, tinygrad_path):
    """Convert PyTorch .pt checkpoint to tinygrad safetensor format."""
    
    # Load PyTorch checkpoint
    pt_state = torch.load(pytorch_path, map_location='cpu')
    
    # Convert each parameter
    tg_state = {}
    for name, param in pt_state.items():
        # Convert torch tensor → numpy → tinygrad tensor
        tg_state[name] = Tensor(param.numpy())
    
    # Save in safetensor format
    safe_save(tg_state, tinygrad_path)
    print(f"Converted {len(tg_state)} parameters")

# Usage:
# python convert_checkpoint.py \
#   --input checkpoints/pytorch/gsm8k_best.pt \
#   --output checkpoints/tinygrad/gsm8k_best.safetensor
```

---

## Risk Assessment

```
LOW RISK:
  - Basic operations (matmul, linear, attention) — well tested in tinygrad
  - Llama loading — tinygrad has a working llama.py example
  - AdamW optimizer — built into tinygrad
  - NVIDIA backend (CUDA) — mature and stable

MEDIUM RISK:
  - Complex training loop (flexible loss, consumption, text injection)
    More code to port, more places for bugs. Validate carefully.
  - Performance on A10G — tinygrad might be slower than PyTorch on NVIDIA
    (tinygrad is optimized for AMD, NVIDIA is mature in PyTorch)
  - Gradient checkpointing if needed — may need manual implementation

LOW-MEDIUM RISK:
  - AMD backend (HIP) — tinybox uses this in production, but consumer
    7900 XTX drivers can have rough edges. The tinygrad community
    actively supports this GPU specifically.

MITIGATION:
  - Test on A10G FIRST (same GPU, direct comparison)
  - Keep PyTorch code untouched (can always fall back)
  - Port incrementally, validate each component
  - Don't buy hardware until port is validated
```

---

## What NOT to Do

```
- Do NOT modify the PyTorch code during porting.
  The PyTorch version is our known-good baseline.
  Keep it exactly as-is for comparison.

- Do NOT try to share code between PyTorch and tinygrad.
  Clean separation is easier to debug than clever abstractions.
  Each framework has its own idioms. Respect them.

- Do NOT port everything at once.
  Port component by component. Test each one.
  A bug in the LoRA injection is easier to find in isolation.

- Do NOT buy hardware before validating the port.
  Test on AWS A10G first. If tinygrad has issues, we find out for $1/hr
  not $2,200.

- Do NOT optimize prematurely.
  Get it CORRECT first. Make it FAST later.
  Tinygrad's lazy evaluation and kernel fusion handle most optimization.
```

---

## Success Criteria

```
The port is DONE when:
  1. validate_port.py shows outputs match within float precision
  2. One full epoch on GSM8K produces similar loss trajectory
  3. Memory fits in 24GB with batch_size=4
  4. Speed on A10G is within 1.5x of PyTorch (ideally faster)
  
The hardware purchase is JUSTIFIED when:
  5. Port works on A10G (CUDA backend)
  6. Estimated speedup on 7900 XTX is >= 1.5x
  7. Breakeven vs AWS is < 6 months at current usage rate
```
