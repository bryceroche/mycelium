# Breathing Models — Dual LoRA Verification & Dynamic Cycles

## Overview

Two sets of LoRA templates — forward (computation) and verification (consistency checking) — blended smoothly by a learned weight. The model gradually transitions from building an answer to checking it. A confidence head reads the page notebook and learns when to stop — after the model has both computed AND verified.

This is the geometric mirror: forward computation and its reflection (verification) are two modes of attention operating on the same problem. The blend weight controls the transition between them, like rotating from one hemisphere of the hypersphere to the other.

---

## The Dual LoRA Architecture

### Two Template Sets, One Hypernetwork

```
Forward templates (rank 4, ~1.1M params):
  Learned attention patterns for sequential computation.
  "Focus on this number, apply this operation, produce a result."
  Narrow, step-by-step attention.

Verification templates (rank 4, ~1.1M params):
  Learned attention patterns for consistency checking.
  "Look at all numbers simultaneously, check relationships hold."
  Broad, relational attention.

Hypernetwork generates scales for BOTH + a blend weight:
  page_summary + strategy → forward_scales (256) + verify_scales (256) + blend (1)
```

### Blended Application

```python
class DualLoRA(nn.Module):
    def __init__(self, d_model=2048, rank=4, num_layers=16):
        super().__init__()
        proj_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        
        # Forward templates: computation-focused attention
        self.A_forward = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, d_model, rank) * 0.01)
            for _ in proj_names
        ])
        self.B_forward = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, rank, self._proj_dim(name, d_model)) * 0.01)
            for name in proj_names
        ])
        
        # Verification templates: consistency-checking attention
        self.A_verify = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, d_model, rank) * 0.01)
            for _ in proj_names
        ])
        self.B_verify = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, rank, self._proj_dim(name, d_model)) * 0.01)
            for name in proj_names
        ])
    
    def _proj_dim(self, proj_name, d_model):
        """Handle GQA: K,V projections are smaller in Llama 3.2 1B."""
        if proj_name in ['k_proj', 'v_proj']:
            return 512  # 8 KV heads × 64 dim
        return d_model   # 32 Q heads × 64 dim or full O projection
```

### Hypernetwork With Blend

```python
class BlendedHypernetwork(nn.Module):
    def __init__(self, page_summary_dim=1024, strategy_dim=512, num_scales=256):
        super().__init__()
        input_dim = page_summary_dim + strategy_dim  # 1536
        
        self.scales_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, num_scales + num_scales + 1),  # forward + verify + blend
        )
    
    def forward(self, page_summary, strategy):
        """
        Returns forward_scales, verify_scales, and blend weight.
        blend: 0 = pure forward, 1 = pure verification
        """
        out = self.scales_net(torch.cat([page_summary, strategy], dim=-1))
        
        forward_scales = torch.tanh(out[:, :256])
        verify_scales = torch.tanh(out[:, 256:512])
        blend = torch.sigmoid(out[:, 512:513])  # smooth 0-1 transition
        
        return forward_scales, verify_scales, blend
```

### Additive LoRA With Blending

```python
def apply_blended_lora(hidden, layer_idx, proj_name, proj_idx,
                        forward_scales, verify_scales, blend,
                        A_forward, B_forward, A_verify, B_verify):
    """
    Apply blended forward + verification LoRA as additive term.
    No hooks, no weight modification.
    """
    # Forward path
    A_f = A_forward[proj_idx][layer_idx]          # (d_model, rank)
    B_f = B_forward[proj_idx][layer_idx]          # (rank, proj_dim)
    s_f = forward_scales_for_layer_proj            # (batch, rank)
    q_forward = (hidden @ A_f) * s_f @ B_f        # (batch, seq, proj_dim)
    
    # Verification path
    A_v = A_verify[proj_idx][layer_idx]            # (d_model, rank)
    B_v = B_verify[proj_idx][layer_idx]            # (rank, proj_dim)
    s_v = verify_scales_for_layer_proj             # (batch, rank)
    q_verify = (hidden @ A_v) * s_v @ B_v          # (batch, seq, proj_dim)
    
    # Blend: smooth transition between computation and verification
    q_lora = (1 - blend) * q_forward + blend * q_verify
    
    return q_lora  # added to the original projection output
```

---

## The Blend Weight In Action

The blend weight creates a smooth trajectory from computation to verification:

```
Cycle 1: blend ≈ 0.05  (95% forward, 5% verify)
         Pure computation. Parse the problem, extract numbers.
         Verification templates barely active.

Cycle 2: blend ≈ 0.15  (85% forward, 15% verify)
         Mostly computing. Starting to notice consistency.
         "I'm computing 48/2=24 and already checking it makes sense."

Cycle 3: blend ≈ 0.40  (60% forward, 40% verify)
         Computing the final step while actively checking intermediates.
         "48+24=72. Does 24 = half of 48? Yes."

Cycle 4: blend ≈ 0.85  (15% forward, 85% verify)
         Mostly verifying. Broad attention over all numbers.
         "72 total. 48 April + 24 May. 24 = 48/2. All consistent."

Cycle 5: blend ≈ 0.95  (5% forward, 95% verify)
         Pure verification. Double-checking everything.
         Confidence head: "I've computed AND verified. Ready."
```

The model doesn't hard-switch between modes. It GRADUALLY transitions from building to checking. The blend is learned end-to-end — the model discovers when to shift through gradient signal from correct/incorrect answers.

---

## Dynamic Cycle Count (Confidence Head Returns)

### Why We Disabled It Before

The confidence head was disabled because:
1. The efficiency penalty (0.01 × num_passes) made the model always stop at 1 pass
2. The confidence head learned to output 0.9+ immediately to avoid the penalty
3. Fixed passes at 3 was more stable

### Why It Works Now

With the page-based architecture and dual LoRA, the confidence head has better information:

```
Before: confidence read a single 64-float state (limited signal)
Now:    confidence reads ALL pages via attention (rich signal)
        + the blend weight tells it whether verification has happened
```

The confidence head can learn a more nuanced policy:

```
"I see 3 pages of forward computation (blend < 0.5) → NOT confident
 even if the numbers look right — haven't verified yet."

"I see 3 forward pages + 1 verification page (blend > 0.8) 
 AND the verification page is consistent → CONFIDENT"
```

The confidence head effectively learns: **don't stop until you've verified.**

### Implementation

```python
class PageConfidenceHead(nn.Module):
    def __init__(self, page_size=64, hidden=128):
        super().__init__()
        self.page_project = nn.Linear(page_size, hidden)
        self.blend_project = nn.Linear(1, hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, hidden))
        self.output = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, state_pages, blend_history):
        """
        state_pages: list of (batch, 64) page tensors
        blend_history: list of (batch, 1) blend values from each cycle
        """
        pages = torch.stack(state_pages, dim=1)
        pages_proj = self.page_project(pages)  # (batch, num_pages, hidden)
        
        # Inject blend information into each page's representation
        blends = torch.stack(blend_history, dim=1)
        blend_proj = self.blend_project(blends)  # (batch, num_pages, hidden)
        pages_proj = pages_proj + blend_proj  # pages know their computation/verification mix
        
        batch_size = pages.size(0)
        q = self.query.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.attn(query=q, key=pages_proj, value=pages_proj)
        
        return self.output(attended.squeeze(1))  # (batch, 1) confidence
```

### Training The Confidence Head

```
NO efficiency penalty. The model should think as long as it needs.

Instead, train confidence with a CORRECTNESS signal:

1. Run N cycles (e.g., max 8)
2. At each cycle, generate an answer using pseudo-tokens from current pages
3. Check if the answer is correct
4. Confidence target = 1.0 if answer is correct, 0.0 if wrong

The confidence head learns: "at what point in the thinking process
does the answer become correct?" It naturally discovers that
answers become correct AFTER verification cycles (blend > 0.5).
```

```python
def train_confidence(model, problem, gold_answer, max_cycles=8):
    state_pages = []
    blend_history = []
    confidence_targets = []
    
    for cycle in range(max_cycles):
        # Run thinking cycle
        page, strategy, blend = model.think_one_cycle(state_pages)
        state_pages.append(page)
        blend_history.append(blend)
        
        # Check: would the current pages produce a correct answer?
        with torch.no_grad():
            pseudo_tokens = model.page_to_tokens(state_pages)
            answer = model.generate(pseudo_tokens, problem)
            is_correct = (extract_number(answer) == gold_answer)
        
        confidence_targets.append(float(is_correct))
    
    # Train confidence head to predict correctness at each cycle
    confidence_loss = 0
    for i in range(len(state_pages)):
        partial_pages = state_pages[:i+1]
        partial_blends = blend_history[:i+1]
        predicted_conf = model.confidence_head(partial_pages, partial_blends)
        target = torch.tensor([[confidence_targets[i]]], device=device)
        confidence_loss += F.binary_cross_entropy(predicted_conf, target)
    
    return confidence_loss / len(state_pages)
```

### Inference With Dynamic Cycles

```python
def solve(model, problem, max_cycles=8, confidence_threshold=0.85):
    state_pages = []
    blend_history = []
    min_cycles = 2  # always think at least twice
    
    for cycle in range(max_cycles):
        page, strategy, blend = model.think_one_cycle(state_pages)
        state_pages.append(page)
        blend_history.append(blend)
        
        # Only check confidence after minimum cycles
        if cycle >= min_cycles - 1:
            conf = model.confidence_head(state_pages, blend_history)
            if conf > confidence_threshold:
                break
    
    # Generate with pseudo-tokens (LoRA OFF)
    pseudo_tokens = model.page_to_tokens(state_pages)
    answer = model.generate(pseudo_tokens, problem)
    
    return answer, len(state_pages), conf
```

---

## The Geometric Interpretation

```
FORWARD COMPUTATION (blend ≈ 0):
  The state trajectory moves through "computation space"
  Each page is a step forward in the solution
  Attention is narrow, sequential, step-by-step
  
TRANSITION (blend ≈ 0.5):
  The state crosses the "equator" between computation and verification
  Attention broadens, starting to check relationships
  The model is simultaneously finishing computation and beginning verification
  
VERIFICATION (blend ≈ 1):
  The state moves through "verification space"
  Each page is a consistency check
  Attention is broad, relational, holistic
  
REFLECTION = the moment blend crosses 0.5
  Forward thinking reflects into backward checking
  The mirror point on the hypersphere
```

The blend weight traces a sigmoid curve across cycles — starting near 0, crossing 0.5 at the reflection point, approaching 1. The model's thinking has a natural arc: build, transition, verify.

---

## Expected Behavior Across Problem Difficulty

```
EASY problem (e.g., "5 + 3 = ?"):
  Cycle 1: blend=0.1, compute 5+3=8
  Cycle 2: blend=0.8, verify 8=5+3 ✓
  Confidence: 0.95 → STOP after 2 cycles

MEDIUM problem (e.g., "48 clips, half in May, total?"):
  Cycle 1: blend=0.05, parse numbers (48, half)
  Cycle 2: blend=0.15, compute 48/2=24
  Cycle 3: blend=0.35, compute 48+24=72
  Cycle 4: blend=0.80, verify: 24=48/2 ✓, 72=48+24 ✓
  Confidence: 0.90 → STOP after 4 cycles

HARD problem (e.g., 5-step GSM8K):
  Cycle 1-3: blend=0.05-0.20, forward computation
  Cycle 4-5: blend=0.30-0.50, computing while checking
  Cycle 6-7: blend=0.70-0.90, mostly verifying
  Cycle 8: blend=0.95, final verification
  Confidence: 0.88 → STOP after 8 cycles
```

The model allocates compute proportional to difficulty — not because we told it to, but because the confidence head learned when verification is sufficient. Easy problems verify fast. Hard problems need more cycles.

---

## Parameter Budget

```
Component                         Params      % of total
──────────────────────────────────────────────────────────
Llama 3.2 1B (frozen)             1.23B       91.7%
7-Layer Perceiver                 105M        7.8%
Forward LoRA templates            1.1M        0.08%
Verification LoRA templates       1.1M        0.08%  ← NEW
Blended Hypernetwork              ~650K       0.05%
Page-to-Tokens                    ~550K       0.04%
Page Confidence Head              ~20K        0.00%
──────────────────────────────────────────────────────────
Total:                            ~1.34B
New verification params:          ~1.1M       <0.1%
```

---

## When to Implement

```
PHASE 1 (NOW): 
  Train page-based architecture on GSM8K with single LoRA
  Prove pages work on word problems
  Establish GSM8K accuracy baseline

PHASE 2 (AFTER GSM8K BASELINE):
  Add verification LoRA templates + blend weight
  Retrain on GSM8K
  Compare: does dual LoRA beat single LoRA?
  Does the blend naturally shift from computation to verification?

PHASE 3 (AFTER VERIFICATION PROVEN):
  Re-enable confidence head with page attention + blend awareness
  Train with correctness signal (not efficiency penalty)
  Dynamic cycles: easy=2, medium=4, hard=8
  
PHASE 4 (MATH-500):
  Full architecture: pages + dual LoRA + dynamic cycles
  Competition math benefits most from verification
  (verification catches errors on hard multi-step problems)
```

---

## What to Monitor When Implemented

```
1. Blend trajectory: does blend naturally rise across cycles? (sigmoid shape expected)
2. Forward vs verify template norms: do they diverge? (should specialize differently)
3. Accuracy at verification cycles: does accuracy INCREASE when blend > 0.5?
4. Confidence calibration: does confidence correlate with actual correctness?
5. Cycle allocation: do hard problems use more cycles than easy ones?
6. Error recovery: does the model catch errors during verification that it missed during computation?
```
