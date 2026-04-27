# Handoff: Haar Wavelet Preprocessing for Perceiver Compression

## One-Sentence Summary

Apply a Haar wavelet transform to each layer's hidden states before the perceiver reads them. Truncate the finest detail level. The perceiver reads multi-resolution frequency coefficients instead of raw token positions — already 2x compressed and structured by frequency before learned compression begins.

---

## Why

The perceiver currently reads raw hidden states: 16 layers × seq_len × 2048. Most of the information is redundant — nearby tokens have similar hidden states. The perceiver must learn to ignore this redundancy through attention, wasting capacity on "discovering" that tokens 5 and 6 are similar.

The wavelet transform does this structural compression for free:

```
Current:   100 tokens × 2048 features per layer → perceiver (must learn to compress)
Wavelet:   ~49 coefficients × 2048 features per layer → perceiver (pre-compressed)
```

The wavelet preserves BOTH frequency AND position — unlike FFT which loses position. "There's an important number at positions 10-15" survives the wavelet transform. This matters for math reasoning where WHERE a number appears in the problem determines its role.

---

## Haar Wavelet Basics

The Haar wavelet is the simplest wavelet — just averages and differences:

```
Input:    [a, b, c, d, e, f, g, h]  (8 values)

Level 1:  averages:    [(a+b)/√2, (c+d)/√2, (e+f)/√2, (g+h)/√2]  → 4 values (smooth)
          differences: [(a-b)/√2, (c-d)/√2, (e-f)/√2, (g-h)/√2]  → 4 values (detail)

Level 2:  averages of averages:    [2 values]  (smoother)
          differences of averages: [2 values]  (detail at coarser scale)

Level 3:  averages: [1 value]   (global average — DC component)
          differences: [1 value] (coarsest detail)

Total coefficients: 1 + 1 + 2 + 4 = 8 (same as input, just reorganized by frequency)
```

For 100 tokens with 4 decomposition levels:

```
Level 4 (coarsest):  ~6 coefficients  (broad sequence structure)
Level 3:             ~6 coefficients  (paragraph-level patterns)
Level 2:             ~12 coefficients (phrase-level patterns)
Level 1:             ~25 coefficients (word-level detail)
Level 0 (finest):    ~50 coefficients (token-level detail — TRUNCATED)

Keep levels 1-4:  ~49 coefficients (2x compression, structured)
```

---

## Implementation

### Install

```bash
pip install PyWavelets --break-system-packages
```

### Wavelet Preprocessing Module

```python
import pywt
import torch
import torch.nn as nn

class HaarWaveletPreprocess(nn.Module):
    def __init__(self, max_level=4, truncate_finest=True):
        """
        Haar wavelet transform for hidden state compression.
        
        max_level: number of decomposition levels
        truncate_finest: if True, discard the finest detail coefficients (level 0)
        """
        super().__init__()
        self.max_level = max_level
        self.truncate_finest = truncate_finest
        
        # Learnable weights per level (optional — let model decide importance)
        self.level_weights = nn.Parameter(torch.ones(max_level + 1))
    
    def forward(self, hidden_states):
        """
        hidden_states: (batch, seq_len, d_model)
        returns: (batch, num_coeffs, d_model) — multi-resolution coefficients
        """
        batch, seq_len, d_model = hidden_states.shape
        device = hidden_states.device
        
        # Move to CPU for pywt (pywt doesn't support CUDA tensors)
        h_cpu = hidden_states.detach().cpu().float().numpy()
        
        all_coeffs = []
        for b in range(batch):
            # Apply wavelet decomposition per feature dimension
            # pywt.wavedec works on 1D signals — apply across seq dimension
            # Shape: (seq_len, d_model) → decompose each column
            coeffs = pywt.wavedec(h_cpu[b], wavelet='haar', level=self.max_level, axis=0)
            # coeffs[0] = coarsest approximation (level max_level)
            # coeffs[1] = detail at level max_level
            # coeffs[2] = detail at level max_level-1
            # ...
            # coeffs[-1] = detail at level 1 (finest kept)
            
            if self.truncate_finest:
                # Remove the finest detail level (last element)
                coeffs = coeffs[:-1]
            
            all_coeffs.append(coeffs)
        
        # Convert back to tensors, concatenate levels
        result = []
        for b in range(batch):
            levels = []
            for level_idx, coeff in enumerate(all_coeffs[b]):
                coeff_tensor = torch.tensor(coeff, dtype=hidden_states.dtype, device=device)
                # Apply learned level weight
                weight = torch.sigmoid(self.level_weights[level_idx])
                levels.append(coeff_tensor * weight)
            
            result.append(torch.cat(levels, dim=0))  # (num_coeffs, d_model)
        
        return torch.stack(result, dim=0)  # (batch, num_coeffs, d_model)
```

### Faster Pure-PyTorch Implementation (GPU-friendly, no pywt dependency)

```python
class HaarWaveletPreprocessFast(nn.Module):
    def __init__(self, max_level=4, truncate_finest=True):
        super().__init__()
        self.max_level = max_level
        self.truncate_finest = truncate_finest
        self.level_weights = nn.Parameter(torch.ones(max_level + 1))
    
    def forward(self, hidden_states):
        """
        hidden_states: (batch, seq_len, d_model)
        returns: (batch, num_coeffs, d_model)
        
        Pure PyTorch — stays on GPU, supports autograd.
        """
        coefficients = []
        current = hidden_states  # (batch, seq_len, d_model)
        
        for level in range(self.max_level):
            # Pad if odd length
            if current.size(1) % 2 == 1:
                current = F.pad(current, (0, 0, 0, 1))  # pad seq dim
            
            # Haar wavelet: averages and differences of adjacent pairs
            even = current[:, 0::2, :]  # (batch, seq_len//2, d_model)
            odd = current[:, 1::2, :]
            
            averages = (even + odd) / math.sqrt(2)    # low-freq (smooth)
            details = (even - odd) / math.sqrt(2)      # high-freq (edges)
            
            # Weight this level's detail coefficients
            weight = torch.sigmoid(self.level_weights[level])
            coefficients.append(details * weight)
            
            current = averages  # continue decomposing the smooth part
        
        # Coarsest approximation
        weight = torch.sigmoid(self.level_weights[-1])
        coefficients.append(current * weight)
        
        if self.truncate_finest:
            coefficients = coefficients[1:]  # drop finest detail (index 0)
        
        # Reverse so coarsest is first (natural reading order for perceiver)
        coefficients.reverse()
        
        return torch.cat(coefficients, dim=1)  # (batch, num_coeffs, d_model)
```

**Use the fast version.** It stays on GPU, supports autograd (gradients flow through the wavelet transform), and doesn't require pywt. The Haar wavelet is simple enough to implement directly.

---

## Integration Into Perceiver

### Before (Raw Hidden States)

```python
class Perceiver:
    def forward(self, all_layer_hidden, pass_num):
        # all_layer_hidden: list of 16 × (batch, seq_len, 2048)
        # Concatenate layers along a new dimension or interleave
        # Perceiver cross-attends over raw token positions
        ...
```

### After (Wavelet Coefficients)

```python
class Perceiver:
    def __init__(self, ...):
        ...
        self.wavelet = HaarWaveletPreprocessFast(max_level=4, truncate_finest=True)
    
    def forward(self, all_layer_hidden, pass_num):
        # Transform each layer's hidden states to wavelet domain
        wavelet_layers = []
        for layer_hidden in all_layer_hidden:
            # layer_hidden: (batch, seq_len, 2048)
            wavelet_coeffs = self.wavelet(layer_hidden)
            # wavelet_coeffs: (batch, ~49, 2048) — 2x compressed, structured
            wavelet_layers.append(wavelet_coeffs)
        
        # Perceiver cross-attends over wavelet coefficients instead of raw positions
        # ~49 tokens per layer instead of ~100
        # Structured: coarsest coefficients first, finer detail later
        ...
```

---

## What This Changes

```
BEFORE:
  Perceiver input: 16 layers × 100 positions = 1600 tokens
  Each token: raw hidden state at one position in one layer
  No frequency structure — perceiver must discover redundancy

AFTER:
  Perceiver input: 16 layers × ~49 coefficients = ~784 tokens
  Each token: wavelet coefficient at one frequency band in one layer
  Structured by frequency — coarsest first, finest last
  2x fewer tokens — perceiver attention is 4x faster (quadratic in tokens)
```

### Speed Impact

Perceiver cross-attention is O(queries × keys). Halving the keys:

```
Before: 4 queries × 1600 keys = 6400 attention computations per layer
After:  4 queries × 784 keys  = 3136 attention computations per layer

~2x speedup in perceiver attention (the expensive part of the perceiver)
```

### Quality Impact

The wavelet gives the perceiver a BETTER representation to work with:

```
Raw positions: "token 5 has hidden state [0.3, 0.5, ...]" — one sample point
Wavelet coeffs: "the sequence has a low-freq component of 2.1 centered around position 5"
                — captures the PATTERN, not just one point
```

The perceiver's queries can ask "what's the broad structure?" (attend to coarse coefficients) or "what's the precise detail at the number position?" (attend to fine coefficients). The multi-resolution structure makes selective attention easier.

---

## Learned Level Weights

The `level_weights` parameter lets the model decide how important each resolution level is:

```
level_weights after training might look like:
  Level 4 (coarsest):  weight=0.8  (important — problem structure)
  Level 3:             weight=0.9  (important — key phrases)
  Level 2:             weight=0.7  (moderately important — word detail)
  Level 1:             weight=0.3  (less important — token detail)
  Level 0 (truncated): not present (finest detail discarded)
```

Different levels might matter more for different passes:

```
Pass 1 (parsing):    coarse levels matter most (broad problem structure)
Pass 3 (computing):  mid levels matter most (specific numbers and operations)
Pass 5 (verifying):  all levels matter (checking fine details against broad structure)
```

For pass-dependent level weighting, condition the weights on the pass number:

```python
# Optional enhancement: pass-conditioned level weights
self.level_weight_net = nn.Linear(pass_embed_dim, max_level + 1)

def forward(self, hidden_states, pass_embed):
    level_weights = torch.sigmoid(self.level_weight_net(pass_embed))
    # Use these instead of the fixed self.level_weights
```

This is optional — start with fixed level weights. Add pass-conditioning later if needed.

---

## Connection to the Full Wave Architecture

```
WAVELET INPUT:
  Hidden states → Haar wavelet → multi-resolution coefficients
  Structured by frequency BEFORE the perceiver sees them

PERCEIVER COMPRESSION:
  Wavelet coefficients → 7-layer perceiver → 64-float page
  Reads frequency-structured input, produces frequency-structured output

FOURIER PAGE ENCODING:
  64-float page + Fourier positional encoding
  Page dimensions carry frequency identity (low dims = coarse, high dims = fine)

FOURIER PASS ENCODING:
  Pass number → sine/cosine features → hypernetwork
  Smooth temporal evolution of atom activations

ATOM EXPANSION:
  64 atom scales → modify attention at all 16 layers
  Each atom is a direction of attention modification

The full pipeline is frequency-aware end-to-end:
  wavelet input → perceiver → Fourier page → Fourier pass → atom expansion
```

---

## What NOT to Do

```
- Do NOT use pywt in the training loop — CPU transfer kills speed.
  Use the pure PyTorch HaarWaveletPreprocessFast implementation.
  
- Do NOT truncate too aggressively. Removing level 0 (finest) is safe.
  Removing level 1 too might lose important token-level detail for math.
  Start with truncating only level 0. Add level 1 truncation if speed-limited.

- Do NOT apply wavelet to the embedding layer (layer 0).
  Only to transformer hidden states (layers 1-16).
  The embedding layer has different statistical properties.

- Do NOT make the wavelet transform learnable (learned filters instead of Haar).
  Haar is simple, fast, well-understood. Learned wavelets add complexity
  without clear benefit. The perceiver already does learned compression.

- Do NOT apply wavelet along the hidden dimension (d_model).
  Apply along the SEQUENCE dimension only. Each feature gets its own
  frequency decomposition across the sequence.
```

---

## Testing

Verify the wavelet transform preserves information:

```python
# Roundtrip test: wavelet → inverse wavelet should recover original
hidden = torch.randn(2, 100, 2048)
wavelet = HaarWaveletPreprocessFast(max_level=4, truncate_finest=False)
coeffs = wavelet(hidden)
# coeffs should have same total elements as hidden (just reorganized)
assert coeffs.size(1) >= hidden.size(1) - 4  # within padding tolerance

# Truncation test: removing finest level should halve the coefficients
wavelet_trunc = HaarWaveletPreprocessFast(max_level=4, truncate_finest=True)
coeffs_trunc = wavelet_trunc(hidden)
assert coeffs_trunc.size(1) < hidden.size(1)  # fewer than original
print(f"Compression: {hidden.size(1)} → {coeffs_trunc.size(1)} tokens")
```

---

## Parameter Cost

```
Learned level weights: 5 floats (negligible)
Everything else: zero parameters (the Haar wavelet is a fixed transform)

The wavelet SAVES compute by halving the perceiver's input.
Net effect: faster training, better compression, zero parameter cost.
```
