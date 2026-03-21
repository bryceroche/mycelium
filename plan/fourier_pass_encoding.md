# Handoff: Fourier Pass Encoding

## One-Sentence Summary

Replace the discrete pass embedding (nn.Embedding) with continuous Fourier features of the pass number. The hypernetwork learns to generate atom scales as smooth waveforms across passes rather than arbitrary jumps. One change, encourages rhythmic thinking.

---

## Why

The current pass embedding is a lookup table — pass 3 and pass 4 have no inherent relationship. The hypernetwork treats each pass independently, producing potentially unrelated atom configurations at consecutive passes.

With Fourier features, the pass number becomes a smooth, periodic signal. The hypernetwork learns atom activations as continuous functions of the pass number. Atoms activate and deactivate in wave patterns:

```
Discrete embedding:
  Pass 1: [random vector A]
  Pass 2: [random vector B]  ← no relationship to A
  Pass 3: [random vector C]  ← no relationship to B

Fourier features:
  Pass 1: [sin(1·f₁), cos(1·f₁), sin(1·f₂), cos(1·f₂), ...]
  Pass 2: [sin(2·f₁), cos(2·f₁), sin(2·f₂), cos(2·f₂), ...]  ← smoothly related
  Pass 3: [sin(3·f₁), cos(3·f₁), sin(3·f₂), cos(3·f₂), ...]  ← smoothly related
```

The result: the atom activation spectrogram across passes becomes smooth rather than noisy. The model develops rhythmic thinking patterns — atoms activate in waves.

---

## Implementation

Replace the pass embedding in AtomHypernetwork:

```python
# BEFORE:
class AtomHypernetwork(nn.Module):
    def __init__(self, ..., pass_embed_dim=512, max_passes=20):
        ...
        self.pass_embed = nn.Embedding(max_passes, pass_embed_dim)
    
    def forward(self, state_pages, pass_num):
        ...
        pass_ctx = self.pass_embed(torch.tensor(pass_num, device=device))
        ...

# AFTER:
class AtomHypernetwork(nn.Module):
    def __init__(self, ..., pass_embed_dim=512):
        ...
        # Fourier frequencies (fixed, not learned)
        self.register_buffer(
            'fourier_freqs',
            torch.exp(torch.arange(0, pass_embed_dim, 2) * -(math.log(10000.0) / pass_embed_dim))
        )
        # Optional: learned linear projection on top of Fourier features
        self.pass_project = nn.Linear(pass_embed_dim, pass_embed_dim)
    
    def fourier_encode(self, pass_num, device):
        """Encode pass number as smooth Fourier features."""
        t = pass_num * self.fourier_freqs  # (dim/2,)
        encoding = torch.cat([torch.sin(t), torch.cos(t)])  # (dim,)
        return self.pass_project(encoding)  # optional learned transformation
    
    def forward(self, state_pages, pass_num):
        ...
        pass_ctx = self.fourier_encode(pass_num, device)
        pass_ctx = pass_ctx.unsqueeze(0).expand(batch_size, -1)
        ...
```

---

## What This Changes

```
BEFORE: pass_embed(3) → arbitrary learned vector for pass 3
AFTER:  fourier_encode(3) → [sin(3f₁), cos(3f₁), sin(3f₂), cos(3f₂), ...]
        Smoothly interpolates between pass 2 and pass 4
        Naturally periodic — the model can learn cyclic attention patterns
```

The Fourier frequencies span multiple scales:

```
Low frequencies (f₁):   slow oscillation across passes
  → captures broad phase: "early thinking" vs "late thinking"
  
Mid frequencies (f₂-f₄): medium oscillation
  → captures rhythm: "compute, check, compute, check"
  
High frequencies (f₁₂₈+): fast oscillation
  → captures fine timing: "this specific pass needs this specific atom"
```

The hypernetwork's linear layers learn which frequencies matter for generating useful atom scales. Low frequencies might drive the parse→compute→verify→answer arc. High frequencies might drive pass-specific adjustments.

---

## Properties

**Smooth interpolation.** Pass 3.5 (not used in training) would produce a meaningful encoding — halfway between pass 3 and pass 4. This means the model could generalize to pass counts it wasn't trained on. Train with 5 passes, inference with 8 — the Fourier encoding smoothly extends.

**No max_passes limit.** The embedding table required a fixed max_passes. Fourier features work for any pass number. Pass 100 is just sin(100·f), cos(100·f). No reallocation needed.

**Periodicity as a prior.** The sin/cos basis encourages the model to learn periodic patterns. "Activate atom 17 every 3 passes" is naturally representable. This matches the expand-collapse rhythm — the model might learn to alternate between computation-focused and verification-focused atom sets.

**The learned projection (pass_project) adds flexibility.** Raw Fourier features are a fixed basis. The linear projection lets the model rotate and scale the features into the most useful representation. If pure Fourier isn't optimal, the projection compensates.

---

## Parameter Changes

```
REMOVED: nn.Embedding(20, 512) = 10,240 params
ADDED:   nn.Linear(512, 512) = 262,656 params (pass_project)
         fourier_freqs = 256 floats (buffer, not parameter)

Net change: +252K params (negligible vs 10M hypernetwork)
```

Alternatively, skip the learned projection and use raw Fourier features:

```
REMOVED: nn.Embedding(20, 512) = 10,240 params
ADDED:   fourier_freqs = 256 floats (buffer only)

Net change: -10K params (even simpler)
```

Start with the learned projection. If it doesn't help, simplify to raw Fourier.

---

## No Other Changes

Everything else stays the same:
- Page attention mechanism (unchanged)
- Scale MLP (unchanged — still takes page_summary + strategy + pass_features as input)
- Tanh output (unchanged)
- Atom templates (unchanged)
- Training recipe (unchanged)
- All regularization (unchanged)

This is a drop-in replacement for the pass embedding. One component swapped.

---

## What to Monitor

```
1. Atom spectrogram smoothness:
   Plot atom_scales across passes for several problems.
   With Fourier: should see smooth wave patterns.
   Without Fourier: arbitrary jumps between passes.

2. Cross-pass cosine:
   cos(scales_pass_N, scales_pass_N+1) should be HIGHER than with discrete embedding.
   Consecutive passes should have related (but not identical) atom configurations.

3. Generalization to more passes:
   Train with 5 passes, eval with 8.
   Fourier encoding should gracefully extend.
   Discrete embedding would crash (no embedding for pass 6+).
```

---

## Fourier Page Encoding (Structural Identity for Pages)

In addition to Fourier pass encoding in the hypernetwork, add Fourier structure to the PAGES themselves. Each page dimension gets a frequency identity from alternating sine/cosine waves. Each page carries both content (what was compressed) and structure (which dimension, which pass).

### Why

Right now pages are flat 64-float vectors. Dimension 0 and dimension 63 have no structural relationship. The perceiver uses them however it wants with no inductive bias about which dimensions should encode what.

With Fourier encoding, each dimension has a position in frequency space:

```
dim 0 (sin):  lowest frequency  → coarse information
dim 1 (cos):  lowest frequency  → coarse information (90° phase shift)
dim 2 (sin):  next frequency    → slightly finer
dim 3 (cos):  next frequency    → slightly finer (90° phase shift)
...
dim 62 (sin): highest frequency → finest detail
dim 63 (cos): highest frequency → finest detail (90° phase shift)

32 frequency pairs × 2 (sin + cos) = 64 dimensions
```

Adjacent dimensions form sine/cosine PAIRS — together they encode both magnitude and phase at that frequency. The 64 floats become 32 frequency bands, each capturing a different scale of information.

### Implementation: Pi-Harmonic Frequencies

The frequencies use pi-harmonic spacing — the same mathematical basis as DCT (discrete cosine transform), which is what JPEG uses for image compression. Each frequency pair is an independent harmonic of pi. This creates an orthogonal basis where the page IS a proper frequency decomposition.

```python
class PiHarmonicPageEncoding(nn.Module):
    def __init__(self, page_size=64):
        super().__init__()
        n = page_size // 2  # 32 frequency pairs
        
        # Pi-harmonic frequencies (DCT-like basis)
        # freq_k = k * π / page_size
        # Creates orthogonal harmonics: wavelength decreases with k
        # k=1:  wavelength = 128 dims (lowest frequency — broadest pattern)
        # k=16: wavelength = 8 dims (mid frequency)
        # k=32: wavelength = 4 dims (highest frequency — finest detail)
        freqs = torch.arange(1, n + 1, dtype=torch.float32) * math.pi / page_size
        self.register_buffer('freqs', freqs)
    
    def encode(self, pass_num):
        """
        Encode pass number using pi-harmonic frequencies.
        Each dimension pair (sin, cos) is one harmonic of pi.
        Orthogonal basis — independent frequency channels.
        """
        t = pass_num * self.freqs  # (32,)
        return torch.cat([torch.sin(t), torch.cos(t)])  # (64,)
    
    def apply(self, page, pass_num):
        """Add pi-harmonic positional structure to a page."""
        encoding = self.encode(pass_num).to(page.device)
        return page + encoding  # content + harmonic structure
```

### Why Pi-Harmonic Instead of Transformer-Style

```
Transformer-style: freqs = exp(-i * log(10000) / d)
  Geometric spacing. Borrowed from token position encoding.
  Works but not principled for frequency decomposition.
  The constant 10000 is arbitrary.

Pi-harmonic: freqs = k * π / d
  Arithmetic spacing at multiples of π.
  Same basis as DCT (proven optimal for energy compaction).
  Each frequency pair is orthogonal by construction.
  Pi is the natural constant — not arbitrary, mathematical.

The page becomes a DCT-like decomposition:
  dims 0-1:   1st harmonic of π (coarsest — problem type)
  dims 2-3:   2nd harmonic of π (next coarsest)
  ...
  dims 62-63: 32nd harmonic of π (finest — exact corrections)
```

### Apéry-Weighted Wavelet Initialization

The wavelet level weights start with a 1/k³ power law — coarse levels carry more power than fine levels. The total power converges to Apéry's constant ζ(3) ≈ 1.202.

This is a principled initialization: natural signals have power spectra that decay with frequency. 1/k³ is a common decay rate in physical systems. The model quickly learns its own weights, but starting from a physically motivated prior is better than starting uniform.

```python
# In HaarWaveletPreprocessFast.__init__:

# Apéry-weighted initialization (1/k³ power law, coarse > fine)
apery_weights = torch.tensor([1.0 / (k + 1)**3 for k in range(max_level + 1)])
apery_weights = apery_weights / apery_weights.sum() * (max_level + 1)  # normalize
self.level_weights = nn.Parameter(apery_weights)

# Result for max_level=4:
#   Level 0 (coarsest): weight ∝ 1/1³ = 1.000  (strongest)
#   Level 1:            weight ∝ 1/2³ = 0.125
#   Level 2:            weight ∝ 1/3³ = 0.037
#   Level 3:            weight ∝ 1/4³ = 0.016
#   Level 4 (finest):   weight ∝ 1/5³ = 0.008  (weakest)
#
# Total: Σ 1/k³ → ζ(3) ≈ 1.202 (Apéry's constant)
```

### Integration Into Thinking Loop

```python
fourier_page = FourierPageEncoding(page_size=64)

for pass_num in range(max_passes):
    # ... atom LoRA, Llama forward, perceiver compresses ...
    page = perceiver(hidden_states, pass_num)  # raw 64 floats
    page = F.normalize(page, dim=-1) * math.sqrt(64)
    
    # Add Fourier structure AFTER normalization
    page = fourier_page.apply(page, pass_num)
    
    state_pages.append(page)
```

### What This Gives the Hypernetwork

When the hypernetwork attends over pages, it can distinguish:

```
"page 1, dims 0-7"   → low-frequency content from pass 1 (coarse, early)
"page 1, dims 56-63"  → high-frequency content from pass 1 (fine, early)
"page 3, dims 0-7"   → low-frequency content from pass 3 (coarse, late)
"page 3, dims 56-63"  → high-frequency content from pass 3 (fine, late)
```

The cross-attention naturally learns to query specific frequency bands from specific passes. "For this computation, I need exact numbers from pass 2's high-frequency band and problem type from pass 1's low-frequency band."

### Natural Coarse-to-Fine Pressure

The encoding creates an inductive bias for progressive refinement:

```
Pass 1 writes heavily to dims 0-15:   "addition problem, numbers around 50"
Pass 2 writes heavily to dims 16-40:  "specifically 48 and 24, divide then add"
Pass 3 writes heavily to dims 40-63:  "48/2=24, 24+48=72, answer exactly 72"
```

Low dimensions → slow-changing, coarse information (problem type, magnitude)
High dimensions → fast-changing, precise information (exact values, corrections)

This matches Fourier series: low frequencies capture broad shape, high frequencies capture details. The model starts with this bias and refines from there.

### 64 Atoms ↔ 64 Page Dimensions

The Fourier structure creates natural atom-dimension correspondence:

```
Low-freq page dims (0-15)   ↔ parsing atoms (broad understanding)
Mid-freq page dims (16-40)  ↔ computing atoms (key operations)
High-freq page dims (40-63) ↔ precision atoms (exact numbers)
```

Atom 5 might learn to modify attention in ways that help the perceiver fill low-frequency page dimensions. Atom 50 might help fill high-frequency dimensions. The shared frequency structure aligns atoms with page bands.

### Parameter Cost

Zero learnable parameters. The Fourier frequencies are a fixed buffer, not trained. The encoding is deterministic — same pass number always produces the same encoding. No parameters, no training, just an inductive bias.

### What NOT to Do

```
- Do NOT make the frequencies learnable. Fixed frequencies provide a stable basis.
  Learned frequencies would drift during training and lose their structural meaning.
- Do NOT apply encoding BEFORE hypersphere normalization.
  Normalize the content first, then add structure.
- Do NOT use large encoding magnitudes. The encoding should be a gentle bias,
  not overwhelm the content. If the Fourier encoding norms are much larger than
  the content norms, scale them down (multiply by 0.1 or 0.01).
```

---

## The Wave Interpretation

After training, the atom spectrogram reveals the model's thinking rhythm:

```
        Pass 1   Pass 2   Pass 3   Pass 4   Pass 5
Atom 0:  ████░    ███░░    ██░░░    ███░░    ████░    ← low-frequency wave
Atom 7:  ░░░░░    ████░    ░░░░░    ████░    ░░░░░    ← period-2 oscillation
Atom 22: ░░░░░    ░░░░░    █████    ░░░░░    ░░░░░    ← single peak at pass 3
Atom 41: ██░░░    ██░░░    ██░░░    ██░░░    ██░░░    ← constant (always on)
Atom 58: ░░░░░    ░░░░░    ░░░░░    ░░░░░    █████    ← late activator
```

Each atom's activation pattern IS a wave. The Fourier encoding encourages these patterns to be smooth and periodic. The model breathes in waves — some atoms pulse with the expand-collapse rhythm, others provide steady background processing, others activate only at specific phases.
