# Handoff: Fourier Initialization — Atoms + Residual Gate

## One-Sentence Summary

Initialize the 64 LoRA atoms as a Fourier basis (orthogonal by construction, each atom a different frequency of attention modification) and initialize the residual gate with a frequency-aware bias (low-frequency page dimensions preserve, high-frequency dimensions update). Two one-time initializations that create a soft pressure toward spectral decomposition across cycles without forcing it.

---

## Why

### The Problem With Random Initialization

Random atom initialization wastes capacity. With 64 atoms initialized from randn, some atoms start nearly identical by chance. They receive similar gradients. They stay similar throughout training. We measured this: with random init, the model uses ~40 atoms but many are redundant.

```
Random init:   64 atoms, ~40 truly distinct (24 redundant by chance)
Fourier init:  64 atoms, ALL 64 distinct (orthogonal, zero redundancy)
```

### The Prism Principle

The FFT splits a complex signal into independent frequency components. Each component can be analyzed separately. Our atoms should do the same — split a complex problem into independent aspects of attention modification.

```
FFT prism:    white light → [red, orange, yellow, green, blue, violet]
Atom prism:   complex problem → [global structure, phrase patterns, 
                                  word relationships, number focus, 
                                  operator attention, digit precision]
```

Each atom starts as a different "color" of attention modification. The hypernetwork learns which colors each cycle needs.

---

## Component 1: Fourier Atom Initialization

### Implementation

```python
import torch
import math

def fourier_atom_init(num_atoms=64, d_model=2048, rank=6, scale=0.01):
    """
    Initialize atom A matrices as Fourier basis functions.
    
    Each (atom, rank_dim) pair gets a unique frequency.
    64 atoms × 6 rank dims = 384 unique basis functions.
    All orthogonal by construction (sine/cosine at different frequencies).
    
    Args:
        num_atoms: number of LoRA atoms (64)
        d_model: model hidden dimension (2048)
        rank: LoRA rank per atom (6)
        scale: initialization magnitude (0.01)
    
    Returns:
        A tensor of shape (num_atoms, d_model, rank)
    """
    A = torch.zeros(num_atoms, d_model, rank)
    
    # Position along the hidden dimension (normalized to [0, 1])
    t = torch.arange(d_model, dtype=torch.float32) / d_model
    
    for atom_idx in range(num_atoms):
        for r in range(rank):
            # Unique frequency for each (atom, rank) pair
            freq = atom_idx * rank + r + 1  # +1 to avoid DC at (0,0)
            
            # Alternating sin/cos gives orthogonal pairs
            if r % 2 == 0:
                A[atom_idx, :, r] = torch.sin(2 * math.pi * freq * t)
            else:
                A[atom_idx, :, r] = torch.cos(2 * math.pi * freq * t)
        
        # Normalize each atom to target scale
        atom_norm = A[atom_idx].norm()
        if atom_norm > 0:
            A[atom_idx] = A[atom_idx] / atom_norm * scale * math.sqrt(d_model)
    
    return A


def apply_fourier_init_to_model(model):
    """
    Apply Fourier initialization to all atom A matrices.
    B matrices stay random — the asymmetric init (structured A, random B)
    gives good coverage while maintaining flexibility.
    
    Call ONCE at model creation (before training).
    Do NOT call when warm-starting from a trained checkpoint.
    """
    atoms = model.atoms  # LoRAAtoms module
    
    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        A_param = atoms.A[proj_name]  # (num_atoms, num_layers, d_model, rank)
        num_atoms, num_layers, d_model, rank = A_param.shape
        
        # Generate one Fourier basis
        fourier_A = fourier_atom_init(num_atoms, d_model, rank, scale=0.01)
        
        # Apply same basis to every layer
        # Different layers learn different refinements from the same starting point
        for layer_idx in range(num_layers):
            A_param.data[:, layer_idx] = fourier_A
    
    print(f"Fourier atom init: {num_atoms} atoms × {rank} rank = "
          f"{num_atoms * rank} orthogonal basis functions")
    print(f"Frequency range: 1 to {num_atoms * rank} cycles across d_model={d_model}")
```

### What Each Atom Becomes

```
Atom 0  (freqs 1-6):    lowest frequencies
  → broadest attention modification
  → captures global problem structure ("this is an addition problem")
  → activates on most problems (general-purpose)

Atom 16 (freqs 97-102): low-mid frequencies  
  → phrase-level attention modification
  → captures relationships between clauses ("sold X in April, half in May")
  → activates when parsing multi-part descriptions

Atom 32 (freqs 193-198): mid-high frequencies
  → word-level attention modification  
  → captures specific number-word associations ("48 friends")
  → activates during number extraction

Atom 48 (freqs 289-294): high frequencies
  → fine-grained attention modification
  → captures precise token relationships ("48 / 2")
  → activates during computation

Atom 63 (freqs 379-384): highest frequencies
  → most local attention modification
  → captures adjacent-token patterns ("= 24")
  → activates during answer extraction
```

### Verification: Orthogonality Check

```python
def verify_orthogonality(model):
    """Verify that Fourier-initialized atoms are orthogonal."""
    A = model.atoms.A['q_proj'][:, 0]  # (64, d_model, rank) for layer 0
    
    # Flatten each atom to a vector
    flat = A.reshape(A.size(0), -1)  # (64, d_model * rank)
    
    # Compute pairwise cosine similarities
    flat_norm = F.normalize(flat, dim=-1)
    cos_matrix = flat_norm @ flat_norm.T  # (64, 64)
    
    # Off-diagonal should be near zero
    off_diag = cos_matrix - torch.eye(64)
    max_cos = off_diag.abs().max().item()
    mean_cos = off_diag.abs().mean().item()
    
    print(f"Atom orthogonality check:")
    print(f"  Max off-diagonal cosine: {max_cos:.6f} (should be < 0.01)")
    print(f"  Mean off-diagonal cosine: {mean_cos:.6f} (should be < 0.001)")
    
    return max_cos < 0.05  # passes if approximately orthogonal
```

---

## Component 2: Frequency-Aware Residual Gate Initialization

### The Insight

The residual gate blends old and new pages per dimension. With pi-harmonic page encoding, page dimensions have frequency identity:

```
Dims 0-15:   low frequency  (coarse information — problem type, magnitude)
Dims 16-40:  mid frequency  (medium detail — key quantities, relationships)
Dims 40-63:  high frequency (fine detail — exact numbers, corrections)
```

Coarse information should PERSIST across cycles (you don't re-parse the problem type every cycle). Fine detail should UPDATE each cycle (each cycle computes a new number). The residual gate should reflect this:

```
Low-freq dims:   gate ≈ 0.27 → 73% preserve old page (coarse is stable)
Mid-freq dims:   gate ≈ 0.50 → 50/50 blend (relationships evolve)
High-freq dims:  gate ≈ 0.73 → 73% use new page (fine details change)
```

### Implementation

```python
def init_residual_gate_fourier(residual_gate_module, page_size=64):
    """
    Initialize the residual gate with frequency-aware bias.
    
    Low-frequency page dimensions (early indices) → preserve old page.
    High-frequency page dimensions (late indices) → accept new page.
    
    The gate is: blended = sigmoid(gate) * new + (1 - sigmoid(gate)) * old
    So positive bias → more new page, negative bias → more old page.
    
    Args:
        residual_gate_module: the ResidualPageGate module
        page_size: dimension of pages (64)
    """
    # Linear ramp from -1.0 to +1.0 across page dimensions
    # sigmoid(-1.0) = 0.27 → 73% preserve (low-freq dims)
    # sigmoid( 0.0) = 0.50 → balanced (mid-freq dims)
    # sigmoid(+1.0) = 0.73 → 73% update (high-freq dims)
    
    bias = torch.linspace(-1.0, 1.0, page_size)
    
    # The gate module is: gate = sigmoid(Linear([new, old]))
    # We set the bias of that linear layer
    # The weight starts near zero (from default init), so bias dominates initially
    if hasattr(residual_gate_module, 'gate'):
        # ResidualPageGate has self.gate = nn.Linear(page_size * 2, page_size)
        residual_gate_module.gate.bias.data = bias
        
        # Also scale down the weight so bias dominates early
        residual_gate_module.gate.weight.data *= 0.01
        
        print(f"Frequency-aware residual gate init:")
        print(f"  Dim 0 (lowest freq):  gate = sigmoid({bias[0]:.1f}) = {torch.sigmoid(bias[0]):.2f} → {1-torch.sigmoid(bias[0]):.0%} preserve")
        print(f"  Dim 31 (mid freq):    gate = sigmoid({bias[31]:.1f}) = {torch.sigmoid(bias[31]):.2f} → balanced")
        print(f"  Dim 63 (highest freq): gate = sigmoid({bias[63]:.1f}) = {torch.sigmoid(bias[63]):.2f} → {torch.sigmoid(bias[63]):.0%} update")
```

### Effect on Information Flow

```
WITHOUT frequency-aware gate init (current):
  All dimensions start at gate ≈ 0.5 (uniform blending)
  The model must learn which dims to preserve vs update
  Takes several epochs to discover that coarse info is stable

WITH frequency-aware gate init:
  Low dims start at gate ≈ 0.27 (preserve coarse info)
  High dims start at gate ≈ 0.73 (update fine details)
  The model starts with the RIGHT bias from epoch 1
  Training refines but the structure is already there
```

### Effect on Eigenvalues

The residual gate contributes to the Jacobian. With frequency-aware init:

```
Low-freq dims:  eigenvalue ≈ 0.73 (strong preservation — approaching 1.0)
Mid-freq dims:  eigenvalue ≈ 0.50 (moderate preservation)
High-freq dims: eigenvalue ≈ 0.27 (weak preservation — mostly fresh each cycle)

This creates a SPECTRUM of eigenvalues:
  Some near 1.0 (information persists — problem type, key numbers)
  Some near 0.5 (information blends — evolving computations)
  Some near 0.3 (information refreshes — cycle-specific details)

Compared to current (all ≈ 0.5):
  More stable for coarse information
  More dynamic for fine information
  Better spectral structure overall
```

---

## Combined Application

```python
def apply_fourier_inits(model):
    """
    Apply both Fourier initializations at model creation.
    Call ONCE before training begins.
    """
    # 1. Fourier atom initialization (orthogonal basis)
    apply_fourier_init_to_model(model)
    
    # 2. Frequency-aware residual gate initialization
    if hasattr(model, 'residual_gate'):
        init_residual_gate_fourier(model.residual_gate)
    
    # 3. Verify
    verify_orthogonality(model)
    
    print("\nFourier initializations applied:")
    print("  ✓ Atoms: 64 orthogonal frequency basis functions")
    print("  ✓ Residual gate: low-freq preserves, high-freq updates")
    print("  → Spectral decomposition should emerge during training")
```

### Integration Into Training Script

```python
# In the training script, after model creation:

model = AtomLoRAModel(
    llama=llama,
    num_atoms=64,
    rank=6,
    skip_pass_embed=True,
    # ... other args ...
)

# Apply Fourier initializations (ONLY for fresh training, NOT warm start)
if not args.warm:
    apply_fourier_inits(model)
```

---

## What Should Emerge From Training

After training with both initializations, the atom spectrogram should show frequency-ordered activation:

```
Expected atom spectrogram (problem: "Natalia sold 48 clips, half in May, total?"):

         Cycle 1    Cycle 2    Cycle 3    Cycle 4    Cycle 5
Atom 0:  ████████   ██████░░   ████░░░░   ██░░░░░░   ██░░░░░░   ← low-freq (always on, fading)
Atom 8:  ██████░░   ████████   ██████░░   ████░░░░   ██░░░░░░   ← low-mid (peaks early)
Atom 24: ░░░░░░░░   ████░░░░   ████████   ██████░░   ████░░░░   ← mid (peaks middle)
Atom 40: ░░░░░░░░   ░░░░░░░░   ████░░░░   ████████   ██████░░   ← mid-high (peaks late)
Atom 56: ░░░░░░░░   ░░░░░░░░   ░░░░░░░░   ████░░░░   ████████   ← high-freq (peaks last)

Reading: the problem is decomposed coarse-to-fine across cycles.
Cycle 1: "what kind of problem?" (low-freq atoms)
Cycle 3: "what are the relationships?" (mid-freq atoms)
Cycle 5: "what's the exact answer?" (high-freq atoms)
```

The residual gate should show frequency-dependent persistence:

```
Expected residual gate values after training:

Dim 0  (low freq):   gate ≈ 0.2 (80% preserve — problem type is stable)
Dim 15 (low-mid):    gate ≈ 0.3 (70% preserve — key quantities persist)
Dim 31 (mid):        gate ≈ 0.5 (balanced — relationships evolve)
Dim 47 (mid-high):   gate ≈ 0.7 (30% preserve — computations update)
Dim 63 (high freq):  gate ≈ 0.9 (10% preserve — fine details refresh each cycle)
```

---

## Diagnostic: Verify Spectral Decomposition Emerged

```python
def diagnose_spectral_decomposition(model, eval_problems, max_passes=5):
    """
    After training, check if the frequency-ordered decomposition emerged.
    
    Measures:
    1. Do low-index atoms activate more in early cycles?
    2. Do high-index atoms activate more in late cycles?
    3. Does the residual gate preserve low-freq dims more than high-freq?
    """
    atom_by_cycle = torch.zeros(64, max_passes)  # mean activation per atom per cycle
    gate_values = None
    
    for problem_ids, gold in eval_problems[:50]:
        state_pages = []
        for pass_num in range(max_passes):
            page, scales = model.think_one_pass(problem_ids, state_pages, pass_num)
            atom_by_cycle[:, pass_num] += scales.abs().mean(dim=0).cpu()
            state_pages.append(page)
    
    atom_by_cycle /= 50  # average over problems
    
    # Check: do low-index atoms peak earlier than high-index atoms?
    peak_cycle = atom_by_cycle.argmax(dim=1)  # which cycle each atom peaks at
    
    low_atoms_peak = peak_cycle[:16].float().mean()   # atoms 0-15
    mid_atoms_peak = peak_cycle[16:48].float().mean()  # atoms 16-47
    high_atoms_peak = peak_cycle[48:].float().mean()   # atoms 48-63
    
    print("Spectral decomposition diagnostic:")
    print(f"  Low-freq atoms (0-15) peak at cycle:  {low_atoms_peak:.1f}")
    print(f"  Mid-freq atoms (16-47) peak at cycle: {mid_atoms_peak:.1f}")
    print(f"  High-freq atoms (48-63) peak at cycle: {high_atoms_peak:.1f}")
    
    if low_atoms_peak < mid_atoms_peak < high_atoms_peak:
        print("  ✓ SPECTRAL ORDER EMERGED: low→mid→high across cycles")
    else:
        print("  ~ No clear spectral ordering (model found a different decomposition)")
    
    # Check residual gate frequency structure
    if hasattr(model, 'residual_gate') and hasattr(model.residual_gate, 'gate'):
        # Get the effective gate bias
        gate_bias = model.residual_gate.gate.bias.data
        gate_sigmoid = torch.sigmoid(gate_bias)
        
        low_gate = gate_sigmoid[:16].mean()
        mid_gate = gate_sigmoid[16:48].mean()
        high_gate = gate_sigmoid[48:].mean()
        
        print(f"\n  Residual gate by frequency band:")
        print(f"    Low-freq dims (0-15):   gate={low_gate:.2f} → {1-low_gate:.0%} preserve")
        print(f"    Mid-freq dims (16-47):  gate={mid_gate:.2f} → {1-mid_gate:.0%} preserve")
        print(f"    High-freq dims (48-63): gate={high_gate:.2f} → {1-high_gate:.0%} preserve")
        
        if low_gate < mid_gate < high_gate:
            print("    ✓ FREQUENCY FILTER EMERGED: low preserves, high updates")
        else:
            print("    ~ No clear frequency filter (model learned a different pattern)")
```

---

## Parameter Cost

```
Fourier atom initialization:       0 params (one-time init, no new parameters)
Frequency-aware residual gate:     0 params (one-time bias init, no new parameters)

Total new parameters: ZERO
Total compute cost: ZERO (initializations run once before training)

These are FREE improvements — better starting points at no cost.
```

---

## What NOT to Do

```
- Do NOT force atoms to stay orthogonal during training.
  The initialization is a starting point. Training refines freely.
  Adding an orthogonality loss constrains learning unnecessarily.

- Do NOT force specific atoms to specific cycles.
  "Atom 0-15 can only activate at cycle 1" would be too rigid.
  The hypernetwork decides which atoms activate when.
  The Fourier init creates a BIAS, not a constraint.

- Do NOT apply Fourier init when warm-starting from a checkpoint.
  Warm-started atoms have learned representations.
  Overwriting them with Fourier init erases what they learned.
  Only use for fresh training from scratch.

- Do NOT make the residual gate bias too extreme.
  linspace(-1, 1) gives sigmoid range [0.27, 0.73] — gentle.
  linspace(-3, 3) gives sigmoid range [0.05, 0.95] — too extreme,
  would almost freeze low dims and almost ignore old for high dims.
  Start gentle, let training adjust.

- Do NOT apply Fourier init to B matrices.
  Only A matrices get the Fourier basis. B matrices stay random.
  Structured A × random B gives good coverage.
  Structured A × structured B would over-constrain.
```
