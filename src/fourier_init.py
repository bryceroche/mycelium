"""
Fourier Initialization — Atoms + Residual Gate (v24.9)

Two zero-cost, one-time initializations:

1. Fourier atom init: Initialize 64 LoRA atoms as orthogonal frequency basis
   functions. Each (atom, rank_dim) pair gets a unique frequency. All 384
   basis functions are orthogonal by construction — zero redundancy.

2. Frequency-aware residual gate: Low-freq page dims preserve (gate ~ 0.27),
   high-freq dims update (gate ~ 0.73). Creates natural spectral filter
   aligned with pi-harmonic page encoding.

Call ONCE at model creation for fresh training. Do NOT apply when warm-starting
from a trained checkpoint — that would erase learned representations.

Parameter cost: ZERO (initializations only, no new parameters)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier_atom_init(num_atoms: int = 64, d_model: int = 2048,
                      rank: int = 6, scale: float = 0.01) -> torch.Tensor:
    """
    Initialize atom A matrices as Fourier basis functions.

    Each (atom, rank_dim) pair gets a unique frequency.
    64 atoms x 6 rank dims = 384 unique basis functions.
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


def apply_fourier_atom_init(atoms_module) -> None:
    """
    Apply Fourier initialization to all atom A matrices.
    B matrices stay random — asymmetric init (structured A, random B)
    gives good coverage while maintaining flexibility.

    Args:
        atoms_module: LoRAAtoms module with A and B ParameterDicts
    """
    for proj_name in atoms_module.PROJ_NAMES:
        A_param = atoms_module.A[proj_name]  # (num_atoms, num_layers, d_model, rank)
        num_atoms, num_layers, d_model, rank = A_param.shape

        # Generate one Fourier basis
        fourier_A = fourier_atom_init(num_atoms, d_model, rank, scale=0.01)

        # Apply same basis to every layer
        with torch.no_grad():
            for layer_idx in range(num_layers):
                A_param[:, layer_idx] = fourier_A

    print(f"  Fourier atom init: {num_atoms} atoms x {rank} rank = "
          f"{num_atoms * rank} orthogonal basis functions")
    print(f"  Frequency range: 1 to {num_atoms * rank} cycles across d_model={d_model}")


def init_residual_gate_fourier(residual_gate_module, page_size: int = 64) -> None:
    """
    Initialize the residual gate with frequency-aware bias.

    Low-frequency page dimensions (early indices) → preserve old page.
    High-frequency page dimensions (late indices) → accept new page.

    The gate is: blended = sigmoid(gate) * new + (1 - sigmoid(gate)) * old
    So positive bias → more new page, negative bias → more old page.

    Args:
        residual_gate_module: ResidualPageGate module
        page_size: dimension of pages (64)
    """
    # Linear ramp from -1.0 to +1.0 across page dimensions
    # sigmoid(-1.0) = 0.27 → 73% preserve (low-freq dims)
    # sigmoid( 0.0) = 0.50 → balanced (mid-freq dims)
    # sigmoid(+1.0) = 0.73 → 73% update (high-freq dims)
    bias = torch.linspace(-1.0, 1.0, page_size)

    if hasattr(residual_gate_module, 'gate'):
        with torch.no_grad():
            residual_gate_module.gate.bias.data = bias
            # Scale down weight so bias dominates early training
            residual_gate_module.gate.weight.data *= 0.01

        low = torch.sigmoid(bias[0])
        mid = torch.sigmoid(bias[page_size // 2])
        high = torch.sigmoid(bias[-1])
        print(f"  Frequency-aware residual gate:")
        print(f"    Dim 0 (low freq):  gate={low:.2f} → {1-low:.0%} preserve")
        print(f"    Dim {page_size//2} (mid freq): gate={mid:.2f} → balanced")
        print(f"    Dim {page_size-1} (high freq): gate={high:.2f} → {high:.0%} update")


def verify_atom_orthogonality(atoms_module) -> bool:
    """Verify that Fourier-initialized atoms are approximately orthogonal."""
    A = atoms_module.A['q_proj'][:, 0]  # (64, d_model, rank) for layer 0
    flat = A.reshape(A.size(0), -1)  # (64, d_model * rank)
    flat_norm = F.normalize(flat, dim=-1)
    cos_matrix = flat_norm @ flat_norm.T  # (64, 64)

    off_diag = cos_matrix - torch.eye(cos_matrix.size(0), device=cos_matrix.device)
    max_cos = off_diag.abs().max().item()
    mean_cos = off_diag.abs().mean().item()

    print(f"  Orthogonality check:")
    print(f"    Max off-diagonal cosine:  {max_cos:.6f} (target < 0.01)")
    print(f"    Mean off-diagonal cosine: {mean_cos:.6f} (target < 0.001)")

    return max_cos < 0.05


def apply_fourier_inits(model) -> None:
    """
    Apply both Fourier initializations at model creation.
    Call ONCE before training begins. Do NOT call when warm-starting.

    Args:
        model: AtomLoRAModel with .atoms (LoRAAtoms) and .residual_gate (ResidualPageGate)
    """
    print("Applying Fourier initializations:")

    # 1. Fourier atom initialization (orthogonal basis)
    apply_fourier_atom_init(model.atoms)

    # 2. Frequency-aware residual gate initialization
    if hasattr(model, 'residual_gate'):
        init_residual_gate_fourier(model.residual_gate)

    # 3. Verify
    verify_atom_orthogonality(model.atoms)

    print("Fourier initializations complete.")
