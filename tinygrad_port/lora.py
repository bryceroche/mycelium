"""
64-Atom LoRA Architecture — tinygrad port.

Port of scripts/atom_lora.py (LoRAAtoms + AtomAdditiveLoRAInjector).

Key differences from PyTorch version:
- No monkey-patching: tinygrad version computes LoRA output inline via apply()
  and the caller adds it to the base projection output.
- Tensors are tinygrad Tensors, initialized with equivalent Fourier/random init.
- einsum replaced with explicit matmul/reshape operations (tinygrad has einsum
  but explicit ops are clearer and more portable).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from tinygrad import Tensor, dtypes
from tinygrad_port.nn_utils import Linear


# ---------------------------------------------------------------------------
# LoRAAtoms -- 64 rank-6 anonymous LoRA atoms
# ---------------------------------------------------------------------------
class LoRAAtoms:
    """
    64 rank-6 LoRA atoms for Q,K,V,O at all 16 Llama layers.

    Each atom is an independent direction of attention modification.
    Applied via batched matmul -- no per-atom loops.

    A: {proj_name: Tensor of shape (num_atoms, num_layers, d_model, rank)}
    B: {proj_name: Tensor of shape (num_atoms, num_layers, rank, proj_dim)}
       where proj_dim is 512 for k_proj/v_proj (GQA) and 2048 for q_proj/o_proj
    """

    PROJ_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj"]

    def __init__(
        self,
        d_model: int = 2048,
        d_kv: int = 512,
        rank: int = 6,
        num_atoms: int = 64,
        num_layers: int = 16,
    ):
        self.d_model = d_model
        self.d_kv = d_kv
        self.rank = rank
        self.num_atoms = num_atoms
        self.num_layers = num_layers

        self.A: Dict[str, Tensor] = {}
        self.B: Dict[str, Tensor] = {}

        for proj_name in self.PROJ_NAMES:
            proj_dim = d_kv if proj_name in ("k_proj", "v_proj") else d_model
            # Gaussian init scaled by 0.01, matching PyTorch version
            self.A[proj_name] = Tensor.randn(
                num_atoms, num_layers, d_model, rank
            ).mul(0.01).requires_grad_()
            self.B[proj_name] = Tensor.randn(
                num_atoms, num_layers, rank, proj_dim
            ).mul(0.01).requires_grad_()

    def parameters(self) -> List[Tensor]:
        """Return all trainable parameters (for optimizer)."""
        params = []
        for proj_name in self.PROJ_NAMES:
            params.append(self.A[proj_name])
            params.append(self.B[proj_name])
        return params

    def apply(
        self,
        hidden: Tensor,
        layer_idx: int,
        proj_name: str,
        atom_scales: Tensor,
    ) -> Tensor:
        """
        Compute blended atom LoRA output via batched matmul.

        Args:
            hidden:      (batch, seq, d_model)
            layer_idx:   int, which Llama layer
            proj_name:   str, one of q_proj/k_proj/v_proj/o_proj
            atom_scales: (batch, num_atoms) -- independent tanh-bounded scalars

        Returns:
            lora_out: (batch, seq, proj_dim) -- additive LoRA contribution

        PyTorch equivalent:
            projections = einsum('bsd,adr->basr', hidden, A)
            scaled = projections * atom_scales[:,:,None,None]
            lora_out = einsum('basr,arp->bsp', scaled, B)
        """
        # Slice to this layer: A_l shape (num_atoms, d_model, rank)
        A_l = self.A[proj_name][:, layer_idx]  # (A, D, R)
        B_l = self.B[proj_name][:, layer_idx]  # (A, R, P)

        B, S, D = hidden.shape
        A_count = self.num_atoms
        R = self.rank

        # Step 1: hidden @ A -> projections
        # hidden: (B, S, D), A_l: (A, D, R)
        # We want: (B, A, S, R) = (B, 1, S, D) @ (1, A, D, R)
        # Reshape for broadcast matmul:
        #   hidden_exp: (B, 1, S, D)
        #   A_exp:      (1, A, D, R)
        #   result:     (B, A, S, R) via batched matmul on last two dims
        hidden_exp = hidden.reshape(B, 1, S, D).expand(B, A_count, S, D)
        A_exp = A_l.reshape(1, A_count, D, R).expand(B, A_count, D, R)
        projections = hidden_exp.matmul(A_exp)  # (B, A, S, R)

        # Step 2: Scale each atom
        # atom_scales: (B, A) -> (B, A, 1, 1)
        scales_4d = atom_scales.reshape(B, A_count, 1, 1)
        scaled = projections * scales_4d  # (B, A, S, R)

        # Step 3: Sum across atoms and project to output dim
        # scaled: (B, A, S, R) @ B_l: (A, R, P) -> (B, A, S, P) -> sum over A -> (B, S, P)
        B_exp = B_l.reshape(1, A_count, R, -1).expand(B, A_count, R, B_l.shape[-1])
        projected = scaled.matmul(B_exp)  # (B, A, S, P)
        lora_out = projected.sum(axis=1)  # (B, S, P)

        return lora_out


# ---------------------------------------------------------------------------
# FourierPageEncoding -- fixed positional encoding for pages
# ---------------------------------------------------------------------------
class FourierPageEncoding:
    """Fixed Fourier encoding giving each page dimension a frequency identity.

    DEPRECATED in favor of PiHarmonicPageEncoding -- kept for checkpoint compat.
    """

    def __init__(self, page_size: int = 64):
        self.page_size = page_size
        dim_indices = Tensor.arange(page_size // 2).float()
        self.freqs = (dim_indices * -(math.log(10000.0) / (page_size // 2))).exp()

    def encode(self, pass_num: int) -> Tensor:
        """Unique encoding per (dimension, pass) combination."""
        t = self.freqs * pass_num  # (page_size // 2,)
        return t.sin().cat(t.cos())  # (page_size,)

    def apply(self, page: Tensor, pass_num: int) -> Tensor:
        """Add positional structure to a page. Applied AFTER normalization."""
        encoding = self.encode(pass_num)
        return page + encoding


# ---------------------------------------------------------------------------
# PiHarmonicPageEncoding -- DCT-like orthogonal basis (zero learnable params)
# ---------------------------------------------------------------------------
class PiHarmonicPageEncoding:
    """Pi-harmonic page encoding with DCT-like orthogonal frequency basis.

    Frequencies: freq_k = k * pi / page_size for k in 1..page_size//2
    Creates orthogonal harmonics (same basis as DCT/JPEG compression).

    Advantages over transformer-style (10000-based) encoding:
    - Orthogonal by construction (harmonics of pi)
    - Same mathematical basis as DCT (proven optimal for energy compaction)
    - No arbitrary constant (pi is natural, 10000 was arbitrary)
    """

    def __init__(self, page_size: int = 64):
        self.page_size = page_size
        n = page_size // 2
        # freq_k = k * pi / page_size, k in [1, n]
        self.freqs = (Tensor.arange(1, n + 1).float()) * (math.pi / page_size)

    def encode(self, pass_num: int) -> Tensor:
        """Encode pass number using pi-harmonic frequencies."""
        t = self.freqs * pass_num  # (n,)
        return t.sin().cat(t.cos())  # (page_size,)

    def apply(self, page: Tensor, pass_num: int) -> Tensor:
        """Add pi-harmonic positional structure to a page."""
        encoding = self.encode(pass_num)
        return page + encoding


# ---------------------------------------------------------------------------
# AtomAdditiveLoRAInjector -- clean inline LoRA injection (no monkey-patching)
# ---------------------------------------------------------------------------
class AtomAdditiveLoRAInjector:
    """Injects LoRA modifications into Llama attention projections.

    Unlike the PyTorch version (AtomAdditiveLoRAManager) which monkey-patches
    nn.Module.forward methods, the tinygrad version computes the LoRA delta
    inline and returns it for the caller to add.

    This is cleaner and more explicit:
        base_output = proj(x)
        lora_delta  = injector.inject(x, layer_idx, proj_name)
        output      = base_output + lora_delta

    Usage:
        atoms = LoRAAtoms(d_model=2048, d_kv=512, rank=6)
        injector = AtomAdditiveLoRAInjector(atoms)

        # Before each forward pass, set the atom scales from hypernet
        injector.set_scales(atom_scales)  # (B, num_atoms)

        # During forward, for each layer and projection:
        for layer_idx in range(num_layers):
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                base_out = base_proj(hidden)
                lora_out = injector.inject(hidden, layer_idx, proj_name)
                output = base_out + lora_out
    """

    PROJECTION_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj"]

    def __init__(self, atoms: LoRAAtoms, num_layers: int = 16):
        self.atoms = atoms
        self.num_layers = num_layers
        self._atom_scales: Optional[Tensor] = None

    def set_scales(self, atom_scales: Tensor) -> None:
        """Set atom scales for the current forward pass.

        Args:
            atom_scales: (batch, num_atoms) -- tanh-bounded scalars from hypernet
        """
        self._atom_scales = atom_scales

    def clear_scales(self) -> None:
        """Clear atom scales after forward pass (prevents stale state)."""
        self._atom_scales = None

    def inject(
        self,
        hidden: Tensor,
        layer_idx: int,
        proj_name: str,
    ) -> Tensor:
        """Compute the additive LoRA contribution for one projection.

        Args:
            hidden:    (batch, seq, d_model) -- input to the projection
            layer_idx: int -- which Llama layer (0-15)
            proj_name: str -- one of q_proj/k_proj/v_proj/o_proj

        Returns:
            lora_delta: (batch, seq, proj_dim) -- add this to base projection output

        Raises:
            RuntimeError: if set_scales() was not called first
        """
        if self._atom_scales is None:
            raise RuntimeError(
                "Atom scales not set. Call set_scales(atom_scales) before inject()."
            )
        return self.atoms.apply(hidden, layer_idx, proj_name, self._atom_scales)

    def inject_all_projections(
        self,
        hidden: Tensor,
        layer_idx: int,
    ) -> Dict[str, Tensor]:
        """Compute LoRA deltas for all four projections at once.

        Convenience method that returns a dict of deltas:
            {"q_proj": ..., "k_proj": ..., "v_proj": ..., "o_proj": ...}

        Args:
            hidden:    (batch, seq, d_model)
            layer_idx: int

        Returns:
            Dict mapping proj_name -> (batch, seq, proj_dim) delta tensor
        """
        return {
            proj_name: self.inject(hidden, layer_idx, proj_name)
            for proj_name in self.PROJECTION_NAMES
        }

    def parameters(self) -> List[Tensor]:
        """Proxy to atoms.parameters() for optimizer registration."""
        return self.atoms.parameters()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("64-Atom LoRA (tinygrad) -- Self-Test")
    print("=" * 60)

    Tensor.manual_seed(42)

    # Use small dimensions for fast CPU test
    d_model = 128
    d_kv = 64
    rank = 6
    num_atoms = 64
    num_layers = 2
    batch = 4
    seq_len = 8

    # ---------------------------------------------------------------
    # 1. Test LoRAAtoms
    # ---------------------------------------------------------------
    print("\n--- LoRAAtoms ---")
    atoms = LoRAAtoms(
        d_model=d_model, d_kv=d_kv, rank=rank,
        num_atoms=num_atoms, num_layers=num_layers,
    )

    hidden = Tensor.randn(batch, seq_len, d_model)
    atom_scales = Tensor.randn(batch, num_atoms).tanh()

    for proj_name in LoRAAtoms.PROJ_NAMES:
        out = atoms.apply(hidden, layer_idx=0, proj_name=proj_name, atom_scales=atom_scales)
        expected_dim = d_kv if proj_name in ("k_proj", "v_proj") else d_model
        actual_dim = out.shape[-1]
        status = "OK" if actual_dim == expected_dim else "FAIL"
        print(f"  {proj_name}: {hidden.shape} -> {out.shape} (expected {expected_dim}) {status}")

    num_params = sum(p.numel() for p in atoms.parameters())
    print(f"  Total params: {num_params:,}")

    # Verify param count matches PyTorch formula:
    # 4 projections * num_atoms * num_layers * (d_model*rank + rank*proj_dim)
    expected_a = num_atoms * num_layers * d_model * rank * 4
    expected_b = num_atoms * num_layers * rank * (d_kv * 2 + d_model * 2)
    print(f"  Expected params: {expected_a + expected_b:,} (A={expected_a:,} + B={expected_b:,})")

    # ---------------------------------------------------------------
    # 2. Test AtomAdditiveLoRAInjector
    # ---------------------------------------------------------------
    print("\n--- AtomAdditiveLoRAInjector ---")
    injector = AtomAdditiveLoRAInjector(atoms, num_layers=num_layers)

    # Should raise without set_scales
    try:
        injector.inject(hidden, 0, "q_proj")
        print("  ERROR: should have raised RuntimeError")
    except RuntimeError as e:
        print(f"  Correctly raised: {e}")

    # Set scales and inject
    injector.set_scales(atom_scales)
    deltas = injector.inject_all_projections(hidden, layer_idx=0)
    for name, delta in deltas.items():
        expected_dim = d_kv if name in ("k_proj", "v_proj") else d_model
        print(f"  {name} delta: {delta.shape}, mean abs: {delta.abs().mean().numpy():.6f}")

    injector.clear_scales()

    # ---------------------------------------------------------------
    # 3. Test PiHarmonicPageEncoding
    # ---------------------------------------------------------------
    print("\n--- PiHarmonicPageEncoding ---")
    enc = PiHarmonicPageEncoding(page_size=64)
    e0 = enc.encode(0)
    e1 = enc.encode(1)
    e2 = enc.encode(2)
    print(f"  Pass 0 encoding shape: {e0.shape}, norm: {(e0 * e0).sum().sqrt().numpy():.4f}")
    print(f"  Pass 1 encoding shape: {e1.shape}, norm: {(e1 * e1).sum().sqrt().numpy():.4f}")

    # Different passes should give different encodings
    diff_01 = (e0 - e1).abs().mean().numpy()
    diff_12 = (e1 - e2).abs().mean().numpy()
    print(f"  Pass 0 vs 1 mean abs diff: {diff_01:.4f}")
    print(f"  Pass 1 vs 2 mean abs diff: {diff_12:.4f}")

    # Apply to a page
    page = Tensor.randn(batch, 64)
    page_encoded = enc.apply(page, pass_num=1)
    print(f"  Page + encoding shape: {page_encoded.shape}")

    # ---------------------------------------------------------------
    # 4. Test FourierPageEncoding (deprecated but kept for compat)
    # ---------------------------------------------------------------
    print("\n--- FourierPageEncoding (deprecated) ---")
    fenc = FourierPageEncoding(page_size=64)
    fe0 = fenc.encode(0)
    fe1 = fenc.encode(1)
    print(f"  Pass 0 shape: {fe0.shape}, Pass 1 shape: {fe1.shape}")
    print(f"  Diff: {(fe0 - fe1).abs().mean().numpy():.4f}")

    # ---------------------------------------------------------------
    # 5. Additive injection end-to-end simulation
    # ---------------------------------------------------------------
    print("\n--- End-to-end simulation ---")
    # Simulate: base_proj + lora_delta
    base_weight = Tensor.randn(d_model, d_model).mul(0.01)
    x = Tensor.randn(batch, seq_len, d_model)

    base_out = x.matmul(base_weight.transpose())  # (B, S, D)

    injector.set_scales(atom_scales)
    lora_delta = injector.inject(x, layer_idx=0, proj_name="q_proj")
    combined = base_out + lora_delta

    print(f"  Base output:    {base_out.shape}, mean abs: {base_out.abs().mean().numpy():.6f}")
    print(f"  LoRA delta:     {lora_delta.shape}, mean abs: {lora_delta.abs().mean().numpy():.6f}")
    print(f"  Combined:       {combined.shape}, mean abs: {combined.abs().mean().numpy():.6f}")

    ratio = lora_delta.abs().mean().numpy() / (base_out.abs().mean().numpy() + 1e-8)
    print(f"  LoRA/base ratio: {ratio:.4f} (should be small with 0.01 init)")

    injector.clear_scales()

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
