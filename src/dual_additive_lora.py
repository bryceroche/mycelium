"""
DualAdditiveLoRAManager: applies blended forward+verify LoRA inline.

At each projection layer:
    q_forward = (x @ A_forward) * forward_scales @ B_forward
    q_verify  = (x @ A_verify)  * verify_scales  @ B_verify
    q_lora    = (1 - blend) * q_forward + blend * q_verify
    output    = W @ x + q_lora

The blend weight is per-batch (B, 1), allowing different examples in a batch
to have different compute/verify ratios. In practice, blend is constant within
a batch (same pass_num) but varies across passes.
"""

import torch
import torch.nn as nn
from typing import Dict


class DualAdditiveLoRAManager:
    """
    Manages inline blended dual LoRA (forward + verify) on Llama attention.

    Same monkey-patching approach as AdditiveLoRAManager, but the patched
    forward computes two LoRA terms and blends them.
    """

    PROJECTION_NAMES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    def __init__(self, model: nn.Module):
        self.model = model
        self._original_forwards: Dict[int, Dict[str, callable]] = {}
        self._active = False

        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.num_layers = len(model.model.layers)
        else:
            self.num_layers = 16

    def _get_projection(self, layer_idx: int, proj_name: str) -> nn.Module:
        return getattr(self.model.model.layers[layer_idx].self_attn, proj_name)

    @staticmethod
    def _make_dual_lora_forward(
        original_forward: callable,
        A_fwd: torch.Tensor,
        B_fwd: torch.Tensor,
        scales_fwd: torch.Tensor,
        A_ver: torch.Tensor,
        B_ver: torch.Tensor,
        scales_ver: torch.Tensor,
        blend: torch.Tensor,
    ) -> callable:
        """
        Create a patched forward that blends two LoRA terms.

        blend: (B, 1) — broadcast over seq and proj dims.
        """
        def forward(x: torch.Tensor) -> torch.Tensor:
            base_output = original_forward(x)

            dtype = x.dtype
            device = x.device

            # Forward LoRA
            a_f = A_fwd.to(dtype=dtype, device=device)
            b_f = B_fwd.to(dtype=dtype, device=device)
            s_f = scales_fwd.to(dtype=dtype, device=device)
            lora_fwd = (x @ a_f * s_f.unsqueeze(1)) @ b_f

            # Verify LoRA
            a_v = A_ver.to(dtype=dtype, device=device)
            b_v = B_ver.to(dtype=dtype, device=device)
            s_v = scales_ver.to(dtype=dtype, device=device)
            lora_ver = (x @ a_v * s_v.unsqueeze(1)) @ b_v

            # Blend: (B, 1, 1) for broadcasting over (B, seq, proj_dim)
            b = blend.to(dtype=dtype, device=device).unsqueeze(1)
            lora_out = (1 - b) * lora_fwd + b * lora_ver

            return base_output + lora_out

        return forward

    def apply(
        self,
        forward_mods: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
        verify_mods: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
        blend: torch.Tensor,
    ) -> None:
        """
        Apply blended dual LoRA.

        Args:
            forward_mods: lora_mods dict for forward (computation) templates
            verify_mods:  lora_mods dict for verify (checking) templates
            blend:        (B, 1) blend weight [0=all forward, 1=all verify]
        """
        if self._active:
            raise RuntimeError(
                "Dual LoRA already applied. Call remove() before applying again."
            )

        self._original_forwards = {}

        for layer_idx in forward_mods:
            self._original_forwards[layer_idx] = {}

            for proj_name in forward_mods[layer_idx]:
                if proj_name not in self.PROJECTION_NAMES:
                    continue

                proj_module = self._get_projection(layer_idx, proj_name)
                self._original_forwards[layer_idx][proj_name] = proj_module.forward

                fwd = forward_mods[layer_idx][proj_name]
                ver = verify_mods[layer_idx][proj_name]

                proj_module.forward = self._make_dual_lora_forward(
                    proj_module.forward,
                    fwd['A'], fwd['B'], fwd['scales'],
                    ver['A'], ver['B'], ver['scales'],
                    blend,
                )

        self._active = True

    def remove(self) -> None:
        if not self._active:
            return
        for layer_idx, layer_forwards in self._original_forwards.items():
            for proj_name, original_forward in layer_forwards.items():
                proj_module = self._get_projection(layer_idx, proj_name)
                proj_module.forward = original_forward
        self._original_forwards = {}
        self._active = False

    def is_active(self) -> bool:
        return self._active

    def __del__(self):
        if self._active:
            self.remove()


if __name__ == '__main__':
    print("Testing DualAdditiveLoRAManager...")

    # Mock model
    class MockSelfAttn(nn.Module):
        def __init__(self, d, dk):
            super().__init__()
            self.q_proj = nn.Linear(d, d, bias=False)
            self.k_proj = nn.Linear(d, dk, bias=False)
            self.v_proj = nn.Linear(d, dk, bias=False)
            self.o_proj = nn.Linear(d, d, bias=False)

    class MockLayer(nn.Module):
        def __init__(self, d, dk):
            super().__init__()
            self.self_attn = MockSelfAttn(d, dk)

    class MockLlama(nn.Module):
        def __init__(self, d=64, dk=16, n=2):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([MockLayer(d, dk) for _ in range(n)])

    d, dk, rank, B, S = 64, 16, 4, 2, 10
    model = MockLlama(d, dk, 2)
    x = torch.randn(B, S, d)

    proj_dims = {'q_proj': d, 'k_proj': dk, 'v_proj': dk, 'o_proj': d}

    def make_mods():
        mods = {}
        for li in range(2):
            mods[li] = {}
            for pn, pd in proj_dims.items():
                mods[li][pn] = {
                    'A': torch.randn(d, rank) * 0.1,
                    'B': torch.randn(rank, pd) * 0.1,
                    'scales': torch.randn(B, rank).tanh(),
                }
        return mods

    fwd_mods = make_mods()
    ver_mods = make_mods()

    # Test blend=0 (pure forward)
    baseline = model.model.layers[0].self_attn.q_proj(x)

    blend_0 = torch.zeros(B, 1)
    mgr = DualAdditiveLoRAManager(model)
    mgr.apply(fwd_mods, ver_mods, blend_0)
    out_blend0 = model.model.layers[0].self_attn.q_proj(x)
    mgr.remove()

    # Test blend=1 (pure verify)
    blend_1 = torch.ones(B, 1)
    mgr.apply(fwd_mods, ver_mods, blend_1)
    out_blend1 = model.model.layers[0].self_attn.q_proj(x)
    mgr.remove()

    # Test blend=0.5 (mixed)
    blend_half = torch.full((B, 1), 0.5)
    mgr.apply(fwd_mods, ver_mods, blend_half)
    out_blend_half = model.model.layers[0].self_attn.q_proj(x)
    mgr.remove()

    print(f"  baseline norm: {baseline.norm():.4f}")
    print(f"  blend=0 diff from baseline: {(out_blend0 - baseline).norm():.4f}")
    print(f"  blend=1 diff from baseline: {(out_blend1 - baseline).norm():.4f}")
    print(f"  blend=0 vs blend=1 diff: {(out_blend0 - out_blend1).norm():.4f}")

    # blend=0.5 should be midpoint
    expected_mid = (out_blend0 + out_blend1) / 2
    mid_err = (out_blend_half - expected_mid).norm().item()
    print(f"  blend=0.5 vs midpoint error: {mid_err:.6f}")
    assert mid_err < 1e-5, f"blend=0.5 should be midpoint, error={mid_err}"

    # Gradient flow
    blend_g = torch.tensor([[0.3], [0.7]], requires_grad=True)
    x_g = torch.randn(B, S, d, requires_grad=True)
    mgr.apply(fwd_mods, ver_mods, blend_g)
    out = model.model.layers[0].self_attn.q_proj(x_g)
    out.sum().backward()
    mgr.remove()
    assert blend_g.grad is not None, "blend should receive gradients"
    assert x_g.grad is not None, "input should receive gradients"
    print(f"  blend grad: {blend_g.grad.squeeze().tolist()}")
    print("  gradients flow: OK")

    print("\nAll dual LoRA tests passed!")
