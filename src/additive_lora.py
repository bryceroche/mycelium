"""
Additive LoRA: Inline LoRA without forward hooks.

Replaces the slow hook-based approach (~13 min/epoch) with monkey-patched
forward methods on Q, K, V, O projection layers. The LoRA modification
happens INSIDE the forward pass, so output_hidden_states=True returns
hidden states that reflect the LoRA modification.

The formula per projection:
    output = W @ x + bias + (x @ A * scales) @ B

Where:
    - W: Original weight matrix of the projection
    - A: Down-projection template (d_model, rank)
    - B: Up-projection template (rank, proj_dim)
    - scales: State-derived per-batch scaling factors (batch, rank)

Usage:
    from src.additive_lora import AdditiveLoRAManager

    manager = AdditiveLoRAManager(model.transformer)

    # Before each thinking pass:
    lora_mods = lora_module(state)           # or lora_module(state, strategy)
    manager.apply(lora_mods)

    # Forward pass (LoRA is active inline)
    outputs = model.transformer(input_ids=..., output_hidden_states=True)

    # After pass:
    manager.remove()

Why this is faster than hooks:
    - Hooks have Python-level overhead per call (function dispatch, tuple packing)
    - Monkey-patching replaces the forward method directly — same Python overhead
      as a normal forward call
    - More importantly: no hook registration/removal overhead per pass
    - The patched forward is a single fused computation, not original + hook delta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from functools import partial


class AdditiveLoRAManager:
    """
    Manages inline additive LoRA on Llama attention projection layers.

    Instead of registering forward hooks (slow), this manager monkey-patches
    the forward() method of each nn.Linear projection to include the LoRA
    additive term. The original forward is saved and restored on remove().

    This ensures that output_hidden_states=True returns hidden states that
    reflect the LoRA modification, because the modification happens INSIDE
    the layer's forward pass.

    Attributes:
        model: The transformer model (LlamaForCausalLM)
        _original_forwards: Saved original forward methods for restoration
        _active: Whether LoRA patches are currently applied

    Example:
        manager = AdditiveLoRAManager(model.transformer)
        lora_mods = lora_module(state)
        manager.apply(lora_mods)
        outputs = model.transformer(input_ids=ids, output_hidden_states=True)
        manager.remove()
    """

    PROJECTION_NAMES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    def __init__(self, model: nn.Module):
        """
        Initialize the manager.

        Args:
            model: The transformer model (e.g., LlamaForCausalLM).
                   Expects layers at model.model.layers[i].self_attn
        """
        self.model = model
        self._original_forwards: Dict[int, Dict[str, callable]] = {}
        self._active = False

        # Determine number of layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.num_layers = len(model.model.layers)
        else:
            self.num_layers = 16  # Default for Llama 3.2 1B

    def _get_projection(self, layer_idx: int, proj_name: str) -> nn.Module:
        """Get the projection module for a given layer and projection name."""
        return getattr(self.model.model.layers[layer_idx].self_attn, proj_name)

    @staticmethod
    def _make_lora_forward(
        original_forward: callable,
        A: torch.Tensor,
        B: torch.Tensor,
        scales: torch.Tensor,
    ) -> callable:
        """
        Create a patched forward that adds the LoRA term inline.

        The patched forward computes:
            output = original_forward(x) + (x @ A * scales) @ B

        Args:
            original_forward: The original forward method of the nn.Linear
            A: Down-projection template (d_model, rank)
            B: Up-projection template (rank, proj_dim)
            scales: Per-batch scaling factors (batch, rank)

        Returns:
            Patched forward function
        """
        def forward(x: torch.Tensor) -> torch.Tensor:
            # Original linear output
            base_output = original_forward(x)

            # LoRA additive term:
            # x: (batch, seq, d_model)
            # A: (d_model, rank)
            # scales: (batch, rank) -> (batch, 1, rank) for broadcasting over seq
            # B: (rank, proj_dim)
            A_dev = A.to(dtype=x.dtype, device=x.device)
            B_dev = B.to(dtype=x.dtype, device=x.device)
            scales_dev = scales.to(dtype=x.dtype, device=x.device)

            # (batch, seq, d_model) @ (d_model, rank) = (batch, seq, rank)
            lora_down = x @ A_dev

            # Scale: (batch, seq, rank) * (batch, 1, rank) = (batch, seq, rank)
            lora_scaled = lora_down * scales_dev.unsqueeze(1)

            # (batch, seq, rank) @ (rank, proj_dim) = (batch, seq, proj_dim)
            lora_out = lora_scaled @ B_dev

            return base_output + lora_out

        return forward

    def apply(self, lora_mods: Dict[int, Dict[str, Dict[str, torch.Tensor]]]) -> None:
        """
        Apply LoRA modifications by monkey-patching projection forward methods.

        Args:
            lora_mods: Dictionary mapping layer_idx -> projection_name -> {A, B, scales}
                       Same format as StateConditionedLoRA output.

        Raises:
            RuntimeError: If LoRA is already applied (must remove first)
        """
        if self._active:
            raise RuntimeError(
                "Additive LoRA already applied. Call remove() before applying again."
            )

        self._original_forwards = {}

        for layer_idx, layer_mods in lora_mods.items():
            self._original_forwards[layer_idx] = {}

            for proj_name, params in layer_mods.items():
                if proj_name not in self.PROJECTION_NAMES:
                    continue

                proj_module = self._get_projection(layer_idx, proj_name)

                # Save original forward
                self._original_forwards[layer_idx][proj_name] = proj_module.forward

                # Create and set patched forward
                A = params['A']       # (d_model, rank)
                B = params['B']       # (rank, proj_dim)
                scales = params['scales']  # (batch, rank)

                proj_module.forward = self._make_lora_forward(
                    proj_module.forward, A, B, scales,
                )

        self._active = True

    def remove(self) -> None:
        """
        Remove all LoRA patches, restoring original forward methods.

        Safe to call even if no patches are applied.
        """
        if not self._active:
            return

        for layer_idx, layer_forwards in self._original_forwards.items():
            for proj_name, original_forward in layer_forwards.items():
                proj_module = self._get_projection(layer_idx, proj_name)
                proj_module.forward = original_forward

        self._original_forwards = {}
        self._active = False

    def is_active(self) -> bool:
        """Check if LoRA patches are currently applied."""
        return self._active

    def __del__(self):
        """Cleanup patches on deletion."""
        if self._active:
            self.remove()


# Module-level convenience functions (mirror lora_hooks.py API)

_global_managers: Dict[int, AdditiveLoRAManager] = {}


def apply_lora_additive(
    model: nn.Module,
    lora_module: nn.Module,
    state: torch.Tensor,
    strategy: Optional[torch.Tensor] = None,
) -> AdditiveLoRAManager:
    """
    Apply state-conditioned LoRA as additive term (no hooks).

    Works with both v1 (state only) and v3 (state + strategy) hypernetworks.

    Args:
        model: The transformer model (LlamaForCausalLM)
        lora_module: StateConditionedLoRA module (v1 or v3)
        state: State vector on hypersphere (batch, state_size)
        strategy: Optional strategy vector (batch, strategy_size) for v3

    Returns:
        AdditiveLoRAManager instance
    """
    manager = AdditiveLoRAManager(model)

    # v3 hypernetwork accepts strategy kwarg; v1 does not
    if strategy is not None:
        lora_mods = lora_module(state, strategy=strategy)
    else:
        lora_mods = lora_module(state)

    manager.apply(lora_mods)

    # Store in global registry for remove_lora_additive()
    model_id = id(model)
    _global_managers[model_id] = manager

    return manager


def remove_lora_additive(model: nn.Module) -> None:
    """
    Remove all additive LoRA patches from a model.

    Safe to call even if no patches are applied.

    Args:
        model: The transformer model
    """
    model_id = id(model)

    if model_id in _global_managers:
        _global_managers[model_id].remove()
        del _global_managers[model_id]


def _test_additive_lora():
    """
    Test additive LoRA with a mock model.

    Validates:
    1. Patches can be applied and removed
    2. LoRA modification changes output
    3. Output returns to baseline after removal
    4. Gradient flow works through the additive term
    5. Double-apply protection works
    6. Works with v3 strategy argument
    """
    print("Testing Additive LoRA (no hooks)...")
    print("=" * 60)

    # Create minimal mock of Llama's attention structure
    class MockSelfAttn(nn.Module):
        def __init__(self, d_model, d_kv):
            super().__init__()
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_kv, bias=False)
            self.v_proj = nn.Linear(d_model, d_kv, bias=False)
            self.o_proj = nn.Linear(d_model, d_model, bias=False)

    class MockLayer(nn.Module):
        def __init__(self, d_model, d_kv):
            super().__init__()
            self.self_attn = MockSelfAttn(d_model, d_kv)

    class MockLlama(nn.Module):
        def __init__(self, d_model=64, d_kv=16, num_layers=2):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([
                MockLayer(d_model, d_kv) for _ in range(num_layers)
            ])

    # Setup
    d_model = 64
    d_kv = 16
    num_layers = 2
    rank = 4
    batch_size = 2
    seq_len = 10

    mock_model = MockLlama(d_model=d_model, d_kv=d_kv, num_layers=num_layers)
    print(f"Mock model: {num_layers} layers, d_model={d_model}, d_kv={d_kv}, rank={rank}")

    # Create dummy LoRA modifications (GQA-aware)
    proj_dims = {'q_proj': d_model, 'k_proj': d_kv, 'v_proj': d_kv, 'o_proj': d_model}

    def create_dummy_lora_mods():
        lora_mods = {}
        for layer_idx in range(num_layers):
            lora_mods[layer_idx] = {}
            for proj_name, proj_dim in proj_dims.items():
                lora_mods[layer_idx][proj_name] = {
                    'A': torch.randn(d_model, rank) * 0.1,
                    'B': torch.randn(rank, proj_dim) * 0.1,
                    'scales': torch.randn(batch_size, rank).tanh(),
                }
        return lora_mods

    # Test input
    x = torch.randn(batch_size, seq_len, d_model)

    # 1. Baseline output (no LoRA)
    print("\n1. Baseline output (no LoRA):")
    q_proj = mock_model.model.layers[0].self_attn.q_proj
    k_proj = mock_model.model.layers[0].self_attn.k_proj
    output_baseline_q = q_proj(x)
    output_baseline_k = k_proj(x)
    print(f"   Q shape: {output_baseline_q.shape}, K shape: {output_baseline_k.shape}")

    # 2. Apply additive LoRA
    print("\n2. Apply additive LoRA:")
    lora_mods = create_dummy_lora_mods()
    manager = AdditiveLoRAManager(mock_model)
    manager.apply(lora_mods)
    print(f"   is_active: {manager.is_active()}")
    assert manager.is_active(), "Should be active"

    # 3. Output with LoRA
    print("\n3. Output with LoRA:")
    output_with_lora_q = q_proj(x)
    output_with_lora_k = k_proj(x)
    print(f"   Q shape: {output_with_lora_q.shape}, K shape: {output_with_lora_k.shape}")

    diff_q = (output_with_lora_q - output_baseline_q).abs().mean().item()
    diff_k = (output_with_lora_k - output_baseline_k).abs().mean().item()
    print(f"   Q diff from baseline: {diff_q:.6f}")
    print(f"   K diff from baseline: {diff_k:.6f}")
    assert diff_q > 0, "Q LoRA should change output"
    assert diff_k > 0, "K LoRA should change output"
    print("   Output changed: OK")

    # 4. Remove LoRA
    print("\n4. Remove LoRA patches:")
    manager.remove()
    print(f"   is_active: {manager.is_active()}")
    assert not manager.is_active(), "Should be inactive"

    output_after_q = q_proj(x)
    diff_after = (output_after_q - output_baseline_q).abs().mean().item()
    print(f"   Q diff from baseline after remove: {diff_after:.8f}")
    assert diff_after < 1e-6, "Output should match baseline after removing LoRA"
    print("   Output matches baseline: OK")

    # 5. Double-apply protection
    print("\n5. Double-apply protection:")
    manager2 = AdditiveLoRAManager(mock_model)
    manager2.apply(lora_mods)
    try:
        manager2.apply(lora_mods)
        print("   ERROR: Should have raised RuntimeError")
        assert False
    except RuntimeError as e:
        print(f"   Caught expected error: {e}")
    manager2.remove()

    # 6. Gradient flow
    print("\n6. Gradient flow through additive LoRA:")
    x_grad = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    # Create LoRA mods with gradient-tracked scales (leaf tensors)
    scales_leaf = torch.randn(batch_size, rank, requires_grad=True)
    A_leaf = nn.Parameter(torch.randn(d_model, rank) * 0.1)
    B_leaf = nn.Parameter(torch.randn(rank, d_model) * 0.1)
    lora_mods_grad = {}
    for layer_idx in range(num_layers):
        lora_mods_grad[layer_idx] = {}
        for proj_name, proj_dim in proj_dims.items():
            lora_mods_grad[layer_idx][proj_name] = {
                'A': A_leaf,
                'B': B_leaf if proj_dim == d_model else nn.Parameter(torch.randn(rank, proj_dim) * 0.1),
                'scales': scales_leaf,
            }

    manager3 = AdditiveLoRAManager(mock_model)
    manager3.apply(lora_mods_grad)

    output = q_proj(x_grad)
    loss = output.sum()
    loss.backward()

    assert x_grad.grad is not None, "No gradient on input"
    assert scales_leaf.grad is not None, "No gradient on scales"
    assert A_leaf.grad is not None, "No gradient on A template"
    print(f"   Input grad norm: {x_grad.grad.norm().item():.4f}")
    print(f"   Scales grad norm: {scales_leaf.grad.norm().item():.4f}")
    print(f"   A template grad norm: {A_leaf.grad.norm().item():.4f}")
    print("   Gradients flow: OK")

    manager3.remove()

    # 7. Module-level convenience functions
    print("\n7. Module-level apply/remove functions:")

    class MockLoRAModule(nn.Module):
        def __init__(self, lora_mods):
            super().__init__()
            self._mods = lora_mods

        def forward(self, state, strategy=None):
            return self._mods

    mock_lora = MockLoRAModule(create_dummy_lora_mods())
    state = torch.randn(batch_size, 64)

    mgr = apply_lora_additive(mock_model, mock_lora, state)
    assert mgr.is_active()
    output_conv = q_proj(x)
    diff_conv = (output_conv - output_baseline_q).abs().mean().item()
    assert diff_conv > 0, "Convenience function should apply LoRA"
    print(f"   Applied via convenience function, diff={diff_conv:.6f}")

    remove_lora_additive(mock_model)
    assert not mgr.is_active()
    print("   Removed via convenience function: OK")

    # 8. Test with strategy argument (v3 compat)
    print("\n8. v3 strategy argument:")
    strategy = torch.randn(batch_size, 512)
    mgr2 = apply_lora_additive(mock_model, mock_lora, state, strategy=strategy)
    assert mgr2.is_active()
    remove_lora_additive(mock_model)
    print("   Strategy argument accepted: OK")

    print("\n" + "=" * 60)
    print("All additive LoRA tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    _test_additive_lora()
