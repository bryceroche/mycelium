"""
StateConditionedLoRA: Hypernetwork that rewires transformer attention via state-scaled LoRA templates.

The state vector (64 floats) generates scaling factors for learned LoRA A/B templates.
Templates are shared across all thinking passes - they form a learned vocabulary of
attention modification styles. The hypernetwork maps state to 256 scales
(16 layers x 4 projections x 4 rank), bounded to [-1, 1] via Tanh.

Architecture:
- A_templates: 4 projections (Q,K,V,O) x 16 layers x 2048 x rank = 524K params
- B_templates: 4 projections (Q,K,V,O) x 16 layers x rank x 2048 = 524K params
- Hypernetwork: Linear(64 -> 256) + Tanh = 16K params
- Total: ~1.1M params

The templates define KINDS of attention modification for mathematical reasoning.
The state-derived scales select the MIX for each thinking pass.
Different state = different attention = different thinking style.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class StateConditionedLoRA(nn.Module):
    """
    Hypernetwork that generates state-conditioned LoRA modifications.

    Takes a 64-float state vector and produces scaling factors for learned
    LoRA A/B templates. Each rank dimension represents a "type" of attention
    modification. The state controls the mix of these types at each layer.

    Args:
        d_model: Hidden dimension of the transformer (2048 for Llama 3.2 1B)
        state_size: Size of the input state vector (64)
        rank: LoRA rank - number of attention modification styles per projection (4)
        num_layers: Number of transformer layers to modify (16 for Llama 3.2 1B)
        num_projections: Number of attention projections per layer (4: Q, K, V, O)
    """

    def __init__(
        self,
        d_model: int = 2048,
        d_kv: int = 512,  # For GQA: num_kv_heads * head_dim (8 * 64 for Llama 3.2 1B)
        state_size: int = 64,
        rank: int = 4,
        num_layers: int = 16,
        num_projections: int = 4,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_kv = d_kv  # K, V projections have smaller output due to GQA
        self.state_size = state_size
        self.rank = rank
        self.num_layers = num_layers
        self.num_projections = num_projections

        # Learned LoRA templates (shared across all thinking passes)
        # These learn "useful ways to modify attention for mathematical reasoning"
        #
        # GQA (Grouped Query Attention) in Llama 3.2 1B:
        # - Q, O projections: d_model -> d_model (2048 -> 2048)
        # - K, V projections: d_model -> d_kv (2048 -> 512 for 8 KV heads)
        #
        # proj_dims[i] = output dimension for projection i (Q, K, V, O)
        # GQA (Grouped Query Attention) in Llama 3.2 1B:
        # - Q, O projections: d_model -> d_model (2048 -> 2048)
        # - K, V projections: d_model -> d_kv (2048 -> 512 for 8 KV heads)
        self.proj_dims = [d_model, d_kv, d_kv, d_model]  # Q=2048, K=512, V=512, O=2048

        # A templates: (num_layers, d_model, rank) - down-projection from input
        # B templates: (num_layers, rank, proj_dim) - up-projection to output
        self.A_templates = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, d_model, rank) * 0.01)
            for _ in range(num_projections)
        ])
        self.B_templates = nn.ParameterList([
            nn.Parameter(torch.randn(num_layers, rank, self.proj_dims[i]) * 0.01)
            for i in range(num_projections)
        ])

        # Hypernetwork: state -> scaling factors
        # Maps 64 floats to (16 layers x 4 projections x 4 rank) = 256 scales
        # Tanh bounds scales to [-1, 1] for stable LoRA modification
        num_scales = num_layers * num_projections * rank
        self.state_to_scales = nn.Sequential(
            nn.Linear(state_size, num_scales),
            nn.Tanh(),  # scales bounded to [-1, 1]
        )

        # Projection names for output dictionary
        self.proj_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    def forward(self, state: torch.Tensor) -> Dict[int, Dict[str, Dict[str, torch.Tensor]]]:
        """
        Generate state-conditioned LoRA modifications for all layers.

        Args:
            state: (batch, 64) state vector on hypersphere

        Returns:
            Dictionary mapping layer_idx -> projection_name -> {'A', 'B', 'scales'}

            lora_mods[layer_idx][proj_name] = {
                'A': (d_model, rank) - LoRA down-projection template
                'B': (rank, d_model) - LoRA up-projection template
                'scales': (batch, rank) - state-dependent scaling factors
            }

            To apply LoRA modification:
                delta_W = A @ diag(scales) @ B
                output = (W + delta_W) @ input
        """
        batch_size = state.size(0)

        # Generate all scales from state
        # (batch, 256) where 256 = 16 layers x 4 projections x 4 rank
        all_scales = self.state_to_scales(state)

        # Reshape to (batch, num_layers, num_projections, rank)
        all_scales = all_scales.reshape(
            batch_size,
            self.num_layers,
            self.num_projections,
            self.rank
        )

        # Build output dictionary
        lora_mods: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}

        for layer_idx in range(self.num_layers):
            lora_mods[layer_idx] = {}

            for proj_idx, proj_name in enumerate(self.proj_names):
                # Extract scales for this layer and projection
                # (batch, rank)
                scales = all_scales[:, layer_idx, proj_idx, :]

                # Get the learned templates for this projection and layer
                # A: (d_model, rank), B: (rank, d_model)
                A = self.A_templates[proj_idx][layer_idx]
                B = self.B_templates[proj_idx][layer_idx]

                # Store for use in forward hooks
                # Each rank dimension is a "type" of attention modification
                # The scales control the mix of these types
                lora_mods[layer_idx][proj_name] = {
                    'A': A,           # (d_model, rank) = (2048, 4)
                    'B': B,           # (rank, proj_dim) - varies by projection due to GQA
                    'scales': scales, # (batch, rank) = (batch, 4)
                }

        return lora_mods

    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters by component.

        Returns:
            Dictionary with parameter counts:
            - a_templates: A template parameters (4 x 16 x 2048 x 4 = 524,288)
            - b_templates: B template parameters (4 x 16 x 4 x 2048 = 524,288)
            - hypernetwork: Linear(64 -> 256) + bias (64*256 + 256 = 16,640)
            - total: Sum of all parameters (~1.1M)
        """
        a_params = sum(p.numel() for p in self.A_templates)
        b_params = sum(p.numel() for p in self.B_templates)
        hyper_params = sum(p.numel() for p in self.state_to_scales.parameters())

        return {
            'a_templates': a_params,
            'b_templates': b_params,
            'hypernetwork': hyper_params,
            'total': a_params + b_params + hyper_params,
        }

    def get_scale_statistics(self, state: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistics on the generated scales for monitoring.

        Args:
            state: (batch, 64) state vector

        Returns:
            Dictionary with scale statistics for diagnostics
        """
        with torch.no_grad():
            all_scales = self.state_to_scales(state)

            return {
                'mean': all_scales.mean().item(),
                'std': all_scales.std().item(),
                'min': all_scales.min().item(),
                'max': all_scales.max().item(),
                'abs_mean': all_scales.abs().mean().item(),
            }


def test_state_conditioned_lora():
    """
    Test function to verify shapes and parameter count.

    Validates:
    1. Output shapes match expected dimensions
    2. Parameter count is approximately 1.1M
    3. Scales are bounded to [-1, 1] via Tanh
    4. Forward pass works with batched input
    """
    print("Testing StateConditionedLoRA...")
    print("=" * 60)

    # Create module with default parameters (matching Llama 3.2 1B)
    lora = StateConditionedLoRA(
        d_model=2048,
        d_kv=512,  # 8 KV heads * 64 head_dim for GQA
        state_size=64,
        rank=4,
        num_layers=16,
        num_projections=4,
    )

    # Test parameter count
    param_counts = lora.count_parameters()
    print("\nParameter counts:")
    print(f"  A templates:  {param_counts['a_templates']:,} (expected: 524,288)")
    print(f"  B templates:  {param_counts['b_templates']:,} (expected: 524,288)")
    print(f"  Hypernetwork: {param_counts['hypernetwork']:,} (expected: 16,640)")
    print(f"  Total:        {param_counts['total']:,} (expected: ~1,065,216)")

    assert param_counts['a_templates'] == 4 * 16 * 2048 * 4, "A template params mismatch"
    assert param_counts['b_templates'] == 4 * 16 * 4 * 2048, "B template params mismatch"
    assert param_counts['hypernetwork'] == 64 * 256 + 256, "Hypernetwork params mismatch"
    print("  [PASS] Parameter counts match expected values")

    # Test forward pass with batch
    batch_size = 4
    state = torch.randn(batch_size, 64)
    lora_mods = lora(state)

    print(f"\nForward pass with batch_size={batch_size}:")
    print(f"  Number of layers in output: {len(lora_mods)}")

    # Check structure
    assert len(lora_mods) == 16, f"Expected 16 layers, got {len(lora_mods)}"

    for layer_idx in range(16):
        assert layer_idx in lora_mods, f"Missing layer {layer_idx}"
        layer_mods = lora_mods[layer_idx]

        # Expected B shapes: Q=2048, K=512, V=512, O=2048 (GQA-aware)
        expected_b_dims = {'q_proj': 2048, 'k_proj': 512, 'v_proj': 512, 'o_proj': 2048}

        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            assert proj_name in layer_mods, f"Missing {proj_name} in layer {layer_idx}"

            A = layer_mods[proj_name]['A']
            B = layer_mods[proj_name]['B']
            scales = layer_mods[proj_name]['scales']

            expected_b_dim = expected_b_dims[proj_name]
            assert A.shape == (2048, 4), f"A shape mismatch: {A.shape}"
            assert B.shape == (4, expected_b_dim), f"B shape mismatch for {proj_name}: {B.shape}, expected (4, {expected_b_dim})"
            assert scales.shape == (batch_size, 4), f"scales shape mismatch: {scales.shape}"

    print("  [PASS] Output structure is correct")
    print("  [PASS] A shapes: (2048, 4) for all layers/projections")
    print("  [PASS] B shapes: Q=(4,2048), K=(4,512), V=(4,512), O=(4,2048) - GQA-aware")
    print(f"  [PASS] Scales shapes: ({batch_size}, 4) for all layers/projections")

    # Verify scales are bounded by Tanh
    all_scales = lora.state_to_scales(state)
    assert all_scales.min() >= -1.0, f"Scales below -1: {all_scales.min()}"
    assert all_scales.max() <= 1.0, f"Scales above 1: {all_scales.max()}"
    print("  [PASS] Scales bounded to [-1, 1] via Tanh")

    # Test scale statistics
    stats = lora.get_scale_statistics(state)
    print(f"\nScale statistics:")
    print(f"  Mean:     {stats['mean']:.4f}")
    print(f"  Std:      {stats['std']:.4f}")
    print(f"  Min:      {stats['min']:.4f}")
    print(f"  Max:      {stats['max']:.4f}")
    print(f"  Abs mean: {stats['abs_mean']:.4f}")

    # Test that different states produce different scales
    state2 = torch.randn(batch_size, 64)
    lora_mods2 = lora(state2)

    scales1 = lora_mods[0]['q_proj']['scales']
    scales2 = lora_mods2[0]['q_proj']['scales']

    # Scales should be different for different states
    assert not torch.allclose(scales1, scales2, atol=1e-6), "Different states produced same scales"
    print("\n  [PASS] Different states produce different scales")

    # Test gradient flow
    state_grad = torch.randn(batch_size, 64, requires_grad=True)
    lora_mods_grad = lora(state_grad)

    # Sum all scales and backprop
    total = sum(
        lora_mods_grad[layer_idx][proj_name]['scales'].sum()
        for layer_idx in range(16)
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    )
    total.backward()

    assert state_grad.grad is not None, "No gradient on state"
    assert state_grad.grad.shape == (batch_size, 64), "Gradient shape mismatch"
    print("  [PASS] Gradients flow back to state vector")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

    return lora


if __name__ == "__main__":
    test_state_conditioned_lora()
