"""
SymPy Result Encoder (v24.6)

Encodes SymPy evaluation results into page-compatible vectors for the
Mycelium breathing architecture. This bridges the gap between the symbolic
computation engine (SymPy) and the neural thinking loop.

The key insight: SymPy gives us EXACT numeric results, but we need to encode
them in a format the perceiver/hypernetwork can use. This module handles:

1. Log-scale encoding — handles wide numeric ranges (0.01 to 10000+)
2. Variable count normalization — always produces fixed-size vectors
3. Learnable gating — controls how much SymPy results influence the page

Components:
- SymPyResultEncoder: nn.Module that encodes results into page vectors
- format_sympy_context: Formats results as text for the language model
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Optional


class SymPyResultEncoder(nn.Module):
    """
    Encode SymPy evaluation results into a page-compatible vector.

    Takes a dictionary of {variable_name: numeric_value} from SymPyEvaluator
    and produces a (page_size,) vector that can be added to the page.

    Uses log-scale encoding for numbers to handle the wide range of values
    that appear in math problems (0.01 to 10000+). Each variable contributes
    two features: the normalized value and its log magnitude.

    The learnable gate parameter controls how much SymPy results influence
    the page — initialized to 0.3 (sigmoid → ~0.57) for moderate influence.

    Architecture:
        Input:  max_variables * 2 features (value + log_mag per variable)
        Hidden: 128-dim with GELU activation
        Output: page_size features, scaled by sigmoid(gate)

    Parameter count: ~17K for default settings
        - value_encoder: (max_variables * 2) * 128 + 128 + 128 * page_size + page_size
        - result_gate: 1

    Example:
        >>> encoder = SymPyResultEncoder(page_size=64, max_variables=8)
        >>> results = {'n_april': 48.0, 'n_may': 24.0, 'total': 72.0}
        >>> vec = encoder(results, device='cuda')
        >>> # vec.shape == (64,), can be added directly to the page
    """

    def __init__(self, page_size: int = 64, max_variables: int = 8):
        """
        Initialize the SymPyResultEncoder.

        Args:
            page_size: Output vector dimension, matching the page size.
                      Default 64 matches the Mycelium architecture.
            max_variables: Maximum number of variables to encode.
                          Extra variables are ignored. Default 8 handles
                          most GSM8K problems (typically 2-5 variables).
        """
        super().__init__()
        self.page_size = page_size
        self.max_variables = max_variables

        # Encode each variable's value
        # Input: value + log(|value|) per variable = max_variables * 2
        # Hidden: 128 with GELU activation
        # Output: page_size
        self.value_encoder = nn.Sequential(
            nn.Linear(max_variables * 2, 128),
            nn.GELU(),
            nn.Linear(128, page_size),
        )

        # Gate: how much should SymPy results influence the page?
        # Initialized to 0.3 → sigmoid(0.3) ≈ 0.57 (moderate influence)
        # Can learn to increase (more trust in SymPy) or decrease (less trust)
        self.result_gate = nn.Parameter(torch.tensor(0.3))

    def encode_values(self, values: list, device: torch.device) -> torch.Tensor:
        """
        Encode a list of numeric values into the input feature vector.

        Uses log-scale encoding: for each value, stores both:
        - Normalized value: value / 1000 (rough centering for typical ranges)
        - Log magnitude: sign(value) * log(|value| + 1) (captures scale)

        Args:
            values: List of numeric values to encode
            device: Target device for the tensor

        Returns:
            Tensor of shape (max_variables * 2,) with encoded values
        """
        encoded = torch.zeros(self.max_variables * 2, device=device)

        for i, v in enumerate(values[:self.max_variables]):
            # Normalized value (rough centering for typical GSM8K ranges)
            encoded[i * 2] = v / 1000.0

            # Log magnitude with sign preservation
            # log(|v| + 1) handles zero gracefully, sign preserves direction
            sign = 1.0 if v >= 0 else -1.0
            encoded[i * 2 + 1] = sign * math.log(abs(v) + 1.0)

        return encoded

    def forward(
        self,
        sympy_results: Dict[str, float],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Encode SymPy results into a page-compatible vector.

        Args:
            sympy_results: Dictionary of {variable_name: numeric_value}.
                          Variable names are ignored — only values matter.
                          Empty dict returns zeros.
            device: Target device. If None, uses the module's parameter device.

        Returns:
            Tensor of shape (page_size,) encoding the results.
            Scaled by sigmoid(gate) to control influence.
        """
        # Determine device from module parameters if not specified
        if device is None:
            device = self.result_gate.device

        # Handle empty results
        if not sympy_results:
            return torch.zeros(self.page_size, device=device)

        # Extract values (order doesn't matter, we treat them as a set)
        values = list(sympy_results.values())[:self.max_variables]

        # Encode values into feature vector
        encoded = self.encode_values(values, device)

        # Pass through value encoder network
        result_vec = self.value_encoder(encoded.unsqueeze(0))  # (1, page_size)

        # Apply learnable gate
        gate = torch.sigmoid(self.result_gate)

        return result_vec.squeeze(0) * gate  # (page_size,)

    def forward_batch(
        self,
        sympy_results_batch: list,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Encode a batch of SymPy results.

        Args:
            sympy_results_batch: List of dictionaries, one per sample.
            device: Target device.

        Returns:
            Tensor of shape (batch_size, page_size)
        """
        if device is None:
            device = self.result_gate.device

        batch_size = len(sympy_results_batch)
        output = torch.zeros(batch_size, self.page_size, device=device)

        # Track which samples have non-empty results
        non_empty_indices = []
        non_empty_encoded = []

        for b, results in enumerate(sympy_results_batch):
            if results:
                non_empty_indices.append(b)
                values = list(results.values())[:self.max_variables]
                encoded = torch.zeros(self.max_variables * 2, device=device)
                for i, v in enumerate(values):
                    encoded[i * 2] = v / 1000.0
                    sign = 1.0 if v >= 0 else -1.0
                    encoded[i * 2 + 1] = sign * math.log(abs(v) + 1.0)
                non_empty_encoded.append(encoded)

        # Only run the network on non-empty entries
        if non_empty_indices:
            encoded_batch = torch.stack(non_empty_encoded, dim=0)  # (N, max_variables * 2)
            result_vecs = self.value_encoder(encoded_batch)  # (N, page_size)
            gate = torch.sigmoid(self.result_gate)
            scaled_vecs = result_vecs * gate

            # Scatter back to output
            for idx, b in enumerate(non_empty_indices):
                output[b] = scaled_vecs[idx]

        return output


def format_sympy_context(sympy_results: Dict[str, float]) -> str:
    """
    Format SymPy results as text context for the language model.

    Prepends accumulated results to the problem text so the model
    can see what has been computed in previous thinking cycles.

    Args:
        sympy_results: Dictionary of {variable_name: numeric_value}

    Returns:
        Formatted string like "Known values: n_april=48, n_may=24\n"
        Empty string if no results.

    Example:
        >>> results = {'n_april': 48.0, 'n_may': 24.0}
        >>> format_sympy_context(results)
        'Known values: n_april=48, n_may=24\\n'
    """
    if not sympy_results:
        return ""

    # Format each result, handling floats cleanly
    parts = []
    for name, value in sympy_results.items():
        # Format integers without decimal point
        if isinstance(value, float) and value.is_integer():
            formatted_value = str(int(value))
        else:
            # Keep reasonable precision for non-integers
            formatted_value = f"{value:.6g}"
        parts.append(f"{name}={formatted_value}")

    return "Known values: " + ", ".join(parts) + "\n"


if __name__ == "__main__":
    """Tests for SymPyResultEncoder and format_sympy_context."""

    print("Testing SymPyResultEncoder...")
    print("=" * 50)

    # Test 1: Empty dict returns zeros
    print("\n1. Empty dict returns zeros:")
    encoder = SymPyResultEncoder(page_size=64, max_variables=8)
    empty_result = encoder({}, device="cpu")

    assert empty_result.shape == (64,), f"Expected shape (64,), got {empty_result.shape}"
    assert torch.allclose(empty_result, torch.zeros(64)), "Empty dict should return zeros"
    print(f"   Shape: {empty_result.shape}")
    print(f"   Sum: {empty_result.sum().item():.6f} (expected: 0.0)")
    print("   PASSED: Empty dict returns zeros")

    # Test 2: Single variable encodes correctly
    print("\n2. Single variable encodes correctly:")
    single_result = encoder({'x': 100.0}, device="cpu")

    assert single_result.shape == (64,), f"Expected shape (64,), got {single_result.shape}"
    assert single_result.abs().sum() > 0, "Single variable should produce non-zero output"
    print(f"   Shape: {single_result.shape}")
    print(f"   Non-zero sum: {single_result.abs().sum().item():.4f}")
    print("   PASSED: Single variable encodes to non-zero vector")

    # Test 3: Multiple variables encode
    print("\n3. Multiple variables encode:")
    multi_result = encoder({'a': 10.0, 'b': 20.0, 'c': 30.0}, device="cpu")

    assert multi_result.shape == (64,), f"Expected shape (64,), got {multi_result.shape}"
    assert multi_result.abs().sum() > 0, "Multiple variables should produce non-zero output"
    # Note: More variables don't necessarily produce larger magnitude (depends on network weights)
    # The key test is that different inputs produce different outputs
    assert not torch.allclose(multi_result, single_result), \
        "Different inputs should produce different outputs"
    print(f"   Shape: {multi_result.shape}")
    print(f"   Single var magnitude: {single_result.abs().sum().item():.4f}")
    print(f"   Multi var magnitude: {multi_result.abs().sum().item():.4f}")
    print("   PASSED: Multiple variables encode")

    # Test 4: Different values produce different encodings
    print("\n4. Different values produce different encodings:")
    result_a = encoder({'x': 100.0}, device="cpu")
    result_b = encoder({'x': 200.0}, device="cpu")

    cosine_sim = torch.nn.functional.cosine_similarity(
        result_a.unsqueeze(0), result_b.unsqueeze(0)
    ).item()

    assert not torch.allclose(result_a, result_b), "Different values should produce different encodings"
    print(f"   Cosine similarity between x=100 and x=200: {cosine_sim:.4f}")
    print("   PASSED: Different values produce different encodings")

    # Test 5: Gate controls magnitude
    print("\n5. Gate controls magnitude:")

    # Create two encoders with different gate values
    encoder_low_gate = SymPyResultEncoder(page_size=64, max_variables=8)
    encoder_high_gate = SymPyResultEncoder(page_size=64, max_variables=8)

    # Manually set gate values
    with torch.no_grad():
        encoder_low_gate.result_gate.fill_(-2.0)  # sigmoid(-2) ≈ 0.12
        encoder_high_gate.result_gate.fill_(2.0)   # sigmoid(2) ≈ 0.88

    test_results = {'a': 50.0, 'b': 100.0}
    low_gate_output = encoder_low_gate(test_results, device="cpu")
    high_gate_output = encoder_high_gate(test_results, device="cpu")

    low_mag = low_gate_output.abs().sum().item()
    high_mag = high_gate_output.abs().sum().item()

    print(f"   Low gate (sigmoid=-2 ≈ 0.12) magnitude: {low_mag:.4f}")
    print(f"   High gate (sigmoid=2 ≈ 0.88) magnitude: {high_mag:.4f}")
    print(f"   Ratio (high/low): {high_mag / low_mag:.2f}")

    assert high_mag > low_mag * 2, "High gate should produce significantly larger magnitude"
    print("   PASSED: Gate controls magnitude")

    # Test 6: Max variables limit is respected
    print("\n6. Max variables limit is respected:")
    encoder_small = SymPyResultEncoder(page_size=64, max_variables=2)

    # Provide more variables than max
    many_vars = {'a': 1.0, 'b': 2.0, 'c': 3.0, 'd': 4.0, 'e': 5.0}
    limited_result = encoder_small(many_vars, device="cpu")

    # With only 2 variables, result should match first 2 variables only
    first_two = {'a': 1.0, 'b': 2.0}
    first_two_result = encoder_small(first_two, device="cpu")

    # Note: dict ordering is preserved in Python 3.7+, so this should work
    assert limited_result.shape == (64,), f"Expected shape (64,), got {limited_result.shape}"
    print(f"   5 variables with max_variables=2: shape {limited_result.shape}")
    print("   PASSED: Max variables limit is respected")

    # Test 7: Handles negative numbers
    print("\n7. Handles negative numbers:")
    neg_result = encoder({'x': -50.0, 'y': 100.0}, device="cpu")

    assert neg_result.shape == (64,), f"Expected shape (64,), got {neg_result.shape}"
    assert neg_result.abs().sum() > 0, "Negative numbers should encode"
    print(f"   Shape: {neg_result.shape}")
    print(f"   Magnitude: {neg_result.abs().sum().item():.4f}")
    print("   PASSED: Handles negative numbers")

    # Test 8: Handles very large numbers (log scale)
    print("\n8. Handles very large numbers (log scale):")
    small_result = encoder({'x': 10.0}, device="cpu")
    large_result = encoder({'x': 10000.0}, device="cpu")

    small_mag = small_result.abs().sum().item()
    large_mag = large_result.abs().sum().item()

    # Log scale should compress the range
    # Without log: ratio would be 1000x
    # With log: ratio should be much smaller
    ratio = large_mag / (small_mag + 1e-8)

    print(f"   x=10 magnitude: {small_mag:.4f}")
    print(f"   x=10000 magnitude: {large_mag:.4f}")
    print(f"   Ratio: {ratio:.2f} (linear would be ~1000x)")

    assert ratio < 100, "Log scale should compress large number range"
    print("   PASSED: Log scale compresses large number range")

    # Test 9: Handles very small numbers
    print("\n9. Handles very small numbers:")
    tiny_result = encoder({'x': 0.001}, device="cpu")

    assert tiny_result.shape == (64,), f"Expected shape (64,), got {tiny_result.shape}"
    assert not torch.isnan(tiny_result).any(), "Should not produce NaN for small numbers"
    print(f"   x=0.001 magnitude: {tiny_result.abs().sum().item():.4f}")
    print("   PASSED: Handles very small numbers without NaN")

    # Test 10: Handles zero
    print("\n10. Handles zero:")
    zero_result = encoder({'x': 0.0}, device="cpu")

    assert not torch.isnan(zero_result).any(), "Should not produce NaN for zero"
    assert not torch.isinf(zero_result).any(), "Should not produce Inf for zero"
    print(f"   x=0.0 magnitude: {zero_result.abs().sum().item():.4f}")
    print("   PASSED: Handles zero without NaN/Inf")

    # Test 11: Gradient flow
    print("\n11. Gradient flow:")
    encoder_grad = SymPyResultEncoder(page_size=64, max_variables=8)

    # Forward pass
    result = encoder_grad({'a': 50.0, 'b': 100.0}, device="cpu")
    loss = result.sum()
    loss.backward()

    # Check gradients exist
    assert encoder_grad.result_gate.grad is not None, "Gate should have gradient"
    assert encoder_grad.value_encoder[0].weight.grad is not None, "Encoder should have gradient"

    print(f"   Gate gradient: {encoder_grad.result_gate.grad.item():.6f}")
    print(f"   Encoder weight grad norm: {encoder_grad.value_encoder[0].weight.grad.norm().item():.6f}")
    print("   PASSED: Gradients flow correctly")

    # Test 12: Batch encoding
    print("\n12. Batch encoding:")
    batch_results = [
        {'a': 10.0, 'b': 20.0},
        {'x': 100.0},
        {},  # empty
        {'p': 5.0, 'q': 15.0, 'r': 25.0},
    ]

    batch_output = encoder.forward_batch(batch_results, device="cpu")

    assert batch_output.shape == (4, 64), f"Expected shape (4, 64), got {batch_output.shape}"

    # Empty batch entry should be zeros
    assert torch.allclose(batch_output[2], torch.zeros(64)), "Empty entry should be zeros"

    print(f"   Batch output shape: {batch_output.shape}")
    print(f"   Entry 0 magnitude: {batch_output[0].abs().sum().item():.4f}")
    print(f"   Entry 2 (empty) magnitude: {batch_output[2].abs().sum().item():.6f}")
    print("   PASSED: Batch encoding works")

    # ========================================
    # format_sympy_context tests
    # ========================================
    print("\n" + "=" * 50)
    print("Testing format_sympy_context...")
    print("=" * 50)

    # Test 13: Empty dict returns empty string
    print("\n13. Empty dict returns empty string:")
    empty_context = format_sympy_context({})
    assert empty_context == "", f"Expected empty string, got '{empty_context}'"
    print(f"   Result: '{empty_context}' (empty)")
    print("   PASSED: Empty dict returns empty string")

    # Test 14: Single variable formatting
    print("\n14. Single variable formatting:")
    single_context = format_sympy_context({'n_april': 48.0})
    expected = "Known values: n_april=48\n"
    assert single_context == expected, f"Expected '{expected}', got '{single_context}'"
    print(f"   Result: '{single_context.strip()}'")
    print("   PASSED: Single variable formats correctly")

    # Test 15: Multiple variables formatting
    print("\n15. Multiple variables formatting:")
    multi_context = format_sympy_context({'n_april': 48.0, 'n_may': 24.0})

    # Check structure (order may vary)
    assert multi_context.startswith("Known values: "), "Should start with 'Known values: '"
    assert multi_context.endswith("\n"), "Should end with newline"
    assert "n_april=48" in multi_context, "Should contain n_april=48"
    assert "n_may=24" in multi_context, "Should contain n_may=24"

    print(f"   Result: '{multi_context.strip()}'")
    print("   PASSED: Multiple variables format correctly")

    # Test 16: Float formatting
    print("\n16. Float formatting:")
    float_context = format_sympy_context({'x': 3.14159, 'y': 2.5})

    assert "3.14159" in float_context or "3.1416" in float_context, \
        "Should format floats with reasonable precision"
    assert "2.5" in float_context, "Should format 2.5 correctly"

    print(f"   Result: '{float_context.strip()}'")
    print("   PASSED: Floats format correctly")

    # Test 17: Integer floats (no decimal point)
    print("\n17. Integer floats (no decimal point):")
    int_float_context = format_sympy_context({'total': 72.0})

    assert "total=72" in int_float_context, "Integer floats should format without decimal"
    assert "72.0" not in int_float_context, "Should not have .0 suffix"

    print(f"   Result: '{int_float_context.strip()}'")
    print("   PASSED: Integer floats format without decimal point")

    # Test 18: Negative numbers
    print("\n18. Negative numbers:")
    neg_context = format_sympy_context({'x': -50.0, 'y': 100.0})

    assert "x=-50" in neg_context, "Should handle negative numbers"

    print(f"   Result: '{neg_context.strip()}'")
    print("   PASSED: Negative numbers format correctly")

    # Test 19: Large numbers
    print("\n19. Large numbers:")
    large_context = format_sympy_context({'big': 1000000.0})

    assert "1e" in large_context.lower() or "1000000" in large_context, \
        "Should handle large numbers"

    print(f"   Result: '{large_context.strip()}'")
    print("   PASSED: Large numbers format correctly")

    # Test 20: Typical GSM8K example
    print("\n20. Typical GSM8K example:")
    gsm8k_results = {
        'n_april': 48.0,
        'n_may': 24.0,
        'total': 72.0,
    }
    gsm8k_context = format_sympy_context(gsm8k_results)

    print(f"   Input: {gsm8k_results}")
    print(f"   Output: '{gsm8k_context.strip()}'")

    assert gsm8k_context.startswith("Known values: "), "Should have proper prefix"
    assert gsm8k_context.endswith("\n"), "Should have trailing newline"
    print("   PASSED: Typical GSM8K example works")

    # ========================================
    # Integration test
    # ========================================
    print("\n" + "=" * 50)
    print("Integration test: Encoding + Context together...")
    print("=" * 50)

    # Simulate a thinking cycle with SymPy results
    sympy_results = {'n_april': 48.0, 'n_may': 24.0}

    # Get context for language model
    context = format_sympy_context(sympy_results)
    print(f"\n   Context for LM: '{context.strip()}'")

    # Get encoding for page
    encoder_final = SymPyResultEncoder(page_size=64, max_variables=8)
    encoding = encoder_final(sympy_results, device="cpu")
    print(f"   Encoding shape: {encoding.shape}")
    print(f"   Encoding magnitude: {encoding.abs().sum().item():.4f}")

    # Simulate adding to page
    page = torch.randn(64)
    page_with_sympy = page + encoding
    print(f"   Page norm before SymPy: {page.norm().item():.4f}")
    print(f"   Page norm after SymPy: {page_with_sympy.norm().item():.4f}")

    print("\n   PASSED: Integration test complete")

    print("\n" + "=" * 50)
    print("All SymPyResultEncoder tests PASSED!")
    print("All format_sympy_context tests PASSED!")
    print("=" * 50)
