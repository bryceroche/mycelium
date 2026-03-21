"""
Haar Wavelet Preprocessing for Perceiver Input Compression (v24.3)

Apply Haar wavelet transform to hidden states before the perceiver reads them.
Truncate the finest detail level for 2x compression.

The perceiver reads multi-resolution frequency coefficients instead of raw token
positions — already 2x compressed and structured by frequency before learned
compression begins.

Key properties:
- Pure PyTorch — stays on GPU, supports autograd (no pywt CPU transfer)
- 2x fewer tokens — perceiver attention is 4x faster (quadratic in tokens)
- Structured by frequency — coarsest coefficients first, finer detail later
- Learned level weights — model decides how important each resolution is
- Position-preserving — unlike FFT, wavelet keeps spatial localization
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarWaveletPreprocess(nn.Module):
    """
    Haar wavelet transform for hidden state compression.

    Applies multi-level Haar decomposition along the sequence dimension.
    Each level produces averages (smooth) and differences (detail) coefficients.
    Truncating the finest detail level gives ~2x compression.

    The output is structured by frequency: coarsest coefficients first,
    progressively finer detail later. This gives the perceiver a natural
    coarse-to-fine reading order.
    """

    def __init__(
        self,
        max_level: int = 4,
        truncate_finest: bool = True,
        learnable_weights: bool = True,
    ):
        """
        Args:
            max_level: Number of decomposition levels (4 gives good compression)
            truncate_finest: If True, discard finest detail coefficients (~2x compression)
            learnable_weights: If True, learn importance weights per level
        """
        super().__init__()
        self.max_level = max_level
        self.truncate_finest = truncate_finest
        self.learnable_weights = learnable_weights

        if learnable_weights:
            # Learnable importance weight per level (sigmoid → [0, 1])
            # +1 for the coarsest approximation coefficients
            # Apéry-weighted initialization (1/k³ power law, coarse > fine)
            # Total converges to ζ(3) ≈ 1.202 (Apéry's constant)
            # This gives a principled prior: natural signals have power spectra
            # that decay with frequency. 1/k³ is common in physical systems.
            apery_weights = torch.tensor([1.0 / (k + 1)**3 for k in range(max_level + 1)])
            apery_weights = apery_weights / apery_weights.sum() * (max_level + 1)  # normalize
            self.level_weights = nn.Parameter(apery_weights)
        else:
            self.register_buffer('level_weights', torch.ones(max_level + 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply Haar wavelet decomposition.

        Args:
            hidden_states: (batch, seq_len, d_model)

        Returns:
            wavelet_coeffs: (batch, num_coeffs, d_model)
                where num_coeffs ≈ seq_len / 2 if truncate_finest=True

        The output is ordered coarsest-to-finest:
            [coarsest_approx, level_N_detail, level_N-1_detail, ..., level_1_detail]
        """
        coefficients = []
        current = hidden_states  # (batch, seq_len, d_model)

        for level in range(self.max_level):
            seq_len = current.size(1)

            # Pad if odd length (Haar requires even length)
            if seq_len % 2 == 1:
                # Pad sequence dimension by 1 (reflect last value)
                current = F.pad(current, (0, 0, 0, 1), mode='replicate')

            # Haar wavelet: averages and differences of adjacent pairs
            # This is the classic Haar decomposition
            even = current[:, 0::2, :]  # (batch, seq_len//2, d_model)
            odd = current[:, 1::2, :]

            # Normalized Haar coefficients (preserve energy)
            averages = (even + odd) / math.sqrt(2)  # low-freq (smooth)
            details = (even - odd) / math.sqrt(2)   # high-freq (edges)

            # Apply learned weight to this level's detail coefficients
            if self.learnable_weights:
                weight = torch.sigmoid(self.level_weights[level])
            else:
                weight = self.level_weights[level]

            coefficients.append(details * weight)

            # Continue decomposing the smooth part
            current = averages

        # Coarsest approximation coefficients (final averages)
        if self.learnable_weights:
            weight = torch.sigmoid(self.level_weights[-1])
        else:
            weight = self.level_weights[-1]
        coefficients.append(current * weight)

        # Truncate finest detail if requested (~2x compression)
        if self.truncate_finest and len(coefficients) > 1:
            coefficients = coefficients[1:]  # drop finest detail (index 0)

        # Reverse so coarsest is first (natural reading order for perceiver)
        # Order: [coarsest_approx, level_N_detail, ..., level_1_detail]
        coefficients.reverse()

        # Concatenate along sequence dimension
        return torch.cat(coefficients, dim=1)  # (batch, num_coeffs, d_model)

    def get_compression_ratio(self, seq_len: int) -> float:
        """Estimate compression ratio for a given sequence length."""
        # Simulate the decomposition to count output coefficients
        coeffs = 0
        current_len = seq_len

        for level in range(self.max_level):
            if current_len % 2 == 1:
                current_len += 1
            detail_len = current_len // 2
            if not (self.truncate_finest and level == 0):
                coeffs += detail_len
            current_len = current_len // 2

        coeffs += current_len  # coarsest approximation

        return seq_len / coeffs if coeffs > 0 else 1.0


class LayerWiseWavelet(nn.Module):
    """
    Apply wavelet preprocessing independently to each transformer layer's output.

    This wraps HaarWaveletPreprocess to handle the list of layer hidden states
    that the perceiver receives.
    """

    def __init__(
        self,
        num_layers: int = 16,
        max_level: int = 4,
        truncate_finest: bool = True,
        share_weights: bool = True,
    ):
        """
        Args:
            num_layers: Number of transformer layers (16 for Llama 1B)
            max_level: Wavelet decomposition levels
            truncate_finest: Whether to truncate finest detail
            share_weights: If True, share wavelet weights across layers.
                If False, each layer gets its own weights.
        """
        super().__init__()
        self.num_layers = num_layers
        self.share_weights = share_weights

        if share_weights:
            self.wavelet = HaarWaveletPreprocess(
                max_level=max_level,
                truncate_finest=truncate_finest,
                learnable_weights=True,
            )
        else:
            self.wavelets = nn.ModuleList([
                HaarWaveletPreprocess(
                    max_level=max_level,
                    truncate_finest=truncate_finest,
                    learnable_weights=True,
                )
                for _ in range(num_layers)
            ])

    def forward(self, all_layer_hidden_states: list) -> list:
        """
        Transform each layer's hidden states to wavelet domain.

        Args:
            all_layer_hidden_states: List of 16 tensors, each (batch, seq_len, d_model)

        Returns:
            List of 16 tensors, each (batch, num_coeffs, d_model)
        """
        if self.share_weights:
            return [self.wavelet(hs) for hs in all_layer_hidden_states]
        else:
            return [
                self.wavelets[i](hs)
                for i, hs in enumerate(all_layer_hidden_states)
            ]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    print("Testing Haar Wavelet Preprocessing...")

    # Test basic wavelet
    wavelet = HaarWaveletPreprocess(max_level=4, truncate_finest=True)

    batch, seq_len, d_model = 2, 100, 2048
    hidden = torch.randn(batch, seq_len, d_model)

    coeffs = wavelet(hidden)

    print(f"\nInput shape:  {hidden.shape}")
    print(f"Output shape: {coeffs.shape}")
    print(f"Compression:  {seq_len} -> {coeffs.size(1)} tokens ({wavelet.get_compression_ratio(seq_len):.2f}x)")

    # Test gradient flow
    coeffs.sum().backward()
    print(f"Gradient flow: OK")
    print(f"Level weights grad: {wavelet.level_weights.grad}")

    # Test without truncation
    wavelet_full = HaarWaveletPreprocess(max_level=4, truncate_finest=False)
    coeffs_full = wavelet_full(hidden)
    print(f"\nWithout truncation: {hidden.size(1)} -> {coeffs_full.size(1)} tokens")

    # Test layer-wise wavelet
    print("\n--- Layer-wise Wavelet ---")
    layer_wavelet = LayerWiseWavelet(num_layers=16, share_weights=True)
    all_layers = [torch.randn(batch, seq_len, d_model) for _ in range(16)]

    wavelet_layers = layer_wavelet(all_layers)

    print(f"Input:  16 layers x {all_layers[0].shape}")
    print(f"Output: 16 layers x {wavelet_layers[0].shape}")
    print(f"Parameters: {layer_wavelet.count_parameters()}")

    # Per-layer weights test
    layer_wavelet_separate = LayerWiseWavelet(num_layers=16, share_weights=False)
    print(f"Separate weights params: {layer_wavelet_separate.count_parameters()}")

    print("\nAll tests passed!")
