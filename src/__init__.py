# Mycelium v25.4 - Growing Notebook Architecture
#
# Components:
#   Compressor (compressor_v3) - 7-layer perceiver
#   HaarWaveletPreprocess      - wavelet input compression
#   per_page_contrastive_loss  - notebook diversity loss
#
# The model lives in scripts/atom_lora.py (AtomLoRAModel).

from .compressor_v3 import Compressor
from .haar_wavelet import HaarWaveletPreprocess, LayerWiseWavelet
from .contrastive_page_loss import per_page_contrastive_loss

__all__ = [
    "Compressor",
    "HaarWaveletPreprocess",
    "LayerWiseWavelet",
    "per_page_contrastive_loss",
]
