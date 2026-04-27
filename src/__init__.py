# Mycelium v24 - 64-Atom LoRA Architecture
#
# Components:
#   Compressor (compressor_v3) - 7-layer perceiver with skip connection
#   AdditiveLoRAManager        - Fast inline LoRA via monkey-patched forwards
#   HaarWaveletPreprocess      - v24.3 wavelet input compression
#   PageCache/ReplayBuffer     - v24.4 cache graduated cycles for faster training
#
# The v24 model lives in scripts/atom_lora.py (AtomLoRAModel).

from .compressor_v3 import Compressor
from .additive_lora import AdditiveLoRAManager, apply_lora_additive, remove_lora_additive
from .haar_wavelet import HaarWaveletPreprocess, LayerWiseWavelet
from .page_cache import (
    PageCache,
    ReplayBuffer,
    GraduationTracker,
    measure_per_step_accuracy,
    train_step_with_cache,
    populate_cache,
)
from .entropy_flow import (
    EntropyTracker,
    SurpriseDetector,
    EntropyFlowConfidence,
    compute_smoothness_target,
)

__all__ = [
    "Compressor",
    "AdditiveLoRAManager",
    "apply_lora_additive",
    "remove_lora_additive",
    "HaarWaveletPreprocess",
    "LayerWiseWavelet",
    "PageCache",
    "ReplayBuffer",
    "GraduationTracker",
    "measure_per_step_accuracy",
    "train_step_with_cache",
    "populate_cache",
    "EntropyTracker",
    "SurpriseDetector",
    "EntropyFlowConfidence",
    "compute_smoothness_target",
]
