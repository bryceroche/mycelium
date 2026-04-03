# Mycelium v19 - Symmetric Hourglass Architecture
#
# Components:
#   Decompressor    - 7-layer perceiver expanding state → bias (105M params)
#   Compressor      - 7-layer perceiver compressing all layers → state (105M params)
#   ConfidenceHead  - Decides when to stop thinking (~2K params)
#   ThinkingModel   - Full thinking loop (DECOMPRESSOR → Llama → COMPRESSOR)
#
# Legacy (v18):
#   AllLayerPerceiver - Previous compressor implementation
#   StateInjector     - Previous state → pseudo-tokens (replaced by Decompressor)

from .decompressor import Decompressor
from .compressor import Compressor
from .confidence_head import ConfidenceHead
from .thinking_model import ThinkingModel

# Legacy exports for backward compatibility
from .all_layer_perceiver import AllLayerPerceiver
from .state_injector import StateInjector

__all__ = [
    # v19 Symmetric Hourglass
    "Decompressor",
    "Compressor",
    "ConfidenceHead",
    "ThinkingModel",
    # Legacy (v18)
    "AllLayerPerceiver",
    "StateInjector",
]
