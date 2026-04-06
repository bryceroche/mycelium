# Mycelium v20 - State-Conditioned LoRA Architecture
#
# The state vector REWIRES the transformer through state-conditioned LoRA.
# 64 floats generate scaling factors for learned LoRA templates, changing
# HOW the transformer attends at every layer on every thinking pass.
#
# Components:
#   StateConditionedLoRA - Hypernetwork: state -> LoRA scales (~1.1M params)
#   Compressor          - 7-layer perceiver: all layers -> state delta (~105M params)
#   ConfidenceHead      - Decides when to stop thinking (~2K params)
#   ThinkingModel       - Full thinking loop (LoRA -> Llama -> Compress -> Rotate)
#   AdditiveLoRAManager  - Fast inline LoRA via monkey-patched forwards (default)
#   LoRAHookManager     - Legacy hook-based LoRA (slower, kept for compatibility)
#
# Legacy (v19):
#   Decompressor        - Previous state -> bias approach (replaced by LoRA)
#   AllLayerPerceiver   - Previous compressor implementation

from .state_conditioned_lora import StateConditionedLoRA
from .compressor import Compressor
from .confidence_head import ConfidenceHead
from .thinking_model import ThinkingModel
from .lora_hooks import LoRAHookManager, apply_lora, remove_lora
from .additive_lora import AdditiveLoRAManager, apply_lora_additive, remove_lora_additive

# Legacy exports for backward compatibility
try:
    from .decompressor import Decompressor
except ImportError:
    Decompressor = None

try:
    from .all_layer_perceiver import AllLayerPerceiver
except ImportError:
    AllLayerPerceiver = None

try:
    from .state_injector import StateInjector
except ImportError:
    StateInjector = None

__all__ = [
    # v20 State-Conditioned LoRA Architecture
    "StateConditionedLoRA",
    "Compressor",
    "ConfidenceHead",
    "ThinkingModel",
    # LoRA application (state-conditioned attention modification)
    "LoRAHookManager",
    "apply_lora",
    "remove_lora",
    # Additive LoRA (fast, no hooks — default)
    "AdditiveLoRAManager",
    "apply_lora_additive",
    "remove_lora_additive",
    # Legacy (v19)
    "Decompressor",
    "AllLayerPerceiver",
    "StateInjector",
]
