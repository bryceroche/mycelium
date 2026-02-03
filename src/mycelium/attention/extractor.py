"""Extract attention matrices from transformer models.

NOTE: This is legacy/experimental code not actively used by the main pipeline.
The active SpanDetector in dual_signal_templates.py uses a different approach.

This module provides utilities to extract raw attention matrices from
transformer models like DeepSeek-Math for research/experimentation.

Requires: transformers, torch, and model weights (GPU recommended).

Why DeepSeek?
- SOTA math accuracy = attention patterns encode actual math structure
- A model that gets wrong answers has misleading attention patterns
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Global cache for lazy-loaded models (singleton pattern)
_model_cache: Dict[str, Tuple["AutoModelForCausalLM", "AutoTokenizer"]] = {}


def _get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model(model_name: str) -> Tuple["AutoModelForCausalLM", "AutoTokenizer"]:
    """Load model and tokenizer with caching (singleton pattern).

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Tuple of (model, tokenizer)
    """
    if model_name in _model_cache:
        logger.debug(f"Using cached model: {model_name}")
        return _model_cache[model_name]

    logger.info(f"Loading model: {model_name}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _get_device()
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load model with attention output enabled
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    # Move to device if not using device_map="auto"
    if device != "cuda":
        model = model.to(device)

    model.eval()

    _model_cache[model_name] = (model, tokenizer)
    logger.info(f"Model loaded successfully: {model_name}")

    return model, tokenizer


@dataclass
class AttentionResult:
    """Result of attention extraction."""
    tokens: List[str]
    # Shape: (num_layers, num_heads, seq_len, seq_len)
    attention: np.ndarray
    # Which layers/heads to use (determined empirically)
    semantic_layers: List[int] = field(default_factory=list)
    semantic_heads: List[int] = field(default_factory=list)


def extract_attention(
    text: str,
    model_name: str = "deepseek-ai/deepseek-math-7b-base",
    layers: Optional[List[int]] = None,
) -> AttentionResult:
    """Extract attention matrices from text.

    Args:
        text: Math problem text
        model_name: HuggingFace model to use
        layers: Which layers to extract (default: last 8 layers, more semantic)

    Returns:
        AttentionResult with tokens and attention matrices
    """
    # Load model (cached singleton)
    model, tokenizer = _load_model(model_name)

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")

    # Move inputs to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass with attention output
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # outputs.attentions is tuple of (batch, num_heads, seq_len, seq_len) per layer
    # Stack into (num_layers, batch, num_heads, seq_len, seq_len)
    all_attentions = torch.stack(outputs.attentions)

    # Remove batch dimension: (num_layers, num_heads, seq_len, seq_len)
    all_attentions = all_attentions.squeeze(1)

    # Determine which layers to use
    num_layers = all_attentions.shape[0]
    if layers is None:
        # Default: last 8 layers (more semantic, less syntactic)
        layers = list(range(max(0, num_layers - 8), num_layers))

    # Extract only requested layers
    selected_attentions = all_attentions[layers]

    # Convert to numpy
    attention_np = selected_attentions.cpu().float().numpy()

    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())

    logger.debug(
        f"Extracted attention: {len(tokens)} tokens, "
        f"{attention_np.shape[0]} layers, {attention_np.shape[1]} heads"
    )

    return AttentionResult(
        tokens=tokens,
        attention=attention_np,
        semantic_layers=layers,
        semantic_heads=list(range(attention_np.shape[1])),  # All heads by default
    )


def aggregate_attention(
    attention: np.ndarray,
    layers: List[int],
    heads: List[int],
    method: str = "mean",
) -> np.ndarray:
    """Aggregate attention across selected layers and heads.

    Args:
        attention: Full attention tensor (layers, heads, seq, seq)
        layers: Which layers to use
        heads: Which heads to use
        method: "mean" or "max"

    Returns:
        Aggregated attention matrix (seq, seq)
    """
    selected = attention[layers][:, heads]  # (L, H, seq, seq)

    if method == "mean":
        return selected.mean(axis=(0, 1))
    elif method == "max":
        return selected.max(axis=(0, 1))
    else:
        raise ValueError(f"Unknown method: {method}")


def clear_model_cache() -> None:
    """Clear the cached models to free memory.

    Useful for testing or when switching between models.
    """
    global _model_cache
    _model_cache.clear()
    logger.info("Model cache cleared")
