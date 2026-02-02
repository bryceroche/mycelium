"""Extract attention matrices from transformer models.

Uses DeepSeek-Math (or similar) to get attention patterns that
encode mathematical understanding.

Why DeepSeek?
- SOTA math accuracy = attention patterns encode actual math structure
- A model that gets wrong answers has misleading attention patterns
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AttentionResult:
    """Result of attention extraction."""
    tokens: List[str]
    # Shape: (num_layers, num_heads, seq_len, seq_len)
    attention: np.ndarray
    # Which layers/heads to use (determined empirically)
    semantic_layers: List[int] = None
    semantic_heads: List[int] = None


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
    # TODO: Implement with transformers library
    #
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    #
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     output_attentions=True,
    #     device_map="auto",
    # )
    #
    # inputs = tokenizer(text, return_tensors="pt")
    # outputs = model(**inputs, output_attentions=True)
    #
    # # outputs.attentions is tuple of (batch, heads, seq, seq) per layer
    # attention = torch.stack(outputs.attentions).numpy()
    # tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

    raise NotImplementedError("Attention extraction not yet implemented")


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
