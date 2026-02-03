#!/usr/bin/env python3
"""Extract attention patterns from Qwen-2.5-7B for teacher distillation.

This script processes GSM8K problems through Qwen-2.5-7B and extracts attention
patterns that encode mathematical understanding. These patterns serve as "teacher"
signal for training a smaller student model (MiniLM-22M).

HARDWARE REQUIREMENTS:
- GPU: ~16GB VRAM for full precision, ~8GB for 8-bit quantization, ~6GB for 4-bit
- RAM: ~16GB for loading model + dataset
- Storage: ~15GB for Qwen model weights + ~2GB for extracted attention data

ARCHITECTURE (Qwen-2.5-7B):
- 28 layers
- 28 attention heads per layer
- Hidden dimension: 3584
- Head dimension: 128

USAGE:
    # On GPU VM (full precision, fastest):
    python scripts/qwen_attention_extraction.py --batch-size 4

    # On GPU VM (8-bit quantization, lower VRAM):
    python scripts/qwen_attention_extraction.py --quantize 8bit --batch-size 8

    # On GPU VM (4-bit quantization, lowest VRAM):
    python scripts/qwen_attention_extraction.py --quantize 4bit --batch-size 16

    # Test mode (small subset):
    python scripts/qwen_attention_extraction.py --test-mode --num-samples 10

OUTPUT:
    Saves to /data/qwen_attention/ directory:
    - attention_patterns_{batch_id}.npz: Compressed attention matrices
    - metadata_{batch_id}.json: Problem IDs, tokens, layer/head info
    - processing_stats.json: Overall statistics

DISTILLATION TARGET:
    The extracted attention patterns will be used to train MiniLM-22M to predict
    attention-derived features (not raw attention matrices, which are too large).
    Specifically, we extract:
    1. Per-token attention entropy (which tokens are "important")
    2. Span connectivity scores (which tokens group together)
    3. Cross-clause attention (how clauses relate)
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExtractionConfig:
    """Configuration for attention extraction."""
    # Model settings
    model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    quantize: Optional[str] = None  # None, "4bit", or "8bit"

    # Layer/head selection
    # By default, extract from last 8 layers (more semantic) and all heads
    # Middle layers (9-18) often have more semantic info for math
    layer_strategy: str = "semantic"  # "all", "last_n", "middle", "semantic"
    num_layers: int = 8  # For "last_n" strategy

    # Dataset settings
    dataset: str = "both"  # "gsm8k", "math", or "both"
    math_levels: List[str] = field(default_factory=lambda: ["Level 1", "Level 2"])

    # Processing settings
    batch_size: int = 1
    max_seq_length: int = 512
    num_samples: Optional[int] = None  # None = all

    # Output settings
    output_dir: str = "/data/qwen_attention"
    save_every_n_batches: int = 100
    compress_output: bool = True  # Use npz instead of npy

    # Memory optimization
    clear_cache_every_n: int = 10
    fp16_attention: bool = True  # Keep attention in fp16 to save memory

    # Feature extraction (what to save)
    save_raw_attention: bool = False  # Full attention matrices (LARGE!)
    save_attention_entropy: bool = True
    save_connectivity_scores: bool = True
    save_hidden_states: bool = False  # Final layer hidden states


@dataclass
class AttentionFeatures:
    """Extracted attention features for a single problem."""
    problem_id: str
    problem_text: str
    tokens: List[str]
    seq_length: int

    # Per-token features (seq_length,)
    attention_entropy: Optional[np.ndarray] = None  # How focused each token's attention is
    attention_received: Optional[np.ndarray] = None  # How much attention each token receives

    # Connectivity features (seq_length, seq_length)
    span_connectivity: Optional[np.ndarray] = None  # Token-pair grouping scores

    # Raw attention (optional, very large)
    # Shape: (num_layers, num_heads, seq_length, seq_length)
    raw_attention: Optional[np.ndarray] = None

    # Hidden states (optional)
    hidden_states: Optional[np.ndarray] = None  # (seq_length, hidden_dim)

    # Metadata
    layer_indices: List[int] = field(default_factory=list)
    extraction_time_ms: float = 0.0


# =============================================================================
# Model Loading
# =============================================================================

def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(config: ExtractionConfig):
    """Load Qwen model with attention output enabled.

    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device = get_device()
    logger.info(f"Loading {config.model_name} on {device}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="left"  # For batch processing
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model loading kwargs
    model_kwargs = {
        "output_attentions": True,
        "output_hidden_states": config.save_hidden_states,
        "trust_remote_code": True,
        "attn_implementation": "eager",  # Required for attention output
    }

    # Quantization config
    if config.quantize == "4bit":
        logger.info("Using 4-bit quantization (bitsandbytes)")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model_kwargs["device_map"] = "auto"
    elif config.quantize == "8bit":
        logger.info("Using 8-bit quantization (bitsandbytes)")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model_kwargs["device_map"] = "auto"
    else:
        # Full precision
        model_kwargs["torch_dtype"] = torch.float16 if device != "cpu" else torch.float32
        model_kwargs["device_map"] = "auto" if device == "cuda" else None

    # Load model
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    # Move to device if not using device_map
    if "device_map" not in model_kwargs or model_kwargs["device_map"] is None:
        model = model.to(device)

    model.eval()

    # Log model info
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {num_params/1e9:.2f}B parameters")
    logger.info(f"Layers: {model.config.num_hidden_layers}")
    logger.info(f"Attention heads: {model.config.num_attention_heads}")

    return model, tokenizer


# =============================================================================
# Layer Selection Strategies
# =============================================================================

def get_layer_indices(config: ExtractionConfig, total_layers: int) -> List[int]:
    """Determine which layers to extract attention from.

    Different strategies optimize for different use cases:
    - "all": All layers (most complete, highest storage)
    - "last_n": Last N layers (more semantic/abstract)
    - "middle": Middle third (often best for semantic grouping)
    - "semantic": Heuristic mix of middle and late layers
    """
    if config.layer_strategy == "all":
        return list(range(total_layers))

    elif config.layer_strategy == "last_n":
        n = min(config.num_layers, total_layers)
        return list(range(total_layers - n, total_layers))

    elif config.layer_strategy == "middle":
        # Middle third of layers
        start = total_layers // 3
        end = 2 * total_layers // 3
        return list(range(start, end))

    elif config.layer_strategy == "semantic":
        # Heuristic: mix of middle and late layers
        # Middle layers (semantic structure) + late layers (task-specific)
        middle_start = total_layers // 3
        middle_end = total_layers // 2
        late_start = total_layers - total_layers // 4

        middle = list(range(middle_start, middle_end))
        late = list(range(late_start, total_layers))

        # Combine and dedupe
        return sorted(set(middle + late))

    else:
        raise ValueError(f"Unknown layer strategy: {config.layer_strategy}")


# =============================================================================
# Attention Feature Extraction
# =============================================================================

def compute_attention_entropy(attention: np.ndarray) -> np.ndarray:
    """Compute per-token attention entropy.

    For each token, computes the entropy of its attention distribution
    over previous tokens. High entropy = diffuse attention, low = focused.

    Args:
        attention: (num_layers, num_heads, seq_len, seq_len) attention weights

    Returns:
        (seq_len,) array of entropy values averaged across layers/heads
    """
    from scipy.stats import entropy as scipy_entropy

    # Average across layers and heads first
    # (num_layers, num_heads, seq_len, seq_len) -> (seq_len, seq_len)
    attn_avg = attention.mean(axis=(0, 1))

    seq_len = attn_avg.shape[0]
    entropies = np.zeros(seq_len, dtype=np.float32)

    for i in range(seq_len):
        # Attention from token i to positions 0..i (causal)
        row = attn_avg[i, :i+1]
        if len(row) > 1 and row.sum() > 1e-8:
            # Normalize and compute entropy
            row = row / row.sum()
            entropies[i] = scipy_entropy(row + 1e-10)
        else:
            entropies[i] = 0.0

    return entropies


def compute_attention_received(attention: np.ndarray) -> np.ndarray:
    """Compute how much attention each token receives from later tokens.

    Tokens that receive high attention from many later tokens are likely
    important (numbers, entities, operators).

    Args:
        attention: (num_layers, num_heads, seq_len, seq_len)

    Returns:
        (seq_len,) array of received attention (averaged across layers/heads)
    """
    # Average across layers and heads
    attn_avg = attention.mean(axis=(0, 1))  # (seq_len, seq_len)

    # Sum columns (attention received by each token)
    received = attn_avg.sum(axis=0)  # (seq_len,)

    # Normalize by sequence length
    received = received / (attn_avg.shape[0] + 1e-8)

    return received.astype(np.float32)


def compute_span_connectivity(attention: np.ndarray) -> np.ndarray:
    """Compute token-pair connectivity scores for span detection.

    Mutual attention between token pairs indicates they should group together.
    High connectivity between adjacent tokens suggests they form a span.

    Args:
        attention: (num_layers, num_heads, seq_len, seq_len)

    Returns:
        (seq_len, seq_len) symmetric connectivity matrix
    """
    # Average across layers and heads
    attn_avg = attention.mean(axis=(0, 1))  # (seq_len, seq_len)

    # Compute mutual attention (average of A->B and B->A)
    connectivity = (attn_avg + attn_avg.T) / 2

    return connectivity.astype(np.float32)


def extract_features_from_attention(
    attention_tuple: Tuple[torch.Tensor, ...],
    config: ExtractionConfig,
    layer_indices: List[int]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract features from attention tensors.

    Args:
        attention_tuple: Tuple of attention tensors, one per layer
            Each tensor: (batch, num_heads, seq_len, seq_len)
        config: Extraction configuration
        layer_indices: Which layers to use

    Returns:
        Tuple of (entropy, received, connectivity, raw_attention)
    """
    # Stack selected layers: (num_selected_layers, batch, heads, seq, seq)
    selected = torch.stack([attention_tuple[i] for i in layer_indices])

    # Remove batch dimension (assume batch size 1 for now)
    # -> (num_layers, heads, seq, seq)
    selected = selected.squeeze(1)

    # Convert to numpy
    if config.fp16_attention:
        attention_np = selected.cpu().half().numpy().astype(np.float32)
    else:
        attention_np = selected.cpu().float().numpy()

    # Compute features
    entropy = None
    received = None
    connectivity = None
    raw = None

    if config.save_attention_entropy:
        entropy = compute_attention_entropy(attention_np)
        received = compute_attention_received(attention_np)

    if config.save_connectivity_scores:
        connectivity = compute_span_connectivity(attention_np)

    if config.save_raw_attention:
        raw = attention_np

    return entropy, received, connectivity, raw


# =============================================================================
# Dataset Loading
# =============================================================================

def load_gsm8k(split: str = "train", num_samples: Optional[int] = None) -> List[Dict]:
    """Load GSM8K dataset.

    Args:
        split: "train" or "test"
        num_samples: Limit number of samples (None = all)

    Returns:
        List of problem dicts with 'question' and 'answer' keys
    """
    from datasets import load_dataset

    logger.info(f"Loading GSM8K {split} split...")
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    problems = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break
        problems.append({
            "id": f"gsm8k_{split}_{i}",
            "question": item["question"],
            "answer": item["answer"],
            "source": "gsm8k"
        })

    logger.info(f"Loaded {len(problems)} GSM8K problems")
    return problems


def load_math_levels(levels: List[str] = ["Level 1", "Level 2"], split: str = "train") -> List[Dict]:
    """Load MATH dataset filtered by difficulty levels.

    Args:
        levels: List of levels to include (e.g., ["Level 1", "Level 2"])
        split: "train" or "test"

    Returns:
        List of problem dicts with 'question' and 'answer' keys
    """
    from datasets import load_dataset

    configs = ['algebra', 'counting_and_probability', 'geometry',
               'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']

    problems = []
    for config in configs:
        try:
            logger.info(f"Loading MATH/{config} {split}...")
            dataset = load_dataset("EleutherAI/hendrycks_math", config, split=split)

            for i, item in enumerate(dataset):
                if item.get("level") in levels:
                    problems.append({
                        "id": f"math_{config}_{i}",
                        "question": item["problem"],
                        "answer": item["solution"],
                        "level": item.get("level"),
                        "type": config,
                        "source": "math"
                    })
        except Exception as e:
            logger.warning(f"Failed to load MATH/{config}: {e}")

    logger.info(f"Loaded {len(problems)} MATH L1/L2 problems")
    return problems


def load_combined_dataset(
    include_gsm8k: bool = True,
    include_math: bool = True,
    math_levels: List[str] = ["Level 1", "Level 2"],
    num_samples: Optional[int] = None
) -> List[Dict]:
    """Load combined GSM8K + MATH dataset.

    Args:
        include_gsm8k: Include GSM8K problems
        include_math: Include MATH problems
        math_levels: Which MATH levels to include
        num_samples: Limit total samples (None = all)

    Returns:
        Combined list of problems
    """
    problems = []

    if include_gsm8k:
        problems.extend(load_gsm8k("train"))

    if include_math:
        problems.extend(load_math_levels(math_levels, "train"))

    logger.info(f"Combined dataset: {len(problems)} total problems")

    if num_samples and len(problems) > num_samples:
        import random
        random.shuffle(problems)
        problems = problems[:num_samples]
        logger.info(f"Sampled down to {len(problems)} problems")

    return problems


# =============================================================================
# Main Extraction Loop
# =============================================================================

def process_batch(
    model,
    tokenizer,
    problems: List[Dict],
    config: ExtractionConfig,
    layer_indices: List[int]
) -> List[AttentionFeatures]:
    """Process a batch of problems and extract attention features.

    Args:
        model: Loaded Qwen model
        tokenizer: Tokenizer
        problems: List of problem dicts
        config: Extraction config
        layer_indices: Which layers to extract

    Returns:
        List of AttentionFeatures for each problem
    """
    device = next(model.parameters()).device
    results = []

    for problem in problems:
        start_time = time.time()

        # Tokenize
        text = problem["question"]
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=config.max_seq_length,
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                output_hidden_states=config.save_hidden_states
            )

        # Extract features from attention
        entropy, received, connectivity, raw = extract_features_from_attention(
            outputs.attentions,
            config,
            layer_indices
        )

        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())

        # Hidden states (optional)
        hidden = None
        if config.save_hidden_states and outputs.hidden_states:
            hidden = outputs.hidden_states[-1].squeeze(0).cpu().float().numpy()

        # Create result
        features = AttentionFeatures(
            problem_id=problem["id"],
            problem_text=text,
            tokens=tokens,
            seq_length=len(tokens),
            attention_entropy=entropy,
            attention_received=received,
            span_connectivity=connectivity,
            raw_attention=raw,
            hidden_states=hidden,
            layer_indices=layer_indices,
            extraction_time_ms=(time.time() - start_time) * 1000
        )
        results.append(features)

        # Clear GPU cache periodically
        del outputs

    return results


def save_batch(
    features_list: List[AttentionFeatures],
    batch_id: int,
    config: ExtractionConfig
):
    """Save a batch of extracted features to disk.

    Saves:
    - Compressed numpy arrays for numerical data
    - JSON metadata for text/tokens
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare arrays and metadata
    metadata = []
    entropy_list = []
    received_list = []
    connectivity_list = []

    for feat in features_list:
        metadata.append({
            "problem_id": feat.problem_id,
            "problem_text": feat.problem_text,
            "tokens": feat.tokens,
            "seq_length": feat.seq_length,
            "layer_indices": feat.layer_indices,
            "extraction_time_ms": feat.extraction_time_ms
        })

        if feat.attention_entropy is not None:
            entropy_list.append(feat.attention_entropy)
        if feat.attention_received is not None:
            received_list.append(feat.attention_received)
        if feat.span_connectivity is not None:
            connectivity_list.append(feat.span_connectivity)

    # Save metadata
    meta_path = output_dir / f"metadata_{batch_id:04d}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save arrays (compressed)
    if entropy_list:
        arrays_path = output_dir / f"features_{batch_id:04d}.npz"
        np.savez_compressed(
            arrays_path,
            entropy=np.array(entropy_list, dtype=object),  # Variable length
            received=np.array(received_list, dtype=object),
            connectivity=np.array(connectivity_list, dtype=object)
        )

    logger.info(f"Saved batch {batch_id}: {len(features_list)} problems to {output_dir}")


def run_extraction(config: ExtractionConfig):
    """Main extraction pipeline."""
    logger.info("=" * 60)
    logger.info("Qwen-2.5-7B Attention Extraction")
    logger.info("=" * 60)
    logger.info(f"Config: {asdict(config)}")

    # Load model
    model, tokenizer = load_model(config)

    # Determine layers
    total_layers = model.config.num_hidden_layers
    layer_indices = get_layer_indices(config, total_layers)
    logger.info(f"Extracting from layers: {layer_indices}")

    # Load dataset
    include_gsm8k = config.dataset in ["gsm8k", "both"]
    include_math = config.dataset in ["math", "both"]
    problems = load_combined_dataset(
        include_gsm8k=include_gsm8k,
        include_math=include_math,
        math_levels=config.math_levels,
        num_samples=config.num_samples
    )

    # Processing stats
    stats = {
        "start_time": datetime.now().isoformat(),
        "config": asdict(config),
        "total_problems": len(problems),
        "layer_indices": layer_indices,
        "batches_processed": 0,
        "total_time_sec": 0,
        "problems_per_sec": 0
    }

    # Process in batches
    all_features = []
    batch_id = 0
    start_time = time.time()

    pbar = tqdm(range(0, len(problems), config.batch_size), desc="Processing")

    for i in pbar:
        batch = problems[i:i + config.batch_size]

        # Extract features
        features = process_batch(model, tokenizer, batch, config, layer_indices)
        all_features.extend(features)

        # Save periodically
        if len(all_features) >= config.save_every_n_batches:
            save_batch(all_features, batch_id, config)
            all_features = []
            batch_id += 1

        # Clear GPU cache
        if i % (config.clear_cache_every_n * config.batch_size) == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Update progress
        elapsed = time.time() - start_time
        problems_done = i + len(batch)
        rate = problems_done / elapsed if elapsed > 0 else 0
        pbar.set_postfix({
            "rate": f"{rate:.1f}/s",
            "batch": batch_id
        })

        stats["batches_processed"] = batch_id

    # Save remaining
    if all_features:
        save_batch(all_features, batch_id, config)

    # Final stats
    total_time = time.time() - start_time
    stats["end_time"] = datetime.now().isoformat()
    stats["total_time_sec"] = total_time
    stats["problems_per_sec"] = len(problems) / total_time if total_time > 0 else 0

    # Save stats
    output_dir = Path(config.output_dir)
    stats_path = output_dir / "processing_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("=" * 60)
    logger.info("Extraction Complete!")
    logger.info(f"Processed {len(problems)} problems in {total_time:.1f}s")
    logger.info(f"Rate: {stats['problems_per_sec']:.2f} problems/sec")
    logger.info(f"Output: {config.output_dir}")
    logger.info("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract attention patterns from Qwen-2.5-7B for distillation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model settings
    parser.add_argument(
        "--model", "-m",
        default="Qwen/Qwen2.5-Math-7B-Instruct",
        help="Model name (default: Qwen/Qwen2.5-Math-7B-Instruct)"
    )
    parser.add_argument(
        "--quantize", "-q",
        choices=["4bit", "8bit"],
        help="Quantization mode (reduces VRAM, slightly slower)"
    )

    # Processing settings
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Batch size (default: 1, increase for 4bit/8bit)"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        help="Number of samples to process (default: all)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )

    # Layer selection
    parser.add_argument(
        "--layer-strategy",
        choices=["all", "last_n", "middle", "semantic"],
        default="semantic",
        help="Layer selection strategy (default: semantic)"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=8,
        help="Number of layers for 'last_n' strategy (default: 8)"
    )

    # Output settings
    parser.add_argument(
        "--output-dir", "-o",
        default="/data/qwen_attention",
        help="Output directory (default: /data/qwen_attention)"
    )
    parser.add_argument(
        "--save-raw-attention",
        action="store_true",
        help="Save raw attention matrices (WARNING: very large!)"
    )
    parser.add_argument(
        "--save-hidden-states",
        action="store_true",
        help="Save final layer hidden states"
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        choices=["gsm8k", "math", "both"],
        default="both",
        help="Dataset to process: gsm8k, math (L1+L2), or both (default: both)"
    )
    parser.add_argument(
        "--math-levels",
        nargs="+",
        default=["Level 1", "Level 2"],
        help="MATH levels to include (default: 'Level 1' 'Level 2')"
    )

    # Test mode
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Test mode: process small subset and print results"
    )

    args = parser.parse_args()

    # Build config
    config = ExtractionConfig(
        model_name=args.model,
        quantize=args.quantize,
        dataset=args.dataset,
        math_levels=args.math_levels,
        batch_size=args.batch_size,
        num_samples=args.num_samples if not args.test_mode else 5,
        max_seq_length=args.max_seq_length,
        layer_strategy=args.layer_strategy,
        num_layers=args.num_layers,
        output_dir=args.output_dir if not args.test_mode else "/tmp/qwen_test",
        save_raw_attention=args.save_raw_attention,
        save_hidden_states=args.save_hidden_states
    )

    if args.test_mode:
        logger.info("TEST MODE - processing small subset")

    run_extraction(config)


if __name__ == "__main__":
    main()
