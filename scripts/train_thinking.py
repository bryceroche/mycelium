#!/usr/bin/env python3
"""
Mycelium v20 Training Script: Deep Supervision for State-Conditioned LoRA Thinking

This script trains the ThinkingModel on GSM8K with deep supervision, where every
thinking pass predicts the answer and receives direct gradient signal.

Architecture Overview (v20 - State-Conditioned LoRA):
    STATE (64 floats on hypersphere)
        |
        v
    HYPERNETWORK (StateConditionedLoRA, ~1.1M) -> LoRA scales
        |
        v
    Scale learned A/B templates -> Apply LoRA to Q, K, V, O in ALL 16 layers
        |
        v
    [problem tokens] -> Llama layers 1-16 (WITH LoRA modifications)
        |          |         |              |
      (all 16 layer hidden states saved)
        |          |         |              |
        v          v         v              v
    7-LAYER PERCEIVER COMPRESSOR (105M params) -> 64-float state delta
        |
        v
    state = normalize(state + delta) * sqrt(64)  <- HYPERSPHERE
        |
        +---> ConfidenceHead -> ready?

Key Features:
1. Deep supervision: Every intermediate state tries to answer (not just final)
2. Loss weighted by pass number (later passes weighted more)
3. Confidence loss to teach when to stop
4. Efficiency penalty for too many passes
5. Two-phase training:
   - Phase 1: Freeze Llama, train LoRA templates + Perceiver + Confidence (5-10 epochs)
   - Phase 2: Unfreeze Llama, end-to-end (10-20 epochs)

Usage:
    # Phase 1: Train LoRA templates + Perceiver + Confidence only
    python scripts/train_thinking.py --phase 1

    # Phase 2: End-to-end training
    python scripts/train_thinking.py --phase 2 --resume checkpoints/phase1_best.pt

    # Quick test
    python scripts/train_thinking.py --debug

    # Resume from checkpoint
    python scripts/train_thinking.py --resume checkpoints/latest.pt
"""

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ThinkingModel

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available, logging to console only")

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("datasets not available, will use local JSONL files")


# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    # Model
    "model_name": "meta-llama/Llama-3.2-1B-Instruct",

    # Architecture
    "state_size": 64,
    "lora_rank": 4,
    "num_queries": 4,
    "num_perceiver_layers": 7,
    "d_perceiver": 1024,
    "max_passes": 20,

    # Thinking - FIXED 3 PASSES (no confidence head, no early stopping)
    "confidence_threshold": 1.0,  # Never early stop - always do all passes
    "curriculum_max_passes": [3],  # Always 3 passes (add confidence head back later)
    "curriculum_epochs": [0],      # No curriculum - fixed at 3

    # Training
    "batch_size": 8,           # Increased - was only using 12% GPU
    "grad_accum_steps": 2,     # Effective batch size = 16
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "warmup_steps": 100,

    # Phase-specific
    "epochs": 20,
    "phase1_epochs": 5,        # Freeze Llama
    "phase2_epochs": 15,       # End-to-end

    # Loss weights (DISABLED - only using answer loss with fixed 3 passes)
    "confidence_loss_weight": 0.0,  # Not training confidence head yet
    "efficiency_penalty_weight": 0.0,  # No penalty - always 3 passes

    # Evaluation
    "eval_every_n_epochs": 1,
    "num_eval_samples": 100,
    "num_train_samples": None,  # None = use all

    # Logging
    "log_every_n_steps": 10,
    "save_every_n_epochs": 1,
    "project_name": "mycelium-v20-thinking",
    "run_name": f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}",

    # Checkpoints
    "checkpoint_dir": "checkpoints/v20",

    # Hardware
    "device": "cuda",
}


# ============================================================================
# Data Loading
# ============================================================================

def load_gsm8k_hf(split: str = "train", max_samples: Optional[int] = None) -> List[Dict]:
    """Load GSM8K from HuggingFace datasets."""
    if not DATASETS_AVAILABLE:
        raise RuntimeError("datasets library required: pip install datasets")

    dataset = load_dataset("gsm8k", "main", split=split)

    data = []
    for i, item in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        # Extract numeric answer from GSM8K format: reasoning ... #### answer
        answer_text = item["answer"]
        gold_answer = ""
        if "####" in answer_text:
            gold_answer = answer_text.split("####")[-1].strip().replace(",", "")

        data.append({
            "question": item["question"],
            "answer": answer_text,
            "gold_answer": gold_answer,
        })

    print(f"Loaded GSM8K {split}: {len(data)} examples")
    return data


def load_gsm8k_jsonl(filepath: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load GSM8K from local JSONL file."""
    data = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)
            data.append({
                "question": item.get("question", item.get("problem", "")),
                "answer": item.get("answer", item.get("solution", "")),
                "gold_answer": item.get("gold_answer", item.get("final_answer", "")),
            })

    print(f"Loaded {filepath}: {len(data)} examples")
    return data


def load_data(split: str, config: Dict) -> List[Dict]:
    """Load training or evaluation data."""
    max_samples = config.get("num_train_samples") if split == "train" else config.get("num_eval_samples")

    # Try HuggingFace first
    if DATASETS_AVAILABLE:
        try:
            return load_gsm8k_hf(split, max_samples)
        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")

    # Fallback to local files
    local_paths = {
        "train": ["data/gsm8k_train.jsonl", "data/gsm8k/train.jsonl"],
        "test": ["data/gsm8k_test.jsonl", "data/gsm8k/test.jsonl"],
    }

    for path in local_paths.get(split, []):
        if os.path.exists(path):
            return load_gsm8k_jsonl(path, max_samples)

    raise RuntimeError(f"Could not load GSM8K {split} data. Install datasets or provide local files.")


# ============================================================================
# Answer Extraction and Verification
# ============================================================================

def extract_answer_from_text(text: str) -> Optional[str]:
    """
    Extract numeric answer from model output.

    Looks for:
    1. \\boxed{answer}
    2. #### answer (GSM8K format)
    3. "The answer is X"
    4. Final number in the response
    """
    # Try boxed format first
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1).replace(",", "").strip()

    # Try GSM8K #### format
    if "####" in text:
        answer = text.split("####")[-1].strip().replace(",", "")
        number_match = re.search(r'-?[\d.]+', answer)
        if number_match:
            return number_match.group()
        return answer

    # Try "the answer is X" pattern
    answer_match = re.search(r'answer\s+is\s+[:\s]*(-?[\d,]+(?:\.\d+)?)', text.lower())
    if answer_match:
        return answer_match.group(1).replace(",", "")

    # Last number in text
    numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def check_answer(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer."""
    if predicted is None or not gold:
        return False

    predicted = predicted.strip().lower()
    gold = gold.strip().lower()

    try:
        pred_num = float(predicted)
        gold_num = float(gold)

        # Integer comparison
        if pred_num == int(pred_num) and gold_num == int(gold_num):
            return int(pred_num) == int(gold_num)

        # Float comparison with tolerance
        return abs(pred_num - gold_num) < 1e-5
    except ValueError:
        pass

    return predicted == gold


# ============================================================================
# Training Utilities
# ============================================================================

def get_curriculum_max_passes(epoch: int, config: Dict) -> int:
    """Get max_passes for current epoch based on curriculum."""
    curriculum_epochs = config.get("curriculum_epochs", [0, 5, 10, 15])
    curriculum_passes = config.get("curriculum_max_passes", [3, 5, 7, 10])

    for i in range(len(curriculum_epochs) - 1, -1, -1):
        if epoch >= curriculum_epochs[i]:
            return curriculum_passes[i]

    return curriculum_passes[0]


def compute_cosine_similarity(states: List[torch.Tensor]) -> List[float]:
    """Compute cosine similarity between consecutive states."""
    if len(states) < 2:
        return []

    similarities = []
    for i in range(1, len(states)):
        s1 = states[i - 1].flatten()
        s2 = states[i].flatten()
        sim = F.cosine_similarity(s1.unsqueeze(0), s2.unsqueeze(0)).item()
        similarities.append(sim)

    return similarities


# ============================================================================
# Deep Supervision Loss
# ============================================================================

def compute_deep_supervision_loss(
    model: ThinkingModel,
    problem_text: str,
    gold_answer: str,
    all_states: List[torch.Tensor],
    confidences: List[float],
    max_passes: int,
    config: Dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute deep supervision loss across all thinking passes.

    Every intermediate state tries to predict the answer. Loss is weighted
    by pass number so later passes (which should be better) contribute more.

    Gradient flows: answer_loss -> logits -> LoRA -> state -> compressor

    Args:
        model: The ThinkingModel
        problem_text: The input problem
        gold_answer: The target answer
        all_states: List of state tensors from model.think() [initial, pass1, pass2, ...]
        confidences: List of confidence values from model.think()
        max_passes: Maximum passes for this curriculum stage
        config: Training configuration

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    device = model.device

    # Prepare question prompt (with instruction for step-by-step reasoning)
    formatted_prompt = f"""Solve this math problem step by step. Put your final answer in \\boxed{{}}.

Problem: {problem_text}

Solution:"""
    messages = [{"role": "user", "content": formatted_prompt}]
    prompt_text = model.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    # Tokenize prompt
    prompt_encoding = model.tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=512,
    ).to(device)
    prompt_ids = prompt_encoding.input_ids  # (1, prompt_len)

    # Prepare gold answer tokens for teacher forcing
    # Format: "The answer is X."
    answer_text = f"The answer is {gold_answer}."
    answer_encoding = model.tokenizer(
        answer_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)
    answer_ids = answer_encoding.input_ids  # (1, answer_len)

    # Concatenate for teacher-forced loss
    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)

    # Deep supervision: compute loss for each intermediate state
    # Skip the first state (initial random) which is all_states[0]
    intermediate_states = all_states[1:]  # States after each thinking pass
    num_passes = len(intermediate_states)

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    pass_losses = []

    for i, state in enumerate(intermediate_states):
        # Apply LoRA from this intermediate state
        model.apply_lora(state)

        # Forward with teacher forcing
        outputs = model.transformer(
            input_ids=full_ids,
            use_cache=False,
        )

        # Remove LoRA after forward
        model.remove_lora()

        # Compute loss on answer tokens only
        prompt_len = prompt_ids.size(1)
        answer_len = answer_ids.size(1)

        # Shift logits and labels for next-token prediction
        logits = outputs.logits[:, prompt_len - 1:-1, :]  # (1, answer_len, vocab_size)
        targets = answer_ids.squeeze(0)  # (answer_len,)

        pass_loss = F.cross_entropy(logits.squeeze(0), targets)

        # Weight increases with pass number (later = better = higher weight)
        # weight = (i + 1) / num_passes
        weight = (i + 1) / num_passes
        weighted_loss = weight * pass_loss

        total_loss = total_loss + weighted_loss
        pass_losses.append(pass_loss.item())

    # Normalize by sum of weights: (1 + 2 + ... + n) / n = (n+1)/2
    normalization = (num_passes + 1) / 2
    total_loss = total_loss / normalization

    # SIMPLIFIED: No confidence loss or efficiency penalty
    # Just train on answer loss with fixed 3 passes
    # Add confidence head back once accuracy improves
    combined_loss = total_loss

    metrics = {
        "answer_loss": total_loss.item(),
        "confidence_loss": 0.0,  # Not training confidence head
        "efficiency_penalty": 0.0,  # No penalty for passes
        "num_passes": num_passes,
        "early_pass_loss": pass_losses[0] if pass_losses else 0.0,
        "late_pass_loss": pass_losses[-1] if pass_losses else 0.0,
        "pass_losses": pass_losses,
    }

    return combined_loss, metrics


def compute_confidence_loss(
    confidences: List[float],
    max_passes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute confidence loss to teach when to stop thinking.

    Target: confidence should increase with pass number, reaching
    high values after sufficient passes.

    Target progression: min((pass + 1) / max_passes + 0.1, 0.95)
    """
    if not confidences:
        return torch.tensor(0.0, device=device)

    total_loss = 0.0
    for i, conf in enumerate(confidences):
        # Target confidence increases with pass number
        # Clamp at 0.95 to avoid pushing for 1.0
        target_conf = min((i + 1) / max_passes + 0.1, 0.95)
        loss = (conf - target_conf) ** 2
        total_loss += loss

    return torch.tensor(total_loss / len(confidences), device=device, requires_grad=True)


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate(
    model: ThinkingModel,
    eval_data: List[Dict],
    config: Dict,
    epoch: int = 0,
) -> Dict[str, float]:
    """
    Evaluate model on GSM8K.

    Computes:
    - thinking_accuracy: Accuracy after N thinking passes
    - pass_N_accuracy: Accuracy from intermediate state at pass N
    - avg_passes: Average number of thinking passes
    - avg_confidence: Average final confidence
    - cosine_similarities: State rotation between passes
    """
    model.eval()

    max_passes = get_curriculum_max_passes(epoch, config)
    confidence_threshold = config.get("confidence_threshold", 0.8)
    num_samples = min(config.get("num_eval_samples", 100), len(eval_data))

    # Metrics
    thinking_correct = 0
    total_passes = 0
    total_confidence = 0.0

    # Per-pass accuracy tracking
    pass_correct = {i: 0 for i in range(1, max_passes + 1)}
    pass_counts = {i: 0 for i in range(1, max_passes + 1)}

    # Cosine similarities
    all_cosine_sims = []

    pbar = tqdm(range(num_samples), desc="Evaluating", leave=False)
    for idx in pbar:
        item = eval_data[idx]
        question = item["question"]
        gold = item["gold_answer"]

        if not gold:
            continue

        try:
            # Think
            state, all_states, confidences = model.think(
                problem_text=question,
                max_passes=max_passes,
                confidence_threshold=confidence_threshold,
            )

            # Generate answer
            answer = model.generate_answer(
                problem_text=question,
                state=state,
                max_new_tokens=256,
            )

            predicted = extract_answer_from_text(answer)
            correct = check_answer(predicted, gold)

            if correct:
                thinking_correct += 1

            num_passes = len(confidences)
            total_passes += num_passes

            if confidences:
                total_confidence += confidences[-1]

            # Track per-pass
            if num_passes in pass_counts:
                pass_counts[num_passes] += 1
                if correct:
                    pass_correct[num_passes] += 1

            # Cosine similarities
            cosine_sims = compute_cosine_similarity(all_states)
            if cosine_sims:
                all_cosine_sims.extend(cosine_sims)

        except Exception as e:
            print(f"Evaluation error: {e}")
            continue

    # Compute metrics
    metrics = {
        "thinking_accuracy": thinking_correct / num_samples if num_samples > 0 else 0.0,
        "avg_passes": total_passes / num_samples if num_samples > 0 else 0.0,
        "avg_confidence": total_confidence / num_samples if num_samples > 0 else 0.0,
        "num_samples": num_samples,
    }

    # Per-pass accuracy
    for num_pass in pass_correct:
        if pass_counts[num_pass] > 0:
            metrics[f"pass_{num_pass}_accuracy"] = pass_correct[num_pass] / pass_counts[num_pass]
            metrics[f"pass_{num_pass}_count"] = pass_counts[num_pass]

    # Cosine similarity
    if all_cosine_sims:
        metrics["avg_cosine_similarity"] = sum(all_cosine_sims) / len(all_cosine_sims)

    model.train()
    return metrics


# ============================================================================
# Checkpointing
# ============================================================================

def save_checkpoint(
    model: ThinkingModel,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    step: int,
    config: Dict,
    metrics: Dict[str, float],
    path: str,
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": config,
        "metrics": metrics,
        "architecture": "v20_state_conditioned_lora",
    }

    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    model: ThinkingModel,
    optimizer: Optional[AdamW] = None,
    scheduler: Optional[CosineAnnealingLR] = None,
) -> Dict[str, Any]:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"Loaded checkpoint from {path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Step: {checkpoint.get('step', 'unknown')}")
    print(f"  Architecture: {checkpoint.get('architecture', 'unknown')}")

    return checkpoint


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(
    model: ThinkingModel,
    train_data: List[Dict],
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    config: Dict,
    epoch: int,
    global_step: int,
) -> Tuple[int, Dict[str, float]]:
    """
    Train for one epoch with deep supervision.

    Args:
        model: The ThinkingModel
        train_data: Training data
        optimizer: AdamW optimizer
        scheduler: Learning rate scheduler
        config: Training configuration
        epoch: Current epoch number
        global_step: Global step counter

    Returns:
        Tuple of (new_global_step, epoch_metrics)
    """
    model.train()

    batch_size = config.get("batch_size", 1)
    grad_accum_steps = config.get("grad_accum_steps", 4)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    log_every = config.get("log_every_n_steps", 10)

    # Get curriculum max_passes
    max_passes = get_curriculum_max_passes(epoch, config)
    confidence_threshold = config.get("confidence_threshold", 0.8)

    # Metrics accumulators
    total_loss = 0.0
    total_answer_loss = 0.0
    total_conf_loss = 0.0
    total_passes = 0
    num_batches = 0

    # Shuffle data
    import random
    indices = list(range(len(train_data)))
    random.shuffle(indices)

    # Progress bar
    pbar = tqdm(range(0, len(indices), batch_size), desc=f"Epoch {epoch + 1} (max_passes={max_passes})")

    optimizer.zero_grad()

    for batch_idx, start_idx in enumerate(pbar):
        batch_indices = indices[start_idx:start_idx + batch_size]

        batch_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
        batch_answer_loss = 0.0
        batch_conf_loss = 0.0
        batch_passes = 0
        valid_samples = 0

        for idx in batch_indices:
            item = train_data[idx]
            question = item["question"]
            gold_answer = item["gold_answer"]

            # Skip if no valid gold answer
            if not gold_answer:
                continue

            try:
                # Think about the problem
                state, all_states, confidences = model.think(
                    problem_text=question,
                    max_passes=max_passes,
                    confidence_threshold=confidence_threshold,
                )

                # Deep supervision loss
                loss, metrics = compute_deep_supervision_loss(
                    model=model,
                    problem_text=question,
                    gold_answer=gold_answer,
                    all_states=all_states,
                    confidences=confidences,
                    max_passes=max_passes,
                    config=config,
                )

                batch_loss = batch_loss + loss
                batch_answer_loss += metrics["answer_loss"]
                batch_conf_loss += metrics["confidence_loss"]
                batch_passes += metrics["num_passes"]
                valid_samples += 1

            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue

        # Skip if no valid samples
        if valid_samples == 0:
            continue

        # Average over batch
        batch_loss = batch_loss / valid_samples

        # Backward pass with gradient accumulation
        batch_loss = batch_loss / grad_accum_steps
        batch_loss.backward()

        # Update weights after accumulation steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.get_trainable_parameters(),
                max_grad_norm,
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            # Accumulate metrics
            total_loss += batch_loss.item() * grad_accum_steps
            total_answer_loss += batch_answer_loss / valid_samples
            total_conf_loss += batch_conf_loss / valid_samples
            total_passes += batch_passes / valid_samples
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{batch_loss.item() * grad_accum_steps:.4f}",
                "passes": f"{batch_passes / valid_samples:.1f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            # Logging
            if global_step % log_every == 0 and WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    "train/loss": batch_loss.item() * grad_accum_steps,
                    "train/answer_loss": batch_answer_loss / valid_samples,
                    "train/confidence_loss": batch_conf_loss / valid_samples,
                    "train/avg_passes": batch_passes / valid_samples,
                    "train/max_passes": max_passes,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/step": global_step,
                })

    # Epoch metrics
    epoch_metrics = {
        "loss": total_loss / num_batches if num_batches > 0 else 0.0,
        "answer_loss": total_answer_loss / num_batches if num_batches > 0 else 0.0,
        "confidence_loss": total_conf_loss / num_batches if num_batches > 0 else 0.0,
        "avg_passes": total_passes / num_batches if num_batches > 0 else 0.0,
        "max_passes": max_passes,
    }

    return global_step, epoch_metrics


def train(config: Dict, phase: int, resume_path: Optional[str] = None, weights_only: bool = False) -> None:
    """
    Main training function.

    Args:
        config: Training configuration
        phase: Training phase (1 = freeze transformer, 2 = end-to-end)
        resume_path: Path to checkpoint to resume from
    """
    # Device setup
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    print("\nCreating ThinkingModel (v20 State-Conditioned LoRA)...")
    model = ThinkingModel(
        model_name=config.get("model_name", "meta-llama/Llama-3.2-1B-Instruct"),
        state_size=config.get("state_size", 64),
        lora_rank=config.get("lora_rank", 4),
        num_queries=config.get("num_queries", 4),
        num_perceiver_layers=config.get("num_perceiver_layers", 7),
        d_perceiver=config.get("d_perceiver", 1024),
        max_passes=config.get("max_passes", 20),
    )
    model = model.to(device)

    # Count parameters
    param_counts = model.count_parameters()
    print("\nParameter counts:")
    for name, count in param_counts.items():
        if count > 1e6:
            print(f"  {name}: {count / 1e6:.1f}M")
        else:
            print(f"  {name}: {count:,}")

    # Phase-specific setup
    if phase == 1:
        model.freeze_transformer()
        num_epochs = config.get("phase1_epochs", 5)
        print(f"\nPhase 1: Transformer FROZEN, training LoRA + Perceiver + Confidence")
    else:
        model.unfreeze_transformer()
        num_epochs = config.get("phase2_epochs", 15)
        print(f"\nPhase 2: Transformer UNFROZEN, end-to-end training")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # Load data
    print("\nLoading data...")
    train_data = load_data("train", config)
    eval_data = load_data("test", config)

    # Optimizer and scheduler
    # Different LRs for different components to handle gradient imbalance
    # Perceiver is close to loss (small LR ok), templates are far (need bigger push)
    base_lr = config.get("learning_rate", 1e-4)
    template_lr = config.get("template_lr", 1e-3)      # 10x higher (100x was too aggressive)
    hypernetwork_lr = config.get("hypernetwork_lr", 1e-3)  # 10x higher
    weight_decay = config.get("weight_decay", 0.01)
    warmup_steps = config.get("warmup_steps", 100)

    # Separate parameter groups
    template_params = []
    hypernetwork_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "A_templates" in name or "B_templates" in name:
            template_params.append(param)
        elif "state_to_scales" in name:
            hypernetwork_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": other_params, "lr": base_lr},           # Perceiver, confidence
        {"params": template_params, "lr": template_lr},     # LoRA templates (10x)
        {"params": hypernetwork_params, "lr": hypernetwork_lr},  # Hypernetwork (10x)
    ]

    print(f"\nOptimizer parameter groups:")
    print(f"  Perceiver/other: {len(other_params)} params, lr={base_lr}")
    print(f"  LoRA templates:  {len(template_params)} params, lr={template_lr}")
    print(f"  Hypernetwork:    {len(hypernetwork_params)} params, lr={hypernetwork_lr}")

    optimizer = AdamW(
        param_groups,
        weight_decay=weight_decay,
    )

    # Calculate total steps for scheduler
    batch_size = config.get("batch_size", 1)
    grad_accum = config.get("grad_accum_steps", 4)
    steps_per_epoch = len(train_data) // (batch_size * grad_accum)
    total_steps = steps_per_epoch * num_epochs

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=base_lr * 0.1,
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0

    if resume_path and os.path.exists(resume_path):
        if weights_only:
            # Only load model weights, not optimizer/scheduler (for LR scheme changes)
            checkpoint = load_checkpoint(resume_path, model, None, None)
            print("Loaded model weights only (fresh optimizer/scheduler)")
        else:
            checkpoint = load_checkpoint(resume_path, model, optimizer, scheduler)
            start_epoch = checkpoint.get("epoch", 0) + 1
            global_step = checkpoint.get("step", 0)
            print(f"Resuming from epoch {start_epoch}, step {global_step}")

    # Initialize wandb
    if WANDB_AVAILABLE:
        try:
            wandb.init(
                project=config.get("project_name", "mycelium-v20-thinking"),
                name=f"{config.get('run_name', 'train')}-phase{phase}",
                config=config,
            )
        except Exception as e:
            print(f"wandb init failed: {e}")

    # Checkpoint directory
    save_dir = Path(config.get("checkpoint_dir", "checkpoints/v20"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\nStarting training: Phase {phase}, {num_epochs} epochs")
    print(f"  Batch size: {batch_size} x {grad_accum} = {batch_size * grad_accum} effective")
    print(f"  Learning rates: perceiver={base_lr}, templates={template_lr}, hypernetwork={hypernetwork_lr}")
    print(f"  Total steps: {total_steps}")

    best_accuracy = 0.0

    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        # Get curriculum max_passes
        curr_max_passes = get_curriculum_max_passes(epoch, config)
        print(f"Curriculum max_passes: {curr_max_passes}")

        # Train
        global_step, epoch_metrics = train_epoch(
            model=model,
            train_data=train_data,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            epoch=epoch,
            global_step=global_step,
        )

        print(f"\nEpoch {epoch + 1} training metrics:")
        for name, value in epoch_metrics.items():
            print(f"  {name}: {value:.4f}")

        # Evaluate
        if (epoch + 1) % config.get("eval_every_n_epochs", 1) == 0:
            print("\nEvaluating...")
            eval_metrics = evaluate(
                model=model,
                eval_data=eval_data,
                config=config,
                epoch=epoch,
            )

            print(f"\nEpoch {epoch + 1} evaluation metrics:")
            for name, value in eval_metrics.items():
                if isinstance(value, float):
                    print(f"  {name}: {value:.4f}")
                else:
                    print(f"  {name}: {value}")

            # Log to wandb
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    f"eval/{k}": v for k, v in eval_metrics.items()
                    if isinstance(v, (int, float))
                })
                wandb.log({f"epoch/{k}": v for k, v in epoch_metrics.items()})
                wandb.log({"epoch": epoch + 1})

            # Check for best model
            is_best = eval_metrics.get("thinking_accuracy", 0) > best_accuracy
            if is_best:
                best_accuracy = eval_metrics["thinking_accuracy"]
                print(f"\nNew best accuracy: {best_accuracy:.4f}")
        else:
            eval_metrics = {}
            is_best = False

        # Save checkpoints
        if (epoch + 1) % config.get("save_every_n_epochs", 1) == 0:
            # Save latest
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=global_step,
                config=config,
                metrics=eval_metrics,
                path=str(save_dir / f"phase{phase}_latest.pt"),
            )

            # Save best
            if is_best:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=global_step,
                    config=config,
                    metrics=eval_metrics,
                    path=str(save_dir / f"phase{phase}_best.pt"),
                )

            # Save periodic
            if (epoch + 1) % 5 == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=global_step,
                    config=config,
                    metrics=eval_metrics,
                    path=str(save_dir / f"phase{phase}_epoch{epoch + 1}.pt"),
                )

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    final_metrics = evaluate(
        model=model,
        eval_data=eval_data,
        config=config,
        epoch=num_epochs - 1,
    )

    print("\nFinal metrics:")
    for name, value in final_metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")

    # Log final metrics
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({f"final/{k}": v for k, v in final_metrics.items() if isinstance(v, (int, float))})
        wandb.finish()

    print(f"\nTraining complete!")
    print(f"Best thinking accuracy: {best_accuracy:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train ThinkingModel v20 on GSM8K with deep supervision"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=1,
        help="Training phase: 1=freeze transformer, 2=end-to-end",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: smaller dataset, fewer epochs",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Only load model weights from checkpoint, not optimizer/scheduler (use when changing LR scheme)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=None,
        help="Override number of evaluation samples",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Limit training samples for faster iteration",
    )

    args = parser.parse_args()

    # Copy config and apply overrides
    config = CONFIG.copy()

    if args.epochs is not None:
        if args.phase == 1:
            config["phase1_epochs"] = args.epochs
        else:
            config["phase2_epochs"] = args.epochs

    if args.lr is not None:
        config["learning_rate"] = args.lr

    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    if args.eval_samples is not None:
        config["num_eval_samples"] = args.eval_samples

    if args.train_samples is not None:
        config["num_train_samples"] = args.train_samples

    # Debug mode adjustments
    if args.debug:
        print("\n*** DEBUG MODE ***")
        config["phase1_epochs"] = 2
        config["phase2_epochs"] = 2
        config["num_eval_samples"] = 20
        config["num_train_samples"] = 100
        config["log_every_n_steps"] = 1
        config["curriculum_max_passes"] = [2, 3]
        config["curriculum_epochs"] = [0, 1]

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Run training
    train(config, args.phase, args.resume, args.weights_only)


if __name__ == "__main__":
    main()
