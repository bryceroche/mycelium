#!/usr/bin/env python3
"""
Mycelium v18 Training Script: Deep Supervision for Integrated Thinking

This script trains the ThinkingModel on GSM8K with deep supervision, where every
thinking pass predicts the answer and receives direct gradient signal.

Key features:
1. Two-phase training:
   - Phase 1: Freeze transformer, train perceiver + injector + confidence
   - Phase 2: Unfreeze everything for end-to-end training

2. Deep supervision:
   - Every thinking pass predicts the answer from current state
   - Weight increases with pass number: weight = (pass_num + 1) / num_passes
   - Sum weighted losses across all passes
   - Every pass gets direct gradient, no vanishing gradient

3. State scale warmup:
   - Scale pseudo-tokens by factor that ramps 0.1 -> 1.0 over 5 epochs
   - Gives transformer time to adjust to pseudo-token distribution

4. Confidence loss:
   - Confidence should increase over passes
   - Target confidence = (pass_num + 1) / max_passes

Usage:
    # Phase 1: Train compression components only
    python scripts/train_thinking.py --config configs/thinking_gsm8k.yaml --phase 1

    # Phase 2: End-to-end training
    python scripts/train_thinking.py --config configs/thinking_gsm8k.yaml --phase 2

    # Resume from checkpoint
    python scripts/train_thinking.py --config configs/thinking_gsm8k.yaml --resume checkpoint.pt
"""

import argparse
import os
import sys
import re
import math
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import yaml
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import ThinkingModel

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available, logging to console only")

# Optional datasets import (HuggingFace)
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("datasets not available, cannot load GSM8K")


# ============================================================================
# Data Loading
# ============================================================================

class GSM8KDataset(Dataset):
    """
    GSM8K dataset wrapper for training.

    Each item contains:
        - question: The problem text
        - answer: The full solution with reasoning
        - gold_answer: The extracted numeric answer
    """

    def __init__(self, split: str = "train", max_samples: Optional[int] = None):
        """
        Load GSM8K dataset.

        Args:
            split: "train" or "test"
            max_samples: Limit number of samples (for debugging)
        """
        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets library required: pip install datasets")

        self.data = load_dataset("gsm8k", "main", split=split)

        if max_samples is not None:
            self.data = self.data.select(range(min(max_samples, len(self.data))))

        print(f"Loaded GSM8K {split}: {len(self.data)} examples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        item = self.data[idx]

        # Extract the numeric answer from the solution
        # GSM8K format: reasoning ... #### answer
        answer_text = item["answer"]
        gold_answer = self._extract_gold_answer(answer_text)

        return {
            "question": item["question"],
            "answer": answer_text,
            "gold_answer": gold_answer,
        }

    @staticmethod
    def _extract_gold_answer(answer_text: str) -> str:
        """Extract the numeric answer from GSM8K format."""
        # GSM8K uses #### to mark the final answer
        if "####" in answer_text:
            answer = answer_text.split("####")[-1].strip()
            # Remove commas from numbers
            answer = answer.replace(",", "")
            return answer
        return ""


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
        answer = text.split("####")[-1].strip()
        answer = answer.replace(",", "")
        # Extract just the number
        number_match = re.search(r'-?[\d.]+', answer)
        if number_match:
            return number_match.group()
        return answer

    # Try "the answer is X" pattern
    answer_match = re.search(r'answer\s+is\s+[:\s]*(-?[\d,]+(?:\.\d+)?)', text.lower())
    if answer_match:
        return answer_match.group(1).replace(",", "")

    # Try to find the last number in the text
    numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def check_answer(predicted: str, gold: str) -> bool:
    """
    Check if predicted answer matches gold answer.

    Handles:
    - Integer comparison
    - Float comparison with tolerance
    - String comparison as fallback
    """
    if predicted is None:
        return False

    # Normalize
    predicted = predicted.strip().lower()
    gold = gold.strip().lower()

    # Try numeric comparison
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

    # String comparison
    return predicted == gold


# ============================================================================
# Training Utilities
# ============================================================================

def compute_state_scale(epoch: int, config: Dict) -> float:
    """
    Compute state scale for warmup.

    Scale ramps from start_scale to end_scale over warmup_epochs.
    After warmup, scale is fixed at end_scale.
    """
    warmup_cfg = config["training"]["state_scale_warmup"]

    if not warmup_cfg["enabled"]:
        return 1.0

    start_scale = warmup_cfg["start_scale"]
    end_scale = warmup_cfg["end_scale"]
    warmup_epochs = warmup_cfg["warmup_epochs"]

    if epoch >= warmup_epochs:
        return end_scale

    # Linear interpolation
    progress = epoch / warmup_epochs
    return start_scale + progress * (end_scale - start_scale)


def compute_deep_supervision_loss(
    model: ThinkingModel,
    problem_text: str,
    gold_answer: str,
    all_states: List[torch.Tensor],
    config: Dict,
    state_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute deep supervision loss across all thinking passes.

    Every intermediate state predicts the answer. Weight increases
    with pass number so later passes (which should be better)
    contribute more to the loss.

    Args:
        model: The ThinkingModel
        problem_text: The input problem
        gold_answer: The target answer
        all_states: List of state tensors from model.think()
        config: Training configuration
        state_scale: Current state scale for warmup

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    device = model.device
    dtype = next(model.transformer.parameters()).dtype

    # Prepare prompt
    messages = [{"role": "user", "content": problem_text}]
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
    ).to(device)
    prompt_ids = prompt_encoding.input_ids  # (1, prompt_len)

    # Get prompt embeddings
    prompt_embeds = model.transformer.get_input_embeddings()(prompt_ids)

    # Prepare gold answer tokens for teacher forcing
    # Format: "The answer is X."
    answer_text = f"The answer is {gold_answer}."
    answer_encoding = model.tokenizer(
        answer_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)
    answer_ids = answer_encoding.input_ids  # (1, answer_len)
    answer_embeds = model.transformer.get_input_embeddings()(answer_ids)

    # Deep supervision: compute loss for each intermediate state
    # Skip the first state (zeros) which is all_states[0]
    intermediate_states = all_states[1:]  # States after each thinking pass
    num_passes = len(intermediate_states)

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    pass_losses = []

    for i, state in enumerate(intermediate_states):
        # Apply state scale warmup
        scaled_state = state * state_scale

        # Inject state as pseudo-tokens
        state_tokens = model.injector(scaled_state)
        state_tokens = state_tokens.to(dtype=dtype)

        # Build input: [state_tokens] + [prompt] + [answer]
        # We want to compute loss on the answer tokens only
        input_embeds = torch.cat([state_tokens, prompt_embeds, answer_embeds], dim=1)

        # Create labels: -100 for state+prompt (ignore), actual IDs for answer
        num_state_tokens = state_tokens.size(1)
        prompt_len = prompt_ids.size(1)
        answer_len = answer_ids.size(1)

        # Labels: [-100, -100, ..., answer_token_1, answer_token_2, ...]
        labels = torch.full(
            (1, num_state_tokens + prompt_len + answer_len),
            -100,
            dtype=torch.long,
            device=device,
        )
        labels[0, num_state_tokens + prompt_len:] = answer_ids[0]

        # Forward pass with labels
        outputs = model.transformer(
            inputs_embeds=input_embeds,
            labels=labels,
            return_dict=True,
        )

        # Get loss for this pass
        pass_loss = outputs.loss

        # Weight increases with pass number (later = better = higher weight)
        # weight = (i + 1) / num_passes
        weight = (i + 1) / num_passes
        weighted_loss = weight * pass_loss

        total_loss = total_loss + weighted_loss
        pass_losses.append(pass_loss.item())

    # Normalize by sum of weights (1 + 2 + ... + n) / n = (n+1)/2 / n
    # Actually, keep as is - the weights already sum to roughly 1
    # sum of (i+1)/n for i in 0..n-1 = sum(1..n)/n = (n+1)/2 / n * n = (n+1)/2
    # So normalize by (n+1)/2
    normalization = (num_passes + 1) / 2
    total_loss = total_loss / normalization

    metrics = {
        "pass_losses": pass_losses,
        "num_passes": num_passes,
        "early_pass_loss": pass_losses[0] if pass_losses else 0.0,
        "late_pass_loss": pass_losses[-1] if pass_losses else 0.0,
    }

    return total_loss, metrics


def compute_confidence_loss(
    confidences: List[float],
    max_passes: int,
) -> torch.Tensor:
    """
    Compute confidence loss to encourage increasing confidence over passes.

    Target: confidence should increase with pass number.
    Target confidence at pass i = (i + 1) / max_passes

    Args:
        confidences: List of confidence values from model.think()
        max_passes: Maximum number of passes (for normalization)

    Returns:
        MSE loss between actual and target confidences
    """
    if not confidences:
        return torch.tensor(0.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_loss = 0.0
    for i, conf in enumerate(confidences):
        # Target confidence increases with pass number
        # Clamp at 0.95 to avoid pushing for 1.0
        target_conf = min((i + 1) / max_passes + 0.1, 0.95)
        loss = (conf - target_conf) ** 2
        total_loss += loss

    return torch.tensor(total_loss / len(confidences), device=device)


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate(
    model: ThinkingModel,
    eval_dataset: GSM8KDataset,
    config: Dict,
    state_scale: float = 1.0,
    num_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate model on GSM8K.

    Computes:
    - thinking_accuracy: Accuracy after N thinking passes
    - single_shot_accuracy: Accuracy without thinking (direct generation)
    - zeros_accuracy: Thinking with zeros state (ablation)
    - avg_passes: Average number of thinking passes
    - avg_confidence: Average final confidence

    Args:
        model: The ThinkingModel to evaluate
        eval_dataset: GSM8K evaluation dataset
        config: Configuration dict
        state_scale: Current state scale
        num_samples: Number of samples to evaluate (defaults to config)

    Returns:
        Dictionary of metrics
    """
    model.eval()

    if num_samples is None:
        num_samples = config["evaluation"]["num_eval_samples"]

    num_samples = min(num_samples, len(eval_dataset))

    max_passes = config["thinking"]["max_passes"]
    confidence_threshold = config["thinking"]["confidence_threshold"]
    max_new_tokens = config["generation"]["max_new_tokens"]

    # Metrics accumulators
    thinking_correct = 0
    single_shot_correct = 0
    zeros_correct = 0
    total_passes = 0
    total_confidence = 0.0

    # Per-pass accuracy tracking
    pass_correct = {i: 0 for i in range(1, max_passes + 1)}
    pass_counts = {i: 0 for i in range(1, max_passes + 1)}

    for idx in tqdm(range(num_samples), desc="Evaluating", leave=False):
        item = eval_dataset[idx]
        question = item["question"]
        gold = item["gold_answer"]

        # 1. Thinking accuracy: use solve()
        try:
            result = model.solve(
                problem_text=question,
                max_passes=max_passes,
                confidence_threshold=confidence_threshold,
                max_new_tokens=max_new_tokens,
            )

            predicted = extract_answer_from_text(result["answer"])
            if check_answer(predicted, gold):
                thinking_correct += 1

            num_passes = result["num_passes"]
            total_passes += num_passes

            if result["confidences"]:
                total_confidence += result["confidences"][-1]

            # Track per-pass accuracy
            if num_passes in pass_correct:
                pass_counts[num_passes] += 1
                if check_answer(predicted, gold):
                    pass_correct[num_passes] += 1

        except Exception as e:
            print(f"Error in thinking evaluation: {e}")

        # 2. Single-shot accuracy: generate without thinking
        try:
            messages = [{"role": "user", "content": question}]
            prompt_text = model.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            inputs = model.tokenizer(
                prompt_text,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                outputs = model.transformer.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=model.tokenizer.pad_token_id,
                )

            single_shot_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted = extract_answer_from_text(single_shot_text)

            if check_answer(predicted, gold):
                single_shot_correct += 1

        except Exception as e:
            print(f"Error in single-shot evaluation: {e}")

        # 3. Zeros ablation: thinking with zeros state
        try:
            zeros_state = model.injector.get_empty_state(
                batch_size=1,
                device=model.device,
            )

            # Generate from zeros state
            answer = model.generate_answer(
                problem_text=question,
                state=zeros_state,
                max_new_tokens=max_new_tokens,
            )

            predicted = extract_answer_from_text(answer)
            if check_answer(predicted, gold):
                zeros_correct += 1

        except Exception as e:
            print(f"Error in zeros evaluation: {e}")

    # Compute metrics
    metrics = {
        "thinking_accuracy": thinking_correct / num_samples if num_samples > 0 else 0.0,
        "single_shot_accuracy": single_shot_correct / num_samples if num_samples > 0 else 0.0,
        "zeros_accuracy": zeros_correct / num_samples if num_samples > 0 else 0.0,
        "avg_passes": total_passes / num_samples if num_samples > 0 else 0.0,
        "avg_confidence": total_confidence / num_samples if num_samples > 0 else 0.0,
        "num_samples": num_samples,
    }

    # Add per-pass accuracy
    for num_pass in pass_correct:
        if pass_counts[num_pass] > 0:
            metrics[f"pass_{num_pass}_accuracy"] = pass_correct[num_pass] / pass_counts[num_pass]
            metrics[f"pass_{num_pass}_count"] = pass_counts[num_pass]

    # Compute ablation delta (the key metric)
    metrics["ablation_delta"] = metrics["thinking_accuracy"] - metrics["zeros_accuracy"]

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

    return checkpoint


# ============================================================================
# Main Training Loop
# ============================================================================

def train_epoch(
    model: ThinkingModel,
    train_dataset: GSM8KDataset,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    config: Dict,
    epoch: int,
    global_step: int,
    state_scale: float,
) -> Tuple[int, Dict[str, float]]:
    """
    Train for one epoch.

    Args:
        model: The ThinkingModel
        train_dataset: Training dataset
        optimizer: AdamW optimizer
        scheduler: Learning rate scheduler
        config: Training configuration
        epoch: Current epoch number
        global_step: Global step counter
        state_scale: Current state scale for warmup

    Returns:
        Tuple of (new_global_step, epoch_metrics)
    """
    model.train()

    batch_size = config["training"]["batch_size"]
    grad_accum_steps = config["training"]["gradient_accumulation_steps"]
    max_grad_norm = config["training"]["max_grad_norm"]
    max_passes = config["thinking"]["max_passes"]
    confidence_threshold = config["thinking"]["confidence_threshold"]
    log_every = config["logging"]["log_every_n_steps"]
    eval_every = config["evaluation"]["eval_every_n_steps"]
    save_every = config["logging"]["save_every_n_steps"]

    # Metrics accumulators
    total_loss = 0.0
    total_answer_loss = 0.0
    total_conf_loss = 0.0
    total_passes = 0
    num_batches = 0

    # Shuffle indices
    indices = torch.randperm(len(train_dataset)).tolist()

    # Progress bar
    pbar = tqdm(range(0, len(indices), batch_size), desc=f"Epoch {epoch}")

    optimizer.zero_grad()

    for batch_idx, start_idx in enumerate(pbar):
        batch_indices = indices[start_idx:start_idx + batch_size]

        batch_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
        batch_answer_loss = 0.0
        batch_conf_loss = 0.0
        batch_passes = 0

        for idx in batch_indices:
            item = train_dataset[idx]
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
                answer_loss, metrics = compute_deep_supervision_loss(
                    model=model,
                    problem_text=question,
                    gold_answer=gold_answer,
                    all_states=all_states,
                    config=config,
                    state_scale=state_scale,
                )

                # Confidence loss
                conf_loss = compute_confidence_loss(confidences, max_passes)
                conf_loss = conf_loss.to(model.device)

                # Total loss
                # Weight confidence loss lower (0.1) as it's auxiliary
                loss = answer_loss + 0.1 * conf_loss

                # Accumulate for gradient accumulation
                batch_loss = batch_loss + loss / len(batch_indices)
                batch_answer_loss += answer_loss.item() / len(batch_indices)
                batch_conf_loss += conf_loss.item() / len(batch_indices)
                batch_passes += metrics["num_passes"] / len(batch_indices)

            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue

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
            total_answer_loss += batch_answer_loss
            total_conf_loss += batch_conf_loss
            total_passes += batch_passes
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{batch_loss.item() * grad_accum_steps:.4f}",
                "passes": f"{batch_passes:.1f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "scale": f"{state_scale:.2f}",
            })

            # Logging
            if global_step % log_every == 0 and WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    "train/loss": batch_loss.item() * grad_accum_steps,
                    "train/answer_loss": batch_answer_loss,
                    "train/confidence_loss": batch_conf_loss,
                    "train/avg_passes": batch_passes,
                    "train/state_scale": state_scale,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/alpha": model.alpha.item(),
                    "train/epoch": epoch,
                    "train/step": global_step,
                })

    # Epoch metrics
    epoch_metrics = {
        "loss": total_loss / num_batches if num_batches > 0 else 0.0,
        "answer_loss": total_answer_loss / num_batches if num_batches > 0 else 0.0,
        "confidence_loss": total_conf_loss / num_batches if num_batches > 0 else 0.0,
        "avg_passes": total_passes / num_batches if num_batches > 0 else 0.0,
        "state_scale": state_scale,
        "alpha": model.alpha.item(),
    }

    return global_step, epoch_metrics


def train(config: Dict, phase: int, resume_path: Optional[str] = None) -> None:
    """
    Main training function.

    Args:
        config: Training configuration
        phase: Training phase (1 = freeze transformer, 2 = end-to-end)
        resume_path: Path to checkpoint to resume from
    """
    # Device setup
    device = torch.device(config["hardware"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    print("\nCreating ThinkingModel...")
    model = ThinkingModel(
        model_name=config["model"]["name"],
        state_size=config["architecture"]["state_size"],
        num_pseudo_tokens=config["architecture"]["num_tokens"],
        num_queries=config["architecture"]["num_queries"],
        num_perceiver_layers=config["architecture"]["num_perceiver_layers"],
        d_perceiver=config["architecture"]["d_perceiver"],
        max_passes=config["architecture"]["max_passes"],
        alpha_init=config["architecture"]["alpha_init"],
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
    phase_key = f"phase{phase}"
    phase_config = config["training"][phase_key]

    if phase_config["freeze_transformer"]:
        model.freeze_transformer()
        print(f"\nPhase {phase}: Transformer FROZEN")
    else:
        model.unfreeze_transformer()
        print(f"\nPhase {phase}: Transformer UNFROZEN (end-to-end)")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # Load datasets
    print("\nLoading GSM8K...")
    train_dataset = GSM8KDataset(split="train")
    eval_dataset = GSM8KDataset(split="test")

    # Optimizer and scheduler
    learning_rate = phase_config["learning_rate"]
    weight_decay = phase_config["weight_decay"]
    num_epochs = phase_config["epochs"]
    warmup_steps = config["training"]["warmup_steps"]

    optimizer = AdamW(
        model.get_trainable_parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Calculate total steps for scheduler
    batch_size = config["training"]["batch_size"]
    grad_accum = config["training"]["gradient_accumulation_steps"]
    steps_per_epoch = len(train_dataset) // (batch_size * grad_accum)
    total_steps = steps_per_epoch * num_epochs

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=learning_rate * 0.1,
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0

    if resume_path and os.path.exists(resume_path):
        checkpoint = load_checkpoint(resume_path, model, optimizer, scheduler)
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("step", 0)
        print(f"Resuming from epoch {start_epoch}, step {global_step}")

    # Initialize wandb
    if WANDB_AVAILABLE:
        try:
            wandb.init(
                project=config["logging"]["project"],
                name=f"{config['logging']['run_name']}-phase{phase}",
                config=config,
            )
        except Exception as e:
            print(f"wandb init failed: {e}")

    # Checkpoint directory
    save_dir = Path(config["checkpoints"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\nStarting training: Phase {phase}, {num_epochs} epochs")
    print(f"  Batch size: {batch_size} x {grad_accum} = {batch_size * grad_accum} effective")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Total steps: {total_steps}")

    best_accuracy = 0.0

    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        # Compute state scale for warmup
        state_scale = compute_state_scale(epoch, config)
        print(f"State scale: {state_scale:.2f}")

        # Train
        global_step, epoch_metrics = train_epoch(
            model=model,
            train_dataset=train_dataset,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            epoch=epoch,
            global_step=global_step,
            state_scale=state_scale,
        )

        print(f"\nEpoch {epoch + 1} training metrics:")
        for name, value in epoch_metrics.items():
            print(f"  {name}: {value:.4f}")

        # Evaluate
        print("\nEvaluating...")
        eval_metrics = evaluate(
            model=model,
            eval_dataset=eval_dataset,
            config=config,
            state_scale=state_scale,
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

        # Save checkpoint
        is_best = eval_metrics["thinking_accuracy"] > best_accuracy
        if is_best:
            best_accuracy = eval_metrics["thinking_accuracy"]
            print(f"\nNew best accuracy: {best_accuracy:.4f}")

        # Save latest
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=global_step,
            config=config,
            metrics=eval_metrics,
            path=str(save_dir / f"checkpoint_phase{phase}_latest.pt"),
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
                path=str(save_dir / f"checkpoint_phase{phase}_best.pt"),
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
                path=str(save_dir / f"checkpoint_phase{phase}_epoch{epoch + 1}.pt"),
            )

    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)

    final_metrics = evaluate(
        model=model,
        eval_dataset=eval_dataset,
        config=config,
        state_scale=1.0,
        num_samples=min(200, len(eval_dataset)),  # More samples for final eval
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
        description="Train ThinkingModel on GSM8K with deep supervision"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/thinking_gsm8k.yaml",
        help="Path to config file",
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

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from: {config_path}")

    # Debug mode adjustments
    if args.debug:
        print("\n*** DEBUG MODE ***")
        config["training"]["phase1"]["epochs"] = 2
        config["training"]["phase2"]["epochs"] = 2
        config["evaluation"]["num_eval_samples"] = 20
        config["logging"]["log_every_n_steps"] = 1
        config["logging"]["save_every_n_steps"] = 100

    # Run training
    train(config, args.phase, args.resume)


if __name__ == "__main__":
    main()
