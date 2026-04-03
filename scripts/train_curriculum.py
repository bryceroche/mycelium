#!/usr/bin/env python3
"""
Curriculum training for Mycelium v18.

Starts with VERY easy problems (single-step arithmetic) and advances
through levels based on accuracy thresholds.

Usage:
    python scripts/train_curriculum.py --config configs/curriculum.yaml --phase 1
    python scripts/train_curriculum.py --config configs/curriculum.yaml --phase 2 --resume checkpoint.pt
"""

import os
import sys
import json
import yaml
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.thinking_model import ThinkingModel


@dataclass
class CurriculumLevel:
    """Configuration for a curriculum level."""
    level: int
    name: str
    data_file: str
    max_passes: int
    accuracy_threshold: float
    min_epochs: int


class CurriculumDataset(Dataset):
    """Dataset for a single curriculum level."""

    def __init__(self, data_file: str, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(data_file, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)

        print(f"Loaded {len(self.data)} problems from {data_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "question": item["question"],
            "answer": item["answer"],
            "solution": item.get("solution", item["answer"]),
            "level": item.get("level", 0),
            "num_steps": item.get("num_steps", 1),
        }


def extract_answer(text: str) -> Optional[str]:
    """Extract numerical answer from model output."""
    import re

    # Try \boxed{} first
    boxed = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed.group(1).strip()

    # Try #### format
    hash_match = re.search(r'####\s*(\d+)', text)
    if hash_match:
        return hash_match.group(1).strip()

    # Try "answer is X" or "= X" at end
    answer_match = re.search(r'(?:answer is|=)\s*(\d+)', text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()

    # Try to find last number
    numbers = re.findall(r'\d+', text)
    if numbers:
        return numbers[-1]

    return None


def check_answer(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold."""
    if predicted is None:
        return False

    try:
        pred_num = float(predicted.replace(',', ''))
        gold_num = float(gold.replace(',', ''))
        return abs(pred_num - gold_num) < 0.01
    except (ValueError, TypeError):
        return predicted.strip() == gold.strip()


class CurriculumTrainer:
    """Trainer with curriculum learning."""

    def __init__(
        self,
        config: Dict,
        model: ThinkingModel,
        phase: int = 1,
        device: str = "cuda",
    ):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.phase = phase

        # Parse curriculum levels
        self.levels = self._parse_levels()
        self.current_level = config["curriculum"]["start_level"]
        self.epochs_at_level = 0
        self.consecutive_passes = 0

        # Safety limits
        self.max_passes_limit = config["safety"]["max_passes_hard_limit"]
        self.timeout_seconds = config["safety"]["timeout_seconds"]

        # Set up optimizer
        self._setup_training()

        # Tracking
        self.global_step = 0
        self.best_accuracy = 0.0

        # Logging
        self.wandb_run = None
        self._setup_logging()

    def _parse_levels(self) -> Dict[int, CurriculumLevel]:
        """Parse curriculum levels from config."""
        levels = {}
        for level_id, level_config in self.config["curriculum"]["levels"].items():
            if level_config.get("data_file"):  # Skip levels without data
                levels[int(level_id)] = CurriculumLevel(
                    level=int(level_id),
                    name=level_config["name"],
                    data_file=level_config["data_file"],
                    max_passes=min(level_config["max_passes"], self.config["safety"]["max_passes_hard_limit"]),
                    accuracy_threshold=level_config["accuracy_threshold"],
                    min_epochs=level_config["min_epochs"],
                )
        return levels

    def _setup_training(self):
        """Set up optimizer and scheduler."""
        phase_config = self.config["training"][f"phase{self.phase}"]

        # Freeze/unfreeze transformer
        if phase_config["freeze_transformer"]:
            self.model.freeze_transformer()
            print("Phase 1: Transformer FROZEN")
        else:
            self.model.unfreeze_transformer()
            print("Phase 2: Transformer UNFROZEN")

        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=phase_config["learning_rate"],
            weight_decay=phase_config["weight_decay"],
        )

    def _setup_logging(self):
        """Set up wandb logging."""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.config["logging"]["project"],
                name=f"curriculum_phase{self.phase}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                config=self.config,
            )
            print("Wandb logging enabled")
        except Exception as e:
            print(f"Wandb not available: {e}")
            self.wandb_run = None

    def get_state_scale(self, epoch: int) -> float:
        """Get state scale for current epoch (warmup)."""
        warmup_config = self.config["training"]["state_scale_warmup"]
        if not warmup_config["enabled"]:
            return 1.0

        warmup_epochs = warmup_config["warmup_epochs"]
        start_scale = warmup_config["start_scale"]
        end_scale = warmup_config["end_scale"]

        if epoch >= warmup_epochs:
            return end_scale

        progress = epoch / warmup_epochs
        return start_scale + (end_scale - start_scale) * progress

    def train_epoch(
        self,
        dataloader: DataLoader,
        level: CurriculumLevel,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch at current level."""
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_passes = 0

        state_scale = self.get_state_scale(self.epochs_at_level)
        print(f"State scale: {state_scale:.2f}")

        pbar = tqdm(dataloader, desc=f"Level {level.level} Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()

            batch_loss = 0.0
            batch_correct = 0

            for i in range(len(batch["question"])):
                question = batch["question"][i]
                gold_answer = batch["answer"][i]

                try:
                    # Think with timeout protection
                    start_time = time.time()

                    # SAFETY: Hard limit on passes
                    max_passes = min(level.max_passes, self.max_passes_limit)

                    state, all_states, confidences = self.model.think(
                        question,
                        max_passes=max_passes,
                        confidence_threshold=self.config["thinking"]["confidence_threshold"],
                    )

                    elapsed = time.time() - start_time
                    if elapsed > self.timeout_seconds:
                        print(f"WARNING: Problem took {elapsed:.1f}s (limit: {self.timeout_seconds}s)")

                    total_passes += len(confidences)

                    # Deep supervision: loss at each pass
                    if self.config["training"]["deep_supervision"]["enabled"]:
                        for pass_idx, intermediate_state in enumerate(all_states[1:]):
                            # Scale state
                            scaled_state = intermediate_state * state_scale

                            # Get logits for gold answer
                            logits = self.model.forward_with_state(
                                question,
                                scaled_state,
                                gold_answer=batch["solution"][i],
                            )

                            # Compute loss
                            pass_loss = F.cross_entropy(
                                logits.view(-1, logits.size(-1)),
                                self.model.tokenizer.encode(
                                    batch["solution"][i],
                                    return_tensors="pt"
                                ).to(self.device).view(-1),
                                ignore_index=-100,
                            )

                            # Weight by pass number (later = higher weight)
                            weight = (pass_idx + 1) / len(all_states[1:])
                            batch_loss += weight * pass_loss
                    else:
                        # Just final state loss
                        scaled_state = state * state_scale
                        logits = self.model.forward_with_state(
                            question,
                            scaled_state,
                            gold_answer=batch["solution"][i],
                        )
                        batch_loss += F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            self.model.tokenizer.encode(
                                batch["solution"][i],
                                return_tensors="pt"
                            ).to(self.device).view(-1),
                            ignore_index=-100,
                        )

                    # Check if answer is correct (for monitoring)
                    with torch.no_grad():
                        generated = self.model.generate_answer(question, state * state_scale)
                        pred_answer = extract_answer(generated)
                        if check_answer(pred_answer, gold_answer):
                            batch_correct += 1

                except Exception as e:
                    print(f"Error on problem: {e}")
                    continue

                total_samples += 1

            if total_samples == 0:
                continue

            # Average loss
            batch_loss = batch_loss / len(batch["question"])

            # Backward
            batch_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["training"]["max_grad_norm"],
            )

            self.optimizer.step()
            self.global_step += 1

            total_loss += batch_loss.item()
            total_correct += batch_correct

            # Update progress bar
            acc = total_correct / total_samples if total_samples > 0 else 0
            avg_passes = total_passes / total_samples if total_samples > 0 else 0
            pbar.set_postfix({
                "loss": f"{batch_loss.item():.4f}",
                "acc": f"{acc:.2%}",
                "passes": f"{avg_passes:.1f}",
            })

            # Log to wandb
            if self.wandb_run and self.global_step % 10 == 0:
                self.wandb_run.log({
                    "train/loss": batch_loss.item(),
                    "train/accuracy": acc,
                    "train/avg_passes": avg_passes,
                    "train/level": level.level,
                    "train/state_scale": state_scale,
                    "train/step": self.global_step,
                })

        return {
            "loss": total_loss / len(dataloader) if len(dataloader) > 0 else 0,
            "accuracy": total_correct / total_samples if total_samples > 0 else 0,
            "avg_passes": total_passes / total_samples if total_samples > 0 else 0,
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, level: CurriculumLevel) -> Dict[str, float]:
        """Evaluate on current level."""
        self.model.eval()

        total_correct = 0
        zeros_correct = 0
        total_samples = 0
        total_passes = 0
        pass_accuracies = {}

        max_passes = min(level.max_passes, self.max_passes_limit)

        for batch in tqdm(dataloader, desc="Evaluating"):
            for i in range(len(batch["question"])):
                question = batch["question"][i]
                gold_answer = batch["answer"][i]

                try:
                    # Thinking accuracy
                    state, all_states, confidences = self.model.think(
                        question,
                        max_passes=max_passes,
                    )
                    generated = self.model.generate_answer(question, state)
                    pred_answer = extract_answer(generated)

                    if check_answer(pred_answer, gold_answer):
                        total_correct += 1

                    total_passes += len(confidences)

                    # Track accuracy by pass count
                    n_passes = len(confidences)
                    if n_passes not in pass_accuracies:
                        pass_accuracies[n_passes] = {"correct": 0, "total": 0}
                    pass_accuracies[n_passes]["total"] += 1
                    if check_answer(pred_answer, gold_answer):
                        pass_accuracies[n_passes]["correct"] += 1

                    # Zeros ablation
                    zeros_state = torch.zeros_like(state)
                    zeros_generated = self.model.generate_answer(question, zeros_state)
                    zeros_pred = extract_answer(zeros_generated)
                    if check_answer(zeros_pred, gold_answer):
                        zeros_correct += 1

                except Exception as e:
                    print(f"Eval error: {e}")

                total_samples += 1

        thinking_acc = total_correct / total_samples if total_samples > 0 else 0
        zeros_acc = zeros_correct / total_samples if total_samples > 0 else 0

        results = {
            "thinking_accuracy": thinking_acc,
            "zeros_accuracy": zeros_acc,
            "ablation_delta": thinking_acc - zeros_acc,
            "avg_passes": total_passes / total_samples if total_samples > 0 else 0,
            "num_samples": total_samples,
        }

        # Add per-pass accuracy
        for n_passes, data in sorted(pass_accuracies.items()):
            if data["total"] > 0:
                results[f"pass_{n_passes}_accuracy"] = data["correct"] / data["total"]

        return results

    def should_advance_level(self, eval_results: Dict[str, float]) -> bool:
        """Check if we should advance to next level."""
        level = self.levels[self.current_level]

        # Must complete minimum epochs
        if self.epochs_at_level < level.min_epochs:
            return False

        # Must meet accuracy threshold
        if eval_results["thinking_accuracy"] >= level.accuracy_threshold:
            self.consecutive_passes += 1
            print(f"Passed threshold ({self.consecutive_passes}/{self.config['curriculum']['advancement']['require_consecutive_passes']})")
        else:
            self.consecutive_passes = 0

        # Require consecutive passes
        required = self.config["curriculum"]["advancement"]["require_consecutive_passes"]
        return self.consecutive_passes >= required

    def advance_level(self):
        """Advance to next curriculum level."""
        next_level = self.current_level + 1

        if next_level not in self.levels:
            print(f"Completed all curriculum levels!")
            return False

        print(f"\n{'='*60}")
        print(f"ADVANCING: Level {self.current_level} -> Level {next_level}")
        print(f"{'='*60}\n")

        # Save checkpoint at level transition
        self.save_checkpoint(f"level_{self.current_level}_complete.pt")

        self.current_level = next_level
        self.epochs_at_level = 0
        self.consecutive_passes = 0

        return True

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        save_dir = Path(self.config["checkpoints"]["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_level": self.current_level,
            "epochs_at_level": self.epochs_at_level,
            "global_step": self.global_step,
            "best_accuracy": self.best_accuracy,
            "config": self.config,
        }

        torch.save(checkpoint, save_dir / filename)
        print(f"Saved checkpoint: {save_dir / filename}")

    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_level = checkpoint["current_level"]
        self.epochs_at_level = checkpoint["epochs_at_level"]
        self.global_step = checkpoint["global_step"]
        self.best_accuracy = checkpoint.get("best_accuracy", 0.0)

        print(f"Loaded checkpoint from {filepath}")
        print(f"  Level: {self.current_level}, Epoch: {self.epochs_at_level}, Step: {self.global_step}")

    def train(self):
        """Main training loop with curriculum advancement."""
        phase_config = self.config["training"][f"phase{self.phase}"]
        max_epochs_per_level = phase_config["epochs_per_level"]

        print(f"\n{'='*60}")
        print(f"CURRICULUM TRAINING - Phase {self.phase}")
        print(f"Starting at level {self.current_level}: {self.levels[self.current_level].name}")
        print(f"{'='*60}\n")

        while self.current_level in self.levels:
            level = self.levels[self.current_level]

            # Load data for current level
            dataset = CurriculumDataset(
                level.data_file,
                self.model.tokenizer,
                self.config["data"]["max_seq_length"],
            )

            # Split into train/eval
            train_size = int(0.9 * len(dataset))
            eval_size = len(dataset) - train_size
            train_dataset, eval_dataset = torch.utils.data.random_split(
                dataset, [train_size, eval_size]
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config["training"]["batch_size"],
                shuffle=True,
                collate_fn=lambda x: {k: [d[k] for d in x] for k in x[0].keys()},
            )

            eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.config["evaluation"]["eval_batch_size"],
                shuffle=False,
                collate_fn=lambda x: {k: [d[k] for d in x] for k in x[0].keys()},
            )

            # Train epochs at this level
            for epoch in range(self.epochs_at_level, max_epochs_per_level):
                print(f"\n--- Level {level.level} ({level.name}) - Epoch {epoch + 1}/{max_epochs_per_level} ---")

                # Train
                train_metrics = self.train_epoch(train_loader, level, epoch)
                print(f"Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.2%}, passes={train_metrics['avg_passes']:.1f}")

                # Evaluate
                eval_metrics = self.evaluate(eval_loader, level)
                print(f"Eval: thinking={eval_metrics['thinking_accuracy']:.2%}, zeros={eval_metrics['zeros_accuracy']:.2%}, delta={eval_metrics['ablation_delta']:.2%}")

                self.epochs_at_level = epoch + 1

                # Log to wandb
                if self.wandb_run:
                    self.wandb_run.log({
                        "eval/thinking_accuracy": eval_metrics["thinking_accuracy"],
                        "eval/zeros_accuracy": eval_metrics["zeros_accuracy"],
                        "eval/ablation_delta": eval_metrics["ablation_delta"],
                        "eval/avg_passes": eval_metrics["avg_passes"],
                        "level": level.level,
                        "epoch": self.epochs_at_level,
                    })

                # Check if we should advance
                if self.should_advance_level(eval_metrics):
                    if not self.advance_level():
                        print("Training complete - all levels finished!")
                        return
                    break  # Start fresh at new level

                # Save periodic checkpoint
                if self.epochs_at_level % 3 == 0:
                    self.save_checkpoint(f"level_{level.level}_epoch_{self.epochs_at_level}.pt")

            # If we hit max epochs without advancing, still try next level
            if self.epochs_at_level >= max_epochs_per_level:
                print(f"Max epochs reached at level {level.level}, advancing anyway...")
                if not self.advance_level():
                    break

        print("\n" + "="*60)
        print("CURRICULUM TRAINING COMPLETE")
        print("="*60)
        self.save_checkpoint("curriculum_final.pt")


def main():
    parser = argparse.ArgumentParser(description="Curriculum training for Mycelium v18")
    parser.add_argument("--config", type=str, default="configs/curriculum.yaml")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--level", type=int, default=None, help="Override starting level")
    parser.add_argument("--debug", action="store_true", help="Debug mode (small data)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model
    print("Loading model...")
    model = ThinkingModel(
        model_name=config["model"]["name"],
        state_size=config["architecture"]["state_size"],
        num_tokens=config["architecture"]["num_tokens"],
        num_queries=config["architecture"]["num_queries"],
        num_perceiver_layers=config["architecture"]["num_perceiver_layers"],
        max_passes=config["architecture"]["max_passes"],
    )

    # Create trainer
    trainer = CurriculumTrainer(
        config=config,
        model=model,
        phase=args.phase,
        device=device,
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Override level if specified
    if args.level is not None:
        trainer.current_level = args.level
        trainer.epochs_at_level = 0
        print(f"Overriding start level to {args.level}")

    # Generate curriculum data if needed
    data_dir = Path(config["data"]["curriculum_dir"])
    if not data_dir.exists() or not list(data_dir.glob("*.jsonl")):
        print("Generating curriculum data...")
        from generate_curriculum import generate_all_levels
        generate_all_levels(
            samples_per_level=1000 if not args.debug else 100,
            output_dir=str(data_dir),
        )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
