#!/usr/bin/env python3
"""
Mycelium Phase 2: Joint Constraint Training

Trains the full pipeline with belief propagation:
  - C2/C3 encoders: frozen or very low lr (1e-6)
  - Message gates, C4, messages: normal lr (1e-4)
  - Loss = component_losses + REINFORCE execution reward

Key features:
  - Differential learning rates per component
  - REINFORCE with moving average baseline
  - Gumbel temperature annealing for C4
  - Convergence regularization

Usage:
    python train/train_phase2_joint.py \\
        --c2-checkpoint models/c2_checkpoint.pt \\
        --c3-checkpoint models/c3_checkpoint.pt \\
        --data-path s3://mycelium-data/phase2_train.json \\
        --output-dir outputs/phase2
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel, AutoTokenizer

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.constraint_pipeline import (
    MyceliumConstraintPipeline,
    SpanGroup,
    PipelineOutput,
    count_parameters,
)
from models.c5_sympy_executor import OP_LABELS, N_OPS


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class Phase2Example:
    """Training example for Phase 2."""
    text: str
    gold_ops: List[int]           # operation indices per step
    gold_adjacency: List[List[int]]  # binary adjacency matrix
    gold_answer: float


class Phase2Dataset(Dataset):
    """Dataset for Phase 2 joint training with curriculum and oversampling."""

    def __init__(
        self,
        examples: List[Dict],
        tokenizer,
        max_length: int = 256,
        max_steps: int = None,  # Curriculum: filter by max steps
        oversample_rare_ops: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.op_to_idx = {op: i for i, op in enumerate(OP_LABELS)}

        # Filter by max steps for curriculum learning
        if max_steps is not None:
            examples = [ex for ex in examples if ex.get("n_steps", len(ex.get("span_groups", []))) <= max_steps]

        # Compute operation frequencies for inverse weighting
        op_counts = {op: 0 for op in OP_LABELS}
        for ex in examples:
            for op in ex.get("gold_ops", []):
                if op in op_counts:
                    op_counts[op] += 1

        total_ops = sum(op_counts.values()) + 1e-6
        self.op_weights = {op: total_ops / (count + 1) for op, count in op_counts.items()}

        # Normalize weights (mean=1)
        mean_weight = sum(self.op_weights.values()) / len(self.op_weights)
        self.op_weights = {op: w / mean_weight for op, w in self.op_weights.items()}

        # Oversample examples with rare operations
        if oversample_rare_ops:
            examples = self._oversample_rare(examples)

        self.examples = examples
        print(f"Dataset: {len(self.examples)} examples (max_steps={max_steps})")
        print(f"Op weights (top 5 rare): {sorted(self.op_weights.items(), key=lambda x: -x[1])[:5]}")

    def _oversample_rare(self, examples: List[Dict]) -> List[Dict]:
        """Oversample examples containing rare operations."""
        from collections import Counter
        import random

        # Count examples per dominant operation
        op_example_counts = Counter()
        for ex in examples:
            ops = ex.get("gold_ops", [])
            if ops:
                # Use most rare op in the example
                rarest_op = max(ops, key=lambda o: self.op_weights.get(o, 1))
                op_example_counts[rarest_op] += 1

        # Target: bring each operation to at least median count
        median_count = sorted(op_example_counts.values())[len(op_example_counts) // 2] if op_example_counts else 100

        # Collect examples by their rarest op
        op_examples = {op: [] for op in OP_LABELS}
        for ex in examples:
            ops = ex.get("gold_ops", [])
            if ops:
                rarest_op = max(ops, key=lambda o: self.op_weights.get(o, 1))
                op_examples[rarest_op].append(ex)

        # Oversample
        oversampled = list(examples)
        for op, exs in op_examples.items():
            if exs and len(exs) < median_count:
                n_oversample = min(median_count - len(exs), len(exs) * 2)  # Cap at 3x
                oversampled.extend(random.choices(exs, k=n_oversample))

        random.shuffle(oversampled)
        print(f"Oversampled: {len(examples)} → {len(oversampled)} examples")
        return oversampled

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Parse span groups (for multi-step problems)
        # Format: list of text chunks, each corresponding to one step
        span_texts = ex.get("span_groups", [ex.get("text", "")])
        if isinstance(span_texts, str):
            span_texts = [span_texts]

        # Tokenize each span group
        span_groups = []
        for text in span_texts:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            span_groups.append(SpanGroup(
                input_ids=encoding["input_ids"].squeeze(0),
                attention_mask=encoding["attention_mask"].squeeze(0),
                text=text,
            ))

        # Parse gold labels
        gold_ops = ex.get("gold_ops", [])
        if isinstance(gold_ops, list) and gold_ops and isinstance(gold_ops[0], str):
            gold_ops = [self.op_to_idx.get(op, 0) for op in gold_ops]

        gold_adjacency = ex.get("gold_adjacency", [])
        gold_answer = ex.get("gold_answer", 0.0)

        return {
            "span_groups": span_groups,
            "gold_ops": torch.tensor(gold_ops, dtype=torch.long),
            "gold_adjacency": torch.tensor(gold_adjacency, dtype=torch.float),
            "gold_answer": gold_answer,
        }


def collate_fn(batch):
    """Custom collate - don't stack span groups."""
    return {
        "span_groups": [b["span_groups"] for b in batch],
        "gold_ops": [b["gold_ops"] for b in batch],
        "gold_adjacency": [b["gold_adjacency"] for b in batch],
        "gold_answer": [b["gold_answer"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Phase2Trainer:
    """
    Joint constraint training with belief propagation.

    Key: different learning rates for different components.
    - Pretrained encoders (MiniLM, RoBERTa): frozen or 1e-6
    - New heads (message gates, C4): 1e-4
    - Message networks: 1e-4

    Supports DDP for multi-GPU training.
    """

    def __init__(
        self,
        pipeline: MyceliumConstraintPipeline,
        lr_config: Optional[Dict] = None,
        loss_weights: Optional[Dict] = None,
        device: torch.device = None,
        local_rank: int = -1,
        use_data_parallel: bool = False,
    ):
        self.local_rank = local_rank
        self.is_main = local_rank <= 0  # True for single-GPU or rank 0
        self.use_data_parallel = use_data_parallel

        if local_rank >= 0:
            # DDP mode
            self.device = torch.device(f"cuda:{local_rank}")
            pipeline = pipeline.to(self.device)
            self.pipeline = DDP(pipeline, device_ids=[local_rank], find_unused_parameters=True)
            self.pipeline_unwrapped = pipeline
        elif use_data_parallel:
            # DataParallel mode - simpler multi-GPU
            self.device = torch.device("cuda:0")
            pipeline = pipeline.to(self.device)
            self.pipeline = nn.DataParallel(pipeline)
            self.pipeline_unwrapped = pipeline
        else:
            # Single GPU mode
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pipeline.to(self.device)
            self.pipeline = pipeline
            self.pipeline_unwrapped = pipeline

        # Default lr config
        if lr_config is None:
            lr_config = {
                "encoder": 1e-6,       # MiniLM + RoBERTa encoders
                "heads": 1e-4,         # classification/extraction heads
                "assembler": 1e-4,     # C4 graph assembler
                "messages": 1e-4,      # message networks
            }
        self.lr_config = lr_config

        # Parameter groups
        param_groups = self._build_param_groups()
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

        # Loss weights
        if loss_weights is None:
            loss_weights = {
                "c2_loss": 1.0,
                "c4_loss": 1.0,
                "reinforce_loss": 0.1,     # start low
                "convergence_loss": 0.5,
            }
        self.loss_weights = loss_weights

        # REINFORCE baseline (moving average of rewards)
        self.reward_baseline = 0.5
        self.baseline_momentum = 0.99

        # Training state
        self.global_step = 0
        self.best_loss = float("inf")

    def _build_param_groups(self) -> List[Dict]:
        """Build optimizer parameter groups with different learning rates."""
        groups = []
        p = self.pipeline_unwrapped  # Use unwrapped for param access

        # C2 encoder (low lr or frozen)
        c2_encoder_params = []
        if hasattr(p.c2.c2, 'backbone'):
            c2_encoder_params = list(p.c2.c2.backbone.parameters())
        if c2_encoder_params:
            groups.append({
                "params": c2_encoder_params,
                "lr": self.lr_config["encoder"],
                "name": "c2_encoder",
            })

        # C3 encoder (low lr or frozen)
        c3_encoder_params = []
        if hasattr(p.c3.c3, 'backbone'):
            c3_encoder_params = list(p.c3.c3.backbone.parameters())
        if c3_encoder_params:
            groups.append({
                "params": c3_encoder_params,
                "lr": self.lr_config["encoder"],
                "name": "c3_encoder",
            })

        # C2/C3 heads and message gates (normal lr)
        head_params = []
        if hasattr(p.c2.c2, 'classifier'):
            head_params.extend(p.c2.c2.classifier.parameters())
        head_params.extend(p.c2.msg_gate.parameters())
        head_params.extend(p.c3.msg_gate.parameters())
        head_params.extend(p.c3.op_embedding.parameters())
        if head_params:
            groups.append({
                "params": head_params,
                "lr": self.lr_config["heads"],
                "name": "heads_and_gates",
            })

        # C4 assembler (normal lr)
        groups.append({
            "params": list(p.c4.parameters()),
            "lr": self.lr_config["assembler"],
            "name": "c4_assembler",
        })

        # Message networks (normal lr)
        groups.append({
            "params": list(p.messages.parameters()),
            "lr": self.lr_config["messages"],
            "name": "message_networks",
        })

        return groups

    def train_step(self, batch: Dict) -> Dict:
        """
        One training step.

        Processes one example at a time due to variable span group counts.
        """
        self.pipeline.train()
        self.optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=self.device)
        loss_log = {}
        n_examples = len(batch["span_groups"])

        for i in range(n_examples):
            span_groups = batch["span_groups"][i]
            gold_ops = batch["gold_ops"][i]
            gold_adjacency = batch["gold_adjacency"][i]
            gold_answer = batch["gold_answer"][i]

            # Move to device
            for sg in span_groups:
                sg.input_ids = sg.input_ids.to(self.device)
                sg.attention_mask = sg.attention_mask.to(self.device)

            if gold_ops.numel() > 0:
                gold_ops = gold_ops.to(self.device)
            else:
                gold_ops = None

            if gold_adjacency.numel() > 0:
                gold_adjacency = gold_adjacency.to(self.device)
            else:
                gold_adjacency = None

            # Forward
            output = self.pipeline(
                span_groups=span_groups,
                gold_ops=gold_ops,
                gold_adjacency=gold_adjacency,
                gold_answer=gold_answer,
            )

            # Accumulate weighted losses
            for loss_name, loss_val in output.losses.items():
                weight = self.loss_weights.get(loss_name, 1.0)
                total_loss = total_loss + weight * loss_val / n_examples

                # Log
                if loss_name not in loss_log:
                    loss_log[loss_name] = 0.0
                loss_log[loss_name] += loss_val.item() / n_examples

            # Update REINFORCE baseline
            if output.success:
                correct = abs(output.final_answer - gold_answer) < 1e-6 if output.final_answer else False
                reward = 1.0 if correct else 0.0
                self.reward_baseline = (
                    self.baseline_momentum * self.reward_baseline +
                    (1 - self.baseline_momentum) * reward
                )

        # Backward
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.pipeline.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Anneal Gumbel temperature (use unwrapped for non-DDP methods)
        self.pipeline_unwrapped.c4.discretizer.anneal()

        # Update step count
        self.global_step += 1

        loss_log["total_loss"] = total_loss.item()
        loss_log["gumbel_tau"] = self.pipeline_unwrapped.c4.discretizer.get_tau()
        loss_log["reward_baseline"] = self.reward_baseline

        return loss_log

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> List[Dict]:
        """Full training epoch with belief_shift monitoring."""
        epoch_losses = []

        for batch_idx, batch in enumerate(dataloader):
            loss_log = self.train_step(batch)
            epoch_losses.append(loss_log)

            if batch_idx % 50 == 0 and self.is_main:
                avg = {
                    k: sum(d.get(k, 0) for d in epoch_losses[-50:]) / min(50, len(epoch_losses))
                    for k in loss_log
                }
                print(
                    f"Epoch {epoch} | Step {batch_idx} | "
                    f"Loss: {avg['total_loss']:.4f} | "
                    f"C2: {avg.get('c2_loss', 0):.4f} | "
                    f"C4: {avg.get('c4_loss', 0):.4f} | "
                    f"RL: {avg.get('reinforce_loss', 0):.4f} | "
                    f"Δbelief: {avg.get('belief_shift', 0):.4f} | "
                    f"τ: {avg.get('gumbel_tau', 0):.3f}"
                )

        return epoch_losses

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """Evaluate on validation set."""
        self.pipeline.eval()

        total_correct = 0
        total_examples = 0
        total_exec_success = 0

        for batch in dataloader:
            for i in range(len(batch["span_groups"])):
                span_groups = batch["span_groups"][i]
                gold_answer = batch["gold_answer"][i]

                # Move to device
                for sg in span_groups:
                    sg.input_ids = sg.input_ids.to(self.device)
                    sg.attention_mask = sg.attention_mask.to(self.device)

                # Inference
                output = self.pipeline(span_groups=span_groups)

                total_examples += 1
                if output.success:
                    total_exec_success += 1
                    if output.final_answer is not None:
                        if abs(output.final_answer - gold_answer) < 1e-6:
                            total_correct += 1

        return {
            "accuracy": total_correct / total_examples if total_examples > 0 else 0,
            "exec_success_rate": total_exec_success / total_examples if total_examples > 0 else 0,
            "n_examples": total_examples,
        }

    def save_checkpoint(self, path: str, extra_state: Dict = None):
        """Save training checkpoint (only rank 0 saves)."""
        if not self.is_main:
            return

        state = {
            "pipeline_state_dict": self.pipeline_unwrapped.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "reward_baseline": self.reward_baseline,
            "gumbel_tau": self.pipeline_unwrapped.c4.discretizer.get_tau(),
            "lr_config": self.lr_config,
            "loss_weights": self.loss_weights,
        }
        if extra_state:
            state.update(extra_state)

        torch.save(state, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        state = torch.load(path, map_location=self.device, weights_only=False)

        self.pipeline_unwrapped.load_state_dict(state["pipeline_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.global_step = state.get("global_step", 0)
        self.reward_baseline = state.get("reward_baseline", 0.5)

        if "gumbel_tau" in state:
            self.pipeline_unwrapped.c4.discretizer.tau.fill_(state["gumbel_tau"])

        if self.is_main:
            print(f"Loaded checkpoint from {path} (step {self.global_step})")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(path: str) -> List[Dict]:
    """Load training data from local file or S3."""
    if path.startswith("s3://"):
        import boto3
        import io

        # Parse S3 path
        path_parts = path[5:].split("/", 1)
        bucket = path_parts[0]
        key = path_parts[1]

        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        content = obj["Body"].read().decode("utf-8")
        return json.loads(content)
    else:
        with open(path) as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Joint Training")

    # Model paths
    parser.add_argument("--c2-checkpoint", type=str, required=True,
                        help="Path to pretrained C2 checkpoint")
    parser.add_argument("--c3-checkpoint", type=str, required=True,
                        help="Path to pretrained C3 checkpoint")

    # Data
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to training data (local or s3://)")
    parser.add_argument("--val-path", type=str, default=None,
                        help="Path to validation data")

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/phase2",
                        help="Output directory for checkpoints")

    # Training config
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr-encoder", type=float, default=1e-6)
    parser.add_argument("--lr-heads", type=float, default=1e-4)
    parser.add_argument("--lr-assembler", type=float, default=1e-4)
    parser.add_argument("--lr-messages", type=float, default=1e-4)

    # Loss weights
    parser.add_argument("--weight-c2", type=float, default=1.0)
    parser.add_argument("--weight-c4", type=float, default=1.0)
    parser.add_argument("--weight-reinforce", type=float, default=0.1)
    parser.add_argument("--weight-convergence", type=float, default=0.5)

    # Pipeline config
    parser.add_argument("--n-belief-rounds", type=int, default=3)
    parser.add_argument("--msg-dim", type=int, default=64)

    # Curriculum learning
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable curriculum: epoch 1 ≤8 steps, epoch 2 ≤15, then all")
    parser.add_argument("--curriculum-steps", type=str, default="8,15",
                        help="Comma-separated max steps per curriculum stage")

    # Memory optimization
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing for C3")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")

    # Multi-GPU support
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (set by torchrun)")
    parser.add_argument("--data-parallel", action="store_true",
                        help="Use DataParallel for single-node multi-GPU")

    args = parser.parse_args()

    # Initialize distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    use_ddp = local_rank >= 0
    use_dp = args.data_parallel and torch.cuda.device_count() > 1

    if use_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        is_main = local_rank == 0
    else:
        is_main = True
        if use_dp:
            print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")

    # Create output dir
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer (use C2's tokenizer for simplicity)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Load pretrained C2
    if is_main:
        print(f"Loading C2 from {args.c2_checkpoint}...")
    c2_state = torch.load(args.c2_checkpoint, map_location="cpu", weights_only=False)

    # Reconstruct C2 model
    from models.train_c2 import C2Model
    c2_model = C2Model(
        backbone_name="sentence-transformers/all-MiniLM-L6-v2",
        num_labels=N_OPS,
    )

    # Handle op count mismatch (old checkpoint has fewer ops)
    ckpt_state = c2_state["model_state_dict"]
    ckpt_num_ops = ckpt_state["classifier.4.weight"].shape[0]
    if ckpt_num_ops != N_OPS:
        if is_main:
            print(f"  Adapting C2 checkpoint: {ckpt_num_ops} ops -> {N_OPS} ops")
        # Copy existing op weights, leave new ops at random init
        new_state = c2_model.state_dict()
        for k, v in ckpt_state.items():
            if k in new_state:
                if v.shape == new_state[k].shape:
                    new_state[k] = v
                elif "classifier.4" in k:
                    # Partial copy for classifier head
                    if len(v.shape) == 2:  # weight
                        new_state[k][:ckpt_num_ops] = v
                    else:  # bias
                        new_state[k][:ckpt_num_ops] = v
                else:
                    new_state[k] = v
        c2_model.load_state_dict(new_state)
    else:
        c2_model.load_state_dict(ckpt_state)

    # Load pretrained C3
    from models.c3_extractor import C3SpanExtractor

    if args.c3_checkpoint and os.path.exists(args.c3_checkpoint):
        if is_main:
            print(f"Loading C3 from {args.c3_checkpoint}...")
        c3_state = torch.load(args.c3_checkpoint, map_location="cpu", weights_only=False)
        ckpt_state = c3_state["model_state_dict"]

        # Check if checkpoint architecture matches
        # Our C3 uses Qwen2 backbone.layers, old C3 used roberta encoder.layer
        has_qwen_keys = any("backbone.layers" in k for k in ckpt_state.keys())

        if has_qwen_keys:
            c3_model = C3SpanExtractor()
            c3_model.load_state_dict(ckpt_state)
            if is_main:
                print("  C3 checkpoint loaded successfully")
        else:
            if is_main:
                print("  C3 checkpoint architecture mismatch (RoBERTa vs Qwen2)")
                print("  Initializing C3 fresh from Qwen2-0.5B pretrained weights")
            c3_model = C3SpanExtractor()  # Fresh init from HuggingFace
    else:
        if is_main:
            print("No C3 checkpoint found, initializing fresh from Qwen2-0.5B")
        c3_model = C3SpanExtractor()

    # Create pipeline
    if is_main:
        print("Creating constraint pipeline...")
    pipeline = MyceliumConstraintPipeline(
        c2_model=c2_model,
        c3_model=c3_model,
        n_ops=N_OPS,
        msg_dim=args.msg_dim,
        n_belief_rounds=args.n_belief_rounds,
    )

    # Print parameter counts
    if is_main:
        counts = count_parameters(pipeline)
        print("\nParameter counts:")
        for name, count in counts.items():
            print(f"  {name}: {count:,}")

    # Create trainer
    lr_config = {
        "encoder": args.lr_encoder,
        "heads": args.lr_heads,
        "assembler": args.lr_assembler,
        "messages": args.lr_messages,
    }
    loss_weights = {
        "c2_loss": args.weight_c2,
        "c4_loss": args.weight_c4,
        "reinforce_loss": args.weight_reinforce,
        "convergence_loss": args.weight_convergence,
    }

    trainer = Phase2Trainer(
        pipeline=pipeline,
        lr_config=lr_config,
        loss_weights=loss_weights,
        local_rank=local_rank,
        use_data_parallel=use_dp,
    )

    # Resume if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        p = trainer.pipeline_unwrapped
        if hasattr(p.c3.c3, 'backbone') and hasattr(p.c3.c3.backbone, 'gradient_checkpointing_enable'):
            p.c3.c3.backbone.gradient_checkpointing_enable()
            if is_main:
                print("Enabled gradient checkpointing for C3")

    # Load data
    if is_main:
        print(f"\nLoading training data from {args.data_path}...")
    train_data = load_training_data(args.data_path)
    if is_main:
        print(f"Loaded {len(train_data)} training examples")

    # Curriculum stages
    if args.curriculum:
        curriculum_steps = [int(s) for s in args.curriculum_steps.split(",")]
        curriculum_steps.append(None)  # Final stage: all steps
        if is_main:
            print(f"Curriculum learning enabled: stages {curriculum_steps}")
    else:
        curriculum_steps = [None]  # No curriculum

    # Store op_weights from first dataset for REINFORCE weighting
    op_weights = None

    # Validation data
    val_loader = None
    if args.val_path:
        val_data = load_training_data(args.val_path)
        val_dataset = Phase2Dataset(val_data, tokenizer, max_steps=None)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    # Track belief_shift history for monitoring
    belief_shift_history = []

    # Training loop with curriculum
    if is_main:
        print("\nStarting training...")
    for epoch in range(args.epochs):
        # Determine curriculum stage
        if args.curriculum:
            stage_idx = min(epoch, len(curriculum_steps) - 1)
            max_steps = curriculum_steps[stage_idx]
            stage_name = f"≤{max_steps} steps" if max_steps else "all steps"
        else:
            max_steps = None
            stage_name = "all steps"

        if is_main:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{args.epochs} (Curriculum: {stage_name})")
            print(f"{'='*60}")

        # Create dataset for this curriculum stage
        train_dataset = Phase2Dataset(
            train_data, tokenizer,
            max_steps=max_steps,
            oversample_rare_ops=True,
        )

        # Store op_weights for potential REINFORCE weighting
        if op_weights is None:
            op_weights = train_dataset.op_weights
            trainer.op_weights = op_weights  # Pass to trainer

        # Use DistributedSampler for DDP
        if local_rank >= 0:
            sampler = DistributedSampler(train_dataset, shuffle=True)
            sampler.set_epoch(epoch)  # For proper shuffling
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                collate_fn=collate_fn,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
            )

        # Train
        epoch_losses = trainer.train_epoch(train_loader, epoch)

        # Summarize epoch
        avg_loss = sum(d["total_loss"] for d in epoch_losses) / len(epoch_losses)
        avg_belief_shift = sum(d.get("belief_shift", 0) for d in epoch_losses) / len(epoch_losses)
        belief_shift_history.append(avg_belief_shift)

        if is_main:
            print(f"\nEpoch {epoch + 1} complete:")
            print(f"  Avg loss: {avg_loss:.4f}")
            print(f"  Avg belief_shift: {avg_belief_shift:.4f}")

            # Monitor belief_shift trend
            if len(belief_shift_history) >= 3:
                recent = belief_shift_history[-3:]
                if all(recent[i] > recent[i-1] for i in range(1, len(recent))):
                    print("  [!] belief_shift increasing - forward pass may be degrading")
                elif all(recent[i] < recent[i-1] for i in range(1, len(recent))):
                    print("  [ok] belief_shift decreasing - beliefs converging on round 0")

        # Evaluate
        if val_loader and is_main:
            metrics = trainer.evaluate(val_loader)
            print(f"  Validation: acc={metrics['accuracy']:.4f}, "
                  f"exec_success={metrics['exec_success_rate']:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch + 1}.pt")
        trainer.save_checkpoint(ckpt_path, {
            "epoch": epoch + 1,
            "curriculum_stage": stage_name,
            "belief_shift": avg_belief_shift,
        })

    # Save final model
    final_path = os.path.join(args.output_dir, "final_model.pt")
    trainer.save_checkpoint(final_path, {"epoch": args.epochs})
    if is_main:
        print(f"\nTraining complete. Final model saved to {final_path}")

    # Cleanup DDP
    if local_rank >= 0:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
