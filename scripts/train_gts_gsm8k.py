#!/usr/bin/env python3
"""Train GTS model on GSM8K training data.

Usage:
    uv run python scripts/train_gts_gsm8k.py

This script trains a Goal-driven Tree-Structured (GTS) model on GSM8K data
that was converted to prefix notation format.
"""

import sys
sys.path.insert(0, "src")

import json
import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from mycelium.gts_model import (
    GTSModel, GTSConfig, GTSEmbedder, GTSEncoder,
    GTSDecoder, Merge, NodeGenerator
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    train_path: str = "data/gsm8k_gts/train.jsonl"
    test_path: str = "data/gsm8k_gts/test.jsonl"
    output_dir: str = "trained_model/GTS-gsm8k"

    # Model
    embedding_size: int = 128
    hidden_size: int = 512
    num_layers: int = 2
    dropout_ratio: float = 0.5

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 50
    patience: int = 5
    grad_clip: float = 5.0

    # Vocab
    min_word_freq: int = 2
    max_vocab_size: int = 10000


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_input_vocab(data: list[dict], config: TrainingConfig) -> dict:
    """Build input vocabulary from normalized questions."""
    word_counts = Counter()

    for item in data:
        # Tokenize normalized question
        tokens = item["normalized_question"].lower().split()
        word_counts.update(tokens)

    # Special tokens
    vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>", "NUM"]

    # Add frequent words
    for word, count in word_counts.most_common(config.max_vocab_size):
        if count >= config.min_word_freq and word not in vocab:
            # Skip NUM_X tokens (they become NUM)
            if not word.startswith("num_"):
                vocab.append(word)

    return {"in_idx2word": vocab}


def build_output_vocab(data: list[dict]) -> dict:
    """Build output vocabulary from prefix expressions."""
    # Operators first (in order)
    operators = ["+", "-", "*", "/"]

    # Count all symbols
    symbol_counts = Counter()
    for item in data:
        for prefix in item["prefix_steps"]:
            tokens = prefix.split()
            symbol_counts.update(tokens)

    # Find max NUM_X index
    max_num = 0
    max_const = 0
    max_step = 0
    for symbol in symbol_counts:
        if symbol.startswith("NUM_"):
            idx = int(symbol.split("_")[1])
            max_num = max(max_num, idx)
        elif symbol.startswith("CONST_"):
            idx = int(symbol.split("_")[1])
            max_const = max(max_const, idx)
        elif symbol.startswith("step_"):
            idx = int(symbol.split("_")[1])
            max_step = max(max_step, idx)

    # Build output vocab
    out_symbols = operators.copy()

    # Add NUM_X tokens
    for i in range(max_num + 1):
        out_symbols.append(f"NUM_{i}")

    # Add CONST_X tokens
    for i in range(max_const + 1):
        out_symbols.append(f"CONST_{i}")

    # Add step_X tokens for chaining
    for i in range(1, max_step + 1):
        out_symbols.append(f"step_{i}")

    out_symbols.append("<UNK>")

    # Temp symbols (for generated constants - placeholders)
    temp_symbols = ["<OPT>"] + out_symbols[len(operators):]

    return {
        "out_idx2symbol": out_symbols,
        "temp_idx2symbol": temp_symbols,
    }


class GSM8KDataset(Dataset):
    """Dataset for GSM8K GTS training."""

    def __init__(
        self,
        data: list[dict],
        input_vocab: dict,
        output_vocab: dict,
    ):
        self.data = data
        self.word2idx = {w: i for i, w in enumerate(input_vocab["in_idx2word"])}
        self.symbol2idx = {s: i for i, s in enumerate(output_vocab["out_idx2symbol"])}
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
        self.num_idx = 4  # Generic NUM token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> dict:
        item = self.data[idx]

        # Tokenize input
        input_tokens = self._tokenize_input(item["normalized_question"])

        # Tokenize all prefix steps (we train on final prefix)
        target_tokens = self._tokenize_output(item["final_prefix"])

        return {
            "input_ids": torch.tensor(input_tokens, dtype=torch.long),
            "target_ids": torch.tensor(target_tokens, dtype=torch.long),
            "input_len": len(input_tokens),
            "target_len": len(target_tokens),
            "num_count": len([t for t in input_tokens if t == self.num_idx]),
        }

    def _tokenize_input(self, text: str) -> list[int]:
        """Tokenize input text."""
        tokens = []
        for word in text.lower().split():
            if word.startswith("num_") or word.startswith("const_"):
                # Replace all NUM_X with generic NUM token
                tokens.append(self.num_idx)
            elif word in self.word2idx:
                tokens.append(self.word2idx[word])
            else:
                tokens.append(self.unk_idx)
        return tokens

    def _tokenize_output(self, prefix: str) -> list[int]:
        """Tokenize prefix expression."""
        tokens = []
        for symbol in prefix.split():
            if symbol in self.symbol2idx:
                tokens.append(self.symbol2idx[symbol])
            else:
                tokens.append(self.symbol2idx.get("<UNK>", 0))
        return tokens


def collate_fn(batch: list[dict]) -> dict:
    """Collate batch with padding."""
    input_ids = [item["input_ids"] for item in batch]
    target_ids = [item["target_ids"] for item in batch]
    input_lens = torch.tensor([item["input_len"] for item in batch])
    target_lens = torch.tensor([item["target_len"] for item in batch])
    num_counts = torch.tensor([item["num_count"] for item in batch])

    # Pad sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_ids = pad_sequence(target_ids, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "input_lens": input_lens,
        "target_lens": target_lens,
        "num_counts": num_counts,
    }


class GTSTrainer:
    """Trainer for GTS model."""

    def __init__(
        self,
        model: GTSModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Track best model
        self.best_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            input_lens = batch["input_lens"].to(self.device)

            # Forward pass through encoder
            embedded = self.model.embedder(input_ids)
            encoder_outputs, hidden = self.model.encoder(embedded, input_lens)

            # Get number embeddings
            num_positions = self._get_num_positions(input_ids)
            num_embeddings = self._get_num_embeddings(encoder_outputs, num_positions)

            # Attention mask
            mask = (input_ids != 0).float()

            # Teacher forcing: predict each target token
            batch_size = input_ids.size(0)
            hidden_size = self.config.hidden_size

            # Initial goal from encoder hidden
            goal = hidden[:, :hidden_size]

            # Compute loss over target sequence
            loss = 0.0
            for t in range(target_ids.size(1)):
                # Get context via attention
                context_full = self.model.decoder.attn(goal, encoder_outputs, mask)
                context = context_full[:, :hidden_size]

                # Predict
                op_scores, num_scores = self.model.decoder(
                    goal, context, num_embeddings, mask
                )

                # Combined scores
                all_scores = torch.cat([op_scores, num_scores], dim=1)

                # Target for this step
                target = target_ids[:, t]

                # Cross-entropy loss
                step_loss = self.criterion(all_scores, target)
                loss = loss + step_loss

                # Teacher forcing: use true target for next step
                # Update goal based on target (simplified)
                if t + 1 < target_ids.size(1):
                    # Use node generator to create next goal
                    target_op = target.clamp(0, 5)  # Operators are 0-3
                    left_goal, right_goal = self.model.node_generater(
                        goal, context, target_op
                    )
                    # Alternate between left and right (simplified)
                    goal = left_goal if t % 2 == 0 else right_goal

            # Average loss
            loss = loss / target_ids.size(1)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self) -> float:
        """Validate on held-out data."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                input_lens = batch["input_lens"].to(self.device)

                embedded = self.model.embedder(input_ids)
                encoder_outputs, hidden = self.model.encoder(embedded, input_lens)

                num_positions = self._get_num_positions(input_ids)
                num_embeddings = self._get_num_embeddings(encoder_outputs, num_positions)
                mask = (input_ids != 0).float()

                hidden_size = self.config.hidden_size
                goal = hidden[:, :hidden_size]

                loss = 0.0
                for t in range(target_ids.size(1)):
                    context_full = self.model.decoder.attn(goal, encoder_outputs, mask)
                    context = context_full[:, :hidden_size]

                    op_scores, num_scores = self.model.decoder(
                        goal, context, num_embeddings, mask
                    )
                    all_scores = torch.cat([op_scores, num_scores], dim=1)
                    target = target_ids[:, t]

                    step_loss = self.criterion(all_scores, target)
                    loss = loss + step_loss

                    if t + 1 < target_ids.size(1):
                        target_op = target.clamp(0, 5)
                        left_goal, right_goal = self.model.node_generater(
                            goal, context, target_op
                        )
                        goal = left_goal if t % 2 == 0 else right_goal

                loss = loss / target_ids.size(1)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def _get_num_positions(self, input_ids: torch.Tensor) -> list[list[int]]:
        """Get positions of NUM tokens for each batch item."""
        batch_positions = []
        num_idx = 4  # NUM token index

        for i in range(input_ids.size(0)):
            positions = (input_ids[i] == num_idx).nonzero(as_tuple=True)[0].tolist()
            batch_positions.append(positions)

        return batch_positions

    def _get_num_embeddings(
        self,
        encoder_outputs: torch.Tensor,
        num_positions: list[list[int]],
    ) -> torch.Tensor:
        """Extract number embeddings from encoder outputs."""
        batch_size = encoder_outputs.size(0)
        hidden_size = self.config.hidden_size

        # Find max number of positions
        max_nums = max(len(pos) for pos in num_positions) if num_positions else 1
        max_nums = max(max_nums, 1)

        # Initialize
        num_embeddings = torch.zeros(batch_size, max_nums, hidden_size, device=self.device)

        for i, positions in enumerate(num_positions):
            for j, pos in enumerate(positions):
                if pos < encoder_outputs.size(1):
                    num_embeddings[i, j] = encoder_outputs[i, pos, :hidden_size]

        return num_embeddings

    def train(self):
        """Full training loop."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.scheduler.step(val_loss)

            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )

            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                self.save_model(output_dir)
                logger.info(f"  -> Saved best model (val_loss={val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        logger.info(f"Training complete. Best val_loss={self.best_loss:.4f}")

    def save_model(self, output_dir: Path):
        """Save model checkpoint."""
        torch.save(
            {"model": self.model.state_dict()},
            output_dir / "model.pth",
        )


def main():
    """Main training function."""
    config = TrainingConfig()

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading training data...")
    train_data = load_jsonl(config.train_path)
    test_data = load_jsonl(config.test_path)
    logger.info(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Shuffle and split train into train/val
    random.seed(42)
    random.shuffle(train_data)
    val_size = len(train_data) // 10
    val_data = train_data[:val_size]
    train_data = train_data[val_size:]
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Build vocabularies
    logger.info("Building vocabularies...")
    input_vocab = build_input_vocab(train_data + val_data, config)
    output_vocab = build_output_vocab(train_data + val_data)
    logger.info(f"Input vocab size: {len(input_vocab['in_idx2word'])}")
    logger.info(f"Output vocab size: {len(output_vocab['out_idx2symbol'])}")
    logger.info(f"Output symbols: {output_vocab['out_idx2symbol']}")

    # Create datasets
    train_dataset = GSM8KDataset(train_data, input_vocab, output_vocab)
    val_dataset = GSM8KDataset(val_data, input_vocab, output_vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Create model
    logger.info("Creating model...")
    model_config = GTSConfig(
        vocab_size=len(input_vocab["in_idx2word"]),
        output_size=len(output_vocab["out_idx2symbol"]),
        generate_size=len(output_vocab["temp_idx2symbol"]),
        embedding_size=config.embedding_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout_ratio=config.dropout_ratio,
    )

    model = GTSModel(model_config, input_vocab, output_vocab)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # Create trainer
    trainer = GTSTrainer(model, train_loader, val_loader, config, device)

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save vocabs
    output_dir = Path(config.output_dir)
    with open(output_dir / "input_vocab.json", "w") as f:
        json.dump(input_vocab, f, indent=2)
    with open(output_dir / "output_vocab.json", "w") as f:
        json.dump(output_vocab, f, indent=2)

    # Save config
    config_dict = {
        "final_config_dict": {
            "embedding_size": config.embedding_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "dropout_ratio": config.dropout_ratio,
        }
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
