#!/usr/bin/env python3
"""
Mycelium v6: Train C1 Relevance Scorer (Regression)

Per six_model.md: C1 outputs continuous relevance scores (0-1) per token.
Uses WEIGHTED MSE loss (10x on hot tokens >0.3) to prioritize recall over precision.
Downstream models can't recover missed tokens, so we optimize for top-k recall.

Output: A soft heatmap over problem tokens. Downstream models decide what to attend to.
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
)
from transformers.modeling_outputs import TokenClassifierOutput
import random
import argparse


class Qwen2ForTokenRegression(PreTrainedModel):
    """Qwen2 with a token-level regression head for relevance scoring."""

    def __init__(self, config):
        super().__init__(config)
        self.qwen = AutoModel.from_pretrained(
            config._name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        self.qwen = self.qwen.float()
        self.dropout = nn.Dropout(0.1)
        # Single output per token (relevance score)
        self.regressor = nn.Linear(config.hidden_size, 1)

        # Initialize weights
        self.regressor.weight.data.normal_(mean=0.0, std=0.02)
        self.regressor.bias.data.zero_()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        # Output shape: (batch, seq_len, 1) -> squeeze to (batch, seq_len)
        predictions = self.regressor(sequence_output).squeeze(-1)

        loss = None
        if labels is not None:
            # Weighted MSE loss - 10x weight on hot tokens (relevance > 0.3)
            loss_fct = nn.MSELoss(reduction='none')
            # Mask out padding tokens
            mask = (labels != -100).float()
            # Replace -100 with 0 for MSE computation (won't affect loss due to mask)
            labels_safe = labels.clone()
            labels_safe[labels == -100] = 0

            per_token_loss = loss_fct(predictions, labels_safe)

            # Weight hot tokens 10x more than cold tokens
            # This prevents the model from trading off hot-token accuracy for cold-token accuracy
            hot_token_weight = torch.where(labels_safe > 0.3, 10.0, 1.0)
            weighted_loss = per_token_loss * hot_token_weight

            # Apply mask and compute mean
            loss = (weighted_loss * mask).sum() / (hot_token_weight * mask).sum()

        return TokenClassifierOutput(
            loss=loss,
            logits=predictions.unsqueeze(-1),  # Add dim for compatibility
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


MODEL_NAME = "Qwen/Qwen2-0.5B"


def load_data(data_path: str):
    """Load relevance-scored data."""
    with open(data_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples")

    # Filter out examples with no relevance
    data = [d for d in data if max(d.get('relevance', [0])) > 0]
    print(f"After filtering zero-relevance: {len(data)}")

    return data


def align_relevance(orig_relevance: list, orig_len: int, new_len: int) -> list:
    """
    Align relevance scores from original tokenization to new tokenization.
    Uses proportional mapping.
    """
    if orig_len == new_len:
        return orig_relevance[:new_len]

    aligned = []
    for i in range(new_len):
        orig_pos = int(round(i * (orig_len - 1) / (new_len - 1))) if new_len > 1 else 0
        orig_pos = min(orig_pos, len(orig_relevance) - 1)
        aligned.append(orig_relevance[orig_pos])

    return aligned


def prepare_dataset(data: list, tokenizer, max_length=256):
    """Prepare dataset for training."""
    processed = []
    skipped = 0

    for example in data:
        if "input_ids" in example:
            input_ids = example["input_ids"][:max_length]
            relevance = example["relevance"][:max_length]
        elif "text" in example:
            encoding = tokenizer(
                example["text"],
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            input_ids = encoding["input_ids"]
            orig_relevance = example["relevance"]
            relevance = align_relevance(orig_relevance, len(orig_relevance), len(input_ids))
        else:
            skipped += 1
            continue

        # Pad to max_length
        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            relevance = relevance + [-100.0] * pad_len  # -100 ignores loss

        processed.append({
            "input_ids": input_ids,
            "attention_mask": [1] * (max_length - pad_len) + [0] * pad_len,
            "labels": relevance,  # Continuous scores, not integers
        })

    if skipped > 0:
        print(f"  Skipped {skipped} examples")

    return Dataset.from_list(processed)


def compute_metrics(eval_pred):
    """Compute evaluation metrics for regression."""
    predictions, labels = eval_pred
    predictions = predictions.squeeze(-1)  # Remove last dim

    # Flatten and filter padding
    pred_flat = []
    label_flat = []

    for pred_seq, label_seq in zip(predictions, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                pred_flat.append(p)
                label_flat.append(l)

    pred_flat = np.array(pred_flat)
    label_flat = np.array(label_flat)

    # MSE
    mse = np.mean((pred_flat - label_flat) ** 2)

    # MAE
    mae = np.mean(np.abs(pred_flat - label_flat))

    # Correlation
    if len(pred_flat) > 1:
        correlation = np.corrcoef(pred_flat, label_flat)[0, 1]
    else:
        correlation = 0.0

    # Threshold-based accuracy (predict >0.5 as "relevant")
    pred_binary = (pred_flat > 0.5).astype(int)
    label_binary = (label_flat > 0.5).astype(int)
    threshold_acc = np.mean(pred_binary == label_binary)

    # Top-k recall: of tokens with label > 0.5, what fraction have pred > 0.3?
    high_label_mask = label_flat > 0.5
    if high_label_mask.sum() > 0:
        top_k_recall = np.mean(pred_flat[high_label_mask] > 0.3)
    else:
        top_k_recall = 0.0

    return {
        "mse": mse,
        "mae": mae,
        "correlation": correlation if not np.isnan(correlation) else 0.0,
        "threshold_acc": threshold_acc,
        "top_k_recall": top_k_recall,
    }


class RegressionDataCollator:
    """Custom data collator for regression with float labels."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = {
            "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
            "labels": torch.tensor([f["labels"] for f in features], dtype=torch.float),
        }
        return batch


def main():
    parser = argparse.ArgumentParser(description="Train C1 Relevance Scorer")
    parser.add_argument("--data-path", default="data/c1_relevance/c1_relevance_merged.json",
                        help="Path to training data")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    print("=" * 60)
    print("MYCELIUM V6: C1 RELEVANCE SCORER (REGRESSION)")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print(f"\nLoading training data from {args.data_path}...")
    data = load_data(args.data_path)

    # Split train/eval
    random.seed(42)
    random.shuffle(data)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = prepare_dataset(train_data, tokenizer, args.max_length)
    eval_dataset = prepare_dataset(eval_data, tokenizer, args.max_length)

    # Load model
    print("\nLoading model...")
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config.torch_dtype = torch.float32
    model = Qwen2ForTokenRegression(config)
    model = model.float()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif Path("/opt/dlami/nvme").exists():
        output_dir = "/opt/dlami/nvme/models/c1_relevance"
    else:
        output_dir = "models/c1_relevance"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="top_k_recall",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        save_total_limit=5,
        warmup_ratio=0.1,
    )

    # Data collator
    data_collator = RegressionDataCollator(tokenizer)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nStarting training...")
    print("=" * 60)
    trainer.train()

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    results = trainer.evaluate()
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save model
    final_path = f"{output_dir}/final"
    print(f"\nSaving model to {final_path}...")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Quick inference test
    print("\n" + "=" * 60)
    print("INFERENCE TEST")
    print("=" * 60)

    test_texts = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether?",
        "Find $2^{-1} \\pmod{185}$, as a residue modulo 185.",
    ]

    model.eval()
    for test_text in test_texts:
        inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits.squeeze(-1)[0]

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        print(f"\nInput: {test_text[:80]}...")
        print("High relevance tokens (>0.5):")

        high_rel = [(i, tok, pred.item()) for i, (tok, pred)
                    in enumerate(zip(tokens, predictions)) if pred > 0.5]

        for i, tok, score in sorted(high_rel, key=lambda x: -x[2])[:10]:
            print(f"  [{i:3d}] {tok:20s} {score:.3f}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
