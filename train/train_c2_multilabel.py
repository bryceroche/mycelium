#!/usr/bin/env python3
"""
Mycelium v6: Train C2 Multi-Label Classifier

Trains Qwen-0.5B for multi-label operation classification.
Each problem can have multiple operations (ADD, SUB, MUL, DIV, etc.)

Uses BCEWithLogitsLoss for multi-label classification.
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import f1_score, precision_score, recall_score
import random
import argparse
import sys

MODEL_NAME = "Qwen/Qwen2-0.5B"

# Core operations from IB template discovery
OPERATIONS = [
    "ADD", "SUBTRACT", "MULTIPLY", "DIVIDE",
    "POWER", "SQRT", "MOD", "ABS",
    "MIN", "MAX", "FLOOR", "CEIL", "ROUND",
    "LOG", "EXP", "SIN", "COS", "TAN",
    "FACTORIAL", "GCD", "LCM", "PERCENT",
    "RATIO", "AVERAGE", "SUM", "PRODUCT"
]


def load_multilabel_data(path: str):
    """Load multi-label training data."""
    with open(path) as f:
        data = json.load(f)

    # Discover all unique labels
    all_labels = set()
    for item in data:
        all_labels.update(item.get("labels", []))

    labels = sorted(all_labels)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    print(f"Loaded {len(data)} samples")
    print(f"Label classes: {len(labels)}")
    print(f"Labels: {labels}")

    return data, label2id, id2label


def prepare_multilabel_dataset(data: list, label2id: dict, tokenizer, max_length=256):
    """Prepare dataset with multi-hot label encoding."""
    num_labels = len(label2id)

    texts = [d["input"] for d in data]

    # Multi-hot encoding
    labels = []
    for d in data:
        label_vec = [0.0] * num_labels
        for lbl in d.get("labels", []):
            if lbl in label2id:
                label_vec[label2id[lbl]] = 1.0
        labels.append(label_vec)

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    return Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    })


def compute_multilabel_metrics(eval_pred):
    """Compute multi-label classification metrics."""
    logits, labels = eval_pred

    # Sigmoid + threshold
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs > 0.5).astype(int)
    labels = labels.astype(int)

    # Micro and macro F1
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    precision = precision_score(labels, preds, average='micro', zero_division=0)
    recall = recall_score(labels, preds, average='micro', zero_division=0)

    # Exact match (all labels correct)
    exact_match = np.all(preds == labels, axis=1).mean()

    # At least one correct
    any_correct = (((preds == 1) & (labels == 1)).sum(axis=1) > 0).mean()

    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "precision": precision,
        "recall": recall,
        "exact_match": exact_match,
        "any_correct": any_correct,
    }


class MultiLabelTrainer(Trainer):
    """Trainer with BCEWithLogitsLoss for multi-label classification."""

    def __init__(self, pos_weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # BCEWithLogitsLoss for multi-label - move pos_weight to same device
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=pw)
        loss = loss_fct(logits, labels.float().to(logits.device))

        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/training/c2_math_multilabel.json")
    parser.add_argument("--output-dir", default="models/c2_multilabel")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    print("=" * 70)
    print("MYCELIUM V6: C2 MULTI-LABEL CLASSIFIER TRAINING")
    print("=" * 70)
    sys.stdout.flush()

    # Load data
    print(f"\nLoading multi-label data from {args.data_path}...")
    sys.stdout.flush()

    data, label2id, id2label = load_multilabel_data(args.data_path)
    num_labels = len(label2id)

    # Show label distribution
    print("\nLabel distribution:")
    label_counts = {l: 0 for l in label2id}
    for d in data:
        for lbl in d.get("labels", []):
            if lbl in label_counts:
                label_counts[lbl] += 1
    for lbl, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {lbl}: {count}")
    sys.stdout.flush()

    # Load tokenizer
    print("\nLoading tokenizer...")
    sys.stdout.flush()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Split train/eval
    random.seed(42)
    random.shuffle(data)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    print(f"\nTrain: {len(train_data)}, Eval: {len(eval_data)}")
    sys.stdout.flush()

    # Compute pos_weight for imbalanced labels
    total = len(train_data)
    pos_counts = [0] * num_labels
    for d in train_data:
        for lbl in d.get("labels", []):
            if lbl in label2id:
                pos_counts[label2id[lbl]] += 1

    pos_weight = torch.tensor([
        (total - c) / max(c, 1) for c in pos_counts
    ], dtype=torch.float32)

    # Cap extreme weights
    pos_weight = torch.clamp(pos_weight, max=10.0)

    print(f"\nPos weight range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")
    sys.stdout.flush()

    # Prepare datasets
    print("\nPreparing datasets...")
    sys.stdout.flush()

    train_dataset = prepare_multilabel_dataset(train_data, label2id, tokenizer, args.max_length)
    eval_dataset = prepare_multilabel_dataset(eval_data, label2id, tokenizer, args.max_length)

    # Load model
    print("\nLoading model...")
    sys.stdout.flush()

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Model parameters: {model.num_parameters()/1e6:.1f}M")
    print(f"Classification head: {num_labels} labels (multi-label)")
    sys.stdout.flush()

    # Training arguments
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        logging_steps=50,
        fp16=False,
        report_to="none",
        save_total_limit=2,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # Data collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )

    # MultiLabel Trainer
    trainer = MultiLabelTrainer(
        pos_weight=pos_weight,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_multilabel_metrics,
    )

    # Train
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    sys.stdout.flush()

    trainer.train()

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    sys.stdout.flush()

    results = trainer.evaluate()
    for k, v in sorted(results.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    sys.stdout.flush()

    # Save model
    final_path = f"{args.output_dir}/final"
    print(f"\nSaving model to {final_path}...")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Save label mappings
    with open(f"{final_path}/label_mappings.json", "w") as f:
        json.dump({
            "id2label": {str(k): v for k, v in id2label.items()},
            "label2id": label2id,
            "num_labels": num_labels,
        }, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {final_path}")
    print(f"Final F1 (micro): {results.get('eval_f1_micro', 0):.4f}")
    print(f"Final exact match: {results.get('eval_exact_match', 0):.4f}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
