#!/usr/bin/env python3
"""
Mycelium v6: Train C2 Classifier on IB-Discovered Templates

100-class sequence classification on Qwen-0.5B using 17,101
labeled pairs from Information Bottleneck template discovery.
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
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import random
import argparse
import sys

MODEL_NAME = "Qwen/Qwen2-0.5B"


def load_ib_labels(labels_path: str, template_map_path: str):
    """Load IB-labeled training data."""
    with open(labels_path) as f:
        labels_data = json.load(f)

    with open(template_map_path) as f:
        template_map = json.load(f)

    # Build label mappings
    template_ids = sorted(set(d["template_id"] for d in labels_data))
    id2label = {i: template_map[str(tid)]["name"] for i, tid in enumerate(template_ids)}
    label2id = {name: i for i, name in id2label.items()}
    template_to_idx = {tid: i for i, tid in enumerate(template_ids)}

    print(f"Loaded {len(labels_data)} labeled samples")
    print(f"Template classes: {len(template_ids)}")

    return labels_data, id2label, label2id, template_to_idx


def prepare_input_text(sample: dict) -> str:
    """Format input text with optional span markers."""
    # Use raw_expression as primary input, with category context
    text = sample["raw_expression"]
    category = sample.get("category", "")

    # Add category as context prefix
    if category:
        return f"[{category}] {text}"
    return text


def prepare_dataset(data: list, template_to_idx: dict, tokenizer, max_length=256):
    """Prepare HuggingFace dataset."""
    texts = [prepare_input_text(d) for d in data]
    labels = [template_to_idx[d["template_id"]] for d in data]

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


def compute_metrics(eval_pred, id2label):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)

    # Top-5 accuracy
    top5_correct = 0
    for i, label in enumerate(labels):
        top5_preds = np.argsort(predictions[i])[-5:]
        if label in top5_preds:
            top5_correct += 1
    top5_acc = top5_correct / len(labels) if labels.size > 0 else 0

    return {
        "accuracy": accuracy,
        "top5_accuracy": top5_acc,
    }


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss."""

    def __init__(self, class_weights, num_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.num_labels = num_labels

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device, dtype=logits.dtype)
        )
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-path", default="/home/ubuntu/data/ib_results/classifier_labels.json")
    parser.add_argument("--template-map-path", default="/home/ubuntu/data/ib_results/template_map.json")
    parser.add_argument("--output-dir", default="/home/ubuntu/models/c2_ib_templates")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    print("=" * 70)
    print("MYCELIUM V6: C2 CLASSIFIER TRAINING (IB TEMPLATES)")
    print("=" * 70)
    sys.stdout.flush()

    # Load data
    print("\nLoading IB-labeled data...")
    sys.stdout.flush()

    labels_data, id2label, label2id, template_to_idx = load_ib_labels(
        args.labels_path, args.template_map_path
    )
    num_labels = len(id2label)

    # Show top templates
    print("\nTop 10 templates by frequency:")
    template_counts = Counter(d["template_id"] for d in labels_data)
    for tid, count in template_counts.most_common(10):
        idx = template_to_idx[tid]
        name = id2label[idx]
        print(f"  {name}: {count}")
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
    random.shuffle(labels_data)
    split_idx = int(len(labels_data) * 0.9)
    train_data = labels_data[:split_idx]
    eval_data = labels_data[split_idx:]

    print(f"\nTrain: {len(train_data)}, Eval: {len(eval_data)}")
    sys.stdout.flush()

    # Compute class weights
    train_labels = [template_to_idx[d["template_id"]] for d in train_data]
    unique_labels = sorted(set(train_labels))

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(unique_labels),
        y=train_labels
    )

    # Create full weight tensor (some classes might not appear in train)
    full_weights = torch.ones(num_labels, dtype=torch.float32)
    for i, label in enumerate(unique_labels):
        full_weights[label] = class_weights[i]

    print(f"\nClass weights computed for {len(unique_labels)} classes")
    print(f"Weight range: [{full_weights.min():.3f}, {full_weights.max():.3f}]")
    sys.stdout.flush()

    # Prepare datasets
    print("\nPreparing datasets...")
    sys.stdout.flush()

    train_dataset = prepare_dataset(train_data, template_to_idx, tokenizer, args.max_length)
    eval_dataset = prepare_dataset(eval_data, template_to_idx, tokenizer, args.max_length)

    # Load model
    print("\nLoading model...")
    sys.stdout.flush()

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    print(f"Model parameters: {model.num_parameters()/1e6:.1f}M")
    print(f"Classification head: {num_labels} classes")
    sys.stdout.flush()

    # Training arguments
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=100,
        fp16=False,
        report_to="none",
        save_total_limit=2,
        warmup_ratio=0.1,
    )

    # Data collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )

    # Create compute_metrics function with id2label
    def metrics_fn(eval_pred):
        return compute_metrics(eval_pred, id2label)

    # Weighted Trainer
    trainer = WeightedTrainer(
        class_weights=full_weights,
        num_labels=num_labels,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=metrics_fn,
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
    for k, v in results.items():
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
            "id2label": id2label,
            "label2id": label2id,
            "template_to_idx": template_to_idx,
        }, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {final_path}")
    print(f"Final accuracy: {results.get('eval_accuracy', 0):.4f}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
