#!/usr/bin/env python3
"""
Mycelium v6: Train Operation Classifier on Multi-Span Data

Uses the new multi-span training data where 1, 2, or 3 spans
are marked per operation. This teaches the classifier to work
with candidate groupings.
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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import random


LABELS = ["ADD", "SUB", "MUL", "DIV"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for i, l in enumerate(LABELS)}

MODEL_NAME = "Qwen/Qwen2-0.5B"


def load_data(data_path: str):
    """Load multi-span classifier training pairs."""
    with open(data_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} training pairs")
    return data


def prepare_dataset(data: list, tokenizer, max_length=384):
    """Prepare dataset for training."""
    texts = [d["marked_problem"] for d in data]
    labels = [label2id[d["operation"]] for d in data]

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    })

    return dataset


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = (predictions == labels).mean()

    report = classification_report(
        labels, predictions,
        target_names=LABELS,
        output_dict=True,
        zero_division=0
    )

    macro_f1 = report["macro avg"]["f1-score"]

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "add_f1": report["ADD"]["f1-score"],
        "sub_f1": report["SUB"]["f1-score"],
        "mul_f1": report["MUL"]["f1-score"],
        "div_f1": report["DIV"]["f1-score"],
    }


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss."""

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, len(LABELS)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def main():
    print("=" * 60)
    print("MYCELIUM V6: CLASSIFIER TRAINING (MULTI-SPAN)")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Add span markers
    special_tokens = {"additional_special_tokens": ["<SPAN>", "</SPAN>"]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added} special tokens")

    # Load data
    print("\nLoading training data...")
    data = load_data("data/classification_multispan.json")

    # Show span count distribution
    span_counts = Counter(d["n_spans"] for d in data)
    print("\nSpan count distribution:")
    for n, count in sorted(span_counts.items()):
        pct = 100 * count / len(data)
        print(f"  {n} span(s): {count} ({pct:.1f}%)")

    # Split train/eval
    random.seed(42)
    random.shuffle(data)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    print(f"\nTrain: {len(train_data)}, Eval: {len(eval_data)}")

    # Compute class weights
    train_labels = [label2id[d["operation"]] for d in train_data]
    label_counts = Counter(train_labels)

    print("\nClass distribution in training data:")
    for label_id, count in sorted(label_counts.items()):
        pct = 100 * count / len(train_labels)
        print(f"  {LABELS[label_id]}: {count} ({pct:.1f}%)")

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1, 2, 3]),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    print("\nClass weights (balanced):")
    for i, w in enumerate(class_weights):
        print(f"  {LABELS[i]}: {w:.3f}")

    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = prepare_dataset(train_data, tokenizer)
    eval_dataset = prepare_dataset(eval_data, tokenizer)

    # Load model
    print("\nLoading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    print(f"Model parameters: {model.num_parameters()/1e6:.1f}M")

    # Training arguments
    output_dir = "/opt/dlami/nvme/models/qwen05b_classifier_multispan"
    if not Path("/opt/dlami/nvme").exists():
        output_dir = "models/qwen05b_classifier_multispan"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=3e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        bf16=True,
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

    # Weighted Trainer
    trainer = WeightedTrainer(
        class_weights=class_weights,
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

    # Per-class report
    print("\n" + "-" * 40)
    print("Per-class classification report:")
    print("-" * 40)

    predictions = trainer.predict(eval_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels_true = predictions.label_ids

    print(classification_report(
        labels_true, preds,
        target_names=LABELS,
        digits=3
    ))

    # Confusion matrix
    print("\nConfusion matrix (counts):")
    cm = confusion_matrix(labels_true, preds)
    print(f"         {'  '.join(LABELS)}")
    for i, row in enumerate(cm):
        print(f"  {LABELS[i]:4}: {row}")

    # Evaluate by span count
    print("\n" + "-" * 40)
    print("Accuracy by span count:")
    print("-" * 40)

    eval_preds_with_meta = list(zip(preds, labels_true, eval_data))
    for n_spans in [1, 2, 3]:
        subset = [(p, l) for p, l, d in eval_preds_with_meta if d["n_spans"] == n_spans]
        if subset:
            correct = sum(1 for p, l in subset if p == l)
            acc = correct / len(subset)
            print(f"  {n_spans} span(s): {correct}/{len(subset)} = {100*acc:.1f}%")

    # Save model
    print(f"\nSaving model to {output_dir}/final...")
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    # Inference test
    print("\n" + "=" * 60)
    print("INFERENCE TEST")
    print("=" * 60)

    test_examples = [
        # Single-span
        ("Bob has 5 apples. <SPAN> He gives 2 to Sally </SPAN> and 1 more to Sam.", "SUB"),
        ("<SPAN> John bought 3 pencils at $2 each </SPAN>. How much did he spend?", "MUL"),
        # Multi-span
        ("<SPAN> In the first box he counted 72 raisins </SPAN> and <SPAN> in a second box he counted 74 raisins </SPAN>. How many total?", "ADD"),
        ("<SPAN> sells apples for $2 each </SPAN> and <SPAN> bought 5 apples </SPAN>. How much?", "MUL"),
    ]

    print("\nTest predictions:")
    correct = 0
    for text, expected in test_examples:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=384)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=-1).item()
            pred_label = id2label[pred_id]

        n_spans = text.count("<SPAN>")
        status = "OK" if pred_label == expected else "WRONG"
        if pred_label == expected:
            correct += 1
        print(f"  [{status}] {n_spans}-span -> {pred_label} (expected {expected})")

    print(f"\nInference test: {correct}/{len(test_examples)} correct")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE (MULTI-SPAN)")
    print("=" * 60)


if __name__ == "__main__":
    main()
