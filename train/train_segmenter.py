#!/usr/bin/env python3
"""
Mycelium v6: Train Qwen-0.5B on CLEAN Segmenter Labels

Same as v6_train_segmenter.py but uses filtered data:
- Only correct CoT solutions (95.5% -> 100%)
- Merged short fragments (<=2 chars)
"""

import json
import torch
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import random


# Labels
LABELS = ["O", "B-OP", "I-OP", "B-Q", "I-Q"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for i, l in enumerate(LABELS)}

MODEL_NAME = "Qwen/Qwen2-0.5B"


def load_data(data_path: str, tokenizer):
    """Load and process BIO labeled data."""
    with open(data_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples")

    # Filter out examples with 0 OP spans
    data = [d for d in data if d["n_op_spans"] > 0]
    print(f"After filtering 0-span examples: {len(data)}")

    return data


def prepare_dataset(data: list, tokenizer, max_length=256):
    """Prepare dataset for training."""
    processed = []

    for example in data:
        input_ids = example["input_ids"][:max_length]
        labels = example["bio_labels"][:max_length]

        # Pad to max_length
        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len  # -100 ignores loss

        processed.append({
            "input_ids": input_ids,
            "attention_mask": [1] * (max_length - pad_len) + [0] * pad_len,
            "labels": labels,
        })

    return Dataset.from_list(processed)


def extract_spans(labels: list) -> list:
    """Extract (start, end, type) spans from BIO label sequence."""
    spans = []
    current_start = None
    current_type = None

    for i, label in enumerate(labels):
        if label == -100:
            continue

        label_name = id2label.get(label, "O")

        if label_name.startswith("B-"):
            if current_start is not None:
                spans.append((current_start, i, current_type))
            current_start = i
            current_type = label_name[2:]
        elif label_name.startswith("I-"):
            if current_start is None:
                current_start = i
                current_type = label_name[2:]
        else:  # O
            if current_start is not None:
                spans.append((current_start, i, current_type))
                current_start = None
                current_type = None

    if current_start is not None:
        spans.append((current_start, len(labels), current_type))

    return spans


def compute_span_metrics(pred_labels, true_labels):
    """Compute span-level precision, recall, F1."""
    pred_spans_all = []
    true_spans_all = []

    for preds, trues in zip(pred_labels, true_labels):
        pred_spans = extract_spans(preds)
        true_spans = extract_spans([t for t in trues if t != -100])
        pred_spans_all.append(pred_spans)
        true_spans_all.append(true_spans)

    tp = 0
    total_pred = 0
    total_true = 0

    for pred_spans, true_spans in zip(pred_spans_all, true_spans_all):
        total_pred += len(pred_spans)
        total_true += len(true_spans)

        for pred in pred_spans:
            for true in true_spans:
                if pred[2] == true[2]:  # Same type
                    overlap_start = max(pred[0], true[0])
                    overlap_end = min(pred[1], true[1])
                    overlap = max(0, overlap_end - overlap_start)

                    pred_len = pred[1] - pred[0]
                    true_len = true[1] - true[0]

                    if overlap > 0.5 * min(pred_len, true_len):
                        tp += 1
                        break

    precision = tp / total_pred if total_pred > 0 else 0
    recall = tp / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "span_precision": precision,
        "span_recall": recall,
        "span_f1": f1,
        "total_pred_spans": total_pred,
        "total_true_spans": total_true,
    }


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_flat = []
    pred_flat = []

    for pred_seq, label_seq in zip(predictions, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_flat.append(l)
                pred_flat.append(p)

    correct = sum(1 for p, t in zip(pred_flat, true_flat) if p == t)
    token_accuracy = correct / len(true_flat) if true_flat else 0

    span_metrics = compute_span_metrics(predictions.tolist(), labels.tolist())

    return {
        "token_accuracy": token_accuracy,
        **span_metrics,
    }


def main():
    print("=" * 60)
    print("MYCELIUM V6: SEGMENTER TRAINING (CLEAN LABELS)")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load CLEAN data (filtered to correct solutions only)
    print("\nLoading training data (CLEAN - correct solutions only)...")
    data = load_data("data/bio_segmentation_clean.json", tokenizer)

    # Split train/eval
    random.seed(42)
    random.shuffle(data)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = prepare_dataset(train_data, tokenizer)
    eval_dataset = prepare_dataset(eval_data, tokenizer)

    # Load model
    print("\nLoading model...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    print(f"Model parameters: {model.num_parameters()/1e6:.1f}M")

    # Training arguments - clean labels model
    output_dir = "/opt/dlami/nvme/models/qwen05b_segmenter_clean"
    if not Path("/opt/dlami/nvme").exists():
        output_dir = "models/qwen05b_segmenter_clean"

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
        metric_for_best_model="span_f1",
        greater_is_better=True,
        logging_steps=50,
        bf16=True,
        report_to="none",
        save_total_limit=2,
        warmup_ratio=0.1,
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )

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
    print(f"\nSaving model to {output_dir}/final...")
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    # Quick inference test
    print("\n" + "=" * 60)
    print("INFERENCE TEST")
    print("=" * 60)

    test_text = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether?"

    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)[0]

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels_pred = [id2label[p.item()] for p in predictions]

    print(f"Input: {test_text}\n")
    print("Predictions:")
    current_span = None
    span_tokens = []

    for tok, lab in zip(tokens, labels_pred):
        tok_str = tok.replace("Ġ", " ").replace("▁", " ")
        if lab.startswith("B-"):
            if current_span:
                print(f"  [{current_span}] {''.join(span_tokens).strip()}")
            current_span = lab[2:]
            span_tokens = [tok_str]
        elif lab.startswith("I-") and current_span:
            span_tokens.append(tok_str)
        else:
            if current_span:
                print(f"  [{current_span}] {''.join(span_tokens).strip()}")
                current_span = None
                span_tokens = []

    if current_span:
        print(f"  [{current_span}] {''.join(span_tokens).strip()}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE (CLEAN LABELS)")
    print("=" * 60)


if __name__ == "__main__":
    main()
