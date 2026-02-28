#!/usr/bin/env python3
"""
Mycelium v6: C3 Expression Extractor (HuggingFace Trainer)

Uses HF Trainer for multi-GPU training - handles device placement automatically.
Run with: accelerate launch --num_processes=8 train_c3_hf.py
Or single GPU: python train_c3_hf.py
"""

import json
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
import argparse
import sys
import os
import boto3
import sympy
import warnings
warnings.filterwarnings("ignore")

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"


class C3Dataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=384):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        input_text = ex["input"]
        output_text = ex["output"]

        # Add EOS so model learns when to stop
        eos = self.tokenizer.eos_token or ""
        full_text = f"{input_text}\nExpression: {output_text}{eos}"

        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        # Mask input portion for loss computation
        input_only = f"{input_text}\nExpression: "
        input_encodings = self.tokenizer(input_only, truncation=True, max_length=self.max_length, padding=False)
        input_len = len(input_encodings["input_ids"])

        labels = input_ids.clone()
        labels[:input_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def validate_sympy(expr_str):
    try:
        sympy.sympify(expr_str)
        return True
    except:
        return False


def load_data_from_s3(s3_path):
    s3 = boto3.client('s3')
    bucket = s3_path.split('/')[2]
    key = '/'.join(s3_path.split('/')[3:])

    response = s3.get_object(Bucket=bucket, Key=key)
    examples = []
    for line in response['Body'].iter_lines():
        if line:
            examples.append(json.loads(line.decode('utf-8')))
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="s3://mycelium-data/c3_training/c3_train_clean.jsonl")
    parser.add_argument("--output-dir", default="models/c3_extractor")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()

    print("=" * 70)
    print("MYCELIUM V6: C3 EXPRESSION EXTRACTOR (HF Trainer)")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Data: {args.data_path}")
    sys.stdout.flush()

    # Load data
    print(f"\nLoading data...")
    if args.data_path.startswith("s3://"):
        examples = load_data_from_s3(args.data_path)
    else:
        with open(args.data_path) as f:
            examples = [json.loads(line) for line in f]

    print(f"Loaded {len(examples)} examples")
    sys.stdout.flush()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Train/val split
    import random
    random.seed(42)
    random.shuffle(examples)
    split_idx = int(len(examples) * 0.9)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    print(f"Train: {len(train_examples)}, Val: {len(val_examples)}")
    sys.stdout.flush()

    # Create datasets
    train_dataset = C3Dataset(train_examples, tokenizer, args.max_length)
    val_dataset = C3Dataset(val_examples, tokenizer, args.max_length)

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    sys.stdout.flush()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    sys.stdout.flush()

    trainer.train()

    # Save best model
    print("\nSaving best model...")
    trainer.save_model(f"{args.output_dir}/best")
    tokenizer.save_pretrained(f"{args.output_dir}/best")

    # Upload to S3
    print("\nUploading to S3...")
    s3 = boto3.client('s3')
    save_path = f"{args.output_dir}/best"

    for fname in Path(save_path).iterdir():
        if fname.is_file():
            s3_key = f"models/c3_extractor/{fname.name}"
            s3.upload_file(str(fname), "mycelium-data", s3_key)
            print(f"  Uploaded {fname.name}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
