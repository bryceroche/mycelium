#!/usr/bin/env python3
"""
Train Canonicalizer on Verified Examples

Fresh LoRA from base Qwen-0.5B, r=16, trained on 4,182 execution-verified examples.
Each training target is guaranteed to produce correct results when executed.

Usage:
    python train_canonicalizer_verified.py --output models/canonicalizer_v3_verified
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


class VerifiedDataset(Dataset):
    """Dataset of verified training examples."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # Load data
        if data_path.startswith("s3://"):
            result = subprocess.run(
                ["aws", "s3", "cp", data_path, "-"],
                capture_output=True
            )
            lines = result.stdout.decode().strip().split("\n")
        else:
            with open(data_path) as f:
                lines = f.readlines()

        for line in lines:
            if line.strip():
                self.examples.append(json.loads(line))

        print(f"Loaded {len(self.examples)} verified examples")

        # Show scaffold distribution
        scaffolds = {}
        for ex in self.examples:
            s = ex.get("scaffold", "OTHER")
            scaffolds[s] = scaffolds.get(s, 0) + 1
        print("Scaffold distribution:")
        for s, c in sorted(scaffolds.items(), key=lambda x: -x[1]):
            print(f"  {s}: {c}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Format: input + target
        prompt = ex["input"]
        target = ex["target"]
        full_text = f"{prompt} {target}"

        # Tokenize
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()

        # Create labels (mask prompt tokens with -100)
        prompt_encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        prompt_len = prompt_encoded["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # Don't compute loss on prompt

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="s3://mycelium-data/verified_training/verified_training.jsonl")
    parser.add_argument("--output", default="models/canonicalizer_v3_verified")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    print("=" * 60)
    print("Training Canonicalizer on Verified Examples")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"LoRA r: {args.lora_r}")
    print()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = VerifiedDataset(args.data, tokenizer, args.max_length)

    # Split train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Load base model
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,  # A10G doesn't support bfloat16
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,  # A10G - use fp32
        report_to="none",
        dataloader_num_workers=0,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save
    print(f"\nSaving to {args.output}...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    # Upload to S3
    s3_path = f"s3://mycelium-data/models/canonicalizer_v3_verified/"
    print(f"Uploading to {s3_path}...")
    subprocess.run(["aws", "s3", "sync", args.output, s3_path], check=True)

    print("\nTraining complete!")
    print(f"Model saved to {args.output}")
    print(f"Model uploaded to {s3_path}")


if __name__ == "__main__":
    main()
