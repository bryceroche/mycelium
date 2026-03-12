#!/usr/bin/env python3
"""
Train the canonicalizer model: problem_text + scaffold → telegrams.

Architecture: Qwen-0.5B + LoRA (r=16, alpha=32, Q/K/V/O)
Training: 3 epochs, cosine schedule, lr=2e-4, float32 on A10G

Usage:
    python train_canonicalizer.py \
        --train canonicalizer_train.jsonl \
        --val canonicalizer_val.jsonl \
        --output models/canonicalizer_v1
"""

import argparse
import json
import os
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


def load_training_data(train_path: str, val_path: str = None):
    """Load training data from JSONL files."""

    def load_jsonl(path):
        examples = []
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                examples.append(ex)
        return examples

    train_examples = load_jsonl(train_path)
    val_examples = load_jsonl(val_path) if val_path else []

    print(f"Loaded {len(train_examples)} train, {len(val_examples)} val examples")
    return train_examples, val_examples


def format_example(example: dict) -> str:
    """
    Format a training example as input-target string.

    Input format:
        Problem: <problem_text>
        Scaffold: [GIVEN, EVAL, SOLVE, ANSWER]
        Telegrams:

    Target format:
        GIVEN x^2+y^2=90
        EVAL _prev
        SOLVE x+y=12 x
        ANSWER _prev
    """
    problem_text = example.get("problem_text", "")
    scaffold = example.get("scaffold", [])
    target = example.get("target", "")

    # Format scaffold as bracket list
    scaffold_str = "[" + ", ".join(scaffold) + "]"

    # Build full sequence
    full_text = f"Problem: {problem_text}\nScaffold: {scaffold_str}\nTelegrams:\n{target}"

    return full_text


def tokenize_examples(examples: list, tokenizer, max_length: int = 512):
    """Tokenize examples for training."""

    texts = [format_example(ex) for ex in examples]

    # Tokenize with padding to max_length
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )

    # For causal LM, labels = input_ids (pad tokens will be ignored by loss)
    tokenized["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in ids]
        for ids in tokenized["input_ids"]
    ]

    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Train canonicalizer")
    parser.add_argument("--train", required=True, help="Training data JSONL")
    parser.add_argument("--val", default=None, help="Validation data JSONL")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")

    args = parser.parse_args()

    # Load data
    train_examples, val_examples = load_training_data(args.train, args.val)

    # Load tokenizer
    base_model = "Qwen/Qwen2.5-0.5B"
    print(f"Loading tokenizer from {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize
    print("Tokenizing...")
    train_tokenized = tokenize_examples(train_examples, tokenizer, args.max_length)
    val_tokenized = tokenize_examples(val_examples, tokenizer, args.max_length) if val_examples else None

    # Create datasets
    train_dataset = Dataset.from_dict({
        "input_ids": train_tokenized["input_ids"],
        "attention_mask": train_tokenized["attention_mask"],
        "labels": train_tokenized["labels"],
    })

    val_dataset = None
    if val_tokenized:
        val_dataset = Dataset.from_dict({
            "input_ids": val_tokenized["input_ids"],
            "attention_mask": val_tokenized["attention_mask"],
            "labels": val_tokenized["labels"],
        })

    print(f"Train dataset: {len(train_dataset)} examples")
    if val_dataset:
        print(f"Val dataset: {len(val_dataset)} examples")

    # Load model
    print(f"Loading model from {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,  # A10G doesn't support bf16
        trust_remote_code=True,
    )

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        save_total_limit=2,
        load_best_model_at_end=True if val_dataset else False,
        report_to="none",
        fp16=False,  # A10G: use float32
        bf16=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
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
    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()
