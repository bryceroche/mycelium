#!/usr/bin/env python3
"""
Train LoRA C (operand extraction) on IAF-grounded data.

The IAF data contains operands that the 7B teacher ACTUALLY attended to
when solving each step. This is ground truth operand selection.

Architecture: Qwen-0.5B + LoRA (r=16, alpha=32, Q/K/V/O)
Training: 3 epochs, cosine schedule, lr=2e-4, float32

Usage:
    python scripts/train_lora_c_iaf.py \
        --train data/iaf_training_filtered.jsonl \
        --output models/lora_c_iaf_v1

    # On VM with GPU:
    python scripts/train_lora_c_iaf.py \
        --train /tmp/iaf_training_filtered.jsonl \
        --output /tmp/lora_c_iaf_v1 \
        --epochs 3
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


def load_training_data(train_path: str, val_split: float = 0.1):
    """Load training data from JSONL file."""
    examples = []
    with open(train_path) as f:
        for line in f:
            ex = json.loads(line)
            examples.append(ex)

    # Split into train/val by problem_id
    problem_ids = list(set(ex['problem_id'] for ex in examples))
    val_ids = set(problem_ids[:int(len(problem_ids) * val_split)])

    train_examples = [ex for ex in examples if ex['problem_id'] not in val_ids]
    val_examples = [ex for ex in examples if ex['problem_id'] in val_ids]

    print(f"Loaded {len(examples)} total examples")
    print(f"Train: {len(train_examples)}, Val: {len(val_examples)}")
    return train_examples, val_examples


def format_example(example: dict) -> str:
    """
    Format a training example.

    The prompt already contains the full input format.
    We just need to append the target.
    """
    prompt = example.get("prompt", "")
    target = example.get("target", "")

    # Full sequence: prompt + target
    return prompt + target


def tokenize_examples(examples: list, tokenizer, max_length: int = 384):
    """Tokenize examples for training."""

    texts = [format_example(ex) for ex in examples]

    # Tokenize with padding
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )

    # For causal LM, labels = input_ids (pad tokens ignored by loss)
    tokenized["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in ids]
        for ids in tokenized["input_ids"]
    ]

    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Train LoRA C on IAF operand data")
    parser.add_argument("--train", required=True, help="Path to training JSONL")
    parser.add_argument("--output", default="models/lora_c_iaf_v1", help="Output path")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B", help="Base model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=384, help="Max sequence length")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Training data: {args.train}")

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load training data
    print("\nLoading training data...")
    train_examples, val_examples = load_training_data(args.train)

    # Tokenize
    print("\nTokenizing...")
    train_tokenized = tokenize_examples(train_examples, tokenizer, args.max_length)
    val_tokenized = tokenize_examples(val_examples, tokenizer, args.max_length) if val_examples else None

    train_dataset = Dataset.from_dict(train_tokenized)
    val_dataset = Dataset.from_dict(val_tokenized) if val_tokenized else None

    print(f"Train dataset: {len(train_dataset)} examples")
    if val_dataset:
        print(f"Val dataset: {len(val_dataset)} examples")

    # Load model
    print(f"\nLoading base model {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,  # float32 on A10G per CLAUDE.md
        trust_remote_code=True,
    )

    # Configure LoRA
    print(f"\nConfiguring LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,  # Effective batch = 32
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        load_best_model_at_end=True if val_dataset else False,
        save_total_limit=2,
        report_to="none",
        fp16=False,  # Use float32
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
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer.train()

    # Save
    print(f"\nSaving model to {args.output}...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print("\nDone!")

    # Print sample generation
    print("\n" + "=" * 60)
    print("Sample generation test:")
    print("=" * 60)

    model.eval()
    test_prompt = """Problem: If x^2 + y^2 = 90 and xy = 27, what is (x+y)^2?
Step: SETUP (step 1 of 3)
What values does the teacher attend to for this step?
Operands:"""

    inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(out[0], skip_special_tokens=True)
    generated = response[len(test_prompt):]
    print(f"Prompt: {test_prompt[:100]}...")
    print(f"Generated: {generated}")


if __name__ == "__main__":
    main()
