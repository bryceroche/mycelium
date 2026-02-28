#!/usr/bin/env python3
"""
Mycelium v6: Train C3 Extractor on Full-Text CoT Data

Trains Qwen-0.5B-Instruct to extract mathematical expressions from
problem text and CoT reasoning clauses.

Input format (JSONL):
{"problem": "...", "clause": "...", "expression": "48.0+24.0", "result": "72.0", "op_type": "add"}

Model input: [OP_TYPE] Problem: {problem}\nClause: {clause}
Model output: {expression}
"""

import json
import torch
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from collections import Counter
import random
import argparse
import sys

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"


def load_jsonl(path: str):
    """Load JSONL training data."""
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} training samples")
    return data


def format_training_sample(item: dict) -> dict:
    """Format a sample for extraction training.

    Input format at inference: [OP_TYPE] <problem text>
    Output: expression (e.g., "48 / 2")

    No CoT clause - at inference we only have problem + template from C2.
    """
    op_type = item.get("op_type", "").upper()
    problem = item.get("problem", "")
    expression = item.get("expression", "")

    # Format: [TEMPLATE] problem text
    input_text = f"[{op_type}] {problem}"
    output_text = expression

    return {"input": input_text, "output": output_text, "op_type": op_type}


def prepare_dataset(data: list, tokenizer, max_length=512):
    """Prepare dataset for causal LM fine-tuning."""
    processed = []

    for item in data:
        formatted = format_training_sample(item)
        input_text = formatted["input"]
        output_text = formatted["output"]

        # Skip if no valid expression
        if not output_text or output_text == "":
            continue

        # Format: input + output with loss only on output
        full_text = f"{input_text}\nExpression: {output_text}"

        # Tokenize full sequence
        encodings = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        input_ids = encodings["input_ids"]

        # Find where the output starts
        input_only = f"{input_text}\nExpression: "
        input_encodings = tokenizer(
            input_only,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        input_len = len(input_encodings["input_ids"])

        # Create labels: -100 for input tokens, actual ids for output
        labels = [-100] * input_len + input_ids[input_len:]
        labels = labels[:len(input_ids)]

        processed.append({
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "op_type": formatted["op_type"],
        })

    print(f"Prepared {len(processed)} valid samples (skipped {len(data) - len(processed)} invalid)")
    return Dataset.from_list(processed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/training/c3_math_fulltext.jsonl")
    parser.add_argument("--output-dir", default="models/c3_extractor")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--freeze-layers", type=int, default=16,
                        help="Freeze first N transformer layers (Qwen2-0.5B has 24)")
    args = parser.parse_args()

    print("=" * 70)
    print("MYCELIUM V6: C3 EXTRACTOR TRAINING (FULL-TEXT)")
    print("=" * 70)
    sys.stdout.flush()

    # Load tokenizer
    print("\nLoading tokenizer...")
    sys.stdout.flush()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load data
    print(f"\nLoading training data from {args.data_path}...")
    sys.stdout.flush()

    data = load_jsonl(args.data_path)

    # Show operation distribution
    print("\nOperation distribution:")
    op_counts = Counter(d.get("op_type", "UNK") for d in data)
    for op, count in op_counts.most_common(15):
        pct = 100 * count / len(data)
        print(f"  {op}: {count} ({pct:.1f}%)")
    sys.stdout.flush()

    # Split train/eval
    random.seed(42)
    random.shuffle(data)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    print(f"\nTrain: {len(train_data)}, Eval: {len(eval_data)}")
    sys.stdout.flush()

    # Show example
    print("\nExample training sample:")
    ex = format_training_sample(train_data[0])
    print(f"Input:\n{ex['input'][:200]}...")
    print(f"Output: {ex['output']}")
    sys.stdout.flush()

    # Prepare datasets
    print("\nPreparing datasets...")
    sys.stdout.flush()

    train_dataset = prepare_dataset(train_data, tokenizer, args.max_length)
    eval_dataset = prepare_dataset(eval_data, tokenizer, args.max_length)

    # Load model
    print("\nLoading model...")
    sys.stdout.flush()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    # Partial layer freezing
    freeze_layers = args.freeze_layers
    for name, param in model.named_parameters():
        # Always train embeddings and LM head
        if "embed" in name or "lm_head" in name:
            param.requires_grad = True
        # Freeze early layers
        elif "layers." in name:
            layer_num = int(name.split("layers.")[1].split(".")[0])
            param.requires_grad = (layer_num >= freeze_layers)
        else:
            # Final layer norm etc - train these
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = model.num_parameters()
    print(f"Partial freeze: {trainable:,} / {total:,} params trainable ({100*trainable/total:.1f}%)")
    print(f"  Frozen: layers 0-{freeze_layers-1}, Training: layers {freeze_layers}-23 + LM head")
    sys.stdout.flush()

    # Training arguments
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        fp16=False,  # Disabled - CUBLAS issues
        report_to="none",
        save_total_limit=2,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
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

    # Save metadata
    with open(f"{final_path}/training_metadata.json", "w") as f:
        json.dump({
            "base_model": MODEL_NAME,
            "freeze_layers": freeze_layers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "final_loss": results.get("eval_loss", 0),
        }, f, indent=2)

    # Inference test
    print("\n" + "=" * 70)
    print("INFERENCE TEST")
    print("=" * 70)
    sys.stdout.flush()

    test_examples = [
        {
            "input": "[ADD] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "expected": "48+24",
        },
        {
            "input": "[DIVIDE] Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            "expected": "50/60",
        },
        {
            "input": "[MULTIPLY] Betty is saving money for a new wallet which costs $100. Her parents give her $15 and her grandparents twice as much as her parents.",
            "expected": "15*2",
        },
    ]

    print("\nTest predictions:")
    model.eval()
    device = next(model.parameters()).device

    for ex in test_examples:
        prompt = f"{ex['input']}\nExpression: "
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        generated = generated.strip().split("\n")[0]  # Take first line only

        print(f"\n  Input: ...{ex['input'][-80:]}...")
        print(f"  Expected: {ex['expected']}")
        print(f"  Got:      {generated}")
    sys.stdout.flush()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {final_path}")
    print(f"Final loss: {results.get('eval_loss', 0):.4f}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
