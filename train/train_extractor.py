#!/usr/bin/env python3
"""
Mycelium v6: Train Argument Extractor on Multi-Span Data

Uses the new multi-span training data where arguments are
extracted from groups of 1, 2, or 3 marked spans.

Input: [OP_LABEL]\n<problem with marked spans>
Output: arg0|source\narg1|source
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


MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"


def load_data(data_path: str):
    """Load multi-span extraction training pairs."""
    with open(data_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} training pairs")
    return data


def prepare_dataset(data: list, tokenizer, max_length=384):
    """Prepare dataset for causal LM fine-tuning."""
    processed = []

    for item in data:
        input_text = item["input"]
        output_text = item["output"]

        # Format: input + output with loss only on output
        full_text = f"{input_text}\nArguments:\n{output_text}"

        # Tokenize full sequence
        encodings = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        input_ids = encodings["input_ids"]

        # Find where the output starts
        input_only = f"{input_text}\nArguments:\n"
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
            "n_spans": item.get("n_spans", 1),
        })

    return Dataset.from_list(processed)


def main():
    print("=" * 60)
    print("MYCELIUM V6: EXTRACTOR TRAINING (MULTI-SPAN)")
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
    data = load_data("data/extraction_multispan.json")

    # Show span count distribution
    span_counts = Counter(d["n_spans"] for d in data)
    print("\nSpan count distribution:")
    for n, count in sorted(span_counts.items()):
        pct = 100 * count / len(data)
        print(f"  {n} span(s): {count} ({pct:.1f}%)")

    # Show source distribution
    print("\nArgument source distribution:")
    sources = Counter()
    for d in data:
        sources[d.get("arg1_source", "UNK").split(":")[0]] += 1
        if d.get("arg2_source"):
            sources[d.get("arg2_source", "UNK").split(":")[0]] += 1
    total = sum(sources.values())
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count} ({100*count/total:.1f}%)")

    # Split train/eval
    random.seed(42)
    random.shuffle(data)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    print(f"\nTrain: {len(train_data)}, Eval: {len(eval_data)}")

    # Show example
    print("\nExample training pair:")
    ex = train_data[0]
    print(f"Input (first 150 chars):\n{ex['input'][:150]}...")
    print(f"Output:\n{ex['output']}")

    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = prepare_dataset(train_data, tokenizer)
    eval_dataset = prepare_dataset(eval_data, tokenizer)

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    print(f"Model parameters: {model.num_parameters()/1e6:.1f}M")

    # Training arguments
    output_dir = "/opt/dlami/nvme/models/qwen05b_extractor_multispan"
    if not Path("/opt/dlami/nvme").exists():
        output_dir = "models/qwen05b_extractor_multispan"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        bf16=True,
        report_to="none",
        save_total_limit=2,
        warmup_ratio=0.1,
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

    # Inference test
    print("\n" + "=" * 60)
    print("INFERENCE TEST")
    print("=" * 60)

    test_examples = [
        # Single-span
        {
            "input": "[SUB]\nBob has 5 apples. <SPAN> He gives 2 to Sally </SPAN> and 1 more to Sam.",
            "expected": "5|PROB\n2|PROB",
        },
        # Multi-span
        {
            "input": "[ADD]\n<SPAN> In the first box he counted 72 raisins </SPAN> and <SPAN> in a second box he counted 74 raisins </SPAN>. How many total?",
            "expected": "72|PROB\n74|PROB",
        },
        {
            "input": "[MUL]\n<SPAN> sells apples for $2 each </SPAN> Maria <SPAN> bought 5 apples </SPAN>.",
            "expected": "2|PROB\n5|PROB",
        },
        {
            "input": "[DIV]\nNatalia sold clips to 48 of her friends in April, <SPAN> and then she sold half as many clips in May </SPAN>.",
            "expected": "48|PROB\n2|IMP:half",
        },
    ]

    print("\nTest predictions:")
    model.eval()

    for ex in test_examples:
        prompt = f"{ex['input']}\nArguments:\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        generated = generated.strip().split("\n")[:2]
        generated = "\n".join(generated)

        n_spans = ex["input"].count("<SPAN>")
        print(f"\n  {n_spans}-span input")
        print(f"  Expected: {ex['expected']}")
        print(f"  Got:      {generated}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE (MULTI-SPAN)")
    print("=" * 60)


if __name__ == "__main__":
    main()
