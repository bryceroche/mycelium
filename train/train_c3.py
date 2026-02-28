#!/usr/bin/env python3
"""
Mycelium v6: C3 Per-Operation Expression Extractor

Task: Given full problem text + ONE template tag, extract the sympy expression.
Architecture: MiniLM-22M encoder + small decoder head (or Qwen-0.5B causal LM)

Input format: [TEMPLATE: DIVIDE] Natalia sold clips to 48 of her friends...
Output format: 48 / 2

NO CoT anywhere - just problem text and template tag.
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
import random
import argparse
import sys
import boto3
import sympy

# Use Qwen-0.5B for generation (causal LM)
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"


class C3Dataset(Dataset):
    """Dataset for C3 expression extraction."""

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

        # Format: input\nExpression: output
        full_text = f"{input_text}\nExpression: {output_text}"

        # Tokenize full sequence
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        # Find where output starts (after "Expression: ")
        input_only = f"{input_text}\nExpression: "
        input_encodings = self.tokenizer(
            input_only,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        input_len = len(input_encodings["input_ids"])

        # Labels: -100 for input tokens, actual ids for output
        labels = input_ids.clone()
        labels[:input_len] = -100
        # Also mask padding
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def validate_sympy(expr_str):
    """Check if expression is valid sympy."""
    try:
        sympy.sympify(expr_str)
        return True
    except:
        return False


def compute_metrics(predictions, references):
    """Compute C3-specific metrics."""
    exact_match = 0
    sympy_equiv = 0
    parse_success = 0
    total = len(predictions)

    for pred, ref in zip(predictions, references):
        # Normalize whitespace
        pred = pred.strip()
        ref = ref.strip()

        # Exact match
        if pred == ref:
            exact_match += 1

        # Parse rate
        try:
            pred_expr = sympy.sympify(pred)
            parse_success += 1

            # Sympy equivalence
            try:
                ref_expr = sympy.sympify(ref)
                if sympy.simplify(pred_expr - ref_expr) == 0:
                    sympy_equiv += 1
            except:
                pass
        except:
            pass

    return {
        "exact_match": exact_match / total if total > 0 else 0,
        "sympy_equiv": sympy_equiv / total if total > 0 else 0,
        "parse_rate": parse_success / total if total > 0 else 0,
    }


def load_data_from_s3(s3_path):
    """Load JSONL training data from S3."""
    s3 = boto3.client('s3')
    bucket = s3_path.split('/')[2]
    key = '/'.join(s3_path.split('/')[3:])

    print(f"  Bucket: {bucket}, Key: {key}")
    response = s3.get_object(Bucket=bucket, Key=key)

    examples = []
    for line in response['Body'].iter_lines():
        if line:
            examples.append(json.loads(line.decode('utf-8')))

    return examples


def filter_valid_examples(examples):
    """Filter to examples with valid sympy expressions."""
    valid = []
    invalid_count = 0

    for ex in examples:
        output = ex.get("output", "").strip()
        if not output:
            invalid_count += 1
            continue

        # Try to parse as sympy
        if validate_sympy(output):
            valid.append(ex)
        else:
            invalid_count += 1

    print(f"  Filtered: {len(valid)} valid, {invalid_count} invalid (sympy parse failed)")
    return valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="s3://mycelium-data/c3_training/c3_train_fulltext.jsonl")
    parser.add_argument("--output-dir", default="models/c3_extractor")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--freeze-layers", type=int, default=16,
                        help="Freeze first N transformer layers (Qwen2-0.5B has 24)")
    args = parser.parse_args()

    print("=" * 70)
    print("MYCELIUM V6: C3 EXPRESSION EXTRACTOR")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Freeze layers: 0-{args.freeze_layers-1}")
    sys.stdout.flush()

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    sys.stdout.flush()

    if args.data_path.startswith("s3://"):
        examples = load_data_from_s3(args.data_path)
    else:
        examples = []
        with open(args.data_path) as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples")

    # Filter valid sympy expressions
    print("\nFiltering valid sympy expressions...")
    examples = filter_valid_examples(examples)
    sys.stdout.flush()

    # Show template distribution
    print("\nTemplate distribution:")
    template_counts = Counter(ex.get("template", "UNK") for ex in examples)
    for template, count in template_counts.most_common(15):
        pct = 100 * count / len(examples)
        print(f"  {template:12}: {count:5} ({pct:5.1f}%)")
    sys.stdout.flush()

    # Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    sys.stdout.flush()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Train/val split
    random.seed(42)
    random.shuffle(examples)
    split_idx = int(len(examples) * 0.9)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    print(f"\nTrain: {len(train_examples)}, Val: {len(val_examples)}")

    # Show example
    print("\nExample training sample:")
    ex = train_examples[0]
    print(f"  Input: {ex['input'][:100]}...")
    print(f"  Output: {ex['output']}")
    sys.stdout.flush()

    # Create datasets
    train_dataset = C3Dataset(train_examples, tokenizer, args.max_length)
    val_dataset = C3Dataset(val_examples, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    sys.stdout.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Device: {device}, GPUs available: {n_gpus}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    # Multi-GPU with DataParallel
    if n_gpus > 1:
        print(f"Using DataParallel across {n_gpus} GPUs")
        model = nn.DataParallel(model)

    # Partial layer freezing
    for name, param in model.named_parameters():
        if "embed" in name or "lm_head" in name:
            param.requires_grad = True
        elif "layers." in name:
            layer_num = int(name.split("layers.")[1].split(".")[0])
            param.requires_grad = (layer_num >= args.freeze_layers)
        else:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {trainable:,} / {total:,} trainable ({100*trainable/total:.1f}%)")
    sys.stdout.flush()

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate
    )

    # Training loop
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')
    patience = 3
    patience_counter = 0

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    sys.stdout.flush()

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            if loss.dim() > 0:  # DataParallel returns tensor per GPU
                loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
                sys.stdout.flush()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_refs = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                vloss = outputs.loss
                if vloss.dim() > 0:
                    vloss = vloss.mean()
                val_loss += vloss.item()

                # Generate predictions for first few batches
                if batch_idx < 5:
                    for i in range(min(4, input_ids.size(0))):
                        # Find where input ends (before "Expression:")
                        ex = val_examples[batch_idx * args.batch_size + i]
                        prompt = f"{ex['input']}\nExpression: "

                        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
                        prompt_ids = {k: v.to(device) for k, v in prompt_ids.items()}

                        gen_model = model.module if hasattr(model, 'module') else model
                        gen_ids = gen_model.generate(
                            **prompt_ids,
                            max_new_tokens=50,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )

                        pred = tokenizer.decode(gen_ids[0][prompt_ids["input_ids"].shape[1]:], skip_special_tokens=True)
                        pred = pred.strip().split("\n")[0]

                        all_preds.append(pred)
                        all_refs.append(ex["output"])

        avg_val_loss = val_loss / len(val_loader)

        # Compute metrics on sampled predictions
        metrics = compute_metrics(all_preds, all_refs) if all_preds else {}

        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        if metrics:
            print(f"  Exact Match: {metrics['exact_match']:.4f}")
            print(f"  Sympy Equiv: {metrics['sympy_equiv']:.4f}")
            print(f"  Parse Rate:  {metrics['parse_rate']:.4f}")
        sys.stdout.flush()

        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0

            save_path = f"{args.output_dir}/best"
            Path(save_path).mkdir(parents=True, exist_ok=True)

            # Handle DataParallel wrapper
            save_model = model.module if hasattr(model, 'module') else model
            save_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

            # Save config
            with open(f"{save_path}/config_c3.json", "w") as f:
                json.dump({
                    "model_name": MODEL_NAME,
                    "freeze_layers": args.freeze_layers,
                    "best_val_loss": best_loss,
                    "metrics": metrics,
                }, f, indent=2)

            print(f"  >>> New best model saved (loss: {best_loss:.4f})")
            sys.stdout.flush()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered (patience={patience})")
                break

    # Final inference test
    print("\n" + "=" * 70)
    print("INFERENCE TEST")
    print("=" * 70)
    sys.stdout.flush()

    test_examples = [
        {"input": "[TEMPLATE: ADD] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether?", "expected": "48+24"},
        {"input": "[TEMPLATE: DIV] Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "expected": "50/60"},
        {"input": "[TEMPLATE: MUL] Betty's parents gave her $15 and her grandparents gave her twice as much.", "expected": "15*2"},
    ]

    model.eval()
    gen_model = model.module if hasattr(model, 'module') else model
    for ex in test_examples:
        prompt = f"{ex['input']}\nExpression: "
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = gen_model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        generated = generated.strip().split("\n")[0]

        print(f"\n  Input: ...{ex['input'][-60:]}...")
        print(f"  Expected: {ex['expected']}")
        print(f"  Got:      {generated}")
    sys.stdout.flush()

    # Upload to S3
    print("\n" + "=" * 70)
    print("UPLOADING TO S3")
    print("=" * 70)

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
    print(f"Best Val Loss: {best_loss:.4f}")
    print(f"Model saved to: {args.output_dir}/best")
    print(f"Uploaded to: s3://mycelium-data/models/c3_extractor/")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
