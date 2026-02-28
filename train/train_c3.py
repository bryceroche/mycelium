#!/usr/bin/env python3
"""
Mycelium v6: C3 Per-Operation Expression Extractor (Multi-GPU DDP)

Run with: torchrun --nproc_per_node=8 train_c3.py

Task: Given full problem text + ONE template tag, extract the sympy expression.
Uses DistributedDataParallel for proper multi-GPU training.
"""

import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
import random
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

        # Add EOS token so model learns when to stop generating
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


def filter_valid_examples(examples):
    valid = []
    for ex in examples:
        output = ex.get("output", "").strip()
        if output and validate_sympy(output):
            valid.append(ex)
    return valid


def setup_ddp():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="s3://mycelium-data/c3_training/c3_train_clean.jsonl")
    parser.add_argument("--output-dir", default="models/c3_extractor")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)  # Per GPU
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--freeze-layers", type=int, default=16)
    args = parser.parse_args()

    # Setup DDP
    local_rank = setup_ddp()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    if is_main_process():
        print("=" * 70)
        print("MYCELIUM V6: C3 EXPRESSION EXTRACTOR (8-GPU DDP)")
        print("=" * 70)
        print(f"World size: {world_size} GPUs")
        print(f"Per-GPU batch: {args.batch_size}, Effective batch: {args.batch_size * world_size}")
        sys.stdout.flush()

    # Load data (only on main process, then broadcast)
    if is_main_process():
        print(f"\nLoading data from {args.data_path}...")
        examples = load_data_from_s3(args.data_path)
        print(f"Loaded {len(examples)} examples")
        print("Filtering valid sympy expressions...")
        examples = filter_valid_examples(examples)
        print(f"Valid: {len(examples)}")
        sys.stdout.flush()
    else:
        examples = None

    # Broadcast examples to all processes
    examples = [examples]
    dist.broadcast_object_list(examples, src=0)
    examples = examples[0]

    # Load tokenizer
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

    if is_main_process():
        print(f"\nTrain: {len(train_examples)}, Val: {len(val_examples)}")
        sys.stdout.flush()

    # Create datasets
    train_dataset = C3Dataset(train_examples, tokenizer, args.max_length)
    val_dataset = C3Dataset(val_examples, tokenizer, args.max_length)

    # Distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)

    # Load model
    if is_main_process():
        print(f"\nLoading model: {MODEL_NAME}")
        sys.stdout.flush()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Partial layer freezing
    for name, param in model.named_parameters():
        if "embed" in name or "lm_head" in name:
            param.requires_grad = True
        elif "layers." in name:
            layer_num = int(name.split("layers.")[1].split(".")[0])
            param.requires_grad = (layer_num >= args.freeze_layers)
        else:
            param.requires_grad = True

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_main_process():
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Params: {trainable:,} / {total:,} trainable ({100*trainable/total:.1f}%)")
        sys.stdout.flush()

    # Optimizer
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)

    # Training loop
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')
    patience = 3
    patience_counter = 0

    if is_main_process():
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        sys.stdout.flush()

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

            if is_main_process() and (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
                sys.stdout.flush()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

        # Average across processes
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Reduce losses across all GPUs
        train_loss_tensor = torch.tensor(avg_train_loss, device=device)
        val_loss_tensor = torch.tensor(avg_val_loss, device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)

        if is_main_process():
            print(f"\nEpoch {epoch+1}/{args.epochs}:")
            print(f"  Train Loss: {train_loss_tensor.item():.4f}")
            print(f"  Val Loss:   {val_loss_tensor.item():.4f}")
            sys.stdout.flush()

            # Save best model
            if val_loss_tensor.item() < best_loss:
                best_loss = val_loss_tensor.item()
                patience_counter = 0

                save_path = f"{args.output_dir}/best"
                Path(save_path).mkdir(parents=True, exist_ok=True)
                model.module.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"  >>> New best model saved (loss: {best_loss:.4f})")
                sys.stdout.flush()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered")
                    break

        dist.barrier()

    # Upload to S3 (main process only)
    if is_main_process():
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
        print(f"Best Val Loss: {best_loss:.4f}")
        print(f"Model: s3://mycelium-data/models/c3_extractor/")
        sys.stdout.flush()

    cleanup_ddp()


if __name__ == "__main__":
    main()
