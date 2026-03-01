#!/usr/bin/env python3
"""
Mycelium v6: C3-Extract Span Extraction Training

Trains the span extraction model to find operand spans in problem text.
Uses the reformatted C3 training data with span annotations.

Run with: torchrun --nproc_per_node=8 train_c3_extract.py
Or single GPU: python train_c3_extract.py
"""

import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import random
import argparse
import sys
import os
import boto3
import warnings
warnings.filterwarnings("ignore")

# Add parent to path for model import
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.c3_extract import C3ExtractModel, C3ExtractDataset, compute_loss, MAX_OPERANDS

MODEL_NAME = "Qwen/Qwen2-0.5B"


def load_data_from_s3(s3_path):
    """Load JSONL data from S3."""
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
    """
    Filter to examples where at least one operand was found in text.
    Skip examples where all operands are 'generated' (can't extract from text).
    """
    valid = []
    for ex in examples:
        spans = ex.get('spans', [])
        # Check if at least one operand has a valid span (not generated)
        has_extractable = any(
            s.get('source') not in ['generated', 'domain_constant']
            and s.get('span_start', -1) >= 0
            for s in spans
        )
        if has_extractable:
            valid.append(ex)
    return valid


def setup_ddp():
    """Initialize distributed training."""
    if 'LOCAL_RANK' in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, True
    return 0, False


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def evaluate(model, dataloader, device):
    """Evaluate model and compute metrics."""
    model.eval()
    total_loss = 0
    total_correct_starts = 0
    total_correct_ends = 0
    total_operands = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            operand_mask = batch['operand_mask'].to(device)
            num_operands = batch['num_operands'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = compute_loss(outputs, start_positions, end_positions, operand_mask, num_operands)
            total_loss += loss.item()

            # Compute span accuracy
            pred_starts = outputs['start_logits'].argmax(dim=-1)  # (batch, max_ops)
            pred_ends = outputs['end_logits'].argmax(dim=-1)

            for i in range(MAX_OPERANDS):
                mask = operand_mask[:, i].bool()
                if mask.sum() > 0:
                    correct_starts = (pred_starts[:, i][mask] == start_positions[:, i][mask]).sum().item()
                    correct_ends = (pred_ends[:, i][mask] == end_positions[:, i][mask]).sum().item()
                    total_correct_starts += correct_starts
                    total_correct_ends += correct_ends
                    total_operands += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    start_acc = total_correct_starts / max(total_operands, 1)
    end_acc = total_correct_ends / max(total_operands, 1)

    return {
        'loss': avg_loss,
        'start_acc': start_acc,
        'end_acc': end_acc,
        'span_acc': (start_acc + end_acc) / 2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="s3://mycelium-data/c3_span_training/c3_span_train.jsonl")
    parser.add_argument("--output-dir", default="models/c3_extract")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--freeze-layers", type=int, default=12)
    args = parser.parse_args()

    # Setup DDP
    local_rank, is_distributed = setup_ddp()
    world_size = dist.get_world_size() if is_distributed else 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        print("=" * 70)
        print("MYCELIUM V6: C3-EXTRACT SPAN EXTRACTION TRAINING")
        print("=" * 70)
        if is_distributed:
            print(f"World size: {world_size} GPUs")
        print(f"Per-GPU batch: {args.batch_size}, Effective batch: {args.batch_size * world_size}")
        sys.stdout.flush()

    # Load data
    if is_main_process():
        print(f"\nLoading data from {args.data_path}...")
        if args.data_path.startswith("s3://"):
            examples = load_data_from_s3(args.data_path)
        else:
            with open(args.data_path) as f:
                examples = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {len(examples)} examples")

        print("Filtering to extractable examples...")
        examples = filter_valid_examples(examples)
        print(f"Valid: {len(examples)}")
        sys.stdout.flush()
    else:
        examples = None

    # Broadcast examples in distributed mode
    if is_distributed:
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

        # Show sample
        sample = train_examples[0]
        print(f"\nSample training example:")
        print(f"  Template: {sample['template']}")
        print(f"  Expression: {sample['expression']}")
        print(f"  Operands: {sample['operands']}")
        print(f"  Spans: {len([s for s in sample['spans'] if s.get('source') != 'generated'])} extractable")
        sys.stdout.flush()

    # Create datasets
    train_dataset = C3ExtractDataset(train_examples, tokenizer, args.max_length)
    val_dataset = C3ExtractDataset(val_examples, tokenizer, args.max_length)

    # Data loaders
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2,
        sampler=val_sampler, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Load model
    if is_main_process():
        print(f"\nLoading model: {MODEL_NAME}")
        sys.stdout.flush()

    model = C3ExtractModel(backbone_name=MODEL_NAME)

    # Partial layer freezing
    for name, param in model.named_parameters():
        if "backbone" in name:
            if "embed" in name:
                param.requires_grad = True
            elif "layers." in name:
                layer_num = int(name.split("layers.")[1].split(".")[0])
                param.requires_grad = (layer_num >= args.freeze_layers)
            else:
                param.requires_grad = True
        else:
            # Always train extraction heads
            param.requires_grad = True

    model = model.to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_main_process():
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Params: {trainable:,} / {total:,} trainable ({100*trainable/total:.1f}%)")
        sys.stdout.flush()

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01
    )

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
        if is_distributed:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            operand_mask = batch['operand_mask'].to(device)
            num_operands = batch['num_operands'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = compute_loss(outputs, start_positions, end_positions, operand_mask, num_operands)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

            if is_main_process() and (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
                sys.stdout.flush()

        # Validation
        val_metrics = evaluate(model, val_loader, device)

        # Reduce metrics across GPUs
        if is_distributed:
            for key in ['loss', 'start_acc', 'end_acc', 'span_acc']:
                tensor = torch.tensor(val_metrics[key], device=device)
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
                val_metrics[key] = tensor.item()

        avg_train_loss = train_loss / len(train_loader)

        if is_main_process():
            print(f"\nEpoch {epoch+1}/{args.epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  Start Acc:  {val_metrics['start_acc']:.3f}")
            print(f"  End Acc:    {val_metrics['end_acc']:.3f}")
            print(f"  Span Acc:   {val_metrics['span_acc']:.3f}")
            sys.stdout.flush()

            # Save best model
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                patience_counter = 0

                save_path = f"{args.output_dir}/best"
                Path(save_path).mkdir(parents=True, exist_ok=True)

                # Save model
                model_to_save = model.module if is_distributed else model
                torch.save({
                    'model_state_dict': model_to_save.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_metrics['loss'],
                    'val_span_acc': val_metrics['span_acc'],
                }, f"{save_path}/model.pt")

                # Save tokenizer
                tokenizer.save_pretrained(save_path)

                # Save config
                config = {
                    'model_name': MODEL_NAME,
                    'max_length': args.max_length,
                    'freeze_layers': args.freeze_layers,
                }
                with open(f"{save_path}/config.json", 'w') as f:
                    json.dump(config, f)

                print(f"  >>> New best model saved (loss: {best_loss:.4f}, span_acc: {val_metrics['span_acc']:.3f})")
                sys.stdout.flush()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered")
                    break

        if is_distributed:
            dist.barrier()

    # Upload to S3 (main process only)
    if is_main_process():
        print("\nUploading to S3...")
        s3 = boto3.client('s3')
        save_path = f"{args.output_dir}/best"

        for fname in Path(save_path).iterdir():
            if fname.is_file():
                s3_key = f"models/c3_extract/{fname.name}"
                s3.upload_file(str(fname), "mycelium-data", s3_key)
                print(f"  Uploaded {fname.name}")

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Best Val Loss: {best_loss:.4f}")
        print(f"Model: s3://mycelium-data/models/c3_extract/")
        sys.stdout.flush()

    cleanup_ddp()


if __name__ == "__main__":
    main()
