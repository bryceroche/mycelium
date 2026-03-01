#!/usr/bin/env python3
"""
C3-Pointer Training Script

Trains the pointer model to select operand sources from a closed set:
- TEXT_<position> - operand from problem text
- PRIOR_<N> - result from prior computation step
- IMPLICIT_<word> - implied value ("half"→0.5, etc.)
- CONSTANT_<value> - domain constant

Run: python train/train_c3_pointer.py
"""

import json
import torch
import torch.nn as nn
import boto3
import random
from pathlib import Path
from transformers import AutoTokenizer
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.c3_pointer import C3PointerModel, IMPLICIT_VALUES, DOMAIN_CONSTANTS, MAX_OPERANDS

MODEL_NAME = "Qwen/Qwen2-0.5B"

# Source type indices
SOURCE_TYPES = ['TEXT', 'PRIOR', 'IMPLICIT', 'CONSTANT', 'NONE']
SOURCE_TYPE_TO_IDX = {s: i for i, s in enumerate(SOURCE_TYPES)}

# Implicit and constant indices
IMPLICIT_KEYS = list(IMPLICIT_VALUES.keys())
IMPLICIT_TO_IDX = {k: i for i, k in enumerate(IMPLICIT_KEYS)}
CONSTANT_KEYS = list(DOMAIN_CONSTANTS.keys())
CONSTANT_TO_IDX = {k: i for i, k in enumerate(CONSTANT_KEYS)}


class C3PointerDataset(torch.utils.data.Dataset):
    """Dataset for C3-Pointer training."""

    def __init__(self, examples: list, tokenizer, max_length: int = 384):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        problem_text = ex['problem_text']
        template = ex['template']
        provenance = ex['provenance']

        # Build input text
        input_text = f"[TEMPLATE: {template}] {problem_text}"
        prefix_len = len(f"[TEMPLATE: {template}] ")

        # Tokenize with offset mapping
        encodings = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
            return_offsets_mapping=True,
        )

        input_ids = encodings['input_ids'].squeeze(0)
        attention_mask = encodings['attention_mask'].squeeze(0)
        offset_mapping = encodings['offset_mapping'].squeeze(0)

        # Build targets
        source_type_targets = torch.zeros(MAX_OPERANDS, dtype=torch.long)
        text_pointer_targets = torch.zeros(MAX_OPERANDS, dtype=torch.long)
        implicit_targets = torch.zeros(MAX_OPERANDS, dtype=torch.long)
        constant_targets = torch.zeros(MAX_OPERANDS, dtype=torch.long)
        operand_mask = torch.zeros(MAX_OPERANDS, dtype=torch.float)

        for i, prov in enumerate(provenance[:MAX_OPERANDS]):
            operand_mask[i] = 1.0
            source_type = prov['source_type']
            source_type_targets[i] = SOURCE_TYPE_TO_IDX.get(source_type, 0)

            if source_type == 'TEXT':
                # Convert char position to token position
                char_start = prov.get('char_start', 0)
                # Adjust for prefix
                adj_char_start = char_start + prefix_len

                token_pos = 0
                for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping.tolist()):
                    if tok_start <= adj_char_start < tok_end:
                        token_pos = tok_idx
                        break

                text_pointer_targets[i] = token_pos

            elif source_type == 'IMPLICIT':
                word = prov.get('word', 'half')
                implicit_targets[i] = IMPLICIT_TO_IDX.get(word, 0)

            elif source_type == 'CONSTANT':
                # Find constant index
                value = prov.get('value', 100)
                if isinstance(value, float):
                    value = int(value)
                constant_targets[i] = CONSTANT_TO_IDX.get(value, 0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'source_type_targets': source_type_targets,
            'text_pointer_targets': text_pointer_targets,
            'implicit_targets': implicit_targets,
            'constant_targets': constant_targets,
            'operand_mask': operand_mask,
            'n_operands': len(provenance),
        }


def compute_loss(outputs, batch, device):
    """Compute combined loss for pointer model."""
    source_type_logits = outputs['source_type_logits']  # (batch, max_ops, 5)
    text_pointer_logits = outputs['text_pointer_logits']  # (batch, max_ops, seq_len)
    implicit_logits = outputs['implicit_logits']
    constant_logits = outputs['constant_logits']

    source_type_targets = batch['source_type_targets'].to(device)
    text_pointer_targets = batch['text_pointer_targets'].to(device)
    implicit_targets = batch['implicit_targets'].to(device)
    constant_targets = batch['constant_targets'].to(device)
    operand_mask = batch['operand_mask'].to(device)

    batch_size = source_type_targets.shape[0]
    total_loss = 0.0
    n_valid = 0

    for i in range(MAX_OPERANDS):
        mask = operand_mask[:, i]
        if mask.sum() == 0:
            continue

        # Source type loss
        source_type_ce = nn.functional.cross_entropy(
            source_type_logits[:, i, :][mask == 1],
            source_type_targets[:, i][mask == 1],
            reduction='sum'
        )
        total_loss += source_type_ce

        # Text pointer loss (only for TEXT source type)
        text_mask = (source_type_targets[:, i] == SOURCE_TYPE_TO_IDX['TEXT']) & (mask == 1)
        if text_mask.sum() > 0:
            text_ptr_ce = nn.functional.cross_entropy(
                text_pointer_logits[:, i, :][text_mask],
                text_pointer_targets[:, i][text_mask],
                reduction='sum'
            )
            total_loss += text_ptr_ce

        # Implicit loss (only for IMPLICIT source type)
        impl_mask = (source_type_targets[:, i] == SOURCE_TYPE_TO_IDX['IMPLICIT']) & (mask == 1)
        if impl_mask.sum() > 0:
            impl_ce = nn.functional.cross_entropy(
                implicit_logits[:, i, :][impl_mask],
                implicit_targets[:, i][impl_mask],
                reduction='sum'
            )
            total_loss += impl_ce

        # Constant loss (only for CONSTANT source type)
        const_mask = (source_type_targets[:, i] == SOURCE_TYPE_TO_IDX['CONSTANT']) & (mask == 1)
        if const_mask.sum() > 0:
            const_ce = nn.functional.cross_entropy(
                constant_logits[:, i, :][const_mask],
                constant_targets[:, i][const_mask],
                reduction='sum'
            )
            total_loss += const_ce

        n_valid += mask.sum().item()

    if n_valid > 0:
        total_loss = total_loss / n_valid

    return total_loss


def evaluate(model, dataloader, device):
    """Evaluate model and compute metrics."""
    model.eval()
    total_loss = 0
    total_source_correct = 0
    total_text_ptr_correct = 0
    total_operands = 0
    total_text_operands = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = compute_loss(outputs, batch, device)
            total_loss += loss.item()

            # Compute accuracies
            source_type_preds = outputs['source_type_logits'].argmax(dim=-1)
            text_ptr_preds = outputs['text_pointer_logits'].argmax(dim=-1)

            source_type_targets = batch['source_type_targets'].to(device)
            text_pointer_targets = batch['text_pointer_targets'].to(device)
            operand_mask = batch['operand_mask'].to(device)

            for i in range(MAX_OPERANDS):
                mask = operand_mask[:, i].bool()
                if mask.sum() > 0:
                    # Source type accuracy
                    correct = (source_type_preds[:, i][mask] == source_type_targets[:, i][mask]).sum().item()
                    total_source_correct += correct
                    total_operands += mask.sum().item()

                    # Text pointer accuracy (only for TEXT type)
                    text_mask = (source_type_targets[:, i] == SOURCE_TYPE_TO_IDX['TEXT']) & mask
                    if text_mask.sum() > 0:
                        text_correct = (text_ptr_preds[:, i][text_mask] == text_pointer_targets[:, i][text_mask]).sum().item()
                        total_text_ptr_correct += text_correct
                        total_text_operands += text_mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    source_acc = total_source_correct / max(total_operands, 1)
    text_ptr_acc = total_text_ptr_correct / max(total_text_operands, 1)

    return {
        'loss': avg_loss,
        'source_acc': source_acc,
        'text_ptr_acc': text_ptr_acc,
    }


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="s3://mycelium-data/c3_pointer_training/track_a_b_pointer.jsonl")
    parser.add_argument("--output-dir", default="models/c3_pointer")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--freeze-layers", type=int, default=12)
    args = parser.parse_args()

    # DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1

    if is_ddp:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = local_rank == 0
    if is_main:
        print(f"Using device: {device}, DDP: {is_ddp}, World size: {world_size}")

    if is_main:
        print("=" * 70)
        print("C3-POINTER TRAINING (DDP)" if is_ddp else "C3-POINTER TRAINING")
        print("=" * 70)
        print(f"\nLoading data from {args.data_path}...")
    if args.data_path.startswith("s3://"):
        s3 = boto3.client('s3')
        bucket = args.data_path.split('/')[2]
        key = '/'.join(args.data_path.split('/')[3:])
        response = s3.get_object(Bucket=bucket, Key=key)
        examples = [json.loads(line) for line in response['Body'].iter_lines() if line]
    else:
        with open(args.data_path) as f:
            examples = [json.loads(line) for line in f if line.strip()]

    if is_main:
        print(f"Loaded {len(examples)} examples")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Split data
    random.seed(42)
    random.shuffle(examples)
    split_idx = int(len(examples) * 0.9)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    if is_main:
        print(f"Train: {len(train_examples)}, Val: {len(val_examples)}")

    # Create datasets
    train_dataset = C3PointerDataset(train_examples, tokenizer, args.max_length)
    val_dataset = C3PointerDataset(val_examples, tokenizer, args.max_length)

    # Use DistributedSampler for DDP
    if is_ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size * 2, sampler=val_sampler, num_workers=4, pin_memory=True
        )
    else:
        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True
        )

    # Load model
    if is_main:
        print(f"\nLoading model: {MODEL_NAME}")
    model = C3PointerModel(backbone_name=MODEL_NAME)

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
            param.requires_grad = True

    model = model.to(device)

    # Wrap in DDP
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"Params: {trainable:,} / {total:,} trainable ({100*trainable/total:.1f}%)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01
    )

    # Training loop
    if is_main:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')
    patience = 3
    patience_counter = 0

    if is_main:
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)

    for epoch in range(args.epochs):
        # Set epoch for sampler (important for DDP shuffling)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = compute_loss(outputs, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

            if is_main and (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Validation (only on main process for DDP)
        if is_main:
            # Unwrap model for evaluation if DDP
            eval_model = model.module if is_ddp else model
            val_metrics = evaluate(eval_model, val_loader, device)
            avg_train_loss = train_loss / len(train_loader)

            print(f"\nEpoch {epoch+1}/{args.epochs}:")
            print(f"  Train Loss:    {avg_train_loss:.4f}")
            print(f"  Val Loss:      {val_metrics['loss']:.4f}")
            print(f"  Source Acc:    {val_metrics['source_acc']:.3f}")
            print(f"  Text Ptr Acc:  {val_metrics['text_ptr_acc']:.3f}")

            # Save best model
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                patience_counter = 0

                save_path = f"{args.output_dir}/best"
                Path(save_path).mkdir(parents=True, exist_ok=True)

                # Save unwrapped model state
                model_to_save = model.module if is_ddp else model
                torch.save({
                    'model_state_dict': model_to_save.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_metrics['loss'],
                    'source_acc': val_metrics['source_acc'],
                    'text_ptr_acc': val_metrics['text_ptr_acc'],
                }, f"{save_path}/model.pt")

                tokenizer.save_pretrained(save_path)

                print(f"  >>> New best model saved (loss: {best_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered")
                    break

    # Upload to S3 (only main process)
    if is_main:
        print("\nUploading to S3...")
        s3 = boto3.client('s3')
        save_path = f"{args.output_dir}/best"

        for fname in Path(save_path).iterdir():
            if fname.is_file():
                s3_key = f"models/c3_pointer/{fname.name}"
                s3.upload_file(str(fname), "mycelium-data", s3_key)
                print(f"  Uploaded {fname.name}")

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Best Val Loss: {best_loss:.4f}")
        print(f"Model: s3://mycelium-data/models/c3_pointer/")

    # Clean up DDP
    if is_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
