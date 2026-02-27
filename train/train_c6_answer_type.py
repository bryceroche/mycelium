#!/usr/bin/env python3
"""
C6: Answer Type Classification Training

Qwen-0.5B-Instruct with frozen backbone, multi-class classification head.
Input: problem text
Output: INTEGER, FRACTION, DECIMAL, SET, EXPRESSION (5 classes)

Uses fp16, batch_size=64, train for 5 epochs.
Save best checkpoint by F1 score.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from sklearn.metrics import f1_score, classification_report
import json
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import boto3

# Labels from MATH dataset extraction
LABELS = ["integer", "fraction", "decimal", "expression", "radical", "other", "letter"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


class C6Dataset(Dataset):
    """Dataset for answer type classification."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # Load data (can be local file or S3 path)
        if data_path.startswith('s3://'):
            s3 = boto3.client('s3')
            bucket = data_path.split('/')[2]
            prefix = '/'.join(data_path.split('/')[3:])

            # List all files
            paginator = s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if key.endswith('.json'):
                        response = s3.get_object(Bucket=bucket, Key=key)
                        chunk_data = json.loads(response['Body'].read().decode('utf-8'))
                        self.examples.extend(chunk_data)
        else:
            # Local file or directory
            path = Path(data_path)
            if path.is_dir():
                for f in path.glob('*.json'):
                    with open(f) as fp:
                        self.examples.extend(json.load(fp))
            else:
                with open(path) as f:
                    self.examples = json.load(f)

        # Filter out invalid examples
        orig_count = len(self.examples)
        self.examples = [ex for ex in self.examples if ex.get('label') in LABELS]
        print(f"Loaded {len(self.examples)} examples (filtered {orig_count - len(self.examples)} invalid)")

        # Print label distribution and store for class weights
        label_counts = {}
        for ex in self.examples:
            label = ex.get('label', 'other')
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"Label distribution: {label_counts}")

        # Store counts in label order for class weight computation
        self.class_counts = [label_counts.get(l, 0) for l in LABELS]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Input is the problem text (supports both 'text' and 'problem' fields)
        text = ex.get('text', ex.get('problem', ex.get('question', '')))

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        label = LABEL2ID.get(ex.get('label', ex.get('answer_type', 'integer')), 0)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train_epoch(model, dataloader, optimizer, scheduler, device, class_weights):
    """Train for one epoch (fp32) with weighted cross-entropy."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    # Weighted loss for class imbalance
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': loss.item()})

    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / len(dataloader), f1


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)

            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=LABELS))

    return f1, acc


def main():
    parser = argparse.ArgumentParser(description='Train C6 answer type classifier')
    parser.add_argument('--data', default='s3://mycelium-data/training_data/c456/c6/',
                       help='Training data path')
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct',
                       help='Base model')
    parser.add_argument('--output', default='./checkpoints/c6', help='Output directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for head')
    parser.add_argument('--max-length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    args = parser.parse_args()

    # Setup
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.output, exist_ok=True)

    # Load tokenizer and model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        trust_remote_code=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Freeze backbone, only train classification head
    print("Freezing backbone...")
    for name, param in model.named_parameters():
        if 'classifier' not in name and 'score' not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    model.float().to(device)  # Convert to fp32

    # Load data
    print(f"Loading data from: {args.data}")
    dataset = C6Dataset(args.data, tokenizer, args.max_length)

    # Compute class weights for imbalanced data (clipped at 5x max)
    total = sum(dataset.class_counts)
    num_classes = len(LABELS)
    class_weights = []
    for count in dataset.class_counts:
        if count > 0:
            w = min(total / (num_classes * count), 5.0)  # Clip at 5x
        else:
            w = 1.0
        class_weights.append(w)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights (clipped 5x): {dict(zip(LABELS, [f'{w:.2f}' for w in class_weights.tolist()]))}")

    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)

    # Optimizer (only for trainable params)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr
    )

    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_scheduler('linear', optimizer, num_warmup_steps=100,
                             num_training_steps=num_training_steps)

    # Training loop
    best_f1 = 0

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")

        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler,
                                           device, class_weights)
        val_f1, val_acc = evaluate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model by F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            print(f"New best F1! Saving to {args.output}")
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)

    print(f"\nTraining complete. Best F1: {best_f1:.4f}")

    # Upload to S3
    print("Uploading to S3...")
    s3 = boto3.client('s3')
    for f in Path(args.output).glob('*'):
        s3.upload_file(str(f), 'mycelium-data', f'models/c6/{f.name}')
    print("Done!")


if __name__ == '__main__':
    main()
