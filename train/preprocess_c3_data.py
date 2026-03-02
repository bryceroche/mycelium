#!/usr/bin/env python3
"""
Preprocess C3 training data to tokenized tensors.

Loads raw jsonl, tokenizes with RoBERTa, converts char->token positions,
saves as .pt file for fast loading during training.
"""

import json
import torch
import boto3
import argparse
import re
from pathlib import Path
from transformers import AutoTokenizer

MAX_OPERANDS = 4
MAX_SEQ_LEN = 384

MODELS = {
    'roberta': 'deepset/roberta-base-squad2',
    'roberta-large': 'deepset/roberta-large-squad2',
}


def char_to_token_position(char_pos, offset_mapping):
    for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start <= char_pos < tok_end:
            return tok_idx
    for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start >= char_pos:
            return max(0, tok_idx - 1)
    return len(offset_mapping) - 1


def reformat_example(ex, tokenizer, max_length=MAX_SEQ_LEN):
    """Convert raw example to tokenized tensors."""
    # New format: input_text already has [TEMPLATE: ...] and spans array
    if 'input_text' in ex and 'spans' in ex:
        input_text = ex['input_text']
        spans = ex['spans']
        if not spans:
            return None

        enc = tokenizer(
            input_text,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
            return_offsets_mapping=True
        )
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        offset_mapping = enc['offset_mapping'].squeeze(0).tolist()

        start_targets = torch.zeros(MAX_OPERANDS, dtype=torch.long)
        end_targets = torch.zeros(MAX_OPERANDS, dtype=torch.long)
        operand_mask = torch.zeros(MAX_OPERANDS, dtype=torch.float)

        valid = 0
        for i, span in enumerate(spans[:MAX_OPERANDS]):
            cs, ce = span.get('span_start', -1), span.get('span_end', -1)
            if cs < 0 or ce < 0:
                continue
            s_tok = char_to_token_position(cs, offset_mapping)
            e_tok = char_to_token_position(ce - 1, offset_mapping)
            if s_tok >= max_length or e_tok >= max_length:
                continue
            if e_tok < s_tok:
                e_tok = s_tok
            start_targets[i] = s_tok
            end_targets[i] = e_tok
            operand_mask[i] = 1.0
            valid += 1

        if valid == 0:
            return None

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'start_targets': start_targets,
            'end_targets': end_targets,
            'operand_mask': operand_mask
        }

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=['roberta', 'roberta-large'], default='roberta-large')
    parser.add_argument("--input-path", default="s3://mycelium-data/c3_span_training/c3_train_with_priors.jsonl")
    parser.add_argument("--output-path", default="s3://mycelium-data/c3_span_training/c3_train_roberta_large.pt")
    parser.add_argument("--local-cache", default="/tmp/c3_preprocessed.pt")
    args = parser.parse_args()

    model_name = MODELS[args.model]
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load raw data
    print(f"Loading data from: {args.input_path}")
    if args.input_path.startswith("s3://"):
        s3 = boto3.client('s3')
        bucket = args.input_path.split('/')[2]
        key = '/'.join(args.input_path.split('/')[3:])
        response = s3.get_object(Bucket=bucket, Key=key)
        raw = [json.loads(line) for line in response['Body'].iter_lines() if line]
    else:
        with open(args.input_path) as f:
            raw = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(raw)} raw examples")

    # Preprocess all examples
    print("Tokenizing and converting to tensors...")
    processed = []
    for i, ex in enumerate(raw):
        result = reformat_example(ex, tokenizer)
        if result is not None:
            processed.append(result)
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{len(raw)} ({len(processed)} valid)")

    print(f"Valid examples: {len(processed)}/{len(raw)} ({100*len(processed)/len(raw):.1f}%)")

    # Stack into single tensors
    print("Stacking tensors...")
    data = {
        'input_ids': torch.stack([ex['input_ids'] for ex in processed]),
        'attention_mask': torch.stack([ex['attention_mask'] for ex in processed]),
        'start_targets': torch.stack([ex['start_targets'] for ex in processed]),
        'end_targets': torch.stack([ex['end_targets'] for ex in processed]),
        'operand_mask': torch.stack([ex['operand_mask'] for ex in processed]),
    }

    print(f"Tensor shapes:")
    for k, v in data.items():
        print(f"  {k}: {v.shape}")

    # Save locally first
    print(f"Saving to local cache: {args.local_cache}")
    torch.save(data, args.local_cache)

    # Upload to S3
    if args.output_path.startswith("s3://"):
        print(f"Uploading to: {args.output_path}")
        bucket = args.output_path.split('/')[2]
        key = '/'.join(args.output_path.split('/')[3:])
        s3.upload_file(args.local_cache, bucket, key)
        print("Upload complete!")
    else:
        import shutil
        shutil.copy(args.local_cache, args.output_path)

    print(f"\nDone! Preprocessed data saved to: {args.output_path}")
    print(f"Total examples: {len(processed)}")
    print(f"File size: {Path(args.local_cache).stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
