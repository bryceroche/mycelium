#!/usr/bin/env python3
"""
Mycelium v6: Train C1 Multi-Step Relevance (Option A)

Multi-channel output where each channel = one computation step's relevance map.
This fixes the "flat labels" problem - instead of ~90% tokens being "relevant",
each channel has sparse peaks (3-10 tokens per step).

Architecture:
  - Input: problem text
  - Output: (batch, seq_len, N_CHANNELS) relevance scores
  - Each channel is trained to predict one step's attention pattern
  - At inference, active channels have peaks, inactive channels are ~0

Loss: Weighted MSE per channel
  - 10x weight on hot tokens (>0.3) for recall
  - 0.1x weight on inactive channels (step > n_active) to not penalize zeros
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
)
from transformers.modeling_outputs import TokenClassifierOutput
import random
import argparse

N_CHANNELS = 8  # Must match extraction script
MODEL_NAME = "Qwen/Qwen2-0.5B"


class Qwen2ForMultiStepRelevance(PreTrainedModel):
    """Qwen2 with multi-channel token-level regression for per-step relevance."""

    def __init__(self, config):
        super().__init__(config)
        self.n_channels = N_CHANNELS

        self.qwen = AutoModel.from_pretrained(
            config._name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        self.qwen = self.qwen.float()
        self.dropout = nn.Dropout(0.1)

        # Multi-channel output: one relevance map per step
        self.regressor = nn.Linear(config.hidden_size, N_CHANNELS)

        # Initialize
        self.regressor.weight.data.normal_(mean=0.0, std=0.02)
        self.regressor.bias.data.zero_()

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                n_active_steps=None, **kwargs):
        """
        Args:
            labels: (batch, seq_len, N_CHANNELS) per-step relevance targets
            n_active_steps: (batch,) number of active steps per example
        """
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        # Output: (batch, seq_len, N_CHANNELS)
        predictions = self.regressor(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss(reduction='none')

            # Mask padding tokens (where all channels are -100)
            mask = (labels[:, :, 0] != -100).float().unsqueeze(-1)  # (batch, seq, 1)

            # Replace -100 with 0 for computation
            labels_safe = labels.clone()
            labels_safe[labels == -100] = 0

            # Per-token, per-channel loss
            per_elem_loss = loss_fct(predictions, labels_safe)  # (batch, seq, channels)

            # Weight hot tokens 10x (per channel)
            hot_weight = torch.where(labels_safe > 0.3, 10.0, 1.0)

            # Down-weight inactive channels (if n_active_steps provided)
            if n_active_steps is not None:
                # Create channel mask: 1.0 for active, 0.1 for inactive
                channel_mask = torch.ones_like(per_elem_loss)
                for b in range(per_elem_loss.shape[0]):
                    n_active = n_active_steps[b].item() if n_active_steps.dim() > 0 else n_active_steps.item()
                    if n_active < N_CHANNELS:
                        channel_mask[b, :, int(n_active):] = 0.1
                hot_weight = hot_weight * channel_mask

            weighted_loss = per_elem_loss * hot_weight * mask

            # Normalize
            loss = weighted_loss.sum() / (hot_weight * mask).sum()

        return TokenClassifierOutput(
            loss=loss,
            logits=predictions,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def load_data(data_path: str):
    """Load multi-step relevance data."""
    with open(data_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples")

    # Filter examples with at least one active step
    data = [d for d in data if d.get('n_active_steps', 0) > 0]
    print(f"After filtering: {len(data)}")

    return data


def prepare_dataset(data: list, tokenizer, max_length=256):
    """Prepare dataset for training."""
    processed = []

    for example in data:
        encoding = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        input_ids = encoding["input_ids"]
        seq_len = len(input_ids)

        # Get step labels and align to tokenization
        step_labels = example["step_labels"]  # (N_CHANNELS, problem_len)
        orig_len = example.get("problem_len", len(step_labels[0]))

        # Align each channel's labels to tokenized length
        aligned_labels = []
        for channel_labels in step_labels:
            if len(channel_labels) == seq_len:
                aligned = channel_labels[:seq_len]
            else:
                # Proportional mapping
                aligned = []
                for i in range(seq_len):
                    orig_pos = int(round(i * (orig_len - 1) / (seq_len - 1))) if seq_len > 1 else 0
                    orig_pos = min(orig_pos, len(channel_labels) - 1)
                    aligned.append(channel_labels[orig_pos])
            aligned_labels.append(aligned)

        # Pad to max_length
        pad_len = max_length - seq_len
        if pad_len > 0:
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            aligned_labels = [ch + [-100.0] * pad_len for ch in aligned_labels]

        # Transpose: (N_CHANNELS, seq_len) -> (seq_len, N_CHANNELS)
        labels_transposed = [[aligned_labels[c][t] for c in range(N_CHANNELS)]
                            for t in range(max_length)]

        processed.append({
            "input_ids": input_ids,
            "attention_mask": [1] * (max_length - pad_len) + [0] * pad_len,
            "labels": labels_transposed,
            "n_active_steps": example.get("n_active_steps", 1),
        })

    return Dataset.from_list(processed)


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred  # (N, seq, channels), (N, seq, channels)

    # Flatten valid tokens
    pred_flat = []
    label_flat = []
    channel_stats = {c: {'pred': [], 'label': []} for c in range(N_CHANNELS)}

    for pred_seq, label_seq in zip(predictions, labels):
        for t, (p_vec, l_vec) in enumerate(zip(pred_seq, label_seq)):
            if l_vec[0] != -100:  # Valid token
                for c in range(N_CHANNELS):
                    if l_vec[c] != -100:
                        channel_stats[c]['pred'].append(p_vec[c])
                        channel_stats[c]['label'].append(l_vec[c])
                        pred_flat.append(p_vec[c])
                        label_flat.append(l_vec[c])

    pred_flat = np.array(pred_flat)
    label_flat = np.array(label_flat)

    # Global MSE
    mse = np.mean((pred_flat - label_flat) ** 2)

    # Per-channel top-k recall (of tokens with label > 0.5, what fraction have pred > 0.3?)
    recalls = []
    for c in range(N_CHANNELS):
        c_pred = np.array(channel_stats[c]['pred'])
        c_label = np.array(channel_stats[c]['label'])
        if len(c_label) > 0:
            hot_mask = c_label > 0.5
            if hot_mask.sum() > 0:
                recall = np.mean(c_pred[hot_mask] > 0.3)
                recalls.append(recall)

    avg_recall = np.mean(recalls) if recalls else 0.0

    # Sparsity: fraction of predictions > 0.3 (should be low ~5-15%)
    sparsity = np.mean(pred_flat > 0.3)

    return {
        "mse": mse,
        "avg_channel_recall": avg_recall,
        "sparsity": sparsity,
    }


class MultiStepDataCollator:
    """Custom data collator for multi-step labels."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = {
            "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
            "labels": torch.tensor([f["labels"] for f in features], dtype=torch.float),
            "n_active_steps": torch.tensor([f["n_active_steps"] for f in features], dtype=torch.long),
        }
        return batch


def main():
    parser = argparse.ArgumentParser(description="Train C1 Multi-Step Relevance")
    parser.add_argument("--data-path", required=True, help="Path to multi-step training data")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    print("=" * 60)
    print(f"MYCELIUM V6: C1 MULTI-STEP RELEVANCE (N_CHANNELS={N_CHANNELS})")
    print("=" * 60)

    # Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Data
    print(f"\nLoading data from {args.data_path}...")
    data = load_data(args.data_path)

    # Split
    random.seed(42)
    random.shuffle(data)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Datasets
    print("\nPreparing datasets...")
    train_dataset = prepare_dataset(train_data, tokenizer, args.max_length)
    eval_dataset = prepare_dataset(eval_data, tokenizer, args.max_length)

    # Model
    print("\nLoading model...")
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = Qwen2ForMultiStepRelevance(config)
    model = model.float()

    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Output dir
    if args.output_dir:
        output_dir = args.output_dir
    elif Path("/opt/dlami/nvme").exists():
        output_dir = "/opt/dlami/nvme/models/c1_multistep"
    else:
        output_dir = "models/c1_multistep"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="avg_channel_recall",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        save_total_limit=3,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=MultiStepDataCollator(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    trainer.train()

    # Final eval
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    results = trainer.evaluate()
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save
    final_path = f"{output_dir}/final"
    print(f"\nSaving to {final_path}...")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Test inference
    print("\n" + "=" * 60)
    print("INFERENCE TEST")

    model.eval()
    test_text = "Natalia sold clips to 48 of her friends in April, and then half as many in May."
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=256)

    with torch.no_grad():
        outputs = model(**inputs)
        preds = outputs.logits[0]  # (seq_len, N_CHANNELS)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    print(f"\nInput: {test_text}")
    print("\nPer-channel peaks (>0.3):")

    for c in range(N_CHANNELS):
        channel_preds = preds[:, c].numpy()
        peaks = [(i, tokens[i], channel_preds[i])
                 for i in range(len(tokens)) if channel_preds[i] > 0.3]
        if peaks:
            print(f"  Channel {c}: {peaks}")
        else:
            max_val = np.max(channel_preds)
            print(f"  Channel {c}: (inactive, max={max_val:.3f})")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")


if __name__ == "__main__":
    main()
