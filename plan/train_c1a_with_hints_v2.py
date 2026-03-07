"""
C1-A Coarse Boundary Window Detection with C0 Hints v2

Changes from v6 baseline:
- Load cached hints from JSON file
- In Dataset __getitem__, load hint vector for each problem
- In model forward, concatenate hint features to window hidden states before head
- Window head input size: 896 -> 900 (adds 4 hint dimensions)
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from tqdm import tqdm


def preprocess_latex(text: str) -> str:
    """
    Normalize LaTeX notation to human-readable math.
    """
    result = text

    # Remove \left and \right
    result = re.sub(r'\\left\s*([(\[{|])', r'\1', result)
    result = re.sub(r'\\right\s*([)\]}|])', r'\1', result)

    # Fractions
    def replace_frac(match):
        content = match.group(1)
        depth = 0
        num_start = num_end = denom_start = denom_end = -1
        for i, c in enumerate(content):
            if c == '{':
                if depth == 0 and num_start == -1:
                    num_start = i + 1
                elif depth == 0 and denom_start == -1:
                    denom_start = i + 1
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and num_end == -1:
                    num_end = i
                elif depth == 0 and denom_end == -1:
                    denom_end = i
                    break

        if num_start != -1 and num_end != -1 and denom_start != -1 and denom_end != -1:
            num = content[num_start:num_end]
            denom = content[denom_start:denom_end]
            num = preprocess_latex(num)
            denom = preprocess_latex(denom)
            if ' ' in num or '+' in num or '-' in num:
                num = f'({num})'
            if ' ' in denom or '+' in denom or '-' in denom:
                denom = f'({denom})'
            return f'{num}/{denom}'
        return match.group(0)

    for _ in range(5):
        old = result
        result = re.sub(r'\\[dt]?frac(.{0,200})', replace_frac, result)
        if result == old:
            break

    # Binomial
    def replace_binom(match):
        content = match.group(1)
        depth = 0
        n_start = n_end = k_start = k_end = -1
        for i, c in enumerate(content):
            if c == '{':
                if depth == 0 and n_start == -1:
                    n_start = i + 1
                elif depth == 0 and k_start == -1:
                    k_start = i + 1
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and n_end == -1:
                    n_end = i
                elif depth == 0 and k_end == -1:
                    k_end = i
                    break

        if n_start != -1 and n_end != -1 and k_start != -1 and k_end != -1:
            n = content[n_start:n_end]
            k = content[k_start:k_end]
            return f'binomial({n},{k})'
        return match.group(0)

    result = re.sub(r'\\binom(.{0,100})', replace_binom, result)

    # Square roots
    result = re.sub(r'\\sqrt\[([^\]]+)\]\{([^}]+)\}', r'root(\2,\1)', result)
    result = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', result)

    # Exponents and subscripts
    result = re.sub(r'\^{([^}]+)}', r'^\1', result)
    result = re.sub(r'_{([^}]+)}', r'_\1', result)

    # Operators
    result = result.replace('\\times', '×')
    result = result.replace('\\div', '÷')
    result = result.replace('\\cdot', '·')
    result = result.replace('\\pm', '±')
    result = result.replace('\\leq', '≤')
    result = result.replace('\\geq', '≥')
    result = result.replace('\\neq', '≠')
    result = result.replace('\\le', '≤')
    result = result.replace('\\ge', '≥')
    result = result.replace('\\ldots', '...')
    result = result.replace('\\cdots', '...')
    result = result.replace('\\dots', '...')

    # Text commands
    result = re.sub(r'\\text\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\textbf\{([^}]*)\}', r'\1', result)

    # Spacing
    result = re.sub(r'\\[,;:!]', ' ', result)
    result = re.sub(r'\\quad', ' ', result)
    result = re.sub(r'\\\[', '', result)
    result = re.sub(r'\\\]', '', result)

    # Remaining backslash commands
    result = re.sub(r'\\([a-zA-Z]+)', r'\1', result)

    # Clean up whitespace
    result = re.sub(r'[ \t]+', ' ', result)
    result = result.strip()

    return result


class WindowBoundaryHeadWithHints(nn.Module):
    """Coarse boundary detection via window pooling with hint features."""

    def __init__(self, hidden_dim: int, hint_dim: int = 4,
                 window_size: int = 16, stride: int = 8):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.hint_dim = hint_dim

        # Input is hidden_dim + hint_dim (896 + 4 = 900)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + hint_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor,
                hint_features: torch.Tensor):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len]
            hint_features: [batch, seq_len, hint_dim]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        n_windows = (seq_len - self.window_size) // self.stride + 1
        if n_windows <= 0:
            n_windows = 1

        window_logits_list = []
        window_mask_list = []

        for start in range(0, seq_len, self.stride):
            end = min(start + self.window_size, seq_len)
            if end - start < self.window_size // 2:
                break

            window_hidden = hidden_states[:, start:end, :]
            window_hints = hint_features[:, start:end, :]
            window_attn = attention_mask[:, start:end]

            mask = window_attn.unsqueeze(-1).float()

            # Pool hidden states
            pooled_hidden = (window_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            # Pool hint features
            pooled_hints = (window_hints * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            # Concatenate hidden + hints
            pooled = torch.cat([pooled_hidden, pooled_hints], dim=-1)

            logit = self.mlp(pooled).squeeze(-1)
            window_logits_list.append(logit)

            valid = (window_attn.sum(dim=1) > 0).float()
            window_mask_list.append(valid)

        if not window_logits_list:
            pooled_hidden = hidden_states.mean(dim=1)
            pooled_hints = hint_features.mean(dim=1)
            pooled = torch.cat([pooled_hidden, pooled_hints], dim=-1)
            logit = self.mlp(pooled).squeeze(-1)
            return logit.unsqueeze(1), torch.ones(batch_size, 1, device=hidden_states.device)

        window_logits = torch.stack(window_logits_list, dim=1)
        window_mask = torch.stack(window_mask_list, dim=1)

        return window_logits, window_mask


class C1ACoarseModelWithHints(nn.Module):
    """C1-A with coarse boundary window detection and C0 hints."""

    def __init__(self, backbone: nn.Module, hidden_dim: int = 896,
                 hint_dim: int = 4, window_size: int = 16, stride: int = 8):
        super().__init__()
        self.backbone = backbone
        self.hint_dim = hint_dim
        self.window_head = WindowBoundaryHeadWithHints(
            hidden_dim, hint_dim, window_size, stride
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                hint_features: torch.Tensor):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            hint_features: [batch, seq_len, hint_dim]
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        window_logits, window_mask = self.window_head(
            hidden_states, attention_mask, hint_features
        )
        return window_logits, hidden_states, window_mask


def make_boundary_probs(record: dict) -> list:
    """Convert boundaries format to per-token probability array."""
    n_tokens = record.get('n_input_tokens', record.get('input_len', 100))
    boundary_probs = [0.0] * n_tokens

    for boundary in record.get('boundaries', []):
        for sb in boundary.get('soft_boundaries', []):
            pos = sb['pos']
            prob = sb['prob']
            if 0 <= pos < n_tokens:
                boundary_probs[pos] = max(boundary_probs[pos], prob)

    return boundary_probs


def make_window_labels(boundary_probs: list, n_tokens: int, W: int = 16, S: int = 8):
    """Convert per-token probabilities to per-window labels."""
    windows = []
    for start in range(0, n_tokens, S):
        end = min(start + W, n_tokens)
        if end - start < W // 2:
            break
        window_probs = boundary_probs[start:end]
        label = max(window_probs) if window_probs else 0.0
        windows.append(label)

    if not windows:
        windows = [max(boundary_probs) if boundary_probs else 0.0]

    return windows


class C1ACoarseDatasetWithHints(Dataset):
    """Dataset for coarse boundary window detection with LaTeX preprocessing and C0 hints."""

    def __init__(self, data_file: str, hints_file: str, tokenizer, max_length: int = 512,
                 window_size: int = 16, stride: int = 8, hint_dim: int = 4,
                 split: str = "train", val_ratio: float = 0.1, seed: int = 42,
                 apply_latex_preprocessing: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.window_size = window_size
        self.stride = stride
        self.hint_dim = hint_dim
        self.apply_latex_preprocessing = apply_latex_preprocessing

        # Load cached hints
        print(f"Loading hints from {hints_file}...")
        with open(hints_file) as f:
            self.hints_cache = json.load(f)
        print(f"  Loaded hints for {len(self.hints_cache)} problems")

        # Load all records
        print(f"Loading data from {data_file}...")
        all_records = []
        with open(data_file) as f:
            for line in f:
                if line.strip():
                    all_records.append(json.loads(line))

        n_positive = sum(1 for r in all_records if r.get('has_boundaries', False))
        n_negative = len(all_records) - n_positive
        print(f"  Loaded {len(all_records)} records ({n_positive} positive, {n_negative} negative)")

        # Stratify by n_boundaries
        np.random.seed(seed)
        by_n_bounds = defaultdict(list)
        for i, r in enumerate(all_records):
            n_bounds = len(r.get('boundaries', []))
            by_n_bounds[n_bounds].append(i)

        train_indices, val_indices = [], []
        for n_bounds, indices in by_n_bounds.items():
            np.random.shuffle(indices)
            n_val = max(1, int(len(indices) * val_ratio))
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])

        self.indices = train_indices if split == "train" else val_indices
        np.random.shuffle(self.indices)
        self.all_records = all_records

        # Compute window statistics
        total_windows, positive_windows = 0, 0
        for idx in self.indices[:500]:
            r = all_records[idx]
            bp = make_boundary_probs(r)
            n_tokens = len(bp)
            labels = make_window_labels(bp, n_tokens, window_size, stride)
            total_windows += len(labels)
            positive_windows += sum(1 for l in labels if l > 0.5)

        pos_rate = positive_windows / max(1, total_windows)
        print(f"  {split}: {len(self.indices)} examples, ~{pos_rate*100:.1f}% positive windows")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        record_idx = self.indices[idx]
        record = self.all_records[record_idx]

        # Apply LaTeX preprocessing if enabled
        text = record['problem_text']
        if self.apply_latex_preprocessing:
            text = preprocess_latex(text)

        encoding = self.tokenizer(
            text, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        actual_len = int(attention_mask.sum().item())

        # Get boundary probs and scale to tokenized length
        n_input = record.get('n_input_tokens', record.get('input_len', 100))
        boundary_probs = make_boundary_probs(record)

        if n_input > 0 and actual_len > 0:
            scale = actual_len / n_input
            scaled_probs = [0.0] * actual_len
            for pos, prob in enumerate(boundary_probs):
                if prob > 0:
                    scaled_pos = min(int(pos * scale), actual_len - 1)
                    scaled_probs[scaled_pos] = max(scaled_probs[scaled_pos], prob)
            boundary_probs = scaled_probs

        # Generate window labels
        padded_boundary_probs = boundary_probs + [0.0] * (self.max_length - len(boundary_probs))
        window_labels = make_window_labels(padded_boundary_probs, self.max_length, self.window_size, self.stride)

        # Window mask
        window_mask = []
        for i, start in enumerate(range(0, self.max_length, self.stride)):
            end = min(start + self.window_size, self.max_length)
            if end - start < self.window_size // 2:
                break
            valid = 1.0 if start < actual_len else 0.0
            window_mask.append(valid)

        max_windows = len(window_labels)
        window_mask = window_mask[:max_windows]
        if len(window_mask) < max_windows:
            window_mask = window_mask + [0.0] * (max_windows - len(window_mask))

        # Load hint vectors for this problem
        problem_key = str(record_idx)
        if problem_key in self.hints_cache:
            hint_data = self.hints_cache[problem_key]
            hint_vectors = hint_data['hint_vectors']
            hint_n_tokens = hint_data['n_tokens']

            # Scale hint vectors to tokenized length
            if hint_n_tokens > 0 and actual_len > 0:
                scale = actual_len / hint_n_tokens
                scaled_hints = [[0.0] * self.hint_dim for _ in range(actual_len)]
                for pos, hint_vec in enumerate(hint_vectors):
                    if pos < hint_n_tokens:
                        scaled_pos = min(int(pos * scale), actual_len - 1)
                        # Take max of existing and new hint values
                        for d in range(self.hint_dim):
                            if d < len(hint_vec):
                                scaled_hints[scaled_pos][d] = max(
                                    scaled_hints[scaled_pos][d], hint_vec[d]
                                )
                hint_vectors = scaled_hints
            else:
                hint_vectors = [[0.0] * self.hint_dim for _ in range(actual_len)]
        else:
            # No hints available - use zeros
            hint_vectors = [[0.0] * self.hint_dim for _ in range(actual_len)]

        # Pad hint vectors to max_length
        while len(hint_vectors) < self.max_length:
            hint_vectors.append([0.0] * self.hint_dim)
        hint_vectors = hint_vectors[:self.max_length]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'window_labels': torch.tensor(window_labels, dtype=torch.float32),
            'window_mask': torch.tensor(window_mask, dtype=torch.float32),
            'hint_features': torch.tensor(hint_vectors, dtype=torch.float32),
            'n_boundaries': len(record.get('boundaries', [])),
            'has_boundaries': record.get('has_boundaries', False),
        }


def compute_metrics(window_logits, window_labels, window_mask, n_boundaries, has_boundaries,
                    thresholds=[0.3, 0.5, 0.7]):
    """Compute all metrics for coarse boundary detection."""
    probs = torch.sigmoid(window_logits)
    valid_mask = window_mask.bool()

    metrics = {}

    # Per-threshold metrics
    for thresh in thresholds:
        preds = (probs > thresh).float()
        targets = (window_labels > thresh).float()

        preds_valid = preds[valid_mask]
        targets_valid = targets[valid_mask]

        tp = ((preds_valid == 1) & (targets_valid == 1)).sum().float()
        fp = ((preds_valid == 1) & (targets_valid == 0)).sum().float()
        fn = ((preds_valid == 0) & (targets_valid == 1)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        suffix = f"_{thresh}" if thresh != 0.5 else ""
        metrics[f'precision{suffix}'] = precision.item()
        metrics[f'recall{suffix}'] = recall.item()
        metrics[f'f1{suffix}'] = f1.item()

    # Window counts
    metrics['n_pred_pos'] = (probs[valid_mask] > 0.5).sum().item()
    metrics['n_actual_pos'] = (window_labels[valid_mask] > 0.5).sum().item()

    # Per-problem segment metrics
    batch_size = window_logits.shape[0]
    all_correct, any_correct = 0, 0
    true_neg_correct, true_neg_total = 0, 0

    for i in range(batch_size):
        mask_i = window_mask[i].bool()
        preds_i = (probs[i, mask_i] > 0.5)
        targets_i = (window_labels[i, mask_i] > 0.5)

        if has_boundaries[i]:
            target_positive_idx = targets_i.nonzero(as_tuple=True)[0]

            if len(target_positive_idx) > 0:
                found_all = all(preds_i[idx] for idx in target_positive_idx)
                if found_all:
                    all_correct += 1

                found_any = any(preds_i[idx] for idx in target_positive_idx)
                if found_any:
                    any_correct += 1
        else:
            true_neg_total += 1
            if not preds_i.any():
                true_neg_correct += 1

    n_positive_problems = has_boundaries.sum().item()
    metrics['segment_all_correct'] = all_correct / max(1, n_positive_problems)
    metrics['segment_any_correct'] = any_correct / max(1, n_positive_problems)
    metrics['true_neg_accuracy'] = true_neg_correct / max(1, true_neg_total)

    # Calibration
    pos_mask = (window_labels[valid_mask] > 0.5)
    neg_mask = (window_labels[valid_mask] <= 0.5)

    if pos_mask.sum() > 0:
        metrics['mean_prob_at_positive'] = probs[valid_mask][pos_mask].mean().item()
    else:
        metrics['mean_prob_at_positive'] = 0.0

    if neg_mask.sum() > 0:
        metrics['mean_prob_at_negative'] = probs[valid_mask][neg_mask].mean().item()
    else:
        metrics['mean_prob_at_negative'] = 0.0

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--hints-file', type=str, required=True,
                        help='Path to cached hints JSON file')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2-0.5B')
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--window-size', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--hint-dim', type=int, default=4,
                        help='Dimension of hint vectors (default: 4)')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--no-latex-preprocessing', action='store_true',
                        help='Disable LaTeX preprocessing (data already preprocessed)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use already-preprocessed data (no runtime preprocessing needed)
    apply_preprocessing = not args.no_latex_preprocessing

    # Datasets with hints
    train_dataset = C1ACoarseDatasetWithHints(
        args.data_file, args.hints_file, tokenizer, args.max_length,
        args.window_size, args.stride, args.hint_dim, split='train',
        apply_latex_preprocessing=apply_preprocessing
    )
    val_dataset = C1ACoarseDatasetWithHints(
        args.data_file, args.hints_file, tokenizer, args.max_length,
        args.window_size, args.stride, args.hint_dim, split='val',
        apply_latex_preprocessing=apply_preprocessing
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Save config
    config = {
        'model_name': args.model_name,
        'lora_rank': args.lora_rank,
        'window_size': args.window_size,
        'stride': args.stride,
        'hint_dim': args.hint_dim,
        'batch_size': args.batch_size,
        'grad_accum': args.grad_accum,
        'effective_batch_size': args.batch_size * args.grad_accum,
        'lr': args.lr,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'latex_preprocessing': apply_preprocessing,
        'version': 'v2_with_hints',
    }
    with open(output_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Model with hints
    backbone = AutoModel.from_pretrained(args.model_name, trust_remote_code=True,
                                         torch_dtype=torch.float32)
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )
    backbone = get_peft_model(backbone, lora_config)
    backbone.print_trainable_parameters()

    model = C1ACoarseModelWithHints(
        backbone, hidden_dim=backbone.config.hidden_size,
        hint_dim=args.hint_dim,
        window_size=args.window_size, stride=args.stride
    ).to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = SequentialLR(optimizer, [
        LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
        CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-5)
    ], milestones=[warmup_steps])

    # Training
    training_log = []
    best_f1 = 0.0
    best_f1_07 = 0.0
    patience_counter = 0

    print(f"\nStarting C1-A with Hints v2 training: {len(train_dataset)} train, {len(val_dataset)} val")
    print(f"Window size: {args.window_size}, Stride: {args.stride}")
    print(f"Hint dimension: {args.hint_dim}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"LaTeX preprocessing: {apply_preprocessing}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            window_labels = batch['window_labels'].to(device)
            window_mask_labels = batch['window_mask'].to(device)
            hint_features = batch['hint_features'].to(device)

            window_logits, hidden_states, window_mask_pred = model(
                input_ids, attention_mask, hint_features
            )

            n_windows = min(window_logits.shape[1], window_labels.shape[1])
            window_logits = window_logits[:, :n_windows]
            window_labels = window_labels[:, :n_windows]
            window_mask_labels = window_mask_labels[:, :n_windows]

            valid_mask = window_mask_labels.bool()
            loss = F.binary_cross_entropy_with_logits(
                window_logits[valid_mask], window_labels[valid_mask]
            )
            loss = loss / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.grad_accum
            n_batches += 1

        # Validation
        model.eval()
        val_loss = 0
        all_metrics = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                window_labels = batch['window_labels'].to(device)
                window_mask_labels = batch['window_mask'].to(device)
                hint_features = batch['hint_features'].to(device)
                n_boundaries = batch['n_boundaries']
                has_boundaries = batch['has_boundaries']

                window_logits, hidden_states, window_mask_pred = model(
                    input_ids, attention_mask, hint_features
                )

                n_windows = min(window_logits.shape[1], window_labels.shape[1])
                window_logits = window_logits[:, :n_windows]
                window_labels = window_labels[:, :n_windows]
                window_mask_labels = window_mask_labels[:, :n_windows]

                valid_mask = window_mask_labels.bool()
                loss = F.binary_cross_entropy_with_logits(
                    window_logits[valid_mask], window_labels[valid_mask]
                )
                val_loss += loss.item()

                metrics = compute_metrics(
                    window_logits, window_labels, window_mask_labels,
                    n_boundaries, has_boundaries
                )
                all_metrics.append(metrics)

        val_loss /= len(val_loader)

        # Aggregate metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        # Log
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}")
        print(f"{'='*70}")
        print(f"Train Loss: {total_loss/n_batches:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Window Precision@0.5: {avg_metrics['precision']:.3f} | Recall@0.5: {avg_metrics['recall']:.3f} | F1@0.5: {avg_metrics['f1']:.3f}")
        print(f"Window F1@0.3: {avg_metrics['f1_0.3']:.3f} | F1@0.7: {avg_metrics['f1_0.7']:.3f}")
        print(f"Windows predicted positive: {avg_metrics['n_pred_pos']:.0f} | Actual positive: {avg_metrics['n_actual_pos']:.0f}")
        print(f"Segment all-correct: {avg_metrics['segment_all_correct']*100:.1f}%")
        print(f"Segment any-correct: {avg_metrics['segment_any_correct']*100:.1f}%")
        print(f"True negative accuracy: {avg_metrics['true_neg_accuracy']*100:.1f}%")
        print(f"Mean prob at positive: {avg_metrics['mean_prob_at_positive']:.3f} | at negative: {avg_metrics['mean_prob_at_negative']:.3f}")

        # Track best F1@0.7 separately
        if avg_metrics['f1_0.7'] > best_f1_07:
            best_f1_07 = avg_metrics['f1_0.7']
            print(f"  New best F1@0.7: {best_f1_07:.4f}")

        epoch_log = {
            'epoch': epoch,
            'train_loss': total_loss / n_batches,
            'val_loss': val_loss,
            **avg_metrics
        }
        training_log.append(epoch_log)

        # Early stopping on F1@0.5
        if avg_metrics['f1'] > best_f1:
            best_f1 = avg_metrics['f1']
            patience_counter = 0

            best_path = output_path / 'best_checkpoint'
            best_path.mkdir(parents=True, exist_ok=True)
            model.backbone.save_pretrained(best_path / 'lora_adapters')
            torch.save({
                'window_head': model.window_head.state_dict(),
            }, best_path / 'head_weights.pt')
            print(f"  Saved best (F1@0.5: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        with open(output_path / 'training_log.json', 'w') as f:
            json.dump(training_log, f, indent=2)

    # Final results
    best_epoch = max(training_log, key=lambda x: x['f1'])
    best_epoch_07 = max(training_log, key=lambda x: x['f1_0.7'])

    print(f"\nC1-A with Hints v2 Complete!")
    print(f"Best F1@0.5: {best_f1:.4f} at epoch {best_epoch['epoch']}")
    print(f"Best F1@0.7: {best_f1_07:.4f} at epoch {best_epoch_07['epoch']}")
    print(f"Best segment all-correct: {best_epoch['segment_all_correct']*100:.1f}%")
    print(f"Best true neg accuracy: {best_epoch['true_neg_accuracy']*100:.1f}%")

    results = {
        'best_f1': best_f1,
        'best_f1_0.7': best_f1_07,
        'best_epoch_f1': best_epoch['epoch'],
        'best_epoch_f1_07': best_epoch_07['epoch'],
        'best_segment_all_correct': best_epoch['segment_all_correct'],
        'best_segment_any_correct': best_epoch['segment_any_correct'],
        'best_true_neg_accuracy': best_epoch['true_neg_accuracy'],
        'final_epoch': epoch,
        'version': 'v2_with_hints',
    }
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
