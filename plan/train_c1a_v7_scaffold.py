"""
C1-A v7: Coarse Boundary + Auxiliary Telegraph + Secondary Scaffold Type

Architecture:
    Problem text → Qwen-0.5B + LoRA (rank 16) → hidden_states
        │
        ├── PRIMARY: Coarse boundary windows (weight 1.0)
        │   Per-window → sigmoid → boundary probability
        │
        ├── AUXILIARY: Telegraph prediction (weight 0.3)
        │   Per-token → scalar → teacher's reading/computing signal
        │   (backbone enrichment, NOT used at inference)
        │
        └── SECONDARY: Scaffold type prediction (weight 0.5)
            Per-window → softmax(7) → step type distribution
            (USED at inference — guides LLM generation)

Three heads, one forward pass.
"""

import json
import re
import io
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
import boto3

s3 = boto3.client("s3")
BUCKET = "mycelium-data"

# Scaffold type labels
SCAFFOLD_TYPES = ["SETUP", "SUBSTITUTE", "SIMPLIFY", "SOLVE", "COMPUTE", "THEOREM", "OTHER"]
N_SCAFFOLD_CLASSES = len(SCAFFOLD_TYPES)


def preprocess_latex(text: str) -> str:
    """Normalize LaTeX notation to human-readable math."""
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


class WindowBoundaryHead(nn.Module):
    """Coarse boundary detection via window pooling."""

    def __init__(self, hidden_dim: int, window_size: int = 16, stride: int = 8):
        super().__init__()
        self.window_size = window_size
        self.stride = stride

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        window_logits_list = []
        window_mask_list = []
        window_pooled_list = []

        for start in range(0, seq_len, self.stride):
            end = min(start + self.window_size, seq_len)
            if end - start < self.window_size // 2:
                break

            window_hidden = hidden_states[:, start:end, :]
            window_attn = attention_mask[:, start:end]

            mask = window_attn.unsqueeze(-1).float()
            pooled = (window_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            logit = self.mlp(pooled).squeeze(-1)
            window_logits_list.append(logit)
            window_pooled_list.append(pooled)

            valid = (window_attn.sum(dim=1) > 0).float()
            window_mask_list.append(valid)

        if not window_logits_list:
            pooled = hidden_states.mean(dim=1)
            logit = self.mlp(pooled).squeeze(-1)
            return logit.unsqueeze(1), torch.ones(batch_size, 1, device=hidden_states.device), pooled.unsqueeze(1)

        window_logits = torch.stack(window_logits_list, dim=1)
        window_mask = torch.stack(window_mask_list, dim=1)
        window_pooled = torch.stack(window_pooled_list, dim=1)

        return window_logits, window_mask, window_pooled


class TelegraphHead(nn.Module):
    """Per-token telegraph prediction (auxiliary task)."""

    def __init__(self, hidden_dim: int = 896):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, hidden_states: torch.Tensor):
        return self.mlp(hidden_states).squeeze(-1)


class ScaffoldHead(nn.Module):
    """Per-window scaffold type prediction (secondary task)."""

    def __init__(self, hidden_dim: int = 896, n_classes: int = N_SCAFFOLD_CLASSES):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, window_pooled: torch.Tensor):
        """
        Args:
            window_pooled: [batch, n_windows, hidden_dim]
        Returns:
            scaffold_logits: [batch, n_windows, n_classes]
        """
        return self.mlp(window_pooled)


class C1AV7Model(nn.Module):
    """C1-A v7: boundary + telegraph + scaffold."""

    def __init__(self, backbone: nn.Module, hidden_dim: int = 896,
                 window_size: int = 16, stride: int = 8):
        super().__init__()
        self.backbone = backbone
        self.window_head = WindowBoundaryHead(hidden_dim, window_size, stride)
        self.telegraph_head = TelegraphHead(hidden_dim)
        self.scaffold_head = ScaffoldHead(hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]

        # Primary: window boundary prediction (returns pooled features too)
        window_logits, window_mask, window_pooled = self.window_head(hidden_states, attention_mask)

        # Auxiliary: per-token telegraph prediction
        telegraph_preds = self.telegraph_head(hidden_states)

        # Secondary: per-window scaffold type prediction
        scaffold_logits = self.scaffold_head(window_pooled)

        return window_logits, telegraph_preds, scaffold_logits, hidden_states, window_mask


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


def load_scaffold_labels():
    """Load scaffold labels from S3."""
    print("Loading scaffold labels from S3...")

    try:
        resp = s3.get_object(Bucket=BUCKET, Key="scaffold_training/labels.npy")
        labels = np.load(io.BytesIO(resp["Body"].read()))

        resp = s3.get_object(Bucket=BUCKET, Key="scaffold_training/metadata.json")
        metadata = json.loads(resp["Body"].read().decode("utf-8"))

        # Build lookup: (problem_id, window_idx) -> scaffold_label
        scaffold_lookup = {}
        for i, meta in enumerate(metadata.get("meta", [])):
            key = (str(meta["problem_id"]), meta["window_idx"])
            if i < len(labels):
                scaffold_lookup[key] = int(labels[i])

        print(f"  Loaded {len(scaffold_lookup)} scaffold labels")
        return scaffold_lookup
    except Exception as e:
        print(f"  Warning: could not load scaffold labels: {e}")
        return {}


class C1AV7Dataset(Dataset):
    """Dataset for C1-A v7 with boundary, telegraph, and scaffold labels."""

    def __init__(self, data_file: str, tokenizer, hints_cache: dict,
                 scaffold_lookup: dict,
                 max_length: int = 512, window_size: int = 16, stride: int = 8,
                 split: str = "train", val_ratio: float = 0.1, seed: int = 42,
                 apply_latex_preprocessing: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.window_size = window_size
        self.stride = stride
        self.apply_latex_preprocessing = apply_latex_preprocessing
        self.hints_cache = hints_cache
        self.scaffold_lookup = scaffold_lookup

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

        # Count scaffold label coverage
        scaffold_count = 0
        for idx in self.indices:
            r = all_records[idx]
            prob_idx = str(r.get('problem_idx', idx))
            if any((prob_idx, w) in scaffold_lookup for w in range(64)):
                scaffold_count += 1

        print(f"  {split}: {len(self.indices)} examples, {scaffold_count} with scaffold labels")

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

        # Telegraph targets
        problem_key = str(record.get('problem_idx', record_idx))
        telegraph_targets = [0.0] * self.max_length

        if problem_key in self.hints_cache:
            hint_data = self.hints_cache[problem_key]
            hint_vectors = hint_data['hint_vectors']
            hint_n_tokens = hint_data['n_tokens']

            if hint_n_tokens > 0 and actual_len > 0:
                scale = actual_len / hint_n_tokens
                for pos, hint_vec in enumerate(hint_vectors):
                    if pos < hint_n_tokens and len(hint_vec) > 2:
                        telegraph_val = hint_vec[2]
                        scaled_pos = min(int(pos * scale), actual_len - 1)
                        telegraph_targets[scaled_pos] = max(telegraph_targets[scaled_pos], telegraph_val)

        # Scaffold labels (per-window)
        scaffold_labels = [-1] * max_windows  # -1 means no label
        scaffold_mask = [0.0] * max_windows

        for w_idx in range(max_windows):
            key = (problem_key, w_idx)
            if key in self.scaffold_lookup:
                scaffold_labels[w_idx] = self.scaffold_lookup[key]
                scaffold_mask[w_idx] = 1.0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'window_labels': torch.tensor(window_labels, dtype=torch.float32),
            'window_mask': torch.tensor(window_mask, dtype=torch.float32),
            'telegraph_targets': torch.tensor(telegraph_targets, dtype=torch.float32),
            'scaffold_labels': torch.tensor(scaffold_labels, dtype=torch.long),
            'scaffold_mask': torch.tensor(scaffold_mask, dtype=torch.float32),
            'n_boundaries': len(record.get('boundaries', [])),
            'has_boundaries': record.get('has_boundaries', False),
        }


def compute_boundary_metrics(window_logits, window_labels, window_mask, n_boundaries, has_boundaries,
                             thresholds=[0.3, 0.5, 0.7]):
    """Compute boundary detection metrics."""
    probs = torch.sigmoid(window_logits)
    valid_mask = window_mask.bool()

    metrics = {}

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

    return metrics


def compute_scaffold_metrics(scaffold_logits, scaffold_labels, scaffold_mask):
    """Compute scaffold type prediction metrics."""
    valid_mask = scaffold_mask.bool()

    if not valid_mask.any():
        return {'scaffold_acc': 0.0, 'scaffold_samples': 0}

    logits_flat = scaffold_logits[valid_mask]  # [N, n_classes]
    labels_flat = scaffold_labels[valid_mask]  # [N]

    # Filter out -1 labels
    valid_labels = labels_flat >= 0
    if not valid_labels.any():
        return {'scaffold_acc': 0.0, 'scaffold_samples': 0}

    logits_valid = logits_flat[valid_labels]
    labels_valid = labels_flat[valid_labels]

    preds = logits_valid.argmax(dim=-1)
    correct = (preds == labels_valid).float().sum()
    total = labels_valid.shape[0]

    return {
        'scaffold_acc': (correct / total).item() if total > 0 else 0.0,
        'scaffold_samples': total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--hints-file', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2-0.5B')
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--window-size', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--telegraph-weight', type=float, default=0.3)
    parser.add_argument('--scaffold-weight', type=float, default=0.5)
    parser.add_argument('--no-latex-preprocessing', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load hints cache
    print(f"Loading hints from {args.hints_file}...")
    with open(args.hints_file) as f:
        hints_cache = json.load(f)
    print(f"  Loaded hints for {len(hints_cache)} problems")

    # Load scaffold labels
    scaffold_lookup = load_scaffold_labels()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    apply_preprocessing = not args.no_latex_preprocessing

    # Datasets
    train_dataset = C1AV7Dataset(
        args.data_file, tokenizer, hints_cache, scaffold_lookup, args.max_length,
        args.window_size, args.stride, split='train',
        apply_latex_preprocessing=apply_preprocessing
    )
    val_dataset = C1AV7Dataset(
        args.data_file, tokenizer, hints_cache, scaffold_lookup, args.max_length,
        args.window_size, args.stride, split='val',
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
        'batch_size': args.batch_size,
        'grad_accum': args.grad_accum,
        'effective_batch_size': args.batch_size * args.grad_accum,
        'lr': args.lr,
        'telegraph_weight': args.telegraph_weight,
        'scaffold_weight': args.scaffold_weight,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'latex_preprocessing': apply_preprocessing,
        'version': 'v7_scaffold',
    }
    with open(output_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Model
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

    model = C1AV7Model(
        backbone, hidden_dim=backbone.config.hidden_size,
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
    patience_counter = 0
    baseline_f1 = 0.741  # Previous best with telegraph

    print(f"\n{'='*70}")
    print("C1-A v7: Boundary + Telegraph + Scaffold")
    print(f"{'='*70}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Window size: {args.window_size}, Stride: {args.stride}")
    print(f"Loss weights: boundary=1.0, telegraph={args.telegraph_weight}, scaffold={args.scaffold_weight}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        total_boundary_loss = 0
        total_telegraph_loss = 0
        total_scaffold_loss = 0
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            window_labels = batch['window_labels'].to(device)
            window_mask_labels = batch['window_mask'].to(device)
            telegraph_targets = batch['telegraph_targets'].to(device)
            scaffold_labels = batch['scaffold_labels'].to(device)
            scaffold_mask = batch['scaffold_mask'].to(device)

            window_logits, telegraph_preds, scaffold_logits, hidden_states, window_mask_pred = model(
                input_ids, attention_mask
            )

            # Align dimensions
            n_windows = min(window_logits.shape[1], window_labels.shape[1])
            window_logits = window_logits[:, :n_windows]
            window_labels = window_labels[:, :n_windows]
            window_mask_labels = window_mask_labels[:, :n_windows]
            scaffold_logits = scaffold_logits[:, :n_windows]
            scaffold_labels = scaffold_labels[:, :n_windows]
            scaffold_mask = scaffold_mask[:, :n_windows]

            # Primary loss: boundary BCE
            valid_mask = window_mask_labels.bool()
            boundary_loss = F.binary_cross_entropy_with_logits(
                window_logits[valid_mask], window_labels[valid_mask]
            )

            # Auxiliary loss: telegraph MSE
            token_mask = attention_mask.bool()
            telegraph_loss = F.mse_loss(
                telegraph_preds[token_mask], telegraph_targets[token_mask]
            )

            # Secondary loss: scaffold cross-entropy (only on labeled windows)
            scaffold_valid = scaffold_mask.bool() & (scaffold_labels >= 0)
            if scaffold_valid.any():
                scaffold_loss = F.cross_entropy(
                    scaffold_logits[scaffold_valid],
                    scaffold_labels[scaffold_valid]
                )
            else:
                scaffold_loss = torch.tensor(0.0, device=device)

            # Combined loss
            loss = (boundary_loss +
                    args.telegraph_weight * telegraph_loss +
                    args.scaffold_weight * scaffold_loss)
            loss = loss / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.grad_accum
            total_boundary_loss += boundary_loss.item()
            total_telegraph_loss += telegraph_loss.item()
            total_scaffold_loss += scaffold_loss.item()
            n_batches += 1

        # Validation
        model.eval()
        val_loss = 0
        all_boundary_metrics = []
        all_scaffold_metrics = []
        total_telegraph_mae = 0
        total_telegraph_corr = 0
        n_val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                window_labels = batch['window_labels'].to(device)
                window_mask_labels = batch['window_mask'].to(device)
                telegraph_targets = batch['telegraph_targets'].to(device)
                scaffold_labels = batch['scaffold_labels'].to(device)
                scaffold_mask = batch['scaffold_mask'].to(device)
                n_boundaries = batch['n_boundaries']
                has_boundaries = batch['has_boundaries']

                window_logits, telegraph_preds, scaffold_logits, hidden_states, window_mask_pred = model(
                    input_ids, attention_mask
                )

                n_windows = min(window_logits.shape[1], window_labels.shape[1])
                window_logits = window_logits[:, :n_windows]
                window_labels = window_labels[:, :n_windows]
                window_mask_labels = window_mask_labels[:, :n_windows]
                scaffold_logits = scaffold_logits[:, :n_windows]
                scaffold_labels = scaffold_labels[:, :n_windows]
                scaffold_mask = scaffold_mask[:, :n_windows]

                valid_mask = window_mask_labels.bool()
                loss = F.binary_cross_entropy_with_logits(
                    window_logits[valid_mask], window_labels[valid_mask]
                )
                val_loss += loss.item()

                # Boundary metrics
                boundary_metrics = compute_boundary_metrics(
                    window_logits, window_labels, window_mask_labels,
                    n_boundaries, has_boundaries
                )
                all_boundary_metrics.append(boundary_metrics)

                # Scaffold metrics
                scaffold_metrics = compute_scaffold_metrics(
                    scaffold_logits, scaffold_labels, scaffold_mask
                )
                all_scaffold_metrics.append(scaffold_metrics)

                # Telegraph metrics
                token_mask = attention_mask.bool()
                preds = telegraph_preds[token_mask]
                targets = telegraph_targets[token_mask]
                total_telegraph_mae += (preds - targets).abs().mean().item()
                n_val_batches += 1

        val_loss /= len(val_loader)

        # Aggregate metrics
        avg_boundary = {k: np.mean([m[k] for m in all_boundary_metrics])
                       for k in all_boundary_metrics[0].keys()}

        total_scaffold_samples = sum(m['scaffold_samples'] for m in all_scaffold_metrics)
        if total_scaffold_samples > 0:
            weighted_scaffold_acc = sum(m['scaffold_acc'] * m['scaffold_samples']
                                       for m in all_scaffold_metrics) / total_scaffold_samples
        else:
            weighted_scaffold_acc = 0.0

        avg_telegraph_mae = total_telegraph_mae / n_val_batches

        # Log
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}")
        print(f"{'='*70}")
        print(f"Train Loss: {total_loss/n_batches:.4f} | Boundary: {total_boundary_loss/n_batches:.4f} | "
              f"Telegraph: {total_telegraph_loss/n_batches:.4f} | Scaffold: {total_scaffold_loss/n_batches:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Boundary F1@0.5: {avg_boundary['f1']:.4f} | F1@0.7: {avg_boundary['f1_0.7']:.4f}")
        print(f"Telegraph MAE: {avg_telegraph_mae:.4f}")
        print(f"Scaffold Acc: {weighted_scaffold_acc:.4f} ({total_scaffold_samples} samples)")
        print()
        delta = avg_boundary['f1'] - baseline_f1
        print(f"[vs Baseline 0.741: F1@0.5 delta = {'+' if delta >= 0 else ''}{delta:.4f}]")

        epoch_log = {
            'epoch': epoch,
            'train_loss': total_loss / n_batches,
            'val_loss': val_loss,
            **avg_boundary,
            'scaffold_acc': weighted_scaffold_acc,
            'telegraph_mae': avg_telegraph_mae,
        }
        training_log.append(epoch_log)

        # Early stopping on F1@0.5
        if avg_boundary['f1'] > best_f1:
            best_f1 = avg_boundary['f1']
            patience_counter = 0

            best_path = output_path / 'best_checkpoint'
            best_path.mkdir(parents=True, exist_ok=True)
            model.backbone.save_pretrained(best_path / 'lora_adapters')
            torch.save({
                'window_head': model.window_head.state_dict(),
                'telegraph_head': model.telegraph_head.state_dict(),
                'scaffold_head': model.scaffold_head.state_dict(),
            }, best_path / 'head_weights.pt')
            print(f"  Saved best (F1@0.5: {best_f1:.4f}, Scaffold Acc: {weighted_scaffold_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        with open(output_path / 'training_log.json', 'w') as f:
            json.dump(training_log, f, indent=2)

    # Final results
    best_epoch = max(training_log, key=lambda x: x['f1'])

    print(f"\n{'='*70}")
    print("C1-A v7 Complete!")
    print(f"{'='*70}")
    print(f"Best F1@0.5: {best_f1:.4f} at epoch {best_epoch['epoch']}")
    print(f"Best Scaffold Acc: {best_epoch['scaffold_acc']:.4f}")
    print()
    delta = best_f1 - baseline_f1
    print(f"[vs Baseline 0.741: F1@0.5 delta = {'+' if delta >= 0 else ''}{delta:.4f}]")

    results = {
        'best_f1': best_f1,
        'best_scaffold_acc': best_epoch['scaffold_acc'],
        'best_epoch': best_epoch['epoch'],
        'baseline_f1': baseline_f1,
        'improvement': best_f1 - baseline_f1,
        'version': 'v7_scaffold',
    }
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
