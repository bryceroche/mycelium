"""
C1-A Error Analysis: What's causing misses and false positives?

Analyzes trained C1-A coarse boundary model to understand error patterns.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import numpy as np
from tqdm import tqdm

# Math words for complexity analysis
MATH_WORDS = [
    'add', 'added', 'adding', 'plus', 'sum', 'total', 'combine', 'combined',
    'subtract', 'subtracted', 'minus', 'less', 'difference', 'remain', 'left',
    'multiply', 'multiplied', 'times', 'product', 'of',  # "of" as in "3/4 of"
    'divide', 'divided', 'split', 'share', 'quotient', 'per',
    'percent', 'percentage', '%',
    'half', 'third', 'quarter', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
    'double', 'triple', 'twice', 'thrice',
    'each', 'every', 'equal', 'equally',
]

# Back-reference words
BACKREF_WORDS = [
    'it', 'its', 'they', 'them', 'their', 'this', 'that', 'these', 'those',
    'the result', 'the answer', 'the total', 'the sum', 'the difference',
    'the same', 'same as', 'like before', 'again', 'remaining', 'left over',
]


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
            window_attn = attention_mask[:, start:end]

            mask = window_attn.unsqueeze(-1).float()
            pooled = (window_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            logit = self.mlp(pooled).squeeze(-1)
            window_logits_list.append(logit)

            valid = (window_attn.sum(dim=1) > 0).float()
            window_mask_list.append(valid)

        if not window_logits_list:
            pooled = hidden_states.mean(dim=1)
            logit = self.mlp(pooled).squeeze(-1)
            return logit.unsqueeze(1), torch.ones(batch_size, 1, device=hidden_states.device)

        window_logits = torch.stack(window_logits_list, dim=1)
        window_mask = torch.stack(window_mask_list, dim=1)

        return window_logits, window_mask


class C1ACoarseModel(nn.Module):
    """C1-A with coarse boundary window detection."""

    def __init__(self, backbone: nn.Module, hidden_dim: int = 896,
                 window_size: int = 16, stride: int = 8):
        super().__init__()
        self.backbone = backbone
        self.window_head = WindowBoundaryHead(hidden_dim, window_size, stride)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        window_logits, window_mask = self.window_head(hidden_states, attention_mask)
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


def get_window_text(text: str, tokenizer, window_idx: int, W: int = 16, S: int = 8, max_length: int = 512):
    """Extract the text corresponding to a specific window."""
    encoding = tokenizer(text, max_length=max_length, truncation=True, return_offsets_mapping=True)
    offsets = encoding['offset_mapping']

    start_token = window_idx * S
    end_token = min(start_token + W, len(offsets))

    if start_token >= len(offsets):
        return "[out of range]"

    char_start = offsets[start_token][0] if start_token < len(offsets) else 0
    char_end = offsets[min(end_token - 1, len(offsets) - 1)][1] if end_token > 0 else len(text)

    return text[char_start:char_end].strip()


def categorize_missed_window(window_text: str) -> str:
    """Categorize why a boundary window might have been missed."""
    text_lower = window_text.lower()

    # Check for clear math language
    clear_math_patterns = [
        r'\b(add|subtract|multiply|divide|times|plus|minus)\b',
        r'\b(divided by|multiplied by)\b',
        r'\d+\s*[\+\-\*\/\×÷]\s*\d+',
        r'\b\d+\s+(plus|minus|times)\s+\d+\b',
    ]
    for pattern in clear_math_patterns:
        if re.search(pattern, text_lower):
            return "clear_math"

    # Check for back-references
    backref_patterns = [
        r'\b(the result|the answer|the total|the sum|the difference)\b',
        r'\b(do the same|same as|like before|again)\b',
        r'\bremaining\b',
        r'\bleft over\b',
    ]
    for pattern in backref_patterns:
        if re.search(pattern, text_lower):
            return "back_reference"

    # Check for obtuse/indirect wording
    obtuse_patterns = [
        r'\b(three-fifths|two-thirds|one-half|one-quarter)\b',
        r'\b(take|took)\s+\w+\s+of\b',
        r'\b(portion|part|fraction)\s+of\b',
        r'\b\d+\s*%\s*of\b',
    ]
    for pattern in obtuse_patterns:
        if re.search(pattern, text_lower):
            return "obtuse_wording"

    # Check for implicit operations
    implicit_patterns = [
        r'\beach\b',
        r'\bper\b',
        r'\b(altogether|in total|combined|together)\b',
        r'\b(shared|split|distributed)\b',
    ]
    for pattern in implicit_patterns:
        if re.search(pattern, text_lower):
            return "implicit_op"

    return "other"


def categorize_false_positive(window_text: str) -> str:
    """Categorize why a window was falsely predicted as a boundary."""
    text_lower = window_text.lower()

    # Math-adjacent language
    math_adjacent_patterns = [
        r'\bnumber\b',
        r'\bcount\b',
        r'\bhow many\b',
        r'\bhow much\b',
        r'\btotal\b',  # Could be just asking about a total, not computing
    ]
    for pattern in math_adjacent_patterns:
        if re.search(pattern, text_lower):
            return "math_adjacent"

    # Numerical content without operation
    if re.search(r'\d+', text_lower):
        return "numerical_content"

    # Ambiguous
    if any(word in text_lower for word in ['could', 'would', 'should', 'might', 'if']):
        return "ambiguous"

    return "other"


def compute_text_complexity(text: str) -> dict:
    """Compute text complexity metrics."""
    # Word count
    words = text.split()
    word_count = len(words)

    # Sentence count (simple heuristic)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)

    # Average sentence length
    avg_sentence_len = word_count / max(1, sentence_count)

    # Math words
    text_lower = text.lower()
    math_word_count = sum(1 for word in MATH_WORDS if word in text_lower)

    # Pronouns and back-references
    backref_count = sum(1 for phrase in BACKREF_WORDS if phrase in text_lower)

    # Fractions/percentages written as words
    implicit_ops = len(re.findall(r'\b(half|third|quarter|fifth|percent|%)\b', text_lower))

    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_sentence_len': avg_sentence_len,
        'math_word_count': math_word_count,
        'backref_count': backref_count,
        'implicit_op_count': implicit_ops,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-file', type=str, required=True, help='JSONL data file')
    parser.add_argument('--output-file', type=str, required=True, help='Output JSON file')
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2-0.5B')
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--window-size', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("Loading model...")
    backbone = AutoModel.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.float32)
    backbone = PeftModel.from_pretrained(backbone, f"{args.model_dir}/best_checkpoint/lora_adapters")

    model = C1ACoarseModel(
        backbone, hidden_dim=backbone.config.hidden_size,
        window_size=args.window_size, stride=args.stride
    )

    # Load window head weights
    head_weights = torch.load(f"{args.model_dir}/best_checkpoint/head_weights.pt", map_location='cpu')
    model.window_head.load_state_dict(head_weights['window_head'])
    model = model.to(device)
    model.eval()
    print("Model loaded!")

    # Load validation data (same split as training)
    print(f"Loading data from {args.data_file}...")
    all_records = []
    with open(args.data_file) as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line))

    # Stratify by n_boundaries (same as training)
    np.random.seed(42)
    by_n_bounds = defaultdict(list)
    for i, r in enumerate(all_records):
        n_bounds = len(r.get('boundaries', []))
        by_n_bounds[n_bounds].append(i)

    val_indices = []
    for n_bounds, indices in by_n_bounds.items():
        np.random.shuffle(indices)
        n_val = max(1, int(len(indices) * 0.1))
        val_indices.extend(indices[:n_val])

    print(f"Validation set: {len(val_indices)} problems")

    # Run inference and collect results
    print("Running inference...")
    results = []

    for idx in tqdm(val_indices):
        record = all_records[idx]
        text = record['problem_text']
        has_boundaries = record.get('has_boundaries', False)

        # Tokenize
        encoding = tokenizer(
            text, max_length=args.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        actual_len = int(attention_mask.sum().item())

        # Get ground truth window labels
        n_input = record.get('n_input_tokens', record.get('input_len', 100))
        boundary_probs = make_boundary_probs(record)

        # Scale to tokenized length
        if n_input > 0 and actual_len > 0:
            scale = actual_len / n_input
            scaled_probs = [0.0] * actual_len
            for pos, prob in enumerate(boundary_probs):
                if prob > 0:
                    scaled_pos = min(int(pos * scale), actual_len - 1)
                    scaled_probs[scaled_pos] = max(scaled_probs[scaled_pos], prob)
            boundary_probs = scaled_probs

        # Window labels
        padded_probs = boundary_probs + [0.0] * (args.max_length - len(boundary_probs))
        gt_window_labels = make_window_labels(padded_probs, args.max_length, args.window_size, args.stride)

        # Model prediction
        with torch.no_grad():
            window_logits, _, window_mask = model(input_ids, attention_mask)

        probs = torch.sigmoid(window_logits).cpu().numpy()[0]
        n_windows = min(len(probs), len(gt_window_labels))

        # Find valid windows (within actual sequence)
        valid_windows = []
        for i in range(n_windows):
            start = i * args.stride
            if start < actual_len:
                valid_windows.append(i)

        # Compute errors
        gt_pos = set(i for i in valid_windows if gt_window_labels[i] > 0.5)
        pred_pos = set(i for i in valid_windows if probs[i] > 0.5)

        false_negatives = gt_pos - pred_pos  # Missed
        false_positives = pred_pos - gt_pos  # Hallucinated

        # Text complexity
        complexity = compute_text_complexity(text)

        result = {
            'idx': idx,
            'text': text,
            'has_boundaries': has_boundaries,
            'n_boundaries': len(record.get('boundaries', [])),
            'gt_positive_windows': list(gt_pos),
            'pred_positive_windows': list(pred_pos),
            'false_negatives': list(false_negatives),
            'false_positives': list(false_positives),
            'window_probs': probs[:n_windows].tolist(),
            'complexity': complexity,
            'n_valid_windows': len(valid_windows),
        }
        results.append(result)

    print(f"Analyzed {len(results)} problems")

    # Analysis 1: False Negative Deep Dive
    print("\n=== Analysis 1: False Negatives (Missed Boundaries) ===")
    fn_problems = [(r, len(r['false_negatives'])) for r in results if r['false_negatives']]
    fn_problems.sort(key=lambda x: -x[1])

    fn_examples = []
    fn_categories = defaultdict(int)

    for r, n_fn in fn_problems[:100]:
        missed_details = []
        for win_idx in r['false_negatives']:
            win_text = get_window_text(r['text'], tokenizer, win_idx, args.window_size, args.stride, args.max_length)
            pred_prob = r['window_probs'][win_idx] if win_idx < len(r['window_probs']) else 0.0
            category = categorize_missed_window(win_text)
            fn_categories[category] += 1

            missed_details.append({
                'window_idx': win_idx,
                'window_text': win_text[:200],
                'predicted_prob': round(pred_prob, 3),
                'category': category,
            })

        fn_examples.append({
            'problem_text': r['text'][:500],
            'n_missed': len(r['false_negatives']),
            'missed_windows': missed_details,
        })

    print(f"Total false negatives: {sum(len(r['false_negatives']) for r in results)}")
    print(f"Category distribution: {dict(fn_categories)}")

    # Analysis 2: False Positive Deep Dive
    print("\n=== Analysis 2: False Positives (Hallucinated Boundaries) ===")
    fp_problems = [(r, len(r['false_positives'])) for r in results if r['false_positives']]
    fp_problems.sort(key=lambda x: -x[1])

    fp_examples = []
    fp_categories = defaultdict(int)

    for r, n_fp in fp_problems[:100]:
        fp_details = []
        for win_idx in r['false_positives']:
            win_text = get_window_text(r['text'], tokenizer, win_idx, args.window_size, args.stride, args.max_length)
            pred_prob = r['window_probs'][win_idx] if win_idx < len(r['window_probs']) else 0.0
            category = categorize_false_positive(win_text)
            fp_categories[category] += 1

            fp_details.append({
                'window_idx': win_idx,
                'window_text': win_text[:200],
                'predicted_prob': round(pred_prob, 3),
                'category': category,
            })

        fp_examples.append({
            'problem_text': r['text'][:500],
            'n_false_positive': len(r['false_positives']),
            'hallucinated_windows': fp_details,
        })

    print(f"Total false positives: {sum(len(r['false_positives']) for r in results)}")
    print(f"Category distribution: {dict(fp_categories)}")

    # Analysis 3: True Negative Failures
    print("\n=== Analysis 3: True Negative Failures ===")
    true_neg_problems = [r for r in results if not r['has_boundaries']]

    tn_failed = [r for r in true_neg_problems if r['pred_positive_windows']]
    tn_passed = [r for r in true_neg_problems if not r['pred_positive_windows']]

    print(f"True negatives: {len(true_neg_problems)}")
    print(f"  Failed (predicted boundaries): {len(tn_failed)}")
    print(f"  Passed (correctly no boundaries): {len(tn_passed)}")

    tn_failed_examples = []
    for r in tn_failed[:50]:
        fp_details = []
        for win_idx in r['pred_positive_windows']:
            win_text = get_window_text(r['text'], tokenizer, win_idx, args.window_size, args.stride, args.max_length)
            pred_prob = r['window_probs'][win_idx] if win_idx < len(r['window_probs']) else 0.0
            fp_details.append({
                'window_idx': win_idx,
                'window_text': win_text[:200],
                'predicted_prob': round(pred_prob, 3),
            })

        tn_failed_examples.append({
            'problem_text': r['text'][:500],
            'n_predicted_boundaries': len(r['pred_positive_windows']),
            'false_boundary_windows': fp_details,
            'complexity': r['complexity'],
        })

    tn_passed_examples = []
    for r in tn_passed[:50]:
        tn_passed_examples.append({
            'problem_text': r['text'][:500],
            'complexity': r['complexity'],
        })

    # Analysis 4: Complexity Correlations
    print("\n=== Analysis 4: Complexity Correlations ===")

    # Per-problem F1
    for r in results:
        gt_set = set(r['gt_positive_windows'])
        pred_set = set(r['pred_positive_windows'])

        tp = len(gt_set & pred_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        r['problem_f1'] = f1

    # Correlation analysis
    def compute_correlation(metric_name, complexity_key):
        x = [r['complexity'][complexity_key] for r in results if r['has_boundaries']]
        y = [r['problem_f1'] for r in results if r['has_boundaries']]
        if len(x) < 2:
            return 0.0
        x_mean, y_mean = np.mean(x), np.mean(y)
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = np.sqrt(sum((xi - x_mean)**2 for xi in x) * sum((yi - y_mean)**2 for yi in y))
        return numerator / denominator if denominator > 0 else 0.0

    correlations = {
        'backref_count_vs_f1': compute_correlation('f1', 'backref_count'),
        'math_word_count_vs_f1': compute_correlation('f1', 'math_word_count'),
        'avg_sentence_len_vs_f1': compute_correlation('f1', 'avg_sentence_len'),
        'word_count_vs_f1': compute_correlation('f1', 'word_count'),
        'implicit_op_count_vs_f1': compute_correlation('f1', 'implicit_op_count'),
    }

    print(f"Correlations with problem F1:")
    for k, v in correlations.items():
        print(f"  {k}: {v:.3f}")

    # Bin analysis
    def analyze_by_bins(complexity_key, n_bins=5):
        pos_results = [r for r in results if r['has_boundaries']]
        values = [r['complexity'][complexity_key] for r in pos_results]
        min_val, max_val = min(values), max(values)
        bin_size = (max_val - min_val) / n_bins if max_val > min_val else 1

        bins = defaultdict(list)
        for r in pos_results:
            bin_idx = min(int((r['complexity'][complexity_key] - min_val) / bin_size), n_bins - 1)
            bins[bin_idx].append(r['problem_f1'])

        return {f"bin_{i}": {"mean_f1": np.mean(v), "count": len(v)} for i, v in sorted(bins.items())}

    bin_analysis = {
        'backref_bins': analyze_by_bins('backref_count'),
        'math_word_bins': analyze_by_bins('math_word_count'),
        'sentence_len_bins': analyze_by_bins('avg_sentence_len'),
    }

    # Summary
    print("\n=== Summary ===")

    total_fn = sum(len(r['false_negatives']) for r in results)
    total_fp = sum(len(r['false_positives']) for r in results)

    summary = f"""
C1-A Error Analysis Summary:

FALSE NEGATIVES (Missed Boundaries): {total_fn} windows
- Category breakdown: {dict(fn_categories)}
- Top category: {max(fn_categories.items(), key=lambda x: x[1]) if fn_categories else 'N/A'}

FALSE POSITIVES (Hallucinated Boundaries): {total_fp} windows
- Category breakdown: {dict(fp_categories)}
- Top category: {max(fp_categories.items(), key=lambda x: x[1]) if fp_categories else 'N/A'}

TRUE NEGATIVE ACCURACY: {len(tn_passed)}/{len(true_neg_problems)} ({100*len(tn_passed)/len(true_neg_problems):.1f}%)
- {len(tn_failed)} problems incorrectly predicted as having boundaries

COMPLEXITY CORRELATIONS:
- Back-reference count vs F1: {correlations['backref_count_vs_f1']:.3f}
- Math word count vs F1: {correlations['math_word_count_vs_f1']:.3f}
- Sentence length vs F1: {correlations['avg_sentence_len_vs_f1']:.3f}
"""
    print(summary)

    # C0 recommendation
    obtuse_implicit = fn_categories.get('obtuse_wording', 0) + fn_categories.get('implicit_op', 0)
    backref = fn_categories.get('back_reference', 0)
    clear_math = fn_categories.get('clear_math', 0)

    if (obtuse_implicit + backref) > 0.5 * total_fn:
        c0_recommendation = f"YES - Text normalization (C0) would likely help. {obtuse_implicit + backref}/{total_fn} ({100*(obtuse_implicit+backref)/max(1,total_fn):.0f}%) of missed boundaries are due to obtuse wording, implicit operations, or back-references that a C0 normalizer could clarify."
    elif clear_math > 0.5 * total_fn:
        c0_recommendation = f"NO - Most misses ({clear_math}/{total_fn}) are clear math language that C1-A should detect. C0 wouldn't help - need to improve C1-A directly."
    else:
        c0_recommendation = f"UNCERTAIN - Mixed error types. Consider targeted improvements to C1-A training data or model architecture first."

    print(f"\nC0 RECOMMENDATION: {c0_recommendation}")

    # Compile final output
    output = {
        'false_negatives': {
            'count': total_fn,
            'examples': fn_examples[:50],
            'category_distribution': dict(fn_categories),
        },
        'false_positives': {
            'count': total_fp,
            'examples': fp_examples[:50],
            'category_distribution': dict(fp_categories),
        },
        'true_negative_failures': {
            'total_true_negatives': len(true_neg_problems),
            'failed_count': len(tn_failed),
            'passed_count': len(tn_passed),
            'examples_failed': tn_failed_examples,
            'examples_passed': tn_passed_examples,
            'pattern_summary': "TBD - requires manual review of examples",
        },
        'complexity_correlation': {
            **correlations,
            'bin_analysis': bin_analysis,
        },
        'summary': summary.strip(),
        'c0_recommendation': c0_recommendation,
    }

    # Save output
    print(f"\nSaving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print("Done!")


if __name__ == '__main__':
    main()
