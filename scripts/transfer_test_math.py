#!/usr/bin/env python3
"""
Transfer Test: GSM8K-trained Segmenter on MATH Problems

Tests whether the Qwen-0.5B segmenter trained on GSM8K transfers to MATH.
Uses JSD traces from MATH500 7B as ground truth boundaries.

If F1 > 80%, the full 12K MATH run doesn't need attention capture.
"""

import json
import torch
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
from transformers import AutoModelForTokenClassification, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter


def load_math_problems(data_dir: str, limit: int = None):
    """Load MATH problems with JSD traces."""
    problems = []
    data_path = Path(data_dir)
    files = sorted(data_path.glob("problem_*.json"))

    if limit:
        files = files[:limit]

    for f in files:
        with open(f) as fp:
            data = json.load(fp)
            if data.get("jsd_trace"):
                problems.append(data)

    print(f"Loaded {len(problems)} MATH problems with JSD traces")
    return problems


def jsd_to_boundaries(jsd_trace: list, window: int = 5, height_pct: float = 0.6):
    """Convert JSD trace to boundary positions using savgol + peaks."""
    if len(jsd_trace) < window:
        return []

    jsd = np.array(jsd_trace)
    smoothed = savgol_filter(jsd, window, 2)

    threshold = np.percentile(smoothed, height_pct * 100)
    peaks, _ = find_peaks(smoothed, height=threshold, distance=3)

    return list(peaks)


def bio_to_boundaries(bio_labels: list):
    """Extract boundary positions from BIO labels (B-OP positions)."""
    boundaries = []
    for i, label in enumerate(bio_labels):
        if label == 1:  # B-OP
            boundaries.append(i)
    return boundaries


def compute_boundary_f1(pred_bounds: list, true_bounds: list, tolerance: int = 2):
    """Compute F1 with tolerance for boundary matching."""
    if not true_bounds:
        return 1.0 if not pred_bounds else 0.0
    if not pred_bounds:
        return 0.0

    # Match predictions to ground truth with tolerance
    matched_true = set()
    matched_pred = set()

    for p in pred_bounds:
        for t in true_bounds:
            if abs(p - t) <= tolerance and t not in matched_true:
                matched_pred.add(p)
                matched_true.add(t)
                break

    tp = len(matched_true)
    fp = len(pred_bounds) - len(matched_pred)
    fn = len(true_bounds) - len(matched_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return f1, precision, recall


def run_transfer_test(model_path: str, data_dir: str, limit: int = 80):
    """Run the transfer test."""
    print("=" * 60)
    print("TRANSFER TEST: GSM8K Segmenter on MATH")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}")

    # Load MATH problems
    print(f"\nLoading MATH problems from {data_dir}...")
    problems = load_math_problems(data_dir, limit)

    # Run predictions
    results = []
    f1_scores = []
    precision_scores = []
    recall_scores = []

    print(f"\nProcessing {len(problems)} problems...")
    for i, prob in enumerate(problems):
        text = prob["generated_cot"]
        jsd_trace = prob["jsd_trace"]

        # Get ground truth boundaries from JSD
        true_bounds = jsd_to_boundaries(jsd_trace)

        # Tokenize and predict
        inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()

        # Get predicted boundaries from BIO
        pred_bounds = bio_to_boundaries(preds)

        # Note: JSD trace aligns with generated tokens,
        # but our tokenization may differ slightly
        # Scale predicted boundaries to match JSD trace length
        n_tokens_pred = len(preds)
        n_tokens_jsd = len(jsd_trace)
        if n_tokens_pred > 0 and n_tokens_jsd > 0:
            scale = n_tokens_jsd / n_tokens_pred
            pred_bounds_scaled = [int(b * scale) for b in pred_bounds]
        else:
            pred_bounds_scaled = pred_bounds

        # Compute F1
        f1, prec, rec = compute_boundary_f1(pred_bounds_scaled, true_bounds)
        f1_scores.append(f1)
        precision_scores.append(prec)
        recall_scores.append(rec)

        results.append({
            "idx": prob["idx"],
            "n_pred_bounds": len(pred_bounds),
            "n_true_bounds": len(true_bounds),
            "f1": f1,
            "precision": prec,
            "recall": rec,
        })

        if (i + 1) % 20 == 0:
            avg_f1 = np.mean(f1_scores)
            print(f"  Processed {i+1}/{len(problems)}, running F1: {avg_f1:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    avg_f1 = np.mean(f1_scores)
    avg_prec = np.mean(precision_scores)
    avg_rec = np.mean(recall_scores)

    print(f"\nOverall Boundary Detection:")
    print(f"  F1:        {avg_f1:.3f}")
    print(f"  Precision: {avg_prec:.3f}")
    print(f"  Recall:    {avg_rec:.3f}")

    # Distribution of F1 scores
    print(f"\nF1 Distribution:")
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, _ = np.histogram(f1_scores, bins=bins)
    for i in range(len(bins)-1):
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]} ({100*hist[i]/len(f1_scores):.1f}%)")

    # Boundary count stats
    pred_counts = [r["n_pred_bounds"] for r in results]
    true_counts = [r["n_true_bounds"] for r in results]
    print(f"\nBoundary Counts:")
    print(f"  Predicted mean: {np.mean(pred_counts):.1f} (std: {np.std(pred_counts):.1f})")
    print(f"  True (JSD) mean: {np.mean(true_counts):.1f} (std: {np.std(true_counts):.1f})")

    # Decision
    print("\n" + "=" * 60)
    print("DECISION")
    print("=" * 60)

    if avg_f1 >= 0.8:
        print(f"F1 = {avg_f1:.3f} >= 0.80")
        print(">>> 12K MATH run does NOT need attention capture <<<")
        print(">>> Can use vLLM for 3-5x speedup <<<")
    elif avg_f1 >= 0.6:
        print(f"F1 = {avg_f1:.3f} (0.60-0.80 range)")
        print(">>> Marginal transfer - consider 72B attention for quality <<<")
    else:
        print(f"F1 = {avg_f1:.3f} < 0.60")
        print(">>> Poor transfer - need attention capture for MATH <<<")

    return avg_f1, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/qwen05b_segmenter_clean")
    parser.add_argument("--data", default="/tmp/math_transfer_test")
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    run_transfer_test(args.model, args.data, args.limit)
