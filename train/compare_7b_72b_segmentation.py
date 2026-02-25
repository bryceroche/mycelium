#!/usr/bin/env python3
"""
Compare segmentation quality between Qwen-7B and Qwen-72B attention patterns.

Uses existing 72B JSD data as ground truth and generates 7B JSD for comparison.
"""

import json
import torch
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.ndimage import find_objects, label
from tqdm import tqdm


def compute_jsd(attn_row_i, attn_row_j):
    """Compute Jensen-Shannon divergence between two attention distributions."""
    # Ensure proper probability distributions
    p = attn_row_i / (attn_row_i.sum() + 1e-10)
    q = attn_row_j / (attn_row_j.sum() + 1e-10)

    # JSD = 0.5 * (KL(p||m) + KL(q||m)) where m = (p+q)/2
    m = 0.5 * (p + q)

    # Avoid log(0)
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    m = np.clip(m, 1e-10, 1.0)

    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))

    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd


def find_segments_from_jsd(jsd_trace, window=5, threshold_std=1.0):
    """
    Find segment boundaries from JSD trace using savgol smoothing and peak detection.

    Returns list of (start, end) token ranges.
    """
    if len(jsd_trace) < window:
        return [(0, len(jsd_trace))]

    # Smooth with Savitzky-Golay
    smoothed = savgol_filter(jsd_trace, window, 2)

    # Find peaks (above threshold)
    mean_jsd = np.mean(smoothed)
    std_jsd = np.std(smoothed)
    threshold = mean_jsd + threshold_std * std_jsd

    peaks = smoothed > threshold

    # Convert peaks to segment boundaries
    segments = []
    start = 0
    for i in range(len(peaks)):
        if peaks[i]:
            if i > start:
                segments.append((start, i))
            start = i + 1

    if start < len(jsd_trace):
        segments.append((start, len(jsd_trace)))

    return segments


def segment_f1(pred_segments, gold_segments, tolerance=2):
    """
    Compute F1 score between predicted and gold segments.

    A predicted segment matches gold if boundaries are within tolerance tokens.
    """
    if not pred_segments or not gold_segments:
        return 0.0

    # Convert to boundary sets
    pred_boundaries = set()
    for s, e in pred_segments:
        pred_boundaries.add(s)
        pred_boundaries.add(e)

    gold_boundaries = set()
    for s, e in gold_segments:
        gold_boundaries.add(s)
        gold_boundaries.add(e)

    # Count matches with tolerance
    matched_pred = 0
    for pb in pred_boundaries:
        for gb in gold_boundaries:
            if abs(pb - gb) <= tolerance:
                matched_pred += 1
                break

    matched_gold = 0
    for gb in gold_boundaries:
        for pb in pred_boundaries:
            if abs(pb - gb) <= tolerance:
                matched_gold += 1
                break

    precision = matched_pred / len(pred_boundaries) if pred_boundaries else 0
    recall = matched_gold / len(gold_boundaries) if gold_boundaries else 0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return f1


def load_72b_data(data_path, limit=20):
    """Load existing 72B results."""
    with open(data_path) as f:
        data = json.load(f)

    return data[:limit]


def generate_7b_jsd(model, tokenizer, problems, device="cuda"):
    """Generate JSD traces using 7B model."""
    results = []

    for item in tqdm(problems, desc="Generating 7B JSD"):
        question = item["question"]

        # Build prompt (same as 72B generation)
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]

        # Generate with attention output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                output_attentions=True,
                return_dict_in_generate=True,
                do_sample=False,
            )

        # Compute JSD trace from attention
        jsd_trace = []

        if hasattr(outputs, 'attentions') and outputs.attentions:
            for step_idx, step_attn in enumerate(outputs.attentions):
                if step_attn is None or step_idx == 0:
                    continue

                # Get last layer attention
                prev_attn = outputs.attentions[step_idx - 1][-1]  # Last layer
                curr_attn = step_attn[-1]

                # Average over heads, get last token's attention
                prev_row = prev_attn[0, :, -1, :].mean(dim=0).float().cpu().numpy()
                curr_row = curr_attn[0, :, -1, :].float().mean(dim=0).cpu().numpy()

                # Pad to same length
                max_len = max(len(prev_row), len(curr_row))
                prev_row = np.pad(prev_row, (0, max_len - len(prev_row)))
                curr_row = np.pad(curr_row, (0, max_len - len(curr_row)))

                jsd = compute_jsd(prev_row, curr_row)
                jsd_trace.append(float(jsd))

        results.append({
            "problem_idx": item.get("problem_idx", 0),
            "question": question,
            "jsd_trace_7b": jsd_trace,
            "jsd_trace_72b": item.get("jsd_trace", []),
            "segments_72b": item.get("segment_ranges", []),
        })

    return results


def compare_segmentation(results):
    """Compare 7B vs 72B segmentation quality."""
    f1_scores = []

    for item in results:
        jsd_7b = item.get("jsd_trace_7b", [])
        jsd_72b = item.get("jsd_trace_72b", [])
        gold_segments = item.get("segments_72b", [])

        if not jsd_7b or not gold_segments:
            continue

        # Find segments from 7B JSD
        pred_segments = find_segments_from_jsd(jsd_7b)

        # Compute F1
        f1 = segment_f1(pred_segments, gold_segments)
        f1_scores.append(f1)

        if len(f1_scores) <= 5:
            print(f"Problem {item['problem_idx']}:")
            print(f"  72B segments: {len(gold_segments)}, 7B segments: {len(pred_segments)}")
            print(f"  F1: {f1:.3f}")

    if f1_scores:
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        print(f"\nOverall F1: {mean_f1:.3f} ± {std_f1:.3f}")
        return mean_f1

    return 0.0


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-72b", default="results_worker0.json")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip 7B generation and just analyze existing 72B data")
    args = parser.parse_args()

    print("=" * 60)
    print("7B vs 72B Segmentation Comparison")
    print("=" * 60)

    # Load 72B data
    print(f"\nLoading 72B data from {args.data_72b}...")
    data_72b = load_72b_data(args.data_72b, limit=args.limit)
    print(f"Loaded {len(data_72b)} problems")

    if args.skip_generation:
        # Just analyze 72B segmentation statistics
        print("\nAnalyzing 72B segmentation statistics...")
        seg_counts = []
        jsd_lengths = []
        for item in data_72b:
            segs = item.get("segment_ranges", [])
            jsd = item.get("jsd_trace", [])
            seg_counts.append(len(segs))
            jsd_lengths.append(len(jsd))

        print(f"Average segments per problem: {np.mean(seg_counts):.1f}")
        print(f"Average JSD trace length: {np.mean(jsd_lengths):.1f}")
        return

    # Load 7B model
    print(f"\nLoading {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    # Generate 7B JSD
    print(f"\nGenerating 7B JSD traces...")
    results = generate_7b_jsd(model, tokenizer, data_72b, device=args.device)

    # Compare
    print(f"\nComparing segmentation...")
    mean_f1 = compare_segmentation(results)

    # Save results
    output_path = "7b_vs_72b_comparison.json"
    with open(output_path, "w") as f:
        json.dump({
            "mean_f1": mean_f1,
            "n_problems": len(results),
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
