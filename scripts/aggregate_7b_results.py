#!/usr/bin/env python3
"""
Aggregate 7B results from parallel shards and identify failures for 72B.

Usage:
    # After syncing from S3:
    aws s3 sync s3://mycelium-data/math_full_7b/ ./math_full_7b/

    # Run aggregation:
    python aggregate_7b_results.py --input-dir ./math_full_7b --output-dir ./math_analysis
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory with 7B results")
    parser.add_argument("--output-dir", default="./math_analysis", help="Output directory")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Aggregating 7B Results")
    print("=" * 70)

    # Load all results
    results = []
    for f in sorted(input_dir.glob("problem_*.json")):
        if "error" in f.name:
            continue
        with open(f) as fp:
            results.append(json.load(fp))

    print(f"Loaded {len(results)} results")

    # Separate correct and failures
    correct = [r for r in results if r.get("is_correct", False)]
    failures = [r for r in results if not r.get("is_correct", False)]

    print(f"Correct: {len(correct)} ({len(correct)/len(results)*100:.1f}%)")
    print(f"Failures: {len(failures)} ({len(failures)/len(results)*100:.1f}%)")

    # Category breakdown
    print("\n" + "-" * 70)
    print("Category Breakdown")
    print("-" * 70)

    category_stats = defaultdict(lambda: {"total": 0, "correct": 0, "levels": defaultdict(lambda: {"total": 0, "correct": 0})})

    for r in results:
        cat = r.get("category", "Unknown")
        level = r.get("level", "Unknown")
        is_correct = r.get("is_correct", False)

        category_stats[cat]["total"] += 1
        category_stats[cat]["levels"][level]["total"] += 1

        if is_correct:
            category_stats[cat]["correct"] += 1
            category_stats[cat]["levels"][level]["correct"] += 1

    print(f"\n{'Category':<30} {'Total':>6} {'Correct':>8} {'Rate':>8}")
    print("-" * 60)

    for cat in sorted(category_stats.keys()):
        stats = category_stats[cat]
        rate = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"{cat:<30} {stats['total']:>6} {stats['correct']:>8} {rate:>7.1f}%")

    # Level breakdown
    print("\n" + "-" * 70)
    print("Level Breakdown (failures only)")
    print("-" * 70)

    level_counts = defaultdict(int)
    for f in failures:
        level = f.get("level", "Unknown")
        level_counts[level] += 1

    for level in sorted(level_counts.keys()):
        count = level_counts[level]
        pct = count / len(failures) * 100
        print(f"  Level {level}: {count:>5} ({pct:>5.1f}%)")

    # Save failure indices for 72B
    failure_indices = [f["idx"] for f in failures]
    failure_path = output_dir / "failure_indices.json"
    with open(failure_path, "w") as f:
        json.dump(failure_indices, f)
    print(f"\nSaved {len(failure_indices)} failure indices to {failure_path}")

    # Save category breakdown for strategic planning
    category_summary = {
        cat: {
            "total": stats["total"],
            "correct": stats["correct"],
            "rate": stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0,
            "failures": stats["total"] - stats["correct"],
        }
        for cat, stats in category_stats.items()
    }

    with open(output_dir / "category_summary.json", "w") as f:
        json.dump(category_summary, f, indent=2)

    # Save detailed failure analysis
    failure_analysis = []
    for f in failures:
        failure_analysis.append({
            "idx": f["idx"],
            "category": f.get("category"),
            "level": f.get("level"),
            "predicted": f.get("predicted_answer"),
            "gold": f.get("gold_answer"),
            "n_tokens": f.get("n_tokens_generated", 0),
        })

    with open(output_dir / "failure_analysis.json", "w") as f:
        json.dump(failure_analysis, f, indent=2)

    # JSD statistics
    print("\n" + "-" * 70)
    print("JSD Statistics")
    print("-" * 70)

    jsd_lengths = [len(r.get("jsd_scores", [])) for r in results]
    print(f"  Mean JSD trace length: {np.mean(jsd_lengths):.0f} tokens")
    print(f"  Median JSD trace length: {np.median(jsd_lengths):.0f} tokens")

    # Summary
    summary = {
        "total_problems": len(results),
        "correct": len(correct),
        "failures": len(failures),
        "accuracy": len(correct) / len(results) * 100 if results else 0,
        "category_stats": category_summary,
        "mean_jsd_length": float(np.mean(jsd_lengths)),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAnalysis saved to {output_dir}")

    # Estimate 72B cost
    print("\n" + "=" * 70)
    print("72B Cost Estimate")
    print("=" * 70)
    print(f"Failures to process: {len(failures)}")
    print(f"Estimated time @ 70s/problem: {len(failures) * 70 / 3600:.1f} hours")
    print(f"Estimated cost @ $14.32/hr: ${len(failures) * 70 / 3600 * 14.32:.0f}")


if __name__ == "__main__":
    main()
