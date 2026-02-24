#!/usr/bin/env python3
"""Quick evaluation script for v6 E2E pipeline."""

import json
import argparse
from tqdm import tqdm
from v6_e2e_pipeline import MyceliumPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/ubuntu/mycelium/data/dsl_test_pairs_clean.json")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    # Load pipeline
    pipeline = MyceliumPipeline()
    pipeline.load_models()

    # Load test data
    with open(args.data) as f:
        data = json.load(f)
    data = data[:args.limit]
    print(f"\nEvaluating on {len(data)} problems...")

    correct = 0
    total = 0
    results = []

    for item in tqdm(data):
        problem = item.get("problem_text", item.get("question", ""))
        gold = item.get("gold_answer")

        if gold is None:
            continue

        # Handle string answers
        if isinstance(gold, str):
            try:
                gold = float(gold.replace(",", ""))
            except:
                continue

        result = pipeline.solve(problem, verbose=False)
        pred = result.get("answer")
        n_spans = result["trace"].get("n_spans", 0)

        total += 1
        is_correct = pred is not None and abs(pred - gold) < 0.01

        if is_correct:
            correct += 1

        results.append({
            "problem": problem[:60],
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
            "n_spans": n_spans,
        })

    accuracy = correct / total if total > 0 else 0
    print(f"\n{'=' * 50}")
    print(f"Results: {correct}/{total} = {100*accuracy:.1f}%")
    print(f"{'=' * 50}")

    # Breakdown by n_spans
    print("\nAccuracy by span count:")
    for n in range(0, 6):
        subset = [r for r in results if r["n_spans"] == n]
        if subset:
            sub_correct = sum(1 for r in subset if r["correct"])
            sub_acc = 100 * sub_correct / len(subset)
            print(f"  {n} spans: {sub_correct}/{len(subset)} = {sub_acc:.1f}%")

    # Show some errors
    print("\nSample errors:")
    errors = [r for r in results if not r["correct"]][:5]
    for e in errors:
        print(f"  Q: {e['problem']}...")
        print(f"  Gold: {e['gold']}, Pred: {e['pred']}, Spans: {e['n_spans']}")
        print()


if __name__ == "__main__":
    main()
