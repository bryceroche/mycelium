#!/usr/bin/env python3
"""Analyze step failures from the database.

Usage:
    python scripts/failure_analysis.py                    # Show top failing signatures
    python scripts/failure_analysis.py --sig <sig_id>     # Details for specific signature
    python scripts/failure_analysis.py --problem <pid>    # All failures for a problem
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mycelium.step_signatures import StepSignatureDB


def main():
    parser = argparse.ArgumentParser(description="Analyze step failures")
    parser.add_argument("--sig", type=str, help="Show failures for specific signature ID")
    parser.add_argument("--problem", type=str, help="Show failures for specific problem ID")
    parser.add_argument("--limit", type=int, default=20, help="Number of results")
    parser.add_argument("--db", type=str, default="mycelium.db", help="Path to database (default: mycelium.db)")
    args = parser.parse_args()

    db = StepSignatureDB(db_path=args.db)

    if args.sig:
        # Show all failures for a specific signature
        failures = db.get_failures_for_signature(args.sig)
        if not failures:
            print(f"No failures found for signature {args.sig}")
            return

        print(f"\n{'='*80}")
        print(f"FAILURES FOR SIGNATURE: {args.sig}")
        print(f"Total failures: {len(failures)}")
        print(f"{'='*80}\n")

        for f in failures[:args.limit]:
            print(f"Step: {f['step_id']}")
            print(f"  Path: {f['step_path']}")
            print(f"  Task: {f['task'][:100]}...")
            print(f"  DSL: {f['dsl_script']}")
            print(f"  Result: {f['dsl_result']}")
            print(f"  Ground truth: {f['ground_truth']}")
            print(f"  Predicted: {f['predicted_answer']}")
            print()

    elif args.problem:
        # Show all failures for a specific problem
        with db._connection() as conn:
            cursor = conn.execute(
                """SELECT * FROM step_failures
                   WHERE problem_id = ?
                   ORDER BY depth, sibling_index""",
                (args.problem,),
            )
            failures = [dict(row) for row in cursor.fetchall()]

        if not failures:
            print(f"No failures found for problem {args.problem}")
            return

        print(f"\n{'='*80}")
        print(f"FAILURES FOR PROBLEM: {args.problem}")
        print(f"Problem: {failures[0]['problem_text'][:200]}...")
        print(f"Ground truth: {failures[0]['ground_truth']}")
        print(f"Predicted: {failures[0]['predicted_answer']}")
        print(f"{'='*80}\n")

        for f in failures:
            indent = "  " * f['depth']
            print(f"{indent}[{f['step_id']}] {f['task'][:60]}...")
            print(f"{indent}  DSL: {f['dsl_script']} → {f['dsl_result']}")
            if f['signature_id']:
                print(f"{indent}  Sig: {f['signature_id'][:8]}...")
            print()

    else:
        # Show top failing signatures
        stats = db.get_failure_stats(limit=args.limit)

        if not stats:
            print("No step failures recorded yet.")
            print("\nRun some problems first:")
            print("  python scripts/pipeline_runner.py --problems 10 --levels 5")
            return

        print(f"\n{'='*80}")
        print("TOP FAILING SIGNATURES")
        print(f"{'='*80}\n")

        print(f"{'Signature':<40} {'Type':<20} {'Failures':<10}")
        print("-" * 70)

        for s in stats:
            sig_short = s['signature_id'][:36] + "..." if s['signature_id'] else "N/A"
            step_type = s['step_type'] or "unknown"
            print(f"{sig_short:<40} {step_type:<20} {s['failure_count']:<10}")

        print(f"\n{'='*80}")
        print("Use --sig <signature_id> for details on a specific signature")
        print("Use --problem <problem_id> for all failures in a problem")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
