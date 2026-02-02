"""CLI entry point for Mycelium solver.

Usage:
    python -m mycelium solve "What is 15% of 80?"
    python -m mycelium info
    python -m mycelium stats
"""

import argparse
import sys
import time


def solve_command(args):
    """Solve a single problem."""
    from mycelium.engine import PatternEngine

    problem = args.problem
    engine = PatternEngine()

    start = time.time()
    answer = engine.solve(problem)
    elapsed_ms = (time.time() - start) * 1000

    print(f"\nProblem: {problem}")
    print(f"Answer: {answer}")
    print(f"Time: {elapsed_ms:.0f}ms")


def info_command(args):
    """Show system information."""
    from mycelium.config import EMBEDDING_MODEL, EMBEDDING_DIM
    from mycelium.patterns import PATTERNS

    print("Mycelium System Info")
    print("=" * 40)
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Embedding dimensions: {EMBEDDING_DIM}")
    print(f"Patterns: {len(PATTERNS)}")

    total_examples = sum(len(p.examples) for p in PATTERNS.values())
    print(f"Examples: {total_examples}")
    print()
    print("Patterns:")
    for name, pattern in sorted(PATTERNS.items()):
        print(f"  - {name}: {len(pattern.examples)} examples ({pattern.execution_type})")


def stats_command(args):
    """Show Welford statistics."""
    from mycelium.patterns.welford import (
        get_global_stats,
        get_all_pattern_stats,
        get_high_variance_examples,
    )

    global_stats = get_global_stats()
    pattern_stats = get_all_pattern_stats()
    high_var = get_high_variance_examples()

    print("Welford Statistics")
    print("=" * 40)
    print(f"Global: n={global_stats.n}, mean={global_stats.mean:.3f}, stddev={global_stats.stddev:.3f}")
    print()

    if pattern_stats:
        print("Per-Pattern Stats:")
        for name, stats in sorted(pattern_stats.items()):
            thresh = stats.adaptive_threshold()
            thresh_str = f"{thresh:.3f}" if thresh else "default"
            print(f"  {name}: n={stats.n}, mean={stats.mean:.3f}, stddev={stats.stddev:.3f}, thresh={thresh_str}")
        print()

    if high_var:
        print("High Variance Examples (need attention):")
        for eid, stats in high_var.items():
            flags = []
            if stats.high_embedding_variance:
                flags.append("high-emb-var")
            if stats.high_outcome_variance:
                flags.append("high-out-var")
            print(f"  {eid[:50]}: {', '.join(flags)}")
    else:
        print("No high-variance examples detected.")


def main():
    parser = argparse.ArgumentParser(
        prog="mycelium",
        description="Mycelium pattern-based math problem solver",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # solve command
    solve_parser = subparsers.add_parser("solve", help="Solve a single problem")
    solve_parser.add_argument("problem", help="The math problem to solve")

    # info command
    subparsers.add_parser("info", help="Show system information")

    # stats command
    subparsers.add_parser("stats", help="Show Welford statistics")

    args = parser.parse_args()

    if args.command is None:
        # Backward compatibility: treat positional args as problem
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            args.command = "solve"
            args.problem = " ".join(sys.argv[1:])
        else:
            parser.print_help()
            sys.exit(1)

    if args.command == "solve":
        solve_command(args)
    elif args.command == "info":
        info_command(args)
    elif args.command == "stats":
        stats_command(args)


if __name__ == "__main__":
    main()
