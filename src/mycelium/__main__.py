"""CLI entry point for Mycelium.

Usage:
    python -m mycelium stats    # Show Welford statistics
"""

import argparse
import sys


def stats_command(args):
    """Show Welford statistics."""
    from mycelium.welford import (
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
        description="Mycelium attention-based math decomposer",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # stats command
    subparsers.add_parser("stats", help="Show Welford statistics")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "stats":
        stats_command(args)


if __name__ == "__main__":
    main()
