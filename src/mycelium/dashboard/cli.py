"""CLI interface for the Observability Dashboard.

Usage:
    python -m mycelium.dashboard [command]

Commands:
    health    - Show signature health summary
    routing   - Show routing tree and hot paths
    progress  - Show cold-start progress
    full      - Show complete dashboard (default)
"""

import argparse
import sys

from mycelium.config import DB_PATH
from mycelium.dashboard.health import (
    get_health_summary,
    get_signature_health_report,
    format_health_summary,
    format_health_report,
)
from mycelium.dashboard.routing import (
    get_routing_tree,
    get_hot_paths,
    get_routing_stats,
    format_routing_tree,
    format_hot_paths,
    format_routing_stats,
)
from mycelium.dashboard.cold_start import (
    get_cold_start_progress,
    get_growth_velocity,
    format_cold_start_progress,
    format_growth_velocity,
)


def cmd_health(args):
    """Show signature health metrics."""
    print("Loading health data...")
    print()

    summary = get_health_summary(args.db)
    print(format_health_summary(summary))

    if args.verbose:
        print()
        print("=== DETAILED HEALTH REPORT ===")
        print()
        report = get_signature_health_report(
            args.db,
            limit=args.limit,
            include_cold=args.include_cold,
        )
        print(format_health_report(report, show_description=args.descriptions))


def cmd_routing(args):
    """Show routing tree and hot paths."""
    print("Loading routing data...")
    print()

    # Routing stats
    stats = get_routing_stats(args.db)
    print(format_routing_stats(stats))
    print()

    # Routing tree
    tree = get_routing_tree(args.db, max_depth=args.depth)
    if tree:
        print("=== ROUTING TREE ===")
        print()
        print(format_routing_tree(tree, max_depth=args.depth, show_heat=True))
        print()

    # Hot paths
    if args.hot_paths > 0:
        paths = get_hot_paths(args.db, top_n=args.hot_paths)
        print(format_hot_paths(paths))


def cmd_progress(args):
    """Show cold-start progress metrics."""
    print("Loading progress data...")
    print()

    progress = get_cold_start_progress(args.db)
    print(format_cold_start_progress(progress))

    if args.velocity:
        print()
        velocity = get_growth_velocity(args.db, hours=args.hours)
        print(format_growth_velocity(velocity))


def cmd_full(args):
    """Show complete dashboard."""
    separator = "\n" + "=" * 60 + "\n"

    # Cold-start progress (always first - gives context)
    print("Loading dashboard...")
    print(separator)

    progress = get_cold_start_progress(args.db)
    print(format_cold_start_progress(progress))

    # Health summary
    print(separator)

    summary = get_health_summary(args.db)
    print(format_health_summary(summary))

    # Routing overview
    print(separator)

    stats = get_routing_stats(args.db)
    print(format_routing_stats(stats))
    print()

    # Compact routing tree
    tree = get_routing_tree(args.db, max_depth=3)
    if tree:
        print("=== ROUTING TREE (depth 3) ===")
        print()
        print(format_routing_tree(tree, max_depth=3, show_heat=True))

    # Hot paths
    print(separator)

    paths = get_hot_paths(args.db, top_n=5)
    print(format_hot_paths(paths))

    # Growth velocity
    print(separator)

    velocity = get_growth_velocity(args.db, hours=24)
    print(format_growth_velocity(velocity))


def dashboard_cli(argv=None):
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Mycelium Observability Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--db",
        default=DB_PATH,
        help=f"Path to database (default: {DB_PATH})",
    )

    subparsers = parser.add_subparsers(dest="command", help="Dashboard command")

    # Health command
    health_parser = subparsers.add_parser("health", help="Signature health metrics")
    health_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed health report",
    )
    health_parser.add_argument(
        "-n", "--limit",
        type=int,
        default=20,
        help="Number of signatures to show (default: 20)",
    )
    health_parser.add_argument(
        "--include-cold",
        action="store_true",
        help="Include cold-start signatures",
    )
    health_parser.add_argument(
        "--descriptions",
        action="store_true",
        help="Show signature descriptions",
    )
    health_parser.set_defaults(func=cmd_health)

    # Routing command
    routing_parser = subparsers.add_parser("routing", help="Routing tree visualization")
    routing_parser.add_argument(
        "-d", "--depth",
        type=int,
        default=5,
        help="Maximum depth to display (default: 5)",
    )
    routing_parser.add_argument(
        "--hot-paths",
        type=int,
        default=5,
        help="Number of hot paths to show (default: 5)",
    )
    routing_parser.set_defaults(func=cmd_routing)

    # Progress command
    progress_parser = subparsers.add_parser("progress", help="Cold-start progress")
    progress_parser.add_argument(
        "--velocity",
        action="store_true",
        help="Show growth velocity",
    )
    progress_parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Hours for velocity calculation (default: 24)",
    )
    progress_parser.set_defaults(func=cmd_progress)

    # Full command
    full_parser = subparsers.add_parser("full", help="Complete dashboard")
    full_parser.set_defaults(func=cmd_full)

    args = parser.parse_args(argv)

    if args.command is None:
        # Default to full dashboard
        args.func = cmd_full
        args.command = "full"

    args.func(args)


def main():
    """Entry point for python -m mycelium.dashboard."""
    try:
        dashboard_cli()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
