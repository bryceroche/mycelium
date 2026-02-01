"""CLI entry point for Mycelium solver.

Usage:
    python -m mycelium solve "What is 15% of 80?"
    python -m mycelium seed
    python -m mycelium info
"""

import argparse
import sys
import time


def solve_command(args):
    """Solve a single problem."""
    from mycelium.engine import TemplateEngine

    problem = args.problem
    engine = TemplateEngine()

    start = time.time()
    answer = engine.solve(problem)
    elapsed_ms = (time.time() - start) * 1000

    print(f"\nProblem: {problem}")
    print(f"Answer: {answer}")
    print(f"Time: {elapsed_ms:.0f}ms")


def seed_command(args):
    """Seed the template database."""
    from mycelium.templates import seed_database
    seed_database()


def info_command(args):
    """Show system information."""
    from mycelium.config import EMBEDDING_MODEL, EMBEDDING_DIM
    from mycelium.templates.db import get_all_templates, get_all_examples

    templates = get_all_templates()
    examples = get_all_examples()

    print("Mycelium System Info")
    print("=" * 40)
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Embedding dimensions: {EMBEDDING_DIM}")
    print(f"Templates: {len(templates)}")
    print(f"Examples: {len(examples)}")
    print()
    if templates:
        print("Templates:")
        for t in templates:
            print(f"  - {t.name}: {t.description}")


def clear_command(args):
    """Clear the template database."""
    import os
    from pathlib import Path

    db_path = Path.home() / ".mycelium" / "templates.db"

    if not args.force:
        confirm = input(f"This will delete {db_path}. Are you sure? [y/N]: ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return

    if db_path.exists():
        os.remove(db_path)
        print(f"Deleted: {db_path}")
    else:
        print("Database doesn't exist.")


def main():
    parser = argparse.ArgumentParser(
        prog="mycelium",
        description="Mycelium template-based math problem solver",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # solve command
    solve_parser = subparsers.add_parser("solve", help="Solve a single problem")
    solve_parser.add_argument("problem", help="The math problem to solve")

    # seed command
    subparsers.add_parser("seed", help="Seed the template database")

    # info command
    subparsers.add_parser("info", help="Show system information")

    # clear command
    clear_parser = subparsers.add_parser("clear", help="Clear the template database")
    clear_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip confirmation prompt",
    )

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
    elif args.command == "seed":
        seed_command(args)
    elif args.command == "info":
        info_command(args)
    elif args.command == "clear":
        clear_command(args)


if __name__ == "__main__":
    main()
