"""CLI entry point for Mycelium solver.

Usage:
    python -m mycelium solve "What is 15% of 80?"
    python -m mycelium solve "Calculate 2^10"
    python -m mycelium info
"""

import argparse
import os
import sys
import time


def set_mode(mode: str):
    """Set training/inference mode via environment variable."""
    if mode == "training":
        os.environ["MYCELIUM_TRAINING_MODE"] = "true"
    else:
        os.environ["MYCELIUM_TRAINING_MODE"] = "false"


def solve_command(args):
    """Solve a single problem."""
    set_mode(args.mode)

    # Import after setting env var so config picks it up
    from mycelium.solver import Solver

    problem = args.problem
    solver = Solver()

    start = time.time()
    answer, results = solver.solve_with_trend_and_results(problem)
    elapsed_ms = (time.time() - start) * 1000

    print(f"\nProblem: {problem}")
    print(f"Answer: {answer}")
    print(f"Steps: {len(results)}")
    print(f"Time: {elapsed_ms:.0f}ms")
    print(f"Mode: {args.mode}")


def train_command(args):
    """Run training pipeline on a dataset."""
    set_mode("training")

    # Import after setting env var
    from mycelium.config import TRAINING_MODE

    print(f"Training mode: {TRAINING_MODE}")
    print(f"Dataset: {args.dataset}")
    print(f"Problems: {args.num}")
    print(f"Workers: {args.workers}")
    print("")
    print("Note: Training pipeline not yet updated for simplified architecture.")
    print("Use the solver directly for now.")


def info_command(args):
    """Show system information."""
    from mycelium.config import TRAINING_MODE, EMBEDDING_MODEL, EMBEDDING_DIM
    from mycelium.step_signatures import StepSignatureDB

    step_db = StepSignatureDB()
    sig_count = step_db.count_signatures()
    func_names = step_db.get_all_func_names()

    print("Mycelium System Info")
    print("=" * 40)
    print(f"Training mode: {TRAINING_MODE}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Embedding dimensions: {EMBEDDING_DIM}")
    print(f"Signatures in DB: {sig_count}")
    print(f"Functions covered: {len(func_names)}")


def clear_command(args):
    """Clear the signature database."""
    from mycelium.config import DB_PROTECTED

    # Check protection first
    if DB_PROTECTED and not args.force:
        print("ERROR: Database is protected (DB_PROTECTED=True)")
        print("")
        print("The database contains valuable learned data from 100+ problems.")
        print("To clear it anyway, use one of these options:")
        print("  1. mycelium clear --force")
        print("  2. Set environment variable: MYCELIUM_DB_PROTECTED=false")
        print("")
        print("Consider backing up first: cp mycelium.db mycelium.db.backup")
        return

    if not args.force:
        confirm = input("This will delete all signatures. Are you sure? [y/N]: ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return

    from mycelium.step_signatures import StepSignatureDB
    step_db = StepSignatureDB()
    result = step_db.clear_all_data(force=True)  # Already confirmed, bypass check
    print(f"Database cleared: {result}")


def main():
    parser = argparse.ArgumentParser(
        prog="mycelium",
        description="Mycelium math problem solver with signature learning",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # solve command
    solve_parser = subparsers.add_parser("solve", help="Solve a single problem")
    solve_parser.add_argument("problem", help="The math problem to solve")
    solve_parser.add_argument(
        "--mode", "-m",
        choices=["training", "inference"],
        default="inference",
        help="Operating mode (default: inference)",
    )

    # train command
    train_parser = subparsers.add_parser("train", help="Run training on a dataset")
    train_parser.add_argument(
        "--dataset", "-d",
        choices=["gsm8k", "math"],
        default="gsm8k",
        help="Dataset to train on (default: gsm8k)",
    )
    train_parser.add_argument(
        "--num", "-n",
        type=int,
        default=100,
        help="Number of problems to run (default: 100)",
    )
    train_parser.add_argument(
        "--workers", "-w",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)",
    )
    train_parser.add_argument(
        "--levels",
        help="Comma-separated difficulty levels (e.g., 1,2,3)",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )

    # info command
    subparsers.add_parser("info", help="Show system information")

    # clear command
    clear_parser = subparsers.add_parser("clear", help="Clear the signature database")
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
            args.mode = "inference"
        else:
            parser.print_help()
            sys.exit(1)

    if args.command == "solve":
        solve_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "info":
        info_command(args)
    elif args.command == "clear":
        clear_command(args)


if __name__ == "__main__":
    main()
