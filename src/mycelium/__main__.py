"""CLI entry point for Mycelium solver.

Usage:
    python -m mycelium solve "What is 15% of 80?"
    python -m mycelium solve "Calculate 2^10" --mode=inference
    python -m mycelium train --dataset=gsm8k --num=100
    python -m mycelium info
"""

import argparse
import asyncio
import os
import sys


def set_mode(mode: str):
    """Set training/inference mode via environment variable."""
    if mode == "training":
        os.environ["MYCELIUM_TRAINING_MODE"] = "true"
    else:
        os.environ["MYCELIUM_TRAINING_MODE"] = "false"


async def solve_command(args):
    """Solve a single problem."""
    set_mode(args.mode)

    # Import after setting env var so config picks it up
    from mycelium.solver import Solver
    from mycelium.step_signatures import StepSignatureDB

    problem = args.problem
    step_db = StepSignatureDB()
    solver = Solver(step_db=step_db)

    result = await solver.solve(problem=problem)

    print(f"\nProblem: {problem}")
    print(f"Answer: {result.answer}")
    print(f"Steps: {result.total_steps}")
    print(f"Signatures matched: {result.signatures_matched}")
    print(f"DSL injections: {result.steps_with_injection}")
    print(f"Time: {result.elapsed_ms:.0f}ms")
    print(f"Mode: {args.mode}")


async def train_command(args):
    """Run training pipeline on a dataset."""
    set_mode("training")

    # Import after setting env var
    from mycelium.config import TRAINING_MODE

    print(f"Training mode: {TRAINING_MODE}")
    print(f"Dataset: {args.dataset}")
    print(f"Problems: {args.num}")
    print(f"Workers: {args.workers}")

    # Use pipeline_runner for training
    import subprocess
    cmd = [
        sys.executable, "-m", "scripts.pipeline_runner",
        "--dataset", args.dataset,
        "--problems", str(args.num),
        "--workers", str(args.workers),
    ]
    if args.levels:
        cmd.extend(["--levels", args.levels])
    if args.seed:
        cmd.extend(["--seed", str(args.seed)])

    subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


async def info_command(args):
    """Show system information."""
    from mycelium.config import (
        TRAINING_MODE, EMBEDDING_MODEL, MAX_SIGNATURES,
        TRAINING_ACCURACY_WEIGHT, INFERENCE_ACCURACY_WEIGHT,
    )
    from mycelium.step_signatures import StepSignatureDB

    step_db = StepSignatureDB()
    sig_count = step_db.count_signatures()

    print("Mycelium System Info")
    print("=" * 40)
    print(f"Training mode: {TRAINING_MODE}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Signatures in DB: {sig_count}")
    print(f"Max signatures: {MAX_SIGNATURES}")
    print(f"Training accuracy weight: {TRAINING_ACCURACY_WEIGHT}")
    print(f"Inference accuracy weight: {INFERENCE_ACCURACY_WEIGHT}")


async def clear_command(args):
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
        asyncio.run(solve_command(args))
    elif args.command == "train":
        asyncio.run(train_command(args))
    elif args.command == "info":
        asyncio.run(info_command(args))
    elif args.command == "clear":
        asyncio.run(clear_command(args))


if __name__ == "__main__":
    main()
