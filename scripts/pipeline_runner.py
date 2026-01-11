#!/usr/bin/env python3
"""Pipeline runner for evaluating Solver on math problems.

Usage:
    python scripts/pipeline_runner.py --problems 10 --modes cosine auto --workers 2
    python scripts/pipeline_runner.py --dataset math --levels 1 2 3 --problems 20 --workers 3
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_dataset

from mycelium.solver import Solver
from mycelium.step_signatures import StepSignatureDB
from mycelium.client import GroqClient
from mycelium.answer_norm import answers_equivalent_llm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ProblemResult:
    """Result from solving a single problem."""
    problem_id: str
    problem: str
    ground_truth: str
    predicted: str
    success: bool
    match_mode: str
    elapsed_ms: float
    level: str = ""
    # Signature matching stats
    total_steps: int = 0
    signatures_matched: int = 0
    signatures_new: int = 0
    steps_with_injection: int = 0
    new_signatures_created: int = 0
    error: Optional[str] = None


def extract_boxed_answer(solution: str) -> str:
    """Extract answer from \\boxed{} in MATH solutions."""
    # Find the last \boxed{} in the solution
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', solution)
    if matches:
        return matches[-1].strip()
    # Fallback: try to find any boxed content
    match = re.search(r'\\boxed\{(.+?)\}', solution)
    if match:
        return match.group(1).strip()
    return ""


def load_problems_gsm8k(num_problems: int) -> list[dict]:
    """Load problems from GSM8K dataset."""
    logger.info(f"Loading {num_problems} problems from GSM8K...")

    dataset = load_dataset("gsm8k", "main", split="test")

    problems = []
    for i, item in enumerate(dataset):
        if i >= num_problems:
            break

        answer = item["answer"].split("####")[-1].strip()

        problems.append({
            "id": f"gsm8k_{i}",
            "problem": item["question"],
            "answer": answer,
            "level": "grade_school",
        })

    logger.info(f"Loaded {len(problems)} problems")
    return problems


def load_problems_math(num_problems: int, levels: list[int], seed: int = None) -> list[dict]:
    """Load problems from MATH dataset with level filtering.

    Args:
        num_problems: Number of problems to load
        levels: List of difficulty levels (1-5) to include
        seed: Random seed for reproducible problem selection
    """
    level_strs = [f"Level {l}" for l in levels]
    logger.info(f"Loading {num_problems} problems from MATH (levels {levels})...")

    # MATH dataset has multiple subsets
    subsets = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus"
    ]

    all_problems = []
    for subset in subsets:
        try:
            dataset = load_dataset("EleutherAI/hendrycks_math", subset, split="test")
            for item in dataset:
                if item["level"] in level_strs:
                    answer = extract_boxed_answer(item["solution"])
                    all_problems.append({
                        "id": f"math_{subset}_{len(all_problems)}",
                        "problem": item["problem"],
                        "answer": answer,
                        "level": item["level"],
                        "type": item["type"],
                    })
        except Exception as e:
            logger.warning(f"Failed to load subset {subset}: {e}")

    # Shuffle and limit (use seed for reproducibility)
    import random
    if seed is not None:
        random.seed(seed)
        logger.info(f"Using seed {seed} for reproducible problem selection")
    random.shuffle(all_problems)
    problems = all_problems[:num_problems]

    # Count by level
    level_counts = {}
    for p in problems:
        level_counts[p["level"]] = level_counts.get(p["level"], 0) + 1

    logger.info(f"Loaded {len(problems)} problems: {level_counts}")
    return problems


def load_problems(num_problems: int, dataset: str = "gsm8k", levels: list[int] = None, seed: int = None) -> list[dict]:
    """Load problems from specified dataset."""
    if dataset == "gsm8k":
        return load_problems_gsm8k(num_problems)
    elif dataset == "math":
        levels = levels or [1, 2, 3, 4, 5]
        return load_problems_math(num_problems, levels, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


async def solve_direct(problem: dict) -> ProblemResult:
    """Solve problem directly with Llama (no decomposition) - true baseline."""
    import time
    import re

    start = time.time()

    try:
        async with GroqClient() as client:
            messages = [
                {"role": "system", "content": "You are a math expert. Solve the problem step by step, then give your final answer in \\boxed{}."},
                {"role": "user", "content": problem["problem"]},
            ]
            response = await client.generate(messages, temperature=0.0)

            # Extract answer from \boxed{}
            matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', response)
            predicted = matches[-1].strip() if matches else response.strip()

            # Use LLM judge for fair comparison
            success = await answers_equivalent_llm(predicted, problem["answer"], problem["problem"])

            elapsed = (time.time() - start) * 1000

            return ProblemResult(
                problem_id=problem["id"],
                problem=problem["problem"][:200],
                ground_truth=problem["answer"],
                predicted=predicted,
                success=success,
                match_mode="direct",
                elapsed_ms=elapsed,
                level=problem.get("level", ""),
            )
    except Exception as e:
        logger.error(f"Error in direct solve {problem['id']}: {e}")
        return ProblemResult(
            problem_id=problem["id"],
            problem=problem["problem"][:200],
            ground_truth=problem["answer"],
            predicted="",
            success=False,
            match_mode="direct",
            elapsed_ms=0.0,
            level=problem.get("level", ""),
            error=str(e),
        )


async def solve_problem(
    problem: dict,
    match_mode: str,
    db_path: str,
    injection_mode: str = "all",
    use_hints: bool = True,
) -> ProblemResult:
    """Solve a single problem with the given match mode."""
    try:
        # Pass db_path directly to bypass singleton issues in multiprocessing
        step_db = StepSignatureDB(db_path=db_path)
        solver = Solver(
            step_db=step_db,
            match_mode=match_mode,
            injection_mode=injection_mode,
            use_hints=use_hints,
        )

        result = await solver.solve(
            problem=problem["problem"],
            ground_truth=problem["answer"],
            problem_id=problem["id"],
        )

        return ProblemResult(
            problem_id=problem["id"],
            problem=problem["problem"][:200],
            ground_truth=problem["answer"],
            predicted=result.answer,
            success=result.success,
            match_mode=match_mode,
            elapsed_ms=result.elapsed_ms,
            level=problem.get("level", ""),
            total_steps=result.total_steps,
            signatures_matched=result.signatures_matched,
            signatures_new=result.signatures_new,
            steps_with_injection=result.steps_with_injection,
            new_signatures_created=result.new_signatures_created,
        )

    except Exception as e:
        logger.error(f"Error solving problem {problem['id']}: {e}")
        return ProblemResult(
            problem_id=problem["id"],
            problem=problem["problem"][:200],
            ground_truth=problem["answer"],
            predicted="",
            success=False,
            match_mode=match_mode,
            elapsed_ms=0.0,
            level=problem.get("level", ""),
            error=str(e),
        )


def run_problem_sync(args: tuple) -> dict:
    """Synchronous wrapper for process pool."""
    problem, match_mode, db_path, injection_mode, use_hints = args
    if match_mode == "direct":
        result = asyncio.run(solve_direct(problem))
    else:
        result = asyncio.run(solve_problem(problem, match_mode, db_path, injection_mode, use_hints))
    return asdict(result)


def run_pipeline(
    num_problems: int,
    modes: list[str],
    num_workers: int,
    output_file: str,
    dataset: str = "gsm8k",
    levels: list[int] = None,
    injection_mode: str = "all",
    seed: int = None,
    use_hints: bool = True,
    db_path: str = "mycelium.db",
):
    """Run the evaluation pipeline."""
    logger.info(f"Starting pipeline: {num_problems} problems, modes={modes}, workers={num_workers}, dataset={dataset}, injection={injection_mode}, seed={seed}, hints={use_hints}, db={db_path}")

    # Load problems
    problems = load_problems(num_problems, dataset=dataset, levels=levels, seed=seed)

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Prepare tasks (use single shared database)
    tasks = []
    for mode in modes:
        for problem in problems:
            tasks.append((problem, mode, db_path, injection_mode, use_hints))

    logger.info(f"Running {len(tasks)} total tasks ({len(problems)} problems x {len(modes)} modes)")

    # Run with process pool
    results = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i, result in enumerate(executor.map(run_problem_sync, tasks)):
            results.append(result)
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{len(tasks)} tasks")

    total_time = time.time() - start_time

    # Aggregate results by mode
    mode_stats = {}
    for mode in modes:
        mode_results = [r for r in results if r["match_mode"] == mode]
        successes = sum(1 for r in mode_results if r["success"])
        total = len(mode_results)

        # Signature matching stats
        total_steps = sum(r.get("total_steps", 0) for r in mode_results)
        sigs_matched = sum(r.get("signatures_matched", 0) for r in mode_results)
        sigs_new = sum(r.get("signatures_new", 0) for r in mode_results)
        steps_injected = sum(r.get("steps_with_injection", 0) for r in mode_results)
        match_rate = sigs_matched / total_steps if total_steps > 0 else 0.0
        avg_steps = total_steps / total if total > 0 else 0.0

        mode_stats[mode] = {
            "total": total,
            "successes": successes,
            "accuracy": successes / total if total > 0 else 0.0,
            "avg_elapsed_ms": sum(r["elapsed_ms"] for r in mode_results) / total if total > 0 else 0.0,
            # Signature stats
            "total_steps": total_steps,
            "avg_steps_per_problem": avg_steps,
            "signatures_matched": sigs_matched,
            "signatures_new": sigs_new,
            "steps_with_injection": steps_injected,
            "match_rate": match_rate,
        }

    # Aggregate by level (for MATH dataset)
    level_stats = {}
    for r in results:
        level = r.get("level", "unknown")
        if level not in level_stats:
            level_stats[level] = {"total": 0, "successes": 0}
        level_stats[level]["total"] += 1
        if r["success"]:
            level_stats[level]["successes"] += 1

    for level in level_stats:
        total = level_stats[level]["total"]
        level_stats[level]["accuracy"] = level_stats[level]["successes"] / total if total > 0 else 0.0

    # Save results
    output = {
        "config": {
            "num_problems": num_problems,
            "modes": modes,
            "num_workers": num_workers,
            "dataset": dataset,
            "levels": levels,
            "injection_mode": injection_mode,
            "seed": seed,
        },
        "summary": {
            "total_time_seconds": total_time,
            "mode_stats": mode_stats,
            "level_stats": level_stats,
        },
        "results": results,
    }

    output_path = Path(output_file)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Problems: {num_problems}")
    print(f"Modes: {modes}")
    print(f"Injection: {injection_mode}")
    if levels:
        print(f"Levels: {levels}")
    print()

    print("Results by mode:")
    for mode, stats in mode_stats.items():
        print(f"  {mode:15s} {stats['accuracy']:5.1%} ({stats['successes']}/{stats['total']})  avg {stats['avg_elapsed_ms']:.0f}ms")

    # Print signature matching stats
    print("\nSignature matching stats:")
    for mode, stats in mode_stats.items():
        avg_steps = stats.get('avg_steps_per_problem', 0)
        match_rate = stats.get('match_rate', 0)
        sigs_matched = stats.get('signatures_matched', 0)
        sigs_new = stats.get('signatures_new', 0)
        steps_injected = stats.get('steps_with_injection', 0)
        total_steps = stats.get('total_steps', 0)
        print(f"  {mode:15s} {avg_steps:.1f} steps/prob, {match_rate:5.1%} matched ({sigs_matched}/{total_steps}), {sigs_new} new, {steps_injected} injected")

    if len(level_stats) > 1:
        print("\nResults by level:")
        for level in sorted(level_stats.keys()):
            stats = level_stats[level]
            print(f"  {level:15s} {stats['accuracy']:5.1%} ({stats['successes']}/{stats['total']})")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run Solver evaluation pipeline")
    parser.add_argument(
        "--problems", "-p",
        type=int,
        default=10,
        help="Number of problems to evaluate (default: 10)",
    )
    parser.add_argument(
        "--modes", "-m",
        nargs="+",
        default=["auto"],
        help="Match modes to test (default: auto)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results/pipeline_results.json",
        help="Output file path (default: results/pipeline_results.json)",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "math"],
        help="Dataset to use (default: gsm8k)",
    )
    parser.add_argument(
        "--levels", "-l",
        type=int,
        nargs="+",
        default=None,
        help="Difficulty levels for MATH dataset (1-5, default: all)",
    )
    parser.add_argument(
        "--injection-mode", "-i",
        type=str,
        default="all",
        choices=["none", "dsl", "formula", "procedure", "guidance", "all"],
        help="Injection strategy: none (baseline), dsl, formula, procedure, guidance, all (default: all)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducible problem selection (default: None)",
    )
    parser.add_argument(
        "--hints/--no-hints",
        dest="use_hints",
        action="store_true",
        default=True,
        help="Enable signature-guided decomposition hints (default: enabled)",
    )
    parser.add_argument(
        "--no-hints",
        dest="use_hints",
        action="store_false",
        help="Disable signature-guided decomposition hints",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["training", "benchmark"],
        help="Operating mode: training (explore, collect data) or benchmark (conservative, max accuracy)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="mycelium.db",
        help="Path to signature database (default: mycelium.db in project root)",
    )

    args = parser.parse_args()

    # Apply runtime config overrides
    import mycelium.config as config
    from mycelium.config import Mode

    if args.mode:
        if args.mode == "training":
            config.ACTIVE_MODE = Mode.TRAINING
            config.TRAINING_MODE = True
            config.MIN_MATCH_THRESHOLD = 0.92
            config.DSL_MIN_CONFIDENCE = 0.0
            config.DSL_LLM_THRESHOLD = 1.0
            config.EXPLORATION_RATE = 1.0
            config.EXPLORATION_UNPROVEN_RATE = 1.0
            config.DSL_PROBATION_ENABLED = False
            config.RECURSIVE_DECOMPOSITION_ENABLED = False
            logger.info("Mode: TRAINING (explore signatures, collect data)")
        else:  # benchmark
            config.ACTIVE_MODE = Mode.BENCHMARK
            config.TRAINING_MODE = False
            config.MIN_MATCH_THRESHOLD = 0.95
            config.DSL_MIN_CONFIDENCE = 0.3
            config.DSL_LLM_THRESHOLD = 0.5
            config.EXPLORATION_RATE = 0.5
            config.EXPLORATION_UNPROVEN_RATE = 0.3
            config.DSL_PROBATION_ENABLED = True
            config.RECURSIVE_DECOMPOSITION_ENABLED = True
            logger.info("Mode: BENCHMARK (conservative matching, max accuracy)")

    run_pipeline(
        num_problems=args.problems,
        modes=args.modes,
        num_workers=args.workers,
        output_file=args.output,
        dataset=args.dataset,
        levels=args.levels,
        injection_mode=args.injection_mode,
        seed=args.seed,
        use_hints=args.use_hints,
        db_path=args.db,
    )


if __name__ == "__main__":
    main()
