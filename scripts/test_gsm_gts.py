#!/usr/bin/env python3
"""Test GTS decomposition pipeline on GSM8K problems.

Tests:
1. GTS model loading
2. Prefix parsing and decomposition
3. Full pipeline with Solver

Usage:
    uv run python scripts/test_gsm_gts.py
"""

import sys
sys.path.insert(0, "src")

import asyncio
import logging
from datasets import load_dataset

from mycelium.gts_decomposer import GTSDecomposer
from mycelium.expression_tree import parse_prefix, decompose_to_atomic
from mycelium.solver import Solver
from mycelium.config import USE_GTS_DECOMPOSITION

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_gsm8k_problems(n: int = 10, seed: int = 42) -> list[dict]:
    """Load n GSM8K problems."""
    import random
    random.seed(seed)

    ds = load_dataset("gsm8k", "main", split="test")
    all_problems = []
    for i, item in enumerate(ds):
        answer = item["answer"].split("####")[-1].strip()
        all_problems.append({
            "id": f"gsm8k_{i}",
            "problem": item["question"],
            "answer": answer,
        })

    random.shuffle(all_problems)
    return all_problems[:n]


def test_prefix_parser():
    """Test that prefix parsing works."""
    print("\n" + "=" * 60)
    print("TEST 1: Prefix Parser")
    print("=" * 60)

    test_cases = [
        ("+ NUM_0 NUM_1", {"NUM_0": 5, "NUM_1": 3}),  # 5 + 3 = 8
        ("- NUM_0 NUM_1", {"NUM_0": 10, "NUM_1": 3}),  # 10 - 3 = 7
        ("* + NUM_0 NUM_1 NUM_2", {"NUM_0": 2, "NUM_1": 3, "NUM_2": 4}),  # (2+3)*4 = 20
        ("/ - NUM_0 NUM_1 NUM_2", {"NUM_0": 20, "NUM_1": 5, "NUM_2": 3}),  # (20-5)/3 = 5
    ]

    passed = 0
    for prefix, numbers in test_cases:
        try:
            tree = parse_prefix(prefix)
            steps = decompose_to_atomic(tree)
            print(f"  {prefix:30s} -> {len(steps)} atomic steps, depth={tree.depth}")
            passed += 1
        except Exception as e:
            print(f"  {prefix:30s} -> ERROR: {e}")

    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_gts_decomposer_from_prefix():
    """Test GTSDecomposer with known prefix."""
    print("\n" + "=" * 60)
    print("TEST 2: GTSDecomposer (from prefix)")
    print("=" * 60)

    # Test with a known prefix and number mapping
    decomposer = GTSDecomposer()

    # Multi-step expression: (5 + 3) * 2 = 16
    prefix = "* + NUM_0 NUM_1 NUM_2"
    numbers = {"NUM_0": 5, "NUM_1": 3, "NUM_2": 2}

    try:
        steps = decomposer.decompose_from_prefix(prefix, numbers)
        print(f"  Prefix: {prefix}")
        print(f"  Numbers: {numbers}")
        print(f"  Steps: {len(steps)}")
        for step in steps:
            print(f"    Step {step.step_number}: {step.operation}")
            print(f"      Values: {step.extracted_values}")
            print(f"      Depends on: {step.depends_on}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def test_gts_full_inference():
    """Test GTSDecomposer full inference (may raise NotImplementedError)."""
    print("\n" + "=" * 60)
    print("TEST 3: GTSDecomposer (full inference)")
    print("=" * 60)

    decomposer = GTSDecomposer()
    problem = "John has 5 apples. Mary gives him 3 more. How many apples does John have now?"

    try:
        steps = decomposer.decompose(problem)
        print(f"  Problem: {problem}")
        print(f"  Steps: {len(steps)}")
        for step in steps:
            print(f"    Step {step.step_number}: {step.operation}")
        return True
    except NotImplementedError as e:
        print(f"  Expected NotImplementedError: {e}")
        print("  (GTS beam search not yet implemented)")
        return "expected"  # Expected failure
    except Exception as e:
        print(f"  Unexpected ERROR: {e}")
        return False


async def test_solver_with_gts():
    """Test Solver with GTS decomposition."""
    print("\n" + "=" * 60)
    print("TEST 4: Solver with GTS")
    print("=" * 60)

    print(f"  USE_GTS_DECOMPOSITION = {USE_GTS_DECOMPOSITION}")

    # Use a fresh temporary database
    import tempfile
    import os
    from mycelium.data_layer.connection import reset_db

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        test_db = f.name

    # Reset global DB connection and set env var so mcts.py uses same DB
    old_db_path = os.environ.get("MYCELIUM_DB_PATH")
    os.environ["MYCELIUM_DB_PATH"] = test_db
    reset_db()

    try:
        solver = Solver(db_path=test_db)
        problem = "John has 5 apples. Mary gives him 3 more. How many apples does John have now?"
        expected = "8"

        result = await solver.solve(problem, expected_answer=expected)
        print(f"  Problem: {problem}")
        print(f"  Expected: {expected}")
        print(f"  Answer: {result.answer}")
        print(f"  Success: {result.success}")
        print(f"  Steps executed: {result.steps_executed}")
        if result.error:
            print(f"  Error: {result.error}")
        return result.success
    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    finally:
        # Cleanup
        reset_db()
        if old_db_path:
            os.environ["MYCELIUM_DB_PATH"] = old_db_path
        elif "MYCELIUM_DB_PATH" in os.environ:
            del os.environ["MYCELIUM_DB_PATH"]
        if os.path.exists(test_db):
            os.remove(test_db)


async def test_gsm8k_problems(n: int = 10):
    """Test solver on n GSM8K problems."""
    print("\n" + "=" * 60)
    print(f"TEST 5: GSM8K Problems ({n} problems)")
    print("=" * 60)

    # Use a fresh temporary database
    import tempfile
    import os
    from mycelium.data_layer.connection import reset_db

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        test_db = f.name

    # Reset global DB connection and set env var
    old_db_path = os.environ.get("MYCELIUM_DB_PATH")
    os.environ["MYCELIUM_DB_PATH"] = test_db
    reset_db()

    try:
        problems = load_gsm8k_problems(n)
        solver = Solver(db_path=test_db)

        successes = 0
        failures_decomp = 0
        failures_exec = 0

        for i, p in enumerate(problems):
            try:
                result = await solver.solve(p["problem"], expected_answer=p["answer"])
                status = "PASS" if result.success else "FAIL"
                if result.success:
                    successes += 1
                elif result.error and "decompose" in result.error.lower():
                    failures_decomp += 1
                else:
                    failures_exec += 1

                print(f"  [{i+1:2d}] {status}: expected={p['answer']:>10s}, got={str(result.answer or ''):>10s} ({result.steps_executed} steps)")
            except Exception as e:
                print(f"  [{i+1:2d}] ERROR: {e}")
                failures_decomp += 1

        print(f"\nResults: {successes}/{n} correct")
        print(f"  Decomposition failures: {failures_decomp}")
        print(f"  Execution failures: {failures_exec}")

        return successes, failures_decomp, failures_exec
    finally:
        # Cleanup
        reset_db()
        if old_db_path:
            os.environ["MYCELIUM_DB_PATH"] = old_db_path
        elif "MYCELIUM_DB_PATH" in os.environ:
            del os.environ["MYCELIUM_DB_PATH"]
        if os.path.exists(test_db):
            os.remove(test_db)


async def main():
    """Run all tests."""
    print("=" * 60)
    print("GTS DECOMPOSITION PIPELINE TEST")
    print("=" * 60)

    # Test 1: Prefix parser
    test_prefix_parser()

    # Test 2: GTSDecomposer from prefix
    test_gts_decomposer_from_prefix()

    # Test 3: GTS full inference
    test_gts_full_inference()

    # Test 4: Solver with GTS
    await test_solver_with_gts()

    # Test 5: GSM8K problems
    await test_gsm8k_problems(10)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
