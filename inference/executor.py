#!/usr/bin/env python3
"""
Mycelium v6: Symbolic Executor + Candidate Scorer

Executes candidate groupings through the full pipeline:
1. For each candidate grouping, classify each group
2. Extract arguments for each group
3. Execute symbolically
4. Score the result

The candidate with highest score wins.
"""

import math
from typing import Dict, List, Optional, Tuple, Any


class DSLExecutor:
    """Symbolic executor for arithmetic operations."""

    OPERATIONS = {
        "ADD": lambda a, b: a + b,
        "SUB": lambda a, b: a - b,
        "MUL": lambda a, b: a * b,
        "DIV": lambda a, b: a / b if b != 0 else None,
        "SET": lambda a: a,
        # Extensions for MATH
        "POW": lambda a, b: a ** b if b < 100 else None,  # Limit exponent
        "SQRT": lambda a: a ** 0.5 if a >= 0 else None,
        "MOD": lambda a, b: a % b if b != 0 else None,
        "NEG": lambda a: -a,
    }

    def __init__(self):
        self.trace = {}

    def execute_operation(
        self,
        op_type: str,
        arg1: Optional[float],
        arg2: Optional[float] = None,
    ) -> Optional[float]:
        """Execute a single operation."""
        if op_type not in self.OPERATIONS:
            return None

        op_func = self.OPERATIONS[op_type]

        try:
            if op_type in ["SET", "SQRT", "NEG"]:
                if arg1 is None:
                    return None
                return op_func(arg1)
            else:
                if arg1 is None or arg2 is None:
                    return None
                return op_func(arg1, arg2)
        except (ValueError, OverflowError, ZeroDivisionError):
            return None

    def execute_graph(
        self,
        operations: List[Dict],
        prev_results: Optional[List[float]] = None,
    ) -> Tuple[Optional[float], Dict]:
        """
        Execute a sequence of operations.

        Each operation is a dict with:
            - operation: str (ADD, SUB, MUL, DIV)
            - arg1: float | "PREV" | "PREV:N"
            - arg2: float | "PREV" | "PREV:N" | None

        Returns (answer, trace) where trace contains execution details.
        """
        results = prev_results.copy() if prev_results else []
        trace = {
            "steps": [],
            "all_args_consumed": True,
            "graph_connected": True,
            "final_op": None,
            "n_ops": len(operations),
        }

        for i, op in enumerate(operations):
            op_type = op.get("operation", "SET")
            arg1_spec = op.get("arg1")
            arg2_spec = op.get("arg2")

            # Resolve arguments
            arg1 = self._resolve_arg(arg1_spec, results)
            arg2 = self._resolve_arg(arg2_spec, results) if arg2_spec is not None else None

            # Execute
            result = self.execute_operation(op_type, arg1, arg2)

            if result is None:
                trace["steps"].append({
                    "op": op_type,
                    "arg1": arg1,
                    "arg2": arg2,
                    "result": None,
                    "error": True,
                })
                return None, trace

            results.append(result)
            trace["steps"].append({
                "op": op_type,
                "arg1": arg1,
                "arg2": arg2,
                "result": result,
            })

        if results:
            trace["final_op"] = operations[-1].get("operation") if operations else None
            return results[-1], trace

        return None, trace

    def _resolve_arg(
        self,
        arg_spec: Any,
        results: List[float],
    ) -> Optional[float]:
        """Resolve an argument specification to a value."""
        if arg_spec is None:
            return None

        if isinstance(arg_spec, (int, float)):
            return float(arg_spec)

        if isinstance(arg_spec, str):
            if arg_spec == "PREV":
                return results[-1] if results else None
            if arg_spec.startswith("PREV:"):
                try:
                    n = int(arg_spec.split(":")[1])
                    return results[-n] if len(results) >= n else None
                except (ValueError, IndexError):
                    return None

        return None


def score_candidate(
    answer: Optional[float],
    goal: Optional[Dict],
    trace: Dict,
) -> float:
    """
    Score an execution result for plausibility.

    Higher score = more plausible answer.
    """
    score = 0.0

    if answer is None:
        return -1.0

    # Execution succeeded
    score += 1.0

    # Answer type matches goal (if provided)
    if goal:
        goal_type = goal.get("type", "integer")
        if goal_type == "integer":
            if answer == int(answer):
                score += 0.5
        elif goal_type == "decimal":
            if answer != int(answer):
                score += 0.3

        # Answer operation matches goal hint
        if trace.get("final_op") == goal.get("op"):
            score += 0.5

    # Answer is reasonable magnitude (not astronomically large or tiny)
    if answer is not None:
        abs_answer = abs(answer)
        if 0 < abs_answer < 1e6:
            score += 0.3
        elif abs_answer == 0:
            score += 0.1  # Zero is sometimes valid

    # All arguments consumed (no dangling values)
    if trace.get("all_args_consumed", False):
        score += 0.2

    # Graph is connected (no isolated operations)
    if trace.get("graph_connected", False):
        score += 0.3

    # Simpler graphs are better (fewer operations = cleaner solution)
    n_ops = trace.get("n_ops", 0)
    if n_ops <= 3:
        score += 0.2
    elif n_ops <= 5:
        score += 0.1

    # Integer answers are often preferred for GSM8K
    if answer is not None and answer == int(answer):
        score += 0.1

    return score


def evaluate_candidates(
    candidates: List[Dict],
    goal: Optional[Dict] = None,
) -> List[Dict]:
    """
    Evaluate and rank candidates by score.

    Each candidate dict should have:
        - grouping: the span grouping
        - operations: list of operation dicts

    Returns sorted list of candidates with scores.
    """
    executor = DSLExecutor()
    results = []

    for candidate in candidates:
        operations = candidate.get("operations", [])

        try:
            answer, trace = executor.execute_graph(operations)
            score = score_candidate(answer, goal, trace)

            results.append({
                **candidate,
                "answer": answer,
                "score": score,
                "trace": trace,
            })
        except Exception as e:
            results.append({
                **candidate,
                "answer": None,
                "score": -2.0,
                "trace": {"error": str(e)},
            })

    # Sort by score descending
    results.sort(key=lambda x: x.get("score", -999), reverse=True)
    return results


def main():
    """Test executor with example operations."""
    print("=" * 60)
    print("SYMBOLIC EXECUTOR TEST")
    print("=" * 60)

    executor = DSLExecutor()

    # Example 1: Simple addition
    print("\nExample 1: 72 + 74 = 146")
    ops = [
        {"operation": "ADD", "arg1": 72, "arg2": 74},
    ]
    answer, trace = executor.execute_graph(ops)
    print(f"  Answer: {answer}")
    print(f"  Expected: 146")
    print(f"  Correct: {answer == 146}")

    # Example 2: Multi-step
    print("\nExample 2: (2 * 5) + (3 * 4) = 22")
    ops = [
        {"operation": "MUL", "arg1": 2, "arg2": 5},      # = 10
        {"operation": "MUL", "arg1": 3, "arg2": 4},      # = 12
        {"operation": "ADD", "arg1": "PREV:2", "arg2": "PREV"},  # 10 + 12 = 22
    ]
    answer, trace = executor.execute_graph(ops)
    print(f"  Answer: {answer}")
    print(f"  Expected: 22")
    print(f"  Correct: {answer == 22}")
    print(f"  Steps: {trace['steps']}")

    # Example 3: Change calculation
    print("\nExample 3: 50 - ((2*5) + (3*4)) = 28")
    ops = [
        {"operation": "MUL", "arg1": 2, "arg2": 5},      # = 10
        {"operation": "MUL", "arg1": 3, "arg2": 4},      # = 12
        {"operation": "ADD", "arg1": "PREV:2", "arg2": "PREV"},  # 10 + 12 = 22
        {"operation": "SUB", "arg1": 50, "arg2": "PREV"},       # 50 - 22 = 28
    ]
    answer, trace = executor.execute_graph(ops)
    print(f"  Answer: {answer}")
    print(f"  Expected: 28")
    print(f"  Correct: {answer == 28}")

    # Example 4: Scoring
    print("\n" + "-" * 40)
    print("Candidate Scoring Test")
    print("-" * 40)

    candidates = [
        {
            "name": "Correct grouping",
            "operations": [
                {"operation": "ADD", "arg1": 72, "arg2": 74},
            ],
        },
        {
            "name": "Wrong operation",
            "operations": [
                {"operation": "SUB", "arg1": 72, "arg2": 74},
            ],
        },
        {
            "name": "Over-complex",
            "operations": [
                {"operation": "SET", "arg1": 72},
                {"operation": "SET", "arg1": 74},
                {"operation": "ADD", "arg1": "PREV:2", "arg2": "PREV"},
            ],
        },
    ]

    goal = {"type": "integer", "op": "ADD"}
    results = evaluate_candidates(candidates, goal)

    print(f"\nGoal: type={goal['type']}, op={goal['op']}")
    print("\nRanked candidates:")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['name']}: answer={r['answer']}, score={r['score']:.2f}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
