"""
C5: SymPy Executor — Non-differentiable verification oracle

Takes: operation sequence + arguments + dependency graph (DAG)
Returns: (result, success, plausibility_score)

The plausibility score becomes the REWARD SIGNAL for joint training.
This component has NO learnable parameters — it's pure symbolic execution.

Used in two ways:
  1. Training: provides reward signal for REINFORCE gradient
  2. Inference: final answer verification after belief propagation converges
"""

import torch
import sympy
from sympy import symbols, solve, sqrt, gcd, lcm, binomial, Rational
from typing import Optional, Union
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Operation label vocabulary (must match C2's label set)
# ---------------------------------------------------------------------------

OP_LABELS = [
    "ADD", "SUB", "MUL", "DIV",
    "POW", "SQRT", "MOD",
    "PERCENT_OF", "PERCENT_CHANGE",
    "RATIO", "PROPORTION",
    "SOLVE_LINEAR", "SOLVE_QUADRATIC", "SOLVE_SYSTEM",
    "GCD", "LCM", "COMB", "PERM",
    "ABS", "NEG", "RECIPROCAL",
    "SUM", "PRODUCT", "MEAN",
    "MIN", "MAX",
    "FLOOR", "CEIL", "ROUND",
    # Extend as needed from IB template discovery
]

N_OPS = len(OP_LABELS)
OP_TO_IDX = {op: i for i, op in enumerate(OP_LABELS)}
IDX_TO_OP = {i: op for i, op in enumerate(OP_LABELS)}


# ---------------------------------------------------------------------------
# Helper functions for complex operations
# ---------------------------------------------------------------------------

def _solve_linear(a: float, b: float, c: float) -> Optional[float]:
    """Solve ax + b = c for x."""
    if a == 0:
        return None
    return (c - b) / a


def _solve_quadratic(a: float, b: float, c: float) -> Optional[list]:
    """Solve ax^2 + bx + c = 0. Returns list of real solutions."""
    if a == 0:
        return _solve_linear(1, b, c)

    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None  # Complex roots — not handling for now

    sqrt_d = discriminant ** 0.5
    x1 = (-b + sqrt_d) / (2*a)
    x2 = (-b - sqrt_d) / (2*a)

    if abs(x1 - x2) < 1e-10:
        return [x1]
    return [x1, x2]


def _safe_div(a: float, b: float) -> Optional[float]:
    """Division with zero check."""
    if abs(b) < 1e-15:
        return None
    return a / b


def _safe_mod(a: float, b: float) -> Optional[float]:
    """Modulo with zero check."""
    if abs(b) < 1e-15:
        return None
    return a % b


def _safe_pow(a: float, b: float) -> Optional[float]:
    """Power with overflow protection."""
    try:
        result = a ** b
        if abs(result) > 1e15:
            return None
        return result
    except (OverflowError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Operation dispatch table
# ---------------------------------------------------------------------------

SYMPY_OPS = {
    # Basic arithmetic
    "ADD": lambda a, b: a + b,
    "SUB": lambda a, b: a - b,
    "MUL": lambda a, b: a * b,
    "DIV": _safe_div,
    "POW": _safe_pow,
    "SQRT": lambda a: sqrt(float(a)) if a >= 0 else None,
    "MOD": _safe_mod,

    # Percentage operations
    "PERCENT_OF": lambda a, b: a * b / 100,  # a% of b
    "PERCENT_CHANGE": lambda a, b: _safe_div((b - a), a) * 100 if a != 0 else None,

    # Ratio operations
    "RATIO": _safe_div,
    "PROPORTION": lambda a, b, c: _safe_div(a * c, b),  # a/b = x/c → x = ac/b

    # Equation solving
    "SOLVE_LINEAR": _solve_linear,
    "SOLVE_QUADRATIC": _solve_quadratic,

    # Number theory
    "GCD": lambda a, b: int(gcd(int(a), int(b))),
    "LCM": lambda a, b: int(lcm(int(a), int(b))),
    "COMB": lambda n, k: int(binomial(int(n), int(k))),
    "PERM": lambda n, k: int(binomial(int(n), int(k))) * int(sympy.factorial(int(k))),

    # Unary operations
    "ABS": lambda a: abs(a),
    "NEG": lambda a: -a,
    "RECIPROCAL": lambda a: _safe_div(1, a),

    # Aggregations (variable arity)
    "SUM": lambda *args: sum(args),
    "PRODUCT": lambda *args: sympy.prod(args),
    "MEAN": lambda *args: sum(args) / len(args) if args else None,
    "MIN": lambda *args: min(args),
    "MAX": lambda *args: max(args),

    # Rounding
    "FLOOR": lambda a: int(a // 1),
    "CEIL": lambda a: int(-(-a // 1)),
    "ROUND": lambda a: round(a),
}

# Arity table: how many args each operation expects
# -1 means variable arity (SUM, PRODUCT, etc.)
OP_ARITY = {
    "ADD": 2, "SUB": 2, "MUL": 2, "DIV": 2,
    "POW": 2, "SQRT": 1, "MOD": 2,
    "PERCENT_OF": 2, "PERCENT_CHANGE": 2,
    "RATIO": 2, "PROPORTION": 3,
    "SOLVE_LINEAR": 3, "SOLVE_QUADRATIC": 3,
    "GCD": 2, "LCM": 2, "COMB": 2, "PERM": 2,
    "ABS": 1, "NEG": 1, "RECIPROCAL": 1,
    "SUM": -1, "PRODUCT": -1, "MEAN": -1, "MIN": -1, "MAX": -1,
    "FLOOR": 1, "CEIL": 1, "ROUND": 1,
}


# ---------------------------------------------------------------------------
# Execution result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Result of executing a computation graph."""
    result: Optional[float]       # Final numeric answer
    success: bool                 # Whether execution completed
    plausibility: float           # [0, 1] reward signal
    intermediate: list[float]     # Per-node results (for debugging)
    error: Optional[str] = None   # Error message if failed


# ---------------------------------------------------------------------------
# C5 Executor class
# ---------------------------------------------------------------------------

class C5_SymPyExecutor:
    """
    Pure verification oracle. No parameters. Non-differentiable.

    Takes: operation sequence + arguments + dependency graph
    Returns: ExecutionResult with (result, success, plausibility)

    The plausibility score becomes the REWARD SIGNAL for joint training.
    """

    def __init__(self, result_bounds: tuple = (-1e12, 1e12)):
        """
        Args:
            result_bounds: (min, max) for plausible result range
        """
        self.result_bounds = result_bounds

    def execute(
        self,
        ops: list[str],
        args: list[list[Optional[float]]],
        adjacency: torch.Tensor,
    ) -> ExecutionResult:
        """
        Execute the computation graph.

        Args:
            ops: List of operation names, one per node
            args: List of argument lists, one per node.
                  None values indicate slots filled by dependency results.
            adjacency: (n, n) binary tensor where adjacency[i,j]=1 means
                       node i's result feeds into node j

        Returns:
            ExecutionResult with answer and plausibility score
        """
        n = len(ops)
        if n == 0:
            return ExecutionResult(None, False, 0.0, [], "Empty operation sequence")

        # Topological sort
        order = self._topo_sort(adjacency, n)
        if order is None:
            return ExecutionResult(None, False, 0.0, [], "Cycle detected in DAG")

        results = [None] * n

        try:
            for idx in order:
                op_name = ops[idx]

                # Get operation function
                op_fn = SYMPY_OPS.get(op_name)
                if op_fn is None:
                    return ExecutionResult(
                        None, False, 0.0, results,
                        f"Unknown operation: {op_name}"
                    )

                # Collect arguments: start with extracted literals
                op_args = list(args[idx]) if args[idx] else []

                # Resolve dependencies: find nodes that feed into this node
                # adjacency[i, idx] = 1 means node i's output goes to node idx
                if adjacency.dim() == 2:
                    deps = (adjacency[:, idx] > 0.5).nonzero(as_tuple=True)[0]
                else:
                    deps = []

                # Insert dependency results into None slots
                dep_results = [results[d.item()] for d in deps if results[d.item()] is not None]

                # Fill None slots with dependency results
                filled_args = []
                dep_iter = iter(dep_results)
                for arg in op_args:
                    if arg is None:
                        try:
                            filled_args.append(next(dep_iter))
                        except StopIteration:
                            pass  # No more deps to fill
                    else:
                        filled_args.append(arg)

                # Add remaining dependency results
                for dep_val in dep_iter:
                    filled_args.append(dep_val)

                # Check arity
                expected_arity = OP_ARITY.get(op_name, 0)
                if expected_arity > 0 and len(filled_args) < expected_arity:
                    return ExecutionResult(
                        None, False, 0.0, results,
                        f"Not enough args for {op_name}: got {len(filled_args)}, need {expected_arity}"
                    )

                # Execute
                if expected_arity == -1:  # Variable arity
                    result = op_fn(*filled_args)
                elif expected_arity == 1:
                    result = op_fn(filled_args[0])
                elif expected_arity == 2:
                    result = op_fn(filled_args[0], filled_args[1])
                elif expected_arity == 3:
                    result = op_fn(filled_args[0], filled_args[1], filled_args[2])
                else:
                    result = op_fn(*filled_args[:expected_arity])

                if result is None:
                    return ExecutionResult(
                        None, False, 0.0, results,
                        f"Operation {op_name} returned None"
                    )

                # Convert sympy to float if needed
                if hasattr(result, 'evalf'):
                    result = float(result.evalf())
                elif isinstance(result, list):
                    # For equation solvers that return multiple solutions
                    result = result[0] if result else None

                results[idx] = float(result) if result is not None else None

            # Final result is from the last node in topological order
            final = results[order[-1]]
            if final is None:
                return ExecutionResult(None, False, 0.0, results, "Final result is None")

            plausibility = self._score_plausibility(final, ops, args)
            return ExecutionResult(final, True, plausibility, results)

        except Exception as e:
            return ExecutionResult(None, False, 0.0, results, str(e))

    def _topo_sort(self, adj: torch.Tensor, n: int) -> Optional[list[int]]:
        """
        Kahn's algorithm for topological sort.
        Returns None if cycle detected.
        """
        adj_np = (adj > 0.5).cpu().numpy() if torch.is_tensor(adj) else adj
        in_degree = adj_np.sum(axis=0)

        # Start with nodes that have no incoming edges
        queue = [i for i in range(n) if in_degree[i] == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)

            # Remove edges from this node
            for j in range(n):
                if adj_np[node, j]:
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        queue.append(j)

        # If we didn't visit all nodes, there's a cycle
        return order if len(order) == n else None

    def _score_plausibility(
        self,
        result: float,
        ops: list[str],
        args: list[list[Optional[float]]],
    ) -> float:
        """
        Score how plausible the result is.

        Currently uses simple bounds checking.
        TODO: Replace with learned plausibility scorer trained on
              teacher (correct_answer, pipeline_answer) pairs.

        Returns:
            Float in [0, 1] — higher means more plausible
        """
        if result is None:
            return 0.0

        # Check bounds
        min_bound, max_bound = self.result_bounds
        if not (min_bound < result < max_bound):
            return 0.0

        # Check for NaN/Inf
        if not (result == result):  # NaN check
            return 0.0

        return 1.0  # Placeholder — real version is learned

    def verify_against_gold(
        self,
        ops: list[str],
        args: list[list[Optional[float]]],
        adjacency: torch.Tensor,
        gold_answer: float,
        tolerance: float = 1e-6,
    ) -> tuple[bool, float]:
        """
        Execute and compare against gold answer.

        Returns:
            (is_correct, reward)
            - is_correct: bool, whether result matches gold
            - reward: float, 1.0 if correct, 0.0 otherwise (for REINFORCE)
        """
        result = self.execute(ops, args, adjacency)

        if not result.success:
            return False, 0.0

        # Check if result matches gold
        if abs(result.result - gold_answer) < tolerance:
            return True, 1.0

        # Relative tolerance for large numbers
        if abs(gold_answer) > 1e-6:
            rel_error = abs(result.result - gold_answer) / abs(gold_answer)
            if rel_error < tolerance:
                return True, 1.0

        return False, 0.0


# ---------------------------------------------------------------------------
# Batch execution utilities
# ---------------------------------------------------------------------------

def batch_execute(
    executor: C5_SymPyExecutor,
    batch_ops: list[list[str]],
    batch_args: list[list[list[Optional[float]]]],
    batch_adjacency: list[torch.Tensor],
) -> list[ExecutionResult]:
    """
    Execute a batch of computation graphs.

    Note: This is NOT parallelized — SymPy execution is inherently sequential.
    For training, the batch dimension comes from the dataloader, and each
    sample is executed independently.
    """
    results = []
    for ops, args, adj in zip(batch_ops, batch_args, batch_adjacency):
        results.append(executor.execute(ops, args, adj))
    return results


def compute_reinforce_reward(
    results: list[ExecutionResult],
    gold_answers: list[float],
    baseline: float = 0.5,
    tolerance: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute REINFORCE rewards and advantages for a batch.

    Args:
        results: ExecutionResult list from batch_execute
        gold_answers: Ground truth answers
        baseline: Moving average baseline for variance reduction
        tolerance: Numerical tolerance for correctness check

    Returns:
        rewards: (batch,) tensor of 0/1 rewards
        advantages: (batch,) tensor of (reward - baseline)
    """
    rewards = []
    for result, gold in zip(results, gold_answers):
        if not result.success:
            rewards.append(0.0)
            continue

        # Check correctness
        diff = abs(result.result - gold)
        if diff < tolerance:
            rewards.append(1.0)
        elif abs(gold) > 1e-6 and diff / abs(gold) < tolerance:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    rewards = torch.tensor(rewards)
    advantages = rewards - baseline

    return rewards, advantages


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick sanity check
    executor = C5_SymPyExecutor()

    # Test 1: Simple addition chain
    # (2 + 3) then * 4 = 20
    ops = ["ADD", "MUL"]
    args = [[2.0, 3.0], [None, 4.0]]  # None = filled by dependency
    adj = torch.tensor([
        [0, 1],  # Node 0 feeds into Node 1
        [0, 0],
    ], dtype=torch.float)

    result = executor.execute(ops, args, adj)
    print(f"Test 1 (2+3)*4: {result}")
    assert result.success and abs(result.result - 20.0) < 1e-6

    # Test 2: Division by zero
    ops = ["DIV"]
    args = [[5.0, 0.0]]
    adj = torch.zeros(1, 1)

    result = executor.execute(ops, args, adj)
    print(f"Test 2 (5/0): {result}")
    assert not result.success

    # Test 3: SQRT
    ops = ["SQRT"]
    args = [[16.0]]
    adj = torch.zeros(1, 1)

    result = executor.execute(ops, args, adj)
    print(f"Test 3 sqrt(16): {result}")
    assert result.success and abs(result.result - 4.0) < 1e-6

    # Test 4: GCD
    ops = ["GCD"]
    args = [[12.0, 8.0]]
    adj = torch.zeros(1, 1)

    result = executor.execute(ops, args, adj)
    print(f"Test 4 gcd(12,8): {result}")
    assert result.success and result.result == 4.0

    # Test 5: Multi-step DAG
    # Node 0: 10 + 5 = 15
    # Node 1: 20 - 8 = 12
    # Node 2: Node0 * Node1 = 15 * 12 = 180
    ops = ["ADD", "SUB", "MUL"]
    args = [[10.0, 5.0], [20.0, 8.0], [None, None]]
    adj = torch.tensor([
        [0, 0, 1],  # Node 0 → Node 2
        [0, 0, 1],  # Node 1 → Node 2
        [0, 0, 0],
    ], dtype=torch.float)

    result = executor.execute(ops, args, adj)
    print(f"Test 5 (10+5)*(20-8): {result}")
    assert result.success and abs(result.result - 180.0) < 1e-6

    print("\nAll tests passed!")
