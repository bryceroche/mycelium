"""CLI entry point for Mycelium solver."""

import asyncio
import sys

from mycelium.solver import Solver
from mycelium.step_signatures import StepSignatureDB


async def main():
    if len(sys.argv) < 2:
        print("Usage: python -m mycelium 'Your math problem here'")
        print("Example: python -m mycelium 'What is 15% of 80?'")
        sys.exit(1)

    problem = " ".join(sys.argv[1:])

    step_db = StepSignatureDB()
    solver = Solver(step_db=step_db)

    result = await solver.solve(problem=problem)

    print(f"\nProblem: {problem}")
    print(f"Answer: {result.answer}")
    print(f"Steps: {result.total_steps}")
    print(f"Signatures matched: {result.signatures_matched}")
    print(f"DSL injections: {result.steps_with_injection}")
    print(f"Time: {result.elapsed_ms:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
