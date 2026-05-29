"""v84 SymPy executor for the final-breath 3-list format.

v84's final breath (B4) emits text identical to v82's B6 format:

    "<ops_csv> | <types_csv> | <args_csv>"

So we reuse v82's parser verbatim. This file exists to keep launchers /
eval scripts self-documenting and to make the v84-paradigm wiring explicit.

Re-exports:
  b6_string_to_dag  (final-breath text -> SymPy DAG)
  dag_to_answer     (DAG -> float)
"""
from __future__ import annotations

import os
import sys

# Re-export from v82 (no behavior change — see module docstring for why).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v82_sympy_eval import b6_string_to_dag, dag_to_answer  # type: ignore  # noqa: F401


def main():
    # Run v82's self-test to sanity-check the parser still works in this
    # environment. Same test set — v84 final-breath format == v82 B6 format.
    from v82_sympy_eval import main as v82_main  # type: ignore
    v82_main()


if __name__ == "__main__":
    main()
