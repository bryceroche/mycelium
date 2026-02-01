"""Execute patterns with SymPy or step-based evaluation."""
import json
import logging
import os
import re
from typing import Any, Dict, Optional

import sympy
from sympy import symbols, simplify, factor, expand, sqrt, Rational, solve, Eq, log, I
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

from .registry import Pattern
from mycelium.mathdecomp.llm_api import call_openai, call_anthropic

logger = logging.getLogger(__name__)

# Safe eval allowlist
SAFE_BUILTINS = {
    "__builtins__": {},
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "int": int,
    "float": float,
    "pow": pow,
    "divmod": divmod,
    "log": lambda x, base=None: __import__("math").log(x) if base is None else __import__("math").log(x, base),
}


def _call_llm(prompt: str) -> str:
    """Call LLM using available provider."""
    if os.environ.get("OPENAI_API_KEY"):
        return call_openai(prompt)
    elif os.environ.get("ANTHROPIC_API_KEY"):
        return call_anthropic(prompt)
    else:
        raise ValueError("No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")


def _safe_eval(expression: str, context: Dict[str, Any] = None) -> Any:
    """Safely evaluate a Python expression."""
    allowed = SAFE_BUILTINS.copy()
    if context:
        allowed.update(context)
    return eval(expression, allowed, {})


def _execute_steps(decomposition: Dict[str, Any]) -> Any:
    """Execute step-based decomposition."""
    steps = decomposition.get("steps", [])
    answer_key = decomposition.get("answer", "")

    if not steps:
        # Try direct answer
        if "answer" in decomposition:
            ans = decomposition["answer"]
            if isinstance(ans, (int, float)):
                return ans
            if isinstance(ans, str) and ans.replace(".", "").replace("-", "").isdigit():
                return float(ans) if "." in ans else int(ans)
        return None

    context = {}

    for step in steps:
        expr = step.get("expr", "")
        result_name = step.get("result", "")

        if not expr or not result_name:
            continue

        try:
            # Evaluate expression with context
            value = _safe_eval(expr, context)
            context[result_name] = value
            logger.debug(f"[executor] {result_name} = {expr} = {value}")
        except Exception as e:
            logger.warning(f"[executor] Failed to eval '{expr}': {e}")
            continue

    # Get final answer
    if answer_key and answer_key in context:
        return context[answer_key]

    # Return last computed value
    if context:
        return list(context.values())[-1]

    return None


def _execute_sympy(decomposition: Dict[str, Any]) -> Any:
    """Execute SymPy-based operations (simplify, factor, expand)."""
    operation = decomposition.get("operation", "simplify")
    expression = decomposition.get("expression", "")
    variable = decomposition.get("variable", "x")

    if not expression:
        return None

    try:
        # Parse expression
        transformations = standard_transformations + (implicit_multiplication_application,)

        # Create symbol
        var = symbols(variable)
        local_dict = {variable: var, "sqrt": sqrt, "Rational": Rational, "I": I}

        expr = parse_expr(expression, local_dict=local_dict, transformations=transformations)

        # Apply operation
        if operation == "simplify":
            result = simplify(expr)
        elif operation == "factor":
            result = factor(expr)
        elif operation == "expand":
            result = expand(expr)
        elif operation == "rationalize":
            # Rationalize denominator
            result = simplify(sympy.radsimp(expr))
        else:
            result = simplify(expr)

        logger.info(f"[executor] SymPy {operation}: {expression} = {result}")
        return result

    except Exception as e:
        logger.error(f"[executor] SymPy error: {e}")
        return None


def _execute_sympy_solve(decomposition: Dict[str, Any]) -> Any:
    """Execute SymPy equation solving."""
    # Handle single equation
    if "equation" in decomposition:
        equation = decomposition.get("equation", "")
        variable = decomposition.get("variable", "x")

        if not equation:
            return None

        try:
            var = symbols(variable)
            local_dict = {
                variable: var,
                "Eq": Eq,
                "sqrt": sqrt,
                "Rational": Rational,
                "log": log,
                "I": I,
            }

            # Parse and solve
            eq = eval(equation, {"__builtins__": {}, **local_dict})
            solutions = solve(eq, var)

            logger.info(f"[executor] Solved {equation}: {solutions}")

            if len(solutions) == 1:
                return solutions[0]
            return solutions

        except Exception as e:
            logger.error(f"[executor] SymPy solve error: {e}")
            return None

    # Handle system of equations
    if "equations" in decomposition:
        equations = decomposition.get("equations", [])
        unknowns = decomposition.get("unknowns", [])
        find_expr = decomposition.get("find", "")

        try:
            # Create symbols
            syms = symbols(" ".join(unknowns))
            if len(unknowns) == 1:
                syms = [syms]

            local_dict = {name: sym for name, sym in zip(unknowns, syms)}
            local_dict.update({"Eq": Eq, "sqrt": sqrt, "Rational": Rational})

            # Parse equations
            eqs = []
            for eq_str in equations:
                eq = eval(eq_str, {"__builtins__": {}, **local_dict})
                eqs.append(eq)

            # Solve system
            solutions = solve(eqs, syms)
            logger.info(f"[executor] System solution: {solutions}")

            # Compute requested expression
            if find_expr and solutions:
                if isinstance(solutions, dict):
                    result = eval(find_expr, {"__builtins__": {}, **solutions})
                    return result
                elif isinstance(solutions, list) and len(solutions) == 1:
                    sol_dict = {str(s): v for s, v in zip(syms, solutions[0])}
                    result = eval(find_expr, {"__builtins__": {}, **sol_dict})
                    return result

            return solutions

        except Exception as e:
            logger.error(f"[executor] SymPy system solve error: {e}")
            return None

    return None


def _execute_direct(decomposition: Dict[str, Any]) -> Any:
    """Execute direct answer patterns (circle, midpoint)."""
    answer = decomposition.get("answer", "")

    if not answer:
        return None

    # Try to parse as number
    if isinstance(answer, (int, float)):
        return answer

    answer_str = str(answer).strip()

    # Try numeric
    try:
        if "." in answer_str:
            return float(answer_str)
        return int(answer_str)
    except ValueError:
        pass

    # Return as string (e.g., coordinate pair)
    return answer_str


def execute_pattern(problem: str, pattern: Pattern) -> Any:
    """
    Execute a pattern on a problem.

    Args:
        problem: The problem text
        pattern: The matched pattern

    Returns:
        The computed answer
    """
    # Build prompt
    prompt = pattern.prompt_template.replace("{problem}", problem)

    # Call LLM
    try:
        response = _call_llm(prompt)
    except Exception as e:
        logger.error(f"[executor] LLM call failed: {e}")
        return None

    # Parse response
    try:
        # Strip markdown code blocks if present
        response = re.sub(r"^```json\s*", "", response.strip())
        response = re.sub(r"\s*```$", "", response)

        decomposition = json.loads(response)
        logger.debug(f"[executor] Decomposition: {decomposition}")
    except json.JSONDecodeError as e:
        logger.error(f"[executor] Failed to parse LLM response: {e}")
        logger.debug(f"[executor] Raw response: {response}")
        return None

    # Execute based on type
    if pattern.execution_type == "steps":
        return _execute_steps(decomposition)
    elif pattern.execution_type == "sympy":
        return _execute_sympy(decomposition)
    elif pattern.execution_type == "sympy_solve":
        return _execute_sympy_solve(decomposition)
    elif pattern.execution_type == "direct":
        return _execute_direct(decomposition)
    else:
        logger.warning(f"[executor] Unknown execution type: {pattern.execution_type}")
        return _execute_steps(decomposition)  # Fallback to steps
