"""
Scaffold-Guided Verification Factor Graph

Architecture shift:
    OLD: Factor graph selects templates + extracts operands (FAILED)
    NEW: LLM generates within scaffold, factor graph VERIFIES + CORRECTS

Components:
    1. Scaffold: step types, dependencies, expected output types
    2. LLM: generates mathematical expressions per slot
    3. Factor graph: verifies consistency, localizes errors, routes corrections

Energy terms:
    - structural_consistency: does step match scaffold type?
    - dependency_flow: do outputs flow to dependent steps?
    - execution_validity: does step parse and execute?
    - verification: does answer check backwards?
"""

import re
import json
import sympy
from sympy import Symbol, Eq, solve, simplify, expand, sqrt, Rational, pi
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import boto3

s3 = boto3.client("s3")
BUCKET = "mycelium-data"


class StepType(Enum):
    SETUP = "setup"           # Establish equation/expression
    SUBSTITUTE = "substitute"  # Plug in values
    SIMPLIFY = "simplify"      # Algebraic simplification
    SOLVE = "solve"            # Solve for variable
    EVALUATE = "evaluate"      # Compute numeric result
    VERIFY = "verify"          # Check answer


@dataclass
class ScaffoldStep:
    """One step in the scaffold."""
    idx: int
    step_type: StepType
    depends_on: List[int] = field(default_factory=list)
    expected_output: str = "expression"  # "equation", "expression", "number"

    # Filled by LLM generation
    generated_text: str = ""
    parsed_expr: Any = None

    # Filled by verification
    energy: float = 0.0
    errors: List[str] = field(default_factory=list)


@dataclass
class Scaffold:
    """Complete scaffold for a problem."""
    steps: List[ScaffoldStep]
    final_step_idx: int = -1

    def __post_init__(self):
        if self.final_step_idx < 0:
            self.final_step_idx = len(self.steps) - 1


def extract_scaffold_from_cot(cot_text: str) -> Scaffold:
    """
    Extract scaffold structure from teacher's CoT.
    Infers step types from text patterns.
    """
    # Split into steps (sentences)
    steps_text = re.split(r'(?<=[.!?])\s+', cot_text)
    steps_text = [s.strip() for s in steps_text if len(s.strip()) > 10]

    scaffold_steps = []

    for i, text in enumerate(steps_text[:8]):  # Limit to 8 steps
        # Infer step type from text
        text_lower = text.lower()

        if any(kw in text_lower for kw in ["let", "given", "we have", "start with"]):
            step_type = StepType.SETUP
        elif any(kw in text_lower for kw in ["substitute", "plug", "replacing"]):
            step_type = StepType.SUBSTITUTE
        elif any(kw in text_lower for kw in ["simplif", "combin", "factor", "expand"]):
            step_type = StepType.SIMPLIFY
        elif any(kw in text_lower for kw in ["solve", "solving", "find"]):
            step_type = StepType.SOLVE
        elif any(kw in text_lower for kw in ["evaluat", "compute", "calculat", "="]):
            step_type = StepType.EVALUATE
        elif any(kw in text_lower for kw in ["check", "verify", "confirm"]):
            step_type = StepType.VERIFY
        else:
            step_type = StepType.EVALUATE

        # Dependencies: each step depends on previous (simple chain)
        depends = [i - 1] if i > 0 else []

        # Expected output type
        if step_type in [StepType.SETUP, StepType.SOLVE]:
            expected = "equation"
        elif step_type == StepType.EVALUATE:
            expected = "number"
        else:
            expected = "expression"

        scaffold_steps.append(ScaffoldStep(
            idx=i,
            step_type=step_type,
            depends_on=depends,
            expected_output=expected,
            generated_text=text,
        ))

    return Scaffold(steps=scaffold_steps)


def parse_math_from_text(text: str) -> Optional[Any]:
    """Parse mathematical expression from text."""
    # Extract LaTeX
    latex_patterns = [
        r'\$\$(.+?)\$\$',
        r'\$(.+?)\$',
        r'\\boxed\{(.+?)\}',
    ]

    for pattern in latex_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            latex = match.group(1)
            parsed = latex_to_sympy(latex)
            if parsed is not None:
                return parsed

    # Try to find equation patterns
    eq_match = re.search(r'([a-zA-Z0-9\s\+\-\*\/\^]+)\s*=\s*([a-zA-Z0-9\s\+\-\*\/\^]+)', text)
    if eq_match:
        try:
            lhs = parse_expr(eq_match.group(1).strip(),
                           transformations=standard_transformations + (implicit_multiplication,))
            rhs = parse_expr(eq_match.group(2).strip(),
                           transformations=standard_transformations + (implicit_multiplication,))
            return Eq(lhs, rhs)
        except:
            pass

    return None


def latex_to_sympy(latex: str) -> Optional[Any]:
    """Convert LaTeX to SymPy (simplified version)."""
    s = latex.strip()

    # Clean LaTeX
    s = s.replace('\\left', '').replace('\\right', '')
    s = s.replace('\\cdot', '*').replace('\\times', '*')
    s = s.replace('\\div', '/')

    # Fractions
    for _ in range(5):
        new_s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', s)
        if new_s == s:
            break
        s = new_s

    # Square roots
    s = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', s)

    # Powers
    s = re.sub(r'\^{([^{}]+)}', r'**(\1)', s)
    s = re.sub(r'\^(\d+)', r'**\1', s)

    # Greek/constants
    s = s.replace('\\pi', 'pi')

    # Implicit multiplication
    s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)

    # Clean
    s = re.sub(r'\\text\{[^}]*\}', '', s)
    s = re.sub(r'\s+', ' ', s).strip()

    if not s:
        return None

    try:
        # Check for equation
        if '=' in s:
            parts = s.split('=')
            if len(parts) == 2:
                lhs = parse_expr(parts[0].strip(),
                               transformations=standard_transformations + (implicit_multiplication,))
                rhs = parse_expr(parts[1].strip(),
                               transformations=standard_transformations + (implicit_multiplication,))
                return Eq(lhs, rhs)

        return parse_expr(s, transformations=standard_transformations + (implicit_multiplication,))
    except:
        return None


# =============================================================================
# FACTOR GRAPH VERIFICATION
# =============================================================================

def compute_type_energy(step: ScaffoldStep) -> Tuple[float, List[str]]:
    """Check if step output matches expected type."""
    errors = []

    if step.parsed_expr is None:
        return 1.0, ["no_parse"]

    expr = step.parsed_expr
    expected = step.expected_output

    if expected == "equation":
        if not isinstance(expr, Eq):
            errors.append("expected_equation_got_expression")
            return 0.5, errors

    elif expected == "number":
        try:
            if not expr.is_number:
                errors.append("expected_number_got_expression")
                return 0.5, errors
        except:
            pass

    return 0.0, errors


def compute_dependency_energy(step: ScaffoldStep, all_steps: List[ScaffoldStep]) -> Tuple[float, List[str]]:
    """Check if step uses outputs from dependencies."""
    errors = []

    if not step.depends_on:
        return 0.0, errors

    if step.parsed_expr is None:
        return 1.0, ["no_parse"]

    # Get symbols used in this step
    try:
        step_symbols = step.parsed_expr.free_symbols
    except:
        step_symbols = set()

    # Check if any dependency symbols flow through
    dep_connected = False
    for dep_idx in step.depends_on:
        if dep_idx < len(all_steps):
            dep_step = all_steps[dep_idx]
            if dep_step.parsed_expr is not None:
                try:
                    dep_symbols = dep_step.parsed_expr.free_symbols
                    if step_symbols & dep_symbols:
                        dep_connected = True
                        break
                except:
                    pass

    # Allow numeric steps to not share symbols (they resolve to numbers)
    if not dep_connected and step.expected_output != "number":
        errors.append("no_symbol_flow_from_dependency")
        return 0.3, errors

    return 0.0, errors


def compute_execution_energy(step: ScaffoldStep) -> Tuple[float, List[str]]:
    """Check if step can be executed."""
    errors = []

    if step.parsed_expr is None:
        return 1.0, ["no_parse"]

    expr = step.parsed_expr

    # Try to simplify
    try:
        if isinstance(expr, Eq):
            simplified = Eq(simplify(expr.lhs), simplify(expr.rhs))
        else:
            simplified = simplify(expr)
        return 0.0, errors
    except Exception as e:
        errors.append(f"simplify_failed: {e}")
        return 0.5, errors


def compute_verification_energy(scaffold: Scaffold, expected_answer: str) -> Tuple[float, List[str], Optional[Any]]:
    """
    Check if final answer verifies backwards.
    This is the key consistency check.
    """
    errors = []

    final_step = scaffold.steps[scaffold.final_step_idx]
    if final_step.parsed_expr is None:
        return 1.0, ["no_final_parse"], None

    # Parse expected answer
    expected = latex_to_sympy(expected_answer)
    if expected is None:
        # Try direct parse
        try:
            expected = parse_expr(expected_answer.strip(),
                                transformations=standard_transformations + (implicit_multiplication,))
        except:
            return 0.5, ["expected_parse_failed"], None

    result = final_step.parsed_expr

    # Extract value if equation
    if isinstance(result, Eq):
        try:
            solutions = solve(result)
            if solutions:
                result = list(solutions.values())[0] if isinstance(solutions, dict) else solutions[0]
        except:
            pass

    # Compare
    try:
        diff = simplify(result - expected)
        if diff == 0:
            return 0.0, [], result
    except:
        pass

    # Numeric comparison
    try:
        r_float = float(sympy.N(result))
        e_float = float(sympy.N(expected))
        if abs(r_float - e_float) < 1e-6:
            return 0.0, [], result
    except:
        pass

    errors.append(f"answer_mismatch: got {result}, expected {expected}")
    return 1.0, errors, result


def verify_scaffold(scaffold: Scaffold, expected_answer: str) -> Dict:
    """
    Run full factor graph verification on scaffold.
    Returns energy breakdown and error localization.
    """
    total_energy = 0.0
    all_errors = []
    step_energies = []

    # Parse all steps first
    for step in scaffold.steps:
        step.parsed_expr = parse_math_from_text(step.generated_text)

    # Compute per-step energies
    for step in scaffold.steps:
        step_energy = 0.0
        step_errors = []

        # Type consistency
        e, errs = compute_type_energy(step)
        step_energy += e
        step_errors.extend(errs)

        # Dependency flow
        e, errs = compute_dependency_energy(step, scaffold.steps)
        step_energy += e
        step_errors.extend(errs)

        # Execution validity
        e, errs = compute_execution_energy(step)
        step_energy += e
        step_errors.extend(errs)

        step.energy = step_energy
        step.errors = step_errors
        total_energy += step_energy

        step_energies.append({
            "idx": step.idx,
            "type": step.step_type.value,
            "energy": step_energy,
            "errors": step_errors,
            "parsed": str(step.parsed_expr)[:50] if step.parsed_expr is not None else None,
        })

    # Verification energy (answer check)
    verify_energy, verify_errors, final_result = compute_verification_energy(scaffold, expected_answer)
    total_energy += verify_energy * 2  # Weight answer verification higher

    # Find highest energy step (error localization)
    max_energy_step = max(scaffold.steps, key=lambda s: s.energy)

    return {
        "total_energy": total_energy,
        "step_energies": step_energies,
        "verification_energy": verify_energy,
        "verification_errors": verify_errors,
        "final_result": str(final_result) if final_result else None,
        "expected": expected_answer,
        "correct": verify_energy == 0,
        "error_step_idx": max_energy_step.idx if max_energy_step.energy > 0 else None,
        "error_step_errors": max_energy_step.errors if max_energy_step.energy > 0 else [],
    }


def generate_correction_hint(scaffold: Scaffold, error_step_idx: int) -> str:
    """
    Generate a correction hint for the LLM to regenerate a step.
    This is what gets sent back to the LLM for targeted regeneration.
    """
    step = scaffold.steps[error_step_idx]

    hint_parts = [f"Regenerate Step {error_step_idx + 1} ({step.step_type.value})."]

    # Context from dependencies
    if step.depends_on:
        for dep_idx in step.depends_on:
            dep = scaffold.steps[dep_idx]
            if dep.parsed_expr is not None:
                hint_parts.append(f"Previous step established: {dep.parsed_expr}")

    # Error-specific hints
    if "no_parse" in step.errors:
        hint_parts.append("Your previous attempt didn't contain a valid mathematical expression.")
    if "expected_equation_got_expression" in step.errors:
        hint_parts.append("This step should produce an equation (with = sign).")
    if "expected_number_got_expression" in step.errors:
        hint_parts.append("This step should produce a numeric result.")
    if "no_symbol_flow_from_dependency" in step.errors:
        hint_parts.append("This step should use variables from the previous step.")

    return " ".join(hint_parts)


# =============================================================================
# EVALUATION
# =============================================================================

def load_problems(n=50):
    """Load MATH problems."""
    print(f"Loading {n} problems...")

    paginator = s3.get_paginator("list_objects_v2")
    keys = []

    for page in paginator.paginate(Bucket=BUCKET, Prefix="math500_72b_final/problem_"):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
            if len(keys) >= n:
                break
        if len(keys) >= n:
            break

    problems = []
    for key in keys[:n]:
        try:
            resp = s3.get_object(Bucket=BUCKET, Key=key)
            p = json.loads(resp["Body"].read().decode("utf-8"))
            if p.get("generated_cot") and p.get("gold_answer"):
                problems.append(p)
        except:
            continue

    print(f"  Loaded {len(problems)} problems")
    return problems


def main():
    print("=" * 60)
    print("SCAFFOLD-GUIDED VERIFICATION TEST")
    print("=" * 60)
    print("\nArchitecture: LLM generates within scaffold, factor graph VERIFIES")
    print("Using teacher CoT as LLM generation (to test verification machinery)")

    problems = load_problems(50)

    results = {
        "total": len(problems),
        "scaffolds_created": 0,
        "verification_ran": 0,
        "low_energy": 0,  # Energy < 1.0 (mostly consistent)
        "correct": 0,
        "errors_localized": 0,
        "step_type_distribution": {},
        "error_type_distribution": {},
        "correct_examples": [],
        "error_examples": [],
    }

    for i, problem in enumerate(problems):
        cot = problem.get("generated_cot", "")
        answer = problem.get("gold_answer", "")

        # Extract scaffold from teacher's CoT
        scaffold = extract_scaffold_from_cot(cot)
        if not scaffold.steps:
            continue

        results["scaffolds_created"] += 1

        # Track step types
        for step in scaffold.steps:
            t = step.step_type.value
            results["step_type_distribution"][t] = results["step_type_distribution"].get(t, 0) + 1

        # Run verification
        verification = verify_scaffold(scaffold, answer)
        results["verification_ran"] += 1

        if verification["total_energy"] < 1.0:
            results["low_energy"] += 1

        if verification["correct"]:
            results["correct"] += 1
            if len(results["correct_examples"]) < 5:
                results["correct_examples"].append({
                    "idx": problem.get("idx"),
                    "energy": verification["total_energy"],
                    "result": verification["final_result"],
                    "expected": answer,
                    "n_steps": len(scaffold.steps),
                })
        else:
            if verification["error_step_idx"] is not None:
                results["errors_localized"] += 1

                # Track error types
                for err in verification["error_step_errors"]:
                    err_type = err.split(":")[0]
                    results["error_type_distribution"][err_type] = \
                        results["error_type_distribution"].get(err_type, 0) + 1

            if len(results["error_examples"]) < 5:
                hint = generate_correction_hint(scaffold, verification["error_step_idx"]) \
                    if verification["error_step_idx"] is not None else "N/A"
                results["error_examples"].append({
                    "idx": problem.get("idx"),
                    "energy": verification["total_energy"],
                    "error_step": verification["error_step_idx"],
                    "errors": verification["error_step_errors"],
                    "hint": hint,
                    "result": verification["final_result"],
                    "expected": answer,
                })

    # Report
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    pct_correct = results["correct"] / results["total"] * 100 if results["total"] > 0 else 0
    pct_low_energy = results["low_energy"] / results["total"] * 100 if results["total"] > 0 else 0
    pct_localized = results["errors_localized"] / (results["total"] - results["correct"]) * 100 \
        if (results["total"] - results["correct"]) > 0 else 0

    print(f"\nProblems: {results['total']}")
    print(f"Scaffolds created: {results['scaffolds_created']}")
    print(f"Verification ran: {results['verification_ran']}")
    print(f"Low energy (<1.0): {results['low_energy']} ({pct_low_energy:.1f}%)")
    print(f"Correct answers: {results['correct']} ({pct_correct:.1f}%)")
    print(f"Errors localized: {results['errors_localized']} ({pct_localized:.1f}% of incorrect)")

    print(f"\nStep type distribution:")
    for t, count in sorted(results["step_type_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")

    print(f"\nError type distribution:")
    for t, count in sorted(results["error_type_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")

    # Examples
    if results["correct_examples"]:
        print(f"\nCorrect examples:")
        for ex in results["correct_examples"][:3]:
            print(f"\n  Problem {ex['idx']}: energy={ex['energy']:.2f}")
            print(f"    Result: {ex['result']}")
            print(f"    Expected: {ex['expected']}")

    if results["error_examples"]:
        print(f"\nError examples with localization:")
        for ex in results["error_examples"][:3]:
            print(f"\n  Problem {ex['idx']}: energy={ex['energy']:.2f}")
            print(f"    Error step: {ex['error_step']}")
            print(f"    Errors: {ex['errors']}")
            print(f"    Hint: {ex['hint'][:100]}...")
            print(f"    Result: {ex['result']}")
            print(f"    Expected: {ex['expected']}")

    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    print(f"\nVerification machinery:")
    if pct_correct > 5:
        print(f"  ✓ Detects {pct_correct:.1f}% correct answers")
    else:
        print(f"  ~ Only {pct_correct:.1f}% verified correct (parser needs work)")

    if pct_localized > 50:
        print(f"  ✓ Localizes errors in {pct_localized:.1f}% of failures")
    else:
        print(f"  ~ Error localization at {pct_localized:.1f}% (needs improvement)")

    print(f"\nNext steps:")
    print("  1. If verification works: plug in small LLM for constrained generation")
    print("  2. If localization works: test correction-regeneration loop")
    print("  3. Energy landscape shapes correction budget (C1-B complexity)")

    # Save
    print("\nSaving results...")
    s3.put_object(
        Bucket=BUCKET,
        Key="factor_graph_eval/scaffold_verification_results.json",
        Body=json.dumps({
            "pct_correct": pct_correct,
            "pct_low_energy": pct_low_energy,
            "pct_localized": pct_localized,
            "correct_examples": results["correct_examples"],
            "error_examples": results["error_examples"],
            "step_types": results["step_type_distribution"],
            "error_types": results["error_type_distribution"],
        }, indent=2, default=str).encode("utf-8")
    )


if __name__ == "__main__":
    main()
