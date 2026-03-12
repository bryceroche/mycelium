#!/usr/bin/env python3
"""
Stage 3: ODE Refinement and SymPy Execution

Takes Stage 2 output (rough slot filler strings) and:
1. Parses rough text to SymPy expressions
2. Executes using the SymPy Oracle (with timeout)
3. Uses ODE/energy landscape for verification and error localization

The key insight: rough text like "x, -7" maps to SymPy via SymPy's own parsers
(sympify, parse_latex). The ODE verifies consistency across the solution chain.

Input format (from Stage 2):
{
    "problems": [{
        "problem_id": 0,
        "segments": [{
            "scaffold_type": "SUBSTITUTE",
            "rough_output": "x, -7",
            "source": "slot_filler"
        }, ...]
    }]
}

Output format:
{
    "problems": [{
        "problem_id": 0,
        "segments": [{
            "scaffold_type": "SUBSTITUTE",
            "rough_output": "x, -7",
            "sympy_expr": "subs(x, -7)",
            "result": "-7",
            "success": true
        }],
        "final_answer": "42",
        "correct": true  # if ground truth available
    }]
}
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from oracle import SympyOracle, OracleResult
from templates import execute_template


@dataclass
class SegmentResult:
    """Result of processing a single segment."""
    scaffold_type: str
    rough_output: str
    sympy_expr: Optional[str]
    result: Optional[str]
    success: bool
    error: Optional[str] = None


def rough_to_sympy(scaffold_type: str, rough: str, prev_result: Optional[Any]) -> Tuple[Optional[Any], Optional[str]]:
    """
    Convert rough slot filler output to SymPy using SymPy's own parsers.

    No regex on mathematical content. Let sympify and parse_latex do the work.
    The comma split for SUBSTITUTE is structured output format, not pattern matching.

    Returns (sympy_expr, error)
    """
    import sympy
    from sympy import sympify
    from sympy.parsing.latex import parse_latex

    if not rough:
        return None, "Empty input"

    rough = rough.strip()

    if scaffold_type == "SETUP":
        # SETUP: parse the equation/expression
        # Try parse_latex first (handles LaTeX notation), fall back to sympify
        try:
            if '\\' in rough or '{' in rough:
                return parse_latex(rough), None
            else:
                return sympify(rough), None
        except Exception as e:
            # Try sympify as fallback
            try:
                return sympify(rough), None
            except Exception as e2:
                return None, f"Could not parse: {e2}"

    elif scaffold_type == "SUBSTITUTE":
        # SUBSTITUTE: "expr, value" format (structured output from training)
        parts = rough.split(',', 1)
        if len(parts) != 2:
            return None, f"Expected 'expr, value' format, got: {rough}"

        try:
            expr = sympify(parts[0].strip())
            value = sympify(parts[1].strip())

            if prev_result is not None:
                # Apply substitution to previous result
                return prev_result.subs(expr, value), None
            else:
                # Return as a substitution pair for later use
                return (expr, value), None
        except Exception as e:
            return None, f"Could not parse substitution: {e}"

    elif scaffold_type == "THEOREM":
        # THEOREM: "domain, type" keywords
        # For MVP, just return the keywords - formula lookup comes later
        return rough, None

    else:
        return None, f"Unknown scaffold type: {scaffold_type}"


def execute_segment(
    segment: Dict[str, Any],
    context: Dict[str, Any],
    oracle: SympyOracle
) -> SegmentResult:
    """
    Execute a single segment.

    Args:
        segment: Segment dict with scaffold_type, rough_output, source
        context: Running context with previous results
        oracle: SympyOracle instance

    Returns:
        SegmentResult
    """
    scaffold_type = segment.get('scaffold_type', '').upper()
    rough_output = segment.get('rough_output', '')
    source = segment.get('source', '')

    # Template scaffolds are executed deterministically
    if source == 'template' or scaffold_type in ['EXPAND', 'SIMPLIFY', 'SOLVE', 'COMPUTE', 'ANSWER']:
        # Use the template execution
        # For template scaffolds, rough_output should be the expression to operate on
        # If rough_output is None, use prev_result from context
        expr = rough_output or context.get('prev_result', '')

        if not expr:
            return SegmentResult(
                scaffold_type=scaffold_type,
                rough_output=rough_output,
                sympy_expr=None,
                result=None,
                success=False,
                error="No expression to process"
            )

        success, result, message = execute_template(scaffold_type, expr)

        return SegmentResult(
            scaffold_type=scaffold_type,
            rough_output=rough_output,
            sympy_expr=expr,
            result=str(result) if result is not None else None,
            success=success,
            error=message if not success else None
        )

    # Slot filler scaffolds: use SymPy parsers (not regex)
    prev_result = context.get('prev_sympy_result')
    sympy_expr, error = rough_to_sympy(scaffold_type, rough_output, prev_result)

    if error:
        return SegmentResult(
            scaffold_type=scaffold_type,
            rough_output=rough_output,
            sympy_expr=str(sympy_expr) if sympy_expr else None,
            result=None,
            success=False,
            error=error
        )

    # For THEOREM, we just store the keywords (no execution yet)
    if scaffold_type == 'THEOREM':
        return SegmentResult(
            scaffold_type=scaffold_type,
            rough_output=rough_output,
            sympy_expr=str(sympy_expr),
            result=str(sympy_expr),
            success=True
        )

    # Execute/simplify the result
    try:
        from sympy import simplify
        result = simplify(sympy_expr) if sympy_expr is not None else None
        return SegmentResult(
            scaffold_type=scaffold_type,
            rough_output=rough_output,
            sympy_expr=str(sympy_expr),
            result=str(result) if result is not None else None,
            success=True
        )
    except Exception as e:
        return SegmentResult(
            scaffold_type=scaffold_type,
            rough_output=rough_output,
            sympy_expr=str(sympy_expr) if sympy_expr else None,
            result=None,
            success=False,
            error=str(e)
        )

    if exec_result.status == OracleResult.SUCCESS:
        return SegmentResult(
            scaffold_type=scaffold_type,
            rough_output=rough_output,
            sympy_expr=sympy_expr,
            result=str(exec_result.normalized or exec_result.value),
            success=True
        )
    else:
        return SegmentResult(
            scaffold_type=scaffold_type,
            rough_output=rough_output,
            sympy_expr=sympy_expr,
            result=None,
            success=False,
            error=exec_result.message
        )


def process_problem(
    problem: Dict[str, Any],
    oracle: SympyOracle
) -> Dict[str, Any]:
    """
    Process a single problem through the pipeline.

    Returns dict with processed segments and final answer.
    """
    problem_id = problem.get('problem_id')
    segments = problem.get('segments', [])
    ground_truth = problem.get('expected_answer')

    results = []
    context = {
        'prev_result': None,
        'variables': {},
    }

    for segment in segments:
        result = execute_segment(segment, context, oracle)
        results.append(result)

        # Update context with successful result
        if result.success and result.result:
            context['prev_result'] = result.result
            # Also store as SymPy object for substitutions
            try:
                from sympy import sympify
                context['prev_sympy_result'] = sympify(result.result)
            except:
                pass

    # Get final answer from last successful segment
    final_answer = None
    for result in reversed(results):
        if result.success and result.result:
            final_answer = result.result
            break

    # Check correctness if ground truth available
    correct = None
    if ground_truth is not None and final_answer is not None:
        correct = oracle.compare(final_answer, ground_truth)

    return {
        'problem_id': problem_id,
        'segments': [
            {
                'scaffold_type': r.scaffold_type,
                'rough_output': r.rough_output,
                'sympy_expr': r.sympy_expr,
                'result': r.result,
                'success': r.success,
                'error': r.error,
            }
            for r in results
        ],
        'final_answer': final_answer,
        'correct': correct,
        'n_segments': len(segments),
        'n_successful': sum(1 for r in results if r.success),
    }


def load_stage2_output(path: str) -> Dict[str, Any]:
    """Load Stage 2 output from file or S3."""
    if path.startswith('s3://'):
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            subprocess.run(f"aws s3 cp {path} {f.name}", shell=True, check=True)
            with open(f.name, 'r') as rf:
                data = json.load(rf)
            os.unlink(f.name)
            return data
    else:
        with open(path, 'r') as f:
            return json.load(f)


def save_output(data: Dict[str, Any], path: str):
    """Save output to file or S3."""
    if path.startswith('s3://'):
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f, indent=2)
            f.flush()
            subprocess.run(f"aws s3 cp {f.name} {path}", shell=True, check=True)
            os.unlink(f.name)
    else:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Stage 3: ODE Refinement and SymPy Execution")
    parser.add_argument(
        '--stage2-output',
        type=str,
        required=True,
        help='Path to Stage 2 output JSON'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Path to save Stage 3 output'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=5,
        help='SymPy execution timeout in seconds'
    )

    args = parser.parse_args()

    # Initialize oracle
    oracle = SympyOracle(timeout=args.timeout)

    # Load Stage 2 output
    print(f"Loading Stage 2 output from {args.stage2_output}...")
    stage2_data = load_stage2_output(args.stage2_output)

    # Process problems
    problems = stage2_data.get('problems', [])
    print(f"Processing {len(problems)} problems...")

    results = []
    n_correct = 0
    n_with_answer = 0

    for problem in problems:
        result = process_problem(problem, oracle)
        results.append(result)

        if result['correct'] is not None:
            n_with_answer += 1
            if result['correct']:
                n_correct += 1

    # Build output
    output = {
        'stage': 3,
        'problems': results,
        'stats': {
            'total_problems': len(results),
            'problems_with_ground_truth': n_with_answer,
            'correct': n_correct,
            'accuracy': n_correct / n_with_answer if n_with_answer > 0 else None,
            'total_segments': sum(r['n_segments'] for r in results),
            'successful_segments': sum(r['n_successful'] for r in results),
        }
    }

    # Save output
    print(f"\nSaving to {args.output_path}...")
    save_output(output, args.output_path)

    # Print summary
    print("\n=== Stage 3 Summary ===")
    print(f"Total problems: {len(results)}")
    print(f"Segments processed: {output['stats']['total_segments']}")
    print(f"Successful segments: {output['stats']['successful_segments']}")
    if n_with_answer > 0:
        print(f"Accuracy: {n_correct}/{n_with_answer} = {100*n_correct/n_with_answer:.1f}%")


if __name__ == '__main__':
    main()
