#!/usr/bin/env python3
"""
MATH CoT Feature Extraction for Information Bottleneck

Parses MATH CoT solutions into structural features for template discovery.
Each step in the CoT is characterized by:
  - operator_symbol: +, -, ×, ÷, √, ^, sin, cos, etc.
  - operand_count: number of operands
  - operand_provenance: where operands come from (PROB, PREV, CONST)
  - result_type: INT, FRAC, DECIMAL, EXPR, SPECIAL
  - chain_position: normalized position in reasoning chain (0-1)

Output: JSON with extracted features for IB clustering.
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class CoTStep:
    """Represents one computational step in a CoT."""
    step_idx: int
    text: str
    operator: str
    operand_count: int
    operand_provenance: List[str]  # PROB, PREV, CONST for each operand
    result_type: str  # INT, FRAC, DECIMAL, EXPR, SPECIAL
    chain_position: float  # 0.0 to 1.0
    raw_expression: str
    category: str  # problem category


# Operator patterns (LaTeX and plain text)
OPERATOR_PATTERNS = {
    # Basic arithmetic
    'ADD': [r'\+', r'plus', r'added', r'sum'],
    'SUB': [r'-(?!\d)', r'minus', r'subtract', r'difference'],
    'MUL': [r'\\times', r'\\cdot', r'\*', r'×', r'multiply', r'product'],
    'DIV': [r'\\div', r'\\frac\{', r'/', r'÷', r'divide', r'quotient'],

    # Powers and roots
    'POW': [r'\^', r'\\text\{power\}', r'squared', r'cubed', r'exponent'],
    'SQRT': [r'\\sqrt', r'square root', r'√'],
    'ROOT': [r'\\sqrt\[\d+\]', r'cube root', r'nth root'],

    # Trigonometry
    'SIN': [r'\\sin', r'sine'],
    'COS': [r'\\cos', r'cosine'],
    'TAN': [r'\\tan', r'tangent'],
    'ARCSIN': [r'\\arcsin', r'\\sin\^{-1}', r'inverse sine'],
    'ARCCOS': [r'\\arccos', r'\\cos\^{-1}', r'inverse cosine'],
    'ARCTAN': [r'\\arctan', r'\\tan\^{-1}', r'inverse tangent'],

    # Logarithms
    'LOG': [r'\\log', r'logarithm'],
    'LN': [r'\\ln', r'natural log'],

    # Combinatorics
    'FACTORIAL': [r'!', r'factorial'],
    'CHOOSE': [r'\\binom', r'\\choose', r'combination', r'C\('],
    'PERMUTE': [r'P\(', r'permutation'],

    # Algebra
    'SOLVE': [r'solve', r'find.*=', r'x\s*='],
    'SUBSTITUTE': [r'substitut', r'plug in', r'replace'],
    'SIMPLIFY': [r'simplif', r'reduce', r'cancel'],
    'FACTOR': [r'factor', r'\\left\(.*\\right\)\^'],
    'EXPAND': [r'expand', r'distribute', r'FOIL'],

    # Geometry
    'AREA': [r'area', r'A\s*='],
    'PERIMETER': [r'perimeter', r'circumference'],
    'DISTANCE': [r'distance', r'd\s*=', r'\\sqrt\{.*\^2.*\+.*\^2\}'],
    'ANGLE': [r'angle', r'\\theta', r'degrees', r'radians'],

    # Comparisons
    'COMPARE': [r'<', r'>', r'\\le', r'\\ge', r'\\leq', r'\\geq'],
    'EQUALS': [r'=(?!=)', r'equal'],

    # Special
    'MOD': [r'\\mod', r'remainder', r'modulo'],
    'ABS': [r'\\left\|', r'\\abs', r'absolute'],
    'FLOOR': [r'\\lfloor', r'floor'],
    'CEIL': [r'\\lceil', r'ceiling'],
    'GCD': [r'gcd', r'greatest common'],
    'LCM': [r'lcm', r'least common'],
}

# Result type patterns
RESULT_PATTERNS = {
    'INT': r'^-?\d+$',
    'FRAC': r'\\frac\{-?\d+\}\{-?\d+\}|^\d+/\d+$',
    'DECIMAL': r'^-?\d+\.\d+$',
    'PI': r'\\pi',
    'SQRT_EXPR': r'\\sqrt\{',
    'TRIG_EXPR': r'\\(sin|cos|tan)',
}


def extract_equations(text: str) -> List[str]:
    """Extract equations/expressions from CoT text."""
    equations = []

    # LaTeX display math: \[ ... \] or $$ ... $$
    display_math = re.findall(r'\\\[(.*?)\\\]|\$\$(.*?)\$\$', text, re.DOTALL)
    for match in display_math:
        eq = match[0] or match[1]
        if eq.strip():
            equations.append(eq.strip())

    # LaTeX inline math: $ ... $ (but not $$)
    inline_math = re.findall(r'(?<!\$)\$([^\$]+)\$(?!\$)', text)
    equations.extend([m.strip() for m in inline_math if m.strip()])

    # Equations with = sign outside LaTeX
    plain_eq = re.findall(r'(\w+)\s*=\s*([^\n,\.]+)', text)
    for var, val in plain_eq:
        equations.append(f"{var} = {val}")

    return equations


def classify_operator(text: str) -> str:
    """Identify the primary operator in an expression."""
    text_lower = text.lower()

    # Check patterns in priority order
    priority_ops = ['SQRT', 'POW', 'SIN', 'COS', 'TAN', 'ARCSIN', 'ARCCOS', 'ARCTAN',
                    'LOG', 'LN', 'FACTORIAL', 'CHOOSE', 'PERMUTE', 'GCD', 'LCM',
                    'DIV', 'MUL', 'SUB', 'ADD']

    for op in priority_ops:
        patterns = OPERATOR_PATTERNS.get(op, [])
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return op

    # Check for algebraic operations
    for op in ['SOLVE', 'SUBSTITUTE', 'SIMPLIFY', 'FACTOR', 'EXPAND']:
        patterns = OPERATOR_PATTERNS.get(op, [])
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return op

    # Check for geometry
    for op in ['AREA', 'PERIMETER', 'DISTANCE', 'ANGLE']:
        patterns = OPERATOR_PATTERNS.get(op, [])
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return op

    return 'UNKNOWN'


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text."""
    numbers = []
    # Match integers, decimals, fractions
    for match in re.finditer(r'-?\d+(?:\.\d+)?', text):
        try:
            numbers.append(float(match.group()))
        except ValueError:
            pass
    return numbers


def classify_result_type(expression: str) -> str:
    """Classify the type of result."""
    # Clean expression
    expr = expression.strip()

    if re.match(RESULT_PATTERNS['INT'], expr):
        return 'INT'
    if re.search(RESULT_PATTERNS['FRAC'], expr):
        return 'FRAC'
    if re.match(RESULT_PATTERNS['DECIMAL'], expr):
        return 'DECIMAL'
    if re.search(RESULT_PATTERNS['PI'], expr):
        return 'PI'
    if re.search(RESULT_PATTERNS['SQRT_EXPR'], expr):
        return 'SQRT_EXPR'
    if re.search(RESULT_PATTERNS['TRIG_EXPR'], expr):
        return 'TRIG_EXPR'

    return 'EXPR'


def determine_provenance(expr: str, problem_numbers: List[float], prev_results: List[float]) -> List[str]:
    """Determine where operands come from."""
    expr_numbers = extract_numbers(expr)
    provenance = []

    for num in expr_numbers[:3]:  # Limit to first 3 operands
        if num in problem_numbers:
            provenance.append('PROB')
        elif num in prev_results:
            provenance.append('PREV')
        elif num in [0, 1, 2, 3, 4, 5, 10, 100, 180, 360]:  # Common constants
            provenance.append('CONST')
        else:
            provenance.append('DERIVED')

    return provenance if provenance else ['NONE']


def parse_cot_steps(cot: str, problem_text: str, category: str) -> List[CoTStep]:
    """Parse a CoT into structured steps."""
    steps = []

    # Extract numbers from problem for provenance tracking
    problem_numbers = extract_numbers(problem_text)
    prev_results = []

    # Split CoT into paragraphs/sections
    paragraphs = re.split(r'\n\s*\n|\.\s+(?=[A-Z])', cot)

    # Extract all equations
    all_equations = extract_equations(cot)

    # If we have equations, use those as steps
    if all_equations:
        for i, eq in enumerate(all_equations):
            operator = classify_operator(eq)
            operands = extract_numbers(eq)
            provenance = determine_provenance(eq, problem_numbers, prev_results)
            result_type = classify_result_type(eq)

            # Track results for provenance
            results = extract_numbers(eq.split('=')[-1] if '=' in eq else eq)
            prev_results.extend(results)

            step = CoTStep(
                step_idx=i,
                text=eq[:100],
                operator=operator,
                operand_count=min(len(operands), 3),
                operand_provenance=provenance,
                result_type=result_type,
                chain_position=i / max(len(all_equations), 1),
                raw_expression=eq[:200],
                category=category,
            )
            steps.append(step)
    else:
        # Fall back to paragraph-based parsing
        for i, para in enumerate(paragraphs):
            if len(para.strip()) < 10:
                continue

            operator = classify_operator(para)
            if operator == 'UNKNOWN':
                continue

            operands = extract_numbers(para)
            provenance = determine_provenance(para, problem_numbers, prev_results)
            result_type = classify_result_type(para)

            prev_results.extend(operands[-2:] if operands else [])

            step = CoTStep(
                step_idx=len(steps),
                text=para[:100],
                operator=operator,
                operand_count=min(len(operands), 3),
                operand_provenance=provenance,
                result_type=result_type,
                chain_position=len(steps) / max(len(paragraphs), 1),
                raw_expression=para[:200],
                category=category,
            )
            steps.append(step)

    return steps


def load_math_cots(data_dir: str) -> List[Dict]:
    """Load MATH problems with CoT solutions."""
    problems = []

    # Load 7B solutions
    for f in sorted(Path(data_dir).glob("7b/*.json")):
        with open(f) as fp:
            d = json.load(fp)
            if d.get('generated_cot'):
                problems.append(d)

    # Load 72B solutions (for failures that 72B solved)
    for f in sorted(Path(data_dir).glob("72b/*.json")):
        with open(f) as fp:
            d = json.load(fp)
            if d.get('generated_cot'):
                # Check if we already have this problem
                existing = {p['idx'] for p in problems}
                if d['idx'] not in existing:
                    problems.append(d)

    return problems


def main():
    print("=" * 70)
    print("MATH CoT FEATURE EXTRACTION")
    print("=" * 70)

    # Load MATH CoTs
    data_dir = "/tmp/math_e2e_test"
    print(f"\nLoading MATH CoTs from {data_dir}...")
    problems = load_math_cots(data_dir)
    print(f"Loaded {len(problems)} problems with CoT solutions")

    # Parse all CoTs
    print("\nParsing CoT steps...")
    all_steps = []
    steps_by_problem = {}

    for prob in problems:
        cot = prob.get('generated_cot', '')
        question = prob.get('question', '')
        category = prob.get('category', 'Unknown')
        idx = prob.get('idx')

        steps = parse_cot_steps(cot, question, category)
        all_steps.extend(steps)
        steps_by_problem[idx] = steps

    print(f"Extracted {len(all_steps)} total steps from {len(problems)} problems")

    # Statistics
    print("\n" + "-" * 50)
    print("OPERATOR DISTRIBUTION")
    print("-" * 50)
    op_counts = Counter(s.operator for s in all_steps)
    for op, count in op_counts.most_common(30):
        pct = 100 * count / len(all_steps)
        print(f"  {op:15} {count:5} ({pct:5.1f}%)")

    print("\n" + "-" * 50)
    print("RESULT TYPE DISTRIBUTION")
    print("-" * 50)
    result_counts = Counter(s.result_type for s in all_steps)
    for rt, count in result_counts.most_common():
        pct = 100 * count / len(all_steps)
        print(f"  {rt:15} {count:5} ({pct:5.1f}%)")

    print("\n" + "-" * 50)
    print("OPERAND COUNT DISTRIBUTION")
    print("-" * 50)
    opcount = Counter(s.operand_count for s in all_steps)
    for oc, count in sorted(opcount.items()):
        pct = 100 * count / len(all_steps)
        print(f"  {oc} operands: {count:5} ({pct:5.1f}%)")

    print("\n" + "-" * 50)
    print("CATEGORY DISTRIBUTION")
    print("-" * 50)
    cat_counts = Counter(s.category for s in all_steps)
    for cat, count in cat_counts.most_common():
        pct = 100 * count / len(all_steps)
        print(f"  {cat:25} {count:5} ({pct:5.1f}%)")

    # Build feature vectors for IB
    print("\n" + "-" * 50)
    print("BUILDING FEATURE VECTORS FOR IB")
    print("-" * 50)

    feature_data = []
    for step in all_steps:
        feature = {
            'operator': step.operator,
            'operand_count': step.operand_count,
            'result_type': step.result_type,
            'chain_position_bin': int(step.chain_position * 5),  # 0-4
            'category': step.category,
            'provenance_pattern': '_'.join(step.operand_provenance[:2]),
            'text': step.text,
            'raw_expression': step.raw_expression,
        }
        feature_data.append(feature)

    # Save feature data
    output_path = Path("/tmp/math_cot_features.json")
    with open(output_path, 'w') as f:
        json.dump(feature_data, f, indent=2)
    print(f"\nSaved {len(feature_data)} feature vectors to {output_path}")

    # Also save as steps_by_problem for reference
    steps_dict = {
        idx: [asdict(s) for s in steps]
        for idx, steps in steps_by_problem.items()
    }
    with open("/tmp/math_cot_steps_by_problem.json", 'w') as f:
        json.dump(steps_dict, f, indent=2, default=str)
    print(f"Saved problem-indexed steps to /tmp/math_cot_steps_by_problem.json")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Problems processed: {len(problems)}")
    print(f"Total steps extracted: {len(all_steps)}")
    print(f"Unique operators: {len(op_counts)}")
    print(f"Avg steps per problem: {len(all_steps)/len(problems):.1f}")

    # Preview unique feature combinations (potential templates)
    print("\n" + "-" * 50)
    print("UNIQUE FEATURE COMBINATIONS (preview of templates)")
    print("-" * 50)
    combos = Counter((f['operator'], f['result_type'], f['operand_count']) for f in feature_data)
    for (op, rt, oc), count in combos.most_common(20):
        print(f"  {op:12} + {rt:10} + {oc}ops = {count:4}")

    print(f"\nTotal unique (op, result_type, operand_count) combinations: {len(combos)}")
    print("This gives an estimate of the template count for IB.")


if __name__ == "__main__":
    main()
