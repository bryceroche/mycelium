"""
Lambda Function: C3 Training Data Extractor

Extracts mathematical expressions from MATH CoT for C3 training.
Each training pair: [TEMPLATE: OP] <problem_text> -> <expression>

Input: S3 key to IAF chunk
Output: C3 training pairs (one per operation per problem)
"""

import json
import re
import boto3
from typing import List, Dict, Tuple, Optional

s3 = boto3.client('s3')
BUCKET = 'mycelium-data'
OUTPUT_PREFIX = 'c3_training/chunks/'

# Operation labels (same as C2)
LABELS = [
    "ADD", "MUL", "DIV", "SQRT", "SQUARE", "CUBE", "HIGH_POW", "FRAC_POW",
    "FACTORIAL", "LOG", "TRIG", "MOD", "EQUATION", "OTHER"
]

# Patterns for operation classification
OP_PATTERNS = {
    'FACTORIAL': [r'\d+!', r'\\binom', r'\\frac\{(\d+)!\}'],
    'LOG': [r'\\log', r'\\ln'],
    'TRIG': [r'\\sin', r'\\cos', r'\\tan', r'\\cot', r'\\sec', r'\\csc'],
    'MOD': [r'\\mod', r'\\pmod', r'\\equiv'],
    'SQRT': [r'\\sqrt'],
    'CUBE': [r'\^3(?!\d)', r'\^{3}'],
    'FRAC_POW': [r'\^\{?\\frac', r'\^\{?\d+/\d+\}?'],
    'HIGH_POW': [r'\^[4-9]', r'\^{[4-9]}', r'\^\d{2,}'],
    'SQUARE': [r'\^2(?!\d)', r'\^{2}'],
    'DIV': [r'\\frac\{', r'\\div', r'/'],
    'MUL': [r'\\times', r'\\cdot', r'\*'],
    'ADD': [r'\+', r'-(?!\d+\})'],  # Avoid matching negative exponents
}


def latex_to_sympy(latex: str) -> Optional[str]:
    """Convert LaTeX expression to sympy-parseable format."""
    if not latex:
        return None

    expr = latex.strip()

    # Remove display math delimiters
    expr = re.sub(r'^\\\[|\\\]$', '', expr)
    expr = re.sub(r'^\$|\$$', '', expr)

    # Convert LaTeX to Python/sympy
    expr = expr.replace('\\times', '*')
    expr = expr.replace('\\cdot', '*')
    expr = expr.replace('\\div', '/')
    expr = expr.replace('\\pm', '+')  # Simplify ± to +

    # Handle fractions: \frac{a}{b} -> (a)/(b)
    expr = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', expr)

    # Handle nested fractions (one level)
    expr = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', expr)

    # Handle binomial: \binom{n}{k} -> binomial(n,k)
    expr = re.sub(r'\\binom\{(\d+)\}\{(\d+)\}', r'binomial(\1,\2)', expr)

    # Handle sqrt: \sqrt{x} -> sqrt(x)
    expr = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', expr)

    # Handle powers: x^{n} -> x**n, x^n -> x**n
    expr = re.sub(r'\^{([^{}]+)}', r'**(\1)', expr)
    expr = re.sub(r'\^(\d+)', r'**\1', expr)

    # Handle factorial: n! -> factorial(n)
    expr = re.sub(r'(\d+)!', r'factorial(\1)', expr)

    # Handle trig functions
    for func in ['sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'log', 'ln']:
        expr = expr.replace(f'\\{func}', func)

    # Remove remaining LaTeX commands
    expr = re.sub(r'\\[a-zA-Z]+', '', expr)

    # Clean up spaces and extra characters
    expr = re.sub(r'\s+', '', expr)
    expr = re.sub(r'\\', '', expr)

    # Remove text mode artifacts
    expr = re.sub(r'\([^()]*text[^()]*\)', '', expr)

    return expr if expr else None


def classify_expression(expr: str) -> str:
    """Classify expression by primary operation."""
    for op, patterns in OP_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, expr, re.IGNORECASE):
                return op
    return 'OTHER'


def extract_math_blocks(cot: str) -> List[Tuple[str, str]]:
    """
    Extract math expressions from CoT and classify them.
    Returns list of (expression, operation_type) tuples.
    """
    results = []

    # Find display math: \[ ... \]
    display_matches = re.findall(r'\\\[(.*?)\\\]', cot, re.DOTALL)

    # Find inline math: $ ... $
    inline_matches = re.findall(r'\$([^\$]+)\$', cot)

    all_latex = display_matches + inline_matches

    for latex in all_latex:
        # Skip very short or very long expressions
        if len(latex) < 3 or len(latex) > 200:
            continue

        # Must contain = sign (we want computations, not just symbols)
        if '=' not in latex:
            continue

        # Extract the right side of the equation (the result)
        parts = latex.split('=')
        if len(parts) >= 2:
            # Get the computation (could be on left or right of =)
            left = parts[0].strip()
            right = parts[-1].strip()

            # Prefer the side with operations
            expr = left if any(c in left for c in ['+', '-', '*', '/', '^', '\\']) else right

            # Try to convert to sympy format
            sympy_expr = latex_to_sympy(expr)
            if sympy_expr and len(sympy_expr) >= 3:
                op_type = classify_expression(latex)
                results.append((sympy_expr, op_type))

    return results


def lambda_handler(event, context):
    """
    Process a single IAF chunk and extract C3 training data.

    Event:
        chunk_key: S3 key to IAF chunk file
    """
    chunk_key = event.get('chunk_key')
    if not chunk_key:
        return {'error': 'chunk_key required'}

    # Load chunk
    try:
        response = s3.get_object(Bucket=BUCKET, Key=chunk_key)
        data = json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        return {'error': f'Failed to load {chunk_key}: {str(e)}'}

    # Process each problem
    training_pairs = []
    op_counts = {}

    for problem in data:
        problem_text = problem.get('problem_text', '')
        cot = problem.get('generated_cot', '')
        problem_idx = problem.get('problem_idx')

        if not problem_text or not cot:
            continue

        # Extract math expressions and classify
        expressions = extract_math_blocks(cot)

        for step_idx, (expr, op_type) in enumerate(expressions):
            # Create training pair
            pair = {
                'input': f'[TEMPLATE: {op_type}] {problem_text}',
                'output': expr,
                'template': op_type,
                'problem_idx': problem_idx,
                'step_idx': step_idx,
            }
            training_pairs.append(pair)

            # Count operations
            op_counts[op_type] = op_counts.get(op_type, 0) + 1

    # Extract chunk name
    import os
    chunk_name = os.path.basename(chunk_key).replace('.json', '')

    # Write results to S3
    output_key = f"{OUTPUT_PREFIX}{chunk_name}_c3.json"
    output = {
        'source_chunk': chunk_key,
        'n_problems': len(data),
        'n_pairs': len(training_pairs),
        'op_counts': op_counts,
        'pairs': training_pairs,
    }

    s3.put_object(
        Bucket=BUCKET,
        Key=output_key,
        Body=json.dumps(output),
        ContentType='application/json'
    )

    return {
        'success': True,
        'chunk_key': chunk_key,
        'n_problems': len(data),
        'n_pairs': len(training_pairs),
        'op_counts': op_counts,
        'output_key': output_key,
    }


# Local testing
if __name__ == '__main__':
    test_cot = r"""
    We need to calculate \(\binom{8}{2}\):
    \[
    \binom{8}{2} = \frac{8!}{2!6!} = \frac{8 \times 7}{2 \times 1} = 28
    \]

    Then multiply:
    \[
    28 \times 15 = 420
    \]
    """

    expressions = extract_math_blocks(test_cot)
    print("Extracted expressions:")
    for expr, op in expressions:
        print(f"  {op}: {expr}")
