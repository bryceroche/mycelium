"""
Lambda Function: SymPy Y Extractor

Extracts the PRIMARY SYMPY OPERATOR from each CoT step's math expression.
This is the Y target variable for true Information Bottleneck.

Y = root operator of the sympy AST:
  - 48/2 → Mul (sympy represents division as Mul with Rational)
  - 48-24 → Add (subtraction is Add with negative)
  - sqrt(144) → Pow (sqrt is Pow(x, 1/2))
  - sin(pi/6) → sin
  - solve(x**2-4) → solve

Input: S3 key to aggregated_steps.json chunk
Output: Steps with Y labels written to S3
"""

import json
import re
import boto3
import warnings
from typing import Optional, Tuple

# Suppress sympy deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Lazy import sympy (heavy)
sympy = None

def lazy_import_sympy():
    global sympy
    if sympy is None:
        import sympy as sp
        # Suppress sympy-specific warnings
        sp.init_printing(quiet=True)
        sympy = sp

s3 = boto3.client('s3')
BUCKET = 'mycelium-data'
OUTPUT_PREFIX = 'ib_y_labels/'


def extract_math_from_text(text: str) -> list[str]:
    """
    Extract mathematical expressions from CoT text.
    Returns list of candidate expressions to try parsing.
    Prioritizes COMPUTATION expressions (left side of =) over results.
    """
    expressions = []

    # Equations with = sign - prioritize LEFT side (the computation)
    # Look for patterns like "3 + 4 = 7" and extract "3 + 4"
    equations = re.findall(r'([^=\n]{3,}?)\s*=\s*([^=\n]+)', text)
    for left, right in equations:
        # Clean and check if left side has an operation
        left_clean = left.strip().rstrip('\\')
        if any(op in left_clean for op in ['+', '-', '*', '/', '^', '\\frac', '\\sqrt', '\\cdot', '\\times']):
            expressions.append(left_clean)
        # Also try right side if it has operations
        right_clean = right.strip()
        if any(op in right_clean for op in ['+', '-', '*', '/', '^', '\\frac', '\\sqrt']):
            expressions.append(right_clean)

    # Display math: \[...\] or $$...$$ - extract inner content (handle multiline)
    # First collapse multiline to single line for easier parsing
    text_collapsed = re.sub(r'\s+', ' ', text)
    display = re.findall(r'\\\[(.+?)\\\]', text_collapsed, re.DOTALL)
    for d in display:
        # If it contains =, split and prioritize left
        if '=' in d:
            parts = d.split('=')
            for p in parts[:-1]:  # Skip last part (usually just the answer)
                if any(op in p for op in ['+', '-', '*', '/', '^', '\\frac', '\\sqrt']):
                    expressions.append(p.strip())
        else:
            expressions.append(d)

    display2 = re.findall(r'\$\$(.+?)\$\$', text, re.DOTALL)
    for d in display2:
        if '=' in d:
            parts = d.split('=')
            for p in parts[:-1]:
                if any(op in p for op in ['+', '-', '*', '/', '^', '\\frac', '\\sqrt']):
                    expressions.append(p.strip())
        else:
            expressions.append(d)

    # Inline math: \(...\) or $...$
    inline = re.findall(r'\\\((.+?)\\\)', text)
    for i in inline:
        if '=' in i:
            parts = i.split('=')
            for p in parts[:-1]:
                if any(op in p for op in ['+', '-', '*', '/', '^', '\\frac', '\\sqrt']):
                    expressions.append(p.strip())
        else:
            expressions.append(i)

    inline2 = re.findall(r'(?<!\$)\$([^$]+?)\$(?!\$)', text)
    for i in inline2:
        if '=' in i:
            parts = i.split('=')
            for p in parts[:-1]:
                if any(op in p for op in ['+', '-', '*', '/', '^', '\\frac', '\\sqrt']):
                    expressions.append(p.strip())
        else:
            expressions.append(i)

    # Numeric expressions with operators: 48/2, 3+4, 2^3, etc
    numeric = re.findall(
        r'\b\d+(?:\.\d+)?\s*[+\-*/^]\s*\d+(?:\.\d+)?(?:\s*[+\-*/^]\s*\d+(?:\.\d+)?)*',
        text
    )
    expressions.extend(numeric)

    # Fraction patterns
    fracs = re.findall(r'\\frac\s*\{[^{}]+\}\s*\{[^{}]+\}', text)
    expressions.extend(fracs)

    # Square root patterns
    sqrts = re.findall(r'\\sqrt\s*(?:\[[^\]]+\])?\s*\{[^{}]+\}', text)
    expressions.extend(sqrts)

    return expressions


def clean_latex(expr: str) -> str:
    """Convert LaTeX to sympy-parseable string."""
    s = expr.strip()

    # Remove LaTeX formatting
    s = re.sub(r'\\left|\\right', '', s)
    s = re.sub(r'\\[,;:!]', ' ', s)  # spacing commands
    s = re.sub(r'\\quad|\\qquad', ' ', s)
    s = re.sub(r'\\text\{[^}]*\}', '', s)  # \text{...}
    s = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', s)

    # Convert LaTeX operators to sympy
    s = s.replace('\\cdot', '*')
    s = s.replace('\\times', '*')
    s = s.replace('\\div', '/')
    s = s.replace('\\pm', '+')  # just use +
    s = s.replace('\\mp', '-')

    # Fractions: \frac{a}{b} -> (a)/(b)
    while '\\frac' in s:
        match = re.search(r'\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', s)
        if match:
            s = s[:match.start()] + f'(({match.group(1)})/({match.group(2)}))' + s[match.end():]
        else:
            break

    # Square root: \sqrt{x} -> sqrt(x)
    s = re.sub(r'\\sqrt\s*\{([^{}]*)\}', r'sqrt(\1)', s)

    # nth root: \sqrt[n]{x} -> x**(1/n)
    s = re.sub(r'\\sqrt\s*\[([^\]]*)\]\s*\{([^{}]*)\}', r'((\2)**(1/(\1)))', s)

    # Powers: x^{2} -> x**2, x^2 -> x**2
    s = re.sub(r'\^{([^{}]*)}', r'**(\1)', s)
    s = re.sub(r'\^(\d+)', r'**\1', s)
    s = re.sub(r'\^([a-zA-Z])', r'**\1', s)

    # Trig functions
    for fn in ['sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'arcsin', 'arccos', 'arctan']:
        s = s.replace(f'\\{fn}', fn)

    # Log functions
    s = re.sub(r'\\ln\s*', 'log', s)  # ln -> log (sympy's natural log)
    # \log_2(8) -> log(8, 2) (sympy uses log(x, base))
    s = re.sub(r'\\log_\{?(\d+)\}?\s*\(([^)]+)\)', r'log(\2, \1)', s)
    s = re.sub(r'\\log_\{?(\d+)\}?\s*(\d+)', r'log(\2, \1)', s)
    s = re.sub(r'\\log\s*', 'log', s)

    # Factorial: 5! -> factorial(5)
    s = re.sub(r'(\d+)!', r'factorial(\1)', s)
    s = re.sub(r'([a-zA-Z])!', r'factorial(\1)', s)

    # Modulo: n \mod 3 -> Mod(n, 3) or n % 3
    s = re.sub(r'(\w+)\s*\\mod\s*(\w+)', r'Mod(\1, \2)', s)
    s = re.sub(r'(\w+)\s*\\pmod\s*\{(\w+)\}', r'Mod(\1, \2)', s)

    # Exponential
    s = s.replace('\\exp', 'exp')
    s = re.sub(r'e\^', 'exp', s)

    # Pi and e
    s = s.replace('\\pi', 'pi')

    # Add implicit multiplication
    # Between number and pi/e: 2pi -> 2*pi, 3e -> 3*E
    s = re.sub(r'(\d)(pi|Pi|PI)\b', r'\1*pi', s)

    # Between number and function: 3sin -> 3*sin
    s = re.sub(r'(\d)\s*(sin|cos|tan|cot|sec|csc|log|exp|sqrt|ln)\s*\(', r'\1*\2(', s)
    # Between closing paren and function: )sin -> )*sin
    s = re.sub(r'\)\s*(sin|cos|tan|cot|sec|csc|log|exp|sqrt|ln)\s*\(', r')*\1(', s)
    # Between number and opening paren: 3( -> 3*(
    s = re.sub(r'(\d)\s*\(', r'\1*(', s)
    # Between closing paren and opening paren: )( -> )*(
    s = re.sub(r'\)\s*\(', r')*(', s)
    # Between closing paren and variable: )x -> )*x
    s = re.sub(r'\)\s*([a-zA-Z])', r')*\1', s)
    # Between number and variable: 3x -> 3*x (but not inside function names)
    s = re.sub(r'(\d)\s*([a-zA-Z])(?![a-zA-Z])', r'\1*\2', s)

    # Clean up whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    # Remove trailing punctuation
    s = s.rstrip('.,;:')

    return s


def extract_all_operators(expr_str: str) -> Optional[set]:
    """
    Parse expression with sympy and return ALL interesting operators.
    Walks the entire AST, not just the root.

    Returns set of operator names, or None if parsing fails.

    E.g., "sin(pi/6) + cos(pi/3)" → {"sin", "cos"}
          "3 * sqrt(144)" → {"sqrt"}
          "48/2" → {"Mul"} (fallback to root)
    """
    lazy_import_sympy()

    cleaned = clean_latex(expr_str)
    if not cleaned or len(cleaned) < 2:
        return None

    try:
        # Parse WITHOUT evaluation to preserve operation structure
        expr = sympy.sympify(cleaned, evaluate=False)

        ops = set()

        # Walk the entire AST
        for node in sympy.preorder_traversal(expr):
            node_type = type(node).__name__

            # Check for functions (sin, cos, log, exp, etc.)
            if isinstance(node, sympy.Function):
                ops.add(node_type)

            # Check for Pow variants - split into meaningful subtypes
            elif isinstance(node, sympy.Pow):
                exp = node.exp
                # sqrt: Pow(x, 1/2) or Pow(x, 0.5)
                if exp == sympy.Rational(1, 2) or exp == 0.5:
                    ops.add("sqrt")
                # cbrt: Pow(x, 1/3)
                elif exp == sympy.Rational(1, 3):
                    ops.add("cbrt")
                # square: Pow(x, 2)
                elif exp == 2 or exp == sympy.Integer(2):
                    ops.add("square")
                # cube: Pow(x, 3)
                elif exp == 3 or exp == sympy.Integer(3):
                    ops.add("cube")
                # inverse: Pow(x, -1)
                elif exp == -1 or exp == sympy.Integer(-1):
                    ops.add("inverse")
                # nth root: Pow(x, 1/n) for n > 3
                elif isinstance(exp, sympy.Rational) and exp.p == 1 and exp.q > 1:
                    ops.add("nth_root")
                # negative integer exponent (other than -1): x^-2, x^-3, etc.
                elif isinstance(exp, (int, sympy.Integer)) and exp < 0:
                    ops.add("neg_pow")
                # positive integer exponent > 3: x^4, x^5, etc.
                elif isinstance(exp, (int, sympy.Integer)) and exp > 3:
                    ops.add("high_pow")
                # fractional exponent that's not a root: x^(2/3), x^(3/4), etc.
                elif isinstance(exp, sympy.Rational):
                    ops.add("frac_pow")
                # anything else (symbolic exponents like x^n, etc.)
                else:
                    ops.add("pow_general")

            # Check for Mod
            elif isinstance(node, sympy.Mod):
                ops.add("Mod")

            # Check for factorial
            elif node_type == 'factorial':
                ops.add("factorial")

            # Check for binomial
            elif node_type == 'binomial':
                ops.add("binomial")

            # Check for floor/ceiling
            elif node_type in ('floor', 'ceiling'):
                ops.add(node_type)

            # Check for Abs
            elif node_type == 'Abs':
                ops.add("Abs")

            # Check for Sum/Product/Integral
            elif node_type in ('Sum', 'Product', 'Integral', 'Derivative', 'Limit'):
                ops.add(node_type)

        # If no interesting operators found, fall back to root
        if not ops:
            root_type = type(expr).__name__
            # Only add if it's an actual operation
            if root_type in ('Add', 'Mul', 'Pow', 'Eq', 'Ne', 'Lt', 'Le', 'Gt', 'Ge'):
                ops.add(root_type)

        return ops if ops else None

    except Exception:
        return None


def extract_Y(text: str) -> Tuple[Optional[set], Optional[str]]:
    """
    Extract Y (set of sympy operators) from a CoT step's text.
    Returns (Y_set, matched_expression) or (None, None) if extraction fails.

    Y is a SET of all interesting operators found in the expression AST.
    E.g., "sin(pi/6) + cos(pi/3)" → {"sin", "cos"}

    Prioritizes expressions with special operators over generic function calls.
    """
    expressions = extract_math_from_text(text)

    # Special operators we prioritize (including refined Pow subtypes)
    special_ops = {'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'asin', 'acos', 'atan',
                   'sinh', 'cosh', 'tanh', 'log', 'exp', 'sqrt', 'cbrt', 'Abs',
                   'factorial', 'binomial', 'Mod', 'floor', 'ceiling',
                   'Sum', 'Product', 'Integral', 'Derivative', 'Limit',
                   # Refined Pow subtypes
                   'square', 'cube', 'inverse', 'nth_root', 'neg_pow', 'high_pow',
                   'frac_pow', 'pow_general'}

    # Single-letter function names to deprioritize
    single_letters = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')

    best_result = None
    best_score = -1

    # Score each expression based on operator quality
    for expr in expressions:
        ops = extract_all_operators(expr)
        if not ops:
            continue

        # Calculate score: special ops are worth 10, others worth 1
        # Penalize single-letter function names
        score = 0
        filtered_ops = set()
        for op in ops:
            if op in special_ops:
                score += 10
                filtered_ops.add(op)
            elif op in single_letters:
                score -= 5  # Penalize f(x), g(x), etc.
            elif len(op) > 1:
                score += 1
                filtered_ops.add(op)

        # Use filtered ops if we have any, otherwise use all ops
        result_ops = filtered_ops if filtered_ops else ops

        if score > best_score:
            best_score = score
            best_result = (result_ops, expr)

    if best_result:
        return best_result

    return None, None


def lambda_handler(event, context):
    """
    Process a chunk of steps and extract Y labels.

    Event:
        key: S3 key to steps file (or aggregated_steps.json)
        chunk_start: optional start index for chunked processing
        chunk_size: optional chunk size
    """
    key = event.get('key', 'ib_results_v2/aggregated_steps.json')
    chunk_start = event.get('chunk_start', 0)
    chunk_size = event.get('chunk_size', 5000)  # Process 5K steps per Lambda

    # Load steps
    try:
        response = s3.get_object(Bucket=BUCKET, Key=key)
        data = json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        return {'error': f'Failed to load {key}: {str(e)}'}

    steps = data.get('steps', data) if isinstance(data, dict) else data

    # Get chunk
    chunk_end = min(chunk_start + chunk_size, len(steps))
    chunk_steps = steps[chunk_start:chunk_end]

    # Process each step
    results = []
    y_counts = {}  # Count each operator across all steps
    parse_failures = 0

    for step in chunk_steps:
        text = step.get('text', '')
        y_set, matched_expr = extract_Y(text)

        result = {
            'problem_idx': step.get('problem_idx'),
            'step_idx': step.get('step_idx'),
            'content_hash': step.get('content_hash', ''),
            'Y': list(y_set) if y_set else None,  # Convert set to list for JSON
            'matched_expr': matched_expr[:200] if matched_expr else None,
        }
        results.append(result)

        if y_set:
            for op in y_set:
                y_counts[op] = y_counts.get(op, 0) + 1
        else:
            parse_failures += 1

    # Write to S3
    output_key = f"{OUTPUT_PREFIX}y_labels_chunk_{chunk_start}.json"
    output_data = {
        'chunk_start': chunk_start,
        'chunk_end': chunk_end,
        'n_steps': len(results),
        'parse_failures': parse_failures,
        'y_distribution': y_counts,
        'results': results,
    }

    s3.put_object(
        Bucket=BUCKET,
        Key=output_key,
        Body=json.dumps(output_data),
        ContentType='application/json'
    )

    return {
        'success': True,
        'chunk_start': chunk_start,
        'chunk_end': chunk_end,
        'n_steps': len(results),
        'parse_failures': parse_failures,
        'y_distribution': y_counts,
        'output_key': output_key,
    }


# Local testing
if __name__ == '__main__':
    # Test cases - should extract interesting operators, not just Add/Mul
    test_cases = [
        ("Calculate: $48/2 = 24$", "Mul (division)"),
        ("So we get $48 - 24 = 24$", "Add (subtraction)"),
        ("\\[\\sin(\\pi/6) = \\frac{1}{2}\\]", "sin"),
        ("Thus $\\sqrt{144} = 12$", "sqrt"),
        ("We have $3^2 + 4^2 = 25$", "square"),
        ("The fraction is $\\frac{3}{4}$", "Mul (fraction)"),
        ("Solving: $2x + 3 = 7$", "Add"),
        ("$\\sin(x) + \\cos(x) = 1$", "sin, cos"),
        ("$3 \\cdot \\sqrt{144} = 36$", "sqrt"),
        ("$\\log_2(8) = 3$", "log"),
        ("$5! = 120$", "factorial"),
        ("$n \\mod 3 = 0$", "Mod"),
        ("$x^3 = 27$", "cube"),
        ("$\\frac{1}{x} = x^{-1}$", "inverse"),
        ("$x^5 + x^4$", "high_pow"),
        ("$\\sqrt[4]{16} = 2$", "nth_root"),
        ("$x^{2/3}$", "frac_pow"),
    ]

    for text, expected in test_cases:
        y_set, expr = extract_Y(text)
        print(f"Text: {text[:60]}...")
        print(f"  Expected: {expected}")
        print(f"  Got: {y_set}")
        print(f"  Matched: {expr[:80] if expr else None}...")
        print()
