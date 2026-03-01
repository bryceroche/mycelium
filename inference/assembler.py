"""
Deterministic Assembler for C3

Given template + extracted operands, builds sympy expression.
This is a lookup table + simple rules - NO model inference.

The assembler is deterministic: if operands are correct, expression is correct.
"""

import re
import sympy
from typing import List, Dict, Optional, Union


# Word to number mapping for implicit values
WORD_TO_NUMBER = {
    # Fractions
    'half': '1/2',
    'quarter': '1/4',
    'third': '1/3',
    'fourth': '1/4',
    'fifth': '1/5',
    'sixth': '1/6',
    'eighth': '1/8',
    'tenth': '1/10',

    # Multipliers
    'double': '2',
    'twice': '2',
    'triple': '3',
    'thrice': '3',
    'quadruple': '4',

    # Percentages
    'percent': '100',

    # Common numbers as words
    'zero': '0',
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9',
    'ten': '10',
    'eleven': '11',
    'twelve': '12',
    'thirteen': '13',
    'fourteen': '14',
    'fifteen': '15',
    'sixteen': '16',
    'seventeen': '17',
    'eighteen': '18',
    'nineteen': '19',
    'twenty': '20',
    'thirty': '30',
    'forty': '40',
    'fifty': '50',
    'sixty': '60',
    'seventy': '70',
    'eighty': '80',
    'ninety': '90',
    'hundred': '100',
    'thousand': '1000',
    'million': '1000000',

    # Special
    'dozen': '12',
    'score': '20',
    'gross': '144',
}

# Template to expression format
# {0}, {1}, etc. are placeholders for operands
TEMPLATE_FORMATS = {
    # Binary arithmetic
    'ADD': '{0} + {1}',
    'SUB': '{0} - {1}',
    'MUL': '{0} * {1}',
    'DIV': '{0} / {1}',
    'MOD': '{0} % {1}',

    # Unary operations
    'SQRT': 'sqrt({0})',
    'SQUARE': '({0})**2',
    'CUBE': '({0})**3',
    'LOG': 'log({0})',
    'LN': 'ln({0})',
    'EXP': 'exp({0})',
    'FACTORIAL': 'factorial({0})',
    'ABS': 'abs({0})',

    # Trigonometric
    'SIN': 'sin({0})',
    'COS': 'cos({0})',
    'TAN': 'tan({0})',
    'COT': 'cot({0})',
    'SEC': 'sec({0})',
    'CSC': 'csc({0})',

    # Powers
    'HIGH_POW': '({0})**{1}',
    'FRAC_POW': '({0})**({1})',

    # Comparison / Equation (output as expression, not equation)
    'EQUATION': '{0} - {1}',  # LHS - RHS = 0 form

    # Other (default to first operand or binary)
    'OTHER': '{0}',
}


def normalize_operand(operand: str) -> str:
    """
    Normalize an operand string to a sympy-parseable format.

    Handles:
    - Word to number conversion
    - Cleaning up whitespace
    - Basic formatting
    """
    if not operand:
        return '0'

    # Clean whitespace
    operand = operand.strip()

    # Check word-to-number mapping (case-insensitive)
    lower = operand.lower()
    if lower in WORD_TO_NUMBER:
        return WORD_TO_NUMBER[lower]

    # Handle compound word numbers like "twenty-five"
    if '-' in lower:
        parts = lower.split('-')
        if all(p in WORD_TO_NUMBER for p in parts):
            # Combine: "twenty-five" = 20 + 5 = 25
            total = sum(int(WORD_TO_NUMBER[p]) for p in parts)
            return str(total)

    # Handle "$X" format (currency)
    if operand.startswith('$'):
        operand = operand[1:]

    # Handle "X%" format (percentage)
    if operand.endswith('%'):
        operand = operand[:-1]

    # Clean up LaTeX artifacts
    operand = operand.replace('\\', '')
    operand = re.sub(r'\s+', '', operand)

    # Handle fractions written as "a/b"
    if '/' in operand and not operand.startswith('('):
        parts = operand.split('/')
        if len(parts) == 2:
            try:
                num = float(parts[0])
                den = float(parts[1])
                return f"({parts[0]})/({parts[1]})"
            except:
                pass

    return operand if operand else '0'


def assemble(template: str, operands: List[str]) -> Optional[str]:
    """
    Assemble a sympy expression from template and operands.

    Args:
        template: Operation template (e.g., 'ADD', 'DIV', 'SQRT')
        operands: List of operand strings

    Returns:
        Sympy expression string, or None if assembly fails
    """
    # Get format string for template
    format_str = TEMPLATE_FORMATS.get(template.upper(), TEMPLATE_FORMATS['OTHER'])

    # Normalize operands
    normalized = [normalize_operand(op) for op in operands]

    # Check we have enough operands
    required = format_str.count('{')
    if len(normalized) < required:
        # Pad with placeholder if needed
        normalized.extend(['x'] * (required - len(normalized)))

    # Build expression
    try:
        expr = format_str.format(*normalized[:required])
        return expr
    except Exception as e:
        return None


def assemble_and_validate(template: str, operands: List[str]) -> Dict:
    """
    Assemble expression and validate with sympy.

    Returns dict with:
        - expression: the assembled expression string
        - valid: whether sympy can parse it
        - value: numeric value if evaluable
        - error: error message if invalid
    """
    expr = assemble(template, operands)

    if expr is None:
        return {
            'expression': None,
            'valid': False,
            'value': None,
            'error': 'Assembly failed'
        }

    try:
        parsed = sympy.sympify(expr)
        result = {
            'expression': expr,
            'valid': True,
            'value': None,
            'error': None
        }

        # Try to evaluate numerically
        try:
            if parsed.is_number:
                result['value'] = float(parsed.evalf())
        except:
            pass

        return result

    except Exception as e:
        return {
            'expression': expr,
            'valid': False,
            'value': None,
            'error': str(e)
        }


def batch_assemble(templates: List[str], operands_list: List[List[str]]) -> List[Dict]:
    """
    Assemble multiple expressions in batch.
    """
    return [
        assemble_and_validate(t, ops)
        for t, ops in zip(templates, operands_list)
    ]


# Testing
if __name__ == '__main__':
    print("Testing assembler...")

    test_cases = [
        ('ADD', ['15', '23']),
        ('DIV', ['48', 'half']),
        ('MUL', ['twelve', 'five']),
        ('SQRT', ['49']),
        ('SQUARE', ['7']),
        ('HIGH_POW', ['2', '10']),
        ('SUB', ['100', 'twenty-five']),
        ('DIV', ['$50', '2']),
    ]

    for template, ops in test_cases:
        result = assemble_and_validate(template, ops)
        print(f"\n{template}({ops}):")
        print(f"  Expression: {result['expression']}")
        print(f"  Valid: {result['valid']}")
        if result['value'] is not None:
            print(f"  Value: {result['value']}")
        if result['error']:
            print(f"  Error: {result['error']}")
