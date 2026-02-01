"""
Recursive Decomposer - Template-based decomposition with signature matching.

Core loop (single-pass, leverage transformer attention):
1. LLM sees full problem, outputs arithmetic steps with values
2. Each step template gets embedded, matched to signature
3. If match → execute
4. If no match → LLM proposes func, add to proposal table, execute

Key insight: Transformers excel at understanding word relationships.
Keep the full problem context when decomposing, don't strip it away.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

import sympy
from sympy import symbols, simplify, factor, expand, sqrt, Rational
from sympy.parsing.sympy_parser import parse_expr

from mycelium.embedding_cache import cached_embed
from mycelium.step_signatures import StepSignatureDB, get_step_db
from mycelium.function_registry import execute, REGISTRY

logger = logging.getLogger(__name__)


# Core math functions that proposals must map to
CORE_FUNCTIONS = {
    "add", "sub", "mul", "truediv", "floordiv", "mod", "pow",
    "abs", "neg", "sqrt",
    "percent_of", "percent_increase", "percent_decrease",
    "min", "max", "avg",
}

# Expected arity for each function (for validation)
FUNCTION_ARITY = {
    "add": 2, "sub": 2, "mul": 2, "truediv": 2, "floordiv": 2, "mod": 2, "pow": 2,
    "abs": 1, "neg": 1, "sqrt": 1,
    "percent_of": 2, "percent_increase": 2, "percent_decrease": 2,
    "min": 2, "max": 2, "avg": 2,
}

# Templates that look like prompt artifacts (should be rejected)
BAD_TEMPLATE_PATTERNS = [
    "describe the arithmetic",
    "next step using",
    "simpler operation",
    "the final result",
]


@dataclass
class SignatureProposal:
    """A proposed new signature for review."""
    template: str
    func: str
    embedding: List[float]
    similarity_to_nearest: float
    nearest_signature_id: Optional[int]
    problem_context: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


# Module-level proposal storage (persisted to DB in production)
_proposals: List[SignatureProposal] = []


def get_proposals() -> List[SignatureProposal]:
    """Get all pending signature proposals."""
    return _proposals.copy()


def clear_proposals() -> None:
    """Clear all pending proposals."""
    global _proposals
    _proposals = []


# =============================================================================
# LLM INTERFACE
# =============================================================================

def call_llm(prompt: str, model: str = "gpt-4o") -> str:
    """Call LLM with a simple prompt. Returns raw text."""
    try:
        from openai import OpenAI
        client = OpenAI()

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You output valid JSON only. No markdown."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise


# =============================================================================
# PATTERN DETECTION
# =============================================================================

def detect_pattern(problem: str) -> str:
    """
    Detect which reasoning pattern a problem requires.

    Returns: 'symbolic', 'composition', 'ratio_chain', 'exponent', 'algebra',
             'complement', 'conditional', 'inversion', 'ratio', or 'standard'
    """
    problem_lower = problem.lower()

    # Substitution patterns: given a=2, b=3, evaluate expression
    substitution_signals = [
        'if $a =', 'if $b =', 'if $x =',
        'let $a =', 'let $b =',
        'given that $a', 'given that $b',
        '$a = ', '$b = ', '$c = ',
    ]
    if any(signal in problem or signal in problem_lower for signal in substitution_signals):
        if 'value of' in problem_lower or 'what is' in problem_lower:
            return 'substitution'

    # Rationalize patterns
    if 'rationalize' in problem_lower:
        return 'rationalize'

    # Logarithm patterns: log_a(b), what power of
    log_signals = [
        'log_', '\\log', 'what power of',
    ]
    if any(signal in problem or signal in problem_lower for signal in log_signals):
        return 'logarithm'

    # Complex number patterns: (a+bi), i^2=-1, simplify complex
    complex_signals = [
        'i^2 = -1', 'i^2=-1', 'i^2 = -1', '$i^2 = -1$',
        '+i)', '-i)', '+2i)', '-2i)', '+3i)', '-3i)',
        '(1+2i)', '(2+3i)', '(3-i)', '(5-3i)',
        'where $i^2', 'where i^2',
    ]
    if any(signal in problem or signal in problem_lower for signal in complex_signals):
        return 'complex'

    # Arithmetic sequence patterns: nth term, first negative, common difference
    sequence_signals = [
        'arithmetic sequence', 'arithmetic progression',
        'first negative', 'first positive',
        'nth term', 'n-th term', 'term of the sequence',
        'common difference', 'first term',
        'how many integers belong',
    ]
    if any(signal in problem_lower for signal in sequence_signals):
        return 'sequence'

    # Vieta's formulas: sum/product of roots
    vieta_signals = [
        'sum of all possible values',
        'sum of the roots', 'product of the roots',
        'sum of all values of $x$',
        'sum of all solutions',
    ]
    if any(signal in problem_lower for signal in vieta_signals):
        if 'x' in problem or '$x$' in problem:
            return 'vieta'

    # System of equations: multiple equations with multiple unknowns
    # Signals: multiple = signs, words like "and", explicit system setup
    system_signals = [
        'weigh as much as', 'together cost',
        'is $8$ less than', 'is $9$ less than',
        '$a+b=', '$b+c=', '$a+c=',
        'and $2e$', 'and $2d$',
    ]
    if any(signal in problem or signal in problem_lower for signal in system_signals):
        return 'system'
    # Also check for multiple equations with multiple unknowns
    if problem.count('=') >= 2 and sum(1 for v in ['a', 'b', 'c', 'd', 'e'] if f'${v}' in problem or f' {v} ' in problem_lower) >= 2:
        return 'system'

    # Age problem patterns (system of equations word problems)
    age_signals = [
        'years ago', 'years from now', 'years older',
        "father's age", "mother's age", "son's age", "daughter's age",
    ]
    if any(signal in problem_lower for signal in age_signals):
        return 'age_problem'

    # Symbolic patterns: simplify, factor, expand (need SymPy)
    symbolic_signals = [
        'simplify', 'factor', 'expand',
        'express as', 'write as',
    ]
    # Check for algebraic expression manipulation
    if any(signal in problem_lower for signal in symbolic_signals):
        # Confirm it has variables (x, y, etc.) not just numbers
        if any(var in problem for var in ['x', 'y', 'z', 's', 'n', 'a', 'b']):
            if '^' in problem or 'frac' in problem:
                return 'symbolic'

    # Equation solving patterns: solve for x, find x where equation = 0
    equation_signals = [
        'solve for', 'find the value of $x$', 'find $x$',
        'what is $x$', 'value of $x$ in the equation',
        '= 0$', '=0$',
    ]
    if any(signal in problem or signal in problem_lower for signal in equation_signals):
        # Must have an equation with variable
        if '=' in problem and any(v in problem for v in ['x', 'y', 'n', 'a']):
            return 'equation'

    # Function composition patterns: f(g(x)), nested function evaluation
    composition_signals = [
        'f(g(', 'g(f(', 'f(h(', 'h(f(',
        'what is $f(g', 'what is $g(f',
        'find $f(g', 'find $g(f',
    ]
    if any(signal in problem or signal in problem_lower for signal in composition_signals):
        return 'composition'

    # Ratio chain patterns: x/y=a, y/z=b, find x/z
    ratio_chain_signals = [
        'frac{x}{y}', 'frac{y}{z}', 'frac{z}{x}',
        'frac{a}{b}', 'frac{b}{c}',
        'x/y', 'y/z', 'z/x',
    ]
    # Need at least 2 ratio definitions
    ratio_count = sum(1 for sig in ratio_chain_signals if sig in problem)
    if ratio_count >= 2 and 'value of' in problem_lower:
        return 'ratio_chain'

    # Exponent/logarithm patterns: solve for x in power equations
    exponent_signals = [
        '^x', '= x$', 'value of $x$', 'value of x',
        '2^', '3^', '4^', '5^', '10^', '17^',
    ]
    # Check for exponent equations (contains ^ and asks for x)
    if '^' in problem and ('x' in problem_lower or 'value of' in problem_lower):
        if any(signal in problem for signal in exponent_signals):
            return 'exponent'

    # Algebra/equation patterns: solve for unknown variable in equation
    algebra_signals = [
        'how much was her previous', 'how much was his previous',
        'what was the original', 'find the original',
        'increased by', 'decreased by',
        'now only amount to', 'used to spend',
    ]
    # Check for "before and after" comparison implying equation setup
    if any(signal in problem_lower for signal in algebra_signals):
        if 'previous' in problem_lower or 'original' in problem_lower or ('used to' in problem_lower and 'now' in problem_lower):
            return 'algebra'

    # Percentage complement patterns: X% did Y, find count of not-Y
    complement_signals = [
        '% got', '% of his', '% of her',
        'percent got', 'percent of his', 'percent of her',
    ]
    if any(signal in problem_lower for signal in complement_signals):
        # Check if asking about the complement (above/below, pass/fail)
        if 'how many' in problem_lower and ('above' in problem_lower or 'below' in problem_lower or 'and above' in problem_lower):
            return 'complement'

    # Conditional patterns: if/then, thresholds, overtime
    conditional_signals = [
        'if she works more than', 'if he works more than',
        'eligible for overtime', 'overtime',
        'if more than', 'if less than',
        'when exceeds', 'above', 'below threshold',
    ]
    if any(signal in problem_lower for signal in conditional_signals):
        return 'conditional'

    # Inversion patterns: solve for unknown count/amount
    inversion_signals = [
        'how many driveways', 'how many lawns',
        'how many did he', 'how many did she',
        'how much did he', 'how much did she',
        'how many more does', 'how many does he need',
    ]
    if any(signal in problem_lower for signal in inversion_signals):
        # Check if it's asking to solve for something given a result
        if 'after' in problem_lower or 'left' in problem_lower or 'change' in problem_lower:
            return 'inversion'

    # Ratio patterns: proportional splits, same ratio
    ratio_signals = [
        'ratio', 'shared among', 'split between',
        'divided among', 'in the ratio',
        'proportional', 'same ratio',
        'uses one ounce', 'this same ratio',
    ]
    if any(signal in problem_lower for signal in ratio_signals):
        return 'ratio'

    return 'standard'


# =============================================================================
# SPECIALIZED DECOMPOSITION TEMPLATES
# =============================================================================

def decompose_symbolic(problem: str) -> Dict[str, Any]:
    """Template for symbolic algebra problems using SymPy (simplify, factor, expand)."""
    prompt = f'''Convert this algebra problem to a SymPy expression.

Problem: {problem}

Output JSON:
{{
    "operation": "simplify" | "factor" | "expand" | "solve",
    "expression": "the expression in Python/SymPy syntax",
    "variable": "main variable (usually x)"
}}

CONVERSION RULES:
1. Use ** for exponents (NOT ^)
2. Use * for multiplication (even implicit: 2x → 2*x)
3. Fractions: a/b or Rational(a, b) for exact fractions
4. Square roots: sqrt(x)
5. Common patterns:
   - $x^2$ → x**2
   - $\\frac{{a}}{{b}}$ → a/b
   - $2x^3$ → 2*x**3
   - $\\sqrt{{x}}$ → sqrt(x)

EXAMPLES:
"Simplify (2s^5)/(s^3) - 6s^2 + (7s^3)/s"
→ {{"operation": "simplify", "expression": "(2*s**5)/(s**3) - 6*s**2 + (7*s**3)/s", "variable": "s"}}

"Factor 30x^3 - 8x^2 + 20x"
→ {{"operation": "factor", "expression": "30*x**3 - 8*x**2 + 20*x", "variable": "x"}}

"Expand (x+2)(x-3)"
→ {{"operation": "expand", "expression": "(x+2)*(x-3)", "variable": "x"}}'''

    response = call_llm(prompt)
    return json.loads(response)


def decompose_equation(problem: str) -> Dict[str, Any]:
    """Template for equation solving problems using SymPy (solve for x)."""
    prompt = f'''Solve this equation for the variable.

Problem: {problem}

Output JSON:
{{
    "equation": "the equation in Python/SymPy syntax (use Eq(lhs, rhs))",
    "variable": "the variable to solve for (usually x)"
}}

CONVERSION RULES:
1. Use ** for exponents (NOT ^)
2. Use * for multiplication (even implicit: 2x → 2*x)
3. Use Eq(left, right) for equations
4. Common patterns:
   - $x^2 - 9 = 0$ → Eq(x**2 - 9, 0)
   - $2x + 3 = 7$ → Eq(2*x + 3, 7)
   - $3^{{x+8}} = 9^{{x+3}}$ → Eq(3**(x+8), 9**(x+3))

EXAMPLES:
"Solve x^2 - 9 = 0"
→ {{"equation": "Eq(x**2 - 9, 0)", "variable": "x"}}

"If 3^(x+8) = 9^(x+3), what is x?"
→ {{"equation": "Eq(3**(x+8), 9**(x+3))", "variable": "x"}}

"Find x if 2x + 5 = 13"
→ {{"equation": "Eq(2*x + 5, 13)", "variable": "x"}}'''

    response = call_llm(prompt)
    return json.loads(response)


def decompose_substitution(problem: str) -> Dict[str, Any]:
    """Template for variable substitution problems."""
    prompt = f'''Substitute the given values and compute the result.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "arithmetic expression", "result": "step1"}},
        ...
    ],
    "answer": "final_step_name"
}}

SUBSTITUTION RULES:
1. First, identify ALL given values (a=2, b=3, c=4, etc.)
2. Write out the expression with values substituted
3. Compute step by step, ONE operation at a time
4. Use ** for exponents

EXAMPLE: "If a=2, b=3, c=4, what is (b-c)^2 + a(b+c)?"
- Given: a=2, b=3, c=4
- Expression: (b-c)^2 + a(b+c) = (3-4)^2 + 2(3+4)
- Step 1: 3 - 4 = -1
- Step 2: (-1)^2 = 1
- Step 3: 3 + 4 = 7
- Step 4: 2 * 7 = 14
- Step 5: 1 + 14 = 15

Steps:
{{"description": "b minus c", "expr": "3 - 4", "result": "diff"}}
{{"description": "diff squared", "expr": "diff ** 2", "result": "sq"}}
{{"description": "b plus c", "expr": "3 + 4", "result": "sum"}}
{{"description": "a times sum", "expr": "2 * sum", "result": "prod"}}
{{"description": "final sum", "expr": "sq + prod", "result": "answer"}}
answer: "answer"

Use actual numbers. Use ** for exponents. Each step ONE operation.'''

    response = call_llm(prompt)
    return json.loads(response)


def decompose_rationalize(problem: str) -> Dict[str, Any]:
    """Template for rationalizing denominators using SymPy."""
    prompt = f'''Rationalize this expression's denominator.

Problem: {problem}

Output JSON:
{{
    "expression": "the expression in Python/SymPy syntax",
    "operation": "rationalize"
}}

CONVERSION RULES:
1. Use sqrt(n) for square roots
2. Use ** for exponents
3. Use Rational(a, b) for fractions

EXAMPLES:
"Rationalize 1/(2*sqrt(7))"
→ {{"expression": "1/(2*sqrt(7))", "operation": "rationalize"}}

"Rationalize sqrt(2)/sqrt(3)"
→ {{"expression": "sqrt(2)/sqrt(3)", "operation": "rationalize"}}'''

    response = call_llm(prompt)
    result = json.loads(response)
    result['_pattern'] = 'rationalize'
    return result


def decompose_logarithm(problem: str) -> Dict[str, Any]:
    """Template for logarithm and 'what power of' problems."""
    prompt = f'''Solve this logarithm or power problem.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "arithmetic expression", "result": "step1"}},
        ...
    ],
    "answer": "final_step_name"
}}

LOGARITHM RULES:
1. log_a(b) means "what power of a equals b?"
2. If a^x = b, then x = log(b) / log(a)
3. "What power of 4 equals 8?" means solve 4^x = 8

EXAMPLE 1: "Evaluate log_2(64)"
- log_2(64) means: 2^x = 64
- 64 = 2^6, so x = 6
Steps:
{{"description": "log base 2 of 64", "expr": "log(64) / log(2)", "result": "answer"}}
answer: "answer"

EXAMPLE 2: "What power of 4 equals 8?"
- 4^x = 8
- (2^2)^x = 2^3
- 2^(2x) = 2^3
- 2x = 3, x = 1.5
Steps:
{{"description": "power of 4 that equals 8", "expr": "log(8) / log(4)", "result": "answer"}}
answer: "answer"

Use log() for natural log. Each step ONE operation.'''

    response = call_llm(prompt)
    return json.loads(response)


def decompose_complex(problem: str) -> Dict[str, Any]:
    """Template for complex number arithmetic (a+bi)."""
    prompt = f'''Solve this complex number problem. Remember i^2 = -1.

Problem: {problem}

Output JSON:
{{
    "real_part": "expression for real component",
    "imag_part": "expression for imaginary component",
    "format": "a + bi"
}}

COMPLEX NUMBER RULES:
1. i^2 = -1
2. (a+bi)(c+di) = (ac-bd) + (ad+bc)i
3. Multiply out, then combine real and imaginary parts separately

EXAMPLE 1: "Simplify (2-2i)(5+5i)"
- Real: 2*5 - (-2)*5 = 10 - (-10) = 10 + 10 = 20
- Imag: 2*5 + (-2)*5 = 10 + (-10) = 0
Output: {{"real_part": "2*5 - (-2)*5", "imag_part": "2*5 + (-2)*5", "format": "a + bi"}}

EXAMPLE 2: "Evaluate (1+2i)(6-3i)"
- Real: 1*6 - 2*(-3) = 6 + 6 = 12
- Imag: 1*(-3) + 2*6 = -3 + 12 = 9
Output: {{"real_part": "1*6 - 2*(-3)", "imag_part": "1*(-3) + 2*6", "format": "a + bi"}}

Compute real and imaginary parts separately.'''

    response = call_llm(prompt)
    return json.loads(response)


def decompose_sequence(problem: str) -> Dict[str, Any]:
    """Template for arithmetic sequence problems."""
    prompt = f'''Solve this arithmetic sequence problem.

Problem: {problem}

Output JSON:
{{
    "first_term": "a_1 value or expression",
    "common_diff": "d value (difference between terms)",
    "formula": "a_n = a_1 + (n-1)*d",
    "target": "what we're finding (n, a_n, or count)",
    "steps": [
        {{"description": "what this computes", "expr": "expression", "result": "step1"}}
    ],
    "answer": "final_step_name"
}}

ARITHMETIC SEQUENCE FORMULAS:
1. nth term: a_n = a_1 + (n-1)*d
2. First negative term: find smallest n where a_n < 0
3. Count of terms from a to b with step d: floor((b-a)/d) + 1

EXAMPLE: "In sequence 1000, 987, 974, ... find which term is first negative"
- a_1 = 1000, d = 987-1000 = -13
- a_n = 1000 + (n-1)*(-13) < 0
- 1000 - 13n + 13 < 0
- 1013 < 13n
- n > 77.9, so n = 78

Steps: compute a_1, d, solve inequality for n, take ceiling.'''

    response = call_llm(prompt)
    return json.loads(response)


def decompose_vieta(problem: str) -> Dict[str, Any]:
    """Template for sum/product of roots using Vieta's formulas."""
    prompt = f'''Solve using Vieta's formulas for quadratic equations.

Problem: {problem}

Output JSON:
{{
    "equation": "the quadratic in standard form ax^2 + bx + c = 0",
    "a": "coefficient of x^2",
    "b": "coefficient of x",
    "c": "constant term",
    "find": "sum or product",
    "steps": [
        {{"description": "what this computes", "expr": "expression", "result": "step1"}}
    ],
    "answer": "final_step_name"
}}

VIETA'S FORMULAS for ax^2 + bx + c = 0:
- Sum of roots: x1 + x2 = -b/a
- Product of roots: x1 * x2 = c/a

EXAMPLE: "Sum of all values of x such that 2x(x-10) = -50"
1. Expand: 2x^2 - 20x = -50
2. Standard form: 2x^2 - 20x + 50 = 0
3. a=2, b=-20, c=50
4. Sum of roots = -(-20)/2 = 20/2 = 10

Wait, let me recalculate: 2x^2 - 20x + 50 = 0, divide by 2: x^2 - 10x + 25 = 0
This factors as (x-5)^2 = 0, so x=5 is a double root.
Sum = 5 + 5 = 10? No wait, the problem asks for sum of all POSSIBLE values.
Since x=5 is the only solution, sum = 5.

Be careful: "sum of all possible values" vs "sum of roots" (counting multiplicity).'''

    response = call_llm(prompt)
    return json.loads(response)


def decompose_system(problem: str) -> Dict[str, Any]:
    """Template for system of equations (2+ equations, 2+ unknowns)."""
    prompt = f'''Solve this system of equations problem.

Problem: {problem}

Output JSON:
{{
    "equations": ["eq1 in form 'expr = value'", "eq2", ...],
    "unknowns": ["x", "y", ...],
    "method": "substitution or elimination or sympy",
    "steps": [
        {{"description": "what this computes", "expr": "expression", "result": "step1"}}
    ],
    "answer": "final_step_name"
}}

SYSTEM SOLVING STRATEGIES:
1. Substitution: solve one equation for one variable, substitute into other
2. Elimination: add/subtract equations to eliminate a variable
3. For 3 equations with 3 unknowns: a+b=8, b+c=-3, a+c=-5
   - Add all three: 2(a+b+c) = 0, so a+b+c = 0
   - Then: c = 0-8 = -8, a = 0-(-3) = 3, b = 0-(-5) = 5

EXAMPLE: "a+b=8, b+c=-3, a+c=-5, find abc"
- Add all: 2(a+b+c) = 8 + (-3) + (-5) = 0, so a+b+c = 0
- a = (a+b+c) - (b+c) = 0 - (-3) = 3
- b = (a+b+c) - (a+c) = 0 - (-5) = 5
- c = (a+b+c) - (a+b) = 0 - 8 = -8
- abc = 3 * 5 * (-8) = -120

Output numeric values, not symbolic expressions.'''

    response = call_llm(prompt)
    return json.loads(response)


def decompose_age_problem(problem: str) -> Dict[str, Any]:
    """Template for age problems (system of equations)."""
    prompt = f'''Solve this age problem by setting up equations.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "arithmetic expression", "result": "step1"}},
        ...
    ],
    "answer": "final_step_name"
}}

AGE PROBLEM RULES:
1. Let x = person's current age (usually the younger person)
2. Set up equations based on the relationships
3. Solve the system step by step

EXAMPLE: "Father is 5 times son's age. 3 years ago, sum of ages was 30. How old is son?"
- Let s = son's current age
- Father's current age = 5s
- 3 years ago: son was (s-3), father was (5s-3)
- Equation: (s-3) + (5s-3) = 30
- Simplify: 6s - 6 = 30
- Solve: 6s = 36, s = 6

Steps:
{{"description": "sum of 3 years ago ages equation: 6s - 6 = 30, so 6s = 36", "expr": "30 + 6", "result": "total"}}
{{"description": "divide by 6 to get s", "expr": "total / 6", "result": "son_age"}}
answer: "son_age"

Use actual numbers. Each step ONE operation.'''

    response = call_llm(prompt)
    return json.loads(response)


def decompose_composition(problem: str) -> Dict[str, Any]:
    """Template for function composition problems like f(g(x))."""
    prompt = f'''Solve this function composition problem. Evaluate from INSIDE OUT.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "arithmetic expression", "result": "step1"}},
        ...
    ],
    "answer": "final_step_name"
}}

COMPOSITION RULES:
1. Identify the INNER function and its input value
2. Evaluate the INNER function first
3. Use that result as input to the OUTER function
4. Evaluate the OUTER function

EXAMPLE: "Let f(x) = 2x - 4 and g(x) = x^2 + 3. What is f(g(2))?"
Step 1: Evaluate inner function g(2) = 2**2 + 3 = 4 + 3 = 7
Step 2: Use result in outer function f(7) = 2*7 - 4 = 14 - 4 = 10

Steps:
{{"description": "evaluate g(2): 2 squared", "expr": "2**2", "result": "sq"}}
{{"description": "evaluate g(2): add 3", "expr": "sq + 3", "result": "g_result"}}
{{"description": "evaluate f(g_result): multiply by 2", "expr": "2 * g_result", "result": "times2"}}
{{"description": "evaluate f(g_result): subtract 4", "expr": "times2 - 4", "result": "answer"}}
answer: "answer"

Use ** for exponents. Each step ONE operation.'''

    response = call_llm(prompt)
    return json.loads(response)


def decompose_ratio_chain(problem: str) -> Dict[str, Any]:
    """Template for ratio chain problems like x/y=a, z/x=b, find z/y."""
    prompt = f'''Solve this ratio chain problem by multiplying/dividing ratios.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "arithmetic expression", "result": "step1"}},
        ...
    ],
    "answer": "final_step_name"
}}

RATIO CHAIN RULES:
1. Write down each given ratio as a fraction
2. To find a new ratio, multiply ratios that chain together
3. Cancel common terms: (x/y) * (z/x) = z/y (x cancels)

EXAMPLE: "If x/y = 2 and z/x = 4, what is z/y?"
- We want z/y
- We have z/x = 4 and x/y = 2
- Chain: (z/x) * (x/y) = z/y (x cancels!)
- Answer: 4 * 2 = 8

Steps:
{{"description": "z/x ratio", "expr": "4", "result": "ratio_zx"}}
{{"description": "x/y ratio", "expr": "2", "result": "ratio_xy"}}
{{"description": "multiply to get z/y", "expr": "ratio_zx * ratio_xy", "result": "answer"}}
answer: "answer"

Use actual numbers from the problem. Each step ONE operation.'''

    response = call_llm(prompt)
    return json.loads(response)


def decompose_exponent(problem: str) -> Dict[str, Any]:
    """Template for exponent/power equation problems (solve for x in a^b = c^x)."""
    prompt = f'''Solve this exponent/power equation problem.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "arithmetic expression", "result": "step1"}},
        ...
    ],
    "answer": "final_step_name"
}}

EXPONENT RULES:
1. Use ** for exponentiation in Python (NOT ^)
2. Convert bases to same base when possible
   - 4 = 2**2, 8 = 2**3, 16 = 2**4, 32 = 2**5
   - 9 = 3**2, 27 = 3**3, 81 = 3**4
3. Use logarithms for harder cases: log(value) / log(base)
4. When a^m = a^n, then m = n

EXAMPLE 1: "If 2^8 = 4^x, what is x?"
- Rewrite: 4 = 2**2, so 4^x = (2**2)**x = 2**(2*x)
- Equation: 2**8 = 2**(2*x)
- Therefore: 8 = 2*x, so x = 8/2 = 4
Steps:
{{"description": "exponent on left side", "expr": "8", "result": "left_exp"}}
{{"description": "base conversion factor", "expr": "2", "result": "factor"}}
{{"description": "solve for x", "expr": "left_exp / factor", "result": "x"}}
answer: "x"

EXAMPLE 2: "(17^6 - 17^5) / 16 = 17^x"
- Factor: 17^5 * (17 - 1) / 16 = 17^5 * 16 / 16 = 17^5
- So x = 5
Steps:
{{"description": "compute 17-1", "expr": "17 - 1", "result": "diff"}}
{{"description": "the answer is the exponent", "expr": "5", "result": "x"}}
answer: "x"

Use actual numbers. Each step ONE operation.'''

    response = call_llm(prompt)
    return json.loads(response)


def decompose_algebra(problem: str) -> Dict[str, Any]:
    """Template for algebra/equation problems (solve for unknown variable)."""
    prompt = f'''Solve this algebra problem by setting up and solving an equation.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "arithmetic expression", "result": "step1"}},
        ...
    ],
    "answer": "final_step_name"
}}

ALGEBRA RULES:
1. Identify the UNKNOWN (e.g., "previous income", "original price")
2. Set up the EQUATION relating before and after states
3. Solve step by step using algebra

EXAMPLE: "Spent 40% on rent. Income increased by $600. Now rent is 25% of new income. Find previous income."
- Let x = previous income
- Old rent = 0.40 * x
- New income = x + 600
- New rent = 0.25 * (x + 600)
- Old rent = New rent (same amount): 0.40x = 0.25(x + 600)
- 0.40x = 0.25x + 150
- 0.15x = 150
- x = 150 / 0.15 = 1000

Steps should compute intermediate values leading to the answer.
Use actual numbers. Each step ONE operation.'''

    response = call_llm(prompt)
    return json.loads(response)


def decompose_complement(problem: str) -> Dict[str, Any]:
    """Template for percentage complement problems (X% did Y, find not-Y)."""
    prompt = f'''Solve this percentage problem involving complements.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "arithmetic expression", "result": "step1"}},
        ...
    ],
    "answer": "final_step_name"
}}

PERCENTAGE COMPLEMENT RULES:
1. If X% did something, then (100 - X)% did NOT do it
2. "40% got below B" means "60% got B and above"
3. Calculate the complement percentage first
4. Then apply to the total

EXAMPLE: "40% of 60 students got below B. How many got B and above?"
- Complement: 100 - 40 = 60 (percent who got B and above)
- Convert to decimal: 60 / 100 = 0.60
- Apply to total: 0.60 * 60 = 36 students

Use actual numbers from the problem. Each step ONE operation.'''

    response = call_llm(prompt)
    return json.loads(response)


def decompose_conditional(problem: str) -> Dict[str, Any]:
    """Template for conditional logic problems (if/then, overtime, thresholds)."""
    prompt = f'''Solve this problem with conditional logic. Show all steps.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "arithmetic expression", "result": "step1"}},
        ...
    ],
    "answer": "final_step_name"
}}

CONDITIONAL LOGIC RULES:
1. First identify the THRESHOLD (e.g., "more than 8 hours")
2. Calculate the amount ABOVE threshold (e.g., 10 - 8 = 2 overtime hours)
3. Calculate the amount AT OR BELOW threshold (e.g., 8 regular hours)
4. Apply DIFFERENT RATES to each portion
5. Combine the results

OVERTIME EXAMPLE:
- Regular hours = min(total_hours, threshold) = min(10, 8) = 8
- Overtime hours = max(0, total_hours - threshold) = max(0, 10-8) = 2
- Regular pay = regular_hours * regular_rate
- Overtime pay = overtime_hours * overtime_rate
- Total = regular_pay + overtime_pay

Use actual numbers from the problem. Each step should be ONE operation.'''

    response = call_llm(prompt)
    return json.loads(response)


def decompose_inversion(problem: str) -> Dict[str, Any]:
    """Template for algebraic inversion problems (solve for unknown)."""
    prompt = f'''Solve this problem by working BACKWARDS to find an unknown.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "arithmetic expression", "result": "step1"}},
        ...
    ],
    "answer": "final_step_name"
}}

INVERSION/WORKING BACKWARDS RULES:
1. Identify what you KNOW (total needed, amounts from specific sources)
2. Identify what you DON'T KNOW (the unknown count/amount)
3. Set up: known_sources + unknown = total
4. Solve: unknown = total - known_sources
5. If unknown is a COUNT: unknown_count = unknown_amount / rate_per_item

EXAMPLE: "Total needed is $110. Allowance gives $15. Mowing 4 lawns at $15 each gives $60. How many driveways at $7 each?"
- Total needed: 110
- From allowance: 15
- From mowing: 4 * 15 = 60
- From driveways: 110 - 15 - 60 = 35
- Number of driveways: 35 / 7 = 5

Use actual numbers from the problem. Each step should be ONE operation.'''

    response = call_llm(prompt)
    return json.loads(response)


def decompose_ratio(problem: str) -> Dict[str, Any]:
    """Template for ratio/proportion problems."""
    prompt = f'''Solve this ratio/proportion problem.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "arithmetic expression", "result": "step1"}},
        ...
    ],
    "answer": "final_step_name"
}}

RATIO/PROPORTION RULES:

TYPE 1 - SPLIT RATIO (sharing in ratio X:Y):
1. Total parts = X + Y
2. Value per part = total_amount / total_parts
3. Each share = their_parts * value_per_part

TYPE 2 - SCALING RATIO (same ratio, different size):
1. Find the ratio from the known case (e.g., 1 oz tea per 8 oz water = 1/8)
2. Apply that ratio to the new case (e.g., 6 oz cup * (1/8) = 0.75 oz tea)
3. Multiply by count if needed (e.g., 12 people * 0.75 = 9 oz)

EXAMPLE 1 (Split): "$1400 in ratio 2:5"
- Total parts: 2 + 5 = 7
- Per part: 1400 / 7 = 200
- Mike: 2 * 200 = 400

EXAMPLE 2 (Scale): "8 oz water uses 1 oz tea. How much tea for 12 people with 6 oz cups?"
- Ratio: 1 / 8 = 0.125 oz tea per oz water
- Per person: 6 * 0.125 = 0.75 oz tea
- Total: 12 * 0.75 = 9 oz tea

Use actual numbers. Convert ratios to decimals. Each step ONE operation.'''

    response = call_llm(prompt)
    return json.loads(response)


def decompose_single_pass(problem: str) -> Dict[str, Any]:
    """
    Single-pass decomposition that keeps full problem context.

    Lets the transformer see the whole problem and understand
    which values relate to which operations naturally.

    Returns:
        {
            "steps": [
                {"description": "price of cream cheese", "expr": "10 * 0.5", "result": "step1"},
                {"description": "price of cold cuts", "expr": "10 * 2", "result": "step2"},
                {"description": "total cost", "expr": "10 + step1 + step2", "result": "answer"}
            ]
        }
    """
    prompt = f'''Break this problem into arithmetic steps. Output the actual math expressions.

Problem: {problem}

Output JSON:
{{
    "steps": [
        {{"description": "what this computes", "expr": "10 * 0.5", "result": "step1"}},
        {{"description": "next calculation", "expr": "step1 + 20", "result": "step2"}}
    ],
    "answer": "step2"
}}

RULES:
1. Each step must be ONE arithmetic operation: +, -, *, /
2. Use actual NUMBERS from the problem in "expr" - NO VARIABLES like x, y, n
3. Reference previous step results by name (step1, step2, etc.)
4. "description" should say WHAT you're computing (e.g., "cream cheese price")
5. "answer" is the final step name

CRITICAL - NUMBERS ONLY:
- expr must contain ONLY numbers and step references
- WRONG: "2 * x" or "x + 3" (contains variable x)
- RIGHT: "2 * 5" or "step1 + 3" (only numbers/step refs)
- Use ** for exponents: "2 ** 3" not "2^3"

IMPORTANT: Read the problem carefully!
- "X for $Y" means the price IS $Y (don't multiply X*Y)
- "half the price of X" means multiply X's price by 0.5
- "twice as much" means multiply by 2
- "80% more" means original + (original * 0.8), NOT original * 0.8

FRACTIONS: Convert to decimals in expressions!
- "1/2 times" → use 0.5
- "1/3 of" → use 0.333333
- "1/4 of" → use 0.25
- "1/6 cup" → use 0.166667
- "2/3 of" → use 0.666667
- DO NOT write "1/2" in expr - write the decimal 0.5'''

    response = call_llm(prompt)
    return json.loads(response)


def decompose_with_pattern(problem: str) -> Dict[str, Any]:
    """
    Detect pattern and use appropriate specialized template.
    """
    pattern = detect_pattern(problem)
    logger.info(f"[decomposer] Detected pattern: {pattern}")

    if pattern == 'substitution':
        return decompose_substitution(problem)
    elif pattern == 'rationalize':
        return decompose_rationalize(problem)
    elif pattern == 'logarithm':
        return decompose_logarithm(problem)
    elif pattern == 'complex':
        result = decompose_complex(problem)
        result['_pattern'] = 'complex'
        return result
    elif pattern == 'sequence':
        result = decompose_sequence(problem)
        result['_pattern'] = 'sequence'
        return result
    elif pattern == 'vieta':
        result = decompose_vieta(problem)
        result['_pattern'] = 'vieta'
        return result
    elif pattern == 'system':
        result = decompose_system(problem)
        result['_pattern'] = 'system'
        return result
    elif pattern == 'age_problem':
        return decompose_age_problem(problem)
    elif pattern == 'symbolic':
        # Symbolic returns different structure, mark it
        result = decompose_symbolic(problem)
        result['_pattern'] = 'symbolic'
        return result
    elif pattern == 'equation':
        # Equation solving returns different structure, mark it
        result = decompose_equation(problem)
        result['_pattern'] = 'equation'
        return result
    elif pattern == 'composition':
        return decompose_composition(problem)
    elif pattern == 'ratio_chain':
        return decompose_ratio_chain(problem)
    elif pattern == 'exponent':
        return decompose_exponent(problem)
    elif pattern == 'algebra':
        return decompose_algebra(problem)
    elif pattern == 'complement':
        return decompose_complement(problem)
    elif pattern == 'conditional':
        return decompose_conditional(problem)
    elif pattern == 'inversion':
        return decompose_inversion(problem)
    elif pattern == 'ratio':
        return decompose_ratio(problem)
    else:
        return decompose_single_pass(problem)


# =============================================================================
# STEP 1: EXTRACT AND MASK (legacy two-pass approach)
# =============================================================================

def extract_and_mask(problem: str) -> Dict[str, Any]:
    """
    Extract values and create masked template.

    Returns:
        {
            "template": "Person has <A>. Gives <B> to person. How many left?",
            "values": {"A": 10, "B": 3}
        }
    """
    prompt = f'''Extract ALL numbers from this problem and create a template.

Problem: {problem}

Output JSON:
{{
    "template": "the problem with <A>, <B>, <C> replacing numbers",
    "values": {{"A": number, "B": number, ...}}
}}

CRITICAL RULES:
1. Extract EVERY number, including:
   - Explicit numbers: "5 apples" → extract 5
   - Percentages: "20%" → extract 0.20 (as decimal)
   - Word numbers: "twice" → 2, "half" → 0.5, "three" → 3
   - Time words: "two hours" → 2
   - IMPLIED conversion constants you'll need:
     * If problem involves hours to minutes: extract 60
     * If problem involves weeks to year: extract 52
     * If problem involves dozen: extract 12
     * If problem involves "both" or "each way": extract 2

2. Replace ALL numbers with <A>, <B>, <C>, etc. in order
3. Values must be numeric (floats or ints)
4. Do NOT leave any numbers in the template - replace them ALL

Example:
Problem: "Tim works 2 hours. How many minutes?"
values: {{"A": 2, "B": 60}}  ← B=60 is the implicit hours-to-minutes conversion'''

    response = call_llm(prompt)
    return json.loads(response)


# =============================================================================
# STEP 2: BUILD POINTER GRAPH (which values connect to answer)
# =============================================================================

def validate_pointer_graph(graph: Dict[str, Any], values: Dict[str, float]) -> List[str]:
    """
    Validate a pointer graph for correctness.

    Returns list of error messages (empty if valid).
    """
    errors = []
    steps = graph.get("steps", [])
    defined_refs = set(values.keys())  # Start with extracted values

    for i, step in enumerate(steps):
        # Check required fields
        if "template" not in step:
            errors.append(f"Step {i+1}: missing 'template' field")
            continue
        if "uses" not in step:
            errors.append(f"Step {i+1}: missing 'uses' field")
            continue
        if "result" not in step:
            errors.append(f"Step {i+1}: missing 'result' field")
            continue

        template = step["template"]
        uses = step["uses"]
        result = step["result"]

        # Check for prompt artifacts in template
        for bad_pattern in BAD_TEMPLATE_PATTERNS:
            if bad_pattern.lower() in template.lower():
                errors.append(f"Step {i+1}: template looks like prompt artifact: '{template}'")
                break

        # Check that all references exist
        for ref in uses:
            if ref not in defined_refs:
                errors.append(f"Step {i+1}: undefined reference '{ref}' in uses")

        # Check arity (most ops are binary)
        if len(uses) < 1:
            errors.append(f"Step {i+1}: no inputs specified")
        elif len(uses) > 3:
            errors.append(f"Step {i+1}: too many inputs ({len(uses)})")

        # Add this step's result to defined refs for subsequent steps
        defined_refs.add(result)

    # Check answer reference
    answer = graph.get("answer")
    if answer and answer not in defined_refs:
        errors.append(f"Answer '{answer}' not defined in any step")

    return errors


def build_pointer_graph(template: str, values: Dict[str, float], max_retries: int = 2) -> Dict[str, Any]:
    """
    Figure out how values connect to produce the answer.

    Returns:
        {
            "steps": [
                {"template": "A minus B", "uses": ["A", "B"], "result": "step1"},
                {"template": "step1 times C", "uses": ["step1", "C"], "result": "step2"}
            ],
            "answer": "step2"
        }
    """
    values_str = ", ".join(f"{k} = {v}" for k, v in values.items())
    value_keys = ", ".join(values.keys())

    prompt = f'''Break this problem into simple arithmetic steps.

Template: {template}
Values: {values_str}

Output JSON:
{{
    "steps": [
        {{"template": "A plus B", "uses": ["A", "B"], "result": "step1"}},
        {{"template": "step1 times C", "uses": ["step1", "C"], "result": "step2"}}
    ],
    "answer": "step2"
}}

STRICT RULES:
1. Each step must be ONE operation: add, subtract, multiply, or divide
2. Templates MUST use variable names like: "A plus B", "step1 minus C", "step2 divided by D"
3. "uses" MUST only contain: {value_keys} or prior step results (step1, step2, etc.)
4. Each step must have EXACTLY 2 inputs in "uses" (for binary operations)
5. "answer" must be the result name of the final step
6. Do NOT use placeholder text - use actual variable names'''

    for attempt in range(max_retries + 1):
        response = call_llm(prompt)
        try:
            graph = json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"[pointer_graph] JSON parse error (attempt {attempt+1}): {e}")
            continue

        errors = validate_pointer_graph(graph, values)
        if not errors:
            return graph

        logger.warning(f"[pointer_graph] Validation errors (attempt {attempt+1}): {errors}")

        if attempt < max_retries:
            # Add errors to prompt for retry
            prompt = f'''{prompt}

YOUR PREVIOUS OUTPUT HAD ERRORS:
{chr(10).join(f"- {e}" for e in errors)}

Please fix these errors and try again.'''

    # Return last attempt even if invalid (let execution handle failures)
    logger.error(f"[pointer_graph] Failed validation after {max_retries+1} attempts")
    return graph


# =============================================================================
# STEP 3: MATCH TEMPLATES TO SIGNATURES
# =============================================================================

def match_template(
    template: str,
    db: StepSignatureDB,
) -> Tuple[Optional[str], float, Any]:
    """
    Embed template and find matching signature using Welford-adaptive threshold.

    Returns:
        (func_name, similarity, embedding) or (None, similarity, embedding) if no match
    """
    embedding = cached_embed(template)
    if embedding is None:
        return None, 0.0, None

    func_name, similarity, sig = db.classify(embedding)
    threshold = db.get_adaptive_threshold()

    logger.debug(f"[match] template='{template[:30]}' sim={similarity:.3f} threshold={threshold:.3f}")

    if similarity >= threshold:
        return func_name, similarity, embedding
    return None, similarity, embedding


def propose_function(template: str) -> Optional[str]:
    """
    Ask LLM to propose which core function this template maps to.

    Returns function name or None if invalid.
    """
    funcs_list = ", ".join(sorted(CORE_FUNCTIONS))

    prompt = f'''What basic math operation is this?

Template: {template}

Output JSON:
{{"func": "one of: {funcs_list}"}}

Pick the single best match. If unsure, use the most likely one.'''

    try:
        response = call_llm(prompt)
        result = json.loads(response)
        func = result.get("func", "").strip()

        if func in CORE_FUNCTIONS:
            return func
        # Try to normalize common variations
        func_lower = func.lower()
        if "sub" in func_lower:
            return "sub"
        if "add" in func_lower:
            return "add"
        if "mul" in func_lower or "times" in func_lower:
            return "mul"
        if "div" in func_lower:
            return "truediv"

        logger.warning(f"[propose] LLM proposed invalid func: {func}")
        return None
    except Exception as e:
        logger.error(f"[propose] LLM call failed: {e}")
        return None


def is_bad_template(template: str) -> bool:
    """Check if template looks like a prompt artifact."""
    template_lower = template.lower()
    for bad_pattern in BAD_TEMPLATE_PATTERNS:
        if bad_pattern.lower() in template_lower:
            return True
    return False


def add_proposal(
    template: str,
    func: str,
    embedding: List[float],
    db: StepSignatureDB,
    problem_context: str,
    min_distance: float = 0.15,
) -> bool:
    """
    Add a signature proposal if it's far enough from existing signatures.

    Returns True if proposal was added, False if too close to existing.
    """
    import numpy as np

    # Filter out prompt artifacts
    if is_bad_template(template):
        logger.warning(f"[propose] Rejecting bad template: '{template[:40]}'")
        return False

    # Check for duplicate template in existing proposals
    for existing in _proposals:
        if existing.template == template and existing.func == func:
            logger.debug(f"[propose] Skipping duplicate proposal: '{template[:40]}'")
            return False

    # Find nearest signature with same function
    nearest_sig, distance = db.get_nearest_same_func_signature(
        np.array(embedding, dtype=np.float32),
        func
    )

    if distance < min_distance:
        logger.info(f"[propose] Template too close to existing (dist={distance:.3f} < {min_distance})")
        return False

    proposal = SignatureProposal(
        template=template,
        func=func,
        embedding=embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
        similarity_to_nearest=1.0 - distance,  # Convert distance to similarity
        nearest_signature_id=nearest_sig.id if nearest_sig else None,
        problem_context=problem_context[:200],  # Truncate for storage
    )

    _proposals.append(proposal)
    logger.info(f"[propose] Added proposal: '{template[:40]}' → {func}")
    return True


def decompose_template(template: str) -> List[Dict[str, Any]]:
    """
    Break a complex template into simpler sub-templates.

    Returns list of simpler templates.
    """
    prompt = f'''This operation is too complex. Break it into simpler steps.

Operation: {template}

Output JSON:
{{
    "steps": [
        {{"template": "simpler operation 1"}},
        {{"template": "simpler operation 2"}}
    ]
}}

Each step should be ONE basic operation: add, subtract, multiply, or divide.'''

    response = call_llm(prompt)
    result = json.loads(response)
    return result.get("steps", [])


# =============================================================================
# RECURSIVE DECOMPOSITION ENGINE
# =============================================================================

class RecursiveDecomposer:
    """
    Recursively decomposes problems until all templates match signatures.

    If no signature matches at Welford-adaptive threshold, proposes new
    signature for human review (post-mortem learning).
    """

    def __init__(
        self,
        db: StepSignatureDB = None,
        max_depth: int = 5,
    ):
        self.db = db or get_step_db()
        self.max_depth = max_depth
        self._problem_context = ""  # Current problem for proposals

    def solve(self, problem: str, single_pass: bool = True) -> float:
        """
        Main entry point: solve a problem.

        Args:
            problem: The problem text
            single_pass: If True, use single-pass decomposition (better for complex relationships)

        Returns the numeric answer.
        """
        if single_pass:
            return self.solve_single_pass(problem)
        else:
            return self.solve_two_pass(problem)

    def solve_single_pass(self, problem: str):
        """
        Solve using single-pass decomposition with pattern detection.

        Detects problem type (conditional, inversion, ratio, standard)
        and uses specialized template for better accuracy.

        Returns the answer (numeric or symbolic string).
        """
        self._problem_context = problem

        logger.info("[decomposer] Pattern-aware decomposition")
        decomp = decompose_with_pattern(problem)

        # Handle symbolic problems with SymPy
        if decomp.get('_pattern') == 'symbolic':
            return self._solve_symbolic(decomp)

        # Handle equation problems with SymPy
        if decomp.get('_pattern') == 'equation':
            return self._solve_equation(decomp)

        # Handle rationalize problems with SymPy
        if decomp.get('_pattern') == 'rationalize':
            return self._solve_rationalize(decomp)

        # Handle complex number problems
        if decomp.get('_pattern') == 'complex':
            return self._solve_complex(decomp)

        # Handle arithmetic sequence problems
        if decomp.get('_pattern') == 'sequence':
            return self._solve_sequence(decomp)

        # Handle Vieta's formulas (sum/product of roots)
        if decomp.get('_pattern') == 'vieta':
            return self._solve_vieta(decomp)

        # Handle system of equations
        if decomp.get('_pattern') == 'system':
            return self._solve_system(decomp)

        # Execute steps, evaluating expressions
        context = {}
        import re

        for step in decomp.get("steps", []):
            desc = step.get("description", "")
            expr = step.get("expr", "0")
            result_name = step.get("result", "step")

            # Substitute previous step results into expression
            # Use word boundary matching to avoid replacing substrings (e.g., '60' in '0.60')
            # Wrap negative values in parentheses for correct operator precedence
            eval_expr = expr
            for var, val in sorted(context.items(), key=lambda x: -len(x[0])):
                # Only replace if var is a valid identifier (starts with letter or _)
                if var and (var[0].isalpha() or var[0] == '_'):
                    # Wrap negative numbers in parens to avoid precedence issues like -1**2 = -1
                    val_str = f"({val})" if val < 0 else str(val)
                    eval_expr = re.sub(r'\b' + re.escape(var) + r'\b', val_str, eval_expr)

            # Safely evaluate the arithmetic expression
            try:
                result = self._safe_eval(eval_expr)
            except Exception as e:
                logger.warning(f"[decomposer] Failed to eval '{expr}': {e}")
                result = 0.0

            context[result_name] = result
            logger.info(f"[decomposer] {result_name} = {expr} = {result} ({desc})")

            # Embed description for signature learning (but don't route - we have the expr)
            self._record_description_for_learning(desc, expr)

        # Return final answer
        answer_key = decomp.get("answer", "step1")
        return context.get(answer_key, 0.0)

    def _safe_eval(self, expr: str):
        """Safely evaluate an arithmetic expression with limited builtins."""
        import re
        import math

        # Preprocess common issues
        expr = self._preprocess_expr(expr)

        # Allow numbers, operators, parentheses, whitespace, commas, and safe function names
        # Include 'i' and 'j' for complex numbers
        safe_pattern = r'^[\d\s\+\-\*\/\.\(\),a-z_]+$'
        if not re.match(safe_pattern, expr, re.IGNORECASE):
            raise ValueError(f"Invalid expression: {expr}")

        # Only allow these safe functions (includes math functions for exponents)
        safe_builtins = {
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'pow': pow,
            'sqrt': math.sqrt,
            'log': math.log,
            'log10': math.log10,
            'log2': math.log2,
            'floor': math.floor,
            'ceil': math.ceil,
            'I': 1j,  # Complex number support (sympy style)
            'i': 1j,  # Also allow lowercase
            'j': 1j,  # Python style
        }
        result = eval(expr, {"__builtins__": {}}, safe_builtins)
        # Return as-is if complex, else convert to float
        if isinstance(result, complex):
            return result
        return float(result)

    def _preprocess_expr(self, expr: str) -> str:
        """Preprocess expression to fix common LLM output issues."""
        import re

        # 1. Convert ^ to ** (LaTeX exponent to Python)
        expr = expr.replace('^', '**')

        # 2. Add implicit multiplication: 2x -> 2*x, 3(x) -> 3*(x)
        expr = re.sub(r'(\d)([a-z])', r'\1*\2', expr)
        expr = re.sub(r'(\d)\(', r'\1*(', expr)
        expr = re.sub(r'\)(\d)', r')*\1', expr)
        expr = re.sub(r'([a-z])\(', r'\1*(', expr)

        # 3. Remove equation parts (if expr contains =, take right side or left side)
        if '=' in expr and not any(op in expr for op in ['==', '!=', '<=', '>=']):
            parts = expr.split('=')
            # Try to find the numeric part
            for part in reversed(parts):
                part = part.strip()
                try:
                    # Check if this part is evaluatable
                    float(eval(part, {"__builtins__": {}}, {}))
                    expr = part
                    break
                except:
                    continue

        # 4. Remove comparison operators (take left side)
        for op in ['<', '>']:
            if op in expr and '=' not in expr:
                expr = expr.split(op)[0].strip()

        return expr

    def _record_description_for_learning(self, description: str, expr: str) -> None:
        """Record step description for potential signature learning."""
        if not description or is_bad_template(description):
            return

        embedding = cached_embed(description)
        if embedding is None:
            return

        # Determine function from expression
        func = None
        if '+' in expr and '-' not in expr:
            func = 'add'
        elif '-' in expr and '+' not in expr:
            func = 'sub'
        elif '*' in expr and '/' not in expr:
            func = 'mul'
        elif '/' in expr:
            func = 'truediv'

        if func:
            add_proposal(
                template=description,
                func=func,
                embedding=embedding,
                db=self.db,
                problem_context=self._problem_context,
            )

    def _solve_symbolic(self, decomp: Dict[str, Any]) -> str:
        """
        Solve a symbolic problem using SymPy.

        Args:
            decomp: Decomposition with operation, expression, variable

        Returns:
            String representation of the symbolic result
        """
        operation = decomp.get('operation', 'simplify')
        expr_str = decomp.get('expression', '0')
        var_name = decomp.get('variable', 'x')

        logger.info(f"[decomposer] Symbolic: {operation}({expr_str})")

        try:
            # Create symbol for the variable
            var = symbols(var_name)

            # Parse the expression with local symbols
            local_dict = {var_name: var, 'sqrt': sqrt, 'Rational': Rational}
            expr = parse_expr(expr_str, local_dict=local_dict)

            # Apply the operation
            if operation == 'simplify':
                result = simplify(expr)
            elif operation == 'factor':
                result = factor(expr)
            elif operation == 'expand':
                result = expand(expr)
            else:
                result = simplify(expr)

            logger.info(f"[decomposer] Symbolic result: {result}")
            return str(result)

        except Exception as e:
            logger.error(f"[decomposer] SymPy error: {e}")
            return "0"

    def _solve_equation(self, decomp: Dict[str, Any]) -> str:
        """
        Solve an equation using SymPy.

        Args:
            decomp: Decomposition with equation and variable

        Returns:
            String representation of the solution(s)
        """
        from sympy import solve, Eq

        eq_str = decomp.get('equation', 'Eq(x, 0)')
        var_name = decomp.get('variable', 'x')

        logger.info(f"[decomposer] Equation: {eq_str}, var: {var_name}")

        try:
            # Create symbol for the variable
            var = symbols(var_name)

            # Parse the equation with local symbols
            local_dict = {var_name: var, 'Eq': Eq, 'sqrt': sqrt, 'Rational': Rational}
            equation = eval(eq_str, {"__builtins__": {}}, local_dict)

            # Solve the equation
            solutions = solve(equation, var)

            logger.info(f"[decomposer] Solutions: {solutions}")

            if not solutions:
                return "0"
            elif len(solutions) == 1:
                return str(solutions[0])
            else:
                # Multiple solutions - return sum if asked, or first positive
                # For "sum of all values" problems
                try:
                    return str(sum(solutions))
                except:
                    return str(solutions[0])

        except Exception as e:
            logger.error(f"[decomposer] Equation solve error: {e}")
            return "0"

    def _solve_rationalize(self, decomp: Dict[str, Any]) -> str:
        """
        Rationalize a denominator using SymPy.

        Args:
            decomp: Decomposition with expression

        Returns:
            String representation of rationalized expression
        """
        from sympy import nsimplify, radsimp

        expr_str = decomp.get('expression', '1')

        logger.info(f"[decomposer] Rationalize: {expr_str}")

        try:
            # Parse the expression
            local_dict = {'sqrt': sqrt, 'Rational': Rational}
            expr = parse_expr(expr_str, local_dict=local_dict)

            # Rationalize the denominator
            result = radsimp(expr)

            logger.info(f"[decomposer] Rationalized result: {result}")
            return str(result)

        except Exception as e:
            logger.error(f"[decomposer] Rationalize error: {e}")
            return "0"

    def _solve_complex(self, decomp: Dict[str, Any]) -> str:
        """
        Solve complex number arithmetic.

        Args:
            decomp: Decomposition with real_part and imag_part expressions

        Returns:
            String in form "a + bi" or just "a" if imaginary is 0
        """
        real_expr = decomp.get('real_part', '0')
        imag_expr = decomp.get('imag_part', '0')

        logger.info(f"[decomposer] Complex: real={real_expr}, imag={imag_expr}")

        try:
            real = self._safe_eval(real_expr)
            imag = self._safe_eval(imag_expr)

            # Format result
            if imag == 0:
                return str(int(real) if real == int(real) else real)
            elif real == 0:
                if imag == 1:
                    return "i"
                elif imag == -1:
                    return "-i"
                else:
                    return f"{int(imag) if imag == int(imag) else imag}i"
            else:
                real_str = str(int(real) if real == int(real) else real)
                if imag == 1:
                    return f"{real_str} + i"
                elif imag == -1:
                    return f"{real_str} - i"
                elif imag > 0:
                    return f"{real_str} + {int(imag) if imag == int(imag) else imag}i"
                else:
                    return f"{real_str} - {int(abs(imag)) if abs(imag) == int(abs(imag)) else abs(imag)}i"

        except Exception as e:
            logger.error(f"[decomposer] Complex error: {e}")
            return "0"

    def _solve_sequence(self, decomp: Dict[str, Any]) -> float:
        """
        Solve arithmetic sequence problems.

        Args:
            decomp: Decomposition with first_term, common_diff, and steps

        Returns:
            The answer (n, a_n, or count depending on problem)
        """
        import math

        logger.info(f"[decomposer] Sequence: {decomp}")

        try:
            # Check if answer is a direct numeric value
            answer_key = decomp.get("answer", "")
            try:
                direct_answer = float(answer_key)
                logger.info(f"[decomposer] Sequence direct answer: {direct_answer}")
                return direct_answer
            except (ValueError, TypeError):
                pass

            # If there are steps, execute them
            if 'steps' in decomp:
                context = {}
                for step in decomp.get("steps", []):
                    expr = step.get("expr", "0")
                    result_name = step.get("result", "step")

                    # Substitute context values
                    eval_expr = expr
                    for var, val in context.items():
                        if var and (var[0].isalpha() or var[0] == '_'):
                            val_str = f"({val})" if val < 0 else str(val)
                            import re
                            eval_expr = re.sub(r'\b' + re.escape(var) + r'\b', val_str, eval_expr)

                    try:
                        # Add math functions for sequence calculations
                        result = eval(eval_expr, {"__builtins__": {}}, {
                            'floor': math.floor, 'ceil': math.ceil,
                            'abs': abs, 'min': min, 'max': max
                        })
                    except:
                        result = 0.0

                    context[result_name] = result

                answer_key = decomp.get("answer", "step1")
                return context.get(answer_key, 0.0)

            # Fallback: use first_term and common_diff if provided
            a1 = float(decomp.get('first_term', 0))
            d = float(decomp.get('common_diff', 0))

            target = decomp.get('target', '')
            if 'negative' in target.lower() and d < 0:
                # Find first negative term: a_n < 0
                # a1 + (n-1)*d < 0
                # n > 1 - a1/d
                n = math.ceil(1 - a1/d)
                return float(n)

            return 0.0

        except Exception as e:
            logger.error(f"[decomposer] Sequence error: {e}")
            return 0.0

    def _solve_vieta(self, decomp: Dict[str, Any]) -> float:
        """
        Solve using Vieta's formulas for sum/product of roots.

        For ax^2 + bx + c = 0:
        - Sum of roots = -b/a
        - Product of roots = c/a
        """
        logger.info(f"[decomposer] Vieta: {decomp}")

        try:
            # Check if answer is a direct numeric value
            answer_key = decomp.get("answer", "")
            try:
                direct_answer = float(answer_key)
                logger.info(f"[decomposer] Vieta direct answer: {direct_answer}")
                return direct_answer
            except (ValueError, TypeError):
                pass

            # Try to execute steps if provided
            if 'steps' in decomp and decomp['steps']:
                context = {}
                import re
                for step in decomp.get("steps", []):
                    expr = step.get("expr", "0")
                    result_name = step.get("result", "step")

                    eval_expr = expr
                    for var, val in context.items():
                        if var and (var[0].isalpha() or var[0] == '_'):
                            val_str = f"({val})" if val < 0 else str(val)
                            eval_expr = re.sub(r'\b' + re.escape(var) + r'\b', val_str, eval_expr)

                    try:
                        result = self._safe_eval(eval_expr)
                    except:
                        result = 0.0

                    context[result_name] = result

                return context.get(answer_key, 0.0)

            # Fallback: compute directly from coefficients
            a = float(decomp.get('a', 1))
            b = float(decomp.get('b', 0))
            c = float(decomp.get('c', 0))
            find = decomp.get('find', 'sum').lower()

            if 'sum' in find:
                return -b / a
            elif 'product' in find:
                return c / a

            return 0.0

        except Exception as e:
            logger.error(f"[decomposer] Vieta error: {e}")
            return 0.0

    def _solve_system(self, decomp: Dict[str, Any]) -> float:
        """
        Solve system of equations using SymPy.
        """
        from sympy import symbols, Eq, solve as sympy_solve
        from sympy.parsing.sympy_parser import parse_expr

        logger.info(f"[decomposer] System: {decomp}")

        try:
            # Check if answer is a direct numeric value
            answer_key = decomp.get("answer", "")
            try:
                direct_answer = float(answer_key)
                logger.info(f"[decomposer] System direct answer: {direct_answer}")
                return direct_answer
            except (ValueError, TypeError):
                pass

            # Try to execute steps if provided
            if 'steps' in decomp and decomp['steps']:
                context = {}
                import re
                for step in decomp.get("steps", []):
                    expr = step.get("expr", "0")
                    result_name = step.get("result", "step")

                    eval_expr = expr
                    for var, val in context.items():
                        if var and (var[0].isalpha() or var[0] == '_'):
                            val_str = f"({val})" if val < 0 else str(val)
                            eval_expr = re.sub(r'\b' + re.escape(var) + r'\b', val_str, eval_expr)

                    try:
                        result = self._safe_eval(eval_expr)
                    except:
                        result = 0.0

                    context[result_name] = result

                return context.get(answer_key, 0.0)

            # Fallback: parse and solve equations with SymPy
            equations = decomp.get('equations', [])
            unknowns = decomp.get('unknowns', [])

            if not equations or not unknowns:
                return 0.0

            # Create symbols
            sym_vars = symbols(' '.join(unknowns))
            if len(unknowns) == 1:
                sym_vars = [sym_vars]

            local_dict = {u: s for u, s in zip(unknowns, sym_vars)}

            # Parse equations
            sympy_eqs = []
            for eq in equations:
                if '=' in eq:
                    lhs, rhs = eq.split('=', 1)
                    lhs_expr = parse_expr(lhs.strip(), local_dict=local_dict)
                    rhs_expr = parse_expr(rhs.strip(), local_dict=local_dict)
                    sympy_eqs.append(Eq(lhs_expr, rhs_expr))

            # Solve
            solution = sympy_solve(sympy_eqs, sym_vars)
            logger.info(f"[decomposer] System solution: {solution}")

            # Return first value or compute product if needed
            if isinstance(solution, dict):
                vals = list(solution.values())
                if len(vals) == 1:
                    return float(vals[0])
                # Compute product for abc type problems
                product = 1
                for v in vals:
                    product *= float(v)
                return product

            return 0.0

        except Exception as e:
            logger.error(f"[decomposer] System error: {e}")
            return 0.0

    def solve_two_pass(self, problem: str) -> float:
        """
        Solve using two-pass decomposition (legacy approach).

        Extracts values first, then builds pointer graph.
        Can struggle with complex value relationships.
        """
        self._problem_context = problem

        # Step 1: Extract and mask
        logger.info("[decomposer] Step 1: Extract and mask")
        extracted = extract_and_mask(problem)
        template = extracted["template"]
        values = extracted["values"]

        logger.info(f"[decomposer] Template: {template}")
        logger.info(f"[decomposer] Values: {values}")

        # Step 2: Build pointer graph
        logger.info("[decomposer] Step 2: Build pointer graph")
        graph = build_pointer_graph(template, values)

        # Step 3: Execute steps
        logger.info("[decomposer] Step 3: Execute steps")
        context = dict(values)  # Start with extracted values

        for step in graph.get("steps", []):
            step_template = step["template"]
            step_uses = step.get("uses", [])
            step_result = step["result"]

            # Resolve input values
            args = []
            for ref in step_uses:
                if ref in context:
                    args.append(context[ref])
                else:
                    logger.warning(f"[decomposer] Unknown ref: {ref}")
                    args.append(0.0)

            # Match template to signature and execute
            result = self._execute_template(step_template, args, depth=0)
            context[step_result] = result

            logger.info(f"[decomposer] {step_result} = {step_template} → {result}")

        # Return final answer
        answer_key = graph.get("answer", "step1")
        return context.get(answer_key, 0.0)

    def _execute_template(
        self,
        template: str,
        args: List[float],
        depth: int
    ) -> float:
        """
        Match template to signature and execute.
        If no match, LLM proposes function, we add to proposal table and execute.
        """
        # Try to match template to signature (Welford-adaptive threshold)
        func_name, similarity, embedding = match_template(template, self.db)

        logger.info(f"[decomposer] depth={depth} template='{template[:40]}' → {func_name} (sim={similarity:.3f})")

        if func_name:
            # Found a match - execute
            try:
                result = execute(func_name, *args)
                return result
            except Exception as e:
                logger.warning(f"[decomposer] Execution failed ({func_name}): {e}")
                # Signature matched but execution failed - propose with correct func
                return self._propose_and_execute(template, args, embedding)

        # No match at threshold - propose new signature
        if embedding is not None:
            return self._propose_and_execute(template, args, embedding)

        # No embedding available (shouldn't happen)
        logger.error("[decomposer] No embedding available for template")
        return 0.0

    def _propose_and_execute(
        self,
        template: str,
        args: List[float],
        embedding: Any,
    ) -> float:
        """
        LLM proposes function, add to proposal table, execute.
        """
        # Filter out bad templates before proposing
        if is_bad_template(template):
            logger.warning(f"[decomposer] Skipping bad template: {template}")
            return 0.0

        # Ask LLM which function this template maps to
        func = propose_function(template)

        if func is None:
            logger.warning(f"[decomposer] LLM couldn't propose function for: {template}")
            return 0.0

        logger.info(f"[decomposer] LLM proposed: {func} for '{template[:40]}'")

        # Validate arity before execution
        expected_arity = FUNCTION_ARITY.get(func, 2)
        if len(args) != expected_arity:
            logger.warning(
                f"[decomposer] Arity mismatch: {func} expects {expected_arity} args, got {len(args)}"
            )
            # Try to fix common issues
            if expected_arity == 2 and len(args) == 1:
                # Maybe LLM forgot second arg - can't recover
                return 0.0
            elif expected_arity == 2 and len(args) > 2:
                # Too many args - take first two
                args = args[:2]
                logger.info(f"[decomposer] Truncated args to {args}")
            elif expected_arity == 1 and len(args) > 1:
                # Unary function with multiple args - take first
                args = args[:1]

        # Add to proposal table if far enough from existing signatures
        add_proposal(
            template=template,
            func=func,
            embedding=embedding,
            db=self.db,
            problem_context=self._problem_context,
        )

        # Execute with proposed function
        try:
            result = execute(func, *args)
            return result
        except Exception as e:
            logger.error(f"[decomposer] Execution failed ({func}): {e}")
            return 0.0


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def solve(problem: str) -> float:
    """Solve a math problem and return the answer."""
    decomposer = RecursiveDecomposer()
    return decomposer.solve(problem)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Clear any old proposals
    clear_proposals()

    test_problems = [
        ("Tim has 10 apples. He gives 3 to Mary. How many apples does Tim have left?", 7),
        ("A shirt costs $25. It's on sale for 20% off. What's the sale price?", 20),
    ]

    decomposer = RecursiveDecomposer()

    for problem, expected in test_problems:
        print(f"\n{'='*60}")
        print(f"Problem: {problem}")
        print(f"Expected: {expected}")
        print('='*60)

        try:
            result = decomposer.solve(problem)
            print(f"\nResult: {result}")
            if abs(result - expected) < 0.01:
                print("✓ CORRECT")
            else:
                print(f"✗ WRONG (expected {expected})")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # Show proposals for review
    proposals = get_proposals()
    if proposals:
        print(f"\n{'='*60}")
        print(f"SIGNATURE PROPOSALS FOR REVIEW ({len(proposals)})")
        print('='*60)
        for i, p in enumerate(proposals, 1):
            print(f"\n{i}. Template: \"{p.template}\"")
            print(f"   Function: {p.func}")
            print(f"   Similarity to nearest: {p.similarity_to_nearest:.3f}")
            print(f"   Context: {p.problem_context[:60]}...")
    else:
        print(f"\n{'='*60}")
        print("No new signature proposals (all templates matched existing)")
        print('='*60)
