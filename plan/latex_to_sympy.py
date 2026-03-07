"""
LaTeX-to-SymPy Converter

Converts LaTeX mathematical expressions to SymPy objects for execution.
This is DIFFERENT from the text preprocessor - produces executable objects, not strings.

C3 extracts: "3\sqrt{13}"  (raw string)
Converter:    3*sqrt(13)    (SymPy expression)
Template:     simplify(...)  → executes correctly
"""

import re
import sympy
from sympy import (
    Symbol, Integer, Float, Rational, sqrt, pi, E, I,
    sin, cos, tan, log, exp, Abs, floor, ceiling,
    oo, zoo, nan
)
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication
from typing import Union, List, Tuple, Optional, Any


class LaTeXConverter:
    """Convert LaTeX strings to SymPy objects."""

    def __init__(self):
        # Transformation rules applied in order
        self.transformations = standard_transformations + (implicit_multiplication,)

    def convert(self, latex: str) -> Any:
        """
        Convert LaTeX string to SymPy object.
        Returns None if conversion fails.
        """
        if not latex or not isinstance(latex, str):
            return None

        try:
            # Clean and normalize
            expr = self._clean_latex(latex)

            # Try conversion pipeline
            result = self._convert_pipeline(expr)

            return result

        except Exception:
            return None

    def _clean_latex(self, latex: str) -> str:
        """Remove LaTeX formatting that doesn't affect math."""
        s = latex.strip()

        # Remove display math delimiters
        s = re.sub(r'^\$+|\$+$', '', s)
        s = re.sub(r'^\\[\[\]]|\\[\[\]]$', '', s)

        # Remove \left \right
        s = s.replace('\\left', '').replace('\\right', '')

        # Remove \displaystyle, \textstyle, etc.
        s = re.sub(r'\\(display|text|script|scriptscript)style', '', s)

        # Remove \, \; \: \! (spacing)
        s = re.sub(r'\\[,;:!]', ' ', s)

        # Remove \text{...} but keep content
        s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)

        # Remove \mathrm, \mathbf, etc.
        s = re.sub(r'\\math(rm|bf|it|sf|tt|cal)\{([^}]*)\}', r'\2', s)

        # Normalize whitespace
        s = re.sub(r'\s+', ' ', s).strip()

        return s

    def _convert_pipeline(self, expr: str) -> Any:
        """Try conversion methods in order of specificity."""

        # 1. Check for special values first
        result = self._convert_special(expr)
        if result is not None:
            return result

        # 2. Check for structured types (tuples, sets, equations)
        result = self._convert_structured(expr)
        if result is not None:
            return result

        # 3. Convert LaTeX math notation to SymPy-parseable form
        converted = self._latex_to_parseable(expr)

        # 4. Try sympy parsing
        result = self._parse_sympy(converted)
        if result is not None:
            return result

        # 5. Fallback: try original expression
        result = self._parse_sympy(expr)
        return result

    def _convert_special(self, expr: str) -> Optional[Any]:
        """Handle special mathematical constants and values."""
        expr_lower = expr.lower().strip()

        # Infinity
        if expr_lower in ['\\infty', 'infty', 'infinity', '∞']:
            return oo
        if expr_lower in ['-\\infty', '-infty', '-infinity', '-∞']:
            return -oo

        # Pi
        if expr_lower in ['\\pi', 'pi', 'π']:
            return pi
        if expr_lower in ['-\\pi', '-pi', '-π']:
            return -pi

        # Euler's number
        if expr_lower in ['e', '\\e']:
            return E

        # Imaginary unit
        if expr_lower in ['i', '\\i']:
            return I

        # Undefined/NaN
        if expr_lower in ['undefined', 'undef', 'nan']:
            return nan

        return None

    def _convert_structured(self, expr: str) -> Optional[Any]:
        """Handle tuples, sets, equations, lists."""

        # Tuple: (a, b) or \left(a, b\right)
        tuple_match = re.match(r'^\(([^)]+)\)$', expr)
        if tuple_match:
            inner = tuple_match.group(1)
            elements = self._split_elements(inner)
            converted = [self.convert(e) for e in elements]
            if all(c is not None for c in converted):
                return tuple(converted)

        # Set: {a, b, c} or \{a, b, c\}
        # First normalize: \{ → { and \} → }
        expr_normalized = expr.replace('\\{', '{').replace('\\}', '}')
        set_match = re.match(r'^\{(.+)\}$', expr_normalized)
        if set_match and ',' in expr_normalized:
            inner = set_match.group(1)
            elements = self._split_elements(inner)
            converted = [self.convert(e) for e in elements]
            if all(c is not None for c in converted):
                return set(converted)

        # Equation: x = 5 or a = b
        eq_match = re.match(r'^([a-zA-Z_]\w*)\s*=\s*(.+)$', expr)
        if eq_match:
            var_name = eq_match.group(1)
            value_str = eq_match.group(2)
            value = self.convert(value_str)
            if value is not None:
                # Return just the value (common use case)
                return value

        # Comma-separated values without braces: 1, 2, 3 or -2, 5
        if ',' in expr and not re.search(r'[{}()\[\]]', expr):
            elements = self._split_elements(expr)
            if len(elements) > 1:
                converted = [self.convert(e) for e in elements]
                if all(c is not None for c in converted):
                    return tuple(converted)

        return None

    def _split_elements(self, s: str) -> List[str]:
        """Split comma-separated elements, respecting nested structures."""
        elements = []
        depth = 0
        current = []

        for char in s:
            if char in '({[':
                depth += 1
                current.append(char)
            elif char in ')}]':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                elements.append(''.join(current).strip())
                current = []
            else:
                current.append(char)

        if current:
            elements.append(''.join(current).strip())

        return [e for e in elements if e]

    def _latex_to_parseable(self, expr: str) -> str:
        """Convert LaTeX notation to SymPy-parseable string."""
        s = expr

        # Degree symbol FIRST (before power handling): 90^\circ → 90
        s = re.sub(r'\^\\circ', '', s)
        s = re.sub(r'\\circ', '', s)
        s = re.sub(r'°', '', s)
        s = re.sub(r'\\degree', '', s)

        # Fractions: \frac{a}{b} → (a)/(b)
        # Handle nested fractions by iterating
        for _ in range(5):  # Max nesting depth
            new_s = re.sub(
                r'\\c?frac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
                r'((\1)/(\2))',
                s
            )
            if new_s == s:
                break
            s = new_s

        # Square root: \sqrt{x} → sqrt(x)
        s = re.sub(r'\\sqrt\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'sqrt(\1)', s)

        # Nth root: \sqrt[n]{x} → x**(1/n)
        s = re.sub(r'\\sqrt\[(\d+)\]\{([^{}]*)\}', r'((\2)**(1/\1))', s)

        # Powers: x^{n} → x**(n), x^n → x**n
        s = re.sub(r'\^{([^{}]+)}', r'**(\1)', s)
        s = re.sub(r'\^(\d+)', r'**\1', s)
        s = re.sub(r'\^([a-zA-Z])', r'**\1', s)

        # Subscripts: remove for now (usually variable naming)
        s = re.sub(r'_\{([^{}]+)\}', r'', s)
        s = re.sub(r'_(\d+)', r'', s)

        # Greek letters
        greek = {
            '\\alpha': 'alpha', '\\beta': 'beta', '\\gamma': 'gamma',
            '\\delta': 'delta', '\\epsilon': 'epsilon', '\\theta': 'theta',
            '\\lambda': 'lambda_', '\\mu': 'mu', '\\nu': 'nu',
            '\\pi': 'pi', '\\rho': 'rho', '\\sigma': 'sigma',
            '\\tau': 'tau', '\\phi': 'phi', '\\omega': 'omega',
        }
        for latex_greek, sympy_greek in greek.items():
            s = s.replace(latex_greek, sympy_greek)

        # Trig functions
        trig = ['sin', 'cos', 'tan', 'cot', 'sec', 'csc',
                'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh']
        for fn in trig:
            s = re.sub(rf'\\{fn}\s*\{{([^{{}}]+)\}}', rf'{fn}(\1)', s)
            s = re.sub(rf'\\{fn}\s+', f'{fn}', s)

        # Log and exp
        s = re.sub(r'\\ln\s*\{([^{}]+)\}', r'log(\1)', s)
        s = re.sub(r'\\ln\s+', 'log', s)
        s = re.sub(r'\\log\s*\{([^{}]+)\}', r'log(\1)', s)
        s = re.sub(r'\\exp\s*\{([^{}]+)\}', r'exp(\1)', s)

        # Absolute value: |x| or \|x\| → Abs(x)
        s = re.sub(r'\|([^|]+)\|', r'Abs(\1)', s)
        s = re.sub(r'\\lvert([^\\]+)\\rvert', r'Abs(\1)', s)

        # Floor and ceiling
        s = re.sub(r'\\lfloor([^\\]+)\\rfloor', r'floor(\1)', s)
        s = re.sub(r'\\lceil([^\\]+)\\rceil', r'ceiling(\1)', s)

        # Multiplication: \cdot, \times → *
        s = s.replace('\\cdot', '*').replace('\\times', '*')

        # Division: \div → /
        s = s.replace('\\div', '/')

        # Plus/minus: \pm → + (take positive)
        s = s.replace('\\pm', '+').replace('\\mp', '-')

        # Degree symbol: remove (value is the number)
        s = re.sub(r'\\?°|\\circ|\\degree', '', s)

        # Remove remaining backslashes before letters
        s = re.sub(r'\\([a-zA-Z]+)', r'\1', s)

        # Clean up spacing
        s = re.sub(r'\s+', ' ', s).strip()

        # Handle implicit multiplication: 2x → 2*x, 3(x+1) → 3*(x+1)
        s = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', s)
        s = re.sub(r'(\))(\d)', r'\1*\2', s)
        s = re.sub(r'(\))([a-zA-Z(])', r'\1*\2', s)

        return s

    def _parse_sympy(self, expr: str) -> Optional[Any]:
        """Parse expression using SymPy."""
        if not expr:
            return None

        try:
            # Try parse_expr with transformations
            result = parse_expr(
                expr,
                transformations=self.transformations,
                local_dict={
                    'pi': pi, 'e': E, 'i': I, 'I': I,
                    'sqrt': sqrt, 'Abs': Abs,
                    'sin': sin, 'cos': cos, 'tan': tan,
                    'log': log, 'exp': exp,
                    'floor': floor, 'ceiling': ceiling,
                    'oo': oo, 'inf': oo,
                }
            )
            return result
        except:
            pass

        try:
            # Fallback: sympify
            result = sympy.sympify(expr)
            return result
        except:
            pass

        return None


# Module-level converter instance
_converter = LaTeXConverter()


def latex_to_sympy(latex: str) -> Any:
    """Convert LaTeX string to SymPy object."""
    return _converter.convert(latex)


def extract_operands_from_latex(text: str) -> List[Any]:
    """
    Extract operands from text containing LaTeX.
    Returns list of SymPy objects.
    """
    operands = []

    # Pattern 1: LaTeX fractions \frac{a}{b}
    for match in re.finditer(r'\\c?frac\{([^{}]+)\}\{([^{}]+)\}', text):
        result = latex_to_sympy(match.group(0))
        if result is not None:
            operands.append(result)

    # Pattern 2: Square roots \sqrt{...} or n\sqrt{...}
    for match in re.finditer(r'(\d*)\s*\\sqrt\{([^{}]+)\}', text):
        coef = match.group(1)
        inner = match.group(2)
        if coef:
            expr = f'{coef}*sqrt({inner})'
        else:
            expr = f'sqrt({inner})'
        result = latex_to_sympy(expr)
        if result is not None:
            operands.append(result)

    # Pattern 3: Powers like x^2, x^{n}
    for match in re.finditer(r'(\d+)\^{?(\d+)}?', text):
        base, exp = match.groups()
        result = Integer(int(base)) ** Integer(int(exp))
        operands.append(result)

    # Pattern 4: Decimal numbers
    for match in re.finditer(r'(?<![/\d])-?\d+\.\d+', text):
        try:
            operands.append(Float(match.group()))
        except:
            pass

    # Pattern 5: Plain fractions a/b (not in LaTeX)
    for match in re.finditer(r'(?<![.\d])(-?\d+)/(-?\d+)(?![.\d])', text):
        try:
            num, den = int(match.group(1)), int(match.group(2))
            if den != 0:
                operands.append(Rational(num, den))
        except:
            pass

    # Pattern 6: Integers (not already captured)
    for match in re.finditer(r'(?<![/.\d\w])-?\d+(?![/.\d\w])', text):
        try:
            val = int(match.group())
            # Avoid duplicates
            if Integer(val) not in operands and Rational(val) not in operands:
                operands.append(Integer(val))
        except:
            pass

    # Pattern 7: Pi
    if re.search(r'\\pi|π', text):
        operands.append(pi)

    # Pattern 8: Variables (single letters, excluding common words)
    for match in re.finditer(r'\b([a-zA-Z])\b', text):
        var = match.group(1)
        if var.lower() not in ['a', 'i', 'e', 'o']:  # Skip common words
            sym = Symbol(var)
            if sym not in operands:
                operands.append(sym)

    return operands


# Test suite
def test_converter():
    """Test the LaTeX converter."""
    converter = LaTeXConverter()

    test_cases = [
        # Fractions
        (r'\frac{1}{3}', Rational(1, 3)),
        (r'\frac{14}{3}', Rational(14, 3)),
        (r'\cfrac{-5}{7}', Rational(-5, 7)),

        # Square roots
        (r'\sqrt{13}', sqrt(13)),
        (r'3\sqrt{13}', 3*sqrt(13)),
        (r'\sqrt{2}', sqrt(2)),

        # Powers
        ('2^3', Integer(8)),
        ('x^2', Symbol('x')**2),
        (r'x^{10}', Symbol('x')**10),

        # Constants
        (r'\pi', pi),
        (r'-\pi', -pi),
        ('e', E),

        # Angles (just extract number)
        (r'90^\circ', Integer(90)),
        (r'45\degree', Integer(45)),

        # Tuples
        ('(3, 4)', (Integer(3), Integer(4))),
        (r'(1, \frac{1}{2})', (Integer(1), Rational(1, 2))),

        # Sets
        (r'\{1, 2, 3\}', {Integer(1), Integer(2), Integer(3)}),

        # Equations (extract value)
        ('x = 5', Integer(5)),
        ('n = 10', Integer(10)),

        # Comma-separated
        ('1, -2', (Integer(1), Integer(-2))),
        ('-2, 5', (Integer(-2), Integer(5))),

        # Complex expressions
        (r'\frac{\pi}{2}', pi/2),
        (r'\frac{3\pi}{4}', 3*pi/4),
        (r'\sqrt{\frac{1}{2}}', sqrt(Rational(1, 2))),
    ]

    print("Testing LaTeX Converter")
    print("="*60)

    passed = 0
    failed = 0

    for latex, expected in test_cases:
        result = converter.convert(latex)

        # Compare (handle sets specially)
        if isinstance(expected, set):
            match = isinstance(result, set) and result == expected
        elif isinstance(expected, tuple):
            match = isinstance(result, tuple) and result == expected
        else:
            try:
                match = sympy.simplify(result - expected) == 0
            except:
                match = result == expected

        status = "✓" if match else "✗"
        if match:
            passed += 1
        else:
            failed += 1

        print(f"{status} {latex:30s} → {result} (expected {expected})")

    print(f"\nPassed: {passed}/{passed+failed}")
    return passed, failed


if __name__ == "__main__":
    test_converter()
