"""
Factor Graph v2: Type-Based Dispatch

Operation type is an OUTPUT, not an INPUT. The factor graph discovers
operations through:
1. Type-based filtering (what templates CAN apply to these operands?)
2. Execution filtering (which templates actually run without crashing?)
3. Result chaining (use segment N-1's result in segment N)
4. Neighbor consistency (does output type match downstream expectations?)

No keyword heuristics. No learned classifier. Just type dispatch + execution.
"""

import re
import sympy
from sympy import Symbol, Integer, Float, Rational, Eq, sqrt
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

# Import the LaTeX converter
from latex_to_sympy import latex_to_sympy, extract_operands_from_latex, LaTeXConverter


# ================================================================
# TEMPLATE LIBRARY (executable functions)
# ================================================================

class Templates:
    """All templates as static methods."""

    # ARITHMETIC (2 numeric operands)
    @staticmethod
    def arithmetic_add(a, b):
        return a + b

    @staticmethod
    def arithmetic_sub(a, b):
        return a - b

    @staticmethod
    def arithmetic_mul(a, b):
        return a * b

    @staticmethod
    def arithmetic_div(a, b):
        if b == 0:
            raise ValueError("Division by zero")
        return a / b

    @staticmethod
    def arithmetic_pow(a, b):
        return a ** b

    # SINGLE NUMERIC
    @staticmethod
    def compute_sqrt(a):
        return sqrt(a)

    @staticmethod
    def compute_floor(a):
        return sympy.floor(a)

    @staticmethod
    def compute_ceiling(a):
        return sympy.ceiling(a)

    @staticmethod
    def compute_abs(a):
        return sympy.Abs(a)

    @staticmethod
    def identity(a):
        """Pass through - useful for setup steps."""
        return a

    # EQUATION SOLVING
    @staticmethod
    def solve_equation(equation, variable):
        """Solve equation for variable."""
        if isinstance(equation, Eq):
            solutions = sympy.solve(equation, variable)
        else:
            solutions = sympy.solve(equation, variable)
        return solutions[0] if solutions else None

    # EXPRESSION MANIPULATION
    @staticmethod
    def simplify(expr):
        return sympy.simplify(expr)

    @staticmethod
    def expand(expr):
        return sympy.expand(expr)

    @staticmethod
    def factor(expr):
        return sympy.factor(expr)

    # SUBSTITUTION
    @staticmethod
    def substitute(expr, var, value):
        return expr.subs(var, value)

    @staticmethod
    def evaluate(expr):
        """Try to evaluate to a number."""
        try:
            return sympy.nsimplify(expr)
        except:
            return expr


# Template registry with metadata
TEMPLATE_REGISTRY = {
    # Arithmetic (requires 2 numeric)
    "arithmetic_add": {"fn": Templates.arithmetic_add, "n_args": 2, "requires": "numeric"},
    "arithmetic_sub": {"fn": Templates.arithmetic_sub, "n_args": 2, "requires": "numeric"},
    "arithmetic_mul": {"fn": Templates.arithmetic_mul, "n_args": 2, "requires": "numeric"},
    "arithmetic_div": {"fn": Templates.arithmetic_div, "n_args": 2, "requires": "numeric"},
    "arithmetic_pow": {"fn": Templates.arithmetic_pow, "n_args": 2, "requires": "numeric"},

    # Single numeric
    "compute_sqrt": {"fn": Templates.compute_sqrt, "n_args": 1, "requires": "numeric"},
    "compute_floor": {"fn": Templates.compute_floor, "n_args": 1, "requires": "numeric"},
    "compute_ceiling": {"fn": Templates.compute_ceiling, "n_args": 1, "requires": "numeric"},
    "compute_abs": {"fn": Templates.compute_abs, "n_args": 1, "requires": "numeric"},
    "identity": {"fn": Templates.identity, "n_args": 1, "requires": "any"},

    # Expression (requires symbolic)
    "simplify": {"fn": Templates.simplify, "n_args": 1, "requires": "expression"},
    "expand": {"fn": Templates.expand, "n_args": 1, "requires": "expression"},
    "factor": {"fn": Templates.factor, "n_args": 1, "requires": "expression"},
    "evaluate": {"fn": Templates.evaluate, "n_args": 1, "requires": "any"},

    # Equation solving (requires equation + symbol)
    "solve_equation": {"fn": Templates.solve_equation, "n_args": 2, "requires": "equation"},

    # Substitution (requires expression + symbol + value)
    "substitute": {"fn": Templates.substitute, "n_args": 3, "requires": "expression"},
}


# ================================================================
# OPERAND EXTRACTION
# ================================================================

def extract_operands_from_text(text: str) -> List[Any]:
    """Extract operands from text using the LaTeX converter."""
    # Use the new LaTeX-aware extractor
    return extract_operands_from_latex(text)


def classify_operands(operands: List[Any]) -> Dict[str, Any]:
    """Classify operands by type."""
    info = {
        "all_numeric": True,
        "has_symbol": False,
        "has_equation": False,
        "has_expression": False,
        "numeric_count": 0,
        "symbol_count": 0,
        "types": [],
    }

    for op in operands:
        if isinstance(op, Symbol):
            info["has_symbol"] = True
            info["all_numeric"] = False
            info["symbol_count"] += 1
            info["types"].append("symbol")
        elif isinstance(op, Eq):
            info["has_equation"] = True
            info["all_numeric"] = False
            info["types"].append("equation")
        elif hasattr(op, 'is_number') and op.is_number:
            info["numeric_count"] += 1
            info["types"].append("number")
        elif hasattr(op, 'free_symbols') and op.free_symbols:
            info["has_expression"] = True
            info["all_numeric"] = False
            info["types"].append("expression")
        else:
            info["types"].append("unknown")

    return info


# ================================================================
# TYPE-BASED TEMPLATE DISPATCH
# ================================================================

def get_candidate_templates(operands: List[Any], has_backref: bool = False) -> List[str]:
    """
    Filter templates by operand types.
    This is type dispatch, not heuristics.
    """
    info = classify_operands(operands)
    candidates = []

    # Equation present → solve
    if info["has_equation"]:
        candidates.extend(["solve_equation"])

    # Expression with free symbols → expression manipulation
    if info["has_expression"] or (info["has_symbol"] and not info["all_numeric"]):
        candidates.extend(["simplify", "expand", "factor", "evaluate"])
        if info["numeric_count"] > 0:
            candidates.append("substitute")

    # All numeric, 2 operands → arithmetic
    if info["all_numeric"] and info["numeric_count"] >= 2:
        # Order by likelihood: mul > add > sub > div > pow
        candidates.extend([
            "arithmetic_mul", "arithmetic_add",
            "arithmetic_sub", "arithmetic_div", "arithmetic_pow"
        ])

    # All numeric, 1 operand → single-value operations
    if info["all_numeric"] and info["numeric_count"] == 1:
        candidates.extend([
            "identity", "compute_sqrt", "compute_abs",
            "compute_floor", "compute_ceiling"
        ])

    # With backref, try using previous result
    if has_backref:
        if info["numeric_count"] >= 1:
            # Previous result + current number → arithmetic
            candidates.extend([
                "arithmetic_mul", "arithmetic_add",
                "arithmetic_sub", "arithmetic_div"
            ])
        candidates.append("evaluate")

    # Fallbacks
    if not candidates:
        candidates = ["identity", "evaluate"]

    # Remove duplicates preserving order
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return unique


def try_template(template_name: str, operands: List[Any],
                 previous_result: Any = None) -> Tuple[bool, Any]:
    """
    Try executing a template with given operands.
    Returns (success, result).
    """
    if template_name not in TEMPLATE_REGISTRY:
        return False, None

    template = TEMPLATE_REGISTRY[template_name]
    fn = template["fn"]
    n_args = template["n_args"]

    # Prepare arguments
    args = []

    # Filter to appropriate types
    numbers = [op for op in operands if hasattr(op, 'is_number') and op.is_number]
    symbols = [op for op in operands if isinstance(op, Symbol)]
    expressions = [op for op in operands
                   if hasattr(op, 'free_symbols') and op.free_symbols and not isinstance(op, Symbol)]

    # Build args based on template requirements
    requires = template["requires"]

    if requires == "numeric":
        if previous_result is not None and hasattr(previous_result, 'is_number') and previous_result.is_number:
            args = [previous_result] + numbers[:n_args-1]
        else:
            args = numbers[:n_args]

    elif requires == "equation":
        # Need equation + symbol
        equations = [op for op in operands if isinstance(op, Eq)]
        if equations and symbols:
            args = [equations[0], symbols[0]]
        elif expressions and symbols:
            # Treat expression as implicit equation = 0
            args = [expressions[0], symbols[0]]

    elif requires == "expression":
        if n_args == 1:
            if expressions:
                args = [expressions[0]]
            elif previous_result is not None:
                args = [previous_result]
        elif n_args == 3:  # substitute
            if expressions and symbols and numbers:
                args = [expressions[0], symbols[0], numbers[0]]
            elif previous_result is not None and symbols and numbers:
                args = [previous_result, symbols[0], numbers[0]]

    else:  # "any"
        if n_args == 1:
            if numbers:
                args = [numbers[0]]
            elif expressions:
                args = [expressions[0]]
            elif previous_result is not None:
                args = [previous_result]

    # Check we have enough args
    if len(args) < n_args:
        return False, None

    # Try execution
    try:
        result = fn(*args[:n_args])
        if result is not None:
            return True, result
        return False, None
    except Exception:
        return False, None


# ================================================================
# SEGMENT PROCESSING
# ================================================================

@dataclass
class Segment:
    """A reasoning segment."""
    idx: int
    text: str
    operands: List[Any] = field(default_factory=list)
    selected_template: Optional[str] = None
    result: Any = None
    success: bool = False


def process_segment(segment: Segment, previous_result: Any = None) -> Segment:
    """
    Process a segment: extract operands, filter templates by type,
    try execution, select first that works.
    """
    # Extract operands from text
    segment.operands = extract_operands_from_text(segment.text)

    # Get candidate templates based on operand types
    has_backref = previous_result is not None
    candidates = get_candidate_templates(segment.operands, has_backref)

    # Try each candidate
    for template_name in candidates:
        success, result = try_template(template_name, segment.operands, previous_result)
        if success:
            segment.selected_template = template_name
            segment.result = result
            segment.success = True
            return segment

    # Nothing worked
    segment.success = False
    return segment


# ================================================================
# FULL PIPELINE
# ================================================================

def run_pipeline(windows: List[Dict]) -> Tuple[Any, List[Segment]]:
    """
    Run type-based dispatch pipeline.
    Returns (final_result, segments).
    """
    segments = []
    previous_result = None

    for i, window in enumerate(windows):
        segment = Segment(idx=i, text=window.get("text", ""))
        segment = process_segment(segment, previous_result)

        # Chain result to next segment
        if segment.success and segment.result is not None:
            previous_result = segment.result

        segments.append(segment)

    # Final result is the last successful segment's result
    final_result = None
    for seg in reversed(segments):
        if seg.success and seg.result is not None:
            final_result = seg.result
            break

    return final_result, segments


def print_trace(segments: List[Segment], final_result: Any):
    """Print execution trace."""
    print("\n" + "="*60)
    print("EXECUTION TRACE")
    print("="*60)

    for seg in segments:
        print(f"\nSegment {seg.idx}: {seg.text[:50]}...")
        print(f"  Operands: {seg.operands[:4]}")
        print(f"  Template: {seg.selected_template}")
        print(f"  Result: {seg.result}")
        print(f"  Success: {seg.success}")

    print(f"\nFinal result: {final_result}")


# ================================================================
# EVALUATION
# ================================================================

def evaluate_answer(result: Any, ground_truth: str) -> Dict[str, Any]:
    """Compare result to ground truth using LaTeX converter."""
    if result is None:
        return {"correct": False, "status": "no_result"}

    try:
        # Strip $ delimiters
        gt = ground_truth.strip()
        gt = re.sub(r'^\$+|\$+$', '', gt)

        # Use LaTeX converter to parse ground truth
        expected = latex_to_sympy(gt)

        if expected is None:
            # Fallback to sympify
            gt_clean = gt.replace('\\frac{', '(').replace('}{', ')/(').replace('}', ')')
            gt_clean = gt_clean.replace('\\cdot', '*').replace('\\times', '*')
            expected = sympy.sympify(gt_clean)

        # Handle tuple/set comparisons
        if isinstance(expected, (tuple, set)):
            if isinstance(result, (tuple, set)):
                if set(expected) == set(result) if isinstance(expected, set) else expected == result:
                    return {"correct": True, "status": "exact"}
            return {"correct": False, "status": "type_mismatch",
                    "result": str(result), "expected": str(expected)}

        # Compare expressions
        try:
            diff = sympy.simplify(result - expected)
            if diff == 0:
                return {"correct": True, "status": "exact"}
        except:
            pass

        # Numeric comparison
        try:
            r_float = float(sympy.N(result))
            e_float = float(sympy.N(expected))
            if abs(r_float - e_float) < 1e-6:
                return {"correct": True, "status": "numeric"}
        except:
            pass

        return {"correct": False, "status": "wrong",
                "result": str(result), "expected": str(expected)}

    except Exception as e:
        return {"correct": False, "status": "error", "error": str(e)}


if __name__ == "__main__":
    # Test
    windows = [
        {"text": "The store has 3/5 of 40 croissants left."},
        {"text": "So the remaining croissants are 24."},
    ]

    result, segments = run_pipeline(windows)
    print_trace(segments, result)
