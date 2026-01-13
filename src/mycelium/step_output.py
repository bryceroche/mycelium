"""Rich typed step outputs for symbolic DSL execution.

The Problem:
    Traditional step outputs are strings: "42" or "x^2 - 16 = 0"
    DSL scripts like `solve(equation, x)` need the SYMBOLIC FORM,
    not just the string representation.

The Solution:
    StepOutput preserves type information:
    - value_type: "number" | "equation" | "expression" | "list" | "text"
    - numeric: parsed float for numbers
    - sympy_expr: sympy-compatible string for equations/expressions
    - variables: list of variable names found

This enables DSL to use appropriate representations:
    - Arithmetic DSL: uses numeric value
    - Symbolic DSL: uses sympy_expr
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class StepOutput:
    """Rich typed output from a step.

    Instead of storing just "42" or "x^2 - 16 = 0" as strings,
    we preserve type information so DSL can use appropriate form.
    """
    raw: str                           # Original output: "x² - 16 = 0"
    value_type: str                    # "number" | "equation" | "expression" | "list" | "text"
    numeric: Optional[float] = None    # 42.0 if it's a number
    sympy_expr: Optional[str] = None   # "Eq(x**2 - 16, 0)" for equations
    variables: list[str] = field(default_factory=list)  # ["x", "k"] - symbols

    def __str__(self) -> str:
        """For backwards compatibility, string conversion returns raw."""
        return self.raw

    def __repr__(self) -> str:
        return f"StepOutput({self.value_type}: {self.raw[:30]}...)"

    def for_dsl(self, prefer_type: str = "auto") -> Any:
        """Return the best representation for DSL execution.

        Args:
            prefer_type: "numeric" | "symbolic" | "auto"
                - numeric: prefer numeric value if available
                - symbolic: prefer sympy expression if available
                - auto: return most specific form available
        """
        if prefer_type == "numeric" or (prefer_type == "auto" and self.value_type == "number"):
            if self.numeric is not None:
                return self.numeric

        if prefer_type == "symbolic" or self.value_type in ("equation", "expression"):
            if self.sympy_expr:
                return self.sympy_expr

        return self.raw

    def is_numeric(self) -> bool:
        """Check if this output can be used as a number."""
        return self.numeric is not None

    def is_symbolic(self) -> bool:
        """Check if this output has symbolic form."""
        return self.sympy_expr is not None


def detect_output_type(raw: str) -> StepOutput:
    """Detect the type of a step output and parse accordingly.

    Examples:
        "42" → number
        "x^2 - 16 = 0" → equation
        "3x + 2y" → expression
        "[1, 2, 3]" → list
        "The answer is..." → text
    """
    if not raw:
        return StepOutput(raw="", value_type="text")

    raw = raw.strip()

    # Try numeric first
    numeric = try_parse_numeric(raw)
    if numeric is not None:
        return StepOutput(
            raw=raw,
            value_type="number",
            numeric=numeric,
        )

    # Check for equation (contains =, but not ==, <=, >=, !=)
    # Must have variable-like content, not just "Answer = 42"
    if re.search(r'(?<![=!<>])=(?![=])', raw):
        sympy_expr, variables = try_parse_equation(raw)
        if variables:  # Only treat as equation if it has variables
            return StepOutput(
                raw=raw,
                value_type="equation",
                sympy_expr=sympy_expr,
                variables=variables,
            )

    # Check for inequality with variables
    if re.search(r'[<>]=?|≤|≥', raw):
        sympy_expr, variables = try_parse_expression(raw)
        if variables:
            return StepOutput(
                raw=raw,
                value_type="expression",
                sympy_expr=sympy_expr,
                variables=variables,
            )

    # Check for list/set notation
    if re.match(r'^\s*[\[\{].*[\]\}]\s*$', raw, re.DOTALL):
        items = try_parse_list(raw)
        return StepOutput(
            raw=raw,
            value_type="list",
        )

    # Check for mathematical expression (has variables/operators but no =)
    # More aggressive detection: any single letter followed/preceded by operator or number
    if has_mathematical_content(raw):
        sympy_expr, variables = try_parse_expression(raw)
        if variables:
            return StepOutput(
                raw=raw,
                value_type="expression",
                sympy_expr=sympy_expr,
                variables=variables,
            )

    # Default to text
    return StepOutput(
        raw=raw,
        value_type="text",
    )


def has_mathematical_content(raw: str) -> bool:
    """Check if string contains mathematical expression patterns."""
    # Pattern: variable with operator nearby
    patterns = [
        r'\b[a-zA-Z]\s*[\+\-\*\/\^]',      # x +
        r'[\+\-\*\/\^]\s*[a-zA-Z]\b',      # + x
        r'\d+\s*[a-zA-Z]\b',                # 3x
        r'\b[a-zA-Z]\s*\d+',                # x2 (like x^2 written as x2)
        r'\b[a-zA-Z]\s*\(',                 # f(
        r'\)\s*[\+\-\*\/\^]',               # ) +
        r'[\+\-\*\/\^]\s*\(',               # + (
    ]
    return any(re.search(p, raw) for p in patterns)


def try_parse_numeric(raw: str) -> Optional[float]:
    """Try to parse as a number.

    Also extracts numbers from text like "The answer is 25" or "result: 42".
    """
    # Clean common formatting
    cleaned = raw.replace(',', '').replace('$', '').replace('%', '').strip()

    # Try direct parse first (fast path)
    try:
        return float(cleaned)
    except ValueError:
        pass

    # Handle fractions like "3/4" FIRST (before expression check)
    # Must check before general expression detection
    if '/' in cleaned and cleaned.count('/') == 1:
        parts = cleaned.split('/')
        if len(parts) == 2:
            try:
                num = float(parts[0].strip())
                denom = float(parts[1].strip())
                if denom != 0:
                    return num / denom
            except ValueError:
                pass  # Not a simple fraction, continue with other checks

    # Handle LaTeX fractions: \frac{6}{23}
    frac_match = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', cleaned)
    if frac_match:
        try:
            return float(frac_match.group(1)) / float(frac_match.group(2))
        except (ValueError, ZeroDivisionError):
            pass

    # Handle sqrt, pi, etc. - these are expressions, not simple numbers
    if re.search(r'sqrt|π|\\pi|\bpi\b', cleaned, re.IGNORECASE):
        return None

    # Try to extract number from text like "The answer is 25" or "result: 42"
    # Aggressive extraction - try many patterns
    answer_patterns = [
        # Explicit answer markers
        r'(?:answer|result|value|total|sum|product|equals?|is)\s*[=:]?\s*(-?\d+\.?\d*)',
        r'(?:therefore|thus|so|hence)\s*[,:]?\s*(-?\d+\.?\d*)',
        # Boxed answers (LaTeX)
        r'\\boxed\{(-?\d+\.?\d*)\}',
        r'\\box\{(-?\d+\.?\d*)\}',
        # Final answer patterns
        r'final\s+(?:answer|result)\s*[=:]?\s*(-?\d+\.?\d*)',
        r'the\s+(?:answer|result|total|sum)\s+is\s+(-?\d+\.?\d*)',
        # Equation results
        r'[a-zA-Z_]\s*=\s*(-?\d+\.?\d*)\s*$',  # "x = 25" at end
        r'=\s*(-?\d+\.?\d*)\s*$',  # "= 25" at end
        # Step output formats
        r'step\s*\d*\s*[=:]\s*(-?\d+\.?\d*)',
        r'output\s*[=:]\s*(-?\d+\.?\d*)',
        # General: last number in string (fallback)
        r'(-?\d+\.?\d*)\s*$',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                # Sanity check - avoid extracting year-like numbers from text
                if val != 0 or '0' in match.group(1):
                    return val
            except ValueError:
                continue

    return None


def try_parse_equation(raw: str) -> tuple[Optional[str], list[str]]:
    """Try to convert equation to sympy format.

    "x^2 - 16 = 0" → ("Eq(x**2 - 16, 0)", ["x"])
    """
    try:
        # Split on = (but not ==, <=, >=, !=)
        parts = re.split(r'(?<![=!<>])=(?![=])', raw)
        if len(parts) != 2:
            return None, []

        lhs, rhs = parts[0].strip(), parts[1].strip()

        # Convert common notation to sympy
        lhs_sympy = to_sympy_notation(lhs)
        rhs_sympy = to_sympy_notation(rhs)

        # Extract variables
        variables = extract_variables(lhs + " " + rhs)

        if not variables:
            return None, []

        sympy_expr = f"Eq({lhs_sympy}, {rhs_sympy})"
        return sympy_expr, variables

    except Exception as e:
        logger.debug("Failed to parse equation '%s': %s", raw[:50], e)
        return None, []


def try_parse_expression(raw: str) -> tuple[Optional[str], list[str]]:
    """Try to convert expression to sympy format.

    "3x^2 + 2x - 5" → ("3*x**2 + 2*x - 5", ["x"])
    """
    try:
        sympy_expr = to_sympy_notation(raw)
        variables = extract_variables(raw)
        return sympy_expr, variables
    except Exception as e:
        logger.debug("Failed to parse expression '%s': %s", raw[:50], e)
        return None, []


def try_parse_list(raw: str) -> list:
    """Try to parse a list/set notation."""
    try:
        # Remove brackets and split
        inner = raw.strip()[1:-1]
        items = [x.strip() for x in inner.split(',')]
        return items
    except Exception:
        return []


def to_sympy_notation(expr: str) -> str:
    """Convert mathematical notation to sympy format.

    "x^2" → "x**2"
    "3x" → "3*x"
    "sin(x)" → "sin(x)"  # already valid
    """
    result = expr

    # Remove LaTeX formatting
    result = re.sub(r'\\left|\\right', '', result)
    result = result.replace('\\cdot', '*')
    result = result.replace('\\times', '*')
    result = result.replace('\\div', '/')

    # ^ to **
    result = result.replace('^', '**')

    # Implicit multiplication: 3x → 3*x, 2(x+1) → 2*(x+1)
    result = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', result)
    result = re.sub(r'(\d)\(', r'\1*(', result)
    result = re.sub(r'\)([a-zA-Z])', r')*\1', result)
    result = re.sub(r'\)\(', r')*(', result)
    result = re.sub(r'([a-zA-Z])\(', r'\1*(', result)

    # Handle common symbols
    result = result.replace('√', 'sqrt')
    result = result.replace('π', 'pi')
    result = result.replace('≤', '<=')
    result = result.replace('≥', '>=')

    return result


def extract_variables(expr: str) -> list[str]:
    """Extract variable names from expression.

    "3x^2 + 2y - z" → ["x", "y", "z"]
    """
    # Known function names to exclude
    function_names = {
        'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
        'log', 'ln', 'exp', 'sqrt', 'abs',
        'min', 'max', 'sum', 'prod',
        'floor', 'ceil', 'round',
    }

    # Known constants to exclude
    constants = {'pi', 'e', 'i', 'inf'}

    # Find all identifiers (single letters are most likely variables)
    # Multi-letter identifiers might be functions or variable names
    # Use lookbehind/lookahead to handle "3x" (digit followed by letter)
    single_letters = set(re.findall(r'(?<![a-zA-Z])([a-zA-Z])(?![a-zA-Z])', expr))
    multi_letter = set(re.findall(r'\b([a-zA-Z][a-zA-Z0-9_]+)\b', expr))

    variables = []

    # Add single letters (most likely variables)
    for var in sorted(single_letters):
        if var.lower() not in constants:
            variables.append(var)

    # Add multi-letter identifiers that aren't functions
    for ident in sorted(multi_letter):
        if ident.lower() not in function_names and ident.lower() not in constants:
            # Only add if it looks like a variable (e.g., x1, area_ABC)
            if re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', ident):
                variables.append(ident)

    return variables


def validate_llm_output_type(
    result: str,
    context: dict,
    step_task: str,
) -> tuple[bool, str]:
    """Validate that LLM output type matches what context suggests.

    When context has concrete numeric values and the task asks for computation,
    the output should be numeric, not symbolic.

    Args:
        result: The LLM's output string
        context: Dict of previous step results (step_id -> value)
        step_task: The task description for this step

    Returns:
        (is_valid, message) - True if valid, False with explanation if not
    """
    if not result or not context:
        return True, ""

    # Parse the result to see what type it is
    output = detect_output_type(result)

    # Check if context has numeric values
    context_has_numbers = False
    numeric_context_values = []
    for key, value in context.items():
        if key.startswith("_"):  # Skip internal keys
            continue
        ctx_output = detect_output_type(str(value))
        if ctx_output.is_numeric():
            context_has_numbers = True
            numeric_context_values.append((key, ctx_output.numeric))

    if not context_has_numbers:
        # No numeric context, can't validate
        return True, ""

    # Check if task suggests numeric computation
    compute_keywords = [
        "calculate", "compute", "find", "determine", "evaluate",
        "what is", "how many", "how much", "sum", "product",
        "total", "multiply", "divide", "add", "subtract",
        "result", "value", "answer"
    ]
    task_lower = step_task.lower()
    task_expects_numeric = any(kw in task_lower for kw in compute_keywords)

    if not task_expects_numeric:
        # Task doesn't seem to expect numeric output
        return True, ""

    # Now validate: if context has numbers and task expects computation,
    # output should be numeric (not symbolic with unresolved variables)
    if output.value_type == "number":
        return True, ""

    if output.value_type in ("expression", "equation") and output.variables:
        # Check if variables appear in the task (problem variables - OK)
        # vs truly unresolved variables (BAD)
        task_variables = set(extract_variables(step_task.lower()))
        output_variables = set(v.lower() for v in output.variables)

        # Variables that appear in output but NOT in task are suspicious
        unresolved_vars = output_variables - task_variables

        # Also allow common answer variables (x, y, n, etc.) that appear in "solve for x" type tasks
        common_answer_vars = {'x', 'y', 'z', 'n', 'm', 'k', 't', 'r', 'a', 'b', 'c'}
        truly_unresolved = unresolved_vars - common_answer_vars

        if truly_unresolved:
            # Output has variables that don't appear in task and aren't common answer vars
            logger.warning(
                "[llm_output] Symbolic output with numeric context: "
                "result='%s' (unresolved=%s) but context has %s",
                result[:50], truly_unresolved, numeric_context_values[:3]
            )
            return False, f"Output has unresolved variables ({truly_unresolved}) but context has numeric values"

    # Text output when expecting numeric
    if output.value_type == "text" and task_expects_numeric:
        # Try to extract a number from the text
        extracted = try_parse_numeric(result)
        if extracted is None:
            logger.warning(
                "[llm_output] Non-numeric output for computation task: "
                "result='%s' task='%s'",
                result[:50], step_task[:50]
            )
            return False, "Expected numeric output for computation task"

    return True, ""
