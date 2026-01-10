"""I/O Schema: Standardized input/output specifications for step types.

This module defines the schema system for step inputs and outputs,
enabling:
- Structured context formatting from previous steps
- More specific prompts based on expected I/O
- Output validation
- Direct formula execution (when possible)
"""

import ast
import json
import math
import operator
import re
from dataclasses import dataclass, field
from typing import Optional


# Common value types for step I/O
VALUE_TYPES = {
    "numeric": "A single numeric value (integer or decimal)",
    "expression": "A mathematical expression (e.g., '2x + 3')",
    "equation": "An equation with equals sign (e.g., 'x = 5')",
    "boolean": "True/False or Yes/No",
    "text": "Free-form text",
    "list": "A list of values",
    "symbol": "A variable or symbol name",
}

# Common output formats
OUTPUT_FORMATS = {
    "integer": "Whole number (e.g., 42)",
    "decimal": "Decimal number (e.g., 3.14)",
    "fraction": "Fraction (e.g., 3/4)",
    "percentage": "Percentage (e.g., 25%)",
    "currency": "Currency value (e.g., $10.50)",
    "simplified": "Simplified expression",
    "solved": "Solved equation (variable isolated)",
}

# Execution modes for step schemas
EXECUTION_MODES = ["guidance", "formula", "procedure", "dsl"]


@dataclass
class InputSpec:
    """Specification for a step input.

    Describes what value a step needs and where it comes from.
    """
    name: str  # Identifier for this input (e.g., "percentage", "base_value")
    value_type: str = "numeric"  # One of VALUE_TYPES
    source: str = "previous"  # "previous" (from prior step), "problem" (from original), "literal"
    description: str = ""  # Human-readable description
    required: bool = True  # Whether this input is required
    default: Optional[str] = None  # Default value if not required

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value_type": self.value_type,
            "source": self.source,
            "description": self.description,
            "required": self.required,
            "default": self.default,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InputSpec":
        return cls(**d)


@dataclass
class OutputSpec:
    """Specification for a step output.

    Describes what type and format of value a step produces.
    """
    value_type: str = "numeric"  # One of VALUE_TYPES
    format: str = ""  # One of OUTPUT_FORMATS (optional, more specific)
    unit: str = ""  # Unit of measurement (e.g., "$", "%", "meters")
    description: str = ""  # Human-readable description

    def to_dict(self) -> dict:
        return {
            "value_type": self.value_type,
            "format": self.format,
            "unit": self.unit,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OutputSpec":
        return cls(**d)

    def format_hint(self) -> str:
        """Generate a hint string for the expected output format."""
        parts = []
        if self.value_type:
            parts.append(self.value_type)
        if self.format:
            parts.append(f"({self.format})")
        if self.unit:
            parts.append(f"in {self.unit}")
        return " ".join(parts) if parts else "any value"


@dataclass
class StepIOSchema:
    """Input/Output schema for a step type.

    Defines what inputs a step needs and what output it produces.
    Used to:
    - Format context from previous steps
    - Generate more specific prompts
    - Validate step outputs
    - Execute formulas directly (when possible)
    """
    inputs: list[InputSpec] = field(default_factory=list)
    output: OutputSpec = field(default_factory=OutputSpec)

    # Enhanced method template with placeholders (fallback/guidance)
    # e.g., "Given {percentage} and {base_value}, divide percentage by 100, then multiply."
    method_template_v2: str = ""

    # Execution mode: "guidance" | "formula" | "procedure"
    execution_mode: str = "guidance"

    # For formula mode: Python-evaluable expression
    # e.g., "(percentage / 100) * base"
    formula: Optional[str] = None

    # For procedure mode: ordered list of steps
    # e.g., ["1. Identify variable terms", "2. Move to left side", ...]
    procedure: Optional[list[str]] = None

    def to_json(self) -> str:
        """Serialize to JSON for database storage."""
        data = {
            "inputs": [i.to_dict() for i in self.inputs],
            "output": self.output.to_dict(),
            "method_template_v2": self.method_template_v2,
            "execution_mode": self.execution_mode,
        }
        if self.formula:
            data["formula"] = self.formula
        if self.procedure:
            data["procedure"] = self.procedure
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> "StepIOSchema":
        """Deserialize from JSON."""
        if not json_str:
            return cls()
        try:
            d = json.loads(json_str)
            return cls(
                inputs=[InputSpec.from_dict(i) for i in d.get("inputs", [])],
                output=OutputSpec.from_dict(d.get("output", {})),
                method_template_v2=d.get("method_template_v2", ""),
                execution_mode=d.get("execution_mode", "guidance"),
                formula=d.get("formula"),
                procedure=d.get("procedure"),
            )
        except (json.JSONDecodeError, KeyError):
            return cls()

    def format_inputs(self, context: dict[str, str]) -> str:
        """Format input values from context according to schema.

        Args:
            context: Dict mapping step_id -> result string

        Returns:
            Formatted string describing available inputs
        """
        if not self.inputs:
            # Fallback: format context as-is
            lines = [f"- {k}: {v}" for k, v in context.items()]
            return "\n".join(lines) if lines else "No previous results."

        lines = []
        for inp in self.inputs:
            # Try to find matching value in context
            value = self._extract_value(inp, context)
            if value:
                lines.append(f"- {inp.name}: {value}")
                if inp.description:
                    lines.append(f"  ({inp.description})")
            elif inp.required:
                lines.append(f"- {inp.name}: [not available - required]")
            elif inp.default:
                lines.append(f"- {inp.name}: {inp.default} (default)")

        return "\n".join(lines) if lines else "No inputs specified."

    def _extract_value(self, inp: InputSpec, context: dict[str, str]) -> Optional[str]:
        """Extract value for an input from context."""
        # Try direct match by name
        for key, value in context.items():
            if inp.name.lower() in key.lower():
                return self._parse_numeric(value) if inp.value_type == "numeric" else value

        # Try to extract numeric from any context value
        if inp.value_type == "numeric" and context:
            for value in context.values():
                extracted = self._parse_numeric(value)
                if extracted:
                    return extracted

        return None

    def _parse_numeric(self, text: str) -> Optional[str]:
        """Extract numeric value from text."""
        # Try to find a number in the text
        patterns = [
            r"(?:^|=\s*)(-?\d+\.?\d*)",  # Leading or after equals
            r"(-?\d+\.?\d*)\s*$",  # Trailing number
            r"(-?\d+\.?\d*)",  # Any number
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    def validate_output(self, result: str) -> tuple[bool, str]:
        """Validate that a result conforms to the output schema.

        Args:
            result: The step result to validate

        Returns:
            (is_valid, message) tuple
        """
        if not result:
            return False, "Empty result"

        if self.output.value_type == "numeric":
            # Check if result contains a number
            if not re.search(r'-?\d+\.?\d*', result):
                return False, f"Expected numeric output, got: {result[:50]}"

        if self.output.value_type == "boolean":
            lower = result.lower()
            if not any(b in lower for b in ["true", "false", "yes", "no"]):
                return False, f"Expected boolean output, got: {result[:50]}"

        if self.output.value_type == "equation":
            if "=" not in result:
                return False, f"Expected equation with '=', got: {result[:50]}"

        return True, "OK"

    def format_procedure(self) -> str:
        """Format procedure steps for prompt injection."""
        if not self.procedure:
            return ""
        return "\n".join(self.procedure)

    def extract_numeric_inputs(self, context: dict[str, str]) -> dict[str, float]:
        """Extract numeric values from context for formula evaluation.

        Supports two extraction modes:
        1. By name: Match input names to context keys (e.g., 'percentage' in 'step_percentage')
        2. Positional: When names don't match, extract values in order from context

        Returns:
            Dict mapping input names to float values.
            Only includes inputs that could be parsed as numbers.
        """
        result = {}
        numeric_inputs = [inp for inp in self.inputs if inp.value_type == "numeric"]

        # First pass: try name matching
        for inp in numeric_inputs:
            value_str = self._extract_value_by_name(inp, context)
            if value_str:
                try:
                    result[inp.name] = float(value_str)
                except ValueError:
                    pass

        # If name matching didn't find all inputs, try positional
        if len(result) < len(numeric_inputs):
            # Extract all numeric values from context in order
            context_values = []
            for key in sorted(context.keys()):  # Sort for consistent ordering
                value_str = self._parse_numeric(context[key])
                if value_str:
                    try:
                        context_values.append(float(value_str))
                    except ValueError:
                        pass

            # Map positionally to inputs that weren't found by name
            value_idx = 0
            for inp in numeric_inputs:
                if inp.name not in result and value_idx < len(context_values):
                    result[inp.name] = context_values[value_idx]
                    value_idx += 1

        return result

    def _extract_value_by_name(self, inp: InputSpec, context: dict[str, str]) -> Optional[str]:
        """Extract value for an input by matching name to context keys."""
        for key, value in context.items():
            if inp.name.lower() in key.lower():
                return self._parse_numeric(value) if inp.value_type == "numeric" else value
        return None


def try_execute_formula(formula: str, inputs: dict[str, float]) -> Optional[float]:
    """Safely execute a formula with given inputs.

    Uses AST-based evaluation with only safe math operations.
    Does NOT use eval() - parses and evaluates the AST directly.

    Args:
        formula: Python-evaluable expression (e.g., "(percentage / 100) * base")
        inputs: Dict mapping variable names to float values

    Returns:
        Computed result if successful, None if formula needs LLM interpretation.
    """
    if not formula or not inputs:
        return None

    try:
        result = _safe_eval_formula(formula, inputs)
        return float(result) if result is not None else None
    except (ValueError, TypeError, SyntaxError, ZeroDivisionError, KeyError, AttributeError):
        return None  # Fall back to LLM


def _safe_eval_formula(formula: str, inputs: dict[str, float]) -> Optional[float]:
    """AST-based safe formula evaluator. No eval() used.

    Only allows: numbers, variables from inputs, basic math ops, and safe functions.
    """
    # Allowed binary operators
    BINOPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    # Allowed unary operators
    UNARYOPS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    # Allowed functions (name -> callable)
    FUNCTIONS = {
        "abs": abs,
        "min": min,
        "max": max,
        "round": round,
        "pow": pow,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
    }

    # Allowed constants
    CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
    }

    def _eval_node(node: ast.AST) -> float:
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)

        elif isinstance(node, ast.Constant):  # Python 3.8+
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

        elif isinstance(node, ast.Num):  # Python 3.7 compatibility
            return float(node.n)

        elif isinstance(node, ast.Name):
            name = node.id
            if name in inputs:
                return float(inputs[name])
            if name in CONSTANTS:
                return CONSTANTS[name]
            raise KeyError(f"Unknown variable: {name}")

        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in BINOPS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            return BINOPS[op_type](left, right)

        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in UNARYOPS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            operand = _eval_node(node.operand)
            return UNARYOPS[op_type](operand)

        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed")
            func_name = node.func.id
            if func_name not in FUNCTIONS:
                raise ValueError(f"Unsupported function: {func_name}")
            args = [_eval_node(arg) for arg in node.args]
            return FUNCTIONS[func_name](*args)

        else:
            raise ValueError(f"Unsupported AST node: {type(node).__name__}")

    # Parse and evaluate
    tree = ast.parse(formula, mode='eval')
    return _eval_node(tree)


# Default schemas for common step types
DEFAULT_IO_SCHEMAS: dict[str, StepIOSchema] = {
    "compute_percentage": StepIOSchema(
        inputs=[
            InputSpec(name="percentage", value_type="numeric", description="The percentage value"),
            InputSpec(name="base_value", value_type="numeric", description="The value to take percentage of"),
        ],
        output=OutputSpec(value_type="numeric", format="decimal"),
        execution_mode="formula",
        formula="(percentage / 100) * base_value",
        method_template_v2="Convert percentage to decimal (divide by 100), then multiply by base value.",
    ),
    "compute_sum": StepIOSchema(
        inputs=[
            InputSpec(name="a", value_type="numeric", description="First value"),
            InputSpec(name="b", value_type="numeric", description="Second value"),
        ],
        output=OutputSpec(value_type="numeric"),
        execution_mode="formula",
        formula="a + b",
        method_template_v2="Add all the given values together to get the sum.",
    ),
    "compute_product": StepIOSchema(
        inputs=[
            InputSpec(name="a", value_type="numeric", description="First value"),
            InputSpec(name="b", value_type="numeric", description="Second value"),
        ],
        output=OutputSpec(value_type="numeric"),
        execution_mode="formula",
        formula="a * b",
        method_template_v2="Multiply all the given values together to get the product.",
    ),
    "compute_difference": StepIOSchema(
        inputs=[
            InputSpec(name="a", value_type="numeric", description="Value to subtract from"),
            InputSpec(name="b", value_type="numeric", description="Value to subtract"),
        ],
        output=OutputSpec(value_type="numeric"),
        execution_mode="formula",
        formula="a - b",
        method_template_v2="Subtract the second value from the first.",
    ),
    "compute_quotient": StepIOSchema(
        inputs=[
            InputSpec(name="a", value_type="numeric", description="Dividend"),
            InputSpec(name="b", value_type="numeric", description="Divisor"),
        ],
        output=OutputSpec(value_type="numeric"),
        execution_mode="formula",
        formula="a / b",
        method_template_v2="Divide the first value by the second.",
    ),
    "solve_equation": StepIOSchema(
        inputs=[
            InputSpec(name="equation", value_type="equation", description="The equation to solve"),
            InputSpec(name="variable", value_type="symbol", description="Variable to solve for"),
        ],
        output=OutputSpec(value_type="equation", format="solved"),
        execution_mode="procedure",
        procedure=[
            "1. Identify all terms containing the variable",
            "2. Move all variable terms to the left side of the equation",
            "3. Move all constant terms to the right side",
            "4. Combine like terms on each side",
            "5. Divide both sides by the coefficient of the variable",
            "6. Write the solution as: variable = value",
        ],
        method_template_v2="Isolate the variable by performing inverse operations on both sides.",
    ),
    "substitute_value": StepIOSchema(
        inputs=[
            InputSpec(name="expression", value_type="expression", description="Expression with variable"),
            InputSpec(name="variable", value_type="symbol", description="Variable to replace"),
            InputSpec(name="value", value_type="numeric", description="Value to substitute"),
        ],
        output=OutputSpec(value_type="numeric"),
        execution_mode="procedure",
        procedure=[
            "1. Identify all occurrences of the variable in the expression",
            "2. Replace each occurrence with the given value",
            "3. Evaluate the resulting arithmetic expression",
            "4. Return the numeric result",
        ],
        method_template_v2="Replace the variable in the expression with the given value, then evaluate.",
    ),
    "simplify_expression": StepIOSchema(
        inputs=[
            InputSpec(name="expression", value_type="expression", description="Expression to simplify"),
        ],
        output=OutputSpec(value_type="expression", format="simplified"),
        execution_mode="procedure",
        procedure=[
            "1. Identify like terms (terms with the same variable and exponent)",
            "2. Combine coefficients of like terms",
            "3. Simplify any arithmetic operations",
            "4. Write the result in standard form",
        ],
        method_template_v2="Combine like terms and reduce to simplest form.",
    ),
    "synthesize_answer": StepIOSchema(
        inputs=[
            InputSpec(name="results", value_type="list", description="Results from previous steps"),
        ],
        output=OutputSpec(value_type="text"),
        execution_mode="guidance",
        method_template_v2="Combine all previous results to form the final answer to the original problem.",
    ),
}


def get_default_schema(step_type: str) -> Optional[StepIOSchema]:
    """Get the default I/O schema for a step type."""
    return DEFAULT_IO_SCHEMAS.get(step_type)
