"""Expression tree data structures for GTS-based decomposition.

Per CLAUDE.md Big 5 #4 (True Atomic Decomposition) and #5 (Primitive vs Chain Nodes),
these classes support prefix notation parsing and atomic decomposition.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class NodeType(Enum):
    OPERATOR = 'operator'  # +, -, *, /
    NUMBER = 'number'      # Literal values
    VARIABLE = 'variable'  # NUM_0, NUM_1, step_1, etc.


@dataclass
class ExprNode:
    """Expression tree node supporting depth calculation and atomicity checks."""

    type: NodeType
    value: str
    left: Optional['ExprNode'] = None
    right: Optional['ExprNode'] = None

    @property
    def depth(self) -> int:
        """Tree depth. Depth 1 = single operation (atomic)."""
        if self.type != NodeType.OPERATOR:
            return 0
        left_depth = self.left.depth if self.left else 0
        right_depth = self.right.depth if self.right else 0
        return 1 + max(left_depth, right_depth)

    @property
    def is_atomic(self) -> bool:
        """Per CLAUDE.md: depth <= 1 is atomic during cold start."""
        return self.depth <= 1

    def to_operation_string(self) -> str:
        """Convert to operation description for graph embedding."""
        op_names = {
            '+': 'add',
            '-': 'subtract',
            '*': 'multiply',
            '/': 'divide',
            '^': 'power',
            '**': 'power',
        }
        if self.type == NodeType.OPERATOR:
            return f"{op_names.get(self.value, self.value)} two numbers"
        return self.value

    def to_prefix_string(self) -> str:
        """Convert tree back to prefix notation string."""
        if self.type != NodeType.OPERATOR:
            return self.value
        left_str = self.left.to_prefix_string() if self.left else ""
        right_str = self.right.to_prefix_string() if self.right else ""
        return f"{self.value} {left_str} {right_str}"

    def __repr__(self) -> str:
        if self.type != NodeType.OPERATOR:
            return f"ExprNode({self.type.value}, {self.value!r})"
        return f"ExprNode({self.type.value}, {self.value!r}, left={self.left!r}, right={self.right!r})"


def parse_prefix(prefix_str: str) -> ExprNode:
    """Parse GTS prefix output to expression tree.

    Example: "- + NUM_0 NUM_1 NUM_2" -> (- (+ NUM_0 NUM_1) NUM_2)

    Args:
        prefix_str: Prefix notation string with operators and operands.

    Returns:
        ExprNode: Root of the parsed expression tree.

    Raises:
        ValueError: If the expression is malformed.
    """
    # Clean up the string - remove parentheses and normalize whitespace
    tokens = prefix_str.replace('(', '').replace(')', '').split()

    if not tokens:
        raise ValueError("Empty expression")

    idx = 0

    def parse() -> ExprNode:
        nonlocal idx
        if idx >= len(tokens):
            raise ValueError("Unexpected end of expression")

        token = tokens[idx]
        idx += 1

        # Check if it's an operator
        if token in ['+', '-', '*', '/', '^', '**']:
            left = parse()
            right = parse()
            return ExprNode(
                type=NodeType.OPERATOR,
                value=token,
                left=left,
                right=right
            )
        # Check if it's a variable (NUM_*, step_*, x, y, etc.)
        elif (token.startswith('NUM') or
              token.startswith('step_') or
              token in ('x', 'y', 'z', 'n', 'm')):
            return ExprNode(type=NodeType.VARIABLE, value=token)
        else:
            # Try to parse as number
            try:
                float(token)
                return ExprNode(type=NodeType.NUMBER, value=token)
            except ValueError:
                # Treat as variable if not a number
                return ExprNode(type=NodeType.VARIABLE, value=token)

    result = parse()

    # Check for unconsumed tokens
    if idx < len(tokens):
        raise ValueError(f"Unconsumed tokens: {tokens[idx:]}")

    return result


def decompose_to_atomic(tree: ExprNode) -> list[tuple[str, ExprNode, dict]]:
    """Recursively decompose until all steps are depth 1.

    This implements the GTS-style decomposition where complex expressions
    are broken down into atomic operations (single operations on primitives).

    Args:
        tree: The expression tree to decompose.

    Returns:
        List of (step_name, atomic_tree, metadata) tuples where:
        - step_name: Name for this atomic step (e.g., "step_1", "step_2")
        - atomic_tree: An ExprNode with depth <= 1
        - metadata: Dict with 'operation', 'left_operand', 'right_operand'
    """
    if tree.is_atomic:
        # Already atomic, return as single step
        metadata = _build_metadata(tree)
        return [("step_1", tree, metadata)]

    # Need to decompose: extract sub-expressions and replace with step references
    steps: list[tuple[str, ExprNode, dict]] = []
    step_counter = [0]  # Use list for mutable reference in nested function

    def decompose_recursive(node: ExprNode) -> ExprNode:
        """Recursively decompose, returning atomic node or step reference."""
        if node.type != NodeType.OPERATOR:
            # Leaf node (number or variable) - already atomic
            return node

        # Check if this operation is already atomic
        if node.is_atomic:
            # This is an atomic operation, record it and return step reference
            step_counter[0] += 1
            step_name = f"step_{step_counter[0]}"
            metadata = _build_metadata(node)
            steps.append((step_name, node, metadata))
            return ExprNode(type=NodeType.VARIABLE, value=step_name)

        # Not atomic - decompose children first
        new_left = decompose_recursive(node.left) if node.left else None
        new_right = decompose_recursive(node.right) if node.right else None

        # Create new atomic operation with decomposed children
        new_node = ExprNode(
            type=NodeType.OPERATOR,
            value=node.value,
            left=new_left,
            right=new_right
        )

        # Record this as a step
        step_counter[0] += 1
        step_name = f"step_{step_counter[0]}"
        metadata = _build_metadata(new_node)
        steps.append((step_name, new_node, metadata))

        return ExprNode(type=NodeType.VARIABLE, value=step_name)

    # Start decomposition from root
    decompose_recursive(tree)

    return steps


def _build_metadata(node: ExprNode) -> dict:
    """Build metadata dict for an expression node."""
    if node.type != NodeType.OPERATOR:
        return {
            'operation': None,
            'left_operand': node.value if node.type == NodeType.VARIABLE else None,
            'right_operand': None,
            'is_leaf': True,
        }

    op_names = {
        '+': 'add',
        '-': 'subtract',
        '*': 'multiply',
        '/': 'divide',
        '^': 'power',
        '**': 'power',
    }

    return {
        'operation': op_names.get(node.value, node.value),
        'left_operand': node.left.value if node.left else None,
        'right_operand': node.right.value if node.right else None,
        'is_leaf': False,
    }
