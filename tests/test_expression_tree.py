"""Tests for expression tree data structures."""

import pytest
from mycelium.expression_tree import (
    ExprNode,
    NodeType,
    parse_prefix,
    decompose_to_atomic,
)


class TestDepthCalculation:
    """Tests for tree depth calculation."""

    def test_leaf_number_has_depth_zero(self):
        """Number nodes have depth 0."""
        node = ExprNode(type=NodeType.NUMBER, value="42")
        assert node.depth == 0

    def test_leaf_variable_has_depth_zero(self):
        """Variable nodes have depth 0."""
        node = ExprNode(type=NodeType.VARIABLE, value="NUM_0")
        assert node.depth == 0

    def test_single_operation_has_depth_one(self):
        """Single operation on two leaves has depth 1."""
        node = ExprNode(
            type=NodeType.OPERATOR,
            value="+",
            left=ExprNode(type=NodeType.VARIABLE, value="NUM_0"),
            right=ExprNode(type=NodeType.VARIABLE, value="NUM_1"),
        )
        assert node.depth == 1

    def test_nested_operation_has_depth_two(self):
        """Nested operations increase depth."""
        # (+ (+ NUM_0 NUM_1) NUM_2) has depth 2
        inner = ExprNode(
            type=NodeType.OPERATOR,
            value="+",
            left=ExprNode(type=NodeType.VARIABLE, value="NUM_0"),
            right=ExprNode(type=NodeType.VARIABLE, value="NUM_1"),
        )
        outer = ExprNode(
            type=NodeType.OPERATOR,
            value="+",
            left=inner,
            right=ExprNode(type=NodeType.VARIABLE, value="NUM_2"),
        )
        assert outer.depth == 2

    def test_deeply_nested_depth(self):
        """Test depth with multiple levels of nesting."""
        # (+ (+ (+ NUM_0 NUM_1) NUM_2) NUM_3) has depth 3
        level1 = ExprNode(
            type=NodeType.OPERATOR,
            value="+",
            left=ExprNode(type=NodeType.VARIABLE, value="NUM_0"),
            right=ExprNode(type=NodeType.VARIABLE, value="NUM_1"),
        )
        level2 = ExprNode(
            type=NodeType.OPERATOR,
            value="+",
            left=level1,
            right=ExprNode(type=NodeType.VARIABLE, value="NUM_2"),
        )
        level3 = ExprNode(
            type=NodeType.OPERATOR,
            value="+",
            left=level2,
            right=ExprNode(type=NodeType.VARIABLE, value="NUM_3"),
        )
        assert level3.depth == 3

    def test_balanced_tree_depth(self):
        """Test depth with balanced tree."""
        # (+ (+ NUM_0 NUM_1) (+ NUM_2 NUM_3)) has depth 2
        left = ExprNode(
            type=NodeType.OPERATOR,
            value="+",
            left=ExprNode(type=NodeType.VARIABLE, value="NUM_0"),
            right=ExprNode(type=NodeType.VARIABLE, value="NUM_1"),
        )
        right = ExprNode(
            type=NodeType.OPERATOR,
            value="+",
            left=ExprNode(type=NodeType.VARIABLE, value="NUM_2"),
            right=ExprNode(type=NodeType.VARIABLE, value="NUM_3"),
        )
        root = ExprNode(
            type=NodeType.OPERATOR,
            value="+",
            left=left,
            right=right,
        )
        assert root.depth == 2


class TestAtomicityCheck:
    """Tests for atomicity checking."""

    def test_leaf_is_atomic(self):
        """Leaf nodes (depth 0) are atomic."""
        node = ExprNode(type=NodeType.VARIABLE, value="NUM_0")
        assert node.is_atomic is True

    def test_single_operation_is_atomic(self):
        """Single operation (depth 1) is atomic."""
        node = ExprNode(
            type=NodeType.OPERATOR,
            value="+",
            left=ExprNode(type=NodeType.VARIABLE, value="NUM_0"),
            right=ExprNode(type=NodeType.VARIABLE, value="NUM_1"),
        )
        assert node.is_atomic is True

    def test_nested_operation_not_atomic(self):
        """Nested operations (depth > 1) are not atomic."""
        inner = ExprNode(
            type=NodeType.OPERATOR,
            value="+",
            left=ExprNode(type=NodeType.VARIABLE, value="NUM_0"),
            right=ExprNode(type=NodeType.VARIABLE, value="NUM_1"),
        )
        outer = ExprNode(
            type=NodeType.OPERATOR,
            value="+",
            left=inner,
            right=ExprNode(type=NodeType.VARIABLE, value="NUM_2"),
        )
        assert outer.is_atomic is False


class TestPrefixParsing:
    """Tests for prefix notation parsing with GTS-style tokens."""

    def test_parse_single_variable(self):
        """Parse single variable."""
        tree = parse_prefix("NUM_0")
        assert tree.type == NodeType.VARIABLE
        assert tree.value == "NUM_0"

    def test_parse_single_number(self):
        """Parse single number."""
        tree = parse_prefix("42")
        assert tree.type == NodeType.NUMBER
        assert tree.value == "42"

    def test_parse_simple_addition(self):
        """Parse simple addition: + NUM_0 NUM_1."""
        tree = parse_prefix("+ NUM_0 NUM_1")
        assert tree.type == NodeType.OPERATOR
        assert tree.value == "+"
        assert tree.left.type == NodeType.VARIABLE
        assert tree.left.value == "NUM_0"
        assert tree.right.type == NodeType.VARIABLE
        assert tree.right.value == "NUM_1"

    def test_parse_nested_expression(self):
        """Parse nested: - + NUM_0 NUM_1 NUM_2."""
        tree = parse_prefix("- + NUM_0 NUM_1 NUM_2")
        assert tree.type == NodeType.OPERATOR
        assert tree.value == "-"
        assert tree.left.type == NodeType.OPERATOR
        assert tree.left.value == "+"
        assert tree.left.left.value == "NUM_0"
        assert tree.left.right.value == "NUM_1"
        assert tree.right.value == "NUM_2"

    def test_parse_multiplication(self):
        """Parse multiplication: * NUM_0 NUM_1."""
        tree = parse_prefix("* NUM_0 NUM_1")
        assert tree.type == NodeType.OPERATOR
        assert tree.value == "*"
        assert tree.left.value == "NUM_0"
        assert tree.right.value == "NUM_1"

    def test_parse_division(self):
        """Parse division: / NUM_0 NUM_1."""
        tree = parse_prefix("/ NUM_0 NUM_1")
        assert tree.type == NodeType.OPERATOR
        assert tree.value == "/"

    def test_parse_power(self):
        """Parse power: ^ NUM_0 NUM_1."""
        tree = parse_prefix("^ NUM_0 NUM_1")
        assert tree.type == NodeType.OPERATOR
        assert tree.value == "^"

    def test_parse_step_references(self):
        """Parse step references: + step_1 step_2."""
        tree = parse_prefix("+ step_1 step_2")
        assert tree.type == NodeType.OPERATOR
        assert tree.left.type == NodeType.VARIABLE
        assert tree.left.value == "step_1"
        assert tree.right.value == "step_2"

    def test_parse_complex_nested(self):
        """Parse complex nested: * + NUM_0 NUM_1 - NUM_2 NUM_3."""
        tree = parse_prefix("* + NUM_0 NUM_1 - NUM_2 NUM_3")
        assert tree.type == NodeType.OPERATOR
        assert tree.value == "*"
        assert tree.left.value == "+"
        assert tree.right.value == "-"
        assert tree.depth == 2

    def test_parse_with_literal_x(self):
        """Parse expressions with variable x."""
        tree = parse_prefix("+ x NUM_0")
        assert tree.left.type == NodeType.VARIABLE
        assert tree.left.value == "x"

    def test_parse_with_parentheses(self):
        """Parse expression with parentheses (should be stripped)."""
        tree = parse_prefix("(+ (NUM_0) (NUM_1))")
        assert tree.type == NodeType.OPERATOR
        assert tree.value == "+"

    def test_parse_float_number(self):
        """Parse floating point number."""
        tree = parse_prefix("3.14")
        assert tree.type == NodeType.NUMBER
        assert tree.value == "3.14"

    def test_parse_negative_number(self):
        """Parse negative number."""
        tree = parse_prefix("-5")
        # Note: -5 is parsed as a number, not as negation operation
        assert tree.type == NodeType.NUMBER
        assert tree.value == "-5"

    def test_parse_empty_raises(self):
        """Empty expression raises ValueError."""
        with pytest.raises(ValueError, match="Empty expression"):
            parse_prefix("")

    def test_parse_incomplete_raises(self):
        """Incomplete expression raises ValueError."""
        with pytest.raises(ValueError, match="Unexpected end"):
            parse_prefix("+ NUM_0")

    def test_parse_unconsumed_tokens_raises(self):
        """Extra tokens raise ValueError."""
        with pytest.raises(ValueError, match="Unconsumed tokens"):
            parse_prefix("+ NUM_0 NUM_1 NUM_2")


class TestRecursiveDecomposition:
    """Tests for recursive decomposition to atomic steps."""

    def test_atomic_returns_single_step(self):
        """Already atomic expression returns single step."""
        tree = ExprNode(
            type=NodeType.OPERATOR,
            value="+",
            left=ExprNode(type=NodeType.VARIABLE, value="NUM_0"),
            right=ExprNode(type=NodeType.VARIABLE, value="NUM_1"),
        )
        steps = decompose_to_atomic(tree)
        assert len(steps) == 1
        assert steps[0][0] == "step_1"
        assert steps[0][1].is_atomic

    def test_leaf_returns_single_step(self):
        """Leaf node (already atomic) returns single step."""
        tree = ExprNode(type=NodeType.VARIABLE, value="NUM_0")
        steps = decompose_to_atomic(tree)
        assert len(steps) == 1

    def test_nested_decomposes_to_multiple_steps(self):
        """Nested expression decomposes to multiple atomic steps."""
        # - + NUM_0 NUM_1 NUM_2 -> step_1: (+ NUM_0 NUM_1), step_2: (- step_1 NUM_2)
        tree = parse_prefix("- + NUM_0 NUM_1 NUM_2")
        steps = decompose_to_atomic(tree)

        assert len(steps) == 2
        # First step should be the inner addition
        assert steps[0][2]['operation'] == 'add'
        # Second step should be subtraction with step reference
        assert steps[1][2]['operation'] == 'subtract'

    def test_deeply_nested_decomposes(self):
        """Deeply nested expression produces multiple steps."""
        # + + + NUM_0 NUM_1 NUM_2 NUM_3 (depth 3)
        tree = parse_prefix("+ + + NUM_0 NUM_1 NUM_2 NUM_3")
        steps = decompose_to_atomic(tree)

        assert len(steps) == 3
        # All steps should be atomic
        for step_name, step_tree, metadata in steps:
            assert step_tree.is_atomic

    def test_balanced_tree_decomposes(self):
        """Balanced tree decomposes correctly."""
        # * + NUM_0 NUM_1 - NUM_2 NUM_3
        tree = parse_prefix("* + NUM_0 NUM_1 - NUM_2 NUM_3")
        steps = decompose_to_atomic(tree)

        assert len(steps) == 3
        # Should have: add, subtract, multiply
        ops = [s[2]['operation'] for s in steps]
        assert 'add' in ops
        assert 'subtract' in ops
        assert 'multiply' in ops

    def test_metadata_contains_operands(self):
        """Metadata includes left and right operand info."""
        tree = ExprNode(
            type=NodeType.OPERATOR,
            value="+",
            left=ExprNode(type=NodeType.VARIABLE, value="NUM_0"),
            right=ExprNode(type=NodeType.VARIABLE, value="NUM_1"),
        )
        steps = decompose_to_atomic(tree)

        metadata = steps[0][2]
        assert metadata['operation'] == 'add'
        assert metadata['left_operand'] == 'NUM_0'
        assert metadata['right_operand'] == 'NUM_1'
        assert metadata['is_leaf'] is False

    def test_all_operators_decompose(self):
        """All operators can be decomposed."""
        for op in ['+', '-', '*', '/', '^']:
            tree = parse_prefix(f"{op} NUM_0 NUM_1")
            steps = decompose_to_atomic(tree)
            assert len(steps) == 1


class TestToOperationString:
    """Tests for operation string conversion."""

    def test_add_operation_string(self):
        """Addition converts to 'add two numbers'."""
        node = ExprNode(
            type=NodeType.OPERATOR,
            value="+",
            left=ExprNode(type=NodeType.VARIABLE, value="NUM_0"),
            right=ExprNode(type=NodeType.VARIABLE, value="NUM_1"),
        )
        assert node.to_operation_string() == "add two numbers"

    def test_subtract_operation_string(self):
        """Subtraction converts to 'subtract two numbers'."""
        node = ExprNode(
            type=NodeType.OPERATOR,
            value="-",
            left=ExprNode(type=NodeType.VARIABLE, value="NUM_0"),
            right=ExprNode(type=NodeType.VARIABLE, value="NUM_1"),
        )
        assert node.to_operation_string() == "subtract two numbers"

    def test_multiply_operation_string(self):
        """Multiplication converts to 'multiply two numbers'."""
        node = ExprNode(
            type=NodeType.OPERATOR,
            value="*",
            left=ExprNode(type=NodeType.VARIABLE, value="NUM_0"),
            right=ExprNode(type=NodeType.VARIABLE, value="NUM_1"),
        )
        assert node.to_operation_string() == "multiply two numbers"

    def test_divide_operation_string(self):
        """Division converts to 'divide two numbers'."""
        node = ExprNode(
            type=NodeType.OPERATOR,
            value="/",
            left=ExprNode(type=NodeType.VARIABLE, value="NUM_0"),
            right=ExprNode(type=NodeType.VARIABLE, value="NUM_1"),
        )
        assert node.to_operation_string() == "divide two numbers"

    def test_power_operation_string(self):
        """Power converts to 'power two numbers'."""
        node = ExprNode(
            type=NodeType.OPERATOR,
            value="^",
            left=ExprNode(type=NodeType.VARIABLE, value="NUM_0"),
            right=ExprNode(type=NodeType.VARIABLE, value="NUM_1"),
        )
        assert node.to_operation_string() == "power two numbers"

    def test_variable_operation_string(self):
        """Variable returns its value."""
        node = ExprNode(type=NodeType.VARIABLE, value="NUM_0")
        assert node.to_operation_string() == "NUM_0"


class TestToPrefixString:
    """Tests for converting back to prefix notation."""

    def test_variable_to_prefix(self):
        """Variable converts to its value."""
        node = ExprNode(type=NodeType.VARIABLE, value="NUM_0")
        assert node.to_prefix_string() == "NUM_0"

    def test_simple_operation_to_prefix(self):
        """Simple operation converts back to prefix."""
        node = ExprNode(
            type=NodeType.OPERATOR,
            value="+",
            left=ExprNode(type=NodeType.VARIABLE, value="NUM_0"),
            right=ExprNode(type=NodeType.VARIABLE, value="NUM_1"),
        )
        assert node.to_prefix_string() == "+ NUM_0 NUM_1"

    def test_nested_to_prefix(self):
        """Nested operation converts back to prefix."""
        tree = parse_prefix("- + NUM_0 NUM_1 NUM_2")
        assert tree.to_prefix_string() == "- + NUM_0 NUM_1 NUM_2"
