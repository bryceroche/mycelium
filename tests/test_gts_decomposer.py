"""Tests for GTSDecomposer wrapper."""

import pytest
from mycelium.gts_decomposer import GTSDecomposer, DecomposedStep
from mycelium.expression_tree import NodeType


class TestNumberExtraction:
    """Tests for extracting numbers from problem text."""

    def test_extract_single_integer(self):
        """Extract a single integer from text."""
        decomposer = GTSDecomposer.__new__(GTSDecomposer)
        decomposer.model = None  # Skip model loading

        numbers, normalized = decomposer._extract_numbers("I have 5 apples")
        assert numbers == {"NUM_0": 5.0}
        assert "NUM" in normalized
        assert "5" not in normalized

    def test_extract_multiple_integers(self):
        """Extract multiple integers from text."""
        decomposer = GTSDecomposer.__new__(GTSDecomposer)
        decomposer.model = None

        numbers, normalized = decomposer._extract_numbers("I have 5 apples and 3 oranges")
        assert numbers == {"NUM_0": 5.0, "NUM_1": 3.0}

    def test_extract_floats(self):
        """Extract floating point numbers."""
        decomposer = GTSDecomposer.__new__(GTSDecomposer)
        decomposer.model = None

        numbers, normalized = decomposer._extract_numbers("The price is 3.50 dollars")
        assert numbers == {"NUM_0": 3.5}

    def test_extract_mixed_integers_and_floats(self):
        """Extract mix of integers and floats."""
        decomposer = GTSDecomposer.__new__(GTSDecomposer)
        decomposer.model = None

        numbers, normalized = decomposer._extract_numbers("Buy 2 items at 4.99 each")
        assert numbers == {"NUM_0": 2.0, "NUM_1": 4.99}

    def test_extract_preserves_order(self):
        """Numbers are extracted in order of appearance."""
        decomposer = GTSDecomposer.__new__(GTSDecomposer)
        decomposer.model = None

        numbers, _ = decomposer._extract_numbers("First 10, then 20, finally 30")
        assert numbers["NUM_0"] == 10.0
        assert numbers["NUM_1"] == 20.0
        assert numbers["NUM_2"] == 30.0


class TestDecomposeFromPrefixSimple:
    """Tests for decompose_from_prefix with simple expressions."""

    def test_simple_addition(self):
        """Decompose simple addition: + NUM_0 NUM_1."""
        decomposer = GTSDecomposer.__new__(GTSDecomposer)
        decomposer.model = None

        steps = decomposer.decompose_from_prefix(
            "+ NUM_0 NUM_1",
            {"NUM_0": 5.0, "NUM_1": 3.0}
        )

        assert len(steps) == 1
        step = steps[0]
        assert step.step_number == 1
        assert step.operation == "add two numbers"
        assert step.extracted_values == {"NUM_0": 5.0, "NUM_1": 3.0}
        assert step.depends_on == []

    def test_simple_subtraction(self):
        """Decompose simple subtraction: - NUM_0 NUM_1."""
        decomposer = GTSDecomposer.__new__(GTSDecomposer)
        decomposer.model = None

        steps = decomposer.decompose_from_prefix(
            "- NUM_0 NUM_1",
            {"NUM_0": 10.0, "NUM_1": 4.0}
        )

        assert len(steps) == 1
        assert steps[0].operation == "subtract two numbers"

    def test_simple_multiplication(self):
        """Decompose simple multiplication: * NUM_0 NUM_1."""
        decomposer = GTSDecomposer.__new__(GTSDecomposer)
        decomposer.model = None

        steps = decomposer.decompose_from_prefix(
            "* NUM_0 NUM_1",
            {"NUM_0": 6.0, "NUM_1": 7.0}
        )

        assert len(steps) == 1
        assert steps[0].operation == "multiply two numbers"

    def test_simple_division(self):
        """Decompose simple division: / NUM_0 NUM_1."""
        decomposer = GTSDecomposer.__new__(GTSDecomposer)
        decomposer.model = None

        steps = decomposer.decompose_from_prefix(
            "/ NUM_0 NUM_1",
            {"NUM_0": 20.0, "NUM_1": 5.0}
        )

        assert len(steps) == 1
        assert steps[0].operation == "divide two numbers"


class TestDecomposeFromPrefixComplex:
    """Tests for decompose_from_prefix with complex nested expressions."""

    def test_nested_two_operations(self):
        """Decompose nested: - + NUM_0 NUM_1 NUM_2."""
        decomposer = GTSDecomposer.__new__(GTSDecomposer)
        decomposer.model = None

        steps = decomposer.decompose_from_prefix(
            "- + NUM_0 NUM_1 NUM_2",
            {"NUM_0": 10.0, "NUM_1": 5.0, "NUM_2": 3.0}
        )

        assert len(steps) == 2

        # First step: add NUM_0 + NUM_1
        assert steps[0].step_number == 1
        assert steps[0].operation == "add two numbers"
        assert steps[0].extracted_values == {"NUM_0": 10.0, "NUM_1": 5.0}
        assert steps[0].depends_on == []

        # Second step: subtract step_1 - NUM_2
        assert steps[1].step_number == 2
        assert steps[1].operation == "subtract two numbers"
        assert steps[1].extracted_values == {"NUM_2": 3.0}
        assert steps[1].depends_on == [1]

    def test_deeply_nested_three_operations(self):
        """Decompose deeply nested: + + + NUM_0 NUM_1 NUM_2 NUM_3."""
        decomposer = GTSDecomposer.__new__(GTSDecomposer)
        decomposer.model = None

        steps = decomposer.decompose_from_prefix(
            "+ + + NUM_0 NUM_1 NUM_2 NUM_3",
            {"NUM_0": 1.0, "NUM_1": 2.0, "NUM_2": 3.0, "NUM_3": 4.0}
        )

        assert len(steps) == 3
        # All steps should be add operations
        for step in steps:
            assert step.operation == "add two numbers"

    def test_balanced_tree_decomposition(self):
        """Decompose balanced: * + NUM_0 NUM_1 - NUM_2 NUM_3."""
        decomposer = GTSDecomposer.__new__(GTSDecomposer)
        decomposer.model = None

        steps = decomposer.decompose_from_prefix(
            "* + NUM_0 NUM_1 - NUM_2 NUM_3",
            {"NUM_0": 2.0, "NUM_1": 3.0, "NUM_2": 10.0, "NUM_3": 4.0}
        )

        assert len(steps) == 3

        # Collect operations
        operations = [s.operation for s in steps]
        assert "add two numbers" in operations
        assert "subtract two numbers" in operations
        assert "multiply two numbers" in operations

    def test_mixed_operations(self):
        """Decompose mixed: / * NUM_0 NUM_1 + NUM_2 NUM_3."""
        decomposer = GTSDecomposer.__new__(GTSDecomposer)
        decomposer.model = None

        steps = decomposer.decompose_from_prefix(
            "/ * NUM_0 NUM_1 + NUM_2 NUM_3",
            {"NUM_0": 6.0, "NUM_1": 4.0, "NUM_2": 2.0, "NUM_3": 1.0}
        )

        assert len(steps) == 3
        operations = [s.operation for s in steps]
        assert "multiply two numbers" in operations
        assert "add two numbers" in operations
        assert "divide two numbers" in operations


class TestStepDependencies:
    """Tests for tracking step dependencies."""

    def test_no_dependencies_for_first_step(self):
        """First step has no dependencies."""
        decomposer = GTSDecomposer.__new__(GTSDecomposer)
        decomposer.model = None

        steps = decomposer.decompose_from_prefix(
            "- + NUM_0 NUM_1 NUM_2",
            {"NUM_0": 1.0, "NUM_1": 2.0, "NUM_2": 3.0}
        )

        assert steps[0].depends_on == []

    def test_dependency_on_previous_step(self):
        """Later step depends on earlier step."""
        decomposer = GTSDecomposer.__new__(GTSDecomposer)
        decomposer.model = None

        steps = decomposer.decompose_from_prefix(
            "* + NUM_0 NUM_1 NUM_2",
            {"NUM_0": 2.0, "NUM_1": 3.0, "NUM_2": 4.0}
        )

        # Second step should depend on first
        assert steps[1].depends_on == [1]

    def test_multiple_dependencies(self):
        """Step can depend on multiple previous steps."""
        decomposer = GTSDecomposer.__new__(GTSDecomposer)
        decomposer.model = None

        # * + NUM_0 NUM_1 - NUM_2 NUM_3 produces:
        # step_1: + NUM_0 NUM_1
        # step_2: - NUM_2 NUM_3
        # step_3: * step_1 step_2
        steps = decomposer.decompose_from_prefix(
            "* + NUM_0 NUM_1 - NUM_2 NUM_3",
            {"NUM_0": 1.0, "NUM_1": 2.0, "NUM_2": 3.0, "NUM_3": 4.0}
        )

        # Find the multiply step (should be last)
        multiply_step = [s for s in steps if s.operation == "multiply two numbers"][0]
        # It should depend on both step_1 and step_2
        assert 1 in multiply_step.depends_on
        assert 2 in multiply_step.depends_on

    def test_dependencies_sorted(self):
        """Dependencies are returned in sorted order."""
        decomposer = GTSDecomposer.__new__(GTSDecomposer)
        decomposer.model = None

        steps = decomposer.decompose_from_prefix(
            "* + NUM_0 NUM_1 - NUM_2 NUM_3",
            {"NUM_0": 1.0, "NUM_1": 2.0, "NUM_2": 3.0, "NUM_3": 4.0}
        )

        for step in steps:
            assert step.depends_on == sorted(step.depends_on)


class TestExtractedValues:
    """Tests for extracted value tracking."""

    def test_all_values_in_simple_step(self):
        """Simple step includes all its values."""
        decomposer = GTSDecomposer.__new__(GTSDecomposer)
        decomposer.model = None

        steps = decomposer.decompose_from_prefix(
            "+ NUM_0 NUM_1",
            {"NUM_0": 100.0, "NUM_1": 200.0}
        )

        assert steps[0].extracted_values == {"NUM_0": 100.0, "NUM_1": 200.0}

    def test_only_used_values_extracted(self):
        """Only values used in step are extracted, not from dependencies."""
        decomposer = GTSDecomposer.__new__(GTSDecomposer)
        decomposer.model = None

        steps = decomposer.decompose_from_prefix(
            "* + NUM_0 NUM_1 NUM_2",
            {"NUM_0": 1.0, "NUM_1": 2.0, "NUM_2": 3.0}
        )

        # First step uses NUM_0 and NUM_1
        assert steps[0].extracted_values == {"NUM_0": 1.0, "NUM_1": 2.0}

        # Second step uses only NUM_2 (step_1 is a dependency, not a value)
        assert steps[1].extracted_values == {"NUM_2": 3.0}

    def test_no_extracted_values_for_step_refs_only(self):
        """Step with only step references has no extracted values."""
        decomposer = GTSDecomposer.__new__(GTSDecomposer)
        decomposer.model = None

        steps = decomposer.decompose_from_prefix(
            "* + NUM_0 NUM_1 - NUM_2 NUM_3",
            {"NUM_0": 1.0, "NUM_1": 2.0, "NUM_2": 3.0, "NUM_3": 4.0}
        )

        # The multiply step uses step_1 and step_2, no direct NUM values
        multiply_step = [s for s in steps if s.operation == "multiply two numbers"][0]
        assert multiply_step.extracted_values == {}

    def test_extracted_values_preserve_float(self):
        """Extracted values preserve float type."""
        decomposer = GTSDecomposer.__new__(GTSDecomposer)
        decomposer.model = None

        steps = decomposer.decompose_from_prefix(
            "+ NUM_0 NUM_1",
            {"NUM_0": 3.14159, "NUM_1": 2.71828}
        )

        assert steps[0].extracted_values["NUM_0"] == 3.14159
        assert steps[0].extracted_values["NUM_1"] == 2.71828


class TestExprTreeInSteps:
    """Tests for expression tree in decomposed steps."""

    def test_step_has_atomic_tree(self):
        """Each step's expr_tree is atomic."""
        decomposer = GTSDecomposer.__new__(GTSDecomposer)
        decomposer.model = None

        steps = decomposer.decompose_from_prefix(
            "- + NUM_0 NUM_1 NUM_2",
            {"NUM_0": 1.0, "NUM_1": 2.0, "NUM_2": 3.0}
        )

        for step in steps:
            assert step.expr_tree.is_atomic

    def test_tree_preserves_operator(self):
        """Expression tree preserves the operator."""
        decomposer = GTSDecomposer.__new__(GTSDecomposer)
        decomposer.model = None

        steps = decomposer.decompose_from_prefix(
            "* NUM_0 NUM_1",
            {"NUM_0": 2.0, "NUM_1": 3.0}
        )

        assert steps[0].expr_tree.type == NodeType.OPERATOR
        assert steps[0].expr_tree.value == "*"


class TestDecomposeFullPipeline:
    """Tests for the full decompose() method."""

    def test_decompose_raises_not_implemented(self):
        """decompose() raises NotImplementedError for beam search."""
        decomposer = GTSDecomposer.__new__(GTSDecomposer)
        decomposer.model = None
        decomposer._model_path = "trained_model/GTS-mawps"

        with pytest.raises(NotImplementedError, match="beam search"):
            decomposer.decompose("If I have 5 apples and buy 3 more, how many do I have?")


class TestDecomposedStepDataclass:
    """Tests for DecomposedStep dataclass."""

    def test_dataclass_fields(self):
        """DecomposedStep has all required fields."""
        from mycelium.expression_tree import ExprNode, NodeType

        tree = ExprNode(
            type=NodeType.OPERATOR,
            value="+",
            left=ExprNode(type=NodeType.VARIABLE, value="NUM_0"),
            right=ExprNode(type=NodeType.VARIABLE, value="NUM_1"),
        )

        step = DecomposedStep(
            step_number=1,
            operation="add two numbers",
            expr_tree=tree,
            extracted_values={"NUM_0": 5.0, "NUM_1": 3.0},
            depends_on=[],
        )

        assert step.step_number == 1
        assert step.operation == "add two numbers"
        assert step.expr_tree == tree
        assert step.extracted_values == {"NUM_0": 5.0, "NUM_1": 3.0}
        assert step.depends_on == []
