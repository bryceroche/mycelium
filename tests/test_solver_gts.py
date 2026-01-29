"""Tests for GTSDecomposer integration with Solver."""

import pytest
from unittest.mock import MagicMock, patch

from mycelium.solver import Solver
from mycelium.plan_models import Step, DAGPlan
from mycelium.gts_decomposer import DecomposedStep
from mycelium.expression_tree import ExprNode, NodeType


class TestConfigFlagExists:
    """Tests for USE_GTS_DECOMPOSITION config flag."""

    def test_config_flag_exists(self):
        """USE_GTS_DECOMPOSITION config flag exists and is bool."""
        from mycelium import config
        assert hasattr(config, "USE_GTS_DECOMPOSITION")
        assert isinstance(config.USE_GTS_DECOMPOSITION, bool)

    def test_gts_model_path_exists(self):
        """GTS_MODEL_PATH config exists and is string."""
        from mycelium import config
        assert hasattr(config, "GTS_MODEL_PATH")
        assert isinstance(config.GTS_MODEL_PATH, str)

    def test_default_value_is_false(self):
        """USE_GTS_DECOMPOSITION defaults to False."""
        from mycelium import config
        assert config.USE_GTS_DECOMPOSITION is False


class TestDecomposerImport:
    """Tests for GTSDecomposer import and initialization in Solver."""

    def test_solver_has_gts_decomposer_property(self):
        """Solver has gts_decomposer property."""
        solver = Solver()
        assert hasattr(solver, "gts_decomposer")

    def test_gts_decomposer_lazy_loaded(self):
        """GTSDecomposer is lazy-loaded."""
        solver = Solver()
        # Internal attribute should be None before access
        assert solver._gts_decomposer is None

    def test_use_gts_from_parameter(self):
        """use_gts parameter overrides config."""
        solver_true = Solver(use_gts=True)
        assert solver_true._use_gts is True

        solver_false = Solver(use_gts=False)
        assert solver_false._use_gts is False


class TestAtomicStepsToDagConversion:
    """Tests for converting GTS atomic steps to DAGPlan format."""

    def _make_simple_tree(self, op: str) -> ExprNode:
        """Helper to create a simple atomic tree."""
        return ExprNode(
            type=NodeType.OPERATOR,
            value=op,
            left=ExprNode(type=NodeType.VARIABLE, value="NUM_0"),
            right=ExprNode(type=NodeType.VARIABLE, value="NUM_1"),
        )

    def test_convert_single_step(self):
        """Convert single DecomposedStep to DAGPlan."""
        solver = Solver()

        decomposed_steps = [
            DecomposedStep(
                step_number=1,
                operation="add two numbers",
                expr_tree=self._make_simple_tree("+"),
                extracted_values={"NUM_0": 5.0, "NUM_1": 3.0},
                depends_on=[],
            )
        ]

        plan = solver._convert_gts_to_dag(decomposed_steps, "test problem")

        assert isinstance(plan, DAGPlan)
        assert len(plan.steps) == 1
        assert plan.problem == "test problem"

        step = plan.steps[0]
        assert step.id == "step_1"
        assert step.task == "add two numbers"
        assert step.extracted_values == {"NUM_0": 5.0, "NUM_1": 3.0}
        assert step.depends_on == []
        assert step.operation == "add"

    def test_convert_multiple_steps_with_dependencies(self):
        """Convert multiple DecomposedSteps with dependencies."""
        solver = Solver()

        decomposed_steps = [
            DecomposedStep(
                step_number=1,
                operation="add two numbers",
                expr_tree=self._make_simple_tree("+"),
                extracted_values={"NUM_0": 10.0, "NUM_1": 5.0},
                depends_on=[],
            ),
            DecomposedStep(
                step_number=2,
                operation="subtract two numbers",
                expr_tree=self._make_simple_tree("-"),
                extracted_values={"NUM_2": 3.0},
                depends_on=[1],
            ),
        ]

        plan = solver._convert_gts_to_dag(decomposed_steps, "nested problem")

        assert len(plan.steps) == 2

        # First step
        assert plan.steps[0].id == "step_1"
        assert plan.steps[0].depends_on == []
        assert plan.steps[0].operation == "add"

        # Second step
        assert plan.steps[1].id == "step_2"
        assert plan.steps[1].depends_on == ["step_1"]
        assert plan.steps[1].operation == "subtract"

    def test_convert_step_with_multiple_dependencies(self):
        """Convert step that depends on multiple previous steps."""
        solver = Solver()

        decomposed_steps = [
            DecomposedStep(
                step_number=1,
                operation="add two numbers",
                expr_tree=self._make_simple_tree("+"),
                extracted_values={"NUM_0": 1.0, "NUM_1": 2.0},
                depends_on=[],
            ),
            DecomposedStep(
                step_number=2,
                operation="subtract two numbers",
                expr_tree=self._make_simple_tree("-"),
                extracted_values={"NUM_2": 3.0, "NUM_3": 4.0},
                depends_on=[],
            ),
            DecomposedStep(
                step_number=3,
                operation="multiply two numbers",
                expr_tree=self._make_simple_tree("*"),
                extracted_values={},
                depends_on=[1, 2],
            ),
        ]

        plan = solver._convert_gts_to_dag(decomposed_steps, "balanced tree")

        assert len(plan.steps) == 3
        assert plan.steps[2].depends_on == ["step_1", "step_2"]
        assert plan.steps[2].operation == "multiply"


class TestGtsDecompositionFallback:
    """Tests for GTS decomposition with fallback behavior."""

    @pytest.mark.asyncio
    async def test_decompose_with_gts_raises_not_implemented(self):
        """_decompose_with_gts raises NotImplementedError (beam search not impl)."""
        solver = Solver(use_gts=True)

        # Mock the internal decomposer attribute
        mock_decomposer = MagicMock()
        mock_decomposer.decompose.side_effect = NotImplementedError(
            "GTS beam search decoding not yet implemented"
        )
        solver._gts_decomposer = mock_decomposer

        # _decompose should handle the NotImplementedError gracefully
        result = await solver._decompose("test problem")
        assert result is None

    @pytest.mark.asyncio
    async def test_decompose_falls_back_on_not_implemented(self):
        """_decompose falls back gracefully when GTS raises NotImplementedError."""
        solver = Solver(use_gts=True)

        # Directly test _decompose behavior
        with patch.object(solver, "_decompose_with_gts") as mock_gts:
            mock_gts.side_effect = NotImplementedError("beam search not impl")

            result = await solver._decompose("test problem")
            assert result is None

    @pytest.mark.asyncio
    async def test_decompose_falls_back_on_exception(self):
        """_decompose falls back gracefully on any exception."""
        solver = Solver(use_gts=True)

        with patch.object(solver, "_decompose_with_gts") as mock_gts:
            mock_gts.side_effect = RuntimeError("unexpected error")

            result = await solver._decompose("test problem")
            assert result is None

    @pytest.mark.asyncio
    async def test_decompose_without_gts_returns_none(self):
        """_decompose returns None when GTS is disabled."""
        solver = Solver(use_gts=False)

        result = await solver._decompose("test problem")
        assert result is None


class TestGtsDecompositionSuccess:
    """Tests for successful GTS decomposition flow."""

    def _make_simple_tree(self, op: str) -> ExprNode:
        """Helper to create a simple atomic tree."""
        return ExprNode(
            type=NodeType.OPERATOR,
            value=op,
            left=ExprNode(type=NodeType.VARIABLE, value="NUM_0"),
            right=ExprNode(type=NodeType.VARIABLE, value="NUM_1"),
        )

    @pytest.mark.asyncio
    async def test_decompose_with_gts_success(self):
        """_decompose_with_gts returns DAGPlan on success."""
        solver = Solver(use_gts=True)

        mock_steps = [
            DecomposedStep(
                step_number=1,
                operation="add two numbers",
                expr_tree=self._make_simple_tree("+"),
                extracted_values={"NUM_0": 5.0, "NUM_1": 3.0},
                depends_on=[],
            )
        ]

        # Mock the internal decomposer attribute
        mock_decomposer = MagicMock()
        mock_decomposer.decompose.return_value = mock_steps
        solver._gts_decomposer = mock_decomposer

        result = await solver._decompose_with_gts("test problem")

        assert isinstance(result, DAGPlan)
        assert len(result.steps) == 1
        assert result.problem == "test problem"

    @pytest.mark.asyncio
    async def test_decompose_with_gts_empty_steps(self):
        """_decompose_with_gts returns None when GTS returns empty list."""
        solver = Solver(use_gts=True)

        # Mock the internal decomposer attribute
        mock_decomposer = MagicMock()
        mock_decomposer.decompose.return_value = []
        solver._gts_decomposer = mock_decomposer

        result = await solver._decompose_with_gts("test problem")
        assert result is None


class TestStepOperationExtraction:
    """Tests for extracting operation from operation string."""

    def _make_simple_tree(self, op: str) -> ExprNode:
        """Helper to create a simple atomic tree."""
        return ExprNode(
            type=NodeType.OPERATOR,
            value=op,
            left=ExprNode(type=NodeType.VARIABLE, value="NUM_0"),
            right=ExprNode(type=NodeType.VARIABLE, value="NUM_1"),
        )

    def test_extract_add_operation(self):
        """Extract 'add' from 'add two numbers'."""
        solver = Solver()
        steps = [
            DecomposedStep(
                step_number=1,
                operation="add two numbers",
                expr_tree=self._make_simple_tree("+"),
                extracted_values={},
                depends_on=[],
            )
        ]
        plan = solver._convert_gts_to_dag(steps, "test")
        assert plan.steps[0].operation == "add"

    def test_extract_subtract_operation(self):
        """Extract 'subtract' from 'subtract two numbers'."""
        solver = Solver()
        steps = [
            DecomposedStep(
                step_number=1,
                operation="subtract two numbers",
                expr_tree=self._make_simple_tree("-"),
                extracted_values={},
                depends_on=[],
            )
        ]
        plan = solver._convert_gts_to_dag(steps, "test")
        assert plan.steps[0].operation == "subtract"

    def test_extract_multiply_operation(self):
        """Extract 'multiply' from 'multiply two numbers'."""
        solver = Solver()
        steps = [
            DecomposedStep(
                step_number=1,
                operation="multiply two numbers",
                expr_tree=self._make_simple_tree("*"),
                extracted_values={},
                depends_on=[],
            )
        ]
        plan = solver._convert_gts_to_dag(steps, "test")
        assert plan.steps[0].operation == "multiply"

    def test_extract_divide_operation(self):
        """Extract 'divide' from 'divide two numbers'."""
        solver = Solver()
        steps = [
            DecomposedStep(
                step_number=1,
                operation="divide two numbers",
                expr_tree=self._make_simple_tree("/"),
                extracted_values={},
                depends_on=[],
            )
        ]
        plan = solver._convert_gts_to_dag(steps, "test")
        assert plan.steps[0].operation == "divide"

    def test_empty_operation_string(self):
        """Handle empty operation string - converts to None."""
        solver = Solver()
        steps = [
            DecomposedStep(
                step_number=1,
                operation="",
                expr_tree=self._make_simple_tree("+"),
                extracted_values={},
                depends_on=[],
            )
        ]
        plan = solver._convert_gts_to_dag(steps, "test")
        # Empty string split returns [''], first element is '' which is falsy
        # So operation gets set to None via the conditional
        assert plan.steps[0].operation is None

    def test_none_operation_string(self):
        """Handle None operation string."""
        solver = Solver()
        steps = [
            DecomposedStep(
                step_number=1,
                operation=None,
                expr_tree=self._make_simple_tree("+"),
                extracted_values={},
                depends_on=[],
            )
        ]
        plan = solver._convert_gts_to_dag(steps, "test")
        assert plan.steps[0].operation is None
