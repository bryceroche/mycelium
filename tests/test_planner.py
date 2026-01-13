"""Tests for DAG planning and validation."""

import pytest
from mycelium.planner import Step, DAGPlan, PlanValidationError

# Helper to create DAGPlan with default problem
def make_plan(steps, problem="test problem"):
    return DAGPlan(steps=steps, problem=problem)


class TestStep:
    """Tests for Step dataclass."""

    def test_atomic_step(self):
        step = Step(id="step1", task="Do something")
        assert step.is_atomic
        assert not step.is_composite
        assert step.max_depth() == 0
        assert step.total_steps() == 1

    def test_composite_step(self):
        sub_plan = make_plan(steps=[
            Step(id="sub1", task="Sub task 1"),
            Step(id="sub2", task="Sub task 2"),
        ])
        step = Step(id="step1", task="Composite task", sub_plan=sub_plan)
        assert step.is_composite
        assert not step.is_atomic
        assert step.max_depth() == 1
        assert step.total_steps() == 3  # 1 + 2 sub-steps

    def test_deeply_nested_step(self):
        # Create 2 levels of nesting: outer -> middle -> inner
        # outer.max_depth() = 1 + middle.max_depth()
        # middle has one step with sub_plan=inner, so middle.max_depth() = 1
        # outer.max_depth() = 1 + 1 = 2
        inner = make_plan(steps=[Step(id="inner", task="Inner")])
        middle = make_plan(steps=[Step(id="middle", task="Middle", sub_plan=inner)])
        outer = Step(id="outer", task="Outer", sub_plan=middle)

        assert outer.max_depth() == 2
        assert outer.total_steps() == 3

    def test_flatten(self):
        sub_plan = make_plan(steps=[
            Step(id="sub1", task="Sub 1"),
            Step(id="sub2", task="Sub 2"),
        ])
        step = Step(id="main", task="Main", sub_plan=sub_plan)

        flattened = step.flatten()
        paths = [path for path, _ in flattened]

        assert "main" in paths
        assert "main/sub1" in paths
        assert "main/sub2" in paths

    def test_dependencies(self):
        step = Step(id="step2", task="Step 2", depends_on=["step1"])
        assert step.depends_on == ["step1"]


class TestDAGPlan:
    """Tests for DAGPlan validation and execution order."""

    def test_empty_plan_is_valid(self):
        plan = make_plan(steps=[])
        is_valid, errors = plan.validate()
        assert is_valid
        assert errors == []

    def test_simple_linear_plan(self):
        plan = make_plan(steps=[
            Step(id="step1", task="First"),
            Step(id="step2", task="Second", depends_on=["step1"]),
            Step(id="step3", task="Third", depends_on=["step2"]),
        ])
        is_valid, errors = plan.validate()
        assert is_valid
        assert errors == []

    def test_parallel_steps(self):
        plan = make_plan(steps=[
            Step(id="step1", task="First"),
            Step(id="step2a", task="Parallel A", depends_on=["step1"]),
            Step(id="step2b", task="Parallel B", depends_on=["step1"]),
            Step(id="step3", task="Join", depends_on=["step2a", "step2b"]),
        ])
        is_valid, errors = plan.validate()
        assert is_valid
        assert errors == []

    def test_missing_dependency(self):
        plan = make_plan(steps=[
            Step(id="step1", task="First"),
            Step(id="step2", task="Second", depends_on=["nonexistent"]),
        ])
        is_valid, errors = plan.validate()
        assert not is_valid
        assert len(errors) == 1
        assert "nonexistent" in errors[0]

    def test_cycle_detection_simple(self):
        # A -> B -> A (simple cycle)
        plan = make_plan(steps=[
            Step(id="A", task="A", depends_on=["B"]),
            Step(id="B", task="B", depends_on=["A"]),
        ])
        is_valid, errors = plan.validate()
        assert not is_valid
        assert any("cyclic" in e.lower() or "cycle" in e.lower() for e in errors)

    def test_cycle_detection_self_reference(self):
        # A -> A (self cycle)
        plan = make_plan(steps=[
            Step(id="A", task="A", depends_on=["A"]),
        ])
        is_valid, errors = plan.validate()
        assert not is_valid

    def test_cycle_detection_complex(self):
        # A -> B -> C -> A (longer cycle)
        plan = make_plan(steps=[
            Step(id="A", task="A", depends_on=["C"]),
            Step(id="B", task="B", depends_on=["A"]),
            Step(id="C", task="C", depends_on=["B"]),
        ])
        is_valid, errors = plan.validate()
        assert not is_valid
        assert any("cyclic" in e.lower() or "cycle" in e.lower() for e in errors)

    def test_nested_plan_validation(self):
        # Valid outer, invalid inner (cycle)
        inner_plan = make_plan(steps=[
            Step(id="inner1", task="Inner 1", depends_on=["inner2"]),
            Step(id="inner2", task="Inner 2", depends_on=["inner1"]),
        ])
        plan = make_plan(steps=[
            Step(id="outer1", task="Outer", sub_plan=inner_plan),
        ])
        is_valid, errors = plan.validate()
        assert not is_valid
        assert any("outer1" in e for e in errors)  # Error should reference parent

    def test_max_depth(self):
        plan = make_plan(steps=[
            Step(id="step1", task="Simple"),
        ])
        assert plan.max_depth() == 0

        nested = make_plan(steps=[
            Step(id="step1", task="Nested", sub_plan=plan),
        ])
        assert nested.max_depth() == 1

    def test_total_steps(self):
        plan = make_plan(steps=[
            Step(id="step1", task="One"),
            Step(id="step2", task="Two"),
            Step(id="step3", task="Three"),
        ])
        assert plan.total_steps() == 3

    def test_get_execution_order_simple(self):
        plan = make_plan(steps=[
            Step(id="step1", task="First"),
            Step(id="step2", task="Second", depends_on=["step1"]),
        ])
        levels = plan.get_execution_order()

        # step1 should be in level 0, step2 in level 1
        level0_ids = [s.id for s in levels[0]]
        level1_ids = [s.id for s in levels[1]]

        assert "step1" in level0_ids
        assert "step2" in level1_ids

    def test_get_execution_order_parallel(self):
        plan = make_plan(steps=[
            Step(id="step1", task="First"),
            Step(id="step2a", task="Parallel A", depends_on=["step1"]),
            Step(id="step2b", task="Parallel B", depends_on=["step1"]),
        ])
        levels = plan.get_execution_order()

        # step2a and step2b should be in the same level
        level1_ids = [s.id for s in levels[1]]
        assert "step2a" in level1_ids
        assert "step2b" in level1_ids


class TestPlanValidationError:
    """Tests for PlanValidationError exception."""

    def test_exception_raised_on_strict_validation(self):
        plan = make_plan(steps=[
            Step(id="A", task="A", depends_on=["B"]),
            Step(id="B", task="B", depends_on=["A"]),
        ])

        with pytest.raises(PlanValidationError):
            plan.get_execution_order(strict=True)

    def test_no_exception_on_non_strict(self):
        plan = make_plan(steps=[
            Step(id="A", task="A", depends_on=["B"]),
            Step(id="B", task="B", depends_on=["A"]),
        ])

        # Should not raise, but return partial result
        levels = plan.get_execution_order(strict=False)
        # Result may be empty or partial for invalid DAG
        assert isinstance(levels, list)
