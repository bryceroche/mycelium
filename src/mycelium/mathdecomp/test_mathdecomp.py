"""
Tests for mathdecomp module.

Run with: python -m mycelium.mathdecomp.test_mathdecomp
"""

from .schema import Ref, RefType, Extraction, Step, Decomposition
from .executor import execute_decomposition, verify_decomposition, trace_execution
from .decomposer import mock_decompose


def test_ref():
    """Test Ref creation and resolution."""
    print("Testing Ref...")

    # Test shorthand constructors
    r1 = Ref.extraction("num_toys")
    assert r1.type == RefType.EXTRACTION
    assert r1.id == "num_toys"

    r2 = Ref.step("s1")
    assert r2.type == RefType.STEP
    assert r2.id == "s1"

    r3 = Ref.constant(0.5)
    assert r3.type == RefType.CONSTANT
    assert r3.id == "0.5"

    # Test resolution
    extractions = {"num_toys": 3, "toy_price": 2}
    step_results = {"s1": 6}

    assert r1.resolve(extractions, step_results) == 3
    assert r2.resolve(extractions, step_results) == 6
    assert r3.resolve(extractions, step_results) == 0.5

    print("  ✓ Ref tests passed")


def test_step():
    """Test Step creation and dependencies."""
    print("Testing Step...")

    step = Step(
        id="s1",
        func="mul",
        inputs=[Ref.extraction("num_toys"), Ref.extraction("toy_price")],
        result=6,
        semantic="total_cost",
    )

    assert step.dependencies() == []  # No step dependencies

    step2 = Step(
        id="s2",
        func="sub",
        inputs=[Ref.extraction("tim_money"), Ref.step("s1")],
        result=4,
        semantic="remaining",
    )

    assert step2.dependencies() == ["s1"]

    print("  ✓ Step tests passed")


def test_decomposition():
    """Test full decomposition flow."""
    print("Testing Decomposition...")

    decomp = Decomposition(
        problem="Tim has 10 dollars. He buys 3 toys at 2 dollars each. How much left?",
        extractions=[
            Extraction(
                id="tim_money", value=10, span="10 dollars",
                offset_start=8, offset_end=18
            ),
            Extraction(
                id="num_toys", value=3, span="3 toys",
                offset_start=28, offset_end=34
            ),
            Extraction(
                id="toy_price", value=2, span="2 dollars each",
                offset_start=38, offset_end=52
            ),
        ],
        steps=[
            Step(
                id="s1", func="mul",
                inputs=[Ref.extraction("num_toys"), Ref.extraction("toy_price")],
                result=6, semantic="total_cost"
            ),
            Step(
                id="s2", func="sub",
                inputs=[Ref.extraction("tim_money"), Ref.step("s1")],
                result=4, semantic="remaining_money"
            ),
        ],
        answer_ref=Ref.step("s2"),
        answer_value=4,
    )

    # Test dependency order
    order = decomp.dependency_order()
    assert order == ["s1", "s2"], f"Expected ['s1', 's2'], got {order}"

    # Test execution
    step_results, error = execute_decomposition(decomp)
    assert error is None, f"Execution error: {error}"
    assert step_results == {"s1": 6.0, "s2": 4.0}

    # Test verification
    decomp = verify_decomposition(decomp, expected_answer=4)
    assert decomp.verified, f"Verification failed: {decomp.error}"

    print("  ✓ Decomposition tests passed")


def test_serialization():
    """Test JSON serialization/deserialization."""
    print("Testing serialization...")

    decomp = Decomposition(
        problem="5 + 3 = ?",
        extractions=[
            Extraction(id="a", value=5, span="5", offset_start=0, offset_end=1),
            Extraction(id="b", value=3, span="3", offset_start=4, offset_end=5),
        ],
        steps=[
            Step(
                id="s1", func="add",
                inputs=[Ref.extraction("a"), Ref.extraction("b")],
                result=8, semantic="sum"
            ),
        ],
        answer_ref=Ref.step("s1"),
        answer_value=8,
        verified=True,
    )

    # Serialize
    json_str = decomp.to_json()
    print(f"  JSON: {json_str[:100]}...")

    # Deserialize
    decomp2 = Decomposition.from_json(json_str)
    assert decomp2.problem == decomp.problem
    assert decomp2.answer_value == decomp.answer_value
    assert len(decomp2.steps) == len(decomp.steps)

    print("  ✓ Serialization tests passed")


def test_trace():
    """Test execution trace generation."""
    print("Testing trace...")

    decomp = Decomposition(
        problem="Tim has 10 dollars. He buys 3 toys at 2 dollars each. How much left?",
        extractions=[
            Extraction(
                id="tim_money", value=10, span="10 dollars",
                offset_start=8, offset_end=18
            ),
            Extraction(
                id="num_toys", value=3, span="3 toys",
                offset_start=28, offset_end=34
            ),
            Extraction(
                id="toy_price", value=2, span="2 dollars each",
                offset_start=38, offset_end=52
            ),
        ],
        steps=[
            Step(
                id="s1", func="mul",
                inputs=[Ref.extraction("num_toys"), Ref.extraction("toy_price")],
                result=6, semantic="total_cost"
            ),
            Step(
                id="s2", func="sub",
                inputs=[Ref.extraction("tim_money"), Ref.step("s1")],
                result=4, semantic="remaining_money"
            ),
        ],
        answer_ref=Ref.step("s2"),
        answer_value=4,
        verified=True,
    )

    trace = trace_execution(decomp)
    print(f"\n{trace}\n")

    assert "tim_money = 10" in trace
    assert "total_cost" in trace
    assert "Verified: True" in trace

    print("  ✓ Trace tests passed")


def test_mock_decompose():
    """Test mock decomposition."""
    print("Testing mock_decompose...")

    decomp = mock_decompose("John has 5 apples. Mary gives him 3 more. What is the total?")
    print(f"  Mock result: {decomp.answer_value}")
    assert decomp.answer_value == 8  # 5 + 3

    decomp2 = mock_decompose("There are 12 cookies split among 4 kids. How many per kid?")
    print(f"  Mock result: {decomp2.answer_value}")
    assert decomp2.answer_value == 3  # 12 / 4

    print("  ✓ mock_decompose tests passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("mathdecomp module tests")
    print("=" * 60)

    test_ref()
    test_step()
    test_decomposition()
    test_serialization()
    test_trace()
    test_mock_decompose()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
