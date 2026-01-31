"""
Grader for math problem decomposition quality.

Implements 10 core rules for grading decomposition quality:
- Structural Rules (1-7): Auto-gradable structural checks
- Semantic Rules (8-10): Structural checks with semantic awareness

Core principle: Grade against explicit schema, not inferred conventions.
"""

from dataclasses import dataclass
from typing import List, Set, Dict, Optional
from mycelium.mathdecomp.schema import Decomposition, Step, Ref, RefType, Extraction


# Valid operators as defined in schema
VALID_OPERATORS = {"+", "-", "*", "/"}


@dataclass
class GradeResult:
    """Result of grading a single rule."""
    rule_number: int
    rule_name: str
    passed: bool
    message: str


def grade_rule_1_atomic(decomp: Decomposition) -> GradeResult:
    """
    Rule 1: Atomic Operations
    Each step must have exactly one operator (op field is not empty).

    In this schema, atomicity is enforced by structure: each Step has
    exactly one op, one left, one right. We verify op is a valid operator.
    """
    rule_number = 1
    rule_name = "Atomic Operations"

    if not decomp.steps:
        return GradeResult(rule_number, rule_name, True, "No steps to check (trivial pass)")

    invalid_steps = []
    for step in decomp.steps:
        if not step.op or step.op not in VALID_OPERATORS:
            invalid_steps.append(f"{step.id}: op='{step.op}'")

    if invalid_steps:
        return GradeResult(
            rule_number, rule_name, False,
            f"Steps with invalid/missing operator: {', '.join(invalid_steps)}"
        )

    return GradeResult(rule_number, rule_name, True, f"All {len(decomp.steps)} steps have valid atomic operators")


def grade_rule_2_explicit_refs(decomp: Decomposition) -> GradeResult:
    """
    Rule 2: Explicit References
    All inputs (left and right in each step) must be typed Ref objects.

    Verifies that each step's left and right are proper Ref instances
    with valid RefType.
    """
    rule_number = 2
    rule_name = "Explicit References"

    if not decomp.steps:
        return GradeResult(rule_number, rule_name, True, "No steps to check (trivial pass)")

    invalid_refs = []
    for step in decomp.steps:
        # Check left ref
        if not isinstance(step.left, Ref):
            invalid_refs.append(f"{step.id}.left: not a Ref object")
        elif step.left.type not in RefType:
            invalid_refs.append(f"{step.id}.left: invalid type '{step.left.type}'")
        elif not step.left.id:
            invalid_refs.append(f"{step.id}.left: empty id")

        # Check right ref
        if not isinstance(step.right, Ref):
            invalid_refs.append(f"{step.id}.right: not a Ref object")
        elif step.right.type not in RefType:
            invalid_refs.append(f"{step.id}.right: invalid type '{step.right.type}'")
        elif not step.right.id:
            invalid_refs.append(f"{step.id}.right: empty id")

    if invalid_refs:
        return GradeResult(
            rule_number, rule_name, False,
            f"Invalid references: {'; '.join(invalid_refs)}"
        )

    return GradeResult(rule_number, rule_name, True, "All step inputs are valid Ref objects")


def grade_rule_3_no_dangling_refs(decomp: Decomposition) -> GradeResult:
    """
    Rule 3: No Dangling Refs
    Every ref must resolve to a valid extraction id or prior step id.

    For step refs, we only allow references to steps that appear earlier
    in the step list (enforcing DAG ordering).
    """
    rule_number = 3
    rule_name = "No Dangling Refs"

    # Build valid ID sets
    extraction_ids = {e.id for e in decomp.extractions}
    seen_step_ids: Set[str] = set()

    dangling = []

    for step in decomp.steps:
        for ref_name, ref in [("left", step.left), ("right", step.right)]:
            if ref.type == RefType.EXTRACTION:
                if ref.id not in extraction_ids:
                    dangling.append(f"{step.id}.{ref_name}: extraction '{ref.id}' not found")
            elif ref.type == RefType.STEP:
                if ref.id not in seen_step_ids:
                    dangling.append(f"{step.id}.{ref_name}: step '{ref.id}' not found or appears later")
            elif ref.type == RefType.CONSTANT:
                # Constants are self-contained, just verify parseable
                try:
                    float(ref.id)
                except ValueError:
                    dangling.append(f"{step.id}.{ref_name}: constant '{ref.id}' not a valid number")

        # Now this step's result is available for subsequent steps
        seen_step_ids.add(step.id)

    if dangling:
        return GradeResult(
            rule_number, rule_name, False,
            f"Dangling references: {'; '.join(dangling)}"
        )

    return GradeResult(rule_number, rule_name, True, "All references resolve to valid targets")


def grade_rule_4_acyclic_dag(decomp: Decomposition) -> GradeResult:
    """
    Rule 4: Acyclic DAG
    Steps must form a valid DAG with no circular dependencies.

    Uses DFS cycle detection on the dependency graph.
    """
    rule_number = 4
    rule_name = "Acyclic DAG"

    if not decomp.steps:
        return GradeResult(rule_number, rule_name, True, "No steps to check (trivial pass)")

    # Build adjacency list: step_id -> list of step_ids it depends on
    deps: Dict[str, List[str]] = {}
    step_ids = {s.id for s in decomp.steps}

    for step in decomp.steps:
        step_deps = []
        if step.left.type == RefType.STEP and step.left.id in step_ids:
            step_deps.append(step.left.id)
        if step.right.type == RefType.STEP and step.right.id in step_ids:
            step_deps.append(step.right.id)
        deps[step.id] = step_deps

    # DFS cycle detection
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {s_id: WHITE for s_id in step_ids}
    cycle_path = []

    def has_cycle(node: str) -> bool:
        color[node] = GRAY
        cycle_path.append(node)

        for dep in deps.get(node, []):
            if color[dep] == GRAY:
                # Found cycle
                cycle_start = cycle_path.index(dep)
                cycle_path.append(dep)
                return True
            if color[dep] == WHITE:
                if has_cycle(dep):
                    return True

        cycle_path.pop()
        color[node] = BLACK
        return False

    for step_id in step_ids:
        if color[step_id] == WHITE:
            if has_cycle(step_id):
                cycle_str = " -> ".join(cycle_path[cycle_path.index(cycle_path[-1]):])
                return GradeResult(
                    rule_number, rule_name, False,
                    f"Circular dependency detected: {cycle_str}"
                )

    return GradeResult(rule_number, rule_name, True, "Steps form a valid acyclic DAG")


def grade_rule_5_no_orphaned_extractions(decomp: Decomposition) -> GradeResult:
    """
    Rule 5: No Orphaned Extractions
    Every extraction.id must be referenced by at least one step.

    Extractions that are never used are wasted work and may indicate
    parsing errors or incomplete decomposition.
    """
    rule_number = 5
    rule_name = "No Orphaned Extractions"

    if not decomp.extractions:
        return GradeResult(rule_number, rule_name, True, "No extractions to check (trivial pass)")

    # Collect all extraction refs used in steps
    used_extraction_ids: Set[str] = set()
    for step in decomp.steps:
        if step.left.type == RefType.EXTRACTION:
            used_extraction_ids.add(step.left.id)
        if step.right.type == RefType.EXTRACTION:
            used_extraction_ids.add(step.right.id)

    # Find orphans
    all_extraction_ids = {e.id for e in decomp.extractions}
    orphaned = all_extraction_ids - used_extraction_ids

    if orphaned:
        return GradeResult(
            rule_number, rule_name, False,
            f"Orphaned extractions never used: {', '.join(sorted(orphaned))}"
        )

    return GradeResult(
        rule_number, rule_name, True,
        f"All {len(decomp.extractions)} extractions are referenced"
    )


def grade_rule_6_arithmetic_integrity(decomp: Decomposition) -> GradeResult:
    """
    Rule 6: Arithmetic Integrity
    Each step's claimed result must match the computation of op(left, right).

    Tolerance is applied for floating point comparisons.
    """
    rule_number = 6
    rule_name = "Arithmetic Integrity"

    if not decomp.steps:
        return GradeResult(rule_number, rule_name, True, "No steps to check (trivial pass)")

    # Build extraction values lookup
    extraction_values = {e.id: e.value for e in decomp.extractions}

    # Build step results as we go (to resolve step refs)
    step_results: Dict[str, float] = {}

    mismatches = []
    unresolved = []

    TOLERANCE = 1e-9

    def resolve_ref(ref: Ref) -> Optional[float]:
        if ref.type == RefType.EXTRACTION:
            return extraction_values.get(ref.id)
        elif ref.type == RefType.STEP:
            return step_results.get(ref.id)
        elif ref.type == RefType.CONSTANT:
            try:
                return float(ref.id)
            except ValueError:
                return None
        return None

    def compute(op: str, left_val: float, right_val: float) -> Optional[float]:
        if op == "+":
            return left_val + right_val
        elif op == "-":
            return left_val - right_val
        elif op == "*":
            return left_val * right_val
        elif op == "/":
            if right_val == 0:
                return None  # Division by zero
            return left_val / right_val
        return None

    for step in decomp.steps:
        left_val = resolve_ref(step.left)
        right_val = resolve_ref(step.right)

        if left_val is None or right_val is None:
            unresolved.append(f"{step.id}: could not resolve operands")
            # Still record the claimed result for dependent steps
            step_results[step.id] = step.result
            continue

        expected = compute(step.op, left_val, right_val)

        if expected is None:
            unresolved.append(f"{step.id}: computation error (e.g., division by zero)")
            step_results[step.id] = step.result
            continue

        # Store computed result for subsequent steps
        step_results[step.id] = step.result

        # Check against claimed result
        if abs(expected - step.result) > TOLERANCE:
            mismatches.append(
                f"{step.id}: claimed {step.result}, computed {expected} "
                f"({left_val} {step.op} {right_val})"
            )

    issues = mismatches + unresolved
    if issues:
        return GradeResult(
            rule_number, rule_name, False,
            f"Arithmetic issues: {'; '.join(issues)}"
        )

    return GradeResult(rule_number, rule_name, True, "All step results match their computations")


def grade_rule_7_single_answer_path(decomp: Decomposition) -> GradeResult:
    """
    Rule 7: Single Answer Path
    Exactly one answer_ref pointing to a valid step.

    The answer_ref must be a step reference (not extraction or constant)
    and must point to an existing step.
    """
    rule_number = 7
    rule_name = "Single Answer Path"

    # Check answer_ref exists and is valid
    if decomp.answer_ref is None:
        return GradeResult(rule_number, rule_name, False, "No answer_ref specified")

    if not isinstance(decomp.answer_ref, Ref):
        return GradeResult(rule_number, rule_name, False, "answer_ref is not a Ref object")

    # answer_ref should point to a step (the final computation)
    if decomp.answer_ref.type != RefType.STEP:
        return GradeResult(
            rule_number, rule_name, False,
            f"answer_ref type is '{decomp.answer_ref.type.value}', expected 'step'"
        )

    # Check the referenced step exists
    step_ids = {s.id for s in decomp.steps}
    if decomp.answer_ref.id not in step_ids:
        return GradeResult(
            rule_number, rule_name, False,
            f"answer_ref points to non-existent step '{decomp.answer_ref.id}'"
        )

    return GradeResult(
        rule_number, rule_name, True,
        f"answer_ref correctly points to step '{decomp.answer_ref.id}'"
    )


def grade_rule_8_complete_parsing(decomp: Decomposition) -> GradeResult:
    """
    Rule 8: Complete Parsing
    At least one extraction must exist.

    A decomposition without extractions means no values were parsed
    from the problem text, which is almost certainly an error.
    """
    rule_number = 8
    rule_name = "Complete Parsing"

    if not decomp.extractions:
        return GradeResult(
            rule_number, rule_name, False,
            "No extractions found - problem text was not parsed"
        )

    # Check extractions have valid structure
    invalid = []
    for ext in decomp.extractions:
        if not ext.id:
            invalid.append("extraction with empty id")
        if ext.value is None:
            invalid.append(f"{ext.id}: no value")
        if not ext.span:
            invalid.append(f"{ext.id}: no span text")

    if invalid:
        return GradeResult(
            rule_number, rule_name, False,
            f"Incomplete extractions: {'; '.join(invalid)}"
        )

    return GradeResult(
        rule_number, rule_name, True,
        f"Problem text parsed into {len(decomp.extractions)} extractions"
    )


def grade_rule_9_valid_operation_choice(decomp: Decomposition) -> GradeResult:
    """
    Rule 9: Valid Operation Choice
    Each step's op must be a valid operator from the registry.

    In this schema, valid operators are: +, -, *, /
    This is similar to rule 1 but focuses on semantic validity.
    """
    rule_number = 9
    rule_name = "Valid Operation Choice"

    if not decomp.steps:
        return GradeResult(rule_number, rule_name, True, "No steps to check (trivial pass)")

    invalid_ops = []
    for step in decomp.steps:
        if step.op not in VALID_OPERATORS:
            invalid_ops.append(f"{step.id}: '{step.op}' not in {VALID_OPERATORS}")

    if invalid_ops:
        return GradeResult(
            rule_number, rule_name, False,
            f"Invalid operators: {'; '.join(invalid_ops)}"
        )

    # Additional semantic check: ops should match their semantics reasonably
    # (This is a structural check for now, semantic validation would require more context)

    return GradeResult(
        rule_number, rule_name, True,
        f"All {len(decomp.steps)} steps use valid operators"
    )


def grade_rule_10_answer_alignment(decomp: Decomposition) -> GradeResult:
    """
    Rule 10: Answer Alignment
    answer_ref must point to existing step and answer_value must be set.

    Additionally, answer_value should match the result of the referenced step.
    """
    rule_number = 10
    rule_name = "Answer Alignment"

    issues = []

    # Check answer_value is set
    if decomp.answer_value is None:
        issues.append("answer_value is not set")

    # Check answer_ref points to valid step (partially overlaps with rule 7)
    if decomp.answer_ref is None:
        issues.append("answer_ref is not set")
    elif decomp.answer_ref.type != RefType.STEP:
        issues.append(f"answer_ref is not a step reference")
    else:
        # Find the referenced step
        answer_step = decomp.get_step(decomp.answer_ref.id)
        if answer_step is None:
            issues.append(f"answer_ref points to non-existent step '{decomp.answer_ref.id}'")
        elif decomp.answer_value is not None:
            # Check alignment between answer_value and step result
            TOLERANCE = 1e-9
            if abs(answer_step.result - decomp.answer_value) > TOLERANCE:
                issues.append(
                    f"answer_value ({decomp.answer_value}) != "
                    f"step '{answer_step.id}' result ({answer_step.result})"
                )

    if issues:
        return GradeResult(
            rule_number, rule_name, False,
            f"Answer alignment issues: {'; '.join(issues)}"
        )

    return GradeResult(
        rule_number, rule_name, True,
        f"answer_value={decomp.answer_value} matches step '{decomp.answer_ref.id}'"
    )


def grade_decomposition(decomp: Decomposition) -> List[GradeResult]:
    """
    Grade a decomposition against all 10 rules.

    Returns a list of GradeResult objects, one per rule.

    Rules:
    1. Atomic Operations - Each step has a single valid operator
    2. Explicit References - All inputs are typed Ref objects
    3. No Dangling Refs - Every ref resolves to valid target
    4. Acyclic DAG - Steps form valid DAG
    5. No Orphaned Extractions - All extractions are used
    6. Arithmetic Integrity - Computed results match claimed results
    7. Single Answer Path - Exactly one answer_ref to valid step
    8. Complete Parsing - At least one extraction exists
    9. Valid Operation Choice - Operators are in registry
    10. Answer Alignment - answer_ref and answer_value are consistent
    """
    return [
        grade_rule_1_atomic(decomp),
        grade_rule_2_explicit_refs(decomp),
        grade_rule_3_no_dangling_refs(decomp),
        grade_rule_4_acyclic_dag(decomp),
        grade_rule_5_no_orphaned_extractions(decomp),
        grade_rule_6_arithmetic_integrity(decomp),
        grade_rule_7_single_answer_path(decomp),
        grade_rule_8_complete_parsing(decomp),
        grade_rule_9_valid_operation_choice(decomp),
        grade_rule_10_answer_alignment(decomp),
    ]


def summary(results: List[GradeResult]) -> str:
    """
    Return summary like '8/10 rules passed' with failed rule details.

    Example output:
        8/10 rules passed

        Failed:
        - Rule 3 (No Dangling Refs): step s2.left: extraction 'price' not found
        - Rule 6 (Arithmetic Integrity): step s1: claimed 15, computed 12
    """
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    lines = [f"{len(passed)}/{len(results)} rules passed"]

    if failed:
        lines.append("")
        lines.append("Failed:")
        for r in failed:
            lines.append(f"- Rule {r.rule_number} ({r.rule_name}): {r.message}")

    return "\n".join(lines)
