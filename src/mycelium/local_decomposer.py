"""Local Decomposer: Convert problems to atomic steps without iterative LLM negotiation.

Per CLAUDE.md: "Are we going deep enough? If we have low accuracy, the answer is NO."

This module implements fine-grained decomposition:
1. Single LLM call extracts values + computation graph
2. Local decomposer converts to atomic steps using tree vocabulary
3. Each step maps directly to one signature (extract or compute)

Benefits:
- Fewer LLM calls (1 vs 3+)
- Finer granularity (10+ atomic steps vs 3-5 coarse steps)
- Better signature reuse (direct mapping)
- Precise failure attribution (which atomic op failed)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExtractedValue:
    """A value extracted from the problem text."""
    name: str
    value: float
    description: str = ""


@dataclass
class PlanNode:
    """A computation node in the plan graph."""
    id: str
    op: str  # add, subtract, multiply, divide
    inputs: list[str]  # References to values or previous nodes
    description: str = ""


@dataclass
class StructuredPlan:
    """Output from Phase 1: extracted values + computation graph."""
    values: dict[str, float]
    plan: list[PlanNode]
    answer_var: str
    raw_response: str = ""


@dataclass
class AtomicStep:
    """A single atomic step ready for execution."""
    id: str
    step_type: str  # "extract" or "compute"
    signature: str  # Tree signature to use (e.g., "compute_sum")
    params: dict[str, str]  # Parameter bindings
    description: str = ""
    depends_on: list[str] = field(default_factory=list)
    value: Optional[float] = None  # For extract steps


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

EXTRACTION_PROMPT = '''You are a math problem decomposer. Extract all numeric values and create a computation plan.

PROBLEM:
{problem}

Instructions:
1. Extract ALL numeric values from the problem (including implicit ones like fractions, percentages)
2. Create a step-by-step computation plan using ONLY these operations: add, subtract, multiply, divide
3. Each plan step should be ONE atomic operation
4. Reference values and previous steps by their id

Respond in JSON format:
{{
  "values": {{
    "value_name": <number>,
    ...
  }},
  "plan": [
    {{"id": "step_1", "op": "multiply|add|subtract|divide", "inputs": ["value_or_step", "value_or_step"], "description": "what this computes"}},
    ...
  ],
  "answer_var": "step_N"
}}

Example for "Apples cost $3 each. Buy 5 apples and get $2 discount. Total cost?":
{{
  "values": {{
    "price_per_apple": 3,
    "num_apples": 5,
    "discount": 2
  }},
  "plan": [
    {{"id": "subtotal", "op": "multiply", "inputs": ["price_per_apple", "num_apples"], "description": "cost before discount"}},
    {{"id": "total", "op": "subtract", "inputs": ["subtotal", "discount"], "description": "final cost after discount"}}
  ],
  "answer_var": "total"
}}

IMPORTANT:
- Break complex calculations into multiple simple steps
- Each step should have exactly 2 inputs
- Use descriptive value names (not x, y, z)
- The answer_var must be the id of a plan step or a value name
'''


# =============================================================================
# PARSING
# =============================================================================

def parse_structured_plan(response: str) -> Optional[StructuredPlan]:
    """Parse LLM response into StructuredPlan."""
    try:
        # Handle markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        data = json.loads(response.strip())

        values = {}
        for k, v in data.get("values", {}).items():
            try:
                values[k] = float(v)
            except (ValueError, TypeError):
                logger.warning("[local_decomp] Could not parse value %s=%s as float", k, v)
                continue

        plan = []
        for node in data.get("plan", []):
            plan.append(PlanNode(
                id=node.get("id", f"step_{len(plan)+1}"),
                op=node.get("op", "").lower(),
                inputs=node.get("inputs", []),
                description=node.get("description", ""),
            ))

        answer_var = data.get("answer_var", plan[-1].id if plan else "")

        return StructuredPlan(
            values=values,
            plan=plan,
            answer_var=answer_var,
            raw_response=response,
        )

    except json.JSONDecodeError as e:
        logger.error("[local_decomp] Failed to parse JSON: %s\nResponse: %s", e, response[:500])
        return None
    except Exception as e:
        logger.error("[local_decomp] Failed to parse structured plan: %s", e)
        return None


# =============================================================================
# LOCAL DECOMPOSER
# =============================================================================

# Map from LLM operation names to tree signature types
OP_TO_SIGNATURE = {
    "add": "compute_sum",
    "sum": "compute_sum",
    "plus": "compute_sum",
    "subtract": "compute_difference",
    "minus": "compute_difference",
    "difference": "compute_difference",
    "multiply": "compute_product",
    "times": "compute_product",
    "product": "compute_product",
    "divide": "compute_quotient",
    "quotient": "compute_quotient",
    "ratio": "compute_quotient",
}


def decompose_locally(
    structured_plan: StructuredPlan,
    tree_vocabulary: Optional[list[str]] = None,
) -> list[AtomicStep]:
    """Convert structured plan to atomic steps using tree vocabulary.

    This is the core local decomposition - NO LLM calls.

    Args:
        structured_plan: Output from Phase 1 LLM call
        tree_vocabulary: Available signatures in tree (for validation)

    Returns:
        List of atomic steps ready for execution
    """
    steps = []
    tree_vocabulary = tree_vocabulary or list(OP_TO_SIGNATURE.values())

    # Track which IDs are available (values + computed steps)
    available_ids = set(structured_plan.values.keys())

    # 1. Create extract steps for each value
    # These don't need signatures - they just surface values into context
    for name, value in structured_plan.values.items():
        steps.append(AtomicStep(
            id=name,
            step_type="extract",
            signature="extract_value",
            params={},
            description=f"Extract {name} = {value}",
            depends_on=[],
            value=value,
        ))

    # 2. Create compute steps from plan
    for node in structured_plan.plan:
        # Map operation to signature
        sig = OP_TO_SIGNATURE.get(node.op)

        if sig is None:
            logger.warning("[local_decomp] Unknown operation '%s', defaulting to compute_sum", node.op)
            sig = "compute_sum"

        # Validate signature exists in tree
        if tree_vocabulary and sig not in tree_vocabulary:
            # Find closest match
            for vocab_sig in tree_vocabulary:
                if node.op in vocab_sig.lower():
                    sig = vocab_sig
                    break
            else:
                logger.warning("[local_decomp] Signature '%s' not in tree vocabulary", sig)

        # Build param bindings
        params = {}
        depends_on = []

        for i, inp in enumerate(node.inputs):
            param_name = f"param{i+1}"

            # Check if input is a literal number
            if isinstance(inp, (int, float)):
                # Literal number - use directly
                params[param_name] = float(inp)
            elif isinstance(inp, str):
                # Try to parse as number
                try:
                    params[param_name] = float(inp)
                except ValueError:
                    # It's a variable reference
                    params[param_name] = inp
                    if inp in available_ids:
                        depends_on.append(inp)
                    else:
                        logger.warning("[local_decomp] Input '%s' not found in available IDs", inp)
            else:
                params[param_name] = inp

        steps.append(AtomicStep(
            id=node.id,
            step_type="compute",
            signature=sig,
            params=params,
            description=node.description or f"{node.op}({', '.join(node.inputs)})",
            depends_on=depends_on,
        ))

        # This step's output is now available
        available_ids.add(node.id)

    logger.info(
        "[local_decomp] Decomposed into %d steps: %d extracts, %d computes",
        len(steps),
        sum(1 for s in steps if s.step_type == "extract"),
        sum(1 for s in steps if s.step_type == "compute"),
    )

    return steps


# =============================================================================
# EXECUTION HELPERS
# =============================================================================

def atomic_steps_to_dag(steps: list[AtomicStep]) -> dict:
    """Convert atomic steps to DAG format for solver execution.

    Returns dict with:
        - context: Initial values from extract steps
        - compute_steps: List of compute operations to execute
    """
    context = {}
    compute_steps = []

    for step in steps:
        if step.step_type == "extract":
            context[step.id] = step.value
        else:
            compute_steps.append({
                "id": step.id,
                "signature": step.signature,
                "params": step.params,
                "depends_on": step.depends_on,
                "description": step.description,
            })

    return {
        "context": context,
        "compute_steps": compute_steps,
    }


def validate_plan(steps: list[AtomicStep], answer_var: str) -> tuple[bool, str]:
    """Validate that the plan is executable.

    Returns:
        (is_valid, error_message)
    """
    available = set()

    for step in steps:
        # Check dependencies are available
        for dep in step.depends_on:
            if dep not in available:
                return False, f"Step '{step.id}' depends on '{dep}' which is not available"

        # Check compute steps have required params
        if step.step_type == "compute":
            if len(step.params) < 2:
                return False, f"Step '{step.id}' has {len(step.params)} params, need at least 2"

        available.add(step.id)

    # Check answer_var exists
    if answer_var not in available:
        return False, f"answer_var '{answer_var}' not found in steps"

    return True, ""


# =============================================================================
# HIGH-LEVEL API
# =============================================================================

async def extract_and_decompose(
    problem: str,
    client,  # LLMClient
    tree_vocabulary: Optional[list[str]] = None,
) -> tuple[Optional[list[AtomicStep]], Optional[str], dict]:
    """Single entry point: extract values + decompose locally.

    Args:
        problem: The math problem text
        client: LLM client for the single extraction call
        tree_vocabulary: Available signatures in tree

    Returns:
        (atomic_steps, answer_var, metadata)
    """
    # Phase 1: Single LLM call for structured extraction
    prompt = EXTRACTION_PROMPT.format(problem=problem)

    try:
        response = await client.generate(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.error("[local_decomp] LLM extraction failed: %s", e)
        return None, None, {"error": str(e)}

    # Parse response
    structured_plan = parse_structured_plan(response)
    if structured_plan is None:
        return None, None, {"error": "Failed to parse structured plan", "raw": response}

    logger.info(
        "[local_decomp] Extracted %d values, %d plan steps",
        len(structured_plan.values),
        len(structured_plan.plan),
    )

    # Phase 2: Local decomposition (NO LLM)
    atomic_steps = decompose_locally(structured_plan, tree_vocabulary)

    # Validate
    is_valid, error = validate_plan(atomic_steps, structured_plan.answer_var)
    if not is_valid:
        logger.warning("[local_decomp] Plan validation failed: %s", error)
        return None, None, {"error": error, "structured_plan": structured_plan}

    metadata = {
        "values_count": len(structured_plan.values),
        "plan_steps": len(structured_plan.plan),
        "atomic_steps": len(atomic_steps),
        "answer_var": structured_plan.answer_var,
    }

    return atomic_steps, structured_plan.answer_var, metadata


# =============================================================================
# DAGPLAN CONVERSION
# =============================================================================

def atomic_steps_to_dagplan(
    atomic_steps: list[AtomicStep],
    answer_var: str,
    problem: str,
) -> "DAGPlan":
    """Convert atomic steps to a DAGPlan for solver execution.

    This bridges local decomposition output to the existing solver infrastructure.

    Args:
        atomic_steps: List of AtomicStep from local decomposition
        answer_var: The step ID that contains the final answer
        problem: Original problem text

    Returns:
        DAGPlan compatible with solver.solve()
    """
    from mycelium.planner import Step, DAGPlan

    # Build context from extract steps (phase1_values)
    phase1_values = {}
    for step in atomic_steps:
        if step.step_type == "extract" and step.value is not None:
            phase1_values[step.id] = step.value

    # Convert compute steps to Step objects
    dag_steps = []
    step_counter = 1

    for atomic in atomic_steps:
        if atomic.step_type == "extract":
            # Extract steps become the initial context, not explicit DAG steps
            # The values are in phase1_values and will be available to compute steps
            continue

        # Build extracted_values with references to dependencies
        extracted_values = {}
        for param_name, param_ref in atomic.params.items():
            if isinstance(param_ref, (int, float)):
                # Literal number - use directly
                extracted_values[param_name] = float(param_ref)
            elif param_ref in phase1_values:
                # Direct value reference from extracted values
                extracted_values[param_name] = phase1_values[param_ref]
            else:
                # Reference to previous step result
                extracted_values[param_name] = f"{{step_{param_ref}}}"

        # Map signature to dsl_hint
        sig_to_hint = {
            "compute_sum": "+",
            "compute_difference": "-",
            "compute_product": "*",
            "compute_quotient": "/",
        }
        dsl_hint = sig_to_hint.get(atomic.signature, atomic.signature)

        # Create depends_on with proper step_N format
        depends_on = []
        for dep in atomic.depends_on:
            if dep not in phase1_values:
                # It's a compute step dependency
                depends_on.append(f"step_{dep}")

        step = Step(
            id=f"step_{atomic.id}",
            task=atomic.description,
            depends_on=depends_on,
            extracted_values=extracted_values,
            dsl_hint=dsl_hint,
            operation=atomic.signature,
        )
        dag_steps.append(step)
        step_counter += 1

    # Create DAGPlan
    plan = DAGPlan(
        steps=dag_steps,
        problem=problem,
        depth=0,
        phase1_values=phase1_values,
    )

    logger.info(
        "[local_decomp] Converted to DAGPlan: %d steps, %d phase1 values, answer=%s",
        len(dag_steps),
        len(phase1_values),
        answer_var,
    )

    return plan


async def decompose_problem(
    problem: str,
    client,  # LLMClient
    tree_vocabulary: Optional[list[str]] = None,
) -> Optional["DAGPlan"]:
    """Main entry point: decompose problem to DAGPlan using local decomposition.

    Single LLM call for extraction, then local decomposition.
    Returns a DAGPlan ready for solver execution.

    Args:
        problem: The math problem text
        client: LLM client for extraction
        tree_vocabulary: Available signatures in tree

    Returns:
        DAGPlan or None if decomposition failed
    """
    atomic_steps, answer_var, metadata = await extract_and_decompose(
        problem, client, tree_vocabulary
    )

    if atomic_steps is None:
        logger.warning("[local_decomp] Decomposition failed: %s", metadata.get("error"))
        return None

    return atomic_steps_to_dagplan(atomic_steps, answer_var, problem)
