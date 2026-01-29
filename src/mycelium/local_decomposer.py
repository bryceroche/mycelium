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

EXTRACTION_PROMPT = '''You are a math problem decomposer creating MAXIMALLY ATOMIC steps.

PROBLEM:
{problem}

CRITICAL RULES - FOLLOW EXACTLY:
1. Extract EVERY numeric value - explicit AND implicit:
   - "half" = 0.5, "twice" = 2, "triple" = 3, "quarter" = 0.25
   - Percentages: "2.5%" = 0.025 (as decimal)
   - "per" rates, "each" prices, time periods, counts
   - EVEN if a value seems obvious, EXTRACT IT

2. Each plan step is EXACTLY ONE arithmetic operation: add, subtract, multiply, OR divide
   - NEVER combine operations
   - "multiply then add" = TWO separate steps
   - "find the difference then multiply" = TWO separate steps

3. SHOW ALL INTERMEDIATE CALCULATIONS:
   - Convert percentages: amount × percentage = increase (separate step)
   - Then add: original + increase = final (separate step)
   - NEVER do "original × 1.025" - that's combining multiply and implicit add!

4. MINIMUM 5 STEPS for any problem. Most need 8-12 steps.
   - If you have fewer than 5 steps, you're combining operations
   - Break down further until each step is truly atomic

5. Each step has exactly 2 inputs (values or previous step results)

Respond in JSON:
{{
  "values": {{
    "descriptive_name": <number>,
    ...
  }},
  "plan": [
    {{"id": "step_1", "op": "add|subtract|multiply|divide", "inputs": ["input1", "input2"], "description": "what this computes"}},
    ...
  ],
  "answer_var": "final_step_id"
}}

EXAMPLE - "A store sells shirts for $25 each. Tom buys 3 shirts and 2 pants at $40 each. He has a $15 coupon. How much does he pay?"

{{
  "values": {{
    "shirt_price": 25,
    "num_shirts": 3,
    "pants_price": 40,
    "num_pants": 2,
    "coupon_value": 15
  }},
  "plan": [
    {{"id": "shirts_cost", "op": "multiply", "inputs": ["shirt_price", "num_shirts"], "description": "total cost of shirts"}},
    {{"id": "pants_cost", "op": "multiply", "inputs": ["pants_price", "num_pants"], "description": "total cost of pants"}},
    {{"id": "subtotal", "op": "add", "inputs": ["shirts_cost", "pants_cost"], "description": "cost before coupon"}},
    {{"id": "final_total", "op": "subtract", "inputs": ["subtotal", "coupon_value"], "description": "final amount after coupon"}}
  ],
  "answer_var": "final_total"
}}

EXAMPLE - "Dana runs 4x faster than walks. Skip speed is half run speed. Skip = 3mph. Distance in 1hr (1/3 running, 2/3 walking)?"

{{
  "values": {{
    "skip_speed": 3,
    "run_to_skip_ratio": 2,
    "walk_to_run_divisor": 4,
    "total_time_hours": 1,
    "run_time_fraction": 0.333,
    "walk_time_fraction": 0.667
  }},
  "plan": [
    {{"id": "run_speed", "op": "multiply", "inputs": ["skip_speed", "run_to_skip_ratio"], "description": "running speed from skip"}},
    {{"id": "walk_speed", "op": "divide", "inputs": ["run_speed", "walk_to_run_divisor"], "description": "walking speed from run"}},
    {{"id": "run_time", "op": "multiply", "inputs": ["total_time_hours", "run_time_fraction"], "description": "time spent running"}},
    {{"id": "walk_time", "op": "multiply", "inputs": ["total_time_hours", "walk_time_fraction"], "description": "time spent walking"}},
    {{"id": "run_distance", "op": "multiply", "inputs": ["run_speed", "run_time"], "description": "distance covered running"}},
    {{"id": "walk_distance", "op": "multiply", "inputs": ["walk_speed", "walk_time"], "description": "distance covered walking"}},
    {{"id": "total_distance", "op": "add", "inputs": ["run_distance", "walk_distance"], "description": "total distance traveled"}}
  ],
  "answer_var": "total_distance"
}}

Remember: MORE ATOMIC STEPS = BETTER. Break everything down to single operations.
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
