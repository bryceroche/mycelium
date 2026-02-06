"""Computation Graph Embeddings for Mycelium.

Converts SubGraphDSL structures to parameter-agnostic embeddings that capture
what operations DO, not what they SOUND LIKE.

Per CLAUDE.md: "Route by what operations DO, not what they SOUND LIKE."
- Parameter-agnostic: Variable names don't matter, structure does
- Implementation-agnostic: Same graph regardless of Python vs SymPy
- Operationally meaningful: Two DSLs with the same graph do the same thing
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

from mycelium.subgraph_dsl import SubGraphDSL, SubGraphStep


# Operation type indices for embedding
OP_TO_IDX = {
    'SET': 0,
    'ADD': 1,
    'SUB': 2,
    'MUL': 3,
    'DIV': 4,
    'MOD': 5,
    'NEG': 6,
}
NUM_OPS = len(OP_TO_IDX)

# Embedding dimensions
GRAPH_EMBEDDING_DIM = 64


def canonicalize_graph(dsl: Union[SubGraphDSL, Dict[str, Any]]) -> str:
    """Convert SubGraphDSL to parameter-agnostic canonical string.

    The canonical form ignores variable names and captures only:
    - Operation sequence
    - Data flow (which ops feed into which)
    - Input/output structure

    Args:
        dsl: SubGraphDSL instance or dict representation

    Returns:
        Canonical string like "MUL(p0,p1)->v0;ADD(v0,p2)->out"
    """
    if isinstance(dsl, dict):
        steps = dsl.get('steps', [])
        params = dsl.get('params', {})
        inputs = dsl.get('inputs', {})
        output = dsl.get('output', '')
    else:
        steps = dsl.steps
        params = dsl.params
        inputs = dsl.inputs
        output = dsl.output

    if not steps:
        return "EMPTY"

    # Build variable mapping: original name -> canonical name
    var_map = {}

    # Map params to p0, p1, p2...
    for i, param_name in enumerate(sorted(params.keys())):
        var_map[param_name] = f"p{i}"

    # Map inputs to i0, i1, i2...
    for i, input_name in enumerate(sorted(inputs.keys())):
        var_map[input_name] = f"i{i}"

    # Process steps and map intermediate vars to v0, v1, v2...
    canonical_steps = []
    var_counter = 0

    for step in steps:
        if isinstance(step, dict):
            op = step.get('op', 'SET')
            var = step.get('var', '')
            args = step.get('args', [])
        else:
            op = step.op
            var = step.var
            args = step.args

        # Map arguments to canonical names
        canonical_args = []
        for arg in args:
            if isinstance(arg, (int, float)):
                # Literal numbers - keep as-is but mark as constant
                canonical_args.append(f"c{arg}")
            elif arg in var_map:
                canonical_args.append(var_map[arg])
            else:
                # Unknown var - shouldn't happen but handle gracefully
                canonical_args.append(f"?{arg}")

        # Map output variable
        if var not in var_map:
            var_map[var] = f"v{var_counter}"
            var_counter += 1

        canonical_var = var_map[var]
        canonical_steps.append(f"{op}({','.join(canonical_args)})->{canonical_var}")

    # Mark the output variable
    output_canonical = var_map.get(output, "?")

    return f"{';'.join(canonical_steps)}|out={output_canonical}"


def hash_graph_structure(dsl: Union[SubGraphDSL, Dict[str, Any]]) -> str:
    """Get a hash of the graph structure for grouping operationally-similar templates.

    Simpler than full canonicalization - just captures operation types and flow.

    Args:
        dsl: SubGraphDSL instance or dict representation

    Returns:
        Structure hash string like "MUL_True" (ops + has_inputs)
    """
    if isinstance(dsl, dict):
        steps = dsl.get('steps', [])
        inputs = dsl.get('inputs', {})
    else:
        steps = dsl.steps
        inputs = dsl.inputs

    ops = tuple(
        step.get('op', 'SET') if isinstance(step, dict) else step.op
        for step in steps
    )
    has_inputs = bool(inputs)

    return f"{ops}_{has_inputs}"


def graph_to_embedding(dsl: Union[SubGraphDSL, Dict[str, Any]]) -> np.ndarray:
    """Convert SubGraphDSL structure to fixed-size embedding vector.

    The embedding captures:
    - [0:7]   Operation type counts (one-hot per op type)
    - [7:14]  Operation order encoding (position of each op type)
    - [14:28] Adjacency structure (which ops feed into which)
    - [28:32] Structural features (has_inputs, num_params, num_steps, depth)
    - [32:64] Reserved for future graph features

    Args:
        dsl: SubGraphDSL instance or dict representation

    Returns:
        64-dimensional numpy array
    """
    vec = np.zeros(GRAPH_EMBEDDING_DIM, dtype=np.float32)

    if isinstance(dsl, dict):
        steps = dsl.get('steps', [])
        params = dsl.get('params', {})
        inputs = dsl.get('inputs', {})
        output = dsl.get('output', '')
    else:
        steps = dsl.steps
        params = dsl.params
        inputs = dsl.inputs
        output = dsl.output

    if not steps:
        # Empty graph - return zero vector with marker
        vec[31] = 1.0  # Mark as empty
        return vec

    # [0:7] Operation type counts (normalized)
    op_counts = np.zeros(NUM_OPS)
    for step in steps:
        op = step.get('op', 'SET') if isinstance(step, dict) else step.op
        if op in OP_TO_IDX:
            op_counts[OP_TO_IDX[op]] += 1
    if op_counts.sum() > 0:
        vec[0:NUM_OPS] = op_counts / op_counts.sum()

    # [7:14] Operation order encoding (first occurrence position, normalized)
    first_occurrence = np.ones(NUM_OPS) * -1  # -1 = not present
    for i, step in enumerate(steps):
        op = step.get('op', 'SET') if isinstance(step, dict) else step.op
        if op in OP_TO_IDX:
            idx = OP_TO_IDX[op]
            if first_occurrence[idx] < 0:
                first_occurrence[idx] = i / max(1, len(steps) - 1) if len(steps) > 1 else 0
    vec[7:14] = first_occurrence

    # [14:28] Adjacency structure - which variables are used where
    # Build var -> step index map
    var_to_step = {}
    for i, step in enumerate(steps):
        var = step.get('var', '') if isinstance(step, dict) else step.var
        var_to_step[var] = i

    # Encode data flow as adjacency
    adjacency = np.zeros(14)  # 7 ops * 2 (producer/consumer)
    for i, step in enumerate(steps):
        args = step.get('args', []) if isinstance(step, dict) else step.args
        step_op = step.get('op', 'SET') if isinstance(step, dict) else step.op
        step_op_idx = OP_TO_IDX.get(step_op, 0)

        for arg in args:
            if isinstance(arg, str) and arg in var_to_step:
                # This step consumes output from another step
                producer_step = steps[var_to_step[arg]]
                producer_op = producer_step.get('op', 'SET') if isinstance(producer_step, dict) else producer_step.op
                producer_op_idx = OP_TO_IDX.get(producer_op, 0)

                # Mark producer-consumer relationship
                adjacency[producer_op_idx] += 0.5  # Producer
                adjacency[NUM_OPS + step_op_idx] += 0.5  # Consumer

    if adjacency.sum() > 0:
        adjacency = adjacency / adjacency.sum()
    vec[14:28] = adjacency

    # [28:32] Structural features
    vec[28] = 1.0 if inputs else 0.0  # has_inputs
    vec[29] = min(1.0, len(params) / 5.0)  # num_params (normalized to max 5)
    vec[30] = min(1.0, len(steps) / 5.0)  # num_steps (normalized to max 5)

    # Compute depth (longest chain)
    depth = compute_graph_depth(steps, var_to_step)
    vec[31] = min(1.0, depth / 5.0)  # depth (normalized to max 5)

    # [32:64] Reserved - could add:
    # - Hash of canonical form
    # - Learned graph neural network features
    # - Operation argument patterns

    return vec


def compute_graph_depth(steps: List, var_to_step: Dict[str, int]) -> int:
    """Compute the longest chain depth in the computation graph."""
    if not steps:
        return 0

    # Build depth for each step via dynamic programming
    depths = {}

    def get_depth(step_idx: int) -> int:
        if step_idx in depths:
            return depths[step_idx]

        step = steps[step_idx]
        args = step.get('args', []) if isinstance(step, dict) else step.args

        max_arg_depth = 0
        for arg in args:
            if isinstance(arg, str) and arg in var_to_step:
                arg_depth = get_depth(var_to_step[arg])
                max_arg_depth = max(max_arg_depth, arg_depth)

        depths[step_idx] = max_arg_depth + 1
        return depths[step_idx]

    return max(get_depth(i) for i in range(len(steps)))


def embedding_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two graph embeddings."""
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)

    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0

    return float(np.dot(emb1, emb2) / (norm1 * norm2))


def batch_graph_embeddings(templates: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Compute graph embeddings for a batch of templates.

    Args:
        templates: List of template dicts with 'template_id' and 'subgraph' fields

    Returns:
        Dict mapping template_id -> graph_embedding
    """
    embeddings = {}

    for template in templates:
        template_id = template.get('template_id', '')
        subgraph = template.get('subgraph', {})

        if subgraph:
            embeddings[template_id] = graph_to_embedding(subgraph)
        else:
            embeddings[template_id] = np.zeros(GRAPH_EMBEDDING_DIM, dtype=np.float32)

    return embeddings


# ================================================================
# CLI / Main
# ================================================================

if __name__ == "__main__":
    # Demo with sample SubGraphDSLs
    print("=" * 60)
    print("Graph Embedder Demo")
    print("=" * 60)

    # Example 1: Simple SET
    dsl1 = {
        "template_id": "test_set",
        "params": {"value": "the quantity"},
        "inputs": {},
        "steps": [{"var": "result", "op": "SET", "args": ["value"]}],
        "output": "result"
    }

    # Example 2: MUL operation
    dsl2 = {
        "template_id": "test_mul",
        "params": {"rate": "rate", "time": "time"},
        "inputs": {},
        "steps": [{"var": "total", "op": "MUL", "args": ["rate", "time"]}],
        "output": "total"
    }

    # Example 3: Chained operations (ADD then SUB)
    dsl3 = {
        "template_id": "test_chain",
        "params": {"a": "first", "b": "second", "c": "third"},
        "inputs": {},
        "steps": [
            {"var": "sum", "op": "ADD", "args": ["a", "b"]},
            {"var": "result", "op": "SUB", "args": ["sum", "c"]}
        ],
        "output": "result"
    }

    # Example 4: With upstream inputs
    dsl4 = {
        "template_id": "test_inputs",
        "params": {"value": "amount"},
        "inputs": {"upstream": "previous result"},
        "steps": [{"var": "result", "op": "SUB", "args": ["upstream", "value"]}],
        "output": "result"
    }

    examples = [dsl1, dsl2, dsl3, dsl4]

    print("\nCanonical forms:")
    for dsl in examples:
        print(f"  {dsl['template_id']}: {canonicalize_graph(dsl)}")

    print("\nStructure hashes:")
    for dsl in examples:
        print(f"  {dsl['template_id']}: {hash_graph_structure(dsl)}")

    print("\nEmbeddings (first 32 dims):")
    embeddings = []
    for dsl in examples:
        emb = graph_to_embedding(dsl)
        embeddings.append(emb)
        print(f"  {dsl['template_id']}:")
        print(f"    op_counts: {emb[0:7]}")
        print(f"    has_inputs: {emb[28]}, num_params: {emb[29]:.2f}, depth: {emb[31]:.2f}")

    print("\nSimilarity matrix:")
    for i, dsl1 in enumerate(examples):
        row = []
        for j, dsl2 in enumerate(examples):
            sim = embedding_similarity(embeddings[i], embeddings[j])
            row.append(f"{sim:.2f}")
        print(f"  {examples[i]['template_id']}: {' '.join(row)}")
