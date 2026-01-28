"""Computation Graph Extraction from DSL Scripts.

Per CLAUDE.md: Route by what operations DO, not what they SOUND LIKE.

This module extracts structural computation graphs from DSL scripts.
The graph is a parameter-agnostic representation of the computation:
    MUL(param_0, param_1)  - multiplication of two values
    ADD(MUL(param_0, param_1), param_2)  - multiply then add

Graphs are:
- Parameter-agnostic: Variable names don't matter, structure does
- Implementation-agnostic: Same graph regardless of Python vs SymPy
- Operationally meaningful: Two DSLs with the same graph do the same thing
"""

import ast
import json
import logging
from typing import Optional

from mycelium.embedding_cache import cached_embed
from mycelium.config import GRAPH_EMBEDDING_CACHE_MAX_SIZE

logger = logging.getLogger(__name__)

# Map Python AST operators to graph node names
_BINOP_MAP = {
    ast.Add: "ADD",
    ast.Sub: "SUB",
    ast.Mult: "MUL",
    ast.Div: "DIV",
    ast.FloorDiv: "FLOORDIV",
    ast.Mod: "MOD",
    ast.Pow: "POW",
}

_UNARYOP_MAP = {
    ast.UAdd: "POS",
    ast.USub: "NEG",
}

# Built-in functions we recognize
_FUNC_MAP = {
    "sqrt": "SQRT",
    "abs": "ABS",
    "gcd": "GCD",
    "lcm": "LCM",
    "factorial": "FACTORIAL",
    "comb": "COMB",
    "perm": "PERM",
    "sum": "REDUCE_SUM",
    "min": "REDUCE_MIN",
    "max": "REDUCE_MAX",
    "len": "LEN",
    "round": "ROUND",
    "int": "INT",
    "float": "FLOAT",
    "ceil": "CEIL",
    "floor": "FLOOR",
}


class GraphExtractor:
    """Extract computation graphs from Python DSL scripts."""

    def __init__(self):
        self._param_counter = 0
        self._param_map: dict[str, str] = {}  # original name -> param_N

    def extract(self, dsl_script: str) -> Optional[str]:
        """Extract computation graph from DSL script.

        Args:
            dsl_script: Either a raw Python expression like "a * b"
                       or a JSON DSL dict with "script" key

        Returns:
            Graph string like "MUL(param_0, param_1)" or None if parsing fails
        """
        # Reset state
        self._param_counter = 0
        self._param_map = {}

        # Parse DSL format
        script = self._parse_dsl_input(dsl_script)
        if not script:
            return None

        # Parse Python expression
        try:
            tree = ast.parse(script, mode='eval')
            return self._visit(tree.body)
        except SyntaxError as e:
            logger.debug("[graph] Failed to parse DSL script: %s - %s", script[:50], e)
            return None
        except Exception as e:
            logger.warning("[graph] Unexpected error extracting graph: %s", e)
            return None

    def _parse_dsl_input(self, dsl_script: str) -> Optional[str]:
        """Extract the actual script from various DSL formats."""
        if not dsl_script:
            return None

        # Try parsing as JSON first
        try:
            dsl_data = json.loads(dsl_script)
            if isinstance(dsl_data, dict):
                script = dsl_data.get("script", "")
                # Skip decompose type - no meaningful graph
                if dsl_data.get("type") == "decompose":
                    return None
                return script if script else None
        except (json.JSONDecodeError, TypeError):
            pass

        # Assume it's a raw Python expression
        return dsl_script.strip()

    def _visit(self, node: ast.AST) -> str:
        """Visit an AST node and return its graph representation."""
        if isinstance(node, ast.BinOp):
            return self._visit_binop(node)
        elif isinstance(node, ast.UnaryOp):
            return self._visit_unaryop(node)
        elif isinstance(node, ast.Call):
            return self._visit_call(node)
        elif isinstance(node, ast.Name):
            return self._visit_name(node)
        elif isinstance(node, ast.Constant):
            return self._visit_constant(node)
        elif isinstance(node, ast.Num):  # Python 3.7 compat
            return self._visit_num(node)
        elif isinstance(node, ast.Subscript):
            return self._visit_subscript(node)
        elif isinstance(node, ast.Compare):
            return self._visit_compare(node)
        elif isinstance(node, ast.IfExp):
            return self._visit_ifexp(node)
        else:
            logger.debug("[graph] Unhandled AST node type: %s", type(node).__name__)
            return "UNKNOWN"

    def _visit_binop(self, node: ast.BinOp) -> str:
        """Handle binary operations: a + b, x * y, etc."""
        op_name = _BINOP_MAP.get(type(node.op), "BINOP")
        left = self._visit(node.left)
        right = self._visit(node.right)
        return f"{op_name}({left}, {right})"

    def _visit_unaryop(self, node: ast.UnaryOp) -> str:
        """Handle unary operations: -x, +x."""
        op_name = _UNARYOP_MAP.get(type(node.op), "UNARY")
        operand = self._visit(node.operand)
        return f"{op_name}({operand})"

    def _visit_call(self, node: ast.Call) -> str:
        """Handle function calls: sqrt(x), gcd(a, b), sum(items)."""
        # Get function name
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Handle module.func like math.sqrt
            func_name = node.func.attr
        else:
            func_name = "FUNC"

        # Map to graph node name
        graph_name = _FUNC_MAP.get(func_name.lower(), func_name.upper())

        # Process arguments
        args = [self._visit(arg) for arg in node.args]
        return f"{graph_name}({', '.join(args)})"

    def _visit_name(self, node: ast.Name) -> str:
        """Handle variable names: map to param_N."""
        name = node.id

        # Already mapped?
        if name in self._param_map:
            return self._param_map[name]

        # Create new param mapping
        param_id = f"param_{self._param_counter}"
        self._param_counter += 1
        self._param_map[name] = param_id
        return param_id

    def _visit_constant(self, node: ast.Constant) -> str:
        """Handle constants: numbers, strings."""
        val = node.value
        if isinstance(val, (int, float)):
            # Represent constants as CONST(value) for structure
            # But for simple integers like 2, 100, etc., use CONST
            return f"CONST({val})"
        return "CONST"

    def _visit_num(self, node: ast.Num) -> str:
        """Handle numbers (Python 3.7 compat)."""
        return f"CONST({node.n})"

    def _visit_subscript(self, node: ast.Subscript) -> str:
        """Handle subscript: items[0], data['key']."""
        value = self._visit(node.value)
        # Simplify subscript to INDEX operation
        return f"INDEX({value})"

    def _visit_compare(self, node: ast.Compare) -> str:
        """Handle comparisons: a < b, x == y."""
        left = self._visit(node.left)
        # For simplicity, just capture as COMPARE
        # Could extend to GT, LT, EQ, etc.
        comparators = [self._visit(c) for c in node.comparators]
        return f"COMPARE({left}, {', '.join(comparators)})"

    def _visit_ifexp(self, node: ast.IfExp) -> str:
        """Handle ternary: a if cond else b."""
        test = self._visit(node.test)
        body = self._visit(node.body)
        orelse = self._visit(node.orelse)
        return f"IF({test}, {body}, {orelse})"


# Module-level instance for convenience
_extractor = GraphExtractor()


def extract_computation_graph(dsl_script: str) -> Optional[str]:
    """Extract computation graph from DSL script.

    Args:
        dsl_script: Python expression or JSON DSL dict

    Returns:
        Graph string like "MUL(param_0, param_1)" or None

    Examples:
        >>> extract_computation_graph("a * b")
        'MUL(param_0, param_1)'

        >>> extract_computation_graph("base ** exponent")
        'POW(param_0, param_1)'

        >>> extract_computation_graph('{"type": "math", "script": "x + y"}')
        'ADD(param_0, param_1)'

        >>> extract_computation_graph("(price * quantity) + tax")
        'ADD(MUL(param_0, param_1), param_2)'
    """
    return _extractor.extract(dsl_script)


def graphs_equivalent(graph1: Optional[str], graph2: Optional[str]) -> bool:
    """Check if two computation graphs are structurally equivalent.

    Graphs are equivalent if they have the same structure, regardless
    of param numbering (since params are assigned in order of appearance).

    Args:
        graph1: First graph string
        graph2: Second graph string

    Returns:
        True if graphs represent the same computation structure
    """
    if graph1 is None or graph2 is None:
        return graph1 == graph2
    return graph1 == graph2


# Natural language expansions for graph operations
# Used to make graph embeddings comparable to operation embeddings
_OP_TO_NL = {
    "ADD": "add",
    "SUB": "subtract",
    "MUL": "multiply",
    "DIV": "divide",
    "FLOORDIV": "integer divide",
    "MOD": "find remainder of",
    "POW": "raise to power",
    "NEG": "negate",
    "POS": "positive",
    "SQRT": "square root of",
    "ABS": "absolute value of",
    "GCD": "greatest common divisor of",
    "LCM": "least common multiple of",
    "FACTORIAL": "factorial of",
    "COMB": "combinations of",
    "PERM": "permutations of",
    "REDUCE_SUM": "sum of",
    "REDUCE_MIN": "minimum of",
    "REDUCE_MAX": "maximum of",
    "LEN": "length of",
    "ROUND": "round",
    "INT": "convert to integer",
    "FLOAT": "convert to decimal",
    "CEIL": "ceiling of",
    "FLOOR": "floor of",
    "COMPARE": "compare",
    "IF": "if condition then",
    "INDEX": "get element from",
}


def graph_to_natural_language(graph: str) -> str:
    """Convert computation graph to natural language for embedding.

    This makes graph embeddings comparable to operation embeddings
    (which are extracted as natural language from problem text).

    Args:
        graph: Computation graph string (e.g., "MUL(param_0, param_1)")

    Returns:
        Natural language description (e.g., "multiply two values")

    Examples:
        >>> graph_to_natural_language("MUL(param_0, param_1)")
        'multiply two values'

        >>> graph_to_natural_language("POW(param_0, param_1)")
        'raise first value to power of second value'

        >>> graph_to_natural_language("ADD(MUL(param_0, param_1), param_2)")
        'add multiply two values and third value'
    """
    if not graph:
        return ""

    # Simple recursive expansion
    def expand(g: str) -> str:
        g = g.strip()

        # Check for CONST
        if g.startswith("CONST("):
            return "constant"

        # Check for param
        if g.startswith("param_"):
            # Extract param number for ordinal
            try:
                num = int(g.split("_")[1])
                ordinals = ["first", "second", "third", "fourth", "fifth"]
                return f"{ordinals[num] if num < len(ordinals) else f'param {num}'} value"
            except (IndexError, ValueError):
                return "value"

        # Find the operation and arguments
        paren_idx = g.find("(")
        if paren_idx == -1:
            # No arguments - might be unknown op
            return _OP_TO_NL.get(g, g.lower())

        op = g[:paren_idx]
        args_str = g[paren_idx + 1:-1]  # Remove outer parens

        # Parse arguments (handle nested parens)
        args = []
        depth = 0
        current = ""
        for char in args_str:
            if char == "(":
                depth += 1
                current += char
            elif char == ")":
                depth -= 1
                current += char
            elif char == "," and depth == 0:
                if current.strip():
                    args.append(current.strip())
                current = ""
            else:
                current += char
        if current.strip():
            args.append(current.strip())

        # Expand arguments
        expanded_args = [expand(arg) for arg in args]

        # Build natural language based on operation
        op_nl = _OP_TO_NL.get(op, op.lower())

        if len(expanded_args) == 0:
            return op_nl
        elif len(expanded_args) == 1:
            return f"{op_nl} {expanded_args[0]}"
        elif len(expanded_args) == 2:
            return f"{op_nl} {expanded_args[0]} and {expanded_args[1]}"
        else:
            return f"{op_nl} {', '.join(expanded_args[:-1])}, and {expanded_args[-1]}"

    return expand(graph)


# Graph embedding cache
# GRAPH_EMBEDDING_CACHE_MAX_SIZE imported from config.py (per "The Flow": no magic numbers)
_graph_embedding_cache: dict[str, list[float]] = {}


def clear_graph_embedding_cache() -> None:
    """Clear the graph embedding cache."""
    global _graph_embedding_cache
    _graph_embedding_cache.clear()
    logger.debug("[graph_embed] Cache cleared")


def embed_computation_graph_sync(
    embedder,  # Sync embedder with embed(text) method
    graph: str,
) -> Optional[list[float]]:
    """Sync version of embed_computation_graph for use with sync Embedder.

    Converts graph to natural language first so embeddings are comparable
    to operation embeddings.

    Args:
        embedder: Sync embedder with embed(text) -> np.ndarray method
        graph: Computation graph string (e.g., "MUL(param_0, param_1)")

    Returns:
        Embedding as list[float] or None
    """
    global _graph_embedding_cache

    if not graph:
        return None

    # Check cache
    if graph in _graph_embedding_cache:
        logger.debug("[graph_embed] Cache hit for: %s", graph[:50])
        return _graph_embedding_cache[graph]

    # Convert graph to natural language
    nl_description = graph_to_natural_language(graph)
    logger.debug("[graph_embed] Expanded '%s' to '%s'", graph[:30], nl_description[:50])

    try:
        # Use sync embedder
        # Per CLAUDE.md "New Favorite Pattern": Use cached_embed
        embedding_array = cached_embed(nl_description, embedder)
        embedding = embedding_array.tolist()

        # Cache
        if len(_graph_embedding_cache) >= GRAPH_EMBEDDING_CACHE_MAX_SIZE:
            keys_to_remove = list(_graph_embedding_cache.keys())[:GRAPH_EMBEDDING_CACHE_MAX_SIZE // 10]
            for k in keys_to_remove:
                del _graph_embedding_cache[k]

        _graph_embedding_cache[graph] = embedding
        logger.debug("[graph_embed] Cached sync embedding for: %s", graph[:50])

        return embedding

    except Exception as e:
        logger.error("[graph_embed] Failed to embed graph (sync): %s", e)
        return None
