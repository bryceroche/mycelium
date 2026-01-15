"""Routing Decision Visualization.

Shows which paths through the signature hierarchy are hot:
- Tree structure visualization
- Traffic flow through routes
- Hot path highlighting
"""

import sqlite3
from dataclasses import dataclass, field
from typing import Optional

from mycelium.config import DB_PATH


@dataclass
class RouteNode:
    """A node in the routing tree."""

    id: int
    step_type: str
    description: str
    depth: int

    # Traffic metrics
    uses: int
    successes: int
    success_rate: float

    # Node type
    is_umbrella: bool
    is_root: bool
    dsl_type: str

    # Children (for tree structure)
    children: list["RouteNode"] = field(default_factory=list)

    # Traffic share (% of total uses flowing through this node)
    traffic_share: float = 0.0

    # Heat level (0.0 = cold, 1.0 = hottest path)
    heat: float = 0.0


def get_routing_tree(db_path: str = None, max_depth: int = 10) -> Optional[RouteNode]:
    """Build the complete routing tree starting from root.

    Args:
        db_path: Path to database (default from config)
        max_depth: Maximum depth to traverse

    Returns:
        Root RouteNode with children populated, or None if no root
    """
    if db_path is None:
        db_path = DB_PATH

    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row

    try:
        # Get total uses for traffic share calculation
        total_row = conn.execute("SELECT SUM(uses) FROM step_signatures").fetchone()
        total_uses = total_row[0] or 1

        # Get max uses for heat calculation
        max_row = conn.execute("SELECT MAX(uses) FROM step_signatures").fetchone()
        max_uses = max_row[0] or 1

        # Get root
        root_row = conn.execute(
            "SELECT * FROM step_signatures WHERE is_root = 1 LIMIT 1"
        ).fetchone()

        if not root_row:
            return None

        # Build tree recursively
        root = _build_node(conn, root_row, total_uses, max_uses, 0, max_depth)

        return root

    finally:
        conn.close()


def _build_node(
    conn,
    row,
    total_uses: int,
    max_uses: int,
    current_depth: int,
    max_depth: int,
) -> RouteNode:
    """Build a RouteNode from a database row with children."""
    uses = row["uses"] or 0
    successes = row["successes"] or 0
    success_rate = successes / uses if uses > 0 else 0.0

    node = RouteNode(
        id=row["id"],
        step_type=row["step_type"] or "unknown",
        description=(row["description"] or "")[:60],
        depth=row["depth"] or 0,
        uses=uses,
        successes=successes,
        success_rate=success_rate,
        is_umbrella=bool(row["is_semantic_umbrella"]),
        is_root=bool(row["is_root"]),
        dsl_type=row["dsl_type"] or "unknown",
        traffic_share=uses / total_uses if total_uses > 0 else 0.0,
        heat=uses / max_uses if max_uses > 0 else 0.0,
    )

    # Get children if within depth limit
    if current_depth < max_depth and node.is_umbrella:
        cursor = conn.execute(
            """SELECT s.* FROM signature_relationships r
               JOIN step_signatures s ON r.child_id = s.id
               WHERE r.parent_id = ?
               ORDER BY s.uses DESC""",
            (node.id,)
        )

        for child_row in cursor.fetchall():
            child = _build_node(
                conn, child_row, total_uses, max_uses,
                current_depth + 1, max_depth
            )
            node.children.append(child)

    return node


def get_hot_paths(
    db_path: str = None,
    top_n: int = 10,
) -> list[list[RouteNode]]:
    """Get the hottest routing paths (most traffic).

    Args:
        db_path: Path to database (default from config)
        top_n: Number of top paths to return

    Returns:
        List of paths, each path is a list of RouteNodes from root to leaf
    """
    tree = get_routing_tree(db_path)
    if not tree:
        return []

    # Collect all root-to-leaf paths
    paths = []
    _collect_paths(tree, [], paths)

    # Sort by total traffic (sum of uses along path)
    paths.sort(key=lambda p: sum(n.uses for n in p), reverse=True)

    return paths[:top_n]


def _collect_paths(
    node: RouteNode,
    current_path: list[RouteNode],
    all_paths: list[list[RouteNode]],
):
    """Recursively collect all root-to-leaf paths."""
    current_path = current_path + [node]

    if not node.children:
        # Leaf node - save path
        all_paths.append(current_path)
    else:
        # Internal node - recurse to children
        for child in node.children:
            _collect_paths(child, current_path, all_paths)


def format_routing_tree(
    root: RouteNode,
    max_depth: int = 5,
    show_heat: bool = True,
) -> str:
    """Format routing tree as ASCII art.

    Args:
        root: Root RouteNode
        max_depth: Maximum depth to display
        show_heat: Whether to show heat indicators

    Returns:
        Formatted string with tree visualization
    """
    lines = []
    _format_node(root, lines, "", True, 0, max_depth, show_heat)
    return "\n".join(lines)


def _format_node(
    node: RouteNode,
    lines: list[str],
    prefix: str,
    is_last: bool,
    depth: int,
    max_depth: int,
    show_heat: bool,
):
    """Recursively format a node and its children."""
    # Heat indicator
    if show_heat:
        if node.heat >= 0.7:
            heat_char = "[HOT]"
        elif node.heat >= 0.4:
            heat_char = "[~~~]"
        elif node.heat >= 0.1:
            heat_char = "[   ]"
        else:
            heat_char = "[---]"
    else:
        heat_char = ""

    # Node type indicator
    if node.is_root:
        type_char = "(R)"
    elif node.is_umbrella:
        type_char = "(U)"
    else:
        type_char = "(L)"

    # Success rate
    success_str = f"{node.success_rate*100:.0f}%" if node.uses > 0 else "new"

    # Build line
    connector = "\\-- " if is_last else "+-- "
    line = (
        f"{prefix}{connector}{type_char} {node.step_type[:20]:20} "
        f"[{node.uses:>4} uses, {success_str:>4}] {heat_char}"
    )
    lines.append(line)

    # Stop at max depth
    if depth >= max_depth:
        if node.children:
            child_prefix = prefix + ("    " if is_last else "|   ")
            lines.append(f"{child_prefix}... ({len(node.children)} more children)")
        return

    # Format children
    child_prefix = prefix + ("    " if is_last else "|   ")
    for i, child in enumerate(node.children):
        is_last_child = (i == len(node.children) - 1)
        _format_node(
            child, lines, child_prefix, is_last_child,
            depth + 1, max_depth, show_heat
        )


def format_hot_paths(paths: list[list[RouteNode]]) -> str:
    """Format hot paths as text.

    Args:
        paths: List of paths from get_hot_paths()

    Returns:
        Formatted string showing hot paths
    """
    if not paths:
        return "No routing paths found."

    lines = []
    lines.append("=== HOT ROUTING PATHS ===")
    lines.append("")

    for i, path in enumerate(paths, 1):
        total_uses = sum(n.uses for n in path)
        avg_success = (
            sum(n.success_rate for n in path) / len(path)
            if path else 0.0
        )

        lines.append(f"Path #{i} ({total_uses} total uses, {avg_success*100:.0f}% avg success)")

        # Show path as chain
        path_str = " -> ".join(n.step_type[:15] for n in path)
        lines.append(f"  {path_str}")

        # Show individual nodes
        for j, node in enumerate(path):
            indent = "  " * (j + 1)
            type_char = "R" if node.is_root else ("U" if node.is_umbrella else "L")
            lines.append(
                f"{indent}[{type_char}] {node.step_type:20} "
                f"({node.uses} uses, {node.success_rate*100:.0f}%)"
            )

        lines.append("")

    return "\n".join(lines)


def get_routing_stats(db_path: str = None) -> dict:
    """Get aggregate routing statistics.

    Args:
        db_path: Path to database (default from config)

    Returns:
        Dict with routing statistics
    """
    if db_path is None:
        db_path = DB_PATH

    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row

    try:
        stats = {}

        # Total relationships
        row = conn.execute("SELECT COUNT(*) FROM signature_relationships").fetchone()
        stats["total_edges"] = row[0] if row else 0

        # Max depth
        row = conn.execute("SELECT MAX(depth) FROM step_signatures").fetchone()
        stats["max_depth"] = row[0] or 0

        # Average children per umbrella
        row = conn.execute(
            """SELECT AVG(child_count) FROM (
                SELECT COUNT(*) as child_count
                FROM signature_relationships
                GROUP BY parent_id
            )"""
        ).fetchone()
        stats["avg_children_per_umbrella"] = row[0] or 0.0

        # Leaf node count (no children)
        row = conn.execute(
            """SELECT COUNT(*) FROM step_signatures s
               WHERE NOT EXISTS (
                   SELECT 1 FROM signature_relationships r WHERE r.parent_id = s.id
               )"""
        ).fetchone()
        stats["leaf_count"] = row[0] if row else 0

        # Orphan count (no parent, not root)
        row = conn.execute(
            """SELECT COUNT(*) FROM step_signatures s
               WHERE is_root = 0
               AND NOT EXISTS (
                   SELECT 1 FROM signature_relationships r WHERE r.child_id = s.id
               )"""
        ).fetchone()
        stats["orphan_count"] = row[0] if row else 0

        # Traffic concentration (Gini-like metric)
        cursor = conn.execute(
            "SELECT uses FROM step_signatures ORDER BY uses ASC"
        )
        uses_list = [row[0] or 0 for row in cursor.fetchall()]

        if uses_list and sum(uses_list) > 0:
            n = len(uses_list)
            total = sum(uses_list)
            cumsum = 0
            gini_sum = 0
            for i, u in enumerate(uses_list):
                cumsum += u
                gini_sum += (2 * (i + 1) - n - 1) * u
            stats["traffic_gini"] = gini_sum / (n * total) if n * total > 0 else 0.0
        else:
            stats["traffic_gini"] = 0.0

        return stats

    finally:
        conn.close()


def format_routing_stats(stats: dict) -> str:
    """Format routing statistics as text.

    Args:
        stats: Dict from get_routing_stats()

    Returns:
        Formatted string
    """
    lines = []
    lines.append("=== ROUTING STATISTICS ===")
    lines.append("")
    lines.append(f"Total Edges (parent-child): {stats.get('total_edges', 0)}")
    lines.append(f"Maximum Depth:              {stats.get('max_depth', 0)}")
    lines.append(f"Avg Children per Umbrella:  {stats.get('avg_children_per_umbrella', 0):.1f}")
    lines.append(f"Leaf Nodes:                 {stats.get('leaf_count', 0)}")
    lines.append(f"Orphan Nodes:               {stats.get('orphan_count', 0)}")
    lines.append("")

    gini = stats.get("traffic_gini", 0.0)
    if gini < 0.3:
        traffic_desc = "evenly distributed"
    elif gini < 0.6:
        traffic_desc = "moderately concentrated"
    else:
        traffic_desc = "highly concentrated (few hot paths)"

    lines.append(f"Traffic Concentration:      {gini:.2f} ({traffic_desc})")

    return "\n".join(lines)
