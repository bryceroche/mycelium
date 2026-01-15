"""Observability Dashboard for Mycelium.

Provides at-a-glance metrics for:
- Signature health (success rates, traffic, staleness)
- Routing decision visualization (hot paths)
- Cold-start progress metrics
"""

from mycelium.dashboard.health import (
    SignatureHealth,
    get_signature_health_report,
    get_health_summary,
)
from mycelium.dashboard.routing import (
    RouteNode,
    get_hot_paths,
    get_routing_tree,
    format_routing_tree,
)
from mycelium.dashboard.cold_start import (
    ColdStartProgress,
    get_cold_start_progress,
)
from mycelium.dashboard.cli import dashboard_cli

__all__ = [
    "SignatureHealth",
    "get_signature_health_report",
    "get_health_summary",
    "RouteNode",
    "get_hot_paths",
    "get_routing_tree",
    "format_routing_tree",
    "ColdStartProgress",
    "get_cold_start_progress",
    "dashboard_cli",
]
