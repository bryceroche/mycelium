"""csp_search.py — DEPRECATED compatibility shim (Phase-0 refactor, June-19).

The coloring-specific branch-and-propagate skeleton that used to live here has been
SPLIT into the general, predicate-driven search tier
(docs/general_factor_graph_search.md §6.1):

  * mycelium/csp_core.py      — the GENERAL core (Factor/Problem/CSPState, the shared
                                backtrack_search/solve_symbolic, verify_complete,
                                is_consistent_partial, gac_propagate, mrv/lcv,
                                assign_var). ZERO domain identifiers.
  * mycelium/csp_registry.py  — the FactorType registry mechanism.
  * mycelium/csp_domains.py   — the domain content (the coloring not-equal predicate +
                                problem_from_coloring bridge + coloring verifier wrappers).
  * mycelium/csp_coloring_legacy.py — the FROZEN byte-for-byte copy of the OLD module,
                                kept as the parity REFERENCE for the behavior-preservation
                                gate (scripts/test_csp_parity.py).

For backward compatibility this module RE-EXPORTS the legacy coloring API from
mycelium/csp_coloring_legacy.py so any old importer keeps working. NEW code should import
the general core (mycelium/csp_core.py) + the coloring domain (mycelium/csp_domains.py),
NOT this shim. The published coloring run-commands flow through scripts/search_coloring.py,
which already drives the general core.
"""

from __future__ import annotations

# Re-export the legacy coloring API verbatim (the frozen reference module).
from mycelium.csp_coloring_legacy import (  # noqa: F401
    UNASSIGNED,
    Assignment,
    CSPState,
    Edges,
    ac3_propagate,
    assign_vertex,
    backtrack_search,
    build_adjacency,
    dsatur_varorder,
    edges_from_membership,
    has_empty_domain,
    is_complete_proper,
    is_proper_partial,
    lcv_valorder,
    make_initial_state,
    noop_propagate,
    normalize_edges,
    random_valorder,
    random_varorder,
    solve_symbolic,
)
