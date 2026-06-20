"""csp_registry.py — the FactorType REGISTRY mechanism.

This is the registration plumbing (spec §2.5): `register(registry, ftype, predicate,
specialized_propagator=None, arity_hint=None)` + lookup. It is the MECHANISM, not the
content: it contains ZERO domain identifiers (no `edges`, no `color`, no `not_equal`,
no `cage`, no `gate`, no `dsatur`, no `ac3`, no `sudoku`). The domain predicates
themselves live in mycelium/csp_domains.py and are registered THROUGH this module.

A registry is just `dict[int, FactorType]` mapping a factor-type id (the engine's
`latent_type` value) to the FactorType carrying its predicate (+ optional specialized
propagator + arity hint). Adding a domain = a few `register()` calls + a bridge — no
search code, no core change.
"""

from __future__ import annotations

import ast
import os
import sys
from typing import Any, Callable, Optional

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mycelium.csp_core import (
    Consistency,
    FactorType,
    assert_hole_monotone,
)


def new_registry() -> dict:
    """An empty FactorType registry (ftype -> FactorType)."""
    return {}


def register(
    registry: dict,
    ftype: int,
    predicate: Callable[[int, Any, tuple], Consistency],
    name: Optional[str] = None,
    specialized_propagator: Optional[Callable] = None,
    arity_hint: Optional[int] = None,
    check_alphabet: Optional[Any] = None,
    representative_params: Optional[Any] = None,
) -> dict:
    """Register a FactorType under `ftype`. Returns the registry for chaining.

    name   : human-readable label (defaults to f"ftype{ftype}"). Purely descriptive.
    predicate(ftype, params, member_values) -> Consistency : the ONLY domain code.
    specialized_propagator : OPTIONAL large-arity fast path (semantics-preserving).
    arity_hint : OPTIONAL scope size, for the GAC cost guard.
    check_alphabet : OPTIONAL value alphabet; if given (with arity_hint), the L-MONO
        hole-monotonicity contract is ENFORCED at registration (raises on violation) so
        the soundness gate can never be wired to a non-monotone predicate unnoticed.
    representative_params : the params to use for that check on a param-CARRYING predicate
        (e.g. a cage's (op, target)); leave None for param-free types (not-equal).
    """
    registry[ftype] = FactorType(
        name=(name if name is not None else f"ftype{ftype}"),
        predicate=predicate,
        specialized_propagator=specialized_propagator,
        arity_hint=arity_hint,
    )
    if check_alphabet is not None and arity_hint is not None:
        if not assert_hole_monotone(ftype, representative_params, predicate,
                                    check_alphabet, arity_hint):
            raise ValueError(
                f"FactorType {ftype} ({name}) violates the L-MONO hole-monotonicity "
                f"contract the partial-soundness gate requires -- refusing to register."
            )
    return registry


def lookup(registry: dict, ftype: int) -> FactorType:
    """Return the FactorType registered under `ftype` (KeyError if unregistered)."""
    return registry[ftype]


def verify_registry_monotone(registry: dict, alphabet, samples: int = 200,
                             seed: int = 0, params_for: Optional[dict] = None) -> dict:
    """Run the L-MONO randomized monotonicity selftest (csp_core.assert_hole_monotone)
    for every registered type over `alphabet`, using arity_hint as the tuple arity
    (skipping types with no arity_hint). Returns {ftype: bool}. A False is a registered
    predicate that violates the hole-monotonicity contract the soundness gate needs.

    params_for : OPTIONAL {ftype: representative params} for param-CARRYING predicates
    (e.g. a cage's (op, target)); param-FREE types (not-equal) default to None. Without
    it a param-carrying predicate would be checked with None and could misbehave -- supply
    representative params per parameterized type."""
    pf = params_for or {}
    out = {}
    for ftype, ft in registry.items():
        if ft.arity_hint is None:
            continue
        out[ftype] = assert_hole_monotone(
            ftype, pf.get(ftype), ft.predicate, alphabet, ft.arity_hint,
            samples=samples, seed=seed,
        )
    return out


def _ast_parse_ok() -> bool:
    with open(os.path.abspath(__file__)) as f:
        src = f.read()
    try:
        ast.parse(src)
        return True
    except SyntaxError as e:
        print(f"[ast.parse] FAILED: {e}", flush=True)
        return False


if __name__ == "__main__":
    parse_ok = _ast_parse_ok()
    print(f"[ast.parse] ok={parse_ok}", flush=True)
    sys.exit(0 if parse_ok else 1)
