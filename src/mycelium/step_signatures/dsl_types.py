"""DSL Types: Core dataclasses and enums for DSL execution.

This module contains:
- DSLLayer: Execution layer enum
- DSLSpec: DSL script specification
- ValueType: Parameter value classification
"""

__all__ = [
    "ValueType",
    "DSLLayer",
    "DSLSpec",
]

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ValueType(Enum):
    """Classification of a parameter value."""
    NUMBER = "number"
    EXPRESSION = "expression"
    UNCERTAIN = "uncertain"


class DSLLayer(Enum):
    """Execution layer for DSL scripts."""
    MATH = "math"
    SYMPY = "sympy"
    CUSTOM = "custom"
    DECOMPOSE = "decompose"  # Needs decomposition into atomic steps
    ROUTER = "router"  # Routing layer - delegates to children
    NONE = "none"  # No DSL execution, use LLM fallback


@dataclass
class DSLSpec:
    """Specification for a DSL script."""
    layer: DSLLayer
    script: str
    params: list[str]
    aliases: dict[str, list[str]] = field(default_factory=dict)
    param_types: dict[str, str] = field(default_factory=dict)  # param -> "numeric"|"symbolic"|"any"
    param_roles: dict[str, str] = field(default_factory=dict)  # param -> semantic role description
    output_type: str = "numeric"
    fallback: str = "decompose"
    purpose: str = ""  # Human-readable description of what this DSL does
    _purpose_embedding: Optional[Any] = field(default=None, repr=False)  # Cached embedding
    _param_role_embeddings: dict[str, Any] = field(default_factory=dict, repr=False)  # Cached param embeddings

    @classmethod
    def from_json(cls, json_str: str) -> Optional["DSLSpec"]:
        """Parse JSON string into DSLSpec."""
        if not json_str:
            return None
        try:
            d = json.loads(json_str)
            layer = DSLLayer(d.get("type", "math"))
            # Handle params as list or dict (extract keys if dict)
            raw_params = d.get("params", [])
            if isinstance(raw_params, dict):
                params = list(raw_params.keys()) if raw_params else []
            elif isinstance(raw_params, list):
                params = raw_params
            else:
                params = []

            # Infer param types based on DSL layer if not specified
            param_types = d.get("param_types", {})
            if not param_types and layer == DSLLayer.MATH:
                # Math DSLs expect numeric inputs by default
                param_types = {p: "numeric" for p in params}

            # Get param roles (semantic descriptions)
            param_roles = d.get("param_roles", {})

            return cls(
                layer=layer,
                script=d.get("script", ""),
                params=params,
                aliases=d.get("aliases", {}),
                param_types=param_types,
                param_roles=param_roles,
                output_type=d.get("output_type", "numeric"),
                fallback=d.get("fallback", "decompose"),
                purpose=d.get("purpose", ""),
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("Failed to parse DSL spec: %s", e)
            return None

    def to_json(self) -> str:
        """Serialize to JSON."""
        data = {
            "type": self.layer.value,
            "script": self.script,
            "params": self.params,
            "output_type": self.output_type,
            "fallback": self.fallback,
        }
        if self.aliases:
            data["aliases"] = self.aliases
        if self.purpose:
            data["purpose"] = self.purpose
        return json.dumps(data)

    def get_purpose_embedding(self):
        """Get or compute the purpose embedding (lazy).

        If no purpose is set, infers one from the script.
        """
        if self._purpose_embedding is not None:
            return self._purpose_embedding

        # Get purpose text (infer from script if not set)
        purpose_text = self.purpose or self._infer_purpose_from_script()
        if not purpose_text:
            return None

        # Compute embedding
        from mycelium.embedder import Embedder
        embedder = Embedder.get_instance()
        self._purpose_embedding = embedder.embed(purpose_text)
        return self._purpose_embedding

    def _infer_purpose_from_script(self) -> str:
        """Infer a purpose description from the DSL script.

        Uses the script itself as the semantic description. The embedding
        similarity will handle matching - no need for hardcoded mappings.
        """
        # The script IS the purpose - let embeddings find semantic matches
        return f"compute: {self.script[:60]}"

    def get_param_role_embedding(self, param: str):
        """Get or compute embedding for a parameter's semantic role.

        If no role is defined, infers one from the param name and script context.
        """
        if param in self._param_role_embeddings:
            return self._param_role_embeddings[param]

        # Get role text (explicit or inferred)
        role_text = self.param_roles.get(param) or self._infer_param_role(param)
        if not role_text:
            return None

        # Compute embedding
        from mycelium.embedder import Embedder
        embedder = Embedder.get_instance()
        embedding = embedder.embed(role_text)
        self._param_role_embeddings[param] = embedding
        return embedding

    def _infer_param_role(self, param: str) -> str:
        """Infer semantic role description for a parameter.

        Uses the param name itself as the semantic description. The embedding
        similarity will handle matching - no need for hardcoded mappings.
        Human-readable param names already contain semantic meaning.
        """
        # Convert snake_case to readable: "base_value" -> "base value"
        readable = param.replace("_", " ").lower()
        return f"the {readable} parameter"

    def get_dsl_type(self) -> str:
        """Determine the DSL type category.

        Returns 'default' - all DSLs use the same thresholds.
        Per-type thresholds were manual tuning that doesn't scale.
        Let the system learn appropriate confidence from execution data.
        """
        return "default"

    def match_param(self, param: str, context_keys: list[str]) -> Optional[str]:
        """Find matching context key for a param using aliases.

        Matching priority:
        1. Exact match (param == key)
        2. Param is substring of key (e.g., 'value' in 'base_value')
        3. Key is substring of param
        4. Alias exact match
        5. Alias substring match

        Returns the matching context key, or None if no match.
        """
        param_lower = param.lower()
        aliases = [a.lower() for a in self.aliases.get(param, [])]

        for key in context_keys:
            key_lower = key.lower()

            # Exact match
            if param_lower == key_lower:
                return key

            # Param in key (e.g., 'percentage' in 'step_1_percentage')
            if param_lower in key_lower:
                return key

            # Key in param (e.g., 'pct' in 'percentage')
            if key_lower in param_lower:
                return key

            # Alias matches
            for alias in aliases:
                if alias == key_lower or alias in key_lower or key_lower in alias:
                    return key

        return None

    def map_inputs(self, context: dict[str, Any]) -> dict[str, Any]:
        """Map context values to param names using alias matching.

        Matching strategy:
        1. First try name-based matching (exact, substring, alias)
        2. For unfound params, fall back to positional matching from remaining keys
        3. Aggressively match step_N keys to params by position

        Returns dict with param names as keys and context values as values.
        """
        result = {}
        context_keys = list(context.keys())
        used_keys = set()

        # First pass: name-based matching
        for param in self.params:
            matched_key = self.match_param(param, context_keys)
            if matched_key:
                result[param] = context[matched_key]
                used_keys.add(matched_key)
                context_keys.remove(matched_key)  # Don't reuse

        # Second pass: positional matching for unfound params
        # Sort remaining keys (step_1, step_2, etc. will be in order)
        remaining_keys = sorted([k for k in context.keys() if k not in used_keys])
        unfound_params = [p for p in self.params if p not in result]

        for i, param in enumerate(unfound_params):
            if i < len(remaining_keys):
                result[param] = context[remaining_keys[i]]

        # If still no matches and we have short params (single letter or 2-3 chars)
        # Try aggressive positional matching - these are likely generic math params
        if not result and self.params and context:
            if all(len(p) <= 3 for p in self.params):
                sorted_keys = sorted(context.keys())
                for i, param in enumerate(self.params):
                    if i < len(sorted_keys):
                        result[param] = context[sorted_keys[i]]

        return result

    def compute_confidence(self, context: dict[str, Any]) -> float:
        """Compute confidence score for executing DSL with given context.

        Score based on:
        - Percentage of required params found
        - Quality of matches (exact vs fuzzy vs positional)
        - Type compatibility (numeric for math DSL)

        Returns:
            Float between 0.0 and 1.0. Higher = more confident.
        """
        if not self.params:
            return 1.0  # No params required

        context_keys = list(context.keys())
        matched_count = 0
        fuzzy_matches = 0
        type_mismatches = 0
        positional_fallback = False

        # Try name-based matching first
        for param in self.params:
            matched_key = self.match_param(param, context_keys)
            if matched_key:
                matched_count += 1
                context_keys.remove(matched_key)

                # Check if fuzzy match (not exact)
                if param.lower() != matched_key.lower():
                    fuzzy_matches += 1

                # Check type compatibility for math DSL
                if self.layer == DSLLayer.MATH:
                    value = context[matched_key]
                    if not isinstance(value, (int, float)):
                        try:
                            float(str(value))
                        except (ValueError, TypeError):
                            type_mismatches += 1

        # If no name matches, check positional fallback viability
        if matched_count == 0 and len(context) >= len(self.params):
            positional_fallback = True
            matched_count = len(self.params)
            # All positional matches count as fuzzy
            fuzzy_matches = len(self.params)

            # Check type compatibility for positional values
            if self.layer == DSLLayer.MATH:
                sorted_keys = sorted(context.keys())
                for i in range(len(self.params)):
                    if i < len(sorted_keys):
                        value = context[sorted_keys[i]]
                        if not isinstance(value, (int, float)):
                            try:
                                float(str(value))
                            except (ValueError, TypeError):
                                type_mismatches += 1

        # Base score: percentage of params found
        score = matched_count / len(self.params) if self.params else 0.0

        # Penalty for fuzzy matches (20% per fuzzy match)
        score *= (0.8 ** fuzzy_matches)

        # Penalty for type mismatches in math DSL (50% per mismatch)
        if self.layer == DSLLayer.MATH:
            score *= (0.5 ** type_mismatches)

        return score
