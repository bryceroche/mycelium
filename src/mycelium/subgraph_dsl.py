"""
Sub-Graph DSL: Composable computation sub-graphs for span templates.

Each deduplicated template has a 1:1 sub-graph DSL that defines:
- params: values extracted from span text at inference
- inputs: values wired from upstream sub-graphs in the DAG
- steps: ordered computation steps
- output: single value exposed downstream

Sub-graphs compose into a DAG via cross-attention wiring between spans.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Allowed operators for GSM8K
OPERATORS = {
    "SET": lambda args: args[0],
    "ADD": lambda args: args[0] + args[1],
    "SUB": lambda args: args[0] - args[1],
    "MUL": lambda args: args[0] * args[1],
    "DIV": lambda args: args[0] / args[1] if args[1] != 0 else float("inf"),
    "MOD": lambda args: args[0] % args[1] if args[1] != 0 else float("inf"),
    "NEG": lambda args: -args[0],
}


@dataclass
class SubGraphStep:
    """A single computation step within a sub-graph."""

    var: str  # Variable name this step produces
    op: str  # Operator: SET, ADD, SUB, MUL, DIV, MOD, NEG
    args: List[Union[str, float]]  # Variable names or literal numbers

    def to_dict(self) -> dict:
        return {"var": self.var, "op": self.op, "args": self.args}

    @classmethod
    def from_dict(cls, d: dict) -> SubGraphStep:
        return cls(var=d["var"], op=d["op"], args=d["args"])


@dataclass
class SubGraphDSL:
    """A composable sub-graph DSL for a span template.

    Each template has exactly one SubGraphDSL that defines its computation.
    Sub-graphs compose into a DAG: outputs wire to downstream inputs via
    cross-attention between spans.
    """

    template_id: str
    pattern: str
    params: Dict[str, str]  # {var_name: description} — extracted from span text
    inputs: Dict[str, str]  # {var_name: description} — wired from upstream sub-graphs
    steps: List[SubGraphStep]
    output: str  # Which variable is exposed downstream

    def validate(self) -> List[str]:
        """Validate the DSL. Returns list of errors (empty = valid)."""
        errors = []

        # Check operators are valid
        for step in self.steps:
            if step.op not in OPERATORS:
                errors.append(
                    f"Step '{step.var}': unknown operator '{step.op}'. "
                    f"Allowed: {list(OPERATORS.keys())}"
                )

        # Check all variable references resolve
        available = set(self.params.keys()) | set(self.inputs.keys())
        for step in self.steps:
            for arg in step.args:
                if isinstance(arg, str) and arg not in available:
                    errors.append(
                        f"Step '{step.var}': references undefined variable '{arg}'. "
                        f"Available: {available}"
                    )
            available.add(step.var)

        # Check output exists
        if self.output not in available:
            errors.append(
                f"Output '{self.output}' not found in available variables: {available}"
            )

        # Check no duplicate step vars
        step_vars = [s.var for s in self.steps]
        seen = set()
        for v in step_vars:
            if v in seen:
                errors.append(f"Duplicate step variable: '{v}'")
            seen.add(v)

        # Check step vars don't shadow params/inputs
        for v in step_vars:
            if v in self.params:
                errors.append(f"Step var '{v}' shadows param '{v}'")
            if v in self.inputs:
                errors.append(f"Step var '{v}' shadows input '{v}'")

        # Check arity
        arity = {
            "SET": 1,
            "NEG": 1,
            "ADD": 2,
            "SUB": 2,
            "MUL": 2,
            "DIV": 2,
            "MOD": 2,
        }
        for step in self.steps:
            expected = arity.get(step.op)
            if expected is not None and len(step.args) != expected:
                errors.append(
                    f"Step '{step.var}': {step.op} expects {expected} args, "
                    f"got {len(step.args)}"
                )

        return errors

    def execute(
        self,
        param_values: Dict[str, float],
        input_values: Dict[str, float],
    ) -> float:
        """Execute the sub-graph with concrete values.

        Args:
            param_values: {var_name: value} for params extracted from span text
            input_values: {var_name: value} for inputs wired from upstream sub-graphs

        Returns:
            The output value.
        """
        # Build the variable environment
        env: Dict[str, float] = {}
        env.update(input_values)
        env.update(param_values)

        # Execute steps in order
        for step in self.steps:
            resolved_args = []
            for arg in step.args:
                if isinstance(arg, (int, float)):
                    resolved_args.append(float(arg))
                elif isinstance(arg, str):
                    if arg not in env:
                        raise ValueError(
                            f"Step '{step.var}': variable '{arg}' not in environment. "
                            f"Available: {list(env.keys())}"
                        )
                    resolved_args.append(env[arg])
                else:
                    raise TypeError(
                        f"Step '{step.var}': unexpected arg type {type(arg)}: {arg}"
                    )

            op_fn = OPERATORS.get(step.op)
            if op_fn is None:
                raise ValueError(f"Unknown operator: {step.op}")

            env[step.var] = op_fn(resolved_args)

        if self.output not in env:
            raise ValueError(
                f"Output '{self.output}' not in environment after execution. "
                f"Available: {list(env.keys())}"
            )

        return env[self.output]

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "template_id": self.template_id,
            "pattern": self.pattern,
            "params": self.params,
            "inputs": self.inputs,
            "steps": [s.to_dict() for s in self.steps],
            "output": self.output,
        }

    @classmethod
    def from_dict(cls, d: dict, template_id: str = "unknown", pattern: str = "") -> SubGraphDSL:
        """Deserialize from dict.

        Args:
            d: Dict with params, inputs, steps, output (and optionally template_id, pattern)
            template_id: Fallback template_id if not in dict
            pattern: Fallback pattern if not in dict
        """
        return cls(
            template_id=d.get("template_id", template_id),
            pattern=d.get("pattern", pattern),
            params=d.get("params", {}),
            inputs=d.get("inputs", {}),
            steps=[SubGraphStep.from_dict(s) for s in d.get("steps", [])],
            output=d.get("output", "result"),
        )

    def __repr__(self) -> str:
        step_strs = []
        for s in self.steps:
            args_str = ", ".join(str(a) for a in s.args)
            step_strs.append(f"  {s.var} = {s.op}({args_str})")
        steps_block = "\n".join(step_strs)
        return (
            f"SubGraphDSL({self.template_id})\n"
            f"  pattern: {self.pattern}\n"
            f"  params: {self.params}\n"
            f"  inputs: {self.inputs}\n"
            f"{steps_block}\n"
            f"  → {self.output}"
        )


def load_subgraph_dsls(path: Union[str, Path]) -> Dict[str, SubGraphDSL]:
    """Load sub-graph DSLs from a JSON file.

    Returns dict keyed by template_id.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    dsls = {}
    templates = data if isinstance(data, list) else data.values()
    for d in templates:
        # Support both standalone DSL files and template files with embedded DSL
        if "steps" in d:
            dsl = SubGraphDSL.from_dict(d)
        elif "subgraph" in d:
            dsl = SubGraphDSL.from_dict(d["subgraph"])
            dsl.template_id = d.get("template_id", dsl.template_id)
            dsl.pattern = d.get("pattern", dsl.pattern)
        else:
            continue
        dsls[dsl.template_id] = dsl

    return dsls


def save_subgraph_dsls(dsls: Dict[str, SubGraphDSL], path: Union[str, Path]) -> None:
    """Save sub-graph DSLs to a JSON file."""
    path = Path(path)
    data = [dsl.to_dict() for dsl in dsls.values()]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
