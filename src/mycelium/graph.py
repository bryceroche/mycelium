"""Computation graph for math word problems.

Represents mathematical operations as a directed graph with named variables.
Execution walks the graph and computes the final result.
"""

from dataclasses import dataclass
from typing import List, Dict, Union
from enum import Enum


class Op(Enum):
    """Operation types for computation nodes."""
    SET = "SET"       # SET(value) - initialize variable
    ADD = "ADD"       # ADD(a, b) - a + b
    SUB = "SUB"       # SUB(a, b) - a - b
    MUL = "MUL"       # MUL(a, b) - a * b
    DIV = "DIV"       # DIV(a, b) - a / b


@dataclass
class Node:
    """A single computation node."""
    name: str                           # variable name (e.g., "total", "sold")
    op: Op                              # operation type
    inputs: List[Union[str, float]]     # variable names OR literal values
    span: str = ""                      # source text that produced this node

    def __repr__(self):
        inputs_str = ", ".join(str(i) for i in self.inputs)
        return f"{self.name} = {self.op.value}({inputs_str})"


class Graph:
    """Computation graph with named variables and multiple input references."""

    def __init__(self, entity: str = ""):
        self.entity = entity
        self.nodes: List[Node] = []

    def add(self, name: str, op: Op, inputs: List, span: str = "") -> 'Graph':
        """Add a node to the graph. Returns self for chaining."""
        self.nodes.append(Node(name, op, inputs, span))
        return self

    def execute(self) -> Dict[str, float]:
        """Execute graph, return all variable values."""
        env = {}

        for node in self.nodes:
            # Resolve inputs (variable names → values, literals stay as-is)
            resolved = []
            for inp in node.inputs:
                if isinstance(inp, str):
                    resolved.append(env[inp])
                else:
                    resolved.append(inp)

            # Execute operation
            if node.op == Op.SET:
                env[node.name] = resolved[0]
            elif node.op == Op.ADD:
                env[node.name] = resolved[0] + resolved[1]
            elif node.op == Op.SUB:
                env[node.name] = resolved[0] - resolved[1]
            elif node.op == Op.MUL:
                env[node.name] = resolved[0] * resolved[1]
            elif node.op == Op.DIV:
                env[node.name] = resolved[0] / resolved[1]

        return env

    def result(self) -> float:
        """Execute and return final value."""
        env = self.execute()
        return list(env.values())[-1] if env else 0.0

    def __str__(self):
        lines = [f"Graph({self.entity}):"]
        for n in self.nodes:
            lines.append(f"  {n}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize graph to dictionary."""
        return {
            "entity": self.entity,
            "nodes": [
                {
                    "name": n.name,
                    "op": n.op.value,
                    "inputs": n.inputs,
                    "span": n.span,
                }
                for n in self.nodes
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Graph':
        """Deserialize graph from dictionary."""
        g = cls(data.get("entity", ""))
        for n in data.get("nodes", []):
            g.add(n["name"], Op(n["op"]), n["inputs"], n.get("span", ""))
        return g
