"""
Message Networks — Learned constraint propagation channels

These small MLPs convert execution/validation signals into update messages
for upstream components. They're the ONLY thing that encodes constraint
structure, and they learn it entirely from the joint training signal.

Message flow:
  C5 execution result ──► M_5→4 ──► C4 (refine wiring)
  C4 graph validity   ──► M_4→2 ──► C2 (refine op labels)
  C2 refined beliefs  ──► M_2→3 ──► C3 (refine extraction)

Instead of coding rules like "MUL needs 2 args", the message network
learns from training data that when C5 fails on a MUL node, the useful
update is to shift C2's belief toward ADD or DIV.

~10K parameters each. Negligible compute cost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Base message network
# ---------------------------------------------------------------------------

class MessageNetwork(nn.Module):
    """
    Small MLP that converts signals into update messages.

    3-layer MLP with GELU activation.
    ~10K parameters for typical hidden=128.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Constraint message container
# ---------------------------------------------------------------------------

@dataclass
class ConstraintMessages:
    """Container for messages between components."""
    msg_5to4: Optional[torch.Tensor] = None  # C5 → C4
    msg_4to2: Optional[torch.Tensor] = None  # C4 → C2
    msg_2to3: Optional[torch.Tensor] = None  # C2 → C3


# ---------------------------------------------------------------------------
# Full message network ensemble
# ---------------------------------------------------------------------------

class MessageNetworkEnsemble(nn.Module):
    """
    All message networks for the constraint propagation loop.

    Message flow:
      C5 execution result ──► M_5→4 ──► C4 (refine wiring)
      C4 graph validity   ──► M_4→2 ──► C2 (refine op labels)
      C2 refined beliefs  ──► M_2→3 ──► C3 (refine extraction)

    Each message network takes context from the current component
    plus passthrough from the previous message, enabling multi-hop
    constraint propagation.
    """

    def __init__(
        self,
        n_ops: int,
        max_args: int = 3,
        msg_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.n_ops = n_ops
        self.max_args = max_args
        self.msg_dim = msg_dim

        # M_5→4: execution outcome → graph refinement signal
        # Input: per-node (op_probs + arg_values + exec_success + exec_result)
        exec_input_dim = n_ops + max_args + 2  # op_probs + args + success + result
        self.m_5to4 = MessageNetwork(exec_input_dim, msg_dim, hidden_dim)

        # M_4→2: graph structure validity → operation label refinement
        # Input: per-node (in_degree, out_degree, n_edges, validity) + passthrough
        graph_input_dim = 4 + msg_dim  # graph features + msg from C5
        self.m_4to2 = MessageNetwork(graph_input_dim, msg_dim, hidden_dim)

        # M_2→3: refined op beliefs → extraction refinement
        # Input: per-node (op_prob_delta) + passthrough
        self.m_2to3 = MessageNetwork(n_ops + msg_dim, msg_dim, hidden_dim)

    def compute_5to4(
        self,
        op_probs: torch.Tensor,           # (n, n_ops)
        arg_values: torch.Tensor,         # (n, max_args)
        exec_success: torch.Tensor,       # (n,) per-node success
        exec_results: torch.Tensor,       # (n,) per-node result
    ) -> torch.Tensor:
        """
        C5 → C4 message: how should graph wiring change given execution?

        When execution fails, this network learns what kind of wiring
        changes would help. E.g., if a DIV fails (div by zero), maybe
        the dependency should be rewired to get a different operand.

        Returns: (n, msg_dim)
        """
        x = torch.cat([
            op_probs,                              # (n, n_ops)
            arg_values,                            # (n, max_args)
            exec_success.unsqueeze(-1),            # (n, 1)
            exec_results.unsqueeze(-1),            # (n, 1)
        ], dim=-1)
        return self.m_5to4(x)

    def compute_4to2(
        self,
        adjacency: torch.Tensor,          # (n, n) binary
        validity_score: torch.Tensor,     # scalar or (n,)
        msg_from_5: torch.Tensor,         # (n, msg_dim) passthrough
    ) -> torch.Tensor:
        """
        C4 → C2 message: how should op labels change given graph validity?

        When the graph structure is invalid (cycles, disconnected, etc),
        this network learns which op label changes would help. E.g., if
        too many nodes have no dependencies, maybe some should be SOLVE_*
        instead of basic arithmetic.

        Returns: (n, msg_dim)
        """
        n = adjacency.size(0)

        # Compute graph features per node
        in_degree = adjacency.sum(dim=0)           # (n,) incoming edges
        out_degree = adjacency.sum(dim=1)          # (n,) outgoing edges
        n_edges = adjacency.sum().expand(n)        # scalar → (n,)

        # Expand validity if scalar
        if validity_score.dim() == 0:
            validity_score = validity_score.expand(n)

        graph_features = torch.stack([
            in_degree,
            out_degree,
            n_edges,
            validity_score,
        ], dim=-1)  # (n, 4)

        x = torch.cat([graph_features, msg_from_5], dim=-1)  # (n, 4 + msg_dim)
        return self.m_4to2(x)

    def compute_2to3(
        self,
        op_prob_delta: torch.Tensor,      # (n, n_ops) how beliefs shifted
        msg_from_4: torch.Tensor,         # (n, msg_dim) passthrough
    ) -> torch.Tensor:
        """
        C2 → C3 message: how should extraction change given refined ops?

        When C2's beliefs shift (e.g., from ADD to MUL), this network
        learns how C3's extraction should adapt. E.g., MUL might need
        different positional patterns than ADD.

        Returns: (n, msg_dim)
        """
        x = torch.cat([op_prob_delta, msg_from_4], dim=-1)  # (n, n_ops + msg_dim)
        return self.m_2to3(x)

    def forward(
        self,
        op_probs: torch.Tensor,
        op_probs_prev: Optional[torch.Tensor],
        arg_values: torch.Tensor,
        adjacency: torch.Tensor,
        exec_success: torch.Tensor,
        exec_results: torch.Tensor,
        validity_score: torch.Tensor,
    ) -> ConstraintMessages:
        """
        Compute all backward messages in one pass.

        This is the full message computation for one belief prop round.

        Args:
            op_probs: (n, n_ops) current C2 output
            op_probs_prev: (n, n_ops) C2 output from previous round (or None)
            arg_values: (n, max_args) extracted argument values
            adjacency: (n, n) current graph structure
            exec_success: (n,) per-node execution success
            exec_results: (n,) per-node execution results
            validity_score: scalar or (n,) graph validity

        Returns:
            ConstraintMessages with msg_5to4, msg_4to2, msg_2to3
        """
        # M_5→4: execution → graph
        msg_5to4 = self.compute_5to4(op_probs, arg_values, exec_success, exec_results)

        # M_4→2: graph → ops
        msg_4to2 = self.compute_4to2(adjacency, validity_score, msg_5to4)

        # M_2→3: ops → extraction
        if op_probs_prev is not None:
            op_delta = op_probs - op_probs_prev
        else:
            op_delta = torch.zeros_like(op_probs)

        msg_2to3 = self.compute_2to3(op_delta, msg_4to2)

        return ConstraintMessages(
            msg_5to4=msg_5to4,
            msg_4to2=msg_4to2,
            msg_2to3=msg_2to3,
        )

    def param_count(self) -> dict:
        """Count parameters per message network."""
        return {
            "m_5to4": sum(p.numel() for p in self.m_5to4.parameters()),
            "m_4to2": sum(p.numel() for p in self.m_4to2.parameters()),
            "m_2to3": sum(p.numel() for p in self.m_2to3.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }


# ---------------------------------------------------------------------------
# Message gates for C2 and C3
# ---------------------------------------------------------------------------

class MessageGate(nn.Module):
    """
    Gate layer that integrates backward messages into component forward pass.

    Produces an additive bias on the component's output, allowing constraint
    signals to shift beliefs without overwriting pretrained representations.

    Architecture: msg_dim → hidden → output_dim (additive bias)
    """

    def __init__(
        self,
        msg_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(msg_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        base_output: torch.Tensor,
        message: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply message as additive bias.

        Args:
            base_output: (..., output_dim) from component forward
            message: (..., msg_dim) backward message, or None

        Returns:
            base_output + gate(message), or base_output if message is None
        """
        if message is None:
            return base_output

        bias = self.gate(message)
        return base_output + bias


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    # Test parameters
    n_ops = 19
    max_args = 3
    msg_dim = 64
    n_nodes = 4

    # Initialize
    messages = MessageNetworkEnsemble(
        n_ops=n_ops,
        max_args=max_args,
        msg_dim=msg_dim,
    )

    # Print param counts
    counts = messages.param_count()
    print("Parameter counts:")
    for name, count in counts.items():
        print(f"  {name}: {count:,}")

    # Test 1: Individual message computations
    op_probs = F.softmax(torch.randn(n_nodes, n_ops), dim=-1)
    arg_values = torch.randn(n_nodes, max_args)
    exec_success = torch.ones(n_nodes)
    exec_results = torch.randn(n_nodes)
    adjacency = torch.triu(torch.randint(0, 2, (n_nodes, n_nodes)).float(), diagonal=1)
    validity = torch.tensor(0.8)

    msg_5to4 = messages.compute_5to4(op_probs, arg_values, exec_success, exec_results)
    print(f"\nTest 1: M_5→4 shape: {msg_5to4.shape}")
    assert msg_5to4.shape == (n_nodes, msg_dim)

    msg_4to2 = messages.compute_4to2(adjacency, validity, msg_5to4)
    print(f"Test 1: M_4→2 shape: {msg_4to2.shape}")
    assert msg_4to2.shape == (n_nodes, msg_dim)

    op_delta = torch.randn(n_nodes, n_ops)
    msg_2to3 = messages.compute_2to3(op_delta, msg_4to2)
    print(f"Test 1: M_2→3 shape: {msg_2to3.shape}")
    assert msg_2to3.shape == (n_nodes, msg_dim)

    # Test 2: Full forward
    result = messages.forward(
        op_probs=op_probs,
        op_probs_prev=None,
        arg_values=arg_values,
        adjacency=adjacency,
        exec_success=exec_success,
        exec_results=exec_results,
        validity_score=validity,
    )

    print(f"\nTest 2: Full forward")
    print(f"  msg_5to4: {result.msg_5to4.shape}")
    print(f"  msg_4to2: {result.msg_4to2.shape}")
    print(f"  msg_2to3: {result.msg_2to3.shape}")

    # Test 3: With previous op_probs
    op_probs_prev = F.softmax(torch.randn(n_nodes, n_ops), dim=-1)
    result2 = messages.forward(
        op_probs=op_probs,
        op_probs_prev=op_probs_prev,
        arg_values=arg_values,
        adjacency=adjacency,
        exec_success=exec_success,
        exec_results=exec_results,
        validity_score=validity,
    )

    print(f"\nTest 3: With op_probs_prev")
    print(f"  msg_2to3 different: {not torch.equal(result.msg_2to3, result2.msg_2to3)}")

    # Test 4: MessageGate
    gate = MessageGate(msg_dim=msg_dim, hidden_dim=128, output_dim=n_ops)

    base = torch.randn(n_nodes, n_ops)
    gated = gate(base, result.msg_4to2)

    print(f"\nTest 4: MessageGate")
    print(f"  Base output: {base.shape}")
    print(f"  Gated output: {gated.shape}")
    print(f"  Different: {not torch.equal(base, gated)}")

    # Test 5: None message
    gated_none = gate(base, None)
    print(f"\nTest 5: None message")
    print(f"  Same as base: {torch.equal(base, gated_none)}")

    # Test 6: Gradient flow
    op_probs = F.softmax(torch.randn(n_nodes, n_ops, requires_grad=True), dim=-1)
    result = messages.forward(
        op_probs=op_probs,
        op_probs_prev=None,
        arg_values=arg_values,
        adjacency=adjacency,
        exec_success=exec_success,
        exec_results=exec_results,
        validity_score=validity,
    )

    loss = result.msg_2to3.sum()
    loss.backward()

    print(f"\nTest 6: Gradient flow")
    print(f"  op_probs.grad exists: {op_probs.grad is not None}")

    print("\nAll tests passed!")
