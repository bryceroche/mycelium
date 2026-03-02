"""
C4: Graph Assembler — Learns DAG structure with straight-through gradients

Takes: C2 op labels (soft probs) + C3 extracted arg summaries
Returns: (n_nodes, n_nodes) soft adjacency → discretized via straight-through

Key insight: The adjacency structure is LEARNED from teacher CoT step ordering,
not hand-coded. Step 2 using the result of Step 1 means edge 1→2. This is
implicit in teacher CoT — we just extract and supervise it.

Gradient flow:
  - Forward: hard binary edges (argmax)
  - Backward: soft gradients (softmax)
  - Temperature anneals from 1.0 → 0.1 for exploration → exploitation

~100K parameters total.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Straight-through estimator (base version)
# ---------------------------------------------------------------------------

class StraightThrough(torch.autograd.Function):
    """
    Forward: argmax (hard discrete decision)
    Backward: pass gradient through soft distribution

    This is the key trick for making graph assembly differentiable.
    The assembler makes hard wiring decisions, but gradients flow through
    the soft logits that produced those decisions.
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor) -> torch.Tensor:
        # Hard one-hot in forward pass
        indices = logits.argmax(dim=-1)
        hard = F.one_hot(indices, logits.size(-1)).float()
        # Save soft probs for backward
        ctx.save_for_backward(logits)
        return hard

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        logits, = ctx.saved_tensors
        # Backward through softmax, ignoring the argmax
        soft = F.softmax(logits, dim=-1)
        return grad_output * soft


def straight_through(logits: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper. Returns hard decisions with soft gradients."""
    return StraightThrough.apply(logits)


# ---------------------------------------------------------------------------
# Gumbel-Softmax with straight-through
# ---------------------------------------------------------------------------

class GumbelStraightThrough(nn.Module):
    """
    Gumbel-Softmax with straight-through estimator.

    Adds stochastic exploration during training via Gumbel noise.
    Temperature anneals from high (exploration) to low (exploitation).

    At inference (eval mode): pure argmax, no noise.
    """

    def __init__(
        self,
        tau_start: float = 1.0,
        tau_min: float = 0.1,
        anneal_steps: int = 10000,
    ):
        super().__init__()
        self.register_buffer('tau', torch.tensor(tau_start))
        self.tau_min = tau_min
        self.anneal_rate = (tau_start - tau_min) / anneal_steps
        self.step_count = 0

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (..., n_classes) raw scores

        Returns:
            (..., n_classes) one-hot with soft gradients
        """
        if self.training:
            # Sample Gumbel noise
            gumbels = -torch.log(
                -torch.log(torch.rand_like(logits).clamp(min=1e-10)) + 1e-10
            )

            # Soft samples
            soft = F.softmax((logits + gumbels) / self.tau, dim=-1)

            # Straight-through: hard forward, soft backward
            hard = F.one_hot(soft.argmax(dim=-1), logits.size(-1)).float()

            # Trick: hard - soft.detach() + soft
            # Forward sees 'hard', backward sees 'soft'
            return hard - soft.detach() + soft
        else:
            # Inference: pure argmax
            return F.one_hot(logits.argmax(dim=-1), logits.size(-1)).float()

    def anneal(self):
        """Call once per training step to reduce temperature."""
        self.step_count += 1
        new_tau = max(self.tau_min, self.tau.item() - self.anneal_rate)
        self.tau.fill_(new_tau)

    def get_tau(self) -> float:
        return self.tau.item()


# ---------------------------------------------------------------------------
# C4 Graph Assembler
# ---------------------------------------------------------------------------

class C4_GraphAssembler(nn.Module):
    """
    Assembles execution DAG from C2 op labels + C3 extracted args.

    Architecture:
      1. Node encoder: embeds (op_label, arg_summary) → hidden representation
      2. Edge predictor: for each pair (i,j), predicts P(edge | node_i, node_j)
      3. Discretizer: Gumbel-ST converts soft probs to hard edges

    DAG constraint: Only predict edges i→j where i < j (topological order).
    This is structural, not heuristic — teacher CoT is always ordered.

    Input shapes:
      op_labels: (n_nodes, n_ops) soft probabilities or (n_nodes,) hard indices
      arg_summaries: (n_nodes, n_features) extracted arg features

    Output shapes:
      edge_logits: (n_nodes, n_nodes, 2) raw scores for [no_edge, edge]
      adjacency_hard: (n_nodes, n_nodes) binary adjacency matrix
    """

    def __init__(
        self,
        n_ops: int,
        n_arg_features: int = 6,  # e.g., max_args * 2 (position + confidence)
        hidden_dim: int = 128,
        msg_dim: int = 64,
        tau_start: float = 1.0,
        tau_min: float = 0.1,
        anneal_steps: int = 10000,
    ):
        super().__init__()
        self.n_ops = n_ops
        self.hidden_dim = hidden_dim

        # Operation embedding (for both soft and hard labels)
        self.op_embed = nn.Embedding(n_ops, hidden_dim)

        # Argument feature encoder
        self.arg_encoder = nn.Linear(n_arg_features, hidden_dim)

        # Node encoder: combines op embedding + arg features
        self.node_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Pairwise edge predictor
        # Input: [node_i_repr, node_j_repr] → P(edge)
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2),  # [no_edge, edge]
        )

        # Message integration (for belief propagation)
        # Takes backward message from C5 and modulates edge prediction
        self.msg_gate = nn.Sequential(
            nn.Linear(msg_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Straight-through discretizer with Gumbel noise
        self.discretizer = GumbelStraightThrough(
            tau_start=tau_start,
            tau_min=tau_min,
            anneal_steps=anneal_steps,
        )

    def forward(
        self,
        op_labels: torch.Tensor,
        arg_summaries: torch.Tensor,
        backward_msg: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assemble computation graph.

        Args:
            op_labels: (n_nodes, n_ops) soft probs OR (n_nodes,) hard indices
            arg_summaries: (n_nodes, n_arg_features) extracted arg features
            backward_msg: (n_nodes, msg_dim) optional message from C5

        Returns:
            edge_logits: (n_nodes, n_nodes, 2) raw edge scores
            adjacency_hard: (n_nodes, n_nodes) binary edges via straight-through
        """
        n = op_labels.size(0)
        device = op_labels.device

        # Encode operations
        if op_labels.dim() == 1:
            # Hard labels: use embedding lookup
            op_emb = self.op_embed(op_labels)  # (n, hidden)
        else:
            # Soft labels: weighted sum of embeddings
            # This preserves gradient flow through C2's output
            op_emb = op_labels @ self.op_embed.weight  # (n, n_ops) @ (n_ops, hidden)

        # Encode arguments
        arg_emb = self.arg_encoder(arg_summaries)  # (n, hidden)

        # Combine into node representations
        node_repr = self.node_encoder(
            torch.cat([op_emb, arg_emb], dim=-1)
        )  # (n, hidden)

        # Integrate backward message if present (from C5 execution feedback)
        if backward_msg is not None:
            msg_signal = self.msg_gate(backward_msg)  # (n, hidden)
            node_repr = node_repr + msg_signal

        # Pairwise edge prediction
        # Create all pairs: (node_i, node_j) for all i, j
        node_i = node_repr.unsqueeze(1).expand(-1, n, -1)  # (n, n, hidden)
        node_j = node_repr.unsqueeze(0).expand(n, -1, -1)  # (n, n, hidden)
        pair_repr = torch.cat([node_i, node_j], dim=-1)    # (n, n, 2*hidden)

        edge_logits = self.edge_predictor(pair_repr)  # (n, n, 2)

        # DAG constraint: only allow edges i → j where i < j
        # This enforces topological order without being a "heuristic" —
        # it's the mathematical definition of a DAG matching CoT structure
        causal_mask = torch.triu(torch.ones(n, n, device=device), diagonal=1)

        # Apply mask: set backward edge logits to -inf so they're never selected
        edge_logits = edge_logits.clone()
        edge_logits[:, :, 1] = edge_logits[:, :, 1].masked_fill(
            causal_mask == 0, float('-inf')
        )

        # Straight-through discretization: hard forward, soft backward
        # Shape: (n, n, 2) → take the "edge" class → (n, n)
        edge_probs = self.discretizer(edge_logits)  # (n, n, 2) one-hot
        adjacency_hard = edge_probs[:, :, 1]  # (n, n) binary

        return edge_logits, adjacency_hard

    def compute_edge_loss(
        self,
        edge_logits: torch.Tensor,
        gold_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute binary cross-entropy loss against gold adjacency.

        Args:
            edge_logits: (n, n, 2) raw logits from forward()
            gold_adjacency: (n, n) binary ground truth

        Returns:
            Scalar loss
        """
        # Only compute loss on upper triangle (valid edge positions)
        n = edge_logits.size(0)
        mask = torch.triu(torch.ones(n, n, device=edge_logits.device), diagonal=1)

        # Get edge probabilities (softmax over the 2-class dimension)
        edge_probs = F.softmax(edge_logits, dim=-1)[:, :, 1]  # (n, n)

        # Masked BCE
        loss = F.binary_cross_entropy(
            edge_probs * mask,
            gold_adjacency * mask,
            reduction='sum'
        )

        # Normalize by number of valid positions
        n_valid = mask.sum()
        return loss / n_valid if n_valid > 0 else loss


# ---------------------------------------------------------------------------
# Utility: Extract arg summaries from C3 output
# ---------------------------------------------------------------------------

def summarize_extractions(
    extraction_logits: torch.Tensor,
    max_args: int = 3,
) -> torch.Tensor:
    """
    Compress C3 extraction logits into fixed-size features for C4.

    Uses soft argmax for differentiable position extraction.

    Args:
        extraction_logits: (n_nodes, max_args, seq_len, 2) start/end logits

    Returns:
        (n_nodes, max_args * 2) features: [pos_0, conf_0, pos_1, conf_1, ...]
    """
    n_nodes = extraction_logits.size(0)
    seq_len = extraction_logits.size(2)
    device = extraction_logits.device

    # Start logits for each arg slot
    start_logits = extraction_logits[:, :, :, 0]  # (n, args, seq)

    # Soft argmax: expected position (differentiable)
    start_probs = F.softmax(start_logits, dim=-1)  # (n, args, seq)
    positions = torch.arange(seq_len, device=device, dtype=torch.float)
    expected_pos = (start_probs * positions).sum(dim=-1)  # (n, args)

    # Confidence: max probability
    confidence = start_probs.max(dim=-1).values  # (n, args)

    # Interleave position and confidence
    # Result: [pos_0, conf_0, pos_1, conf_1, ...]
    features = torch.stack([expected_pos, confidence], dim=-1)  # (n, args, 2)
    features = features.view(n_nodes, -1)  # (n, args * 2)

    return features


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    torch.manual_seed(42)

    # Test parameters
    n_ops = 19
    n_nodes = 4
    n_arg_features = 6
    hidden_dim = 128

    # Initialize assembler
    assembler = C4_GraphAssembler(
        n_ops=n_ops,
        n_arg_features=n_arg_features,
        hidden_dim=hidden_dim,
    )

    # Test 1: Soft op labels (from C2)
    op_probs = F.softmax(torch.randn(n_nodes, n_ops), dim=-1)
    arg_summaries = torch.randn(n_nodes, n_arg_features)

    edge_logits, adjacency = assembler(op_probs, arg_summaries)

    print(f"Test 1: Soft labels")
    print(f"  Edge logits shape: {edge_logits.shape}")  # (4, 4, 2)
    print(f"  Adjacency shape: {adjacency.shape}")      # (4, 4)
    print(f"  Adjacency (DAG):\n{adjacency}")
    assert edge_logits.shape == (n_nodes, n_nodes, 2)
    assert adjacency.shape == (n_nodes, n_nodes)
    # Check DAG constraint: no edges where i >= j
    assert (adjacency * torch.tril(torch.ones_like(adjacency))).sum() == 0

    # Test 2: Hard op labels
    op_hard = torch.randint(0, n_ops, (n_nodes,))
    edge_logits2, adjacency2 = assembler(op_hard, arg_summaries)

    print(f"\nTest 2: Hard labels")
    print(f"  Adjacency:\n{adjacency2}")
    assert adjacency2.shape == (n_nodes, n_nodes)

    # Test 3: Backward message integration
    msg = torch.randn(n_nodes, 64)
    edge_logits3, adjacency3 = assembler(op_probs, arg_summaries, backward_msg=msg)

    print(f"\nTest 3: With backward message")
    print(f"  Adjacency:\n{adjacency3}")

    # Test 4: Gradient flow
    assembler.train()
    op_probs = F.softmax(torch.randn(n_nodes, n_ops, requires_grad=True), dim=-1)
    arg_summaries = torch.randn(n_nodes, n_arg_features, requires_grad=True)

    edge_logits, adjacency = assembler(op_probs, arg_summaries)

    # Create dummy loss and backprop
    loss = adjacency.sum()
    loss.backward()

    print(f"\nTest 4: Gradient flow")
    print(f"  op_probs.grad exists: {op_probs.grad is not None}")
    print(f"  arg_summaries.grad exists: {arg_summaries.grad is not None}")
    assert op_probs.grad is not None
    assert arg_summaries.grad is not None

    # Test 5: Edge loss computation
    gold_adj = torch.triu(torch.randint(0, 2, (n_nodes, n_nodes)).float(), diagonal=1)
    loss = assembler.compute_edge_loss(edge_logits, gold_adj)
    print(f"\nTest 5: Edge loss")
    print(f"  Loss: {loss.item():.4f}")

    # Test 6: Temperature annealing
    print(f"\nTest 6: Temperature annealing")
    print(f"  Initial tau: {assembler.discretizer.get_tau():.4f}")
    for _ in range(1000):
        assembler.discretizer.anneal()
    print(f"  After 1000 steps: {assembler.discretizer.get_tau():.4f}")

    # Test 7: Eval mode (no Gumbel noise)
    assembler.eval()
    edge_logits_eval1, adj_eval1 = assembler(op_probs.detach(), arg_summaries.detach())
    edge_logits_eval2, adj_eval2 = assembler(op_probs.detach(), arg_summaries.detach())
    print(f"\nTest 7: Eval mode determinism")
    print(f"  Same output: {torch.equal(adj_eval1, adj_eval2)}")
    assert torch.equal(adj_eval1, adj_eval2)

    print("\nAll tests passed!")
