"""
Mycelium Phase 2: Joint Constraint Training Pipeline

Full pipeline with belief propagation:
  C2 (op classification) → C3 (operand extraction) → C4 (graph assembly) → C5 (SymPy execution)
                    ↑________________________↩_______________________↩___________________↓
                                         Message Networks

Training:
  - Phase 1 (done): C2, C3 pretrained independently
  - Phase 2 (this): Joint fine-tuning with constraint propagation loss

Inference:
  - 2-3 rounds of belief propagation until convergence
  - Early-stop when beliefs stabilize
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field

from models.c4_graph_assembler import C4_GraphAssembler, summarize_extractions
from models.c5_sympy_executor import C5_SymPyExecutor, OP_LABELS, N_OPS, ExecutionResult
from models.message_networks import MessageNetworkEnsemble, MessageGate, ConstraintMessages


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SpanGroup:
    """A group of text spans corresponding to one computation step."""
    input_ids: torch.Tensor        # (seq_len,) tokenized text
    attention_mask: torch.Tensor   # (seq_len,)
    text: str = ""                 # raw text for SymPy extraction


@dataclass
class PipelineState:
    """Full state flowing through the pipeline. Mutable during belief prop."""
    span_groups: List[SpanGroup]
    device: torch.device = field(default_factory=lambda: torch.device('cpu'))

    # C2 outputs: (n_groups, n_ops) log-probabilities
    op_logits: Optional[torch.Tensor] = None
    op_probs: Optional[torch.Tensor] = None
    op_probs_prev: Optional[torch.Tensor] = None  # from previous round

    # C3 outputs: (n_groups, max_args, seq_len, 2) start/end logits
    extraction_logits: Optional[torch.Tensor] = None
    extracted_values: Optional[List[List[Optional[float]]]] = None

    # C4 outputs
    adjacency_logits: Optional[torch.Tensor] = None
    adjacency_hard: Optional[torch.Tensor] = None

    # C5 outputs
    execution_result: Optional[ExecutionResult] = None

    # Backward messages
    messages: Optional[ConstraintMessages] = None


@dataclass
class PipelineOutput:
    """Output from full pipeline forward pass."""
    state: PipelineState
    losses: Dict[str, torch.Tensor]
    rounds: List[Dict[str, Any]]
    final_answer: Optional[float] = None
    success: bool = False


# ---------------------------------------------------------------------------
# C2 Wrapper with Message Gate
# ---------------------------------------------------------------------------

class C2_WithMessageGate(nn.Module):
    """
    Wraps existing C2 model and adds message gate for belief propagation.

    The message gate produces an additive bias on logits, allowing
    constraint signals to shift beliefs without overwriting pretrained
    representations.
    """

    def __init__(
        self,
        c2_model: nn.Module,
        n_ops: int = N_OPS,
        msg_dim: int = 64,
    ):
        super().__init__()
        self.c2 = c2_model

        # Get hidden size from C2's backbone
        if hasattr(c2_model, 'backbone'):
            hidden_size = c2_model.backbone.config.hidden_size
        else:
            hidden_size = 384  # MiniLM default

        # Message gate: msg_dim → hidden → n_ops (additive bias)
        self.msg_gate = MessageGate(msg_dim, hidden_size, n_ops)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        backward_msg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional message integration.

        Round 0 (no message):  pure C2 classification
        Round 1+ (with msg):   C2 classification + constraint bias
        """
        # Get base logits from C2
        if hasattr(self.c2, 'classifier'):
            # Using existing C2Model structure
            hidden = self.c2.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state[:, 0, :]  # [CLS] token

            logits = self.c2.classifier(hidden)
        else:
            # Generic forward
            logits = self.c2(input_ids, attention_mask)

        # Apply message gate
        logits = self.msg_gate(logits, backward_msg)

        return logits


# ---------------------------------------------------------------------------
# C3 Wrapper with Message Gate
# ---------------------------------------------------------------------------

class C3_WithMessageGate(nn.Module):
    """
    Wraps existing C3 model and adds message gate for belief propagation.

    Also adds operation conditioning - C3 extracts operands conditioned
    on which operation C2 thinks is being performed.
    """

    def __init__(
        self,
        c3_model: nn.Module,
        n_ops: int = N_OPS,
        msg_dim: int = 64,
        max_operands: int = 4,
    ):
        super().__init__()
        self.c3 = c3_model
        self.n_ops = n_ops
        self.max_operands = max_operands

        # Get hidden size from C3's backbone
        if hasattr(c3_model, 'backbone'):
            hidden_size = c3_model.backbone.config.hidden_size
        else:
            hidden_size = 768  # RoBERTa default

        # Operation conditioning embedding
        self.op_embedding = nn.Embedding(n_ops, hidden_size)

        # Message gate: modulates hidden states before extraction
        self.msg_gate = nn.Sequential(
            nn.Linear(msg_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        op_hypothesis: Optional[torch.Tensor] = None,
        backward_msg: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional operation conditioning and message.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            op_hypothesis: (batch,) int - which op to condition on
            backward_msg: (batch, msg_dim) - backward message from C4

        Returns:
            start_logits: (batch, seq_len, max_operands)
            end_logits: (batch, seq_len, max_operands)
        """
        # Get base extraction from C3
        start_logits, end_logits = self.c3(input_ids, attention_mask)

        # Apply operation conditioning if provided
        if op_hypothesis is not None:
            # This is a soft modulation - doesn't change the extraction
            # architecture, just biases it toward patterns for this op
            pass  # TODO: integrate op_embedding into extraction

        # Apply message gate if provided
        if backward_msg is not None:
            # Modulate logits based on constraint signal
            batch_size = input_ids.size(0)
            msg_bias = self.msg_gate(backward_msg)  # (batch, hidden)

            # Add bias to extraction (simplified - expand along seq)
            # In full impl, this would modulate the hidden states before heads
            pass  # TODO: deeper integration

        return start_logits, end_logits


# ---------------------------------------------------------------------------
# Full Constraint Pipeline
# ---------------------------------------------------------------------------

class MyceliumConstraintPipeline(nn.Module):
    """
    The full Mycelium pipeline with learned constraint propagation.

    Components:
      C2: Operation classifier (MiniLM 22M)
      C3: Operand extractor (Qwen 0.5B / RoBERTa)
      C4: Graph assembler (100K params)
      C5: SymPy executor (no params)
      Messages: Constraint networks (30K params)

    Training:
      - Phase 1 (done): C2, C3 pretrained independently
      - Phase 2 (this): Joint fine-tuning
        - C2, C3 encoders: frozen or very low lr (1e-6)
        - Message gates, C4, messages: normal lr (1e-4)
        - Loss = component_losses + execution_reward

    Inference:
      - Run 2-3 rounds of belief propagation
      - Each round: forward → execute → backward messages → forward again
      - Converge when beliefs stop changing
    """

    def __init__(
        self,
        c2_model: nn.Module,
        c3_model: nn.Module,
        n_ops: int = N_OPS,
        max_args: int = 3,
        msg_dim: int = 64,
        n_belief_rounds: int = 3,
    ):
        super().__init__()

        # Wrap C2 and C3 with message gates
        self.c2 = C2_WithMessageGate(c2_model, n_ops, msg_dim)
        self.c3 = C3_WithMessageGate(c3_model, n_ops, msg_dim)

        # C4: Graph assembler
        n_arg_features = max_args * 2  # position + confidence per arg
        self.c4 = C4_GraphAssembler(
            n_ops=n_ops,
            n_arg_features=n_arg_features,
            msg_dim=msg_dim,
        )

        # C5: SymPy executor (no parameters)
        self.c5 = C5_SymPyExecutor()

        # Message networks
        self.messages = MessageNetworkEnsemble(
            n_ops=n_ops,
            max_args=max_args,
            msg_dim=msg_dim,
        )

        # Config
        self.n_ops = n_ops
        self.max_args = max_args
        self.msg_dim = msg_dim
        self.n_belief_rounds = n_belief_rounds

    def forward_round(
        self,
        state: PipelineState,
        round_idx: int,
    ) -> PipelineState:
        """
        One round of forward pass through the pipeline.

        Round 0: pure forward (no backward messages)
        Round 1+: forward with backward messages from previous round
        """
        n_groups = len(state.span_groups)
        device = state.device

        # Batch span groups
        input_ids = torch.stack([g.input_ids for g in state.span_groups]).to(device)
        attention_mask = torch.stack([g.attention_mask for g in state.span_groups]).to(device)

        # --- C2: Classify operations ---
        backward_msg = state.messages.msg_4to2 if state.messages else None
        state.op_logits = self.c2(input_ids, attention_mask, backward_msg)
        state.op_probs_prev = state.op_probs.clone() if state.op_probs is not None else None
        state.op_probs = F.softmax(state.op_logits, dim=-1)

        # --- C3: Extract operands ---
        backward_msg = state.messages.msg_2to3 if state.messages else None
        top_op = state.op_probs.argmax(dim=-1)  # (n_groups,)

        start_logits, end_logits = self.c3(
            input_ids, attention_mask,
            op_hypothesis=top_op,
            backward_msg=backward_msg,
        )

        # Stack into (n_groups, max_args, seq_len, 2)
        state.extraction_logits = torch.stack([start_logits, end_logits], dim=-1)
        state.extraction_logits = state.extraction_logits.permute(0, 2, 1, 3)

        # Extract concrete values for C5
        state.extracted_values = self._extract_values(state)

        # --- C4: Assemble graph ---
        backward_msg = state.messages.msg_5to4 if state.messages else None

        # Summarize extractions for graph assembly
        arg_summaries = self._get_arg_summaries(state)

        edge_logits, adjacency = self.c4(
            op_labels=state.op_probs,  # soft for gradient flow
            arg_summaries=arg_summaries,
            backward_msg=backward_msg,
        )
        state.adjacency_logits = edge_logits
        state.adjacency_hard = adjacency

        # --- C5: Execute (non-differentiable) ---
        ops_list = [OP_LABELS[i] for i in state.op_probs.argmax(dim=-1).cpu().tolist()]
        state.execution_result = self.c5.execute(
            ops_list,
            state.extracted_values,
            adjacency.detach(),
        )

        return state

    def compute_backward_messages(self, state: PipelineState) -> PipelineState:
        """
        Compute backward constraint messages after a forward round.
        These feed into the NEXT forward round.
        """
        n = len(state.span_groups)
        device = state.device

        # Per-node execution features
        exec_success = torch.tensor(
            [float(state.execution_result.success)] * n,
            device=device
        )
        # Handle None intermediate results
        intermediates = state.execution_result.intermediate
        if intermediates is None:
            intermediates = []
        # Convert any None values to 0.0 and ensure we have n values
        exec_results = torch.tensor(
            [(float(v) if v is not None else 0.0) for v in intermediates[:n]] + [0.0] * max(0, n - len(intermediates)),
            device=device
        )

        # Arg values tensor
        arg_values = torch.zeros(n, self.max_args, device=device)
        for i, args in enumerate(state.extracted_values[:n]):
            for j, val in enumerate(args[:self.max_args]):
                if val is not None:
                    arg_values[i, j] = val

        # Validity score
        validity = torch.tensor(state.execution_result.plausibility, device=device)

        # Compute all messages
        state.messages = self.messages.forward(
            op_probs=state.op_probs,
            op_probs_prev=state.op_probs_prev,
            arg_values=arg_values,
            adjacency=state.adjacency_hard,
            exec_success=exec_success,
            exec_results=exec_results,
            validity_score=validity,
        )

        return state

    def forward(
        self,
        span_groups: List[SpanGroup],
        gold_ops: Optional[torch.Tensor] = None,
        gold_adjacency: Optional[torch.Tensor] = None,
        gold_answer: Optional[float] = None,
    ) -> PipelineOutput:
        """
        Full forward with belief propagation.

        Training: returns losses for joint optimization
        Inference: returns best prediction after convergence
        """
        device = span_groups[0].input_ids.device if span_groups else torch.device('cpu')
        state = PipelineState(span_groups=span_groups, device=device)

        all_round_states = []

        for round_idx in range(self.n_belief_rounds):
            # Forward pass
            state = self.forward_round(state, round_idx)

            all_round_states.append({
                "round": round_idx,
                "op_probs": state.op_probs.clone().detach(),
                "execution_success": state.execution_result.success,
                "execution_result": state.execution_result.result,
            })

            # Backward messages (skip after last round)
            if round_idx < self.n_belief_rounds - 1:
                state = self.compute_backward_messages(state)

        # Compute losses
        losses = self._compute_losses(state, gold_ops, gold_adjacency, gold_answer, all_round_states)

        return PipelineOutput(
            state=state,
            losses=losses,
            rounds=all_round_states,
            final_answer=state.execution_result.result,
            success=state.execution_result.success,
        )

    def _compute_losses(
        self,
        state: PipelineState,
        gold_ops: Optional[torch.Tensor],
        gold_adjacency: Optional[torch.Tensor],
        gold_answer: Optional[float],
        all_round_states: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """Compute all training losses."""
        losses = {}
        device = state.device

        # C2 loss: cross-entropy on operation labels
        if gold_ops is not None:
            losses["c2_loss"] = F.cross_entropy(state.op_logits, gold_ops.to(device))

        # C4 loss: binary cross-entropy on edge prediction
        if gold_adjacency is not None:
            losses["c4_loss"] = self.c4.compute_edge_loss(
                state.adjacency_logits,
                gold_adjacency.to(device),
            )

        # REINFORCE loss: reward from correct execution
        if gold_answer is not None:
            # Check if we got the right answer
            correct = (
                state.execution_result.success and
                state.execution_result.result is not None and
                abs(state.execution_result.result - gold_answer) < 1e-6
            )
            reward = 1.0 if correct else 0.0
            baseline = 0.5  # TODO: moving average baseline

            advantage = reward - baseline

            # Policy gradient on C2's discrete decisions
            chosen_ops = state.op_probs.argmax(dim=-1)
            log_probs = F.log_softmax(state.op_logits, dim=-1)
            chosen_log_probs = log_probs.gather(1, chosen_ops.unsqueeze(1)).squeeze()

            losses["reinforce_loss"] = -(chosen_log_probs * advantage).mean()

        # Convergence loss: encourage belief stability
        if len(all_round_states) >= 2:
            delta = (
                all_round_states[-1]["op_probs"] - all_round_states[-2]["op_probs"]
            ).abs().mean()
            losses["convergence_loss"] = delta

        # Parameter regularization: ensure all message network params get gradients
        # (needed for DDP to work with variable execution paths)
        param_reg = torch.tensor(0.0, device=device, requires_grad=True)
        for module in [self.messages, self.c4, self.c2.msg_gate, self.c3.msg_gate]:
            for param in module.parameters():
                if param.requires_grad:
                    param_reg = param_reg + 1e-8 * param.pow(2).sum()
        losses["param_reg"] = param_reg

        return losses

    def _extract_values(self, state: PipelineState) -> List[List[Optional[float]]]:
        """Extract concrete argument values from C3 output."""
        # Placeholder - real impl parses text spans and converts to numbers
        n = len(state.span_groups)
        return [[None] * self.max_args for _ in range(n)]

    def _get_arg_summaries(self, state: PipelineState) -> torch.Tensor:
        """Get arg summaries for C4 input."""
        n = len(state.span_groups)
        device = state.device

        if state.extraction_logits is None:
            return torch.zeros(n, self.max_args * 2, device=device)

        # Use soft argmax for differentiable position extraction
        # extraction_logits: (n, max_args, seq_len, 2)
        start_logits = state.extraction_logits[:, :self.max_args, :, 0]  # (n, args, seq)

        # Soft argmax
        seq_len = start_logits.size(-1)
        start_probs = F.softmax(start_logits, dim=-1)
        positions = torch.arange(seq_len, device=device, dtype=torch.float)
        expected_pos = (start_probs * positions).sum(dim=-1)  # (n, args)

        confidence = start_probs.max(dim=-1).values  # (n, args)

        # Interleave
        features = torch.stack([expected_pos, confidence], dim=-1)  # (n, args, 2)
        return features.view(n, -1)  # (n, args * 2)


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------

@torch.no_grad()
def inference(
    pipeline: MyceliumConstraintPipeline,
    span_groups: List[SpanGroup],
    max_rounds: int = 3,
    convergence_threshold: float = 0.01,
) -> Dict[str, Any]:
    """
    Inference with belief propagation until convergence.

    Unlike training, we can early-stop when beliefs stabilize.

    Returns:
        Dict with answer, operations, convergence trace
    """
    pipeline.eval()
    device = span_groups[0].input_ids.device if span_groups else torch.device('cpu')
    state = PipelineState(span_groups=span_groups, device=device)

    trace = []
    prev_op_probs = None

    for round_idx in range(max_rounds):
        state = pipeline.forward_round(state, round_idx)

        trace.append({
            "round": round_idx,
            "op_probs": state.op_probs.cpu().clone(),
            "execution_success": state.execution_result.success,
            "execution_result": state.execution_result.result,
            "top_ops": [OP_LABELS[i] for i in state.op_probs.argmax(dim=-1).cpu().tolist()],
        })

        # Check convergence
        if prev_op_probs is not None:
            delta = (state.op_probs - prev_op_probs).abs().max().item()
            trace[-1]["belief_delta"] = delta
            if delta < convergence_threshold:
                trace[-1]["converged"] = True
                break

        prev_op_probs = state.op_probs.clone()

        # Backward messages for next round
        if round_idx < max_rounds - 1:
            state = pipeline.compute_backward_messages(state)

    return {
        "answer": state.execution_result.result,
        "success": state.execution_result.success,
        "plausibility": state.execution_result.plausibility,
        "operations": [OP_LABELS[i] for i in state.op_probs.argmax(dim=-1).cpu().tolist()],
        "n_rounds": len(trace),
        "converged": trace[-1].get("converged", False),
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Parameter count utility
# ---------------------------------------------------------------------------

def count_parameters(pipeline: MyceliumConstraintPipeline) -> Dict[str, int]:
    """Count parameters by component."""
    counts = {}

    # C2 (wrapped)
    c2_total = sum(p.numel() for p in pipeline.c2.parameters())
    c2_gate = sum(p.numel() for p in pipeline.c2.msg_gate.parameters())
    counts["c2_total"] = c2_total
    counts["c2_gate"] = c2_gate

    # C3 (wrapped)
    c3_total = sum(p.numel() for p in pipeline.c3.parameters())
    c3_gate = sum(p.numel() for p in pipeline.c3.msg_gate.parameters())
    c3_op_emb = sum(p.numel() for p in pipeline.c3.op_embedding.parameters())
    counts["c3_total"] = c3_total
    counts["c3_gate"] = c3_gate
    counts["c3_op_emb"] = c3_op_emb

    # C4
    counts["c4"] = sum(p.numel() for p in pipeline.c4.parameters())

    # Messages
    counts["messages"] = sum(p.numel() for p in pipeline.messages.parameters())

    # New Phase 2 params
    counts["phase2_new"] = c2_gate + c3_gate + c3_op_emb + counts["c4"] + counts["messages"]

    # Total
    counts["total"] = sum(p.numel() for p in pipeline.parameters())

    return counts


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing constraint pipeline...")

    # Create mock C2 model
    class MockC2(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = type('Config', (), {'config': type('C', (), {'hidden_size': 384})()})()
            self.classifier = nn.Linear(384, N_OPS)

        def forward(self, input_ids, attention_mask):
            batch = input_ids.size(0)
            return torch.randn(batch, N_OPS)

    # Create mock C3 model
    class MockC3(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = type('Config', (), {'config': type('C', (), {'hidden_size': 768})()})()

        def forward(self, input_ids, attention_mask):
            batch, seq_len = input_ids.shape
            start = torch.randn(batch, seq_len, 4)
            end = torch.randn(batch, seq_len, 4)
            return start, end

    # Initialize
    c2 = MockC2()
    c3 = MockC3()
    pipeline = MyceliumConstraintPipeline(c2, c3)

    # Print param counts
    counts = count_parameters(pipeline)
    print("\nParameter counts:")
    for name, count in counts.items():
        print(f"  {name}: {count:,}")

    # Create mock span groups
    span_groups = [
        SpanGroup(
            input_ids=torch.randint(0, 1000, (128,)),
            attention_mask=torch.ones(128),
            text="What is 2 + 3?",
        )
        for _ in range(3)
    ]

    # Forward pass
    output = pipeline(span_groups)

    print(f"\nForward pass:")
    print(f"  Final answer: {output.final_answer}")
    print(f"  Success: {output.success}")
    print(f"  Rounds: {len(output.rounds)}")
    print(f"  Losses: {list(output.losses.keys())}")

    # Inference
    result = inference(pipeline, span_groups)
    print(f"\nInference:")
    print(f"  Answer: {result['answer']}")
    print(f"  Operations: {result['operations']}")
    print(f"  Converged: {result['converged']} in {result['n_rounds']} rounds")

    print("\nAll tests passed!")
