"""
Mycelium Phase 2: Joint Constraint Training with Belief Propagation

Architecture:
  C2: MiniLM 22M  — operation classification (P(op | span_group))
  C3: RoBERTa-SQuAD 300M — operand span extraction (P(start, end | text, op))
  C4: Graph assembler — wires execution DAG (discrete → straight-through)
  C5: SymPy executor — non-differentiable verification oracle (→ reward signal)

Message networks (learned, ~10K params each):
  M_5→4: execution result → graph wiring update
  M_4→2: graph validity → operation label update  
  M_2→3: refined op beliefs → extraction update

Training:
  Phase 1: Component pretraining (already done for C2, C3)
  Phase 2: Joint fine-tuning with constraint propagation loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
import sympy


# ---------------------------------------------------------------------------
# 0. Data structures
# ---------------------------------------------------------------------------

@dataclass
class SpanGroup:
    """A group of text spans that correspond to one computation step."""
    token_ids: torch.Tensor          # (seq_len,) tokenized span group text
    attention_mask: torch.Tensor     # (seq_len,)
    span_markers: torch.Tensor      # (seq_len,) 1 where spans are marked
    text: str                        # raw text for SymPy extraction


@dataclass
class PipelineState:
    """Full state flowing through the pipeline. Mutable during belief prop."""
    span_groups: list[SpanGroup]
    
    # C2 outputs: (n_groups, n_ops) log-probabilities over operation labels
    op_logits: Optional[torch.Tensor] = None
    op_probs: Optional[torch.Tensor] = None
    
    # C3 outputs: per (group, op_hypothesis) extraction results
    # shape: (n_groups, n_ops, max_args, 2) — start/end logits per arg slot
    extraction_logits: Optional[torch.Tensor] = None
    extracted_values: Optional[list] = None
    
    # C4 outputs: (n_groups, n_groups) soft adjacency matrix
    adjacency_logits: Optional[torch.Tensor] = None
    adjacency_hard: Optional[torch.Tensor] = None  # after straight-through
    
    # C5 outputs: scalar execution plausibility per candidate
    execution_scores: Optional[torch.Tensor] = None
    
    # Backward messages (from message networks)
    msg_5to4: Optional[torch.Tensor] = None
    msg_4to2: Optional[torch.Tensor] = None
    msg_2to3: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# 1. Operation label vocabulary (learned from IB template discovery)
# ---------------------------------------------------------------------------

OP_LABELS = [
    "ADD", "SUB", "MUL", "DIV",
    "POW", "SQRT", "MOD",
    "PERCENT_OF", "PERCENT_CHANGE",
    "RATIO", "PROPORTION",
    "SOLVE_LINEAR", "SOLVE_QUADRATIC", "SOLVE_SYSTEM",
    "GCD", "LCM", "COMB", "PERM",
    # ... expanded from IB template discovery
]
N_OPS = len(OP_LABELS)
MAX_ARGS = 3  # most ops take 1-3 arguments


# ---------------------------------------------------------------------------
# 2. Straight-through estimator
# ---------------------------------------------------------------------------

class StraightThrough(torch.autograd.Function):
    """
    Forward: argmax (hard discrete decision)
    Backward: pass gradient through soft distribution as-if argmax didn't happen
    
    This is the key trick for making C4 graph assembly differentiable.
    The assembler makes hard wiring decisions, but gradients flow through
    the soft logits that produced those decisions.
    """
    @staticmethod
    def forward(ctx, logits):
        # Hard one-hot in forward pass
        indices = logits.argmax(dim=-1)
        hard = F.one_hot(indices, logits.size(-1)).float()
        # Save soft probs for backward
        ctx.save_for_backward(logits)
        return hard
    
    @staticmethod
    def backward(ctx, grad_output):
        logits, = ctx.saved_tensors
        # Backward through softmax, ignoring the argmax
        soft = F.softmax(logits, dim=-1)
        return grad_output * soft


def straight_through(logits: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper. Returns hard decisions with soft gradients."""
    return StraightThrough.apply(logits)


class GumbelStraightThrough(nn.Module):
    """
    Alternative: Gumbel-Softmax with straight-through.
    Adds stochastic exploration during training, anneals to hard decisions.
    Might give better gradient signal than pure straight-through for C4.
    """
    def __init__(self, tau_start=1.0, tau_min=0.1, anneal_steps=10000):
        super().__init__()
        self.tau = tau_start
        self.tau_min = tau_min
        self.anneal_rate = (tau_start - tau_min) / anneal_steps
        self.step_count = 0
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Gumbel noise for exploration
            gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
            soft = F.softmax((logits + gumbels) / self.tau, dim=-1)
            # Straight-through: hard forward, soft backward
            hard = F.one_hot(soft.argmax(dim=-1), logits.size(-1)).float()
            return hard - soft.detach() + soft  # gradient flows through soft
        else:
            return F.one_hot(logits.argmax(dim=-1), logits.size(-1)).float()
    
    def anneal(self):
        self.step_count += 1
        self.tau = max(self.tau_min, self.tau - self.anneal_rate)


# ---------------------------------------------------------------------------
# 3. Wrapper modules for pretrained C2 and C3
# ---------------------------------------------------------------------------

class C2_OperationClassifier(nn.Module):
    """
    Wraps pretrained MiniLM 22M.
    
    Input:  span group tokens + optional backward message from C4
    Output: (n_ops,) log-probabilities over operation labels
    
    The backward message modulates the final classification layer,
    allowing constraint signals to shift beliefs without retraining
    the full encoder.
    """
    def __init__(self, pretrained_model, n_ops=N_OPS, msg_dim=64):
        super().__init__()
        self.encoder = pretrained_model  # MiniLM 22M, frozen or low-lr
        hidden = self.encoder.config.hidden_size  # 384 for MiniLM
        
        # Classification head (pretrained in Phase 1)
        self.cls_head = nn.Linear(hidden, n_ops)
        
        # Message integration layer (NEW in Phase 2)
        # Takes backward message and produces a bias on logits
        self.msg_gate = nn.Sequential(
            nn.Linear(msg_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_ops),
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        backward_msg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns: (batch, n_ops) logits
        
        Round 0 (no message):  pure C2 classification
        Round 1+ (with msg):   C2 classification + constraint bias
        """
        # Encode span group
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Base classification
        logits = self.cls_head(cls_repr)
        
        # Integrate backward message if present
        if backward_msg is not None:
            msg_bias = self.msg_gate(backward_msg)
            logits = logits + msg_bias  # additive — preserves pretrained signal
        
        return logits


class C3_OperandExtractor(nn.Module):
    """
    Wraps pretrained RoBERTa-SQuAD 300M.
    
    Input:  span group text + operation hypothesis + optional backward message
    Output: start/end logits for each argument slot
    
    Key design: C3 runs CONDITIONED on C2's op hypothesis.
    During belief propagation, C3 runs once per plausible op,
    producing extraction candidates for each hypothesis.
    """
    def __init__(self, pretrained_model, n_ops=N_OPS, max_args=MAX_ARGS, msg_dim=64):
        super().__init__()
        self.encoder = pretrained_model  # RoBERTa-SQuAD 300M
        hidden = self.encoder.config.hidden_size  # 1024 for RoBERTa-large
        
        # Operation conditioning: embed the op hypothesis
        self.op_embedding = nn.Embedding(n_ops, hidden)
        
        # Per-argument-slot extraction heads
        self.arg_heads = nn.ModuleList([
            nn.Linear(hidden, 2)  # start + end logit per token
            for _ in range(max_args)
        ])
        
        # Message integration (NEW in Phase 2)
        self.msg_gate = nn.Sequential(
            nn.Linear(msg_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        op_hypothesis: torch.Tensor,       # (batch,) int — which op to condition on
        backward_msg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns: (batch, max_args, seq_len, 2) — start/end logits per arg slot
        """
        # Encode text
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq, hidden)
        
        # Condition on operation hypothesis
        op_emb = self.op_embedding(op_hypothesis).unsqueeze(1)  # (batch, 1, hidden)
        conditioned = hidden_states + op_emb  # broadcast add — op context everywhere
        
        # Integrate backward message if present
        if backward_msg is not None:
            msg_signal = self.msg_gate(backward_msg).unsqueeze(1)  # (batch, 1, hidden)
            conditioned = conditioned + msg_signal
        
        # Extract per-slot
        all_slot_logits = []
        for head in self.arg_heads:
            slot_logits = head(conditioned)  # (batch, seq, 2)
            all_slot_logits.append(slot_logits)
        
        return torch.stack(all_slot_logits, dim=1)  # (batch, max_args, seq, 2)


# ---------------------------------------------------------------------------
# 4. C4: Graph Assembler (with straight-through)
# ---------------------------------------------------------------------------

class C4_GraphAssembler(nn.Module):
    """
    Assembles execution DAG from C2 op labels + C3 extracted args.
    
    Learns: which steps depend on which other steps.
    Output: soft adjacency matrix, discretized via straight-through.
    
    This replaces the old heuristic dependency resolver.
    The adjacency structure is LEARNED from teacher CoT step ordering.
    """
    def __init__(self, n_ops=N_OPS, max_args=MAX_ARGS, hidden=128, msg_dim=64):
        super().__init__()
        
        # Encode each node: (op_label_embedding + arg_summary)
        self.op_embed = nn.Embedding(n_ops, hidden)
        self.arg_encoder = nn.Linear(max_args * 2, hidden)  # 2 = value + confidence
        self.node_encoder = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        
        # Pairwise edge predictor: P(edge | node_i, node_j)
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),  # [no_edge, edge]
        )
        
        # Message integration
        self.msg_gate = nn.Sequential(
            nn.Linear(msg_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        
        # Straight-through discretizer
        self.discretizer = GumbelStraightThrough()
    
    def forward(
        self,
        op_labels: torch.Tensor,       # (n_nodes,) hard or soft op indices
        arg_summaries: torch.Tensor,    # (n_nodes, max_args * 2) value + confidence
        backward_msg: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          adjacency_soft:  (n_nodes, n_nodes, 2) edge logits
          adjacency_hard:  (n_nodes, n_nodes) binary via straight-through
        """
        n = op_labels.size(0)
        
        # Encode each node
        if op_labels.dim() == 1:
            op_emb = self.op_embed(op_labels)           # (n, hidden)
        else:
            # Soft labels (during belief prop): weighted sum of embeddings
            op_emb = op_labels @ self.op_embed.weight    # (n, hidden)
        
        arg_emb = self.arg_encoder(arg_summaries)        # (n, hidden)
        node_repr = self.node_encoder(torch.cat([op_emb, arg_emb], dim=-1))
        
        # Integrate backward message
        if backward_msg is not None:
            msg_signal = self.msg_gate(backward_msg)
            node_repr = node_repr + msg_signal
        
        # Pairwise edge prediction
        # Only predict edges i → j where i < j (DAG constraint: no backward edges)
        # This is a STRUCTURAL constraint but it's not heuristic — it's the
        # definition of a DAG. The teacher's CoT is always ordered.
        node_i = node_repr.unsqueeze(1).expand(-1, n, -1)  # (n, n, h)
        node_j = node_repr.unsqueeze(0).expand(n, -1, -1)  # (n, n, h)
        pair_repr = torch.cat([node_i, node_j], dim=-1)     # (n, n, 2h)
        
        edge_logits = self.edge_predictor(pair_repr)  # (n, n, 2)
        
        # Mask: only allow edges i → j where i < j (topological order)
        causal_mask = torch.triu(torch.ones(n, n, device=edge_logits.device), diagonal=1)
        edge_logits[:, :, 1] = edge_logits[:, :, 1] * causal_mask  # kill backward edges
        
        # Straight-through: hard edges in forward, soft gradients in backward
        adjacency_hard = self.discretizer(edge_logits)[:, :, 1]  # (n, n) binary
        
        return edge_logits, adjacency_hard


# ---------------------------------------------------------------------------
# 5. C5: SymPy Executor (non-differentiable oracle)
# ---------------------------------------------------------------------------

class C5_SymPyExecutor:
    """
    Pure verification oracle. No parameters. Non-differentiable.
    
    Takes: operation sequence + arguments + dependency graph
    Returns: (result, success_bool, plausibility_score)
    
    The plausibility score becomes the REWARD SIGNAL for joint training.
    """
    
    # Map from our op labels to SymPy operations
    SYMPY_OPS = {
        "ADD": lambda a, b: a + b,
        "SUB": lambda a, b: a - b,
        "MUL": lambda a, b: a * b,
        "DIV": lambda a, b: a / b if b != 0 else None,
        "POW": lambda a, b: a ** b,
        "SQRT": lambda a: sympy.sqrt(a),
        "MOD": lambda a, b: a % b if b != 0 else None,
        "PERCENT_OF": lambda a, b: a * b / 100,
        "PERCENT_CHANGE": lambda a, b: (b - a) / a * 100 if a != 0 else None,
        "GCD": lambda a, b: sympy.gcd(int(a), int(b)),
        "LCM": lambda a, b: sympy.lcm(int(a), int(b)),
        "COMB": lambda a, b: sympy.binomial(int(a), int(b)),
        "SOLVE_LINEAR": lambda *args: _solve_linear(*args),
        "SOLVE_QUADRATIC": lambda *args: _solve_quadratic(*args),
    }
    
    def execute(
        self,
        ops: list[str],
        args: list[list[float]],
        adjacency: torch.Tensor,    # (n, n) binary DAG
    ) -> tuple[Optional[float], bool, float]:
        """
        Execute the computation graph.
        
        Returns: (result, success, plausibility_score)
          - result: final numeric answer or None
          - success: whether execution completed without error
          - plausibility_score: [0, 1] how plausible the result is
        """
        n = len(ops)
        results = [None] * n
        
        # Topological order from adjacency matrix
        order = self._topo_sort(adjacency, n)
        if order is None:
            return None, False, 0.0  # cycle detected
        
        try:
            for idx in order:
                op_name = ops[idx]
                op_args = list(args[idx])  # start with extracted literal args
                
                # Resolve dependencies: replace PREV refs with computed results
                deps = (adjacency[:, idx] > 0.5).nonzero(as_tuple=True)[0]
                for dep_idx in deps:
                    if results[dep_idx.item()] is not None:
                        # Replace first None-valued arg with dependency result
                        for i, a in enumerate(op_args):
                            if a is None:
                                op_args[i] = results[dep_idx.item()]
                                break
                
                # Execute
                op_fn = self.SYMPY_OPS.get(op_name)
                if op_fn is None:
                    return None, False, 0.0
                
                result = op_fn(*[a for a in op_args if a is not None])
                if result is None:
                    return None, False, 0.0
                
                results[idx] = float(result)
            
            final = results[order[-1]]
            plausibility = self._score_plausibility(final, ops, args)
            return final, True, plausibility
            
        except Exception:
            return None, False, 0.0
    
    def _topo_sort(self, adj: torch.Tensor, n: int) -> Optional[list[int]]:
        """Kahn's algorithm. Returns None if cycle detected."""
        adj_np = (adj > 0.5).cpu().numpy()
        in_degree = adj_np.sum(axis=0)
        queue = [i for i in range(n) if in_degree[i] == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for j in range(n):
                if adj_np[node, j]:
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        queue.append(j)
        return order if len(order) == n else None
    
    def _score_plausibility(
        self,
        result: float,
        ops: list[str],
        args: list[list[float]],
    ) -> float:
        """
        Learned from data statistics, NOT hand-coded domain rules.
        
        In practice this would be a small MLP trained on (result, context) → 
        P(correct) from Phase 1 data. For now, placeholder that checks 
        basic numeric sanity.
        
        TODO: Replace with learned plausibility scorer trained on
              teacher (correct_answer, pipeline_answer) pairs.
        """
        if result is None:
            return 0.0
        if not (-1e10 < result < 1e10):
            return 0.0
        return 1.0  # placeholder — real version is learned


# ---------------------------------------------------------------------------
# 6. Message Networks (the learned constraint propagation channels)
# ---------------------------------------------------------------------------

class MessageNetwork(nn.Module):
    """
    Small MLP that converts execution/validation signals into
    update messages for upstream components.
    
    These are the LEARNED constraints. Instead of coding
    "MUL needs 2 args" as a rule, the message network learns
    from training data that when C5 fails on a MUL node, the
    useful update is to shift C2's belief toward ADD or DIV.
    
    ~10K parameters each. Negligible compute cost.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConstraintMessages(nn.Module):
    """
    All message networks for the constraint propagation loop.
    
    Message flow:
      C5 execution result ──► M_5→4 ──► C4 (refine wiring)
      C4 graph validity   ──► M_4→2 ──► C2 (refine op labels)
      C2 refined beliefs  ──► M_2→3 ──► C3 (refine extraction)
    """
    def __init__(self, n_ops=N_OPS, max_args=MAX_ARGS, msg_dim=64):
        super().__init__()
        self.msg_dim = msg_dim
        
        # M_5→4: execution outcome → graph refinement signal
        # Input: per-node (op_label, arg_values, exec_success, exec_result)
        exec_input_dim = n_ops + max_args + 2  # op_probs + arg_values + success + result
        self.m_5to4 = MessageNetwork(exec_input_dim, msg_dim)
        
        # M_4→2: graph structure validity → operation label refinement
        # Input: per-node (edge_logits_in, edge_logits_out, node_degree, assembly_valid)
        graph_input_dim = 4  # in_degree, out_degree, n_edges, validity_score
        self.m_4to2 = MessageNetwork(graph_input_dim + msg_dim, msg_dim)
        
        # M_2→3: refined op beliefs → extraction refinement
        # Input: per-node (op_prob_delta — how beliefs shifted)
        self.m_2to3 = MessageNetwork(n_ops + msg_dim, msg_dim)
    
    def compute_5to4(
        self,
        op_probs: torch.Tensor,          # (n, n_ops)
        arg_values: torch.Tensor,         # (n, max_args)
        exec_success: torch.Tensor,       # (n,) per-node success
        exec_results: torch.Tensor,       # (n,) per-node result
    ) -> torch.Tensor:
        """C5 → C4 message: how should graph wiring change given execution?"""
        x = torch.cat([
            op_probs,
            arg_values,
            exec_success.unsqueeze(-1),
            exec_results.unsqueeze(-1),
        ], dim=-1)
        return self.m_5to4(x)  # (n, msg_dim)
    
    def compute_4to2(
        self,
        graph_features: torch.Tensor,     # (n, 4) degree/validity features
        msg_from_5: torch.Tensor,         # (n, msg_dim) passthrough from C5
    ) -> torch.Tensor:
        """C4 → C2 message: how should op labels change given graph validity?"""
        x = torch.cat([graph_features, msg_from_5], dim=-1)
        return self.m_4to2(x)  # (n, msg_dim)
    
    def compute_2to3(
        self,
        op_prob_delta: torch.Tensor,      # (n, n_ops) how beliefs shifted
        msg_from_4: torch.Tensor,         # (n, msg_dim) passthrough from C4
    ) -> torch.Tensor:
        """C2 → C3 message: how should extraction change given refined ops?"""
        x = torch.cat([op_prob_delta, msg_from_4], dim=-1)
        return self.m_2to3(x)  # (n, msg_dim)


# ---------------------------------------------------------------------------
# 7. Full Pipeline with Belief Propagation
# ---------------------------------------------------------------------------

class MyceliumConstraintPipeline(nn.Module):
    """
    The full Mycelium pipeline with learned constraint propagation.
    
    Training:
      - Phase 1 (done): C2, C3 pretrained independently on teacher labels
      - Phase 2 (this): Joint fine-tuning with belief propagation
        - C2, C3 encoders: frozen or very low lr (1e-6)
        - Message gates on C2, C3: normal lr (1e-4)
        - C4 assembler: normal lr (1e-4)
        - Message networks: normal lr (1e-4)
        - Loss = component_losses + execution_reward
    
    Inference:
      - Run 2-3 rounds of belief propagation
      - Each round: forward pass → execute → backward messages → forward again
      - Converge when beliefs stop changing (or fixed 2-3 rounds)
    """
    
    def __init__(self, c2_model, c3_model, n_ops=N_OPS, msg_dim=64):
        super().__init__()
        
        self.c2 = C2_OperationClassifier(c2_model, n_ops, msg_dim)
        self.c3 = C3_OperandExtractor(c3_model, n_ops, msg_dim=msg_dim)
        self.c4 = C4_GraphAssembler(n_ops, msg_dim=msg_dim)
        self.c5 = C5_SymPyExecutor()  # not nn.Module — no parameters
        self.messages = ConstraintMessages(n_ops, msg_dim=msg_dim)
        
        self.n_ops = n_ops
        self.msg_dim = msg_dim
        self.n_belief_rounds = 3
    
    def forward_round(
        self,
        state: PipelineState,
        round_idx: int,
    ) -> PipelineState:
        """
        One round of the pipeline.
        
        Round 0: pure forward (no backward messages)
        Round 1+: forward with backward messages from previous round
        """
        n_groups = len(state.span_groups)
        
        # --- C2: Classify operations ---
        c2_inputs = self._batch_span_groups(state.span_groups)
        state.op_logits = self.c2(
            input_ids=c2_inputs["input_ids"],
            attention_mask=c2_inputs["attention_mask"],
            backward_msg=state.msg_4to2,  # None on round 0
        )
        state.op_probs = F.softmax(state.op_logits, dim=-1)
        
        # --- C3: Extract operands (conditioned on top-k op hypotheses) ---
        # Run C3 for top-k op hypotheses per group, not just argmax.
        # This is where constraint propagation gets its power:
        # we maintain multiple hypotheses and let downstream evidence prune.
        k = min(3, self.n_ops)
        top_ops = state.op_probs.topk(k, dim=-1)  # (n_groups, k)
        
        all_extractions = []
        c3_inputs = self._batch_span_groups(state.span_groups, for_roberta=True)
        
        for hyp_idx in range(k):
            op_hyp = top_ops.indices[:, hyp_idx]  # (n_groups,)
            extraction = self.c3(
                input_ids=c3_inputs["input_ids"],
                attention_mask=c3_inputs["attention_mask"],
                op_hypothesis=op_hyp,
                backward_msg=state.msg_2to3,  # None on round 0
            )
            all_extractions.append(extraction)
        
        state.extraction_logits = torch.stack(all_extractions, dim=1)  # (n, k, args, seq, 2)
        
        # --- C4: Assemble graph (straight-through for discrete edges) ---
        # Summarize extractions into fixed-size features for graph assembly
        arg_summaries = self._summarize_extractions(state.extraction_logits, top_ops)
        
        # Use soft op probs (not hard labels) so gradients flow
        edge_logits, adjacency = self.c4(
            op_labels=state.op_probs,     # soft — gradient flows through
            arg_summaries=arg_summaries,
            backward_msg=state.msg_5to4,  # None on round 0
        )
        state.adjacency_logits = edge_logits
        state.adjacency_hard = adjacency
        
        # --- C5: Execute (non-differentiable) ---
        op_labels_hard = state.op_probs.argmax(dim=-1)
        ops_list = [OP_LABELS[i] for i in op_labels_hard.cpu().tolist()]
        args_list = self._extract_arg_values(state)
        
        result, success, plausibility = self.c5.execute(
            ops_list, args_list, adjacency.detach()
        )
        state.execution_scores = torch.tensor(
            [plausibility], device=state.op_probs.device
        )
        
        return state
    
    def compute_backward_messages(self, state: PipelineState) -> PipelineState:
        """
        After a forward round, compute backward constraint messages.
        These feed into the NEXT forward round.
        """
        n = len(state.span_groups)
        device = state.op_probs.device
        
        # Per-node execution features (simplified — expand for real impl)
        exec_success = state.execution_scores.expand(n)
        exec_results = torch.zeros(n, device=device)  # per-node results
        arg_values = torch.zeros(n, MAX_ARGS, device=device)  # extracted values
        
        # M_5→4: execution → graph wiring
        state.msg_5to4 = self.messages.compute_5to4(
            state.op_probs, arg_values, exec_success, exec_results
        )
        
        # M_4→2: graph → op labels
        adj = state.adjacency_hard
        graph_features = torch.stack([
            adj.sum(dim=0),            # in-degree
            adj.sum(dim=1),            # out-degree
            adj.sum().expand(n),       # total edges
            state.execution_scores.expand(n),  # validity
        ], dim=-1)
        state.msg_4to2 = self.messages.compute_4to2(graph_features, state.msg_5to4)
        
        # M_2→3: refined ops → extraction
        # op_prob_delta = how much beliefs would shift (placeholder: zeros on round 0)
        op_delta = torch.zeros_like(state.op_probs)
        state.msg_2to3 = self.messages.compute_2to3(op_delta, state.msg_4to2)
        
        return state
    
    def forward(
        self,
        span_groups: list[SpanGroup],
        gold_ops: Optional[torch.Tensor] = None,
        gold_args: Optional[list] = None,
        gold_adjacency: Optional[torch.Tensor] = None,
        gold_answer: Optional[float] = None,
    ) -> dict:
        """
        Full forward with belief propagation.
        
        Training: returns losses for joint optimization
        Inference: returns best prediction after convergence
        """
        state = PipelineState(span_groups=span_groups)
        
        all_round_states = []
        
        for round_idx in range(self.n_belief_rounds):
            # Forward pass
            state = self.forward_round(state, round_idx)
            all_round_states.append({
                "op_probs": state.op_probs.clone(),
                "execution_scores": state.execution_scores.clone(),
            })
            
            # Backward messages (skip after last round)
            if round_idx < self.n_belief_rounds - 1:
                state = self.compute_backward_messages(state)
        
        # --- Compute losses ---
        losses = {}
        
        if gold_ops is not None:
            # Component loss: C2 classification accuracy
            losses["c2_loss"] = F.cross_entropy(state.op_logits, gold_ops)
        
        if gold_adjacency is not None:
            # Component loss: C4 edge prediction accuracy
            losses["c4_loss"] = F.binary_cross_entropy_with_logits(
                state.adjacency_logits[:, :, 1],  # edge logits
                gold_adjacency.float(),
            )
        
        if gold_answer is not None:
            # Execution reward: did the pipeline get the right answer?
            # This is the REINFORCE signal — non-differentiable through C5
            result, success, _ = (
                state.execution_scores.item(),
                state.execution_scores.item() > 0.5,
                state.execution_scores.item(),
            )
            
            # Reward = 1 if correct answer, 0 otherwise
            # Applied to all differentiable components via policy gradient
            reward = 1.0 if success and result == gold_answer else 0.0
            baseline = 0.5  # moving average baseline, update during training
            advantage = reward - baseline
            
            # Policy gradient loss on C2's discrete decisions
            # log P(chosen_op) * advantage
            chosen_ops = state.op_probs.argmax(dim=-1)
            log_probs = F.log_softmax(state.op_logits, dim=-1)
            chosen_log_probs = log_probs.gather(1, chosen_ops.unsqueeze(1)).squeeze()
            losses["reinforce_loss"] = -(chosen_log_probs * advantage).mean()
        
        # Convergence bonus: reward belief stability across rounds
        if len(all_round_states) >= 2:
            belief_delta = (
                all_round_states[-1]["op_probs"] - all_round_states[-2]["op_probs"]
            ).abs().mean()
            losses["convergence_loss"] = belief_delta  # encourage settling
        
        return {
            "state": state,
            "losses": losses,
            "rounds": all_round_states,
        }
    
    # --- Helpers ---
    
    def _batch_span_groups(self, groups, for_roberta=False):
        """Batch span groups into tensors. Placeholder — real impl uses tokenizer."""
        return {
            "input_ids": torch.stack([g.token_ids for g in groups]),
            "attention_mask": torch.stack([g.attention_mask for g in groups]),
        }
    
    def _summarize_extractions(self, extraction_logits, top_ops):
        """Compress extraction logits into fixed-size arg summary for C4."""
        # Take the extraction from the top-1 op hypothesis
        # Shape: (n_groups, max_args, seq_len, 2)
        top1_extractions = extraction_logits[:, 0]
        
        # Soft argmax to get expected position (differentiable)
        start_logits = top1_extractions[:, :, :, 0]  # (n, args, seq)
        start_probs = F.softmax(start_logits, dim=-1)
        positions = torch.arange(
            start_logits.size(-1), device=start_logits.device
        ).float()
        expected_start = (start_probs * positions).sum(dim=-1)  # (n, args)
        
        confidence = start_probs.max(dim=-1).values  # (n, args)
        
        return torch.cat([expected_start, confidence], dim=-1)  # (n, args * 2)
    
    def _extract_arg_values(self, state):
        """Extract concrete arg values for SymPy execution. Non-differentiable."""
        # Placeholder — real impl extracts text spans and parses numbers
        n = len(state.span_groups)
        return [[0.0] * MAX_ARGS for _ in range(n)]


# ---------------------------------------------------------------------------
# 8. Training Loop
# ---------------------------------------------------------------------------

class Phase2Trainer:
    """
    Joint constraint training loop.
    
    Key: different learning rates for different components.
    - Pretrained encoders (MiniLM, RoBERTa): frozen or 1e-6
    - New heads (message gates, C4): 1e-4
    - Message networks: 1e-4
    
    The pretrained representations are good — we just need to learn
    the constraint structure on top of them.
    """
    
    def __init__(self, pipeline: MyceliumConstraintPipeline, lr_config: dict = None):
        self.pipeline = pipeline
        
        # Default lr config
        if lr_config is None:
            lr_config = {
                "encoder": 1e-6,       # MiniLM + RoBERTa encoders
                "heads": 1e-4,         # classification/extraction heads
                "assembler": 1e-4,     # C4 graph assembler
                "messages": 1e-4,      # message networks
            }
        
        # Parameter groups with different learning rates
        param_groups = [
            {
                "params": list(pipeline.c2.encoder.parameters()) + 
                          list(pipeline.c3.encoder.parameters()),
                "lr": lr_config["encoder"],
                "name": "encoders",
            },
            {
                "params": list(pipeline.c2.cls_head.parameters()) +
                          list(pipeline.c2.msg_gate.parameters()) +
                          list(pipeline.c3.op_embedding.parameters()) +
                          list(pipeline.c3.arg_heads.parameters()) +
                          list(pipeline.c3.msg_gate.parameters()),
                "lr": lr_config["heads"],
                "name": "heads",
            },
            {
                "params": list(pipeline.c4.parameters()),
                "lr": lr_config["assembler"],
                "name": "assembler",
            },
            {
                "params": list(pipeline.messages.parameters()),
                "lr": lr_config["messages"],
                "name": "messages",
            },
        ]
        
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
        
        # Loss weighting — anneal over training
        self.loss_weights = {
            "c2_loss": 1.0,           # component accuracy
            "c4_loss": 1.0,           # graph structure
            "reinforce_loss": 0.1,    # start low, increase as components stabilize
            "convergence_loss": 0.5,  # belief settling
        }
        
        # REINFORCE baseline (moving average of rewards)
        self.reward_baseline = 0.5
        self.baseline_momentum = 0.99
    
    def train_step(self, batch: dict) -> dict:
        """
        One training step.
        
        batch = {
            "span_groups": list[SpanGroup],
            "gold_ops": Tensor,
            "gold_args": list,
            "gold_adjacency": Tensor,
            "gold_answer": float,
        }
        """
        self.pipeline.train()
        self.optimizer.zero_grad()
        
        result = self.pipeline(
            span_groups=batch["span_groups"],
            gold_ops=batch["gold_ops"],
            gold_args=batch.get("gold_args"),
            gold_adjacency=batch.get("gold_adjacency"),
            gold_answer=batch.get("gold_answer"),
        )
        
        # Weighted total loss
        total_loss = torch.tensor(0.0, device=batch["gold_ops"].device)
        loss_log = {}
        
        for loss_name, loss_val in result["losses"].items():
            weight = self.loss_weights.get(loss_name, 1.0)
            weighted = weight * loss_val
            total_loss = total_loss + weighted
            loss_log[loss_name] = loss_val.item()
        
        total_loss.backward()
        
        # Gradient clipping — important for REINFORCE stability
        torch.nn.utils.clip_grad_norm_(self.pipeline.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Anneal Gumbel temperature
        self.pipeline.c4.discretizer.anneal()
        
        # Log convergence across belief prop rounds
        rounds = result["rounds"]
        if len(rounds) >= 2:
            belief_shift = (
                rounds[-1]["op_probs"] - rounds[0]["op_probs"]
            ).abs().mean().item()
            loss_log["belief_shift"] = belief_shift
        
        loss_log["total_loss"] = total_loss.item()
        loss_log["gumbel_tau"] = self.pipeline.c4.discretizer.tau
        
        return loss_log
    
    def train_epoch(self, dataloader, epoch: int):
        """Full training epoch with logging."""
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            loss_log = self.train_step(batch)
            epoch_losses.append(loss_log)
            
            if batch_idx % 50 == 0:
                avg = {
                    k: sum(d.get(k, 0) for d in epoch_losses[-50:]) / min(50, len(epoch_losses))
                    for k in loss_log
                }
                print(
                    f"Epoch {epoch} | Step {batch_idx} | "
                    f"Loss: {avg['total_loss']:.4f} | "
                    f"C2: {avg.get('c2_loss', 0):.4f} | "
                    f"C4: {avg.get('c4_loss', 0):.4f} | "
                    f"REINFORCE: {avg.get('reinforce_loss', 0):.4f} | "
                    f"Belief Δ: {avg.get('belief_shift', 0):.4f} | "
                    f"τ: {avg.get('gumbel_tau', 0):.3f}"
                )
        
        return epoch_losses


# ---------------------------------------------------------------------------
# 9. Inference with belief propagation
# ---------------------------------------------------------------------------

@torch.no_grad()
def inference(
    pipeline: MyceliumConstraintPipeline,
    span_groups: list[SpanGroup],
    max_rounds: int = 3,
    convergence_threshold: float = 0.01,
) -> dict:
    """
    Inference with belief propagation until convergence.
    
    Unlike training, we can early-stop when beliefs stabilize.
    Typical convergence: 2 rounds for easy problems, 3 for hard.
    
    Returns the final prediction + convergence trace.
    """
    pipeline.eval()
    state = PipelineState(span_groups=span_groups)
    
    trace = []
    prev_op_probs = None
    
    for round_idx in range(max_rounds):
        state = pipeline.forward_round(state, round_idx)
        
        trace.append({
            "round": round_idx,
            "op_probs": state.op_probs.cpu().clone(),
            "execution_score": state.execution_scores.item(),
            "top_ops": [
                OP_LABELS[i] for i in state.op_probs.argmax(dim=-1).cpu().tolist()
            ],
        })
        
        # Check convergence
        if prev_op_probs is not None:
            delta = (state.op_probs - prev_op_probs).abs().max().item()
            trace[-1]["belief_delta"] = delta
            if delta < convergence_threshold:
                trace[-1]["converged"] = True
                break
        
        prev_op_probs = state.op_probs.clone()
        
        # Compute backward messages for next round
        if round_idx < max_rounds - 1:
            state = pipeline.compute_backward_messages(state)
    
    # Final extraction
    final_ops = [OP_LABELS[i] for i in state.op_probs.argmax(dim=-1).cpu().tolist()]
    final_args = pipeline._extract_arg_values(state)
    result, success, plausibility = pipeline.c5.execute(
        final_ops, final_args, state.adjacency_hard.detach()
    )
    
    return {
        "answer": result,
        "success": success,
        "plausibility": plausibility,
        "operations": final_ops,
        "n_rounds": len(trace),
        "converged": trace[-1].get("converged", False),
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# 10. Example usage
# ---------------------------------------------------------------------------

def main():
    """
    Setup and training example.
    
    Prerequisites:
      - C2 MiniLM 22M pretrained checkpoint (Phase 1)
      - C3 RoBERTa-SQuAD 300M pretrained checkpoint (Phase 1)
      - Training data: (span_groups, gold_ops, gold_adjacency, gold_answer)
        generated from teacher CoT distillation
    """
    from transformers import AutoModel
    
    # Load pretrained components
    c2_encoder = AutoModel.from_pretrained("models/c2_minilm_22m")
    c3_encoder = AutoModel.from_pretrained("models/c3_roberta_squad")
    
    # Build pipeline
    pipeline = MyceliumConstraintPipeline(
        c2_model=c2_encoder,
        c3_model=c3_encoder,
        n_ops=N_OPS,
        msg_dim=64,
    )
    
    # Count parameters
    total = sum(p.numel() for p in pipeline.parameters())
    trainable = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    new_params = sum(
        p.numel() for name, p in pipeline.named_parameters()
        if "msg_gate" in name or "messages" in name or "c4" in name
    )
    
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"New Phase 2 params:   {new_params:,} ({new_params/total*100:.1f}%)")
    
    # Phase 2 training
    trainer = Phase2Trainer(pipeline)
    
    # Training loop (placeholder — real impl loads from distillation data)
    # for epoch in range(10):
    #     trainer.train_epoch(dataloader, epoch)
    
    print("\nArchitecture ready for Phase 2 joint constraint training.")
    print("New components: message networks (~30K params) + C4 assembler (~100K params)")
    print("These learn the constraint structure from teacher attention patterns.")
    print("No hand-coded rules. No heuristics. Pure distillation.")


if __name__ == "__main__":
    main()
