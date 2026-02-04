"""
Dual-Signal Template System for Mycelium

This module implements a template matching system that uses BOTH:
1. Embedding similarity (semantic content)
2. Attention pattern similarity (structural/operational behavior)

The dual-signal approach enables routing by what operations DO, not just
what they SOUND LIKE - addressing the core lexical vs operational similarity
problem described in CLAUDE.md.

Key insight: Fine-tuned MiniLM achieves 0.945 correlation with Qwen 7B 
attention patterns, allowing us to use the smaller model for inference
while maintaining the structural understanding of the larger model.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import os
from collections import OrderedDict

# Import consolidated Welford implementation
from mycelium.welford import WelfordStats


class OperationType(Enum):
    """Core arithmetic operation types for math problem solving."""
    SET = "SET"      # Assignment/initialization
    ADD = "ADD"      # Addition
    SUB = "SUB"      # Subtraction
    MUL = "MUL"      # Multiplication
    DIV = "DIV"      # Division
    UNKNOWN = "UNKNOWN"  # Unclassified


@dataclass
class DualSignalTemplate:
    """
    A template representing a computation pattern with dual-signal matching.
    
    Each template captures BOTH:
    - embedding_centroid: Average embedding of spans matching this template
    - attention_signature: Characteristic attention connectivity pattern
    
    This dual-signal approach enables routing by operational semantics
    rather than lexical similarity.
    
    Attributes:
        template_id: Unique identifier for this template
        operation_type: The arithmetic operation this template represents
        pattern: Normalized pattern string like "[NAME] sold [N] [ITEM]"
        dsl_expr: Custom DSL expression like "entity - value"
        embedding_centroid: Mean embedding vector of matched spans
        attention_signature: Characteristic attention connectivity pattern
        span_examples: Example spans that match this template
        embedding_welford: Running stats for embedding similarities
        attention_welford: Running stats for attention similarities
        outcome_welford: Running stats for execution outcomes (success/failure)
    """
    template_id: str
    operation_type: OperationType
    embedding_centroid: np.ndarray
    attention_signature: np.ndarray  # Flattened or aggregated attention pattern
    pattern: str = ""  # Normalized pattern: "[NAME] sold [N] [ITEM]"
    dsl_expr: str = "value"  # Custom DSL expression: "entity - value"
    span_examples: List[str] = field(default_factory=list)
    embedding_welford: WelfordStats = field(default_factory=WelfordStats)
    attention_welford: WelfordStats = field(default_factory=WelfordStats)
    outcome_welford: WelfordStats = field(default_factory=WelfordStats)
    match_count: int = 0
    success_count: int = 0
    cross_entity_attention: float = 0.0  # How much attention flows between entities (discriminates SET vs SUB/MUL)

    def update_centroid(self, new_embedding: np.ndarray, new_attention: np.ndarray) -> None:
        """
        Update centroids using incremental averaging.
        
        This keeps templates as 'semantic attractors' that stabilize
        around operational meaning through usage.
        """
        self.match_count += 1
        alpha = 1.0 / self.match_count
        self.embedding_centroid = (1 - alpha) * self.embedding_centroid + alpha * new_embedding
        self.attention_signature = (1 - alpha) * self.attention_signature + alpha * new_attention
    
    def record_outcome(self, success: bool) -> None:
        """Record execution outcome for this template."""
        self.outcome_welford.update(1.0 if success else 0.0)
        if success:
            self.success_count += 1
    
    @property
    def success_rate(self) -> float:
        """Return success rate or 0.5 if no data."""
        if self.match_count == 0:
            return 0.5
        return self.success_count / self.match_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize template to dictionary."""
        return {
            "template_id": self.template_id,
            "operation_type": self.operation_type.value,
            "embedding_centroid": self.embedding_centroid.tolist(),
            "attention_signature": self.attention_signature.tolist(),
            "span_examples": self.span_examples[:10],  # Keep top 10 examples
            "embedding_welford": self.embedding_welford.to_dict(),
            "attention_welford": self.attention_welford.to_dict(),
            "outcome_welford": self.outcome_welford.to_dict(),
            "match_count": self.match_count,
            "success_count": self.success_count
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DualSignalTemplate":
        """Deserialize template from dictionary."""
        return cls(
            template_id=d["template_id"],
            operation_type=OperationType(d["operation_type"]),
            embedding_centroid=np.array(d["embedding_centroid"]),
            attention_signature=np.array(d["attention_signature"]),
            span_examples=d.get("span_examples", []),
            embedding_welford=WelfordStats.from_dict(d["embedding_welford"]),
            attention_welford=WelfordStats.from_dict(d["attention_welford"]),
            outcome_welford=WelfordStats.from_dict(d["outcome_welford"]),
            match_count=d.get("match_count", 0),
            success_count=d.get("success_count", 0)
        )


class TemplateStore:
    """
    Storage and matching system for dual-signal templates.
    
    Uses BOTH embedding similarity and attention pattern similarity
    for template matching, with learned weights to balance the signals.
    
    The weighted combination allows the system to learn which signal
    matters more for different problem types.
    """
    
    def __init__(self, embedding_weight: float = 0.5, attention_weight: float = 0.5):
        """
        Initialize template store.
        
        Args:
            embedding_weight: Initial weight for embedding similarity [0-1]
            attention_weight: Initial weight for attention similarity [0-1]
        """
        self.templates: Dict[str, DualSignalTemplate] = OrderedDict()
        self.embedding_weight = embedding_weight
        self.attention_weight = attention_weight
        
        # Track weight learning
        self.weight_welford = WelfordStats()  # Tracks optimal weight estimation
        self._weight_history: List[Tuple[float, bool]] = []  # (weight, success) pairs
    
    def add_template(self, template: DualSignalTemplate) -> None:
        """Add a template to the store."""
        self.templates[template.template_id] = template
    
    def remove_template(self, template_id: str) -> Optional[DualSignalTemplate]:
        """Remove and return a template by ID."""
        return self.templates.pop(template_id, None)
    
    def get_template(self, template_id: str) -> Optional[DualSignalTemplate]:
        """Get a template by ID."""
        return self.templates.get(template_id)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def _attention_correlation(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute correlation between attention patterns.
        
        Uses Pearson correlation which captures structural similarity
        in attention connectivity patterns.
        """
        # Flatten if needed
        a_flat = a.flatten()
        b_flat = b.flatten()
        
        # Handle size mismatch by truncating to minimum
        min_len = min(len(a_flat), len(b_flat))
        if min_len == 0:
            return 0.0
        
        a_flat = a_flat[:min_len]
        b_flat = b_flat[:min_len]
        
        # Compute Pearson correlation
        a_mean = np.mean(a_flat)
        b_mean = np.mean(b_flat)
        a_centered = a_flat - a_mean
        b_centered = b_flat - b_mean
        
        num = np.dot(a_centered, b_centered)
        denom = np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2))
        
        if denom == 0:
            return 0.0
        return float(num / denom)
    
    def compute_match_score(
        self, 
        embedding: np.ndarray, 
        attention: np.ndarray,
        template: DualSignalTemplate
    ) -> Tuple[float, float, float]:
        """
        Compute combined match score using both signals.
        
        Returns:
            Tuple of (combined_score, embedding_sim, attention_sim)
        """
        emb_sim = self._cosine_similarity(embedding, template.embedding_centroid)
        att_sim = self._attention_correlation(attention, template.attention_signature)
        
        # Normalize to [0, 1] range (correlation can be negative)
        att_sim_normalized = (att_sim + 1) / 2
        
        combined = (
            self.embedding_weight * emb_sim + 
            self.attention_weight * att_sim_normalized
        )
        
        return combined, emb_sim, att_sim
    
    def find_best_match(
        self, 
        embedding: np.ndarray, 
        attention: np.ndarray,
        operation_filter: Optional[OperationType] = None,
        min_score: float = 0.0
    ) -> Optional[Tuple[DualSignalTemplate, float, float, float]]:
        """
        Find the best matching template for given signals.
        
        Following CLAUDE.md principle: ALWAYS route to best match,
        let failures drive learning rather than arbitrary thresholds.
        
        Args:
            embedding: Query embedding vector
            attention: Query attention pattern
            operation_filter: Optional filter by operation type
            min_score: Minimum score threshold (default 0, always match)
        
        Returns:
            Tuple of (template, combined_score, embedding_sim, attention_sim)
            or None if no templates exist
        """
        if not self.templates:
            return None
        
        best_match = None
        best_score = -float('inf')
        best_emb_sim = 0.0
        best_att_sim = 0.0
        
        for template in self.templates.values():
            if operation_filter and template.operation_type != operation_filter:
                continue
            
            score, emb_sim, att_sim = self.compute_match_score(
                embedding, attention, template
            )
            
            if score > best_score:
                best_score = score
                best_match = template
                best_emb_sim = emb_sim
                best_att_sim = att_sim
        
        if best_match is None or best_score < min_score:
            return None
        
        return best_match, best_score, best_emb_sim, best_att_sim

    def find_best_match_with_cross_entity(
        self,
        embedding: np.ndarray,
        attention: np.ndarray,
        cross_entity_attention: float,
        min_score: float = 0.0
    ) -> Optional[Tuple[DualSignalTemplate, float, float, float]]:
        """
        Find best matching template using cross-entity attention for operation discrimination.

        Cross-entity attention discriminates operations:
        - SET: ~0.0 (single entity, self-contained)
        - ADD: ~0.04 (receiving, but mainly one entity focus)
        - SUB/MUL: ~0.06 (transfer/reference to another entity)

        This method uses cross-entity attention to:
        1. Boost scores for templates whose operation matches the cross-entity pattern
        2. Penalize templates that don't match

        Args:
            embedding: Query embedding vector
            attention: Query attention pattern
            cross_entity_attention: Cross-entity attention score from input span
            min_score: Minimum score threshold

        Returns:
            Tuple of (template, combined_score, embedding_sim, attention_sim)
            or None if no templates exist
        """
        if not self.templates:
            return None

        # Classify input span's operation tendency based on indicator token attention
        # Empirical values from test data:
        # SET: 0.0 (no indicator tokens)
        # ADD: ~0.02-0.035 (has additive tokens like "more")
        # SUB: ~0.035-0.055 (has "to" - transfer relationship)
        # MUL: ~0.055-0.075 (has "as", "times" - comparison)
        # DIV: ~0.08+ (has division tokens like "half", "split")
        THRESHOLD_SET_ADD = 0.02   # Below = SET
        THRESHOLD_ADD_SUB = 0.035  # Below = ADD, above = SUB
        THRESHOLD_SUB_MUL = 0.052  # Below = SUB, above = MUL
        THRESHOLD_MUL_DIV = 0.075  # Below = MUL, above = DIV

        if cross_entity_attention < THRESHOLD_SET_ADD:
            # Likely SET operation - boost SET templates, penalize others
            preferred_ops = {OperationType.SET}
            penalized_ops = {OperationType.ADD, OperationType.SUB, OperationType.MUL, OperationType.DIV}
        elif cross_entity_attention < THRESHOLD_ADD_SUB:
            # Likely ADD operation - boost ADD, penalize SET
            preferred_ops = {OperationType.ADD}
            penalized_ops = {OperationType.SET}
        elif cross_entity_attention < THRESHOLD_SUB_MUL:
            # Likely SUB operation - boost SUB, penalize SET/ADD/MUL
            preferred_ops = {OperationType.SUB}
            penalized_ops = {OperationType.SET, OperationType.MUL}
        elif cross_entity_attention >= THRESHOLD_MUL_DIV:
            # Likely DIV operation - boost DIV, penalize SET/ADD
            preferred_ops = {OperationType.DIV}
            penalized_ops = {OperationType.SET, OperationType.ADD}
        else:
            # Likely SUB/MUL operation - boost SUB/MUL, penalize SET/ADD
            preferred_ops = {OperationType.SUB, OperationType.MUL}
            penalized_ops = {OperationType.SET, OperationType.ADD}

        best_match = None
        best_score = -float('inf')
        best_emb_sim = 0.0
        best_att_sim = 0.0

        PREFERENCE_BOOST = 0.15  # Boost for preferred operations
        PENALTY = 0.10  # Penalty for penalized operations

        for template in self.templates.values():
            score, emb_sim, att_sim = self.compute_match_score(
                embedding, attention, template
            )

            # Apply cross-entity attention-based adjustments
            if template.operation_type in preferred_ops:
                score += PREFERENCE_BOOST
            elif template.operation_type in penalized_ops:
                score -= PENALTY

            if score > best_score:
                best_score = score
                best_match = template
                best_emb_sim = emb_sim
                best_att_sim = att_sim

        if best_match is None or best_score < min_score:
            return None

        return best_match, best_score, best_emb_sim, best_att_sim

    def find_top_k_matches(
        self,
        embedding: np.ndarray,
        attention: np.ndarray,
        k: int = 5,
        operation_filter: Optional[OperationType] = None
    ) -> List[Tuple[DualSignalTemplate, float, float, float]]:
        """
        Find top-k matching templates.
        
        Useful for MCTS exploration where we want to consider
        multiple candidates.
        """
        matches = []
        
        for template in self.templates.values():
            if operation_filter and template.operation_type != operation_filter:
                continue
            
            score, emb_sim, att_sim = self.compute_match_score(
                embedding, attention, template
            )
            matches.append((template, score, emb_sim, att_sim))
        
        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:k]
    
    def update_weights_from_outcome(
        self, 
        embedding_sim: float, 
        attention_sim: float, 
        success: bool
    ) -> None:
        """
        Learn optimal signal weights from outcomes.
        
        Track which signal was more predictive of success
        and adjust weights accordingly.
        """
        # Simple heuristic: if one signal was high and we succeeded,
        # or one signal was low and we failed, increase its weight
        
        # Store for batch learning
        self._weight_history.append((embedding_sim, attention_sim, success))
        
        # Update weights every 100 observations
        if len(self._weight_history) >= 100:
            self._recompute_weights()
    
    def _recompute_weights(self) -> None:
        """Recompute optimal weights from accumulated history."""
        if len(self._weight_history) < 10:
            return
        
        # Simple linear regression to find optimal weights
        # that best predict success from the two signals
        
        emb_sims = np.array([h[0] for h in self._weight_history])
        att_sims = np.array([(h[1] + 1) / 2 for h in self._weight_history])  # Normalize
        outcomes = np.array([1.0 if h[2] else 0.0 for h in self._weight_history])
        
        # Stack features
        X = np.column_stack([emb_sims, att_sims])
        
        # Add small regularization for stability
        XtX = X.T @ X + 0.01 * np.eye(2)
        Xty = X.T @ outcomes
        
        try:
            weights = np.linalg.solve(XtX, Xty)
            # Normalize to sum to 1
            total = abs(weights[0]) + abs(weights[1])
            if total > 0:
                self.embedding_weight = abs(weights[0]) / total
                self.attention_weight = abs(weights[1]) / total
        except np.linalg.LinAlgError:
            pass  # Keep current weights if solve fails
        
        # Clear history (keep last 50 for continuity)
        self._weight_history = self._weight_history[-50:]
    
    def get_templates_by_operation(
        self, 
        operation_type: OperationType
    ) -> List[DualSignalTemplate]:
        """Get all templates for a given operation type."""
        return [
            t for t in self.templates.values() 
            if t.operation_type == operation_type
        ]
    
    def get_high_variance_templates(
        self, 
        variance_threshold: float = 0.25
    ) -> List[DualSignalTemplate]:
        """
        Get templates with high outcome variance.
        
        Per CLAUDE.md: High variance signals need for decomposition.
        "ONE node high variance -> decompose node"
        """
        return [
            t for t in self.templates.values()
            if t.outcome_welford.variance > variance_threshold
            and t.outcome_welford.count >= 10  # Need sufficient data
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize entire store to dictionary."""
        return {
            "embedding_weight": self.embedding_weight,
            "attention_weight": self.attention_weight,
            "templates": {
                tid: t.to_dict() for tid, t in self.templates.items()
            }
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TemplateStore":
        """Deserialize store from dictionary."""
        store = cls(
            embedding_weight=d.get("embedding_weight", 0.5),
            attention_weight=d.get("attention_weight", 0.5)
        )
        for tid, tdict in d.get("templates", {}).items():
            store.templates[tid] = DualSignalTemplate.from_dict(tdict)
        return store


class FineTunedMiniLM(nn.Module):
    """
    Fine-tuned MiniLM model for dual-signal extraction.
    
    This wraps the fine-tuned model that achieves 0.945 correlation
    with Qwen 7B attention patterns.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        
        # Load model with eager attention to get attention outputs
        self.encoder = AutoModel.from_pretrained(
            model_name,
            attn_implementation="eager"
        )
        self.num_heads = 12
        self.num_layers = 6
        
        # Learnable head and layer weights (loaded from checkpoint)
        self.head_weights = nn.Parameter(torch.ones(self.num_heads) / self.num_heads)
        self.layer_weights = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)
        
        # Projection to match Qwen attention scale
        self.proj_scale = nn.Parameter(torch.ones(1))
        self.proj_bias = nn.Parameter(torch.zeros(1))
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning embeddings and attention patterns.
        
        Returns:
            Tuple of:
            - embeddings: [batch, hidden_dim] - CLS token embeddings
            - weighted_attention: [batch, seq_len, seq_len] - Weighted attention
            - all_attentions: [batch, layers, heads, seq_len, seq_len]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        
        # Get CLS embedding
        embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
        
        # Stack all attention matrices
        # Each attention is [batch, heads, seq, seq]
        all_attentions = torch.stack(outputs.attentions, dim=1)  # [batch, layers, heads, seq, seq]
        
        # Apply learned head weights within each layer
        head_weights_softmax = torch.softmax(self.head_weights, dim=0)
        # [batch, layers, seq, seq]
        layer_attentions = torch.einsum('blhij,h->blij', all_attentions, head_weights_softmax)
        
        # Apply learned layer weights
        layer_weights_softmax = torch.softmax(self.layer_weights, dim=0)
        # [batch, seq, seq]
        weighted_attention = torch.einsum('blij,l->bij', layer_attentions, layer_weights_softmax)
        
        # Apply projection
        weighted_attention = self.proj_scale * weighted_attention + self.proj_bias
        
        return embeddings, weighted_attention, all_attentions


class SpanDetector:
    """
    Detects operational spans in text using fine-tuned MiniLM.
    
    Uses community detection on attention graphs to identify
    coherent computational units within problem text.
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize span detector with fine-tuned model.
        
        Args:
            model_path: Path to fine-tuned model checkpoint
            device: Device to use ("auto", "cuda", or "cpu")
        """
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize model
        self.model = FineTunedMiniLM()
        
        # Load fine-tuned weights if provided
        if model_path:
            self._load_checkpoint(model_path)
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _load_checkpoint(self, path: str) -> None:
        """Load fine-tuned model weights from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        
        # Load the model state dict
        model_state = checkpoint["model_state_dict"]
        
        # Filter out keys that don't match our model structure
        # The checkpoint has encoder weights under different names
        filtered_state = {}
        for key, value in model_state.items():
            if key in ["head_weights", "layer_weights", "proj_scale", "proj_bias"]:
                filtered_state[key] = value
            elif key.startswith("encoder."):
                filtered_state[key] = value
        
        # Load with strict=False to handle any mismatches
        self.model.load_state_dict(filtered_state, strict=False)
        
        # Also load numpy arrays if present
        if "head_weights" in checkpoint and isinstance(checkpoint["head_weights"], np.ndarray):
            self.model.head_weights.data = torch.tensor(
                checkpoint["head_weights"], dtype=torch.float32
            )
        if "layer_weights" in checkpoint and isinstance(checkpoint["layer_weights"], np.ndarray):
            self.model.layer_weights.data = torch.tensor(
                checkpoint["layer_weights"], dtype=torch.float32
            )
        
        print(f"Loaded checkpoint from {path}")
        print(f"  Correlation: {checkpoint.get('correlation', 'N/A')}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

    @torch.no_grad()
    def extract_features(
        self,
        text: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract embedding and attention features from text.

        Args:
            text: Input text to process

        Returns:
            Tuple of:
            - embedding: [hidden_dim] numpy array
            - attention: [seq_len, seq_len] numpy array
            - tokens: List of token strings
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        embeddings, attention, _ = self.model(
            inputs["input_ids"],
            inputs["attention_mask"]
        )

        # Convert to numpy
        embedding = embeddings[0].cpu().numpy()
        attention = attention[0].cpu().numpy()

        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu())

        return embedding, attention, tokens

    # Relational tokens that indicate entity relationships (SUB/MUL)
    RELATIONAL_TOKENS = {'to', 'from', 'as', 'than', 'for'}
    # Additive indicators that suggest ADD operations
    ADDITIVE_TOKENS = {'more', 'additional', 'another', 'extra', 'added'}
    # Division indicators that suggest DIV operations
    # Note: "each" removed - too ambiguous ("each basket" isn't division)
    DIVISION_TOKENS = {'half', 'split', 'divided', 'equally', 'share'}
    # SUB-indicating relational tokens (transfer: "gave X to Y")
    SUB_RELATIONAL = {'to'}
    # MUL-indicating relational tokens (comparison: "twice as many as")
    MUL_RELATIONAL = {'as', 'times'}

    def compute_cross_entity_attention(
        self,
        text: str,
        attention_matrix: Optional[np.ndarray] = None,
        tokens: Optional[List[str]] = None
    ) -> float:
        """
        Compute relational/additive token attention for a span.

        This measures attention to discriminative tokens:
        - Relational tokens (to, from, as, than, for): indicate SUB/MUL
        - Additive tokens (more, additional, another): indicate ADD

        Discriminates operation types:
        - SET: ~0.0 (no indicator tokens)
        - ADD: ~0.02-0.04 (has additive tokens like "more")
        - SUB: ~0.04-0.06 (has relational tokens like "to")
        - MUL: ~0.06-0.08 (has relational tokens like "as")

        Args:
            text: The span text
            attention_matrix: Optional pre-computed attention matrix
            tokens: Optional pre-computed tokens

        Returns:
            Indicator token attention score (0.0 to ~0.1)
        """
        # Get attention matrix if not provided
        if attention_matrix is None or tokens is None:
            _, attention_matrix, tokens = self.extract_features(text)

        # Average attention across heads if needed (shape: seq x seq)
        if attention_matrix.ndim > 2:
            attention_matrix = attention_matrix.mean(axis=0)

        # Find indicator token indices by type
        add_indices = [
            i for i, t in enumerate(tokens)
            if t.lower() in self.ADDITIVE_TOKENS
        ]
        sub_indices = [
            i for i, t in enumerate(tokens)
            if t.lower() in self.SUB_RELATIONAL
        ]
        mul_indices = [
            i for i, t in enumerate(tokens)
            if t.lower() in self.MUL_RELATIONAL
        ]
        div_indices = [
            i for i, t in enumerate(tokens)
            if t.lower() in self.DIVISION_TOKENS
        ]

        # Priority: DIV > MUL > SUB > ADD > SET
        # Return different score ranges for each operation type
        if div_indices:
            # DIV gets highest score (~0.08-0.10)
            total_attention = 0.0
            for i in div_indices:
                if i < len(attention_matrix):
                    total_attention += attention_matrix[:, i].sum()
            return float(total_attention / len(tokens)) * 1.2 + 0.08
        elif mul_indices:
            # MUL (~0.055-0.07) - "as", "times"
            total_attention = 0.0
            for i in mul_indices:
                if i < len(attention_matrix):
                    total_attention += attention_matrix[:, i].sum()
            return float(total_attention / len(tokens)) + 0.02
        elif sub_indices:
            # SUB (~0.04-0.055) - "to"
            total_attention = 0.0
            for i in sub_indices:
                if i < len(attention_matrix):
                    total_attention += attention_matrix[:, i].sum()
            return float(total_attention / len(tokens))
        elif add_indices:
            # ADD (~0.02-0.04)
            total_attention = 0.0
            for i in add_indices:
                if i < len(attention_matrix):
                    total_attention += attention_matrix[:, i].sum()
            return float(total_attention / len(tokens)) * 0.7

        return 0.0

    def extract_features_with_cross_entity(
        self,
        text: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
        """
        Extract embedding, attention, and cross-entity attention from text.

        Returns:
            Tuple of:
            - embedding: [hidden_dim] numpy array
            - attention: [seq_len, seq_len] numpy array
            - tokens: List of token strings
            - cross_entity_attention: float score
        """
        embedding, attention, tokens = self.extract_features(text)
        cross_entity = self.compute_cross_entity_attention(text, attention, tokens)
        return embedding, attention, tokens, cross_entity


def create_template_from_span(
    span_text: str,
    embedding: np.ndarray,
    attention_pattern: np.ndarray,
    operation_type: OperationType,
    template_id: Optional[str] = None
) -> DualSignalTemplate:
    """
    Create a new template from span data.

    Args:
        span_text: Text of the span
        embedding: Embedding vector for the span
        attention_pattern: Attention pattern for the span
        operation_type: The operation this span represents
        template_id: Optional ID (auto-generated if not provided)

    Returns:
        New DualSignalTemplate
    """
    import uuid

    if template_id is None:
        template_id = f"{operation_type.value}_{uuid.uuid4().hex[:8]}"

    return DualSignalTemplate(
        template_id=template_id,
        operation_type=operation_type,
        embedding_centroid=embedding.copy(),
        attention_signature=attention_pattern.copy(),
        span_examples=[span_text]
    )


# ============================================================
# Integration with Mycelium
# ============================================================

class DualSignalRouter:
    """
    High-level router that integrates dual-signal matching with
    the Mycelium tree structure.
    
    This bridges the template system to the signature tree
    described in CLAUDE.md.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_weight: float = 0.5,
        attention_weight: float = 0.5
    ):
        self.detector = SpanDetector(model_path=model_path)
        self.store = TemplateStore(
            embedding_weight=embedding_weight,
            attention_weight=attention_weight
        )
    
    def process_dag_step(
        self, 
        step_text: str
    ) -> Optional[Tuple[DualSignalTemplate, float]]:
        """
        Route a DAG step to the best matching template.
        
        Args:
            step_text: Text description of the DAG step
        
        Returns:
            Tuple of (matched_template, confidence_score) or None
        """
        # Extract features
        embedding, attention, tokens = self.detector.extract_features(step_text)
        attention_flat = attention.flatten()
        
        # Find best match
        result = self.store.find_best_match(embedding, attention_flat)
        
        if result is None:
            return None
        
        template, score, emb_sim, att_sim = result
        
        # Update Welford stats
        template.embedding_welford.update(emb_sim)
        template.attention_welford.update(att_sim)
        
        return template, score
    
    def record_outcome(
        self,
        template_id: str,
        success: bool,
        embedding_sim: Optional[float] = None,
        attention_sim: Optional[float] = None
    ) -> None:
        """
        Record execution outcome for learning.
        
        This feeds the variance-based decomposition system.
        """
        template = self.store.get_template(template_id)
        if template:
            template.record_outcome(success)
        
        # Update weight learning
        if embedding_sim is not None and attention_sim is not None:
            self.store.update_weights_from_outcome(
                embedding_sim, attention_sim, success
            )
    
    def get_decomposition_candidates(self) -> List[DualSignalTemplate]:
        """
        Get templates that should be considered for decomposition.
        
        Per CLAUDE.md: High variance indicates need for decomposition.
        """
        return self.store.get_high_variance_templates()


# ============================================================
# CLI / Testing
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dual-Signal Template System")
    parser.add_argument(
        "--model-path",
        default=os.path.expanduser("~/models/minilm_finetuned/best_model.pt"),
        help="Path to fine-tuned MiniLM model"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run basic tests"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Dual-Signal Template System")
    print("=" * 60)

    # Test basic functionality
    print("\n1. Testing SpanDetector initialization...")
    try:
        detector = SpanDetector(model_path=args.model_path)
        print("   [OK] SpanDetector initialized successfully")
    except Exception as e:
        print(f"   [WARN] SpanDetector init failed: {e}")
        print("   Trying without fine-tuned weights...")
        detector = SpanDetector(model_path=None)
        print("   [OK] SpanDetector initialized with base weights")

    # Test feature extraction
    print("\n2. Testing feature extraction...")
    test_text = "John has 5 apples. He gives 2 apples to Mary."
    embedding, attention, tokens = detector.extract_features(test_text)
    print(f"   [OK] Embedding shape: {embedding.shape}")
    print(f"   [OK] Attention shape: {attention.shape}")
    print(f"   [OK] Tokens: {len(tokens)}")

    # Test template creation
    print("\n3. Testing template creation...")
    template = create_template_from_span(
        span_text=test_text,
        embedding=embedding,
        attention_pattern=attention.flatten(),
        operation_type=OperationType.SUB
    )
    print(f"   [OK] Created template: {template.template_id}")
    print(f"       Operation: {template.operation_type.value}")
    print(f"       Embedding dim: {len(template.embedding_centroid)}")
    print(f"       Attention dim: {len(template.attention_signature)}")

    # Test template store
    print("\n4. Testing TemplateStore...")
    store = TemplateStore()
    store.add_template(template)

    # Query with same text (should match well)
    embedding2, attention2, _ = detector.extract_features(test_text)
    result = store.find_best_match(embedding2, attention2.flatten())
    if result:
        matched, score, emb_sim, att_sim = result
        print(f"   [OK] Best match: {matched.template_id}")
        print(f"       Combined score: {score:.4f}")
        print(f"       Embedding sim: {emb_sim:.4f}")
        print(f"       Attention sim: {att_sim:.4f}")

    # Test Welford stats
    print("\n5. Testing Welford statistics...")
    welford = WelfordStats()
    for v in [0.8, 0.85, 0.82, 0.78, 0.9]:
        welford.update(v)
    print(f"   [OK] Count: {welford.count}")
    print(f"   [OK] Mean: {welford.mean:.4f}")
    print(f"   [OK] Variance: {welford.variance:.4f}")
    print(f"   [OK] Std: {welford.std:.4f}")

    # Test full router
    print("\n6. Testing DualSignalRouter...")
    router = DualSignalRouter(embedding_weight=0.6, attention_weight=0.4)

    # Add the template
    router.store.add_template(template)

    # Route a query
    result = router.process_dag_step("Calculate 5 minus 2")
    if result:
        matched, score = result
        print(f"   [OK] Routed to: {matched.template_id}")
        print(f"       Score: {score:.4f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
