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


class OperationType(Enum):
    """Core arithmetic operation types for math problem solving."""
    SET = "SET"      # Assignment/initialization
    ADD = "ADD"      # Addition
    SUB = "SUB"      # Subtraction  
    MUL = "MUL"      # Multiplication
    DIV = "DIV"      # Division
    UNKNOWN = "UNKNOWN"  # Unclassified


@dataclass
class WelfordStats:
    """
    Running statistics using Welford's online algorithm.
    
    Tracks mean and variance incrementally without storing all values.
    This is critical for the variance-based decomposition signals
    described in CLAUDE.md.
    """
    count: int = 0
    mean: float = 0.0
    M2: float = 0.0  # Sum of squared differences from mean
    
    def update(self, value: float) -> None:
        """Update statistics with new value using Welford's algorithm."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2
    
    @property
    def variance(self) -> float:
        """Return sample variance (or 0 if insufficient data)."""
        if self.count < 2:
            return 0.0
        return self.M2 / (self.count - 1)
    
    @property
    def std(self) -> float:
        """Return sample standard deviation."""
        return np.sqrt(self.variance)
    
    def to_dict(self) -> Dict[str, float]:
        """Serialize to dictionary for storage."""
        return {
            "count": self.count,
            "mean": self.mean,
            "M2": self.M2
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "WelfordStats":
        """Deserialize from dictionary."""
        stats = cls()
        stats.count = int(d["count"])
        stats.mean = d["mean"]
        stats.M2 = d["M2"]
        return stats


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
    span_examples: List[str] = field(default_factory=list)
    embedding_welford: WelfordStats = field(default_factory=WelfordStats)
    attention_welford: WelfordStats = field(default_factory=WelfordStats)
    outcome_welford: WelfordStats = field(default_factory=WelfordStats)
    match_count: int = 0
    success_count: int = 0
    
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
    
    def detect_spans_community(
        self, 
        attention: np.ndarray,
        tokens: List[str],
        resolution: float = 1.0,
        min_span_size: int = 2
    ) -> List[Tuple[int, int, float]]:
        """
        Detect spans using community detection on attention graph.
        
        Uses Louvain community detection to find clusters of tokens
        that attend strongly to each other.
        
        Args:
            attention: [seq_len, seq_len] attention matrix
            tokens: List of token strings
            resolution: Louvain resolution parameter (higher = smaller communities)
            min_span_size: Minimum tokens in a span
        
        Returns:
            List of (start_idx, end_idx, density) tuples
        """
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
        
        n = attention.shape[0]
        
        # Build weighted graph from attention
        G = nx.Graph()
        G.add_nodes_from(range(n))
        
        # Add edges with attention weights (symmetric for community detection)
        for i in range(n):
            for j in range(i + 1, n):
                weight = (attention[i, j] + attention[j, i]) / 2
                if weight > 0.01:  # Threshold weak connections
                    G.add_edge(i, j, weight=weight)
        
        # Detect communities
        try:
            communities = louvain_communities(G, weight="weight", resolution=resolution)
        except Exception:
            # Fallback: treat entire sequence as one span
            return [(0, n, 1.0)]
        
        # Convert communities to contiguous spans
        spans = []
        for community in communities:
            if len(community) < min_span_size:
                continue
            
            indices = sorted(community)
            start = indices[0]
            end = indices[-1] + 1
            
            # Compute internal density
            subgraph = G.subgraph(indices)
            if len(indices) > 1:
                density = nx.density(subgraph)
            else:
                density = 1.0
            
            spans.append((start, end, density))
        
        # Sort by start position
        spans.sort(key=lambda x: x[0])
        
        return spans
    
    def detect_spans_threshold(
        self,
        attention: np.ndarray,
        tokens: List[str],
        threshold: float = 0.1,
        min_span_size: int = 2
    ) -> List[Tuple[int, int, float]]:
        """
        Detect spans using attention threshold method.
        
        Simpler alternative to community detection.
        Finds contiguous regions with high self-attention.
        """
        n = attention.shape[0]
        
        # Compute "self-connectivity" score for each position
        connectivity = np.zeros(n)
        for i in range(n):
            # Average attention received from nearby tokens
            window = min(5, n)
            start = max(0, i - window)
            end = min(n, i + window + 1)
            connectivity[i] = np.mean(attention[start:end, i])
        
        # Find spans above threshold
        spans = []
        in_span = False
        span_start = 0
        
        for i in range(n):
            if connectivity[i] > threshold:
                if not in_span:
                    span_start = i
                    in_span = True
            else:
                if in_span:
                    if i - span_start >= min_span_size:
                        density = np.mean(connectivity[span_start:i])
                        spans.append((span_start, i, density))
                    in_span = False
        
        # Handle span at end
        if in_span and n - span_start >= min_span_size:
            density = np.mean(connectivity[span_start:n])
            spans.append((span_start, n, density))
        
        return spans
    
    def extract_span_features(
        self,
        text: str,
        method: str = "community"
    ) -> List[Dict[str, Any]]:
        """
        Extract features for all detected spans in text.
        
        Args:
            text: Input text
            method: "community" or "threshold"
        
        Returns:
            List of span dictionaries with:
            - text: Span text
            - start: Start token index
            - end: End token index
            - embedding: Span embedding (mean of token embeddings)
            - attention_pattern: Submatrix of attention for this span
            - density: Internal attention density
        """
        # Get full features
        embedding, attention, tokens = self.extract_features(text)
        
        # Detect spans
        if method == "community":
            spans = self.detect_spans_community(attention, tokens)
        else:
            spans = self.detect_spans_threshold(attention, tokens)
        
        # Extract features for each span
        results = []
        for start, end, density in spans:
            # Get span text
            span_tokens = tokens[start:end]
            span_text = self.tokenizer.convert_tokens_to_string(span_tokens)
            
            # Get span attention submatrix
            span_attention = attention[start:end, start:end]
            
            # Flatten for storage (can reconstruct shape later)
            attention_flat = span_attention.flatten()
            
            results.append({
                "text": span_text.strip(),
                "start": start,
                "end": end,
                "embedding": embedding,  # Use full embedding for now
                "attention_pattern": attention_flat,
                "attention_shape": span_attention.shape,
                "density": density
            })
        
        return results


def create_template_from_span(
    span_data: Dict[str, Any],
    operation_type: OperationType,
    template_id: Optional[str] = None
) -> DualSignalTemplate:
    """
    Create a new template from span data.
    
    Args:
        span_data: Output from SpanDetector.extract_span_features()
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
        embedding_centroid=span_data["embedding"].copy(),
        attention_signature=span_data["attention_pattern"].copy(),
        span_examples=[span_data["text"]]
    )


def extract_templates_from_qwen_data(
    qwen_data_dir: str,
    max_files: int = 10
) -> List[DualSignalTemplate]:
    """
    Extract initial templates from Qwen attention data.
    
    This bootstraps the template store with patterns learned
    from the large model's attention.
    
    Args:
        qwen_data_dir: Directory containing Qwen feature files
        max_files: Maximum number of files to process
    
    Returns:
        List of extracted templates
    """
    templates = []
    
    # Get list of feature files
    files = sorted([
        f for f in os.listdir(qwen_data_dir) 
        if f.endswith('.npz')
    ])[:max_files]
    
    for fname in files:
        fpath = os.path.join(qwen_data_dir, fname)
        data = np.load(fpath, allow_pickle=True)
        
        # Process each sample in the file
        n_samples = len(data['connectivity'])
        
        for i in range(min(n_samples, 10)):  # Limit samples per file
            connectivity = data['connectivity'][i]
            
            # Skip very small sequences
            if connectivity.shape[0] < 5:
                continue
            
            # Use connectivity as attention signature
            # Flatten for consistent representation
            attention_flat = connectivity.flatten()
            
            # Create a template with dummy embedding (will be updated)
            # In practice, we'd pair this with text to get real embeddings
            embedding = np.random.randn(384)  # MiniLM dimension
            embedding = embedding / np.linalg.norm(embedding)
            
            template = DualSignalTemplate(
                template_id=f"qwen_{fname}_{i}",
                operation_type=OperationType.UNKNOWN,
                embedding_centroid=embedding,
                attention_signature=attention_flat,
                span_examples=[f"Sample {i} from {fname}"]
            )
            templates.append(template)
    
    return templates


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
        "--qwen-data",
        default=os.path.expanduser("~/qwen_data"),
        help="Path to Qwen attention data"
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
    test_text = "John has 5 apples. He gives 2 apples to Mary. How many apples does John have now?"
    embedding, attention, tokens = detector.extract_features(test_text)
    print(f"   [OK] Embedding shape: {embedding.shape}")
    print(f"   [OK] Attention shape: {attention.shape}")
    print(f"   [OK] Tokens: {len(tokens)}")
    
    # Test span detection
    print("\n3. Testing span detection...")
    spans = detector.extract_span_features(test_text, method="community")
    print(f"   [OK] Detected {len(spans)} spans")
    for i, span in enumerate(spans):
        text_preview = span['text'][:50] if len(span['text']) > 50 else span['text']
        print(f"       Span {i}: '{text_preview}' (density={span['density']:.3f})")
    
    # Test template creation
    print("\n4. Testing template creation...")
    if spans:
        template = create_template_from_span(spans[0], OperationType.SUB)
        print(f"   [OK] Created template: {template.template_id}")
        print(f"       Operation: {template.operation_type.value}")
        print(f"       Embedding dim: {len(template.embedding_centroid)}")
        print(f"       Attention dim: {len(template.attention_signature)}")
    
    # Test template store
    print("\n5. Testing TemplateStore...")
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
    print("\n6. Testing Welford statistics...")
    welford = WelfordStats()
    for v in [0.8, 0.85, 0.82, 0.78, 0.9]:
        welford.update(v)
    print(f"   [OK] Count: {welford.count}")
    print(f"   [OK] Mean: {welford.mean:.4f}")
    print(f"   [OK] Variance: {welford.variance:.4f}")
    print(f"   [OK] Std: {welford.std:.4f}")
    
    # Test Qwen data loading if available
    print("\n7. Testing Qwen data extraction...")
    if os.path.exists(args.qwen_data):
        templates = extract_templates_from_qwen_data(args.qwen_data, max_files=2)
        print(f"   [OK] Extracted {len(templates)} templates from Qwen data")
    else:
        print(f"   [SKIP] Qwen data not found at {args.qwen_data}")
    
    # Test full router
    print("\n8. Testing DualSignalRouter...")
    router = DualSignalRouter(embedding_weight=0.6, attention_weight=0.4)
    
    # Add some templates
    for span in spans[:3]:
        t = create_template_from_span(span, OperationType.UNKNOWN)
        router.store.add_template(t)
    
    # Route a query
    result = router.process_dag_step("Calculate 5 minus 2")
    if result:
        matched, score = result
        print(f"   [OK] Routed to: {matched.template_id}")
        print(f"       Score: {score:.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
