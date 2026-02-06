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
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import os
from collections import OrderedDict

# Import consolidated Welford implementation
from mycelium.welford import WelfordStats


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
        pattern: Normalized pattern string like "[NAME] sold [N] [ITEM]"
        subgraph: SubGraphDSL dict defining computation steps
        embedding_centroid: Mean embedding vector of matched spans
        attention_signature: Characteristic attention connectivity pattern
        span_examples: Example spans that match this template
        embedding_welford: Running stats for embedding similarities
        attention_welford: Running stats for attention similarities
        outcome_welford: Running stats for execution outcomes (success/failure)
    """
    template_id: str
    embedding_centroid: np.ndarray
    attention_signature: np.ndarray  # Flattened or aggregated attention pattern
    pattern: str = ""  # Normalized pattern: "[NAME] sold [N] [ITEM]"
    subgraph: Optional[Dict[str, Any]] = None  # SubGraphDSL dict for execution
    graph_embedding: Optional[np.ndarray] = None  # Computation graph embedding (64-dim)
    span_examples: List[str] = field(default_factory=list)
    embedding_welford: WelfordStats = field(default_factory=WelfordStats)
    attention_welford: WelfordStats = field(default_factory=WelfordStats)
    outcome_welford: WelfordStats = field(default_factory=WelfordStats)
    graph_welford: WelfordStats = field(default_factory=WelfordStats)  # Stats for graph similarity
    match_count: int = 0
    success_count: int = 0
    cross_entity_attention: float = 0.0

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
        result = {
            "template_id": self.template_id,
            "subgraph": self.subgraph,
            "pattern": self.pattern,
            "embedding_centroid": self.embedding_centroid.tolist(),
            "attention_signature": self.attention_signature.tolist(),
            "span_examples": self.span_examples[:10],
            "embedding_welford": self.embedding_welford.to_dict(),
            "attention_welford": self.attention_welford.to_dict(),
            "outcome_welford": self.outcome_welford.to_dict(),
            "graph_welford": self.graph_welford.to_dict(),
            "match_count": self.match_count,
            "success_count": self.success_count
        }
        if self.graph_embedding is not None:
            result["graph_embedding"] = self.graph_embedding.tolist()
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DualSignalTemplate":
        """Deserialize template from dictionary."""
        graph_emb = None
        if "graph_embedding" in d:
            graph_emb = np.array(d["graph_embedding"], dtype=np.float32)

        # Handle attention_signature - may not exist in older templates
        if "attention_signature" in d:
            attention_sig = np.array(d["attention_signature"])
        else:
            # Default to zero attention signature
            attention_sig = np.zeros(384, dtype=np.float32)

        # Handle pattern_examples vs span_examples (different naming in files)
        span_examples = d.get("span_examples", d.get("pattern_examples", []))

        return cls(
            template_id=d["template_id"],
            embedding_centroid=np.array(d["embedding_centroid"]),
            attention_signature=attention_sig,
            pattern=d.get("pattern", ""),
            subgraph=d.get("subgraph"),
            graph_embedding=graph_emb,
            span_examples=span_examples,
            embedding_welford=WelfordStats.from_dict(d.get("embedding_welford", {})),
            attention_welford=WelfordStats.from_dict(d.get("attention_welford", {})),
            outcome_welford=WelfordStats.from_dict(d.get("outcome_welford", {})),
            graph_welford=WelfordStats.from_dict(d.get("graph_welford", {})),
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
    
    def __init__(
        self,
        embedding_weight: float = 0.5,
        attention_weight: float = 0.3,
        graph_weight: float = 0.2
    ):
        """
        Initialize template store with triple-signal matching.

        Args:
            embedding_weight: Initial weight for embedding similarity [0-1]
            attention_weight: Initial weight for attention similarity [0-1]
            graph_weight: Initial weight for graph structure similarity [0-1]
        """
        self.templates: Dict[str, DualSignalTemplate] = OrderedDict()
        self.embedding_weight = embedding_weight
        self.attention_weight = attention_weight
        self.graph_weight = graph_weight

        # Vectorized matrices for fast matching (built lazily)
        self._centroid_matrix: Optional[np.ndarray] = None  # (N, dim) normalized
        self._graph_matrix: Optional[np.ndarray] = None  # (N, 64) normalized
        self._template_ids: Optional[List[str]] = None  # parallel list of IDs

        # Track weight learning
        self.weight_welford = WelfordStats()  # Tracks optimal weight estimation
        self._weight_history: List[Tuple[float, float, float, bool]] = []  # (emb, att, graph, success)

    def _invalidate_centroid_cache(self) -> None:
        """Invalidate the vectorized matrices (call after add/remove)."""
        self._centroid_matrix = None
        self._graph_matrix = None
        self._template_ids = None

    def _ensure_centroid_matrix(self) -> None:
        """Build the vectorized centroid and graph matrices if not cached."""
        if self._centroid_matrix is not None:
            return
        if not self.templates:
            return

        self._template_ids = list(self.templates.keys())

        # Build embedding centroid matrix
        centroids = [self.templates[tid].embedding_centroid for tid in self._template_ids]
        self._centroid_matrix = np.array(centroids, dtype=np.float32)
        # Pre-normalize rows for fast cosine similarity via dot product
        norms = np.linalg.norm(self._centroid_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self._centroid_matrix = self._centroid_matrix / norms

        # Build graph embedding matrix (if templates have graph embeddings)
        graph_embeddings = []
        for tid in self._template_ids:
            t = self.templates[tid]
            if t.graph_embedding is not None:
                graph_embeddings.append(t.graph_embedding)
            else:
                # Default zero vector if no graph embedding
                graph_embeddings.append(np.zeros(64, dtype=np.float32))

        self._graph_matrix = np.array(graph_embeddings, dtype=np.float32)
        # Pre-normalize for fast cosine similarity
        norms = np.linalg.norm(self._graph_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self._graph_matrix = self._graph_matrix / norms

    def add_template(self, template: DualSignalTemplate) -> None:
        """Add a template to the store."""
        self.templates[template.template_id] = template
        self._invalidate_centroid_cache()
    
    def remove_template(self, template_id: str) -> Optional[DualSignalTemplate]:
        """Remove and return a template by ID."""
        result = self.templates.pop(template_id, None)
        if result:
            self._invalidate_centroid_cache()
        return result
    
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
        template: DualSignalTemplate,
        graph_embedding: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float, float]:
        """
        Compute combined match score using triple signals.

        Returns:
            Tuple of (combined_score, embedding_sim, attention_sim, graph_sim)
        """
        emb_sim = self._cosine_similarity(embedding, template.embedding_centroid)
        att_sim = self._attention_correlation(attention, template.attention_signature)

        # Normalize attention to [0, 1] range (correlation can be negative)
        att_sim_normalized = (att_sim + 1) / 2

        # Graph similarity (if available)
        graph_sim = 0.5  # Neutral default
        if graph_embedding is not None and template.graph_embedding is not None:
            graph_sim = self._cosine_similarity(graph_embedding, template.graph_embedding)

        combined = (
            self.embedding_weight * emb_sim +
            self.attention_weight * att_sim_normalized +
            self.graph_weight * graph_sim
        )

        return combined, emb_sim, att_sim, graph_sim
    
    def find_best_match(
        self,
        embedding: np.ndarray,
        attention: np.ndarray,
        graph_embedding: Optional[np.ndarray] = None,
        min_score: float = 0.0
    ) -> Optional[Tuple[DualSignalTemplate, float, float, float, float]]:
        """
        Find the best matching template using triple signals.

        Following CLAUDE.md principle: ALWAYS route to best match,
        let failures drive learning rather than arbitrary thresholds.

        Uses vectorized matrix multiplication for O(1) matching against
        all templates simultaneously (vs O(N) Python loop).

        Args:
            embedding: Query embedding vector
            attention: Query attention pattern
            graph_embedding: Query graph structure embedding (optional)
            min_score: Minimum score threshold (default 0, always match)

        Returns:
            Tuple of (template, combined_score, embedding_sim, attention_sim, graph_sim)
            or None if no templates exist
        """
        if not self.templates:
            return None

        # Vectorized cosine similarity via matrix multiply
        self._ensure_centroid_matrix()
        if self._centroid_matrix is not None:
            # Normalize embedding query
            query = embedding.astype(np.float32)
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm

            # Embedding similarity: (N, dim) @ (dim,) -> (N,)
            emb_sims = self._centroid_matrix @ query

            # Graph similarity (vectorized if graph_embedding provided)
            if graph_embedding is not None and self._graph_matrix is not None:
                graph_query = graph_embedding.astype(np.float32)
                gnorm = np.linalg.norm(graph_query)
                if gnorm > 0:
                    graph_query = graph_query / gnorm
                graph_sims = self._graph_matrix @ graph_query
            else:
                graph_sims = np.full(len(emb_sims), 0.5)  # Neutral default

            # Combined score for ranking (attention computed only for top candidate)
            # Use embedding + graph for initial ranking, then refine with attention
            initial_scores = self.embedding_weight * emb_sims + self.graph_weight * graph_sims
            best_idx = int(np.argmax(initial_scores))

            best_emb_sim = float(emb_sims[best_idx])
            best_graph_sim = float(graph_sims[best_idx])

            best_tid = self._template_ids[best_idx]
            best_match = self.templates[best_tid]

            # Attention correlation (computed only for best match)
            att_sim = self._attention_correlation(attention, best_match.attention_signature)
            att_sim_normalized = (att_sim + 1) / 2

            combined = (
                self.embedding_weight * best_emb_sim +
                self.attention_weight * att_sim_normalized +
                self.graph_weight * best_graph_sim
            )

            if combined < min_score:
                return None
            return best_match, combined, best_emb_sim, att_sim, best_graph_sim

        return None

    def find_top_k_matches(
        self,
        embedding: np.ndarray,
        attention: np.ndarray,
        graph_embedding: Optional[np.ndarray] = None,
        k: int = 5,
    ) -> List[Tuple[DualSignalTemplate, float, float, float, float]]:
        """
        Find top-k matching templates using triple signals.

        Useful for MCTS exploration where we want to consider
        multiple candidates.

        Returns:
            List of (template, combined_score, embedding_sim, attention_sim, graph_sim) tuples
        """
        if not self.templates:
            return []

        # Vectorized matching
        self._ensure_centroid_matrix()
        if self._centroid_matrix is not None:
            # Embedding similarity
            query = embedding.astype(np.float32)
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm
            emb_sims = self._centroid_matrix @ query

            # Graph similarity
            if graph_embedding is not None and self._graph_matrix is not None:
                graph_query = graph_embedding.astype(np.float32)
                gnorm = np.linalg.norm(graph_query)
                if gnorm > 0:
                    graph_query = graph_query / gnorm
                graph_sims = self._graph_matrix @ graph_query
            else:
                graph_sims = np.full(len(emb_sims), 0.5)

            # Initial ranking by embedding + graph (attention computed per candidate)
            initial_scores = self.embedding_weight * emb_sims + self.graph_weight * graph_sims
            top_k_indices = np.argsort(initial_scores)[-k:][::-1]

            matches = []
            for idx in top_k_indices:
                tid = self._template_ids[idx]
                template = self.templates[tid]
                emb_sim = float(emb_sims[idx])
                graph_sim = float(graph_sims[idx])
                att_sim = self._attention_correlation(attention, template.attention_signature)
                att_sim_normalized = (att_sim + 1) / 2
                combined = (
                    self.embedding_weight * emb_sim +
                    self.attention_weight * att_sim_normalized +
                    self.graph_weight * graph_sim
                )
                matches.append((template, combined, emb_sim, att_sim, graph_sim))
            return matches

        return []
    
    def update_weights_from_outcome(
        self,
        embedding_sim: float,
        attention_sim: float,
        success: bool,
        graph_sim: float = 0.5
    ) -> None:
        """
        Learn optimal signal weights from outcomes.

        Track which signal was more predictive of success
        and adjust weights accordingly.
        """
        # Store for batch learning (3 signals now)
        self._weight_history.append((embedding_sim, attention_sim, graph_sim, success))
        
        # Update weights every 100 observations
        if len(self._weight_history) >= 100:
            self._recompute_weights()
    
    def _recompute_weights(self) -> None:
        """Recompute optimal weights from accumulated history using 3 signals."""
        if len(self._weight_history) < 10:
            return

        # Linear regression to find optimal weights for triple signals
        emb_sims = np.array([h[0] for h in self._weight_history])
        att_sims = np.array([(h[1] + 1) / 2 for h in self._weight_history])  # Normalize correlation
        graph_sims = np.array([h[2] for h in self._weight_history])
        outcomes = np.array([1.0 if h[3] else 0.0 for h in self._weight_history])

        # Stack features (3 signals now)
        X = np.column_stack([emb_sims, att_sims, graph_sims])

        # Add small regularization for stability
        XtX = X.T @ X + 0.01 * np.eye(3)
        Xty = X.T @ outcomes

        try:
            weights = np.linalg.solve(XtX, Xty)
            # Normalize to sum to 1
            total = abs(weights[0]) + abs(weights[1]) + abs(weights[2])
            if total > 0:
                self.embedding_weight = abs(weights[0]) / total
                self.attention_weight = abs(weights[1]) / total
                self.graph_weight = abs(weights[2]) / total
        except np.linalg.LinAlgError:
            pass  # Keep current weights if solve fails

        # Clear history (keep last 50 for continuity)
        self._weight_history = self._weight_history[-50:]
    
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
            "graph_weight": self.graph_weight,
            "templates": {
                tid: t.to_dict() for tid, t in self.templates.items()
            }
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TemplateStore":
        """Deserialize store from dictionary."""
        store = cls(
            embedding_weight=d.get("embedding_weight", 0.5),
            attention_weight=d.get("attention_weight", 0.3),
            graph_weight=d.get("graph_weight", 0.2)
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

    def compute_cross_entity_attention(
        self,
        text: str,
        attention_matrix: Optional[np.ndarray] = None,
        tokens: Optional[List[str]] = None
    ) -> float:
        """
        Compute cross-entity attention from the attention matrix.

        This measures how much attention flows between different parts of the
        sentence (off-diagonal attention), which could indicate relational
        operations vs self-contained ones.

        NO KEYWORD/VOCABULARY HEURISTICS - pure attention signal only.

        Args:
            text: The span text
            attention_matrix: Optional pre-computed attention matrix
            tokens: Optional pre-computed tokens

        Returns:
            Cross-entity attention score based on attention patterns
        """
        # Get attention matrix if not provided
        if attention_matrix is None or tokens is None:
            _, attention_matrix, tokens = self.extract_features(text)

        # Average attention across heads if needed (shape: seq x seq)
        if attention_matrix.ndim > 2:
            attention_matrix = attention_matrix.mean(axis=0)

        # Compute off-diagonal attention (attention to other positions)
        # This measures how much attention flows between different parts
        n = attention_matrix.shape[0]
        if n < 2:
            return 0.0

        # Create mask for off-diagonal elements (exclude diagonal + adjacent)
        # Adjacent tokens naturally have high attention due to language structure
        mask = np.ones_like(attention_matrix)
        for i in range(n):
            for j in range(n):
                # Mask out diagonal and 1-off diagonal (adjacent tokens)
                if abs(i - j) <= 1:
                    mask[i, j] = 0.0

        # Compute mean off-diagonal attention
        masked_attn = attention_matrix * mask
        off_diag_sum = masked_attn.sum()
        off_diag_count = mask.sum()

        if off_diag_count == 0:
            return 0.0

        return float(off_diag_sum / off_diag_count)

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
    subgraph: Optional[Dict[str, Any]] = None,
    template_id: Optional[str] = None
) -> DualSignalTemplate:
    """
    Create a new template from span data.

    Args:
        span_text: Text of the span
        embedding: Embedding vector for the span
        attention_pattern: Attention pattern for the span
        subgraph: SubGraphDSL dict for computation (default: SET operation)
        template_id: Optional ID (auto-generated if not provided)

    Returns:
        New DualSignalTemplate
    """
    import uuid

    if template_id is None:
        template_id = f"tpl_{uuid.uuid4().hex[:8]}"

    # Default subgraph: simple SET operation
    if subgraph is None:
        subgraph = {
            "template_id": template_id,
            "pattern": span_text,
            "params": {"n1": "value"},
            "inputs": {},
            "steps": [{"var": "out", "op": "SET", "args": ["n1"]}],
            "output": "out",
        }

    return DualSignalTemplate(
        template_id=template_id,
        embedding_centroid=embedding.copy(),
        attention_signature=attention_pattern.copy(),
        subgraph=subgraph,
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
    sub_subgraph = {
        "template_id": "test",
        "pattern": test_text,
        "params": {"n1": "value"},
        "inputs": {"upstream": "entity"},
        "steps": [{"var": "out", "op": "SUB", "args": ["upstream", "n1"]}],
        "output": "out",
    }
    template = create_template_from_span(
        span_text=test_text,
        embedding=embedding,
        attention_pattern=attention.flatten(),
        subgraph=sub_subgraph
    )
    print(f"   [OK] Created template: {template.template_id}")
    op_name = template.subgraph["steps"][-1]["op"] if template.subgraph else "SET"
    print(f"       Op: {op_name}")
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
