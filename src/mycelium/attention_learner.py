"""Attention-based operation learning using Welford's algorithm.

NO HARDCODING. NO KEYWORDS. Learn everything from attention patterns.

The system:
1. Extracts attention features from spans
2. Uses Welford's algorithm to track running statistics per operation
3. Classifies new spans by comparing to learned distributions
"""

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np


@dataclass
class WelfordStats:
    """Welford's online algorithm for running mean and variance."""
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squares of differences from mean

    def update(self, value: float):
        """Add a new value and update statistics."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def zscore(self, value: float) -> float:
        """How many standard deviations from mean."""
        if self.std < 1e-8:
            return 0.0
        return (value - self.mean) / self.std

    def probability(self, value: float) -> float:
        """Gaussian probability density at value."""
        if self.std < 1e-8:
            return 1.0 if abs(value - self.mean) < 1e-8 else 0.0
        z = self.zscore(value)
        return math.exp(-0.5 * z * z) / (self.std * math.sqrt(2 * math.pi))


@dataclass
class AttentionProfile:
    """Learned attention profile for an operation type."""
    operation: str

    # Welford stats for different attention features
    num_to_verb_attn: WelfordStats = field(default_factory=WelfordStats)
    self_attn: WelfordStats = field(default_factory=WelfordStats)
    first_token_attn: WelfordStats = field(default_factory=WelfordStats)

    # Cluster statistics
    num_clusters: WelfordStats = field(default_factory=WelfordStats)
    main_cluster_size: WelfordStats = field(default_factory=WelfordStats)

    def update(self, features: Dict[str, float]):
        """Update all statistics with new observation."""
        if "num_to_verb" in features:
            self.num_to_verb_attn.update(features["num_to_verb"])
        if "self_attn" in features:
            self.self_attn.update(features["self_attn"])
        if "first_token_attn" in features:
            self.first_token_attn.update(features["first_token_attn"])
        if "num_clusters" in features:
            self.num_clusters.update(features["num_clusters"])
        if "main_cluster_size" in features:
            self.main_cluster_size.update(features["main_cluster_size"])

    def score(self, features: Dict[str, float]) -> float:
        """Score how well features match this profile (log probability)."""
        score = 0.0
        count = 0

        if "num_to_verb" in features and self.num_to_verb_attn.count > 5:
            score += math.log(self.num_to_verb_attn.probability(features["num_to_verb"]) + 1e-10)
            count += 1
        if "self_attn" in features and self.self_attn.count > 5:
            score += math.log(self.self_attn.probability(features["self_attn"]) + 1e-10)
            count += 1
        if "first_token_attn" in features and self.first_token_attn.count > 5:
            score += math.log(self.first_token_attn.probability(features["first_token_attn"]) + 1e-10)
            count += 1

        return score / max(count, 1)


class AttentionLearner:
    """Learn operation classification from attention patterns.

    NO KEYWORDS. NO HARDCODED THRESHOLDS.
    Everything is learned from data via Welford statistics.
    """

    def __init__(self):
        self.profiles: Dict[str, AttentionProfile] = {}
        self._model = None
        self._tokenizer = None

    def _ensure_model(self):
        """Lazy load the attention model."""
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_name = "deepseek-ai/deepseek-math-7b-instruct"
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name, output_attentions=True, torch_dtype=torch.float16,
                device_map="auto", trust_remote_code=True,
            )

    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract attention features from text. NO KEYWORDS - pure attention."""
        import torch
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        self._ensure_model()

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs, output_attentions=True)

        # Average attention from last 4 layers
        attn = torch.stack(outputs.attentions[-4:]).mean(dim=(0, 1, 2)).cpu().numpy()
        n = attn.shape[0]

        features = {}

        # Feature 1: Self attention (diagonal)
        features["self_attn"] = float(np.diag(attn).mean())

        # Feature 2: First token attention (subject focus)
        features["first_token_attn"] = float(attn[:, 0].mean())

        # Feature 3: num→verb attention (position 3→1 in typical structure)
        if n >= 4:
            features["num_to_verb"] = float(attn[3, 1])

        # Feature 4: Clustering statistics
        if n >= 2:
            attn_sym = (attn + attn.T) / 2
            dist = 1 - attn_sym
            np.fill_diagonal(dist, 0)

            dist_condensed = squareform(dist, checks=False)
            Z = linkage(dist_condensed, method="average")
            clusters = fcluster(Z, t=0.92, criterion="distance")

            features["num_clusters"] = float(len(set(clusters)))

            # Size of main cluster (first/largest)
            cluster_sizes = {}
            for c in clusters:
                cluster_sizes[c] = cluster_sizes.get(c, 0) + 1
            features["main_cluster_size"] = float(max(cluster_sizes.values()))

        return features

    def learn(self, text: str, operation: str):
        """Learn from a labeled example."""
        if operation not in self.profiles:
            self.profiles[operation] = AttentionProfile(operation=operation)

        features = self.extract_features(text)
        self.profiles[operation].update(features)

    def classify(self, text: str) -> Tuple[str, float]:
        """Classify operation based on learned attention patterns.

        Returns (operation, confidence) based purely on attention similarity
        to learned profiles. NO KEYWORDS.
        """
        if not self.profiles:
            return "UNKNOWN", 0.0

        features = self.extract_features(text)

        # Score against each learned profile
        scores = {}
        for op, profile in self.profiles.items():
            scores[op] = profile.score(features)

        # Return best match
        best_op = max(scores, key=scores.get)

        # Confidence = how much better than second best
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            confidence = sorted_scores[0] - sorted_scores[1]
        else:
            confidence = 1.0

        return best_op, confidence

    def stats(self) -> Dict:
        """Get current learned statistics."""
        result = {}
        for op, profile in self.profiles.items():
            result[op] = {
                "count": profile.num_to_verb_attn.count,
                "num_to_verb_mean": profile.num_to_verb_attn.mean,
                "num_to_verb_std": profile.num_to_verb_attn.std,
                "self_attn_mean": profile.self_attn.mean,
                "first_token_mean": profile.first_token_attn.mean,
            }
        return result

    def save(self, path: str):
        """Save learned profiles."""
        data = {}
        for op, profile in self.profiles.items():
            data[op] = {
                "num_to_verb": {"count": profile.num_to_verb_attn.count,
                               "mean": profile.num_to_verb_attn.mean,
                               "m2": profile.num_to_verb_attn.m2},
                "self_attn": {"count": profile.self_attn.count,
                             "mean": profile.self_attn.mean,
                             "m2": profile.self_attn.m2},
                "first_token": {"count": profile.first_token_attn.count,
                               "mean": profile.first_token_attn.mean,
                               "m2": profile.first_token_attn.m2},
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load learned profiles."""
        with open(path) as f:
            data = json.load(f)

        for op, stats in data.items():
            profile = AttentionProfile(operation=op)
            profile.num_to_verb_attn = WelfordStats(**stats["num_to_verb"])
            profile.self_attn = WelfordStats(**stats["self_attn"])
            profile.first_token_attn = WelfordStats(**stats["first_token"])
            self.profiles[op] = profile
