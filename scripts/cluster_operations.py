#!/usr/bin/env python3
"""
Cluster-then-label system for operation classification.

This script:
1. Extracts spans from problems using SpanDetector
2. Clusters spans using k-means or DBSCAN
3. Generates a review file for manual labeling
4. Provides functions to apply labels

Usage:
    python cluster_operations.py --extract --cluster --review
    python cluster_operations.py --apply-labels labels.json
"""

import json
import os
import sys
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path.home()))

# Import SpanDetector from dual_signal_templates
from dual_signal_templates import SpanDetector, OperationType


@dataclass
class ExtractedSpan:
    """A single extracted span with its features."""
    problem_id: str
    span_text: str
    embedding: np.ndarray
    attention_pattern: np.ndarray
    attention_shape: Tuple[int, int]
    density: float
    start_idx: int
    end_idx: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "problem_id": self.problem_id,
            "span_text": self.span_text,
            "embedding": self.embedding.tolist(),
            "attention_pattern": self.attention_pattern.tolist(),
            "attention_shape": self.attention_shape,
            "density": self.density,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExtractedSpan":
        """Reconstruct from dictionary."""
        return cls(
            problem_id=d["problem_id"],
            span_text=d["span_text"],
            embedding=np.array(d["embedding"]),
            attention_pattern=np.array(d["attention_pattern"]),
            attention_shape=tuple(d["attention_shape"]),
            density=d["density"],
            start_idx=d["start_idx"],
            end_idx=d["end_idx"]
        )


class SpanExtractor:
    """Extract spans from problems using SpanDetector."""
    
    def __init__(
        self, 
        model_path: str = "~/models/minilm_finetuned/best_model.pt",
        qwen_data_dir: str = "~/qwen_data"
    ):
        self.model_path = Path(model_path).expanduser()
        self.qwen_data_dir = Path(qwen_data_dir).expanduser()
        self.detector: Optional[SpanDetector] = None
        
    def _init_detector(self):
        """Lazy initialization of SpanDetector."""
        if self.detector is None:
            print(f"Loading SpanDetector from {self.model_path}...")
            self.detector = SpanDetector(
                model_path=str(self.model_path),
                device="auto"
            )
            print(f"SpanDetector loaded on {self.detector.device}")
    
    def load_problems(self, max_files: int = 10, max_problems: int = 1000) -> List[Dict[str, str]]:
        """Load problems from metadata files."""
        problems = []
        
        # Find all metadata files
        meta_files = sorted(self.qwen_data_dir.glob("metadata_*.json"))
        print(f"Found {len(meta_files)} metadata files")
        
        for meta_file in meta_files[:max_files]:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            
            for entry in meta:
                problems.append({
                    "problem_id": entry["problem_id"],
                    "text": entry["problem_text"]
                })
                
                if len(problems) >= max_problems:
                    break
            
            if len(problems) >= max_problems:
                break
        
        print(f"Loaded {len(problems)} problems")
        return problems
    
    def extract_spans(
        self, 
        problems: List[Dict[str, str]],
        method: str = "community",
        batch_save_interval: int = 100
    ) -> List[ExtractedSpan]:
        """Extract spans from all problems."""
        self._init_detector()
        
        all_spans = []
        
        for i, problem in enumerate(tqdm(problems, desc="Extracting spans")):
            try:
                # Extract span features
                span_features = self.detector.extract_span_features(
                    problem["text"],
                    method=method
                )
                
                # Convert to ExtractedSpan objects
                for sf in span_features:
                    # Skip very short or empty spans
                    if len(sf["text"].strip()) < 3:
                        continue
                    
                    span = ExtractedSpan(
                        problem_id=problem["problem_id"],
                        span_text=sf["text"],
                        embedding=sf["embedding"],
                        attention_pattern=sf["attention_pattern"],
                        attention_shape=sf["attention_shape"],
                        density=sf["density"],
                        start_idx=sf["start"],
                        end_idx=sf["end"]
                    )
                    all_spans.append(span)
                    
            except Exception as e:
                print(f"Error processing {problem['problem_id']}: {e}")
                continue
        
        print(f"Extracted {len(all_spans)} spans from {len(problems)} problems")
        return all_spans
    
    def save_spans(self, spans: List[ExtractedSpan], output_path: str):
        """Save extracted spans to file."""
        output_path = Path(output_path).expanduser()
        
        # Convert to list of dicts
        span_dicts = [s.to_dict() for s in spans]
        
        # Save as JSON (embeddings converted to lists)
        with open(output_path, 'w') as f:
            json.dump(span_dicts, f)
        
        print(f"Saved {len(spans)} spans to {output_path}")
    
    def load_spans(self, input_path: str) -> List[ExtractedSpan]:
        """Load spans from file."""
        input_path = Path(input_path).expanduser()
        
        with open(input_path, 'r') as f:
            span_dicts = json.load(f)
        
        spans = [ExtractedSpan.from_dict(d) for d in span_dicts]
        print(f"Loaded {len(spans)} spans from {input_path}")
        return spans


class SpanClusterer:
    """Cluster spans by their embeddings."""
    
    def __init__(self, spans: List[ExtractedSpan]):
        self.spans = spans
        self.embeddings = np.array([s.embedding for s in spans])
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_centers: Optional[np.ndarray] = None
        
    def cluster_kmeans(self, n_clusters: int = 20) -> np.ndarray:
        """Cluster using k-means."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize
        
        # Normalize embeddings for cosine-like similarity
        embeddings_norm = normalize(self.embeddings)
        
        print(f"Running k-means with {n_clusters} clusters on {len(self.embeddings)} spans...")
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        
        self.cluster_labels = kmeans.fit_predict(embeddings_norm)
        self.cluster_centers = kmeans.cluster_centers_
        
        # Compute cluster sizes
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        print(f"Cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
        
        return self.cluster_labels
    
    def cluster_dbscan(self, eps: float = 0.3, min_samples: int = 5) -> np.ndarray:
        """Cluster using DBSCAN for automatic cluster count."""
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import normalize
        
        # Normalize embeddings
        embeddings_norm = normalize(self.embeddings)
        
        print(f"Running DBSCAN with eps={eps}, min_samples={min_samples}...")
        
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="cosine"
        )
        
        self.cluster_labels = dbscan.fit_predict(embeddings_norm)
        
        # Compute cluster statistics
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        n_clusters = len(unique) - (1 if -1 in unique else 0)  # Exclude noise
        n_noise = np.sum(self.cluster_labels == -1)
        
        print(f"Found {n_clusters} clusters, {n_noise} noise points")
        if n_clusters > 0:
            cluster_counts = counts[unique != -1]
            print(f"Cluster sizes: min={cluster_counts.min()}, max={cluster_counts.max()}, mean={cluster_counts.mean():.1f}")
        
        return self.cluster_labels
    
    def get_cluster_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics for each cluster."""
        if self.cluster_labels is None:
            raise ValueError("Must run clustering first")
        
        stats = {}
        unique_labels = np.unique(self.cluster_labels)
        
        for label in unique_labels:
            mask = self.cluster_labels == label
            cluster_spans = [s for s, m in zip(self.spans, mask) if m]
            
            stats[int(label)] = {
                "size": int(np.sum(mask)),
                "spans": cluster_spans
            }
        
        return stats
    
    def suggest_label(self, span_texts: List[str]) -> str:
        """Suggest an operation label based on keywords."""
        text_combined = " ".join(span_texts).lower()
        
        # Keyword patterns for each operation
        patterns = {
            "SET": ["has", "is", "was", "were", "start", "initial", "begin", "total"],
            "ADD": ["add", "plus", "more", "increase", "gain", "together", "sum", "additional", "bought", "got", "receive"],
            "SUB": ["sub", "minus", "less", "decrease", "lose", "spent", "gave", "sold", "left", "remain", "ate", "used"],
            "MUL": ["times", "multiply", "each", "every", "per", "double", "triple", "twice"],
            "DIV": ["divide", "split", "share", "half", "quarter", "part", "ratio", "average"]
        }
        
        scores = {}
        for op, keywords in patterns.items():
            score = sum(1 for kw in keywords if kw in text_combined)
            scores[op] = score
        
        # Get best match
        best_op = max(scores, key=scores.get)
        if scores[best_op] == 0:
            return "UNKNOWN"
        return best_op
    
    def save_cluster_assignments(self, output_path: str):
        """Save cluster assignments."""
        output_path = Path(output_path).expanduser()
        
        assignments = {
            "n_spans": len(self.spans),
            "n_clusters": len(np.unique(self.cluster_labels)),
            "assignments": [
                {
                    "span_idx": i,
                    "problem_id": self.spans[i].problem_id,
                    "span_text": self.spans[i].span_text,
                    "cluster_id": int(self.cluster_labels[i])
                }
                for i in range(len(self.spans))
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(assignments, f, indent=2)
        
        print(f"Saved cluster assignments to {output_path}")


def generate_review_file(
    clusterer: SpanClusterer,
    output_path: str,
    n_examples: int = 10
):
    """Generate a review file for manual labeling."""
    output_path = Path(output_path).expanduser()
    
    stats = clusterer.get_cluster_stats()
    
    review = {
        "summary": {
            "n_spans": len(clusterer.spans),
            "n_clusters": len(stats),
            "cluster_sizes": {int(k): v["size"] for k, v in stats.items()}
        },
        "clusters": []
    }
    
    for cluster_id, cluster_data in sorted(stats.items()):
        spans = cluster_data["spans"]
        span_texts = [s.span_text for s in spans]
        
        # Sample examples
        if len(span_texts) > n_examples:
            examples = random.sample(span_texts, n_examples)
        else:
            examples = span_texts
        
        # Get suggested label
        suggested = clusterer.suggest_label(span_texts)
        
        cluster_info = {
            "cluster_id": int(cluster_id),
            "size": cluster_data["size"],
            "suggested_label": suggested,
            "label": None,  # To be filled by reviewer
            "examples": examples
        }
        
        review["clusters"].append(cluster_info)
    
    # Sort by size descending
    review["clusters"].sort(key=lambda x: x["size"], reverse=True)
    
    with open(output_path, 'w') as f:
        json.dump(review, f, indent=2)
    
    print(f"Generated review file at {output_path}")
    return review


def apply_labels(
    labels: Dict[int, str],
    spans_path: str,
    cluster_assignments_path: str,
    output_path: str
):
    """
    Apply labels to spans and save labeled data.
    
    Args:
        labels: Dict mapping cluster_id -> operation_type
        spans_path: Path to extracted spans JSON
        cluster_assignments_path: Path to cluster assignments JSON
        output_path: Output path for labeled spans
    """
    spans_path = Path(spans_path).expanduser()
    cluster_path = Path(cluster_assignments_path).expanduser()
    output_path = Path(output_path).expanduser()
    
    # Load spans
    with open(spans_path, 'r') as f:
        spans = json.load(f)
    
    # Load cluster assignments
    with open(cluster_path, 'r') as f:
        assignments = json.load(f)
    
    # Build cluster lookup
    span_to_cluster = {}
    for a in assignments["assignments"]:
        span_to_cluster[a["span_idx"]] = a["cluster_id"]
    
    # Apply labels
    labeled_spans = []
    for i, span in enumerate(spans):
        cluster_id = span_to_cluster.get(i, -1)
        label = labels.get(cluster_id, "UNKNOWN")
        
        labeled_span = {
            **span,
            "cluster_id": cluster_id,
            "operation_type": label
        }
        labeled_spans.append(labeled_span)
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(labeled_spans, f, indent=2)
    
    print(f"Applied labels to {len(labeled_spans)} spans")
    print(f"Saved to {output_path}")
    
    # Print statistics
    label_counts = defaultdict(int)
    for ls in labeled_spans:
        label_counts[ls["operation_type"]] += 1
    
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Cluster-then-label system for operation classification")
    
    parser.add_argument("--extract", action="store_true", help="Extract spans from problems")
    parser.add_argument("--cluster", action="store_true", help="Cluster extracted spans")
    parser.add_argument("--review", action="store_true", help="Generate review file")
    parser.add_argument("--apply-labels", type=str, help="Apply labels from JSON file")
    
    parser.add_argument("--max-problems", type=int, default=1000, help="Max problems to process")
    parser.add_argument("--max-files", type=int, default=20, help="Max metadata files to load")
    parser.add_argument("--n-clusters", type=int, default=20, help="Number of clusters for k-means")
    parser.add_argument("--clustering-method", type=str, default="kmeans", choices=["kmeans", "dbscan"])
    parser.add_argument("--dbscan-eps", type=float, default=0.3, help="DBSCAN eps parameter")
    parser.add_argument("--span-method", type=str, default="community", choices=["community", "threshold"])
    
    parser.add_argument("--spans-file", type=str, default="~/extracted_spans.json")
    parser.add_argument("--clusters-file", type=str, default="~/cluster_assignments.json")
    parser.add_argument("--review-file", type=str, default="~/cluster_review.json")
    parser.add_argument("--labeled-file", type=str, default="~/labeled_spans.json")
    
    args = parser.parse_args()
    
    # Handle apply-labels case
    if args.apply_labels:
        with open(args.apply_labels, 'r') as f:
            labels = json.load(f)
        
        # Convert string keys to int
        labels = {int(k): v for k, v in labels.items()}
        
        apply_labels(
            labels,
            args.spans_file,
            args.clusters_file,
            args.labeled_file
        )
        return
    
    # Extract spans
    if args.extract:
        extractor = SpanExtractor()
        problems = extractor.load_problems(
            max_files=args.max_files,
            max_problems=args.max_problems
        )
        spans = extractor.extract_spans(problems, method=args.span_method)
        extractor.save_spans(spans, args.spans_file)
    
    # Cluster spans
    if args.cluster:
        # Load spans
        extractor = SpanExtractor()
        spans = extractor.load_spans(args.spans_file)
        
        # Cluster
        clusterer = SpanClusterer(spans)
        
        if args.clustering_method == "kmeans":
            clusterer.cluster_kmeans(n_clusters=args.n_clusters)
        else:
            clusterer.cluster_dbscan(eps=args.dbscan_eps)
        
        clusterer.save_cluster_assignments(args.clusters_file)
        
        # Generate review if requested
        if args.review:
            generate_review_file(clusterer, args.review_file)
    
    # Generate review only (if clustering already done)
    elif args.review and not args.cluster:
        print("Review generation requires --cluster flag or pre-existing cluster assignments")


if __name__ == "__main__":
    main()
