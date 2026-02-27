#!/usr/bin/env python3
"""
MATH Template Discovery via Information Bottleneck Clustering

Takes the extracted CoT features and discovers template clusters.
Uses hierarchical clustering with optimal cut based on silhouette score.

Output: Template assignments for each step, ready for classifier training.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import warnings
warnings.filterwarnings('ignore')


def load_features(path: str) -> List[Dict]:
    """Load extracted feature data."""
    with open(path) as f:
        return json.load(f)


def encode_features(features: List[Dict]) -> Tuple[np.ndarray, List[str]]:
    """Encode categorical features as one-hot vectors."""
    # Extract categorical columns
    operators = [f['operator'] for f in features]
    result_types = [f['result_type'] for f in features]
    operand_counts = [str(f['operand_count']) for f in features]
    chain_bins = [str(f['chain_position_bin']) for f in features]
    categories = [f['category'] for f in features]
    provenance = [f['provenance_pattern'] for f in features]

    # One-hot encode each
    enc_op = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    enc_rt = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    enc_oc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    enc_cb = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    enc_cat = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    enc_prov = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    X_op = enc_op.fit_transform(np.array(operators).reshape(-1, 1))
    X_rt = enc_rt.fit_transform(np.array(result_types).reshape(-1, 1))
    X_oc = enc_oc.fit_transform(np.array(operand_counts).reshape(-1, 1))
    X_cb = enc_cb.fit_transform(np.array(chain_bins).reshape(-1, 1))
    X_cat = enc_cat.fit_transform(np.array(categories).reshape(-1, 1))
    X_prov = enc_prov.fit_transform(np.array(provenance).reshape(-1, 1))

    # Weight features differently (operator and result_type are most important)
    X = np.hstack([
        X_op * 2.0,      # Operator is most important
        X_rt * 1.5,      # Result type is important
        X_oc * 1.0,      # Operand count
        X_cb * 0.5,      # Chain position (less important)
        X_cat * 0.3,     # Category (minor influence)
        X_prov * 0.8,    # Provenance pattern
    ])

    # Build feature names for interpretation
    feature_names = (
        [f"op_{c}" for c in enc_op.categories_[0]] +
        [f"rt_{c}" for c in enc_rt.categories_[0]] +
        [f"oc_{c}" for c in enc_oc.categories_[0]] +
        [f"cb_{c}" for c in enc_cb.categories_[0]] +
        [f"cat_{c}" for c in enc_cat.categories_[0]] +
        [f"prov_{c}" for c in enc_prov.categories_[0]]
    )

    return X, feature_names


def find_optimal_clusters(X: np.ndarray, min_k: int = 20, max_k: int = 150) -> Tuple[int, List[float]]:
    """Find optimal number of clusters using silhouette score."""
    print("\nFinding optimal cluster count...")
    scores = []
    ks = list(range(min_k, min(max_k + 1, len(X) // 10)))  # Don't test more than N/10

    for k in ks:
        if k % 20 == 0:
            print(f"  Testing k={k}...")
        clustering = AgglomerativeClustering(n_clusters=k)
        labels = clustering.fit_predict(X)
        score = silhouette_score(X, labels, sample_size=min(5000, len(X)))
        scores.append((k, score))

    # Find k with best silhouette score
    best_k, best_score = max(scores, key=lambda x: x[1])

    return best_k, scores


def cluster_features(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """Run hierarchical clustering."""
    print(f"\nClustering into {n_clusters} templates...")
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(X)
    return labels


def analyze_templates(features: List[Dict], labels: np.ndarray) -> Dict:
    """Analyze discovered templates."""
    templates = defaultdict(list)
    for i, (f, label) in enumerate(zip(features, labels)):
        templates[int(label)].append(f)

    # Summarize each template
    template_summaries = {}
    for tid, members in templates.items():
        # Most common operator
        ops = Counter(m['operator'] for m in members)
        top_op = ops.most_common(1)[0][0]

        # Most common result type
        rts = Counter(m['result_type'] for m in members)
        top_rt = rts.most_common(1)[0][0]

        # Most common operand count
        ocs = Counter(m['operand_count'] for m in members)
        top_oc = ocs.most_common(1)[0][0]

        # Most common category
        cats = Counter(m['category'] for m in members)
        top_cat = cats.most_common(1)[0][0]

        # Purity scores
        op_purity = ops.most_common(1)[0][1] / len(members)
        rt_purity = rts.most_common(1)[0][1] / len(members)

        template_summaries[tid] = {
            'count': len(members),
            'top_operator': top_op,
            'top_result_type': top_rt,
            'top_operand_count': top_oc,
            'top_category': top_cat,
            'op_purity': op_purity,
            'rt_purity': rt_purity,
            'name': f"T{tid}_{top_op}_{top_rt}",
            'examples': [m['raw_expression'][:80] for m in members[:3]],
        }

    return template_summaries


def generate_template_labels(features: List[Dict], labels: np.ndarray, summaries: Dict) -> List[Dict]:
    """Generate training labels for classifier."""
    labeled_data = []
    for f, label in zip(features, labels):
        summary = summaries[int(label)]  # Convert numpy int to Python int
        labeled_data.append({
            'text': f['text'],
            'raw_expression': f['raw_expression'],
            'template_id': int(label),  # Convert numpy int to Python int
            'template_name': summary['name'],
            'operator': f['operator'],
            'result_type': f['result_type'],
            'category': f['category'],
        })
    return labeled_data


def main():
    print("=" * 70)
    print("MATH TEMPLATE DISCOVERY (Information Bottleneck)")
    print("=" * 70)

    # Load features
    feature_path = "/tmp/math_cot_features.json"
    print(f"\nLoading features from {feature_path}...")
    features = load_features(feature_path)
    print(f"Loaded {len(features)} feature vectors")

    # Encode features
    print("\nEncoding features...")
    X, feature_names = encode_features(features)
    print(f"Encoded feature matrix: {X.shape}")

    # Find optimal cluster count
    # Start with a reasonable range based on 121 unique combinations
    best_k, scores = find_optimal_clusters(X, min_k=30, max_k=120)
    print(f"\nOptimal cluster count: {best_k}")
    print(f"Silhouette score: {max(s[1] for s in scores):.3f}")

    # Print silhouette curve
    print("\nSilhouette scores by k:")
    for k, score in scores[::10]:  # Every 10th
        bar = "█" * int(score * 50)
        print(f"  k={k:3d}: {score:.3f} {bar}")

    # Cluster with optimal k
    labels = cluster_features(X, best_k)

    # Analyze templates
    print("\nAnalyzing discovered templates...")
    summaries = analyze_templates(features, labels)

    # Print top templates
    print("\n" + "-" * 70)
    print(f"TOP 30 TEMPLATES (of {len(summaries)} total)")
    print("-" * 70)

    sorted_templates = sorted(summaries.items(), key=lambda x: x[1]['count'], reverse=True)
    for tid, s in sorted_templates[:30]:
        print(f"\n  {s['name']}")
        print(f"    Count: {s['count']} | Op purity: {s['op_purity']:.2f} | RT purity: {s['rt_purity']:.2f}")
        print(f"    Category: {s['top_category']}")
        print(f"    Examples:")
        for ex in s['examples']:
            print(f"      - {ex}")

    # Statistics
    print("\n" + "-" * 70)
    print("TEMPLATE STATISTICS")
    print("-" * 70)

    template_sizes = [s['count'] for s in summaries.values()]
    print(f"Total templates: {len(summaries)}")
    print(f"Largest template: {max(template_sizes)} steps")
    print(f"Smallest template: {min(template_sizes)} steps")
    print(f"Median template size: {np.median(template_sizes):.0f} steps")

    # Purity analysis
    op_purities = [s['op_purity'] for s in summaries.values()]
    rt_purities = [s['rt_purity'] for s in summaries.values()]
    print(f"\nOperator purity: mean={np.mean(op_purities):.2f}, min={min(op_purities):.2f}")
    print(f"Result type purity: mean={np.mean(rt_purities):.2f}, min={min(rt_purities):.2f}")

    # Template distribution by operator
    print("\n" + "-" * 70)
    print("TEMPLATES BY DOMINANT OPERATOR")
    print("-" * 70)
    op_templates = defaultdict(list)
    for tid, s in summaries.items():
        op_templates[s['top_operator']].append((tid, s))

    for op, templates in sorted(op_templates.items(), key=lambda x: -len(x[1])):
        total_steps = sum(t[1]['count'] for t in templates)
        print(f"  {op:15} {len(templates):3d} templates, {total_steps:5d} steps")

    # Generate labeled data
    print("\n" + "-" * 70)
    print("GENERATING CLASSIFIER TRAINING LABELS")
    print("-" * 70)

    labeled_data = generate_template_labels(features, labels, summaries)

    # Save outputs
    output_dir = Path("/tmp")

    # Save template summaries
    with open(output_dir / "math_templates.json", 'w') as f:
        json.dump({str(k): v for k, v in summaries.items()}, f, indent=2)
    print(f"Saved template summaries to {output_dir / 'math_templates.json'}")

    # Save labeled training data
    with open(output_dir / "math_classifier_labels.json", 'w') as f:
        json.dump(labeled_data, f, indent=2)
    print(f"Saved classifier labels to {output_dir / 'math_classifier_labels.json'}")

    # Save template mapping (for inference)
    template_map = {
        tid: {
            'name': s['name'],
            'operator': s['top_operator'],
            'result_type': s['top_result_type'],
            'operand_count': s['top_operand_count'],
        }
        for tid, s in summaries.items()
    }
    with open(output_dir / "math_template_map.json", 'w') as f:
        json.dump(template_map, f, indent=2)
    print(f"Saved template map to {output_dir / 'math_template_map.json'}")

    print("\n" + "=" * 70)
    print("DISCOVERY COMPLETE")
    print("=" * 70)
    print(f"\nDiscovered {len(summaries)} templates from {len(features)} CoT steps")
    print(f"Ready for classifier training with MATH-specific labels")


if __name__ == "__main__":
    main()
