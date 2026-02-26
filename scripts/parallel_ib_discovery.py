#!/usr/bin/env python3
"""
Parallel IB Template Discovery - Optimized for Large IAF Data

Key optimizations:
1. Sequential shard loading (avoids RAM explosion from parallel JSON parsing)
2. GPU clustering with RAPIDS cuML
3. Vectorized feature encoding with numpy
4. Progress output at each step
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import time
import re
import sys
import warnings
warnings.filterwarnings('ignore')

# Try GPU clustering first, fall back to CPU
try:
    from cuml.cluster import AgglomerativeClustering as GPUClustering
    from cuml.metrics.cluster import silhouette_score as gpu_silhouette
    import cupy as cp
    GPU_AVAILABLE = True
    print("RAPIDS cuML available - using GPU clustering")
except ImportError as e:
    GPU_AVAILABLE = False
    print(f"RAPIDS not available ({e}) - using CPU clustering")

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder

sys.stdout.flush()


# ============================================================================
# OPERATOR PATTERNS
# ============================================================================
OPERATOR_PATTERNS = {
    'ADD': [r'\+', r'plus', r'added', r'sum'],
    'SUB': [r'-(?!\d)', r'minus', r'subtract', r'difference'],
    'MUL': [r'\\times', r'\\cdot', r'\*', r'×', r'multiply', r'product'],
    'DIV': [r'\\div', r'\\frac\{', r'/', r'÷', r'divide', r'quotient'],
    'POW': [r'\^', r'squared', r'cubed', r'exponent'],
    'SQRT': [r'\\sqrt', r'square root', r'√'],
    'SIN': [r'\\sin', r'sine'],
    'COS': [r'\\cos', r'cosine'],
    'TAN': [r'\\tan', r'tangent'],
    'LOG': [r'\\log', r'logarithm'],
    'LN': [r'\\ln', r'natural log'],
    'FACTORIAL': [r'!', r'factorial'],
    'CHOOSE': [r'\\binom', r'\\choose', r'combination'],
    'SOLVE': [r'solve', r'find.*=', r'x\s*='],
    'SUBSTITUTE': [r'substitut', r'plug in', r'replace'],
    'SIMPLIFY': [r'simplif', r'reduce', r'cancel'],
    'FACTOR': [r'factor'],
    'EXPAND': [r'expand', r'distribute', r'FOIL'],
    'MOD': [r'\\mod', r'remainder', r'modulo'],
    'GCD': [r'gcd', r'greatest common'],
    'LCM': [r'lcm', r'least common'],
    'EQUALS': [r'=(?!=)', r'equal'],
}


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_equations(text: str) -> List[str]:
    """Extract equations from CoT text."""
    equations = []
    display_math = re.findall(r'\\\[(.*?)\\\]|\$\$(.*?)\$\$', text, re.DOTALL)
    for match in display_math:
        eq = match[0] or match[1]
        if eq.strip():
            equations.append(eq.strip())
    inline_math = re.findall(r'(?<!\$)\$([^\$]+)\$(?!\$)', text)
    equations.extend([m.strip() for m in inline_math if m.strip()])
    return equations


def classify_operator(text: str) -> str:
    """Identify primary operator in expression."""
    text_lower = text.lower()
    priority = ['SQRT', 'POW', 'SIN', 'COS', 'TAN', 'LOG', 'LN',
                'FACTORIAL', 'CHOOSE', 'GCD', 'LCM', 'DIV', 'MUL', 'SUB', 'ADD',
                'SOLVE', 'SUBSTITUTE', 'SIMPLIFY', 'FACTOR', 'EXPAND', 'MOD']

    for op in priority:
        patterns = OPERATOR_PATTERNS.get(op, [])
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return op
    return 'UNKNOWN'


def extract_numbers(text: str) -> List[float]:
    """Extract numbers from text."""
    numbers = []
    for match in re.finditer(r'-?\d+(?:\.\d+)?', text):
        try:
            numbers.append(float(match.group()))
        except ValueError:
            pass
    return numbers


def classify_result_type(expr: str) -> str:
    """Classify result type."""
    expr = expr.strip()
    if re.match(r'^-?\d+$', expr):
        return 'INT'
    if re.search(r'\\frac\{-?\d+\}\{-?\d+\}|^\d+/\d+$', expr):
        return 'FRAC'
    if re.match(r'^-?\d+\.\d+$', expr):
        return 'DECIMAL'
    if re.search(r'\\pi', expr):
        return 'PI'
    if re.search(r'\\sqrt\{', expr):
        return 'SQRT_EXPR'
    return 'EXPR'


def process_record(record: Dict) -> List[Dict]:
    """Extract features from a single IAF record."""
    features = []

    cot = record.get('generated_cot', '')
    problem = record.get('problem_text', '')
    level = record.get('level', 'Unknown')
    prob_type = record.get('type', 'Unknown')

    if not cot:
        return features

    equations = extract_equations(cot)
    problem_numbers = extract_numbers(problem)
    prev_results = []

    for i, eq in enumerate(equations):
        operator = classify_operator(eq)
        operands = extract_numbers(eq)
        result_type = classify_result_type(eq)

        prov = []
        for num in operands[:2]:
            if num in problem_numbers:
                prov.append('PROB')
            elif num in prev_results:
                prov.append('PREV')
            else:
                prov.append('OTHER')
        prov_pattern = '_'.join(prov) if prov else 'NONE'

        results = extract_numbers(eq.split('=')[-1] if '=' in eq else eq)
        prev_results.extend(results)

        features.append({
            'operator': operator,
            'operand_count': min(len(operands), 3),
            'result_type': result_type,
            'chain_position_bin': min(int(i / max(len(equations), 1) * 5), 4),
            'category': prob_type,
            'level': level,
            'provenance_pattern': prov_pattern,
            'text': eq[:100],
            'raw_expression': eq[:200],
        })

    return features


# ============================================================================
# CLUSTERING
# ============================================================================

def encode_features_vectorized(features: List[Dict]) -> Tuple[np.ndarray, List[str]]:
    """Encode features as one-hot vectors."""
    operators = np.array([f['operator'] for f in features]).reshape(-1, 1)
    result_types = np.array([f['result_type'] for f in features]).reshape(-1, 1)
    operand_counts = np.array([str(f['operand_count']) for f in features]).reshape(-1, 1)
    chain_bins = np.array([str(f['chain_position_bin']) for f in features]).reshape(-1, 1)
    categories = np.array([f['category'] for f in features]).reshape(-1, 1)
    provenances = np.array([f['provenance_pattern'] for f in features]).reshape(-1, 1)

    encoders = {}
    encoded = []
    weights = [2.0, 1.5, 1.0, 0.5, 0.3, 0.8]

    for name, data, weight in zip(
        ['op', 'rt', 'oc', 'cb', 'cat', 'prov'],
        [operators, result_types, operand_counts, chain_bins, categories, provenances],
        weights
    ):
        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X = enc.fit_transform(data) * weight
        encoded.append(X)
        encoders[name] = enc

    X = np.hstack(encoded).astype(np.float32)
    feature_names = []
    for name, enc in encoders.items():
        feature_names.extend([f"{name}_{c}" for c in enc.categories_[0]])

    return X, feature_names


def find_optimal_clusters(X: np.ndarray, min_k: int = 30, max_k: int = 100) -> Tuple[int, float]:
    """Find optimal clusters."""
    best_k, best_score = min_k, -1

    # Limit search range based on data size
    max_k = min(max_k, len(X) // 5)
    if max_k < min_k:
        return min_k, 0.0

    print(f"  Searching k in [{min_k}, {max_k}]...")
    sys.stdout.flush()

    for k in range(min_k, max_k + 1, 5):
        if GPU_AVAILABLE:
            X_gpu = cp.asarray(X)
            clustering = GPUClustering(n_clusters=k)
            labels = clustering.fit_predict(X_gpu)
            score = float(gpu_silhouette(X_gpu, labels))
        else:
            clustering = AgglomerativeClustering(n_clusters=k)
            labels = clustering.fit_predict(X)
            sample_size = min(5000, len(X))
            score = silhouette_score(X, labels, sample_size=sample_size)

        if score > best_score:
            best_k, best_score = k, score

        if k % 10 == 0:
            print(f"    k={k}: silhouette={score:.3f}")
            sys.stdout.flush()

    return best_k, best_score


def cluster_features(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """Run final clustering."""
    if GPU_AVAILABLE:
        X_gpu = cp.asarray(X)
        clustering = GPUClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(X_gpu)
        return cp.asnumpy(labels)
    else:
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        return clustering.fit_predict(X)


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_templates(features: List[Dict], labels: np.ndarray) -> Dict:
    """Analyze discovered templates."""
    templates = defaultdict(list)
    for f, label in zip(features, labels):
        templates[int(label)].append(f)

    summaries = {}
    for tid, members in templates.items():
        ops = Counter(m['operator'] for m in members)
        rts = Counter(m['result_type'] for m in members)
        cats = Counter(m['category'] for m in members)

        top_op = ops.most_common(1)[0][0]
        top_rt = rts.most_common(1)[0][0]

        summaries[tid] = {
            'count': len(members),
            'top_operator': top_op,
            'top_result_type': top_rt,
            'top_category': cats.most_common(1)[0][0],
            'op_purity': ops.most_common(1)[0][1] / len(members),
            'rt_purity': rts.most_common(1)[0][1] / len(members),
            'name': f"T{tid}_{top_op}_{top_rt}",
            'examples': [m['raw_expression'][:80] for m in members[:3]],
        }

    return summaries


def generate_classifier_labels(features: List[Dict], labels: np.ndarray, summaries: Dict) -> List[Dict]:
    """Generate training labels for C2 classifier."""
    labeled = []
    for f, label in zip(features, labels):
        label_int = int(label)
        labeled.append({
            'text': f['text'],
            'raw_expression': f['raw_expression'],
            'template_id': label_int,
            'template_name': summaries[label_int]['name'],
            'operator': f['operator'],
            'result_type': f['result_type'],
            'category': f['category'],
        })
    return labeled


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Parallel IB Template Discovery')
    parser.add_argument('--data-dir', type=str, default='/home/ubuntu/data/iaf_v3')
    parser.add_argument('--output-dir', type=str, default='/home/ubuntu/data/ib_results')
    parser.add_argument('--min-k', type=int, default=30)
    parser.add_argument('--max-k', type=int, default=100)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PARALLEL IB TEMPLATE DISCOVERY")
    print("=" * 70)
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"GPU clustering: {GPU_AVAILABLE}")
    sys.stdout.flush()

    shards = sorted(data_dir.glob("*.json"))
    print(f"Found {len(shards)} IAF shards")
    sys.stdout.flush()

    # ========================================================================
    # PHASE 1: Sequential feature extraction (to avoid RAM explosion)
    # ========================================================================
    print("\n" + "-" * 70)
    print("PHASE 1: Feature Extraction (sequential)")
    print("-" * 70)
    sys.stdout.flush()

    start_time = time.time()
    all_features = []
    total_records = 0

    for shard in shards:
        shard_start = time.time()
        print(f"  Loading {shard.name}...", end=" ")
        sys.stdout.flush()

        with open(shard, 'r') as f:
            records = json.load(f)

        shard_features = []
        for record in records:
            shard_features.extend(process_record(record))

        all_features.extend(shard_features)
        total_records += len(records)

        print(f"{len(records)} records -> {len(shard_features)} features ({time.time()-shard_start:.1f}s)")
        sys.stdout.flush()

    extract_time = time.time() - start_time
    print(f"\nExtracted {len(all_features)} features from {total_records} records")
    print(f"Extraction time: {extract_time:.1f}s")
    sys.stdout.flush()

    if not all_features:
        print("ERROR: No features extracted!")
        return

    # ========================================================================
    # PHASE 2: Feature encoding
    # ========================================================================
    print("\n" + "-" * 70)
    print("PHASE 2: Feature Encoding")
    print("-" * 70)
    sys.stdout.flush()

    start_time = time.time()
    X, feature_names = encode_features_vectorized(all_features)
    encode_time = time.time() - start_time

    print(f"Encoded matrix shape: {X.shape}")
    print(f"Encoding time: {encode_time:.1f}s")
    print(f"Memory: {X.nbytes / 1e6:.1f} MB")
    sys.stdout.flush()

    # ========================================================================
    # PHASE 3: Find optimal cluster count
    # ========================================================================
    print("\n" + "-" * 70)
    print("PHASE 3: Finding Optimal Cluster Count")
    print("-" * 70)
    sys.stdout.flush()

    start_time = time.time()
    best_k, best_score = find_optimal_clusters(X, args.min_k, args.max_k)
    cluster_search_time = time.time() - start_time

    print(f"\nOptimal k={best_k} (silhouette={best_score:.3f})")
    print(f"Search time: {cluster_search_time:.1f}s")
    sys.stdout.flush()

    # ========================================================================
    # PHASE 4: Final clustering
    # ========================================================================
    print("\n" + "-" * 70)
    print("PHASE 4: Final Clustering")
    print("-" * 70)
    sys.stdout.flush()

    start_time = time.time()
    labels = cluster_features(X, best_k)
    cluster_time = time.time() - start_time

    print(f"Clustering time: {cluster_time:.1f}s")
    sys.stdout.flush()

    # ========================================================================
    # PHASE 5: Analysis
    # ========================================================================
    print("\n" + "-" * 70)
    print("PHASE 5: Template Analysis")
    print("-" * 70)
    sys.stdout.flush()

    summaries = analyze_templates(all_features, labels)

    sorted_templates = sorted(summaries.items(), key=lambda x: x[1]['count'], reverse=True)
    print(f"\nDiscovered {len(summaries)} templates")
    print("\nTop 20 templates:")
    for tid, s in sorted_templates[:20]:
        print(f"  {s['name']:30} count={s['count']:4} op_purity={s['op_purity']:.2f}")
    sys.stdout.flush()

    op_purities = [s['op_purity'] for s in summaries.values()]
    print(f"\nOperator purity: mean={np.mean(op_purities):.2f}, min={min(op_purities):.2f}")
    sys.stdout.flush()

    # ========================================================================
    # PHASE 6: Save outputs
    # ========================================================================
    print("\n" + "-" * 70)
    print("PHASE 6: Saving Results")
    print("-" * 70)
    sys.stdout.flush()

    with open(output_dir / "templates.json", 'w') as f:
        json.dump({str(k): v for k, v in summaries.items()}, f, indent=2)
    print(f"Saved: {output_dir / 'templates.json'}")

    labeled_data = generate_classifier_labels(all_features, labels, summaries)
    with open(output_dir / "classifier_labels.json", 'w') as f:
        json.dump(labeled_data, f, indent=2)
    print(f"Saved: {output_dir / 'classifier_labels.json'}")

    template_map = {
        str(tid): {
            'name': s['name'],
            'operator': s['top_operator'],
            'result_type': s['top_result_type'],
        }
        for tid, s in summaries.items()
    }
    with open(output_dir / "template_map.json", 'w') as f:
        json.dump(template_map, f, indent=2)
    print(f"Saved: {output_dir / 'template_map.json'}")

    stats = {
        'total_features': len(all_features),
        'total_records': total_records,
        'num_templates': len(summaries),
        'optimal_k': best_k,
        'silhouette_score': best_score,
        'extract_time_sec': extract_time,
        'cluster_time_sec': cluster_time,
        'gpu_used': GPU_AVAILABLE,
    }
    with open(output_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {output_dir / 'stats.json'}")
    sys.stdout.flush()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    total_time = extract_time + encode_time + cluster_search_time + cluster_time
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Records processed: {total_records}")
    print(f"Features extracted: {len(all_features)}")
    print(f"Templates discovered: {len(summaries)}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Results saved to: {output_dir}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
