#!/usr/bin/env python3
"""
MapReduce IB Template Discovery with Plateau Detection

Processes 33GB of IAF data across 16 shards using true Information Bottleneck
clustering with β-annealing. Lets the data tell us the optimal cluster count
by finding plateaus in the I(X;T) curve.

Architecture:
  MAP:    Each shard → extract features → local clustering at multiple β
  REDUCE: Merge cluster assignments → anneal β → find plateaus → final templates

Usage:
  # On EC2 worker (processes assigned shards):
  python mapreduce_ib_discovery.py map --shard-ids 0,1 --s3-prefix s3://mycelium-data/iaf_extraction/instance1/

  # On coordinator (merges results):
  python mapreduce_ib_discovery.py reduce --results-dir /tmp/ib_map_results/

  # Full local run (single machine with enough RAM):
  python mapreduce_ib_discovery.py full --data-dir /path/to/shards/
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Iterator, Optional
from dataclasses import dataclass, asdict
import time
import re
import sys
import warnings
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

warnings.filterwarnings('ignore')

# Streaming JSON - critical for 2GB files
try:
    import ijson
    IJSON_AVAILABLE = True
except ImportError:
    IJSON_AVAILABLE = False
    print("WARNING: ijson not available. Install with: pip install ijson")

from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mutual_info_score
import scipy.sparse as sp


# ============================================================================
# OPERATOR PATTERNS (from existing code)
# ============================================================================
OPERATOR_PATTERNS = {
    'ADD': [r'\+', r'plus', r'added', r'sum', r'total'],
    'SUB': [r'-(?!\d)', r'minus', r'subtract', r'difference', r'remaining'],
    'MUL': [r'\\times', r'\\cdot', r'\*', r'multiply', r'product'],
    'DIV': [r'\\div', r'\\frac\{', r'/', r'divide', r'quotient'],
    'POW': [r'\^', r'\*\*', r'squared', r'cubed', r'exponent'],
    'SQRT': [r'\\sqrt', r'square root'],
    'SIN': [r'\\sin', r'sine'],
    'COS': [r'\\cos', r'cosine'],
    'TAN': [r'\\tan', r'tangent'],
    'LOG': [r'\\log', r'logarithm'],
    'LN': [r'\\ln', r'natural log'],
    'FACTORIAL': [r'!', r'factorial'],
    'CHOOSE': [r'\\binom', r'\\choose', r'combination', r'C\('],
    'PERMUTE': [r'P\(', r'permutation'],
    'SOLVE': [r'solve', r'find.*=', r'x\s*='],
    'SUBSTITUTE': [r'substitut', r'plug in', r'replace', r'let.*='],
    'SIMPLIFY': [r'simplif', r'reduce', r'cancel'],
    'FACTOR': [r'factor'],
    'EXPAND': [r'expand', r'distribute', r'FOIL'],
    'MOD': [r'\\mod', r'\\pmod', r'remainder', r'modulo'],
    'GCD': [r'gcd', r'greatest common'],
    'LCM': [r'lcm', r'least common'],
    'ABS': [r'\|.*\|', r'\\abs', r'absolute'],
    'FLOOR': [r'\\lfloor', r'floor'],
    'CEIL': [r'\\lceil', r'ceil'],
    'EQUALS': [r'=(?!=)', r'equal'],
}


# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class Feature:
    """Single extracted feature from a CoT step."""
    operator: str
    operand_count: int
    result_type: str
    chain_position_bin: int
    category: str
    provenance: str
    text: str
    problem_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IBResult:
    """Result from IB clustering at a specific β."""
    beta: float
    n_clusters: int
    i_xt: float  # I(X;T) - complexity
    i_ty: float  # I(T;Y) - relevance
    labels: np.ndarray

    def ib_objective(self) -> float:
        """IB objective: maximize I(T;Y) - β*I(X;T)"""
        return self.i_ty - self.beta * self.i_xt


@dataclass
class PlateauRegion:
    """A detected plateau in the IB curve."""
    beta_start: float
    beta_end: float
    n_clusters: int
    stability: float  # How stable the cluster count is in this region


# ============================================================================
# STREAMING JSON PARSER
# ============================================================================
def stream_json_records(path: str) -> Iterator[dict]:
    """
    Stream JSON records without loading entire file into memory.
    Critical for 2GB+ files that would freeze the process.
    """
    if not IJSON_AVAILABLE:
        # Fallback: load entire file (will be slow/dangerous for large files)
        print(f"  WARNING: Loading entire file into memory: {path}")
        with open(path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                yield from data
            else:
                yield data
        return

    with open(path, 'rb') as f:
        # Try to detect if it's a JSON array or JSONL
        first_char = f.read(1)
        f.seek(0)

        if first_char == b'[':
            # JSON array - stream items
            parser = ijson.items(f, 'item')
            yield from parser
        else:
            # Assume JSONL (one JSON object per line)
            for line in f:
                if line.strip():
                    yield json.loads(line)


def stream_s3_json(s3_path: str) -> Iterator[dict]:
    """Stream JSON records directly from S3 without downloading."""
    if not IJSON_AVAILABLE:
        raise RuntimeError("ijson required for S3 streaming. pip install ijson")

    # Use aws s3 cp to stream to stdout
    cmd = ['aws', 's3', 'cp', s3_path, '-']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        parser = ijson.items(proc.stdout, 'item')
        yield from parser
    finally:
        proc.terminate()


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
def extract_equations(text: str) -> List[str]:
    """Extract equations from CoT text."""
    equations = []
    # Display math
    display_math = re.findall(r'\\\[(.*?)\\\]|\$\$(.*?)\$\$', text, re.DOTALL)
    for match in display_math:
        eq = match[0] or match[1]
        if eq.strip():
            equations.append(eq.strip())
    # Inline math
    inline_math = re.findall(r'(?<!\$)\$([^\$]+)\$(?!\$)', text)
    equations.extend([m.strip() for m in inline_math if m.strip()])
    return equations


def classify_operator(text: str) -> str:
    """Identify primary operator in expression."""
    text_lower = text.lower()
    priority = ['SQRT', 'POW', 'SIN', 'COS', 'TAN', 'LOG', 'LN', 'ABS', 'FLOOR', 'CEIL',
                'FACTORIAL', 'CHOOSE', 'PERMUTE', 'GCD', 'LCM', 'MOD',
                'DIV', 'MUL', 'SUB', 'ADD',
                'SOLVE', 'SUBSTITUTE', 'SIMPLIFY', 'FACTOR', 'EXPAND']

    for op in priority:
        patterns = OPERATOR_PATTERNS.get(op, [])
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return op
    return 'UNKNOWN'


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
    if re.search(r'[a-zA-Z]', expr):
        return 'SYMBOLIC'
    return 'EXPR'


def extract_numbers(text: str) -> List[float]:
    """Extract numbers from text."""
    numbers = []
    for match in re.finditer(r'-?\d+(?:\.\d+)?', text):
        try:
            numbers.append(float(match.group()))
        except ValueError:
            pass
    return numbers


def process_record(record: dict) -> List[Feature]:
    """Extract features from a single IAF record."""
    features = []

    cot = record.get('generated_cot', '')
    problem = record.get('problem_text', '')
    level = record.get('level', 'Unknown')
    prob_type = record.get('type', 'Unknown')
    problem_id = record.get('problem_id', hashlib.md5(problem.encode()).hexdigest()[:8])

    if not cot:
        return features

    equations = extract_equations(cot)
    problem_numbers = extract_numbers(problem)
    prev_results = []

    for i, eq in enumerate(equations):
        operator = classify_operator(eq)
        operands = extract_numbers(eq)
        result_type = classify_result_type(eq)

        # Determine provenance pattern
        prov = []
        for num in operands[:2]:
            if num in problem_numbers:
                prov.append('PROB')
            elif num in prev_results:
                prov.append('PREV')
            else:
                prov.append('OTHER')
        prov_pattern = '_'.join(prov) if prov else 'NONE'

        # Track results for next iteration
        results = extract_numbers(eq.split('=')[-1] if '=' in eq else eq)
        prev_results.extend(results)

        features.append(Feature(
            operator=operator,
            operand_count=min(len(operands), 3),
            result_type=result_type,
            chain_position_bin=min(int(i / max(len(equations), 1) * 5), 4),
            category=prob_type,
            provenance=prov_pattern,
            text=eq[:200],
            problem_id=problem_id,
        ))

    return features


# ============================================================================
# FEATURE ENCODING
# ============================================================================
def encode_features(features: List[Feature]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode features for IB clustering.
    Returns (X, Y) where:
      X = feature embeddings (input to IB)
      Y = target labels for computing I(T;Y) - using result_type as proxy for execution result
    """
    if not features:
        return np.array([]), np.array([])

    # Extract categorical columns
    operators = np.array([f.operator for f in features]).reshape(-1, 1)
    result_types = np.array([f.result_type for f in features]).reshape(-1, 1)
    operand_counts = np.array([str(f.operand_count) for f in features]).reshape(-1, 1)
    chain_bins = np.array([str(f.chain_position_bin) for f in features]).reshape(-1, 1)
    categories = np.array([f.category for f in features]).reshape(-1, 1)
    provenances = np.array([f.provenance for f in features]).reshape(-1, 1)

    # One-hot encode with weights
    encoders = {}
    encoded = []
    weights = [2.0, 1.5, 1.0, 0.5, 0.3, 0.8]

    for name, data, weight in zip(
        ['op', 'rt', 'oc', 'cb', 'cat', 'prov'],
        [operators, result_types, operand_counts, chain_bins, categories, provenances],
        weights
    ):
        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_enc = enc.fit_transform(data) * weight
        encoded.append(X_enc)
        encoders[name] = enc

    X = np.hstack(encoded).astype(np.float32)

    # Y = result type as integer labels (for computing I(T;Y))
    rt_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    rt_encoder.fit(result_types)
    Y = rt_encoder.transform(result_types).argmax(axis=1)

    return X, Y


# ============================================================================
# INFORMATION BOTTLENECK CLUSTERING
# ============================================================================
def compute_mutual_info(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Compute mutual information I(A;B)."""
    return mutual_info_score(labels_a, labels_b)


def ib_cluster(X: np.ndarray, Y: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, float, float]:
    """
    Perform clustering and compute IB metrics.

    Returns: (labels, I(X;T), I(T;Y))
    """
    if len(X) < n_clusters:
        n_clusters = max(2, len(X) // 2)

    # Use MiniBatchKMeans for speed on large datasets
    if len(X) > 10000:
        clusterer = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, n_init=3, random_state=42)
    else:
        clusterer = MiniBatchKMeans(n_clusters=n_clusters, n_init=10, random_state=42)

    labels = clusterer.fit_predict(X)

    # Compute I(X;T) - discretize X for MI computation
    # Use cluster assignment as discretization of X
    X_discrete = (X * 10).astype(int).sum(axis=1) % 1000  # Simple hash
    i_xt = compute_mutual_info(X_discrete, labels)

    # Compute I(T;Y) - how much cluster assignment tells us about result type
    i_ty = compute_mutual_info(labels, Y)

    return labels, i_xt, i_ty


def anneal_beta(X: np.ndarray, Y: np.ndarray,
                beta_range: Tuple[float, float] = (0.001, 10.0),
                n_steps: int = 50) -> List[IBResult]:
    """
    Anneal β from low to high and track cluster structure.

    At low β: compression dominates → few clusters
    At high β: relevance dominates → many clusters

    Plateaus in I(X;T) vs β indicate natural cluster counts.
    """
    results = []
    betas = np.logspace(np.log10(beta_range[0]), np.log10(beta_range[1]), n_steps)

    print(f"  Annealing β from {beta_range[0]} to {beta_range[1]} ({n_steps} steps)...")
    sys.stdout.flush()

    for i, beta in enumerate(betas):
        # At low β, want few clusters. At high β, want many.
        # Heuristic: n_clusters scales with β
        n_clusters = max(5, min(200, int(10 + 20 * np.log10(beta + 1))))

        labels, i_xt, i_ty = ib_cluster(X, Y, n_clusters)
        actual_clusters = len(np.unique(labels))

        results.append(IBResult(
            beta=beta,
            n_clusters=actual_clusters,
            i_xt=i_xt,
            i_ty=i_ty,
            labels=labels,
        ))

        if (i + 1) % 10 == 0:
            print(f"    β={beta:.4f}: {actual_clusters} clusters, I(X;T)={i_xt:.3f}, I(T;Y)={i_ty:.3f}")
            sys.stdout.flush()

    return results


def find_plateaus(results: List[IBResult], min_plateau_width: int = 3) -> List[PlateauRegion]:
    """
    Find plateaus in the IB curve where cluster count is stable.
    These represent natural groupings in the data.
    """
    plateaus = []

    # Group consecutive results by cluster count
    i = 0
    while i < len(results):
        n_clusters = results[i].n_clusters
        start_idx = i

        # Find end of plateau
        while i < len(results) and abs(results[i].n_clusters - n_clusters) <= 2:
            i += 1

        plateau_width = i - start_idx

        if plateau_width >= min_plateau_width:
            plateaus.append(PlateauRegion(
                beta_start=results[start_idx].beta,
                beta_end=results[i-1].beta,
                n_clusters=int(np.median([results[j].n_clusters for j in range(start_idx, i)])),
                stability=plateau_width / len(results),
            ))

    return sorted(plateaus, key=lambda p: p.stability, reverse=True)


# ============================================================================
# MAP PHASE
# ============================================================================
def map_shard(shard_path: str, output_dir: Path, shard_id: str) -> dict:
    """
    MAP phase: Process a single shard and save intermediate results.
    """
    print(f"\n{'='*70}")
    print(f"MAP: Processing shard {shard_id}")
    print(f"{'='*70}")
    sys.stdout.flush()

    start_time = time.time()

    # Extract features (streaming)
    print(f"  Loading from {shard_path}...")
    sys.stdout.flush()

    features = []
    record_count = 0

    if shard_path.startswith('s3://'):
        record_iter = stream_s3_json(shard_path)
    else:
        record_iter = stream_json_records(shard_path)

    for record in record_iter:
        record_features = process_record(record)
        features.extend(record_features)
        record_count += 1

        if record_count % 500 == 0:
            print(f"    Processed {record_count} records -> {len(features)} features")
            sys.stdout.flush()

    extract_time = time.time() - start_time
    print(f"  Extracted {len(features)} features from {record_count} records ({extract_time:.1f}s)")
    sys.stdout.flush()

    if not features:
        print(f"  WARNING: No features extracted from shard {shard_id}")
        return {'shard_id': shard_id, 'status': 'empty'}

    # Encode features
    print(f"  Encoding features...")
    sys.stdout.flush()
    X, Y = encode_features(features)
    print(f"  Encoded matrix: {X.shape}")

    # Run IB annealing
    print(f"  Running IB annealing...")
    sys.stdout.flush()
    ib_results = anneal_beta(X, Y, beta_range=(0.01, 10.0), n_steps=30)

    # Find plateaus
    plateaus = find_plateaus(ib_results)
    print(f"  Found {len(plateaus)} plateaus")
    for p in plateaus[:5]:
        print(f"    β=[{p.beta_start:.3f}, {p.beta_end:.3f}]: {p.n_clusters} clusters (stability={p.stability:.2f})")
    sys.stdout.flush()

    # Save intermediate results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save features
    features_path = output_dir / f"features_{shard_id}.json"
    with open(features_path, 'w') as f:
        json.dump([f.to_dict() for f in features], f)

    # Save IB curve
    curve_path = output_dir / f"ib_curve_{shard_id}.json"
    curve_data = [{
        'beta': r.beta,
        'n_clusters': r.n_clusters,
        'i_xt': r.i_xt,
        'i_ty': r.i_ty,
        'ib_objective': r.ib_objective(),
    } for r in ib_results]
    with open(curve_path, 'w') as f:
        json.dump(curve_data, f, indent=2)

    # Save plateaus
    plateaus_path = output_dir / f"plateaus_{shard_id}.json"
    with open(plateaus_path, 'w') as f:
        json.dump([asdict(p) for p in plateaus], f, indent=2)

    total_time = time.time() - start_time
    print(f"  Shard {shard_id} complete in {total_time:.1f}s")
    sys.stdout.flush()

    return {
        'shard_id': shard_id,
        'status': 'success',
        'n_features': len(features),
        'n_records': record_count,
        'plateaus': [asdict(p) for p in plateaus[:3]],
        'time_sec': total_time,
    }


# ============================================================================
# REDUCE PHASE
# ============================================================================
def reduce_results(results_dir: Path, output_dir: Path) -> dict:
    """
    REDUCE phase: Merge results from all shards and find global plateaus.
    """
    print(f"\n{'='*70}")
    print("REDUCE: Merging shard results")
    print(f"{'='*70}")
    sys.stdout.flush()

    # Load all features
    print("  Loading features from all shards...")
    all_features = []
    feature_files = sorted(results_dir.glob("features_*.json"))

    for fpath in feature_files:
        print(f"    Loading {fpath.name}...")
        with open(fpath) as f:
            shard_features = json.load(f)
            all_features.extend([Feature(**d) for d in shard_features])

    print(f"  Total features: {len(all_features)}")
    sys.stdout.flush()

    # Load and merge IB curves
    print("  Merging IB curves...")
    all_curves = []
    curve_files = sorted(results_dir.glob("ib_curve_*.json"))

    for cpath in curve_files:
        with open(cpath) as f:
            all_curves.append(json.load(f))

    # Average I(X;T) and I(T;Y) across shards at each β
    merged_curve = []
    n_shards = len(all_curves)
    n_betas = len(all_curves[0]) if all_curves else 0

    for i in range(n_betas):
        beta = all_curves[0][i]['beta']
        avg_i_xt = np.mean([c[i]['i_xt'] for c in all_curves])
        avg_i_ty = np.mean([c[i]['i_ty'] for c in all_curves])
        avg_n_clusters = int(np.mean([c[i]['n_clusters'] for c in all_curves]))

        merged_curve.append({
            'beta': beta,
            'n_clusters': avg_n_clusters,
            'i_xt': avg_i_xt,
            'i_ty': avg_i_ty,
        })

    # Find global plateaus
    print("  Finding global plateaus...")
    ib_results = [IBResult(
        beta=c['beta'],
        n_clusters=c['n_clusters'],
        i_xt=c['i_xt'],
        i_ty=c['i_ty'],
        labels=np.array([]),  # Not needed for plateau detection
    ) for c in merged_curve]

    plateaus = find_plateaus(ib_results)

    print(f"\n  GLOBAL PLATEAUS (data-driven cluster counts):")
    print(f"  " + "-" * 60)
    for i, p in enumerate(plateaus[:5]):
        print(f"  #{i+1}: {p.n_clusters:3d} clusters | β=[{p.beta_start:.3f}, {p.beta_end:.3f}] | stability={p.stability:.2%}")
    sys.stdout.flush()

    # Use the most stable plateau
    if plateaus:
        optimal_k = plateaus[0].n_clusters
        print(f"\n  SELECTED: {optimal_k} clusters (most stable plateau)")
    else:
        optimal_k = 50  # Fallback
        print(f"\n  WARNING: No clear plateaus found. Using default k={optimal_k}")
    sys.stdout.flush()

    # Final clustering with optimal k
    print(f"\n  Running final clustering with k={optimal_k}...")
    X, Y = encode_features(all_features)

    if len(X) > 50000:
        # Use MiniBatchKMeans for very large datasets
        clusterer = MiniBatchKMeans(n_clusters=optimal_k, batch_size=2048, n_init=10, random_state=42)
    else:
        clusterer = MiniBatchKMeans(n_clusters=optimal_k, n_init=20, random_state=42)

    final_labels = clusterer.fit_predict(X)

    # Analyze templates
    print("  Analyzing discovered templates...")
    templates = defaultdict(list)
    for f, label in zip(all_features, final_labels):
        templates[int(label)].append(f)

    template_summaries = {}
    for tid, members in templates.items():
        ops = Counter(m.operator for m in members)
        rts = Counter(m.result_type for m in members)
        cats = Counter(m.category for m in members)

        top_op = ops.most_common(1)[0][0]
        top_rt = rts.most_common(1)[0][0]

        template_summaries[tid] = {
            'count': len(members),
            'top_operator': top_op,
            'top_result_type': top_rt,
            'top_category': cats.most_common(1)[0][0],
            'op_purity': ops.most_common(1)[0][1] / len(members),
            'rt_purity': rts.most_common(1)[0][1] / len(members),
            'name': f"T{tid}_{top_op}_{top_rt}",
            'examples': [m.text[:100] for m in members[:5]],
        }

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save templates
    with open(output_dir / "templates.json", 'w') as f:
        json.dump({str(k): v for k, v in template_summaries.items()}, f, indent=2)

    # Save classifier training data
    labeled_data = [{
        'text': f.text,
        'template_id': int(label),
        'template_name': template_summaries[int(label)]['name'],
        'operator': f.operator,
        'result_type': f.result_type,
        'category': f.category,
    } for f, label in zip(all_features, final_labels)]

    with open(output_dir / "classifier_labels.json", 'w') as f:
        json.dump(labeled_data, f)

    # Save plateau analysis
    with open(output_dir / "plateau_analysis.json", 'w') as f:
        json.dump({
            'merged_curve': merged_curve,
            'plateaus': [asdict(p) for p in plateaus],
            'selected_k': optimal_k,
            'selection_reason': 'most_stable_plateau',
        }, f, indent=2)

    # Save stats
    op_purities = [s['op_purity'] for s in template_summaries.values()]
    stats = {
        'total_features': len(all_features),
        'num_templates': len(template_summaries),
        'optimal_k': optimal_k,
        'mean_op_purity': float(np.mean(op_purities)),
        'min_op_purity': float(min(op_purities)),
        'plateaus_found': len(plateaus),
    }
    with open(output_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print("REDUCE COMPLETE")
    print(f"{'='*70}")
    print(f"  Total features: {len(all_features)}")
    print(f"  Templates discovered: {len(template_summaries)}")
    print(f"  Mean operator purity: {np.mean(op_purities):.2%}")
    print(f"\n  Top 10 templates:")
    sorted_templates = sorted(template_summaries.items(), key=lambda x: x[1]['count'], reverse=True)
    for tid, s in sorted_templates[:10]:
        print(f"    {s['name']:35} count={s['count']:5} purity={s['op_purity']:.2f}")
    print(f"\n  Results saved to: {output_dir}")
    sys.stdout.flush()

    return stats


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='MapReduce IB Template Discovery')
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # MAP subcommand
    map_parser = subparsers.add_parser('map', help='Process individual shards')
    map_parser.add_argument('--shard-paths', type=str, help='Comma-separated shard paths')
    map_parser.add_argument('--s3-prefix', type=str, help='S3 prefix for shards')
    map_parser.add_argument('--shard-ids', type=str, help='Comma-separated shard IDs (for S3)')
    map_parser.add_argument('--output-dir', type=str, default='/tmp/ib_map_results')

    # REDUCE subcommand
    reduce_parser = subparsers.add_parser('reduce', help='Merge shard results')
    reduce_parser.add_argument('--results-dir', type=str, required=True)
    reduce_parser.add_argument('--output-dir', type=str, default='/tmp/ib_final')

    # FULL subcommand (single-machine mode)
    full_parser = subparsers.add_parser('full', help='Full run on single machine')
    full_parser.add_argument('--data-dir', type=str, required=True)
    full_parser.add_argument('--output-dir', type=str, default='/tmp/ib_results')
    full_parser.add_argument('--max-shards', type=int, default=None, help='Limit shards for testing')

    args = parser.parse_args()

    if args.mode == 'map':
        output_dir = Path(args.output_dir)

        if args.shard_paths:
            # Local paths provided
            paths = args.shard_paths.split(',')
            for i, path in enumerate(paths):
                shard_id = Path(path).stem
                map_shard(path, output_dir, shard_id)

        elif args.s3_prefix and args.shard_ids:
            # S3 paths
            shard_ids = args.shard_ids.split(',')
            for sid in shard_ids:
                s3_path = f"{args.s3_prefix.rstrip('/')}/iaf_v3_gpu{sid}_valid.json"
                map_shard(s3_path, output_dir, f"gpu{sid}")

        else:
            print("ERROR: Must provide --shard-paths or --s3-prefix with --shard-ids")
            sys.exit(1)

    elif args.mode == 'reduce':
        reduce_results(Path(args.results_dir), Path(args.output_dir))

    elif args.mode == 'full':
        # Single-machine full run
        data_dir = Path(args.data_dir)
        output_dir = Path(args.output_dir)
        map_output = output_dir / "map_results"

        shards = sorted(data_dir.glob("*.json"))
        if args.max_shards:
            shards = shards[:args.max_shards]

        print(f"Found {len(shards)} shards")

        # Run MAP on all shards
        for shard in shards:
            map_shard(str(shard), map_output, shard.stem)

        # Run REDUCE
        reduce_results(map_output, output_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
