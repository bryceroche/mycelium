#!/usr/bin/env python3
"""
Generalize spans before deduplication.

ATTENTION-BASED GENERALIZATION (no hardcoded lists):
- Numbers → [N] (regex - syntactic detection)
- High attention entities → [ENTITY] (detected via attention signals)
- Everything else kept as-is (verbs emerge naturally)

This uses the SAME approach as dual_signal_pipeline.py at inference time.

This script:
1. Loads 17k labeled spans
2. Runs each through the model to get attention signals
3. Generalizes using attention (entities detected dynamically)
4. Deduplicates by pattern + embedding similarity
5. Assigns custom DSL based on operation + pattern structure
"""

import json
import re
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path


# Global detector (lazy loaded)
_detector = None


def get_detector():
    """Lazy load the span detector."""
    global _detector
    if _detector is None:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
        from mycelium.dual_signal_templates import SpanDetector
        _detector = SpanDetector()
    return _detector


def generalize_span_with_attention(span_text: str) -> str:
    """Generalize a span using attention signals - NO hardcoded word lists.

    SAME method used at inference time (dual_signal_pipeline.py).

    Uses attention signals to discriminate:
    - Numbers → [N] (regex)
    - High connectivity → verb/structural (KEEP - connects subject to object)
    - High attention_received + lower connectivity → entity (REPLACE with [ENTITY])
    - Everything else → keep as-is

    Args:
        span_text: Raw span text like "John has 5 apples"

    Returns:
        Generalized pattern like "[ENTITY] has [N] [ENTITY]"
    """
    # Clean up special tokens
    text = span_text.replace('[CLS]', '').replace('[SEP]', '').strip()

    if not text or len(text) < 3:
        return ""

    # Get attention signals from model
    detector = get_detector()
    embedding, attention, tokens = detector.extract_features(text)

    if attention.ndim > 2:
        attention = attention.mean(axis=0)

    n = min(len(tokens), attention.shape[0])
    if n == 0:
        return ""

    # Compute attention signals (same as dual_signal_pipeline.py)
    special_mask = np.array([
        (t.startswith('[') and t.endswith(']')) or t in '.!?,:;'
        for t in tokens[:n]
    ])

    # Attention received
    mask = 1 - np.eye(n)
    attn_received = (attention[:n, :n] * mask).sum(axis=0)

    # Connectivity (bidirectional attention strength)
    attn_given = (attention[:n, :n] * mask).sum(axis=1)
    connectivity = attn_received * attn_given

    # Normalize
    def normalize(arr):
        non_special = arr[~special_mask]
        if len(non_special) > 0 and non_special.max() > 0:
            return arr / non_special.max()
        return arr

    attn_received = normalize(attn_received)
    connectivity = normalize(connectivity)

    # Thresholds (same as inference)
    connectivity_threshold = 0.5
    entity_threshold = 0.6

    # Build generalized pattern
    result = []
    seen_entity = False

    for i in range(n):
        token = tokens[i]

        # Skip special tokens
        if token.startswith('[') and token.endswith(']'):
            continue
        if token in '.!?,:;':
            continue

        # Handle subword tokens
        if token.startswith('##'):
            token_clean = token[2:].lower()
        else:
            token_clean = token.lower().lstrip('$').rstrip('.,!?;:')

        # Numbers → [N]
        if re.match(r'^\d+\.?\d*%?$', token_clean):
            result.append('[N]')
            continue

        # Skip very short tokens
        if len(token_clean) <= 1:
            continue

        # Classification using attention signals
        is_verb = connectivity[i] >= connectivity_threshold
        is_entity = (
            attn_received[i] >= entity_threshold and
            connectivity[i] < connectivity_threshold
        )

        if is_verb:
            result.append(token_clean)
        elif is_entity:
            if not seen_entity:
                result.append('[ENTITY]')
                seen_entity = True
        else:
            result.append(token_clean)

    pattern = ' '.join(result)
    pattern = re.sub(r'\s+', ' ', pattern).strip()

    return pattern if len(pattern) > 2 else ""


def generalize_span(span_text: str) -> str:
    """Wrapper that uses attention-based generalization."""
    return generalize_span_with_attention(span_text)


def infer_operation_from_pattern(pattern: str) -> str:
    """Infer operation type from the generalized pattern.

    Based on verb semantics, not arbitrary labels.

    Args:
        pattern: Generalized pattern like "[SUBJ] sold [N]"

    Returns:
        Operation type: SET, ADD, SUB, MUL, or DIV
    """
    pattern_lower = pattern.lower()

    # MUL indicators (check first - "times" is strong signal)
    if any(v in pattern_lower for v in ['times', 'multiplied', 'doubled', 'tripled', 'twice', 'thrice']):
        return 'MUL'

    # DIV indicators
    if any(v in pattern_lower for v in ['divided', 'split', 'halved', 'half']):
        return 'DIV'

    # SUB indicators (loss/decrease verbs)
    if any(v in pattern_lower for v in [
        'sold', 'sells', 'gave', 'gives', 'spent', 'spends', 'paid', 'pays',
        'lost', 'loses', 'ate', 'eats', 'drank', 'drinks', 'used', 'uses',
        'removed', 'removes', 'donated', 'donates', 'traded', 'trades',
        'discarded', 'discards', 'shared', 'shares'
    ]):
        return 'SUB'

    # ADD indicators (gain/increase verbs)
    if any(v in pattern_lower for v in [
        'received', 'receives', 'found', 'finds', 'bought', 'buys',
        'purchased', 'purchases', 'got', 'gets', 'earned', 'earns',
        'won', 'wins', 'collected', 'collects', 'picked', 'picks',
        'gained', 'gains', 'added', 'adds', 'increased', 'increases',
        'grew', 'grows'
    ]):
        return 'ADD'

    # Default to SET (state/value establishment)
    return 'SET'


def deduplicate_patterns(spans: List[Dict]) -> Dict[str, Dict]:
    """Deduplicate spans by their generalized patterns.

    Groups by pattern, then infers operation from pattern content.

    Args:
        spans: List of span dicts with span_text, operation_type, etc.

    Returns:
        Dict mapping pattern -> template info
    """
    # Group by generalized pattern only (not by operation)
    groups = defaultdict(list)

    for span in spans:
        pattern = generalize_span(span['span_text'])
        if pattern:  # Skip empty patterns
            groups[pattern].append(span)

    # Create templates from groups
    templates = {}

    for pattern, group_spans in groups.items():
        # Skip patterns that are too short or just placeholders
        if len(pattern) < 3:
            continue

        # Skip patterns that are mostly placeholders
        words = pattern.split()
        placeholder_count = sum(1 for w in words if w.startswith('['))
        if len(words) > 0 and placeholder_count / len(words) > 0.7:
            continue

        # Infer operation from pattern content (not from original labels)
        op = infer_operation_from_pattern(pattern)

        # Create template ID from pattern
        template_id = pattern.replace(' ', '_').replace('[', '').replace(']', '')
        template_id = re.sub(r'[^a-z0-9_]', '', template_id.lower())[:50]
        template_id = f"{op.lower()}_{template_id}"

        # Get example spans (original text)
        examples = [s['span_text'].replace('[CLS]', '').replace('[SEP]', '').strip()
                    for s in group_spans[:5]]

        # Determine DSL based on operation and pattern
        dsl = determine_dsl(pattern, op)

        templates[template_id] = {
            'template_id': template_id,
            'operation': op,
            'pattern': pattern,
            'base_dsl': dsl,
            'count': len(group_spans),
            'pattern_examples': examples,
            'description': f"{op} operation: {pattern}"
        }

    return templates


def determine_dsl(pattern: str, operation: str) -> str:
    """Determine the DSL expression based on pattern and operation.

    Args:
        pattern: Generalized pattern like "[SUBJ] has [N] [ITEM]"
        operation: Operation type (SET, ADD, SUB, MUL, DIV)

    Returns:
        DSL expression like "entity + value" or "ref * 2"
    """
    pattern_lower = pattern.lower()

    # Check for reference patterns (comparisons)
    has_ref = '[obj]' in pattern_lower or 'than' in pattern_lower

    # Check for specific multipliers
    if 'twice' in pattern_lower or 'double' in pattern_lower:
        return 'ref * 2' if has_ref else 'entity * 2'
    if 'triple' in pattern_lower or 'thrice' in pattern_lower:
        return 'ref * 3' if has_ref else 'entity * 3'
    if 'half' in pattern_lower:
        return 'entity / 2'
    if 'quarter' in pattern_lower:
        return 'entity / 4'

    # Check for relative patterns
    if 'more than' in pattern_lower:
        return 'ref + value'
    if 'less than' in pattern_lower or 'fewer than' in pattern_lower:
        return 'ref - value'
    if 'times as' in pattern_lower or 'times more' in pattern_lower:
        return 'ref * value'

    # Default DSL based on operation
    dsl_defaults = {
        'SET': 'value',
        'ADD': 'entity + value',
        'SUB': 'entity - value',
        'MUL': 'entity * value',
        'DIV': 'entity / value'
    }

    return dsl_defaults.get(operation, 'value')


def load_specialized_templates(path: Path) -> List[Dict]:
    """Load and convert specialized_templates.json format."""
    with open(path) as f:
        data = json.load(f)

    spans = []
    for template_id, tpl in data.items():
        spans.append({
            'span_text': tpl.get('pattern', ''),
            'operation_type': tpl.get('operation_type', 'SET'),
            'original_dsl': tpl.get('dsl_expr', 'value'),
            'template_id': template_id
        })
    return spans


def embed_patterns_batch(patterns: List[str], batch_size: int = 32) -> List[List[float]]:
    """Embed patterns using MiniLM for similarity-based deduplication."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

    from mycelium.dual_signal_templates import SpanDetector
    import numpy as np

    print("Loading embedding model...")
    detector = SpanDetector()

    embeddings = []
    for i in range(0, len(patterns), batch_size):
        batch = patterns[i:i + batch_size]
        if i % 500 == 0:
            print(f"  Embedding {i}/{len(patterns)}...")
        for pattern in batch:
            emb, _, _ = detector.extract_features(pattern)
            embeddings.append(emb.tolist())

    return embeddings


def deduplicate_by_embedding(
    templates: List[Dict],
    similarity_threshold: float = 0.85
) -> List[Dict]:
    """Deduplicate templates by embedding similarity within each operation type.

    Groups templates by operation, then clusters by embedding similarity.
    Keeps the most common pattern from each cluster.

    Args:
        templates: List of template dicts with 'pattern', 'operation', etc.
        similarity_threshold: Cosine similarity threshold for clustering

    Returns:
        Deduplicated list of templates
    """
    import numpy as np
    from collections import defaultdict

    # Group by operation first (never merge different operations)
    by_op = defaultdict(list)
    for t in templates:
        by_op[t['operation']].append(t)

    deduplicated = []

    for op, op_templates in by_op.items():
        print(f"\n  Processing {op} ({len(op_templates)} patterns)...")

        # Get patterns and their embeddings
        patterns = [t['pattern'] for t in op_templates]
        embeddings = embed_patterns_batch(patterns)
        embeddings = np.array(embeddings)

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings_norm = embeddings / norms

        # Simple greedy clustering by similarity
        used = set()
        clusters = []

        # Sort by count (most common first) to pick best representatives
        sorted_indices = sorted(range(len(op_templates)), key=lambda i: -op_templates[i]['count'])

        for idx in sorted_indices:
            if idx in used:
                continue

            # Start new cluster with this template
            cluster = [idx]
            used.add(idx)

            # Find all similar templates
            for other_idx in range(len(op_templates)):
                if other_idx in used:
                    continue
                sim = np.dot(embeddings_norm[idx], embeddings_norm[other_idx])
                if sim >= similarity_threshold:
                    cluster.append(other_idx)
                    used.add(other_idx)

            clusters.append(cluster)

        print(f"    Clustered into {len(clusters)} groups")

        # Keep the most common pattern from each cluster
        for cluster in clusters:
            # Sort by count within cluster
            best_idx = max(cluster, key=lambda i: op_templates[i]['count'])
            template = op_templates[best_idx].copy()

            # Aggregate count from cluster
            template['count'] = sum(op_templates[i]['count'] for i in cluster)
            template['cluster_size'] = len(cluster)

            # Store the embedding centroid
            cluster_embeddings = embeddings[cluster]
            template['embedding_centroid'] = np.mean(cluster_embeddings, axis=0).tolist()

            deduplicated.append(template)

    return deduplicated


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Generalize spans before deduplication')
    parser.add_argument('--input', default='specialized_templates.json',
                        help='Input templates file (specialized_templates.json or labeled_spans.json)')
    parser.add_argument('--output', default='deduplicated_templates.json',
                        help='Output templates file')
    parser.add_argument('--min-count', type=int, default=1,
                        help='Minimum span count to include template')
    parser.add_argument('--similarity-threshold', type=float, default=0.85,
                        help='Cosine similarity threshold for clustering (0.0-1.0)')
    parser.add_argument('--skip-embedding-dedup', action='store_true',
                        help='Skip embedding-based deduplication')
    args = parser.parse_args()

    # Load spans
    input_path = Path(__file__).parent.parent / args.input
    print(f"Loading spans from {input_path}...")

    with open(input_path) as f:
        raw_data = json.load(f)

    # Detect format and load appropriately
    if isinstance(raw_data, dict):
        # specialized_templates.json format
        spans = load_specialized_templates(input_path)
        print(f"Loaded {len(spans)} templates from specialized format")
    else:
        # labeled_spans.json format
        spans = raw_data
        print(f"Loaded {len(spans)} spans from labeled format")

    # Generalize and deduplicate by exact pattern
    print("Generalizing and deduplicating by pattern...")
    templates = deduplicate_patterns(spans)

    # Filter by minimum count
    if args.min_count > 1:
        templates = {k: v for k, v in templates.items() if v['count'] >= args.min_count}

    # Sort by count (most common first)
    sorted_templates = sorted(templates.values(), key=lambda x: -x['count'])

    print(f"\nAfter pattern deduplication: {len(sorted_templates)} templates")

    # Embedding-based deduplication
    if not args.skip_embedding_dedup:
        print("\nDoing embedding-based deduplication...")
        sorted_templates = deduplicate_by_embedding(sorted_templates, args.similarity_threshold)
        print(f"\nAfter embedding deduplication: {len(sorted_templates)} templates")

    # Sort by count again
    sorted_templates = sorted(sorted_templates, key=lambda x: -x['count'])

    # Print stats
    print(f"\nFinal Results:")
    print(f"  Total templates: {len(sorted_templates)}")

    op_counts = defaultdict(int)
    for t in sorted_templates:
        op_counts[t['operation']] += 1

    print(f"  By operation:")
    for op, count in sorted(op_counts.items()):
        print(f"    {op}: {count}")

    # Show top 20 templates
    print(f"\nTop 20 most common patterns:")
    for t in sorted_templates[:20]:
        print(f"  [{t['operation']:3}] {t['pattern'][:50]:50} (n={t['count']:4}, cluster={t.get('cluster_size', 1):3}) DSL: {t['base_dsl']}")

    # Save output
    output_path = Path(__file__).parent.parent / args.output
    with open(output_path, 'w') as f:
        json.dump(sorted_templates, f, indent=2)

    print(f"\nSaved {len(sorted_templates)} templates to {output_path}")


if __name__ == '__main__':
    main()
