#!/usr/bin/env python3
"""Rebuild span pipeline with corrected coarse-grained spans.

Number cascade:
  7,473 GSM8K problems
    → 61K coarse spans (Qwen hidden-state boundaries)
      → ~150K fine-grained atoms (vocab-reduce + atomic-split)
        → ~1K atomic templates (cosine threshold collapse)
          → ~1K SubGraphDSLs (Claude Opus fleet)

  61K coarse spans (separately)
    → ~200-500 coarse templates (cosine collapse)
      → each maps to sequence of atomic templates

Usage:
    python scripts/rebuild_pipeline.py --step vocab-reduce
    python scripts/rebuild_pipeline.py --step atomic-split
    python scripts/rebuild_pipeline.py --step collapse
    python scripts/rebuild_pipeline.py --step generate-dsls
    python scripts/rebuild_pipeline.py --step coarse-templates
    python scripts/rebuild_pipeline.py --step finetune
    python scripts/rebuild_pipeline.py --step all
"""

import argparse
import asyncio
import json
import re
import sys
import time
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"

QWEN_MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"

# Global vLLM engine (loaded once, reused across steps)
_vllm_engine = None


def get_vllm_engine(tp_size: int = 4):
    """Load or return cached vLLM engine."""
    global _vllm_engine
    if _vllm_engine is None:
        from vllm import LLM
        print(f"\nLoading Qwen via vLLM (TP={tp_size})...")
        _vllm_engine = LLM(
            model=QWEN_MODEL,
            tensor_parallel_size=tp_size,
            trust_remote_code=True,
            max_model_len=2048,
            gpu_memory_utilization=0.85,
        )
        print("  Model loaded.")
    return _vllm_engine


def unload_vllm():
    """Free GPU memory."""
    global _vllm_engine
    if _vllm_engine is not None:
        import gc, torch
        del _vllm_engine
        _vllm_engine = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# =============================================================================
# Shared utilities
# =============================================================================

def call_vllm_batch(prompts: List[str], max_tokens: int = 100,
                    temperature: float = 0.1, tp_size: int = 4) -> List[str]:
    """Native vLLM batch inference — sends all prompts at once."""
    from vllm import SamplingParams

    engine = get_vllm_engine(tp_size)
    params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["\n\n", "```"],
    )

    t0 = time.time()
    print(f"  Generating {len(prompts)} completions...")
    outputs = engine.generate(prompts, params)
    elapsed = time.time() - t0
    rate = len(prompts) / elapsed if elapsed > 0 else 0
    print(f"  Done: {len(prompts)} in {elapsed:.1f}s ({rate:.0f} req/s)")

    return [o.outputs[0].text.strip() for o in outputs]


def greedy_cluster(centroids_norm: np.ndarray, threshold: float) -> list:
    """Greedy cosine similarity clustering (from find_1k_threshold.py)."""
    n = len(centroids_norm)
    used = set()
    clusters = []

    for i in range(n):
        if i in used:
            continue
        cluster = [i]
        used.add(i)

        remaining = [j for j in range(i + 1, n) if j not in used]
        if remaining:
            remaining_arr = np.array(remaining)
            sims = centroids_norm[remaining_arr] @ centroids_norm[i]
            for idx, sim in zip(remaining_arr, sims):
                if sim >= threshold:
                    cluster.append(idx)
                    used.add(idx)

        clusters.append(cluster)

    return clusters


def binary_search_threshold(embeddings_norm: np.ndarray, target_count: int,
                            lo: float = 0.5, hi: float = 0.99,
                            max_iter: int = 30) -> float:
    """Binary search for cosine threshold yielding ~target_count clusters."""
    best_thresh = lo
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        clusters = greedy_cluster(embeddings_norm, mid)
        count = len(clusters)
        if count > target_count:
            lo = mid
        else:
            hi = mid
        best_thresh = mid
        if abs(count - target_count) < 50:
            break
    return best_thresh


# =============================================================================
# Step 1: Vocab-reduce coarse spans
# =============================================================================

FORCE_VOCAB = {
    'has', 'have', 'had', 'gave', 'gives', 'give', 'got', 'gets', 'get',
    'bought', 'buys', 'buy', 'sold', 'sells', 'sell',
    'spent', 'spends', 'spend', 'paid', 'pays', 'pay',
    'earned', 'earns', 'earn', 'makes', 'made', 'make',
    'ate', 'eats', 'eat', 'drinks', 'drank', 'drink',
    'lost', 'loses', 'lose', 'found', 'finds', 'find',
    'took', 'takes', 'take', 'added', 'adds', 'add',
    'left', 'remaining', 'rest',
    'each', 'every', 'per', 'total', 'all',
    'more', 'less', 'fewer', 'extra', 'additional',
    'twice', 'half', 'double', 'triple', 'third', 'quarter',
    'times', 'percent', 'fraction',
    'cost', 'costs', 'price', 'worth', 'value',
    'how', 'many', 'much', 'what', 'number',
    'than', 'as', 'if', 'then', 'after', 'before',
    'first', 'second', 'third', 'last', 'next',
    'day', 'week', 'month', 'year', 'hour', 'minute',
    'split', 'divide', 'divided', 'share', 'shared',
    'together', 'combined', 'equal', 'equally',
    'needs', 'need', 'wants', 'want',
    'from', 'to', 'for', 'with', 'on', 'in', 'at', 'by',
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'will', 'be',
    'and', 'or', 'but', 'not', 'no',
    'he', 'she', 'it', 'they', 'his', 'her', 'their', 'its',
    'of', 'that', 'this', 'there', 'does', 'did', 'do',
    'can', 'one', 'two', 'three', 'four', 'five',
    'six', 'seven', 'eight', 'nine', 'ten',
    'some', 'same', 'other', 'another', 'both',
    'now', 'still', 'also', 'only', 'already',
    'person', 'people', 'items', 'things', 'N',
}


def build_vocab(spans: List[dict], vocab_size: int = 300) -> List[str]:
    """Build top-N vocabulary from span texts."""
    word_counts = Counter()
    for s in spans:
        words = re.findall(r'[a-z]+', s.get('span_text', '').lower())
        word_counts.update(words)

    top_words = [w for w, _ in word_counts.most_common(vocab_size)]
    vocab = list(set(top_words) | FORCE_VOCAB)
    vocab.sort(key=lambda w: word_counts.get(w, 0), reverse=True)
    return vocab


REWRITE_PROMPT_TEMPLATE = """Rewrite this math problem span as a short summary using ONLY common words.
Rules:
- Replace all names with "person"
- Replace all specific objects with "items"
- Replace all numbers/amounts with N
- Replace locations with "place"
- Keep the math operation words (bought, sold, gave, earned, split, each, per, more, less, twice, half, etc.)
- Make it as SHORT as possible while preserving what arithmetic operation happens
- Use ONLY these allowed words: {vocab_words}
- Output ONLY the rewritten span, nothing else

Examples:
"Janet's ducks lay 16 eggs per day" → person has N items per day
"She eats three for breakfast every morning" → person eats N each day
"He buys 3 sodas at $2 each" → person buys N items at N each
"The cost of the dessert is 25% of the price of the second course" → items cost N percent of items
"How many apples does John have left?" → how many items does person have left

Span: "{span}"
Summary: """


def clean_rewrite(raw: str) -> str:
    """Clean Qwen's rewrite output."""
    p = raw.strip()
    p = p.split('\n')[0].strip()
    p = p.strip('"\'')
    p = re.sub(r'^(Summary:\s*|Output:\s*|Rewrite:\s*)', '', p, flags=re.IGNORECASE)
    p = p.strip('"\'')
    p = ' '.join(p.split())
    p = p.lower()
    return p.strip()


def is_valid_rewrite(rewrite: str, raw_span: str) -> bool:
    """Check if rewrite is valid."""
    if not rewrite or len(rewrite) < 3:
        return False
    if len(rewrite) > len(raw_span) * 1.5:
        return False
    if not re.search(r'[a-z]', rewrite):
        return False
    return True


def programmatic_reduce(span: str, vocab_set: set) -> str:
    """Fallback: programmatic vocab reduction without Qwen."""
    result = re.sub(r'\d+[\d,.]*', 'N', span.lower())
    result = re.sub(r'\$\s*N', 'N', result)
    tokens = result.split()
    reduced = []
    for t in tokens:
        clean_t = re.sub(r'[^a-z]', '', t)
        if clean_t in vocab_set or t == 'N' or not clean_t:
            reduced.append(t)
    return ' '.join(reduced) if reduced else 'items'


def step_vocab_reduce(args):
    """Step 1: Vocab-reduce coarse spans via vLLM API."""
    print("=" * 60)
    print("STEP 1: Vocab-reduce coarse spans")
    print("=" * 60)

    # Load coarse spans
    input_path = DATA_DIR / "qwen_coarse_spans.json"
    print(f"\nLoading from {input_path}...")
    with open(input_path) as f:
        data = json.load(f)
    spans = data["spans"] if isinstance(data, dict) else data
    print(f"Loaded {len(spans)} coarse spans")

    # Check for checkpoint
    output_path = DATA_DIR / "vocab_reduced_coarse.json"
    if output_path.exists() and not args.force:
        print(f"Checkpoint found at {output_path}, loading...")
        with open(output_path) as f:
            existing = json.load(f)
        done_count = sum(1 for r in existing if r.get("reduced"))
        print(f"  {done_count}/{len(existing)} already reduced")
        if done_count == len(existing):
            print("  All done, skipping.")
            return existing
        spans = existing

    # Build vocab
    print("\nBuilding vocabulary...")
    vocab = build_vocab(spans, vocab_size=300)
    print(f"  Vocabulary size: {len(vocab)}")
    vocab_words = ", ".join(vocab[:200])
    vocab_set = set(vocab)

    # Build prompts for spans that need reduction
    to_reduce = [(i, s) for i, s in enumerate(spans) if not s.get("reduced")]
    print(f"\nReducing {len(to_reduce)} spans via vLLM...")

    prompts = [
        REWRITE_PROMPT_TEMPLATE.format(vocab_words=vocab_words, span=s["span_text"])
        for _, s in to_reduce
    ]

    # Process in batches
    responses = call_vllm_batch(prompts, max_tokens=80)

    failures = 0
    for (idx, span), response in zip(to_reduce, responses):
        rewrite = clean_rewrite(response)
        if is_valid_rewrite(rewrite, span["span_text"]):
            spans[idx]["reduced"] = rewrite
        else:
            spans[idx]["reduced"] = programmatic_reduce(span["span_text"], vocab_set)
            failures += 1

    # Stats
    patterns = Counter(s.get("reduced", "") for s in spans)
    print(f"\nVocab-reduce results:")
    print(f"  Total spans: {len(spans)}")
    print(f"  Unique patterns: {len(patterns)}")
    print(f"  Failures (programmatic fallback): {failures}")
    print(f"  Top 15 patterns:")
    for pat, count in patterns.most_common(15):
        print(f"    [{count:5d}] {pat}")

    # Save
    with open(output_path, "w") as f:
        json.dump(spans, f)
    print(f"\nSaved to {output_path}")

    return spans


# =============================================================================
# Step 2: Atomic-split vocab-reduced spans
# =============================================================================

SPLIT_PROMPT = """Split this math problem span into atomic operations. Each atomic span should describe ONE thing:
- An assignment (person has N)
- A transfer (person gives N, person receives N)
- A rate/unit (per day, each hour, N each)
- A multiplication (N times, N groups of)
- A comparison (N more than, N less than, twice as many)
- A fraction (half of, N percent of)
- A query (how many, how much, what total)
- A condition (if, remaining, left over)

Rules:
- One atomic operation per line
- Keep it SHORT (2-5 words each)
- Use the same words from the input (person, items, N, place, etc.)
- Do NOT add new words or explanations
- If the span is already atomic (2-5 words), output it unchanged
- Output ONLY the atomic spans, one per line, nothing else

Examples:
"person buys n items at n each" →
person buys n items
at n each

"person has n more items than person" →
person has items
n more than person

"how much money does person have left" →
how much money
person has left

"person earns n per hour for n hours" →
person earns n
per hour
for n hours

"person has n items" →
person has n items

"per day" →
per day

Input: "{span}"
Atomic spans:
"""


def parse_atomic_spans(response: str, original_span: str) -> List[str]:
    """Parse Qwen's response into a list of atomic spans."""
    lines = response.strip().split('\n')
    atoms = []
    for line in lines:
        line = line.strip()
        line = re.sub(r'^[\d]+[.)]\s*', '', line)
        line = re.sub(r'^[-*]\s*', '', line)
        line = line.strip('"\'')
        line = ' '.join(line.split()).lower()
        if not line or len(line) < 2:
            continue
        if len(line) > len(original_span) * 1.5:
            continue
        if not re.search(r'[a-z]', line):
            continue
        atoms.append(line)

    if not atoms:
        atoms = [original_span.lower().strip()]

    return atoms


def filter_atoms(atoms: List[dict]) -> List[dict]:
    """Remove garbage atoms."""
    filtered = []
    rejected = 0
    reject_reasons = Counter()

    for a in atoms:
        text = a["atomic"].strip()

        if any(leak in text for leak in [
            "ai assistant", "your response", "generate", "atomic spans",
            "atomic span", "input:", "output:", "example", "summary:",
            "rewrite", "rules:", "split this",
        ]):
            rejected += 1
            reject_reasons["prompt_leakage"] += 1
            continue

        if len(text) < 4 or len(text) > 80:
            rejected += 1
            reject_reasons["bad_length"] += 1
            continue

        if not re.search(r'[a-z]', text):
            rejected += 1
            reject_reasons["no_alpha"] += 1
            continue

        words = text.split()
        if len(words) < 2:
            rejected += 1
            reject_reasons["single_word"] += 1
            continue

        if text in {"and then", "and also", "at the", "in the", "on the",
                     "of the", "for the", "to the", "with the", "from the",
                     "is the", "are the", "was the", "has the", "had the",
                     "it is", "there is", "there are", "that is", "this is"}:
            rejected += 1
            reject_reasons["filler"] += 1
            continue

        filtered.append(a)

    print(f"  Filtered: {len(atoms)} → {len(filtered)} ({rejected} rejected)")
    for reason, count in reject_reasons.most_common():
        print(f"    {reason}: {count}")
    return filtered


def step_atomic_split(args):
    """Step 2: Atomic-split vocab-reduced spans via vLLM API."""
    print("\n" + "=" * 60)
    print("STEP 2: Atomic-split vocab-reduced spans")
    print("=" * 60)

    # Load vocab-reduced spans
    input_path = DATA_DIR / "vocab_reduced_coarse.json"
    print(f"\nLoading from {input_path}...")
    with open(input_path) as f:
        spans = json.load(f)
    print(f"Loaded {len(spans)} spans")

    # Check checkpoint
    output_path = DATA_DIR / "atomic_spans_v2.json"
    if output_path.exists() and not args.force:
        print(f"Checkpoint found at {output_path}, skipping.")
        with open(output_path) as f:
            return json.load(f)

    # Build prompts
    items = []
    for i, s in enumerate(spans):
        reduced = s.get("reduced", s.get("span_text", ""))
        if reduced:
            items.append((i, s, reduced))

    print(f"\nSplitting {len(items)} spans into atomic operations...")

    prompts = [SPLIT_PROMPT.format(span=span) for _, _, span in items]
    responses = call_vllm_batch(prompts, max_tokens=120)

    all_atoms = []
    failures = 0
    for (idx, record, original_span), response in zip(items, responses):
        atoms = parse_atomic_spans(response, original_span)

        for atom in atoms:
            all_atoms.append({
                "parent_idx": idx,
                "parent_problem_id": record.get("problem_id", ""),
                "parent_span_idx": record.get("span_idx", 0),
                "parent_coarse_text": record.get("span_text", ""),
                "reduced": original_span,
                "atomic": atom,
                # Inherit attention signals from parent coarse span
                "connectivity": record.get("connectivity", 0.0),
                "backward_attention": record.get("backward_attention", 0.0),
                "span_position": record.get("span_position", 0.0),
            })

        if len(atoms) == 1 and atoms[0] == original_span.lower().strip():
            failures += 1

    # Filter garbage
    print(f"\nBefore filter: {len(all_atoms)} atoms")
    all_atoms = filter_atoms(all_atoms)
    print(f"After filter: {len(all_atoms)} atoms")

    # Stats
    atom_patterns = Counter(a["atomic"] for a in all_atoms)
    print(f"\nAtomic split results:")
    print(f"  Input spans: {len(items)}")
    print(f"  Output atoms: {len(all_atoms)}")
    print(f"  Avg atoms/span: {len(all_atoms) / max(1, len(items)):.2f}")
    print(f"  Unique atom patterns: {len(atom_patterns)}")
    print(f"  Unchanged: {failures}")
    print(f"  Top 20 atom patterns:")
    for pat, count in atom_patterns.most_common(20):
        print(f"    [{count:5d}] {pat}")

    # Save
    with open(output_path, "w") as f:
        json.dump(all_atoms, f)
    print(f"\nSaved {len(all_atoms)} atoms to {output_path}")

    return all_atoms


# =============================================================================
# Step 3: Collapse atomic spans at cosine threshold
# =============================================================================

def step_collapse(args):
    """Step 3: Embed + sklearn clustering → ~1K atomic templates."""
    print("\n" + "=" * 60)
    print("STEP 3: Collapse atomic spans at cosine threshold")
    print("=" * 60)

    # Load atoms
    input_path = DATA_DIR / "atomic_spans_v2.json"
    print(f"\nLoading from {input_path}...")
    with open(input_path) as f:
        all_atoms = json.load(f)
    print(f"Loaded {len(all_atoms)} atoms")

    # Check checkpoint
    output_path = DATA_DIR / "atomic_templates_v2.json"
    if output_path.exists() and not args.force:
        print(f"Checkpoint found at {output_path}, skipping.")
        with open(output_path) as f:
            return json.load(f)

    # Group by atomic pattern
    pattern_groups = defaultdict(list)
    for a in all_atoms:
        pattern_groups[a["atomic"]].append(a)

    patterns = list(pattern_groups.keys())
    print(f"  Unique patterns: {len(patterns)}")

    # Embed with MiniLM
    print(f"\nEmbedding {len(patterns)} patterns with MiniLM...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embeddings = model.encode(patterns, batch_size=256, show_progress_bar=True,
                              normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Use sklearn AgglomerativeClustering for efficient clustering
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_distances

    target_count = args.target_templates
    print(f"\nClustering {len(patterns)} patterns into ~{target_count} clusters via sklearn...")

    # Compute cosine distance matrix (1 - cosine_similarity)
    print("  Computing cosine distance matrix...")
    dist_matrix = cosine_distances(embeddings)
    print(f"  Distance matrix shape: {dist_matrix.shape}")

    # Agglomerative clustering with precomputed distance
    print(f"  Running AgglomerativeClustering (n_clusters={target_count})...")
    clustering = AgglomerativeClustering(
        n_clusters=target_count,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(dist_matrix)

    # Group indices by cluster label
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
    clusters = list(clusters.values())
    print(f"  Got {len(clusters)} clusters")

    # Build templates
    templates = []
    for cid, member_indices in enumerate(clusters):
        # Pick most common pattern as representative
        rep_idx = max(member_indices, key=lambda i: len(pattern_groups[patterns[i]]))
        rep_pattern = patterns[rep_idx]

        # Collect all atoms in this cluster
        cluster_atoms = []
        all_raw_spans = []
        all_patterns = []
        connectivities = []
        backward_attns = []
        span_positions = []
        parent_refs = []

        for mi in member_indices:
            p = patterns[mi]
            all_patterns.append(p)
            for a in pattern_groups[p]:
                cluster_atoms.append(a)
                all_raw_spans.append(a.get("parent_coarse_text", ""))
                connectivities.append(a.get("connectivity", 0.0))
                backward_attns.append(a.get("backward_attention", 0.0))
                span_positions.append(a.get("span_position", 0.0))
                parent_refs.append({
                    "problem_id": a.get("parent_problem_id", ""),
                    "span_idx": a.get("parent_span_idx", 0),
                    "coarse_text": a.get("parent_coarse_text", ""),
                })

        # Average centroid
        centroid = np.mean(embeddings[member_indices], axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        templates.append({
            "template_id": f"atom_v2_{cid:04d}",
            "pattern": rep_pattern,
            "all_patterns": all_patterns[:20],
            "centroid": centroid.tolist(),
            "member_count": len(cluster_atoms),
            "span_examples": list(set(all_raw_spans))[:20],
            "avg_connectivity": float(np.mean(connectivities)) if connectivities else 0.0,
            "avg_backward_attention": float(np.mean(backward_attns)) if backward_attns else 0.0,
            "avg_span_position": float(np.mean(span_positions)) if span_positions else 0.0,
            "parent_coarse_spans": parent_refs[:50],  # Cap for file size
        })

    # Sort by member count
    templates.sort(key=lambda t: t["member_count"], reverse=True)

    # Stats
    sizes = [t["member_count"] for t in templates]
    print(f"\n  Templates: {len(templates)}")
    print(f"  Total atoms assigned: {sum(sizes)}")
    print(f"  Mean size: {np.mean(sizes):.1f}")
    print(f"  Median size: {np.median(sizes):.0f}")
    print(f"  Singletons: {sum(1 for s in sizes if s == 1)}")
    print(f"\n  Top 20 templates:")
    for t in templates[:20]:
        print(f"    [{t['member_count']:5d}] {t['pattern']}")

    # Attention signal ranges
    print(f"\n  Attention signal ranges:")
    print(f"    connectivity: {min(t['avg_connectivity'] for t in templates):.3f} - {max(t['avg_connectivity'] for t in templates):.3f}")
    print(f"    backward_attn: {min(t['avg_backward_attention'] for t in templates):.3f} - {max(t['avg_backward_attention'] for t in templates):.3f}")
    print(f"    span_position: {min(t['avg_span_position'] for t in templates):.3f} - {max(t['avg_span_position'] for t in templates):.3f}")

    # Save
    with open(output_path, "w") as f:
        json.dump(templates, f)
    print(f"\nSaved {len(templates)} templates to {output_path}")

    return templates


# =============================================================================
# Step 4: Generate DSLs via Qwen (vLLM)
# =============================================================================

DSL_PROMPT_QWEN = """You are writing a SubGraphDSL for a math word problem pattern.

A SubGraphDSL defines the computation a span performs as JSON:
- "params": values extracted from the span text (numbers: n1, n2, ...)
- "inputs": values from upstream spans (use "upstream" for running entity value)
- "steps": ordered computation steps, each with "var", "op", "args"
- "output": which variable is exposed downstream

Allowed operators: SET (1 arg), ADD (2 args), SUB (2 args), MUL (2 args), DIV (2 args), NEG (1 arg)
Args can be variable names (strings) or literal numbers (floats).

IMPORTANT: Identify the COMPUTATION, not just values.
- "N items at N each" → MUL, NOT SET
- "gave N items" → SUB with upstream, NOT SET
- "N more than" → ADD with upstream, NOT SET
- "half of" → DIV by 2 with upstream, NOT SET
- "N percent" → MUL with upstream and DIV by 100, NOT SET

Output ONLY valid JSON. No explanation.

Examples:

Pattern: "person has N items"
{{"params": {{"n1": "quantity"}}, "inputs": {{}}, "steps": [{{"var": "out", "op": "SET", "args": ["n1"]}}], "output": "out"}}

Pattern: "person gave N items to person"
{{"params": {{"n1": "quantity given"}}, "inputs": {{"upstream": "giver total"}}, "steps": [{{"var": "out", "op": "SUB", "args": ["upstream", "n1"]}}], "output": "out"}}

Pattern: "person buys N items at N each"
{{"params": {{"n1": "quantity", "n2": "price each"}}, "inputs": {{}}, "steps": [{{"var": "out", "op": "MUL", "args": ["n1", "n2"]}}], "output": "out"}}

Pattern: "person earns N percent more"
{{"params": {{"n1": "percentage"}}, "inputs": {{"upstream": "base amount"}}, "steps": [{{"var": "pct", "op": "DIV", "args": ["n1", 100]}}, {{"var": "bonus", "op": "MUL", "args": ["upstream", "pct"]}}, {{"var": "out", "op": "ADD", "args": ["upstream", "bonus"]}}], "output": "out"}}

Pattern: "N items split equally N"
{{"params": {{"n1": "total", "n2": "groups"}}, "inputs": {{}}, "steps": [{{"var": "out", "op": "DIV", "args": ["n1", "n2"]}}], "output": "out"}}

Pattern: "how many items does person have left"
{{"params": {{}}, "inputs": {{"upstream": "current total"}}, "steps": [{{"var": "out", "op": "SET", "args": ["upstream"]}}], "output": "out"}}

Pattern: "{pattern}"
Example spans:
{examples}
"""


def parse_dsl_json(response: str) -> Optional[dict]:
    """Parse JSON from Qwen's response."""
    cleaned = response.strip()
    brace_start = cleaned.find('{')
    if brace_start < 0:
        return None
    depth = 0
    for i in range(brace_start, len(cleaned)):
        if cleaned[i] == '{':
            depth += 1
        elif cleaned[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[brace_start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def step_generate_dsls(args):
    """Step 4: Generate SubGraphDSLs via Qwen (vLLM)."""
    from mycelium.subgraph_dsl import SubGraphDSL, SubGraphStep

    print("\n" + "=" * 60)
    print("STEP 4: Generate DSLs via Qwen (vLLM)")
    print("=" * 60)

    # Load templates
    input_path = DATA_DIR / "atomic_templates_v2.json"
    print(f"\nLoading from {input_path}...")
    with open(input_path) as f:
        templates = json.load(f)
    print(f"Loaded {len(templates)} templates")

    # Check checkpoint
    output_path = DATA_DIR / "atomic_templates_v2_with_dsl.json"
    if output_path.exists() and not args.force:
        print(f"Checkpoint found at {output_path}")
        with open(output_path) as f:
            existing = json.load(f)
        need_dsl = [t for t in existing if not t.get("subgraph")]
        if not need_dsl:
            print("  All templates have DSLs, skipping.")
            return existing
        print(f"  {len(need_dsl)} templates still need DSLs, resuming...")
        templates = existing

    to_process = [(i, t) for i, t in enumerate(templates) if not t.get("subgraph")]
    print(f"\nGenerating DSLs for {len(to_process)} templates via Qwen...")

    # Build prompts
    prompts = []
    for idx, tpl in to_process:
        examples = tpl.get("span_examples", [])[:5]
        examples_str = "\n".join(f'  "{ex}"' for ex in examples) or "  (no examples)"
        prompts.append(DSL_PROMPT_QWEN.format(
            pattern=tpl["pattern"],
            examples=examples_str,
        ))

    # Call vLLM in native batch mode
    responses = call_vllm_batch(prompts, max_tokens=300)

    valid_count = 0
    fallback_count = 0

    for (idx, tpl), response in zip(to_process, responses):
        parsed = parse_dsl_json(response)

        if parsed:
            parsed["template_id"] = tpl["template_id"]
            parsed["pattern"] = tpl["pattern"]
            try:
                dsl = SubGraphDSL(
                    template_id=parsed["template_id"],
                    pattern=parsed.get("pattern", ""),
                    params=parsed.get("params", {}),
                    inputs=parsed.get("inputs", {}),
                    steps=[SubGraphStep.from_dict(s) for s in parsed.get("steps", [])],
                    output=parsed.get("output", "out"),
                )
                errors = dsl.validate()
                if not errors:
                    templates[idx]["subgraph"] = dsl.to_dict()
                    valid_count += 1
                    continue
            except Exception:
                pass

        # Fallback to SET
        fb = {
            "template_id": tpl["template_id"],
            "pattern": tpl["pattern"],
            "params": {"n1": "value"},
            "inputs": {},
            "steps": [{"var": "out", "op": "SET", "args": ["n1"]}],
            "output": "out",
        }
        templates[idx]["subgraph"] = fb
        fallback_count += 1

    # Stats
    op_counts = Counter()
    for t in templates:
        sg = t.get("subgraph", {})
        steps = sg.get("steps", [])
        ops = tuple(s.get("op") for s in steps)
        op_counts[ops] += 1

    print(f"\nDSL generation results:")
    print(f"  Valid: {valid_count}")
    print(f"  Fallback (SET): {fallback_count}")
    print(f"  Fallback rate: {fallback_count / max(1, len(to_process)) * 100:.1f}%")
    print(f"  Operation distribution:")
    for ops, count in op_counts.most_common(15):
        print(f"    {ops}: {count}")

    # Save
    with open(output_path, "w") as f:
        json.dump(templates, f, indent=2)
    print(f"\nSaved to {output_path}")

    return templates


# =============================================================================
# Step 5: Build coarse templates + coarse↔fine mapping
# =============================================================================

def step_coarse_templates(args):
    """Step 5: Build coarse templates with mapping to atomic templates."""
    print("\n" + "=" * 60)
    print("STEP 5: Build coarse templates + coarse↔fine mapping")
    print("=" * 60)

    # Load coarse spans
    coarse_path = DATA_DIR / "qwen_coarse_spans.json"
    print(f"\nLoading coarse spans from {coarse_path}...")
    with open(coarse_path) as f:
        data = json.load(f)
    coarse_spans = data["spans"] if isinstance(data, dict) else data
    print(f"Loaded {len(coarse_spans)} coarse spans")

    # Load atoms (for lineage)
    atoms_path = DATA_DIR / "atomic_spans_v2.json"
    print(f"Loading atoms from {atoms_path}...")
    with open(atoms_path) as f:
        all_atoms = json.load(f)
    print(f"Loaded {len(all_atoms)} atoms")

    # Load atomic templates (for matching atoms to templates)
    templates_path = DATA_DIR / "atomic_templates_v2.json"
    print(f"Loading atomic templates from {templates_path}...")
    with open(templates_path) as f:
        atomic_templates = json.load(f)
    print(f"Loaded {len(atomic_templates)} atomic templates")

    # Check checkpoint
    output_path = DATA_DIR / "coarse_templates.json"
    if output_path.exists() and not args.force:
        print(f"Checkpoint found at {output_path}, skipping.")
        with open(output_path) as f:
            return json.load(f)

    # Build atom→template lookup
    # First, embed all atomic template patterns
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    template_centroids = np.array([t["centroid"] for t in atomic_templates], dtype=np.float32)
    norms = np.linalg.norm(template_centroids, axis=1, keepdims=True)
    template_centroids = template_centroids / np.maximum(norms, 1e-8)

    # Build coarse span → atom children lookup
    # Key: (problem_id, span_idx) → list of atom dicts
    coarse_to_atoms = defaultdict(list)
    for a in all_atoms:
        key = (a.get("parent_problem_id", ""), a.get("parent_span_idx", 0))
        coarse_to_atoms[key].append(a)

    # For each atom, find its nearest atomic template
    unique_atom_texts = list(set(a["atomic"] for a in all_atoms))
    print(f"\nEmbedding {len(unique_atom_texts)} unique atom texts...")
    atom_embeddings = model.encode(unique_atom_texts, batch_size=256,
                                    normalize_embeddings=True, show_progress_bar=True)
    atom_emb_lookup = dict(zip(unique_atom_texts, atom_embeddings))

    # Match each atom text to nearest atomic template
    atom_to_template = {}
    for text, emb in atom_emb_lookup.items():
        sims = template_centroids @ emb
        best_idx = int(np.argmax(sims))
        atom_to_template[text] = atomic_templates[best_idx]["template_id"]

    # Now embed all coarse span texts for clustering
    coarse_texts = [s["span_text"] for s in coarse_spans]
    print(f"\nEmbedding {len(coarse_texts)} coarse span texts...")
    coarse_embeddings = model.encode(coarse_texts, batch_size=256,
                                      normalize_embeddings=True, show_progress_bar=True)
    coarse_embeddings = np.array(coarse_embeddings, dtype=np.float32)

    # Cluster coarse spans using MiniBatchKMeans (61K embeddings too large for distance matrix)
    from sklearn.cluster import MiniBatchKMeans

    target = args.coarse_target
    print(f"\nClustering {len(coarse_embeddings)} coarse spans into ~{target} clusters via MiniBatchKMeans...")
    kmeans = MiniBatchKMeans(n_clusters=target, batch_size=2048, n_init=3, random_state=42)
    labels = kmeans.fit_predict(coarse_embeddings)

    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
    clusters = list(clusters.values())
    print(f"  Got {len(clusters)} clusters")

    # Build coarse templates
    coarse_templates = []
    for cid, member_indices in enumerate(clusters):
        rep_idx = max(member_indices, key=lambda i: 1)  # First member as representative
        rep_span = coarse_spans[member_indices[0]]

        examples = []
        connectivities = []
        backward_attns = []
        span_positions = []
        atomic_sequences = Counter()

        for mi in member_indices:
            s = coarse_spans[mi]
            examples.append(s["span_text"])
            connectivities.append(s.get("connectivity", 0.0))
            backward_attns.append(s.get("backward_attention", 0.0))
            span_positions.append(s.get("span_position", 0.0))

            # Find this span's atomic decomposition
            key = (s.get("problem_id", ""), s.get("span_idx", 0))
            child_atoms = coarse_to_atoms.get(key, [])
            if child_atoms:
                seq = tuple(atom_to_template.get(a["atomic"], "unknown") for a in child_atoms)
                atomic_sequences[seq] += 1

        # Most common atomic template sequence
        if atomic_sequences:
            best_seq = atomic_sequences.most_common(1)[0][0]
            # Look up template details for each in sequence
            template_lookup = {t["template_id"]: t for t in atomic_templates}
            decomposition = []
            for tid in best_seq:
                t = template_lookup.get(tid)
                if t:
                    decomposition.append({
                        "template_id": tid,
                        "pattern": t["pattern"],
                    })
        else:
            decomposition = []

        centroid = np.mean(coarse_embeddings[member_indices], axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        coarse_templates.append({
            "template_id": f"coarse_{cid:04d}",
            "pattern": examples[0] if examples else "",
            "centroid": centroid.tolist(),
            "member_count": len(member_indices),
            "span_examples": list(set(examples))[:20],
            "avg_connectivity": float(np.mean(connectivities)),
            "avg_backward_attention": float(np.mean(backward_attns)),
            "avg_span_position": float(np.mean(span_positions)),
            "atomic_decomposition": decomposition,
            "decomposition_coverage": atomic_sequences.most_common(1)[0][1] / len(member_indices) if atomic_sequences else 0.0,
        })

    coarse_templates.sort(key=lambda t: t["member_count"], reverse=True)

    # Stats
    sizes = [t["member_count"] for t in coarse_templates]
    decomp_lengths = [len(t["atomic_decomposition"]) for t in coarse_templates]
    print(f"\n  Coarse templates: {len(coarse_templates)}")
    print(f"  Mean size: {np.mean(sizes):.1f}")
    print(f"  Mean decomposition length: {np.mean(decomp_lengths):.1f}")
    print(f"\n  Top 15:")
    for t in coarse_templates[:15]:
        decomp_str = " → ".join(d["pattern"] for d in t["atomic_decomposition"][:3])
        print(f"    [{t['member_count']:4d}] {t['pattern'][:50]} → [{decomp_str[:60]}]")

    # Save
    with open(output_path, "w") as f:
        json.dump(coarse_templates, f, indent=2)
    print(f"\nSaved {len(coarse_templates)} coarse templates to {output_path}")

    return coarse_templates


# =============================================================================
# Step 6: Fine-tune MiniLM on new coarse templates
# =============================================================================

def step_finetune(args):
    """Step 6: Contrastive fine-tuning of MiniLM on coarse template clusters."""
    print("\n" + "=" * 60)
    print("STEP 6: Fine-tune MiniLM on new coarse templates")
    print("=" * 60)

    import random
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    # Load coarse spans with their reduced forms
    coarse_path = DATA_DIR / "vocab_reduced_coarse.json"
    print(f"\nLoading from {coarse_path}...")
    with open(coarse_path) as f:
        coarse_spans = json.load(f)
    print(f"Loaded {len(coarse_spans)} spans")

    # Load coarse templates for cluster assignment
    templates_path = DATA_DIR / "coarse_templates.json"
    print(f"Loading coarse templates from {templates_path}...")
    with open(templates_path) as f:
        coarse_templates = json.load(f)
    print(f"Loaded {len(coarse_templates)} templates")

    # Build centroid matrix
    centroids = np.array([t["centroid"] for t in coarse_templates], dtype=np.float32)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(norms, 1e-8)

    # Load base model
    print("\nLoading base MiniLM...")
    base_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Encode all span texts and assign to templates
    span_texts = [s.get("span_text", "") for s in coarse_spans]
    print(f"Encoding {len(span_texts)} spans...")
    embeddings = base_model.encode(span_texts, batch_size=512,
                                    normalize_embeddings=True, show_progress_bar=True)

    # Assign to clusters
    by_cluster = defaultdict(list)
    for i, emb in enumerate(embeddings):
        sims = centroids @ emb
        best = int(np.argmax(sims))
        by_cluster[best].append(span_texts[i])

    # Build training pairs
    valid_clusters = {k: v for k, v in by_cluster.items() if len(v) >= 2}
    print(f"\nBuilding pairs from {len(valid_clusters)} clusters (>= 2 members)")

    examples = []
    for cid, texts in valid_clusters.items():
        n_pairs = min(len(texts) * 3, 500)
        for _ in range(n_pairs):
            a, b = random.sample(texts, 2)
            examples.append(InputExample(texts=[a, b]))

    random.shuffle(examples)
    max_pairs = args.max_pairs
    if len(examples) > max_pairs:
        examples = random.sample(examples, max_pairs)
    print(f"  Training pairs: {len(examples)}")

    # Train
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=base_model)

    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(total_steps * 0.1)

    output_path = MODELS_DIR / "minilm_v2_finetuned"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining: {args.epochs} epochs, {total_steps} steps, batch {args.batch_size}")
    base_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": 2e-5},
        output_path=str(output_path),
        show_progress_bar=True,
    )

    print(f"\nModel saved to {output_path}")
    return str(output_path)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Rebuild span pipeline")
    parser.add_argument("--step", required=True,
                        choices=["vocab-reduce", "atomic-split", "collapse",
                                 "generate-dsls", "coarse-templates", "finetune", "all"])
    parser.add_argument("--force", action="store_true", help="Overwrite checkpoints")
    parser.add_argument("--tp-size", type=int, default=4, help="Tensor parallel size for vLLM")
    parser.add_argument("--target-templates", type=int, default=1000,
                        help="Target atomic template count for collapse")
    parser.add_argument("--coarse-target", type=int, default=300,
                        help="Target coarse template count")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-pairs", type=int, default=200000)
    args = parser.parse_args()

    steps = {
        "vocab-reduce": step_vocab_reduce,
        "atomic-split": step_atomic_split,
        "collapse": step_collapse,
        "generate-dsls": step_generate_dsls,
        "coarse-templates": step_coarse_templates,
        "finetune": step_finetune,
    }

    if args.step == "all":
        for name, func in steps.items():
            func(args)
    else:
        steps[args.step](args)

    print("\nDone!")


if __name__ == "__main__":
    main()
