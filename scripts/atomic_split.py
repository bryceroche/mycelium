#!/usr/bin/env python3
"""
Atomic span splitting: break vocab-reduced spans into single-operation primitives.

Takes the output of vocab_reduce.py (vocab_reduced_records.json) and asks Qwen
to split each reduced span into atomic operations that will collapse into
a small number of templates.

Pipeline: raw_span → vocab_reduce → atomic_split → cluster → fine-tune MiniLM

USAGE:
    python scripts/atomic_split.py --tp-size 4
    python scripts/atomic_split.py --tp-size 4 --skip-to-cluster  # resume from split
"""

import argparse
import json
import re
import sys
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

OUTPUT_DIR = Path(__file__).parent.parent

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

"items cost n percent more than items" →
items cost more than items
n percent more

"person gave n items to person at place" →
person gave n items
to person
at place

"person has n items" →
person has n items

"per day" →
per day

Input: "{span}"
Atomic spans:
"""


def load_qwen(tp_size: int = 4):
    """Load Qwen via vLLM."""
    from vllm import LLM
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        max_model_len=1024,
        gpu_memory_utilization=0.85,
    )
    return llm


def unload_qwen(engine):
    """Free GPU memory."""
    import gc, torch
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def batch_qwen(llm, prompts: List[str], max_tokens: int = 100) -> List[str]:
    """Run batch inference."""
    from vllm import SamplingParams
    params = SamplingParams(
        temperature=0.1,
        max_tokens=max_tokens,
        stop=["\n\n", "```", "Input:", "Example"],
    )
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text.strip() for o in outputs]


def parse_atomic_spans(response: str, original_span: str) -> List[str]:
    """Parse Qwen's response into a list of atomic spans."""
    lines = response.strip().split('\n')
    atoms = []
    for line in lines:
        line = line.strip()
        # Remove numbering like "1.", "- ", "* "
        line = re.sub(r'^[\d]+[.)]\s*', '', line)
        line = re.sub(r'^[-*]\s*', '', line)
        # Remove quotes
        line = line.strip('"\'')
        # Clean whitespace
        line = ' '.join(line.split()).lower()
        # Skip empty or too long
        if not line or len(line) < 2:
            continue
        if len(line) > len(original_span) * 1.5:
            continue
        # Skip if it's just punctuation or artifacts
        if not re.search(r'[a-z]', line):
            continue
        atoms.append(line)

    # If parsing failed, return original as single atom
    if not atoms:
        atoms = [original_span.lower().strip()]

    return atoms


def split_spans(llm, records: List[Dict], batch_size: int = 2000) -> List[Dict]:
    """Split all reduced spans into atomic operations."""
    print(f"\n{'='*60}")
    print(f"ATOMIC SPLITTING: {len(records)} spans")
    print(f"{'='*60}")

    all_atoms = []
    failures = 0

    # Build prompts for spans that have a reduced form
    items = []
    for i, r in enumerate(records):
        span = r.get('reduced', r.get('raw_span', ''))
        if span:
            items.append((i, r, span))

    print(f"  Splitting {len(items)} spans into atomic operations...")

    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start:batch_start + batch_size]
        prompts = [SPLIT_PROMPT.format(span=span) for _, _, span in batch]

        responses = batch_qwen(llm, prompts, max_tokens=120)

        for (idx, record, original_span), response in zip(batch, responses):
            atoms = parse_atomic_spans(response, original_span)

            for atom in atoms:
                all_atoms.append({
                    "parent_idx": idx,
                    "raw_span": record.get("raw_span", ""),
                    "reduced": original_span,
                    "atomic": atom,
                    "problem_id": record.get("problem_id", ""),
                    "span_idx": record.get("span_idx", 0),
                })

            if len(atoms) == 1 and atoms[0] == original_span.lower().strip():
                failures += 1

        done = min(batch_start + batch_size, len(items))
        pct_fail = failures / done * 100 if done > 0 else 0
        print(f"  [{done}/{len(items)}] atoms={len(all_atoms)} "
              f"avg={len(all_atoms)/done:.1f}/span failures={failures} ({pct_fail:.1f}%)")

    # Stats
    atom_patterns = Counter(a["atomic"] for a in all_atoms)
    singletons = sum(1 for c in atom_patterns.values() if c == 1)

    print(f"\n  Atomic split results:")
    print(f"    Input spans: {len(items)}")
    print(f"    Output atoms: {len(all_atoms)}")
    print(f"    Avg atoms/span: {len(all_atoms)/len(items):.2f}")
    print(f"    Unique atom patterns: {len(atom_patterns)}")
    print(f"    Singletons: {singletons} ({singletons/len(atom_patterns)*100:.1f}%)")
    print(f"    Failures (unchanged): {failures}")
    print(f"    Top 30 atom patterns:")
    for pat, count in atom_patterns.most_common(30):
        print(f"      [{count:5d}] {pat}")

    return all_atoms


def filter_atoms(atoms: List[Dict]) -> List[Dict]:
    """Remove garbage atoms: prompt leakage, corruption, too short/long, single words."""
    filtered = []
    rejected = 0
    reject_reasons = Counter()

    for a in atoms:
        text = a["atomic"].strip()

        # Reject prompt leakage from Qwen
        if any(leak in text for leak in [
            "ai assistant", "your response", "generate", "atomic spans",
            "atomic span", "input:", "output:", "example", "summary:",
            "rewrite", "rules:", "split this",
        ]):
            rejected += 1
            reject_reasons["prompt_leakage"] += 1
            continue

        # Reject corruption (weird fragments, too short/long)
        if len(text) < 4 or len(text) > 80:
            rejected += 1
            reject_reasons["bad_length"] += 1
            continue

        # Reject if no alpha chars
        if not re.search(r'[a-z]', text):
            rejected += 1
            reject_reasons["no_alpha"] += 1
            continue

        # Reject single words — not meaningful atomic operations
        words = text.split()
        if len(words) < 2:
            rejected += 1
            reject_reasons["single_word"] += 1
            continue

        # Reject common filler phrases with no operation
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


def cluster_atoms(atoms: List[Dict], target_count: int = 200) -> List[Dict]:
    """Cluster atomic spans using K-means (no transitive chaining problem)."""
    print(f"\n{'='*60}")
    print(f"CLUSTERING: {len(set(a['atomic'] for a in atoms))} unique atoms → ~{target_count} templates")
    print(f"{'='*60}")

    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import MiniBatchKMeans

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Filter garbage atoms first
    atoms = filter_atoms(atoms)

    # Group by atomic pattern
    pattern_groups = defaultdict(list)
    for a in atoms:
        pattern_groups[a["atomic"]].append(a)

    patterns = list(pattern_groups.keys())
    pattern_counts = [len(pattern_groups[p]) for p in patterns]
    print(f"  Unique patterns after filter: {len(patterns)}")

    # Batch embed all unique patterns
    print(f"  Embedding {len(patterns)} patterns...")
    embeddings = model.encode(patterns, batch_size=256, show_progress_bar=True,
                              normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"  Embeddings shape: {embeddings.shape}")

    # K-means clustering — fast, no chaining, deterministic k
    print(f"  Running MiniBatchKMeans with k={target_count}...")
    kmeans = MiniBatchKMeans(
        n_clusters=target_count,
        batch_size=4096,
        n_init=3,
        random_state=42,
    )
    labels = kmeans.fit_predict(embeddings)

    # Build cluster dict
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(i)

    print(f"  Clusters formed: {len(clusters)}")

    # Build templates
    templates = []
    for cid, member_indices in clusters.items():
        # Pick the most common pattern as representative
        rep_idx = max(member_indices, key=lambda i: len(pattern_groups[patterns[i]]))
        rep_pattern = patterns[rep_idx]

        all_raw_spans = []
        all_patterns = []
        member_count = 0
        for mi in member_indices:
            p = patterns[mi]
            all_patterns.append(p)
            for a in pattern_groups[p]:
                all_raw_spans.append(a.get("raw_span", ""))
                member_count += 1

        centroid = np.mean(embeddings[member_indices], axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        templates.append({
            "template_id": f"atom_{cid:04d}",
            "pattern": rep_pattern,
            "all_patterns": all_patterns[:20],
            "centroid": centroid.tolist(),
            "span_examples": list(set(all_raw_spans))[:20],
            "member_count": member_count,
        })

    # Size distribution
    sizes = [t["member_count"] for t in templates]
    print(f"\n  Template size distribution:")
    print(f"    Total atoms assigned: {sum(sizes)}")
    print(f"    Mean size: {np.mean(sizes):.1f}")
    print(f"    Median size: {np.median(sizes):.0f}")
    print(f"    Max size: {max(sizes)} ({templates[np.argmax(sizes)]['pattern']})")
    print(f"    Singletons: {sum(1 for s in sizes if s == 1)}")
    print(f"\n  Top 20 templates:")
    sorted_tpls = sorted(templates, key=lambda t: t["member_count"], reverse=True)
    for t in sorted_tpls[:20]:
        print(f"    [{t['member_count']:5d}] {t['pattern']}")

    # Save
    output_path = OUTPUT_DIR / "atomic_templates.json"
    with open(output_path, "w") as f:
        json.dump(templates, f, indent=2)
    print(f"\n  Saved {len(templates)} templates to {output_path}")

    return templates


def main():
    parser = argparse.ArgumentParser(description="Atomic Span Splitting")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--target-templates", type=int, default=200)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--skip-to-cluster", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Atomic Span Splitting Pipeline")
    print("=" * 60)

    atoms_path = OUTPUT_DIR / "atomic_spans.json"

    if args.skip_to_cluster:
        print(f"\nLoading atoms from: {atoms_path}")
        with open(atoms_path) as f:
            all_atoms = json.load(f)
        print(f"Atoms: {len(all_atoms)}")
    else:
        # Load vocab-reduced records
        input_path = Path(args.input) if args.input else OUTPUT_DIR / "vocab_reduced_records.json"
        print(f"\nLoading from: {input_path}")
        with open(input_path) as f:
            records = json.load(f)
        print(f"Records: {len(records)}")

        # Split into atomic spans
        llm = load_qwen(args.tp_size)
        all_atoms = split_spans(llm, records)
        unload_qwen(llm)

        # Save checkpoint
        with open(atoms_path, "w") as f:
            json.dump(all_atoms, f, indent=2)
        print(f"\n  Checkpoint saved to {atoms_path}")

    # Cluster
    templates = cluster_atoms(all_atoms, target_count=args.target_templates)

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Input records: {len(all_atoms)} atoms")
    print(f"  Final templates: {len(templates)}")
    singleton_tpls = sum(1 for t in templates if t["member_count"] == 1)
    print(f"  Singleton templates: {singleton_tpls}")
    print(f"  Output: {OUTPUT_DIR / 'atomic_templates.json'}")


if __name__ == "__main__":
    main()
