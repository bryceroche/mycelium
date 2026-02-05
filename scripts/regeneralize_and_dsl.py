#!/usr/bin/env python3
"""
Three-phase pipeline for template creation on AWS VM with Qwen.

Phase 1: Re-generalize raw spans with expanded dimensions
  - Existing: [PERSON1], [ITEM1], [N]
  - New: [LOC1], [TIME1]

Phase 2: Embed + cluster to ~1K templates
  - MiniLM embeddings on raw spans (not patterns)
  - Greedy cosine clustering (threshold auto-tuned for ~1K)

Phase 3: Qwen writes custom sub-graph DSL for each template
  - Pattern + examples → DSL expression

Output: templates_1k_with_dsl.json
Raw spans preserved in generalized_v2.json for future re-runs.

USAGE:
    # Full pipeline on VM with 4 GPUs:
    python scripts/regeneralize_and_dsl.py --tp-size 4

    # Resume from phase 2 (skip re-generalization):
    python scripts/regeneralize_and_dsl.py --skip-phase1

    # Resume from phase 3 (skip re-gen + clustering):
    python scripts/regeneralize_and_dsl.py --skip-phase1 --skip-phase2
"""

import argparse
import json
import re
import os
import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Phase 1: Re-generalize with expanded dimensions
# =============================================================================

GENERALIZATION_PROMPT_V2 = """Replace names/pronouns with [PERSON1],[PERSON2],etc. Replace objects/nouns with [ITEM1],[ITEM2],etc. Replace numbers/$amounts with [N]. Replace locations/places with [LOC1],[LOC2],etc. Replace time references with [TIME1],[TIME2],etc. Keep verbs and structure. Output ONLY JSON.

{{"input": "John has 5 apples", "output": "[PERSON1] has [N] [ITEM1]"}}
{{"input": "She gave 2 cookies to Mary at the store", "output": "[PERSON1] gave [N] [ITEM1] to [PERSON2] at [LOC1]"}}
{{"input": "Each bag contains 5 items", "output": "each [ITEM1] contains [N] [ITEM2]"}}
{{"input": "She earned $12 per hour for 8 hours on Monday", "output": "[PERSON1] earned [N] per [ITEM1] for [N] [ITEM1] on [TIME1]"}}
{{"input": "He walks 3 miles to school every morning", "output": "[PERSON1] walks [N] [ITEM1] to [LOC1] every [TIME1]"}}
{{"input": "The farmers market sells oranges for $2 each", "output": "[LOC1] sells [ITEM1] for [N] each"}}
{{"input": "On Tuesday, she bought 4 boxes at the grocery store", "output": "on [TIME1], [PERSON1] bought [N] [ITEM1] at [LOC1]"}}
{{"input": "Last week he spent $50 at the gym", "output": "[TIME1] [PERSON1] spent [N] at [LOC1]"}}
{{"input": "{span}", "output": """


def _clean_pattern(raw: str) -> str:
    """Strip common artifacts from extracted patterns."""
    p = raw.strip()
    p = re.sub(r'["\s}]+$', '', p)
    p = p.lstrip('"').lstrip("'")
    p = p.rstrip('.')
    return p.strip()


def parse_qwen_response(response: str) -> Optional[str]:
    """Parse Qwen's JSON response to extract the generalized pattern."""
    cleaned = response.strip()

    for attempt in [
        '{"output": ' + cleaned,
        cleaned,
        '{"output": ' + cleaned + '}',
        '{"output": "' + cleaned,
    ]:
        try:
            fixed = attempt.replace("'", '"').rstrip(',').rstrip()
            if not fixed.endswith('}'):
                fixed = fixed.rstrip('"') + '"}'
            obj = json.loads(fixed)
            pattern = _clean_pattern(obj.get('output', ''))
            if pattern and len(pattern) > 3:
                return pattern
        except (json.JSONDecodeError, AttributeError):
            continue

    match = re.search(r'"output"\s*:\s*"([^"]+)"', cleaned)
    if match:
        pattern = _clean_pattern(match.group(1))
        if pattern and len(pattern) > 3:
            return pattern

    # Fallback: look for placeholder pattern
    placeholder_re = re.compile(r'\[(?:PERSON|ITEM|LOC|TIME)\d+\]|\[N\]', re.IGNORECASE)
    for line in cleaned.split('\n'):
        line = _clean_pattern(line)
        if placeholder_re.search(line) and len(line) > 5:
            if any(x in line.lower() for x in ['replace', 'should be', 'we need']):
                continue
            return line

    return None


def regex_fallback(span: str) -> str:
    """Fallback generalization when Qwen fails."""
    pattern = re.sub(r'\$[\d,]+\.?\d*', '[N]', span)
    pattern = re.sub(r'\b\d{1,3}(?:,\d{3})+\.?\d*\b', '[N]', pattern)
    pattern = re.sub(r'\b\d+\.?\d*\b', '[N]', pattern)
    return pattern.lower()


def phase1_regeneralize(records: List[Dict], llm, tokenizer, checkpoint_dir: str = ".") -> List[Dict]:
    """Re-generalize all raw spans with expanded dimensions."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(max_tokens=60, temperature=0)
    chunk_size = 3200

    # Check for checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "regen_checkpoint.json")
    results = []
    start_idx = 0

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            cp = json.load(f)
        results = cp["results"]
        start_idx = cp["next_idx"]
        print(f"  Resuming from checkpoint: {len(results)} done, starting at {start_idx}")

    raw_spans = [r['raw_span'] for r in records]
    total_chunks = (len(raw_spans) - start_idx + chunk_size - 1) // chunk_size

    for chunk_num, chunk_start in enumerate(range(start_idx, len(raw_spans), chunk_size)):
        chunk = raw_spans[chunk_start:chunk_start + chunk_size]

        prompts = []
        for span in chunk:
            prompt = GENERALIZATION_PROMPT_V2.format(span=span)
            messages = [
                {"role": "system", "content": "Complete the JSON. Output ONLY the pattern value and closing brace. No explanation."},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)

        print(f"  Chunk {chunk_num + 1}/{total_chunks}: generating {len(prompts)} spans...")
        outputs = llm.generate(prompts, sampling_params)

        for i, (span, output) in enumerate(zip(chunk, outputs)):
            response = output.outputs[0].text.strip()
            pattern = parse_qwen_response(response) if response else None

            idx = chunk_start + i
            result = {
                'problem_id': records[idx].get('problem_id', ''),
                'problem_text': records[idx].get('problem_text', ''),
                'raw_span': span,
                'pattern_v1': records[idx].get('pattern', ''),  # Keep old pattern
                'pattern': pattern if pattern else regex_fallback(span),
                'qwen_failed': pattern is None,
                'qwen_response': response,
            }
            results.append(result)

        # Checkpoint
        with open(checkpoint_path, 'w') as f:
            json.dump({"results": results, "next_idx": chunk_start + len(chunk)}, f)
        print(f"  Checkpoint: {len(results)}/{len(raw_spans)}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return results


# =============================================================================
# Phase 2: Embed + Cluster
# =============================================================================

def phase2_embed_and_cluster(records: List[Dict], target_count: int = 1000) -> List[Dict]:
    """Embed PATTERNS (not raw spans) with MiniLM and cluster to target_count templates.

    We embed the generalized patterns (e.g., "[PERSON1] has [N] [ITEM1] at [LOC1]")
    rather than raw spans to avoid lexical noise from specific words like
    "farmers market", "cookies", etc. polluting the embedding space.
    """
    from mycelium.dual_signal_templates import SpanDetector

    print(f"\nPhase 2: Embedding patterns (not raw spans)...")
    detector = SpanDetector()

    # Stage 1: GROUP BY exact pattern
    groups = defaultdict(list)
    for r in records:
        if len(r['pattern']) >= 5:
            groups[r['pattern']].append(r)

    print(f"  Exact pattern groups: {len(groups)}")

    # Embed each PATTERN (one embedding per unique pattern)
    raw_templates = []
    group_items = list(groups.items())

    for batch_start in range(0, len(group_items), 500):
        batch = group_items[batch_start:batch_start + 500]
        if batch_start % 2000 == 0:
            print(f"  Embedding pattern {batch_start}/{len(group_items)}...")

        for pattern, group in batch:
            # Embed the PATTERN itself, not the raw spans
            emb, _, _ = detector.extract_features(pattern)
            centroid = emb / (np.linalg.norm(emb) + 1e-8)

            raw_templates.append({
                'pattern': pattern,
                'centroid': centroid,
                'count': len(group),
                'raw_examples': [r['raw_span'] for r in group[:20]],
                'qwen_fail_rate': sum(1 for r in group if r.get('qwen_failed')) / len(group),
            })

    print(f"  Templates before clustering: {len(raw_templates)}")

    # Sort by count for greedy clustering (high-count clusters first)
    raw_templates.sort(key=lambda t: -t['count'])

    centroids = np.array([t['centroid'] for t in raw_templates], dtype=np.float32)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids_norm = centroids / np.maximum(norms, 1e-8)

    # Binary search for threshold that gives ~target_count clusters
    print(f"\n  Finding threshold for ~{target_count} clusters...")
    low_t, high_t = 0.50, 0.95

    for iteration in range(12):
        mid_t = (low_t + high_t) / 2
        n_clusters = _count_clusters(centroids_norm, mid_t)
        print(f"    threshold={mid_t:.4f} → {n_clusters} clusters")

        if n_clusters > target_count:
            high_t = mid_t
        elif n_clusters < target_count:
            low_t = mid_t
        else:
            break

        if high_t - low_t < 0.001:
            break

    # Use final threshold
    final_threshold = (low_t + high_t) / 2
    clusters = _greedy_cluster(centroids_norm, final_threshold)
    print(f"  Final: threshold={final_threshold:.4f} → {len(clusters)} clusters")

    # Build final templates
    final_templates = []
    for cluster_idx, cluster in enumerate(clusters):
        best = raw_templates[cluster[0]]  # Highest count (pre-sorted)

        total_count = sum(raw_templates[i]['count'] for i in cluster)
        all_examples = []
        all_patterns = []
        for i in cluster:
            all_examples.extend(raw_templates[i]['raw_examples'])
            all_patterns.append(raw_templates[i]['pattern'])

        # Average centroid
        cluster_centroids = centroids[cluster]
        avg_centroid = np.mean(cluster_centroids, axis=0)
        avg_centroid = avg_centroid / (np.linalg.norm(avg_centroid) + 1e-8)

        final_templates.append({
            'template_id': f"tpl_{cluster_idx:04d}",
            'pattern': best['pattern'],
            'embedding_centroid': avg_centroid.tolist(),
            'count': total_count,
            'cluster_size': len(cluster),
            'raw_examples': all_examples[:20],  # Keep raw spans!
            'member_patterns': list(set(all_patterns))[:10],
            'qwen_fail_rate': best['qwen_fail_rate'],
            'dsl_expr': 'value',  # Placeholder — Phase 3 fills this in
        })

    final_templates.sort(key=lambda t: -t['count'])

    return final_templates


def _count_clusters(centroids_norm: np.ndarray, threshold: float) -> int:
    """Fast cluster count without building full cluster lists."""
    n = len(centroids_norm)
    used = set()
    count = 0
    for i in range(n):
        if i in used:
            continue
        count += 1
        used.add(i)
        remaining = [j for j in range(i + 1, n) if j not in used]
        if remaining:
            remaining_arr = np.array(remaining)
            sims = centroids_norm[remaining_arr] @ centroids_norm[i]
            for idx, sim in zip(remaining_arr, sims):
                if sim >= threshold:
                    used.add(idx)
    return count


def _greedy_cluster(centroids_norm: np.ndarray, threshold: float) -> List[List[int]]:
    """Greedy cosine similarity clustering."""
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


# =============================================================================
# Phase 3: Qwen writes DSL for each template
# =============================================================================

DSL_PROMPT = """You are classifying math word problem patterns into computation types.

Given a pattern template and example sentences, determine what arithmetic operation this pattern performs on a tracked entity (like a person's money, items, score, etc.).

Respond with EXACTLY one of these DSL expressions:
- "value" — introduces/assigns a new quantity (e.g., "has 5 apples", "costs $10", "there are 20 students")
- "entity + value" — adds to existing quantity (e.g., "gets 3 more", "earns additional 5", "found 2 extra")
- "entity - value" — subtracts from existing quantity (e.g., "gave away 3", "lost 2", "spent $5")
- "entity * value" — multiplies existing quantity (e.g., "twice as many", "tripled", "3 times")
- "entity / value" — divides existing quantity (e.g., "split among 3", "half of", "divided into 4")

Think about the COMPUTATION, not the words. "Each costs $5" is "value" (introducing a rate). "Sold half" is "entity / value".

Pattern: {pattern}
Examples:
{examples}

DSL expression (one word/phrase only): """


def phase3_generate_dsl(templates: List[Dict], llm, tokenizer) -> List[Dict]:
    """Have Qwen write DSL expressions for each template."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(max_tokens=10, temperature=0)

    print(f"\nPhase 3: Generating DSL for {len(templates)} templates...")

    # Build all prompts
    prompts = []
    for t in templates:
        examples = t.get('raw_examples', t.get('pattern_examples', []))[:5]
        examples_str = "\n".join(f"  - {ex}" for ex in examples) if examples else "  (no examples)"

        prompt = DSL_PROMPT.format(pattern=t['pattern'], examples=examples_str)
        messages = [
            {"role": "system", "content": "Respond with exactly one DSL expression. No explanation."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)

    # Generate all at once (only ~1K prompts)
    print(f"  Generating {len(prompts)} DSL classifications...")
    outputs = llm.generate(prompts, sampling_params)

    valid_dsls = {"value", "entity + value", "entity - value", "entity * value", "entity / value"}

    for t, output in zip(templates, outputs):
        response = output.outputs[0].text.strip().lower()
        # Clean up response
        response = response.strip('"').strip("'").strip()

        # Try to match a valid DSL
        if response in valid_dsls:
            t['dsl_expr'] = response
        else:
            # Fuzzy matching
            for dsl in valid_dsls:
                if dsl in response:
                    t['dsl_expr'] = dsl
                    break
            else:
                t['dsl_expr'] = 'value'  # Default

    # Stats
    dsl_dist = defaultdict(int)
    for t in templates:
        dsl_dist[t['dsl_expr']] += 1

    print(f"\n  DSL Distribution:")
    for dsl, count in sorted(dsl_dist.items(), key=lambda x: -x[1]):
        pct = count / len(templates) * 100
        print(f"    {dsl:<20} {count:>5} ({pct:.1f}%)")

    return templates


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Re-generalize + cluster + DSL pipeline")
    parser.add_argument("--input", default="gsm8k_generalized.json",
                        help="Input: existing generalized spans (with raw_span)")
    parser.add_argument("--output-generalized", default="generalized_v2.json",
                        help="Output: re-generalized spans (preserves raw_span)")
    parser.add_argument("--output-templates", default="templates_1k_with_dsl.json",
                        help="Output: ~1K templates with DSL expressions")
    parser.add_argument("--target-templates", type=int, default=1000,
                        help="Target number of templates after clustering")
    parser.add_argument("--tp-size", type=int, default=1,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--skip-phase1", action="store_true",
                        help="Skip re-generalization (use existing generalized_v2.json)")
    parser.add_argument("--skip-phase2", action="store_true",
                        help="Skip embedding+clustering (use existing clustered templates)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    input_path = project_root / args.input
    gen_v2_path = project_root / args.output_generalized
    templates_path = project_root / args.output_templates

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    def load_qwen():
        """Load Qwen with vLLM."""
        from vllm import LLM
        from transformers import AutoTokenizer
        print(f"Loading {model_name} with vLLM (tp={args.tp_size})...")
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        engine = LLM(
            model=model_name,
            tensor_parallel_size=args.tp_size,
            dtype="float16",
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
        )
        print(f"Loaded {model_name}")
        return engine, tok

    def unload_qwen(engine):
        """Unload vLLM to free GPU memory for MiniLM."""
        import gc, torch
        print("Unloading Qwen to free GPU memory...")
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("GPU memory freed")

    # =========================================================================
    # Phase 1: Re-generalize
    # =========================================================================
    if not args.skip_phase1:
        print("=" * 60)
        print("PHASE 1: Re-generalizing with expanded dimensions")
        print("=" * 60)

        llm, tokenizer = load_qwen()

        print(f"Loading {input_path}...")
        with open(input_path) as f:
            records = json.load(f)
        print(f"Loaded {len(records)} records")

        records_v2 = phase1_regeneralize(records, llm, tokenizer, str(project_root))

        # Save
        print(f"\nSaving {len(records_v2)} re-generalized records to {gen_v2_path}...")
        with open(gen_v2_path, 'w') as f:
            json.dump(records_v2, f, indent=2)

        # Stats
        fail_count = sum(1 for r in records_v2 if r.get('qwen_failed'))
        print(f"Qwen failures: {fail_count}/{len(records_v2)} ({100*fail_count/max(len(records_v2),1):.1f}%)")

        # Compare v1 vs v2 pattern uniqueness
        v1_patterns = len(set(r.get('pattern_v1', '') for r in records_v2))
        v2_patterns = len(set(r['pattern'] for r in records_v2))
        print(f"Unique patterns: v1={v1_patterns} → v2={v2_patterns}")

        # Free GPU for Phase 2
        unload_qwen(llm)
        llm = None
    else:
        print("Skipping Phase 1 (loading existing generalized_v2.json)")
        with open(gen_v2_path) as f:
            records_v2 = json.load(f)
        print(f"Loaded {len(records_v2)} records")

    # =========================================================================
    # Phase 2: Embed + Cluster (MiniLM on CPU — separate from Qwen)
    # =========================================================================
    if not args.skip_phase2:
        print("\n" + "=" * 60)
        print("PHASE 2: Embed + Cluster to ~{} templates".format(args.target_templates))
        print("=" * 60)

        templates = phase2_embed_and_cluster(records_v2, target_count=args.target_templates)
        print(f"\nGot {len(templates)} templates")

        # Save intermediate (before DSL)
        intermediate_path = project_root / "templates_clustered_no_dsl.json"
        with open(intermediate_path, 'w') as f:
            json.dump(templates, f, indent=2)
        print(f"Saved intermediate to {intermediate_path}")
    else:
        print("\nSkipping Phase 2 (loading existing clustered templates)")
        intermediate_path = project_root / "templates_clustered_no_dsl.json"
        with open(intermediate_path) as f:
            templates = json.load(f)
        print(f"Loaded {len(templates)} templates")

    # =========================================================================
    # Phase 3: Qwen writes DSL (reload Qwen after MiniLM is done)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Qwen writes custom DSL expressions")
    print("=" * 60)

    llm, tokenizer = load_qwen()
    templates = phase3_generate_dsl(templates, llm, tokenizer)

    # Save final
    print(f"\nSaving {len(templates)} templates to {templates_path}...")
    with open(templates_path, 'w') as f:
        json.dump(templates, f, indent=2)

    # Final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Templates: {len(templates)}")
    print(f"Output: {templates_path}")
    print(f"Raw spans preserved in: {gen_v2_path}")

    print(f"\nTop 15 templates:")
    for t in templates[:15]:
        print(f"  [{t['dsl_expr']:<18}] {t['pattern'][:50]:<50} (n={t['count']})")


if __name__ == "__main__":
    main()
